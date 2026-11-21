#!/usr/bin/env python
"""Score stored attack checkpoints through a real defense (docs/phase3_plan.md section 2).

    # every checkpoint of a run, through the in-repo defense (GPU host)
    python scripts/score_checkpoints.py --run results/runs/conv_jpeg50_e16_s1000 \
        --defense jpeg50 --max-hours 4

    # locally sanitized Dangerzone checkpoints (GPU host; PNGs from
    # scripts/dangerzone_sanitize.py --ckpt-steps, uploaded into the run dir)
    python scripts/score_checkpoints.py --run results/runs/conv_dz_ll_e16_s2000 \
        --dz-dir /path/to/run/dz_ll --max-hours 4

One job per (image, epsilon, checkpoint step, defense), resumable like every other job.
Rows land in jobs/ckpt_score/ and are collated into ckpt_results.jsonl by aggregate().
The stored `target_success` flag uses the historical substring rule, like every raw row;
the exact-match metric is applied at analysis time (scripts/analyze_convergence.py).
Reads the run's config for the model; nothing outside the run dir is written.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.resume import atomic_write_json, is_done, sanitize_id  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.evaluation.metrics import normalize_answer, contains_target  # noqa: E402
from src.evaluation.runner import build_model, build_defenses  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment import store  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("score_ckpt")


def _iter_attack_jobs(lay: RunLayout):
    for jp in sorted(lay.attack_jobs.glob("*.json")):
        rec = json.loads(jp.read_text())
        if rec and not rec.get("error"):
            yield rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (pulled from Drive on Colab)")
    ap.add_argument("--defense", default="jpeg50",
                    help="in-repo defense to apply to each checkpoint array (array mode)")
    ap.add_argument("--dz-dir", default=None,
                    help="score pre-sanitized Dangerzone checkpoints from this directory "
                         "(contains ckpt/<sid>__eps<e>__s<step>.png); overrides --defense")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args(argv)

    lay = RunLayout(Path(args.run))
    cfg = load_config(lay.config)
    manifest = Path(cfg["dataset"]["manifest"])
    manifest = manifest if manifest.exists() else lay.dataset_manifest
    prompt = json.loads(manifest.read_text())["prompt"]
    max_new_tokens = int(cfg["evaluation"]["max_new_tokens"])

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    model = build_model(cfg)
    defense_obj = None
    defense_name = "dangerzone_ll" if args.dz_dir else args.defense
    if not args.dz_dir:
        defenses = build_defenses(cfg)
        if defense_name not in defenses:
            raise SystemExit(f"defense {defense_name!r} is not available")
        defense_obj = defenses[defense_name]

    metas = list(_iter_attack_jobs(lay))
    by_sid = {sanitize_id(m["image_id"]): m for m in metas}

    jobs = []  # (meta, eps, step, input array or None, sanitized png path or None)
    if args.dz_dir:
        dz_ckpt = Path(args.dz_dir) / "ckpt"
        for png in sorted(dz_ckpt.glob("*.png")):
            # <sid>__eps<e>__s<step>: the sid itself contains "__" (the "/" is doubled),
            # so split from the right, never from the middle of the sid
            parts = png.stem.rsplit("__", 2)
            if len(parts) != 3 or not parts[2].startswith("s"):
                continue
            sid, eps_tag, step_tag = parts
            m = by_sid.get(sid)
            if m is None:
                log.warning("no attack job for sanitized checkpoint %s", png.name)
                continue
            eps = float(m["epsilon"])
            step = int(step_tag[1:])
            jobs.append((m, eps, step, None, png))
    else:
        for m in metas:
            eps = float(m["epsilon"])
            for step in store.list_ckpt_steps(lay, m["image_id"], eps):
                jobs.append((m, eps, step, None, None))

    made, failed = 0, 0
    for m, eps, step, _, png in jobs:
        jp = store.ckpt_score_job_path(lay, m["image_id"], eps, step, defense_name)
        if is_done(jp):
            continue
        if deadline is not None and time.time() >= deadline:
            log.info("time budget reached")
            break
        arr = None
        try:
            if png is not None:
                from src.defenses.base import pil_to_array
                from PIL import Image
                arr = pil_to_array(Image.open(png)).astype(np.float32)
                runtime = None
            else:
                arr = store.load_ckpt_image(lay, m["image_id"], eps, step)
                if arr is None:
                    log.warning("missing ckpt array for %s eps=%.4f s=%d", m["image_id"], eps, step)
                    continue
                res = defense_obj.sanitize(arr)
                arr = res.image
                runtime = res.runtime_s
            ans = model.generate(arr, prompt, max_new_tokens)
            rec = {"image_id": m["image_id"], "class": m["class"], "target": m["target"],
                   "epsilon": eps, "ckpt_step": step, "defense": defense_name,
                   "clean_answer": m.get("clean_answer"), "answer": ans,
                   "target_success": int(contains_target(ans, m["target"])),
                   "restored": int(normalize_answer(ans) == normalize_answer(m.get("clean_answer") or "")),
                   "adv_sha256": store.adv_hash(arr), "defense_runtime_s": runtime,
                   "error": None}
        except Exception as e:
            rec = {"image_id": m["image_id"], "epsilon": eps, "ckpt_step": step,
                   "defense": defense_name, "error": repr(e),
                   "traceback": traceback.format_exc()[-1000:]}
            failed += 1
        atomic_write_json(jp, rec)
        made += 1
        if made % 50 == 0:
            log.info("scored %d checkpoints (%d failed)", made, failed)

    summary = aggregate(lay)
    log.info("ckpt scoring done: %d jobs this session, %d failed; run coverage: "
             "n_ckpt_score_jobs=%s", made, failed, summary.get("n_ckpt_score_jobs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
