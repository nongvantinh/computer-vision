#!/usr/bin/env python
"""Stage 2 of the decoupled Dangerzone (D5) evaluation: model answers on the GPU host.

Runs on the same GPU host and SAME 4-bit LLaVA that produced the rest of the run, so
the preservation/restoration comparison stays valid. Reads the sanitized images from
`<run>/dz/` (produced by scripts/dangerzone_sanitize.py) plus the run's clean answers
and targets, generates the model answer on each sanitized image, and writes the D5
clean and defense job files. Then aggregate() folds D5 into the results.

    python scripts/dangerzone_finalize.py --run results/runs/pilot_12 \
        --config configs/experiment.yaml

The attack is never recomputed; this only adds the D5 defense jobs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402
from src.evaluation.runner import build_model  # noqa: E402
from src.evaluation.metrics import normalize_answer, contains_target  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment import store  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.utils.resume import sanitize_id, atomic_write_json, is_done  # noqa: E402

log = get_logger("dz_finalize")
NAME = "dangerzone"


def _load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    lay = RunLayout(Path(args.run))
    dz_dir = lay.root / "dz"
    prompt = json.loads(lay.dataset_manifest.read_text())["prompt"]
    mnt = cfg["evaluation"]["max_new_tokens"]

    sidecar = {}
    for l in (dz_dir / "sidecar.jsonl").open():
        if l.strip():
            r = json.loads(l)
            sidecar[(r["image_id"], r.get("epsilon"))] = r
    attack = {(r["image_id"], r["epsilon"]): r
              for r in (json.loads(l) for l in lay.attack_results.open() if l.strip())}

    model = build_model(cfg)
    made = 0
    for (iid, eps), meta in sidecar.items():
        if meta.get("error"):
            continue
        sid = sanitize_id(iid)
        atk = attack.get((iid, eps))
        if not atk:
            continue
        clean_answer, target = atk["clean_answer"], atk["target"]

        # clean D5 (preservation)
        cjp = store.clean_job_path(lay, iid, NAME)
        cpng = dz_dir / meta["clean_sanitized"]
        if not is_done(cjp) and cpng.exists():
            ans = model.generate(_load_png(cpng), prompt, mnt)
            atomic_write_json(cjp, {
                "image_id": iid, "defense": NAME, "clean_answer": clean_answer,
                "answer": ans, "preserved": int(normalize_answer(ans) == clean_answer),
                "error": None})
            made += 1

        # adversarial D5 (security)
        djp = store.defense_job_path(lay, iid, eps, NAME)
        apng = dz_dir / meta["adv_sanitized"]
        if not is_done(djp) and apng.exists():
            ans = model.generate(_load_png(apng), prompt, mnt)
            atomic_write_json(djp, {
                "image_id": iid, "class": atk.get("class"), "target": target,
                "epsilon": eps, "defense": NAME, "clean_answer": clean_answer,
                "answer": ans,
                "target_success": int(contains_target(ans, target)),
                "restored": int(normalize_answer(ans) == clean_answer),
                "psnr_vs_clean": meta.get("psnr_vs_clean"),
                "ssim_vs_clean": meta.get("ssim_vs_clean"),
                "mechanism": meta.get("mechanism"), "error": None})
            made += 1

    summary = aggregate(lay)
    log.info("wrote %d D5 job(s). Coverage: %s", made, summary.get("coverage_by_defense"))
    log.info("next: python scripts/run_analysis.py --run %s", lay.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
