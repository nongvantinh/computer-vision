#!/usr/bin/env python
"""Stage 1 of the decoupled Dangerzone (D5) evaluation: sanitize only (no model).

Runs on a host where Dangerzone works (rootless Podman). Reads a run directory's
persisted adversarial images (`adv/*.npy`) and clean images (`examples/*_clean.png`),
pushes each through the real Dangerzone tool, and writes the sanitized images plus a
per-image sidecar (perturbation mechanism, PSNR/SSIM, runtime). No GPU or model here.

    python scripts/dangerzone_sanitize.py --run results/runs/pilot_12 [--limit N]

Then move `<run>/dz/` to the GPU host and run scripts/dangerzone_finalize.py, which
generates the model answers on the SAME 4-bit LLaVA so the metric stays consistent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.defenses.dangerzone import DangerzoneDefense  # noqa: E402
from src.defenses.base import array_to_pil  # noqa: E402
from src.analysis.perturbation import analyze_defense_controlled  # noqa: E402
from src.evaluation.metrics import psnr, ssim  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment.store import load_adv_image  # noqa: E402
from src.utils.resume import sanitize_id, atomic_write_json, is_done  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("dz_sanitize")


def _load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dangerzone-cmd", default="dangerzone-cli")
    ap.add_argument("--embed", choices=["jpeg", "lossless"], default="jpeg",
                    help="how the image enters the PDF: 'jpeg' is the legacy baseline "
                         "adapter (contains a JPEG 75), 'lossless' embeds it byte-exact")
    args = ap.parse_args()
    lay = RunLayout(Path(args.run))
    dz_dir = lay.root / ("dz" if args.embed == "jpeg" else "dz_ll")
    dz_dir.mkdir(parents=True, exist_ok=True)
    defense = DangerzoneDefense(cli=args.dangerzone_cmd, embed=args.embed)
    log.info("adapter embed=%s -> %s", args.embed, dz_dir)

    rows = [json.loads(l) for l in lay.attack_results.open() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    seen_clean, sidecar, done, made, failed = set(), [], 0, 0, 0
    for r in rows:
        iid, eps = r["image_id"], r["epsilon"]
        sid = sanitize_id(iid)
        adv_png = dz_dir / f"{sid}__eps{eps:.4f}__adv.png"
        clean_png_out = dz_dir / f"{sid}__clean.png"
        rec = {"image_id": iid, "epsilon": eps, "sid": sid,
               "adv_sanitized": adv_png.name, "clean_sanitized": clean_png_out.name}

        clean = _load_png(lay.examples / f"{sid}_clean.png")
        adv = load_adv_image(lay, iid, eps)
        if adv is None:
            log.warning("no adv image for %s eps=%s; skipping", iid, eps)
            continue
        try:
            # sanitized clean (once per image) — preservation input + mechanism control
            if not is_done(clean_png_out):
                t0 = time.perf_counter()
                dz_clean = defense.sanitize(clean)
                array_to_pil(dz_clean.image).save(clean_png_out)
                rec["clean_runtime_s"] = time.perf_counter() - t0
            defended_clean = _load_png(clean_png_out)
            seen_clean.add(iid)
            # adversarial — security input
            if is_done(adv_png):
                done += 1
                defended_adv = _load_png(adv_png)
            else:
                t0 = time.perf_counter()
                dz_adv = defense.sanitize(adv)
                array_to_pil(dz_adv.image).save(adv_png)
                rec["adv_runtime_s"] = time.perf_counter() - t0
                defended_adv = dz_adv.image
                made += 1
            # defense-controlled mechanism: defended_adv - defended_clean
            rec["mechanism"] = analyze_defense_controlled(
                clean, adv, defended_clean, defended_adv)
            rec["psnr_vs_clean"] = psnr(clean, defended_adv)
            rec["ssim_vs_clean"] = ssim(clean, defended_adv)
            sidecar.append(rec)
        except Exception as e:
            failed += 1
            log.error("Dangerzone failed on %s eps=%s: %s", iid, eps, e)
            sidecar.append({**rec, "error": repr(e)})

    with (dz_dir / "sidecar.jsonl").open("w") as f:
        for rec in sidecar:
            f.write(json.dumps(rec) + "\n")
    log.info("sanitized: %d new, %d already done, %d failed. Wrote %s",
             made, done, failed, dz_dir)
    log.info("next (on the GPU host): move %s to the run dir there, then "
             "python scripts/dangerzone_finalize.py --run <run>", dz_dir)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
