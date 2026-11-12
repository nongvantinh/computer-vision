#!/usr/bin/env python
"""Directional check of the Dangerzone surrogate along the actual attack directions.

    python scripts/surrogate_directional_check.py --run results/runs/conf_dz_ll_e16 --out results/surrogate_check

The attack follows the Jacobian of the surrogate. This compares, for every image, the response of
the REAL tool to the adversarial perturbation with the response the surrogate predicts:

    d_real = real(adv) - real(clean)          from the sanitized PNGs in <run>/dz_ll
    d_sur  = S(adv) - S(clean)                S = the fitted affine map, before clamping and rounding
    d_id   = adv - clean                      baseline: a defense that does nothing

Reports cosine similarity and relative L2 error between d_real and each prediction, per image and
summarized. This validates the linear response over the finite perturbation actually used, which
is what the attack exploits. It does not test the surrogate where the real tool is nonlinear
(for example under rounding at very small perturbations) and says nothing about directions the
attack never visits. Read-only with respect to the run directory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import store  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.utils.resume import sanitize_id  # noqa: E402


def load_png(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def linear_part(d, x):
    """The surrogate's affine map without clamp or rounding. x: HxWx3 in [0,1]."""
    B, C, b = d["B"].astype(np.float32), d["C"].astype(np.float32), d["b"].astype(np.float32)
    z = np.stack([B @ x[:, :, c] @ B.T for c in range(3)], axis=-1)   # B x B^T per channel
    return z @ C.T + b.reshape(1, 1, 3)


def stats(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    return cos, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eps", type=float, required=True, help="epsilon of the run, e.g. 0.0627450980")
    ap.add_argument("--surrogate", default="configs/surrogates/dangerzone_lossless.npz")
    ap.add_argument("--dz-dir", default=None)
    ap.add_argument("--out", default="results/surrogate_check")
    a = ap.parse_args()
    lay = RunLayout(Path(a.run))
    dz = Path(a.dz_dir) if a.dz_dir else lay.root / "dz_ll"
    d = np.load(a.surrogate)
    rows = []
    for npy in sorted(lay.adv.glob("*.npy")):
        stem = npy.name[:-4]                                   # <id>__eps0.0627
        img = stem.rsplit("__eps", 1)[0]
        adv_png, clean_png = dz / f"{stem}__adv.png", dz / f"{img}__clean.png"
        clean_ex = lay.examples / f"{img}_clean.png"
        if not (adv_png.exists() and clean_png.exists() and clean_ex.exists()):
            continue
        adv = np.load(npy).astype(np.float32)
        clean = load_png(clean_ex)
        d_real = load_png(adv_png) - load_png(clean_png)
        d_sur = linear_part(d, adv) - linear_part(d, clean)
        d_id = adv - clean
        c_s, r_s = stats(d_real, d_sur)
        c_i, r_i = stats(d_real, d_id)
        rows.append({"image": img, "cos_surrogate": c_s, "relerr_surrogate": r_s, "cos_identity": c_i,
                     "relerr_identity": r_i, "norm_real": float(np.linalg.norm(d_real)),
                     "norm_surrogate": float(np.linalg.norm(d_sur)), "norm_input": float(np.linalg.norm(d_id))})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = lambda k, p: float(np.percentile([r[k] for r in rows], p))
    summ = {"run": lay.root.name, "n": len(rows),
            **{f"{k}_{n}": q(k, p) for k in ("cos_surrogate", "relerr_surrogate", "cos_identity", "relerr_identity")
               for n, p in (("median", 50), ("p05", 5), ("p95", 95))}}
    (out / f"{lay.root.name}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
