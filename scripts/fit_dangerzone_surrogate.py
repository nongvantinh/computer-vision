#!/usr/bin/env python
"""Fit and validate a differentiable surrogate of the Dangerzone adapter.

The adapter chain is: [JPEG 75 if embed=jpeg] -> poppler raster at 150 dpi (a 1.5625x
upscale) -> Dangerzone -> 150 dpi raster -> Lanczos downscale to the input size. Apart
from the optional JPEG, every stage is a resampling, so the whole chain is close to a
separable linear operator

    y = C( B x B^T ) + b          B: (H x H) banded, C: 3x3 colour matrix, b: bias

which is differentiable by construction. B, C and b are FITTED to real (input, output)
pairs produced by the real tool, then validated on held-out pairs, including
adversarial inputs. Nothing here replaces the real defense: the final attack success
is always scored through the real Dangerzone pipeline.

    python scripts/fit_dangerzone_surrogate.py --run results/runs/full_200 --variant lossless
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.defenses.base import array_to_pil, pil_to_array  # noqa: E402
from src.defenses.jpeg import JpegDefense  # noqa: E402
from src.evaluation.metrics import psnr, ssim  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment.store import load_adv_image  # noqa: E402
from src.utils.resume import sanitize_id  # noqa: E402


def load_pairs(lay: RunLayout, variant: str):
    """[(kind, image_id, eps, input_to_R, real_output)] as float32 HxWx3 in [0,1].

    `input_to_R` is what enters the resampling stage: the 8-bit image for the lossless
    adapter, or its JPEG-75 round trip for the legacy adapter (the JPEG is handled by
    the JPEG surrogate, not fitted here).
    """
    dz = lay.root / ("dz" if variant == "jpeg" else "dz_ll")
    pairs = []
    jpeg = JpegDefense(75)

    def feed(x):
        x8 = pil_to_array(array_to_pil(x))                  # the 8-bit image
        return jpeg.sanitize(x8).image if variant == "jpeg" else x8

    for sc in sorted((dz).glob("*__clean.png")):
        sid = sc.name[: -len("__clean.png")]
        cl = lay.examples / f"{sid}_clean.png"
        if cl.exists():
            x = pil_to_array(Image.open(cl))
            pairs.append(("clean", sid, None, feed(x), pil_to_array(Image.open(sc))))
    manifest = [json.loads(l) for l in lay.attack_results.open() if l.strip()]
    for r in manifest:
        sid = sanitize_id(r["image_id"])
        p = dz / f"{sid}__eps{r['epsilon']:.4f}__adv.png"
        adv = load_adv_image(lay, r["image_id"], r["epsilon"])
        if p.exists() and adv is not None:
            pairs.append(("adv", sid, r["epsilon"], feed(adv), pil_to_array(Image.open(p))))
    return pairs


def fit(pairs, band: int, steps: int, seed: int):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = pairs[0][3].shape[0]
    X = torch.from_numpy(np.stack([p[3] for p in pairs])).permute(0, 3, 1, 2).contiguous()
    Y = torch.from_numpy(np.stack([p[4] for p in pairs])).permute(0, 3, 1, 2).contiguous()
    idx = np.arange(len(pairs))
    # hold out whole images (clean + all its adversarial versions) so no image leaks
    ids = sorted({p[1] for p in pairs})
    held = set(rng.choice(ids, size=max(1, len(ids) // 4), replace=False))
    tr = np.array([i for i in idx if pairs[i][1] not in held])
    te = np.array([i for i in idx if pairs[i][1] in held])

    mask = torch.zeros(n, n)
    for i in range(n):
        mask[i, max(0, i - band): i + band + 1] = 1.0
    W = torch.eye(n).clone().requires_grad_(True)           # B starts at the identity
    C = torch.eye(3).clone().requires_grad_(True)
    b = torch.zeros(3).requires_grad_(True)
    opt = torch.optim.Adam([W, C, b], lr=2e-3)

    def forward(x):
        B = W * mask
        z = B @ x @ B.T
        return torch.einsum("cd,ndhw->nchw", C, z) + b.view(1, 3, 1, 1)

    for step in range(steps):
        sel = torch.from_numpy(rng.choice(tr, size=min(16, len(tr)), replace=False))
        loss = ((forward(X[sel]) - Y[sel]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"step {step:5d} mse {float(loss):.3e}")
    B = (W * mask).detach().numpy().astype(np.float32)
    return B, C.detach().numpy(), b.detach().numpy(), tr, te, forward


def evaluate(pairs, idx, forward, label):
    import torch
    rows = {"clean": [], "adv": []}
    for i in idx:
        kind, _, eps, x, y = pairs[i]
        with torch.no_grad():
            s = forward(torch.from_numpy(x).permute(2, 0, 1)[None])
        s8 = np.clip(np.rint(s[0].permute(1, 2, 0).numpy() * 255), 0, 255) / 255.0
        rows[kind].append((psnr(s8, y), ssim(s8, y), float(np.abs(s8 - y).max() * 255),
                           psnr(x, y)))
    out = {}
    for kind, v in rows.items():
        if v:
            a = np.array(v)
            out[kind] = {"n": len(v), "psnr_db_mean": float(a[:, 0].mean()),
                         "psnr_db_min": float(a[:, 0].min()),
                         "ssim_mean": float(a[:, 1].mean()),
                         "max_abs_diff_levels_mean": float(a[:, 2].mean()),
                         "psnr_input_vs_real_db_mean": float(a[:, 3].mean())}
    print(label, json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["jpeg", "lossless"], required=True)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lay = RunLayout(Path(args.run))
    pairs = load_pairs(lay, args.variant)
    print(f"{len(pairs)} (input, real-Dangerzone-output) pairs, variant={args.variant}")
    B, C, b, tr, te, forward = fit(pairs, args.band, args.steps, args.seed)
    rep = {"variant": args.variant, "band": args.band, "steps": args.steps,
           "seed": args.seed, "n_pairs": len(pairs), "n_train": int(len(tr)),
           "n_heldout": int(len(te)),
           "train": evaluate(pairs, tr, forward, "train:"),
           "heldout": evaluate(pairs, te, forward, "heldout:")}
    out = Path(args.out) if args.out else (Path(__file__).resolve().parents[1] / "configs"
                                           / "surrogates" / f"dangerzone_{args.variant}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, B=B, C=C, b=b)
    (out.with_suffix(".json")).write_text(json.dumps(rep, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
