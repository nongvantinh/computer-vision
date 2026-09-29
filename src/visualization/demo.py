"""Visual demo (proposal §4.4). Reads a completed run and shows, per example:
clean/adversarial/perturbation/FFT + each defense's answer.

Static mode (no extra deps) writes contact-sheet PNGs:
    python -m src.visualization.demo --run results/<run> --static
Interactive mode needs gradio:
    python -m src.visualization.demo --run results/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..analysis.frequency import fft_magnitude


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _rows_for(run_dir: Path, image_id: str, eps: float) -> dict[str, dict]:
    out = {}
    with (run_dir / "attacks" / "rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["image_id"] == image_id and abs(r["epsilon"] - eps) < 1e-6:
                out[r["defense"]] = r
    return out


def contact_sheet(run_dir: Path, image_id: str, eps: float, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = image_id.replace("/", "__") + f"_eps{eps:.4f}"
    ex = run_dir / "examples"
    clean = _load(ex / f"{stem}_clean.png")
    adv = _load(ex / f"{stem}_adv.png")
    delta = adv - clean
    rows = _rows_for(run_dir, image_id, eps)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(clean); axs[0].set_title("clean")
    axs[1].imshow(adv); axs[1].set_title("adversarial")
    axs[2].imshow(np.clip((delta - delta.min()) / (np.ptp(delta) + 1e-9), 0, 1))
    axs[2].set_title(f"perturbation (Linf={np.max(np.abs(delta)):.3f})")
    mag = np.log1p(fft_magnitude(delta))
    axs[3].imshow(mag, cmap="magma"); axs[3].set_title("FFT(perturbation)")
    for a in axs:
        a.axis("off")
    caption = "  |  ".join(
        f"{d}: {r['prediction']} {'[HIT]' if r['target_success'] else ''}"
        for d, r in sorted(rows.items()))
    fig.suptitle(f"{image_id}  eps={eps:.4f}\n{caption}", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run)

    ids = []
    with (run_dir / "attacks" / "rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            ids.append((r["image_id"], r["epsilon"]))
    ids = sorted(set(ids))

    if args.static:
        for image_id, eps in ids[: args.limit]:
            out = run_dir / "examples" / "sheets" / (
                image_id.replace("/", "__") + f"_eps{eps:.4f}.png")
            contact_sheet(run_dir, image_id, eps, out)
            print("wrote", out)
        return 0

    import gradio as gr  # noqa

    def show(sel):
        image_id, eps = sel.rsplit("@", 1)
        out = run_dir / "examples" / "sheets" / "live.png"
        contact_sheet(run_dir, image_id, float(eps), out)
        return str(out)

    choices = [f"{i}@{e:.4f}" for i, e in ids]
    gr.Interface(fn=show, inputs=gr.Dropdown(choices), outputs="image",
                 title="CDR vs adversarial VLM").launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
