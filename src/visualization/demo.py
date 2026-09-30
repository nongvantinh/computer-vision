"""Visual demo (proposal §4.4, item 20). Reads a completed run under
results/runs/<run_id>/ and shows, per example: clean / adversarial / perturbation /
FFT, plus each defense's answer and whether it restored the clean answer.

Static mode (no extra deps) writes contact-sheet PNGs:
    python -m src.visualization.demo --run results/runs/mvp_001 --static
Interactive mode needs gradio:
    python -m src.visualization.demo --run results/runs/mvp_001

The MVP does not implement defense-aware adaptive attacks; the demo says so, so it
never implies a stronger security guarantee than the experiment establishes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..analysis.frequency import fft_magnitude
from ..experiment.layout import RunLayout
from ..experiment import store

DISCLAIMER = "Defense-aware adaptive attacks are NOT implemented in this MVP."


def _load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _defense_rows(lay: RunLayout, image_id: str, eps: float) -> dict[str, dict]:
    out = {}
    if not lay.defense_results.exists():
        return out
    for line in lay.defense_results.open():
        r = json.loads(line)
        if r.get("error"):
            continue
        if r["image_id"] == image_id and abs(r["epsilon"] - eps) < 1e-6:
            out[r["defense"]] = r
    return out


def contact_sheet(lay: RunLayout, image_id: str, eps: float, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adv = store.load_adv_image(lay, image_id, eps)
    clean_png = lay.examples / f"{image_id.replace('/', '__')}_clean.png"
    clean = _load_png(clean_png) if clean_png.exists() else np.zeros_like(adv)
    delta = adv - clean
    rows = _defense_rows(lay, image_id, eps)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(clean); axs[0].set_title("clean")
    axs[1].imshow(adv); axs[1].set_title("adversarial")
    axs[2].imshow(np.clip((delta - delta.min()) / (np.ptp(delta) + 1e-9), 0, 1))
    axs[2].set_title(f"perturbation (Linf={np.max(np.abs(delta)):.3f})")
    axs[3].imshow(np.log1p(fft_magnitude(delta)), cmap="magma")
    axs[3].set_title("FFT(perturbation)")
    for a in axs:
        a.axis("off")
    caption = "  |  ".join(
        f"{d}: {r['answer']} {'[HIT]' if r['target_success'] else ('[restored]' if r['restored'] else '')}"
        for d, r in sorted(rows.items()))
    fig.suptitle(f"{image_id}  eps={eps:.4f}\n{caption}\n{DISCLAIMER}", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _example_ids(lay: RunLayout) -> list[tuple[str, float]]:
    ids = []
    if lay.attack_results.exists():
        for line in lay.attack_results.open():
            r = json.loads(line)
            ids.append((r["image_id"], r["epsilon"]))
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    lay = RunLayout(Path(args.run))
    ids = _example_ids(lay)

    if args.static:
        for image_id, eps in ids[: args.limit]:
            out = lay.examples / "sheets" / (
                image_id.replace("/", "__") + f"_eps{eps:.4f}.png")
            contact_sheet(lay, image_id, eps, out)
            print("wrote", out)
        return 0

    import gradio as gr  # noqa

    def show(sel):
        image_id, eps = sel.rsplit("@", 1)
        out = lay.examples / "sheets" / "live.png"
        contact_sheet(lay, image_id, float(eps), out)
        return str(out)

    choices = [f"{i}@{e:.4f}" for i, e in ids]
    gr.Interface(fn=show, inputs=gr.Dropdown(choices), outputs="image",
                 title="CDR vs adversarial VLM", description=DISCLAIMER).launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
