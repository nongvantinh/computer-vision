#!/usr/bin/env python
"""Figures for the adaptive pilot, read straight from the run directories.

    python scripts/plot_pilot.py --baseline results/runs/full_200 --out results/pilot

Panel A: target loss of each pilot image through the defense in the attack loop, for the
non-adaptive attack (stored baseline image) and the adaptive attack (defended_loss_*.json).
Panel B: targeted success on the same images, non-adaptive vs adaptive, through every
defense the pilot run was scored with. Intervals are exact (Clopper-Pearson).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.statistics import clopper_pearson  # noqa: E402

EPS = 16 / 255
RUNS = [("pilot_adaptive_jpeg50", "jpeg50", "defended_loss_jpeg50.json", "attack through JPEG 50"),
        ("pilot_adaptive_dz_ll", "dangerzone_ll", "defended_loss_dangerzone_ll.json",
         "attack through Dangerzone (lossless adapter)"),
        ("pilot_adaptive_dz", "dangerzone", "defended_loss_dangerzone.json",
         "attack through Dangerzone (legacy adapter)")]


def rows(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "defense_results.jsonl").open() if l.strip()]


def rate(rs: list[dict], defense: str, ids: set) -> tuple[int, int]:
    sel = [r for r in rs if r["defense"] == defense and r["image_id"] in ids and not r.get("error")
           and abs(float(r["epsilon"]) - EPS) < 1e-3]
    return sum(int(r["target_success"]) for r in sel), len(sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--baseline", default="results/runs/full_200")
    ap.add_argument("--out", default="results/pilot")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = rows(Path(a.baseline))

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (run, defense, dl, title) in zip(axs, RUNS):
        d = json.loads((runs / run / dl).read_text())
        keys = sorted(d["adaptive"])
        for k in keys:
            ax.plot([0, 1], [d["oblivious"][k], d["adaptive"][k]], color="0.55", lw=1, marker="o", ms=3)
        ax.set_xticks([0, 1], ["non-adaptive", "adaptive"])
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.3, 1.3)
    axs[0].set_ylabel(f"target loss through the defense\n(eps = 16/255, n = 10)")
    fig.tight_layout()
    fig.savefig(out / "pilot_defended_loss.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    summary = {}
    for ax, (run, defense, dl, title) in zip(axs, RUNS):
        ada = rows(runs / run)
        ids = {r["image_id"] for r in ada}
        names = sorted({r["defense"] for r in ada}, key=lambda n: (n != "D0", n))
        for j, n in enumerate(names):
            for off, label, rs, col in ((-0.2, "non-adaptive", base, "#4c78a8"), (0.2, "adaptive", ada, "#e45756")):
                k, m = rate(rs, n, ids)
                p, lo, hi = clopper_pearson(k, m, 0.05)
                ax.bar(j + off, p, 0.38, color=col, label=label if j == 0 else None)
                ax.errorbar(j + off, p, yerr=[[p - lo], [hi - p]], color="k", capsize=2, lw=1)
                ax.text(j + off, min(hi, 1) + 0.02, f"{k}/{m}", ha="center", fontsize=7)
                summary[f"{run}|{n}|{label}"] = {"k": k, "n": m, "rate": p, "ci95": [lo, hi]}
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_title(title + "\n(scored through the defenses shown)", fontsize=8)
        ax.set_ylim(0, 1.15)
    axs[0].set_ylabel("targeted success (95% exact CI)")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "pilot_success.png", dpi=160)
    (out / "pilot_success.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
