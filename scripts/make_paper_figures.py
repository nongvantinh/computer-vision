#!/usr/bin/env python
"""Figures for the paper from results/confirmatory (run analyze_confirmatory.py first).

    python scripts/make_paper_figures.py --runs results/runs --conf results/confirmatory \
        --out docs/paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EPS = {"e08": 0.0313725490, "e16": 0.0627450980}
COL = {"obl": "#4c78a8", "ada": "#e45756"}
DEF = {"jpeg50": "JPEG 50", "dangerzone_ll": "Dangerzone\n(lossless adapter)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--conf", default="results/confirmatory")
    ap.add_argument("--out", default="docs/paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = json.loads((Path(a.conf) / "rates.json").read_text())
    tests = json.loads((Path(a.conf) / "tests.json").read_text())["tests"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: targeted success through the real defense, oblivious vs adaptive
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, eps in zip(axes, EPS.values()):
        for di, d in enumerate(DEF):
            for ai, kind in enumerate(("obl", "ada")):
                r = next(x for x in rates if x["condition"] == f"{kind}_{d}" and abs(x["epsilon"] - eps) < 1e-3)
                x = di + (ai - 0.5) * 0.36
                ax.bar(x, r["rate"], 0.34, color=COL[kind],
                       label=("non-adaptive attack" if kind == "obl" else "adaptive attack") if di == 0 else None)
                ax.errorbar(x, r["rate"], yerr=[[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]],
                            color="k", capsize=2, lw=0.8)
                ax.text(x, r["ci95"][1] + 0.02, f"{r['k']}/{r['n']}", ha="center", fontsize=7)
        ax.set_xticks(range(len(DEF)), list(DEF.values()))
        ax.set_ylim(0, 1.0)
        ax.set_title(f"$\\epsilon$ = {round(eps * 255)}/255")
    axes[0].set_ylabel("targeted success (95% exact CI)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_confirmatory_asr.pdf")
    plt.close(fig)

    # Fig 2: paired differences with the minimum effect of interest
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    names = {"adaptive_vs_oblivious[jpeg50]": "JPEG 50: adaptive minus non-adaptive",
             "adaptive_vs_oblivious[dangerzone_ll]": "Dangerzone: adaptive minus non-adaptive",
             "cdr_vs_jpeg_adaptive[dangerzone_ll vs jpeg50]": "Dangerzone minus JPEG 50, both adaptive"}
    y, ticks, labels = 0, [], []
    for eps in EPS.values():
        for key, lab in names.items():
            t = next(x for x in tests if x["id"].startswith(key) and abs(x["epsilon"] - eps) < 1e-3)
            d, (lo, hi) = t["diff_a_minus_b"] * 100, [v * 100 for v in t["diff_ci95"]]
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color="#e45756", capsize=2)
            ticks.append(y)
            labels.append(f"{lab} ({round(eps * 255)}/255)")
            y -= 1
        y -= 0.5
    ax.axvline(15, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.text(15.5, 0.3, "minimum effect\nof interest", fontsize=7, color="gray", va="bottom")
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_xlabel("difference in targeted success (points, 95% interval)")
    fig.tight_layout()
    fig.savefig(out / "fig_paired_differences.pdf")
    plt.close(fig)

    # Fig 3: median target loss through the defense during the attack
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    sty = {"conf_jpeg50_e16": ("JPEG 50, 16/255", "#f58518", "-"), "conf_jpeg50_e08": ("JPEG 50, 8/255", "#f58518", "--"),
           "conf_dz_ll_e16": ("Dangerzone, 16/255", "#e45756", "-"), "conf_dz_ll_e08": ("Dangerzone, 8/255", "#e45756", "--")}
    for rid, (lab, c, ls) in sty.items():
        cs = [json.loads(p.read_text()).get("loss_curve") for p in sorted((Path(a.runs) / rid / "jobs" / "attack").glob("*.json"))]
        cs = [c_ for c_ in cs if c_]
        n = min(map(len, cs))
        m = np.median(np.array([c_[:n] for c_ in cs]), axis=0)
        sm = np.convolve(m, np.ones(10) / 10, mode="valid")
        ax.plot(sm, color=c, ls=ls, lw=1.1, label=lab)
    ax.set_xlabel("attack step")
    ax.set_ylabel("target loss (median, 10-step mean)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_loss_curves.pdf")
    plt.close(fig)
    print("wrote", sorted(p.name for p in out.glob("*.pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
