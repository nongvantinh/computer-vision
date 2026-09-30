"""Publication figures from a run's statistics.json (proposal §4, item 21).

Every figure is generated from the machine-readable statistics file, never from
hand-entered numbers, and saved as both PNG and PDF. Missing data yields a skipped
figure, not a fabricated one. The security-utility trade-off plot is produced
separately by `analysis.frontier.plot_frontier`; these are the bar charts.
"""
from __future__ import annotations

from pathlib import Path


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)
    return paths


def asr_by_defense(stats: dict, out_dir: Path) -> list[Path]:
    """Grouped bar chart of targeted ASR per defense, one group of bars per epsilon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    asr = stats.get("asr", {})
    if not asr:
        return []
    defenses, epsilons = [], []
    for key in asr:
        d, e = key.rsplit("@", 1)
        defenses.append(d)
        epsilons.append(e)
    defenses = sorted(set(defenses))
    epsilons = sorted(set(epsilons))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(defenses)), 4.5))
    width = 0.8 / max(1, len(epsilons))
    x = np.arange(len(defenses))
    for i, e in enumerate(epsilons):
        vals = [asr.get(f"{d}@{e}", {}).get("asr", float("nan")) for d in defenses]
        los = [asr.get(f"{d}@{e}", {}).get("asr_ci95", [None, None])[0] for d in defenses]
        his = [asr.get(f"{d}@{e}", {}).get("asr_ci95", [None, None])[1] for d in defenses]
        yerr = None
        if all(l is not None for l in los):
            yerr = [[max(0, v - l) for v, l in zip(vals, los)],
                    [max(0, h - v) for v, h in zip(vals, his)]]
        ax.bar(x + i * width, vals, width, yerr=yerr, capsize=3,
               label=f"eps={float(e):.4f}")
    ax.set_xticks(x + width * (len(epsilons) - 1) / 2)
    ax.set_xticklabels(defenses, rotation=20, ha="right")
    ax.set_ylabel("Targeted ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Targeted attack success by defense")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig_asr_by_defense")


def preservation_by_defense(stats: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pres = stats.get("preservation", {})
    if not pres:
        return []
    defenses = sorted(pres)
    vals = [pres[d]["preservation"] for d in defenses]
    cis = [pres[d].get("ci95", [None, None]) for d in defenses]
    yerr = None
    if all(c[0] is not None for c in cis):
        yerr = [[max(0, v - c[0]) for v, c in zip(vals, cis)],
                [max(0, c[1] - v) for v, c in zip(vals, cis)]]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(defenses)), 4.5))
    ax.bar(range(len(defenses)), vals, yerr=yerr, capsize=3, color="#54a24b")
    ax.set_xticks(range(len(defenses)))
    ax.set_xticklabels(defenses, rotation=20, ha="right")
    ax.set_ylabel("Preservation (clean answer kept)")
    ax.set_ylim(0, 1)
    ax.set_title("Utility: clean-answer preservation by defense")
    return _save(fig, out_dir, "fig_preservation")


def mechanism_by_defense(stats: dict, out_dir: Path) -> list[Path]:
    """Perturbation removal and residual high-frequency fraction per defense."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mech = stats.get("mechanism", {})
    if not mech:
        return []
    defenses = sorted(mech)
    removed = [mech[d].get("energy_removed_frac_mean", float("nan")) for d in defenses]
    res_hf = [mech[d].get("residual_highfreq_frac_mean", float("nan")) for d in defenses]
    x = np.arange(len(defenses))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(defenses)), 4.5))
    ax.bar(x - 0.2, removed, 0.4, label="energy removed (frac)", color="#4c78a8")
    ax.bar(x + 0.2, res_hf, 0.4, label="residual high-freq (frac)", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, rotation=20, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Perturbation removal (mechanism evidence, not causal)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig_mechanism")


def all_figures(stats: dict, out_dir: Path) -> dict:
    """Generate every available figure; return {name: [paths]} for the ones made."""
    out_dir = Path(out_dir)
    made = {}
    for name, fn in (("asr", asr_by_defense), ("preservation", preservation_by_defense),
                     ("mechanism", mechanism_by_defense)):
        paths = fn(stats, out_dir)
        if paths:
            made[name] = [str(p) for p in paths]
    return made
