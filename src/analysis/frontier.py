"""Security-utility frontier plot (proposal §3.6, §4.3).

Each defense is a point in (clean accuracy retained, robust accuracy = 1 - ASR).
JPEG qualities trace the frontier curve; CDR tools are plotted as separate markers
so we can see whether they lie on or beyond the JPEG curve.
"""
from __future__ import annotations

from pathlib import Path


def plot_frontier(points: dict[str, dict], out_path: str | Path,
                  title: str = "Security-utility frontier"):
    """points: {defense_name: {"clean_acc": float, "robust_acc": float,
                               "group": "jpeg"|"cdr"|"none"}}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    jpeg = {k: v for k, v in points.items() if v.get("group") == "jpeg"}
    if jpeg:
        js = sorted(jpeg.items(), key=lambda kv: kv[1]["clean_acc"])
        ax.plot([v["clean_acc"] for _, v in js],
                [v["robust_acc"] for _, v in js],
                "-o", color="#4c78a8", label="JPEG sweep (frontier)")

    markers = {"cdr": ("*", "#e45756", 240), "none": ("s", "#54a24b", 90)}
    for name, v in points.items():
        g = v.get("group", "cdr")
        if g == "jpeg":
            lbl = None
        else:
            m, c, s = markers.get(g, ("^", "#b279a2", 120))
            ax.scatter(v["clean_acc"], v["robust_acc"], marker=m, color=c, s=s,
                       zorder=5, label=name)
        ax.annotate(name, (v["clean_acc"], v["robust_acc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Clean accuracy retained")
    ax.set_ylabel("Robust accuracy (1 - targeted ASR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
