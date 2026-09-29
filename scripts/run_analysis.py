#!/usr/bin/env python
"""Analysis for a completed run: ASR + CIs, McNemar, Holm, frontier, mechanism.

    python scripts/run_analysis.py --run results/2026-09-29_mvp_001
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logging import get_logger  # noqa: E402
from src.evaluation.statistics import mcnemar, bootstrap_ci, holm_bonferroni  # noqa: E402
from src.analysis.frontier import plot_frontier  # noqa: E402

log = get_logger("analysis")


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "attacks" / "rows.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="epsilon for the frontier plot (default: max)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    run_dir = Path(args.run)
    rows = load_rows(run_dir)
    clean_acc = json.loads((run_dir / "metrics" / "clean_accuracy.json").read_text())

    defenses = sorted({r["defense"] for r in rows})
    epsilons = sorted({r["epsilon"] for r in rows})
    eps_plot = args.epsilon if args.epsilon is not None else max(epsilons)

    # per-image target-success vectors, keyed (defense, epsilon), aligned by image_id
    def vec(defense, eps):
        sub = sorted([r for r in rows if r["defense"] == defense and r["epsilon"] == eps],
                     key=lambda r: r["image_id"])
        return np.array([r["target_success"] for r in sub]), [r["image_id"] for r in sub]

    # 1) ASR + bootstrap CI per (defense, epsilon)
    asr = {}
    for d in defenses:
        for e in epsilons:
            v, _ = vec(d, e)
            point, lo, hi = bootstrap_ci(v, seed=args.seed)
            asr[f"{d}@{e:.4f}"] = {"asr": point, "ci95": [lo, hi], "n": int(v.size)}

    # 2) McNemar family (at eps_plot): each CDR vs each JPEG, and each defense vs D0
    jpeg = [d for d in defenses if d.startswith("jpeg")]
    cdr = [d for d in defenses if d in ("icdr", "dangerzone")]
    comparisons, pvals = {}, {}
    for d in defenses:
        if d == "D0":
            continue
        a, ids_a = vec(d, eps_plot)
        b, ids_b = vec("D0", eps_plot)
        if ids_a == ids_b and a.size:
            r = mcnemar(a.astype(bool), b.astype(bool))
            comparisons[f"{d}_vs_D0"] = r.to_dict(); pvals[f"{d}_vs_D0"] = r.p_value
    for c in cdr:
        for j in jpeg:
            a, ids_a = vec(c, eps_plot)
            b, ids_b = vec(j, eps_plot)
            if ids_a == ids_b and a.size:
                r = mcnemar(a.astype(bool), b.astype(bool))
                comparisons[f"{c}_vs_{j}"] = r.to_dict(); pvals[f"{c}_vs_{j}"] = r.p_value
    holm = holm_bonferroni(pvals) if pvals else {}

    # 3) frontier at eps_plot: clean acc vs robust acc (1 - ASR)
    def group(d):
        return "none" if d == "D0" else ("jpeg" if d.startswith("jpeg") else "cdr")
    points = {}
    for d in defenses:
        a, _ = vec(d, eps_plot)
        points[d] = {"clean_acc": float(clean_acc.get(d, float("nan"))),
                     "robust_acc": float(1 - a.mean()) if a.size else float("nan"),
                     "group": group(d)}
    plot_path = plot_frontier(points, run_dir / "plots" / f"frontier_eps{eps_plot:.4f}.png",
                              title=f"Security-utility frontier (eps={eps_plot:.4f})")

    # 4) mechanism aggregation
    mech = defaultdict(list)
    mpath = run_dir / "metrics" / "perturbation.jsonl"
    if mpath.exists():
        with mpath.open() as f:
            for line in f:
                m = json.loads(line)
                mech[m["defense"]].append(m)
    mech_summary = {}
    for d, ms in mech.items():
        mech_summary[d] = {
            "energy_removed_frac_mean": float(np.mean([m["energy_removed_frac"] for m in ms])),
            "residual_highfreq_frac_mean": float(np.mean([m["residual_highfreq_frac"] for m in ms])),
            "injected_highfreq_frac_mean": float(np.mean([m["injected_highfreq_frac"] for m in ms])),
        }

    summary = {"epsilons": epsilons, "eps_plot": eps_plot, "clean_accuracy": clean_acc,
               "asr": asr, "mcnemar": comparisons, "holm_bonferroni": holm,
               "mechanism": mech_summary, "frontier_plot": str(plot_path)}
    out = run_dir / "statistics" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s and %s", out, plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
