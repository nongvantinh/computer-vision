#!/usr/bin/env python
"""Analysis for a completed (or partial) run: ASR + CIs, McNemar, Holm, frontier,
mechanism. Reads the derived result files under results/runs/<run_id>/.

    python scripts/run_analysis.py --run results/runs/mvp_001
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logging import get_logger  # noqa: E402
from src.utils.resume import atomic_write_json  # noqa: E402
from src.evaluation.statistics import (mcnemar, bootstrap_ci, paired_diff_ci,  # noqa: E402
                                       holm_bonferroni, clopper_pearson)
from src.evaluation.analysis_v2 import analyze_run  # noqa: E402
from src.analysis.frontier import plot_frontier  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402

log = get_logger("analysis")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="epsilon for the frontier/McNemar family (default: max)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parents[1]
                                          / "configs" / "analysis_plan.yaml"),
                    help="pre-declared analysis plan (families of comparisons)")
    args = ap.parse_args()
    lay = RunLayout(Path(args.run))
    aggregate(lay)  # refresh derived files from jobs first

    defrows = [r for r in _read_jsonl(lay.defense_results) if not r.get("error")]
    cleanrows = [r for r in _read_jsonl(lay.clean_results) if not r.get("error")]
    if not defrows:
        log.warning("no successful defense jobs yet; nothing to analyze")
        return 1

    defenses = sorted({r["defense"] for r in defrows})
    epsilons = sorted({r["epsilon"] for r in defrows})
    eps_plot = args.epsilon if args.epsilon is not None else max(epsilons)

    # preservation (utility): mean of `preserved` per defense on clean images
    preservation = {}
    for d in defenses:
        sub = [r["preserved"] for r in cleanrows if r["defense"] == d]
        if sub:
            p, lo, hi = clopper_pearson(int(np.sum(sub)), len(sub))
            preservation[d] = {"preservation": p, "ci95": [lo, hi], "n": len(sub),
                               "ci_method": "clopper-pearson"}

    # per-image target-success vectors keyed (defense, epsilon), aligned by image_id
    def vec(defense, eps):
        sub = sorted([r for r in defrows
                      if r["defense"] == defense and abs(r["epsilon"] - eps) < 1e-9],
                     key=lambda r: r["image_id"])
        return (np.array([r["target_success"] for r in sub]),
                np.array([r["restored"] for r in sub]),
                [r["image_id"] for r in sub])

    # 1) ASR + restoration + bootstrap CI per (defense, epsilon)
    asr = {}
    for d in defenses:
        for e in epsilons:
            tv, rv, _ = vec(d, e)
            if tv.size == 0:
                continue
            ap_, alo, ahi = clopper_pearson(int(tv.sum()), int(tv.size))
            rp_, rlo, rhi = clopper_pearson(int(rv.sum()), int(rv.size))
            asr[f"{d}@{e:.4f}"] = {"asr": ap_, "asr_ci95": [alo, ahi],
                                   "restoration": rp_, "restoration_ci95": [rlo, rhi],
                                   "n": int(tv.size), "ci_method": "clopper-pearson"}

    # 2) McNemar family at eps_plot: each defense vs D0, and each other vs each JPEG
    jpeg = [d for d in defenses if d.startswith("jpeg")]
    non_jpeg = [d for d in defenses if d not in jpeg and d != "D0"]
    comparisons, pvals, diffs = {}, {}, {}
    d0_t, _, d0_ids = vec("D0", eps_plot)
    for d in defenses:
        if d == "D0":
            continue
        a, _, ids_a = vec(d, eps_plot)
        if d0_ids and ids_a == d0_ids and a.size:
            r = mcnemar(a.astype(bool), d0_t.astype(bool))
            comparisons[f"{d}_vs_D0"] = r.to_dict()
            pvals[f"{d}_vs_D0"] = r.p_value
            diffs[f"{d}_vs_D0"] = paired_diff_ci(a, d0_t, seed=args.seed)
    for c in non_jpeg:
        for j in jpeg:
            a, _, ids_a = vec(c, eps_plot)
            b, _, ids_b = vec(j, eps_plot)
            if ids_a == ids_b and a.size:
                r = mcnemar(a.astype(bool), b.astype(bool))
                comparisons[f"{c}_vs_{j}"] = r.to_dict()
                pvals[f"{c}_vs_{j}"] = r.p_value
                diffs[f"{c}_vs_{j}"] = paired_diff_ci(a, b, seed=args.seed)
    holm = holm_bonferroni(pvals) if pvals else {}

    # 3) trade-off plot at eps_plot: preservation retained vs robust rate (1 - ASR)
    def group(d):
        return "none" if d == "D0" else ("jpeg" if d.startswith("jpeg") else "cdr")
    points = {}
    for d in defenses:
        a, _, _ = vec(d, eps_plot)
        pres = preservation.get(d, {}).get("preservation", float("nan"))
        points[d] = {"clean_acc": float(pres),
                     "robust_acc": float(1 - a.mean()) if a.size else float("nan"),
                     "group": group(d)}
    plot_path = plot_frontier(
        points, lay.root / "plots" / f"tradeoff_eps{eps_plot:.4f}.png",
        title=f"Security-utility trade-off (eps={eps_plot:.4f})")

    # 4) mechanism aggregation from defense rows
    mech = defaultdict(list)
    mech_legacy = defaultdict(list)
    for r in defrows:
        m = r.get("mechanism")
        if not m:
            continue
        # rows written by the old runner (defended_adv - clean) lack "post_defense";
        # keep the two formats apart so one never poisons the other's mean
        (mech if "post_defense" in m else mech_legacy)[r["defense"]].append(m)

    def _mean(ms, *path, default=float("nan")):
        vals = []
        for m in ms:
            v = m
            for k in path:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            vals.append(v if isinstance(v, (int, float)) else default)
        return float(np.mean(vals)) if vals else default

    mech_summary = {}
    for d, ms in mech.items():
        if ms:   # defense-controlled format (final runs)
            mech_summary[d] = {
                "raw_highfreq_frac_mean": _mean(ms, "raw", "highfreq_frac"),
                "post_highfreq_frac_mean": _mean(ms, "post_defense", "highfreq_frac"),
                "raw_energy_mean": _mean(ms, "raw", "energy"),
                "post_energy_mean": _mean(ms, "post_defense", "energy"),
                "post_linf_mean": _mean(ms, "post_defense", "linf"),
                "post_l2_mean": _mean(ms, "post_defense", "l2"),
                "energy_ratio_mean": _mean(ms, "energy_ratio"),
                "n": len(ms)}
    for d, ms in mech_legacy.items():     # legacy confounded format (pilot rows)
        mech_summary.setdefault(d, {})["legacy"] = {
            "residual_highfreq_frac_mean": _mean(ms, "residual_highfreq_frac"),
            "residual_energy_mean": _mean(ms, "residual_energy"),
            "n": len(ms), "note": "legacy confounded metric (defended_adv - clean)"}

    stats = {"epsilons": epsilons, "eps_plot": eps_plot,
             "preservation": preservation, "asr": asr,
             "mcnemar": comparisons,
             "paired_diff_ci": {k: list(v) for k, v in diffs.items()},
             "holm_bonferroni": holm, "mechanism": mech_summary,
             "tradeoff_plot": str(plot_path)}
    atomic_write_json(lay.statistics, stats)

    # v2: plan-driven families, exact tests, all epsilons, reproducible from rows
    import yaml
    plan = yaml.safe_load(Path(args.plan).read_text())
    cfg = json.loads(lay.config.read_text()) if lay.config.exists() else {}
    meta = dict(cfg.get("meta", {}))
    meta.update({"run_dir": lay.root.name, "plan_file": Path(args.plan).name,
                 "n_defense_rows": len(defrows), "n_clean_rows": len(cleanrows)})
    v2 = analyze_run(defrows, cleanrows, plan, meta)
    atomic_write_json(lay.root / "analysis_v2.json", v2)
    log.info("wrote %s (primary family %d, exploratory %d)",
             lay.root / "analysis_v2.json", v2["family_sizes"]["primary"],
             v2["family_sizes"]["exploratory"])
    from src.analysis.figures import all_figures
    made = all_figures(stats, lay.root / "plots")
    log.info("wrote %s, %s, figures: %s", lay.statistics, plot_path, list(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
