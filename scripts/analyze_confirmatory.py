#!/usr/bin/env python
"""Confirmatory N = 200 analysis: the six pre-registered tests (configs/analysis_plan.yaml).

    python scripts/analyze_confirmatory.py --runs results/runs --controls results/runs/controls_v1 \
        --out results/confirmatory

Adaptive rows come from conf_jpeg50_e{16,08} and conf_dz_ll_e{16,08}; the oblivious condition is
`controls_v1` (the stored non-adaptive arrays scored on the stored pixels, see the screen plan,
section 2). Every condition is scored through its real defense. Holm runs over the six tests.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.adaptive_compare import condition_vector, run_confirmatory  # noqa: E402
from src.evaluation.statistics import clopper_pearson  # noqa: E402

EPS = {"e16": 0.0627450980, "e08": 0.0313725490}
RUNS = {"jpeg50": "conf_jpeg50_", "dangerzone_ll": "conf_dz_ll_"}


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--controls", default="results/runs/controls_v1")
    ap.add_argument("--plan", default="configs/analysis_plan.yaml")
    ap.add_argument("--out", default="results/confirmatory")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = yaml.safe_load(Path(a.plan).read_text())
    non = rows(Path(a.controls) / "defense_results.jsonl")
    cond, defense_of = {}, {}
    for d, prefix in RUNS.items():
        ada = []
        for tag in EPS:
            ada += [r for r in rows(Path(a.runs) / f"{prefix}{tag}" / "defense_results.jsonl")
                    if r["defense"] == d]
        cond[f"ada_{d}"], defense_of[f"ada_{d}"] = ada, d
        cond[f"obl_{d}"], defense_of[f"obl_{d}"] = non, d
    res = run_confirmatory(plan, cond, defense_of)
    (out / "tests.json").write_text(json.dumps(res, indent=1))
    rates = []
    for label, rs in cond.items():
        for eps in EPS.values():
            vec = condition_vector(rs, defense_of[label], eps)
            k, n = sum(vec.values()), len(vec)
            p, lo, hi = clopper_pearson(k, n)
            rates.append({"condition": label, "defense": defense_of[label], "epsilon": eps,
                          "k": k, "n": n, "rate": p, "ci95": [lo, hi]})
    (out / "rates.json").write_text(json.dumps(rates, indent=1))
    with (out / "rates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "epsilon", "k", "n", "rate", "ci_lo", "ci_hi"])
        for r in rates:
            w.writerow([r["condition"], f"{r['epsilon']:.4f}", r["k"], r["n"],
                        f"{r['rate']:.4f}", f"{r['ci95'][0]:.4f}", f"{r['ci95'][1]:.4f}"])
    for t in res["tests"]:
        print(f"{t['id']:62s} {t['a_rate']:.3f} vs {t['b_rate']:.3f}  diff {t['diff_a_minus_b']:+.3f} "
              f"[{t['diff_ci95'][0]:+.3f},{t['diff_ci95'][1]:+.3f}] b/c {t['b_only']}/{t['c_only']} "
              f"p_adj {t['p_adjusted']:.2g} reject {t['reject_h0']}")
    print("skipped:", res["skipped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
