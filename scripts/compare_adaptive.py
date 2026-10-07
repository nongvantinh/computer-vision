#!/usr/bin/env python
"""Confirmatory adaptive-vs-oblivious comparison across run directories.

    python scripts/compare_adaptive.py \
        --cond obl_jpeg50=results/runs/full_200:jpeg50 \
        --cond ada_jpeg50=results/runs/adaptive_jpeg50:jpeg50 \
        --cond obl_dangerzone_ll=results/runs/controls_v1:dangerzone_ll \
        --cond ada_dangerzone_ll=results/runs/adaptive_dz_ll:dangerzone_ll \
        --out results/confirmatory_adaptive.json

LABEL is ada_<defense> or obl_<defense> (see src/evaluation/adaptive_compare.py). Runs
are read, never modified: existing derived tables are used as they are.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.adaptive_compare import run_confirmatory  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.utils.resume import atomic_write_json  # noqa: E402


def load_rows(run_dir: str) -> list[dict]:
    lay = RunLayout(Path(run_dir))
    if not lay.defense_results.exists():
        aggregate(lay)
    return [json.loads(l) for l in lay.defense_results.open() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", required=True,
                    help="LABEL=RUN_DIR:DEFENSE (repeatable)")
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parents[1]
                                          / "configs" / "analysis_plan.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    conditions, defense_of, sources = {}, {}, {}
    for spec in args.cond:
        label, rest = spec.split("=", 1)
        run_dir, defense = rest.rsplit(":", 1)
        conditions[label] = load_rows(run_dir)
        defense_of[label] = defense
        sources[label] = {"run_dir": run_dir, "defense": defense,
                          "n_rows": len(conditions[label])}
    plan = yaml.safe_load(Path(args.plan).read_text())
    out = run_confirmatory(plan, conditions, defense_of)
    out["sources"] = sources
    text = json.dumps(out, indent=1)
    if args.out:
        atomic_write_json(Path(args.out), out)
    for t in out["tests"]:
        print(f"{t['id']:70s} diff={t['diff_a_minus_b']:+.3f} "
              f"[{t['diff_ci95'][0]:+.3f},{t['diff_ci95'][1]:+.3f}] p_adj={t['p_adjusted']:.2e} "
              f"reject={t['reject_h0']}")
    for s in out["skipped"]:
        print("skipped:", s["id"], "missing", s["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
