#!/usr/bin/env python
"""Regenerate publication figures from a run's statistics.json (item 21).

    python scripts/make_figures.py --run results/runs/mvp_001

Figures come only from the machine-readable statistics file, so they never contain
hand-entered numbers. `run_analysis.py` also calls this at the end of a run; this
script is for regenerating figures without recomputing the statistics.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.figures import all_figures  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    lay = RunLayout(Path(args.run))
    if not lay.statistics.exists():
        print(f"no statistics.json at {lay.statistics}; run run_analysis.py first")
        return 1
    stats = json.loads(lay.statistics.read_text())
    made = all_figures(stats, lay.root / "plots")
    for name, paths in made.items():
        print(f"{name}: {paths}")
    if not made:
        print("no figures produced (statistics.json has no asr/preservation/mechanism)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
