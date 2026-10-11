#!/usr/bin/env python
"""Apply the staged-attack stopping rule (src/evaluation/staging.py) to a finished stage.

    python scripts/stage_decision.py --run results/runs/pilot_jpeg50_s500 --defense jpeg50 \
        --eps 0.0627450980 --previous-loss results/runs/pilot_adaptive_jpeg50/defended_loss_jpeg50.json \
        --expect 10 --out results/runs/pilot_jpeg50_s500/stage_decision.json

Exit 0 and writes the decision; exits 3 if the stage is incomplete (fewer attacks than
--expect), so a time-limited session never decides on partial data.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import staging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--previous-loss", required=True, help="defended_loss_*.json of the previous stage")
    ap.add_argument("--expect", type=int, required=True, help="number of images in the stage")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    summ = staging.stage_summary(run, a.defense, a.eps, run / f"defended_loss_{a.defense}.json")
    if summ["n_attacks"] < a.expect or summ["n_scored"] < a.expect:
        print(f"stage incomplete: {summ['n_attacks']} attacks, {summ['n_scored']} scored of {a.expect}")
        return 3
    prev = staging.median_defended_loss(Path(a.previous_loss), a.eps)
    d = staging.decide(summ, prev)
    d["previous_median_defended_loss"] = prev
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(json.dumps({k: d[k] for k in ("action", "reason", "success_share", "rel_improvement")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
