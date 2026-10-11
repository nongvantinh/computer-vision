"""Stopping rule for the staged adaptive attack (docs/phase2_screen_plan.md, rule S1 to S3).

Fixed before any extended run. After each stage (cumulative budget in the schedule) the
attack is scored through the REAL defense and the rule decides whether another stage is
run. The rule reads losses and success counts of the stage just finished; the budget
(epsilon, target, success definition) never changes between stages.

  S1 success      at least `success_stop` of the images succeed through the real defense
  S2 converged    the median defended target loss of the final iterates (from
                  scripts/eval_defended_loss.py, the same estimator at every stage)
                  improved by less than `min_rel_improvement` relative to the
                  previous stage
  S3 max budget   the last budget in the schedule has been run
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEDULE = (200, 500, 1000, 2000)       # cumulative steps; 200 is the existing pilot
SUCCESS_STOP = 0.80
MIN_REL_IMPROVEMENT = 0.10
TAIL = 50                               # steps averaged for the optimization curve figure


def smoothed_loss(job: dict, tail: int = TAIL) -> float | None:
    curve = job.get("loss_curve") or []
    if not curve:
        return job.get("loss_last")
    return float(np.mean(curve[-tail:]))


def median_defended_loss(path: Path, eps: float, tag: str = "adaptive") -> float | None:
    """Median over images of the defended final-iterate loss at one epsilon."""
    d = json.loads(Path(path).read_text())[tag]
    vals = [v for k, v in d.items() if abs(float(k.rsplit("@", 1)[1]) - eps) < 1e-4]
    return float(np.median(vals)) if vals else None


def stage_summary(run_dir: Path, defense: str, eps: float, defended_loss: Path) -> dict:
    run_dir = Path(run_dir)
    jobs = [json.loads(p.read_text()) for p in sorted((run_dir / "jobs" / "attack").glob("*.json"))]
    jobs = [j for j in jobs if abs(float(j["epsilon"]) - eps) < 1e-6]
    rows = [json.loads(l) for l in (run_dir / "defense_results.jsonl").open() if l.strip()]
    rows = [r for r in rows if r["defense"] == defense and not r.get("error")
            and abs(float(r["epsilon"]) - eps) < 1e-6]
    return {"run": run_dir.name, "defense": defense, "epsilon": eps,
            "n_attacks": len(jobs), "n_scored": len(rows),
            "n_success": int(sum(int(r["target_success"]) for r in rows)),
            "cumulative_steps": max((j.get("steps_cumulative", j.get("steps", 0)) for j in jobs),
                                    default=0),
            "median_defended_loss": median_defended_loss(defended_loss, eps)}


def decide(summary: dict, previous_median: float | None, schedule=SCHEDULE,
           success_stop: float = SUCCESS_STOP, min_rel: float = MIN_REL_IMPROVEMENT) -> dict:
    n = summary["n_scored"]
    share = summary["n_success"] / n if n else 0.0
    cur, prev = summary["median_defended_loss"], previous_median
    rel = None if (prev is None or cur is None or prev == 0) else (prev - cur) / prev
    if share >= success_stop:
        action, reason = "stop", "S1 success"
    elif rel is not None and rel < min_rel:
        action, reason = "stop", "S2 converged"
    elif summary["cumulative_steps"] >= max(schedule):
        action, reason = "stop", "S3 max budget"
    else:
        later = [b for b in schedule if b > summary["cumulative_steps"]]
        action, reason = "continue", f"next budget {later[0]}"
    return {"action": action, "reason": reason, "success_share": share,
            "rel_improvement": rel, "summary": summary}
