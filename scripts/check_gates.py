#!/usr/bin/env python
"""Evaluate validity gates G1 to G5 for an adaptive run (docs/phase2_preregistration.md).

    python scripts/check_gates.py --adaptive-run results/runs/pilot_jpeg50 --steps 200 \
        --images 10 --eps-count 1 [--surrogate-report configs/surrogates/dangerzone_lossless.json] \
        [--defended-loss defended_loss_jpeg50.json] [--controls-run RUN --baseline-run RUN]

Gates never read attack success. A gate whose input is not supplied is reported as
"not evaluated", never as passed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import gates  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402


def _jobs(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-run", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--images", type=int, required=True)
    ap.add_argument("--eps-count", type=int, required=True)
    ap.add_argument("--surrogate-report", default=None)
    ap.add_argument("--defended-loss", default=None)
    ap.add_argument("--controls-run", default=None)
    ap.add_argument("--baseline-run", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lay = RunLayout(Path(args.adaptive_run))
    aggregate(lay)
    attack = _jobs(lay.attack_jobs)
    rows = [json.loads(l) for l in lay.defense_results.open() if l.strip()]
    res = [gates.g1_defense_in_loop(attack, args.steps),
           gates.g4_no_silent_loss(attack, rows, args.images, args.eps_count)]
    if args.surrogate_report:
        res.append(gates.g2_surrogate_fidelity(json.loads(Path(args.surrogate_report).read_text())))
    else:
        res.append({"gate": "G2", "passed": None, "detail": "not evaluated"})
    if args.defended_loss:
        d = json.loads(Path(args.defended_loss).read_text())
        res.append(gates.g3_attack_optimizes(d["adaptive"], d["oblivious"]))
    else:
        res.append({"gate": "G3", "passed": None, "detail": "not evaluated"})
    if args.controls_run and args.baseline_run:
        def rows_of(p):
            l = RunLayout(Path(p))
            return [json.loads(x) for x in l.defense_results.open() if x.strip()]
        res.append(gates.g5_baseline_reproduction(rows_of(args.controls_run), rows_of(args.baseline_run)))
    else:
        res.append({"gate": "G5", "passed": None, "detail": "not evaluated"})
    res.sort(key=lambda g: g["gate"])
    for g in res:
        print(f"{g['gate']}: {'PASS' if g['passed'] else ('not evaluated' if g['passed'] is None else 'FAIL')}"
              f"  {json.dumps(g['detail'])[:200]}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0 if all(g["passed"] is not False for g in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
