#!/usr/bin/env python
"""MVP experiment: resumable, persistent, time-budgeted (proposal §3.6, §11).

    # first launch
    python scripts/run_experiment.py --config configs/mvp.yaml --run-id mvp_001
    # continue after a kill / VM reclaim (skips finished jobs)
    python scripts/run_experiment.py --config configs/mvp.yaml --run-id mvp_001 --resume

Results live under results/runs/<run_id>/ and are self-describing (config,
environment, git commit, dataset manifest, model/defense info). The attack is
crafted once per (image, epsilon), persisted as a lossless .npy, and read back by
every defense job, so an interrupted session loses at most one job and adding a
defense never recomputes the attack. `--max-hours` stops cleanly before a runtime
is reclaimed. Pilot knobs (`--limit/--steps/--epsilon/--defenses`) narrow scope
without changing the scientific definitions.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402
from src.evaluation.runner import build_model, build_defenses, build_dataset  # noqa: E402
from src.experiment.layout import create_run, load_run, RUNS_ROOT  # noqa: E402
from src.experiment.run import run_experiment  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.experiment import store  # noqa: E402
from src.utils.resume import atomic_write_json  # noqa: E402

log = get_logger("run_experiment")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--attack", default="configs/attack.yaml")
    ap.add_argument("--run-id", required=True, help="stable id under results/runs/")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing run (skips finished jobs)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="session time budget; stops cleanly between jobs")
    ap.add_argument("--limit", type=int, default=None, help="first N images (pilot)")
    ap.add_argument("--steps", type=int, default=None, help="override PGD steps (pilot)")
    ap.add_argument("--epsilon", type=float, action="append", default=None,
                    help="override epsilon(s); repeatable (pilot)")
    ap.add_argument("--defenses", default=None,
                    help="comma-separated defense override, e.g. D0,jpeg90,dangerzone")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = load_config(args.attack)
    if args.steps is not None:
        acfg["steps"] = args.steps
    if args.epsilon:
        acfg["epsilon"] = args.epsilon
    if args.defenses:
        cfg["defenses"] = [d.strip() for d in args.defenses.split(",") if d.strip()]
    seed_everything(cfg["seed"])

    if args.resume:
        lay = load_run(args.run_id)
        create_run(args.run_id, cfg, acfg, cfg["dataset"]["manifest"])  # refresh env
        log.info("resuming run %s (%s)", args.run_id, lay.root)
    else:
        lay = create_run(args.run_id, cfg, acfg, cfg["dataset"]["manifest"])
        log.info("created run %s (%s)", args.run_id, lay.root)

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    log.info("deadline: %s", f"{args.max_hours}h" if deadline else "none")

    model = build_model(cfg)
    ds = build_dataset(cfg)
    defenses = build_defenses(cfg)
    unavailable = getattr(build_defenses, "unavailable", {})
    if unavailable:
        atomic_write_json(lay.root / "defenses_unavailable.json", unavailable)
    size = cfg["dataset"]["image_size"]
    mnt = cfg["evaluation"]["max_new_tokens"]

    log.info("running %d images x %d eps x %d defenses (limit=%s)",
             len(ds), len(acfg["epsilon"]), len(defenses), args.limit)
    run_experiment(model, ds, defenses, lay,
                   epsilons=acfg["epsilon"], pgd_steps=acfg["steps"],
                   pgd_step_size=acfg["step_size"],
                   pgd_random_start=acfg["random_start"], seed=cfg["seed"],
                   image_size=size, max_new_tokens=mnt, deadline=deadline,
                   limit=args.limit)

    summary = aggregate(lay)
    log.info("coverage: %s", summary)
    log.info("next: python scripts/run_analysis.py --run %s", lay.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
