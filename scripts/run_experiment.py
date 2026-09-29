#!/usr/bin/env python
"""Full 200-image MVP experiment (proposal §3.6, §11) — resumable & time-budgeted.

    python scripts/run_experiment.py --out results/mvp --max-hours 3

Resumable: writes one job file per (image) clean eval and per (image, epsilon)
attack; a rerun skips finished jobs, so a Colab session killed by a usage limit
continues where it stopped. `--max-hours` stops cleanly before the runtime is
reclaimed. Derived rows.jsonl/metrics are rebuilt from the job files each session.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, snapshot_configs  # noqa: E402
from src.utils.logging import get_logger, seed_everything, write_manifest  # noqa: E402
from src.evaluation.runner import (build_model, build_defenses, build_dataset,  # noqa: E402
                                   run_clean, run_full, aggregate)

log = get_logger("run_experiment")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--attack", default="configs/attack.yaml")
    ap.add_argument("--out", default="results/mvp",
                    help="stable, resumable output dir (reused across sessions)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="session time budget; stops cleanly between jobs")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images (pilot)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override PGD steps (pilot; faster)")
    ap.add_argument("--epsilon", type=float, action="append", default=None,
                    help="override epsilon(s); repeatable (pilot)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    acfg = load_config(args.attack)
    if args.steps is not None:
        acfg["steps"] = args.steps
    if args.epsilon:
        acfg["epsilon"] = args.epsilon
    seed_everything(cfg["seed"])

    run_dir = Path(args.out)
    for sub in ("jobs/clean", "jobs/adv", "clean", "attacks", "metrics",
                "statistics", "plots", "examples", "logs", "configs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    snapshot_configs(run_dir, [args.config, args.attack])
    write_manifest(run_dir, {"stage": "full", "config": cfg, "attack": acfg,
                             "resumable": True})
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    log.info("run dir: %s (deadline: %s)", run_dir,
             f"{args.max_hours}h" if deadline else "none")

    model = build_model(cfg)
    ds = build_dataset(cfg)
    defenses = build_defenses(cfg)
    size = cfg["dataset"]["image_size"]
    mnt = cfg["evaluation"]["max_new_tokens"]

    log.info("stage 1/2: clean baseline")
    run_clean(model, ds, defenses, run_dir, image_size=size, max_new_tokens=mnt,
              deadline=deadline, limit=args.limit)

    log.info("stage 2/2: attack + defenses (steps=%s, eps=%s, limit=%s)",
             acfg["steps"], acfg["epsilon"], args.limit)
    run_full(model, ds, defenses, run_dir,
             epsilons=acfg["epsilon"], pgd_steps=acfg["steps"],
             pgd_step_size=acfg["step_size"], pgd_random_start=acfg["random_start"],
             seed=cfg["seed"], image_size=size, max_new_tokens=mnt, deadline=deadline,
             limit=args.limit)

    stats = aggregate(run_dir, cfg["defenses"])
    log.info("aggregated: %s", stats)
    log.info("next: python scripts/run_analysis.py --run %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
