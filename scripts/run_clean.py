#!/usr/bin/env python
"""Checkpoint 2: clean accuracy under every defense (no attack yet). Resumable.

    python scripts/run_clean.py --config configs/experiment.yaml --out results/mvp
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, snapshot_configs  # noqa: E402
from src.utils.logging import get_logger, seed_everything, write_manifest  # noqa: E402
from src.evaluation.runner import (build_model, build_defenses, build_dataset,  # noqa: E402
                                   run_clean, aggregate)

log = get_logger("run_clean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="results/mvp")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    run_dir = Path(args.out)
    for sub in ("jobs/clean", "clean", "metrics", "configs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    snapshot_configs(run_dir, [args.config])
    write_manifest(run_dir, {"stage": "clean"})

    model = build_model(cfg)
    ds = build_dataset(cfg)
    defenses = build_defenses(cfg)
    run_clean(model, ds, defenses, run_dir,
              image_size=cfg["dataset"]["image_size"],
              max_new_tokens=cfg["evaluation"]["max_new_tokens"])
    stats = aggregate(run_dir, cfg["defenses"])
    log.info("CHECKPOINT 2 — clean accuracy per defense:")
    for k, v in stats["clean_accuracy"].items():
        log.info("  %-12s %.3f", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
