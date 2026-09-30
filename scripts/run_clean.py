#!/usr/bin/env python
"""Checkpoint 2: clean-answer preservation under every defense (no attack). Resumable.

Runs only the clean stage of the experiment (epsilons empty), so it writes clean
jobs into a normal run directory and reports per-defense preservation. Use it to
confirm clean generation and the defenses work before spending GPU time on attacks.

    python scripts/run_clean.py --config configs/experiment.yaml --run-id clean_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402
from src.evaluation.runner import build_model, build_defenses, build_dataset  # noqa: E402
from src.experiment.layout import create_run  # noqa: E402
from src.experiment.run import run_experiment  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402

log = get_logger("run_clean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--run-id", default="clean_check")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    lay = create_run(args.run_id, cfg, {}, cfg["dataset"]["manifest"])
    model = build_model(cfg)
    ds = build_dataset(cfg)
    defenses = build_defenses(cfg)
    run_experiment(model, ds, defenses, lay, epsilons=[], pgd_steps=0,
                   pgd_step_size=0.0, pgd_random_start=False, seed=cfg["seed"],
                   image_size=cfg["dataset"]["image_size"],
                   max_new_tokens=cfg["evaluation"]["max_new_tokens"],
                   limit=args.limit)
    summary = aggregate(lay)
    import json
    clean = [json.loads(l) for l in lay.clean_results.open() if l.strip()]
    log.info("CHECKPOINT 2 — clean-answer preservation per defense:")
    for d in sorted({r["defense"] for r in clean if not r.get("error")}):
        sub = [r["preserved"] for r in clean if r["defense"] == d and not r.get("error")]
        if sub:
            log.info("  %-12s %.3f (n=%d)", d, sum(sub) / len(sub), len(sub))
    log.info("coverage: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
