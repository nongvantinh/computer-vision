#!/usr/bin/env python
"""Checkpoint 3: prove the targeted PGD attack works on D0 before the full run.

Attacks the first N images (default 10) at one epsilon and reports targeted ASR.
Gates the full experiment on attack.yaml:validation.min_asr_to_proceed.

    python scripts/validate_attack.py --config configs/experiment.yaml \
        --attack configs/attack.yaml --epsilon 0.0313725490
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, new_run_dir, snapshot_configs  # noqa: E402
from src.utils.logging import get_logger, seed_everything, write_manifest  # noqa: E402
from src.evaluation.runner import build_model, build_dataset  # noqa: E402
from src.attacks.targeted_pgd import PGDConfig, targeted_pgd  # noqa: E402
from src.evaluation.predict import predict  # noqa: E402
from src.evaluation.metrics import targeted_asr  # noqa: E402

log = get_logger("validate_attack")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--attack", default="configs/attack.yaml")
    ap.add_argument("--epsilon", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    acfg = load_config(args.attack)
    seed_everything(cfg["seed"])
    eps = args.epsilon if args.epsilon is not None else acfg["epsilon"][1]

    run_dir = new_run_dir("validate")
    snapshot_configs(run_dir, [args.config, args.attack])
    write_manifest(run_dir, {"stage": "validate", "epsilon": eps})

    model = build_model(cfg)
    ds = build_dataset(cfg)
    n = acfg["validation"]["n_images"]
    size = cfg["dataset"]["image_size"]

    preds, targets, rows = [], [], []
    for s in itertools.islice(ds, n):
        clean = ds.load_image(s, size=size)
        _, clean_pred = predict(model, clean, ds.prompt, ds.class_names)
        res = targeted_pgd(model, clean, ds.prompt, s.target_class_name,
                           PGDConfig(epsilon=eps, steps=acfg["steps"],
                                     step_size=acfg["step_size"],
                                     random_start=acfg["random_start"],
                                     seed=cfg["seed"]))
        _, adv_pred = predict(model, res.adv_image, ds.prompt, ds.class_names)
        preds.append(adv_pred)
        targets.append(s.target_class_name)
        rows.append({"image_id": s.image_id, "gt": s.class_name,
                     "target": s.target_class_name, "clean_pred": clean_pred,
                     "adv_pred": adv_pred, "linf": res.linf,
                     "final_loss": res.losses[-1] if res.losses else None})
        log.info("%s: clean=%r -> adv=%r (target=%r)",
                 s.image_id, clean_pred, adv_pred, s.target_class_name)

    asr = targeted_asr(preds, targets)
    (run_dir / "metrics" / "validation.json").write_text(
        json.dumps({"epsilon": eps, "n": n, "targeted_asr": asr, "rows": rows}, indent=2))
    thresh = acfg["validation"]["min_asr_to_proceed"]
    ok = asr >= thresh
    log.info("CHECKPOINT 3 — targeted ASR on D0 = %.2f (threshold %.2f): %s",
             asr, thresh, "PASS" if ok else "FAIL — debug before the full run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
