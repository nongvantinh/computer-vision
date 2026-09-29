#!/usr/bin/env python
"""Build the deterministic 200-image ImageNet-VQA manifest (configs/dataset_200.json).

    python scripts/build_dataset.py --config configs/experiment.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402
from src.datasets.imagenet_vqa import build_manifest  # noqa: E402

log = get_logger("build_dataset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    d = cfg["dataset"]
    samples = build_manifest(
        imagenet_val_dir=d["imagenet_val_dir"], out_path=d["manifest"],
        class_index_cache=d["class_index_cache"],
        n_classes=d["n_classes"], per_class=d["per_class"], seed=cfg["seed"])
    log.info("wrote %d samples to %s", len(samples), d["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
