#!/usr/bin/env python
"""Fixed subsets for the convergence study (docs/phase3_plan.md section 2).

    python scripts/make_conv_manifests.py            # writes configs/conv_100.json, conv_25.json
    python scripts/make_conv_manifests.py --check    # verifies the committed files

Rule (decided 2026-11-20, before any convergence run):
  - source: the confirmatory 200-image manifest (dataset_manifest.json of the conf runs,
    identical to full_200), 20 classes, 10 images each, in manifest order;
  - rng = numpy.random.default_rng(4321);
  - conv_100: for each class in sorted wnid order, permute that class's 10 samples and
    take the first 5;
  - conv_25: continue the same rng, permute the 100 selected ids and take the first 25.

The output files carry the rule and the seed, so the subsets are reproducible from this
script alone. `--check` recomputes and compares against the committed files; it does not
write.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/runs/conf_jpeg50_e16/dataset_manifest.json"
OUT100 = ROOT / "configs/conv_100.json"
OUT25 = ROOT / "configs/conv_25.json"
RULE_100 = ("5 per class: for each class in sorted wnid order, permute the class's 10 "
            "samples with default_rng(4321) and take the first 5")
RULE_25 = ("continue the same rng, permute the 100 selected ids and take the first 25")


def build(src: dict) -> tuple[list, list]:
    by_class = defaultdict(list)
    for s in src["samples"]:
        by_class[s["wnid"]].append(s)
    rng = np.random.default_rng(4321)
    picked = []
    for wnid in sorted(by_class):
        pool = by_class[wnid]                       # manifest order within the class
        idx = rng.permutation(len(pool))[:5]
        picked.extend(pool[i] for i in idx)
    perm = rng.permutation(len(picked))[:25]
    probe = [picked[i] for i in perm]
    return picked, probe


def wrap(src: dict, samples: list, n: int, rule: str) -> dict:
    return {"seed": src["seed"], "n_classes": src["n_classes"], "per_class": 10,
            "prompt": src["prompt"],
            "conv": {"parent_manifest": "results/runs/conf_jpeg50_e16/dataset_manifest.json",
                     "rng": 4321, "n": n, "rule": rule},
            "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify existing files, write nothing")
    args = ap.parse_args()

    src = json.loads(SRC.read_text())
    picked, probe = build(src)
    assert len(picked) == 100 and len(probe) == 25
    assert len({s["image_id"] for s in picked}) == 100
    assert len({s["image_id"] for s in probe}) == 25
    assert {s["image_id"] for s in probe} <= {s["image_id"] for s in picked}

    if args.check:
        ok = True
        for path, samples, rule in ((OUT100, picked, RULE_100), (OUT25, probe, RULE_25)):
            have = json.loads(path.read_text())
            want = wrap(src, samples, len(samples), rule)
            if have != want:
                ok = False
                print(f"MISMATCH {path.relative_to(ROOT)}", file=sys.stderr)
        print("conv manifests: OK" if ok else "conv manifests: MISMATCH", file=sys.stderr)
        return 0 if ok else 1

    OUT100.write_text(json.dumps(wrap(src, picked, 100, RULE_100), indent=1))
    OUT25.write_text(json.dumps(wrap(src, probe, 25, RULE_25), indent=1))
    print(f"wrote {OUT100.relative_to(ROOT)} ({len(picked)} samples) and "
          f"{OUT25.relative_to(ROOT)} ({len(probe)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
