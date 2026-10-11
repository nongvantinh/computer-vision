#!/usr/bin/env python
"""Select the 50-image screen: stratified over classes, fixed seed, subset of the benchmark.

    python scripts/make_screen_manifest.py --manifest results/runs/full_200/dataset_manifest.json

Two images from every class (40) plus one more from 10 randomly chosen classes (50), all
drawn with a seeded generator from the 200-image benchmark. The output has the same format
as the benchmark manifest, so `--manifest configs/screen_50.json` works with every script.
The screen is a subset of the confirmatory set, so non-adaptive results for these images
already exist in `controls_v1`.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="configs/screen_50.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    m = json.loads(Path(a.manifest).read_text())
    by_class: dict[str, list[dict]] = {}
    for s in m["samples"]:
        by_class.setdefault(s["wnid"], []).append(s)
    classes = sorted(by_class)
    rng = np.random.default_rng(a.seed)
    base, extra = divmod(a.n, len(classes))
    extra_cls = set(rng.choice(classes, size=extra, replace=False).tolist())
    chosen = []
    for c in classes:
        k = base + (1 if c in extra_cls else 0)
        idx = sorted(rng.choice(len(by_class[c]), size=k, replace=False).tolist())
        chosen += [by_class[c][i] for i in idx]
    out = {k: v for k, v in m.items() if k != "samples"}
    out.update({"screen": {"parent_manifest": Path(a.manifest).name, "seed": a.seed,
                           "n": len(chosen), "classes": len(classes),
                           "rule": f"{base} per class plus 1 from {extra} seeded classes"},
                "samples": chosen})
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: {len(chosen)} images, {len({s['wnid'] for s in chosen})} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
