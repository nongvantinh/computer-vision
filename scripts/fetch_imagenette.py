#!/usr/bin/env python
"""Fetch a free 20-class ImageNet subset for the benchmark.

Imagenette (10 classes) + Imagewoof (10 classes) from fast.ai are real ImageNet
classes with real wnid folder names, so they drop straight into the ImageFolder
loader and the standard imagenet_class_index.json. No login required.

    python scripts/fetch_imagenette.py --out data/imagenet/val

Populates <out>/<wnid>/*.JPEG for 20 wnids. Documented deviation from the
proposal: the 20 classes are this fixed Imagenette+Imagewoof set, not a random
20 of 1000. Uses the 320px variants (~325 MB each); images are resized to 336.
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URLS = {
    "imagenette2-320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "imagewoof2-320": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/imagenet/val")
    ap.add_argument("--cache", default="/content/imagenette_cache")
    args = ap.parse_args()
    out = Path(args.out)
    cache = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    wnids = 0
    for name, url in URLS.items():
        tgz = cache / f"{name}.tgz"
        _download(url, tgz)
        with tarfile.open(tgz) as t:
            top = t.getnames()[0].split("/")[0]
        root = cache / top
        if not root.exists():
            print(f"[tar ] extracting {tgz.name}")
            with tarfile.open(tgz) as t:
                t.extractall(cache)
        val = root / "val"
        for wnid_dir in sorted(p for p in val.iterdir() if p.is_dir()):
            dst = out / wnid_dir.name
            if dst.exists() or dst.is_symlink():
                continue
            dst.symlink_to(wnid_dir.resolve(), target_is_directory=True)
            wnids += 1
            n = len(list(wnid_dir.glob("*.JPEG")))
            print(f"  {wnid_dir.name} -> {n} images")

    total_classes = len([p for p in out.iterdir() if p.is_dir() or p.is_symlink()])
    print(f"[ok  ] {total_classes} classes available under {out} (added {wnids})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
