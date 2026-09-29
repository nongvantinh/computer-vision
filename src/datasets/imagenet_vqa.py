"""Deterministic ImageNet-VQA benchmark (proposal §3.4).

200 images: 20 classes x 10 images, fixed seed, fixed target label per image.
The manifest (configs/dataset_200.json) fully determines the benchmark and is
versioned. The same images are used for every defense condition (paired design).

Expected ImageNet layout (ImageFolder / standard val):
    <imagenet_val_dir>/<wnid>/<image>.JPEG
Class names come from the standard imagenet_class_index.json (downloaded once and
cached to configs/imagenet_class_index.json, or provided by torchvision).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/"
    "imagenet_class_index.json")
PROMPT = ("What is the main object in this image? "
          "Answer with a single lowercase word.")


def load_class_index(cache: Path) -> dict[str, tuple[str, str]]:
    """Return {index_str: (wnid, class_name)}. Cache locally for reproducibility."""
    cache = Path(cache)
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        with urllib.request.urlopen(CLASS_INDEX_URL, timeout=30) as r:
            raw = json.loads(r.read().decode())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw, indent=2))
    return {k: (v[0], v[1]) for k, v in raw.items()}


@dataclass
class Sample:
    image_id: str
    file: str
    class_id: str       # ImageNet index as string
    wnid: str
    class_name: str
    target_class_id: str
    target_wnid: str
    target_class_name: str


def build_manifest(imagenet_val_dir: str | Path, out_path: str | Path,
                   class_index_cache: str | Path,
                   n_classes: int = 20, per_class: int = 10,
                   seed: int = 1234) -> list[Sample]:
    """Deterministically select classes, images, and target labels; write JSON."""
    val_dir = Path(imagenet_val_dir)
    idx = load_class_index(class_index_cache)
    wnid_to_id = {wnid: cid for cid, (wnid, _) in idx.items()}

    available = sorted(d.name for d in val_dir.iterdir()
                       if d.is_dir() and d.name in wnid_to_id)
    if len(available) < n_classes:
        raise RuntimeError(
            f"only {len(available)} known wnid folders under {val_dir}; "
            f"need {n_classes}")

    rng = np.random.default_rng(seed)
    chosen_wnids = sorted(rng.choice(available, size=n_classes, replace=False).tolist())
    # target map: rotate the chosen classes by one -> guaranteed different class
    target_of = {w: chosen_wnids[(i + 1) % n_classes]
                 for i, w in enumerate(chosen_wnids)}

    samples: list[Sample] = []
    for wnid in chosen_wnids:
        files = sorted(p.name for p in (val_dir / wnid).glob("*")
                       if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
        if len(files) < per_class:
            raise RuntimeError(f"class {wnid} has only {len(files)} images")
        pick = sorted(rng.choice(files, size=per_class, replace=False).tolist())
        tgt = target_of[wnid]
        for fname in pick:
            samples.append(Sample(
                image_id=f"{wnid}/{fname}",
                file=str(Path(wnid) / fname),
                class_id=wnid_to_id[wnid], wnid=wnid,
                class_name=idx[wnid_to_id[wnid]][1],
                target_class_id=wnid_to_id[tgt], target_wnid=tgt,
                target_class_name=idx[wnid_to_id[tgt]][1],
            ))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"seed": seed, "n_classes": n_classes, "per_class": per_class,
         "prompt": PROMPT, "samples": [asdict(s) for s in samples]}, indent=2))
    return samples


class ImageNetVQA:
    """Loads the manifest and serves (image_array, prompt, labels, meta)."""

    def __init__(self, manifest_path: str | Path, imagenet_val_dir: str | Path):
        data = json.loads(Path(manifest_path).read_text())
        self.prompt = data["prompt"]
        self.samples = [Sample(**s) for s in data["samples"]]
        self.root = Path(imagenet_val_dir)
        # class-name vocabulary present in this benchmark, for label matching
        names = set()
        for s in self.samples:
            names.add(s.class_name)
            names.add(s.target_class_name)
        self.class_names = sorted(names)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, s: Sample, size: int = 336) -> np.ndarray:
        img = Image.open(self.root / s.file).convert("RGB").resize(
            (size, size), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __iter__(self):
        for s in self.samples:
            yield s
