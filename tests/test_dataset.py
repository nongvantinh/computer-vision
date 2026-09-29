"""Dataset builder: determinism and correct (incorrect-by-design) target labels."""
import json

import numpy as np
from PIL import Image

from src.datasets.imagenet_vqa import build_manifest, ImageNetVQA


def _fake_imagenet(tmp_path, n_classes=6, per_class=5):
    """Create a fake ImageFolder val dir + class-index cache."""
    val = tmp_path / "val"
    idx = {}
    wnids = [f"n{1000000 + i}" for i in range(n_classes + 2)]  # a couple extra
    for i, w in enumerate(wnids):
        idx[str(i)] = [w, f"class_{i}"]
        d = val / w
        d.mkdir(parents=True)
        for j in range(per_class + 3):
            Image.new("RGB", (8, 8), (j, j, j)).save(d / f"img_{j}.JPEG")
    cache = tmp_path / "class_index.json"
    cache.write_text(json.dumps(idx))
    return val, cache


def test_manifest_is_deterministic(tmp_path):
    val, cache = _fake_imagenet(tmp_path)
    out1 = tmp_path / "m1.json"
    out2 = tmp_path / "m2.json"
    build_manifest(val, out1, cache, n_classes=4, per_class=3, seed=7)
    build_manifest(val, out2, cache, n_classes=4, per_class=3, seed=7)
    assert out1.read_text() == out2.read_text()


def test_manifest_size_and_target_differs_from_gt(tmp_path):
    val, cache = _fake_imagenet(tmp_path)
    out = tmp_path / "m.json"
    samples = build_manifest(val, out, cache, n_classes=4, per_class=3, seed=1)
    assert len(samples) == 4 * 3
    for s in samples:
        assert s.target_wnid != s.wnid                  # target is deliberately wrong
        assert s.target_class_name != s.class_name


def test_loader_reads_images_and_vocab(tmp_path):
    val, cache = _fake_imagenet(tmp_path)
    out = tmp_path / "m.json"
    build_manifest(val, out, cache, n_classes=4, per_class=3, seed=1)
    ds = ImageNetVQA(out, val)
    assert len(ds) == 12
    s = next(iter(ds))
    img = ds.load_image(s, size=16)
    assert img.shape == (16, 16, 3)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert s.class_name in ds.class_names
