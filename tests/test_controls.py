"""Control transformations: deterministic, shape-preserving, and each isolates one effect."""
import shutil

import numpy as np
import pytest

from src.defenses import build_defense
from src.defenses.controls import (AdapterOnlyDefense, ChromaOnlyDefense,
                                   ResampleOnlyDefense, TinyNoiseDefense)


def _img(seed=0, n=48):
    return np.random.default_rng(seed).random((n, n, 3)).astype(np.float32)


def _psnr(a, b):
    m = float(np.mean((a - b) ** 2))
    return 99.0 if m == 0 else 10 * np.log10(1 / m)


CONTROLS = [ResampleOnlyDefense(), ChromaOnlyDefense(), TinyNoiseDefense(1.0)]


@pytest.mark.parametrize("d", CONTROLS, ids=lambda d: d.name)
def test_deterministic_shape_and_range(d):
    x = _img()
    a, b = d.sanitize(x).image, d.sanitize(x).image
    assert np.array_equal(a, b)
    assert a.shape == x.shape and a.min() >= 0.0 and a.max() <= 1.0


def test_registry_builds_every_control_and_extra_jpeg_qualities():
    names = ["adapter_only", "resample_only", "chroma_only", "noise1", "noise2",
             "jpeg95", "jpeg30", "jpeg10"]
    built = {n: build_defense(n) for n in names}
    assert built["noise2"].level == 2.0 and built["jpeg30"].quality == 30
    assert built["noise1"].name == "noise1"
    with pytest.raises(ValueError):
        build_defense("noise")           # no level given


def test_chroma_only_leaves_a_gray_image_unchanged():
    gray = np.repeat(np.linspace(0, 1, 48, dtype=np.float32)[None, :, None], 48, 0)
    gray = np.repeat(gray, 3, axis=2)
    out = ChromaOnlyDefense().sanitize(gray).image
    assert np.abs(out - gray).max() <= 2 / 255          # no chroma, so nothing to subsample


def test_chroma_only_changes_a_colour_checkerboard():
    x = np.zeros((48, 48, 3), np.float32)
    x[::2, ::2, 0] = 1.0                                 # per-pixel red/black pattern
    out = ChromaOnlyDefense().sanitize(x).image
    assert _psnr(x, out) < 30


def test_resample_only_is_lossy_on_noise_and_near_lossless_on_smooth():
    smooth = np.tile(np.linspace(0, 1, 48, dtype=np.float32)[None, :, None], (48, 1, 3))
    assert _psnr(smooth, ResampleOnlyDefense().sanitize(smooth).image) > 40
    assert _psnr(_img(), ResampleOnlyDefense().sanitize(_img()).image) < 30


def test_tiny_noise_magnitude_is_small_and_scales_with_level():
    x = np.full((64, 64, 3), 0.5, np.float32)
    d1 = np.abs(TinyNoiseDefense(1.0).sanitize(x).image - x).mean() * 255
    d2 = np.abs(TinyNoiseDefense(2.0).sanitize(x).image - x).mean() * 255
    assert 0.3 < d1 < 1.5 and d2 > d1
    assert np.abs(TinyNoiseDefense(1.0).sanitize(x).image - x).max() * 255 <= 6


def test_tiny_noise_is_content_seeded_not_order_dependent():
    a, b = _img(1), _img(2)
    d = TinyNoiseDefense(1.0)
    first = (d.sanitize(a).image, d.sanitize(b).image)
    second = (d.sanitize(b).image, d.sanitize(a).image)
    assert np.array_equal(first[0], second[1]) and np.array_equal(first[1], second[0])
    n1 = d.sanitize(a).image - a
    n2 = d.sanitize(b).image - b
    assert not np.allclose(n1, n2)                       # different images, different noise


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="poppler not installed")
def test_adapter_only_matches_the_dangerzone_adapter_resolution_path():
    x = _img(n=336)
    out = AdapterOnlyDefense().sanitize(x)
    assert out.image.shape == x.shape
    assert out.meta["intermediate_size"] == (525, 525)   # 336 px at 96 dpi -> 150 dpi
    assert np.array_equal(out.image, AdapterOnlyDefense().sanitize(x).image)
