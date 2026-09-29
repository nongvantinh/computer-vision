"""JPEG defense: determinism, shape preservation, bounded change."""
import numpy as np

from src.defenses.jpeg import JpegDefense
from src.defenses.base import Identity


def _img():
    return np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)


def test_jpeg_deterministic():
    img = _img()
    d = JpegDefense(quality=75)
    a = d.sanitize(img).image
    b = d.sanitize(img).image
    assert np.array_equal(a, b)


def test_jpeg_preserves_shape_and_range():
    img = _img()
    out = JpegDefense(quality=90).sanitize(img).image
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_lower_quality_changes_more():
    img = _img()
    hi = JpegDefense(quality=90).sanitize(img).image
    lo = JpegDefense(quality=50).sanitize(img).image
    assert np.mean(np.abs(lo - img)) > np.mean(np.abs(hi - img))


def test_identity_is_lossless_uint8_roundtrip():
    img = _img()
    out = Identity().sanitize(img).image
    # identity goes through uint8 quantization only
    assert np.max(np.abs(out - img)) <= 1 / 255 + 1e-6
