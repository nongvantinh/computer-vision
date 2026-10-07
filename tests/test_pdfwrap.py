"""PDF embedding: the legacy adapter hides a JPEG 75; the lossless adapter does not."""
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from src.defenses import build_defense
from src.defenses.base import pil_to_array
from src.defenses.jpeg import JpegDefense
from src.defenses.pdfwrap import write_pdf

needs_poppler = pytest.mark.skipif(
    shutil.which("pdfimages") is None or shutil.which("pdftoppm") is None,
    reason="poppler-utils not installed")


def _photo_like(n=96, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    a = np.stack([0.5 + 0.4 * np.sin(7 * xx), 0.5 + 0.4 * np.cos(6 * yy), xx * yy], -1)
    a += rng.normal(0, 0.02, a.shape)
    return Image.fromarray(np.clip(np.rint(a * 255), 0, 255).astype(np.uint8))


def _extract(pdf, tmp_path, flag):
    stem = tmp_path / "img"
    subprocess.run(["pdfimages", flag, str(pdf), str(stem)], check=True)
    return sorted(tmp_path.glob("img-*"))[0]


@needs_poppler
def test_lossless_embedding_is_byte_exact(tmp_path):
    img = _photo_like()
    write_pdf(img, tmp_path / "a.pdf", embed="lossless")
    out = np.asarray(Image.open(_extract(tmp_path / "a.pdf", tmp_path, "-png")).convert("RGB"))
    assert np.array_equal(out, np.asarray(img))


@needs_poppler
def test_jpeg_embedding_is_bit_identical_to_jpeg_quality_75(tmp_path):
    """The finding behind the adapter confound, pinned as a test."""
    img = _photo_like()
    write_pdf(img, tmp_path / "a.pdf", embed="jpeg")
    emb = pil_to_array(Image.open(_extract(tmp_path / "a.pdf", tmp_path, "-j")))
    ref = JpegDefense(75).sanitize(pil_to_array(img)).image
    assert np.array_equal(emb, ref)


@needs_poppler
def test_both_embeddings_share_page_geometry(tmp_path):
    img = _photo_like(96)
    for mode in ("jpeg", "lossless"):
        write_pdf(img, tmp_path / f"{mode}.pdf", embed=mode)
        info = subprocess.run(["pdfinfo", str(tmp_path / f"{mode}.pdf")],
                              capture_output=True, text=True).stdout
        assert "Page size:       72 x 72 pts" in info


def test_unknown_embed_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_pdf(_photo_like(), tmp_path / "x.pdf", embed="png")


def test_registry_names_and_embed_modes():
    assert build_defense("dangerzone").name == "dangerzone"
    assert build_defense("dangerzone").embed == "jpeg"
    assert build_defense("dangerzone_ll").name == "dangerzone_ll"
    assert build_defense("dangerzone_ll").embed == "lossless"
    assert build_defense("adapter_only").name == "adapter_only"
    assert build_defense("adapter_only_ll").embed == "lossless"


@needs_poppler
def test_lossless_adapter_is_closer_to_the_input_than_the_jpeg_adapter():
    x = pil_to_array(_photo_like(96))
    legacy = build_defense("adapter_only").sanitize(x).image
    lossless = build_defense("adapter_only_ll").sanitize(x).image

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    assert mse(x, lossless) < mse(x, legacy)       # the hidden JPEG adds distortion
