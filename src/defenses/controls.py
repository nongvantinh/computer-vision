"""Control transformations that isolate what a defense does to an image.

None of these is a defense proposal. Each strips one ingredient out of a real
defense so the study can say which ingredient suppresses the attack:

  adapter_only   our image -> PDF -> 150 dpi raster -> Lanczos resize pipeline,
                 i.e. the Dangerzone adapter WITHOUT Dangerzone
  resample_only  Lanczos up-then-down by the adapter's scale factor, no PDF
  chroma_only    JPEG-style 4:2:0 chroma subsampling, no DCT quantization
  noiseN         i.i.d. Gaussian noise of N/255 standard deviation, independent of
                 the model and the label; estimates answer instability under a tiny,
                 benign change and cannot be tuned against the VLM

All are deterministic: the noise seed is derived from the image bytes and a fixed
base seed, so a rerun reproduces the same pixels in any order.
"""
from __future__ import annotations

import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Defense

PDF_DPI = 96.0
RASTER_DPI = 150.0
SCALE = RASTER_DPI / PDF_DPI          # 1.5625: 336 px -> 525 px


class AdapterOnlyDefense(Defense):
    """The Dangerzone adapter with the sanitizer left out (control)."""

    name = "adapter_only"

    def __init__(self, pdftoppm: str = "pdftoppm", timeout_s: int = 120):
        self.pdftoppm = pdftoppm
        self.timeout_s = timeout_s

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        w, h = img.size
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "in.pdf"
            img.convert("RGB").save(pdf, "PDF", resolution=PDF_DPI)
            stem = Path(td) / "page"
            subprocess.run([self.pdftoppm, "-png", "-singlefile", "-r",
                            str(int(RASTER_DPI)), str(pdf), str(stem)],
                           check=True, capture_output=True, timeout=self.timeout_s)
            page = Image.open(str(stem) + ".png").convert("RGB")
            mid = page.size
            page = page.resize((w, h), Image.LANCZOS)
        return page, {"op": "adapter_only", "intermediate_size": mid, "size": page.size}


class ResampleOnlyDefense(Defense):
    """Lanczos resize up by the adapter scale and back down (no PDF involved)."""

    name = "resample_only"

    def __init__(self, scale: float = SCALE):
        self.scale = float(scale)

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        w, h = img.size
        up = img.resize((round(w * self.scale), round(h * self.scale)), Image.LANCZOS)
        out = up.resize((w, h), Image.LANCZOS)
        return out, {"op": "resample_only", "scale": self.scale, "size": out.size}


class ChromaOnlyDefense(Defense):
    """JPEG-style chroma subsampling (4:2:0) with no DCT quantization."""

    name = "chroma_only"

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        w, h = img.size
        ycc = img.convert("YCbCr")
        y, cb, cr = ycc.split()
        half = (max(1, w // 2), max(1, h // 2))
        cb2 = cb.resize(half, Image.BOX).resize((w, h), Image.BILINEAR)
        cr2 = cr.resize(half, Image.BOX).resize((w, h), Image.BILINEAR)
        out = Image.merge("YCbCr", (y, cb2, cr2)).convert("RGB")
        return out, {"op": "chroma_only", "subsampling": "4:2:0", "size": out.size}


class TinyNoiseDefense(Defense):
    """Additive Gaussian noise, sigma = `level`/255, seeded from the image bytes."""

    def __init__(self, level: float = 1.0, base_seed: int = 1234):
        self.level = float(level)
        self.base_seed = int(base_seed)
        self.name = f"noise{int(level) if float(level).is_integer() else level}"

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        seed = (zlib.crc32(arr.astype(np.uint8).tobytes()) ^ self.base_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        noisy = arr + rng.normal(0.0, self.level, size=arr.shape).astype(np.float32)
        out = Image.fromarray(np.clip(np.rint(noisy), 0, 255).astype(np.uint8))
        return out, {"op": "tiny_noise", "sigma_255": self.level, "size": out.size}
