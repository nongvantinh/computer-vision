"""Common defense interface: sanitize(image) -> image.

Canonical in-memory image is a float32 numpy array HxWxC in [0, 1] so it lines up
with the attack and metrics. File-based tools (ICDR, Dangerzone) round-trip
through a temporary PNG/JPEG on disk.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def array_to_pil(img: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def pil_to_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


@dataclass
class DefenseResult:
    image: np.ndarray
    name: str
    runtime_s: float
    meta: dict = field(default_factory=dict)


class Defense(abc.ABC):
    """Base class. Subclasses implement `_apply` on a PIL image."""

    name: str = "base"

    @abc.abstractmethod
    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        ...

    def sanitize(self, img: np.ndarray) -> DefenseResult:
        t0 = time.perf_counter()
        out_pil, meta = self._apply(array_to_pil(img))
        out = pil_to_array(out_pil)
        return DefenseResult(out, self.name, time.perf_counter() - t0, meta)


class Identity(Defense):
    """D0 - no transformation."""

    name = "D0"

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        return img, {"op": "identity"}
