"""JPEG re-encoding defense (D1-D3). Deterministic Pillow round-trip in memory."""
from __future__ import annotations

import io

from PIL import Image

from .base import Defense


class JpegDefense(Defense):
    def __init__(self, quality: int = 90):
        self.quality = int(quality)
        self.name = f"jpeg{self.quality}"

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality, optimize=False,
                 subsampling="4:2:0")
        size_bytes = buf.tell()
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        return out, {
            "op": "jpeg",
            "quality": self.quality,
            "subsampling": "4:2:0",
            "bytes": size_bytes,
            "size": out.size,
        }
