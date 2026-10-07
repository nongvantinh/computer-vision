"""Wrap a raster image into a one-page PDF, with the pixel encoding made explicit.

The Dangerzone adapter needs a PDF. How the image is embedded decides what the
defense actually measures:

  embed="jpeg"      Pillow's PDF writer (the baseline adapter). It re-encodes RGB
                    images with DCT, quality 75, 4:2:0. That is bit-identical to
                    JpegDefense(75) (verified), so every result obtained through
                    this path includes a JPEG 75 round trip BEFORE the sanitizer.
  embed="lossless"  A Flate (zlib) image, byte-exact. The page geometry is the same
                    as the baseline (336 px at 96 dpi = 252 pt), so the only change
                    is that no JPEG is introduced.

Both embeddings are used on purpose: the difference between them is the contribution
of the hidden JPEG.
"""
from __future__ import annotations

import zlib
from pathlib import Path

from PIL import Image

PDF_DPI = 96.0


def write_pdf(img: Image.Image, path: Path, embed: str = "jpeg",
              dpi: float = PDF_DPI) -> None:
    img = img.convert("RGB")
    if embed == "jpeg":
        img.save(path, "PDF", resolution=dpi)
    elif embed == "lossless":
        _write_lossless(img, Path(path), dpi)
    else:
        raise ValueError(f"unknown embed mode {embed!r} (use 'jpeg' or 'lossless')")


def _write_lossless(img: Image.Image, path: Path, dpi: float) -> None:
    w, h = img.size
    w_pt, h_pt = w * 72.0 / dpi, h * 72.0 / dpi
    data = zlib.compress(img.tobytes(), 9)
    content = f"q {w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.4f} {h_pt:.4f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
         f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(data)} >>").encode() + b"\nstream\n" + data + b"\nendstream",
        f"<< /Length {len(content)} >>".encode() + b"\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    Path(path).write_bytes(bytes(out))
