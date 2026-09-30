"""Dangerzone defense (D5): the real Freedom of the Press Dangerzone tool.

Dangerzone renders a document to raw RGB pixels in a sandbox and rebuilds a clean
PDF, discarding all non-pixel structure. To use it as an *image* defense we:
    image -> (wrap into a single-page PDF) -> dangerzone-cli -> safe PDF
          -> rasterize page back to an image at the original resolution.
The rasterize step uses pdftoppm (Poppler). This adapter is the smallest honest
way to push an image through Dangerzone's real sanitization pipeline; it is
documented in experiment-protocol.md rather than hidden. Dangerzone also accepts
raster images natively, so the PDF wrap is valid but optional (see dangerzone.md).

Configure the CLI via DANGERZONE_CMD (default "dangerzone-cli"). On Linux the tool
drives Podman (not Docker); install it on the run host (e.g. Colab has root). See
docs/implementation/dangerzone.md for the verified setup and the container engine
requirement.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .base import Defense


class DangerzoneDefense(Defense):
    name = "dangerzone"

    def __init__(self, cli: str | None = None, timeout_s: int = 300):
        self.cli = cli or os.environ.get("DANGERZONE_CMD", "dangerzone-cli")
        self.timeout_s = timeout_s

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        w, h = img.size
        with tempfile.TemporaryDirectory() as td:
            src_pdf = Path(td) / "in.pdf"
            safe_pdf = Path(td) / "in-safe.pdf"
            img.convert("RGB").save(src_pdf, "PDF", resolution=96.0)
            cmd = [self.cli, "--output-filename", str(safe_pdf), str(src_pdf)]
            proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s)
            if proc.returncode != 0 or not safe_pdf.exists():
                out = proc.stdout.decode(errors="ignore")
                err = proc.stderr.decode(errors="ignore")
                raise RuntimeError(
                    f"Dangerzone failed (rc={proc.returncode}). "
                    f"stdout: {out[-900:]} || stderr: {err[-900:]}")
            # rasterize the sanitized PDF back to a raster at original size
            stem = Path(td) / "page"
            subprocess.run(
                ["pdftoppm", "-png", "-singlefile", "-r", "150",
                 str(safe_pdf), str(stem)],
                check=True, capture_output=True, timeout=self.timeout_s)
            page = Image.open(str(stem) + ".png").convert("RGB")
            page = page.resize((w, h), Image.LANCZOS)
        return page, {"op": "dangerzone", "cli": self.cli, "size": page.size}
