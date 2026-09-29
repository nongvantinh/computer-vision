"""ICDR defense (D4): the real ArielCyber/ICDR tool.

ICDR is the open-source Image Content Disarm and Reconstruction implementation
behind the Open Image CDR paper. Its documented pipeline (`final_fun`) is
    transCode  -> resize (97% compress/restore) -> AdvFilter (Gaussian blur + sharpen)
operating on JPEG files. We invoke the *real* tool as an external process on a
saved image and load the reconstructed result. We do NOT reimplement a look-alike
and call it ICDR.

Setup (documented in experiment-protocol.md):
    git clone https://github.com/ArielCyber/ICDR external/ICDR
    # build/prepare per its README (Java 21 present); expose a runner that maps
    # one input image file to one output image file.
Configure the runner command via the ICDR_CMD env var or the constructor. The
command must contain the placeholders {input} and {output}, e.g.:
    ICDR_CMD="java -jar external/ICDR/icdr.jar {input} {output}"
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .base import Defense


class IcdrDefense(Defense):
    name = "icdr"

    def __init__(self, cmd: str | None = None, timeout_s: int = 120):
        self.cmd = cmd or os.environ.get("ICDR_CMD")
        self.timeout_s = timeout_s
        if not self.cmd:
            raise RuntimeError(
                "ICDR is not configured. Set ICDR_CMD (with {input} and {output} "
                "placeholders) or pass cmd=. See src/defenses/icdr.py docstring. "
                "Do not substitute a homemade pipeline for the real ICDR tool.")
        if "{input}" not in self.cmd or "{output}" not in self.cmd:
            raise ValueError("ICDR_CMD must contain {input} and {output} placeholders")

    def _apply(self, img: Image.Image) -> tuple[Image.Image, dict]:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jpg"
            out = Path(td) / "out.jpg"
            img.save(inp, format="JPEG", quality=100, subsampling="4:4:4")
            cmd = self.cmd.format(input=shlex.quote(str(inp)),
                                  output=shlex.quote(str(out)))
            proc = subprocess.run(cmd, shell=True, capture_output=True,
                                  timeout=self.timeout_s)
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(
                    f"ICDR failed (rc={proc.returncode}): "
                    f"{proc.stderr.decode(errors='ignore')[:500]}")
            result = Image.open(out).convert("RGB")
            result.load()
        return result, {"op": "icdr", "cmd": self.cmd, "size": result.size}
