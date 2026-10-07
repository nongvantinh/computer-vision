"""Defense registry. Build a defense by its config name."""
from __future__ import annotations

from .base import Defense, DefenseResult, Identity, array_to_pil, pil_to_array
from .jpeg import JpegDefense


def build_defense(name: str, **kwargs) -> Defense:
    name = name.lower()
    if name in ("d0", "identity", "none"):
        return Identity()
    if name.startswith("jpeg"):
        quality = int(name[4:]) if name[4:].isdigit() else kwargs.get("quality", 90)
        return JpegDefense(quality=quality)
    if name == "icdr":
        from .icdr import IcdrDefense
        return IcdrDefense(**kwargs)
    if name == "dangerzone":
        from .dangerzone import DangerzoneDefense
        return DangerzoneDefense(**kwargs)
    # controls: strip one ingredient out of a real defense (see controls.py)
    if name == "adapter_only":
        from .controls import AdapterOnlyDefense
        return AdapterOnlyDefense(**kwargs)
    if name == "resample_only":
        from .controls import ResampleOnlyDefense
        return ResampleOnlyDefense(**kwargs)
    if name == "chroma_only":
        from .controls import ChromaOnlyDefense
        return ChromaOnlyDefense(**kwargs)
    if name.startswith("noise") and name[5:].replace(".", "", 1).isdigit():
        from .controls import TinyNoiseDefense
        return TinyNoiseDefense(level=float(name[5:]), **kwargs)
    raise ValueError(f"unknown defense: {name}")


__all__ = ["Defense", "DefenseResult", "Identity", "JpegDefense",
           "build_defense", "array_to_pil", "pil_to_array"]
