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
    raise ValueError(f"unknown defense: {name}")


__all__ = ["Defense", "DefenseResult", "Identity", "JpegDefense",
           "build_defense", "array_to_pil", "pil_to_array"]
