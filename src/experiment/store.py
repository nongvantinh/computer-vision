"""Job store: idempotent paths, adversarial-image persistence, progress counts.

Job granularity (item 13):
    clean job     one (image, defense): model answer on the sanitized clean image
    attack job    one (image, epsilon): the adversarial image + its metadata
    defense job   one (image, epsilon, defense): the sanitized-adversarial answer

The attack job is the expensive unit and runs once. Its adversarial image is saved
losslessly as .npy (plus a .png view), so defense jobs, added defenses, and later
analysis all read the exact same pixels without recrafting the attack (item 11).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.resume import sanitize_id, atomic_write_json, read_json, is_done
from .layout import RunLayout


def _eps_tag(eps: float) -> str:
    return f"eps{eps:.4f}"


def clean_job_path(lay: RunLayout, image_id: str, defense: str) -> Path:
    return lay.clean_jobs / f"{sanitize_id(image_id)}__{defense}.json"


def attack_job_path(lay: RunLayout, image_id: str, eps: float) -> Path:
    return lay.attack_jobs / f"{sanitize_id(image_id)}__{_eps_tag(eps)}.json"


def defense_job_path(lay: RunLayout, image_id: str, eps: float, defense: str) -> Path:
    return lay.defense_jobs / f"{sanitize_id(image_id)}__{_eps_tag(eps)}__{defense}.json"


def adv_artifact_paths(lay: RunLayout, image_id: str, eps: float) -> tuple[Path, Path]:
    """(.npy lossless, .png view) for one adversarial image."""
    stem = f"{sanitize_id(image_id)}__{_eps_tag(eps)}"
    return lay.adv / f"{stem}.npy", lay.adv / f"{stem}.png"


def save_adv_image(lay: RunLayout, image_id: str, eps: float,
                   clean: np.ndarray, adv: np.ndarray) -> tuple[Path, Path]:
    """Persist the adversarial image (lossless .npy + .png) and a clean .png once.

    The .npy is the source of truth for defense evaluation and analysis; the .png is
    for the demo and figures. Written atomically via a temp file + replace.
    """
    from ..defenses.base import array_to_pil
    npy, png = adv_artifact_paths(lay, image_id, eps)
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy.parent / (npy.name + ".tmp")
    with open(tmp, "wb") as f:            # file handle: np.save won't append .npy
        np.save(f, np.asarray(adv, dtype=np.float32))
    tmp.replace(npy)
    array_to_pil(adv).save(png)
    clean_png = lay.examples / f"{sanitize_id(image_id)}_clean.png"
    if not clean_png.exists():
        clean_png.parent.mkdir(parents=True, exist_ok=True)
        array_to_pil(clean).save(clean_png)
    return npy, png


def load_adv_image(lay: RunLayout, image_id: str, eps: float) -> np.ndarray | None:
    npy, _ = adv_artifact_paths(lay, image_id, eps)
    if not npy.exists():
        return None
    return np.load(npy).astype(np.float32)


def read_attack_meta(lay: RunLayout, image_id: str, eps: float) -> dict | None:
    return read_json(attack_job_path(lay, image_id, eps))


def update_progress(lay: RunLayout, **counts) -> dict:
    """Merge live counts into progress.json (best-effort, atomic)."""
    prog = read_json(lay.progress) or {}
    prog.update(counts)
    atomic_write_json(lay.progress, prog)
    return prog


def count_jobs(lay: RunLayout) -> dict:
    return {"clean": len(list(lay.clean_jobs.glob("*.json"))),
            "attack": len(list(lay.attack_jobs.glob("*.json"))),
            "defense": len(list(lay.defense_jobs.glob("*.json"))),
            "adv_images": len(list(lay.adv.glob("*.npy")))}


__all__ = ["clean_job_path", "attack_job_path", "defense_job_path",
           "adv_artifact_paths", "save_adv_image", "load_adv_image",
           "read_attack_meta", "update_progress", "count_jobs", "is_done"]
