"""Resume primitives: per-job idempotency so a killed session loses at most one job.

Mirrors the knowledge-tracing pattern (skip any unit whose result file exists),
adapted to this project where a job is one (image_id, epsilon) or one clean image.
Writes are atomic (temp + os.replace) so an interrupted write never leaves a
half-file that would be wrongly treated as "done".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def sanitize_id(image_id: str) -> str:
    """Filesystem-safe stem for an ImageNet id like 'n01440764/img_0.JPEG'."""
    return image_id.replace("/", "__").replace(" ", "_")


def atomic_write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic on POSIX; a crash leaves either old or new
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def clean_job_path(run_dir: str | Path, image_id: str) -> Path:
    return Path(run_dir) / "jobs" / "clean" / f"{sanitize_id(image_id)}.json"


def adv_job_path(run_dir: str | Path, image_id: str, epsilon: float) -> Path:
    return Path(run_dir) / "jobs" / "adv" / f"{sanitize_id(image_id)}__eps{epsilon:.4f}.json"


def is_done(path: str | Path) -> bool:
    return Path(path).exists()
