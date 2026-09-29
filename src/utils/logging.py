"""Logging, seeding, and environment capture for reproducibility."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def get_logger(name: str = "cdr-vlm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (if present) for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def capture_environment() -> dict:
    """Snapshot the environment for a run manifest."""
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except Exception:
        env["torch"] = None
    for pkg in ("transformers", "numpy", "scipy", "PIL"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[pkg] = None
    return env


def write_manifest(run_dir: Path, extra: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = capture_environment()
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
