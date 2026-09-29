"""Config loading and run-directory management."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; needed for YAML configs")
        return yaml.safe_load(text)
    return json.loads(text)


def new_run_dir(tag: str = "run", base: Path | None = None) -> Path:
    """Create a fresh, timestamped run directory; never overwrite an old one."""
    base = Path(base) if base else RESULTS_DIR
    stamp = datetime.now().strftime("%Y-%m-%d")
    n = 1
    while True:
        run_dir = base / f"{stamp}_{tag}_{n:03d}"
        if not run_dir.exists():
            break
        n += 1
    for sub in ("clean", "attacks", "defenses", "metrics", "statistics",
                "plots", "examples", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_configs(run_dir: Path, config_paths: list[str | Path]) -> None:
    """Copy the exact configs used into the run dir for reproducibility."""
    dst = Path(run_dir) / "configs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in config_paths:
        p = Path(p)
        if p.exists():
            shutil.copy2(p, dst / p.name)


@dataclass
class ExperimentConfig:
    """Top-level experiment settings (defaults mirror the proposal)."""
    model: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 1234
    n_classes: int = 20
    per_class: int = 10
    epsilons: list[float] = field(default_factory=lambda: [4 / 255, 8 / 255, 16 / 255])
    pgd_steps: int = 200
    pgd_step_size: float = 1 / 255
    pgd_random_start: bool = True
    defenses: list[str] = field(default_factory=lambda: [
        "D0", "jpeg90", "jpeg75", "jpeg50", "icdr", "dangerzone"])
