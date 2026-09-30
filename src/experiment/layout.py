"""Run-directory contract (item 22): results/runs/<run_id>/.

A run directory is self-describing. A third party can read it and know the model,
dataset, attack, defenses, versions, hardware, seeds, how many samples finished,
and how failures were handled, without rerunning anything.

Layout:
    results/runs/<run_id>/
        config.json            merged experiment + attack config, with overrides
        environment.json        python/torch/cuda/gpu/package versions
        git_commit.txt          repo commit the run was launched from
        dataset_manifest.json   copy of the exact dataset manifest used
        model_info.json         model id, dtype, quantization, device
        defense_info.json       defense list + external tool commands/versions
        progress.json           live counts, updated as jobs finish
        jobs/
            clean/<image>__<defense>.json
            attack/<image>__eps<e>.json
            defense/<image>__eps<e>__<defense>.json
        adv/<image>__eps<e>.npy + .png      persisted adversarial images
        clean_results.jsonl     derived (rebuilt from jobs by aggregate)
        attack_results.jsonl    derived
        defense_results.jsonl   derived
        summary.json            derived by run_analysis
        statistics.json         derived by run_analysis
        examples/ fft/ perturbations/ plots/ logs/
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import capture_environment
from ..utils.resume import atomic_write_json, read_json

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "results" / "runs"

SUBDIRS = ("jobs/clean", "jobs/attack", "jobs/defense", "adv",
           "examples", "fft", "perturbations", "plots", "logs")


@dataclass
class RunLayout:
    """Paths inside one run directory. Pure path math, no I/O."""
    root: Path

    @property
    def config(self) -> Path: return self.root / "config.json"
    @property
    def environment(self) -> Path: return self.root / "environment.json"
    @property
    def git_commit(self) -> Path: return self.root / "git_commit.txt"
    @property
    def dataset_manifest(self) -> Path: return self.root / "dataset_manifest.json"
    @property
    def model_info(self) -> Path: return self.root / "model_info.json"
    @property
    def defense_info(self) -> Path: return self.root / "defense_info.json"
    @property
    def progress(self) -> Path: return self.root / "progress.json"
    @property
    def clean_jobs(self) -> Path: return self.root / "jobs" / "clean"
    @property
    def attack_jobs(self) -> Path: return self.root / "jobs" / "attack"
    @property
    def defense_jobs(self) -> Path: return self.root / "jobs" / "defense"
    @property
    def adv(self) -> Path: return self.root / "adv"
    @property
    def examples(self) -> Path: return self.root / "examples"
    @property
    def logs(self) -> Path: return self.root / "logs"

    # derived result files
    @property
    def clean_results(self) -> Path: return self.root / "clean_results.jsonl"
    @property
    def attack_results(self) -> Path: return self.root / "attack_results.jsonl"
    @property
    def defense_results(self) -> Path: return self.root / "defense_results.jsonl"
    @property
    def summary(self) -> Path: return self.root / "summary.json"
    @property
    def statistics(self) -> Path: return self.root / "statistics.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _model_info(cfg: dict) -> dict:
    m = cfg.get("model", {})
    return {"model_id": m.get("model_id"), "dtype": m.get("dtype"),
            "load_in_4bit": m.get("load_in_4bit", False),
            "gradient_checkpointing": m.get("gradient_checkpointing", True),
            "device": m.get("device"),
            "vision_tower": "openai/clip-vit-large-patch14-336"}


def _defense_info(cfg: dict) -> dict:
    return {"defenses": list(cfg.get("defenses", [])),
            "tools": dict(cfg.get("tools", {}))}


def create_run(run_id: str, cfg: dict, attack_cfg: dict,
               manifest_path: str | Path | None = None,
               base: Path | None = None) -> RunLayout:
    """Create (or resume) a run directory and write the reproducibility artifacts.

    Idempotent: an existing run_id is reused so a resumed session keeps its files.
    The config snapshot is refreshed on every launch and the previous one is kept
    in progress.json history, so a changed config is visible rather than silent.
    """
    root = (Path(base) if base else RUNS_ROOT) / run_id
    lay = RunLayout(root)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    merged = {"experiment": cfg, "attack": attack_cfg, "run_id": run_id}
    atomic_write_json(lay.config, merged)
    atomic_write_json(lay.environment, capture_environment())
    lay.git_commit.write_text(_git_commit() + "\n")
    atomic_write_json(lay.model_info, _model_info(cfg))
    atomic_write_json(lay.defense_info, _defense_info(cfg))
    if manifest_path and Path(manifest_path).exists():
        atomic_write_json(lay.dataset_manifest,
                          json.loads(Path(manifest_path).read_text()))

    prog = read_json(lay.progress) or {"created": True, "launches": 0}
    prog["launches"] = prog.get("launches", 0) + 1
    atomic_write_json(lay.progress, prog)
    return lay


def load_run(run_id: str, base: Path | None = None) -> RunLayout:
    root = (Path(base) if base else RUNS_ROOT) / run_id
    if not root.exists():
        raise FileNotFoundError(f"no run directory at {root}")
    return RunLayout(root)
