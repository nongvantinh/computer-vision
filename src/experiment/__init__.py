"""Resumable, persistent experiment layer.

Splits the expensive attack (one adversarial image per image+epsilon, crafted
against D0, persisted as a first-class artifact) from the cheap defense evaluation
(one job per image+epsilon+defense that reads the persisted adversarial image).
Every finished unit is written to disk immediately, so a reclaimed VM or a killed
process loses at most one job and a rerun continues where it stopped.
"""
from .layout import RunLayout, create_run, load_run
from .store import (attack_job_path, defense_job_path, clean_job_path,
                    adv_artifact_paths, save_adv_image, load_adv_image,
                    update_progress)

__all__ = [
    "RunLayout", "create_run", "load_run",
    "attack_job_path", "defense_job_path", "clean_job_path",
    "adv_artifact_paths", "save_adv_image", "load_adv_image", "update_progress",
]
