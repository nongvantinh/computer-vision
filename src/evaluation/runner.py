"""Orchestration: resumable, job-idempotent, time-budgeted (Colab-friendly).

Each unit of work writes its own result file and is skipped if that file exists,
so a session killed by a Colab usage limit resumes by re-running only unfinished
jobs. Derived files (rows.jsonl, clean.jsonl, accuracy) are rebuilt by
`aggregate()` from the per-job files, so a new account that restored only the
light `jobs/` tree can reconstruct everything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..defenses import build_defense
from ..datasets.imagenet_vqa import ImageNetVQA
from ..attacks.targeted_pgd import PGDConfig, targeted_pgd
from ..analysis.perturbation import analyze_defense_on_perturbation
from ..utils.resume import (atomic_write_json, read_json, clean_job_path,
                            adv_job_path, is_done, sanitize_id)
from ..utils.logging import get_logger
from .predict import predict
from .metrics import accuracy

log = get_logger("runner")


def build_model(cfg: dict):
    from ..models.llava import LlavaWrapper, LlavaConfig
    m = cfg["model"]
    return LlavaWrapper(LlavaConfig(
        model_id=m["model_id"], device=m["device"], dtype=m["dtype"],
        gradient_checkpointing=m.get("gradient_checkpointing", True),
        load_in_4bit=m.get("load_in_4bit", False)))


def build_defenses(cfg: dict) -> dict:
    tools = cfg.get("tools", {})
    out = {}
    for name in cfg["defenses"]:
        kwargs = {}
        if name == "icdr":
            kwargs["cmd"] = tools.get("icdr_cmd")
        elif name == "dangerzone":
            kwargs["cli"] = tools.get("dangerzone_cmd", "dangerzone-cli")
        out[name] = build_defense(name, **kwargs)
    return out


def build_dataset(cfg: dict) -> ImageNetVQA:
    d = cfg["dataset"]
    return ImageNetVQA(d["manifest"], d["imagenet_val_dir"])


def _deadline_passed(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def run_clean(model, ds: ImageNetVQA, defenses: dict, run_dir: Path,
              image_size: int = 336, max_new_tokens: int = 16,
              deadline: float | None = None) -> int:
    """Clean accuracy under each defense. One job file per image; resumable."""
    run_dir = Path(run_dir)
    done = new = 0
    for s in ds:
        jp = clean_job_path(run_dir, s.image_id)
        if is_done(jp):
            done += 1
            continue
        if _deadline_passed(deadline):
            log.info("clean: time budget reached, stopping (%d done this session)", new)
            break
        img = ds.load_image(s, size=image_size)
        rec = {"image_id": s.image_id, "gt": s.class_name, "defenses": {}}
        for dname, defense in defenses.items():
            dimg = defense.sanitize(img).image
            raw, pred = predict(model, dimg, ds.prompt, ds.class_names, max_new_tokens)
            rec["defenses"][dname] = {"prediction": pred, "raw": raw}
        atomic_write_json(jp, rec)
        new += 1
    log.info("clean: %d already done, %d new this session", done, new)
    return new


def run_full(model, ds: ImageNetVQA, defenses: dict, run_dir: Path,
             epsilons: list[float], pgd_steps: int, pgd_step_size: float,
             pgd_random_start: bool, seed: int, image_size: int = 336,
             max_new_tokens: int = 16, save_images: bool = True,
             deadline: float | None = None) -> int:
    """Attack (defense-unaware) then evaluate every defense. One job file per
    (image, epsilon); resumable and time-budgeted."""
    run_dir = Path(run_dir)
    img_dir = run_dir / "examples"
    done = new = 0
    for s in ds:
        clean = None
        for eps in epsilons:
            jp = adv_job_path(run_dir, s.image_id, eps)
            if is_done(jp):
                done += 1
                continue
            if _deadline_passed(deadline):
                log.info("attack: time budget reached, stopping (%d new this session)", new)
                return new
            if clean is None:
                clean = ds.load_image(s, size=image_size)
            res = targeted_pgd(
                model, clean, ds.prompt, s.target_class_name,
                PGDConfig(epsilon=eps, steps=pgd_steps, step_size=pgd_step_size,
                          random_start=pgd_random_start, seed=seed))
            adv = res.adv_image
            _, pred_d0 = predict(model, adv, ds.prompt, ds.class_names, max_new_tokens)
            rec = {"image_id": s.image_id, "class": s.class_name,
                   "target": s.target_class_name, "epsilon": eps,
                   "attack_success_D0": int(pred_d0 == s.target_class_name),
                   "linf": res.linf, "model": model.cfg.model_id, "defenses": {}}
            for dname, defense in defenses.items():
                dadv = defense.sanitize(adv).image
                raw, pred = predict(model, dadv, ds.prompt, ds.class_names, max_new_tokens)
                mech = analyze_defense_on_perturbation(clean, adv, dadv)
                rec["defenses"][dname] = {
                    "prediction": pred, "raw": raw,
                    "target_success": int(pred == s.target_class_name),
                    "gt_recovered": int(pred == s.class_name),
                    "mechanism": mech}
            atomic_write_json(jp, rec)
            new += 1
            if save_images:
                _save_example(img_dir, s.image_id, eps, clean, adv)
    log.info("attack: %d already done, %d new this session", done, new)
    return new


def _save_example(img_dir: Path, image_id: str, eps: float,
                  clean: np.ndarray, adv: np.ndarray) -> None:
    from ..defenses.base import array_to_pil
    img_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_id(image_id) + f"_eps{eps:.4f}"
    if not (img_dir / f"{stem}_clean.png").exists():
        array_to_pil(clean).save(img_dir / f"{stem}_clean.png")
    array_to_pil(adv).save(img_dir / f"{stem}_adv.png")


# --------------------------------------------------------------------------- #
# Aggregate per-job files into the derived files the analysis step reads.
# Safe to call repeatedly; rebuilt from jobs/ so it works after a partial restore.
# --------------------------------------------------------------------------- #
def aggregate(run_dir: Path, defense_names: list[str]) -> dict:
    run_dir = Path(run_dir)
    clean_dir = run_dir / "jobs" / "clean"
    adv_dir = run_dir / "jobs" / "adv"

    # clean -> clean.jsonl + accuracy
    clean_rows = []
    for jp in sorted(clean_dir.glob("*.json")):
        rec = read_json(jp)
        if not rec:
            continue
        for dname, r in rec["defenses"].items():
            clean_rows.append({"image_id": rec["image_id"], "defense": dname,
                               "condition": "clean", "gt": rec["gt"],
                               "prediction": r["prediction"], "raw": r["raw"]})
    _write_jsonl(run_dir / "clean" / "clean.jsonl", clean_rows)
    acc = {}
    for dname in defense_names:
        sub = [r for r in clean_rows if r["defense"] == dname]
        if sub:
            acc[dname] = accuracy([r["prediction"] for r in sub], [r["gt"] for r in sub])
    atomic_write_json(run_dir / "metrics" / "clean_accuracy.json", acc)

    # adv -> rows.jsonl + perturbation.jsonl
    rows, mech = [], []
    for jp in sorted(adv_dir.glob("*.json")):
        rec = read_json(jp)
        if not rec:
            continue
        for dname, r in rec["defenses"].items():
            rows.append({"image_id": rec["image_id"], "class": rec["class"],
                         "target": rec["target"], "epsilon": rec["epsilon"],
                         "attack_success_D0": rec["attack_success_D0"],
                         "defense": dname, "condition": "adversarial",
                         "model": rec["model"], "prediction": r["prediction"],
                         "raw": r["raw"], "target_success": r["target_success"],
                         "gt_recovered": r["gt_recovered"], "linf": rec["linf"]})
            m = dict(r["mechanism"]); m.update(
                {"image_id": rec["image_id"], "epsilon": rec["epsilon"], "defense": dname})
            mech.append(m)
    _write_jsonl(run_dir / "attacks" / "rows.jsonl", rows)
    _write_jsonl(run_dir / "metrics" / "perturbation.jsonl", mech)
    return {"clean_jobs": len(list(clean_dir.glob('*.json'))),
            "adv_jobs": len(list(adv_dir.glob('*.json'))),
            "rows": len(rows), "clean_accuracy": acc}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
