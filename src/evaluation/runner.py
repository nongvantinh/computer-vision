"""Orchestration: resumable, job-idempotent, time-budgeted (Colab-friendly).

Metric design (revised after the pilot): a generic VLM will not emit fine-grained
ImageNet class names ("tench", "Shih-Tzu") for a clean image, so accuracy against
the ImageNet label is meaningless. Instead we use the model's OWN clean answer as
the reference:
  - targeted ASR: the (defended) adversarial answer contains the target label.
  - preserved:    a defense on the CLEAN image keeps the model's clean answer.
  - restored:     a defense on the ADVERSARIAL image brings the answer back to the
                  clean answer (the defense undid the attack).
This is label-free and robust to the model's phrasing.

Each unit writes its own result file and is skipped if it exists, so a session
killed by a Colab usage limit resumes by re-running only unfinished jobs.
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
from .metrics import normalize_answer, contains_target

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


def _samples(ds: ImageNetVQA, limit: int | None):
    for i, s in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield s


def run_clean(model, ds: ImageNetVQA, defenses: dict, run_dir: Path,
              image_size: int = 336, max_new_tokens: int = 16,
              deadline: float | None = None, limit: int | None = None) -> int:
    """Per image: model's clean answer is the reference; measure preservation."""
    run_dir = Path(run_dir)
    done = new = 0
    for s in _samples(ds, limit):
        jp = clean_job_path(run_dir, s.image_id)
        if is_done(jp):
            done += 1
            continue
        if _deadline_passed(deadline):
            log.info("clean: time budget reached (%d new)", new)
            break
        img = ds.load_image(s, size=image_size)
        ref = normalize_answer(model.generate(img, ds.prompt, max_new_tokens))
        rec = {"image_id": s.image_id, "gt": s.class_name,
               "clean_answer": ref, "defenses": {}}
        for dname, defense in defenses.items():
            ans = model.generate(defense.sanitize(img).image, ds.prompt, max_new_tokens)
            rec["defenses"][dname] = {"answer": ans,
                                      "preserved": int(normalize_answer(ans) == ref)}
        atomic_write_json(jp, rec)
        new += 1
    log.info("clean: %d done, %d new", done, new)
    return new


def run_full(model, ds: ImageNetVQA, defenses: dict, run_dir: Path,
             epsilons: list[float], pgd_steps: int, pgd_step_size: float,
             pgd_random_start: bool, seed: int, image_size: int = 336,
             max_new_tokens: int = 16, save_images: bool = True,
             deadline: float | None = None, limit: int | None = None) -> int:
    """Attack (defense-unaware) then evaluate every defense. One job per
    (image, epsilon); resumable and time-budgeted."""
    run_dir = Path(run_dir)
    img_dir = run_dir / "examples"
    done = new = 0
    for s in _samples(ds, limit):
        clean = ref = None
        for eps in epsilons:
            jp = adv_job_path(run_dir, s.image_id, eps)
            if is_done(jp):
                done += 1
                continue
            if _deadline_passed(deadline):
                log.info("attack: time budget reached (%d new)", new)
                return new
            if clean is None:
                clean = ds.load_image(s, size=image_size)
                ref = normalize_answer(model.generate(clean, ds.prompt, max_new_tokens))
            res = targeted_pgd(
                model, clean, ds.prompt, s.target_class_name,
                PGDConfig(epsilon=eps, steps=pgd_steps, step_size=pgd_step_size,
                          random_start=pgd_random_start, seed=seed))
            adv = res.adv_image
            rec = {"image_id": s.image_id, "class": s.class_name,
                   "target": s.target_class_name, "epsilon": eps,
                   "clean_answer": ref, "linf": res.linf,
                   "model": model.cfg.model_id, "defenses": {}}
            for dname, defense in defenses.items():
                ans = model.generate(defense.sanitize(adv).image, ds.prompt, max_new_tokens)
                mech = analyze_defense_on_perturbation(
                    clean, adv, defense.sanitize(adv).image)
                rec["defenses"][dname] = {
                    "answer": ans,
                    "target_success": int(contains_target(ans, s.target_class_name)),
                    "restored": int(normalize_answer(ans) == ref),
                    "mechanism": mech}
            rec["attack_success_D0"] = rec["defenses"].get("D0", {}).get("target_success", 0)
            atomic_write_json(jp, rec)
            new += 1
            if save_images:
                _save_example(img_dir, s.image_id, eps, clean, adv)
    log.info("attack: %d done, %d new", done, new)
    return new


def _save_example(img_dir: Path, image_id: str, eps: float,
                  clean: np.ndarray, adv: np.ndarray) -> None:
    from ..defenses.base import array_to_pil
    img_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_id(image_id) + f"_eps{eps:.4f}"
    if not (img_dir / f"{stem}_clean.png").exists():
        array_to_pil(clean).save(img_dir / f"{stem}_clean.png")
    array_to_pil(adv).save(img_dir / f"{stem}_adv.png")


def aggregate(run_dir: Path, defense_names: list[str]) -> dict:
    """Rebuild derived files (rows.jsonl, preservation, perturbation) from jobs/."""
    run_dir = Path(run_dir)
    clean_dir = run_dir / "jobs" / "clean"
    adv_dir = run_dir / "jobs" / "adv"

    clean_rows = []
    for jp in sorted(clean_dir.glob("*.json")):
        rec = read_json(jp)
        if not rec:
            continue
        for dname, r in rec["defenses"].items():
            clean_rows.append({"image_id": rec["image_id"], "defense": dname,
                               "answer": r["answer"], "preserved": r["preserved"]})
    _write_jsonl(run_dir / "clean" / "clean.jsonl", clean_rows)
    preservation = {}
    for dname in defense_names:
        sub = [r["preserved"] for r in clean_rows if r["defense"] == dname]
        if sub:
            preservation[dname] = sum(sub) / len(sub)
    # keep the filename analysis reads; semantics = clean-answer preservation
    atomic_write_json(run_dir / "metrics" / "clean_accuracy.json", preservation)

    rows, mech = [], []
    for jp in sorted(adv_dir.glob("*.json")):
        rec = read_json(jp)
        if not rec:
            continue
        for dname, r in rec["defenses"].items():
            rows.append({"image_id": rec["image_id"], "class": rec["class"],
                         "target": rec["target"], "epsilon": rec["epsilon"],
                         "attack_success_D0": rec["attack_success_D0"],
                         "defense": dname, "model": rec["model"],
                         "answer": r["answer"], "target_success": r["target_success"],
                         "restored": r["restored"], "linf": rec["linf"]})
            m = dict(r["mechanism"]); m.update(
                {"image_id": rec["image_id"], "epsilon": rec["epsilon"], "defense": dname})
            mech.append(m)
    _write_jsonl(run_dir / "attacks" / "rows.jsonl", rows)
    _write_jsonl(run_dir / "metrics" / "perturbation.jsonl", mech)
    return {"clean_jobs": len(list(clean_dir.glob('*.json'))),
            "adv_jobs": len(list(adv_dir.glob('*.json'))),
            "rows": len(rows), "preservation": preservation}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
