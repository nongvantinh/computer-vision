"""Resumable scheduler (items 3, 11, 13).

Image-major: each image is carried fully through clean evaluation, the attack, and
every defense before the next image starts, so progress is complete per image and a
reclaimed VM loses at most the image in flight. Every unit is a skippable job, so a
resume continues at the exact job that did not finish.

The attack is defense-unaware: one adversarial image per (image, epsilon) crafted
against D0, persisted, then read back by every defense job (item 11). Adding a
defense later runs only new defense jobs; the attack is never recomputed.

The model, dataset, and defenses are passed in, so this module imports no torch and
stays unit-testable with stubs.
"""
from __future__ import annotations

import time
import traceback

import numpy as np

from ..utils.resume import atomic_write_json, is_done
from ..utils.logging import get_logger
from ..evaluation.metrics import normalize_answer, contains_target, psnr, ssim
from ..analysis.perturbation import analyze_defense_controlled
from ..attacks.targeted_pgd import PGDConfig, targeted_pgd
from . import store
from .layout import RunLayout

log = get_logger("experiment")


def _deadline_passed(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def _samples(ds, limit):
    for i, s in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield s


def _clean_reference(model, ds, s, image_size, max_new_tokens, cache):
    if s.image_id not in cache:
        img = ds.load_image(s, size=image_size)
        cache[s.image_id] = (img, normalize_answer(
            model.generate(img, ds.prompt, max_new_tokens)))
    return cache[s.image_id]


def run_experiment(model, ds, defenses: dict, lay: RunLayout, *,
                   epsilons, pgd_steps, pgd_step_size, pgd_random_start, seed,
                   image_size=336, max_new_tokens=16,
                   deadline=None, limit=None) -> dict:
    """Run clean + attack + defense jobs, image by image, resumably."""
    ref_cache: dict = {}
    made = {"clean": 0, "attack": 0, "defense": 0, "defense_failed": 0}

    for s in _samples(ds, limit):
        if _deadline_passed(deadline):
            log.info("time budget reached before image %s", s.image_id)
            break

        # --- clean evaluation: preservation per defense --------------------- #
        for dname, defense in defenses.items():
            jp = store.clean_job_path(lay, s.image_id, dname)
            if is_done(jp):
                continue
            if _deadline_passed(deadline):
                break
            clean, ref = _clean_reference(model, ds, s, image_size,
                                          max_new_tokens, ref_cache)
            try:
                res = defense.sanitize(clean)
                ans = model.generate(res.image, ds.prompt, max_new_tokens)
                rec = {"image_id": s.image_id, "defense": dname, "gt": s.class_name,
                       "clean_answer": ref, "answer": ans,
                       "preserved": int(normalize_answer(ans) == ref),
                       "defense_runtime_s": res.runtime_s, "defense_meta": res.meta,
                       "error": None}
            except Exception as e:  # a defense failure must not kill the run
                rec = {"image_id": s.image_id, "defense": dname, "error": repr(e),
                       "traceback": traceback.format_exc()[-1000:]}
                made["defense_failed"] += 1
            atomic_write_json(jp, rec)
            made["clean"] += 1

        # --- attack + defense-on-adversarial per epsilon ------------------- #
        for eps in epsilons:
            if _deadline_passed(deadline):
                break
            clean, ref = _clean_reference(model, ds, s, image_size,
                                          max_new_tokens, ref_cache)
            ajp = store.attack_job_path(lay, s.image_id, eps)
            adv = store.load_adv_image(lay, s.image_id, eps)
            if not (is_done(ajp) and adv is not None):
                t0 = time.perf_counter()
                r = targeted_pgd(model, clean, ds.prompt, s.target_class_name,
                                 PGDConfig(epsilon=eps, steps=pgd_steps,
                                           step_size=pgd_step_size,
                                           random_start=pgd_random_start, seed=seed))
                adv = r.adv_image
                store.save_adv_image(lay, s.image_id, eps, clean, adv)
                atomic_write_json(ajp, {
                    "image_id": s.image_id, "class": s.class_name,
                    "target": s.target_class_name, "epsilon": eps,
                    "clean_answer": ref, "linf": r.linf,
                    "l2": float(np.sqrt(np.mean((adv - clean) ** 2))),
                    "steps": pgd_steps, "step_size": pgd_step_size,
                    "loss_first": r.losses[0] if r.losses else None,
                    "loss_last": r.losses[-1] if r.losses else None,
                    "attack_runtime_s": time.perf_counter() - t0,
                    "model": model.cfg.model_id})
                made["attack"] += 1
                log.info("attack done %s eps=%.4f loss %.2f -> %.2f (%.0fs, %d this session)",
                         s.image_id, eps, r.losses[0] if r.losses else float("nan"),
                         r.losses[-1] if r.losses else float("nan"),
                         time.perf_counter() - t0, made["attack"])

            for dname, defense in defenses.items():
                djp = store.defense_job_path(lay, s.image_id, eps, dname)
                if is_done(djp):
                    continue
                if _deadline_passed(deadline):
                    break
                try:
                    res = defense.sanitize(adv)
                    res_clean = defense.sanitize(clean)   # defense footprint on clean
                    ans = model.generate(res.image, ds.prompt, max_new_tokens)
                    mech = analyze_defense_controlled(clean, adv, res_clean.image, res.image)
                    rec = {"image_id": s.image_id, "class": s.class_name,
                           "target": s.target_class_name, "epsilon": eps,
                           "defense": dname, "clean_answer": ref, "answer": ans,
                           "target_success": int(contains_target(ans, s.target_class_name)),
                           "restored": int(normalize_answer(ans) == ref),
                           "defense_runtime_s": res.runtime_s,
                           "psnr_vs_clean": psnr(clean, res.image),
                           "ssim_vs_clean": ssim(clean, res.image),
                           "mechanism": mech, "error": None}
                except Exception as e:
                    rec = {"image_id": s.image_id, "epsilon": eps, "defense": dname,
                           "error": repr(e), "traceback": traceback.format_exc()[-1000:]}
                    made["defense_failed"] += 1
                atomic_write_json(djp, rec)
                made["defense"] += 1

        store.update_progress(lay, **store.count_jobs(lay), last_image=s.image_id)

    log.info("jobs this session: %s", made)
    store.update_progress(lay, **store.count_jobs(lay), last_session_made=made)
    return made
