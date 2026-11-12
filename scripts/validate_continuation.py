#!/usr/bin/env python
"""Continuation-equivalence check for the matched-budget (500-step) oblivious control.

The control continues the stored 200-step attacks for 300 more steps. This script tests, on a
documented sample, whether continuing from a stored array gives the same result as one
uninterrupted run, under the same environment, and measures run-to-run variation as the noise
floor. Tolerances are constants below and in docs/phase3_plan.md section 1.2. They were fixed
before the first run.

Per sampled image (oblivious attack, eps 16/255, seed 1234, LLaVA-1.5-7B 4-bit):
  U1  uninterrupted 500 steps from the seeded random start
  U2  the same again (noise floor)
  R   200 steps from the same start, array saved with the run store (float32 .npy) and read back,
      then 300 steps continued with init_adv
  P   300 steps continued from the array stored in the Phase 1 run (full_200), different session

Writes <out>/continuation.json and the final arrays as <out>/<id>_{U1,U2,R,P}.npy. Read-only with
respect to every run directory (the Phase 1 arrays are only read).

    python scripts/validate_continuation.py --out results/continuation --phase1 results/runs/full_200
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks.targeted_pgd import PGDConfig, targeted_pgd  # noqa: E402
from src.evaluation import labels as L  # noqa: E402
from src.evaluation.runner import build_dataset, build_model  # noqa: E402
from src.experiment import store  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.defenses.jpeg import JpegDefense  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402

log = get_logger("validate_continuation")

EPS = 16 / 255
N_IMAGES = 4
STEPS_TOTAL, STEPS_FIRST = 500, 200
# Pre-specified equivalence rule (see docs/phase3_plan.md 1.2)
REL_LOSS_TOL = 0.05        # final-loss relative difference allowed beyond the noise floor
NOISE_FACTOR = 1.25        # continuation may differ from U1 by at most this times |U1 - U2|
ABS_LINF_TOL = 1e-6        # when U1 and U2 are bitwise equal, R must equal U1 up to this


def pick_images(ds, seed: int, n: int):
    """First image (manifest order) of n classes drawn with a seeded RNG: no outcome is used."""
    samples = list(ds)
    classes = sorted({s.class_name for s in samples})
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(classes, size=n, replace=False))
    return [next(s for s in samples if s.class_name == c) for c in chosen]


def verdict_for(r: dict) -> dict:
    """Pre-specified equivalence rule applied to one image's row (docs/phase3_plan.md 1.2)."""
    floor_linf = r["linf_U1_U2"]
    floor_loss = abs(r["loss_tail"]["U1"] - r["loss_tail"]["U2"]) / max(r["loss_tail"]["U1"], 1e-9)
    loss_gap = abs(r["loss_tail"]["U1"] - r["loss_tail"]["R"]) / max(r["loss_tail"]["U1"], 1e-9)
    deterministic = floor_linf <= ABS_LINF_TOL
    linf_ok = (r["linf_U1_R"] <= ABS_LINF_TOL) if deterministic else (r["linf_U1_R"] <= NOISE_FACTOR * floor_linf)
    loss_ok = loss_gap <= max(REL_LOSS_TOL, NOISE_FACTOR * floor_loss)
    sc = r["score"]
    succ_ok = (sc["R"]["D0"] == sc["U1"]["D0"]) or (sc["U1"]["D0"] != sc["U2"]["D0"])
    return {"image_id": r["image_id"], "bitwise_deterministic_floor": deterministic, "linf_ok": linf_ok,
            "loss_ok": loss_ok, "success_ok": succ_ok, "pass": linf_ok and loss_ok and succ_ok}


def tail_loss(curve, k=10):
    return float(np.mean(curve[-k:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--attack", default="configs/attack.yaml")
    ap.add_argument("--phase1", default="results/runs/full_200")
    ap.add_argument("--out", default="results/continuation")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    cfg, acfg = load_config(a.config), load_config(a.attack)
    if a.manifest:
        cfg["dataset"]["manifest"] = a.manifest
    seed_everything(cfg["seed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ds = build_model(cfg), build_dataset(cfg)
    size = cfg["dataset"]["image_size"]
    phase1 = RunLayout(Path(a.phase1))
    jpeg50 = JpegDefense(50)

    def pgd(clean, target, steps, init=None):
        return targeted_pgd(model, clean, ds.prompt, target,
                            PGDConfig(epsilon=EPS, steps=steps, step_size=acfg["step_size"],
                                      random_start=acfg["random_start"], seed=cfg["seed"],
                                      init_adv=init))

    def score(adv, target):
        d0 = model.generate(adv, ds.prompt, cfg["evaluation"]["max_new_tokens"])
        jp = model.generate(jpeg50.sanitize(adv).image, ds.prompt, cfg["evaluation"]["max_new_tokens"])
        return {"D0": L.tiers(d0, target)["exact"], "jpeg50": L.tiers(jp, target)["exact"],
                "answer_D0": d0, "answer_jpeg50": jp}

    rows = []
    for s in pick_images(ds, cfg["seed"], N_IMAGES):
        clean = ds.load_image(s, size=size)
        t0 = time.perf_counter()
        u1 = pgd(clean, s.target_class_name, STEPS_TOTAL)
        u2 = pgd(clean, s.target_class_name, STEPS_TOTAL)
        first = pgd(clean, s.target_class_name, STEPS_FIRST)
        # round-trip through the run store format exactly as a stored 200-step array would be
        tmp = out / f"{store.sanitize_id(s.image_id)}_R200.npy"
        np.save(tmp, np.asarray(first.adv_image, dtype=np.float32))
        reloaded = np.load(tmp)
        r2 = pgd(clean, s.target_class_name, STEPS_TOTAL - STEPS_FIRST, init=reloaded)
        stored = store.load_adv_image(phase1, s.image_id, EPS)
        p2 = pgd(clean, s.target_class_name, STEPS_TOTAL - STEPS_FIRST, init=stored) if stored is not None else None
        r_curve = list(first.losses) + list(r2.losses)
        row = {
            "image_id": s.image_id, "class": s.class_name, "target": s.target_class_name,
            "runtime_s": time.perf_counter() - t0,
            "prefix200_curve_equal_U1": bool(np.allclose(first.losses, u1.losses[:STEPS_FIRST], rtol=0, atol=1e-6)),
            "prefix200_array_linf_vs_U1_curve_ref": None,
            "linf_U1_U2": float(np.max(np.abs(u1.adv_image - u2.adv_image))),
            "linf_U1_R": float(np.max(np.abs(u1.adv_image - r2.adv_image))),
            "linf_U1_P": float(np.max(np.abs(u1.adv_image - p2.adv_image))) if p2 is not None else None,
            "loss_tail": {"U1": tail_loss(u1.losses), "U2": tail_loss(u2.losses), "R": tail_loss(r_curve),
                          "P": tail_loss(list(p2.losses)) if p2 is not None else None},
            "hash": {k: hashlib.sha256(np.asarray(v.adv_image, dtype=np.float32).tobytes()).hexdigest()[:16]
                     for k, v in {"U1": u1, "U2": u2, "R": r2}.items()},
            "score": {"U1": score(u1.adv_image, s.target_class_name), "U2": score(u2.adv_image, s.target_class_name),
                      "R": score(r2.adv_image, s.target_class_name),
                      "P": score(p2.adv_image, s.target_class_name) if p2 is not None else None},
            "stored_phase1_array_found": stored is not None,
            "stored_array_in_ball": bool(stored is not None and float(np.max(np.abs(stored - clean))) <= EPS + 1e-4),
        }
        for k, v in {"U1": u1, "U2": u2, "R": r2}.items():
            np.save(out / f"{store.sanitize_id(s.image_id)}_{k}.npy", np.asarray(v.adv_image, dtype=np.float32))
        tmp.unlink()
        rows.append(row)
        log.info("%s Linf U1-U2 %.2e U1-R %.2e tail loss U1 %.3f U2 %.3f R %.3f",
                 s.image_id, row["linf_U1_U2"], row["linf_U1_R"], *[row["loss_tail"][k] for k in ("U1", "U2", "R")])
        (out / "continuation.json").write_text(json.dumps({"rows": rows}, indent=1))

    # pre-specified verdict
    verdicts = [verdict_for(r) for r in rows]
    overall = all(v["pass"] for v in verdicts)
    (out / "continuation.json").write_text(json.dumps({
        "rule": {"rel_loss_tol": REL_LOSS_TOL, "noise_factor": NOISE_FACTOR, "abs_linf_tol": ABS_LINF_TOL,
                 "n_images": N_IMAGES, "eps": EPS, "steps": [STEPS_FIRST, STEPS_TOTAL]},
        "rows": rows, "verdicts": verdicts, "continuation_equivalent": overall}, indent=1))
    log.info("CONTINUATION EQUIVALENT: %s", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
