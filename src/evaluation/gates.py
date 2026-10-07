"""Validity gates G1 to G5 (docs/phase2_preregistration.md), as code.

Each gate returns {"gate", "passed", "detail"}. Gates look at attack-job metadata and
at counts; they never look at attack success, so they cannot be tuned to a result.
"""
from __future__ import annotations

import numpy as np


def g1_defense_in_loop(attack_jobs: list[dict], steps: int) -> dict:
    bad = []
    for j in attack_jobs:
        ok = (j.get("defense_in_loop") is not None
              and j.get("defense_calls") == steps
              and j.get("diag_grad_finite") is True
              and (j.get("diag_grad_abs_mean") or 0) > 0
              and j.get("diag_loss_raw_clean") != j.get("diag_loss_defended_clean"))
        if not ok:
            bad.append(j.get("image_id"))
    return {"gate": "G1", "passed": bool(attack_jobs) and not bad,
            "detail": {"n_jobs": len(attack_jobs), "failing_images": bad}}


def g2_surrogate_fidelity(report: dict, min_psnr_db: float = 40.0) -> dict:
    """`report` is the fit report (heldout clean + adv PSNR means)."""
    held = report.get("heldout", {})
    vals = {k: v.get("psnr_db_mean") for k, v in held.items()}
    ok = bool(vals) and all(v is not None and v >= min_psnr_db for v in vals.values())
    return {"gate": "G2", "passed": ok, "detail": {"heldout_psnr_db": vals,
                                                   "threshold_db": min_psnr_db}}


def g3_attack_optimizes(adaptive_loss: dict, oblivious_loss: dict,
                        min_share: float = 0.80) -> dict:
    """Both dicts map image key -> loss through the same defense. Lower is stronger."""
    keys = sorted(set(adaptive_loss) & set(oblivious_loss))
    if not keys:
        return {"gate": "G3", "passed": False, "detail": {"n": 0}}
    wins = sum(adaptive_loss[k] < oblivious_loss[k] for k in keys)
    share = wins / len(keys)
    return {"gate": "G3", "passed": share >= min_share,
            "detail": {"n": len(keys), "adaptive_lower_share": share,
                       "median_adaptive": float(np.median([adaptive_loss[k] for k in keys])),
                       "median_oblivious": float(np.median([oblivious_loss[k] for k in keys]))}}


def g4_no_silent_loss(attack_jobs: list[dict], defense_rows: list[dict],
                      expected_images: int, expected_eps: int, eps_tol: float = 1e-4) -> dict:
    failed = [r for r in defense_rows if r.get("error")]
    over = [j.get("image_id") for j in attack_jobs
            if j.get("linf") is not None and j["linf"] > j["epsilon"] + eps_tol]
    complete = len(attack_jobs) == expected_images * expected_eps
    return {"gate": "G4", "passed": not failed and not over and complete,
            "detail": {"failed_real_defense_rows": len(failed), "outside_ball": over,
                       "attack_jobs": len(attack_jobs),
                       "expected": expected_images * expected_eps}}


def g5_baseline_reproduction(new_rows: list[dict], base_rows: list[dict],
                             min_agreement: float = 0.99) -> dict:
    key = lambda r: (r["image_id"], round(float(r["epsilon"]), 6))   # noqa: E731
    b = {key(r): r["answer"] for r in base_rows if r.get("defense") == "D0" and not r.get("error")}
    n = {key(r): r["answer"] for r in new_rows if r.get("defense") == "D0" and not r.get("error")}
    shared = sorted(set(b) & set(n))
    agree = sum(b[k] == n[k] for k in shared) / len(shared) if shared else 0.0
    return {"gate": "G5", "passed": bool(shared) and agree >= min_agreement,
            "detail": {"n_shared": len(shared), "agreement": agree}}
