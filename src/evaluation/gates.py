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


def g5_reproducibility(new_rows: list[dict], base_rows: list[dict], *,
                       psnr_tol_db: float = 1e-3, min_success_agreement: float = 0.99,
                       min_answer_agreement: float = 0.99) -> list[dict]:
    """G5 as originally written, plus G5a and G5b (protocol revision 2026-10-11).

    D0 rows of two runs are matched on (image, epsilon). The originally registered G5
    (raw-answer agreement over ALL matched rows) is returned unchanged and may fail.

    G5a is the primary reproducibility gate for the endpoint of the study: agreement of
    `target_success` on rows whose model input is verifiably the same. Input identity is
    `adv_sha256` when both rows carry it, otherwise equal `psnr_vs_clean` within
    `psnr_tol_db` (a pixel-level proxy for runs recorded before hashes existed). Rows with
    different inputs are listed, not scored: a different image is not a reproduction test.
    G5b keeps raw-answer agreement as a diagnostic on the same identical-input rows.
    """
    key = lambda r: (r["image_id"], round(float(r["epsilon"]), 6))   # noqa: E731
    ok = lambda r: r.get("defense") == "D0" and not r.get("error")  # noqa: E731
    b = {key(r): r for r in base_rows if ok(r)}
    n = {key(r): r for r in new_rows if ok(r)}
    shared = sorted(set(b) & set(n))

    def same_input(x, y):
        if x.get("adv_sha256") and y.get("adv_sha256"):
            return x["adv_sha256"] == y["adv_sha256"]
        return abs(x["psnr_vs_clean"] - y["psnr_vs_clean"]) <= psnr_tol_db

    ident = [k for k in shared if same_input(n[k], b[k])]
    differ = [k for k in shared if k not in set(ident)]
    ans = lambda k: n[k]["answer"] == b[k]["answer"]                # noqa: E731
    suc = lambda k: int(n[k]["target_success"]) == int(b[k]["target_success"])  # noqa: E731
    frac = lambda f, ks: (sum(f(k) for k in ks) / len(ks)) if ks else 0.0       # noqa: E731

    g5 = {"gate": "G5", "passed": bool(shared) and frac(ans, shared) >= 0.99,
          "detail": {"n_shared": len(shared), "agreement": frac(ans, shared),
                     "definition": "original: raw-answer agreement over all matched rows"}}
    disc_s = [k for k in ident if not suc(k)]
    g5a = {"gate": "G5a", "passed": bool(ident) and frac(suc, ident) >= min_success_agreement,
           "detail": {"n_identical_input": len(ident), "agreement": frac(suc, ident),
                      "discordant_identical_input": [list(k) for k in disc_s],
                      "n_input_differs": len(differ),
                      "agreement_all_matched_rows": frac(suc, shared),
                      "threshold": min_success_agreement,
                      "input_differs_keys": [list(k) for k in differ]}}
    g5b = {"gate": "G5b", "passed": bool(ident) and frac(ans, ident) >= min_answer_agreement,
           "detail": {"n_identical_input": len(ident), "agreement": frac(ans, ident),
                      "differing_answers_identical_input": sum(not ans(k) for k in ident),
                      "differing_answers_input_differs": sum(not ans(k) for k in differ),
                      "threshold": min_answer_agreement, "diagnostic": True}}
    return [g5, g5a, g5b]
