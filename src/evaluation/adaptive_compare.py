"""Confirmatory comparisons across attack conditions (adaptive vs oblivious).

Each condition is a set of per-image results for one (attack, REAL defense) pair.
Tests pair the same images across two conditions at one budget. Pure functions: rows
in, dicts out, so the confirmatory numbers are reproducible from raw result files.
"""
from __future__ import annotations

import numpy as np

from .statistics import (clopper_pearson, holm_bonferroni, mcnemar_exact,
                         newcombe_paired_diff)


def condition_vector(rows: list[dict], defense: str, eps: float) -> dict:
    """{image_id: target_success} for one defense at one epsilon (errored rows dropped)."""
    out = {}
    for r in rows:
        if r.get("error") or r.get("defense") != defense:
            continue
        if abs(float(r["epsilon"]) - eps) < 1e-6:
            out[r["image_id"]] = int(r["target_success"])
    return out


def paired_test(va: dict, vb: dict, alpha: float = 0.05) -> dict | None:
    ids = sorted(set(va) & set(vb))
    if not ids:
        return None
    a = np.array([va[i] for i in ids], dtype=bool)
    b = np.array([vb[i] for i in ids], dtype=bool)
    mc = mcnemar_exact(a, b)
    d, lo, hi = newcombe_paired_diff(a, b, alpha)
    ra, rb = clopper_pearson(int(a.sum()), a.size, alpha), clopper_pearson(int(b.sum()), b.size, alpha)
    return {"n_paired": len(ids), "n_a_only_images": len(set(va) - set(vb)),
            "n_b_only_images": len(set(vb) - set(va)),
            "a_rate": ra[0], "a_ci95": [ra[1], ra[2]], "b_rate": rb[0], "b_ci95": [rb[1], rb[2]],
            "b_only": mc.b, "c_only": mc.c, "p_raw": mc.p_value,
            "diff_a_minus_b": d, "diff_ci95": [lo, hi]}


def run_confirmatory(plan: dict, conditions: dict[str, list[dict]],
                     defense_of: dict[str, str]) -> dict:
    """Evaluate the plan's `confirmatory_adaptive` family.

    conditions : label -> result rows (e.g. "ada_jpeg50", "obl_jpeg50")
    defense_of : label -> name of the (real) defense those rows were scored through
    Labels follow the convention  ada_<defense> / obl_<defense>.
    """
    spec = plan["confirmatory_adaptive"]
    alpha = float(plan.get("alpha", 0.05))
    mei = float(spec.get("minimum_effect_of_interest", 0.15))
    tests, skipped = [], []
    for eps in spec["epsilons"]:
        for t in spec["tests"]:
            if t["id"] == "adaptive_vs_oblivious":
                la, lb = f"ada_{t['defense']}", f"obl_{t['defense']}"
                tid = f"adaptive_vs_oblivious[{t['defense']}]@{eps:.4f}"
            elif t["id"] == "cdr_vs_jpeg_adaptive":
                la, lb = f"ada_{t['a']}", f"ada_{t['b']}"
                tid = f"{t['id']}[{t['a']} vs {t['b']}]@{eps:.4f}"
            else:
                raise ValueError(f"unknown confirmatory test {t['id']!r}")
            if la not in conditions or lb not in conditions:
                skipped.append({"id": tid, "missing": [x for x in (la, lb) if x not in conditions]})
                continue
            res = paired_test(condition_vector(conditions[la], defense_of[la], eps),
                              condition_vector(conditions[lb], defense_of[lb], eps), alpha)
            if res is None:
                skipped.append({"id": tid, "missing": ["no paired images"]})
                continue
            res.update({"id": tid, "epsilon": eps, "a": la, "b": lb})
            tests.append(res)
    holm = holm_bonferroni({t["id"]: t["p_raw"] for t in tests}, alpha) if tests else {}
    for t in tests:
        t["p_adjusted"] = holm[t["id"]]["p_adjusted"]
        t["reject_h0"] = holm[t["id"]]["reject"]
        lo, hi = t["diff_ci95"]
        # a null is only informative if the interval excludes effects as large as the MEI
        t["resolves_mei"] = bool(lo > -mei and hi < mei) or t["reject_h0"]
    return {"schema": "confirmatory_adaptive_v1", "plan_version": plan.get("version"),
            "alpha": alpha, "minimum_effect_of_interest": mei,
            "family_size": len(tests), "tests": tests, "skipped": skipped}
