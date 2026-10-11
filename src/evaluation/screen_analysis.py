"""Analysis of the 50-image screen (docs/phase2_screen_plan.md section 4).

Pure functions on result rows. Everything computed here is exploratory: one Holm family
over the screen's paired tests, labelled as such. The pre-registered confirmatory family
stays in configs/analysis_plan.yaml and N = 200.

The two Dangerzone adapters are separate conditions throughout: `dangerzone_ll`
(lossless adapter) and `dangerzone` (legacy adapter, which embeds a JPEG 75).
"""
from __future__ import annotations

import numpy as np

from .adaptive_compare import condition_vector, paired_test
from .statistics import clopper_pearson, holm_bonferroni

ATTACK_LABEL = {"jpeg50": "adaptive through JPEG 50",
                "dangerzone_ll": "adaptive through Dangerzone (lossless adapter)",
                "dangerzone": "adaptive through Dangerzone (legacy adapter)"}
DEFENSE_LABEL = {"D0": "no defense", "jpeg90": "JPEG 90", "jpeg75": "JPEG 75", "jpeg50": "JPEG 50",
                 "dangerzone_ll": "Dangerzone (lossless adapter)",
                 "dangerzone": "Dangerzone (legacy adapter)"}


def _only(rows: list[dict], ids: set) -> list[dict]:
    return [r for r in rows if r["image_id"] in ids]


def rate_rows(non_adaptive: list[dict], adaptive: dict[str, list[dict]], eps_list,
              ids: set, alpha: float = 0.05) -> list[dict]:
    """One row per (attack, epsilon, scored defense): k, n, exact interval."""
    out = []
    sets = [("non_adaptive", _only(non_adaptive, ids))] + \
           [(f"adaptive_{k}", _only(v, ids)) for k, v in adaptive.items()]
    for label, rows in sets:
        for eps in eps_list:
            for d in sorted({r["defense"] for r in rows}):
                vec = condition_vector(rows, d, eps)
                if not vec:
                    continue
                k, n = sum(vec.values()), len(vec)
                p, lo, hi = clopper_pearson(k, n, alpha)
                out.append({"attack": label, "epsilon": eps, "scored_through": d,
                            "k": k, "n": n, "rate": p, "ci95": [lo, hi]})
    return out


def exploratory_family(non_adaptive: list[dict], adaptive: dict[str, list[dict]], eps_list,
                       ids: set, alpha: float = 0.05, mei: float = 0.15) -> dict:
    """Paired tests: adaptive vs non-adaptive through the same real defense, and
    Dangerzone vs JPEG 50 under adaptive attack (each adapter separately)."""
    non = _only(non_adaptive, ids)
    ada = {k: _only(v, ids) for k, v in adaptive.items()}
    tests, skipped = [], []

    def add(tid, va, vb, meta):
        res = paired_test(va, vb, alpha)
        if res is None:
            skipped.append({"id": tid, "missing": ["no paired images"]})
            return
        res.update({"id": tid, **meta})
        tests.append(res)

    for eps in eps_list:
        for atk, rows in ada.items():
            add(f"adaptive_vs_non_adaptive[{atk}]@{eps:.4f}",
                condition_vector(rows, atk, eps), condition_vector(non, atk, eps),
                {"epsilon": eps, "kind": "adaptivity", "defense": atk})
        for dz in ("dangerzone_ll", "dangerzone"):
            if dz in ada and "jpeg50" in ada:
                add(f"{dz}_vs_jpeg50_under_adaptive@{eps:.4f}",
                    condition_vector(ada[dz], dz, eps), condition_vector(ada["jpeg50"], "jpeg50", eps),
                    {"epsilon": eps, "kind": "cdr_vs_jpeg", "a": dz, "b": "jpeg50"})
    holm = holm_bonferroni({t["id"]: t["p_raw"] for t in tests}, alpha) if tests else {}
    for t in tests:
        t["p_adjusted"] = holm[t["id"]]["p_adjusted"]
        t["reject_h0"] = holm[t["id"]]["reject"]
        lo, hi = t["diff_ci95"]
        t["resolves_mei"] = bool((lo > -mei and hi < mei) or t["reject_h0"])
    return {"schema": "screen_exploratory_v1", "status": "exploratory", "alpha": alpha,
            "family_size": len(tests), "minimum_effect_of_interest": mei,
            "tests": tests, "skipped": skipped}


def discordant_rate(tests: list[dict]) -> dict:
    """Discordant-pair share per test, the input the N = 200 power calculation needs."""
    out = {}
    for t in tests:
        n = t["n_paired"]
        out[t["id"]] = (t["b_only"] + t["c_only"]) / n if n else None
    return out


def preservation(clean_rows: list[dict], ids: set, alpha: float = 0.05) -> list[dict]:
    """Share of clean images whose answer is unchanged by the defense (model-relative)."""
    out = []
    rows = _only(clean_rows, ids)
    for d in sorted({r["defense"] for r in rows}):
        sel = [r for r in rows if r["defense"] == d and not r.get("error")]
        k = sum(int(r["preserved"]) for r in sel)
        p, lo, hi = clopper_pearson(k, len(sel), alpha)
        out.append({"defense": d, "k": k, "n": len(sel), "rate": p, "ci95": [lo, hi]})
    return out


def runtime_summary(attack_jobs: list[dict]) -> dict:
    t = [j["attack_runtime_s"] for j in attack_jobs if j.get("attack_runtime_s") is not None]
    return {"n": len(t), "median_s": float(np.median(t)) if t else None,
            "total_h": float(np.sum(t) / 3600) if t else 0.0}
