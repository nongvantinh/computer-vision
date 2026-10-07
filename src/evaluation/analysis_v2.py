"""Plan-driven statistical analysis of one run (v2).

Pure functions over result rows, no file I/O and no model, so every statistic is
reproducible from raw rows and unit-testable. The families of comparisons come from
`configs/analysis_plan.yaml`, declared before the data are seen.

Row fields used (as written by the experiment runner):
  defense rows : image_id, epsilon, defense, target_success, restored, error
  clean rows   : image_id, defense, preserved, error
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .statistics import (clopper_pearson, mcnemar_exact, newcombe_paired_diff,
                         holm_bonferroni)


def _rate(k: int, n: int, alpha: float) -> dict:
    p, lo, hi = clopper_pearson(k, n, alpha)
    return {"k": int(k), "n": int(n), "rate": p, "ci95": [lo, hi],
            "ci_method": "clopper-pearson"}


def _vectors(defrows: list[dict]) -> dict:
    """{(defense, epsilon_key): {image_id: (target_success, restored)}}."""
    out: dict = {}
    for r in defrows:
        if r.get("error"):
            continue
        key = (r["defense"], round(float(r["epsilon"]), 6))
        out.setdefault(key, {})[r["image_id"]] = (int(r["target_success"]),
                                                  int(r["restored"]))
    return out


def primary_pairs(plan: dict, defenses: set[str]) -> tuple[list[tuple], list[dict]]:
    """Resolve the plan's primary family against the defenses present in a run.

    Returns (pairs, skipped); each pair is (a, b, label) meaning "a vs b".
    """
    base = plan.get("baseline_defense", "D0")
    prim = plan.get("primary", {})
    pairs, skipped, seen = [], [], set()

    def add(a, b, why):
        if a in defenses and b in defenses:
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b, why))
        else:
            missing = [x for x in (a, b) if x not in defenses]
            skipped.append({"pair": [a, b], "reason": why, "missing": missing})

    for d in prim.get("defense_vs_baseline", []):
        add(d, base, "defense_vs_baseline")
    cg = prim.get("cdr_vs_generic", {})
    for c in cg.get("cdr", []):
        for g in cg.get("generic", []):
            add(c, g, "cdr_vs_generic")
    for a, b in prim.get("cdr_vs_control", []):
        add(a, b, "cdr_vs_control")
    return pairs, skipped


def analyze_run(defrows: list[dict], cleanrows: list[dict], plan: dict,
                meta: dict | None = None) -> dict:
    alpha = float(plan.get("alpha", 0.05))
    defrows = [r for r in defrows if not r.get("error")]
    cleanrows = [r for r in cleanrows if not r.get("error")]
    vec = _vectors(defrows)
    defenses = sorted({d for d, _ in vec})
    epsilons = sorted({e for _, e in vec})

    # per-cell rates (ASR and restoration) with exact intervals
    cells = {}
    for (d, e), per_img in sorted(vec.items()):
        ts = [v[0] for v in per_img.values()]
        rs = [v[1] for v in per_img.values()]
        cells[f"{d}@{e:.4f}"] = {"defense": d, "epsilon": e,
                                 "asr": _rate(sum(ts), len(ts), alpha),
                                 "restoration": _rate(sum(rs), len(rs), alpha)}

    preservation = {}
    for d in sorted({r["defense"] for r in cleanrows}):
        ks = [int(r["preserved"]) for r in cleanrows if r["defense"] == d]
        preservation[d] = _rate(sum(ks), len(ks), alpha)

    def compare(a: str, b: str, e: float):
        va, vb = vec.get((a, e)), vec.get((b, e))
        if not va or not vb:
            return None
        ids = sorted(set(va) & set(vb))      # pair by image; keep only shared images
        if not ids:
            return None
        xa = np.array([va[i][0] for i in ids], dtype=bool)
        xb = np.array([vb[i][0] for i in ids], dtype=bool)
        mc = mcnemar_exact(xa, xb)
        diff, lo, hi = newcombe_paired_diff(xa, xb, alpha)
        return {"a": a, "b": b, "epsilon": e, "n_paired": len(ids),
                "a_successes": int(xa.sum()), "b_successes": int(xb.sum()),
                "b_only": mc.b, "c_only": mc.c, "paired": True,
                "test": "exact-mcnemar", "p_raw": mc.p_value,
                "diff_a_minus_b": diff, "diff_ci95": [lo, hi],
                "diff_ci_method": "newcombe-10"}

    prim_pairs, skipped = primary_pairs(plan, set(defenses))
    primary, prim_keys = [], set()
    for a, b, why in prim_pairs:
        for e in epsilons:
            c = compare(a, b, e)
            if c:
                c["family"] = "primary"; c["hypothesis"] = why
                c["id"] = f"{a}_vs_{b}@{e:.4f}"
                primary.append(c); prim_keys.add((a, b))
    exploratory = []
    if plan.get("exploratory", {}).get("all_other_pairs", False):
        for a, b in combinations(defenses, 2):
            if (a, b) in prim_keys or (b, a) in prim_keys:
                continue
            for e in epsilons:
                c = compare(a, b, e)
                if c:
                    c["family"] = "exploratory"; c["hypothesis"] = "exploratory"
                    c["id"] = f"{a}_vs_{b}@{e:.4f}"
                    exploratory.append(c)

    for fam in (primary, exploratory):
        holm = holm_bonferroni({c["id"]: c["p_raw"] for c in fam}, alpha) if fam else {}
        for c in fam:
            c["p_adjusted"] = holm[c["id"]]["p_adjusted"]
            c["reject_h0"] = holm[c["id"]]["reject"]

    return {"schema": "analysis_v2", "plan_version": plan.get("version"),
            "alpha": alpha, "meta": meta or {},
            "defenses": defenses, "epsilons": epsilons,
            "cells": cells, "preservation": preservation,
            "primary": primary, "exploratory": exploratory, "skipped": skipped,
            "family_sizes": {"primary": len(primary), "exploratory": len(exploratory)}}
