"""v2 analysis: exact intervals, exact McNemar, plan-driven families, determinism."""
import json
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.statistics import (clopper_pearson, mcnemar_exact,
                                       newcombe_paired_diff)
from src.evaluation.analysis_v2 import analyze_run, primary_pairs

PLAN = yaml.safe_load((Path(__file__).resolve().parents[1]
                       / "configs" / "analysis_plan.yaml").read_text())


def test_clopper_pearson_zero_successes_is_not_degenerate():
    p, lo, hi = clopper_pearson(0, 200)
    assert p == 0.0 and lo == 0.0
    assert abs(hi - (1 - 0.025 ** (1 / 200))) < 1e-9     # closed form for k=0
    p, lo, hi = clopper_pearson(200, 200)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 200)) < 1e-9


def test_clopper_pearson_brackets_the_rate():
    for k in (1, 7, 100, 199):
        p, lo, hi = clopper_pearson(k, 200)
        assert lo < p < hi


def test_mcnemar_exact_always_exact_even_for_many_pairs():
    a = np.array([True] * 111 + [False] * 89)
    b = np.zeros(200, bool)
    r = mcnemar_exact(a, b)
    assert r.method == "exact-binomial"
    assert (r.b, r.c) == (111, 0)
    assert r.p_value < 1e-30


def test_mcnemar_exact_no_discordant_is_one():
    z = np.zeros(50, bool)
    assert mcnemar_exact(z, z).p_value == 1.0


def test_newcombe_interval_contains_zero_for_identical_outcomes():
    z = np.zeros(200, bool)
    d, lo, hi = newcombe_paired_diff(z, z)
    assert d == 0.0 and lo < 0.0 < hi          # not the degenerate [0, 0]
    assert hi < 0.03


def test_newcombe_sign_and_bounds():
    a = np.array([True] * 60 + [False] * 140)
    b = np.zeros(200, bool)
    d, lo, hi = newcombe_paired_diff(a, b)
    assert abs(d - 0.3) < 1e-12 and 0.2 < lo < d < hi < 0.4


def _rows(n=200, eps=(0.0157, 0.0627)):
    """Synthetic run: D0 succeeds on 30%/60% of images; defenses defeat nearly all."""
    rng = np.random.default_rng(0)
    defrows, cleanrows = [], []
    for e in eps:
        d0 = rng.random(n) < (0.3 if e < 0.03 else 0.6)
        for i in range(n):
            iid = f"img{i:03d}"
            for d, succ in (("D0", d0[i]),
                            ("jpeg50", d0[i] and i == 0),       # one survivor
                            ("dangerzone", False),
                            ("adapter_only", d0[i] and i < 3)):
                defrows.append({"image_id": iid, "epsilon": e, "defense": d,
                                "target_success": int(succ), "restored": int(not succ),
                                "error": None})
    for i in range(n):
        for d in ("D0", "jpeg50", "dangerzone", "adapter_only"):
            cleanrows.append({"image_id": f"img{i:03d}", "defense": d,
                              "preserved": int(d == "D0" or i % 12 != 0), "error": None})
    return defrows, cleanrows


def test_plan_resolution_skips_missing_defenses():
    pairs, skipped = primary_pairs(PLAN, {"D0", "jpeg50", "dangerzone", "adapter_only"})
    names = {(a, b) for a, b, _ in pairs}
    assert ("dangerzone", "D0") in names and ("jpeg50", "D0") in names
    assert ("dangerzone", "jpeg50") in names
    assert ("dangerzone", "adapter_only") in names
    assert not any("mdcore" in p for p in names)
    assert any("mdcore" in s["missing"] for s in skipped)


def test_analyze_run_families_cells_and_holm():
    defrows, cleanrows = _rows()
    out = analyze_run(defrows, cleanrows, PLAN, {"run": "synthetic"})
    assert out["schema"] == "analysis_v2" and out["meta"]["run"] == "synthetic"
    # primary: (jpeg50, dangerzone vs D0) + (dangerzone vs jpeg50) + (dangerzone vs adapter)
    # = 4 pairs, at 2 epsilons
    assert out["family_sizes"]["primary"] == 8
    zero = out["cells"]["dangerzone@0.0627"]["asr"]
    assert zero["k"] == 0 and zero["n"] == 200 and zero["ci95"][1] > 0.01
    vs_d0 = next(c for c in out["primary"] if c["id"] == "dangerzone_vs_D0@0.0627")
    assert vs_d0["family"] == "primary" and vs_d0["test"] == "exact-mcnemar"
    assert vs_d0["b_only"] == 0 and vs_d0["c_only"] == vs_d0["b_successes"] > 0
    assert vs_d0["reject_h0"] is True and vs_d0["p_adjusted"] >= vs_d0["p_raw"]
    # identical outcomes between the two stable defenses cannot be rejected
    same = next(c for c in out["primary"] if c["id"] == "dangerzone_vs_adapter_only@0.0157")
    assert same["p_adjusted"] <= 1.0
    # preservation uses exact intervals and excludes nothing silently
    assert out["preservation"]["D0"]["rate"] == 1.0
    assert out["preservation"]["jpeg50"]["n"] == 200


def test_exploratory_family_is_separate_and_labelled():
    defrows, cleanrows = _rows()
    out = analyze_run(defrows, cleanrows, PLAN)
    assert all(c["family"] == "exploratory" for c in out["exploratory"])
    prim_ids = {c["id"] for c in out["primary"]}
    assert not prim_ids & {c["id"] for c in out["exploratory"]}


def test_analysis_is_deterministic_and_json_serialisable():
    defrows, cleanrows = _rows()
    a = analyze_run(defrows, cleanrows, PLAN)
    b = analyze_run(list(reversed(defrows)), list(reversed(cleanrows)), PLAN)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_errored_rows_are_excluded_not_silently_counted_as_failures():
    defrows, cleanrows = _rows(n=20)
    defrows[0]["error"] = "boom"
    out = analyze_run(defrows, cleanrows, PLAN)
    n_cell = out["cells"][f"{defrows[0]['defense']}@{defrows[0]['epsilon']:.4f}"]["asr"]["n"]
    assert n_cell == 19
