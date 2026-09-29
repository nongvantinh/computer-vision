"""Statistics: McNemar, bootstrap CI, Holm-Bonferroni."""
import numpy as np

from src.evaluation.statistics import mcnemar, bootstrap_ci, paired_diff_ci, holm_bonferroni


def test_mcnemar_no_discordant():
    a = np.array([1, 1, 0, 0], bool)
    r = mcnemar(a, a.copy())
    assert r.b == 0 and r.c == 0
    assert r.p_value == 1.0


def test_mcnemar_all_discordant_small_uses_exact():
    # A succeeds everywhere, B fails everywhere -> strong asymmetry
    a = np.ones(10, bool)
    b = np.zeros(10, bool)
    r = mcnemar(a, b)
    assert r.method == "exact-binomial"
    assert r.b == 10 and r.c == 0
    assert r.p_value < 0.01


def test_mcnemar_large_uses_chi2():
    a = np.array([True] * 40 + [False] * 60)
    b = np.array([False] * 100)
    r = mcnemar(a, b)
    assert r.method == "chi2-continuity"


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    x = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], float)
    p1, lo1, hi1 = bootstrap_ci(x, n_boot=2000, seed=42)
    p2, lo2, hi2 = bootstrap_ci(x, n_boot=2000, seed=42)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)          # deterministic
    assert lo1 <= p1 <= hi1
    assert abs(p1 - x.mean()) < 1e-9


def test_paired_diff_ci_sign():
    a = np.ones(50, float)     # ASR 1.0
    b = np.zeros(50, float)    # ASR 0.0
    point, lo, hi = paired_diff_ci(a, b, n_boot=1000, seed=1)
    assert point == 1.0 and lo == hi == 1.0


def test_holm_bonferroni_orders_and_rejects():
    p = {"c1": 0.001, "c2": 0.04, "c3": 0.5}
    out = holm_bonferroni(p, alpha=0.05)
    assert out["c1"]["reject"] is True         # 0.001*3 = 0.003 < 0.05
    assert out["c3"]["reject"] is False
    # adjusted p is monotonic non-decreasing along sorted raw p
    assert out["c1"]["p_adjusted"] <= out["c2"]["p_adjusted"] <= out["c3"]["p_adjusted"]
