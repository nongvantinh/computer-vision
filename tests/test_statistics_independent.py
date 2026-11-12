"""Independent verification of the confirmatory statistics.

Each check re-derives a quantity with a different method than the code under test
(math.comb sums, a bootstrap, a hand-written step-down, exact enumeration), so a shared
bug in `src/evaluation/statistics.py` would not pass both. The expected values were
fixed before the checks were run against the implementation.
"""
import math

import numpy as np
import pytest

from src.evaluation import statistics as S


def _exact_two_sided(x: int, n: int) -> float:
    """Two-sided exact binomial p at 0.5 by direct summation (no scipy)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[x] * (1 + 1e-12)))


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (5, 0), (3, 3), (18, 0), (43, 3), (2, 9), (113, 2)])
def test_exact_mcnemar_matches_direct_summation(b, c):
    a = np.array([1] * b + [0] * c + [1] * 4 + [0] * 7, dtype=bool)
    bb = np.array([0] * b + [1] * c + [1] * 4 + [0] * 7, dtype=bool)
    r = S.mcnemar_exact(a, bb)
    assert (r.b, r.c) == (b, c)
    assert r.p_value == pytest.approx(_exact_two_sided(min(b, c), b + c), rel=1e-9, abs=1e-15)


def test_zero_discordant_gives_p_one_and_a_nondegenerate_interval():
    a = np.array([1] * 10 + [0] * 190, dtype=bool)
    assert S.mcnemar_exact(a, a.copy()).p_value == 1.0
    d, lo, hi = S.newcombe_paired_diff(a, a.copy())
    assert d == 0.0 and lo < 0.0 < hi            # interval does not collapse at zero discordance


def test_zero_successes_both_arms():
    z = np.zeros(200, dtype=bool)
    d, lo, hi = S.newcombe_paired_diff(z, z)
    assert d == 0.0 and lo <= 0.0 <= hi and lo >= -1 and hi <= 1


def test_clopper_pearson_known_values():
    # 0 of 200: upper bound = 1 - (alpha/2)^(1/n), closed form
    _, lo, hi = S.clopper_pearson(0, 200)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 200), rel=1e-9)
    _, lo, hi = S.clopper_pearson(200, 200)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 200), rel=1e-9)


def _wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _newcombe10(n11, n10, n01, n00):
    """Newcombe (1998) method 10, written from the paper's formulas, not the repo code."""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = _wilson(n11 + n10, n)
    l2, u2 = _wilson(n11 + n01, n)
    a, b_, c, d = n11, n10, n01, n00
    den = (a + b_) * (c + d) * (a + c) * (b_ + d)
    phi = 0.0 if den == 0 else (a * d - b_ * c) / math.sqrt(den)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return diff, max(-1, lo), min(1, hi)


@pytest.mark.parametrize("cells", [(0, 0, 0, 200), (0, 18, 0, 182), (0, 42, 0, 158), (3, 150, 0, 47),
                                   (2, 111, 2, 85), (40, 20, 10, 130), (5, 0, 12, 183)])
def test_newcombe_matches_hand_implementation(cells):
    n11, n10, n01, n00 = cells
    a = np.array([1] * (n11 + n10) + [0] * (n01 + n00), dtype=bool)
    b = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=bool)
    got = S.newcombe_paired_diff(a, b)
    want = _newcombe10(n11, n10, n01, n00)
    assert got == pytest.approx(want, abs=1e-9)


def test_newcombe_is_antisymmetric():
    rng = np.random.default_rng(0)
    a, b = rng.random(200) < 0.3, rng.random(200) < 0.1
    d1, lo1, hi1 = S.newcombe_paired_diff(a, b)
    d2, lo2, hi2 = S.newcombe_paired_diff(b, a)
    assert (d1, lo1, hi1) == pytest.approx((-d2, -hi2, -lo2), abs=1e-12)


def test_newcombe_agrees_with_a_bootstrap_where_the_bootstrap_is_valid():
    rng = np.random.default_rng(1)
    a = rng.random(200) < 0.45
    b = np.where(rng.random(200) < 0.7, a, rng.random(200) < 0.2)
    d, lo, hi = S.newcombe_paired_diff(a, b)
    boots = []
    for _ in range(20000):
        i = rng.integers(0, 200, 200)
        boots.append(a[i].mean() - b[i].mean())
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    assert abs(lo - blo) < 0.025 and abs(hi - bhi) < 0.025


def test_newcombe_paired_coverage_is_near_nominal():
    """Simulated coverage of the 95% interval for correlated binary pairs (N=200)."""
    rng = np.random.default_rng(2)
    n, hits, reps = 200, 0, 1500
    pa, pb, rho = 0.30, 0.10, 0.6
    # true difference under a common-latent construction
    for _ in range(reps):
        shared = rng.random(n) < rho
        u = rng.random(n)
        a = np.where(shared, u < pa, rng.random(n) < pa)
        b = np.where(shared, u < pb, rng.random(n) < pb)
        d, lo, hi = S.newcombe_paired_diff(a, b)
        hits += lo <= (pa - pb) <= hi
    assert 0.925 <= hits / reps <= 0.975


def _holm(p: dict, alpha=0.05):
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, adj, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        adj[k] = run
    return adj


def test_holm_matches_hand_stepdown():
    p = {"a": 7.6e-6, "b": 1e-17, "c": 4.6e-10, "d": 4.5e-13, "e": 1e-45, "f": 0.3}
    got = S.holm_bonferroni(p)
    want = _holm(p)
    for k in p:
        assert got[k]["p_adjusted"] == pytest.approx(want[k], rel=1e-12)
    assert got["f"]["reject"] is False and got["a"]["reject"] is True


def test_holm_is_monotone_and_never_below_raw():
    rng = np.random.default_rng(3)
    p = {str(i): float(x) for i, x in enumerate(rng.random(12) ** 3)}
    got = S.holm_bonferroni(p)
    order = sorted(p, key=p.get)
    adj = [got[k]["p_adjusted"] for k in order]
    assert adj == sorted(adj)
    assert all(got[k]["p_adjusted"] >= p[k] for k in p)


def test_registered_power_claim_by_exact_enumeration():
    """Power of the exact McNemar test at N=200, discordant rate 0.3, effect 0.15, alpha 0.05/6.

    Registered value (simulated, 4000 runs): 0.89. Re-derived here by exact enumeration.
    """
    n, pd, eff, alpha = 200, 0.30, 0.15, 0.05 / 6
    p_b = (pd + eff) / 2 / pd                      # share of discordant pairs favouring A
    # per-D critical region of the exact two-sided test
    from scipy.stats import binom
    power = 0.0
    for d in range(1, n + 1):
        w = binom.pmf(d, n, pd)
        if w < 1e-12:
            continue
        pmf0 = [math.comb(d, i) / 2 ** d for i in range(d + 1)]
        rej = 0.0
        for x in range(d + 1):
            p_two = min(1.0, sum(q for q in pmf0 if q <= pmf0[x] * (1 + 1e-12)))
            if p_two < alpha:
                rej += binom.pmf(x, d, p_b)
        power += w * rej
    assert 0.84 <= power <= 0.93, power
