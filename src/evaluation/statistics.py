"""Paired statistics for the CDR-vs-defense comparison (proposal §4.3).

- McNemar's test for paired binary outcomes (same images through every defense).
- Bootstrap CIs for ASR / accuracy.
- Holm-Bonferroni correction across the predefined comparison family.

Pure numpy/scipy; deterministic given a seed. Testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class McNemarResult:
    b: int          # defense A success, defense B failure
    c: int          # defense A failure, defense B success
    statistic: float
    p_value: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar(a_success: np.ndarray, b_success: np.ndarray,
            exact_threshold: int = 25) -> McNemarResult:
    """Paired McNemar test comparing two defenses on the SAME images.

    a_success, b_success: boolean arrays (True = attack succeeded / whatever the
    binary outcome is), aligned per image.
    Uses the exact binomial test when discordant pairs are few, else the
    continuity-corrected chi-square.
    """
    a = np.asarray(a_success).astype(bool)
    b = np.asarray(b_success).astype(bool)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    b_only = int(np.sum(a & ~b))   # A yes, B no
    c_only = int(np.sum(~a & b))   # A no, B yes
    n = b_only + c_only
    if n == 0:
        return McNemarResult(b_only, c_only, 0.0, 1.0, "no-discordant-pairs")
    if n < exact_threshold:
        from scipy.stats import binomtest
        p = binomtest(min(b_only, c_only), n, 0.5, alternative="two-sided").pvalue
        stat = float(min(b_only, c_only))
        return McNemarResult(b_only, c_only, stat, float(p), "exact-binomial")
    from scipy.stats import chi2
    stat = (abs(b_only - c_only) - 1) ** 2 / n  # continuity correction
    p = float(chi2.sf(stat, df=1))
    return McNemarResult(b_only, c_only, float(stat), p, "chi2-continuity")


def bootstrap_ci(outcomes: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 1234, statistic=np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a per-sample statistic (default: mean = rate).

    Returns (point_estimate, lo, hi) for a (1-alpha) interval.
    """
    x = np.asarray(outcomes, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(x))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = statistic(x[idx], axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_diff_ci(a_outcomes: np.ndarray, b_outcomes: np.ndarray,
                   n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 1234) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference in rate (a - b), resampling images."""
    a = np.asarray(a_outcomes, dtype=float)
    b = np.asarray(b_outcomes, dtype=float)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of comparisons.

    Returns per-comparison {p_raw, p_adjusted, reject}.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_adj = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev_adj)  # enforce monotonic non-decreasing adjusted p
        prev_adj = adj
        out[name] = {"p_raw": float(p), "p_adjusted": float(adj),
                     "reject": bool(adj < alpha)}
    return out


# --------------------------------------------------------------------------- #
# Exact methods used by the v2 analysis (no bootstrap degeneracy at 0 or n).
# --------------------------------------------------------------------------- #

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Exact binomial (Clopper-Pearson) interval. Returns (rate, lo, hi).

    Unlike a percentile bootstrap, the interval is not degenerate at k=0 or k=n:
    for k=0 and n=200 the 95% upper bound is about 0.018.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    from scipy.stats import beta
    k = int(k)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (k / n, lo, hi)


def mcnemar_exact(a_success: np.ndarray, b_success: np.ndarray) -> McNemarResult:
    """Exact two-sided McNemar test (binomial on the discordant pairs), always.

    b = A succeeded and B failed; c = A failed and B succeeded. The v2 analysis uses
    the exact form for every n so the reported test does not switch method with the
    number of discordant pairs.
    """
    a = np.asarray(a_success).astype(bool)
    b = np.asarray(b_success).astype(bool)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    b_only = int(np.sum(a & ~b))
    c_only = int(np.sum(~a & b))
    n = b_only + c_only
    if n == 0:
        return McNemarResult(b_only, c_only, 0.0, 1.0, "exact-binomial (no discordant pairs)")
    from scipy.stats import binomtest
    p = float(binomtest(min(b_only, c_only), n, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(b_only, c_only, float(min(b_only, c_only)), p, "exact-binomial")


def _wilson(k: float, n: int, z: float) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def newcombe_paired_diff(a_success: np.ndarray, b_success: np.ndarray,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Newcombe (1998, method 10) interval for a paired difference in proportions.

    Returns (diff, lo, hi) for rate(A) - rate(B) on the SAME items. It stays
    non-degenerate when there are no discordant pairs, which a resampling interval
    does not.
    """
    from scipy.stats import norm
    a = np.asarray(a_success).astype(bool)
    b = np.asarray(b_success).astype(bool)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    n = a.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    z = float(norm.ppf(1 - alpha / 2))
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b)); n00 = int(np.sum(~a & ~b))
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = _wilson(n11 + n10, n, z)
    l2, u2 = _wilson(n11 + n01, n, z)
    denom = float(np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return float(d), float(max(-1.0, lo)), float(min(1.0, hi))
