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
