#!/usr/bin/env python
"""Power of the paired exact McNemar test for the primary comparisons (pre-declared).

For N paired images, a discordant-pair rate d, and a true difference in success rates
delta (A minus B), the number of discordant pairs is Binomial(N, d) and, given n of
them, the count where only A succeeds is Binomial(n, (d + delta) / (2 d)). Power is the
share of simulated datasets whose exact two-sided McNemar p-value is below alpha / m
(Holm's most stringent step with m primary tests).

    python scripts/power_analysis.py
"""
import argparse
import json

import numpy as np
from scipy.stats import binomtest


def power(n_images, d, delta, alpha, sims, rng):
    p_b = (d + delta) / (2 * d)
    disc = rng.binomial(n_images, d, size=sims)
    b = rng.binomial(disc, p_b)
    cache = {}
    hits = 0
    for n, bb in zip(disc.tolist(), b.tolist()):
        if n == 0:
            continue
        k = (n, min(bb, n - bb))
        if k not in cache:
            cache[k] = binomtest(k[1], n, 0.5).pvalue
        hits += cache[k] <= alpha
    return hits / sims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 6, 12])
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 200, 300])
    ap.add_argument("--d", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--delta", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for m in args.m:
        for d in args.d:
            for delta in args.delta:
                if delta > d:
                    continue                      # discordant rate must cover the effect
                for n in args.n:
                    rows.append({"m": m, "alpha_per_test": args.alpha / m, "d": d,
                                 "delta": delta, "N": n,
                                 "power": power(n, d, delta, args.alpha / m, args.sims, rng)})
    for r in rows:
        print(f"m={r['m']:2d} d={r['d']:.1f} delta={r['delta']:.2f} N={r['N']:3d} power={r['power']:.2f}")
    if args.out:
        json.dump({"seed": args.seed, "sims": args.sims, "rows": rows},
                  open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
