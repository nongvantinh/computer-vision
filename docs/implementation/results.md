# Results

This file separates **hypotheses** (what we expect, no numbers invented) from
**measured results**. Both the preliminary pilot and the final full_200 run below are
real, measured data.

## Preliminary pilot — N=12, eps=8/255 (measured 2026-09-30)

Real end-to-end result. Raw run: `results/runs/pilot_n12_eps8/` (also on Drive).
LLaVA-1.5-7B in 4-bit on a Colab T4; targeted PGD (200 steps, step 1/255); the
Dangerzone condition was produced by the decoupled workflow (sanitize on a rootless
Podman workstation, generate on the same 4-bit model). 12 images x 5 defenses, 0
failures. This is a validation pilot, not the final study; do not over-read N=12.

Targeted ASR and utility:

| Defense | Targeted ASR | 95% CI | Preservation | Restoration |
|---|---|---|---|---|
| D0 (none) | 0.417 | [0.167, 0.667] | 1.00 | 0.33 |
| jpeg90 | 0.000 | [0, 0] | 1.00 | 1.00 |
| jpeg75 | 0.000 | [0, 0] | 1.00 | 1.00 |
| jpeg50 | 0.000 | [0, 0] | 1.00 | 0.92 |
| dangerzone | 0.000 | [0, 0] | 1.00 | 0.92 |

Paired tests at eps=8/255 (McNemar, Holm across the 7-comparison family):

| Comparison | discordant (b,c) | ASR diff [95% CI] | McNemar p | Holm p | reject |
|---|---|---|---|---|---|
| each defense vs D0 | (0, 5) | -0.417 [-0.667, -0.167] | 0.0625 | 0.4375 | no |
| dangerzone vs each jpeg | (0, 0) | 0.0 [0, 0] | 1.0 | 1.0 | no |

Reading: every defense blocked all 5 successful D0 attacks (0 reversed the other way),
but 5 discordant pairs floor the exact McNemar test at p=0.0625, so nothing reaches
significance at N=12. Dangerzone and JPEG have identical binary security outcomes here
(no discordant pairs), so the pilot shows no CDR advantage over JPEG.

Mechanism (pilot used the confounded `defended_adv - clean` residual; the final run uses
the defense-controlled `defended_adv - defended_clean`, see statistical-analysis.md).
Even so, the high-frequency fraction of the residual is informative: Dangerzone is the
only defense that lowers it (0.70 vs the injected 0.83), consistent with heavy
rasterization attenuating high-frequency adversarial structure; JPEG keeps it ~0.85-0.87
(its blocking adds high-frequency edges). Evidence, not proof of causation.

Pilot limitations: N=12 (underpowered, nothing significant); attack modest (5/12 on D0);
single epsilon; model-relative metric; mechanism metric confounded (fixed for the final
run); non-adaptive attacker only.

---

The final run's tables are further down, in the section "Final run: full_200".

## Hypotheses (from the proposal, not results)
- **H1:** a defense reduces targeted ASR vs. D0.
- **H2:** Dangerzone lies beyond the JPEG trade-off curve for >=1 operating point.
- **H3:** any robustness is consistent with removal of high-frequency perturbation
  energy (evidence, not proof of causation).
- **H4:** aggressive sanitization (Dangerzone) costs measurable preservation.
- **H5:** a defense-aware attack would recover ASR (extension; out of MVP scope).

## Defense coverage
ICDR is BLOCKED (see `icdr.md`): commercial Aspose dependency, non-compiling code,
no OSS fallback. It is not run and nothing is substituted. The measured comparison is
D0, JPEG 90/75/50, and Dangerzone. CDR-category claims scope to Dangerzone.

## Final run: full_200 (measured 2026-10-07)

Source: `results/runs/full_200/statistics.json` (sha256 `97fbf7f2...28c4`, also on Drive).
200 images x 3 budgets = 600 attacks, 5 conditions, 3000 defense evaluations, 0 failed.
LLaVA-1.5-7B in 4-bit on Colab T4, defense-unaware targeted PGD. The full write-up is
`docs/report/report.pdf`.

### Targeted ASR (percent, N=200 per cell)
| Defense | eps=4/255 | eps=8/255 | eps=16/255 |
|---|---|---|---|
| D0 | 11.0 [7.0, 15.5] | 26.0 [20.0, 32.5] | 55.5 [48.5, 62.5] |
| jpeg90 | 0.0 | 0.5 [0.0, 1.5] | 0.5 [0.0, 1.5] |
| jpeg75 | 0.0 | 0.0 | 0.5 [0.0, 1.5] |
| jpeg50 | 0.0 | 0.0 | 0.0 |
| dangerzone | 0.0 | 0.0 | 0.0 |

Of 185 attacks that worked on D0 (22, 52, 111), dangerzone and jpeg50 neutralized all,
jpeg75 left 1, jpeg90 left 2. No defense succeeded where D0 failed.

### Preservation (clean images) and restoration (attacked images), percent
| Defense | Preservation [95% CI] | Restoration 4/255 | 8/255 | 16/255 |
|---|---|---|---|---|
| D0 | 100.0 | 49.5 | 35.5 | 17.0 |
| jpeg90 | 92.5 [88.5, 96.0] | 84.5 | 77.0 | 68.5 |
| jpeg75 | 92.0 [88.0, 95.5] | 88.5 | 85.0 | 78.5 |
| jpeg50 | 92.0 [88.0, 95.5] | 89.5 | 88.5 | 83.0 |
| dangerzone | 91.5 [87.5, 95.0] | 88.0 | 87.5 | 81.5 |

### Paired tests at eps=16/255 (McNemar, Holm over 7 comparisons)
| Comparison | (b, c) | ASR diff [95% CI], points | Holm-adjusted p | Reject |
|---|---|---|---|---|
| dangerzone vs D0 | (0, 111) | -55.5 [-62.5, -48.5] | 1.1e-24 | yes |
| jpeg50 vs D0 | (0, 111) | -55.5 [-62.5, -48.5] | 1.1e-24 | yes |
| jpeg75 vs D0 | (0, 110) | -55.0 [-62.0, -48.0] | 1.3e-24 | yes |
| jpeg90 vs D0 | (0, 110) | -55.0 [-62.0, -48.0] | 1.3e-24 | yes |
| dangerzone vs jpeg50 | (0, 0) | 0.0 [0.0, 0.0] | 1.0 | no |
| dangerzone vs jpeg75 | (0, 1) | -0.5 [-1.5, 0.0] | 1.0 | no |
| dangerzone vs jpeg90 | (0, 1) | -0.5 [-1.5, 0.0] | 1.0 | no |

At eps 4/255 and 8/255 every defense beats D0 on the exact binomial McNemar test
(uncorrected p <= 4.8e-7 and <= 8.9e-16). These two were computed outside
`run_analysis.py` from `defense_results.jsonl`; they are not in `statistics.json`.

### Mechanism (defense-controlled residual, n=588 per defense)
| Defense | energy ratio (after/before) | high-freq share before | after |
|---|---|---|---|
| D0 | 1.00 | 0.82 | 0.82 |
| jpeg90 | 0.99 | 0.82 | 0.85 |
| jpeg75 | 1.14 | 0.82 | 0.80 |
| jpeg50 | 1.26 | 0.82 | 0.73 |
| dangerzone | 0.59 | 0.82 | 0.61 |

The other 12 images carry the earlier confounded metric from the pilot and are excluded.

### Scientific conclusion
- H1 supported: every defense cuts ASR from 55.5% to 0-0.5% at 16/255 (adjusted p < 1e-23).
- H2 not supported: dangerzone is indistinguishable from JPEG (no significant
  difference, same preservation within intervals). ASR sits at the floor, so the test
  has no room to show a small advantage.
- H3 only partly consistent: dangerzone removes the most high-frequency energy, but
  jpeg90 defeats 183 of 185 attacks with unchanged residual energy.
- H4 not supported as a separate cost: preservation is about 92% for every defense,
  including jpeg90, so the cost is not specific to aggressive sanitization.
- H5 not tested. The attack is brittle to any re-encoding, so a defense-aware attacker
  is the informative next experiment.
