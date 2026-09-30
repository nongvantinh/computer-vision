# Results

This file separates **hypotheses** (what we expect, no numbers invented) from
**measured results**. The preliminary pilot below is real, measured data. The final
experiment tables stay as placeholders until that run produces its `statistics.json`.

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

The final experiment tables below stay as placeholders until that run produces
`results/runs/<final>/statistics.json`.

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

Fill the tables only from `results/runs/<run>/statistics.json`. Every rate reports N
(completed jobs). State images attempted / completed / failed. Never extrapolate.

## Measured: preservation (utility)  — *pending run*
| Defense | Preservation | 95% CI | N |
|---|---|---|---|
| D0 | _ | _ | _ |
| jpeg90 | _ | _ | _ |
| jpeg75 | _ | _ | _ |
| jpeg50 | _ | _ | _ |
| dangerzone | _ | _ | _ |

## Measured: targeted ASR by defense x epsilon  — *pending run*
| Defense | eps=4/255 | eps=8/255 | eps=16/255 |
|---|---|---|---|
| D0 | _ | _ | _ |
| jpeg90 | _ | _ | _ |
| jpeg75 | _ | _ | _ |
| jpeg50 | _ | _ | _ |
| dangerzone | _ | _ | _ |

## Measured: statistical comparisons (McNemar + Holm)  — *pending run*
| Comparison | ASR diff [CI] | McNemar p | Holm-adjusted | Reject H0 |
|---|---|---|---|---|
| dangerzone vs D0 | _ | _ | _ | _ |
| dangerzone vs jpeg75 | _ | _ | _ | _ |
| jpeg50 vs D0 | _ | _ | _ | _ |

## Measured: mechanism (perturbation removal)  — *pending run*
| Defense | energy removed (mean) | residual high-freq frac | injected high-freq frac |
|---|---|---|---|
| jpeg75 | _ | _ | _ |
| dangerzone | _ | _ | _ |

## Scientific conclusion  — *pending run*
State plainly whether CDR > JPEG, CDR ~= JPEG, or CDR < JPEG, or whether the
attack itself was unreliable (ASR on D0 too low to draw a defense conclusion).
Do not force agreement with H1-H3.
