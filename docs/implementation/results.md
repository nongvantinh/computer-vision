# Results

This file separates **hypotheses** (what we expect, no numbers invented) from
**measured results** (filled in only after the cloud run). Do not populate the
"Measured" tables until `scripts/run_analysis.py` has produced
`results/<run>/statistics/summary.json`.

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
