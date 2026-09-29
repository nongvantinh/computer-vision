# Results

This file separates **hypotheses** (what we expect, no numbers invented) from
**measured results** (filled in only after the cloud run). Do not populate the
"Measured" tables until `scripts/run_analysis.py` has produced
`results/<run>/statistics/summary.json`.

## Hypotheses (from the proposal, not results)
- **H1:** CDR reduces targeted ASR vs. D0.
- **H2:** CDR lies beyond the JPEG frontier for >=1 operating point.
- **H3:** any robustness is consistent with removal of high-frequency perturbation
  energy (evidence, not proof of causation).
- **H4:** aggressive CDR (Dangerzone) costs measurable clean accuracy.
- **H5:** a defense-aware attack would recover ASR (extension; out of MVP scope).

## Measured: clean accuracy (utility)  — *pending run*
| Defense | Clean accuracy | 95% CI |
|---|---|---|
| D0 | _ | _ |
| jpeg90 | _ | _ |
| jpeg75 | _ | _ |
| jpeg50 | _ | _ |
| icdr | _ | _ |
| dangerzone | _ | _ |

## Measured: targeted ASR by defense x epsilon  — *pending run*
| Defense | eps=4/255 | eps=8/255 | eps=16/255 |
|---|---|---|---|
| D0 | _ | _ | _ |
| jpeg90 | _ | _ | _ |
| jpeg75 | _ | _ | _ |
| jpeg50 | _ | _ | _ |
| icdr | _ | _ | _ |
| dangerzone | _ | _ | _ |

## Measured: statistical comparisons (McNemar + Holm)  — *pending run*
| Comparison | ASR diff | McNemar p | Holm-adjusted | Reject H0 |
|---|---|---|---|---|
| icdr vs D0 | _ | _ | _ | _ |
| icdr vs jpeg75 | _ | _ | _ | _ |
| dangerzone vs jpeg75 | _ | _ | _ | _ |

## Measured: mechanism (perturbation removal)  — *pending run*
| Defense | energy removed (mean) | residual high-freq frac | injected high-freq frac |
|---|---|---|---|
| jpeg75 | _ | _ | _ |
| icdr | _ | _ | _ |
| dangerzone | _ | _ | _ |

## Scientific conclusion  — *pending run*
State plainly whether CDR > JPEG, CDR ~= JPEG, or CDR < JPEG, or whether the
attack itself was unreliable (ASR on D0 too low to draw a defense conclusion).
Do not force agreement with H1-H3.
