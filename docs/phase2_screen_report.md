# Phase 2 screen report

Status: exploratory. All six runs finished on 2026-10-30. Plan and criteria: `docs/phase2_screen_plan.md`.
Figures and tables are in `results/screen/` (not versioned; regenerate with
`scripts/analyze_screen.py`). Gate files are in `results/screen/gates/`.

## Design

50 images over 20 ImageNet classes, one seed, LLaVA-1.5-7B 4-bit. Adaptive BPDA attack, PGD sign
gradient with step 1/255, 500 steps, targeted at a benign output. Three defenses, each attacked
through its own surrogate and scored through the real defense: JPEG 50, Dangerzone with a lossless
adapter (`dangerzone_ll`) and Dangerzone with the legacy adapter (`dangerzone`, embeds a JPEG 75).
Two budgets, 16/255 and 8/255. The two adapters are never pooled. 4/255 was not run (compute).

## Targeted success through the real defense

| Defense | ε | Non-adaptive | Adaptive | Paired diff (95% CI) | Discordant | Holm p |
|---|---|---|---|---|---|---|
| JPEG 50 | 16/255 | 0/50 | 12/50 (24%) | 24 pts (12 to 37) | 12 vs 0 | 0.003 |
| Dangerzone, legacy | 16/255 | 0/50 | 21/50 (42%) | 42 pts (27 to 56) | 21 vs 0 | 1e-5 |
| Dangerzone, lossless | 16/255 | 2/50 | 40/50 (80%) | 76 pts (61 to 85) | 38 vs 0 | <1e-10 |
| JPEG 50 | 8/255 | 0/50 | 6/50 (12%) | 12 pts (2 to 24) | 6 vs 0 | 0.094 |
| Dangerzone, legacy | 8/255 | 0/50 | 7/50 (14%) | 14 pts (4 to 26) | 7 vs 0 | 0.063 |
| Dangerzone, lossless | 8/255 | 1/50 | 13/50 (26%) | 24 pts (10 to 38) | 13 vs 1 | 0.011 |

Holm p is within the screen family of 10 and is exploratory. At 8/255 only the lossless Dangerzone
effect is below 0.05 after correction. Adaptive images optimized for one defense do not transfer to
the other JPEG qualities (0 to 3 of 50 at JPEG 75 and 90).

## Defense against defense, adaptive attacker, equal 500 steps

| Comparison | ε | Rates | Diff (95% CI) | Holm p |
|---|---|---|---|---|
| Lossless Dangerzone vs JPEG 50 | 16/255 | 80% vs 24% | 56 pts (39 to 68) | <1e-6 |
| Legacy Dangerzone vs JPEG 50 | 16/255 | 42% vs 24% | 18 pts (6 to 30) | 0.059 |
| Lossless Dangerzone vs JPEG 50 | 8/255 | 26% vs 12% | 14 pts (0.1 to 28) | 0.18 |
| Legacy Dangerzone vs JPEG 50 | 8/255 | 14% vs 12% | 2 pts (-8 to 12) | 1.0 |

## Gates and criteria

- G1 (jobs complete), G3 (defended loss of the adaptive image lower than the non-adaptive one on
  at least 80% of images; 50/50 in all six runs) and G4 (no failed real-defense rows, perturbation
  inside the ball) pass in all six runs.
- G2 (surrogate held-out PSNR at least 40 dB) passes for both Dangerzone surrogates (about 48 to 50
  dB). It does not apply to JPEG 50, which is attacked through a differentiable JPEG model.
- G5 stays a recorded failure from the pilot (39 rows). G5a and G5b pass. Neither reads attack success.
- S2 proxy: the median training loss fell by 1 to 7% over the last 100 steps in five runs and by
  11% in `screen_dz_e16`. The 10% rule was written for stage-to-stage defended loss, so this is
  only a proxy, and the 16/255 JPEG 50 success rate also kept rising from 500 to 2000 steps in the
  pilot (4/10, 6/10, 7/10). Every number above is therefore at or below what a longer attack reaches.
- Runtime on a T4: median 11 to 12 minutes per image-attack, about 10 hours per 50-image run. One
  Colab session (about 9.3 hours observed) does not finish a run, so each needs two launches.

Applying the criteria: G1 to G4 and G3 pass and the surrogate PSNR is above 40 dB, so the attack
configurations meet the advance rule. The 16/255 legacy Dangerzone run is flagged for the S2
proxy. Revisions would be new attack ids.

## What the data supports

- An adaptive attacker with a fitted surrogate defeats real Dangerzone and JPEG 50 on a sizable
  share of images at 16/255, and the effect over the non-adaptive attack is clear (paired, same images).
- At 8/255 the effect is smaller and only the lossless Dangerzone case survives correction.
- The legacy adapter is harder to attack than the lossless one (42% vs 80% at 16/255). Its JPEG 75
  stage is the likely reason, which is a JPEG effect and not a CDR effect.

## What it does not support

- No defense is robust. No ranking of CDR against JPEG: the JPEG 50 attack is unconverged, the
  Dangerzone surrogates are fitted, and lossless Dangerzone is not weaker or stronger than JPEG 50
  beyond equal-effort differences at the cap.
- Nothing about 4/255, other models, or typographic attacks.

## Next stage

The confirmatory N=200 run is decided by the gates, which pass, not by the success rates. It costs
about 40 GPU hours per attack condition at 500 steps. Open choices for the user: which conditions
and budget (for example 16/255 with 1000 steps for JPEG 50 and both Dangerzone adapters), and the
Qwen2-VL-2B replication.
