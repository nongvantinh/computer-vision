# Phase 2 pre-registration: adaptive attacks on JPEG and CDR

Status: written before any adaptive result exists. It is an internal registration kept
in the repository (commit history is the timestamp), not an external registry entry.

What was already known when this was written: the non-adaptive baseline `full_200`
(tag `baseline-nonadaptive-v1`), and the finding that the Dangerzone adapter embeds the
image as JPEG 75 (see "Adapter confound" below). No adaptive attack had been run.

## Question

Does CDR protect a vision-language model beyond generic re-encoding when the attacker
knows the defense will be applied?

## Adapter confound (already established)

The baseline `dangerzone` condition is JPEG 75, then a 150 dpi raster, then Dangerzone,
then a Lanczos downscale. The JPEG comes from Pillow's PDF writer; the embedded image
is bit-identical to `JpegDefense(75)`. Plain JPEG 75 already defeated 184 of the 185
attacks that worked undefended. Baseline results for `dangerzone` therefore cannot be
credited to Dangerzone. This phase adds `dangerzone_ll` (byte-exact embedding, same
page geometry) and the controls `adapter_only`, `adapter_only_ll`, `resample_only`,
`chroma_only`, `noise1`, `noise2`, plus JPEG 95, 30 and 10.

## Confirmatory hypotheses (primary family, six tests)

Estimand: targeted ASR, the share of the 200 images whose greedy answer through the
REAL defense contains the target label, per attack condition and budget.

- **H1 (adaptivity).** For D in {JPEG 50, Dangerzone-lossless}: ASR of the attack
  optimized through D differs from ASR of the baseline (oblivious) attack, both scored
  through D, same images.
- **H2 (CDR versus generic, under adaptive attack).** ASR of the attack optimized
  through Dangerzone-lossless differs from ASR of the attack optimized through JPEG 50,
  each scored through its own real defense, same images.

Each at 8/255 and 16/255, which gives m = 6 tests. All tests are two-sided. Direction
is not assumed: both "adaptivity recovers success" and "adaptivity does not" are
admissible outcomes, as are CDR better, equal or worse than JPEG.

Everything else is secondary and labelled so: JPEG 90, legacy `dangerzone`, the
controls, MDCore, Qwen2-VL, typographic attacks, 4/255.

## Test, intervals, multiplicity

- Paired exact McNemar test on discordant pairs; Holm-Bonferroni over the six tests,
  alpha = 0.05.
- Rates: Clopper-Pearson exact 95% interval. Paired differences: Newcombe (method 10)
  95% interval.
- Zero successes are reported as such with the exact upper bound (about 1.8% for 0 of
  200). A zero is a result, not a missing value.
- All statistics come from `scripts/run_analysis.py` under `configs/analysis_plan.yaml`.

## Minimum effect and sample size

Minimum effect of interest: 0.15 absolute difference in targeted ASR.

Power of the paired exact test (simulated, `scripts/power_analysis.py`, 4000 runs,
seed 1234), per-test alpha = 0.05 / 6:

| N images | discordant rate | power for 0.15 | power for 0.20 | power for 0.10 |
|---|---|---|---|---|
| 100 | 0.3 | 0.51 | 0.86 | 0.16 |
| 200 | 0.3 | 0.89 | 1.00 | 0.43 |
| 200 | 0.4 | 0.75 | 0.97 | 0.30 |

N = 200 is therefore the confirmatory sample. Differences below about 0.10 cannot be
resolved with this design, and the report will say so for any null result.

## Stages and the expansion rule

1. **Controls** on the stored baseline adversarial images (no new attacks).
2. **Pilot:** 10 images, 16/255, JPEG 50, Dangerzone-lossless and legacy Dangerzone,
   30 attacks. Purpose: validity gates only. No inference.
3. **Screen:** 50 images (first 50 in manifest order), 8 and 16/255, the primary
   conditions. Purpose: confirm the gates at scale and estimate discordant rates. At
   N = 50 power is at most 0.3, so nothing here is a confirmatory claim.
4. **Confirmatory:** N = 200 for all primary conditions at both budgets.

**Expansion is decided by the validity gates, not by the ASR values.** If the gates
pass, the full N = 200 run goes ahead whether the screen ASR is zero, intermediate or
saturated, because a zero under adaptive attack is the informative outcome.

## Validity gates (all must pass before the next stage)

- **G1, defense in the loop.** For every attack job: `defense_calls` equals the number
  of steps, the diagnostic gradient is finite and non-zero, and the defended and raw
  clean losses differ.
- **G2, surrogate fidelity.** Held-out PSNR to the real output at least 40 dB for the
  Dangerzone surrogates (clean and adversarial inputs), and at least 42 dB (quality
  90) and 39 dB (quality 50) for the JPEG surrogate against Pillow.
- **G3, the attack optimizes.** Scored through the defense, the adaptive attack's final
  loss is lower than the oblivious baseline attack's loss through the same defense on
  at least 80% of the pilot images.
- **G4, no silent loss.** Zero failed real-defense evaluations; every adversarial array
  lies inside its L-infinity ball; image counts match the manifest.
- **G5, baseline reproduction.** The defense-only controls run reproduces the baseline
  D0 answers on at least 99% of the stored adversarial images.

If G3 fails the attack is called ineffective, not the defense robust. Pre-declared
remedies, each recorded as a new attack id and never mixed with the old one: more
steps (400), best-iterate selection by defended loss, then EOT over small input noise.
The budgets do not change.

## Threat model of the adaptive attacker

White-box access to LLaVA-1.5-7B (4-bit) and knowledge of the defense and its
parameters. Deterministic defenses, so EOT is not needed for them. Same budget,
steps (200), step size (1/255), random start and seed as the baseline; the only change
is the defense in the loss path. JPEG: the real Pillow output in the forward pass and a
differentiable JPEG in the backward pass (BPDA). Dangerzone: a surrogate in both
passes, because the real tool cannot run on the GPU host; final success is always
measured through the real tool. MDCore: only if a defensible surrogate exists,
otherwise it stays a non-adaptive black-box condition.

## Interpretation map (declared, not a prediction)

A: CDR holds under adaptive attack while JPEG recovers. B: both recover. C: the CDR
implementations differ. D: Dangerzone adds nothing beyond its adapter. E: Qwen2-VL
differs from LLaVA. F: typographic attacks survive where pixel attacks fail. Every
cell is reportable.

## Deviation log

| Date | Change | Reason |
|---|---|---|
| 2026-10-07 | Analysis plan v1 to v2 | Adapter confound found before any adaptive run |
| 2026-10-07 | Pilot includes legacy `dangerzone` (30 attacks, not about 20) | Directly tests whether the baseline 0% was an artifact of the hidden JPEG |
