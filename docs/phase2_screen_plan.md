# Phase 2 protocol revision 1 and the 50-image screen

Written 2026-10-11, after the 10-image adaptive pilot and before any screen run. It
extends `docs/phase2_preregistration.md`; the original text and the pilot results are
unchanged. Everything here is labelled as to when it was decided and what had been seen.

## 1. What the pilot showed (preliminary, N = 10, one class)

16/255, images 1 to 10 of the benchmark (all tench), 200 steps.

| Attack through | Scored through (real defense) | Non-adaptive | Adaptive |
|---|---|---|---|
| JPEG 50 | JPEG 50 | 0/10 | 0/10 |
| Dangerzone, lossless adapter | Dangerzone, lossless adapter | 0/10 | 8/10 |
| Dangerzone, legacy adapter | Dangerzone, legacy adapter | 0/10 | 4/10 |
| Dangerzone, legacy adapter | JPEG 75 | 0/10 | 2/10 |

Gates G1 to G4 passed. G5 failed (section 2). The adaptive JPEG 50 attack lowered the
defended loss from a median of 6.3 to 2.6. The two Dangerzone attacks ended at medians of
0.74 (8/10 succeeded) and 1.7 (4/10), so the JPEG 50 attack stopped well above the loss
levels where success appeared. 0/10 is an optimization result and does not show robustness
of JPEG 50. The Dangerzone pilot result is a signal from ten images of one class.

## 2. Revision of G5 (decided after seeing G5 fail)

The registered G5 required at least 99% raw-answer agreement between the D0 rows of
`controls_v1` and the baseline `full_200`. Observed: 96.8% (581 of 600).

**The original G5 still fails and stays in the record as written.**

### Investigation

- 19 answers differ. In all 19 cases the recorded `psnr_vs_clean` of the D0 row also
  differs between the two runs, so the model was shown different pixels.
- 39 of the 600 rows have different pixels (13 images times 3 budgets: the 10 tench
  images and 3 images of `n02086240`). On the other 561 rows the two runs agree on every
  answer (561 of 561), and therefore on every target-success outcome.
- The PSNR of the stored adversarial array against the clean image matches the
  `controls_v1` value for these rows, not the original `full_200` value. The original
  defense rows for these 13 images were therefore computed on adversarial arrays that
  are no longer the stored ones. The Dangerzone sanitized images match the stored arrays
  (checked through the sidecar energy for all 1200 rows).
- Cause: not recorded in any log. It is consistent with those attacks having been
  crafted again after the first defense evaluation, because commit 5792229 (2026-10-04)
  fixed a bug where finished adversarial arrays were not restored at session start. A
  second craft with the same seed gives slightly different pixels on a GPU. This is a
  hypothesis about provenance, not a finding.
- Effect on Phase 1: only D0 changes. D0 successes are 186 in `controls_v1` against 185
  in `full_200` (4/255: 21 vs 22, 8/255: 53 vs 52, 16/255: 112 vs 111). JPEG 90, 75 and
  50 are identical in both runs (0 of 600 target-success outcomes differ). The Phase 1
  report needs an erratum; its conclusions do not change.

### New gates (post hoc, disclosed as such)

- **G5a, reproducibility of targeted success.** Agreement of `target_success` between two
  evaluations of the same D0 inputs. Input identity is `adv_sha256` when both rows have
  it, otherwise equal `psnr_vs_clean` within 0.001 dB. Rows with different inputs are
  listed and not scored, because a different image is not a reproduction test.
  Threshold: 99%. Justification: the model decodes greedily and the 4-bit pipeline gave
  identical answers on 561 of 561 identical inputs, so the expected agreement for a sound
  run is 100%. At 561 pairs a 99% threshold allows at most 5 discordant pairs, which is
  already above the 0 observed and below the 7 that a pixel change produced. It was not
  chosen to make the existing runs pass: the same threshold applied to all 600 rows gives
  98.8% and would fail.
- **G5b, raw-answer agreement (diagnostic).** The original criterion restricted to
  identical inputs. Reported with its denominator and the 19 excluded rows.

Result on the existing runs: G5 fail (96.8%), G5a pass (561 of 561, 0 discordant,
39 rows excluded for different inputs), G5b pass (561 of 561). Source:
`scripts/check_gates.py`, `results/pilot/gates_g5_revised.json`.

From this point every attack job and defense row records `adv_sha256`, so input identity
is checked by hash and not by a PSNR proxy. Phase 2 comparisons use `controls_v1` as the
non-adaptive condition because it was scored on the stored arrays.

## 3. Extended adaptive JPEG 50 attack

Question: does a stronger optimizer reach success through JPEG 50? The threat model does
not change: epsilon 16/255, same 10 images, same target, same success definition, same
step size (1/255), BPDA through the differentiable JPEG with the real Pillow forward.

Update rule: sign-gradient with a constant step and no momentum, so the iterate is the
whole optimizer state. Each stage therefore continues from the stored image of the stage
before, and N steps followed by M steps equals N + M steps in one run (unit test
`test_continuing_from_a_stored_image_equals_one_longer_run`). The 200-step pilot is never
modified; stage runs are new run directories (`pilot_jpeg50_s500`, `_s1000`, `_s2000`).

Cumulative budgets: 500, 1000, 2000 steps. After each stage the final images are scored
through real JPEG 50 (also JPEG 75 and 90 and D0) and the defended loss is recomputed.

Stopping rule, fixed before any extended run (`src/evaluation/staging.py`):

- **S1 success:** 8 or more of the 10 images succeed through real JPEG 50. Stop.
- **S2 converged:** the median defended loss (final iterates, same estimator at every
  stage, 200-step pilot value 2.55 as the first reference) improved by less than 10%
  relative to the previous stage. Stop.
- **S3 maximum:** 2000 cumulative steps have run. Stop.

The 8-of-10 threshold is the rate the lossless Dangerzone attack reached in the pilot, so
JPEG 50 gets the same bar for "the attack works". The 10% threshold is a plateau
criterion for a doubling of steps.

Recorded per stage: success counts, per-step loss curves (`loss_curve` in every attack
job), defended loss, perturbation norms, runtime, failures.

Reporting rule: if JPEG 50 stays at 0/10 at the stopping point, the report says so with
the convergence evidence and calls it an unresolved optimization result. It is not
described as robustness of JPEG 50, and epsilon and target are not changed to produce a
success.

## 4. The 50-image screen

**Purpose.** A decision gate for the confirmatory N = 200 run: find implementation
defects, check that the adaptive attacks are credible at scale, estimate discordant-pair
rates for the power calculation, and decide which conditions get the full run. Nothing in
the screen is a confirmatory claim, and a screen result does not replace the full
experiment.

**Images.** `configs/screen_50.json`, produced by `scripts/make_screen_manifest.py` with
seed 1234: 2 images from each of the 20 classes plus 1 more from 10 seeded classes. All
are members of the 200-image benchmark. This replaces the pre-registered rule "first 50
in manifest order", which selects only 5 classes (deviation, section 7). Two of the ten
pilot images are in the screen.

**Conditions.**

| Attack | Optimized through | Notes |
|---|---|---|
| non-adaptive | none (stored `full_200` arrays) | already scored in `controls_v1`; the 50 images are subset by id |
| adaptive | JPEG 50 | real Pillow forward, differentiable backward |
| adaptive | Dangerzone, lossless adapter | fitted surrogate; scored through the real tool |
| adaptive | Dangerzone, legacy adapter | its adapter embeds JPEG 75; kept separate |

Every adaptive run is scored through D0, JPEG 90, 75 and 50, and the two Dangerzone runs
are also scored through the real Dangerzone tool with their own adapter. The lossless and
legacy adapter conditions appear as separate rows in every table and figure. Nothing is
pooled into one "Dangerzone" number.

**Budgets.** 16/255 first, then 8/255. 4/255 is not in the screen: the non-adaptive attack
reaches only 10.5% undefended at 4/255 (21 of 200), so there is little room to see a
defense effect, and the compute (below) does not allow it. It stays in the confirmatory
plan as a secondary budget.

**Steps.** One step budget for all adaptive attacks in the screen, so that the defenses are
compared at equal optimizer effort. It equals the cumulative budget at which the extended
JPEG 50 attack stops under S1 to S3, capped at 500 steps for compute. If the cap binds,
the screen reports at the cap and says so.

**Compute and order.** One 200-step attack takes about 285 s on a T4 and cost scales with
steps. The runs execute in this priority order and each is resumable and persisted after
every job: lossless Dangerzone 16/255, JPEG 50 16/255, legacy Dangerzone 16/255, then the
same three at 8/255. A screen that ends early because credits run out is reported as
incomplete, with the finished runs.

**Measured per condition.** Targeted success with Clopper-Pearson 95% intervals; paired
comparisons on the same images (exact McNemar, Newcombe interval for the difference);
Holm correction inside the screen family, labelled exploratory; optimization curves
(`loss_curve`); attack and defense runtime; clean-image utility and model-relative
preservation and restoration from `controls_v1` on the same 50 images; gates G1 to G5b.

**Primary versus exploratory.** Every screen comparison is exploratory. The pre-registered
confirmatory family (`configs/analysis_plan.yaml`, six tests at N = 200) is untouched.

**Criteria, written before the screen runs.**

- *Advance* an attack configuration to the confirmatory run if G1 to G4 pass on the screen,
  the defended loss is lower than the non-adaptive loss on at least 80% of the images (G3),
  and the surrogate PSNR is at least 40 dB.
- *Revise* it if G3 fails, if S2 shows the loss still falling at the step cap, or if the
  surrogate PSNR is below 40 dB. Revisions are new attack ids and are never mixed with the
  old ones.
- *Abandon* it as a primary condition only if it cannot be completed within the compute
  limit at the capped budget. Abandonment is reported, not hidden.
- The decision to run N = 200 depends on the gates, not on the ASR values, as already
  registered. A zero under a converged adaptive attack is an informative result.

## 5. What each outcome may be called

No defense is called robust because an adaptive attack that has not converged failed. CDR
is not called better than JPEG unless the paired evidence supports it at equal optimizer
effort. A result obtained with the lossless adapter is never attributed to the legacy
adapter or the reverse.

## 6. Preserved records

`full_200`, `pilot_n12_eps8`, `controls_v1` and the three N = 10 pilot runs are not
modified. Their figures and the commits that produced them stay in the history. The
screen adds run directories and does not overwrite any.

## 7. Deviation log (this revision)

| Date | Change | Reason | Seen before deciding |
|---|---|---|---|
| 2026-10-11 | G5 kept as written; G5a and G5b added | G5 failed at 96.8%; cause traced to 39 rows with different input pixels | G5 failure, the 19 differing answers, the pilot success rates |
| 2026-10-11 | Extended JPEG 50 schedule 500, 1000, 2000 (pre-registered remedy was 400) | Pilot JPEG 50 loss still falling at 200 steps | pilot loss and 0/10 result |
| 2026-10-11 | Screen images stratified over classes, not first 50 | First 50 are 5 classes | manifest order, pilot images all tench |
| 2026-10-11 | 4/255 left out of the screen | Low non-adaptive ASR, compute | non-adaptive ASR per budget |
| 2026-10-11 | Screen step budget tied to the JPEG 50 stopping point, capped at 500 | Equal optimizer effort, compute | none beyond the above |
