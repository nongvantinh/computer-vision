# Statistical analysis

Plain-English description of every test the analysis runs, what it assumes, and how
to read its output. Implementation: `src/evaluation/statistics.py`, driven by
`scripts/run_analysis.py`. All tests are seeded (default 1234) and deterministic.

## The design the tests rely on

Every defense sees the same images and the same adversarial examples. The
adversarial image is crafted once per (image, epsilon) against D0, then passed
through each defense, so the outcomes across defenses are paired by image. Paired
tests are the right family here, and treating the defenses as independent samples
would be wrong.

The primary per-image outcome is binary: for a given defense and epsilon, did the
model emit the attacker's target label (targeted success = 1) or not (0). Utility
uses a second binary outcome: on a clean image, did the defense keep the model's own
clean answer (preserved = 1).

## Bootstrap confidence intervals

`bootstrap_ci` resamples the per-image outcomes with replacement (10,000 times) and
reports the 2.5th and 97.5th percentiles of the resampled mean. This gives a 95%
interval for a rate (targeted ASR, restoration, or preservation) without assuming a
normal distribution. With a small pilot (for example 12 images) the interval is
wide, and that width is the honest signal that the pilot cannot settle a close
comparison. A confidence interval is a description of uncertainty, not a hypothesis
test, so we never use "the intervals overlap" as a substitute for the paired test
below.

## Paired difference interval

`paired_diff_ci` resamples images (not the two defenses independently) and reports
the interval for the difference in rate between two defenses. Because it resamples
the shared image index, it respects the pairing. Read it as the plausible range for
"how much lower is this defense's ASR than that defense's ASR".

## McNemar's test

`mcnemar` is the paired test for two binary outcomes on the same items. It looks
only at the images where the two defenses disagree (one succeeded, the other did
not) and asks whether the split between the two kinds of disagreement is consistent
with chance. When discordant pairs are few (fewer than 25) it uses the exact
binomial test; otherwise it uses the chi-square form with continuity correction.
The concordant images (both defenses gave the same outcome) carry no information
about which defense is better, which is exactly why a paired test, not a comparison
of two separate rates, is appropriate.

## Holm-Bonferroni correction

We run a family of comparisons: each defense against D0, and each non-JPEG defense
against each JPEG quality. Testing many pairs inflates the chance of at least one
false positive. `holm_bonferroni` orders the raw p-values, scales each by the number
of remaining comparisons in a step-down way, and enforces a monotone adjusted
sequence. A comparison is reported as significant only if its adjusted p-value is
below alpha (0.05). The adjusted p-value, not the raw one, is what supports any
claim of a real difference.

## Effect size

The effect size for a defense comparison is the difference in targeted ASR, reported
with its paired bootstrap interval. A significant McNemar result with a tiny ASR
difference is a real but small effect, and the write-up should say so rather than
leaning on the p-value alone.

## How to read a result

1. Is the attack strong enough to test a defense at all? Check ASR on D0. If ASR on
   D0 is low, the images were barely attacked and no defense conclusion follows.
2. For each defense, read ASR and its interval, and preservation and its interval.
3. For each comparison, read the McNemar adjusted p-value and the paired ASR
   difference interval together.
4. State the conclusion at the strength the numbers support, including "inconclusive"
   when the interval is wide or the adjusted p-value is above alpha.

## Mechanism analysis (defense-controlled)

The mechanism analysis is descriptive, not a hypothesis test, but it must not be
confounded. The naive residual `defended_adv - clean` mixes two things: the attack
energy that survives the defense, and the defense's own footprint on the image (JPEG
blocking, Dangerzone re-rendering). A heavy defense then looks like it "added" energy,
which is meaningless. `analyze_defense_controlled` fixes this by comparing:

- raw perturbation: `delta = adv - clean`
- post-defense perturbation: `delta_D = defended_adv - defended_clean`

Both images pass through the same defense, so the defense's footprint cancels and
`delta_D` isolates the adversarial component that survives. For each it reports L-inf,
L2, per-pixel energy, and the high-frequency energy fraction; `energy_ratio =
energy(delta_D) / energy(delta)` summarizes how much attack energy remains. The writeup
compares the raw and post-defense high-frequency fraction across defenses and says
"mechanistic evidence consistent with high-frequency attenuation", never a causal claim.
The N=12 pilot used the old confounded measure (flagged in its results); the final run
uses the controlled one.

## Coverage and honesty

`scripts/run_analysis.py` first rebuilds the derived tables from the job files, so
it analyzes exactly the jobs that finished. The run summary records images attempted,
completed, and failed per defense. Any rate is computed over completed jobs only, and
the report states N. A partial run yields a real but smaller-N result, never an
extrapolation to the full 200.
