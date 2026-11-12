# Phase 3 plan: closing the internal-validity gaps before expanding the study

Written 2026-11-12 after external review of the paper draft. Everything here is labelled as decided
before the corresponding experiment runs. Results of the confirmatory study (Phase 2, N=200) are not
changed by this plan; their numbers are re-derived under the corrected metric and, for H1, re-tested
at equal attack budget. Goal: a manuscript whose conclusions the evidence fully supports. No result is
preferred. A null, a reversal, or an inconclusive outcome is reported as found.

## 0. Status of the evidence

| Item | State | Where |
|---|---|---|
| Exact-match success metric, corrected counts | done | `src/evaluation/labels.py`, `scripts/rescore_predictions.py`, `results/corrected/` |
| Clean-task correctness against ground truth | done | `configs/label_map.json`, `results/corrected/clean_task.json` |
| Independent statistical verification | done | `tests/test_statistics_independent.py` |
| Provenance of the 39 baseline rows | done | `docs/provenance/g5_provenance.md` |
| Read-only analysis scripts, raw-data checksums | done | `ensure_derived`, `scripts/checksum_runs.py`, `docs/provenance/raw_checksums.json` |
| Surrogate directional check (attack directions) | done | `scripts/surrogate_directional_check.py` |
| Continuation-equivalence check | prepared, needs GPU | `scripts/validate_continuation.py`, stage `cont_validate` |
| 500-step oblivious control, both budgets | prepared, needs GPU | stage `obl`, `obl_dz` |
| Convergence and best-so-far study | designed below | section 2 |
| Adapter-only adaptive ablation | designed below | section 3 |
| Second VLM, mechanism, utility frontier, literature | later | sections 5 to 7 |

### Corrected numbers (exact normalized match; historical substring in brackets)

The substring rule counted answers such as "tenchu" or "english springer spaniel" as target hits.
Across the registered conditions 53 historical successes are lost under exact match and 0 rows
disagree with the stored historical flags. Adaptive attacks, N=200, real defense: JPEG 50 17 (18) at
8/255 and 42 (42) at 16/255; Dangerzone lossless adapter 56 (58) and 146 (153). Non-adaptive baseline
(`controls_v1`): undefended 21 (21), 52 (53), 104 (112) at 4, 8, 16/255; JPEG 50 0; Dangerzone
lossless 0, 1, 3. The six registered tests all still reject (exact metric, Holm), with smaller effects
for Dangerzone (+71.5 points at 16/255 against +75.0). No clean answer already contains the target
(0 of 200 in every run), so conditional and unconditional success coincide.

Clean accuracy against ground truth is reported at two levels because the model answers "dog" for
almost every breed: strict equality with the class name (12.5%) and membership in the accepted set of
`configs/label_map.json` (68.5%). The accepted-set map was written after the clean answer
distribution had been seen; it is disclosed as such. JPEG 50 and lossless Dangerzone leave
accepted-set accuracy unchanged at 68.5%.

## 1. P0 items

### 1.1 Success measurement (done)
Primary: exact normalized equality with the target label. Diagnostics: whole-word and substring
(the historical rule). Normalization, multiword labels, punctuation and malformed answers are
documented in `src/evaluation/labels.py` and unit tested. Historical files are not modified.

### 1.2 Continuation equivalence (fixed before the first run)
The control continues stored 200-step attacks for 300 steps. The PGD update is sign-gradient with a
constant step, no momentum and no per-step randomness, and the stored array is float32, so the
iterate is the whole state. That is an argument, and the experiment tests it.

Sample: 4 classes drawn with `numpy.random.default_rng(1234)` from the 20 sorted class names, first
manifest image of each, epsilon 16/255, same model, quantization, prompt, seed and step size as the
Phase 1 attack. Per image: U1 (500 uninterrupted steps), U2 (same again, the noise floor), R (200
steps, saved and reloaded through the float32 `.npy` format, then 300 steps with `init_adv`), P (300
steps from the array stored in `full_200`, written in a different session).

Pass rule, per image (all must hold on all 4 images):
- L-infinity distance between the final arrays U1 and R is at most 1e-6 if U1 and U2 are bitwise
  equal; otherwise at most 1.25 times the U1 to U2 distance.
- The mean of the last 10 target losses differs between U1 and R by at most max(5%, 1.25 times the
  U1 to U2 relative difference).
- Exact-match success of R through D0 equals that of U1, unless U1 and U2 already disagree.

P is reported, not gated: it measures how far a continuation from an array crafted in another
session lands from an uninterrupted run, which is the situation of the real control. The check also
records whether the stored array lies inside the epsilon ball. The GPU stage `obl` refuses to start
unless `results/continuation/continuation.json` on Drive says `continuation_equivalent: true`.
If it fails, the control is recomputed from scratch for 500 steps (about 39 GPU hours per budget)
or the approximation is characterized in the paper; resumed attacks are not called exact controls.
Original 200-step results remain unchanged: new runs write only to new run directories, and local
checksums of every historical run are recorded in `docs/provenance/raw_checksums.json`.

### 1.3 Matched-budget control
Runs `obl500_e16` and `obl500_e08`: continue the stored `full_200` arrays by 300 steps with no
defense in the loop, then score every array through D0, JPEG 90/75/50 on the GPU host and through
lossless Dangerzone after a local sanitize. One oblivious attack per (image, epsilon) serves both
defenses. Cost: about 7 minutes per image, 24 GPU hours per budget, about 10 Colab sessions in all.
Primary H1 is then 500-step adaptive against 500-step oblivious, exact match, six tests, Holm. The
200-step comparison stays as a secondary analysis. Whatever the outcome, including a smaller or
insignificant effect, it replaces the current H1 claim.

### 1.4 Statistical verification (done)
Independent re-derivations: exact McNemar by direct summation, Newcombe method 10 written from the
paper's formulas, a bootstrap cross-check, simulated coverage of the paired interval (nominal 95%),
Holm by a hand step-down, Clopper-Pearson closed forms at 0 and n, zero-discordant and all-zero
edge cases, and the registered power claim re-derived by exact enumeration. All pass. The original
registration is untouched. Post-hoc additions are the G5a and G5b gates, the exact-match metric
(an amendment after seeing the substring-matching issue in review), the equal-budget H1 control, and
the 200-step secondary analysis; the paper marks each as not registered.

### 1.5 G5 provenance (done)
See `docs/provenance/g5_provenance.md`. The original failure stays reported.

## 2. Attack strength and convergence (after 1.3)

Subset: 100 images drawn from the 200 with `default_rng(4321)`, 5 per class, rule fixed now.
Conditions: both adaptive attacks at 16/255 (the budget with the most headroom), continued from the
stored 500-step arrays to 1000 and 2000 cumulative steps. Checkpoint every 100 steps (stored as
float32), scored through the real defense: JPEG 50 on the GPU host, Dangerzone by local sanitize at a
coarser grid. Reported separately: final-iterate success, best-so-far success (nondecreasing by
construction), steps at which a success is lost. Two extra random starts at 500 steps on 25 of the
images to measure restart variance. An optimizer variant (momentum, step decay) on the same 25 images
to test whether the constant-step sign update limits success. Decision rule: if final-iterate success
at 2000 steps exceeds 500 steps by more than the 15-point effect, the confirmatory budget is not
adequate for a ranking claim and the paper narrows to "demonstrated attack success at budget B".
"Lower bound" is not used; the wording is "final-iterate rate at this budget".

## 3. What does Dangerzone itself contribute (after 2)

Conditions on the same images and budgets, adaptive through each, scored on real output pixels:
(1) no preprocessing, (2) the PDF wrap and raster path without Dangerzone (`adapter_only_ll`),
(3) JPEG 50 and 75, (4) full Dangerzone with the lossless adapter, (5) the legacy adapter, always
separate. Needs a surrogate for (2), fitted the same way and held to the same 40 dB gate. Sample:
100 images at 16/255 first; the sample is enlarged if the claim rests on a difference that 100
images cannot resolve (power calculation before running). Question: does Dangerzone give any
measurable benefit beyond the transformations its pipeline needs? A zero or negative answer is a
result.

## 4. Surrogate validation (partly done)

Three separate properties. Forward fidelity: held-out PSNR 48.9 dB clean and 48.1 dB adversarial.
Optimization effect: real-tool defended loss lower for the adaptive image on 200 of 200 images in each
run. Directional fidelity along the attack directions: the real tool's response to the final
perturbation against the surrogate's linear prediction. Median cosine 0.993 at 16/255 and 0.977 at
8/255 (identity baseline 0.78), median relative error 12% and 21%, 200 images per run (rounding of the
8-bit real output contributes to the larger error at the smaller budget). This validates the linear
response over the perturbation actually used. It does not test directions the attack never visits or
the real tool's local gradient. Differentiating through the real tool is not feasible and is stated as
a limitation.

## 5. Second VLM

Qwen2-VL-2B after sections 1 to 3. A fixed manifest (the same 200 images), same budgets, exact-match
success, real-defense scoring, model-specific preprocessing, attack strength calibrated per model
(steps to reach a stable loss on a 20-image pilot) and reported. The central comparisons are
replicated, not the whole grid. Disagreement between models is investigated and reported.

## 6. Mechanism and utility

Clean accuracy (done above), clean-answer preservation, PSNR and SSIM, perturbation residuals,
frequency analysis, preprocessing latency. Correlations stay descriptive; no causal claim from
frequency energy without an intervention. RQ3 is answered only if an experiment supports it,
otherwise it is declared out of scope.

## 7. Literature and novelty

A systematic search before the related-work section, citing only verified publications (adaptive
attacks and gradient obfuscation, input transformation defenses, defense evaluation, VLM robustness,
multimodal prompt injection, CDR and document sanitization, purification, security and utility
tradeoffs). Two 2026 papers named in the review are unverified and are not cited until checked.

## 8. Reproducibility

Raw files of every run are checksummed (`scripts/checksum_runs.py`); analysis scripts do not write to
run directories (`ensure_derived`, regression test `tests/test_read_only_analysis.py`). Every table in
the paper is produced by a script from stored predictions. Environment: Colab T4, `environment.json`
per run, model revision and 4-bit settings in `config.json` and `model_info.json`.

## 9. GPU plan

| Experiment | Question it settles | GPU hours | Sessions |
|---|---|---|---|
| Continuation check | Is resuming equal to an uninterrupted run? | about 3 | 1 |
| Oblivious 500-step control | How much of H1 is knowledge of the defense? | about 48 | about 10 |
| Convergence subset | Are 500 steps adequate? | to be sized after the control | |
| Adapter-only ablation | Does Dangerzone add anything? | about 20 | about 4 |
| Qwen2-VL-2B | Does it replicate? | to be sized | |

Each run is resumable from Drive and syncs every 2 minutes. If the control reverses or erases H1, the
later stages are re-planned on that evidence.

## 10. Standing rules for the paper

The paper is rewritten from the final evidence, not edited. Title, abstract and conclusion state only
supported claims; hypotheses are labelled; the CDR tool's intended file-sanitization function is not
evaluated and is not claimed to fail. An adversarial internal review is run before any submission
recommendation, with every substantial criticism resolved, mitigated or openly acknowledged.
