# Phase 2 progress

Live status file. Updated at each step; the commit history has the details.

## Done

- Baseline `full_200` preserved (tags `phase1-nonadaptive-baseline`, `baseline-nonadaptive-v1`).
- `controls_v1`: 13 defenses on the stored baseline images (600 attacks, 0 failures).
- Adaptive pilot, N = 10, 16/255, 200 steps: JPEG 50, Dangerzone lossless, Dangerzone legacy.
  Real Dangerzone scored. Results in `results/pilot/` (not versioned).
- Protocol revision 1 (`docs/phase2_screen_plan.md`): G5a and G5b, extended JPEG 50 stopping
  rule, screen design and criteria.
- Code: same-input reproducibility gates, attack continuation (`--init-from`), loss curves and
  `adv_sha256` in every job, `--manifest`, staged-attack decision rule, screen stages.
- Screen manifest `configs/screen_50.json` (50 images, 20 classes, seed 1234).

## In progress

- Extended adaptive JPEG 50 pilot (stage `jpeg_ext`, Colab, started 2026-10-11 14:53 UTC).
  - 500 cumulative steps done: 4/10 succeed through real JPEG 50 (0/10 at 200 steps), median
    defended loss 2.55 to 1.87 (26.6% lower), decision `continue`. JPEG 75 and 90 and D0 stay 0/10.
  - 1000 cumulative steps done (2026-10-12): 6/10 succeed through real JPEG 50, median defended
    loss 1.92 to 1.42 (26.1% lower), decision `continue` (S1 needs 8/10, S2 needs under 10%
    improvement).
  - 2000 cumulative steps done (2026-10-12, second attempt; the first was lost with its
    Colab session before reaching Drive): 7/10 succeed through real JPEG 50, median defended
    loss 1.42 to 1.19 (16.1% lower). Stop reason S3, the maximum budget. S1 (8/10) was not
    reached and S2 (under 10% improvement) did not fire, so the attack had not converged.
    Success by budget through JPEG 50: 0/10 (200), 4/10 (500), 6/10 (1000), 7/10 (2000).
    JPEG 75 and JPEG 90 reach 1/10 each at 2000 steps (0/10 before). D0 stays 0/10.
    Cost: about 22 to 24 minutes per image for the last 1000 steps.
  - Incident: the first resume in a new session re-crafted images 1 and 2 of the 500-step
    stage, because the stage script pulled job files without the stored images (fixed in
    112b0ba). The Drive copy of `pilot_jpeg50_s500` now holds re-crafted pixels for those two
    images (loss 2.43 to 2.37, 1.30 to 1.38; GPU runs are not bit-reproducible). The original
    500-step result (4/10, median 1.87) is in the local `results/runs/pilot_jpeg50_s500` and in
    the earlier commit f9bed45. The 1000-step stage continues from the Drive copy, so it is a
    valid 500+500 continuation of those images, but its 500-step reference median is 1.92.

- Screen run 1 of 6, `screen_dz_ll_e16` (lossless Dangerzone adapter, 16/255, 500 steps, 50 images),
  started 2026-10-12 09:34 UTC on Colab: about 755 s per image, so about 10.5 hours. First
  image loss 6.88 to 0.37 on the surrogate. Runs after it: `screen_jpeg50_e16`, `screen_dz_e16`,
  then the three 8/255 runs, each one Colab session with `SCREEN_ONLY` set.
  - Lost (2026-10-12): that run reached 20 of 50 images (last seen 13:51 UTC) and was never copied
    to Drive. Cause: the notebook's shell autosync never ran, because Colab's `/bin/sh` is dash
    and rejects `exec -a`; with no browser tab open, Colab then released the runtime. No screen
    result exists. Fixed in c644c6c: the stage script copies every run directory to Drive every
    2 minutes and logs `SYNC <run>: adv local N, on Drive M`. Run 1 restarts from image 1.
  - Run 1 finished (2026-10-26): 50/50 attacks, 200/200 defense rows, 0 failures. Through the
    standard defenses D0 7/50, JPEG 90/75/50 0/50. Real Dangerzone (lossless adapter) sanitize
    done locally: 50 new, 0 failed, uploaded to Drive as `dz_ll/`. Next Colab session:
    `screen_eval,screen_dz` for run 1, then run 2 (`screen_jpeg50_e16`).
  - Run 1 scored on real Dangerzone (2026-10-27, `screen_dz`, exploratory): adaptive through the
    lossless adapter succeeds on 40/50 images (80%, 95% CI 66.3 to 90.0%) against 2/50 for the
    non-adaptive attack (4%, CI 0.5 to 13.7%); paired difference 76 points (CI 60.7 to 84.9%),
    38 discordant pairs one way and 0 the other, exact McNemar p 7e-12 (single-test family).
    The same images give D0 7/50 and JPEG 90/75/50 0/50, so this attack works on Dangerzone and
    does not transfer to JPEG. Replicates the 8/10 pilot at 50 images over 20 classes. The
    result is for the lossless adapter at 16/255 and 500 steps and says nothing about the legacy
    adapter. Output in `results/screen/` (not versioned).
  - Defect: `screen_eval` pulled run 1 without the stored images, so the first
    `defended_loss_dangerzone_ll.json` had 0 adaptive entries. Fixed (pull_npy, pushed) and rerun.
  - Run 2 `screen_jpeg50_e16` finished (2026-10-27, after 4 launches): 50/50 attacks, 200/200
    defense rows. Adaptive JPEG 50 at 500 steps succeeds through real JPEG 50 on 12/50 (24%, CI
    13.1 to 38.2%) against 0/50 non-adaptive (paired difference 24 points, CI 12 to 37%, 12
    discordant one way, 0 the other, Holm-adjusted p 5e-4). JPEG 75/90 3/50 each, D0 2/50.
    JPEG 50 is not robust to this attacker; 24% is a lower bound, since the 10-image pilot rose
    from 4/10 at 500 steps to 7/10 at 2000.
  - Gate G3 (defended loss, scored with the fixed stage): adaptive below non-adaptive on 50/50
    images for both runs. Median defended loss: lossless Dangerzone surrogate 0.40 vs 4.83;
    JPEG 50 3.0 vs 7.6.
  - Exploratory comparison, lossless Dangerzone vs JPEG 50, each attacked through its own defense
    at the same 500 steps: 40/50 vs 12/50, difference 56 points (CI 39 to 68%), Holm p 1e-7.
    Read with care: the JPEG 50 attack had not converged at 500 steps and the Dangerzone surrogate
    is fitted, so the comparison measures how far each attack got at equal effort. It does not say
    Dangerzone is weaker than JPEG at convergence, and it does not support CDR over JPEG.
  - Run 3 `screen_dz_e16` (legacy Dangerzone adapter, 16/255, 500 steps) attack finished (2026-10-28,
    2 launches): 50/50 attacks, 200/200 standard-defense rows. Real Dangerzone sanitize done locally
    with the legacy embedding (`--embed jpeg`): 50 new, 0 failed, uploaded to Drive as `dz/` (101
    files).
  - Run 3 scored (2026-10-28). Real Dangerzone with the legacy adapter (embeds a JPEG 75): adaptive
    21/50 (42%, CI 28.2 to 56.8%), non-adaptive 0/50; paired diff 42 points (CI 27.5 to 55.8%),
    21 vs 0 discordant, Holm p about 3e-6. G3 passes (50/50 images, medians 1.12 vs 7.45). Same
    attacker and budget on the lossless adapter reached 40/50, so the JPEG 75 stage in the legacy
    adapter roughly halves attack success at 500 steps. Exploratory legacy Dangerzone vs JPEG 50
    (adaptive, equal 500 steps): 21/50 vs 12/50, diff 18 points (CI 5.8 to 29.5%), 10 vs 1
    discordant, Holm p about 0.012. The JPEG 50 attack is unconverged and the Dangerzone
    surrogate is fitted, so this does not rank the defenses. Lossless and legacy stay separate.
  - Next: the three 8/255 runs (about 11 GPU hours each), starting with `screen_dz_ll_e08`.
  - Run 4 `screen_dz_ll_e08` (lossless adapter, 8/255, 500 steps) attack finished (3 launches): 50/50
    attacks, 200/200 standard-defense rows. Local lossless sanitize: 50 new, 0 failed, uploaded as
    `dz_ll/` (101 files).
  - Run 4 scored (2026-10-29), lossless adapter, 8/255, 500 steps, exploratory: through real
    Dangerzone adaptive 13/50 (26%, CI 14.6 to 40.3%) vs non-adaptive 1/50 (2%); paired diff 24
    points (CI 10.2 to 37.8%), 13 vs 1 discordant, Holm p about 0.004. G3 passes (50/50, medians
    1.74 vs 5.46). Transfer to JPEG 50/75/90 is 0/50. Success at 8/255 (26%) is about a third of
    16/255 (80%) for the same attacker and budget. Undefended D0 reaches 6/50 under the adaptive
    images and 13/50 under the non-adaptive ones at 8/255.
  - Next: `screen_jpeg50_e08` attack.

## Pending

- Screen step budget: the JPEG 50 stop point is 2000, so the 500 cap binds (plan section 4). The screen reports at 500 steps, where the pilot JPEG 50 attack had not converged. A 500-step attack takes about 14 minutes per image, so one 50-image run is about 11.5 GPU hours.
- Screen runs in priority order (stage `screen`), local Dangerzone sanitize, stage `screen_dz`.
- Screen analysis, figures, report; Phase 1 report erratum for the 39 rows.

## Blockers

None.
