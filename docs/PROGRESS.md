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

## Pending

- Screen step budget: the JPEG 50 stop point is 2000, so the 500 cap binds (plan section 4). The screen reports at 500 steps, where the pilot JPEG 50 attack had not converged. A 500-step attack takes about 14 minutes per image, so one 50-image run is about 11.5 GPU hours.
- Screen runs in priority order (stage `screen`), local Dangerzone sanitize, stage `screen_dz`.
- Screen analysis, figures, report; Phase 1 report erratum for the 39 rows.

## Blockers

None.
