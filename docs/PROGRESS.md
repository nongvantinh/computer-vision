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
  - 1000 cumulative steps running, about 14 minutes per image.

## Pending

- Choose the screen step budget from the JPEG 50 stopping point (cap 500).
- Screen runs in priority order (stage `screen`), local Dangerzone sanitize, stage `screen_dz`.
- Screen analysis, figures, report; Phase 1 report erratum for the 39 rows.

## Blockers

None.
