# Provenance of the 39 baseline rows that fail the original G5

Status: investigation record. Machine-readable rows: `docs/provenance/g5_mismatch_rows.json`.
The original gate (G5, at least 99% raw-answer agreement between `controls_v1` and `full_200`)
failed at 96.8% (581 of 600) and is reported as failed everywhere. This file records what is known
about why, and what is not.

## What is established

- 39 of the 600 `full_200` D0 rows (13 images at three budgets: the 10 tench images in manifest
  order and 3 images of `n02086240`) were scored on pixels that differ from the arrays now stored
  in `full_200/adv/`. The test is the recorded `psnr_vs_clean`, which differs between the two runs
  for exactly these rows and for no others. On the other 561 rows the two runs agree on every
  answer (561 of 561).
- Drive write times (`rclone lsl`, UTC as reported by Drive) separate the two groups. For all 39
  differing rows the stored adversarial array and its attack job were written **after** the D0
  defense job that produced the original answer (D0 jobs 2026-09-30 to 2026-10-03; arrays
  2026-10-03 and 2026-10-04). For the 561 agreeing rows the array predates the D0 job in 560
  cases. The one exception is a row whose array was written 13 seconds after its D0 job in the
  same session, and its two evaluations agree.
- The write times are therefore consistent with these 13 attacks having been crafted again after
  the first defense evaluation, which is the earlier hypothesis. The commit that restores finished
  arrays at session start (5792229) is dated 2026-10-04 11:42, after the 2026-10-03 re-crafts and
  after the 11:00 to 11:20 re-crafts on 2026-10-04, so the bug was live when they happened.
- The Dangerzone sanitized images match the stored arrays (checked earlier through sidecar energy
  for all 1200 rows), so the Dangerzone rows are not affected.
- Effect on the baseline counts: 19 of the 39 answers differ and 7 target-success flags differ.
  Undefended success is 186 in `controls_v1` and 185 in `full_200` (substring criterion). JPEG 90,
  75 and 50 outcomes are identical in both runs.

## What is not established

- The original arrays that produced the 39 original answers were overwritten and are not stored.
  Their measurements can be read from `full_200/defense_results.jsonl` but cannot be recomputed.
- No log records why the attacks were re-run, so the cause (a session that did not restore its
  arrays) is an inference from the timestamps and the commit history.
- A second craft with the same seed on a GPU gives slightly different pixels. How different has
  not been measured on these images. The continuation-equivalence experiment in
  `docs/phase3_plan.md` measures run-to-run variation on fresh attacks.

## Decision

The non-adaptive baseline used in the confirmatory analysis is `controls_v1`: the stored arrays,
rescored, with `adv_sha256` recorded per row. `full_200` is kept unchanged for history. The
original G5 failure, the replacement gates G5a and G5b, and this provenance record are all
reported in the paper's appendix. G5a and G5b are post hoc.
