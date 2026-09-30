# Colab: resuming across usage limits and accounts

The full run (200 images x 3 epsilons, white-box PGD through LLaVA-1.5-7B) exceeds
one Google Colab account's GPU quota. This is the design that lets you **hit the
limit, switch account, and continue** instead of restarting. It is adapted from
`knowledge-tracing-comparison`'s Colab notebook, simplified for a single experiment.

Notebook: `notebooks/colab_cdr_vlm.ipynb`. Logic module: `src/colab/sync.py`
(unit-tested in `tests/test_colab_sync.py`). Resume primitives: `src/utils/resume.py`.

## The four resume layers
1. **Job idempotency.** Every unit writes one file and is skipped if it exists:
   `results/runs/<run_id>/jobs/clean/<id>__<defense>.json`,
   `.../jobs/attack/<id>__eps<e>.json`, and
   `.../jobs/defense/<id>__eps<e>__<defense>.json`. The expensive attack is one job
   per (image, epsilon) and its adversarial image is persisted losslessly as
   `adv/<id>__eps<e>.npy`, which every defense job reads back, so adding a defense
   never recomputes the attack. A rerun recomputes only unfinished jobs. Writes are
   atomic (temp + `os.replace`) so a killed process never leaves a half-file.
2. **Merge-only restore.** At session start the notebook pulls state from Drive and
   from the fallback zip and **adds missing files only**. A finished job's file is
   immutable, so the two sources can never conflict and restore is order-independent.
3. **Only small state travels; model and dataset are ephemeral.** Free Google Drive
   is 15 GB and LLaVA-1.5-7B is ~14 GB, so the model does not live on Drive. `HF_HOME`
   points at ephemeral `/content/hf_cache` and the model re-downloads each session
   (~10 min). The dataset (free imagenette+imagewoof) is fetched to ephemeral disk
   each session; its selection is deterministic (fixed wnids + seed 1234), so every
   account builds the identical 200-image manifest. Only the light job JSON, tables,
   and plots sync to Drive, which keeps the shared folder far under quota.
4. **Time budget + autosync.** `run_experiment.py --max-hours` stops cleanly between
   jobs before Colab reclaims the runtime, and a background thread pushes the run
   directory to Drive every few minutes, so a timeout costs at most that interval.

## Two transport paths (used together, merged)
- **Path A — shared Drive folder (recommended).** From the main account, share the
  `DRIVE_DIR` folder with Editor rights to the other accounts. On each other account,
  open the link and *Add shortcut to Drive* into My Drive, **keeping the name**, so
  `DRIVE_DIR` still resolves. Every account then reads/writes one store — nothing to
  copy. Storage counts against the main account.
- **Path B — state zip (fallback, no sharing).** After each sync the notebook writes
  `DRIVE_DIR/cdr_state.zip` (light: the `jobs/` tree + tables + `state/`). Copy it
  into the new account's `DRIVE_DIR` and Run all; every finished job is then skipped.

## The `state/` store
- `profile.json` — the attack/eval config (model, dtype, epsilons, steps, seed),
  **locked by the first session** and reused by all others, so heterogeneous GPUs /
  accounts still produce one comparable results table.
- `sessions.jsonl` — append-only session log, deduped by session id.
- `session.lock` — advisory lock with a 20-minute stale-heartbeat timeout, so two
  accounts don't write the same store at once (set `FORCE_UNLOCK` only if one is dead).

## Drive layout
Small by design: the model and dataset are not here (they re-fetch to ephemeral
disk each session), so the shared folder stays well under the 15 GB free quota.
```
DRIVE_DIR/
  results/runs/<run_id>/   # durable run: jobs/, adv/, *_results.jsonl, summary.json, plots/
  state/                   # profile.json, sessions.jsonl, session.lock
  reports/                 # figures/tables for the writeup
  cdr_state.zip            # Path B fallback bundle
```

## Running a second account
1. First account: run the notebook; it locks `profile.json` and fills the store.
2. Second account: set `ACCOUNT_LABEL` and `EXPECT_EXISTING_STATE = True` in CELL 1
   (so an empty store errors instead of silently restarting from zero), attach the
   shared folder (Path A) or copy `cdr_state.zip` (Path B), then Run all.
3. Verify the restored-jobs count in CELL 4 is non-zero before training continues.

## Empty-state guard
A brand-new study and a state that failed to follow you look identical from inside the
notebook. `EXPECT_EXISTING_STATE` makes the difference explicit: leave it `False` for
the very first session, set it `True` afterwards so a missing store is a hard error.
