# Final status audit

State of the implementation as of 2026-09-30, verified by reading and running the
code rather than by assuming a component works because a file exists. This is the
working ground truth for finishing the project. It supersedes the status checklist
in AGENTS.md where the two disagree.

## How each row was checked

- Unit tests: `.venv-exp/bin/python -m pytest -q` -> 41 passed (CPU only).
- Local hardware: RTX 3050, 4 GB VRAM. LLaVA-1.5-7B does not fit here even in
  4-bit, so any row that needs the model is verified on cloud GPU, not locally.
- A prior cloud (Colab T4) session confirmed checkpoint 1 (4-bit backprop to the
  image) and ran the attack on real images. That VM was reclaimed and its outputs
  were on ephemeral disk, so those numbers are gone and are marked "not persisted".

## Status table

| Component | Status | Evidence | Remaining work |
|---|---|---|---|
| LLaVA-1.5-7B | PARTIAL | `src/models/llava.py`; 4-bit NF4 load path present; prior cloud ckpt-1 loaded the model and backprop'd to the image | Re-verify on a fresh cloud GPU; result of ckpt-1 not persisted |
| Image gradients | PARTIAL | `to_pixel_values` keeps CLIP normalization inside the graph; prior ckpt-1 reported max abs grad ~0.78 | Re-confirm on cloud; add a persisted ckpt-1 artifact |
| Dataset | PARTIAL | `src/datasets/imagenet_vqa.py` + `scripts/fetch_imagenette.py`; manifest builder and loader unit-tested (`test_dataset`) | Images not downloaded locally (`data/` empty); validate the prompt/label protocol on the pilot (see Metric risk) |
| Targeted PGD | PARTIAL | `src/attacks/targeted_pgd.py`; L-inf projection unit-tested (`test_pgd_constraint`); ran on real images in the prior session | Efficiency pass (per-step CPU round-trip); confirm generation-based success at scale |
| D0 baseline | PASS | `Identity` in `src/defenses/base.py`; exercised by defense tests | none |
| JPEG 90 | PASS | `src/defenses/jpeg.py`; unit-tested (`test_jpeg`), deterministic Pillow round-trip | none |
| JPEG 75 | PASS | same | none |
| JPEG 50 | PASS | same | none |
| ICDR | BLOCKED | Real `ArielCyber/ICDR` (commit `6449269`) is built entirely on Aspose.Imaging (paid commercial library, watermarks output in eval mode); the committed `icdr.java` does not compile, has no build system, and is not a single-image runner; no OSS fallback exists in the project. Honest wrapper `src/defenses/icdr.py` left as-is, no substitute. See `docs/implementation/icdr.md` | Out of scope unless a licensed Aspose build is supplied via `ICDR_CMD`; CDR-specific claims scope to Dangerzone |
| Dangerzone | BLOCKED on available compute | Real 0.11.0 installs on Colab; image pulled + cosign-verified; adapter has no bug. Conversion fails because Dangerzone needs rootless Podman (`--userns nomap`) and Colab is root-only with a locked-down kernel (podman run rc 125/126); local box is Docker-only/no-root. Excluded from Colab runs (`--defenses D0,jpeg90,jpeg75,jpeg50`). See `docs/implementation/dangerzone.md` | Add D5 later on a rootless-Podman host; attack/defense split means the persisted `adv/*.npy` are reused without recomputing the attack |
| Persistence | REBUILDING | Atomic per-job writes exist (`src/utils/resume.py`, `test_resume`) but there is no run-directory contract; the prior pilot was lost to ephemeral disk | New `results/runs/<run_id>/` layout with config/env/git/manifest snapshots; adversarial images persisted as first-class artifacts (this session) |
| Resume | PARTIAL | `is_done` skip logic unit-tested; `run_experiment.py` reuses a stable `--out` dir | Add `--run-id`/`--resume`; split attack jobs from defense-eval jobs so an added defense does not recompute the attack |
| 200-image experiment | NOT RUN | needs cloud GPU; prior pilot not persisted | Run the persisted pilot first, then scale as compute allows |
| Statistics | PASS (code) | `src/evaluation/statistics.py`; McNemar (exact + chi2), bootstrap CI, paired-diff CI, Holm-Bonferroni; unit-tested (`test_statistics`) | Runs once real result files exist; add `statistical-analysis.md` |
| Frequency analysis | PASS (code) | `src/analysis/frequency.py` + `perturbation.py`; unit-tested (`test_frequency`) | Runs on real adversarial images |
| Frontier | PASS (code) | `src/analysis/frontier.py` plots clean-accuracy vs robust-accuracy | Language stays "trade-off / JPEG baseline curve" until enough points justify more |
| Demo | PARTIAL | `src/visualization/demo.py` (static contact sheet + gradio) | Bug: reads `r['prediction']`, but rows store `answer`; align to item-20 spec |
| Final report | NOT STARTED | `docs/report/` has a README only | Write after results exist |

## Bugs found during the audit

1. `src/visualization/demo.py:57` reads `r['prediction']`, but `aggregate` writes
   the field as `answer`. The static demo would raise `KeyError`. Fix when the
   demo is reworked for the item-20 spec.
2. `scripts/validate_attack.py` still scores success with `match_label` (the old
   label-matching metric), which the runner already replaced with
   `contains_target`. The two paths disagree on what counts as a hit. Align the
   validator to the runner's metric.
3. `docs/implementation/experiment-protocol.md` still describes the old metric
   ("Ground truth = ImageNet class name", "Success = match_label"). Update it to
   the model-relative preservation/restoration metric now in `runner.py`.

## Metric risk (item 8)

A general VLM will not reliably emit a fine-grained ImageNet class name ("tench",
"Shih-Tzu") for a clean image, so scoring against the ImageNet label is not
meaningful. The runner already uses a label-free, model-relative metric:

- preservation: a defense on the clean image keeps the model's own clean answer.
- restoration: a defense on the adversarial image returns the answer to the clean
  answer.
- targeted success: the answer contains the attacker's target label
  (`contains_target`, normalized).

This is the validated primary metric and stays primary. The pilot must still print,
per sample, the clean answer, target label, target loss curve, adversarial answer,
and each defended answer, so the metric can be sanity-checked on real output before
scaling (item 9).

## What blocks the measured result

The only hard blocker is GPU compute for the attack itself. The sanitizers (JPEG,
ICDR, Dangerzone) and the whole persistence, statistics, analysis, and demo layer
run without a GPU. LLaVA-1.5-7B does not fit the local 4 GB card, so producing real
ASR / preservation numbers needs either a fresh cloud GPU session (free T4 with
Drive persistence, resumable) or a paid L4/A100.
