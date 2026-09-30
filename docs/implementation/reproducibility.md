# Reproducibility

What a third party needs to reproduce the experiment, and where every fact about a
run is recorded.

## Environment

Python is managed with uv (see `pyproject.toml`, `requires-python >=3.11,<3.14`).

```bash
# CPU-only, for unit tests and the sanitizers (no GPU needed)
uv venv --python 3.12 .venv-exp
.venv-exp/bin/python -m pip install numpy scipy pillow pyyaml pytest
.venv-exp/bin/python -m pytest -q            # 46 tests

# GPU box, for the attack and model evaluation
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # match CUDA
uv pip install transformers accelerate bitsandbytes pillow numpy scipy pyyaml matplotlib
```

## Fixed inputs

- Model: `llava-hf/llava-1.5-7b-hf`, vision tower CLIP ViT-L/14-336. Weights frozen.
  4-bit NF4 quantization is enabled by default so the model fits a 16 GB GPU; set
  `model.load_in_4bit: false` in `configs/experiment.yaml` for full fp16 on a larger
  card (this changes the numbers, so it is recorded in `model_info.json`).
- Dataset: 20 classes x 10 images = 200, chosen with `numpy.default_rng(1234)`.
  Build it with `python scripts/build_dataset.py`; the manifest
  (`configs/dataset_200.json`) fully determines the benchmark and is copied into
  each run as `dataset_manifest.json`.
- Attack: targeted L-inf PGD, `configs/attack.yaml`. epsilon in {4,8,16}/255,
  200 steps, step size 1/255, random start, seed 1234.
- Defenses: `configs/experiment.yaml`. D0, JPEG 90/75/50, Dangerzone. ICDR is
  documented as BLOCKED (`icdr.md`) and left out of the default set; it runs only if
  a licensed `ICDR_CMD` is supplied.

## Running

```bash
# checkpoints (fast gates)
python scripts/test_llava.py                       # ckpt 1: 4-bit backprop to image
python scripts/run_clean.py --run-id clean_check   # ckpt 2: clean preservation
python scripts/validate_attack.py --epsilon 0.0313725490   # ckpt 3: attack works on D0

# full or pilot run (resumable)
python scripts/run_experiment.py --config configs/experiment.yaml --run-id mvp_001
python scripts/run_experiment.py --run-id mvp_001 --resume          # after any kill
python scripts/run_experiment.py --run-id pilot_12 --limit 12 --epsilon 0.0313725490

# analysis (safe to run on a partial run)
python scripts/run_analysis.py --run results/runs/mvp_001
python -m src.visualization.demo --run results/runs/mvp_001 --static
```

## What a run directory records

`results/runs/<run_id>/` is self-describing:

- `config.json` — merged experiment + attack config, including pilot overrides.
- `environment.json` — python, torch, CUDA, GPU name and VRAM, package versions.
- `git_commit.txt` — the repo commit the run was launched from.
- `dataset_manifest.json` — the exact images, classes, and target labels.
- `model_info.json` — model id, dtype, quantization, device.
- `defense_info.json` and `defenses_unavailable.json` — defenses run and skipped.
- `adv/*.npy` — the exact adversarial pixels each defense saw (lossless).
- `jobs/{clean,attack,defense}/*.json` — one file per finished unit.
- `clean_results.jsonl`, `attack_results.jsonl`, `defense_results.jsonl` — derived
  tables, rebuilt from jobs each session.
- `summary.json` — coverage: attempted, completed, failed, per defense.
- `statistics.json` and `plots/` — produced by the analysis step.

## Resume and persistence

Every finished unit is written atomically (temp file plus rename), so a killed
process or a reclaimed VM loses at most the one job in flight. A rerun with the same
`--run-id` skips finished jobs. On free Colab, point `--run-id` at a directory on a
mounted Drive so the run survives the session, and pass `--max-hours` to stop
cleanly before the runtime is reclaimed. See `colab-resume.md`.

## Determinism notes

Seeds are set for python, numpy, and torch (`seed_everything`). Greedy decoding
(`do_sample=false`, `num_beams=1`) makes generation deterministic. Two caveats: GPU
kernels and library versions can shift low-order bits, and 4-bit quantization changes
results relative to fp16. Both are recorded per run in `environment.json` and
`model_info.json`, so a rerun on the same stack reproduces the numbers and a rerun on
a different stack is comparable rather than identical.
