# Experiment Protocol (exact, reproducible)

Frozen protocol for the MVP. Source of truth for design: `docs/proposal/proposal.tex`.

## Model
- LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`), fp16, weights frozen.
- Vision tower: CLIP ViT-L/14-336; images normalized with CLIP mean/std inside the
  differentiable path so gradients reach the raw [0,1] image.
- Gradient checkpointing on. Greedy decoding (`do_sample=False`, `num_beams=1`).

## Dataset (`configs/dataset_200.json`)
- ImageNet-1k validation, ImageFolder layout `val/<wnid>/*.JPEG`.
- 20 classes x 10 images = 200, selected with `numpy.default_rng(seed=1234)`.
- Class names from the standard `imagenet_class_index.json` (cached to
  `configs/imagenet_class_index.json`).
- Prompt: "What is the main object in this image? Answer with a single lowercase word."
- Target label = the next chosen class in sorted order (rotation), guaranteed
  different from the ground truth. The ImageNet class name is recorded for reference
  but is not the scoring key (see Metrics).
- Images resized to 336x336 (bicubic). The **same** images pass through every defense.

## Attack (`configs/attack.yaml`)
- Targeted L-inf PGD, teacher-forcing cross-entropy toward the target label.
- epsilon in {4/255, 8/255, 16/255} = {0.01569, 0.03137, 0.06275}.
- steps = 200, step_size = 1/255 = 0.003922, random_start = true, seed = 1234.
- Defense-unaware: one adversarial image per (image, epsilon), crafted against D0,
  persisted losslessly, then read back by each defense (no attack recomputation).
- Success = the greedy answer contains the target label (`contains_target`), measured
  on the actual generated output, not on teacher-forcing loss.

## Defenses (`configs/experiment.yaml`)
- D0: identity (uint8 round-trip only).
- D1-D3: JPEG quality 90 / 75 / 50, Pillow, subsampling 4:2:0, deterministic.
- D4: **ICDR** (`ArielCyber/ICDR`) is **BLOCKED** (commercial Aspose dependency,
  non-compiling code, no OSS fallback; see `icdr.md`). Left out of the default set;
  runs only if a licensed `ICDR_CMD` is supplied. No substitute is used.
- D5: **Dangerzone** (`dangerzone-cli`, Docker), image -> single-page PDF ->
  sanitized PDF -> rasterized back to 336x336. See `dangerzone.md` for the verified
  setup. This image<->PDF adapter is the documented way to push a raster through
  Dangerzone's real pipeline.

## Metrics (model-relative, label-free)
- Primary: targeted ASR (`contains_target`) per defense, per epsilon.
- Utility: preservation (a defense on the clean image keeps the model's clean answer)
  and restoration (a defense on the adversarial image returns the clean answer).
- Fidelity (descriptive): PSNR, SSIM, LPIPS(optional).
- Mechanism: residual perturbation energy; radial-FFT high-frequency energy fraction.
- Cost: defense latency, attack runtime.

## Statistics (seed 1234)
- Paired McNemar per comparison (each defense vs D0; each CDR vs each JPEG).
- 95% percentile bootstrap CIs (10k resamples) for ASR and clean accuracy.
- Holm-Bonferroni across the predefined comparison family.
- Effect size = ASR difference (with paired bootstrap CI).

## Hardware / software
- Cloud GPU (>=16-24 GB), CUDA PyTorch build. Python 3.12.
- Local machine (RTX 3050, 4 GB) used only for CPU unit tests with CPU PyTorch.

## Commands
```bash
# cloud GPU
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # match CUDA
uv pip install transformers accelerate pillow numpy scipy matplotlib pyyaml gradio

python scripts/build_dataset.py    --config configs/experiment.yaml
python scripts/test_llava.py                                   # checkpoint 1
python scripts/run_clean.py        --run-id clean_check         # checkpoint 2
python scripts/validate_attack.py  --epsilon 0.0313725490       # checkpoint 3
python scripts/run_experiment.py   --run-id mvp_001            # full MVP (resumable)
python scripts/run_experiment.py   --run-id mvp_001 --resume    # continue after a kill
python scripts/run_analysis.py     --run results/runs/mvp_001
python -m src.visualization.demo   --run results/runs/mvp_001 --static

# external tools (once): see dangerzone.md (verified setup) and icdr.md (BLOCKED)

# local CPU tests
uv pip install --python .venv-exp/bin/python pytest
.venv-exp/bin/python -m pytest tests/ -q
```
