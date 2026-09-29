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
- Ground truth = ImageNet class name. Target label = the next chosen class in
  sorted order (rotation), guaranteed different from the ground truth.
- Images resized to 336x336 (bicubic). The **same** images pass through every defense.

## Attack (`configs/attack.yaml`)
- Targeted L-inf PGD, teacher-forcing cross-entropy toward the target label.
- epsilon in {4/255, 8/255, 16/255} = {0.01569, 0.03137, 0.06275}.
- steps = 200, step_size = 1/255 = 0.003922, random_start = true, seed = 1234.
- Defense-unaware: one adversarial image per (image, epsilon), crafted against D0,
  then passed through each defense.
- Success = the greedy answer maps to the target label (`match_label`).

## Defenses (`configs/experiment.yaml`)
- D0: identity (uint8 round-trip only).
- D1-D3: JPEG quality 90 / 75 / 50, Pillow, subsampling 4:2:0, deterministic.
- D4: **ICDR** (`ArielCyber/ICDR`), real tool via `ICDR_CMD` (`{input}`/`{output}`),
  pipeline transCode -> resize -> AdvFilter. Pin the commit in the run manifest.
- D5: **Dangerzone** (`dangerzone-cli`, Docker), image -> single-page PDF ->
  sanitized PDF -> rasterized back to 336x336 (pdftoppm @150 dpi, LANCZOS resize).
  This image<->PDF adapter is the documented, honest way to push a raster through
  Dangerzone's real pipeline.

## Metrics
- Primary: targeted ASR (per defense, per epsilon).
- Utility: clean VQA accuracy per defense.
- Fidelity (descriptive): PSNR, SSIM, LPIPS(optional).
- Mechanism: residual perturbation energy; radial-FFT high-frequency energy fraction.
- Cost: defense latency.

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
python scripts/run_clean.py        --config configs/experiment.yaml   # checkpoint 2
python scripts/validate_attack.py  --config configs/experiment.yaml --attack configs/attack.yaml  # checkpoint 3
python scripts/run_experiment.py   --config configs/experiment.yaml --attack configs/attack.yaml  # full MVP
python scripts/run_analysis.py     --run results/<run_dir>
python -m src.visualization.demo   --run results/<run_dir> --static

# external tools (once)
git clone https://github.com/ArielCyber/ICDR external/ICDR   # build per its README
export ICDR_CMD="java -jar external/ICDR/icdr.jar {input} {output}"
pipx install dangerzone   # or distro package; needs Docker

# local CPU tests
uv pip install --python .venv-exp/bin/python pytest
.venv-exp/bin/python -m pytest tests/ -q
```
