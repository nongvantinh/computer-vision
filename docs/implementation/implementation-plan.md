# Implementation Plan — CDR as an Adversarial Defense for VLMs (MVP)

Source of truth: `docs/proposal/proposal.tex`. This plan records the environment,
the strategy for each MVP component, and **two hardware/software blockers** found
during inspection that affect the proposal's primary model choice.

Status: **inspection done; blocked on a hardware decision before checkpoint 1.**

## 1. Current environment (measured 2026-09-29)

| Item | Value |
|---|---|
| Machine | ASUS TUF Gaming F15, Ubuntu, kernel 6.17 |
| CPU | 16 cores |
| RAM | 37 GB total, ~15 GB available |
| **GPU** | **NVIDIA RTX 3050 Laptop, 4 GB VRAM** (driver 590.48, CUDA 13.1) |
| Disk | 55 GB free (94% used on the 879 GB root) |
| Python (.venv) | 3.14.3 — no ML packages installed |
| System Python | 3.13.7 |
| uv | 0.10.2 |
| Java | OpenJDK 21 (for ICDR) ✅ |
| Docker | 29.2.1 (for Dangerzone) ✅ |
| Internet | huggingface.co and pypi.org reachable (HTTP 200) ✅ |

## 2. Blockers found

### Blocker A — GPU VRAM (4 GB) vs. the proposed model (LLaVA-1.5-7B)
LLaVA-1.5-7B needs ~14 GB just for fp16 weights, and white-box PGD additionally
needs the forward activations and the input-gradient path — realistically ~16–24 GB.
**It cannot run here.** The proposal's documented fallback, Qwen2-VL-2B, needs
~4–5 GB for fp16 weights *alone*, so it also will not fit the 4 GB budget once the
backward pass is added. This is a genuine incompatibility beyond what the proposal
anticipated, so it needs an explicit decision (see §9).

### Blocker B — Python 3.14 vs. PyTorch
The existing `.venv` is Python 3.14, for which PyTorch has no released wheels
(torch currently supports 3.9–3.13). We will create a **Python 3.12 venv** for the
experiment regardless of the model decision. `.python-version` will be pinned to
3.12 for the experiment env; `pyproject.toml` already relaxed `requires-python` to
`>=3.11`.

## 3. Model-loading strategy (depends on §9 decision)
The attack is white-box and needs `∂loss/∂image`, so the model must run with a
differentiable pixel→logits path (no CLIP surrogate, per the proposal). Options,
ranked by fidelity to the proposal:

1. **Bigger GPU (cloud/lab, ≥16 GB):** keep LLaVA-1.5-7B exactly as proposed. Best
   scientific fidelity. Requires hardware the user supplies.
2. **This 4 GB GPU, smaller VLM:** substitute a small open VLM that fits with
   4-bit weights + gradient checkpointing (e.g. TinyLLaVA ~1–1.5B, or moondream2).
   Preserves the scientific intent ("targeted adversarial attack on a real VLM,
   defended by CDR") but changes the headline model; the proposal's model line
   would be updated with the reason.
3. **CPU, tiny VLM, pilot only:** viable for a 10-image debug pilot, far too slow
   for the 200-image × 3-ε experiment.

Whichever is chosen: freeze weights (no weight grads), enable gradient
checkpointing, fp16/4-bit as needed, deterministic inference where possible.

## 4. CDR integration strategy
- **JPEG (D1–D3):** Pillow `save(quality=…)` round-trip; deterministic. Log
  quality, size, dims, runtime.
- **ICDR (D4):** real tool, `ArielCyber/ICDR` (Java + Aspose.Imaging). Java 21 is
  present. Run as an external process on saved images (batch), pin the commit,
  record config. JPEG-only — our ImageNet inputs are JPEG, so this is compatible.
- **Dangerzone (D5):** real tool, Docker-based (Docker present). Run its
  container on each image (image→pixels→rebuilt PDF/image), then load the
  reconstructed raster. If the image→image path needs an adapter (Dangerzone
  targets PDFs), build the smallest reproducible adapter and document it; if
  genuinely impractical, record the blocker before substituting.
- Common interface: `sanitize(image, defense_config) -> sanitized_image`.

## 5. Dataset strategy
ImageNet-1k validation, 20 classes × 10 = 200 images, fixed seed, deterministic ID
list in `configs/dataset_200.json` (image_id, class_id/name, target_class_id/name).
Target label is a fixed, deliberately-incorrect class per image. Same images
through every defense (paired). Needs an ImageNet val source; if the full val set
is unavailable locally, download only the 200 selected files (documented).

## 6. Attack strategy
Targeted L∞ PGD, teacher-forcing CE toward the predetermined wrong label, through
the full VLM. ε ∈ {4,8,16}/255, configurable steps/step-size/random-start in
`configs/attack.yaml`. Defense-unaware in the MVP: one adversarial image per
(image, ε) generated against D0, then passed through every defense.

## 7. Expected computational cost (rough, model-dependent)
On a ≥16 GB GPU with a 7B model: ~a few seconds/PGD-step × ~100 steps × 200 images
× 3 ε ≈ several GPU-hours. On the 4 GB GPU with a ~1B model: slower per useful
result and memory-fragile. On CPU: impractical beyond a pilot.

## 8. MVP milestones (depth-first, per proposal §22)
1. Env built (Py 3.12 + torch) and model loads with an input-gradient path. **[checkpoint 1]**
2. Clean-accuracy baseline across D0–D5. **[checkpoint 2]**
3. Targeted PGD validated on ~10 images at D0 (high ASR). **[checkpoint 3]**
4. Full 200-image paired experiment across D0–D5 × ε.
5. Statistics (McNemar, bootstrap CI, Holm) + mechanism (residual energy, FFT).
6. Security–utility frontier plot + visual demo.

## 9. Decision (resolved 2026-09-29)
**Path 1 chosen: run on a cloud GPU provider (user-supplied), keeping
LLaVA-1.5-7B exactly as proposed.** The local 4 GB laptop GPU is disregarded per
the user ("I will use the other cloud provider … don't care about the current
machine"). Consequences:

- **No change to the proposal** — the primary model stays LLaVA-1.5-7B. Blocker A
  is retired.
- The code is written device-agnostic (bf16/fp16 + gradient checkpointing) and
  assumes a single ≥16–24 GB CUDA GPU on the cloud.
- **Local role of this machine:** build the codebase and run every non-7B unit
  test on CPU (dataset/JPEG determinism, metrics, statistics, PGD L∞ constraint,
  FFT). GPU stages (checkpoints 1–3 and the 200-image run) execute on the cloud.
- **Two environments:** local `.venv-exp` uses **CPU** PyTorch for tests; the
  cloud uses the CUDA PyTorch build. Both are Python 3.12 (Blocker B). Selection
  is documented in `docs/implementation/experiment-protocol.md`.

## 10. Exact commands (to be finalized after §9)
```bash
# environment (Python 3.12 for torch compatibility)
uv venv --python 3.12 .venv-exp
# torch + libs added once the target device/model is fixed (§9)
```
Stage commands for dataset build, clean eval, attack validation, full run,
analysis, and demo will be filled in as each stage lands.
