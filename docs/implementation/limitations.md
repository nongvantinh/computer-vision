# Limitations

Read alongside the proposal's threat model and the measured results. These are the
constraints a reviewer should weigh before generalizing anything from this study.

## Scope of the defenses tested

The measured study compares D0, three JPEG qualities and one CDR tool, Dangerzone.
Claims are scoped to that: this is an evaluation of lossy input transforms plus one
deployed sanitizer, not of CDR as a category.

ICDR is BLOCKED in principle (see `icdr.md`): the real `ArielCyber/ICDR` is built on
the commercial Aspose.Imaging library, its committed code does not compile, and it has
no open-source fallback. Nothing is substituted for it.

Dangerzone cannot run on Colab (it needs rootless Podman with `--userns nomap`, and
the Colab host is root-only with a locked-down kernel). It was run with the decoupled
workflow: the attack and the model answers on Colab, the sanitizing locally on a
rootless-Podman workstation. The adversarial images persist as `adv/*.npy`, so no input
changes between the two hosts and the attack was never recomputed.

## The Dangerzone adapter

Dangerzone sanitizes documents by rendering pages to pixels in a sandbox and
rebuilding a clean file. To use it on a raster image we wrap the image in a
single-page PDF, run Dangerzone, and rasterize the result back. This exercises its
real pixel-rendering pipeline, but the PDF round-trip and the rasterization DPI are
part of the measured transform, so the result reflects "Dangerzone applied to an
image through this adapter", not a native image-to-image mode. The adapter is
documented in `dangerzone.md` and is not hidden inside a metric.

## 4-bit quantization

The model runs in 4-bit NF4 by default so it fits a 16 GB GPU. Quantization changes
the model's outputs and its gradients, so both the attack and the clean answers
differ from full fp16. This is recorded per run and can be turned off on a larger
card, but the default numbers are 4-bit numbers.

## Single model, single dataset

One model (LLaVA-1.5-7B) and one dataset (a 200-image ImageNet subset, 20 classes)
are used. Results may not transfer to other VLMs, other prompts, or other image
distributions. This is an MVP boundary, not a claim that the finding is universal.

## Metric is model-relative

Success and utility are defined against the model's own clean answer, not against the
ImageNet label, because a general VLM does not reliably emit fine-grained class names.
Targeted success means the answer contains the attacker's target label; preservation
and restoration mean the answer matches the model's clean answer. This is the right
metric for this model and prompt, but it measures behavior relative to the model, not
ground-truth correctness.

## No adaptive attacker in the MVP

The attack is defense-unaware: it is crafted against D0 and then passed through each
defense. A defense-aware (BPDA/EOT) attacker is not implemented in the MVP. Any
robustness observed here is robustness against a non-adaptive attacker only, and the
demo says so explicitly. Robustness against an adaptive attacker is an open question
this study does not answer.

## Compute and sample size

The attack is the expensive step (a full forward and backward through the VLM per PGD
step, per image, per epsilon). On a free T4 this is slow enough that a full 200-image,
three-epsilon matrix needs either many resumed sessions or a stronger GPU. When
compute limits the run, sample count is reduced rather than the scientific definition,
and the report states images attempted, completed, and failed. A smaller well-recorded
N is preferred to an overstated full run.

## What the frequency analysis can and cannot show

The residual-energy and high-frequency analysis compares the injected perturbation to
what survives a defense. It is observational. It can show that a defense removes most
of the perturbation's high-frequency energy and that robustness tends to track this,
but it does not prove causation. The write-up uses "evidence consistent with" rather
than causal language.
