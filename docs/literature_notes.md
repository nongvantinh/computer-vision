# Literature notes (working document, 2026-11-12)

Rule: a paper is cited only after its record has been opened. "Verified" below means the arXiv
abstract page was fetched on the date above and title, authors, date and venue were read from it.
This is not a systematic review and no priority claim can rest on it. A full search (databases,
backward and forward citation chasing) is still required before the related-work section.

## Verified

| Paper | Venue | What it does | Relevance to this study |
|---|---|---|---|
| Räber, Aczel, Plesner, Wattenhofer. *Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification* (arXiv 2508.05489) | NeurIPS 2025 workshop (Reliable ML from Unreliable Data) | White-box and adaptive attacks on learned-compression purification; high-fidelity, realistic reconstructions resist attack better than low-quality ones, independent of gradient masking | Closest in spirit to "does reconstruction give robustness against an adaptive attacker"; images, not VLMs; learned compression, not deployed CDR |
| Zhang et al. *On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression* (arXiv 2601.21531) | ICML 2026 | CAGE attack aligns optimization with token-compressed inference (optimization-inference mismatch) | Same principle (match the attack to the inference pipeline) for token compression, not input sanitization |
| Yang et al. *Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models* (arXiv 2511.16110) | AAAI 2026 | Jailbreak framework that bypasses input and output safety filters of commercial VLMs | Not about preprocessing or pixel-space sanitization. The review's suggestion that it overlaps with this study is not supported by the abstract |
| Das et al. *Shield: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression* (arXiv 1802.06816) | KDD 2018 submission | JPEG compression as a defense, with retraining | JPEG as a baseline defense; the abstract does not describe adaptive attacks |

## Named in the review, not found

- *Adaptive Ensemble Attack* (2026, preprocessing defenses including JPEG on VLMs): one search found
  no matching record. Not cited. Re-search by title and author before drawing any conclusion.

## Already in `docs/proposal/references.bib` (re-verify each before submission)

Athalye et al. 2018 (BPDA), Tramèr et al. 2020 (adaptive attacks), Carlini et al. 2019 (evaluating
robustness), Guo et al. 2018 (input transformations), Qi et al. 2024 (visual adversarial examples
against aligned LLMs), Madry et al. 2018, LLaVA, CLIP, Qwen2-VL, DiffPure, ImageNet, Dangerzone,
SAF-M-49, ICDR.

## Searches still to run

1. Adaptive or BPDA attacks on VLMs with image-preprocessing defenses (JPEG, resize, purification).
2. Evaluations of CDR or document sanitization against adversarial examples and prompt injection.
3. Multimodal prompt injection through images (typographic, adversarial pixels), defenses, benchmarks.
4. Adversarial purification for VLMs (diffusion, compression), including CoDefend (arXiv 2510.11096,
   seen in search results, not yet opened) and MirrorCheck (arXiv 2406.09250, same).
5. Statistical practice for paired attack evaluations and pre-registration in adversarial ML.
6. Gradient obfuscation, BPDA and surrogate-gradient practice for non-differentiable pipelines.

## Positioning implied so far (subject to the full search)

The defensible contribution is a controlled evaluation of a deployed CDR tool, with the real tool in
the scoring path, an attack that is matched to the tool, and an ablation that separates what the
tool does from what its adapter does. The general fact that adaptive attacks beat input
transformations is established and is cited, not claimed.
