# Literature notes (working document, 2026-11-12, searches added 2026-11-20)

Rule: a paper is cited only after its record has been opened. "Verified" below means the arXiv
abstract page was fetched on the date above and title, authors, date and venue were read from it.
For the 2026-11-20 additions marked (API), the arXiv API Atom record (export.arxiv.org) was read
instead, which carries the same title, author, date and comment fields. This is not a systematic
review and no priority claim can rest on it. A full search (databases, backward and forward
citation chasing) is still required before the related-work section.

Search-result summaries are not sources: on 2026-11-20 two different summaries gave three arXiv
identifiers for known papers and all three resolved to unrelated records (1902.00235 is an
X-ray astronomy paper, 1705.02902 a 5G network overview, 2007.15079 a nanophotonics paper).
Only opened records count.

## Verified

| Paper | Venue | What it does | Relevance to this study |
|---|---|---|---|
| Räber, Aczel, Plesner, Wattenhofer. *Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification* (arXiv 2508.05489) | NeurIPS 2025 workshop (Reliable ML from Unreliable Data) | White-box and adaptive attacks on learned-compression purification; high-fidelity, realistic reconstructions resist attack better than low-quality ones, independent of gradient masking | Closest in spirit to "does reconstruction give robustness against an adaptive attacker"; images, not VLMs; learned compression, not deployed CDR |
| Zhang et al. *On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression* (arXiv 2601.21531) | ICML 2026 | CAGE attack aligns optimization with token-compressed inference (optimization-inference mismatch) | Same principle (match the attack to the inference pipeline) for token compression, not input sanitization |
| Yang et al. *Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models* (arXiv 2511.16110) | AAAI 2026 | Jailbreak framework that bypasses input and output safety filters of commercial VLMs | Not about preprocessing or pixel-space sanitization. The review's suggestion that it overlaps with this study is not supported by the abstract |
| Das et al. *Shield: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression* (arXiv 1802.06816) | KDD 2018 submission | JPEG compression as a defense, with retraining | JPEG as a baseline defense; the abstract does not describe adaptive attacks |
| Zhu, Liu, Hua, Zhong. *CoDefend: Cross-Modal Collaborative Defense via Diffusion Purification and Prompt Optimization* (arXiv 2510.11096) | none stated (v1 Oct 2025, cs.CV) | Supervised diffusion purification with directional guidance for MLLMs, contrasted with unsupervised DiffPure, plus prompt optimization; tested on captioning and VQA | Purification defense for multimodal models; no CDR or document sanitization; the abstract does not evidence adaptive (BPDA-style) evaluation, which would need the full text |
| Fares, Ziu, Aremu, Durasov, Takáč, Fua, Laptev, Nandakumar. *MirrorCheck: Efficient Adversarial Defense for Vision-Language Models* (arXiv 2406.09250, v5 Jun 2026) | none stated | Detection framework: regenerate the image from the model's caption with randomized T2I generators and compare embeddings; reports robustness under strong adaptive conditions | Detection, not sanitization in the input path; nothing on CDR; shows the field evaluates defenses against adaptive attackers |
| Tramèr, Carlini, Brendel, Madry. *On Adaptive Attacks to Adversarial Example Defenses* (arXiv 2002.08347) (API) | NeurIPS 2020 | Adaptive attacks constructed for each defense defeat it; argues evaluations must tailor the attack to the defense | Methodological backbone of this study's adaptive arm; already in references.bib, now verified |
| Carlini, Athalye, Papernot, Brendel, Rauber, Tsipras, Goodfellow, Madry, Kurakin. *On Evaluating Adversarial Robustness* (arXiv 1902.06705, living document) (API) | living document | How to evaluate defenses, including adaptive attacks and reporting standards | Already in references.bib, now verified; the reporting standards this study follows |
| Nasr, Carlini, Sitawarin, Schulhoff, Hayes, et al. *The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections* (arXiv 2510.09023, Oct 2025) (API) | none stated | Adaptive attackers that adapt to each defense bypass LLM-side jailbreak and prompt-injection defenses | The attacker-moves-second posture, for text-side defenses; supports the framing that a defense claim needs an attack adapted to that defense |

## Named in the review, not found

- *Adaptive Ensemble Attack* (2026, preprocessing defenses including JPEG on VLMs): searched again
  by exact title on 2026-11-20 (web search and the arXiv API); no matching record. Not cited.
- *JPEG-resistant Adversarial Images* (Shin and Song): the exact phrase returns zero records in the
  arXiv API by title and by all fields (2026-11-20), and two different web-search summaries gave
  two different arXiv identifiers that both resolved to unrelated papers. If it exists it is not on
  arXiv under that title; a workshop proceedings page would have to be opened before citing it.
- *Adaptive Attacks Against AI Defenses* (Carlini): no arXiv record by that title (API, 2026-11-20);
  possibly a blog post, not a paper. Not cited; the Tramèr 2020 and Carlini 2019 records above cover
  the methodology.

## Already in `docs/proposal/references.bib` (re-verify each before submission)

Athalye et al. 2018 (BPDA), Tramèr et al. 2020 (adaptive attacks), Carlini et al. 2019 (evaluating
robustness), Guo et al. 2018 (input transformations), Qi et al. 2024 (visual adversarial examples
against aligned LLMs), Madry et al. 2018, LLaVA, CLIP, Qwen2-VL, DiffPure, ImageNet, Dangerzone,
SAF-M-49, ICDR.

## Searches still to run

1. Adaptive or BPDA attacks on VLMs with image-preprocessing defenses (JPEG, resize, purification).
2. Evaluations of CDR or document sanitization against adversarial examples and prompt injection.
   A 2026-11-20 web search found no academic paper applying image CDR as an adversarial defense;
   only commercial vendor material. A database search (Scholar, Semantic Scholar) still has to run
   before this becomes a claim.
3. Multimodal prompt injection through images (typographic, adversarial pixels), defenses, benchmarks.
4. Adversarial purification for VLMs: CoDefend (2510.11096) and MirrorCheck (2406.09250) are now
   verified above; the broader family (diffusion, compression) still needs a systematic pass.
5. Statistical practice for paired attack evaluations and pre-registration in adversarial ML.
6. Gradient obfuscation, BPDA and surrogate-gradient practice for non-differentiable pipelines.

## Positioning implied so far (subject to the full search)

The defensible contribution is a controlled evaluation of a deployed CDR tool, with the real tool in
the scoring path, an attack that is matched to the tool, and an ablation that separates what the
tool does from what its adapter does. The general fact that adaptive attacks beat input
transformations is established and is cited, not claimed.
