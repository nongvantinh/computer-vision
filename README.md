# CDR as an Adversarial Defense for Vision-Language Models

Advanced Computer Vision (ACV / MCS) course project.

> **Content Disarm and Reconstruction as an Adversarial Defense for
> Vision-Language Models: An Empirical Evaluation and Adaptive-Attack Analysis**

Does image **Content Disarm and Reconstruction (CDR)** — structural file
sanitization from the security world — defend **vision-language models** against
adversarial and typographic prompt-injection attacks better than ordinary image
transforms like JPEG? This project measures it, explains *why*, and stress-tests
it with an adaptive attacker.

## Deliverables

| Deliverable  | Location                     | Status |
|--------------|------------------------------|--------|
| Proposal     | `docs/proposal/proposal.pdf` | ✅ built |
| Final report | `docs/report/`               | planned |
| Presentation | `slides/`                    | planned |
| Experiments  | `experiments/` → `results/`  | planned |

## Quick start

```bash
make proposal      # build docs/proposal/proposal.pdf (XeLaTeX + biber)
make docs          # build proposal + report + slides (as available)
make help          # list all targets
uv sync            # set up the Python environment for experiments
```

See **[AGENTS.md](AGENTS.md)** for the full project structure, conventions, and
build details.
