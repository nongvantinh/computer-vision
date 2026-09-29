#!/usr/bin/env python
"""Checkpoint 1: prove the LLaVA wrapper is differentiable w.r.t. the input image.

Run on the cloud GPU:
    python scripts/test_llava.py

Passes when: model loads, an image loads, a normal answer generates, the
teacher-forcing loss computes, and loss.backward() yields a non-zero image grad.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.llava import LlavaWrapper, LlavaConfig  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("checkpoint1")


def main() -> int:
    model = LlavaWrapper(LlavaConfig())
    log.info("model loaded: %s", model.cfg.model_id)

    img = np.random.default_rng(0).random((336, 336, 3)).astype(np.float32)

    ans = model.generate(img, "What is the main object in this image?")
    log.info("generate() -> %r", ans)

    x = model.make_image_tensor(img, requires_grad=True)
    loss = model.target_loss(x, "What is the main object in this image?", "banana")
    log.info("target_loss = %.4f", float(loss))

    grad = model.torch.autograd.grad(loss, x)[0]
    gmax = float(grad.abs().max())
    log.info("max |dloss/dimage| = %.3e", gmax)

    ok = np.isfinite(float(loss)) and gmax > 0
    log.info("CHECKPOINT 1: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
