"""Targeted L-inf PGD against the VLM (proposal §3.3, §3.6).

Minimizes the teacher-forcing cross-entropy toward a predetermined incorrect
target answer, under an L-inf budget around the clean image. Defense-unaware:
the adversarial image is crafted against D0 and later passed through each defense.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def project_linf(adv: np.ndarray, clean: np.ndarray, epsilon: float) -> np.ndarray:
    """Clip adv into the L-inf ball around clean, then to the valid [0,1] range.

    Pure numpy so it is testable without torch. Works elementwise.
    """
    lo = np.clip(clean - epsilon, 0.0, 1.0)
    hi = np.clip(clean + epsilon, 0.0, 1.0)
    return np.clip(adv, lo, hi)


@dataclass
class PGDConfig:
    epsilon: float = 8 / 255
    steps: int = 200
    step_size: float = 1 / 255
    random_start: bool = True
    seed: int = 1234


@dataclass
class AttackResult:
    adv_image: np.ndarray
    losses: list = field(default_factory=list)
    epsilon: float = 0.0
    steps: int = 0
    linf: float = 0.0


def targeted_pgd(model, image_np: np.ndarray, question: str, target_answer: str,
                 cfg: PGDConfig) -> AttackResult:
    """Run PGD on one image. `model` is a LlavaWrapper-like object."""
    torch = model.torch
    clean = np.asarray(image_np, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)

    if cfg.random_start:
        noise = rng.uniform(-cfg.epsilon, cfg.epsilon, size=clean.shape).astype(np.float32)
        adv = project_linf(clean + noise, clean, cfg.epsilon)
    else:
        adv = clean.copy()

    clean_t = torch.tensor(clean, device=model.device)
    losses: list[float] = []
    for _ in range(cfg.steps):
        x = model.make_image_tensor(adv, requires_grad=True)
        loss = model.target_loss(x, question, target_answer)   # minimize (targeted)
        grad = torch.autograd.grad(loss, x)[0]
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            x = x - cfg.step_size * grad.sign()          # descend toward target
            # project in-tensor, then hand back to numpy for the next iteration
            lo = torch.clamp(clean_t - cfg.epsilon, 0.0, 1.0)
            hi = torch.clamp(clean_t + cfg.epsilon, 0.0, 1.0)
            x = torch.max(torch.min(x, hi), lo)
        adv = x.detach().cpu().numpy().astype(np.float32)

    adv = project_linf(adv, clean, cfg.epsilon)  # final safety projection
    return AttackResult(adv_image=adv, losses=losses, epsilon=cfg.epsilon,
                        steps=cfg.steps,
                        linf=float(np.max(np.abs(adv - clean))))
