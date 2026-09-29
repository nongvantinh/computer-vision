"""Metrics: targeted ASR, accuracy, label matching, image fidelity, L-inf.

Pure numpy so these are testable without a GPU or torch.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Security / utility metrics
# --------------------------------------------------------------------------- #
def targeted_asr(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Fraction of samples where the prediction equals the attacker's target.

    This is the single primary security metric (proposal §3.3): success means the
    model emitted the predetermined incorrect target label.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")
    if not predictions:
        return 0.0
    hits = sum(int(p == t) for p, t in zip(predictions, targets))
    return hits / len(predictions)


def accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Clean/utility accuracy: prediction equals the ground-truth label."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must be the same length")
    if not predictions:
        return 0.0
    return sum(int(p == l) for p, l in zip(predictions, labels)) / len(predictions)


def match_label(answer: str, class_names: Sequence[str]) -> str | None:
    """Map a free-text VLM answer to one of the known class names.

    Deterministic: lowercase substring match; on multiple matches, the longest
    class name wins (most specific). Returns None if nothing matches.
    """
    a = answer.lower()
    matches = [c for c in class_names if c.lower() in a]
    if not matches:
        # try token overlap on the first synonym word
        for c in class_names:
            head = re.split(r"[,/]", c)[0].strip().lower()
            if head and head in a:
                matches.append(c)
    if not matches:
        return None
    return max(matches, key=len)


# --------------------------------------------------------------------------- #
# Image fidelity (inputs are HxWxC arrays in [0, 1])
# --------------------------------------------------------------------------- #
def _as_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.5:  # looks like 0..255
        img = img / 255.0
    return img


def linf(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance between two images in [0, 1]."""
    return float(np.max(np.abs(_as_float(a) - _as_float(b))))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a, b = _as_float(a), _as_float(b)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity, grayscale, single-scale (Wang et al. 2004)."""
    from scipy.signal import fftconvolve

    a, b = _as_float(a), _as_float(b)
    if a.ndim == 3:
        a = a.mean(axis=2)
        b = b.mean(axis=2)
    win = _gaussian_window()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = fftconvolve(a, win, mode="valid")
    mu_b = fftconvolve(b, win, mode="valid")
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = fftconvolve(a * a, win, mode="valid") - mu_a2
    sb = fftconvolve(b * b, win, mode="valid") - mu_b2
    sab = fftconvolve(a * b, win, mode="valid") - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(ssim_map.mean())


def lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    """Perceptual LPIPS distance if the `lpips` package is available, else None.

    Descriptive covariate only (proposal §4.2), so it is optional.
    """
    try:
        import lpips as lpips_lib
        import torch
    except Exception:
        return None
    net = lpips_lib.LPIPS(net="alex", verbose=False)

    def to_t(x):
        x = _as_float(x)
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=2)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        return t * 2 - 1  # LPIPS expects [-1, 1]

    with torch.no_grad():
        return float(net(to_t(a), to_t(b)).item())
