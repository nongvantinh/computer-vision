"""Frequency-domain analysis of the residual perturbation (proposal §3.6, §4.2).

Tests the mechanistic hypothesis (RQ3) that CDR suppresses the high-frequency
component of the perturbation. Provides evidence, not proof of causation.
"""
from __future__ import annotations

import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def fft_magnitude(img: np.ndarray) -> np.ndarray:
    """Centered 2D FFT magnitude spectrum of a (grayscale) image/residual."""
    return np.abs(np.fft.fftshift(np.fft.fft2(_gray(img))))


def radial_profile(mag: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged magnitude as a function of radius (low->high freq)."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def high_freq_energy_fraction(residual: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of spectral energy at radius > cutoff * max_radius.

    cutoff in (0,1); 0.25 keeps the outer 75% of the radius as 'high frequency'.
    """
    mag = fft_magnitude(residual)
    energy = mag ** 2
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max()
    total = energy.sum()
    if total <= 0:
        return 0.0
    high = energy[r > cutoff * r_max].sum()
    return float(high / total)


def residual_energy(residual: np.ndarray) -> float:
    """Total L2 energy (sum of squares) of a residual, normalized per pixel."""
    d = np.asarray(residual, dtype=np.float64)
    return float(np.mean(d ** 2))
