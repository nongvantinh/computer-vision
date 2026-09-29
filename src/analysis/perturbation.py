"""Perturbation-removal analysis (proposal §4.2).

For an adversarial image and a defense, measure how much of the crafted
perturbation survives sanitization, in total energy and in its high-frequency
component. Evidence for/against RQ3.
"""
from __future__ import annotations

import numpy as np

from .frequency import high_freq_energy_fraction, residual_energy


def analyze_defense_on_perturbation(clean: np.ndarray, adv: np.ndarray,
                                    defended_adv: np.ndarray,
                                    cutoff: float = 0.25) -> dict:
    """Compare the injected perturbation to what remains after a defense.

    clean:         original image
    adv:           adversarial image (clean + delta)
    defended_adv:  sanitize(adv)
    """
    delta = np.asarray(adv, np.float64) - np.asarray(clean, np.float64)
    residual = np.asarray(defended_adv, np.float64) - np.asarray(clean, np.float64)

    e_delta = residual_energy(delta)
    e_res = residual_energy(residual)
    return {
        "injected_energy": e_delta,
        "residual_energy": e_res,
        "energy_removed_frac": float(1 - e_res / e_delta) if e_delta > 0 else 0.0,
        "injected_highfreq_frac": high_freq_energy_fraction(delta, cutoff),
        "residual_highfreq_frac": high_freq_energy_fraction(residual, cutoff),
        "injected_linf": float(np.max(np.abs(delta))),
        "residual_linf": float(np.max(np.abs(residual))),
    }
