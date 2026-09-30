"""Perturbation-removal analysis (proposal §4.2).

For an adversarial image and a defense, measure how much of the crafted
perturbation survives sanitization, in total energy and in its high-frequency
component. Evidence for/against RQ3.
"""
from __future__ import annotations

import numpy as np

from .frequency import high_freq_energy_fraction, residual_energy


def _residual_metrics(delta: np.ndarray, cutoff: float) -> dict:
    d = np.asarray(delta, np.float64)
    return {
        "linf": float(np.max(np.abs(d))),
        "l2": float(np.sqrt(np.sum(d ** 2))),
        "energy": residual_energy(d),                 # mean square, per pixel
        "highfreq_frac": high_freq_energy_fraction(d, cutoff),
    }


def analyze_defense_controlled(clean: np.ndarray, adv: np.ndarray,
                               defended_clean: np.ndarray, defended_adv: np.ndarray,
                               cutoff: float = 0.25) -> dict:
    """Defense-controlled perturbation analysis (proposal §4.2, corrected).

    The confounded measure subtracts the raw clean image from the defended
    adversarial image, so it counts the defense's OWN footprint (JPEG blocking,
    Dangerzone re-rendering) as if it were surviving attack energy. Controlling for
    that footprint isolates the adversarial component that survives the defense:

        raw perturbation:          delta   = adv - clean
        post-defense perturbation: delta_D = defended_adv - defended_clean

    Both are reported (L-inf, L2, energy, high-frequency fraction) so the writeup can
    compare the attack signal before and after the defense. This is mechanistic
    evidence, not proof of causation.
    """
    raw = _residual_metrics(np.asarray(adv, np.float64) - np.asarray(clean, np.float64),
                            cutoff)
    post = _residual_metrics(
        np.asarray(defended_adv, np.float64) - np.asarray(defended_clean, np.float64),
        cutoff)
    return {
        "raw": raw, "post_defense": post,
        "energy_ratio": (post["energy"] / raw["energy"]) if raw["energy"] > 0 else 0.0,
        "highfreq_frac_raw": raw["highfreq_frac"],
        "highfreq_frac_post": post["highfreq_frac"],
    }


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
