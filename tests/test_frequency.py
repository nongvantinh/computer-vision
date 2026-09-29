"""Frequency analysis: high-frequency energy fraction and residual energy."""
import numpy as np

from src.analysis.frequency import high_freq_energy_fraction, residual_energy
from src.analysis.perturbation import analyze_defense_on_perturbation


def test_checkerboard_is_more_highfreq_than_gradient():
    n = 64
    checker = np.indices((n, n)).sum(axis=0) % 2  # highest possible frequency
    grad = np.tile(np.linspace(0, 1, n), (n, 1))  # smooth, low frequency
    assert high_freq_energy_fraction(checker) > high_freq_energy_fraction(grad)


def test_residual_energy_zero_for_no_change():
    z = np.zeros((16, 16))
    assert residual_energy(z) == 0.0


def test_analyze_defense_removes_energy():
    rng = np.random.default_rng(0)
    clean = rng.random((32, 32, 3)).astype(np.float32)
    delta = rng.uniform(-0.03, 0.03, clean.shape).astype(np.float32)
    adv = np.clip(clean + delta, 0, 1)
    # a defense that removes half the perturbation
    defended = np.clip(clean + 0.5 * delta, 0, 1)
    out = analyze_defense_on_perturbation(clean, adv, defended)
    assert 0.0 < out["energy_removed_frac"] < 1.0
    assert out["residual_energy"] < out["injected_energy"]
