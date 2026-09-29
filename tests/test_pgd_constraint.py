"""The attack must respect the L-inf budget and the valid pixel range."""
import numpy as np

from src.attacks.targeted_pgd import project_linf


def test_project_linf_respects_epsilon_and_range():
    rng = np.random.default_rng(0)
    clean = rng.random((16, 16, 3)).astype(np.float32)
    for eps in (4 / 255, 8 / 255, 16 / 255):
        adv_raw = clean + rng.uniform(-1, 1, clean.shape).astype(np.float32)
        adv = project_linf(adv_raw, clean, eps)
        assert adv.min() >= 0.0 - 1e-6
        assert adv.max() <= 1.0 + 1e-6
        assert np.max(np.abs(adv - clean)) <= eps + 1e-6


def test_project_linf_is_identity_within_ball():
    clean = np.full((4, 4, 3), 0.5, np.float32)
    small = clean + 0.001
    out = project_linf(small, clean, 8 / 255)
    assert np.allclose(out, small, atol=1e-6)
