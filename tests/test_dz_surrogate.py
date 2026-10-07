"""The committed Dangerzone surrogates: loadable, banded, differentiable, and passing G2."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.defenses.differentiable import (FittedLinearSurrogate, build_adaptive_defense,  # noqa: E402
                                         SURROGATE_DIR)
from src.evaluation.gates import g2_surrogate_fidelity  # noqa: E402

VARIANTS = ["lossless", "jpeg"]


@pytest.mark.parametrize("v", VARIANTS)
def test_committed_fit_report_passes_gate_g2(v):
    report = json.loads((SURROGATE_DIR / f"dangerzone_{v}.json").read_text())
    g = g2_surrogate_fidelity(report, min_psnr_db=40.0)
    assert g["passed"], g
    assert report["n_heldout"] > 0 and report["n_train"] > report["n_heldout"]


@pytest.mark.parametrize("v", VARIANTS)
def test_surrogate_matrix_is_banded(v):
    B = np.load(SURROGATE_DIR / f"dangerzone_{v}.npz")["B"]
    assert B.shape == (336, 336)
    i, j = np.indices(B.shape)
    assert np.all(B[np.abs(i - j) > 8] == 0)            # banded: local resampling only


@pytest.mark.parametrize("v", VARIANTS)
def test_surrogate_preserves_the_brightness_of_flat_images(v):
    """B and the colour matrix share a gain (B acts on both sides), so the invariant
    to check is the composite: a flat gray stays gray to within about two levels."""
    s = FittedLinearSurrogate(SURROGATE_DIR / f"dangerzone_{v}.npz")
    for g in (0.2, 0.5, 0.8):
        with torch.no_grad():
            y = s.surrogate(torch.full((1, 3, 336, 336), g))
        centre = y[..., 20:-20, 20:-20] * 255
        assert abs(float(centre.mean()) - g * 255) < 2.5


@pytest.mark.parametrize("name", ["dangerzone_ll", "dangerzone"])
def test_adaptive_dangerzone_defenses_are_differentiable_and_stay_in_range(name):
    d = build_adaptive_defense(name)
    x = torch.rand(1, 3, 336, 336, requires_grad=True)
    y = d(x)
    assert y.shape == x.shape and float(y.detach().min()) >= 0.0 and float(y.detach().max()) <= 1.0
    (y * torch.randn_like(y)).sum().backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().mean()) > 0


def test_surrogate_rejects_the_wrong_image_size():
    s = FittedLinearSurrogate(SURROGATE_DIR / "dangerzone_lossless.npz")
    with pytest.raises(ValueError):
        s.surrogate(torch.rand(1, 3, 224, 224))


def test_unknown_adaptive_defense_is_an_error_not_a_silent_identity():
    with pytest.raises(ValueError):
        build_adaptive_defense("mdcore")
