"""Differentiable JPEG surrogate: faithful forward, useful gradient, real forward in the loop."""
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from src.defenses.differentiable import DifferentiableJpeg, quant_tables  # noqa: E402
from src.defenses.jpeg import JpegDefense  # noqa: E402


def _structured(n=96, seed=0):
    """Smooth gradients + edges + mild texture, like a photo and unlike white noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.stack([0.5 + 0.4 * np.sin(6 * xx + yy), 0.5 + 0.4 * np.cos(5 * yy - xx),
                    xx * yy], axis=-1)
    img[n // 3: n // 2, n // 4: 3 * n // 4] = [0.9, 0.2, 0.1]            # a hard edge
    img += rng.normal(0, 0.01, img.shape)
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def _psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 99.0 if m == 0 else 10 * np.log10(1 / m)


@pytest.mark.parametrize("q", [10, 30, 50, 75, 90, 95])
def test_quantization_tables_equal_the_ones_pillow_writes(q):
    probe = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    buf = io.BytesIO()
    probe.save(buf, "JPEG", quality=q, subsampling="4:2:0")
    buf.seek(0)
    qt = Image.open(buf).quantization
    ql, qc = quant_tables(q)
    assert list(qt[0]) == list(ql.reshape(-1)) and list(qt[1]) == list(qc.reshape(-1))


@pytest.mark.parametrize("q,min_psnr", [(90, 42.0), (50, 39.0)])
def test_surrogate_forward_tracks_pillow(q, min_psnr):
    x = _structured()
    j = DifferentiableJpeg(q)
    with torch.no_grad():
        assert _psnr(j.surrogate(x), j.real(x)) > min_psnr


def test_real_forward_is_the_repository_jpeg_defense():
    x = _structured()
    ours = DifferentiableJpeg(75).real(x)[0].permute(1, 2, 0).numpy()
    ref = JpegDefense(75).sanitize(x[0].permute(1, 2, 0).numpy()).image
    assert np.array_equal(ours, ref)


def test_hybrid_forward_value_is_the_real_defense_output():
    x = _structured()
    j = DifferentiableJpeg(50)
    with torch.no_grad():
        assert torch.equal(j(x), j.real(x))


def test_hybrid_gradient_is_the_surrogate_gradient_and_is_nonzero():
    x = _structured().requires_grad_(True)
    w = torch.randn_like(x)
    j = DifferentiableJpeg(50)
    (j(x) * w).sum().backward()
    g_hybrid = x.grad.clone()
    x.grad = None
    (j.surrogate(x) * w).sum().backward()
    assert torch.isfinite(g_hybrid).all() and float(g_hybrid.abs().mean()) > 0
    assert torch.allclose(g_hybrid, x.grad, atol=1e-6)


def test_gradient_depends_on_the_defense_quality():
    x = _structured().requires_grad_(True)
    w = torch.randn_like(x)
    grads = []
    for q in (90, 10):
        x.grad = None
        (DifferentiableJpeg(q)(x) * w).sum().backward()
        grads.append(x.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


def _pgd_gain(x0, w, defense, steps=25, eps=16 / 255, step=1 / 255):
    """Maximise <w, defense(x)> inside the L-inf ball; defense=None ignores the defense."""
    real = DifferentiableJpeg(50)
    x = x0.clone()
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        y = x if defense is None else defense(x)
        (w * y).sum().backward()
        with torch.no_grad():
            x = torch.min(torch.max(x + step * x.grad.sign(), x0 - eps), x0 + eps).clamp(0, 1)
    return float((w * real.real(x)).sum() - (w * real.real(x0)).sum())


def test_optimising_through_the_surrogate_beats_ignoring_the_defense():
    """The point of an adaptive attack: scored through the REAL JPEG, it keeps more."""
    torch.manual_seed(0)
    x0 = _structured()
    w = torch.randn_like(x0)
    adaptive = _pgd_gain(x0, w, DifferentiableJpeg(50))
    oblivious = _pgd_gain(x0, w, None)
    assert adaptive > 1.2 * oblivious > 0
