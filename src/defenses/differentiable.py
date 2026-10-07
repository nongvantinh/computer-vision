"""Differentiable surrogates of defenses, for defense-aware (adaptive) attacks.

An adaptive attacker optimizes THROUGH the defense. The real defenses are
numpy/Pillow/subprocess code with no gradient, so each one gets a torch surrogate
and the attack uses the standard BPDA pattern (Athalye et al., 2018):

    y = real(x) in the forward pass, d surrogate(x) / dx in the backward pass

(implemented as a custom autograd function, so the forward is bit-exact). The forward value equals the REAL defense output where a real forward is available
in the attack process (JPEG), so the model is attacked on exactly what it will see.
The backward pass uses the surrogate's Jacobian. For a defense whose real forward
cannot run in the attack process (Dangerzone needs rootless Podman, which the GPU
host lacks) the surrogate is also the forward, and its fidelity to the real tool is
measured and reported separately; final success is always scored through the real
defense, never through the surrogate.

Tensors are (N, 3, H, W) float32 in [0, 1], the layout of LlavaWrapper.
"""
from __future__ import annotations

import io

import numpy as np

# libjpeg standard tables (ITU T.81 Annex K), natural row-major order.
_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
         14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
         49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
           24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def quant_tables(quality: int) -> tuple[np.ndarray, np.ndarray]:
    """libjpeg jpeg_set_quality(quality, force_baseline=TRUE) tables, 8x8 natural order."""
    q = int(min(max(quality, 1), 100))
    scale = 5000 // q if q < 50 else 200 - q * 2
    out = []
    for base in (_LUMA, _CHROMA):
        t = [min(max((b * scale + 50) // 100, 1), 255) for b in base]
        out.append(np.array(t, dtype=np.float32).reshape(8, 8))
    return out[0], out[1]


def _dct_matrix(torch):
    """Orthonormal 8-point DCT-II matrix C, so F = C X C^T equals the JPEG DCT."""
    k = torch.arange(8, dtype=torch.float64).unsqueeze(1)
    n = torch.arange(8, dtype=torch.float64).unsqueeze(0)
    c = torch.cos((2 * n + 1) * k * torch.pi / 16)
    c[0] *= 1 / np.sqrt(2)
    return (c * 0.5).to(torch.float32)


def _round(torch, x, mode: str):
    """Rounding with a usable gradient: straight-through, or Shin & Song's cubic.

    Use "ste" (the default). The cubic variant has gradient 3(x - round(x))^2, which is
    ~0 for any input already on the 8-bit grid, i.e. every real image; on a proxy
    objective it kept 0.0 of the gain that "ste" kept (0.56 at JPEG 90). It stays
    available only so that finding can be reproduced.
    """
    r = torch.round(x)
    if mode == "ste":
        return x + (r - x).detach()
    if mode == "cubic":                       # x + (x - round(x))^3 in the backward
        return r.detach() + (x - r.detach()) ** 3
    raise ValueError(f"unknown rounding mode {mode!r}")


def bpda_apply(defense, x):
    """Backward Pass Differentiable Approximation, with a bit-exact real forward.

    `defense` needs `.real(x)` (the true transform, no gradient) and `.surrogate(x)`
    (differentiable). The returned tensor equals `defense.real(x)` exactly; the
    gradient is the vector-Jacobian product of `defense.surrogate`.
    """
    import torch

    class _Bpda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            ctx.save_for_backward(inp)
            return defense.real(inp)

        @staticmethod
        def backward(ctx, grad_out):
            (inp,) = ctx.saved_tensors
            with torch.enable_grad():
                xd = inp.detach().requires_grad_(True)
                (gx,) = torch.autograd.grad(defense.surrogate(xd), xd, grad_out)
            return gx

    return _Bpda.apply(x)


class DifferentiableJpeg:
    """Torch JPEG round trip (4:2:0, baseline) mirroring Pillow/libjpeg stages.

    Stages: RGB->YCbCr, 2x2 chroma average, level shift, 8x8 DCT, quantize with the
    libjpeg quality tables, dequantize, inverse DCT, fancy (bilinear) chroma
    upsampling, YCbCr->RGB, rounding to 8 bits. `round_mode` controls the gradient
    through every rounding.
    """

    def __init__(self, quality: int = 90, round_mode: str = "ste", torch_module=None):
        import torch
        self.torch = torch_module or torch
        self.quality = int(quality)
        self.round_mode = round_mode
        self.name = f"jpeg{self.quality}_diff"
        t = self.torch
        ql, qc = quant_tables(self.quality)
        self._ql, self._qc = t.tensor(ql), t.tensor(qc)
        self._dct = _dct_matrix(t)

    # -- helpers -------------------------------------------------------------- #
    def _blocks(self, plane):
        t = self.torch
        n, h, w = plane.shape
        b = plane.reshape(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
        return b                                            # (N, H/8, W/8, 8, 8)

    def _unblocks(self, b, h, w):
        n = b.shape[0]
        return b.permute(0, 1, 3, 2, 4).reshape(n, h, w)

    def _codec(self, plane, qtab):
        t = self.torch
        n, h, w = plane.shape
        c = self._dct.to(plane.device)
        q = qtab.to(plane.device)
        b = self._blocks(plane - 128.0)
        f = c @ b @ c.T                                      # DCT
        f = _round(t, f / q, self.round_mode) * q            # quantize + dequantize
        rec = c.T @ f @ c                                    # inverse DCT
        return self._unblocks(rec, h, w) + 128.0

    def surrogate(self, x):
        """Differentiable approximation of the JPEG round trip; x is in [0, 1]."""
        t = self.torch
        F = t.nn.functional
        n, _, h, w = x.shape
        if h % 16 or w % 16:
            raise ValueError("height and width must be multiples of 16 for 4:2:0")
        x8 = _round(t, x * 255.0, self.round_mode)           # the real path starts at uint8
        r, g, b = x8[:, 0], x8[:, 1], x8[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        cb = F.avg_pool2d(cb.unsqueeze(1), 2)                # 4:2:0 chroma subsampling
        cr = F.avg_pool2d(cr.unsqueeze(1), 2)
        y = self._codec(y, self._ql)
        cb = self._codec(cb.squeeze(1), self._qc).unsqueeze(1)
        cr = self._codec(cr.squeeze(1), self._qc).unsqueeze(1)
        cb = F.interpolate(cb, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
        cr = F.interpolate(cr, scale_factor=2, mode="bilinear", align_corners=False).squeeze(1)
        r2 = y + 1.402 * (cr - 128.0)
        g2 = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
        b2 = y + 1.772 * (cb - 128.0)
        out = t.stack([r2, g2, b2], dim=1)
        out = _round(t, out.clamp(0.0, 255.0), self.round_mode)
        return out / 255.0

    def real(self, x):
        """The REAL Pillow JPEG round trip on a (N,3,H,W) tensor; no gradient."""
        from PIL import Image
        t = self.torch
        arr = x.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        outs = []
        for a in arr:
            img = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.quality, optimize=False,
                     subsampling="4:2:0")
            buf.seek(0)
            outs.append(np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0)
        out = t.from_numpy(np.stack(outs)).permute(0, 3, 1, 2)
        return out.to(x.device, x.dtype)

    def __call__(self, x):
        """BPDA: forward = real Pillow JPEG (bit-exact), backward = surrogate Jacobian."""
        return bpda_apply(self, x)
