"""``splat_stamp`` (fused 2D Gaussian stamping for the splat tomography renderer) vs a torch
reference that copies the stamping of ``quantem.core.ml.splat.renderer.render_projections`` in
pixel units (round-half-to-even integer window centre, Gaussian at the continuous centre
evaluated at pixel centres, index-space window clipped to the image, scatter-add).

Forward images must agree to 1e-5 of the image maximum; gradients are anchored to an fp64
reference exactly as in ``test_kplanes_tilted3.py`` (``_grad_bar``). GPU tests skip without a
CUDA device; the validation test runs on the CPU."""

import math

import pytest
import torch

from quantem.cuda.core.ml import splat_stamp
from quantem.cuda.core.ml._ops import _splat_stamp

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def reference(centers, amps, sigmas, image_shape, radius):
    """Torch stamp in pixel units (any dtype / device); identical to the renderer's torch path."""
    T = int(centers.shape[0])
    H, W = image_shape
    dtype, device = centers.dtype, centers.device
    rng = torch.arange(-radius, radius + 1, device=device)
    o0, o1 = torch.meshgrid(rng, rng, indexing="ij")
    off = torch.stack((o0.reshape(-1), o1.reshape(-1)), dim=1)  # (W2, 2)
    ci = torch.round(centers.detach()).long()
    pix = ci[:, :, None, :] + off[None, None]  # (T, N, W2, 2)
    d = pix.to(dtype) - centers[:, :, None, :]
    r2 = (d * d).sum(-1)
    norm = amps / (2.0 * math.pi * sigmas * sigmas)
    inv2s2 = 0.5 / (sigmas * sigmas)
    val = norm[None, :, None] * torch.exp(-r2 * inv2s2[None, :, None])
    valid = (pix[..., 0] >= 0) & (pix[..., 0] < H) & (pix[..., 1] >= 0) & (pix[..., 1] < W)
    val = val * valid.to(dtype)
    prow = pix[..., 0].clamp(0, H - 1)
    pcol = pix[..., 1].clamp(0, W - 1)
    tidx = torch.arange(T, device=device).view(T, 1, 1)
    flat = (tidx * H + prow) * W + pcol
    acc = torch.zeros(T * H * W, device=device, dtype=dtype)
    return acc.scatter_add(0, flat.reshape(-1), val.reshape(-1)).view(T, H, W)


def _inputs(T, N, H, W, seed, margin=3):
    """Random atoms, some of them partly or fully outside the image (``margin`` px beyond
    the border), widths 0.8-2.5 px, radius = ceil(4 sigma_max); fp32 on the GPU."""
    gen = torch.Generator().manual_seed(seed)
    rows = torch.rand(T, N, generator=gen) * (H + 2 * margin) - margin
    cols = torch.rand(T, N, generator=gen) * (W + 2 * margin) - margin
    centers = torch.stack((rows, cols), dim=-1)
    sigmas = 0.8 + 1.7 * torch.rand(N, generator=gen)
    amps = 0.5 + 1.5 * torch.rand(N, generator=gen)
    radius = int(math.ceil(4.0 * float(sigmas.max())))
    return (centers.cuda(), amps.cuda(), sigmas.cuda(), (H, W), radius)


def _grad_bar(err_torch: float, g64: torch.Tensor) -> float:
    """Tolerance against the fp64 reference: 2x torch's own fp32 error or 1e-5 of the largest
    gradient entry, whichever is larger (the forward's atomics make the kernel's round-off
    independent of torch's on the same draw; see test_kplanes_tilted3._grad_bar)."""
    return max(2.0 * err_torch, max(1e-6, 1e-5 * g64.abs().max().item()))


@requires_cuda
def test_forward_matches_torch_stamp():
    """Random shapes, windows larger than the image, one tilt, radius 0."""
    cases = [
        ("random", 3, 50, 24, 32, 0),
        ("many atoms", 2, 300, 40, 48, 1),
        ("window wider than the image", 2, 20, 9, 7, 2),
        ("one tilt, one atom", 1, 1, 16, 16, 3),
    ]
    for name, T, N, H, W, seed in cases:
        centers, amps, sigmas, shape, radius = _inputs(T, N, H, W, seed)
        if name == "window wider than the image":
            radius = max(H, W)
        expected = reference(centers.double(), amps.double(), sigmas.double(), shape, radius)
        actual = splat_stamp(centers, amps, sigmas, shape, radius)
        assert actual.shape == (T, H, W), name
        scale = expected.abs().max().item()
        assert scale > 0, name
        err = (actual.double() - expected).abs().max().item() / scale
        assert err < 1e-5, f"{name}: relative image error {err:.2e}"
    # radius 0: only the pixel under the rounded centre, no neighbours
    centers, amps, sigmas, shape, _ = _inputs(2, 10, 12, 12, 4)
    expected = reference(centers.double(), amps.double(), sigmas.double(), shape, 0)
    actual = splat_stamp(centers, amps, sigmas, shape, 0)
    assert (actual > 0).sum() <= 20 and torch.allclose(actual.double(), expected, rtol=1e-5)


@requires_cuda
def test_special_atoms_edge_offimage_zero_amplitude_nonfinite():
    """Half-integer centres (round half to even), an atom straddling the border, one fully
    off-image, one with zero amplitude (nonzero amplitude gradient), one non-finite (skipped,
    zero gradients, no NaN in the image)."""
    H, W, r = 16, 20, 3
    centers = torch.tensor(
        [
            [[2.5, 3.5], [0.3, 10.2], [-50.0, 5.0], [8.0, 8.0], [float("nan"), 4.0], [9.6, 1.2]],
        ],
        dtype=torch.float32,
    ).cuda()
    amps = torch.tensor([1.0, 1.5, 2.0, 0.0, 1.0, 0.7]).cuda()
    sigmas = torch.tensor([0.9, 1.1, 1.0, 1.2, 1.0, 0.8]).cuda()
    keep = [0, 1, 2, 3, 5]  # the reference cannot take the NaN atom
    expected = reference(
        centers[:, keep].double(), amps[keep].double(), sigmas[keep].double(), (H, W), r
    )
    actual = splat_stamp(centers, amps, sigmas, (H, W), r)
    assert torch.isfinite(actual).all(), "NaN atom poisoned the image"
    err = (actual.double() - expected).abs().max().item() / expected.max().item()
    assert err < 1e-5, f"relative image error {err:.2e}"
    # half-even rounding: (2.5, 3.5) -> window centre (2, 4), so row 2 - r = -1 is clipped and
    # row 2 + r = 5 is stamped, column 4 + r = 7 is stamped and 4 + r + 1 = 8 is not
    single = splat_stamp(centers[:, :1], amps[:1], sigmas[:1], (H, W), r)[0]
    assert single[5, 4] > 0 and single[6, 4] == 0 and single[2, 7] > 0 and single[2, 8] == 0
    # gradients
    c = centers.clone().requires_grad_(True)
    a = amps.clone().requires_grad_(True)
    s = sigmas.clone().requires_grad_(True)
    out = splat_stamp(c, a, s, (H, W), r)
    out.sum().backward()
    assert torch.isfinite(c.grad).all() and torch.isfinite(a.grad).all()
    assert a.grad[3] > 0, "zero-amplitude atom must still get an amplitude gradient"
    assert float(c.grad[0, 3].abs().sum()) == 0.0, "uniform upstream grad: centre grad is 0"
    for idx, what in ((2, "fully off-image"), (4, "non-finite")):
        assert float(c.grad[0, idx].abs().sum()) == 0.0 and float(a.grad[idx]) == 0.0, what
        assert float(s.grad[idx]) == 0.0, what
    assert float(a.grad[1]) < float(a.grad[5]), "straddling atom loses the clipped pixels"


@requires_cuda
def test_grads_match_fp64_reference():
    centers, amps, sigmas, shape, radius = _inputs(3, 80, 28, 36, 11)
    upstream = torch.randn(
        3, *shape, device="cuda", generator=torch.Generator("cuda").manual_seed(1)
    )

    def grads(fn, dtype):
        ins = tuple(t.detach().to(dtype).requires_grad_(True) for t in (centers, amps, sigmas))
        out = fn(*ins, shape, radius)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), ins)

    g64 = grads(reference, torch.float64)
    g_torch = grads(reference, torch.float32)
    g_kernel = grads(splat_stamp, torch.float32)
    for name, gt, gk, gd in zip(("centers", "amps", "sigmas"), g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        assert err_kernel <= _grad_bar(err_torch, gd), (
            f"grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )
    # the backward is deterministic (no atomics): two runs are bit-identical
    g_again = grads(splat_stamp, torch.float32)
    assert all(torch.equal(x, y) for x, y in zip(g_kernel, g_again))


@requires_cuda
def test_opcheck_fakes_and_autograd():
    """torch.library.opcheck: schema, fake kernels and the autograd registration."""
    centers, amps, sigmas, (H, W), radius = _inputs(2, 12, 10, 14, 5)
    args = (
        centers.requires_grad_(True),
        amps.requires_grad_(True),
        sigmas.requires_grad_(True),
        H,
        W,
        radius,
    )
    torch.library.opcheck(_splat_stamp, args)


def test_validation_errors():
    """Shape, dtype and device checks raise before any kernel call (the message names the
    check, so a different check cannot pass for it)."""
    c = torch.zeros(2, 5, 2)
    a = torch.ones(5)
    s = torch.ones(5)
    bad = [
        ((torch.zeros(2, 5, 3), a, s, (8, 8), 2), ValueError, r"centers \[T, N, 2\]"),
        ((c, torch.ones(4), s, (8, 8), 2), ValueError, r"amps \[N\]"),
        ((c, a, torch.ones(5, 1), (8, 8), 2), ValueError, r"sigmas \[N\]"),
        ((c.double(), a.double(), s.double(), (8, 8), 2), TypeError, "fp32-only"),
        ((c, a, s, (8, 8), 2), ValueError, "requires CUDA tensors"),
    ]
    if torch.cuda.is_available():
        cc, ac, sc = c.cuda(), a.cuda(), s.cuda()
        bad += [
            ((cc, ac, sc, (8, 8), -1), ValueError, "radius must be"),
            ((cc, ac, sc, (0, 8), 2), ValueError, "must be positive"),
        ]
    for args, exc, msg in bad:
        with pytest.raises(exc, match=msg):
            splat_stamp(*args)
