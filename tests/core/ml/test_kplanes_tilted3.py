"""Per-plane-type TILTED K-Planes kernels (``*_fuse3``) vs the pure-torch reference.

``kplanes_tilted_fuse3`` / ``kplanes_tilted_tv_fuse3`` are the single-tensor
kernels with one tensor per plane TYPE, so each type can carry its own
resolution. Plane type ``p`` samples the coordinate pair of quantem's
``idx = [[0, 1], [2, 0], [1, 2]]`` (first entry → grid_sample's x / the W
axis, second → y / the H axis), exactly as in
``core/ml/models/kplanes.py::interpolate_ms_features_tilted``.

Tolerances mirror ``test_kplanes_tilted.py``: forward rtol 1e-4 / atol 5e-6,
gradients within 2x torch's own fp32 error against an fp64 reference (4x for
the TV rotation grads, whose atomics accumulate over all points and taps).
"""

import pytest
import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import (
    kplanes_tilted_fuse,
    kplanes_tilted_fuse3,
    kplanes_tilted_tv_fuse,
    kplanes_tilted_tv_fuse3,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

# (B, T, C, (H0, W0), (H1, W1), (H2, W2)) — three DIFFERENT plane shapes, so a
# per-type (H, W) mix-up cannot pass; C not a multiple of 4; odd sizes.
CONFIGS3 = [
    (257, 3, 5, (7, 9), (9, 11), (11, 7)),
    (4096, 4, 8, (64, 200), (200, 24), (24, 64)),
    (2000, 8, 32, (50, 33), (33, 17), (17, 50)),
]

IDX = [[0, 1], [2, 0], [1, 2]]


def reference3(
    pts: torch.Tensor,
    rotations: torch.Tensor,
    plane0: torch.Tensor,
    plane1: torch.Tensor,
    plane2: torch.Tensor,
) -> torch.Tensor:
    """Three grid_sample calls, one per plane type, Hadamard-multiplied.

    Same semantics as one level of quantem's interpolate_ms_features_tilted,
    with the (3T, C, H, W) tensor split by plane type.
    """
    T = rotations.shape[0]
    B = pts.shape[0]
    C = plane0.shape[1]
    rotated = torch.einsum("tij,bj->tbi", rotations, pts)  # (T, B, 3)
    prod = None
    for plane, (i, j) in zip((plane0, plane1, plane2), IDX):
        coords = torch.stack((rotated[..., i], rotated[..., j]), dim=-1)  # (T, B, 2)
        sampled = F.grid_sample(
            plane,
            coords.unsqueeze(1),  # (T, 1, B, 2)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        ).squeeze(2)  # (T, C, B)
        prod = sampled if prod is None else prod * sampled
    assert prod is not None
    return prod.permute(2, 0, 1).reshape(B, T * C)


def reference3_tv(
    pts: torch.Tensor,
    rotations: torch.Tensor,
    plane0: torch.Tensor,
    plane1: torch.Tensor,
    plane2: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """reference3 evaluated at the 4 TV taps; returns [4, B, T*C]."""
    taps = []
    for axis in range(4):
        if axis == 0:
            p = pts
        else:
            delta = torch.zeros_like(pts)
            delta[:, axis - 1] = h
            p = pts + delta
        taps.append(reference3(p, rotations, plane0, plane1, plane2))
    return torch.stack(taps, dim=0)


def _inputs3(B, T, C, hw0, hw1, hw2, seed=0, requires_grad=False, pts_scale=1.1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, lo, hi):
        return torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(
            lo, hi, generator=gen
        )

    # pts_scale > 1 exercises the border clamp on a fraction of the points
    pts = rand((B, 3), -pts_scale, pts_scale)
    rotations = rand((T, 3, 3), -1.0, 1.0)
    planes = [rand((T, C, h, w), 0.1, 0.5) for (h, w) in (hw0, hw1, hw2)]
    tensors = [pts, rotations, *planes]
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)
    return tuple(tensors)


# ── kplanes_tilted_fuse3 ──────────────────────────────────────────────────


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS3)
def test_fuse3_forward_matches_reference(cfg):
    inp = _inputs3(*cfg)
    expected = reference3(*inp)
    actual = kplanes_tilted_fuse3(*inp)
    assert actual.shape == (cfg[0], cfg[1] * cfg[2])
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-6)


def _grad_bar(name: str, err_torch: float, g64: torch.Tensor) -> float:
    """Tolerance for one gradient against its fp64 reference.

    The kernel accumulates with atomic adds, so its rounding error changes from
    run to run and is not tied to torch's fp32 error on the same draw. Two
    allowances are combined: a multiple of torch's own fp32 error (4x for the
    rotation gradients, which sum over every point and tap; 2x elsewhere), and
    a floor of 1e-5 of the gradient's largest entry, which is far above the
    round-off of a few-thousand-term fp32 sum (~1e-6 relative) and far below
    any real error (a wrong axis or a wrong tap is order one). Without the
    floor the bar was set by a single random draw and failed about one run in
    twelve.
    """
    k = 4.0 if name == "rotations" else 2.0
    floor = max(1e-6, 1e-5 * g64.abs().max().item())
    return max(k * err_torch, floor)


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS3)
def test_fuse3_grads_match_fp64_reference(cfg):
    """fp64-anchored, exactly as the single-tensor test: the kernel's deviation
    from the fp64 reference gradients must be no worse than 2x torch's own
    fp32 deviation."""
    inp = _inputs3(*cfg)
    upstream = torch.randn_like(reference3(*inp))

    def grads(fn, dtype):
        ins = tuple(t.detach().to(dtype).requires_grad_(True) for t in inp)
        out = fn(*ins)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), ins)

    g64 = grads(reference3, torch.float64)
    g_torch = grads(reference3, torch.float32)
    g_kernel = grads(kplanes_tilted_fuse3, torch.float32)

    names = ("pts", "rotations", "plane0", "plane1", "plane2")
    for name, gt, gk, gd in zip(names, g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        assert err_kernel <= _grad_bar(name, err_torch, gd), (
            f"grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )


@requires_cuda
def test_fuse3_points_exactly_on_border():
    """±1 coords sit exactly on the clip boundary, where torch zeroes the
    coordinate gradient — with a different (H, W) per plane type."""
    T, C = 2, 3
    gen = torch.Generator(device="cuda").manual_seed(7)
    rotations = torch.stack(
        [torch.eye(3, device="cuda"), torch.eye(3, device="cuda").roll(1, 0)]
    ).float()
    planes = [
        torch.empty((T, C, h, w), device="cuda").uniform_(0.1, 0.5, generator=gen)
        for (h, w) in ((11, 7), (7, 13), (13, 11))
    ]
    base = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [2.0, 0.5, -3.0],
        ],
        device="cuda",
    )
    for grad_rotations in (False, True):
        pts = base.clone().requires_grad_(True)
        rot = rotations.clone().requires_grad_(True) if grad_rotations else rotations
        expected = reference3(pts, rot, *planes)
        actual = kplanes_tilted_fuse3(pts, rot, *planes)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        gref = torch.autograd.grad(expected.sum(), pts, retain_graph=True)[0]
        gact = torch.autograd.grad(actual.sum(), pts, retain_graph=True)[0]
        torch.testing.assert_close(gact, gref, rtol=1e-4, atol=1e-6)


@requires_cuda
def test_fuse3_non_contiguous_inputs():
    cfg = (128, 2, 4, (9, 11), (11, 5), (5, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg)
    pts_nc = pts.t().contiguous().t()
    planes_nc = [p.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2) for p in (p0, p1, p2)]
    assert not pts_nc.is_contiguous() and not any(p.is_contiguous() for p in planes_nc)
    torch.testing.assert_close(
        kplanes_tilted_fuse3(pts_nc, rotations, *planes_nc),
        reference3(pts, rotations, p0, p1, p2),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_cuda
def test_fuse3_channels_last_planes():
    """Channels-last planes (how quantem stores K-Planes grids) give the same output and plane
    gradients as NCHW planes, and the gradients come back with the planes' own strides."""
    cfg = (300, 3, 6, (9, 11), (11, 5), (5, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg)
    planes = [p.detach().requires_grad_(True) for p in (p0, p1, p2)]
    planes_cl = [
        p.detach().contiguous(memory_format=torch.channels_last).requires_grad_(True)
        for p in (p0, p1, p2)
    ]
    out = kplanes_tilted_fuse3(pts, rotations, *planes)
    out_cl = kplanes_tilted_fuse3(pts, rotations, *planes_cl)
    torch.testing.assert_close(out_cl, out, rtol=0, atol=0)
    g = torch.autograd.grad(out.square().sum(), planes)
    g_cl = torch.autograd.grad(out_cl.square().sum(), planes_cl)
    for a, b, p in zip(g_cl, g, planes_cl):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        assert a.stride() == p.stride()


@requires_cuda
def test_fuse3_upstream_grad_scaling():
    cfg = (64, 2, 4, (9, 7), (7, 11), (11, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg, requires_grad=True)
    kplanes_tilted_fuse3(pts, rotations, p0, p1, p2).sum().backward()
    g1 = p1.grad.clone()
    p1.grad = None
    (kplanes_tilted_fuse3(pts, rotations, p0, p1, p2) * 3.0).sum().backward()
    torch.testing.assert_close(p1.grad, g1 * 3.0, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_fuse3_compile_fullgraph():
    cfg = (512, 2, 4, (17, 9), (9, 21), (21, 17))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg, requires_grad=True)

    @torch.compile(fullgraph=True)
    def f(pt, r, a, b, c):
        return kplanes_tilted_fuse3(pt, r, a, b, c).square().mean()

    f(pts, rotations, p0, p1, p2).backward()
    for t in (pts, rotations, p0, p1, p2):
        assert t.grad is not None


@requires_cuda
def test_fuse3_input_validation():
    cfg = (16, 2, 4, (9, 7), (7, 11), (11, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg)
    with pytest.raises(ValueError, match=r"pts \[B, 3\]"):
        kplanes_tilted_fuse3(pts[:, :2], rotations, p0, p1, p2)
    with pytest.raises(ValueError, match=r"rotations \[T, 3, 3\]"):
        kplanes_tilted_fuse3(pts, rotations[:, :2, :], p0, p1, p2)
    with pytest.raises(ValueError, match=r"plane1 \[T, C, H, W\]"):
        kplanes_tilted_fuse3(pts, rotations, p0, p1[:1], p2)
    with pytest.raises(ValueError, match="same channel count"):
        kplanes_tilted_fuse3(pts, rotations, p0, p1[:, :2], p2)
    with pytest.raises(TypeError, match="fp32-only"):
        kplanes_tilted_fuse3(pts.double(), rotations, p0, p1, p2)


def test_fuse3_cpu_rejected():
    pts = torch.rand(8, 3)
    rotations = torch.rand(2, 3, 3)
    planes = [torch.rand(2, 4, h, w) for (h, w) in ((9, 7), (7, 11), (11, 9))]
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_tilted_fuse3(pts, rotations, *planes)
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_tilted_tv_fuse3(pts, rotations, *planes, h=0.01)


# ── equal shapes: fuse3 must reproduce the single-tensor path exactly ──────

EQUAL_CONFIGS = [
    (37, 1, 1, 5, 5),
    (257, 3, 5, 7, 9),
    (2000, 8, 32, 50, 33),
]


def _split_planes(plane: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(3T, C, H, W) with plane index t*3 + p → three (T, C, H, W) tensors."""
    assert plane.shape[0] % 3 == 0
    return tuple(plane[p::3].contiguous() for p in range(3))


@requires_cuda
@pytest.mark.parametrize("cfg", EQUAL_CONFIGS)
def test_fuse3_equals_single_tensor_path(cfg):
    """With all three plane types the same shape, fuse3 and fuse do exactly the
    same arithmetic on exactly the same addresses, so the forwards must be
    bit-identical (not merely close)."""
    B, T, C, H, W = cfg
    gen = torch.Generator(device="cuda").manual_seed(3)
    pts = torch.empty((B, 3), device="cuda").uniform_(-1.1, 1.1, generator=gen)
    rotations = torch.empty((T, 3, 3), device="cuda").uniform_(-1.0, 1.0, generator=gen)
    plane = torch.empty((3 * T, C, H, W), device="cuda").uniform_(0.1, 0.5, generator=gen)
    p0, p1, p2 = _split_planes(plane)

    old = kplanes_tilted_fuse(pts, rotations, plane)
    new = kplanes_tilted_fuse3(pts, rotations, p0, p1, p2)
    if not torch.equal(new, old):
        raise AssertionError(
            f"fuse3 != fuse for {cfg}: max abs diff {(new - old).abs().max().item():.3e}"
        )

    old_tv = kplanes_tilted_tv_fuse(pts, rotations, plane, h=0.03)
    new_tv = kplanes_tilted_tv_fuse3(pts, rotations, p0, p1, p2, h=0.03)
    if not torch.equal(new_tv, old_tv):
        raise AssertionError(
            f"tv_fuse3 != tv_fuse for {cfg}: "
            f"max abs diff {(new_tv - old_tv).abs().max().item():.3e}"
        )


@requires_cuda
def test_fuse3_grads_equal_single_tensor_path():
    """Gradients of the two paths agree to float noise. Not bit-exact by
    construction: both accumulate into the plane grads with atomics, whose
    summation order is not reproducible even run to run."""
    B, T, C, H, W = 512, 3, 8, 17, 13
    gen = torch.Generator(device="cuda").manual_seed(11)
    pts = torch.empty((B, 3), device="cuda").uniform_(-1.1, 1.1, generator=gen)
    rotations = torch.empty((T, 3, 3), device="cuda").uniform_(-1.0, 1.0, generator=gen)
    plane = torch.empty((3 * T, C, H, W), device="cuda").uniform_(0.1, 0.5, generator=gen)
    upstream = torch.randn((B, T * C), device="cuda", generator=gen)

    pl = plane.clone().requires_grad_(True)
    pt = pts.clone().requires_grad_(True)
    rt = rotations.clone().requires_grad_(True)
    (kplanes_tilted_fuse(pt, rt, pl) * upstream).sum().backward()

    parts = [p.clone().requires_grad_(True) for p in _split_planes(plane)]
    pt3 = pts.clone().requires_grad_(True)
    rt3 = rotations.clone().requires_grad_(True)
    (kplanes_tilted_fuse3(pt3, rt3, *parts) * upstream).sum().backward()

    torch.testing.assert_close(pt3.grad, pt.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rt3.grad, rt.grad, rtol=1e-5, atol=1e-6)
    for p in range(3):
        torch.testing.assert_close(parts[p].grad, pl.grad[p::3], rtol=1e-5, atol=1e-6)


# ── kplanes_tilted_tv_fuse3 ───────────────────────────────────────────────


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS3)
def test_tv_fuse3_tap0_matches_base(cfg):
    """tap 0 must be bitwise-identical to kplanes_tilted_fuse3."""
    B, T, C = cfg[0], cfg[1], cfg[2]
    inp = _inputs3(*cfg, seed=42)
    base = kplanes_tilted_fuse3(*inp)
    tv_out = kplanes_tilted_tv_fuse3(*inp, h=0.01)
    assert tv_out.shape == (4, B, T * C)
    torch.testing.assert_close(tv_out[0], base, rtol=0.0, atol=0.0)


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS3)
def test_tv_fuse3_taps_match_reference(cfg):
    inp = _inputs3(*cfg, seed=7)
    h = 0.05
    expected = reference3_tv(*inp, h=h)
    actual = kplanes_tilted_tv_fuse3(*inp, h=h)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-6)


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS3)
def test_tv_fuse3_grads_match_fp64_reference(cfg):
    """fp64-anchored, as in the single-tensor TV test: 4x for the rotation
    grads (atomics over all points and taps), 2x elsewhere."""
    B, T, C = cfg[0], cfg[1], cfg[2]
    inp = _inputs3(*cfg, seed=99)
    h = 0.02
    gen = torch.Generator(device="cuda").manual_seed(7)
    upstream = torch.randn(4, B, T * C, device="cuda", generator=gen)

    def grads(fn, dtype):
        ins = tuple(t.detach().to(dtype).requires_grad_(True) for t in inp)
        out = fn(*ins, h=float(h))
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), ins)

    g64 = grads(reference3_tv, torch.float64)
    g_torch = grads(reference3_tv, torch.float32)
    g_kernel = grads(kplanes_tilted_tv_fuse3, torch.float32)

    names = ("pts", "rotations", "plane0", "plane1", "plane2")
    for name, gt, gk, gd in zip(names, g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        assert err_kernel <= _grad_bar(name, err_torch, gd), (
            f"tv grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )


@requires_cuda
def test_tv_fuse3_border_points():
    """Points at ±1 and beyond, including taps crossing the border."""
    T, C = 2, 4
    gen = torch.Generator(device="cuda").manual_seed(13)
    rotations = torch.stack(
        [torch.eye(3, device="cuda"), torch.eye(3, device="cuda").roll(1, 0)]
    ).float()
    planes = [
        torch.empty((T, C, h, w), device="cuda").uniform_(0.1, 0.5, generator=gen)
        for (h, w) in ((11, 7), (7, 13), (13, 11))
    ]
    base = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.98, 0.0, 0.0],  # tap 1 (x+h) crosses the border at h=0.05
            [0.0, 0.98, 0.0],  # tap 2 (y+h) crosses
            [0.0, 0.0, 0.98],  # tap 3 (z+h) crosses
            [2.0, -3.0, 0.5],  # well outside
        ],
        device="cuda",
    )
    h = 0.05
    pts = base.clone().requires_grad_(True)
    expected = reference3_tv(pts, rotations, *planes, h=h)
    actual = kplanes_tilted_tv_fuse3(pts, rotations, *planes, h=h)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    gref = torch.autograd.grad(expected.sum(), pts, retain_graph=True)[0]
    gact = torch.autograd.grad(actual.sum(), pts, retain_graph=True)[0]
    assert torch.isfinite(gact).all(), "tv grad_pts contains non-finite values"
    torch.testing.assert_close(gact, gref, rtol=1e-4, atol=1e-6)


@requires_cuda
def test_tv_fuse3_non_contiguous_inputs():
    cfg = (128, 2, 4, (9, 11), (11, 5), (5, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg)
    pts_nc = pts.t().contiguous().t()
    planes_nc = [p.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2) for p in (p0, p1, p2)]
    torch.testing.assert_close(
        kplanes_tilted_tv_fuse3(pts_nc, rotations, *planes_nc, h=0.04),
        reference3_tv(pts, rotations, p0, p1, p2, h=0.04),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_cuda
@pytest.mark.parametrize("h_val", [2.0, 1e-4])
def test_tv_fuse3_edge_case_h(h_val):
    cfg = (256, 2, 8, (17, 9), (9, 21), (21, 17))
    tensors = _inputs3(*cfg, seed=55, requires_grad=True)
    out = kplanes_tilted_tv_fuse3(*tensors, h=h_val)
    assert torch.isfinite(out).all(), f"output has non-finite values for h={h_val}"
    out.sum().backward()
    names = ("pts", "rotations", "plane0", "plane1", "plane2")
    for name, t in zip(names, tensors):
        assert torch.isfinite(t.grad).all(), f"grad_{name} has non-finite values for h={h_val}"


@requires_cuda
def test_tv_fuse3_compile_fullgraph():
    cfg = (512, 2, 4, (17, 9), (9, 21), (21, 17))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg, requires_grad=True)

    @torch.compile(fullgraph=True)
    def f(pt, r, a, b, c):
        return kplanes_tilted_tv_fuse3(pt, r, a, b, c, h=0.01).square().mean()

    f(pts, rotations, p0, p1, p2).backward()
    for t in (pts, rotations, p0, p1, p2):
        assert t.grad is not None


@requires_cuda
def test_tv_fuse3_input_validation():
    cfg = (16, 2, 4, (9, 7), (7, 11), (11, 9))
    pts, rotations, p0, p1, p2 = _inputs3(*cfg)
    with pytest.raises(ValueError, match=r"pts \[B, 3\]"):
        kplanes_tilted_tv_fuse3(pts[:, :2], rotations, p0, p1, p2, h=0.01)
    with pytest.raises(ValueError, match=r"rotations \[T, 3, 3\]"):
        kplanes_tilted_tv_fuse3(pts, rotations[:, :2, :], p0, p1, p2, h=0.01)
    with pytest.raises(ValueError, match=r"plane2 \[T, C, H, W\]"):
        kplanes_tilted_tv_fuse3(pts, rotations, p0, p1, p2[:1], h=0.01)
    with pytest.raises(TypeError, match="fp32-only"):
        kplanes_tilted_tv_fuse3(pts.double(), rotations, p0, p1, p2, h=0.01)
