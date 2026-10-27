"""torch-native registration of the core.ml kernels (K-Planes models).

Same conventions as the package's other ``_ops.py`` layers: each kernel
pair is a ``torch.library.custom_op`` (forward) plus a companion backward
op, wired together with ``register_autograd`` and given fake-tensor
implementations so the ops compose with autograd and ``torch.compile``
without graph breaks. The compiled module (``quantem.cuda._core``) is
torch-free; everything torch-facing happens here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from quantem.cuda import _core

# ── fused TILTED K-Planes interpolation ───────────────────────────────────


def _plane_grad_like(plane: Tensor) -> Tensor:
    """Fake-tensor counterpart of a plane gradient: an NCHW view of a contiguous (T, H, W, C)
    buffer, i.e. the strides the real backward ops return (= channels-last ``plane``)."""
    t, c, h, w = plane.shape
    return plane.new_empty((t, h, w, c)).permute(0, 3, 1, 2)


def _channels_last(plane: Tensor) -> Tensor:
    """(3T, C, H, W) → contiguous (3T, H, W, C) for the kernels (contiguous per-cell
    channel reads). A free view when the plane is stored channels-last; otherwise a full copy of
    the plane in EVERY forward and backward call, which for large grids (1e9+ parameters) costs
    more than the kernels — store K-Planes grids channels-last
    (``torch.empty(..., memory_format=torch.channels_last)``)."""
    return plane.permute(0, 2, 3, 1).contiguous()


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse", mutates_args=())
def _kplanes_tilted_fuse(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_cuda(
            p.data_ptr(), r.data_ptr(), g.data_ptr(), out.data_ptr(), b, t, c, h, w, stream
        )
    return out


@_kplanes_tilted_fuse.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    # the real op always allocates a fresh fp32 output, whatever pts' dtype
    return pts.new_empty((pts.shape[0], rotations.shape[0] * plane.shape[1]), dtype=torch.float32)


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_bwd", mutates_args=())
def _kplanes_tilted_fuse_bwd(
    pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)  # accumulated across the T threads per point
    grad_r = torch.zeros_like(r)
    grad_plane_cl = torch.zeros_like(g)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            gout.data_ptr(),
            grad_plane_cl.data_ptr(),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            c,
            h,
            w,
            stream,
        )
    # (3T, H, W, C) → (3T, C, H, W) as a VIEW (channels-last strides): no copy, and when the
    # planes are stored channels-last it matches their strides, so AccumulateGrad keeps it as is
    return [grad_pts, grad_r, grad_plane_cl.permute(0, 3, 1, 2)]


@_kplanes_tilted_fuse_bwd.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor) -> list[Tensor]:
    # new_empty (not empty_like): the real op returns pts / rotation grads CONTIGUOUS
    # even for non-contiguous inputs (it zeros_like the .contiguous() copies), and plane
    # grads channels-last (NCHW views of the kernel's (T, H, W, C) buffers), whatever the
    # input strides.
    return [
        pts.new_empty(pts.shape),
        rotations.new_empty(rotations.shape),
        _plane_grad_like(plane),
    ]


def _kpt_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane = inputs
    ctx.save_for_backward(pts, rotations, plane)


def _kpt_backward(ctx, grad_out):
    pts, rotations, plane = ctx.saved_tensors
    grad_pts, grad_r, grad_plane = _kplanes_tilted_fuse_bwd(pts, rotations, plane, grad_out)
    return grad_pts, grad_r, grad_plane


_kplanes_tilted_fuse.register_autograd(_kpt_backward, setup_context=_kpt_setup_context)


# ── TV-specialized fused TILTED K-Planes interpolation ───────────────────
#
# kplanes_tilted_tv_fuse evaluates kplanes_tilted_fuse at 4 tap locations:
#   tap 0: R·x              (identical in layout to kplanes_tilted_fuse output)
#   tap 1: R·x + h·R[:,0]  (world x finite-difference)
#   tap 2: R·x + h·R[:,1]  (world y finite-difference)
#   tap 3: R·x + h·R[:,2]  (world z finite-difference)
# Output: [4, B, T*C] — tap dimension outermost.
# h is a fixed hyperparameter; no gradient is tracked for it.


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse", mutates_args=())
def _kplanes_tilted_tv_fuse(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, hw, w = plane.shape[1], plane.shape[2], plane.shape[3]
    h_dim = hw  # H dimension
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((4, b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            out.data_ptr(),
            b,
            t,
            c,
            h_dim,
            w,
            float(h),
            stream,
        )
    return out


@_kplanes_tilted_tv_fuse.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    # the real op always allocates a fresh fp32 output, whatever pts' dtype
    return pts.new_empty(
        (4, pts.shape[0], rotations.shape[0] * plane.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse_bwd", mutates_args=())
def _kplanes_tilted_tv_fuse_bwd(
    pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor, h: float
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h_dim, w = plane.shape[1], plane.shape[2], plane.shape[3]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)
    grad_r = torch.zeros_like(r)
    grad_plane_cl = torch.zeros_like(g)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            gout.data_ptr(),
            grad_plane_cl.data_ptr(),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            c,
            h_dim,
            w,
            float(h),
            stream,
        )
    # (3T, H, W, C) → (3T, C, H, W) as a VIEW (channels-last strides): no copy, and when the
    # planes are stored channels-last it matches their strides, so AccumulateGrad keeps it as is
    return [grad_pts, grad_r, grad_plane_cl.permute(0, 3, 1, 2)]


@_kplanes_tilted_tv_fuse_bwd.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor, h: float) -> list[Tensor]:
    # matches the real op's grad layouts (see the fwd bwd fake)
    return [
        pts.new_empty(pts.shape),
        rotations.new_empty(rotations.shape),
        _plane_grad_like(plane),
    ]


def _kpt_tv_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane, h = inputs
    ctx.save_for_backward(pts, rotations, plane)
    ctx.h = h


def _kpt_tv_backward(ctx, grad_out):
    pts, rotations, plane = ctx.saved_tensors
    grad_pts, grad_r, grad_plane = _kplanes_tilted_tv_fuse_bwd(
        pts, rotations, plane, grad_out, ctx.h
    )
    # h has no gradient
    return grad_pts, grad_r, grad_plane, None


_kplanes_tilted_tv_fuse.register_autograd(_kpt_tv_backward, setup_context=_kpt_tv_setup_context)


# ── per-plane-type resolution variants (fuse3) ────────────────────────────
#
# Same computation as the two ops above, but each plane TYPE gets its own
# tensor and its own (H, W) instead of one [3T, C, H, W] tensor shared by all
# three. Plane type p samples the coordinate pair of quantem's
# idx = [[0, 1], [2, 0], [1, 2]] (first entry → grid_sample x / the W axis,
# second → y / the H axis) of the rotated point:
#   plane0 [T, C, H0, W0]: (comp 0 → W, comp 1 → H)
#   plane1 [T, C, H1, W1]: (comp 2 → W, comp 0 → H)
#   plane2 [T, C, H2, W2]: (comp 1 → W, comp 2 → H)
# Output layout and channel order are identical to kplanes_tilted_fuse.


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse3", mutates_args=())
def _kplanes_tilted_fuse3(
    pts: Tensor, rotations: Tensor, plane0: Tensor, plane1: Tensor, plane2: Tensor
) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c = plane0.shape[1]
    g = [_channels_last(pl) for pl in (plane0, plane1, plane2)]
    hw = [(int(pl.shape[2]), int(pl.shape[3])) for pl in (plane0, plane1, plane2)]
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse3_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g[0].data_ptr(),
            g[1].data_ptr(),
            g[2].data_ptr(),
            out.data_ptr(),
            b,
            t,
            c,
            hw[0][0],
            hw[0][1],
            hw[1][0],
            hw[1][1],
            hw[2][0],
            hw[2][1],
            stream,
        )
    return out


@_kplanes_tilted_fuse3.register_fake
def _(pts: Tensor, rotations: Tensor, plane0: Tensor, plane1: Tensor, plane2: Tensor) -> Tensor:
    # the real op always allocates a fresh fp32 output, whatever pts' dtype
    return pts.new_empty((pts.shape[0], rotations.shape[0] * plane0.shape[1]), dtype=torch.float32)


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse3_bwd", mutates_args=())
def _kplanes_tilted_fuse3_bwd(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c = plane0.shape[1]
    g = [_channels_last(pl) for pl in (plane0, plane1, plane2)]
    hw = [(int(pl.shape[2]), int(pl.shape[3])) for pl in (plane0, plane1, plane2)]
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)  # accumulated across the T threads per point
    grad_r = torch.zeros_like(r)
    grad_planes_cl = [torch.zeros_like(gi) for gi in g]
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse3_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g[0].data_ptr(),
            g[1].data_ptr(),
            g[2].data_ptr(),
            gout.data_ptr(),
            grad_planes_cl[0].data_ptr(),
            grad_planes_cl[1].data_ptr(),
            grad_planes_cl[2].data_ptr(),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            c,
            hw[0][0],
            hw[0][1],
            hw[1][0],
            hw[1][1],
            hw[2][0],
            hw[2][1],
            stream,
        )
    # (T, H, W, C) → (T, C, H, W) as a VIEW (channels-last strides; see the single-tensor op)
    return [grad_pts, grad_r] + [gp.permute(0, 3, 1, 2) for gp in grad_planes_cl]


@_kplanes_tilted_fuse3_bwd.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
) -> list[Tensor]:
    # new_empty (not empty_like): the real op returns pts / rotation grads CONTIGUOUS
    # even for non-contiguous inputs (it zeros_like the .contiguous() copies), and plane
    # grads channels-last (NCHW views of the kernel's (T, H, W, C) buffers), whatever the
    # input strides.
    return [
        pts.new_empty(pts.shape),
        rotations.new_empty(rotations.shape),
        _plane_grad_like(plane0),
        _plane_grad_like(plane1),
        _plane_grad_like(plane2),
    ]


def _kpt3_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane0, plane1, plane2 = inputs
    ctx.save_for_backward(pts, rotations, plane0, plane1, plane2)


def _kpt3_backward(ctx, grad_out):
    pts, rotations, plane0, plane1, plane2 = ctx.saved_tensors
    grads = _kplanes_tilted_fuse3_bwd(pts, rotations, plane0, plane1, plane2, grad_out)
    return grads[0], grads[1], grads[2], grads[3], grads[4]


_kplanes_tilted_fuse3.register_autograd(_kpt3_backward, setup_context=_kpt3_setup_context)


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse3", mutates_args=())
def _kplanes_tilted_tv_fuse3(
    pts: Tensor, rotations: Tensor, plane0: Tensor, plane1: Tensor, plane2: Tensor, h: float
) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c = plane0.shape[1]
    g = [_channels_last(pl) for pl in (plane0, plane1, plane2)]
    hw = [(int(pl.shape[2]), int(pl.shape[3])) for pl in (plane0, plane1, plane2)]
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((4, b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse3_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g[0].data_ptr(),
            g[1].data_ptr(),
            g[2].data_ptr(),
            out.data_ptr(),
            b,
            t,
            c,
            hw[0][0],
            hw[0][1],
            hw[1][0],
            hw[1][1],
            hw[2][0],
            hw[2][1],
            float(h),
            stream,
        )
    return out


@_kplanes_tilted_tv_fuse3.register_fake
def _(
    pts: Tensor, rotations: Tensor, plane0: Tensor, plane1: Tensor, plane2: Tensor, h: float
) -> Tensor:
    # the real op always allocates a fresh fp32 output, whatever pts' dtype
    return pts.new_empty(
        (4, pts.shape[0], rotations.shape[0] * plane0.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse3_bwd", mutates_args=())
def _kplanes_tilted_tv_fuse3_bwd(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    h: float,
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c = plane0.shape[1]
    g = [_channels_last(pl) for pl in (plane0, plane1, plane2)]
    hw = [(int(pl.shape[2]), int(pl.shape[3])) for pl in (plane0, plane1, plane2)]
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)
    grad_r = torch.zeros_like(r)
    grad_planes_cl = [torch.zeros_like(gi) for gi in g]
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse3_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g[0].data_ptr(),
            g[1].data_ptr(),
            g[2].data_ptr(),
            gout.data_ptr(),
            grad_planes_cl[0].data_ptr(),
            grad_planes_cl[1].data_ptr(),
            grad_planes_cl[2].data_ptr(),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            c,
            hw[0][0],
            hw[0][1],
            hw[1][0],
            hw[1][1],
            hw[2][0],
            hw[2][1],
            float(h),
            stream,
        )
    # (T, H, W, C) → (T, C, H, W) as a VIEW (channels-last strides; see the single-tensor op)
    return [grad_pts, grad_r] + [gp.permute(0, 3, 1, 2) for gp in grad_planes_cl]


@_kplanes_tilted_tv_fuse3_bwd.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    h: float,
) -> list[Tensor]:
    # matches the real op's grad layouts (see the fwd bwd fake)
    return [
        pts.new_empty(pts.shape),
        rotations.new_empty(rotations.shape),
        _plane_grad_like(plane0),
        _plane_grad_like(plane1),
        _plane_grad_like(plane2),
    ]


def _kpt3_tv_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane0, plane1, plane2, h = inputs
    ctx.save_for_backward(pts, rotations, plane0, plane1, plane2)
    ctx.h = h


def _kpt3_tv_backward(ctx, grad_out):
    pts, rotations, plane0, plane1, plane2 = ctx.saved_tensors
    grads = _kplanes_tilted_tv_fuse3_bwd(pts, rotations, plane0, plane1, plane2, grad_out, ctx.h)
    # h has no gradient
    return grads[0], grads[1], grads[2], grads[3], grads[4], None


_kplanes_tilted_tv_fuse3.register_autograd(_kpt3_tv_backward, setup_context=_kpt3_tv_setup_context)


# ── public wrappers ───────────────────────────────────────────────────────


def kplanes_tilted_fuse(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    """Fused TILTED K-Planes feature interpolation for one multiscale level.

    For each point ``p`` and each rotation ``R_t``, bilinearly samples the
    three planes of rotation ``t`` at the (XY, ZX, YZ) pairs of ``R_t @ p``
    and Hadamard-multiplies them — equivalent to (but much faster than)::

        rotated = torch.einsum("tij,bj->tbi", rotations, pts)
        coords = rotated[..., [[0, 1], [2, 0], [1, 2]]]      # (T, B, 3, 2)
        sampled = F.grid_sample(plane, coords.permute(0, 2, 1, 3).reshape(3*T, B, 1, 2),
                                align_corners=True, mode="bilinear",
                                padding_mode="border")
        out = sampled.squeeze(-1).view(T, 3, C, B).prod(1).permute(2, 0, 1).reshape(B, T*C)

    Differentiable w.r.t. all three inputs (analytic CUDA backward, with
    torch's border-clip zero-gradient convention for coordinates).

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane:     fp32 CUDA tensor ``[3*T, C, H, W]`` (plane ``t*3 + p``).

    Returns:
        fp32 tensor ``[B, T*C]`` (``out[b, t*C + c]``), differentiable.
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"kplanes_tilted_fuse expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"kplanes_tilted_fuse expects rotations [T, 3, 3], got {tuple(rotations.shape)}"
        )
    if plane.ndim != 4 or plane.shape[0] != 3 * rotations.shape[0]:
        raise ValueError(
            "kplanes_tilted_fuse expects plane [3*T, C, H, W] with T = "
            f"rotations.shape[0]; got plane {tuple(plane.shape)} for T={rotations.shape[0]}"
        )
    for name, t in (("pts", pts), ("rotations", rotations), ("plane", plane)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_tilted_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(f"kplanes_tilted_fuse requires CUDA tensors ({name} on {t.device}).")
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_tilted_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_tilted_fuse(pts, rotations, plane)


def kplanes_tilted_tv_fuse(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    """TV-specialized fused TILTED K-Planes feature interpolation for one multiscale level.

    Evaluates ``kplanes_tilted_fuse`` at 4 tap locations per point to support
    total-variation regularization via finite differences. For each point ``p``
    and rotation ``R_t``:

        tap 0: feature(R_t · x)              — base, identical layout to
                                               kplanes_tilted_fuse output
        tap 1: feature(R_t · x + h·R_t[:,0]) — world-x finite difference
        tap 2: feature(R_t · x + h·R_t[:,1]) — world-y finite difference
        tap 3: feature(R_t · x + h·R_t[:,2]) — world-z finite difference

    Key identities used by the kernel:
        R·(x + h·e_i) = R·x + h·R[:,i]
    so the base rotation is computed once per (point, rotation), and the three
    column offsets ``h·R[:,i]`` are per-rotation constants shared across all
    points. This saves 3 matrix-vector multiplications per point.

    Output layout: ``[4, B, T*C]`` — tap dimension outermost. ``out[0]`` is
    bit-for-bit compatible in layout with ``kplanes_tilted_fuse(pts, ...)``,
    enabling ``out.unbind(0)`` to yield four ``[B, T*C]`` tensors.

    ``h`` is treated as a fixed hyperparameter: no gradient is returned for it.
    Differentiable w.r.t. ``pts``, ``rotations``, and ``plane`` (analytic CUDA
    backward, with torch's border-clip zero-gradient convention).

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane:     fp32 CUDA tensor ``[3*T, C, H, W]`` (plane ``t*3 + p``).
        h:         finite-difference step in world ``[-1, 1]`` units.

    Returns:
        fp32 tensor ``[4, B, T*C]``, differentiable w.r.t. pts/rotations/plane.
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"kplanes_tilted_tv_fuse expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"kplanes_tilted_tv_fuse expects rotations [T, 3, 3], got {tuple(rotations.shape)}"
        )
    if plane.ndim != 4 or plane.shape[0] != 3 * rotations.shape[0]:
        raise ValueError(
            "kplanes_tilted_tv_fuse expects plane [3*T, C, H, W] with T = "
            f"rotations.shape[0]; got plane {tuple(plane.shape)} for "
            f"T={rotations.shape[0]}"
        )
    for name, t in (("pts", pts), ("rotations", rotations), ("plane", plane)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_tilted_tv_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(
                f"kplanes_tilted_tv_fuse requires CUDA tensors ({name} on {t.device})."
            )
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_tilted_tv_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_tilted_tv_fuse(pts, rotations, plane, float(h))


def _validate3(
    fn: str, pts: Tensor, rotations: Tensor, planes: tuple[Tensor, Tensor, Tensor]
) -> None:
    """Shared validation for the per-plane-type (``fuse3``) entry points."""
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"{fn} expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"{fn} expects rotations [T, 3, 3], got {tuple(rotations.shape)}")
    t = rotations.shape[0]
    for i, plane in enumerate(planes):
        name = f"plane{i}"
        if plane.ndim != 4 or plane.shape[0] != t:
            raise ValueError(
                f"{fn} expects {name} [T, C, H, W] with T = rotations.shape[0]; "
                f"got {name} {tuple(plane.shape)} for T={t}"
            )
        if plane.shape[1] != planes[0].shape[1]:
            raise ValueError(
                f"{fn} expects the same channel count C in all three planes; got "
                f"plane0 C={planes[0].shape[1]} and {name} C={plane.shape[1]}"
            )
        if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
            raise ValueError(
                f"{fn} uses 32-bit per-plane offsets; C*H*W must be < 2^31, "
                f"got {name} {tuple(plane.shape)}"
            )
    named = (("pts", pts), ("rotations", rotations)) + tuple(
        (f"plane{i}", p) for i, p in enumerate(planes)
    )
    for name, tensor in named:
        if tensor.dtype != torch.float32:
            raise TypeError(f"{fn} is fp32-only ({name} is {tensor.dtype}).")
        if not tensor.is_cuda:
            raise ValueError(f"{fn} requires CUDA tensors ({name} on {tensor.device}).")


def kplanes_tilted_fuse3(
    pts: Tensor, rotations: Tensor, plane0: Tensor, plane1: Tensor, plane2: Tensor
) -> Tensor:
    """:func:`kplanes_tilted_fuse` with a separate resolution per plane type.

    Identical computation, output layout and channel order as
    :func:`kplanes_tilted_fuse`, except that the three plane types live in
    three tensors of their own and may have different ``(H, W)``. Plane type
    ``p`` samples the coordinate pair of quantem's ``idx = [[0, 1], [2, 0],
    [1, 2]]`` (first entry → ``grid_sample``'s x / the ``W`` axis, second →
    y / the ``H`` axis) of the rotated point ``R_t @ p``::

        plane0: (component 0 → W, component 1 → H)
        plane1: (component 2 → W, component 0 → H)
        plane2: (component 1 → W, component 2 → H)

    Equivalent to::

        rotated = torch.einsum("tij,bj->tbi", rotations, pts)       # (T, B, 3)
        idx = [[0, 1], [2, 0], [1, 2]]
        samples = [
            F.grid_sample(plane, rotated[..., ij].unsqueeze(1),
                          align_corners=True, mode="bilinear",
                          padding_mode="border").squeeze(2)         # (T, C, B)
            for plane, ij in zip((plane0, plane1, plane2), idx)
        ]
        out = (samples[0] * samples[1] * samples[2]).permute(2, 0, 1).reshape(B, T * C)

    With all three planes the same shape this returns bit-for-bit the same
    values as ``kplanes_tilted_fuse`` on the equivalent ``[3T, C, H, W]``
    tensor (``plane[t*3 + p] == plane<p>[t]``).

    Differentiable w.r.t. all five inputs (analytic CUDA backward, with
    torch's border-clip zero-gradient convention for coordinates).

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane0:    fp32 CUDA tensor ``[T, C, H0, W0]``.
        plane1:    fp32 CUDA tensor ``[T, C, H1, W1]``.
        plane2:    fp32 CUDA tensor ``[T, C, H2, W2]``.

    Returns:
        fp32 tensor ``[B, T*C]`` (``out[b, t*C + c]``), differentiable.
    """
    _validate3("kplanes_tilted_fuse3", pts, rotations, (plane0, plane1, plane2))
    return _kplanes_tilted_fuse3(pts, rotations, plane0, plane1, plane2)


def kplanes_tilted_tv_fuse3(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    h: float,
) -> Tensor:
    """:func:`kplanes_tilted_tv_fuse` with a separate resolution per plane type.

    Same 4-tap contract as :func:`kplanes_tilted_tv_fuse` — ``tap 0`` at
    ``R_t · x`` and taps 1-3 at ``R_t · x + h·R_t[:, i]`` for the world x, y
    and z directions — with the three plane types in three tensors of their
    own, each with its own ``(H, W)``; see :func:`kplanes_tilted_fuse3` for
    the coordinate pair each plane type uses.

    Output layout: ``[4, B, T*C]`` — tap dimension outermost, ``out[0]``
    identical in layout to ``kplanes_tilted_fuse3(pts, ...)``.

    ``h`` is a fixed hyperparameter: no gradient is returned for it.
    Differentiable w.r.t. ``pts``, ``rotations`` and the three planes.

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane0:    fp32 CUDA tensor ``[T, C, H0, W0]``.
        plane1:    fp32 CUDA tensor ``[T, C, H1, W1]``.
        plane2:    fp32 CUDA tensor ``[T, C, H2, W2]``.
        h:         finite-difference step in world ``[-1, 1]`` units.

    Returns:
        fp32 tensor ``[4, B, T*C]``, differentiable w.r.t. pts/rotations/planes.
    """
    _validate3("kplanes_tilted_tv_fuse3", pts, rotations, (plane0, plane1, plane2))
    return _kplanes_tilted_tv_fuse3(pts, rotations, plane0, plane1, plane2, float(h))


# ── splat tomography: fused 2D Gaussian stamping ─────────────────────────
#
# splat_stamp renders T tilt images of N isotropic Gaussians from their
# projected pixel centres (the pose logic, u = pi(R p + s), stays in torch
# in quantem.core.ml.splat.renderer). Pixel units throughout: amps are the
# pixel-integrated masses (A / sampling²) and sigmas the widths in pixels.
# The backward returns grads to centres, amps and sigmas; the image-shape
# ints and the window radius carry no gradient.


@torch.library.custom_op("quantem_cuda::splat_stamp", mutates_args=())
def _splat_stamp(
    centers: Tensor, amps: Tensor, sigmas: Tensor, height: int, width: int, radius: int
) -> Tensor:
    u = centers.contiguous()
    a = amps.contiguous()
    s = sigmas.contiguous()
    t, n = int(u.shape[0]), int(u.shape[1])
    out = torch.zeros((t, height, width), dtype=torch.float32, device=u.device)
    stream = torch.cuda.current_stream(u.device).cuda_stream
    with torch.cuda.device(u.device):
        _core.splat_stamp_fwd_cuda(
            u.data_ptr(),
            a.data_ptr(),
            s.data_ptr(),
            out.data_ptr(),
            t,
            n,
            int(height),
            int(width),
            int(radius),
            stream,
        )
    return out


@_splat_stamp.register_fake
def _(
    centers: Tensor, amps: Tensor, sigmas: Tensor, height: int, width: int, radius: int
) -> Tensor:
    return centers.new_empty((centers.shape[0], height, width), dtype=torch.float32)


@torch.library.custom_op("quantem_cuda::splat_stamp_bwd", mutates_args=())
def _splat_stamp_bwd(
    centers: Tensor, amps: Tensor, sigmas: Tensor, grad_out: Tensor, radius: int
) -> list[Tensor]:
    u = centers.contiguous()
    a = amps.contiguous()
    s = sigmas.contiguous()
    t, n = int(u.shape[0]), int(u.shape[1])
    gout = grad_out.detach().to(dtype=torch.float32, device=u.device).contiguous()
    h, w = int(gout.shape[1]), int(gout.shape[2])
    grad_u = torch.empty_like(u)  # fully written by the kernel
    ga_part = torch.empty((t, n), dtype=torch.float32, device=u.device)
    gs_part = torch.empty((t, n), dtype=torch.float32, device=u.device)
    stream = torch.cuda.current_stream(u.device).cuda_stream
    with torch.cuda.device(u.device):
        _core.splat_stamp_bwd_cuda(
            u.data_ptr(),
            a.data_ptr(),
            s.data_ptr(),
            gout.data_ptr(),
            grad_u.data_ptr(),
            ga_part.data_ptr(),
            gs_part.data_ptr(),
            t,
            n,
            h,
            w,
            int(radius),
            stream,
        )
    # the per-tilt partials are summed here (deterministic), not with atomics in the kernel
    return [grad_u, ga_part.sum(dim=0), gs_part.sum(dim=0)]


@_splat_stamp_bwd.register_fake
def _(
    centers: Tensor, amps: Tensor, sigmas: Tensor, grad_out: Tensor, radius: int
) -> list[Tensor]:
    return [
        centers.new_empty(centers.shape),
        amps.new_empty(amps.shape),
        sigmas.new_empty(sigmas.shape),
    ]


def _splat_setup_context(ctx, inputs, output) -> None:
    centers, amps, sigmas, height, width, radius = inputs
    ctx.save_for_backward(centers, amps, sigmas)
    ctx.radius = radius


def _splat_backward(ctx, grad_out):
    centers, amps, sigmas = ctx.saved_tensors
    grad_u, grad_a, grad_s = _splat_stamp_bwd(centers, amps, sigmas, grad_out, ctx.radius)
    return grad_u, grad_a, grad_s, None, None, None


_splat_stamp.register_autograd(_splat_backward, setup_context=_splat_setup_context)


def splat_stamp(
    centers: Tensor,
    amps: Tensor,
    sigmas: Tensor,
    image_shape: tuple[int, int],
    radius: int,
) -> Tensor:
    """Stamp ``N`` isotropic 2D Gaussians into ``T`` images (fused forward + backward).

    The image model of ``quantem.core.ml.splat.renderer.render_projections`` in pixel units::

        out[t, i, j] = sum_n amps[n] / (2 pi sigmas[n]^2)
                       * exp(-((i - centers[t, n, 0])^2 + (j - centers[t, n, 1])^2) / (2 sigmas[n]^2))

    over the atoms whose window ``round(centers[t, n]) + {-radius..radius}^2`` (round half to
    even, clipped to the image) contains pixel ``(i, j)``; the Gaussian is evaluated at the
    pixel centres, not integrated. Nothing per (tilt, atom, pixel) is stored, so the memory is
    the output plus the inputs whatever the atom count.

    Parameters
    ----------
    centers : Tensor
        ``(T, N, 2)`` fp32 CUDA, fractional pixel coordinates ordered ``(row, col)``.
    amps : Tensor
        ``(N,)`` fp32 CUDA, pixel-integrated mass per Gaussian (``A / sampling**2``).
    sigmas : Tensor
        ``(N,)`` fp32 CUDA, width in pixels (``sigma / sampling``), positive.
    image_shape : tuple[int, int]
        ``(H, W)``.
    radius : int
        Window half-width in pixels (``>= 0``).

    Returns
    -------
    Tensor
        ``(T, H, W)`` fp32 images; differentiable in ``centers``, ``amps`` and ``sigmas``.
        The backward is deterministic; the forward accumulates with atomics (fp32 round-off).
    """
    fn = "splat_stamp"
    if centers.ndim != 3 or centers.shape[2] != 2:
        raise ValueError(f"{fn} expects centers [T, N, 2], got {tuple(centers.shape)}")
    n = centers.shape[1]
    for name, t in (("amps", amps), ("sigmas", sigmas)):
        if t.ndim != 1 or t.shape[0] != n:
            raise ValueError(f"{fn} expects {name} [N] with N = {n}, got {tuple(t.shape)}")
    for name, t in (("centers", centers), ("amps", amps), ("sigmas", sigmas)):
        if t.dtype != torch.float32:
            raise TypeError(f"{fn} is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(f"{fn} requires CUDA tensors ({name} on {t.device}).")
        if t.device != centers.device:
            raise ValueError(
                f"{fn} requires all tensors on one device ({name} on {t.device}, "
                f"centers on {centers.device})."
            )
    if len(image_shape) != 2:
        raise ValueError(f"{fn} expects image_shape (H, W), got {tuple(image_shape)}")
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0 or h * w >= 2**31:
        raise ValueError(f"{fn}: image shape {(h, w)} must be positive with H * W < 2**31")
    r = int(radius)
    if r < 0:
        raise ValueError(f"{fn}: radius must be >= 0, got {radius}")
    return _splat_stamp(centers, amps, sigmas, h, w, r)
