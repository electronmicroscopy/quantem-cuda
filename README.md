# quantem-cuda

CUDA-accelerated kernels for the [`quantem`](https://github.com/electronmicroscopy/quantem)
electron-microscopy toolkit: hand-written CUDA C++ kernels (with analytic gradients) behind
a torch-native Python API, for the hot spots of tomography, ptychography, and related
reconstruction workflows.

`quantem` declares this package as its optional `cuda` extra and transparently dispatches
to these kernels when they are installed; the package is also fully usable standalone —
every public function takes and returns `torch.Tensor`.

## Design

- **Torch-free compiled core.** The extension (`quantem.cuda._core`) links only against the
  shared CUDA runtime, never libtorch. Tensors cross the boundary as raw device pointers,
  shapes, and the current CUDA stream. One compiled wheel therefore works across PyTorch
  releases — at import time `libcudart` resolves to the copy PyTorch already loaded.
- **Torch-native API layer.** Each kernel is registered via `torch.library.custom_op` with
  a hand-written backward (`register_autograd`) and a fake-tensor implementation, so the
  ops compose with autograd and `torch.compile` without graph breaks. This is the
  gsplat-style custom-gradient pattern, packaged so the main `quantem` repo never has to
  carry compiled code.
- **CUDA-only.** There is no CPU fallback; pure-torch reference implementations live in the
  consuming code (and in `tests/`, where they verify the kernels).
- **Submodules mirror quantem.** Kernels are grouped by the `quantem` module they
  accelerate — `quantem.cuda.tomography`, `quantem.cuda.diffractive_imaging`,
  `quantem.cuda.imaging`, `quantem.cuda.diffraction`, `quantem.cuda.spectroscopy` — with
  `quantem.cuda.core` holding kernels shared across modules: `core` itself for
  regularizers / volume ops, `core.ml` for the ML model kernels mirroring
  `quantem.core.ml` (K-Planes / tensor decompositions). The same split runs through
  `csrc/` (`ops/<module>.h`, `cuda/<module>/*.cu`, `bindings/<module>.cpp`), `tests/`,
  and `benchmarks/`. All submodules exist as scaffolds; a kernel goes in the submodule
  named after the quantem module that owns the code it accelerates.

## Installation

Requires a CUDA 13.x toolkit at build time (PyTorch cu13 builds, `torch>=2.9`, at runtime).
No system or conda CUDA install is needed: with uv, the pinned pip CUDA 13.0 toolchain is
injected into the isolated build env automatically via `[tool.uv.extra-build-dependencies]`
in `pyproject.toml`. With plain pip, install the toolkit wheels first:

```bash
pip install 'nvidia-cuda-nvcc==13.0.*' 'nvidia-cuda-cccl==13.0.*' \
            'nvidia-cuda-runtime==13.0.*' 'nvidia-cuda-crt==13.0.*' 'nvidia-nvvm==13.0.*'
pip install quantem-cuda            # or: pip install quantem[cuda]
```

Keep all toolkit components pinned to the same 13.0.x — the wheels don't pin their own
transitive deps, and a mixed set (e.g. a newer `nvidia-nvvm` with an older `ptxas`) fails
mid-compile. Downstream projects building quantem-cuda from source with uv need the same
`extra-build-dependencies` stanza in their own `pyproject.toml` (uv applies the consumer's
settings, not this package's).

The build auto-detects nvcc from (in order) pip-installed toolkit wheels, an active conda
env, or the system. Compiled architectures default to `75;80;86;90;100;120`; override with
`--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="80;86"`.

## Kernels

### `quantem.cuda.core` — shared

| Function | Description |
| --- | --- |
| `tv_loss_iso_3d(volume, eps)` | Isotropic 3-D total-variation loss (corner-restricted, `sqrt(dd²+dh²+dw²+eps)`), mean-reduced; fused forward + analytic gradient. |
| `tv_loss_sq_3d(volume)` | Squared-anisotropic 3-D TV sum, exactly matching `quantem`'s `tv_vol` regularizer; fused forward + analytic gradient. |
| `tv_loss_l1_3d(volume)` | L1-anisotropic TV: per-axis raw `Σ\|forward diff\|` sums, returned as a shape-`(3,)` tensor so per-axis weights/normalization stay with the caller — composes to exactly `quantem`'s ptychography `_calc_tv_loss` functional; fused forward + analytic gradient (`sign(0) = 0`). |

All accept `[D, H, W]` or `[..., D, H, W]` fp32 CUDA tensors (leading channel/batch dims
are flattened) and return a differentiable 0-dim tensor (`(3,)` for the per-axis L1 sums):

```python
import torch
from quantem.cuda.core import tv_loss_iso_3d

volume = torch.rand(256, 256, 256, device="cuda", requires_grad=True)
loss = data_fidelity + 1e-3 * tv_loss_iso_3d(volume)
loss.backward()
```

### `quantem.cuda.core.ml` — ML model kernels (mirrors `quantem.core.ml`)

| Function | Description |
| --- | --- |
| `kplanes_tilted_fuse(pts, rotations, plane)` | Fused TILTED K-Planes feature interpolation (one multiscale level): rotate → bilinear-sample 3 planes per rotation → Hadamard product, with analytic gradients w.r.t. points, rotations, and plane grids. Exactly matches `quantem`'s `interpolate_ms_features_tilted` per level. |
| `kplanes_tilted_tv_fuse(pts, rotations, plane, h)` | The same computation at 4 tap locations per point (base plus one finite-difference step along each world axis), for TV regularization; output `[4, B, T*C]` with `out[0]` matching `kplanes_tilted_fuse`. |
| `kplanes_tilted_fuse3(pts, rotations, plane0, plane1, plane2)` | `kplanes_tilted_fuse` with one tensor `[T, C, H, W]` per plane type, so each plane type can have its own resolution. Same output layout and channel order; bit-identical to `kplanes_tilted_fuse` when the three planes share a shape. |
| `kplanes_tilted_tv_fuse3(pts, rotations, plane0, plane1, plane2, h)` | The 4-tap TV variant of `kplanes_tilted_fuse3`. |

`quantem`'s `KPlanesTILTED` model lives in `quantem.core.ml.models.kplanes` and is shared
across applications (tomography object models today); its kernels live here for the same
reason.

## Development

```bash
git clone https://github.com/electronmicroscopy/quantem-cuda
cd quantem-cuda
uv sync                              # builds the extension into the venv
uv run pytest tests                  # GPU tests auto-skip without a CUDA device
uv run python benchmarks/core/bench_tv_loss.py
```

Kernel sources live in `csrc/cuda/<module>/`, their raw-pointer bindings in
`csrc/bindings/<module>.cpp` (assembled into the single `_core` extension by
`csrc/bindings.cpp`), and the torch registration layer in
`src/quantem/cuda/<module>/_ops.py`. New kernels should follow the same pattern: a
launcher declared in `csrc/ops/<module>.h`, a `custom_op` pair in the submodule's
`_ops.py`, and a pure-torch reference implementation in `tests/<module>/`. A kernel goes
in the submodule named after the `quantem` module it accelerates; kernels useful across
modules go in `core`.

## AI Disclosure

The bulk of the work in this repo is generated by Anthropic's Claude models referencing the original implementations in `quantem`.
