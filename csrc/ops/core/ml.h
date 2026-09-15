#pragma once
/* ── csrc/ops/core/ml.h ──────────────────────────────────────────────────
 * Host-side launcher declarations for the core.ml CUDA kernels (K-Planes /
 * tensor-decomposition models). All launchers are asynchronous on
 * `stream`; error checking is launch-time only (see CUDA_CHECK_KERNEL in
 * common.cuh).
 */

#include <cuda_runtime.h>

/* Fused TILTED K-Planes feature interpolation (one multiscale level):
 * rotate each point by T matrices, bilinearly sample the 3 planes per
 * rotation (grid_sample align_corners=True / border semantics), Hadamard-
 * multiply, write features. pts [B,3]; R [T,3,3]; grid [3T,H,W,C]
 * CHANNELS-LAST with plane index t*3 + {XY, ZX, YZ}; out [B, T*C] fully
 * written. */
void kplanes_tilted_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float       *d_out,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
);

/* Backward of the fused interpolation. gout [B, T*C]; ggrid [3T,H,W,C]
 * (channels-last, like grid), gR [T,3,3] and gpts [B,3] are accumulated
 * into (caller pre-zeroes all three). Coordinate gradients are zeroed
 * where the border clip engaged, matching torch's grid_sampler. */
void kplanes_tilted_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float       *d_ggrid,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
);

/* TV-specialized fused TILTED K-Planes forward.
 * Evaluates the base computation at 4 tap locations per point:
 *   tap 0: R·x          (base point)
 *   tap 1: R·x + h·R[:,0]  (x + h·ex after rotation)
 *   tap 2: R·x + h·R[:,1]  (x + h·ey after rotation)
 *   tap 3: R·x + h·R[:,2]  (x + h·ez after rotation)
 * pts [B,3]; R [T,3,3]; grid [3T,H,W,C] channels-last;
 * out [4, B, T*C] — tap dim is outermost so out[0] is bit-identical
 * in layout to kplanes_tilted_fuse output. */
void kplanes_tilted_tv_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float       *d_out,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
);

/* Backward of the TV fused interpolation.
 * gout [4, B, T*C]; ggrid [3T,H,W,C], gR [T,3,3], gpts [B,3] accumulated
 * (caller pre-zeroes). h must equal the value used in the forward. */
void kplanes_tilted_tv_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float       *d_ggrid,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
);

/* ── per-plane-type resolution variants (*_fuse3) ─────────────────────────
 * Same computation and semantics as the four launchers above, but each
 * plane TYPE gets its own tensor and its own resolution instead of one
 * [3T, H, W, C] tensor shared by all three:
 *   d_plane0 [T, H0, W0, C]  plane type 0 — coords (comp 0 → W, comp 1 → H)
 *   d_plane1 [T, H1, W1, C]  plane type 1 — coords (comp 2 → W, comp 0 → H)
 *   d_plane2 [T, H2, W2, C]  plane type 2 — coords (comp 1 → W, comp 2 → H)
 * (channels-last, like the single-tensor grid; comp k is component k of the
 * rotated point, matching quantem's idx = [[0,1],[2,0],[1,2]]). C is shared.
 * With H0 = H1 = H2 and W0 = W1 = W2 these produce exactly the same numbers
 * as the single-tensor entry points on the equivalent planes. */
void kplanes_tilted_fuse3_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    float       *d_out,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    cudaStream_t stream
);

/* Backward of kplanes_tilted_fuse3. gout [B, T*C]; gplane0/1/2 (channels-
 * last, like the planes), gR [T,3,3] and gpts [B,3] are accumulated into
 * (caller pre-zeroes all of them). */
void kplanes_tilted_fuse3_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    const float *d_gout,
    float       *d_gplane0,
    float       *d_gplane1,
    float       *d_gplane2,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    cudaStream_t stream
);

/* TV-specialized forward with per-plane-type resolutions (4 taps, exactly
 * as kplanes_tilted_tv_fuse_cuda). out [4, B, T*C] — tap dim outermost. */
void kplanes_tilted_tv_fuse3_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    float       *d_out,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    cudaStream_t stream
);

/* Backward of kplanes_tilted_tv_fuse3. gout [4, B, T*C]; gplane0/1/2, gR
 * and gpts accumulated (caller pre-zeroes). h must equal the forward's. */
void kplanes_tilted_tv_fuse3_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    const float *d_gout,
    float       *d_gplane0,
    float       *d_gplane1,
    float       *d_gplane2,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    cudaStream_t stream
);
