/* ── csrc/cuda/core/ml/kplanes_tilted.cu ────────────────────────────────────────────
 * Fused TILTED K-Planes feature interpolation (one multiscale level).
 *
 * Replaces the torch chain
 *     einsum(rotate) → gather(plane pairs) → F.grid_sample → prod(3 planes)
 *     → permute/reshape
 * with a single kernel per direction. The torch chain materializes a
 * (3T, C, B) sample tensor in HBM just to Hadamard-multiply three planes;
 * here each point's bilinear taps stay in registers and only the final
 * (B, T·C) features are written.
 *
 * Memory design, in order of importance:
 *  1. CHANNELS-LAST grids (3T, H, W, C): a bilinear cell's C channels are
 *     contiguous, so one (t, plane) access touches a handful of cache
 *     lines instead of 4·C of them (stride H·W) in torch's NCHW layout.
 *     The Python wrapper permutes the (3T, C, H, W) parameter per call —
 *     the grid tensor is tiny next to the per-point traffic.
 *  2. LANE-PER-CHANNEL parallelism: one thread per (point, rotation,
 *     channel), channels fastest. Adjacent lanes read adjacent grid
 *     addresses (coalesced loads) and — in the backward — scatter to
 *     adjacent addresses (coalesced atomics). The per-(point, rotation)
 *     tap setup (~80 flops) is recomputed per lane; that is cheap next to
 *     uncoalesced memory traffic.
 *  3. Backward coordinate grads (dL/d ix, dL/d iy per plane) are summed
 *     over channels with warp-segment shuffles (a segment = the lanes of
 *     one (point, rotation)). For C > 32 each 32-channel chunk gets its
 *     own warp; the rotation/point gradients are linear in the per-chunk
 *     sums, so each warp applies its partial contribution independently.
 *     grad_R goes through a per-block shared staging buffer (one global
 *     atomicAdd per (t, i, j) per block); grad_pts via direct atomicAdd.
 *
 * Branchless taps: with align_corners=True border clamping, a clamped +1
 * corner index implies its bilinear weight is exactly 0 (ix == W-1 ⟺
 * tx == 0), so corners are clamped instead of guarded and the weight kills
 * the contribution. The only points whose finite-difference terms would
 * differ under clamping sit exactly on the border, where the coordinate
 * gradient is masked to 0 anyway (torch's clip_coordinates_set_grad
 * convention). Per-plane offsets are 32-bit; the wrapper enforces the
 * grid size this implies.
 *
 * Semantics match F.grid_sample(align_corners=True, padding_mode="border")
 * exactly, gradients included.
 *
 * Two plane layouts, one set of kernels (see PlaneGeom below):
 *   SINGLE  one tensor  grid [3T, H, W, C], plane index t*3 + p — all three
 *           plane types share one resolution (the original entry points,
 *           *_fuse / *_tv_fuse).
 *   PER-TYPE three tensors grid_p [T, H_p, W_p, C], one per plane type, each
 *           with its own resolution (the *_fuse3 / *_tv_fuse3 entry points).
 *
 * Layouts (all contiguous fp32):
 *   pts  [B, 3]         points in [-1, 1]³ (border-clamped outside)
 *   R    [T, 3, 3]      rotation matrices, row-major
 *   out  [B, T*C]       out[b, t*C + c] = Π_p sample(plane(t,p), c)
 *
 * Plane coordinate pairs (gx → W axis, gy → H axis, matching grid_sample's
 * (x, y) ordering and quantem's idx = [[0,1],[2,0],[1,2]]):
 *   p=0 XY: (gx, gy) = (rx, ry)
 *   p=1 ZX: (gx, gy) = (rz, rx)
 *   p=2 YZ: (gx, gy) = (ry, rz)
 * (rx, ry, rz are components 0, 1, 2 of the rotated point.)
 */

#include "common.cuh"
#include "ops/core/ml.h"

namespace {

/* Per-plane-type geometry, the only thing that differs between the two
 * layouts. One struct serves both, so the kernels below are shared:
 *   SINGLE   [3T, H, W, C]:  base[p] = grid + p·H·W·C, tstride = 3·H·W·C
 *   PER-TYPE [T, H_p, W_p, C]: base[p] = grid_p,       tstride = H_p·W_p·C
 * Both give exactly the addresses the single-tensor kernel used before, so
 * the original entry points keep their numerics bit for bit. */
struct PlaneGeom {
    long tstride[3]; // elements between consecutive rotations of plane type p
    int H[3];
    int W[3];
};

struct Tap {
    int off[4];     // corner offsets into a (H, W, C) plane, +1 sides clamped
    float w[4];     // bilinear weights (nw, ne, sw, se); 0 where clamped
    float tx, ty;   // fractional offsets in [0, 1]
    float gxm, gym; // d(ix)/d(gx) including the border-clip mask
};

__device__ __forceinline__ Tap make_tap(float gx, float gy, int H, int W, int C) {
    Tap t;
    float ix = (gx + 1.f) * 0.5f * (float)(W - 1);
    float iy = (gy + 1.f) * 0.5f * (float)(H - 1);
    // torch clip_coordinates_set_grad: gradient is zero at and beyond the
    // border (in <= 0 or in >= size-1), so strict inequalities here.
    t.gxm = (ix > 0.f && ix < (float)(W - 1)) ? 0.5f * (float)(W - 1) : 0.f;
    t.gym = (iy > 0.f && iy < (float)(H - 1)) ? 0.5f * (float)(H - 1) : 0.f;
    ix = fminf(fmaxf(ix, 0.f), (float)(W - 1));
    iy = fminf(fmaxf(iy, 0.f), (float)(H - 1));
    const int x0 = (int)ix;
    const int y0 = (int)iy;
    t.tx = ix - (float)x0;
    t.ty = iy - (float)y0;
    t.w[0] = (1.f - t.tx) * (1.f - t.ty);
    t.w[1] = t.tx * (1.f - t.ty);
    t.w[2] = (1.f - t.tx) * t.ty;
    t.w[3] = t.tx * t.ty;
    const int x1 = min(x0 + 1, W - 1); // clamped ⟺ tx == 0 ⟺ w[1] = w[3] = 0
    const int y1 = min(y0 + 1, H - 1); // clamped ⟺ ty == 0 ⟺ w[2] = w[3] = 0
    t.off[0] = (y0 * W + x0) * C;
    t.off[1] = (y0 * W + x1) * C;
    t.off[2] = (y1 * W + x0) * C;
    t.off[3] = (y1 * W + x1) * C;
    return t;
}

__device__ __forceinline__ void rotate(
    const float *Rt, float px, float py, float pz, float &rx, float &ry, float &rz
) {
    rx = Rt[0] * px + Rt[1] * py + Rt[2] * pz;
    ry = Rt[3] * px + Rt[4] * py + Rt[5] * pz;
    rz = Rt[6] * px + Rt[7] * py + Rt[8] * pz;
}

__global__ void kplanes_tilted_fwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const float *__restrict__ g0, // plane-type bases, channels-last
    const float *__restrict__ g1,
    const float *__restrict__ g2,
    PlaneGeom geom,
    float *__restrict__ out,
    long B, int T, int C
) {
    extern __shared__ float sR[]; // [T*9]
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) sR[i] = R[i];
    __syncthreads();

    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * (long)(T * C)) return;
    const long b = tid / (T * C);
    const int rem = (int)(tid - b * (T * C));
    const int t = rem / C;
    const int c = rem - t * C;

    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    float rx, ry, rz;
    rotate(sR + t * 9, px, py, pz, rx, ry, rz);

    const float *const gbase[3] = {g0, g1, g2};
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};

    float prod = 1.f;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const Tap tp = make_tap(gxs[p], gys[p], geom.H[p], geom.W[p], C);
        const float *pc = gbase[p] + (long)t * geom.tstride[p] + c;
        prod *= tp.w[0] * pc[tp.off[0]] + tp.w[1] * pc[tp.off[1]] + tp.w[2] * pc[tp.off[2]]
              + tp.w[3] * pc[tp.off[3]];
    }
    out[tid] = prod; // out[b, t*C + c] — tid is exactly that flat index
}

__global__ void kplanes_tilted_bwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const float *__restrict__ g0, // plane-type bases, channels-last
    const float *__restrict__ g1,
    const float *__restrict__ g2,
    PlaneGeom geom,
    const float *__restrict__ gout, // [B, T*C]
    float *__restrict__ gg0,        // grads for the plane bases, pre-zeroed
    float *__restrict__ gg1,
    float *__restrict__ gg2,
    float *__restrict__ gR,         // [T, 3, 3], pre-zeroed
    float *__restrict__ gpts,       // [B, 3], pre-zeroed
    long B, int T, int C
) {
    extern __shared__ float smem[]; // [T*9] R copy | [T*9] gR partials
    float *sR = smem;
    float *sgR = smem + T * 9;
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        sgR[i] = 0.f;
    }
    __syncthreads();

    // A segment = the lanes covering one (point, rotation)'s channels
    // (one 32-channel chunk of them when C > 32). Segments never straddle
    // warps; lanes past the last whole segment idle but join the shuffles.
    const int lane = threadIdx.x & 31;
    const long warp_id = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int CW = C < 32 ? C : 32; // channels per segment
    long group;                     // index over B*T (point, rotation) pairs
    int c, lis;                     // channel, lane-in-segment
    bool active;
    if (C <= 32) {
        const int spw = 32 / CW; // segments per warp
        const int seg = lane / CW;
        lis = lane - seg * CW;
        group = warp_id * spw + seg;
        c = lis;
        active = seg < spw && group < B * (long)T;
    } else {
        const int wpg = (C + 31) >> 5; // warps per group
        group = warp_id / wpg;
        lis = lane;
        c = (int)(warp_id - group * wpg) * 32 + lane;
        active = group < B * (long)T && c < C;
    }
    // Inactive (overhanging) threads still execute the addressing below —
    // clamp the whole group index so b and t both stay in range.
    const long g_safe = group < B * (long)T ? group : 0;
    const long b = g_safe / T;
    const int t = (int)(g_safe - b * T);

    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    const float *Rt = sR + t * 9;
    float rx, ry, rz;
    rotate(Rt, px, py, pz, rx, ry, rz);

    const float *const gbase[3] = {g0, g1, g2};
    float *const ggbase[3] = {gg0, gg1, gg2};
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};

    float dgx[3] = {0.f, 0.f, 0.f}; // dL/d(ix_p), pre-unnormalize
    float dgy[3] = {0.f, 0.f, 0.f};

    if (active) {
        Tap tap[3];
        float s[3], dsdix[3], dsdiy[3];
        const float go = gout[group * C + c]; // gout[b, t*C + c]
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            tap[p] = make_tap(gxs[p], gys[p], geom.H[p], geom.W[p], C);
            const Tap tp = tap[p];
            const float *pc = gbase[p] + (long)t * geom.tstride[p] + c;
            const float v0 = pc[tp.off[0]], v1 = pc[tp.off[1]];
            const float v2 = pc[tp.off[2]], v3 = pc[tp.off[3]];
            s[p] = v0 * tp.w[0] + v1 * tp.w[1] + v2 * tp.w[2] + v3 * tp.w[3];
            // Clamped +1 corners read the 0-side value instead of torch's
            // guarded 0, but that only changes dsdix/dsdiy exactly where the
            // point sits on the border — where gxm/gym is 0 anyway.
            dsdix[p] = (1.f - tp.ty) * (v1 - v0) + tp.ty * (v3 - v2);
            dsdiy[p] = (1.f - tp.tx) * (v2 - v0) + tp.tx * (v3 - v1);
        }
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            // d(prod)/d(s_p) = product of the other two plane samples
            const float other = s[(p + 1) % 3] * s[(p + 2) % 3] * go;
            const Tap tp = tap[p];
            float *gpc = ggbase[p] + (long)t * geom.tstride[p] + c;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val = other * tp.w[k];
                if (val != 0.f) atomicAdd(&gpc[tp.off[k]], val);
            }
            dgx[p] += other * dsdix[p];
            dgy[p] += other * dsdiy[p];
        }
    }

    // Segment-reduce the 6 coordinate-grad sums over channels.
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        for (int off = 1; off < CW; off <<= 1) {
            const float ox = __shfl_down_sync(0xffffffffu, dgx[p], off);
            const float oy = __shfl_down_sync(0xffffffffu, dgy[p], off);
            if (lis + off < CW) {
                dgx[p] += ox;
                dgy[p] += oy;
            }
        }
    }

    if (active && lis == 0) {
        // Map plane-coordinate grads back to the rotated point
        // (gx, gy) per plane: XY=(rx,ry), ZX=(rz,rx), YZ=(ry,rz).
        // make_tap is deterministic, so recomputing the masks is exact.
        const Tap t0 = make_tap(gxs[0], gys[0], geom.H[0], geom.W[0], C);
        const Tap t1 = make_tap(gxs[1], gys[1], geom.H[1], geom.W[1], C);
        const Tap t2 = make_tap(gxs[2], gys[2], geom.H[2], geom.W[2], C);
        const float drx = dgx[0] * t0.gxm + dgy[1] * t1.gym;
        const float dry = dgy[0] * t0.gym + dgx[2] * t2.gxm;
        const float drz = dgx[1] * t1.gxm + dgy[2] * t2.gym;

        // dL/dR_t[i, j] = dr_i * p_j (partial sums are fine: linear)
        float *sg = sgR + t * 9;
        atomicAdd(&sg[0], drx * px);
        atomicAdd(&sg[1], drx * py);
        atomicAdd(&sg[2], drx * pz);
        atomicAdd(&sg[3], dry * px);
        atomicAdd(&sg[4], dry * py);
        atomicAdd(&sg[5], dry * pz);
        atomicAdd(&sg[6], drz * px);
        atomicAdd(&sg[7], drz * py);
        atomicAdd(&sg[8], drz * pz);

        // dL/dp = R_t^T @ dr, accumulated across the point's T groups
        atomicAdd(&gpts[3 * b], Rt[0] * drx + Rt[3] * dry + Rt[6] * drz);
        atomicAdd(&gpts[3 * b + 1], Rt[1] * drx + Rt[4] * dry + Rt[7] * drz);
        atomicAdd(&gpts[3 * b + 2], Rt[2] * drx + Rt[5] * dry + Rt[8] * drz);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        if (sgR[i] != 0.f) atomicAdd(&gR[i], sgR[i]);
    }
}

constexpr int kThreads = 256;

inline int n_blocks(long n) { return (int)((n + kThreads - 1) / kThreads); }

/* One tensor [3T, H, W, C]: plane type p of rotation t starts at
 * (t*3 + p)·H·W·C, i.e. base p·H·W·C with a per-rotation stride of 3·H·W·C. */
inline PlaneGeom geom_single(int C, int H, int W) {
    const long hwc = (long)H * W * C;
    PlaneGeom g;
    for (int p = 0; p < 3; ++p) {
        g.tstride[p] = 3 * hwc;
        g.H[p] = H;
        g.W[p] = W;
    }
    return g;
}

/* Three tensors, plane type p being [T, H_p, W_p, C]. */
inline PlaneGeom geom_triple(int C, int H0, int W0, int H1, int W1, int H2, int W2) {
    PlaneGeom g;
    g.H[0] = H0; g.W[0] = W0;
    g.H[1] = H1; g.W[1] = W1;
    g.H[2] = H2; g.W[2] = W2;
    for (int p = 0; p < 3; ++p) g.tstride[p] = (long)g.H[p] * g.W[p] * C;
    return g;
}

inline long bwd_warps(long n_groups, int C) {
    if (C <= 32) {
        const int spw = 32 / (C < 32 ? C : 32);
        return (n_groups + spw - 1) / spw;
    }
    return n_groups * ((C + 31) >> 5);
}

} // namespace

void kplanes_tilted_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float *d_out,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long hwc = (long)H * W * C;
    const size_t shmem = (size_t)T * 9 * sizeof(float);
    kplanes_tilted_fwd_kernel<<<n_blocks(B * (long)(T * C)), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_grid + hwc, d_grid + 2 * hwc, geom_single(C, H, W),
        d_out, B, T, C);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float *d_ggrid,
    float *d_gR,
    float *d_gpts,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long hwc = (long)H * W * C;
    const long n_warps = bwd_warps(B * (long)T, C);
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    kplanes_tilted_bwd_kernel<<<n_blocks(n_warps * 32), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_grid + hwc, d_grid + 2 * hwc, geom_single(C, H, W),
        d_gout, d_ggrid, d_ggrid + hwc, d_ggrid + 2 * hwc, d_gR, d_gpts, B, T, C);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse3_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    float *d_out,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    cudaStream_t stream
) {
    if (B == 0) return;
    const size_t shmem = (size_t)T * 9 * sizeof(float);
    kplanes_tilted_fwd_kernel<<<n_blocks(B * (long)(T * C)), kThreads, shmem, stream>>>(
        d_pts, d_R, d_plane0, d_plane1, d_plane2,
        geom_triple(C, H0, W0, H1, W1, H2, W2), d_out, B, T, C);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse3_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    const float *d_gout,
    float *d_gplane0,
    float *d_gplane1,
    float *d_gplane2,
    float *d_gR,
    float *d_gpts,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long n_warps = bwd_warps(B * (long)T, C);
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    kplanes_tilted_bwd_kernel<<<n_blocks(n_warps * 32), kThreads, shmem, stream>>>(
        d_pts, d_R, d_plane0, d_plane1, d_plane2,
        geom_triple(C, H0, W0, H1, W1, H2, W2),
        d_gout, d_gplane0, d_gplane1, d_gplane2, d_gR, d_gpts, B, T, C);
    CUDA_CHECK_KERNEL();
}

/* ── TV-specialized variant ──────────────────────────────────────────────────
 *
 * kplanes_tilted_tv_fuse evaluates the kplanes_tilted_fuse computation at
 * 4 tap locations per (point, rotation) to support TV-based regularization:
 *
 *   tap 0: rotated = R · x
 *   tap 1: rotated = R · x + h · R[:,0]   (finite-difference along world x)
 *   tap 2: rotated = R · x + h · R[:,1]   (finite-difference along world y)
 *   tap 3: rotated = R · x + h · R[:,2]   (finite-difference along world z)
 *
 * Key structure: R·x is computed ONCE per (point, rotation). The three column
 * offsets h·R[:,i] are shared across ALL points under that rotation, so they
 * are staged in shared memory (T × 3 × 3 floats = T × 9, same as R itself).
 *
 * Output layout: [4, B, T*C] — tap dimension outermost so that out[0] has
 * exactly the same layout as kplanes_tilted_fuse output, enabling direct
 * chunk-based use in downstream code.
 *
 * The backward has a correctness trap in grad_R: R enters the computation
 * both through R·x (all 4 taps) and through h·R[:,i] (taps 1-3). The full
 * gradient is the sum of both contributions — see the backward kernel comment.
 *
 * Both plane layouts go through these kernels via PlaneGeom, exactly as in
 * the base variant.
 */

namespace {  // tv namespace (reuse make_tap / rotate / PlaneGeom from above)

/* ── TV forward kernel ────────────────────────────────────────────────────── */
__global__ void kplanes_tilted_tv_fwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const float *__restrict__ g0, // plane-type bases, channels-last
    const float *__restrict__ g1,
    const float *__restrict__ g2,
    PlaneGeom geom,
    float *__restrict__ out,        // [4, B, T*C] — tap outermost
    long B, int T, int C,
    float h
) {
    // Shared: [T*9] R copy + [T*9] columns h*R[:,0..2] (the 3 tap offsets)
    // Total: T*9 + T*9 = T*18 floats.
    extern __shared__ float smem[];
    float *sR  = smem;          // [T*9]: rotation matrices
    float *sRc = smem + T * 9;  // [T*9]: h * R columns (col-major in groups of 3)
    //   sRc[t*9 + i*3 + j] = h * R[t, j, i]  (column i, row j)
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        // R is row-major [T, 3, 3], stored as R[t*9 + row*3 + col].
        // Column c of R_t is R[t*9 + 0*3+c], R[t*9 + 1*3+c], R[t*9 + 2*3+c].
        // Flatten: t*9 + i = t*9 + col*3 + row  →  col = (i%9)/3, row = (i%9)%3
        // But we write sRc indexed as sRc[t*9 + col*3 + row]:
        const int local = i % 9;
        const int t_idx = i / 9;
        const int row   = local / 3;  // row in R
        const int col   = local % 3;  // column index in R
        // sRc[t*9 + col*3 + row] = h * R[t*9 + row*3 + col]
        sRc[t_idx * 9 + col * 3 + row] = h * R[t_idx * 9 + row * 3 + col];
    }
    __syncthreads();

    // One thread per (tap, point, rotation, channel) — same lane-per-channel
    // parallelism as the base kernel, with an extra tap dimension.
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = 4L * B * (long)(T * C);
    if (tid >= total) return;

    // Decompose: outermost = tap, then same as base kernel.
    const int tap    = (int)(tid / (B * (long)(T * C)));
    const long rem0  = tid - (long)tap * B * (long)(T * C);
    const long b     = rem0 / (T * C);
    const int rem1   = (int)(rem0 - b * (T * C));
    const int t      = rem1 / C;
    const int c      = rem1 - t * C;

    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];

    // Base rotated coordinate (R · x), shared by all 4 taps.
    float rx, ry, rz;
    rotate(sR + t * 9, px, py, pz, rx, ry, rz);

    // Tap offsets: tap 0 = base; taps 1-3 = base + h*R[:,tap-1].
    if (tap >= 1) {
        const float *rc = sRc + t * 9 + (tap - 1) * 3; // column (tap-1) of h*R_t
        rx += rc[0];
        ry += rc[1];
        rz += rc[2];
    }

    const float *const gbase[3] = {g0, g1, g2};
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};

    float prod = 1.f;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const Tap tp = make_tap(gxs[p], gys[p], geom.H[p], geom.W[p], C);
        const float *pc = gbase[p] + (long)t * geom.tstride[p] + c;
        prod *= tp.w[0] * pc[tp.off[0]] + tp.w[1] * pc[tp.off[1]]
              + tp.w[2] * pc[tp.off[2]] + tp.w[3] * pc[tp.off[3]];
    }
    // out[tap, b, t*C+c] — tap outermost, same internal layout as base op
    out[((long)tap * B + b) * (T * C) + (t * C + c)] = prod;
}

/* ── TV backward kernel ───────────────────────────────────────────────────── */
//
// grad_R CORRECTNESS NOTE:
// R enters the output at tap k through two routes:
//
//   Route A: base rotation R·x (all 4 taps)
//     rotated = R·x (+ h·R[:,k-1] for taps 1-3)
//     d(rotated_i)/d(R[i,j]) = x_j  →  the usual outer-product term
//
//   Route B: the column offset h·R[:,k-1] for taps 1, 2, 3.
//     For tap k (k in {1,2,3}): rotated += h · R[:,k-1]
//     d(rotated_i)/d(R[i, k-1]) += h  →  adds h · coord_grad[i] to dL/dR[i, k-1]
//     (only to the R-column used by that tap, not to the other columns)
//
// Concretely for tap k ∈ {1,2,3} with axis axis=k-1:
//   dL/dR[i, axis] += h * coord_grad_i   (on top of Route A's x_j term)
// Route A and B are both linear → we sum them.
//
__global__ void kplanes_tilted_tv_bwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const float *__restrict__ g0, // plane-type bases, channels-last
    const float *__restrict__ g1,
    const float *__restrict__ g2,
    PlaneGeom geom,
    const float *__restrict__ gout, // [4, B, T*C]
    float *__restrict__ gg0,        // grads for the plane bases, pre-zeroed
    float *__restrict__ gg1,
    float *__restrict__ gg2,
    float *__restrict__ gR,         // [T, 3, 3], pre-zeroed
    float *__restrict__ gpts,       // [B, 3], pre-zeroed
    long B, int T, int C,
    float h
) {
    extern __shared__ float smem[]; // [T*9] R | [T*9] h*R cols | [T*9] gR partials
    float *sR  = smem;
    float *sRc = smem + T * 9;
    float *sgR = smem + T * 18;
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        const int local = i % 9;
        const int t_idx = i / 9;
        const int row   = local / 3;
        const int col   = local % 3;
        sRc[t_idx * 9 + col * 3 + row] = h * R[t_idx * 9 + row * 3 + col];
        sgR[i] = 0.f;
    }
    __syncthreads();

    // Same warp-segment decomposition as the base backward, but over 4*B*T
    // groups (one group = one (tap, point, rotation)).
    const int lane = threadIdx.x & 31;
    const long warp_id = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int CW = C < 32 ? C : 32;
    long group;   // index over 4*B*T
    int c, lis;
    bool active;
    if (C <= 32) {
        const int spw = 32 / CW;
        const int seg = lane / CW;
        lis   = lane - seg * CW;
        group = warp_id * spw + seg;
        c     = lis;
        active = seg < spw && group < 4L * B * (long)T;
    } else {
        const int wpg = (C + 31) >> 5;
        group = warp_id / wpg;
        lis   = lane;
        c     = (int)(warp_id - group * wpg) * 32 + lane;
        active = group < 4L * B * (long)T && c < C;
    }

    const long n_groups  = 4L * B * (long)T;
    const long g_safe    = group < n_groups ? group : 0;
    const int  tap       = (int)(g_safe / (B * (long)T));
    const long bt_idx    = g_safe - (long)tap * B * (long)T;
    const long b         = bt_idx / T;
    const int  t         = (int)(bt_idx - b * T);

    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    const float *Rt = sR + t * 9;
    float rx, ry, rz;
    rotate(Rt, px, py, pz, rx, ry, rz);

    // Apply tap column offset (same as forward).
    if (tap >= 1) {
        const float *rc = sRc + t * 9 + (tap - 1) * 3;
        rx += rc[0];
        ry += rc[1];
        rz += rc[2];
    }

    const float *const gbase[3] = {g0, g1, g2};
    float *const ggbase[3] = {gg0, gg1, gg2};
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};

    float dgx[3] = {0.f, 0.f, 0.f};
    float dgy[3] = {0.f, 0.f, 0.f};

    if (active) {
        Tap taps[3];
        float s[3], dsdix[3], dsdiy[3];
        // gout layout: [4, B, T*C] — tap outermost.
        const long gout_off = ((long)tap * B + b) * (T * C) + t * C + c;
        const float go = gout[gout_off];
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            taps[p] = make_tap(gxs[p], gys[p], geom.H[p], geom.W[p], C);
            const Tap tp = taps[p];
            const float *pc = gbase[p] + (long)t * geom.tstride[p] + c;
            const float v0 = pc[tp.off[0]], v1 = pc[tp.off[1]];
            const float v2 = pc[tp.off[2]], v3 = pc[tp.off[3]];
            s[p] = v0 * tp.w[0] + v1 * tp.w[1] + v2 * tp.w[2] + v3 * tp.w[3];
            dsdix[p] = (1.f - tp.ty) * (v1 - v0) + tp.ty * (v3 - v2);
            dsdiy[p] = (1.f - tp.tx) * (v2 - v0) + tp.tx * (v3 - v1);
        }
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            const float other = s[(p + 1) % 3] * s[(p + 2) % 3] * go;
            const Tap tp = taps[p];
            float *gpc = ggbase[p] + (long)t * geom.tstride[p] + c;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val = other * tp.w[k];
                if (val != 0.f) atomicAdd(&gpc[tp.off[k]], val);
            }
            dgx[p] += other * dsdix[p];
            dgy[p] += other * dsdiy[p];
        }
    }

    // Segment-reduce 6 coordinate grads over channels.
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        for (int off = 1; off < CW; off <<= 1) {
            const float ox = __shfl_down_sync(0xffffffffu, dgx[p], off);
            const float oy = __shfl_down_sync(0xffffffffu, dgy[p], off);
            if (lis + off < CW) {
                dgx[p] += ox;
                dgy[p] += oy;
            }
        }
    }

    if (active && lis == 0) {
        // Map plane-coordinate grads to rotated-space components (same as base).
        const Tap t0 = make_tap(gxs[0], gys[0], geom.H[0], geom.W[0], C);
        const Tap t1 = make_tap(gxs[1], gys[1], geom.H[1], geom.W[1], C);
        const Tap t2 = make_tap(gxs[2], gys[2], geom.H[2], geom.W[2], C);
        const float drx = dgx[0] * t0.gxm + dgy[1] * t1.gym;
        const float dry = dgy[0] * t0.gym + dgx[2] * t2.gxm;
        const float drz = dgx[1] * t1.gxm + dgy[2] * t2.gym;

        // ── grad_R: Route A — same outer-product as the base kernel ──────────
        // dL/dR[i,j] += dr_i * x_j   (from R·x, all 4 taps)
        float *sg = sgR + t * 9;
        atomicAdd(&sg[0], drx * px);
        atomicAdd(&sg[1], drx * py);
        atomicAdd(&sg[2], drx * pz);
        atomicAdd(&sg[3], dry * px);
        atomicAdd(&sg[4], dry * py);
        atomicAdd(&sg[5], dry * pz);
        atomicAdd(&sg[6], drz * px);
        atomicAdd(&sg[7], drz * py);
        atomicAdd(&sg[8], drz * pz);

        // ── grad_R: Route B — tap column offset h·R[:,axis] ──────────────────
        // For taps 1,2,3 only: R also enters via h·R[:,axis], axis = tap-1.
        // d(rotated_i) / d(R[i, axis]) += h,  all other R entries unchanged.
        // So: dL/dR[i, axis] += h * dr_i   (for this tap only).
        if (tap >= 1) {
            const int axis = tap - 1; // column index in R that this tap uses
            // R is row-major: R[i, axis] is at index i*3 + axis.
            // dr = (drx, dry, drz) maps to rows i = 0, 1, 2.
            atomicAdd(&sg[0 * 3 + axis], h * drx);
            atomicAdd(&sg[1 * 3 + axis], h * dry);
            atomicAdd(&sg[2 * 3 + axis], h * drz);
        }

        // ── grad_pts: R^T · dr, accumulated over all T rotations AND all 4 taps
        atomicAdd(&gpts[3 * b],     Rt[0] * drx + Rt[3] * dry + Rt[6] * drz);
        atomicAdd(&gpts[3 * b + 1], Rt[1] * drx + Rt[4] * dry + Rt[7] * drz);
        atomicAdd(&gpts[3 * b + 2], Rt[2] * drx + Rt[5] * dry + Rt[8] * drz);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        if (sgR[i] != 0.f) atomicAdd(&gR[i], sgR[i]);
    }
}

} // namespace (tv)

void kplanes_tilted_tv_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float *d_out,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long hwc = (long)H * W * C;
    // shmem: [T*9] R + [T*9] h*R columns = T*18 floats
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    kplanes_tilted_tv_fwd_kernel<<<n_blocks(4L * B * (long)(T * C)), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_grid + hwc, d_grid + 2 * hwc, geom_single(C, H, W),
        d_out, B, T, C, h);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_tv_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float *d_ggrid,
    float *d_gR,
    float *d_gpts,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long hwc = (long)H * W * C;
    const long n_warps = bwd_warps(4L * B * (long)T, C);
    // shmem: [T*9] R + [T*9] h*R cols + [T*9] gR partials = T*27 floats
    const size_t shmem = (size_t)T * 27 * sizeof(float);
    kplanes_tilted_tv_bwd_kernel<<<n_blocks(n_warps * 32), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_grid + hwc, d_grid + 2 * hwc, geom_single(C, H, W),
        d_gout, d_ggrid, d_ggrid + hwc, d_ggrid + 2 * hwc, d_gR, d_gpts, B, T, C, h);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_tv_fuse3_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    float *d_out,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    cudaStream_t stream
) {
    if (B == 0) return;
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    kplanes_tilted_tv_fwd_kernel<<<n_blocks(4L * B * (long)(T * C)), kThreads, shmem, stream>>>(
        d_pts, d_R, d_plane0, d_plane1, d_plane2,
        geom_triple(C, H0, W0, H1, W1, H2, W2), d_out, B, T, C, h);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_tv_fuse3_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_plane0,
    const float *d_plane1,
    const float *d_plane2,
    const float *d_gout,
    float *d_gplane0,
    float *d_gplane1,
    float *d_gplane2,
    float *d_gR,
    float *d_gpts,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    cudaStream_t stream
) {
    if (B == 0) return;
    const long n_warps = bwd_warps(4L * B * (long)T, C);
    const size_t shmem = (size_t)T * 27 * sizeof(float);
    kplanes_tilted_tv_bwd_kernel<<<n_blocks(n_warps * 32), kThreads, shmem, stream>>>(
        d_pts, d_R, d_plane0, d_plane1, d_plane2,
        geom_triple(C, H0, W0, H1, W1, H2, W2),
        d_gout, d_gplane0, d_gplane1, d_gplane2, d_gR, d_gpts, B, T, C, h);
    CUDA_CHECK_KERNEL();
}
