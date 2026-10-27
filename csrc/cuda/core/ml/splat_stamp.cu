/* ── csrc/cuda/core/ml/splat_stamp.cu ──────────────────────────────────────────────
 * Fused 2D Gaussian stamping for the splat tomography renderer
 * (quantem.core.ml.splat.renderer.render_projections, backend="cuda").
 *
 * Replaces the torch chain
 *     round(centres) → window indices (T, N, W2, 2) → exp → mask → scatter_add
 * which materialises every atom's pixel window for every tilt (and keeps it
 * for the backward): O(T N W2) memory, 36 GB at 18 000 atoms. Here nothing
 * per (tilt, atom, pixel) is ever stored.
 *
 * Contract (all fp32, contiguous, pixel units; no pose logic — the
 * projection u = pi(R p + s) and its gradients stay in torch):
 *   u    [T, N, 2]  fractional pixel centre (row, col) of atom n in tilt t
 *   a    [N]        pixel-integrated mass per atom (image units · px²)
 *   s    [N]        width in pixels (> 0)
 *   out  [T, H, W]  out[t,i,j] += a_n/(2π s_n²) · exp(-((i-u_r)² + (j-u_c)²) / (2 s_n²))
 *                   over the atoms whose window rint(u) + {-r..r}² contains (i, j)
 * The window is truncated in INDEX space around the rounded centre (rintf =
 * round half to even = torch.round) and clipped to the image, exactly as the
 * torch path; the Gaussian is evaluated at the continuous centre, at pixel
 * centres, no pixel integration.
 *
 * Forward: one thread block per (tilt, atom); threads stride over the
 * (2r+1)² window pixels and atomicAdd into out (pre-zeroed by the caller).
 * Backward: one block per (tilt, atom) READS its window of grad_out and
 * reduces in-block (registers → warp shuffles → shared) the four sums
 *   S0 = Σ g·e, S1 = Σ g·e·d_r, S2 = Σ g·e·d_c, S3 = Σ g·e·|d|²,
 * e = exp(-|d|²/(2s²))/(2π s²), d = pixel - u, from which (v = a·e)
 *   ∂L/∂u = a·(S1, S2)/s²,  ∂L/∂a = S0,  ∂L/∂s = a·(S3/s³ - 2 S0/s).
 * grad_u[t, n] has a unique owner block and is written directly; the
 * amplitude and width terms are written as per-(tilt, atom) partials
 * ga_part / gs_part [T, N] that the wrapper sums over tilts in torch, so the
 * backward uses no atomics and is bit-reproducible (the forward's atomics
 * are reproducible only to fp32 round-off).
 *
 * Atoms with a non-finite centre, amplitude or width, and atoms whose window
 * lies entirely outside the image, contribute nothing and get zero
 * gradients (torch would NaN-poison pixel (0, 0) through the clamped index
 * of a NaN centre — an implementation artefact, not part of the contract).
 */

#include "common.cuh"
#include "ops/core/ml.h"

namespace {

constexpr int kMaxThreads = 256;  // 8 warps
constexpr float kTwoPi = 6.283185307179586f;

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* Per-(tilt, atom) setup shared by both kernels. Returns false when the atom
 * stamps nothing (non-finite inputs or window fully off-image). */
struct AtomWin {
    float ur, uc, a, s, inv_s2, norm;
    int cr, cc;  // rounded centre
};

__device__ inline bool setup_atom(
    const float *__restrict__ u, const float *__restrict__ a, const float *__restrict__ s,
    long tn, long n, int H, int W, int r, AtomWin &w
) {
    w.ur = u[2 * tn];
    w.uc = u[2 * tn + 1];
    w.a = a[n];
    w.s = s[n];
    if (!isfinite(w.ur) || !isfinite(w.uc) || !isfinite(w.a) || !isfinite(w.s)) return false;
    // clamp before the int cast (UB for huge floats); a centre this far away is off-image anyway
    const float urc = fminf(fmaxf(w.ur, -1.0e9f), 1.0e9f);
    const float ucc = fminf(fmaxf(w.uc, -1.0e9f), 1.0e9f);
    w.cr = (int)rintf(urc);
    w.cc = (int)rintf(ucc);
    if (w.cr + r < 0 || w.cr - r >= H || w.cc + r < 0 || w.cc - r >= W) return false;
    w.inv_s2 = 1.0f / (w.s * w.s);
    w.norm = 1.0f / (kTwoPi * w.s * w.s);
    return true;
}

__global__ void splat_stamp_fwd_kernel(
    const float *__restrict__ u, const float *__restrict__ a, const float *__restrict__ s,
    float *__restrict__ out, long block0, long TN, long N, int H, int W, int r
) {
    const long tn = block0 + (long)blockIdx.x;
    if (tn >= TN) return;
    const long t = tn / N;
    const long n = tn - t * N;
    AtomWin w;
    if (!setup_atom(u, a, s, tn, n, H, W, r, w)) return;
    const float scale = w.a * w.norm;
    const float inv2s2 = 0.5f * w.inv_s2;
    const int side = 2 * r + 1;
    const int w2 = side * side;
    float *o = out + t * (long)H * W;
    for (int p = threadIdx.x; p < w2; p += blockDim.x) {
        const int q = p / side;
        const int i = w.cr + (q - r);
        const int j = w.cc + (p - q * side - r);
        if (i < 0 || i >= H || j < 0 || j >= W) continue;
        const float dr = (float)i - w.ur;
        const float dc = (float)j - w.uc;
        const float v = scale * expf(-(dr * dr + dc * dc) * inv2s2);
        atomicAdd(o + (long)i * W + j, v);
    }
}

__global__ void splat_stamp_bwd_kernel(
    const float *__restrict__ u, const float *__restrict__ a, const float *__restrict__ s,
    const float *__restrict__ gout, float *__restrict__ gu, float *__restrict__ ga_part,
    float *__restrict__ gs_part, long block0, long TN, long N, int H, int W, int r
) {
    __shared__ float sh[4][kMaxThreads / 32];
    const long tn = block0 + (long)blockIdx.x;
    if (tn >= TN) return;  // whole block exits together: no __syncthreads hazard
    const long t = tn / N;
    const long n = tn - t * N;
    AtomWin w;
    const bool active = setup_atom(u, a, s, tn, n, H, W, r, w);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    if (active) {
        const float inv2s2 = 0.5f * w.inv_s2;
        const int side = 2 * r + 1;
        const int w2 = side * side;
        const float *g = gout + t * (long)H * W;
        for (int p = threadIdx.x; p < w2; p += blockDim.x) {
            const int q = p / side;
            const int i = w.cr + (q - r);
            const int j = w.cc + (p - q * side - r);
            if (i < 0 || i >= H || j < 0 || j >= W) continue;
            const float dr = (float)i - w.ur;
            const float dc = (float)j - w.uc;
            const float d2 = dr * dr + dc * dc;
            const float ge = g[(long)i * W + j] * w.norm * expf(-d2 * inv2s2);
            s0 += ge;
            s1 += ge * dr;
            s2 += ge * dc;
            s3 += ge * d2;
        }
    }
    s0 = warp_sum(s0);
    s1 = warp_sum(s1);
    s2 = warp_sum(s2);
    s3 = warp_sum(s3);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        sh[0][warp] = s0;
        sh[1][warp] = s1;
        sh[2][warp] = s2;
        sh[3][warp] = s3;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
        for (int k = 0; k < nwarps; ++k) {
            r0 += sh[0][k];
            r1 += sh[1][k];
            r2 += sh[2][k];
            r3 += sh[3][k];
        }
        if (active) {
            gu[2 * tn] = w.a * r1 * w.inv_s2;
            gu[2 * tn + 1] = w.a * r2 * w.inv_s2;
            ga_part[tn] = r0;
            gs_part[tn] = w.a * (r3 * w.inv_s2 / w.s - 2.0f * r0 / w.s);
        } else {
            gu[2 * tn] = 0.f;
            gu[2 * tn + 1] = 0.f;
            ga_part[tn] = 0.f;
            gs_part[tn] = 0.f;
        }
    }
}

inline int window_threads(int r) {
    const long w2 = (2L * r + 1) * (2L * r + 1);
    const long rounded = ((w2 + 31) / 32) * 32;
    return (int)(rounded < kMaxThreads ? rounded : kMaxThreads);
}

constexpr long kMaxGrid = 1L << 30;  // blocks per launch (grid.x limit is 2^31 - 1)

}  // namespace

void splat_stamp_fwd_cuda(
    const float *d_u, const float *d_a, const float *d_s, float *d_out,
    long T, long N, int H, int W, int r, cudaStream_t stream
) {
    const long TN = T * N;
    if (TN == 0 || H <= 0 || W <= 0) return;
    const int threads = window_threads(r);
    for (long b0 = 0; b0 < TN; b0 += kMaxGrid) {
        const long nb = (TN - b0 < kMaxGrid) ? TN - b0 : kMaxGrid;
        splat_stamp_fwd_kernel<<<(unsigned)nb, threads, 0, stream>>>(
            d_u, d_a, d_s, d_out, b0, TN, N, H, W, r);
        CUDA_CHECK_KERNEL();
    }
}

void splat_stamp_bwd_cuda(
    const float *d_u, const float *d_a, const float *d_s, const float *d_gout,
    float *d_gu, float *d_ga_part, float *d_gs_part,
    long T, long N, int H, int W, int r, cudaStream_t stream
) {
    const long TN = T * N;
    if (TN == 0) return;
    if (H <= 0 || W <= 0) {
        QUANTEM_CUDA_CHECK(cudaMemsetAsync(d_gu, 0, sizeof(float) * 2 * TN, stream));
        QUANTEM_CUDA_CHECK(cudaMemsetAsync(d_ga_part, 0, sizeof(float) * TN, stream));
        QUANTEM_CUDA_CHECK(cudaMemsetAsync(d_gs_part, 0, sizeof(float) * TN, stream));
        return;
    }
    const int threads = window_threads(r);
    for (long b0 = 0; b0 < TN; b0 += kMaxGrid) {
        const long nb = (TN - b0 < kMaxGrid) ? TN - b0 : kMaxGrid;
        splat_stamp_bwd_kernel<<<(unsigned)nb, threads, 0, stream>>>(
            d_u, d_a, d_s, d_gout, d_gu, d_ga_part, d_gs_part, b0, TN, N, H, W, r);
        CUDA_CHECK_KERNEL();
    }
}
