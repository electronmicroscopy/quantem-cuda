/* ── csrc/bindings/core/ml.cpp ───────────────────────────────────────────
 * Raw-pointer bindings for the quantem.cuda.core.ml kernels (K-Planes /
 * tensor-decomposition models). Registered into the single `_core`
 * extension by csrc/bindings.cpp; see csrc/bindings/registry.h for the
 * conventions.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "bindings/registry.h"
#include "ops/core/ml.h"

namespace py = pybind11;

namespace quantem_cuda {

static void py_kplanes_tilted_fuse_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long out_ptr,
    long B, int T, int C, int H, int W,
    long stream_ptr
) {
    kplanes_tilted_fuse_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H, W,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_grad_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long gout_ptr,
    long ggrid_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C, int H, int W,
    long stream_ptr
) {
    kplanes_tilted_fuse_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H, W,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long out_ptr,
    long B, int T, int C, int H, int W,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H, W, h,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse_grad_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long gout_ptr,
    long ggrid_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C, int H, int W,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H, W, h,
        to_stream(stream_ptr)
    );
}

/* ── per-plane-type resolution variants (*_fuse3) ───────────────────────── */

static void py_kplanes_tilted_fuse3_cuda(
    long pts_ptr, long r_ptr, long plane0_ptr, long plane1_ptr, long plane2_ptr,
    long out_ptr,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    long stream_ptr
) {
    kplanes_tilted_fuse3_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(plane0_ptr),
        reinterpret_cast<const float *>(plane1_ptr),
        reinterpret_cast<const float *>(plane2_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H0, W0, H1, W1, H2, W2,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse3_grad_cuda(
    long pts_ptr, long r_ptr, long plane0_ptr, long plane1_ptr, long plane2_ptr,
    long gout_ptr,
    long gplane0_ptr, long gplane1_ptr, long gplane2_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    long stream_ptr
) {
    kplanes_tilted_fuse3_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(plane0_ptr),
        reinterpret_cast<const float *>(plane1_ptr),
        reinterpret_cast<const float *>(plane2_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(gplane0_ptr),
        reinterpret_cast<float *>(gplane1_ptr),
        reinterpret_cast<float *>(gplane2_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H0, W0, H1, W1, H2, W2,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse3_cuda(
    long pts_ptr, long r_ptr, long plane0_ptr, long plane1_ptr, long plane2_ptr,
    long out_ptr,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse3_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(plane0_ptr),
        reinterpret_cast<const float *>(plane1_ptr),
        reinterpret_cast<const float *>(plane2_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H0, W0, H1, W1, H2, W2, h,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse3_grad_cuda(
    long pts_ptr, long r_ptr, long plane0_ptr, long plane1_ptr, long plane2_ptr,
    long gout_ptr,
    long gplane0_ptr, long gplane1_ptr, long gplane2_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C,
    int H0, int W0, int H1, int W1, int H2, int W2,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse3_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(plane0_ptr),
        reinterpret_cast<const float *>(plane1_ptr),
        reinterpret_cast<const float *>(plane2_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(gplane0_ptr),
        reinterpret_cast<float *>(gplane1_ptr),
        reinterpret_cast<float *>(gplane2_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H0, W0, H1, W1, H2, W2, h,
        to_stream(stream_ptr)
    );
}

void register_core_ml_ops(py::module_ &m) {
    m.def("kplanes_tilted_fuse_cuda", &py_kplanes_tilted_fuse_cuda,
          "Fused TILTED K-Planes interpolation (one level). pts_ptr: fp32 "
          "[B,3]; r_ptr: fp32 [T,3,3]; grid_ptr: fp32 [3T,C,H,W]; out_ptr: "
          "fp32 [B,T*C], fully written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_grad_cuda", &py_kplanes_tilted_fuse_grad_cuda,
          "Backward of kplanes_tilted_fuse. gout_ptr: fp32 [B,T*C]. "
          "ggrid/gr are accumulated into (pre-zero them); gpts is fully "
          "written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("gout_ptr"), py::arg("ggrid_ptr"), py::arg("gr_ptr"),
          py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse_cuda", &py_kplanes_tilted_tv_fuse_cuda,
          "TV-specialized fused TILTED K-Planes forward. pts_ptr: fp32 [B,3]; "
          "r_ptr: fp32 [T,3,3]; grid_ptr: fp32 [3T,H,W,C] channels-last; "
          "out_ptr: fp32 [4,B,T*C], fully written. "
          "h: finite-difference step in world [-1,1] units.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("h"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse_grad_cuda", &py_kplanes_tilted_tv_fuse_grad_cuda,
          "Backward of kplanes_tilted_tv_fuse. gout_ptr: fp32 [4,B,T*C]. "
          "ggrid/gr are accumulated into (pre-zero them); gpts is fully "
          "written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("gout_ptr"), py::arg("ggrid_ptr"), py::arg("gr_ptr"),
          py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("h"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse3_cuda", &py_kplanes_tilted_fuse3_cuda,
          "Fused TILTED K-Planes interpolation with per-plane-type "
          "resolutions. pts_ptr: fp32 [B,3]; r_ptr: fp32 [T,3,3]; plane<p>_ptr: "
          "fp32 [T,H<p>,W<p>,C] channels-last; out_ptr: fp32 [B,T*C], fully "
          "written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("plane0_ptr"),
          py::arg("plane1_ptr"), py::arg("plane2_ptr"), py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"),
          py::arg("H0"), py::arg("W0"), py::arg("H1"), py::arg("W1"),
          py::arg("H2"), py::arg("W2"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse3_grad_cuda", &py_kplanes_tilted_fuse3_grad_cuda,
          "Backward of kplanes_tilted_fuse3. gout_ptr: fp32 [B,T*C]. "
          "gplane0/1/2 and gr are accumulated into (pre-zero them); gpts is "
          "accumulated too.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("plane0_ptr"),
          py::arg("plane1_ptr"), py::arg("plane2_ptr"), py::arg("gout_ptr"),
          py::arg("gplane0_ptr"), py::arg("gplane1_ptr"), py::arg("gplane2_ptr"),
          py::arg("gr_ptr"), py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"),
          py::arg("H0"), py::arg("W0"), py::arg("H1"), py::arg("W1"),
          py::arg("H2"), py::arg("W2"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse3_cuda", &py_kplanes_tilted_tv_fuse3_cuda,
          "TV-specialized fused TILTED K-Planes forward with per-plane-type "
          "resolutions. out_ptr: fp32 [4,B,T*C], fully written. "
          "h: finite-difference step in world [-1,1] units.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("plane0_ptr"),
          py::arg("plane1_ptr"), py::arg("plane2_ptr"), py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"),
          py::arg("H0"), py::arg("W0"), py::arg("H1"), py::arg("W1"),
          py::arg("H2"), py::arg("W2"),
          py::arg("h"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse3_grad_cuda", &py_kplanes_tilted_tv_fuse3_grad_cuda,
          "Backward of kplanes_tilted_tv_fuse3. gout_ptr: fp32 [4,B,T*C]. "
          "gplane0/1/2, gr and gpts are accumulated into (pre-zero them).",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("plane0_ptr"),
          py::arg("plane1_ptr"), py::arg("plane2_ptr"), py::arg("gout_ptr"),
          py::arg("gplane0_ptr"), py::arg("gplane1_ptr"), py::arg("gplane2_ptr"),
          py::arg("gr_ptr"), py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"),
          py::arg("H0"), py::arg("W0"), py::arg("H1"), py::arg("W1"),
          py::arg("H2"), py::arg("W2"),
          py::arg("h"), py::arg("stream_ptr"));
}

}  // namespace quantem_cuda
