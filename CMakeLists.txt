cmake_minimum_required(VERSION 3.24)

# quantem-cuda is a CUDA-only library: there is no CPU fallback build.
# The logic below locates an nvcc that users typically have without a
# system-wide CUDA install, then hard-fails with guidance if none is found.

# Pip-installed CUDA 13 toolkit detection.
# `nvidia-cuda-nvcc` (and friends) install at
# <env>/lib/pythonX.Y/site-packages/nvidia/cu13/{bin,include,lib,nvvm};
# CMake's find_package(CUDAToolkit) does not search site-packages, so seed
# CUDAToolkit_ROOT / CMAKE_CUDA_COMPILER before language detection runs.
if(NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED CMAKE_CUDA_COMPILER)
    if(NOT DEFINED Python_EXECUTABLE)
        find_program(Python_EXECUTABLE NAMES python3 python)
    endif()
    if(Python_EXECUTABLE)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c [==[
import os, sys
try:
    import nvidia
except ImportError:
    sys.exit(0)
for base in nvidia.__path__:
    cu13 = os.path.join(base, 'cu13')
    if os.path.isfile(os.path.join(cu13, 'bin', 'nvcc')):
        print(cu13)
        break
]==]
            OUTPUT_VARIABLE _PIP_CUDA_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_PIP_CUDA_ROOT)
            set(_PIP_CUDA_ROOT_USABLE TRUE)
            # cu13 wheels ship versioned libs only (libcudart.so.13); seed the
            # unversioned symlink that CMake's find_library expects.
            file(GLOB _cudart_versioned "${_PIP_CUDA_ROOT}/lib/libcudart.so.*")
            if(_cudart_versioned AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib/libcudart.so")
                list(GET _cudart_versioned 0 _cudart_first)
                file(
                    CREATE_LINK "${_cudart_first}" "${_PIP_CUDA_ROOT}/lib/libcudart.so"
                    SYMBOLIC
                    RESULT _cudart_link_result
                )
                if(NOT _cudart_link_result STREQUAL "0")
                    set(_PIP_CUDA_ROOT_USABLE FALSE)
                    message(WARNING "Unable to create libcudart.so link in ${_PIP_CUDA_ROOT}/lib: ${_cudart_link_result}")
                endif()
            endif()
            # nvcc's device link step searches `<root>/lib64/`; the wheel uses
            # `lib/`. Add a sibling symlink so -lcudadevrt etc. resolve.
            if(_PIP_CUDA_ROOT_USABLE AND NOT EXISTS "${_PIP_CUDA_ROOT}/lib64")
                file(
                    CREATE_LINK "lib" "${_PIP_CUDA_ROOT}/lib64"
                    SYMBOLIC
                    RESULT _lib64_link_result
                )
                if(NOT _lib64_link_result STREQUAL "0")
                    set(_PIP_CUDA_ROOT_USABLE FALSE)
                    message(WARNING "Unable to create lib64 link in ${_PIP_CUDA_ROOT}: ${_lib64_link_result}")
                endif()
            endif()
            if(_PIP_CUDA_ROOT_USABLE)
                message(STATUS "Detected pip-installed CUDA toolkit at ${_PIP_CUDA_ROOT}")
                set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "")
                set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "")
            else()
                message(STATUS "Skipping pip CUDA auto-detect because toolkit links could not be prepared")
            endif()
        endif()
    endif()
endif()

# Conda env CUDA toolkit fallback. If no pip wheel was found and the user is
# inside a conda env that ships nvcc, use that.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CUDAToolkit_ROOT)
    set(_CONDA_ENV_ROOT "$ENV{CONDA_PREFIX}")
    if(_CONDA_ENV_ROOT AND EXISTS "${_CONDA_ENV_ROOT}/bin/nvcc")
        message(STATUS "Detected conda env CUDA toolkit at ${_CONDA_ENV_ROOT}")
        set(CUDAToolkit_ROOT "${_CONDA_ENV_ROOT}" CACHE PATH "")
        set(CMAKE_CUDA_COMPILER "${_CONDA_ENV_ROOT}/bin/nvcc" CACHE FILEPATH "")
    endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "==================================================================\n"
        "  quantem-cuda: no CUDA toolchain found. This package is CUDA-only.\n"
        "  Install a CUDA 13.x toolkit, e.g.\n"
        "      pip install 'nvidia-cuda-nvcc==13.0.*' 'nvidia-cuda-runtime==13.0.*'\n"
        "  or activate a conda env shipping nvcc 13.x, then retry.\n"
        "  To point at an existing install, pass\n"
        "      --config-settings=cmake.define.CUDAToolkit_ROOT=/path/to/cuda\n"
        "==================================================================")
endif()

# Must be set before enable_language(CUDA) / target creation to apply.
# CUDA 13 dropped pre-Turing archs; this list spans Turing (75) through
# Blackwell (100 datacenter, 120 workstation/consumer).
# Override at configure time via -DCMAKE_CUDA_ARCHITECTURES="80;86"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90 100 120)
endif()

project(quantem_cuda LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# PyTorch cu13 wheels (torch >= 2.9) ship libcudart.so.13; pin the toolkit
# major so the extension resolves against the libcudart torch already loaded.
if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 13)
    message(FATAL_ERROR
        "quantem-cuda requires a CUDA 13.x toolkit (found ${CUDAToolkit_VERSION}). "
        "PyTorch cu13 builds ship libcudart.so.13; building against another "
        "major version would mismatch at import time. Install "
        "nvidia-cuda-nvcc==13.0.* or a conda env shipping nvcc 13.x.")
endif()

# Host C++ compiler floor: GCC >= 10 (older toolchains miss C++17 polish and
# aren't tested against CUDA 13).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "quantem-cuda requires GCC >= 10 (found ${CMAKE_CXX_COMPILER_VERSION}). "
        "Install a newer GCC or pass -DCMAKE_CXX_COMPILER=/path/to/g++-N.")
endif()

pybind11_add_module(_core
    csrc/bindings.cpp
    csrc/bindings/core.cpp
    csrc/bindings/core/ml.cpp
    csrc/cuda/core/tv_loss.cu
    csrc/cuda/core/ml/kplanes_tilted.cu
    csrc/cuda/core/ml/splat_stamp.cu
)
target_include_directories(_core PRIVATE csrc)
target_link_libraries(_core PRIVATE CUDA::cudart)

# Force shared libcudart so the symbol resolves to the libcudart already
# loaded by PyTorch at import time. This is what lets one wheel work across
# torch versions; it also avoids bundling cudart_static into the .so.
set_target_properties(_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_RUNTIME_LIBRARY Shared
)

install(TARGETS _core DESTINATION quantem/cuda)
