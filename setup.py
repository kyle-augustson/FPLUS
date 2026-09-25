"""
setup.py
========
Build the _sparse_lu_c Python C extension.

Usage
-----
    # Development (in-place build, single-threaded):
    python setup.py build_ext --inplace

    # With OpenMP (multi-threaded):
    SPARSE_LU_OPENMP=1 python setup.py build_ext --inplace

    # With OpenMP and BLAS (vectorized axpy for dense rows):
    SPARSE_LU_OPENMP=1 SPARSE_LU_BLAS=1 python setup.py build_ext --inplace

    # Override BLAS library (e.g. MKL):
    SPARSE_LU_BLAS=1 BLAS_LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" \\
        python setup.py build_ext --inplace

Requirements
------------
    Python  >= 3.9
    NumPy   >= 1.22
    A C99-capable compiler (gcc >= 7, clang >= 5, MSVC 2019+)
    OpenMP  (optional; gcc -fopenmp, clang -fopenmp, MSVC /openmp)
    BLAS    (optional; libblas, OpenBLAS, MKL, etc., providing CBLAS interface)
"""

import os
import sys
import shlex
import numpy as np
from setuptools import setup, Extension

use_openmp = os.environ.get("SPARSE_LU_OPENMP", "0") == "1"
use_blas   = os.environ.get("SPARSE_LU_BLAS",   "0") == "1"

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/DNDEBUG"]
    extra_link_args    = []
    libraries          = []
    if use_openmp:
        extra_compile_args.append("/openmp")
    if use_blas:
        # On Windows, users typically link MKL or OpenBLAS via vcpkg etc.
        # Honour BLAS_LIBS if set, otherwise default to OpenBLAS.
        extra_compile_args.append("/DHAVE_BLAS")
        blas_libs = os.environ.get("BLAS_LIBS", "")
        if blas_libs:
            extra_link_args.extend(shlex.split(blas_libs))
        else:
            libraries.append("openblas")
else:
    extra_compile_args = ["-O3", "-march=native",
                          "-fno-math-errno", "-fno-signed-zeros",
                          "-DNDEBUG", "-std=c99"]
    extra_link_args    = []
    libraries          = []
    if use_openmp:
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
    if use_blas:
        extra_compile_args.append("-DHAVE_BLAS")
        # Allow caller to override the BLAS link line with arbitrary flags.
        # Otherwise prefer OpenBLAS (faster vectorised axpy on most modern
        # systems), then fall back to the reference BLAS shipped by Linux
        # distros.  We don't actually verify which is present at build time;
        # the linker will fail loudly if neither is installed.
        blas_libs = os.environ.get("BLAS_LIBS", "")
        if blas_libs:
            extra_link_args.extend(shlex.split(blas_libs))
        else:
            # the CBLAS interface and the BLAS behind it, the names conda and
            # the Linux distributions install (OpenBLAS or MKL underneath);
            # BLAS_LIBS="-lopenblas" for a bare OpenBLAS.
            libraries.extend(["cblas", "blas"])

ext = Extension(
    name="_sparse_lu_c",
    sources=["_sparse_lu_c.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
    language="c",
)

omp_note  = " (OpenMP enabled)" if use_openmp else \
            " (set SPARSE_LU_OPENMP=1 to enable OpenMP)"
blas_note = " (BLAS enabled)"   if use_blas   else \
            " (set SPARSE_LU_BLAS=1 to enable the BLAS kernels)"
print(f"Building _sparse_lu_c{omp_note}{blas_note}")

setup(
    name="sparse_lu_full_pivot",
    version="0.4.0",
    description="Sparse LU with full/partial/rook pivoting (C extension)",
    ext_modules=[ext],
    license="GPL-3.0-or-later",
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "scipy>=1.9"],
)
