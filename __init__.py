"""
FPLUS: Full-Pivot LU with Scaling.

A sparse LU decomposition library with full (rook) pivoting, Ruiz
equilibration, and direct-COLAMD fill-reducing ordering.  Designed for
ill-conditioned shift-invert eigenvalue problems where SuperLU's threshold
partial pivoting discards physically important small entries (e.g.
scale-separated PDE eigenproblems with rotation/Lorentz force coupling).

Key features
------------
* Partial pivoting (the default and the production path) on a supernodal
  left-looking kernel with symmetric pruning: work proportional to the
  flops of the factorisation, the dense block updates through BLAS,
  triangular solves in C.
* Full (rook) pivoting: at each elimination step, the globally largest
  entry in the active submatrix is selected as the pivot.  Guarantees
  multipliers |m| <= 1 and a bounded growth factor, on the right-looking
  row kernel, which is slower by an order of magnitude; the strategies for
  matrices on which partial pivoting fails.
* int64 indices throughout: no 2^31 ceiling on total nonzeros.
* Ruiz L-infinity equilibration: handles scale-separated physics by
  normalizing all row and column max-norms to ~1 before pivoting.
* Direct COLAMD via SuiteSparse's libcolamd (no SuperLU dependency).
* Tau row/column pinning: dense boundary-condition lines moved to edges
  before COLAMD analysis.
* OpenMP-parallelized C extension (build with SPARSE_LU_OPENMP=1).
* Drop-in replacement interface for scipy.sparse.linalg.splu via SparseLU.

Public API
----------
SparseLU(A, **kwargs) class:
    Drop-in replacement for scipy.sparse.linalg.splu.  Same .solve(b, trans),
    .L, .U, .perm_r, .perm_c, .shape, .nnz interface.

sparse_lu_full_pivot(A, **kwargs):
    Lower-level functional interface returning (L, U, p, q, Dr, Dc).

lu_solve(L, U, p, q, b, Dr, Dc):
    Solve A x = b given the factors.

lu_residual(A, L, U, p, q, Dr, Dc):
    Relative residual of the factorization.

Build
-----
From this directory (``spdee/solvers/build_ext.sh`` runs both):

    python setup.py build_ext --inplace                    # serial
    SPARSE_LU_OPENMP=1 python setup.py build_ext --inplace # OpenMP

A pure-Python fallback is used automatically if the C extension is missing.
``spdee.solvers.shift_invert(..., factor='fplus')`` is the caller.
"""

from .sparse_lu_full_pivot import (
    SparseLU,
    sparse_lu_full_pivot,
    lu_solve,
    lu_residual,
    _HAVE_C_EXT,
)

__all__ = [
    "SparseLU",
    "sparse_lu_full_pivot",
    "lu_solve",
    "lu_residual",
    "_HAVE_C_EXT",
]
