"""
sparse_lu_full_pivot.py
=======================
Sparse LU decomposition with full (rook) pivoting, optional Ruiz
equilibration, and optional COLAMD fill-reducing column ordering.

Factorisation
-------------
Finds permutation vectors p, q and sparse triangular factors L, U such that

    A[p, :][:, q]  =  L @ U          (exact, or ≈ when drop_tol > 0)

Three optional preprocessing steps can be applied before factorisation,
each controlled by its own keyword argument:

  1. equilibrate=True  (default)
       Ruiz iterative L-inf equilibration: scales rows and columns to
       magnitude ≈ 1, improving pivot selection and condition number.

  2. colamd=False  (default)
       COLAMD fill-reducing column ordering: reorders columns to reduce
       fill-in in L and U.  Particularly effective for matrices with
       block-banded sparsity (e.g. ultraspherical spectral operators
       coupled by Coriolis or other physical terms).

  3. pin_dense_rows=False  (default, only meaningful when colamd=True)
       Tau row detection and pinning: identifies structurally dense rows
       (e.g. boundary condition tau rows in spectral methods), moves them
       to the bottom of the matrix before applying COLAMD so their density
       does not corrupt the fill-reducing ordering for the interior rows.
       Detection is automatic (configurable threshold) or the caller may
       supply an explicit boolean mask via the tau_row_mask parameter.

Permutation composition
-----------------------
All three preprocessing steps are transparent to the caller.  The returned
p, q, Dr, Dc are composed back into the coordinate system of the original
matrix A, so lu_solve() and lu_residual() work unchanged regardless of
which preprocessing options were used.

The composition rules are (derived in the module comments):
    p_final  = row_perm[p_lu]           row perm composition
    q_final  = col_perm[q_lu]           col perm composition
    Dr_final = Dr[inv_row_perm]         re-index row scaling to original rows
    Dc_final = Dc[inv_col_perm]         re-index col scaling to original cols

Backend
-------
When the C extension ``_sparse_lu_c`` has been built (via setup.py), it is
used for the equilibration, the factorisation and the triangular solves.
``pivot_strategy='partial'`` with ``drop_tol=0`` runs the supernodal
left-looking kernel (column by column: a pruned depth-first search over
the supernodes for the pattern, dense block updates with BLAS ``trsv`` /
``gemv`` on a dense accumulator, the largest modulus as pivot, columns
with nesting patterns amalgamated into one dense block); 'rook', 'full'
and any drop tolerance run the right-looking row kernel.  Falls back to
pure Python.

Build:
    python setup.py build_ext --inplace
    SPARSE_LU_OPENMP=1 python setup.py build_ext --inplace  # OpenMP

Public API
----------
    sparse_lu_full_pivot(A, *, drop_tol=0.0, equilibrate=True,
                         colamd=False, pin_dense_rows=False,
                         tau_density=0.5, tau_min_nnz=10,
                         tau_row_mask=None)
        -> L, U, p, q, Dr, Dc

    lu_solve(L, U, p, q, b, Dr=None, Dc=None)
        -> x

    lu_residual(A, L, U, p, q, Dr=None, Dc=None)
        -> float
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve_triangular
from typing import Tuple, Optional

_C_EXT_ERROR = None
try:
    from . import _sparse_lu_c
    _HAVE_C_EXT = True
except (ImportError, ValueError) as _e1:
    # standalone import of the module file
    try:
        import _sparse_lu_c
        _HAVE_C_EXT = True
    except ImportError as _e2:
        _HAVE_C_EXT = False
        _C_EXT_ERROR = f"{_e1}; {_e2}"
# The C extension must be the build of this source: a stale .so (an older in-tree build, or another checkout's module on
# sys.path) would otherwise run with this Python layer and silently route around the missing entry points (audit 2026-10-10).
_C_API_VERSION = 4   # 3: ILU drop-before-use and breakdown pivots; 4: solve plans (solve_plan, lu_solve_plan)
if _HAVE_C_EXT and getattr(_sparse_lu_c, "API_VERSION", 0) != _C_API_VERSION:
    _C_EXT_ERROR = (f"_sparse_lu_c at {getattr(_sparse_lu_c, '__file__', '?')} has API_VERSION "
                    f"{getattr(_sparse_lu_c, 'API_VERSION', 'none')}, this source needs {_C_API_VERSION} (a stale build: rebuild it)")
    _HAVE_C_EXT = False
if not _HAVE_C_EXT:
    # the pure-Python fallback is O(n^3) full pivoting: say so loudly (a mislinked extension, e.g. against the wrong
    # OpenMP runtime, otherwise drops a long factorisation into it without a word)
    import warnings as _w
    _w.warn("FPLUS C extension not loaded; using the pure-Python full-pivoting fallback, which is orders of magnitude "
            f"slower and ignores partial/rook pivoting.  Import error: {_C_EXT_ERROR}", RuntimeWarning, stacklevel=2)

# Pivot strategy enum (must match C-side FPLUS_PIVOT_* macros).
_PIVOT_FULL    = 0
_PIVOT_PARTIAL = 1
_PIVOT_ROOK    = 2
_PIVOT_NAMES = {
    "full":    _PIVOT_FULL,
    "partial": _PIVOT_PARTIAL,
    "rook":    _PIVOT_ROOK,
}

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
_RUIZ_MAX_ITER = 20
# FPLUS_PY_ORDERING=1 forces the Python ordering / permutation / C-kernel route even when the C pipeline
# (analyze, factor_sn_perm) is available -- for comparisons; the two routes give bit-identical factors.
import os as _os
_PY_PIPELINE = _os.environ.get("FPLUS_PY_ORDERING") == "1"
_RUIZ_TOL      = 1e-2


def _ruiz_iters(equilibrate):
    """equilibrate -> the number of Ruiz iterations: True = _RUIZ_MAX_ITER (to tolerance), False / 0 = none, an int k = at most k."""
    if equilibrate is None or isinstance(equilibrate, (bool, np.bool_)):
        return _RUIZ_MAX_ITER if equilibrate else 0
    k = int(equilibrate)
    if k < 0 or k != equilibrate:
        raise ValueError(f"equilibrate must be a bool or a non-negative int; got {equilibrate!r}")
    return k


# ===========================================================================
# Public API
# ===========================================================================

def sparse_lu_full_pivot(
    A: sp.spmatrix,
    *,
    drop_tol: float = 0.0,
    equilibrate: bool = True,
    colamd: bool = False,
    pin_dense_rows: bool = False,
    tau_density: float = 0.5,
    tau_min_nnz: int = 10,
    tau_row_mask: Optional[np.ndarray] = None,
    tau_col_mask: Optional[np.ndarray] = None,
    pivot_strategy: str = "partial",
    dtype=None,
    verbose: int = 0,
    relax_rows: int = 0,
    panel: int = 1,
    cache_ordering: bool = True,
    gpu: bool = False,
    gpu_min_flops: float = 5e7,
    pivot_rel: Optional[float] = None,
    postorder: bool = True,
    _return_csc: bool = False,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Sparse LU decomposition with full pivoting.

    Parameters
    ----------
    A : scipy sparse matrix, shape (n, n)
        General, possibly nonsymmetric, complex or real square matrix.

    drop_tol : float, optional
        Default 0: the exact factorisation.  ``drop_tol > 0``: an incomplete
        LU (threshold ILU, row kernel, any ``pivot_strategy``) for use as a
        preconditioner.  On the equilibrated, ordered matrix it drops, as they
        arise, every multiplier and updated / fill value with
        ``|v| <= drop_tol``, and when a row becomes a pivot row (row k of U)
        its off-pivot entries with ``|v| <= drop_tol`` before it updates the
        rows below; pivots are never dropped.  With Ruiz equilibration the
        entries are O(1), so ``drop_tol`` acts as a relative threshold.  If
        dropping leaves no pivot (the exact factorisation's trailing pivots are
        at or below ``drop_tol``), the pivot is replaced by
        ``drop_tol * max|a_ij|`` as in SuperLU's ILU, with a RuntimeWarning.

    equilibrate : bool or int, optional
        Apply Ruiz iterative L-inf equilibration before factorisation.
        Default True (iterations to tolerance, at most 20).  An int k runs
        at most k iterations (0 = none).  Partial pivoting on the scaled
        matrix picks different rows: on spectral pencils with tau rows the
        fully converged scaling raised nnz(L+U) by 8-117 percent against
        none or one iteration (2026-10-12), at equal backward error.

    colamd : bool, optional
        Apply COLAMD fill-reducing column ordering before factorisation.
        Default False.  Recommended for large sparse matrices with
        exploitable sparsity structure (block-banded, PDE operators, etc.).

    pin_dense_rows : bool, optional
        Only meaningful when colamd=True.  If True, detect structurally
        dense rows and/or columns (tau / boundary-condition lines in
        spectral methods) and pin them to the edges of the matrix before
        applying COLAMD.  Dense rows are moved to the bottom; dense
        columns are moved to the right.  This prevents dense lines from
        inflating degree counts in the COLAMD analysis, which would
        degrade the fill-reducing quality.  Default False.

    tau_density : float, optional
        Fraction of n above which a row or column is considered dense for
        tau detection.  Default 0.5 (must be > 50% filled).
        Only used when pin_dense_rows=True and the corresponding mask
        is None.

    tau_min_nnz : int, optional
        Minimum absolute nonzero count for tau detection.  A row/column
        is only flagged as dense if BOTH nnz > tau_density * n AND
        nnz > tau_min_nnz.  This prevents small matrices from
        triggering false positives.  Default 10.
        Only used when pin_dense_rows=True and the corresponding mask
        is None.

    tau_row_mask : ndarray of bool, shape (n,), optional
        Explicit boolean mask identifying tau (dense) rows.  If supplied,
        automatic row detection via tau_density/tau_min_nnz is skipped.
        Only used when pin_dense_rows=True.

    tau_col_mask : ndarray of bool, shape (n,), optional
        Explicit boolean mask identifying tau (dense) columns.  If
        supplied, automatic column detection via tau_density/tau_min_nnz
        is skipped.  Only used when pin_dense_rows=True.  In the
        ultraspherical spectral method, boundary conditions may appear
        as dense rows (classical tau) or dense columns depending on the
        pencil assembly convention.

    pivot_strategy : str, optional
        Pivoting strategy for the LU factorization.  Default 'partial'.

            'partial': At each step, scan column k for the maximum-magnitude
                       entry; row swap only.  PRESERVES COLAMD ORDERING.
                       Recommended for matrices with exploitable structure
                       (banded, block-banded, COLAMD-ordered).  Combined
                       with Ruiz equilibration this gives numerical
                       robustness comparable to full pivoting at a fraction
                       of the fill-in cost.  Runs on the supernodal
                       left-looking kernel (work proportional to the
                       flops, dense blocks through BLAS) when drop_tol
                       is 0.

            'rook':    Iteratively scan column k then row to convergence.
                       Pivot is the max in BOTH its row and column.
                       Stronger than partial pivoting; useful for
                       asymmetric matrices.  Mostly preserves structure
                       (column swaps occur only when needed).

            'full':    Scan the entire active submatrix M[k:,k:] for the
                       max-magnitude entry; both row AND column swaps.
                       Strongest numerical guarantee but DESTROYS COLAMD
                       ordering.  Use only when the matrix has no
                       exploitable structure or you specifically need
                       maximum robustness for severely ill-conditioned
                       problems.

    dtype : numpy dtype, optional
        Working precision.  Defaults to complex128/float64.

    verbose : int, optional
        Diagnostic verbosity level.  Default 0 = silent.
            0 : no output
            1 : entry parameters + preprocessing summary + final summary
                (no per-step output; safe for production runs)
            2 : adds periodic factorization progress (~20 progress lines)
                and per-iteration Ruiz residual

    relax_rows : int, optional
        Supernodal partial pivoting only.  A column joins the open
        supernode when its L pattern differs from the supernode's by at
        most this many rows; the difference is stored as explicit zeros
        (relaxed supernodes).  Default 0 = exact supernodes.  On complex
        spectral-MHD test pencils 2 merges ~93 percent of the near-misses
        (mean supernode width 1.9 -> 26 at n = 100k) for ~6 percent more
        fill.  A negative value selects the automatic rule: a column
        joins while the share of explicit zeros in the grown supernode
        stays below a bound that tightens with its width (any up to width
        4, 0.8 up to 16, 0.1 up to 48, 0.05 beyond; CHOLMOD's amalgamation
        defaults) and the mismatch is at most 32 rows.  A fixed count suits
        one kind of matrix only (2 on very sparse radial pencils, 32 on
        dense-row meridional ones); the automatic rule was within a few
        percent of the better fixed value on both, and faster than both on
        some (2026-10-12).

    gpu : bool, optional
        panel > 1, complex, BLAS build: run the zgemm of each panel update
        whose flop count is at least gpu_min_flops (default 5e7) on the GPU
        through cuBLAS, loaded at run time, on the host-resident blocks.
        Needs a device that reads pageable host memory (coherent CPU-GPU memory, e.g. NVLink-C2C); otherwise,
        or when CUDA is absent, the CPU path runs.  Default False.

    postorder : bool, optional
        With colamd=True: postorder the column elimination tree of the
        COLAMD order (as SuperLU's sp_preorder).  Same elimination tree and
        fill bound; each subtree's columns become contiguous, so the
        supernodal kernel finds wider supernodes.  Default True (False
        restores the plain COLAMD order of earlier versions).

    pivot_rel : float, optional
        Relative singular-pivot threshold: the factorization stops with
        "Matrix is singular" when the best available pivot is below
        pivot_rel times the column's largest modulus (equilibrated).
        Default None = sqrt(machine epsilon) ~ 1.5e-8, the LAPACK choice.
        Shift-invert pencils shifted close to an eigenvalue are nearly
        singular by design; there a smaller value (e.g. 1e-14) lets the
        factorization finish, and the backward error decides.

    cache_ordering : bool, optional
        Reuse the COLAMD ordering of an earlier call with the same sparsity
        pattern and pinning parameters (an LRU of 8; the ordering depends
        on nothing else, so this is exact).  Default True;
        clear_ordering_cache() empties it.

    panel : int, optional
        Supernodal partial pivoting, BLAS build only (real and complex).
        Columns are processed in panels of this width: each supernode
        complete before the panel is applied to every panel column it
        reaches at once (trsm + gemm), the within-panel updates per
        column.  Default 1 = one column at a time.  Same pivots as
        panel=1 (up to ties); 32 gave 3.2x at n = 100k with relax_rows=2.
        A negative value picks the width from the OpenMP thread count:
        64 x max(1, threads / 2), at most 512 (64 was within 2 percent of
        the best width at one thread for n 1.2k-14k; 512 is the 16-thread
        48 x 144 production setting).

    Returns
    -------
    L  : csr_matrix  — unit lower-triangular factor
    U  : csr_matrix  — upper-triangular factor
    p  : ndarray(int64)  — row permutation  (A[p,:][:,q] ≈ L @ U after scaling)
    q  : ndarray(int64)  — column permutation
    Dr : ndarray(float64) — row scaling (ones if equilibrate=False)
    Dc : ndarray(float64) — column scaling (ones if equilibrate=False)

    Raises
    ------
    ValueError   A is not square.
    RuntimeError A is singular (numerically or structurally).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}.")
    n = A.shape[0]

    is_complex = np.issubdtype(A.dtype, np.complexfloating)
    if dtype is None:
        dtype = np.complex128 if is_complex else np.float64

    # Validate pivot_strategy.
    if pivot_strategy not in _PIVOT_NAMES:
        raise ValueError(
            f"pivot_strategy must be 'partial', 'rook', or 'full'; "
            f"got {pivot_strategy!r}.")
    pivot_code = _PIVOT_NAMES[pivot_strategy]

    # ------------------------------------------------------------------
    # Verbose entry summary (verbose >= 1).  Prints control parameters
    # before any work so the user can verify the configuration on long runs.
    # ------------------------------------------------------------------
    if verbose >= 1:
        backend = "C extension" if _HAVE_C_EXT else "pure-Python fallback"
        print(f"[FPLUS] sparse_lu_full_pivot called: shape={A.shape}, "
              f"dtype={A.dtype}, nnz={A.nnz}, backend={backend}")
        print(f"[FPLUS]   pivot_strategy={pivot_strategy!r}, drop_tol={drop_tol}, "
              f"equilibrate={equilibrate}")
        print(f"[FPLUS]   colamd={colamd}, pin_dense_rows={pin_dense_rows}")
        if pin_dense_rows:
            row_msg = ("explicit mask" if tau_row_mask is not None
                       else f"auto-detect (density>{tau_density}, "
                            f"nnz>{tau_min_nnz})")
            col_msg = ("explicit mask" if tau_col_mask is not None
                       else f"auto-detect (density>{tau_density}, "
                            f"nnz>{tau_min_nnz})")
            print(f"[FPLUS]   tau row detection: {row_msg}")
            print(f"[FPLUS]   tau col detection: {col_msg}")

    # ------------------------------------------------------------------
    # Step 1: Compute static row/column permutations (COLAMD + tau pinning).
    # These operate on the sparsity PATTERN only and are applied before
    # equilibration so that Ruiz sees the reordered matrix.
    # ------------------------------------------------------------------
    # The ordering and the supernodal kernel work on canonical CSC (sorted, no duplicates): at most one conversion,
    # here, and none for CSC input.
    A_csc = _canonical_csc(A)
    if colamd:
        row_perm, col_perm, tr_count, tc_count = _colamd_permutations(
            A_csc, n, pin_dense_rows, tau_density, tau_min_nnz,
            tau_row_mask, tau_col_mask, cache=cache_ordering, postorder=postorder)
        if verbose >= 1:
            if pin_dense_rows:
                print(f"[FPLUS]   COLAMD: pinned {tr_count} dense rows, "
                      f"{tc_count} dense columns")
            else:
                print("[FPLUS]   COLAMD applied (no tau pinning)")
        use_c_pipeline = (_HAVE_C_EXT and hasattr(_sparse_lu_c, "factor_sn_perm") and not _PY_PIPELINE
                          and pivot_code == _PIVOT_PARTIAL and drop_tol == 0.0)
        A_work = None if use_c_pipeline else _permute_csc(A_csc, row_perm, col_perm)
    else:
        use_c_pipeline = False
        row_perm = np.arange(n, dtype=np.int64)
        col_perm = np.arange(n, dtype=np.int64)
        A_work   = A_csc

    # ------------------------------------------------------------------
    # Step 2: Numerical factorisation (with optional equilibration).
    # ------------------------------------------------------------------
    if use_c_pipeline:
        L, U, p_lu, q_lu, Dr, Dc = _lu_factor_c_perm(
            A_csc, row_perm, col_perm, n, equilibrate, is_complex, verbose, _return_csc, relax_rows, panel, gpu,
            gpu_min_flops, pivot_rel)
    elif _HAVE_C_EXT:
        L, U, p_lu, q_lu, Dr, Dc = _lu_factor_c(
            A_work, n, drop_tol, equilibrate, is_complex, dtype,
            verbose, pivot_code, _return_csc, relax_rows, panel, gpu, gpu_min_flops, pivot_rel)
    else:
        if pivot_strategy != "full":
            import warnings
            warnings.warn(
                "Pure-Python fallback only supports pivot_strategy='full'; "
                f"requested {pivot_strategy!r} ignored. Build the C extension "
                "to enable partial/rook pivoting.",
                stacklevel=2)
        L, U, p_lu, q_lu, Dr, Dc = _lu_factor_py(
            A_work, n, drop_tol, equilibrate, dtype)
        if verbose >= 1:
            print("[FPLUS]   (verbose >= 1 only fully supported with the "
                  "C extension; pure-Python fallback ran silently)")

    # ------------------------------------------------------------------
    # Step 3: Compose static and dynamic permutations back to original A.
    #
    # A_work = A[row_perm, :][:, col_perm]
    # LU factored A_work as: diag(Dr[p_lu]) @ A_work[p_lu,:][:,q_lu] @ diag(Dc[q_lu]) = LU
    #
    # Substituting A_work:
    #   A[row_perm[p_lu], :][:, col_perm[q_lu]] after scaling = LU
    #
    # So the permutations in the original coordinate system are:
    #   p_final  = row_perm[p_lu]
    #   q_final  = col_perm[q_lu]
    #
    # Dr[k] was calibrated on row k of A_work = row row_perm[k] of A.
    # lu_solve applies Dr[p_final] to b[p_final].
    # Dr[p_final[k]] = Dr[p_lu[k]], and the LU calibrated Dr on p_lu rows
    # of A_work.  But lu_residual applies Dr to the original A rows by
    # natural index.  We need Dr_final[row_perm[k]] = Dr[k], i.e.:
    #   Dr_final = Dr[inv_row_perm]
    # Similarly:
    #   Dc_final = Dc[inv_col_perm]
    # ------------------------------------------------------------------
    if colamd:
        inv_row_perm = np.empty(n, dtype=np.int64)
        inv_row_perm[row_perm] = np.arange(n, dtype=np.int64)
        inv_col_perm = np.empty(n, dtype=np.int64)
        inv_col_perm[col_perm] = np.arange(n, dtype=np.int64)

        p_final  = row_perm[p_lu]
        q_final  = col_perm[q_lu]
        Dr_final = Dr[inv_row_perm]
        Dc_final = Dc[inv_col_perm]
    else:
        p_final  = p_lu
        q_final  = q_lu
        Dr_final = Dr
        Dc_final = Dc

    return L, U, p_final, q_final, Dr_final, Dc_final


def lu_solve(
    L: sp.spmatrix,
    U: sp.spmatrix,
    p: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
    Dr: Optional[np.ndarray] = None,
    Dc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve A x = b given the factors from sparse_lu_full_pivot.

    Algorithm  (A[p,:][:,q] = L U after scaling by Dr, Dc):
        1. b_eq  = Dr[p] * b[p]
        2. y     = L^{-1} b_eq        (sparse forward solve)
        3. z     = U^{-1} y           (sparse back solve)
        4. x[q]  = Dc[q] * z

    Uses scipy.sparse.linalg.spsolve_triangular for the triangular solves,
    which operates directly on the sparse CSC factors without converting to
    dense.  This preserves the memory and computational advantages of the
    sparse factorisation — critical when L and U are large.

    The permutations p, q and scaling vectors Dr, Dc returned by
    sparse_lu_full_pivot are already composed into the original matrix
    coordinate system, so this function works unchanged regardless of
    which preprocessing options (colamd, pin_dense_rows, equilibrate)
    were used.

    Parameters
    ----------
    L, U : csr_matrix or csc_matrix
        Unit lower-triangular and upper-triangular factors.
    p, q : int64 permutation vectors
    b    : right-hand side, shape (n,) or (n, nrhs)
    Dr   : float64 row scaling (or None for unscaled)
    Dc   : float64 column scaling (or None for unscaled)
    """
    dtype  = np.result_type(L.dtype, b)
    b      = np.asarray(b, dtype=dtype)
    n      = b.shape[0]
    if b.ndim == 2:   # several right-hand sides: one at a time (the scalings broadcast along the rows)
        return np.stack([lu_solve(L, U, p, q, b[:, k], Dr, Dc) for k in range(b.shape[1])], axis=1) if b.shape[1] else b.copy()

    b_perm = b[p]
    if Dr is not None:
        b_perm = Dr[p] * b_perm

    L_csc = L.tocsc() if not isinstance(L, csc_matrix) else L
    U_csc = U.tocsc() if not isinstance(U, csc_matrix) else U

    z = _tri_solve(L_csc, U_csc, b_perm, 0)

    x = np.empty(n, dtype=z.dtype)
    x[q] = (Dc[q] * z) if Dc is not None else z
    return x


def _tri_solve(L_csc, U_csc, b, trans):
    """x = (L U)^-1 b (trans 0), (L U)^-T b (1) or (L U)^-H b (2) on CSC
    factors: the C sweeps when the extension is built, else
    scipy's spsolve_triangular."""
    if _HAVE_C_EXT and hasattr(_sparse_lu_c, "lu_solve"):
        return _tri_solve_c(_factor_arrays(L_csc), _factor_arrays(U_csc), b, trans)
    if trans == 0:
        y = spsolve_triangular(L_csc, b, lower=True, unit_diagonal=True)
        return spsolve_triangular(U_csc, y, lower=False, unit_diagonal=False)
    Lw, Uw = (L_csc.conjugate(), U_csc.conjugate()) if trans == 2 else (L_csc, U_csc)
    y = spsolve_triangular(Uw.T.tocsc(), b, lower=True, unit_diagonal=False)
    return spsolve_triangular(Lw.T.tocsc(), y, lower=False, unit_diagonal=True)


def _factor_arrays(F_csc):
    """(indptr, indices, data) of a CSC factor as int64 / contiguous arrays,
    the form the C sweeps take."""
    return (F_csc.indptr.astype(np.int64, copy=False),
            F_csc.indices.astype(np.int64, copy=False),
            np.ascontiguousarray(F_csc.data))


def _tri_solve_c(L_arrays, U_arrays, b, trans, plan=None):
    """The triangular sweeps in C; with a plan (_sparse_lu_c.solve_plan of these factors) the blocked, threaded solves,
    which also take b of shape (n, nrhs)."""
    Lp, Li, Lx = L_arrays
    Up, Ui, Ux = U_arrays
    is_complex = np.iscomplexobj(Lx) or np.iscomplexobj(b)
    dt = np.complex128 if is_complex else np.float64
    if plan is not None:
        return _sparse_lu_c.lu_solve_plan(
            plan, Lp, Li, np.ascontiguousarray(Lx, dtype=dt), Up, Ui, np.ascontiguousarray(Ux, dtype=dt),
            np.asarray(b, dtype=dt), int(trans), int(is_complex))
    return _sparse_lu_c.lu_solve(
        Lp, Li, np.ascontiguousarray(Lx, dtype=dt), Up, Ui, np.ascontiguousarray(Ux, dtype=dt),
        np.ascontiguousarray(b, dtype=dt), int(trans), int(is_complex))


def _rows_scaled(d, v):
    """d * v for a vector v, d[:, None] * v for the columns of a matrix v"""
    return d * v if v.ndim == 1 else d[:, None] * v


def lu_residual(
    A: sp.spmatrix,
    L: sp.spmatrix,
    U: sp.spmatrix,
    p: np.ndarray,
    q: np.ndarray,
    Dr: Optional[np.ndarray] = None,
    Dc: Optional[np.ndarray] = None,
) -> float:
    """
    Return  ||diag(Dr) A[p,:][:,q] diag(Dc) - L U||_F
            / ||diag(Dr) A[p,:][:,q] diag(Dc)||_F

    Dr and Dc are applied with natural indexing before permuting, which
    is consistent with how lu_solve interprets them.
    """
    A_arr = A.toarray()
    if Dr is not None:
        A_arr = Dr[:, None] * A_arr
    if Dc is not None:
        A_arr = A_arr * Dc[None, :]
    PAQ  = A_arr[np.ix_(p, q)]
    LU   = (L @ U).toarray()
    diff = np.linalg.norm(PAQ - LU, "fro")
    nrm  = np.linalg.norm(PAQ,      "fro")
    return float(diff / (nrm if nrm > 0.0 else 1.0))


# ===========================================================================
# SparseLU wrapper class — drop-in replacement for scipy's SuperLU object
# ===========================================================================

class SparseLU:
    """
    Drop-in replacement for the object returned by scipy.sparse.linalg.splu.

    Provides the same interface as scipy's ``SuperLU`` object so that code
    written for ``splu`` (shift-invert eigenvalue routines) can switch to
    full-pivot LU by replacing::

        lu = scipy.sparse.linalg.splu(A)

    with::

        lu = SparseLU(A)

    The factorisation uses full pivoting with optional Ruiz equilibration
    and COLAMD fill-reducing ordering, providing int64 index support and
    superior numerical stability for ill-conditioned shift-invert matrices.

    Parameters
    ----------
    A : scipy sparse matrix, shape (n, n)
        Matrix to factorise.
    **kwargs
        All keyword arguments are forwarded to ``sparse_lu_full_pivot``.
        Common options: ``equilibrate``, ``colamd``, ``pin_dense_rows``,
        ``tau_row_mask``, ``tau_col_mask``, ``drop_tol``, ``pivot_strategy``,
        ``verbose``.

    Attributes
    ----------
    L : csc_matrix — unit lower-triangular factor
    U : csc_matrix — upper-triangular factor
    perm_r : ndarray(int64) — row permutation
    perm_c : ndarray(int64) — column permutation
    shape : tuple — (n, n)
    nnz : int — total nonzeros in L + U

    Methods
    -------
    solve(b, trans='N')
        Solve A x = b (trans='N'), A^T x = b (trans='T'),
        or A^H x = b (trans='H').

    Examples
    --------
    >>> from spdee.solvers.fplus import SparseLU
    >>> lu = SparseLU(A, colamd=True, equilibrate=True)
    >>> x = lu.solve(b)
    """

    def __init__(self, A, **kwargs):
        A_sp = A if (sp.issparse(A) and A.format in ("csc", "csr")) else sp.csc_matrix(A)
        self._A = A_sp
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A_sp, _return_csc=True, **kwargs)
        self._L  = L.tocsc()
        self._U  = U.tocsc()
        self._L_arrays = _factor_arrays(self._L)
        self._U_arrays = _factor_arrays(self._U)
        # the blocked-solve plan (L supernodes, U row segments), once per factorization
        self._plan = (_sparse_lu_c.solve_plan(self._L_arrays[0], self._L_arrays[1], self._U_arrays[0], self._U_arrays[1])
                      if _HAVE_C_EXT and hasattr(_sparse_lu_c, "solve_plan") else None)
        self._p  = p
        self._q  = q
        self._Dr = Dr
        self._Dc = Dc
        self._n  = A_sp.shape[0]

    @property
    def L(self):
        """Unit lower-triangular factor (CSC)."""
        return self._L

    @property
    def U(self):
        """Upper-triangular factor (CSC)."""
        return self._U

    @property
    def perm_r(self):
        """Row permutation vector (int64)."""
        return self._p

    @property
    def perm_c(self):
        """Column permutation vector (int64)."""
        return self._q

    @property
    def shape(self):
        return (self._n, self._n)

    @property
    def nnz(self):
        """Total nonzeros in L and U."""
        return self._L.nnz + self._U.nnz

    def solve(self, b, trans='N'):
        """
        Solve a linear system using the stored factorisation.

        Parameters
        ----------
        b : ndarray, shape (n,) or (n, nrhs)
            Right-hand side vector or matrix.
        trans : str, optional
            'N' — solve A x = b   (default)
            'T' — solve A^T x = b
            'H' — solve A^H x = b  (conjugate transpose)

        Returns
        -------
        x : ndarray, same shape as b
        """
        b = np.asarray(b)
        if b.ndim == 1 or (b.ndim == 2 and self._plan is not None):
            return self._solve_single(b, trans)     # with a plan, all columns of b in one blocked solve
        else:
            # Handle multiple RHS columns
            return np.column_stack(
                [self._solve_single(b[:, j], trans)
                 for j in range(b.shape[1])])

    def _solve_single(self, b, trans):
        """Solve for a single RHS vector."""
        if trans == 'N':
            if _HAVE_C_EXT and hasattr(_sparse_lu_c, "lu_solve"):
                dtype = np.result_type(self._L.dtype, b)
                b = np.asarray(b, dtype=dtype)
                b_perm = b[self._p]
                if self._Dr is not None:
                    b_perm = _rows_scaled(self._Dr[self._p], b_perm)
                z = _tri_solve_c(self._L_arrays, self._U_arrays, b_perm, 0, self._plan)
                x = np.empty(z.shape, dtype=z.dtype)
                x[self._q] = _rows_scaled(self._Dc[self._q], z) if self._Dc is not None else z
                return x
            return lu_solve(self._L, self._U, self._p, self._q,
                            b, self._Dr, self._Dc)
        elif trans == 'T':
            return self._solve_transposed(b, conjugate=False)
        elif trans == 'H':
            return self._solve_transposed(b, conjugate=True)
        else:
            raise ValueError(
                f"trans must be 'N', 'T', or 'H'; got {trans!r}")

    def _solve_transposed(self, b, conjugate=False):
        """
        Solve A^T x = b  or  A^H x = b.

        From the factorisation identity:
            diag(Dr[p]) @ A[p,:][:,q] @ diag(Dc[q]) = L @ U

        Let A_pq = A[p,:][:,q], Dr_p = diag(Dr[p]), Dc_q = diag(Dc[q]).
        Then: A_pq = Dr_p^{-1} @ L @ U @ Dc_q^{-1}

        Transposing: A_pq^T = Dc_q^{-1} @ U^T @ L^T @ Dr_p^{-1}

        Since A_pq = P_p @ A @ P_q^T, we have A_pq^T = P_q @ A^T @ P_p^T.

        To solve A^T x = b, multiply by P_q:
            Dc_q^{-1} @ U^T @ L^T @ Dr_p^{-1} @ (P_p @ x) = P_q @ b

        Algorithm:
            1. rhs = Dc[q] * b[q]                   (apply P_q and Dc_q)
            2. y   = (U^T)^{-1} rhs                  (lower-triangular solve)
            3. w   = (L^T)^{-1} y                    (upper-triangular solve)
            4. z   = Dr[p] * w                       (undo Dr_p^{-1})
            5. x[p] = z, i.e. x = z[inv_p]           (undo P_p)

        For A^H: conjugate L and U, replace A^T with A^H throughout.
        """
        p, q = self._p, self._q
        Dr, Dc = self._Dr, self._Dc
        n = self._n

        dtype = np.result_type(self._L.dtype, b)
        b = np.asarray(b, dtype=dtype)

        # Steps 1-3: rhs = Dc[q] * b[q], then the transposed sweeps
        rhs = b[q].copy()
        if Dc is not None:
            rhs = _rows_scaled(Dc[q], rhs)
        if _HAVE_C_EXT and hasattr(_sparse_lu_c, "lu_solve"):
            w = _tri_solve_c(self._L_arrays, self._U_arrays, rhs, 2 if conjugate else 1, self._plan)
        else:
            w = _tri_solve(self._L, self._U, rhs, 2 if conjugate else 1)

        # Step 4: z = Dr[p] * w
        if Dr is not None:
            w = _rows_scaled(Dr[p], w)

        # Step 5: x = z[inv_p]
        inv_p = np.empty(n, dtype=np.int64)
        inv_p[p] = np.arange(n, dtype=np.int64)
        x = w[inv_p]

        return x

def _canonical_csc(A) -> csc_matrix:
    """A as CSC with sorted indices and no duplicates, copying only when it has to."""
    A = A if isinstance(A, csc_matrix) else csc_matrix(A)
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    return A


def _permute_csc(A_csc: csc_matrix, row_perm: np.ndarray, col_perm: np.ndarray) -> csc_matrix:
    """A[row_perm, :][:, col_perm] as canonical CSC: a column gather, then a row relabelling (skipped for the identity)."""
    B = A_csc[:, col_perm]
    if not np.array_equal(row_perm, np.arange(len(row_perm))):
        inv = np.empty(len(row_perm), dtype=np.int64)
        inv[row_perm] = np.arange(len(row_perm), dtype=np.int64)
        B.indices = inv[B.indices].astype(B.indices.dtype, copy=False)
        B.has_sorted_indices = False
        B.sort_indices()
    return B


def _detect_dense_lines(A_csr: csr_matrix, n: int,
                        tau_density: float, tau_min_nnz: int,
                        axis: int = 0) -> np.ndarray:
    """
    Identify structurally dense rows (axis=0) or columns (axis=1).

    A line is flagged as dense if:
        nnz_line > tau_density * n   AND   nnz_line > tau_min_nnz

    The AND condition prevents small matrices (where many lines exceed
    the fractional threshold) from being entirely classified as dense.

    Returns a boolean array of shape (n,).
    """
    by_row = isinstance(A_csr, csr_matrix)          # CSR or CSC: counts from indptr or from the indices, no conversion
    if axis == 0:
        nnz_per_line = np.diff(A_csr.indptr) if by_row else np.bincount(A_csr.indices, minlength=n)
    else:
        nnz_per_line = np.bincount(A_csr.indices, minlength=n) if by_row else np.diff(A_csr.indptr)
    threshold = max(float(tau_density) * n, float(tau_min_nnz))
    return nnz_per_line > threshold


def _colamd_ordering(A_csr: csr_matrix, n: int,
                     is_tau_row: np.ndarray,
                     is_tau_col: np.ndarray) -> np.ndarray:
    """
    Compute a COLAMD fill-reducing column permutation.

    Dense (tau) rows AND dense (tau) columns are zeroed out of the
    structural matrix before analysis so their density does not inflate
    degree counts and corrupt the fill-reducing ordering for the
    interior structure.

    Calls the SuiteSparse COLAMD library (colamd_l, int64 variant)
    directly via ctypes when libcolamd is available on the system.

    Falls back to scipy's splu(permc_spec='COLAMD') if libcolamd
    is not installed, and to the identity permutation if both fail.

    Returns col_perm (int64, length n): A[:, col_perm] is the
    column-reordered matrix.
    """
    # The structural pattern in CSC, with the dense rows and columns removed (as zeroing them and eliminating would).
    A_csc = _canonical_csc(A_csr)
    indptr = A_csc.indptr.astype(np.int64, copy=False)
    indices = A_csc.indices.astype(np.int64, copy=False)
    if is_tau_row.any() or is_tau_col.any():
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        keep = ~is_tau_row[indices] & ~is_tau_col[cols]
        indices = indices[keep]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(cols[keep], minlength=n))]).astype(np.int64)

    # --- Primary path: direct call to SuiteSparse colamd_l via ctypes ---
    perm = _colamd_ctypes(indptr, indices, n)
    if perm is not None:
        return perm

    # --- Fallback: scipy splu with permc_spec='COLAMD' ---
    A_struct = csc_matrix((np.ones(len(indices)), indices, indptr), shape=(n, n)).tocsr()
    return _colamd_splu_fallback(A_struct, n)


# COLAMD orderings by sparsity pattern: the ordering is a function of the pattern and the pinning parameters
# only, so a cached one is exact.  Repeated factorizations of one pattern (shift-invert at several shifts, a
# rebuilt Dedalus pencil) skip COLAMD, which took 19 s of 89 s at n = 100k on a complex test pencil.
_ORDER_CACHE: "OrderedDict" = None
import threading as _threading
_ORDER_CACHE_LOCK = _threading.Lock()
_ORDER_CACHE_MAX = 8


def clear_ordering_cache():
    """Forget every cached COLAMD ordering."""
    global _ORDER_CACHE
    with _ORDER_CACHE_LOCK:
        _ORDER_CACHE = None


def _pattern_key(A, n, pin_dense_rows, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask):
    import hashlib
    A_csr = _canonical_csc(A)        # (the name is historical: the key is taken on canonical CSC)
    h = hashlib.blake2b(digest_size=20)
    h.update(b"csc")
    h.update(np.ascontiguousarray(A_csr.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(A_csr.indices, dtype=np.int64).tobytes())
    for tag, m in ((b"row:", tau_row_mask), (b"col:", tau_col_mask)):   # tagged: the two masks cannot alias (audit 2026-10-10)
        h.update(tag + (b"none" if m is None else np.packbits(np.asarray(m, dtype=bool)).tobytes()))
    return (n, A_csr.nnz, bool(pin_dense_rows), float(tau_density), int(tau_min_nnz), h.hexdigest())


def _colamd_permutations(A, n, pin_dense_rows, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask, cache=True, postorder=False):
    """_colamd_permutations_uncached, cached by sparsity pattern (cache=True)."""
    global _ORDER_CACHE
    if not cache:
        return _colamd_permutations_uncached(A, n, pin_dense_rows, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask, postorder)
    from collections import OrderedDict
    key = (_pattern_key(A, n, pin_dense_rows, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask), bool(postorder))
    # the cache is shared by threads (the C core releases the GIL): look up and insert under the lock; the analysis itself runs
    # outside it (two threads missing the same key may both analyse; the second insert is a no-op refresh)
    with _ORDER_CACHE_LOCK:
        if _ORDER_CACHE is None:
            _ORDER_CACHE = OrderedDict()
        hit = _ORDER_CACHE.get(key)
        if hit is not None:
            _ORDER_CACHE.move_to_end(key)
    if hit is None:
        hit = _colamd_permutations_uncached(A, n, pin_dense_rows, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask, postorder)
        with _ORDER_CACHE_LOCK:
            if _ORDER_CACHE is None:
                _ORDER_CACHE = OrderedDict()
            _ORDER_CACHE[key] = hit
            _ORDER_CACHE.move_to_end(key)
            while len(_ORDER_CACHE) > _ORDER_CACHE_MAX:
                _ORDER_CACHE.popitem(last=False)
    return hit[0].copy(), hit[1].copy(), hit[2], hit[3]


def _colamd_permutations_uncached(
    A: sp.spmatrix,
    n: int,
    pin_dense_rows: bool,
    tau_density: float,
    tau_min_nnz: int,
    tau_row_mask: Optional[np.ndarray],
    tau_col_mask: Optional[np.ndarray],
    postorder: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the (row_perm, col_perm) pair for COLAMD preprocessing.

    row_perm : int64[n]   -- moves dense rows to the bottom (identity if
                            pin_dense_rows=False or no dense rows detected)
    col_perm : int64[n]   -- COLAMD fill-reducing column ordering computed
                            on the interior (non-dense) structure, with
                            dense columns appended at the right end

    Returns permutations such that A[row_perm, :][:, col_perm] is the
    matrix to be passed to the numerical factorisation.
    """
    A_csr = _canonical_csc(A)        # (the name is historical: CSC, as the rest of the ordering)
    masks = []
    for name, m in (("tau_row_mask", tau_row_mask), ("tau_col_mask", tau_col_mask)):
        if not pin_dense_rows:
            m = None      # ignored without pinning, on both routes (audit 2026-10-10)
        if pin_dense_rows and m is not None:
            m = np.asarray(m, dtype=bool)
            if m.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},); got {m.shape}.")
        masks.append(None if m is None else np.ascontiguousarray(np.asarray(m, dtype=bool), dtype=np.uint8))

    # -- The C ordering (fplus_analyze): the same steps as below, in C, with COLAMD linked at build time ----
    if _HAVE_C_EXT and hasattr(_sparse_lu_c, "analyze") and not _PY_PIPELINE:
        r = _sparse_lu_c.analyze(A_csr.indptr.astype(np.int64, copy=False), A_csr.indices.astype(np.int64, copy=False), n,
                                 int(bool(pin_dense_rows)), float(tau_density), int(tau_min_nnz), masks[0], masks[1],
                                 int(bool(postorder)))
        return r["row_perm"], r["col_perm"], int(r["n_dense_rows"]), int(r["n_dense_cols"])

    # -- Dense row detection ------------------------------------------------
    if pin_dense_rows:
        if tau_row_mask is not None:
            is_tau_row = np.asarray(tau_row_mask, dtype=bool)
            if is_tau_row.shape != (n,):
                raise ValueError(
                    f"tau_row_mask must have shape ({n},); "
                    f"got {is_tau_row.shape}.")
        else:
            is_tau_row = _detect_dense_lines(A_csr, n, tau_density,
                                             tau_min_nnz, axis=0)
    else:
        is_tau_row = np.zeros(n, dtype=bool)

    # -- Dense column detection ---------------------------------------------
    if pin_dense_rows:
        if tau_col_mask is not None:
            is_tau_col = np.asarray(tau_col_mask, dtype=bool)
            if is_tau_col.shape != (n,):
                raise ValueError(
                    f"tau_col_mask must have shape ({n},); "
                    f"got {is_tau_col.shape}.")
        else:
            is_tau_col = _detect_dense_lines(A_csr, n, tau_density,
                                             tau_min_nnz, axis=1)
    else:
        is_tau_col = np.zeros(n, dtype=bool)

    # -- Row permutation: interior rows first, dense rows last ---------------
    interior_rows = np.where(~is_tau_row)[0].astype(np.int64)
    dense_rows    = np.where( is_tau_row)[0].astype(np.int64)
    row_perm      = np.concatenate([interior_rows, dense_rows])

    # -- COLAMD column ordering (dense rows+cols excluded from analysis) -----
    colamd_perm = _colamd_ordering(A_csr, n, is_tau_row, is_tau_col)
    # -- Optional postorder of the column elimination tree (SuperLU's sp_preorder): same etree, same fill bound, but
    #    every subtree's columns contiguous, so the supernodal kernel can find wide supernodes.  On the interior
    #    (non-dense) rows and columns; dense columns stay pinned at the right end below.
    if postorder and _HAVE_C_EXT and hasattr(_sparse_lu_c, "coletree_postorder"):
        icols = colamd_perm[~is_tau_col[colamd_perm]]
        B = A_csr[:, icols]
        if is_tau_row.any():
            B = B[~is_tau_row, :]
        B = B.tocsc()
        post = _sparse_lu_c.coletree_postorder(B.indptr.astype(np.int64), B.indices.astype(np.int64), B.shape[0], B.shape[1])
        colamd_perm = np.concatenate([icols[post], colamd_perm[is_tau_col[colamd_perm]]]).astype(np.int64)

    # -- Pin dense columns to the right end of the column permutation -------
    if is_tau_col.any():
        interior_mask = ~is_tau_col[colamd_perm]
        col_perm = np.concatenate([
            colamd_perm[interior_mask],
            colamd_perm[~interior_mask],
        ])
    else:
        col_perm = colamd_perm

    return row_perm, col_perm, int(is_tau_row.sum()), int(is_tau_col.sum())



def _colamd_ctypes(indptr: np.ndarray, indices: np.ndarray, n: int) -> Optional[np.ndarray]:
    """
    Call colamd_l directly via ctypes.  Returns the column permutation
    as int64[n], or None if libcolamd is not available.

    colamd_l operates on a CSC matrix:
        A[]  — row indices (int64), must be allocated to at least Alen entries
        p[]  — column pointers (int64, length n_col+1)

    On output, p[0..n_col-1] contains the column permutation.
    A[] is destroyed (used as workspace).

    The knobs parameter controls dense row/column detection.  We use
    defaults, which ignore rows with more than 10*sqrt(n_col) entries.
    Since dense rows and columns have already been removed from A_struct
    before this function is called, the COLAMD-internal dense-line
    detection serves as a secondary safety net.
    """
    import ctypes
    try:
        lib = ctypes.CDLL("libcolamd.so.3")
    except OSError:
        try:
            lib = ctypes.CDLL("libcolamd.so")
        except OSError:
            return None

    c_i64 = ctypes.c_int64

    lib.colamd_l_recommended.restype = ctypes.c_size_t
    lib.colamd_l_recommended.argtypes = [c_i64, c_i64, c_i64]

    lib.colamd_l_set_defaults.restype = None
    lib.colamd_l_set_defaults.argtypes = [ctypes.POINTER(ctypes.c_double)]

    lib.colamd_l.restype = c_i64
    lib.colamd_l.argtypes = [c_i64, c_i64, c_i64,
                              ctypes.POINTER(c_i64), ctypes.POINTER(c_i64),
                              ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(c_i64)]

    n_row = n
    n_col = n
    nnz = int(indptr[n])

    Alen = lib.colamd_l_recommended(c_i64(nnz), c_i64(n_row), c_i64(n_col))
    if Alen == 0:
        return None

    A_work = np.zeros(Alen, dtype=np.int64)
    A_work[:nnz] = indices[:nnz]
    col_ptrs = np.array(indptr, dtype=np.int64)

    knobs = (ctypes.c_double * 20)()
    lib.colamd_l_set_defaults(knobs)

    stats = (c_i64 * 20)()

    result = lib.colamd_l(c_i64(n_row), c_i64(n_col), c_i64(Alen),
                           A_work.ctypes.data_as(ctypes.POINTER(c_i64)),
                           col_ptrs.ctypes.data_as(ctypes.POINTER(c_i64)),
                           knobs, stats)

    if result != 1:
        return None

    return col_ptrs[:n_col].copy().astype(np.int64)


def _colamd_splu_fallback(A_struct: csr_matrix, n: int) -> np.ndarray:
    """
    Fallback: extract COLAMD ordering via scipy's splu.

    This routes through SuperLU which uses int32 internally, so it will
    fail for matrices with more than ~2 billion structural nonzeros.
    Used only when libcolamd.so is not installed on the system.
    """
    try:
        import scipy.sparse.linalg as spla
    except ImportError:
        return np.arange(n, dtype=np.int64)

    A_shifted = (A_struct + sp.eye(n, format="csr", dtype=np.float64) * n).tocsc()

    try:
        lu_struct = spla.splu(A_shifted, permc_spec="COLAMD")
        return np.argsort(lu_struct.perm_c).astype(np.int64)   # perm_c[i] = new position of column i; the order is its inverse (audit 2026-10-10)
    except Exception:
        return np.arange(n, dtype=np.int64)


# ===========================================================================
# C-extension factorisation path
# ===========================================================================

def _ruiz_c(A, n, is_complex, target_dtype, verbose=0, max_iter=None):
    """Ruiz on CSR or CSC arrays (the C routine's CSC variant gives the same Dr, Dc and scaled entries bit for bit)."""
    indices = A.indices.astype(np.int64, copy=False)
    indptr  = A.indptr.astype(np.int64, copy=False)
    data    = np.ascontiguousarray(A.data, dtype=target_dtype)
    raw = _sparse_lu_c.ruiz_equilibrate(
        data, indices, indptr,
        n, _RUIZ_MAX_ITER if max_iter is None else int(max_iter), _RUIZ_TOL, int(is_complex), int(verbose), int(isinstance(A, csc_matrix))
    )
    return raw["Dr"], raw["Dc"], raw["data"]


def _lu_factor_c_perm(A_csc, row_perm, col_perm, n, equilibrate, is_complex, verbose=0, return_csc=False, relax_rows=0,
                      panel=1, gpu=False, gpu_min_flops=5e7, pivot_rel=None):
    """The C pipeline (fplus_factor_sn_perm): A[row_perm, :][:, col_perm], Ruiz on CSC, the supernodal kernel -- the same
    factors as _permute_csc followed by _lu_factor_c, without the Python copies."""
    pivot_rel = float(np.sqrt(np.finfo(np.float64).eps)) if pivot_rel is None else float(pivot_rel)
    target_dtype = np.complex128 if is_complex else np.float64
    A_c = _canonical_csc(A_csc)
    data = np.ascontiguousarray(A_c.data, dtype=target_dtype)
    raw = _sparse_lu_c.factor_sn_perm(
        data, A_c.indices.astype(np.int64, copy=False), A_c.indptr.astype(np.int64, copy=False), n,
        np.ascontiguousarray(row_perm, dtype=np.int64), np.ascontiguousarray(col_perm, dtype=np.int64), int(is_complex),
        int(_ruiz_iters(equilibrate) > 0), max(_ruiz_iters(equilibrate), 1), _RUIZ_TOL, int(verbose), int(relax_rows), int(panel), int(bool(gpu)),
        float(gpu_min_flops), pivot_rel)
    fmt = csc_matrix if return_csc else (lambda *a, **k: csc_matrix(*a, **k).tocsr())   # noqa: E731
    L = fmt((raw["L_data"], raw["L_indices"], raw["L_indptr"]), shape=(n, n), dtype=target_dtype)
    U = fmt((raw["U_data"], raw["U_indices"], raw["U_indptr"]), shape=(n, n), dtype=target_dtype)
    return L, U, raw["p"], raw["q"], raw["Dr"], raw["Dc"]


def _lu_factor_c(A, n, drop_tol, equilibrate, is_complex, dtype,
                 verbose=0, pivot_code=_PIVOT_PARTIAL, return_csc=False, relax_rows=0, panel=1, gpu=False, gpu_min_flops=5e7,
                 pivot_rel=None):
    pivot_rel = float(np.sqrt(np.finfo(np.float64).eps)) if pivot_rel is None else float(pivot_rel)
    target_dtype = np.complex128 if is_complex else np.float64
    A_csc = _canonical_csc(A)
    if A_csc.dtype != target_dtype:
        A_csc = A_csc.astype(target_dtype)

    indices = A_csc.indices.astype(np.int64, copy=False)
    indptr  = A_csc.indptr.astype(np.int64, copy=False)

    if _ruiz_iters(equilibrate) > 0:
        Dr, Dc, data = _ruiz_c(A_csc, n, is_complex, target_dtype, verbose, _ruiz_iters(equilibrate))
    else:
        Dr   = np.ones(n, dtype=np.float64)
        Dc   = np.ones(n, dtype=np.float64)
        data = np.ascontiguousarray(A_csc.data, dtype=target_dtype)

    if pivot_code == _PIVOT_PARTIAL and drop_tol == 0.0 and hasattr(_sparse_lu_c, "lu_factor_sn"):
        # the left-looking kernel takes CSC: the arrays as they are
        raw = _sparse_lu_c.lu_factor_sn(
            data, indices, indptr,
            n, int(is_complex), int(verbose), int(relax_rows), int(panel), int(bool(gpu)), float(gpu_min_flops), pivot_rel)
        # the kernel returns CSC; SparseLU keeps it (return_csc), the public function returns CSR
        fmt = csc_matrix if return_csc else (lambda *a, **k: csc_matrix(*a, **k).tocsr())   # noqa: E731
    else:
        # the row-oriented kernel takes CSR
        A_csr = csc_matrix((data, indices, indptr), shape=(n, n)).tocsr()
        raw = _sparse_lu_c.lu_factor(
            np.ascontiguousarray(A_csr.data, dtype=target_dtype),
            A_csr.indices.astype(np.int64, copy=False),
            A_csr.indptr.astype(np.int64, copy=False),
            n, float(drop_tol), int(is_complex),
            int(verbose), int(pivot_code), pivot_rel
        )
        fmt = csr_matrix
        if raw.get("n_pivots_replaced", 0) > 0:
            import warnings
            warnings.warn(
                f"ILU breakdown: {raw['n_pivots_replaced']} zero or negligible pivots replaced by drop_tol * max|a_ij| "
                f"(the exact factorization's trailing pivots are at or below drop_tol={drop_tol:g}); the factors are a "
                "preconditioner of reduced quality", RuntimeWarning, stacklevel=3)

    L = fmt(
        (raw["L_data"], raw["L_indices"], raw["L_indptr"]),
        shape=(n, n), dtype=target_dtype
    )
    U = fmt(
        (raw["U_data"], raw["U_indices"], raw["U_indptr"]),
        shape=(n, n), dtype=target_dtype
    )
    return L, U, raw["p"], raw["q"], Dr, Dc


# ===========================================================================
# Pure-Python fallback factorisation
# ===========================================================================

def _ruiz_py(A_csr, n, target_dtype, max_iter=None):
    """Pure-Python Ruiz equilibration using vectorised numpy."""
    indptr  = np.asarray(A_csr.indptr,  dtype=np.int64)
    indices = np.asarray(A_csr.indices, dtype=np.int64)
    data    = A_csr.data.astype(target_dtype, copy=True)

    Dr = np.ones(n, dtype=np.float64)
    Dc = np.ones(n, dtype=np.float64)

    row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    nz_rows = np.diff(indptr) > 0

    for _ in range(_RUIZ_MAX_ITER if max_iter is None else max_iter):
        abs_data = np.abs(data)

        row_max = np.zeros(n, dtype=np.float64)
        if nz_rows.any():
            rm = np.maximum.reduceat(abs_data, indptr[:-1])
            row_max = np.where(nz_rows, rm, 0.0)

        if np.any(row_max == 0.0):
            bad = int(np.argmax(row_max == 0.0))
            raise RuntimeError(
                f"Row {bad} is entirely zero (structural singularity).")

        dr = 1.0 / np.sqrt(row_max)
        Dr *= dr
        data *= dr[row_idx]

        abs_data = np.abs(data)

        col_max = np.zeros(n, dtype=np.float64)
        np.maximum.at(col_max, indices, abs_data)

        if np.any(col_max == 0.0):
            bad = int(np.argmax(col_max == 0.0))
            raise RuntimeError(
                f"Column {bad} is entirely zero (structural singularity).")

        dc = 1.0 / np.sqrt(col_max)
        Dc *= dc
        data *= dc[indices]

        abs_data = np.abs(data)
        row_max2 = np.zeros(n, dtype=np.float64)
        if nz_rows.any():
            row_max2 = np.where(nz_rows,
                                np.maximum.reduceat(abs_data, indptr[:-1]),
                                1.0)
        col_max2 = np.zeros(n, dtype=np.float64)
        np.maximum.at(col_max2, indices, abs_data)
        row_max2 = np.maximum(row_max2, 1e-300)
        col_max2 = np.maximum(col_max2, 1e-300)
        res = max(float(np.max(np.abs(np.log(row_max2)))),
                  float(np.max(np.abs(np.log(col_max2)))))
        if res < _RUIZ_TOL:
            break

    return Dr, Dc, data


def _lu_factor_py(A, n, drop_tol, equilibrate, dtype):
    """Pure-Python full-pivot LU with optional equilibration."""
    is_complex   = np.issubdtype(dtype, np.complexfloating)
    target_dtype = np.complex128 if is_complex else np.float64

    A_csr = A.astype(target_dtype).tocsr()
    A_csr.sort_indices()

    if _ruiz_iters(equilibrate) > 0:
        Dr, Dc, eq_data = _ruiz_py(A_csr, n, target_dtype, _ruiz_iters(equilibrate))
        A_work = csr_matrix(
            (eq_data, A_csr.indices.copy(), A_csr.indptr.copy()),
            shape=(n, n)
        )
    else:
        Dr     = np.ones(n, dtype=np.float64)
        Dc     = np.ones(n, dtype=np.float64)
        A_work = A_csr

    M  = [{} for _ in range(n)]
    Lm = [{} for _ in range(n)]
    for i in range(n):
        for ptr in range(A_work.indptr[i], A_work.indptr[i+1]):
            M[i][int(A_work.indices[ptr])] = A_work.data[ptr]

    p = np.arange(n, dtype=np.int64)
    q = np.arange(n, dtype=np.int64)

    _eps       = np.finfo(target_dtype).eps
    _pivot_tol = np.sqrt(_eps)
    _max_entry = 0.0
    n_replaced = 0

    for k in range(n):
        best_val = -1.0; pi = -1; pj = -1
        for i in range(k, n):
            for j, v in M[i].items():
                if j < k: continue
                av = abs(v)
                if av > best_val:
                    best_val = av; pi = i; pj = j
        if k == 0:
            _max_entry = max(best_val, 0.0)
        if (pi == -1 or best_val < _pivot_tol * _max_entry or best_val == 0.0) and drop_tol > 0.0 and drop_tol * _max_entry > 0.0:
            # ILU breakdown (as the C kernel): replace the pivot at (k, k) by drop_tol * max|a_ij|, no swap
            M[k][k] = target_dtype(drop_tol * _max_entry); pi = k; pj = k; best_val = drop_tol * _max_entry
            n_replaced += 1
        if pi == -1 or best_val == 0.0:
            raise RuntimeError(
                f"Zero pivot at step {k}. Matrix is singular.")
        if best_val < _pivot_tol * _max_entry:
            raise RuntimeError(
                f"Near-zero pivot {best_val:.2e} at step {k} "
                f"(threshold {_pivot_tol * _max_entry:.2e}). "
                f"Matrix is singular or very ill-conditioned.")

        if pi != k:
            M[k],  M[pi]  = M[pi],  M[k]
            Lm[k], Lm[pi] = Lm[pi], Lm[k]
            p[k],  p[pi]  = p[pi],  p[k]

        if pj != k:
            for i in range(n):
                v1 = M[i].pop(k,  None)
                v2 = M[i].pop(pj, None)
                if v1 is not None: M[i][pj] = v1
                if v2 is not None: M[i][k]  = v2
            q[k], q[pj] = q[pj], q[k]

        pivot_val = M[k].get(k, target_dtype(0))
        if pivot_val == 0.0:
            raise RuntimeError(
                f"Zero pivot at step {k}. Matrix is singular.")
        if drop_tol > 0.0:
            # ILU: row k is final (row k of U); drop its small off-pivot entries before it updates the rows below
            M[k] = {j: v for j, v in M[k].items() if j == k or abs(v) > drop_tol}

        pk_items = list(M[k].items())
        for i in range(k+1, n):
            Mik = M[i].get(k)
            if Mik is None: continue
            m = Mik / pivot_val
            if drop_tol > 0.0 and abs(m) <= drop_tol:
                del M[i][k]; continue
            Lm[i][k] = m
            for j, vk in pk_items:
                vi    = M[i].get(j, target_dtype(0))
                new_v = vi - m * vk
                if abs(new_v) == 0.0:
                    M[i].pop(j, None)
                elif drop_tol > 0.0 and abs(new_v) <= drop_tol:
                    M[i].pop(j, None)
                else:
                    M[i][j] = new_v
            M[i].pop(k, None)

    def _dod_to_csr(rows, n, upper):
        ip = np.zeros(n+1, dtype=np.int64)
        entries = []
        for i in range(n):
            row_entries = []
            if not upper:
                row_entries.append((i, target_dtype(1)))
            for j, v in rows[i].items():
                if drop_tol > 0.0 and j != i and abs(v) <= drop_tol: continue   # never drop the diagonal (as the C kernel)
                if upper     and j >= i: row_entries.append((j, v))
                if not upper and j <  i: row_entries.append((j, v))
            row_entries.sort(key=lambda x: x[0])
            entries.extend(row_entries)
            ip[i+1] = len(entries)
        if entries:
            c = np.array([e[0] for e in entries], dtype=np.int64)
            v = np.array([e[1] for e in entries], dtype=target_dtype)
        else:
            c = np.empty(0, dtype=np.int64)
            v = np.empty(0, dtype=target_dtype)
        return csr_matrix((v, c, ip), shape=(n, n))

    L = _dod_to_csr(Lm, n, False)
    U = _dod_to_csr(M,  n, True)
    if n_replaced > 0:
        import warnings
        warnings.warn(
            f"ILU breakdown: {n_replaced} zero or negligible pivots replaced by drop_tol * max|a_ij| "
            f"(the exact factorization's trailing pivots are at or below drop_tol={drop_tol:g}); the factors are a "
            "preconditioner of reduced quality", RuntimeWarning, stacklevel=3)
    return L, U, p, q, Dr, Dc
