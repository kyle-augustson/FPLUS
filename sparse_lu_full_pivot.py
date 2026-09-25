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

try:
    from . import _sparse_lu_c
    _HAVE_C_EXT = True
except (ImportError, ValueError):
    # standalone import of the module file
    try:
        import _sparse_lu_c
        _HAVE_C_EXT = True
    except ImportError:
        _HAVE_C_EXT = False

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
_RUIZ_TOL      = 1e-2


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
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Sparse LU decomposition with full pivoting.

    Parameters
    ----------
    A : scipy sparse matrix, shape (n, n)
        General, possibly nonsymmetric, complex or real square matrix.

    drop_tol : float, optional
        ILU drop tolerance.  Default 0 = exact factorisation.

    equilibrate : bool, optional
        Apply Ruiz iterative L-inf equilibration before factorisation.
        Default True.  Disable only if A is already well-scaled.

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
    if colamd:
        row_perm, col_perm = _colamd_permutations(
            A, n, pin_dense_rows, tau_density, tau_min_nnz,
            tau_row_mask, tau_col_mask)
        if verbose >= 1:
            if pin_dense_rows:
                A_csr_v = A.tocsr() if not isinstance(A, csr_matrix) else A
                if tau_row_mask is not None:
                    tr_count = int(np.asarray(tau_row_mask, dtype=bool).sum())
                else:
                    tr_count = int(_detect_dense_lines(
                        A_csr_v, n, tau_density, tau_min_nnz, axis=0).sum())
                if tau_col_mask is not None:
                    tc_count = int(np.asarray(tau_col_mask, dtype=bool).sum())
                else:
                    tc_count = int(_detect_dense_lines(
                        A_csr_v, n, tau_density, tau_min_nnz, axis=1).sum())
                print(f"[FPLUS]   COLAMD: pinned {tr_count} dense rows, "
                      f"{tc_count} dense columns")
            else:
                print("[FPLUS]   COLAMD applied (no tau pinning)")
        A_work = A.tocsr()[row_perm, :][:, col_perm].tocsr()
    else:
        row_perm = np.arange(n, dtype=np.int64)
        col_perm = np.arange(n, dtype=np.int64)
        A_work   = A

    # ------------------------------------------------------------------
    # Step 2: Numerical factorisation (with optional equilibration).
    # ------------------------------------------------------------------
    if _HAVE_C_EXT:
        L, U, p_lu, q_lu, Dr, Dc = _lu_factor_c(
            A_work, n, drop_tol, equilibrate, is_complex, dtype,
            verbose, pivot_code)
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


def _tri_solve_c(L_arrays, U_arrays, b, trans):
    Lp, Li, Lx = L_arrays
    Up, Ui, Ux = U_arrays
    is_complex = np.iscomplexobj(Lx) or np.iscomplexobj(b)
    dt = np.complex128 if is_complex else np.float64
    return _sparse_lu_c.lu_solve(
        Lp, Li, np.ascontiguousarray(Lx, dtype=dt), Up, Ui, np.ascontiguousarray(Ux, dtype=dt),
        np.ascontiguousarray(b, dtype=dt), int(trans), int(is_complex))


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
        A_sp = sp.csr_matrix(A)
        self._A = A_sp
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A_sp, **kwargs)
        self._L  = L.tocsc()
        self._U  = U.tocsc()
        self._L_arrays = _factor_arrays(self._L)
        self._U_arrays = _factor_arrays(self._U)
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
        if b.ndim == 1:
            return self._solve_single(b, trans)
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
                    b_perm = self._Dr[self._p] * b_perm
                z = _tri_solve_c(self._L_arrays, self._U_arrays, b_perm, 0)
                x = np.empty(self._n, dtype=z.dtype)
                x[self._q] = (self._Dc[self._q] * z) if self._Dc is not None else z
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
            rhs = Dc[q] * rhs
        if _HAVE_C_EXT and hasattr(_sparse_lu_c, "lu_solve"):
            w = _tri_solve_c(self._L_arrays, self._U_arrays, rhs, 2 if conjugate else 1)
        else:
            w = _tri_solve(self._L, self._U, rhs, 2 if conjugate else 1)

        # Step 4: z = Dr[p] * w
        if Dr is not None:
            w = Dr[p] * w

        # Step 5: x = z[inv_p]
        inv_p = np.empty(n, dtype=np.int64)
        inv_p[p] = np.arange(n, dtype=np.int64)
        x = w[inv_p]

        return x

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
    if axis == 0:
        nnz_per_line = np.diff(A_csr.indptr)
    else:
        A_csc = A_csr.tocsc()
        nnz_per_line = np.diff(A_csc.indptr)
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
    # Build a structural (pattern-only) CSR matrix with dense lines zeroed.
    A_struct = A_csr.copy().tocsr()
    A_struct.data = np.ones(A_struct.nnz, dtype=np.float64)

    modified = False
    if is_tau_row.any():
        for i in np.where(is_tau_row)[0]:
            A_struct.data[A_struct.indptr[i]:A_struct.indptr[i+1]] = 0.0
        modified = True

    if is_tau_col.any():
        A_csc_struct = A_struct.tocsc()
        for j in np.where(is_tau_col)[0]:
            A_csc_struct.data[A_csc_struct.indptr[j]:A_csc_struct.indptr[j+1]] = 0.0
        A_struct = A_csc_struct.tocsr()
        modified = True

    if modified:
        A_struct.eliminate_zeros()

    # --- Primary path: direct call to SuiteSparse colamd_l via ctypes ---
    perm = _colamd_ctypes(A_struct, n)
    if perm is not None:
        return perm

    # --- Fallback: scipy splu with permc_spec='COLAMD' ---
    return _colamd_splu_fallback(A_struct, n)


def _colamd_permutations(
    A: sp.spmatrix,
    n: int,
    pin_dense_rows: bool,
    tau_density: float,
    tau_min_nnz: int,
    tau_row_mask: Optional[np.ndarray],
    tau_col_mask: Optional[np.ndarray],
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
    A_csr = A.tocsr()

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

    # -- Pin dense columns to the right end of the column permutation -------
    if is_tau_col.any():
        interior_mask = ~is_tau_col[colamd_perm]
        col_perm = np.concatenate([
            colamd_perm[interior_mask],
            colamd_perm[~interior_mask],
        ])
    else:
        col_perm = colamd_perm

    return row_perm, col_perm



def _colamd_ctypes(A_struct: csr_matrix, n: int) -> Optional[np.ndarray]:
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

    A_csc = A_struct.tocsc()
    n_row = n
    n_col = n
    nnz = A_csc.nnz

    Alen = lib.colamd_l_recommended(c_i64(nnz), c_i64(n_row), c_i64(n_col))
    if Alen == 0:
        return None

    A_work = np.zeros(Alen, dtype=np.int64)
    A_work[:nnz] = A_csc.indices.astype(np.int64)
    col_ptrs = A_csc.indptr.astype(np.int64).copy()

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
        return lu_struct.perm_c.astype(np.int64)
    except Exception:
        return np.arange(n, dtype=np.int64)


# ===========================================================================
# C-extension factorisation path
# ===========================================================================

def _ruiz_c(A_csr, n, is_complex, target_dtype, verbose=0):
    indices = A_csr.indices.astype(np.int64, copy=False)
    indptr  = A_csr.indptr.astype(np.int64, copy=False)
    data    = np.ascontiguousarray(A_csr.data, dtype=target_dtype)
    raw = _sparse_lu_c.ruiz_equilibrate(
        data, indices, indptr,
        n, _RUIZ_MAX_ITER, _RUIZ_TOL, int(is_complex), int(verbose)
    )
    return raw["Dr"], raw["Dc"], raw["data"]


def _lu_factor_c(A, n, drop_tol, equilibrate, is_complex, dtype,
                 verbose=0, pivot_code=_PIVOT_PARTIAL):
    target_dtype = np.complex128 if is_complex else np.float64
    A_csr = A.astype(target_dtype).tocsr()
    A_csr.sort_indices()

    indices = A_csr.indices.astype(np.int64, copy=False)
    indptr  = A_csr.indptr.astype(np.int64, copy=False)

    if equilibrate:
        Dr, Dc, data = _ruiz_c(A_csr, n, is_complex, target_dtype, verbose)
    else:
        Dr   = np.ones(n, dtype=np.float64)
        Dc   = np.ones(n, dtype=np.float64)
        data = np.ascontiguousarray(A_csr.data, dtype=target_dtype)

    if pivot_code == _PIVOT_PARTIAL and drop_tol == 0.0 and hasattr(_sparse_lu_c, "lu_factor_sn"):
        # the left-looking kernel takes CSC
        A_csc = csr_matrix((data, indices, indptr), shape=(n, n)).tocsc()
        raw = _sparse_lu_c.lu_factor_sn(
            np.ascontiguousarray(A_csc.data, dtype=target_dtype),
            A_csc.indices.astype(np.int64, copy=False),
            A_csc.indptr.astype(np.int64, copy=False),
            n, int(is_complex), int(verbose))
        fmt = lambda *a, **k: csc_matrix(*a, **k).tocsr()   # noqa: E731
    else:
        raw = _sparse_lu_c.lu_factor(
            data, indices, indptr, n, float(drop_tol), int(is_complex),
            int(verbose), int(pivot_code)
        )
        fmt = csr_matrix

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

def _ruiz_py(A_csr, n, target_dtype):
    """Pure-Python Ruiz equilibration using vectorised numpy."""
    indptr  = np.asarray(A_csr.indptr,  dtype=np.int64)
    indices = np.asarray(A_csr.indices, dtype=np.int64)
    data    = A_csr.data.astype(target_dtype, copy=True)

    Dr = np.ones(n, dtype=np.float64)
    Dc = np.ones(n, dtype=np.float64)

    row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    nz_rows = np.diff(indptr) > 0

    for _ in range(_RUIZ_MAX_ITER):
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

    if equilibrate:
        Dr, Dc, eq_data = _ruiz_py(A_csr, n, target_dtype)
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

    for k in range(n):
        best_val = -1.0; pi = -1; pj = -1
        for i in range(k, n):
            for j, v in M[i].items():
                if j < k: continue
                av = abs(v)
                if av > best_val:
                    best_val = av; pi = i; pj = j
        if pi == -1 or best_val == 0.0:
            raise RuntimeError(
                f"Zero pivot at step {k}. Matrix is singular.")
        if k == 0:
            _max_entry = best_val
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
                if drop_tol > 0.0 and abs(v) <= drop_tol: continue
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
    return L, U, p, q, Dr, Dc
