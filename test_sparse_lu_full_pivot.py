"""
test_sparse_lu_full_pivot.py
============================
Test suite for sparse_lu_full_pivot (C extension and pure-Python paths).

Run with:
    pytest test_sparse_lu_full_pivot.py -v
    python test_sparse_lu_full_pivot.py      # standalone (no pytest needed)

Scaling parameters
------------------
Each test class exposes two class-level constants:

    MULT : int   — multiplies every matrix size n          (default 1)
    DENS : float — multiplies every density parameter      (default 1.0)

Override them to stress-test larger matrices:

    class TestFactorisation(TestFactorisation):
        MULT = 100
        DENS = 0.01

Or edit the defaults directly in each class below.

Notes:
  * Fixed-structure tests (1×1, 2×2, hand-crafted 3×3, singular, all-zero,
    non-square) are intentionally excluded from MULT/DENS scaling because
    they test specific structural behaviour that only makes sense at a
    precise small size.
  * Density is always clamped to min(density * DENS, 1.0) so it stays valid.
  * test_very_sparse_large already targets a large sparse regime; only its
    density is scaled, not its n (which is already 500).
  * test_agrees_with_numpy builds a dense matrix and calls np.linalg.solve —
    fine up to a few thousand but will be slow for very large MULT.
"""

from __future__ import annotations

import sys
import re
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, eye as speye

# ---------------------------------------------------------------------------
# pytest: use the real package when available, install a minimal stub otherwise
# so the file works with  python test_sparse_lu_full_pivot.py  too.
# ---------------------------------------------------------------------------
try:
    import pytest
except ImportError:
    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self.exc = exc; self.match = match
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(
                    f"Expected {self.exc.__name__} but no exception raised")
            if not issubclass(et, self.exc):
                return False
            if self.match and not re.search(self.match, str(ev)):
                raise AssertionError(
                    f"Exception message {str(ev)!r} did not match {self.match!r}")
            return True

    class _PytestStub:
        @staticmethod
        def raises(exc, match=None):
            return _RaisesCtx(exc, match)

    pytest = _PytestStub()
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")

from sparse_lu_full_pivot import (
    sparse_lu_full_pivot,
    lu_solve,
    lu_residual,
    SparseLU,
    _HAVE_C_EXT,
)

RNG = np.random.default_rng(0xDEADBEEF)

print(f"\n  [tests] using {'C extension' if _HAVE_C_EXT else 'pure-Python'} backend\n")


# ---------------------------------------------------------------------------
# Matrix generators
# ---------------------------------------------------------------------------

def rand_sparse_complex(n, density=0.35):
    from scipy.sparse import random as sp_random
    r  = sp_random(n, n, density=density, format="csr", dtype=np.float64,
                   random_state=RNG)
    im = sp_random(n, n, density=density, format="csr", dtype=np.float64,
                   random_state=RNG)
    return (r + 1j * im).astype(np.complex128)


def rand_sparse_real(n, density=0.35):
    from scipy.sparse import random as sp_random
    return sp_random(n, n, density=density, format="csr", dtype=np.float64,
                     random_state=RNG).astype(np.float64)


def diag_dom_complex(n, density=0.25):
    A = rand_sparse_complex(n, density=density)
    row_sums = np.array(np.abs(A).sum(axis=1)).ravel()
    return (A + sp.diags(row_sums + 1.0, dtype=np.complex128)).tocsr()


def diag_dom_real(n, density=0.25):
    A = rand_sparse_real(n, density=density)
    row_sums = np.array(np.abs(A).sum(axis=1)).ravel()
    return (A + sp.diags(row_sums + 1.0, dtype=np.float64)).tocsr()


# ===========================================================================
# 1. Factorisation residual  ||PAQ - LU||_F / ||PAQ||_F
# ===========================================================================

class TestFactorisation:
    # ---- Scaling parameters (edit to stress-test larger problems) ----
    MULT = 1      # multiplies every n
    DENS = 1.0    # multiplies every density  (clamped to 1.0)

    TOL  = 1e-10

    def _n(self, n):
        """Scale a matrix size by MULT."""
        return max(1, int(n * self.MULT))

    def _d(self, density):
        """Scale a density by DENS, clamped to (0, 1]."""
        return min(float(density * self.DENS), 1.0)

    def _check(self, A, tol=None):
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        res = lu_residual(A, L, U, p, q, Dr, Dc)
        t = tol or self.TOL
        assert res < t, f"residual {res:.2e} > {t}"
        Ld = L.toarray()
        np.testing.assert_allclose(np.diag(Ld), np.ones(A.shape[0]),
                                   atol=1e-14, err_msg="L diagonal not 1")
        assert np.max(np.abs(np.triu(Ld, k=1))) < 1e-13, "L has upper entries"
        assert np.max(np.abs(np.tril(U.toarray(), k=-1))) < 1e-13, \
               "U has lower entries"
        return L, U, p, q

    def test_3x3_hand_crafted(self):
        # Fixed small matrix — structural test, not scaled.
        A = csr_matrix(np.array([
            [2+1j, -1+0j,  0   ],
            [-1+0j, 3+0j, -2j  ],
            [ 0,   -2j,    4+0j],
        ], dtype=np.complex128))
        self._check(A)

    def test_5x5_real(self):
        self._check(diag_dom_real(self._n(5), density=self._d(0.8)))

    def test_medium_sparse_complex(self):
        self._check(diag_dom_complex(self._n(25), density=self._d(0.25)))

    def test_dense_complex(self):
        # Fully dense: density is not meaningful, only n scales.
        n = self._n(18)
        M = RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n))
        M += n * np.eye(n)
        self._check(csr_matrix(M))

    def test_identity(self):
        self._check(speye(self._n(12), format="csr", dtype=np.complex128))

    def test_diagonal(self):
        n = self._n(15)
        d = (RNG.random(n) + 0.5) + 1j * RNG.random(n)
        self._check(sp.diags(d, format="csr").astype(np.complex128))

    def test_real_float64(self):
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(
            diag_dom_real(self._n(10), density=self._d(0.5)))
        assert L.dtype == np.float64
        assert U.dtype == np.float64

    def test_permutation_vectors_valid(self):
        n = self._n(16)
        _, _, p, q, _, _ = sparse_lu_full_pivot(diag_dom_complex(n))
        assert sorted(p.tolist()) == list(range(n))
        assert sorted(q.tolist()) == list(range(n))

    def test_output_is_csr(self):
        L, U, _, _, _, _ = sparse_lu_full_pivot(
            diag_dom_complex(self._n(8)))
        assert sp.issparse(L) and L.format == "csr"
        assert sp.issparse(U) and U.format == "csr"

    def test_pure_imaginary(self):
        n = self._n(8)
        M = RNG.standard_normal((n, n))
        np.fill_diagonal(M, np.abs(np.diag(M)) + 1)
        self._check(csr_matrix(1j * M))

    def test_large_sparse(self):
        self._check(diag_dom_complex(self._n(50), density=self._d(0.08)))

    def test_non_symmetric_structure(self):
        n = self._n(9)
        data = np.triu(
            RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n)))
        # Diagonal dominance: diagonal entry > sum of off-diagonal magnitudes
        # in its row.  This guarantees non-singularity at any matrix size.
        row_offdiag_sum = np.sum(np.abs(data), axis=1) - np.abs(np.diag(data))
        np.fill_diagonal(data, row_offdiag_sum + 1.0)
        self._check(csr_matrix(data))


# ===========================================================================
# 2. Solve accuracy
# ===========================================================================

class TestSolve:
    MULT = 1
    DENS = 1.0

    TOL  = 1e-9

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def _check_solve(self, A, n):
        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        x = lu_solve(L, U, p, q, b, Dr, Dc)
        err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-30)
        assert err < self.TOL, f"solve error {err:.2e}"

    def test_small_dense(self):
        n = self._n(6)
        M = RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n))
        M += n * np.eye(n)
        self._check_solve(csr_matrix(M), n)

    def test_medium_sparse(self):
        n = self._n(30)
        self._check_solve(diag_dom_complex(n, density=self._d(0.2)), n)

    def test_real_rhs(self):
        n = self._n(12)
        A = diag_dom_real(n, density=self._d(0.4))
        x_true = RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        x = lu_solve(L, U, p, q, b, Dr, Dc)
        err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-30)
        assert err < self.TOL

    def test_identity_solve(self):
        n = self._n(7)
        A = speye(n, format="csr", dtype=np.complex128)
        b = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        x = lu_solve(L, U, p, q, b, Dr, Dc)
        np.testing.assert_allclose(x, b, atol=1e-13)

    def test_multiple_rhs(self):
        n = self._n(15)
        A = diag_dom_complex(n, density=self._d(0.25))
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        for _ in range(8):
            x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
            b = np.asarray(A @ x_true).ravel()
            x = lu_solve(L, U, p, q, b, Dr, Dc)
            err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-30)
            assert err < self.TOL

    def test_agrees_with_numpy(self):
        # Builds a dense matrix and calls np.linalg.solve — fine for moderate
        # n but slow at large MULT.  Density is not meaningful here.
        n = self._n(10)
        M = RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n))
        M += n * np.eye(n)
        b = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        x_np = np.linalg.solve(M, b)
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(csr_matrix(M))
        x_lu = lu_solve(L, U, p, q, b, Dr, Dc)
        np.testing.assert_allclose(x_lu, x_np, rtol=1e-10)


# ===========================================================================
# 3. Sparsity and ILU mode
# ===========================================================================

class TestSparsity:
    MULT = 1
    DENS = 1.0

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def test_factors_not_fully_dense(self):
        n = self._n(30)
        A = diag_dom_complex(n, density=self._d(0.1))
        L, U, _, _, _, _ = sparse_lu_full_pivot(A)
        dense_tri = n * (n + 1) // 2
        assert L.nnz < dense_tri
        assert U.nnz < dense_tri

    def test_drop_tol_reduces_nnz(self):
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.2))
        L_e, U_e, _, _, _, _ = sparse_lu_full_pivot(A, drop_tol=0.0)
        L_i, U_i, _, _, _, _ = sparse_lu_full_pivot(A, drop_tol=1e-4)
        assert L_i.nnz <= L_e.nnz
        assert U_i.nnz <= U_e.nnz


# ===========================================================================
# 4. COLAMD and tau-row pinning
# ===========================================================================

class TestCOLAMD:
    """
    Tests for COLAMD fill-reducing ordering and tau-row pinning.

    These tests verify:
      - colamd=True gives the same (correct) factorisation as colamd=False
      - pin_dense_rows=True correctly identifies and pins dense rows
      - Explicit tau_row_mask is respected
      - COLAMD does not degrade accuracy for any combination of options
    """
    MULT = 1
    DENS = 1.0

    TOL_RES   = 1e-10   # factorisation residual tolerance
    TOL_SOLVE = 1e-9    # solve accuracy tolerance

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def _check(self, A, **kwargs):
        """Factor with given options and check residual + solve accuracy."""
        n = A.shape[0]
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A, **kwargs)
        res = lu_residual(A, L, U, p, q, Dr, Dc)
        assert res < self.TOL_RES, f"residual {res:.2e} (kwargs={kwargs})"

        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        x = lu_solve(L, U, p, q, b, Dr, Dc)
        err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-30)
        assert err < self.TOL_SOLVE, f"solve error {err:.2e} (kwargs={kwargs})"

    def test_colamd_matches_no_colamd(self):
        """COLAMD and no-COLAMD give the same factorisation quality."""
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.2))
        self._check(A, colamd=False)
        self._check(A, colamd=True)

    def test_colamd_sparse_matrix(self):
        """COLAMD on a genuinely sparse matrix."""
        n = self._n(40)
        A = diag_dom_complex(n, density=self._d(0.08))
        self._check(A, colamd=True)

    def test_colamd_real(self):
        """COLAMD works for real float64 matrices."""
        n = self._n(25)
        A = diag_dom_real(n, density=self._d(0.15))
        self._check(A, colamd=True)

    def test_colamd_with_equilibration(self):
        """COLAMD and Ruiz equilibration can be combined."""
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.2))
        self._check(A, colamd=True, equilibrate=True)

    def test_colamd_without_equilibration(self):
        """COLAMD without equilibration."""
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.2))
        self._check(A, colamd=True, equilibrate=False)

    def test_pin_dense_rows_auto_detection(self):
        """
        Tau rows are detected automatically when pin_dense_rows=True.
        Construct a matrix with 2 deliberately dense rows and verify that
        the factorisation is correct with colamd=True, pin_dense_rows=True.
        """
        n = self._n(20)
        # Sparse banded base matrix
        A_dense = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(max(0, i-2), min(n, i+3)):
                A_dense[i, j] = RNG.standard_normal() + 1j*RNG.standard_normal()
        np.fill_diagonal(A_dense,
                         np.sum(np.abs(A_dense), axis=1) + 1.0)
        # Make rows n//3 and 2*n//3 dense (tau rows)
        tau_rows = [n // 3, 2 * n // 3]
        for r in tau_rows:
            A_dense[r, :] = RNG.standard_normal(n) + 1j*RNG.standard_normal(n)
            # Ensure diagonal dominance in tau row too
            A_dense[r, r] = np.sum(np.abs(A_dense[r, :])) + 1.0
        A = csr_matrix(A_dense)
        self._check(A, colamd=True, pin_dense_rows=True)

    def test_pin_dense_rows_explicit_mask(self):
        """
        An explicit tau_row_mask is respected.
        Mark every other row as tau and verify the factorisation is correct.
        """
        n = self._n(16)
        A = diag_dom_complex(n, density=self._d(0.3))
        mask = np.zeros(n, dtype=bool)
        mask[::3] = True   # every third row is tau
        self._check(A, colamd=True, pin_dense_rows=True,
                    tau_row_mask=mask)

    def test_pin_dense_rows_no_tau_rows(self):
        """pin_dense_rows=True is harmless when no rows are dense."""
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.05))
        self._check(A, colamd=True, pin_dense_rows=True)

    def test_pin_dense_rows_all_tau(self):
        """All rows flagged as tau: degenerates to COLAMD with no row reorder."""
        n = self._n(12)
        A = diag_dom_complex(n, density=self._d(0.4))
        mask = np.ones(n, dtype=bool)   # all rows are tau
        self._check(A, colamd=True, pin_dense_rows=True,
                    tau_row_mask=mask)

    def test_tau_mask_wrong_shape_raises(self):
        """A tau_row_mask with the wrong shape raises ValueError."""
        n = 8
        A = diag_dom_complex(n)
        bad_mask = np.zeros(n + 1, dtype=bool)
        with pytest.raises(ValueError, match="tau_row_mask"):
            sparse_lu_full_pivot(A, colamd=True, pin_dense_rows=True,
                                 tau_row_mask=bad_mask)

    def test_colamd_permutation_vectors_valid(self):
        """p and q are valid permutations of 0..n-1 with COLAMD active."""
        n = self._n(18)
        A = diag_dom_complex(n, density=self._d(0.2))
        _, _, p, q, _, _ = sparse_lu_full_pivot(A, colamd=True,
                                                 pin_dense_rows=False)
        assert sorted(p.tolist()) == list(range(n))
        assert sorted(q.tolist()) == list(range(n))

    def test_colamd_with_tau_permutation_vectors_valid(self):
        """p and q are valid permutations with colamd+pin_dense_rows."""
        n = self._n(18)
        A = diag_dom_complex(n, density=self._d(0.2))
        _, _, p, q, _, _ = sparse_lu_full_pivot(A, colamd=True,
                                                 pin_dense_rows=True)
        assert sorted(p.tolist()) == list(range(n))
        assert sorted(q.tolist()) == list(range(n))

    def test_pin_dense_cols_auto_detection(self):
        """
        Dense columns are detected and pinned when pin_dense_rows=True.
        Construct a matrix with 2 deliberately dense columns and verify
        that the factorisation is correct.
        """
        n = self._n(20)
        A_dense = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(max(0, i-2), min(n, i+3)):
                A_dense[i, j] = RNG.standard_normal() + 1j*RNG.standard_normal()
        np.fill_diagonal(A_dense,
                         np.sum(np.abs(A_dense), axis=1) + 1.0)
        # Make columns n//3 and 2*n//3 dense (tau columns)
        tau_cols = [n // 3, 2 * n // 3]
        for c in tau_cols:
            A_dense[:, c] = RNG.standard_normal(n) + 1j*RNG.standard_normal(n)
            A_dense[c, c] = np.sum(np.abs(A_dense[:, c])) + 1.0
        A = csr_matrix(A_dense)
        self._check(A, colamd=True, pin_dense_rows=True)

    def test_pin_dense_cols_explicit_mask(self):
        """An explicit tau_col_mask is respected."""
        n = self._n(16)
        A = diag_dom_complex(n, density=self._d(0.3))
        col_mask = np.zeros(n, dtype=bool)
        col_mask[::4] = True   # every fourth column is tau
        self._check(A, colamd=True, pin_dense_rows=True,
                    tau_col_mask=col_mask)

    def test_pin_dense_rows_and_cols_together(self):
        """Dense rows and dense columns can be pinned simultaneously."""
        n = self._n(20)
        A_dense = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(max(0, i-2), min(n, i+3)):
                A_dense[i, j] = RNG.standard_normal() + 1j*RNG.standard_normal()
        np.fill_diagonal(A_dense,
                         np.sum(np.abs(A_dense), axis=1) + 1.0)
        # Dense row 0, dense column n-1
        A_dense[0, :] = RNG.standard_normal(n) + 1j*RNG.standard_normal(n)
        A_dense[0, 0] = np.sum(np.abs(A_dense[0, :])) + 1.0
        A_dense[:, n-1] = RNG.standard_normal(n) + 1j*RNG.standard_normal(n)
        A_dense[n-1, n-1] = np.sum(np.abs(A_dense[:, n-1])) + 1.0
        A = csr_matrix(A_dense)
        row_mask = np.zeros(n, dtype=bool)
        row_mask[0] = True
        col_mask = np.zeros(n, dtype=bool)
        col_mask[n-1] = True
        self._check(A, colamd=True, pin_dense_rows=True,
                    tau_row_mask=row_mask, tau_col_mask=col_mask)

    def test_tau_col_mask_wrong_shape_raises(self):
        """A tau_col_mask with the wrong shape raises ValueError."""
        n = 8
        A = diag_dom_complex(n)
        bad_mask = np.zeros(n + 1, dtype=bool)
        with pytest.raises(ValueError, match="tau_col_mask"):
            sparse_lu_full_pivot(A, colamd=True, pin_dense_rows=True,
                                 tau_col_mask=bad_mask)


# ===========================================================================
# 5. Input format compatibility
# ===========================================================================

class TestInputFormats:
    MULT = 1
    DENS = 1.0

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def test_all_scipy_formats(self):
        n = self._n(8)
        A = diag_dom_complex(n, density=self._d(0.4))
        for fmt in ("csr", "csc", "coo", "lil", "dok"):
            try:
                A_fmt = A.asformat(fmt)
            except Exception:
                continue
            L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A_fmt)
            assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-10, \
                f"failed for format {fmt}"

    def test_float32_promoted(self):
        n = self._n(6)
        A = diag_dom_real(n, density=self._d(0.6)).astype(np.float32)
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A.astype(np.float64))
        assert lu_residual(A.astype(np.float64), L, U, p, q, Dr, Dc) < 1e-10


# ===========================================================================
# 5. Edge cases and error handling
# ===========================================================================

class TestEdgeCases:
    # Fixed-structure tests (1×1, 2×2, singular, etc.) are not scaled —
    # they test specific structural behaviour that only makes sense at a
    # precise small size.  The scaling parameters exist only for the tests
    # that generate random matrices of tunable size.
    MULT = 1
    DENS = 1.0

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def test_1x1(self):
        # Fixed: tests 1×1 structural correctness.
        A = csr_matrix([[3+2j]], dtype=np.complex128)
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-14

    def test_2x2_swap_needed(self):
        # Fixed: tests that a column swap is performed when needed.
        A = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-14

    def test_singular_raises(self):
        # Fixed: specific rank-1 matrix.
        A = csr_matrix(np.array([[1, 2], [2, 4]], dtype=np.complex128))
        with pytest.raises(RuntimeError, match="singular"):
            sparse_lu_full_pivot(A)

    def test_all_zero_raises(self):
        # Fixed: all-zero matrix.
        A = csr_matrix((4, 4), dtype=np.complex128)
        with pytest.raises(RuntimeError, match="singular"):
            sparse_lu_full_pivot(A)

    def test_non_square_raises(self):
        # Fixed: rectangular matrix.
        A = csr_matrix(np.ones((3, 4), dtype=np.complex128))
        with pytest.raises(ValueError, match="square"):
            sparse_lu_full_pivot(A)

    def test_permutation_matrix(self):
        n = self._n(6)
        perm = RNG.permutation(n)
        A = csr_matrix((np.ones(n), (np.arange(n), perm)),
                       shape=(n, n), dtype=np.complex128)
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-13

    def test_highly_sparse_with_large_diagonal(self):
        n = self._n(50)
        A = rand_sparse_complex(n, density=self._d(0.03))
        A = A + speye(n, dtype=np.complex128) * 20
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A.tocsr())
        assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-10

    def test_very_sparse_large(self):
        """Target regime: ~1 nnz/row average.  n is fixed at 500;
        only density (of the off-diagonal scatter) is scaled."""
        n = 500   # fixed: already stresses the sparse regime
        cols = RNG.integers(0, n, size=n).astype(np.int32)
        rows = np.arange(n, dtype=np.int32)
        vals = (RNG.random(n) + 0.5) + 1j * RNG.random(n)
        A = sp.csr_matrix((vals, (rows, cols)),
                          shape=(n, n), dtype=np.complex128)
        A = A + speye(n, dtype=np.complex128) * 5
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
        assert lu_residual(A, L, U, p, q, Dr, Dc) < 1e-9


# ===========================================================================
# 7. SparseLU class (drop-in SuperLU replacement)
# ===========================================================================

class TestSparseLU:
    MULT = 1
    DENS = 1.0
    TOL  = 1e-9

    def _n(self, n):   return max(1, int(n * self.MULT))
    def _d(self, d):   return min(float(d * self.DENS), 1.0)

    def test_solve_matches_lu_solve(self):
        """SparseLU.solve gives the same answer as lu_solve."""
        n = self._n(15)
        A = diag_dom_complex(n, density=self._d(0.3))
        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        lu = SparseLU(A)
        x = lu.solve(b)
        err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert err < self.TOL, f"solve error {err:.2e}"

    def test_solve_real(self):
        """SparseLU works for real float64 matrices."""
        n = self._n(12)
        A = diag_dom_real(n, density=self._d(0.4))
        x_true = RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        lu = SparseLU(A)
        x = lu.solve(b)
        err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert err < self.TOL

    def test_solve_trans_T(self):
        """SparseLU.solve(b, trans='T') solves A^T x = b."""
        n = self._n(10)
        A = diag_dom_complex(n, density=self._d(0.4))
        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A.T @ x_true).ravel()
        lu = SparseLU(A)
        x = lu.solve(b, trans='T')
        err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert err < self.TOL, f"trans='T' error {err:.2e}"

    def test_solve_trans_H(self):
        """SparseLU.solve(b, trans='H') solves A^H x = b."""
        n = self._n(10)
        A = diag_dom_complex(n, density=self._d(0.4))
        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A.conj().T @ x_true).ravel()
        lu = SparseLU(A)
        x = lu.solve(b, trans='H')
        err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert err < self.TOL, f"trans='H' error {err:.2e}"

    def test_multiple_rhs(self):
        """SparseLU.solve handles multiple RHS columns."""
        n = self._n(10)
        A = diag_dom_complex(n, density=self._d(0.3))
        X_true = RNG.standard_normal((n, 3)) + 1j * RNG.standard_normal((n, 3))
        B = np.asarray(A @ X_true)
        lu = SparseLU(A)
        X = lu.solve(B)
        assert X.shape == (n, 3)
        err = np.linalg.norm(X - X_true) / np.linalg.norm(X_true)
        assert err < self.TOL

    def test_attributes(self):
        """SparseLU exposes L, U, perm_r, perm_c, shape, nnz."""
        n = self._n(8)
        A = diag_dom_complex(n)
        lu = SparseLU(A)
        assert lu.shape == (n, n)
        assert lu.nnz > 0
        assert lu.L.shape == (n, n)
        assert lu.U.shape == (n, n)
        assert sorted(lu.perm_r.tolist()) == list(range(n))
        assert sorted(lu.perm_c.tolist()) == list(range(n))

    def test_with_colamd(self):
        """SparseLU works with COLAMD options forwarded."""
        n = self._n(20)
        A = diag_dom_complex(n, density=self._d(0.15))
        x_true = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        b = np.asarray(A @ x_true).ravel()
        lu = SparseLU(A, colamd=True, pin_dense_rows=True)
        x = lu.solve(b)
        err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert err < self.TOL

    def test_matches_scipy_splu(self):
        """SparseLU.solve gives comparable accuracy to scipy.sparse.linalg.splu."""
        from scipy.sparse.linalg import splu as scipy_splu
        n = self._n(12)
        A = diag_dom_complex(n, density=self._d(0.4))
        b = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
        x_scipy = scipy_splu(A.tocsc()).solve(b)
        x_ours  = SparseLU(A).solve(b)
        # Both should be close to the true solution
        diff = np.linalg.norm(x_scipy - x_ours) / np.linalg.norm(x_scipy)
        assert diff < 1e-10, f"differs from scipy splu by {diff:.2e}"


# ===========================================================================
# Standalone runner
# ===========================================================================

# ===========================================================================
# 8. Relaxed supernodes and supernode-panel updates (partial pivoting)
# ===========================================================================

def banded_plus_random_complex(n, bw=6, density=0.004):
    """complex, not diagonally dominant: a band (supernodes form) plus scattered entries (near-nesting patterns)"""
    from scipy.sparse import random as sp_random
    offs = list(range(-bw, bw + 1))
    diags = [RNG.standard_normal(n - abs(k)) + 1j * RNG.standard_normal(n - abs(k)) for k in offs]
    B = sp.diags(diags, offs, shape=(n, n), format="csr", dtype=np.complex128)
    R = sp_random(n, n, density=density, format="csr", random_state=RNG) * (1 + 1j)
    return (B + R + 0.5 * speye(n, dtype=np.complex128)).tocsr()


class TestRelaxPanel:
    """relax_rows and panel change only how the factorization is computed: the backward error stays at round-off, the
    solution matches the exact-supernode column-at-a-time path, and panel > 1 keeps the pivots of panel = 1"""
    COMBOS = [(r, p) for r in (0, 2, 8) for p in (1, 4, 32)]

    def _be(self, A, x, b):
        return np.linalg.norm(A @ x - b) / (sp.linalg.norm(A, 1) * np.linalg.norm(x) + np.linalg.norm(b))

    def test_backward_error_and_agreement(self):
        for n in (300, 1200):
            A = banded_plus_random_complex(n)
            b = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
            x0 = SparseLU(A, colamd=True, pin_dense_rows=True).solve(b)
            for r, p in self.COMBOS:
                lu = SparseLU(A, colamd=True, pin_dense_rows=True, relax_rows=r, panel=p)
                x = lu.solve(b)
                be = self._be(A, x, b)
                assert be < 1e-13, f"n {n} relax {r} panel {p}: backward error {be:.1e}"
                d = np.linalg.norm(x - x0) / np.linalg.norm(x0)
                assert d < 1e-8, f"n {n} relax {r} panel {p}: differs from relax 0 panel 1 by {d:.1e}"

    def test_panel_keeps_pivots(self):
        A = banded_plus_random_complex(1500)
        for r in (0, 2):
            ref = SparseLU(A, colamd=True, relax_rows=r, panel=1)
            for p in (4, 32):
                lu = SparseLU(A, colamd=True, relax_rows=r, panel=p)
                assert np.array_equal(lu.perm_r, ref.perm_r), f"relax {r} panel {p}: pivots differ"
                assert lu.L.nnz == ref.L.nnz, f"relax {r} panel {p}: nnz(L) {lu.L.nnz} vs {ref.L.nnz}"

    def test_relax_merges_supernodes_with_bounded_fill(self):
        A = banded_plus_random_complex(1500)
        nz0 = SparseLU(A, colamd=True).nnz
        nz2 = SparseLU(A, colamd=True, relax_rows=2).nnz
        assert nz0 <= nz2 <= 1.5 * nz0, f"fill {nz0} -> {nz2}"

    def test_transposed_solves_after_panel(self):
        A = banded_plus_random_complex(800)
        lu = SparseLU(A, colamd=True, relax_rows=2, panel=16)
        b = RNG.standard_normal(800) + 1j * RNG.standard_normal(800)
        for tr, M in (("T", A.T), ("H", A.conj().T)):
            x = lu.solve(b, trans=tr)
            be = self._be(sp.csr_matrix(M), x, b)
            assert be < 1e-13, f"trans {tr}: backward error {be:.1e}"

    def test_real_matrix_ignores_panel(self):
        A = diag_dom_real(400, density=0.02)
        b = RNG.standard_normal(400)
        x = SparseLU(A, colamd=True, relax_rows=2, panel=8).solve(b)
        assert self._be(A, x, b) < 1e-13


if __name__ == "__main__":
    suites = [TestFactorisation, TestSolve, TestSparsity, TestCOLAMD,
              TestInputFormats, TestEdgeCases, TestSparseLU, TestRelaxPanel]
    passed = failed = 0
    for cls in suites:
        obj = cls()
        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as exc:
                import traceback
                print(f"  FAIL  {cls.__name__}.{name}: {exc}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)
