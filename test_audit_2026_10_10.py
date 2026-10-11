"""
Regression tests for the 2026-10-10 code audit (docs/C_LIBRARY_PLAN.md): the C pipeline actually in use and equal to the
Python route, transposed / conjugate-transposed solves under pinning, pivoting and strong scaling, the row kernel's pivot
strategies and drop tolerance, input validation, multiple right-hand sides, and the ordering cache.
"""
import sys
import threading
import numpy as np
import scipy.sparse as sp
import pytest

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import sparse_lu_full_pivot as M
from sparse_lu_full_pivot import SparseLU, lu_solve, sparse_lu_full_pivot

needs_c = pytest.mark.skipif(not M._HAVE_C_EXT, reason="C extension not built")


def hard_complex(n, seed, dense_rows=2, dense_cols=2, scale=8.0, density=0.03):
    """Complex, not diagonally dominant, with dense rows / columns and rows / columns spanning 10^(+-scale)."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, random_state=rng, format="lil", dtype=float)
    A = A + 1j * sp.random(n, n, density=density, random_state=rng, format="lil", dtype=float)
    A = sp.lil_matrix(A + sp.eye(n) * 0.5)
    for _ in range(dense_rows):
        A[int(rng.integers(n)), :] = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for _ in range(dense_cols):
        A[:, int(rng.integers(n))] = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).reshape(n, 1)
    A = sp.csc_matrix(A)
    r = 10.0 ** rng.uniform(-scale, scale, n)
    c = 10.0 ** rng.uniform(-scale, scale, n)
    return sp.csc_matrix(sp.diags(r) @ A @ sp.diags(c))


def solve_err(A, x, b, trans):
    Ad = A.toarray()
    op = {"N": Ad, "T": Ad.T, "H": Ad.conj().T}[trans]
    return np.linalg.norm(op @ x - b) / (np.linalg.norm(op, 1) * np.linalg.norm(x) + np.linalg.norm(b))


# ---------------------------------------------------------------------------- the C pipeline
@needs_c
def test_c_pipeline_is_the_current_build():
    assert M._sparse_lu_c.API_VERSION == M._C_API_VERSION
    assert hasattr(M._sparse_lu_c, "analyze") and hasattr(M._sparse_lu_c, "factor_sn_perm")


@needs_c
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_c_and_python_routes_bit_identical(seed):
    A = hard_complex(300, seed)
    rng = np.random.default_rng(seed)
    rm = rng.random(300) < 0.01
    out = []
    for py_route in (False, True):
        M._PY_PIPELINE = py_route
        try:
            M.clear_ordering_cache()
            lu = SparseLU(A, colamd=True, pin_dense_rows=True, tau_row_mask=rm, relax_rows=8, panel=16)
            out.append(lu)
        finally:
            M._PY_PIPELINE = False
    a, b = out
    for x, y in ((a._L.data, b._L.data), (a._L.indices, b._L.indices), (a._U.data, b._U.data), (a._p, b._p),
                 (a._q, b._q), (a._Dr, b._Dr), (a._Dc, b._Dc)):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------- solves
@pytest.mark.parametrize("trans", ["N", "T", "H"])
@pytest.mark.parametrize("seed", [4, 5])
def test_trans_solves_pinned_pivoted_scaled(trans, seed):
    A = hard_complex(250, seed)
    b = np.random.default_rng(seed).standard_normal(250) + 1j
    lu = SparseLU(A, colamd=True, pin_dense_rows=True, relax_rows=4, panel=8)
    assert solve_err(A, lu.solve(b, trans=trans), b, trans) < 1e-12


@pytest.mark.parametrize("strategy", ["rook", "full"])
@pytest.mark.parametrize("trans", ["N", "T", "H"])
def test_row_kernel_strategies_solve(strategy, trans):
    A = hard_complex(120, 7, scale=3.0, density=0.06)
    b = np.random.default_rng(7).standard_normal(120) + 0.5j
    lu = SparseLU(A, colamd=True, pin_dense_rows=True, pivot_strategy=strategy)
    assert solve_err(A, lu.solve(b, trans=trans), b, trans) < 1e-12


def test_lu_solve_several_right_hand_sides():
    A = hard_complex(150, 8, scale=2.0)
    L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A)
    B = np.random.default_rng(8).standard_normal((150, 3)) + 0j
    X = lu_solve(L, U, p, q, B, Dr, Dc)
    assert X.shape == (150, 3)
    for k in range(3):
        assert np.allclose(X[:, k], lu_solve(L, U, p, q, B[:, k], Dr, Dc), rtol=0, atol=0)


def test_near_singular_shift_with_pivot_rel_zero():
    """A shift on an eigenvalue: singular to working precision.  pivot_rel 0 factors; the backward error stays small."""
    rng = np.random.default_rng(9)
    n = 200
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    lam = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    A = sp.csc_matrix((Q * lam) @ Q.conj().T)
    C = sp.csc_matrix(A - lam[0] * sp.eye(n))
    lu = SparseLU(C, colamd=True, pivot_rel=0.0, relax_rows=4, panel=8)
    b = rng.standard_normal(n) + 0j
    for trans in ("N", "H"):
        assert solve_err(C, lu.solve(b, trans=trans), b, trans) < 1e-12


# ---------------------------------------------------------------------------- row kernel
@needs_c
def test_drop_tol_keeps_u_diagonal():
    """A pivot above the singular threshold but below drop_tol used to be dropped from U at assembly."""
    A = sp.csc_matrix(np.diag([0.1, 1.0, 1.0]).astype(complex))
    L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A, equilibrate=False, colamd=False, drop_tol=0.5)
    assert np.allclose(lu_solve(L, U, p, q, np.array([0.1, 1.0, 1.0]), Dr, Dc), [1.0, 1.0, 1.0])


@needs_c
@pytest.mark.parametrize("strategy", ["partial", "rook"])
def test_row_kernel_threshold_is_matrix_relative(strategy):
    """The singular test is relative to max |a_ij|, not to the step-0 pivot: a tiny column 0 no longer disarms it."""
    A = sp.csc_matrix(np.diag([1e-12, 1.0]))
    with pytest.raises(RuntimeError, match="singular"):
        sparse_lu_full_pivot(A, equilibrate=False, colamd=False, pivot_strategy=strategy, drop_tol=1e-300)
    L, U, p, q, Dr, Dc = sparse_lu_full_pivot(A, equilibrate=False, colamd=False, pivot_strategy=strategy, drop_tol=1e-300,
                                              pivot_rel=1e-14)
    assert np.allclose(lu_solve(L, U, p, q, np.array([1e-12, 1.0]), Dr, Dc), [1.0, 1.0])


@needs_c
def test_drop_tol_must_be_nonnegative():
    with pytest.raises(ValueError, match="drop_tol"):
        sparse_lu_full_pivot(hard_complex(30, 11, scale=0.0), drop_tol=float("nan"))


# ---------------------------------------------------------------------------- input validation (the extension)
@needs_c
class TestValidation:
    def setup_method(self):
        self.A = sp.csc_matrix(hard_complex(40, 12, scale=0.0))
        self.p = self.A.indptr.astype(np.int64); self.i = self.A.indices.astype(np.int64); self.x = self.A.data.astype(complex)
        self.X = M._sparse_lu_c

    def test_index_out_of_range(self):
        bad = self.i.copy(); bad[3] = 40
        with pytest.raises(ValueError, match="out of range"):
            self.X.lu_factor_sn(self.x, bad, self.p, 40, 1)
        with pytest.raises(ValueError, match="out of range"):
            self.X.analyze(self.p, bad, 40)

    def test_short_indptr_and_negative_n(self):
        with pytest.raises(ValueError, match="indptr must have length"):
            self.X.lu_factor_sn(self.x, self.i, self.p[:-1], 40, 1)
        with pytest.raises(ValueError, match="n must be >= 0"):
            self.X.lu_factor_sn(self.x, self.i, self.p, -1, 1)

    def test_zero_dimensional_array(self):
        with pytest.raises(ValueError, match="1-D"):
            self.X.lu_solve(np.int64(3), self.i, self.x, self.p, self.i, self.x, np.ones(40, complex), 0, 1)

    def test_non_permutation(self):
        rp = np.arange(40, dtype=np.int64); rp[5] = 6
        with pytest.raises(ValueError, match="not a permutation"):
            self.X.factor_sn_perm(self.x, self.i, self.p, 40, rp, np.arange(40, dtype=np.int64), 1)

    def test_non_finite(self):
        x = self.x.copy(); x[2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            self.X.lu_factor_sn(x, self.i, self.p, 40, 1)

    def test_non_canonical_rejected_by_analyze(self):
        i = self.i.copy(); s, e = self.p[5], self.p[6]
        if e - s >= 2:
            i[s], i[s + 1] = i[s + 1], i[s]
            with pytest.raises(ValueError, match="canonical"):
                self.X.analyze(self.p, i, 40)

    def test_bad_trans(self):
        lu = SparseLU(self.A)
        with pytest.raises(ValueError, match="trans"):
            self.X.lu_solve(lu._L.indptr, lu._L.indices, lu._L.data, lu._U.indptr, lu._U.indices, lu._U.data, np.ones(40, complex), 7, 1)

    def test_masks_ignored_without_pinning(self):
        r = self.X.analyze(self.p, self.i, 40, 0, 0.5, 10, np.zeros(3, np.uint8), None, 1)   # wrong length, but unused
        assert len(r["col_perm"]) == 40


def test_panel_wider_than_n_equals_panel_n():
    A = hard_complex(60, 13, scale=0.0)
    a = SparseLU(A, relax_rows=4, panel=60); b = SparseLU(A, relax_rows=4, panel=10 ** 9)
    assert np.array_equal(a._L.data, b._L.data) and np.array_equal(a._U.data, b._U.data)


# ---------------------------------------------------------------------------- the ordering cache
def test_cache_keys_separate_settings_and_match_uncached():
    A = hard_complex(200, 14)
    rng = np.random.default_rng(14)
    rm = rng.random(200) < 0.02
    settings = [dict(pin_dense_rows=True), dict(pin_dense_rows=False), dict(pin_dense_rows=True, tau_row_mask=rm),
                dict(pin_dense_rows=True, tau_col_mask=rm), dict(pin_dense_rows=True, postorder=False)]
    M.clear_ordering_cache()
    for kw in settings:
        cached = SparseLU(A, colamd=True, cache_ordering=True, **kw)
        fresh = SparseLU(A, colamd=True, cache_ordering=False, **kw)
        assert np.array_equal(cached._p, fresh._p) and np.array_equal(cached._q, fresh._q)


def test_cache_returns_copies():
    A = hard_complex(120, 15)
    M.clear_ordering_cache()
    r1 = M._colamd_permutations(M._canonical_csc(A), 120, True, 0.5, 10, None, None, cache=True, postorder=True)
    r1[1][:] = 0                                    # the caller mutates what it got
    r2 = M._colamd_permutations(M._canonical_csc(A), 120, True, 0.5, 10, None, None, cache=True, postorder=True)
    assert len(set(r2[1].tolist())) == 120          # still a permutation


def test_cache_concurrent_threads():
    mats = [hard_complex(80, 100 + k, scale=0.0) for k in range(12)]   # more patterns than the cache holds
    M.clear_ordering_cache()
    errors = []

    def work(k):
        try:
            for j in range(6):
                SparseLU(mats[(k + j) % len(mats)], colamd=True, relax_rows=2, panel=4)
        except Exception as e:   # noqa: BLE001
            errors.append(repr(e))

    th = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in th: t.start()
    for t in th: t.join()
    assert not errors, errors


# ---------------------------------------------------------------- row kernel: drop_tol filters computed values only
def _drop_block(npad, cplx):
    """A 3x3 block whose row 1 holds a12 = 5e-3 < drop_tol that the update from row 0 does not touch but that feeds U22 = 4 - 0.5 a12
    on the next step; npad padding columns on rows 0 and 1 push the row-0 update onto the BLAS axpy path without changing the
    block's arithmetic.  Dominant pivots (8, 6, then U22) with padding diagonals 1, so every strategy picks the same order."""
    n = 3 + 2 * npad
    A = sp.lil_matrix((n, n), dtype=complex if cplx else float)
    A[0, 0] = 8; A[1, 0] = 4; A[1, 1] = 6; A[1, 2] = 5e-3; A[2, 1] = 3; A[2, 2] = 4
    for t in range(npad):
        A[0, 3 + t] = 1; A[1, 3 + npad + t] = 1
    for j in range(3, n):
        A[j, j] = 1
    return A.tocsr()


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("strategy", ["partial", "rook", "full"])
def test_row_kernel_drop_independent_of_axpy_path(cplx, strategy):
    for npad in (0, 40):
        L, U, p, q, Dr, Dc = sparse_lu_full_pivot(_drop_block(npad, cplx), drop_tol=1e-2, equilibrate=False, colamd=False,
                                                  pivot_strategy=strategy)
        U = U.tocsr()
        k2 = int(np.where(p == 2)[0][0]); c2 = int(np.where(q == 2)[0][0])
        assert abs(U[k2, c2] - 3.9975) < 1e-12, (npad, U[k2, c2])
