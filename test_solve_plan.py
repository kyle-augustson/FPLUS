"""Tests of the blocked solves (src/solve_plan.c): the plan's solves against the plain sweeps (lu_solve), several right-hand
sides at once, thread-count independence, the fallback for factors without the supernodal layout, and plan / factor
mismatches."""
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import sparse_lu_full_pivot as M
from sparse_lu_full_pivot import SparseLU

pytestmark = pytest.mark.skipif(not (M._HAVE_C_EXT and hasattr(M._sparse_lu_c, "solve_plan")), reason="needs the C extension")


def _pencil(n, seed, cplx=True, dense_lines=True):
    """A random pencil-like matrix: a band, scattered entries, dense lines, 8 orders of row scaling (wide supernodes)."""
    r = np.random.default_rng(seed)
    A = sp.random(n, n, density=6.0 / n, random_state=r, format="lil")
    if cplx:
        A = A + 1j * sp.random(n, n, density=4.0 / n, random_state=r, format="lil")
    A = (A + sp.diags(r.uniform(1, 2, n)) + sp.diags(r.uniform(-1, 1, n - 1), 1) + sp.diags(r.uniform(-1, 1, n - 1), -1)).tolil()
    if dense_lines:
        A[5, :] = r.uniform(0.5, 1, n); A[:, 11] = r.uniform(0.5, 1, n)[:, None]
    return (sp.diags(10.0 ** r.uniform(-4, 4, n)) @ A).tocsc()


def _quiet_lu(A, **kw):
    """SparseLU without warnings (an ILU breakdown warning is expected for some cases here)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SparseLU(A, **kw)


def _plain(lu, b, trans):
    return M._tri_solve_c(lu._L_arrays, lu._U_arrays, b, trans)


@pytest.mark.parametrize("cplx", [True, False])
@pytest.mark.parametrize("kw", [dict(relax_rows=32, panel=512), dict(relax_rows=0, panel=1), dict(pivot_strategy="rook"),
                                dict(drop_tol=1e-3, pivot_strategy="full")])
def test_plan_matches_plain_sweeps(cplx, kw):
    A = _pencil(700, 3, cplx)
    lu = _quiet_lu(A, colamd=True, pin_dense_rows=True, **kw)
    assert lu._plan is not None
    r = np.random.default_rng(0)
    b = r.standard_normal(700) + (1j * r.standard_normal(700) if cplx else 0)
    for trans in (0, 1, 2):
        x0 = _plain(lu, b, trans)
        x1 = M._tri_solve_c(lu._L_arrays, lu._U_arrays, b, trans, lu._plan)
        if trans == 0:
            assert np.array_equal(x0, x1)                      # the forward solves are bit-identical
        else:
            assert np.max(np.abs(x1 - x0)) <= 1e-9 * np.max(np.abs(x0))


@pytest.mark.parametrize("trans", ["N", "T", "H"])
def test_several_rhs_equal_one_at_a_time(trans):
    A = _pencil(900, 5)
    lu = SparseLU(A, colamd=True, pin_dense_rows=True, relax_rows=32, panel=512)
    r = np.random.default_rng(1)
    B = r.standard_normal((900, 7)) + 1j * r.standard_normal((900, 7))
    X = lu.solve(B, trans=trans)
    cols = np.column_stack([lu.solve(B[:, k], trans=trans) for k in range(7)])
    assert X.shape == (900, 7) and np.array_equal(X, cols)
    Aop = {"N": A, "T": A.T, "H": A.conj().T}[trans]
    nA = abs(A).sum(axis=0).max()
    for k in range(7):
        be = np.linalg.norm(Aop @ X[:, k] - B[:, k]) / (nA * np.linalg.norm(X[:, k]) + np.linalg.norm(B[:, k]))
        assert be < 1e-14


def test_thread_count_independent():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    A = _pencil(1500, 7)
    lu = SparseLU(A, colamd=True, pin_dense_rows=True, relax_rows=32, panel=512)
    b = np.random.default_rng(2).standard_normal((1500, 3)) + 0j
    out = []
    for th in (1, 2, 4):
        with threadpoolctl.threadpool_limits(th, user_api="openmp"):
            out.append([M._tri_solve_c(lu._L_arrays, lu._U_arrays, b, t, lu._plan) for t in (0, 1, 2)])
    for o in out[1:]:
        for t in range(3):
            assert np.array_equal(o[t], out[0][t])


def test_generic_factors_fall_back():
    """Factors without the diagonal first in every L column (here: L given with its rows in descending order) still solve,
    through the plain sweeps."""
    A = _pencil(300, 9)
    lu = SparseLU(A, colamd=True, relax_rows=32, panel=512)
    Lp, Li, Lx = lu._L_arrays
    Li2, Lx2 = Li.copy(), Lx.copy()
    for j in range(300):                                      # reverse each column: the diagonal goes last
        s = slice(Lp[j], Lp[j + 1]); Li2[s] = Li[s][::-1]; Lx2[s] = Lx[s][::-1]
    Up, Ui, Ux = lu._U_arrays
    plan = M._sparse_lu_c.solve_plan(Lp, Li2, Up, Ui)
    b = np.random.default_rng(3).standard_normal(300) + 0j
    x = M._tri_solve_c((Lp, Li2, Lx2), lu._U_arrays, b, 0, plan)
    assert np.array_equal(x, _plain(lu, b, 0))


def test_plan_factor_mismatch_rejected():
    lu1 = SparseLU(_pencil(400, 11), colamd=True, relax_rows=32, panel=512)
    lu2 = SparseLU(_pencil(400, 12), colamd=True, relax_rows=32, panel=512)
    b = np.ones(400, dtype=complex)
    assert lu1._L_arrays[0][-1] != lu2._L_arrays[0][-1]       # different fill: the plan's recorded sizes catch the mix-up
    with pytest.raises(ValueError, match="plan does not match"):
        M._tri_solve_c(lu2._L_arrays, lu2._U_arrays, b, 0, lu1._plan)
    with pytest.raises((ValueError, TypeError)):
        M._sparse_lu_c.lu_solve_plan("not a plan", *lu1._L_arrays, *lu1._U_arrays, b, 0, 1)
