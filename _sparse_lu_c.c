/*
 * _sparse_lu_c.c
 * ==============
 * Python C-extension: sparse LU decomposition with full pivoting.
 *
 * Factorisation
 * -------------
 * Given an n×n sparse matrix A, finds permutation vectors p (rows) and q
 * (columns) and sparse unit-lower-triangular L and upper-triangular U such that
 *
 *     A[p, :][:, q]  =  L @ U        (exact, or ≈ when drop_tol > 0)
 *
 * No LAPACK, SuperLU, or external sparse-solver library is called for the
 * factorisation or the triangular solves.
 *
 * Two kernels: ``lu_factor`` is the right-looking row kernel below, which
 * carries the full and rook pivoting strategies (and the drop tolerance);
 * ``lu_factor_sn`` is the supernodal left-looking kernel for partial
 * pivoting, with work proportional to the flops of the factorisation and
 * the dense block updates through CBLAS (HAVE_BLAS; plain loops without).
 * ``lu_solve`` is the pair of triangular sweeps on the CSC factors.
 *
 * -------------------------------------------------------------------------
 * COMPLEXITY NOTE ON THE RIGHT-LOOKING KERNEL
 * -------------------------------------------------------------------------
 * Full pivoting searches the entire active submatrix at each step k for the
 * entry of maximum absolute value.  Even when the matrix is very sparse, this
 * pivot search visits every stored entry in rows k..n-1 and columns k..n-1,
 * giving O(nnz_active) work per step and O(n * nnz / n) = O(nnz) total for
 * the search alone — BUT the col_swap is O(n) per step regardless of
 * sparsity (every row must be checked).  This makes the total work:
 *
 *     O(n²)  for col_swap alone
 *
 * For n = 1,000,000 this is 10¹² operations and is impractical even with
 * OpenMP.  Full pivoting is appropriate for small-to-medium n (say n ≤ 50,000
 * on a single node).  For very large n with 1 nnz/row consider:
 *   • Partial pivoting (col_swap eliminated, O(n·avg_nnz) total)
 *   • Rook pivoting (stops searching as soon as row AND column max match)
 *   • Threshold pivoting with reordering (COLAMD + partial pivoting)
 *
 * -------------------------------------------------------------------------
 * Why the axpy uses a sparse merge rather than BLAS zaxpy/daxpy
 * -------------------------------------------------------------------------
 * BLAS axpy operates on DENSE contiguous vectors.  Using it for sparse rows
 * requires: gather both rows into dense buffers → BLAS call → scatter back,
 * with malloc overhead per call.  This break-even point is approximately
 * 32–64 nonzeros per row (where AVX-512 throughput exceeds gather/scatter
 * overhead).  At the target density of ~1 nnz/row the pure merge is faster.
 * If your matrices are denser (avg nnz/row > 32), replacing the merge with
 * a gather→zaxpy→scatter path would improve throughput.
 *
 * -------------------------------------------------------------------------
 * OpenMP parallelism
 * -------------------------------------------------------------------------
 * The outer k-loop is sequential (each step depends on the previous pivot).
 * Three inner operations are parallelised with OpenMP:
 *   1. find_full_pivot  — parallel scan with thread-local reduction
 *   2. col_swap         — trivially parallel over rows
 *   3. elimination loop — rows k+1..n-1 are independent (each row_axpy
 *                         reads only the pivot row M[k], which is read-only
 *                         during this phase, and writes to its own row M[i])
 * Build with -fopenmp (uncomment in setup.py) to activate.
 *
 * -------------------------------------------------------------------------
 * Index width
 * -------------------------------------------------------------------------
 * All internal indices use int64_t (npy_intp on 64-bit platforms).
 * Scipy may pass int32 or int64 index arrays; both are accepted and cast
 * to int64_t internally.  Output index arrays are always int64.
 * This supports matrices with up to 2^63 − 1 ≈ 9.2 × 10¹⁸ stored entries.
 *
 * -------------------------------------------------------------------------
 * Python interface
 * -------------------------------------------------------------------------
 *     _sparse_lu_c.lu_factor(data, indices, indptr, n, drop_tol, is_complex)
 *
 *   data      : 1-D ndarray, complex128 or float64, CSR data
 *   indices   : 1-D ndarray, int32 or int64, CSR column indices
 *   indptr    : 1-D ndarray, int32 or int64, CSR row pointers (length n+1)
 *   n         : int
 *   drop_tol  : float  (0 = exact factorisation)
 *   is_complex: int    (1 = complex128, 0 = float64)
 *
 * Returns a dict:
 *   { "L_data", "L_indices", "L_indptr",
 *     "U_data", "U_indices", "U_indptr",
 *     "p", "q" }
 *   All index arrays are int64; data arrays match the input dtype.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "fplus.h"

/* This file is the CPython wrapper only (phase 1 of docs/C_LIBRARY_PLAN.md): argument parsing, numpy arrays, the
 * mapping of FPLUS_ERR_* to exceptions.  The C core (include/fplus.h, src/) never touches the Python API, so the GIL is
 * released around every core call. */

/* Wrap a malloc'd C buffer as an owning 1-D numpy array (NPY_ARRAY_OWNDATA transfers the malloc to numpy's GC). */
static PyObject *wrap_array(void *data, npy_intp len, int typenum)
{
    npy_intp dims[1] = { len };
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, typenum, data);
    if (!arr) return NULL;
    PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

/* Convert to a 1-D array of the given type, or NULL with an exception set.  Callers chain these with || so that no
 * conversion runs while another's exception is pending (audit 2026-10-10). */
static PyArrayObject *as_1d(PyObject *o, int type, int flags, const char *name)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OTF(o, type, flags);
    if (!a) return NULL;
    if (PyArray_NDIM(a) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1-D (got %d dimensions)", name, PyArray_NDIM(a));
        Py_DECREF(a);
        return NULL;
    }
    return a;
}

/* indptr of length n + 1, and indices / data with at least indptr[n] entries; 0, or -1 with ValueError.  The core checks
 * the contents (monotone indptr, indices in range). */
static int check_compressed_lengths(int64_t n, PyArrayObject *ptr, PyArrayObject *idx, PyArrayObject *data, const char *fn)
{
    if (n < 0) { PyErr_Format(PyExc_ValueError, "%s: n must be >= 0", fn); return -1; }
    if (PyArray_DIM(ptr, 0) != n + 1) {
        PyErr_Format(PyExc_ValueError, "%s: indptr must have length n + 1 = %lld (got %lld)", fn, (long long)(n + 1), (long long)PyArray_DIM(ptr, 0));
        return -1;
    }
    const int64_t nnz = ((const int64_t *)PyArray_DATA(ptr))[n];
    if (nnz < 0 || PyArray_DIM(idx, 0) < nnz || (data && PyArray_DIM(data, 0) < nnz)) {
        PyErr_Format(PyExc_ValueError, "%s: indices / data hold fewer than indptr[n] = %lld entries", fn, (long long)nnz);
        return -1;
    }
    return 0;
}

static PyObject *raise_fplus(int rc, const char *err)
{
    if (rc == FPLUS_ERR_NOMEM) return PyErr_NoMemory();
    PyErr_SetString(rc == FPLUS_ERR_BADARG ? PyExc_ValueError : PyExc_RuntimeError, err[0] ? err : "FPLUS error");
    return NULL;
}

/* the factors as the dict the Python layer expects; takes ownership of every array (frees what it cannot wrap) */
static PyObject *factors_to_dict(fplus_factors *f)
{
    int64_t n = f->n, nnzL = f->Lp[n], nnzU = f->Up[n];
    int vt = f->is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyObject *L_dat = wrap_array(f->Lx, nnzL, vt);   if (L_dat) f->Lx = NULL;
    PyObject *L_idx = wrap_array(f->Li, nnzL, NPY_INT64); if (L_idx) f->Li = NULL;
    PyObject *L_ptr = wrap_array(f->Lp, n + 1, NPY_INT64); if (L_ptr) f->Lp = NULL;
    PyObject *U_dat = wrap_array(f->Ux, nnzU, vt);   if (U_dat) f->Ux = NULL;
    PyObject *U_idx = wrap_array(f->Ui, nnzU, NPY_INT64); if (U_idx) f->Ui = NULL;
    PyObject *U_ptr = wrap_array(f->Up, n + 1, NPY_INT64); if (U_ptr) f->Up = NULL;
    PyObject *p_arr = wrap_array(f->p, n, NPY_INT64); if (p_arr) f->p = NULL;
    PyObject *q_arr = wrap_array(f->q, n, NPY_INT64); if (q_arr) f->q = NULL;
    PyObject *result = NULL;
    if (L_dat && L_idx && L_ptr && U_dat && U_idx && U_ptr && p_arr && q_arr) {
        result = f->csc
            ? Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:s}", "L_data", L_dat, "L_indices", L_idx, "L_indptr", L_ptr,
                            "U_data", U_dat, "U_indices", U_idx, "U_indptr", U_ptr, "p", p_arr, "q", q_arr, "format", "csc")
            : Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}", "L_data", L_dat, "L_indices", L_idx, "L_indptr", L_ptr,
                            "U_data", U_dat, "U_indices", U_idx, "U_indptr", U_ptr, "p", p_arr, "q", q_arr);
    } else
        PyErr_NoMemory();
    if (result && f->n_pivots_replaced > 0) {
        PyObject *nr = PyLong_FromLongLong((long long)f->n_pivots_replaced);
        if (!nr || PyDict_SetItemString(result, "n_pivots_replaced", nr)) Py_CLEAR(result);
        Py_XDECREF(nr);
    }
    Py_XDECREF(L_dat); Py_XDECREF(L_idx); Py_XDECREF(L_ptr); Py_XDECREF(U_dat); Py_XDECREF(U_idx); Py_XDECREF(U_ptr);
    Py_XDECREF(p_arr); Py_XDECREF(q_arr);
    fplus_factors_free(f);
    return result;
}

static PyObject *
py_lu_factor(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    double drop_tol;
    int is_complex;
    int verbose = 0;
    int pivot_strategy = FPLUS_PIVOT_PARTIAL;
    double pivot_rel = FPLUS_PIVOT_REL_DEFAULT;
    if (!PyArg_ParseTuple(args, "OOOndi|iid", &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &drop_tol, &is_complex, &verbose, &pivot_strategy, &pivot_rel))
        return NULL;
    int64_t n = (int64_t)n_sst;
    PyArrayObject *data_arr = NULL, *idx_arr = NULL, *ptr_arr = NULL;
    if (!(data_arr = as_1d(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY, "data"))
            || !(idx_arr = as_1d(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || !(ptr_arr = as_1d(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr"))
            || check_compressed_lengths(n, ptr_arr, idx_arr, data_arr, "lu_factor")) {
        Py_XDECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); return NULL;
    }
    fplus_factors f; char err[512] = {0}; int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = fplus_factor_row(n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr), PyArray_DATA(data_arr),
                          drop_tol, is_complex, verbose, pivot_strategy, pivot_rel, &f, err, sizeof(err));
    Py_END_ALLOW_THREADS
    Py_DECREF(data_arr); Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    if (rc != FPLUS_OK) return raise_fplus(rc, err);
    return factors_to_dict(&f);
}

static PyObject *
py_ruiz_equilibrate(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    int max_iter, is_complex;
    double tol;
    int verbose = 0, is_csc = 0;
    if (!PyArg_ParseTuple(args, "OOOnidi|ii", &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &max_iter, &tol, &is_complex, &verbose, &is_csc))
        return NULL;
    int64_t n = (int64_t)n_sst;
    PyArrayObject *idx_arr = NULL, *ptr_arr = NULL, *data_arr = NULL;
    /* the data are copied, so the scaling can work in place */
    if (!(idx_arr = as_1d(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || !(ptr_arr = as_1d(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr"))
            || !(data_arr = as_1d(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_ENSURECOPY | NPY_ARRAY_IN_ARRAY, "data"))
            || check_compressed_lengths(n, ptr_arr, idx_arr, data_arr, "ruiz_equilibrate")) {
        Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); Py_XDECREF(data_arr); return NULL;
    }
    double *Dr = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *Dc = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!Dr || !Dc) { free(Dr); free(Dc); Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr); return PyErr_NoMemory(); }
    char err[512] = {0}; int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = fplus_ruiz(n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr), PyArray_DATA(data_arr),
                    is_complex, is_csc, max_iter, tol, verbose, Dr, Dc, err, sizeof(err));
    Py_END_ALLOW_THREADS
    Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    if (rc != FPLUS_OK) { free(Dr); free(Dc); Py_DECREF(data_arr); return raise_fplus(rc, err); }
    PyObject *Dr_arr = wrap_array(Dr, (npy_intp)n, NPY_FLOAT64); if (!Dr_arr) free(Dr);
    PyObject *Dc_arr = wrap_array(Dc, (npy_intp)n, NPY_FLOAT64); if (!Dc_arr) free(Dc);
    PyObject *result = (Dr_arr && Dc_arr) ? Py_BuildValue("{sOsOsO}", "Dr", Dr_arr, "Dc", Dc_arr, "data", (PyObject *)data_arr) : PyErr_NoMemory();
    Py_XDECREF(Dr_arr); Py_XDECREF(Dc_arr); Py_DECREF(data_arr);
    return result;
}

static PyObject *
py_lu_factor_sn(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    int is_complex, verbose = 0;
    long long relax_rows = 0, panel = 1;
    int use_gpu = 0; double gpu_min_flops = 5e7; double pivot_rel = FPLUS_PIVOT_REL_DEFAULT;
    if (!PyArg_ParseTuple(args, "OOOni|iLLidd", &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &is_complex, &verbose, &relax_rows, &panel, &use_gpu, &gpu_min_flops, &pivot_rel))
        return NULL;
    int64_t n = (int64_t)n_sst;
    PyArrayObject *data_arr = NULL, *idx_arr = NULL, *ptr_arr = NULL;
    if (!(data_arr = as_1d(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY, "data"))
            || !(idx_arr = as_1d(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || !(ptr_arr = as_1d(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr"))
            || check_compressed_lengths(n, ptr_arr, idx_arr, data_arr, "lu_factor_sn")) {
        Py_XDECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); return NULL;
    }
    fplus_factors f; char err[512] = {0}; int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = fplus_factor_sn(n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr), PyArray_DATA(data_arr),
                         is_complex, verbose, (int64_t)relax_rows, (int64_t)panel, use_gpu, gpu_min_flops, pivot_rel, &f, err, sizeof(err));
    Py_END_ALLOW_THREADS
    Py_DECREF(data_arr); Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    if (rc != FPLUS_OK) return raise_fplus(rc, err);
    return factors_to_dict(&f);
}

/* ---- solve plans (a PyCapsule owning an fplus_solve_plan) ---- */
static void plan_capsule_free(PyObject *cap)
{
    fplus_solve_plan_free((fplus_solve_plan *)PyCapsule_GetPointer(cap, "fplus_solve_plan"));
}

static PyObject *
py_solve_plan(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *Lp_o, *Li_o, *Up_o, *Ui_o;
    if (!PyArg_ParseTuple(args, "OOOO", &Lp_o, &Li_o, &Up_o, &Ui_o)) return NULL;
    PyArrayObject *Lp = NULL, *Li = NULL, *Up = NULL, *Ui = NULL;
    PyObject *result = NULL;
    if (!(Lp = as_1d(Lp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Lp")) || !(Li = as_1d(Li_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Li"))
            || !(Up = as_1d(Up_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Up")) || !(Ui = as_1d(Ui_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Ui")))
        goto done;
    {
        int64_t n = PyArray_DIM(Lp, 0) - 1;
        if (check_compressed_lengths(n, Lp, Li, NULL, "solve_plan (L)") || check_compressed_lengths(n, Up, Ui, NULL, "solve_plan (U)"))
            goto done;
        fplus_solve_plan *pl = NULL;
        char err[256] = {0};
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_solve_plan_create(n, (const int64_t *)PyArray_DATA(Lp), (const int64_t *)PyArray_DATA(Li),
                                     (const int64_t *)PyArray_DATA(Up), (const int64_t *)PyArray_DATA(Ui), &pl, err, sizeof(err));
        Py_END_ALLOW_THREADS
        if (rc != FPLUS_OK) { raise_fplus(rc, err); goto done; }
        result = PyCapsule_New(pl, "fplus_solve_plan", plan_capsule_free);
        if (!result) fplus_solve_plan_free(pl);
    }
done:
    Py_XDECREF(Lp); Py_XDECREF(Li); Py_XDECREF(Up); Py_XDECREF(Ui);
    return result;
}

static PyObject *
py_lu_solve_plan(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *cap, *Lp_o, *Li_o, *Lx_o, *Up_o, *Ui_o, *Ux_o, *b_o;
    int trans, is_complex;
    if (!PyArg_ParseTuple(args, "OOOOOOOOii", &cap, &Lp_o, &Li_o, &Lx_o, &Up_o, &Ui_o, &Ux_o, &b_o, &trans, &is_complex))
        return NULL;
    const fplus_solve_plan *pl = (const fplus_solve_plan *)PyCapsule_GetPointer(cap, "fplus_solve_plan");
    if (!pl) return NULL;
    int vt = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyArrayObject *Lp = NULL, *Li = NULL, *Lx = NULL, *Up = NULL, *Ui = NULL, *Ux = NULL, *x = NULL;
    PyObject *result = NULL;
    if (!(Lp = as_1d(Lp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Lp")) || !(Li = as_1d(Li_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Li"))
            || !(Lx = as_1d(Lx_o, vt, NPY_ARRAY_IN_ARRAY, "Lx")) || !(Up = as_1d(Up_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Up"))
            || !(Ui = as_1d(Ui_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Ui")) || !(Ux = as_1d(Ux_o, vt, NPY_ARRAY_IN_ARRAY, "Ux")))
        goto done;
    /* b: (n,) or (n, nrhs), copied column-major */
    x = (PyArrayObject *)PyArray_FROM_OTF(b_o, vt, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ENSURECOPY);
    if (!x) goto done;
    if (trans < 0 || trans > 2) { PyErr_SetString(PyExc_ValueError, "lu_solve_plan: trans must be 0, 1 or 2"); goto done; }
    {
        int64_t n = PyArray_DIM(Lp, 0) - 1;
        if (n < 0 || PyArray_DIM(Up, 0) != n + 1 || check_compressed_lengths(n, Lp, Li, Lx, "lu_solve_plan (L)")
                || check_compressed_lengths(n, Up, Ui, Ux, "lu_solve_plan (U)")) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "lu_solve_plan: Lp and Up must both have length n + 1");
            goto done;
        }
        int nd = PyArray_NDIM(x);
        if ((nd != 1 && nd != 2) || PyArray_DIM(x, 0) != n) {
            PyErr_SetString(PyExc_ValueError, "lu_solve_plan: b must have shape (n,) or (n, nrhs)");
            goto done;
        }
        int64_t nrhs = nd == 2 ? PyArray_DIM(x, 1) : 1;
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_tri_solve_plan(pl, n, (const int64_t *)PyArray_DATA(Lp), (const int64_t *)PyArray_DATA(Li), PyArray_DATA(Lx),
                                  (const int64_t *)PyArray_DATA(Up), (const int64_t *)PyArray_DATA(Ui), PyArray_DATA(Ux),
                                  PyArray_DATA(x), nrhs, n, trans, is_complex);
        Py_END_ALLOW_THREADS
        if (rc == FPLUS_ERR_SINGULAR) { PyErr_SetString(PyExc_RuntimeError, "lu_solve_plan: zero or missing diagonal in U"); goto done; }
        if (rc == FPLUS_ERR_BADARG) { PyErr_SetString(PyExc_ValueError, "lu_solve_plan: the plan does not match these factors"); goto done; }
        if (rc != FPLUS_OK) { raise_fplus(rc, "lu_solve_plan failed"); goto done; }
        result = (PyObject *)x;
        Py_INCREF(result);
    }
done:
    Py_XDECREF(Lp); Py_XDECREF(Li); Py_XDECREF(Lx); Py_XDECREF(Up); Py_XDECREF(Ui); Py_XDECREF(Ux);
    Py_XDECREF(x);
    return result;
}

static PyObject *
py_lu_solve(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *Lp_o, *Li_o, *Lx_o, *Up_o, *Ui_o, *Ux_o, *b_o;
    int trans, is_complex;
    if (!PyArg_ParseTuple(args, "OOOOOOOii", &Lp_o, &Li_o, &Lx_o, &Up_o, &Ui_o, &Ux_o, &b_o, &trans, &is_complex))
        return NULL;
    int vt = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyArrayObject *Lp = NULL, *Li = NULL, *Lx = NULL, *Up = NULL, *Ui = NULL, *Ux = NULL, *x = NULL;
    PyObject *result = NULL;
    if (!(Lp = as_1d(Lp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Lp")) || !(Li = as_1d(Li_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Li"))
            || !(Lx = as_1d(Lx_o, vt, NPY_ARRAY_IN_ARRAY, "Lx")) || !(Up = as_1d(Up_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Up"))
            || !(Ui = as_1d(Ui_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Ui")) || !(Ux = as_1d(Ux_o, vt, NPY_ARRAY_IN_ARRAY, "Ux"))
            || !(x = as_1d(b_o, vt, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY, "b")))
        goto done;
    if (trans < 0 || trans > 2) { PyErr_SetString(PyExc_ValueError, "lu_solve: trans must be 0, 1 or 2"); goto done; }
    {
        int64_t n = PyArray_DIM(Lp, 0) - 1;
        if (n < 0 || PyArray_DIM(Up, 0) != n + 1 || check_compressed_lengths(n, Lp, Li, Lx, "lu_solve (L)") || check_compressed_lengths(n, Up, Ui, Ux, "lu_solve (U)")) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "lu_solve: Lp and Up must both have length n + 1");
            goto done;
        }
        if (PyArray_DIM(x, 0) != n) {
            PyErr_SetString(PyExc_ValueError, "lu_solve: b must be a vector of length n");
            goto done;
        }
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_tri_solve(n, (const int64_t *)PyArray_DATA(Lp), (const int64_t *)PyArray_DATA(Li), PyArray_DATA(Lx),
                             (const int64_t *)PyArray_DATA(Up), (const int64_t *)PyArray_DATA(Ui), PyArray_DATA(Ux),
                             PyArray_DATA(x), trans, is_complex);
        Py_END_ALLOW_THREADS
        if (rc != FPLUS_OK) {
            PyErr_SetString(PyExc_RuntimeError, "lu_solve: zero or missing diagonal in U");
            goto done;
        }
        result = (PyObject *)x;
        Py_INCREF(result);
    }
done:
    Py_XDECREF(Lp); Py_XDECREF(Li); Py_XDECREF(Lx); Py_XDECREF(Up); Py_XDECREF(Ui); Py_XDECREF(Ux);
    Py_XDECREF(x);
    return result;
}

static PyObject *
py_coletree_postorder(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *ptr_o, *idx_o; Py_ssize_t m_s, n_s;
    if (!PyArg_ParseTuple(args, "OOnn", &ptr_o, &idx_o, &m_s, &n_s)) return NULL;
    int64_t m = (int64_t)m_s, n = (int64_t)n_s;
    PyArrayObject *ptr_a = NULL, *idx_a = NULL;
    if (m < 0 || !(ptr_a = as_1d(ptr_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr")) || !(idx_a = as_1d(idx_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || check_compressed_lengths(n, ptr_a, idx_a, NULL, "coletree_postorder")) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "coletree_postorder: m must be >= 0");
        Py_XDECREF(ptr_a); Py_XDECREF(idx_a); return NULL;
    }
    npy_intp dn = (npy_intp)n;
    PyArrayObject *post_a = (PyArrayObject *)PyArray_SimpleNew(1, &dn, NPY_INT64);
    if (!post_a) { Py_DECREF(ptr_a); Py_DECREF(idx_a); return NULL; }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = fplus_coletree_postorder(m, n, (const int64_t *)PyArray_DATA(ptr_a), (const int64_t *)PyArray_DATA(idx_a),
                                  (int64_t *)PyArray_DATA(post_a));
    Py_END_ALLOW_THREADS
    Py_DECREF(ptr_a); Py_DECREF(idx_a);
    if (rc == FPLUS_ERR_NOMEM) { Py_DECREF(post_a); return PyErr_NoMemory(); }
    if (rc != FPLUS_OK) { Py_DECREF(post_a); PyErr_SetString(PyExc_RuntimeError, "coletree_postorder: postorder incomplete"); return NULL; }
    return (PyObject *)post_a;
}

static PyObject *
py_analyze(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *ptr_o, *idx_o, *rmask_o = Py_None, *cmask_o = Py_None;
    Py_ssize_t n_s; int pin_dense = 1, postorder = 1, ordering = FPLUS_ORDER_COLAMD; double tau_density = 0.5; long long tau_min_nnz = 10;
    if (!PyArg_ParseTuple(args, "OOn|idLOOii", &ptr_o, &idx_o, &n_s, &pin_dense, &tau_density, &tau_min_nnz, &rmask_o, &cmask_o, &postorder, &ordering))
        return NULL;
    int64_t n = (int64_t)n_s;
    PyArrayObject *ptr_a = NULL, *idx_a = NULL, *rm_a = NULL, *cm_a = NULL;
    PyObject *result = NULL;
    if (!pin_dense) { rmask_o = Py_None; cmask_o = Py_None; }   /* masks are ignored without pinning (as the Python route) */
    if (!(ptr_a = as_1d(ptr_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr")) || !(idx_a = as_1d(idx_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || check_compressed_lengths(n, ptr_a, idx_a, NULL, "analyze")
            || (rmask_o != Py_None && !(rm_a = as_1d(rmask_o, NPY_UINT8, NPY_ARRAY_IN_ARRAY, "row_mask")))
            || (cmask_o != Py_None && !(cm_a = as_1d(cmask_o, NPY_UINT8, NPY_ARRAY_IN_ARRAY, "col_mask"))))
        goto done;
    if ((rm_a && PyArray_DIM(rm_a, 0) != n) || (cm_a && PyArray_DIM(cm_a, 0) != n)) {
        PyErr_SetString(PyExc_ValueError, "analyze: the masks must have length n"); goto done;
    }
    {
        fplus_symbolic *sym = NULL; char err[512] = {0}; int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_analyze_ord(n, (const int64_t *)PyArray_DATA(ptr_a), (const int64_t *)PyArray_DATA(idx_a), pin_dense, tau_density,
                               (int64_t)tau_min_nnz, rm_a ? (const unsigned char *)PyArray_DATA(rm_a) : NULL,
                               cm_a ? (const unsigned char *)PyArray_DATA(cm_a) : NULL, postorder, ordering, &sym, err, sizeof(err));
        Py_END_ALLOW_THREADS
        if (rc != FPLUS_OK) { raise_fplus(rc, err); goto done; }
        npy_intp dn = (npy_intp)n;
        PyArrayObject *rp = (PyArrayObject *)PyArray_SimpleNew(1, &dn, NPY_INT64);
        PyArrayObject *cp = (PyArrayObject *)PyArray_SimpleNew(1, &dn, NPY_INT64);
        if (rp && cp) {
            memcpy(PyArray_DATA(rp), fplus_symbolic_row_perm(sym), (size_t)n * sizeof(int64_t));
            memcpy(PyArray_DATA(cp), fplus_symbolic_col_perm(sym), (size_t)n * sizeof(int64_t));
            result = Py_BuildValue("{s:O,s:O,s:L,s:L}", "row_perm", rp, "col_perm", cp,
                                   "n_dense_rows", (long long)fplus_symbolic_n_dense_rows(sym), "n_dense_cols", (long long)fplus_symbolic_n_dense_cols(sym));
        }
        Py_XDECREF(rp); Py_XDECREF(cp);
        fplus_symbolic_free(sym);
    }
done:
    Py_XDECREF(ptr_a); Py_XDECREF(idx_a); Py_XDECREF(rm_a); Py_XDECREF(cm_a);
    return result;
}

static PyObject *
py_factor_sn_perm(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj, *rp_o, *cp_o;
    Py_ssize_t n_sst;
    int is_complex, equilibrate = 1, ruiz_max_iter = 20, verbose = 0, use_gpu = 0;
    double ruiz_tol = 1e-2, gpu_min_flops = 5e7, pivot_rel = FPLUS_PIVOT_REL_DEFAULT;
    long long relax_rows = 0, panel = 1;
    if (!PyArg_ParseTuple(args, "OOOnOOi|iidiLLidd", &data_obj, &indices_obj, &indptr_obj, &n_sst, &rp_o, &cp_o, &is_complex,
                          &equilibrate, &ruiz_max_iter, &ruiz_tol, &verbose, &relax_rows, &panel, &use_gpu, &gpu_min_flops, &pivot_rel))
        return NULL;
    int64_t n = (int64_t)n_sst;
    PyArrayObject *data_arr = NULL, *idx_arr = NULL, *ptr_arr = NULL, *rp_arr = NULL, *cp_arr = NULL;
    double *Dr = NULL, *Dc = NULL;
    PyObject *result = NULL;
    if (!(data_arr = as_1d(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY, "data"))
            || !(idx_arr = as_1d(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indices"))
            || !(ptr_arr = as_1d(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY, "indptr"))
            || !(rp_arr = as_1d(rp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "row_perm"))
            || !(cp_arr = as_1d(cp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "col_perm"))
            || check_compressed_lengths(n, ptr_arr, idx_arr, data_arr, "factor_sn_perm"))
        goto done;
    if (PyArray_DIM(rp_arr, 0) != n || PyArray_DIM(cp_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "factor_sn_perm: the permutations must have length n"); goto done;
    }
    Dr = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    Dc = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!Dr || !Dc) { PyErr_NoMemory(); goto done; }
    {
        fplus_factors f; char err[512] = {0}; int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_factor_sn_perm(n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr), PyArray_DATA(data_arr),
                                  is_complex, (const int64_t *)PyArray_DATA(rp_arr), (const int64_t *)PyArray_DATA(cp_arr),
                                  equilibrate, ruiz_max_iter, ruiz_tol, verbose, (int64_t)relax_rows, (int64_t)panel, use_gpu,
                                  gpu_min_flops, pivot_rel, &f, Dr, Dc, err, sizeof(err));
        Py_END_ALLOW_THREADS
        if (rc != FPLUS_OK) { raise_fplus(rc, err); goto done; }
        result = factors_to_dict(&f);
        if (!result) goto done;
        PyObject *Dr_arr = wrap_array(Dr, (npy_intp)n, NPY_FLOAT64); if (Dr_arr) Dr = NULL;
        PyObject *Dc_arr = wrap_array(Dc, (npy_intp)n, NPY_FLOAT64); if (Dc_arr) Dc = NULL;
        if (!Dr_arr || !Dc_arr || PyDict_SetItemString(result, "Dr", Dr_arr) || PyDict_SetItemString(result, "Dc", Dc_arr)) {
            Py_CLEAR(result); if (!PyErr_Occurred()) PyErr_NoMemory();
        }
        Py_XDECREF(Dr_arr); Py_XDECREF(Dc_arr);
    }
done:
    free(Dr); free(Dc);
    Py_XDECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); Py_XDECREF(rp_arr); Py_XDECREF(cp_arr);
    return result;
}

/* lu_solve_perm: SparseLU.solve in one call -- gather b through the row (trans 0) or column permutation with its scaling, the
 * plan's sweeps, scatter through the other permutation with the other scaling; the operations of the Python wrapper, in the
 * same order (x = Dc[q] * z scattered to q; transposed: rhs = Dc[q] * b[q], x[p] = Dr[p] * w), without its temporaries.
 * Dr / Dc None: no scaling (equilibrate off).  b: (n,) or (n, k) of the factors' dtype; returns a new array of b's shape. */
static PyObject *
py_lu_solve_perm(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *cap, *Lp_o, *Li_o, *Lx_o, *Up_o, *Ui_o, *Ux_o, *b_o, *p_o, *q_o, *Dr_o, *Dc_o;
    int trans, is_complex;
    if (!PyArg_ParseTuple(args, "OOOOOOOOiiOOOO", &cap, &Lp_o, &Li_o, &Lx_o, &Up_o, &Ui_o, &Ux_o, &b_o, &trans, &is_complex,
                          &p_o, &q_o, &Dr_o, &Dc_o))
        return NULL;
    const fplus_solve_plan *pl = (const fplus_solve_plan *)PyCapsule_GetPointer(cap, "fplus_solve_plan");
    if (!pl) return NULL;
    const int vt = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyArrayObject *Lp = NULL, *Li = NULL, *Lx = NULL, *Up = NULL, *Ui = NULL, *Ux = NULL, *bin = NULL, *pa = NULL, *qa = NULL;
    PyArrayObject *Dra = NULL, *Dca = NULL, *out = NULL;
    PyObject *result = NULL;
    void *w = NULL;
    if (!(Lp = as_1d(Lp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Lp")) || !(Li = as_1d(Li_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Li"))
            || !(Lx = as_1d(Lx_o, vt, NPY_ARRAY_IN_ARRAY, "Lx")) || !(Up = as_1d(Up_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Up"))
            || !(Ui = as_1d(Ui_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "Ui")) || !(Ux = as_1d(Ux_o, vt, NPY_ARRAY_IN_ARRAY, "Ux"))
            || !(pa = as_1d(p_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "p")) || !(qa = as_1d(q_o, NPY_INT64, NPY_ARRAY_IN_ARRAY, "q"))
            || (Dr_o != Py_None && !(Dra = as_1d(Dr_o, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY, "Dr")))
            || (Dc_o != Py_None && !(Dca = as_1d(Dc_o, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY, "Dc"))))
        goto done;
    if (!(bin = (PyArrayObject *)PyArray_FROM_OTF(b_o, vt, NPY_ARRAY_IN_ARRAY))) goto done;
    if (trans < 0 || trans > 2) { PyErr_SetString(PyExc_ValueError, "lu_solve_perm: trans must be 0, 1 or 2"); goto done; }
    {
        const int64_t n = PyArray_DIM(Lp, 0) - 1;
        if (n < 0 || PyArray_DIM(Up, 0) != n + 1 || check_compressed_lengths(n, Lp, Li, Lx, "lu_solve_perm (L)")
                || check_compressed_lengths(n, Up, Ui, Ux, "lu_solve_perm (U)")) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "lu_solve_perm: Lp and Up must both have length n + 1");
            goto done;
        }
        const int nd = PyArray_NDIM(bin);
        if ((nd != 1 && nd != 2) || PyArray_DIM(bin, 0) != n || PyArray_DIM(pa, 0) != n || PyArray_DIM(qa, 0) != n
                || (Dra && PyArray_DIM(Dra, 0) != n) || (Dca && PyArray_DIM(Dca, 0) != n)) {
            PyErr_SetString(PyExc_ValueError, "lu_solve_perm: b must have shape (n,) or (n, nrhs); p, q, Dr, Dc length n");
            goto done;
        }
        const int64_t k = nd == 2 ? PyArray_DIM(bin, 1) : 1;
        const int64_t *P = (const int64_t *)PyArray_DATA(pa), *Q = (const int64_t *)PyArray_DATA(qa);
        for (int64_t i = 0; i < n; i++)
            if (P[i] < 0 || P[i] >= n || Q[i] < 0 || Q[i] >= n) { PyErr_SetString(PyExc_ValueError, "lu_solve_perm: p or q out of range"); goto done; }
        if (!(out = (PyArrayObject *)PyArray_NewLikeArray(bin, NPY_CORDER, NULL, 0))) goto done;
        const size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
        if (!(w = malloc((size_t)(n > 0 ? n : 1) * (size_t)(k > 0 ? k : 1) * vs))) { PyErr_NoMemory(); goto done; }
        /* in: through G with scaling Dg (trans 0: p, Dr; else q, Dc); out: through S with Ds (trans 0: q, Dc; else p, Dr) */
        const int64_t *G = trans == 0 ? P : Q, *Sc = trans == 0 ? Q : P;
        const double *Dg = trans == 0 ? (Dra ? (const double *)PyArray_DATA(Dra) : NULL) : (Dca ? (const double *)PyArray_DATA(Dca) : NULL);
        const double *Ds = trans == 0 ? (Dca ? (const double *)PyArray_DATA(Dca) : NULL) : (Dra ? (const double *)PyArray_DATA(Dra) : NULL);
        int rc;
        Py_BEGIN_ALLOW_THREADS
        if (is_complex) {
            const double complex *bb = (const double complex *)PyArray_DATA(bin); double complex *ww = (double complex *)w, *xx = (double complex *)PyArray_DATA(out);
            for (int64_t r = 0; r < k; r++)
                for (int64_t i = 0; i < n; i++) ww[r * n + i] = Dg ? Dg[G[i]] * bb[G[i] * k + r] : bb[G[i] * k + r];
            rc = fplus_tri_solve_plan(pl, n, (const int64_t *)PyArray_DATA(Lp), (const int64_t *)PyArray_DATA(Li), PyArray_DATA(Lx),
                                      (const int64_t *)PyArray_DATA(Up), (const int64_t *)PyArray_DATA(Ui), PyArray_DATA(Ux), w, k, n, trans, 1);
            if (rc == FPLUS_OK)
                for (int64_t r = 0; r < k; r++)
                    for (int64_t i = 0; i < n; i++) xx[Sc[i] * k + r] = Ds ? Ds[Sc[i]] * ww[r * n + i] : ww[r * n + i];
        } else {
            const double *bb = (const double *)PyArray_DATA(bin); double *ww = (double *)w, *xx = (double *)PyArray_DATA(out);
            for (int64_t r = 0; r < k; r++)
                for (int64_t i = 0; i < n; i++) ww[r * n + i] = Dg ? Dg[G[i]] * bb[G[i] * k + r] : bb[G[i] * k + r];
            rc = fplus_tri_solve_plan(pl, n, (const int64_t *)PyArray_DATA(Lp), (const int64_t *)PyArray_DATA(Li), PyArray_DATA(Lx),
                                      (const int64_t *)PyArray_DATA(Up), (const int64_t *)PyArray_DATA(Ui), PyArray_DATA(Ux), w, k, n, trans, 0);
            if (rc == FPLUS_OK)
                for (int64_t r = 0; r < k; r++)
                    for (int64_t i = 0; i < n; i++) xx[Sc[i] * k + r] = Ds ? Ds[Sc[i]] * ww[r * n + i] : ww[r * n + i];
        }
        Py_END_ALLOW_THREADS
        if (rc == FPLUS_ERR_SINGULAR) { PyErr_SetString(PyExc_RuntimeError, "lu_solve_perm: zero or missing diagonal in U"); goto done; }
        if (rc == FPLUS_ERR_BADARG) { PyErr_SetString(PyExc_ValueError, "lu_solve_perm: the plan does not match these factors"); goto done; }
        if (rc != FPLUS_OK) { raise_fplus(rc, "lu_solve_perm failed"); goto done; }
        result = (PyObject *)out; out = NULL;
    }
done:
    free(w);
    Py_XDECREF(Lp); Py_XDECREF(Li); Py_XDECREF(Lx); Py_XDECREF(Up); Py_XDECREF(Ui); Py_XDECREF(Ux);
    Py_XDECREF(bin); Py_XDECREF(pa); Py_XDECREF(qa); Py_XDECREF(Dra); Py_XDECREF(Dca); Py_XDECREF(out);
    return result;
}

static PyMethodDef SparseLUMethods[] = {
    {"lu_solve_perm", py_lu_solve_perm, METH_VARARGS,
     "lu_solve_perm(plan, Lp, Li, Lx, Up, Ui, Ux, b, trans, is_complex, p, q, Dr, Dc) -> x\n"
     "SparseLU.solve in one call: permutations and scalings (Dr / Dc None: none) around the plan's sweeps.\n"},
    {"analyze", py_analyze, METH_VARARGS,
     "analyze(indptr, indices, n[, pin_dense, tau_density, tau_min_nnz, row_mask, col_mask, postorder, ordering]) -> dict\n"
     "Static orders of a canonical CSC pattern: row_perm, col_perm (A[row_perm][:, col_perm] is factored), n_dense_rows, n_dense_cols.\n"},
    {"factor_sn_perm", py_factor_sn_perm, METH_VARARGS,
     "factor_sn_perm(data, indices, indptr, n, row_perm, col_perm, is_complex[, equilibrate, ruiz_max_iter, ruiz_tol, verbose,\n"
     "               relax_rows, panel, use_gpu, gpu_min_flops, pivot_rel]) -> dict of the factors of the permuted matrix with Dr, Dc\n"},
    {"coletree_postorder", py_coletree_postorder, METH_VARARGS,
     "coletree_postorder(indptr, indices, m, n) -> post: postorder of the column elimination tree of the m x n CSC matrix\n"},
    {"lu_factor", py_lu_factor, METH_VARARGS,
     "lu_factor(data, indices, indptr, n, drop_tol, is_complex) -> dict\n"
     "\n"
     "Sparse LU with full pivoting.  Accepts int32 or int64 index arrays.\n"
     "Returns dict with keys: L_data, L_indices (int64), L_indptr (int64),\n"
     "                        U_data, U_indices (int64), U_indptr (int64),\n"
     "                        p (int64), q (int64).\n"
    },
    {"ruiz_equilibrate", py_ruiz_equilibrate, METH_VARARGS,
     "ruiz_equilibrate(data, indices, indptr, n, max_iter, tol, is_complex[, verbose[, is_csc]])"
     " -> dict\n"
     "\n"
     "Ruiz iterative L-inf equilibration of a sparse matrix.\n"
     "Returns dict with keys:\n"
     "  Dr   : float64 array, length n  (row scaling)\n"
     "  Dc   : float64 array, length n  (column scaling)\n"
     "  data : scaled CSR data array (same dtype as input)\n"
     "After equilibration: diag(Dr) @ A @ diag(Dc) has all row and column\n"
     "L-inf norms close to 1.  indices and indptr are unchanged.\n"
     "To recover the solution to A x = b:\n"
     "  solve (Dr*A*Dc) y = Dr*b  =>  x = Dc * y\n"
    },
    {"lu_factor_sn", py_lu_factor_sn, METH_VARARGS,
     "lu_factor_sn(data, indices, indptr, n, is_complex[, verbose]) -> dict\n"
     "\n"
     "Supernodal left-looking sparse LU with partial pivoting on a CSC\n"
     "matrix.  Returns dict with keys: L_data, L_indices, L_indptr, U_data,\n"
     "U_indices, U_indptr (CSC, int64 indices), p, q (int64), format='csc'.\n"
    },
    {"lu_solve", py_lu_solve, METH_VARARGS,
     "lu_solve(Lp, Li, Lx, Up, Ui, Ux, b, trans, is_complex) -> x\n"
     "\n"
     "Triangular solves on CSC factors: trans 0 solves L U x = b, 1 solves\n"
     "(L U)^T x = b, 2 solves (L U)^H x = b.\n"
    },
    {"solve_plan", py_solve_plan, METH_VARARGS,
     "solve_plan(Lp, Li, Up, Ui) -> plan\n"
     "\n"
     "The blocked-solve plan of CSC factors (L supernodes, U row segments),\n"
     "for lu_solve_plan with the same factor arrays.\n"
    },
    {"lu_solve_plan", py_lu_solve_plan, METH_VARARGS,
     "lu_solve_plan(plan, Lp, Li, Lx, Up, Ui, Ux, b, trans, is_complex) -> x\n"
     "\n"
     "lu_solve with a plan: b of shape (n,) or (n, nrhs), the right-hand sides\n"
     "solved together, threaded inside large supernodes; trans 0 bit-identical\n"
     "to lu_solve.\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_sparse_lu_c",
    "Sparse LU decomposition with full pivoting (C extension)",
    -1,
    SparseLUMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__sparse_lu_c(void)
{
    import_array();
    PyObject *m = PyModule_Create(&moduledef);
    /* the Python layer requires this version, so a stale build cannot silently stand in for the current one (audit 2026-10-10) */
    if (m && PyModule_AddIntConstant(m, "API_VERSION", 5) != 0) { Py_DECREF(m); return NULL; }
    return m;
}
