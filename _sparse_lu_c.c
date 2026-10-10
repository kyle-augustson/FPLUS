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
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!data_arr) return NULL;
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!idx_arr || !ptr_arr) { Py_DECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); return NULL; }
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
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    /* the data are copied, so the scaling can work in place */
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64,
                                                                NPY_ARRAY_ENSURECOPY | NPY_ARRAY_IN_ARRAY);
    if (!idx_arr || !ptr_arr || !data_arr) { Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); Py_XDECREF(data_arr); return NULL; }
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
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!data_arr) return NULL;
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!idx_arr || !ptr_arr) { Py_DECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); return NULL; }
    fplus_factors f; char err[512] = {0}; int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = fplus_factor_sn(n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr), PyArray_DATA(data_arr),
                         is_complex, verbose, (int64_t)relax_rows, (int64_t)panel, use_gpu, gpu_min_flops, pivot_rel, &f, err, sizeof(err));
    Py_END_ALLOW_THREADS
    Py_DECREF(data_arr); Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    if (rc != FPLUS_OK) return raise_fplus(rc, err);
    return factors_to_dict(&f);
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
    PyArrayObject *Lp = (PyArrayObject *)PyArray_FROM_OTF(Lp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Li = (PyArrayObject *)PyArray_FROM_OTF(Li_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Lx = (PyArrayObject *)PyArray_FROM_OTF(Lx_o, vt, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Up = (PyArrayObject *)PyArray_FROM_OTF(Up_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Ui = (PyArrayObject *)PyArray_FROM_OTF(Ui_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *Ux = (PyArrayObject *)PyArray_FROM_OTF(Ux_o, vt, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *x  = (PyArrayObject *)PyArray_FROM_OTF(b_o, vt, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY);
    PyObject *result = NULL;
    if (!Lp || !Li || !Lx || !Up || !Ui || !Ux || !x) goto done;
    {
        int64_t n = PyArray_DIM(Lp, 0) - 1;
        if (PyArray_NDIM(x) != 1 || PyArray_DIM(x, 0) != n) {
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
    PyArrayObject *ptr_a = (PyArrayObject *)PyArray_FROM_OTF(ptr_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *idx_a = (PyArrayObject *)PyArray_FROM_OTF(idx_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!ptr_a || !idx_a) { Py_XDECREF(ptr_a); Py_XDECREF(idx_a); return NULL; }
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
    Py_ssize_t n_s; int pin_dense = 1, postorder = 1; double tau_density = 0.5; long long tau_min_nnz = 10;
    if (!PyArg_ParseTuple(args, "OOn|idLOOi", &ptr_o, &idx_o, &n_s, &pin_dense, &tau_density, &tau_min_nnz, &rmask_o, &cmask_o, &postorder))
        return NULL;
    int64_t n = (int64_t)n_s;
    PyArrayObject *ptr_a = (PyArrayObject *)PyArray_FROM_OTF(ptr_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *idx_a = (PyArrayObject *)PyArray_FROM_OTF(idx_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *rm_a = rmask_o == Py_None ? NULL : (PyArrayObject *)PyArray_FROM_OTF(rmask_o, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cm_a = cmask_o == Py_None ? NULL : (PyArrayObject *)PyArray_FROM_OTF(cmask_o, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *result = NULL;
    if (!ptr_a || !idx_a || (rmask_o != Py_None && !rm_a) || (cmask_o != Py_None && !cm_a)) goto done;
    if (PyArray_DIM(ptr_a, 0) != n + 1 || (rm_a && PyArray_DIM(rm_a, 0) != n) || (cm_a && PyArray_DIM(cm_a, 0) != n)) {
        PyErr_SetString(PyExc_ValueError, "analyze: indptr must have length n + 1 and the masks length n"); goto done;
    }
    {
        fplus_symbolic *sym = NULL; char err[512] = {0}; int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = fplus_analyze(n, (const int64_t *)PyArray_DATA(ptr_a), (const int64_t *)PyArray_DATA(idx_a), pin_dense, tau_density,
                           (int64_t)tau_min_nnz, rm_a ? (const unsigned char *)PyArray_DATA(rm_a) : NULL,
                           cm_a ? (const unsigned char *)PyArray_DATA(cm_a) : NULL, postorder, &sym, err, sizeof(err));
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
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(data_obj, is_complex ? NPY_COMPLEX128 : NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *rp_arr = (PyArrayObject *)PyArray_FROM_OTF(rp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cp_arr = (PyArrayObject *)PyArray_FROM_OTF(cp_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    double *Dr = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *Dc = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    PyObject *result = NULL;
    if (!data_arr || !idx_arr || !ptr_arr || !rp_arr || !cp_arr) goto done;
    if (!Dr || !Dc) { PyErr_NoMemory(); goto done; }
    if (PyArray_DIM(ptr_arr, 0) != n + 1 || PyArray_DIM(rp_arr, 0) != n || PyArray_DIM(cp_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "factor_sn_perm: indptr must have length n + 1 and the permutations length n"); goto done;
    }
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

static PyMethodDef SparseLUMethods[] = {
    {"analyze", py_analyze, METH_VARARGS,
     "analyze(indptr, indices, n[, pin_dense, tau_density, tau_min_nnz, row_mask, col_mask, postorder]) -> dict\n"
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
    return PyModule_Create(&moduledef);
}
