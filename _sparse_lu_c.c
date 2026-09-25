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

#include <complex.h>    /* C99: double complex, cabs */
#include <math.h>       /* fabs */
#include <stdint.h>     /* int64_t */
#include <string.h>     /* memcpy, memmove */
#include <stdlib.h>     /* malloc, realloc, free */
#include <stdio.h>      /* printf, fflush (verbose mode) */
#include <time.h>       /* clock_gettime (verbose-mode timing) */

/* Activate OpenMP if available (compile with -fopenmp). */
#ifdef _OPENMP
#  include <omp.h>
#endif

/* ------------------------------------------------------------------------- *
 * Optional BLAS support (compile with SPARSE_LU_BLAS=1).
 *
 * When enabled, row_axpy dispatches to cblas_zaxpy / cblas_daxpy for rows
 * whose union size exceeds BLAS_AXPY_THRESHOLD.  For shorter rows, the
 * scalar merge path is used since BLAS overhead (gather + scatter +
 * function-call cost) exceeds the vectorised arithmetic gain.
 *
 * The threshold can be overridden at compile time:
 *   gcc ... -DBLAS_AXPY_THRESHOLD=64 ...
 * ------------------------------------------------------------------------- */
#ifdef HAVE_BLAS
#  ifndef BLAS_AXPY_THRESHOLD
#    define BLAS_AXPY_THRESHOLD 32
#  endif
/* Standard CBLAS prototypes — work with libblas, OpenBLAS, MKL, etc.
 * We declare them here rather than including <cblas.h> because the header
 * location varies across distributions (sometimes /usr/include/cblas.h,
 * sometimes /usr/include/x86_64-linux-gnu/cblas.h, sometimes nested in
 * an openblas/ subdirectory).  The signatures are stable across all
 * standard CBLAS implementations.                                          */
extern void cblas_daxpy(int N, double alpha,
                         const double *X, int incX,
                         double *Y, int incY);
extern void cblas_zaxpy(int N, const void *alpha,
                         const void *X, int incX,
                         void *Y, int incY);
/* the dense kernels of the supernodal update (CBLAS enum values:
 * ColMajor 102, NoTrans 111, Lower 122, Unit 132) */
extern void cblas_dtrsv(int order, int uplo, int trans, int diag, int N,
                        const double *A, int lda, double *X, int incX);
extern void cblas_ztrsv(int order, int uplo, int trans, int diag, int N,
                        const void *A, int lda, void *X, int incX);
extern void cblas_dgemv(int order, int trans, int M, int N, double alpha,
                        const double *A, int lda, const double *X, int incX,
                        double beta, double *Y, int incY);
extern void cblas_zgemv(int order, int trans, int M, int N, const void *alpha,
                        const void *A, int lda, const void *X, int incX,
                        const void *beta, void *Y, int incY);
#  ifndef SN_BLAS_MIN
#    define SN_BLAS_MIN 256   /* segment columns x rest rows below which the loops are used */
#  endif
#endif

/* ------------------------------------------------------------------------- *
 * Verbose-mode helpers
 * Wall-clock seconds since first call to wall_time() in this function.
 * Implementation note: clock_gettime(CLOCK_MONOTONIC) on POSIX, falls back
 * to clock() on Windows / non-POSIX systems.  Only used when verbose >= 1.
 * ------------------------------------------------------------------------- */
static inline double wall_time(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* Extra capacity added to each row on first allocation. */
#define INITIAL_SLACK 4

/* =========================================================================
 * Per-row dynamic sparse vector
 * ========================================================================= */
typedef struct {
    int64_t        *cols;    /* sorted column indices, heap-allocated */
    double complex *vals_z;  /* complex values  (NULL for real)       */
    double         *vals_d;  /* real values     (NULL for complex)    */
    int64_t         nnz;     /* stored entry count                    */
    int64_t         cap;     /* allocated capacity                    */
} SparseRow;

/* =========================================================================
 * Memory helpers
 * ========================================================================= */
static int sr_alloc(SparseRow *r, int64_t cap, int is_complex)
{
    r->cols   = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    r->vals_z = NULL;
    r->vals_d = NULL;
    if (is_complex)
        r->vals_z = (double complex *)malloc(
                        (size_t)cap * sizeof(double complex));
    else
        r->vals_d = (double *)malloc((size_t)cap * sizeof(double));

    r->nnz = 0;
    r->cap = cap;

    if (!r->cols || (is_complex ? !r->vals_z : !r->vals_d)) {
        free(r->cols); free(r->vals_z); free(r->vals_d);
        return -1;
    }
    return 0;
}

static void sr_free(SparseRow *r)
{
    free(r->cols); free(r->vals_z); free(r->vals_d);
    r->cols = NULL; r->vals_z = NULL; r->vals_d = NULL;
    r->nnz = r->cap = 0;
}

static int sr_ensure_cap(SparseRow *r, int64_t needed, int is_complex)
{
    if (r->cap >= needed) return 0;
    int64_t nc = r->cap * 2;
    if (nc < needed) nc = needed;

    int64_t *c = (int64_t *)realloc(r->cols, (size_t)nc * sizeof(int64_t));
    if (!c) return -1;
    r->cols = c;

    if (is_complex) {
        double complex *v = (double complex *)realloc(
                r->vals_z, (size_t)nc * sizeof(double complex));
        if (!v) return -1;
        r->vals_z = v;
    } else {
        double *v = (double *)realloc(r->vals_d, (size_t)nc * sizeof(double));
        if (!v) return -1;
        r->vals_d = v;
    }
    r->cap = nc;
    return 0;
}

static inline void sr_swap(SparseRow *a, SparseRow *b)
{
    SparseRow t = *a; *a = *b; *b = t;
}

static void free_rows(SparseRow *rows, int64_t n)
{
    if (!rows) return;
    for (int64_t i = 0; i < n; i++) sr_free(&rows[i]);
    free(rows);
}

/* =========================================================================
 * Binary search: find target in sorted cols[0..nnz-1].
 * Returns index if found, -1 otherwise.
 * ========================================================================= */
static inline int64_t bsearch_i64(const int64_t * restrict cols,
                                   int64_t target, int64_t nnz)
{
    int64_t lo = 0, hi = nnz - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t c   = cols[mid];
        if      (c == target) return mid;
        else if (c  < target) lo = mid + 1;
        else                  hi = mid - 1;
    }
    return -1;
}

/* =========================================================================
 * Insertion-sort fixup after cols[pos] was renamed.
 * Bubbles in only the direction needed; one direction ever moves.
 * ========================================================================= */
static inline void insertion_fix_z(SparseRow * restrict r, int64_t pos)
{
    int64_t       *cols = r->cols;
    double complex *vals = r->vals_z;
    int64_t nnz = r->nnz, p = pos;

    while (p > 0 && cols[p] < cols[p-1]) {
        int64_t        tc = cols[p]; cols[p] = cols[p-1]; cols[p-1] = tc;
        double complex tv = vals[p]; vals[p] = vals[p-1]; vals[p-1] = tv;
        p--;
    }
    p = pos;
    while (p < nnz-1 && cols[p] > cols[p+1]) {
        int64_t        tc = cols[p]; cols[p] = cols[p+1]; cols[p+1] = tc;
        double complex tv = vals[p]; vals[p] = vals[p+1]; vals[p+1] = tv;
        p++;
    }
}

static inline void insertion_fix_d(SparseRow * restrict r, int64_t pos)
{
    int64_t *cols = r->cols;
    double  *vals = r->vals_d;
    int64_t nnz = r->nnz, p = pos;

    while (p > 0 && cols[p] < cols[p-1]) {
        int64_t tc = cols[p]; cols[p] = cols[p-1]; cols[p-1] = tc;
        double  tv = vals[p]; vals[p] = vals[p-1]; vals[p-1] = tv;
        p--;
    }
    p = pos;
    while (p < nnz-1 && cols[p] > cols[p+1]) {
        int64_t tc = cols[p]; cols[p] = cols[p+1]; cols[p+1] = tc;
        double  tv = vals[p]; vals[p] = vals[p+1]; vals[p+1] = tv;
        p++;
    }
}

/* =========================================================================
 * Delete entry at pos in row r (shift-left by one).
 * ========================================================================= */
static inline void sr_delete_z(SparseRow *r, int64_t pos)
{
    int64_t tail = r->nnz - pos - 1;
    if (tail > 0) {
        memmove(r->cols   + pos, r->cols   + pos + 1,
                (size_t)tail * sizeof(int64_t));
        memmove(r->vals_z + pos, r->vals_z + pos + 1,
                (size_t)tail * sizeof(double complex));
    }
    r->nnz--;
}

static inline void sr_delete_d(SparseRow *r, int64_t pos)
{
    int64_t tail = r->nnz - pos - 1;
    if (tail > 0) {
        memmove(r->cols   + pos, r->cols   + pos + 1,
                (size_t)tail * sizeof(int64_t));
        memmove(r->vals_d + pos, r->vals_d + pos + 1,
                (size_t)tail * sizeof(double));
    }
    r->nnz--;
}

/* =========================================================================
 * Insert (col j, value v) into a sorted row, maintaining order.
 * Assumes j is not already present.
 * ========================================================================= */
static int sr_insert_z(SparseRow *r, int64_t j, double complex v)
{
    if (sr_ensure_cap(r, r->nnz + 1, 1) != 0) return -1;
    int64_t pos = r->nnz;
    while (pos > 0 && r->cols[pos-1] > j) {
        r->cols[pos]   = r->cols[pos-1];
        r->vals_z[pos] = r->vals_z[pos-1];
        pos--;
    }
    r->cols[pos]   = j;
    r->vals_z[pos] = v;
    r->nnz++;
    return 0;
}

static int sr_insert_d(SparseRow *r, int64_t j, double v)
{
    if (sr_ensure_cap(r, r->nnz + 1, 0) != 0) return -1;
    int64_t pos = r->nnz;
    while (pos > 0 && r->cols[pos-1] > j) {
        r->cols[pos]   = r->cols[pos-1];
        r->vals_d[pos] = r->vals_d[pos-1];
        pos--;
    }
    r->cols[pos]   = j;
    r->vals_d[pos] = v;
    r->nnz++;
    return 0;
}

/* =========================================================================
 * Column swap: rename columns j1 <-> j2 across all n rows.
 *
 * Both present in a row: the labels are at their sorted positions already;
 * renaming j1<->j2 is equivalent to swapping the VALUES — no re-sort needed.
 * One present: rename the label and call insertion_fix to restore sort order.
 *
 * OpenMP: each row is independent.
 * ========================================================================= */
static void col_swap_z(SparseRow * restrict M, int64_t n,
                       int64_t j1, int64_t j2)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        SparseRow *r = &M[i];
        int64_t p1 = bsearch_i64(r->cols, j1, r->nnz);
        int64_t p2 = bsearch_i64(r->cols, j2, r->nnz);
        if (p1 < 0 && p2 < 0) continue;
        if (p1 >= 0 && p2 >= 0) {
            double complex tv = r->vals_z[p1];
            r->vals_z[p1] = r->vals_z[p2];
            r->vals_z[p2] = tv;
        } else if (p1 >= 0) {
            r->cols[p1] = j2;
            insertion_fix_z(r, p1);
        } else {
            r->cols[p2] = j1;
            insertion_fix_z(r, p2);
        }
    }
}

static void col_swap_d(SparseRow * restrict M, int64_t n,
                       int64_t j1, int64_t j2)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        SparseRow *r = &M[i];
        int64_t p1 = bsearch_i64(r->cols, j1, r->nnz);
        int64_t p2 = bsearch_i64(r->cols, j2, r->nnz);
        if (p1 < 0 && p2 < 0) continue;
        if (p1 >= 0 && p2 >= 0) {
            double tv = r->vals_d[p1];
            r->vals_d[p1] = r->vals_d[p2];
            r->vals_d[p2] = tv;
        } else if (p1 >= 0) {
            r->cols[p1] = j2;
            insertion_fix_d(r, p1);
        } else {
            r->cols[p2] = j1;
            insertion_fix_d(r, p2);
        }
    }
}

/* =========================================================================
 * Full-pivot scan: find (best_i, best_j) in M[k:n][k:n] with max |value|.
 *
 * OpenMP: each thread scans a contiguous band of rows and tracks a local
 * best.  After the parallel region a serial reduction picks the global best.
 * Returns 1 if a nonzero pivot was found, 0 if the submatrix is all-zero
 * (singular).
 * ========================================================================= */
static int find_full_pivot_z(SparseRow * restrict M, int64_t k, int64_t n,
                              int64_t *best_i_out, int64_t *best_j_out,
                              double complex *best_val_out)
{
    int64_t       g_bi = -1, g_bj = -1;
    double        g_babs = -1.0;
    double complex g_bval = 0.0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    /* Thread-local best: stored in a small array to avoid false sharing. */
    /* Pad each slot to a cache line (64 bytes). */
    typedef struct { int64_t bi, bj; double babs; double complex bval;
                     char pad[64 - sizeof(int64_t)*2
                                  - sizeof(double)
                                  - sizeof(double complex)]; } TLocal;
    TLocal *tl = (TLocal *)calloc((size_t)nthreads, sizeof(TLocal));
    if (!tl) return -1;   /* OOM: caller checks */
    for (int t = 0; t < nthreads; t++) { tl[t].bi = -1; tl[t].babs = -1.0; }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t l_bi = -1, l_bj = -1;
        double  l_babs = -1.0;
        double complex l_bval = 0.0;

#pragma omp for schedule(static) nowait
        for (int64_t i = k; i < n; i++) {
            const SparseRow *r = &M[i];
            for (int64_t p = 0; p < r->nnz; p++) {
                if (r->cols[p] < k) continue;
                double av = cabs(r->vals_z[p]);
                if (av > l_babs) {
                    l_babs = av; l_bi = i;
                    l_bj   = r->cols[p]; l_bval = r->vals_z[p];
                }
            }
        }
        tl[tid].bi = l_bi; tl[tid].bj = l_bj;
        tl[tid].babs = l_babs; tl[tid].bval = l_bval;
    }

    /* Serial reduction over thread-local results. */
    for (int t = 0; t < nthreads; t++) {
        if (tl[t].babs > g_babs) {
            g_babs = tl[t].babs; g_bi   = tl[t].bi;
            g_bj   = tl[t].bj;  g_bval = tl[t].bval;
        }
    }
    free(tl);
#else
    for (int64_t i = k; i < n; i++) {
        const SparseRow *r = &M[i];
        for (int64_t p = 0; p < r->nnz; p++) {
            if (r->cols[p] < k) continue;
            double av = cabs(r->vals_z[p]);
            if (av > g_babs) {
                g_babs = av; g_bi = i;
                g_bj   = r->cols[p]; g_bval = r->vals_z[p];
            }
        }
    }
#endif

    *best_i_out   = g_bi;
    *best_j_out   = g_bj;
    *best_val_out = g_bval;
    return (g_bi >= 0 && g_babs > 0.0);
}

static int find_full_pivot_d(SparseRow * restrict M, int64_t k, int64_t n,
                              int64_t *best_i_out, int64_t *best_j_out,
                              double *best_val_out)
{
    int64_t g_bi = -1, g_bj = -1;
    double  g_babs = -1.0, g_bval = 0.0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    typedef struct { int64_t bi, bj; double babs, bval;
                     char pad[64 - sizeof(int64_t)*2 - sizeof(double)*2]; } TLocal;
    TLocal *tl = (TLocal *)calloc((size_t)nthreads, sizeof(TLocal));
    if (!tl) return -1;
    for (int t = 0; t < nthreads; t++) { tl[t].bi = -1; tl[t].babs = -1.0; }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t l_bi = -1, l_bj = -1;
        double  l_babs = -1.0, l_bval = 0.0;

#pragma omp for schedule(static) nowait
        for (int64_t i = k; i < n; i++) {
            const SparseRow *r = &M[i];
            for (int64_t p = 0; p < r->nnz; p++) {
                if (r->cols[p] < k) continue;
                double av = fabs(r->vals_d[p]);
                if (av > l_babs) {
                    l_babs = av; l_bi = i;
                    l_bj   = r->cols[p]; l_bval = r->vals_d[p];
                }
            }
        }
        tl[tid].bi = l_bi; tl[tid].bj = l_bj;
        tl[tid].babs = l_babs; tl[tid].bval = l_bval;
    }
    for (int t = 0; t < nthreads; t++) {
        if (tl[t].babs > g_babs) {
            g_babs = tl[t].babs; g_bi   = tl[t].bi;
            g_bj   = tl[t].bj;  g_bval = tl[t].bval;
        }
    }
    free(tl);
#else
    for (int64_t i = k; i < n; i++) {
        const SparseRow *r = &M[i];
        for (int64_t p = 0; p < r->nnz; p++) {
            if (r->cols[p] < k) continue;
            double av = fabs(r->vals_d[p]);
            if (av > g_babs) {
                g_babs = av; g_bi = i;
                g_bj   = r->cols[p]; g_bval = r->vals_d[p];
            }
        }
    }
#endif

    *best_i_out   = g_bi;
    *best_j_out   = g_bj;
    *best_val_out = g_bval;
    return (g_bi >= 0 && g_babs > 0.0);
}

/* =========================================================================
 * Partial pivot search (column-only).
 *
 * Find the row index i in [k..n-1] that has the maximum-magnitude entry
 * in column k of the active submatrix.  No column swap is performed, so
 * the column ordering set up by COLAMD is preserved through factorisation.
 *
 * Critical structure-preservation property: because column k is fixed,
 * elimination cannot fill in entries outside the existing column-k pattern
 * combined with row-k's nonzero columns.  This preserves bandedness when
 * COLAMD has produced a fill-reducing ordering — exactly what full pivoting
 * destroys via column swaps.
 *
 * Returns: 1 = pivot found and nonzero, 0 = singular (no nonzero in column k),
 *          -1 = OOM in OpenMP scratch (bubbled up to caller).
 * ========================================================================= */
static int find_partial_pivot_z(SparseRow * restrict M, int64_t k, int64_t n,
                                 int64_t *best_i_out,
                                 double complex *best_val_out)
{
    int64_t       g_bi = -1;
    double        g_babs = -1.0;
    double complex g_bval = 0.0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    typedef struct { int64_t bi; double babs; double complex bval;
                     char pad[64 - sizeof(int64_t) - sizeof(double)
                                  - sizeof(double complex)]; } TLocal;
    TLocal *tl = (TLocal *)calloc((size_t)nthreads, sizeof(TLocal));
    if (!tl) return -1;
    for (int t = 0; t < nthreads; t++) { tl[t].bi = -1; tl[t].babs = -1.0; }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t l_bi = -1;
        double  l_babs = -1.0;
        double complex l_bval = 0.0;

#pragma omp for schedule(static) nowait
        for (int64_t i = k; i < n; i++) {
            int64_t pos = bsearch_i64(M[i].cols, k, M[i].nnz);
            if (pos < 0) continue;
            double av = cabs(M[i].vals_z[pos]);
            if (av > l_babs) {
                l_babs = av; l_bi = i; l_bval = M[i].vals_z[pos];
            }
        }
        tl[tid].bi = l_bi; tl[tid].babs = l_babs; tl[tid].bval = l_bval;
    }

    for (int t = 0; t < nthreads; t++) {
        if (tl[t].babs > g_babs) {
            g_babs = tl[t].babs; g_bi = tl[t].bi; g_bval = tl[t].bval;
        }
    }
    free(tl);
#else
    for (int64_t i = k; i < n; i++) {
        int64_t pos = bsearch_i64(M[i].cols, k, M[i].nnz);
        if (pos < 0) continue;
        double av = cabs(M[i].vals_z[pos]);
        if (av > g_babs) {
            g_babs = av; g_bi = i; g_bval = M[i].vals_z[pos];
        }
    }
#endif

    *best_i_out   = g_bi;
    *best_val_out = g_bval;
    return (g_bi >= 0 && g_babs > 0.0);
}

static int find_partial_pivot_d(SparseRow * restrict M, int64_t k, int64_t n,
                                 int64_t *best_i_out,
                                 double *best_val_out)
{
    int64_t g_bi = -1;
    double  g_babs = -1.0;
    double  g_bval = 0.0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    typedef struct { int64_t bi; double babs; double bval;
                     char pad[64 - sizeof(int64_t) - 2*sizeof(double)]; } TLocal;
    TLocal *tl = (TLocal *)calloc((size_t)nthreads, sizeof(TLocal));
    if (!tl) return -1;
    for (int t = 0; t < nthreads; t++) { tl[t].bi = -1; tl[t].babs = -1.0; }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t l_bi = -1;
        double  l_babs = -1.0;
        double  l_bval = 0.0;

#pragma omp for schedule(static) nowait
        for (int64_t i = k; i < n; i++) {
            int64_t pos = bsearch_i64(M[i].cols, k, M[i].nnz);
            if (pos < 0) continue;
            double av = fabs(M[i].vals_d[pos]);
            if (av > l_babs) {
                l_babs = av; l_bi = i; l_bval = M[i].vals_d[pos];
            }
        }
        tl[tid].bi = l_bi; tl[tid].babs = l_babs; tl[tid].bval = l_bval;
    }

    for (int t = 0; t < nthreads; t++) {
        if (tl[t].babs > g_babs) {
            g_babs = tl[t].babs; g_bi = tl[t].bi; g_bval = tl[t].bval;
        }
    }
    free(tl);
#else
    for (int64_t i = k; i < n; i++) {
        int64_t pos = bsearch_i64(M[i].cols, k, M[i].nnz);
        if (pos < 0) continue;
        double av = fabs(M[i].vals_d[pos]);
        if (av > g_babs) {
            g_babs = av; g_bi = i; g_bval = M[i].vals_d[pos];
        }
    }
#endif

    *best_i_out   = g_bi;
    *best_val_out = g_bval;
    return (g_bi >= 0 && g_babs > 0.0);
}

/* =========================================================================
 * Rook pivot search.
 *
 * Iteratively alternate between scanning column ci for the maximum-magnitude
 * entry (giving a candidate row ri) and scanning row ri for the maximum-
 * magnitude entry (giving a candidate column ci'), until ci' == ci.  At
 * convergence, the pivot is the maximum in BOTH its row and its column —
 * the "rook" condition.
 *
 * Properties:
 *   - Multipliers |m| <= 1 along both row and column dimensions, providing
 *     better numerical stability than partial pivoting for asymmetric or
 *     poorly-scaled matrices.
 *   - Typically converges in 1-2 iterations; bounded above by min(n-k, max_iter).
 *   - Searches a restricted region (one row + one column per iter) rather
 *     than the entire active submatrix, so it preserves much more structure
 *     than full pivoting.
 *
 * Returns: 1 = pivot found, 0 = singular, -1 = OOM.
 * ========================================================================= */
static int find_rook_pivot_z(SparseRow * restrict M, int64_t k, int64_t n,
                              int64_t *best_i_out, int64_t *best_j_out,
                              double complex *best_val_out)
{
    /* Start with column k; find max |entry| in column k over rows >= k. */
    int64_t ci = k;
    int64_t ri = -1;
    double complex bval = 0.0;
    double babs = -1.0;
    const int max_iter = 8;

    for (int iter = 0; iter < max_iter; iter++) {
        /* --- Scan column ci over rows >= k for max |entry|. ------------- */
        int64_t new_ri = -1;
        double  col_babs = -1.0;
        double complex col_bval = 0.0;
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        typedef struct { int64_t ri; double babs; double complex bval;
                         char pad[64 - sizeof(int64_t) - sizeof(double)
                                      - sizeof(double complex)]; } TL;
        TL *tl = (TL *)calloc((size_t)nthreads, sizeof(TL));
        if (!tl) return -1;
        for (int t = 0; t < nthreads; t++) { tl[t].ri = -1; tl[t].babs = -1.0; }
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t l_ri = -1;
            double  l_babs = -1.0;
            double complex l_bval = 0.0;
#pragma omp for schedule(static) nowait
            for (int64_t i = k; i < n; i++) {
                int64_t pos = bsearch_i64(M[i].cols, ci, M[i].nnz);
                if (pos < 0) continue;
                double av = cabs(M[i].vals_z[pos]);
                if (av > l_babs) {
                    l_babs = av; l_ri = i; l_bval = M[i].vals_z[pos];
                }
            }
            tl[tid].ri = l_ri; tl[tid].babs = l_babs; tl[tid].bval = l_bval;
        }
        for (int t = 0; t < nthreads; t++) {
            if (tl[t].babs > col_babs) {
                col_babs = tl[t].babs; new_ri = tl[t].ri; col_bval = tl[t].bval;
            }
        }
        free(tl);
#else
        for (int64_t i = k; i < n; i++) {
            int64_t pos = bsearch_i64(M[i].cols, ci, M[i].nnz);
            if (pos < 0) continue;
            double av = cabs(M[i].vals_z[pos]);
            if (av > col_babs) {
                col_babs = av; new_ri = i; col_bval = M[i].vals_z[pos];
            }
        }
#endif
        if (new_ri < 0) {
            /* No nonzero in column ci; matrix may be structurally singular. */
            *best_i_out = -1;
            *best_j_out = -1;
            *best_val_out = 0.0;
            return 0;
        }
        ri   = new_ri;
        bval = col_bval;
        babs = col_babs;

        /* --- Scan row ri over cols >= k for max |entry|. ---------------- */
        int64_t new_ci = -1;
        double  row_babs = -1.0;
        double complex row_bval = 0.0;
        const SparseRow *r = &M[ri];
        for (int64_t p = 0; p < r->nnz; p++) {
            if (r->cols[p] < k) continue;
            double av = cabs(r->vals_z[p]);
            if (av > row_babs) {
                row_babs = av; new_ci = r->cols[p]; row_bval = r->vals_z[p];
            }
        }
        if (new_ci < 0) return 0;

        /* Rook condition: max-in-column found also max-in-row at same column? */
        if (new_ci == ci) {
            *best_i_out = ri;
            *best_j_out = ci;
            *best_val_out = bval;
            return (babs > 0.0);
        }
        ci   = new_ci;
        bval = row_bval;
        babs = row_babs;
    }

    /* Hit max_iter without converging — accept current candidate.  This is
     * extremely rare; only matrices with cyclic ties of equal magnitude.   */
    *best_i_out = ri;
    *best_j_out = ci;
    *best_val_out = bval;
    return (babs > 0.0);
}

static int find_rook_pivot_d(SparseRow * restrict M, int64_t k, int64_t n,
                              int64_t *best_i_out, int64_t *best_j_out,
                              double *best_val_out)
{
    int64_t ci = k;
    int64_t ri = -1;
    double  bval = 0.0;
    double  babs = -1.0;
    const int max_iter = 8;

    for (int iter = 0; iter < max_iter; iter++) {
        int64_t new_ri = -1;
        double  col_babs = -1.0;
        double  col_bval = 0.0;
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        typedef struct { int64_t ri; double babs; double bval;
                         char pad[64 - sizeof(int64_t) - 2*sizeof(double)]; } TL;
        TL *tl = (TL *)calloc((size_t)nthreads, sizeof(TL));
        if (!tl) return -1;
        for (int t = 0; t < nthreads; t++) { tl[t].ri = -1; tl[t].babs = -1.0; }
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t l_ri = -1;
            double  l_babs = -1.0;
            double  l_bval = 0.0;
#pragma omp for schedule(static) nowait
            for (int64_t i = k; i < n; i++) {
                int64_t pos = bsearch_i64(M[i].cols, ci, M[i].nnz);
                if (pos < 0) continue;
                double av = fabs(M[i].vals_d[pos]);
                if (av > l_babs) {
                    l_babs = av; l_ri = i; l_bval = M[i].vals_d[pos];
                }
            }
            tl[tid].ri = l_ri; tl[tid].babs = l_babs; tl[tid].bval = l_bval;
        }
        for (int t = 0; t < nthreads; t++) {
            if (tl[t].babs > col_babs) {
                col_babs = tl[t].babs; new_ri = tl[t].ri; col_bval = tl[t].bval;
            }
        }
        free(tl);
#else
        for (int64_t i = k; i < n; i++) {
            int64_t pos = bsearch_i64(M[i].cols, ci, M[i].nnz);
            if (pos < 0) continue;
            double av = fabs(M[i].vals_d[pos]);
            if (av > col_babs) {
                col_babs = av; new_ri = i; col_bval = M[i].vals_d[pos];
            }
        }
#endif
        if (new_ri < 0) {
            *best_i_out = -1; *best_j_out = -1; *best_val_out = 0.0;
            return 0;
        }
        ri = new_ri; bval = col_bval; babs = col_babs;

        int64_t new_ci = -1;
        double  row_babs = -1.0;
        double  row_bval = 0.0;
        const SparseRow *r = &M[ri];
        for (int64_t p = 0; p < r->nnz; p++) {
            if (r->cols[p] < k) continue;
            double av = fabs(r->vals_d[p]);
            if (av > row_babs) {
                row_babs = av; new_ci = r->cols[p]; row_bval = r->vals_d[p];
            }
        }
        if (new_ci < 0) return 0;
        if (new_ci == ci) {
            *best_i_out = ri; *best_j_out = ci; *best_val_out = bval;
            return (babs > 0.0);
        }
        ci = new_ci; bval = row_bval; babs = row_babs;
    }
    *best_i_out = ri; *best_j_out = ci; *best_val_out = bval;
    return (babs > 0.0);
}

/* =========================================================================
 * Sparse row merge-axpy:  M[row_i]  +=  alpha * M[k]
 *
 * Why a sparse merge rather than BLAS zaxpy:
 *   BLAS axpy requires dense contiguous vectors.  Using it here would need
 *   a gather-to-dense step, a BLAS call, and a scatter-back step, with a
 *   malloc per call.  At 1–10 nnz/row the merge is faster.  For rows with
 *   > ~32 nonzeros, a gather→zaxpy→scatter path would be faster due to
 *   AVX vectorisation; consider adding that if your matrices are denser.
 *
 * Zero-cancellation handling:
 *   When drop_tol == 0 the condition `av > 0.0` still drops exact zeros
 *   produced by cancellation (e.g. singular matrices).  drop_tol only
 *   governs small-but-nonzero fill-in for ILU approximation.
 *
 * Both rows must be sorted.  tmp_cols/tmp_vals are preallocated scratch
 * buffers of size 2n; the result is written back in-place.
 *
 * OpenMP: each call modifies only M[row_i]; M[k] is read-only.
 *         The caller parallelises over row_i; this function is not parallel.
 * ========================================================================= */
/* -------------------------------------------------------------------------
 * BLAS-path complex axpy (only compiled when HAVE_BLAS is defined).
 *
 * Algorithm:
 *   1. Walk both rows ri and rk once to build the column union and gather
 *      values into two dense buffers (one for ri, one for rk).
 *   2. Call cblas_zaxpy on the dense buffers (vectorised SIMD arithmetic).
 *   3. Walk the union once more applying drop_tol and zero-filtering,
 *      writing back into ri.
 *
 * Win condition: union >= BLAS_AXPY_THRESHOLD entries.  The merge path
 * has tight, branchless inner loops at small sizes; the BLAS path wins
 * once vectorised arithmetic dominates the gather/scatter overhead.
 * ------------------------------------------------------------------------- */
#ifdef HAVE_BLAS
static int row_axpy_z_blas(SparseRow * restrict M,
                            int64_t row_i, int64_t k,
                            double complex alpha, double drop_tol,
                            int64_t * restrict tmp_cols,
                            double complex * restrict tmp_vals_i,
                            double complex * restrict tmp_vals_k)
{
    SparseRow *ri = &M[row_i];
    const SparseRow *rk = &M[k];
    int64_t ni = ri->nnz, nk = rk->nnz;
    int64_t pi = 0, pk = 0, U = 0;

    /* Pass 1: merge to build column union and gather into dense buffers. */
    while (pi < ni && pk < nk) {
        int64_t ci = ri->cols[pi], ck = rk->cols[pk];
        if (ci < ck) {
            tmp_cols[U]    = ci;
            tmp_vals_i[U]  = ri->vals_z[pi];
            tmp_vals_k[U]  = 0.0;
            pi++; U++;
        } else if (ci > ck) {
            tmp_cols[U]    = ck;
            tmp_vals_i[U]  = 0.0;
            tmp_vals_k[U]  = rk->vals_z[pk];
            pk++; U++;
        } else {
            tmp_cols[U]    = ci;
            tmp_vals_i[U]  = ri->vals_z[pi];
            tmp_vals_k[U]  = rk->vals_z[pk];
            pi++; pk++; U++;
        }
    }
    while (pi < ni) {
        tmp_cols[U]    = ri->cols[pi];
        tmp_vals_i[U]  = ri->vals_z[pi];
        tmp_vals_k[U]  = 0.0;
        pi++; U++;
    }
    while (pk < nk) {
        tmp_cols[U]    = rk->cols[pk];
        tmp_vals_i[U]  = 0.0;
        tmp_vals_k[U]  = rk->vals_z[pk];
        pk++; U++;
    }

    /* Pass 2: vectorised axpy: tmp_vals_i += alpha * tmp_vals_k. */
    if (U > 0) {
        /* CBLAS interface accepts a pointer for complex alpha. */
        cblas_zaxpy((int)U, &alpha, tmp_vals_k, 1, tmp_vals_i, 1);
    }

    /* Pass 3: filter, strip column k, and write back to ri. */
    int64_t final_nnz = 0;
    for (int64_t p = 0; p < U; p++) {
        if (tmp_cols[p] == k) continue;
        double complex v = tmp_vals_i[p];
        double av = cabs(v);
        if (av == 0.0) continue;
        if (drop_tol > 0.0 && av <= drop_tol) continue;
        tmp_cols[final_nnz]   = tmp_cols[p];
        tmp_vals_i[final_nnz] = v;
        final_nnz++;
    }

    if (sr_ensure_cap(ri, final_nnz, 1) != 0) return -1;
    memcpy(ri->cols,   tmp_cols,   (size_t)final_nnz * sizeof(int64_t));
    memcpy(ri->vals_z, tmp_vals_i, (size_t)final_nnz * sizeof(double complex));
    ri->nnz = final_nnz;
    return 0;
}

static int row_axpy_d_blas(SparseRow * restrict M,
                            int64_t row_i, int64_t k,
                            double alpha, double drop_tol,
                            int64_t * restrict tmp_cols,
                            double * restrict tmp_vals_i,
                            double * restrict tmp_vals_k)
{
    SparseRow *ri = &M[row_i];
    const SparseRow *rk = &M[k];
    int64_t ni = ri->nnz, nk = rk->nnz;
    int64_t pi = 0, pk = 0, U = 0;

    while (pi < ni && pk < nk) {
        int64_t ci = ri->cols[pi], ck = rk->cols[pk];
        if (ci < ck) {
            tmp_cols[U]    = ci;
            tmp_vals_i[U]  = ri->vals_d[pi];
            tmp_vals_k[U]  = 0.0;
            pi++; U++;
        } else if (ci > ck) {
            tmp_cols[U]    = ck;
            tmp_vals_i[U]  = 0.0;
            tmp_vals_k[U]  = rk->vals_d[pk];
            pk++; U++;
        } else {
            tmp_cols[U]    = ci;
            tmp_vals_i[U]  = ri->vals_d[pi];
            tmp_vals_k[U]  = rk->vals_d[pk];
            pi++; pk++; U++;
        }
    }
    while (pi < ni) {
        tmp_cols[U]    = ri->cols[pi];
        tmp_vals_i[U]  = ri->vals_d[pi];
        tmp_vals_k[U]  = 0.0;
        pi++; U++;
    }
    while (pk < nk) {
        tmp_cols[U]    = rk->cols[pk];
        tmp_vals_i[U]  = 0.0;
        tmp_vals_k[U]  = rk->vals_d[pk];
        pk++; U++;
    }

    if (U > 0) {
        cblas_daxpy((int)U, alpha, tmp_vals_k, 1, tmp_vals_i, 1);
    }

    int64_t final_nnz = 0;
    for (int64_t p = 0; p < U; p++) {
        if (tmp_cols[p] == k) continue;
        double v = tmp_vals_i[p];
        double av = fabs(v);
        if (av == 0.0) continue;
        if (drop_tol > 0.0 && av <= drop_tol) continue;
        tmp_cols[final_nnz]   = tmp_cols[p];
        tmp_vals_i[final_nnz] = v;
        final_nnz++;
    }

    if (sr_ensure_cap(ri, final_nnz, 0) != 0) return -1;
    memcpy(ri->cols,   tmp_cols,   (size_t)final_nnz * sizeof(int64_t));
    memcpy(ri->vals_d, tmp_vals_i, (size_t)final_nnz * sizeof(double));
    ri->nnz = final_nnz;
    return 0;
}
#endif /* HAVE_BLAS */

static int row_axpy_z(SparseRow * restrict M, int64_t row_i, int64_t k,
                      double complex alpha, double drop_tol,
                      int64_t * restrict tmp_cols,
                      double complex * restrict tmp_vals
#ifdef HAVE_BLAS
                      , double complex * restrict tmp_vals2
#endif
                      )
{
    SparseRow *ri = &M[row_i];
    const SparseRow *rk = &M[k];
    int64_t ni = ri->nnz, nk = rk->nnz;

#ifdef HAVE_BLAS
    /* Dispatch to BLAS path when union is likely to exceed threshold.
     * ni + nk is an upper bound on union size; use 2*threshold as proxy. */
    if (tmp_vals2 != NULL && (ni + nk) >= 2 * BLAS_AXPY_THRESHOLD) {
        return row_axpy_z_blas(M, row_i, k, alpha, drop_tol,
                                tmp_cols, tmp_vals, tmp_vals2);
    }
#endif

    int64_t pi = 0, pk = 0, out = 0;

    while (pi < ni && pk < nk) {
        int64_t ci = ri->cols[pi], ck = rk->cols[pk];
        if (ci < ck) {
            tmp_cols[out] = ci; tmp_vals[out] = ri->vals_z[pi];
            pi++; out++;
        } else if (ci > ck) {
            double complex v = alpha * rk->vals_z[pk];
            double av = cabs(v);
            if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
                tmp_cols[out] = ck; tmp_vals[out] = v; out++;
            }
            pk++;
        } else {    /* ci == ck: subtraction may cancel exactly to zero */
            double complex v = ri->vals_z[pi] + alpha * rk->vals_z[pk];
            double av = cabs(v);
            if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
                tmp_cols[out] = ci; tmp_vals[out] = v; out++;
            }
            pi++; pk++;
        }
    }
    while (pi < ni) {
        tmp_cols[out] = ri->cols[pi]; tmp_vals[out] = ri->vals_z[pi];
        pi++; out++;
    }
    while (pk < nk) {
        double complex v = alpha * rk->vals_z[pk];
        double av = cabs(v);
        if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
            tmp_cols[out] = rk->cols[pk]; tmp_vals[out] = v; out++;
        }
        pk++;
    }

    /* Strip column k (the sub-diagonal entry is exactly zero after elimination). */
    int64_t final_nnz = 0;
    for (int64_t p = 0; p < out; p++) {
        if (tmp_cols[p] == k) continue;
        tmp_cols[final_nnz] = tmp_cols[p];
        tmp_vals[final_nnz] = tmp_vals[p];
        final_nnz++;
    }

    if (sr_ensure_cap(ri, final_nnz, 1) != 0) return -1;
    memcpy(ri->cols,   tmp_cols, (size_t)final_nnz * sizeof(int64_t));
    memcpy(ri->vals_z, tmp_vals, (size_t)final_nnz * sizeof(double complex));
    ri->nnz = final_nnz;
    return 0;
}

static int row_axpy_d(SparseRow * restrict M, int64_t row_i, int64_t k,
                      double alpha, double drop_tol,
                      int64_t * restrict tmp_cols,
                      double * restrict tmp_vals
#ifdef HAVE_BLAS
                      , double * restrict tmp_vals2
#endif
                      )
{
    SparseRow *ri = &M[row_i];
    const SparseRow *rk = &M[k];
    int64_t ni = ri->nnz, nk = rk->nnz;

#ifdef HAVE_BLAS
    if (tmp_vals2 != NULL && (ni + nk) >= 2 * BLAS_AXPY_THRESHOLD) {
        return row_axpy_d_blas(M, row_i, k, alpha, drop_tol,
                                tmp_cols, tmp_vals, tmp_vals2);
    }
#endif

    int64_t pi = 0, pk = 0, out = 0;

    while (pi < ni && pk < nk) {
        int64_t ci = ri->cols[pi], ck = rk->cols[pk];
        if (ci < ck) {
            tmp_cols[out] = ci; tmp_vals[out] = ri->vals_d[pi];
            pi++; out++;
        } else if (ci > ck) {
            double v = alpha * rk->vals_d[pk];
            double av = fabs(v);
            if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
                tmp_cols[out] = ck; tmp_vals[out] = v; out++;
            }
            pk++;
        } else {
            double v = ri->vals_d[pi] + alpha * rk->vals_d[pk];
            double av = fabs(v);
            if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
                tmp_cols[out] = ci; tmp_vals[out] = v; out++;
            }
            pi++; pk++;
        }
    }
    while (pi < ni) {
        tmp_cols[out] = ri->cols[pi]; tmp_vals[out] = ri->vals_d[pi];
        pi++; out++;
    }
    while (pk < nk) {
        double v = alpha * rk->vals_d[pk];
        double av = fabs(v);
        if (av > 0.0 && (drop_tol == 0.0 || av > drop_tol)) {
            tmp_cols[out] = rk->cols[pk]; tmp_vals[out] = v; out++;
        }
        pk++;
    }

    int64_t final_nnz = 0;
    for (int64_t p = 0; p < out; p++) {
        if (tmp_cols[p] == k) continue;
        tmp_cols[final_nnz] = tmp_cols[p];
        tmp_vals[final_nnz] = tmp_vals[p];
        final_nnz++;
    }

    if (sr_ensure_cap(ri, final_nnz, 0) != 0) return -1;
    memcpy(ri->cols,   tmp_cols, (size_t)final_nnz * sizeof(int64_t));
    memcpy(ri->vals_d, tmp_vals, (size_t)final_nnz * sizeof(double));
    ri->nnz = final_nnz;
    return 0;
}

/* =========================================================================
 * Assemble CSR output arrays from an array of SparseRow.
 * upper=1: collect col >= row  (U); upper=0: collect col < row plus unit diag (L).
 * Output index arrays are int64.
 * OpenMP: row-counting pass and fill pass are both parallelised.
 * ========================================================================= */
static int assemble_csr_z(SparseRow * restrict rows, int64_t n, int upper,
                           double drop_tol,
                           int64_t **out_indptr, int64_t **out_indices,
                           double complex **out_data)
{
    int64_t *indptr = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    if (!indptr) return -1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        int64_t cnt = upper ? 0 : 1;
        for (int64_t p = 0; p < rows[i].nnz; p++) {
            int64_t j = rows[i].cols[p];
            double complex v = rows[i].vals_z[p];
            if (drop_tol > 0.0 && cabs(v) <= drop_tol) continue;
            if ( upper && j >= i) cnt++;
            if (!upper && j <  i) cnt++;
        }
        indptr[i + 1] = cnt;   /* will be prefix-summed below */
    }

    /* Serial prefix sum (fast for n even up to 1e6). */
    indptr[0] = 0;
    for (int64_t i = 0; i < n; i++) indptr[i+1] += indptr[i];
    int64_t total = indptr[n];

    int64_t       *indices = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    double complex *data   = (double complex *)malloc(
                                 (size_t)total * sizeof(double complex));
    if (!indices || !data) {
        free(indptr); free(indices); free(data); return -1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        int64_t pos = indptr[i];
        if (!upper) { indices[pos] = i; data[pos] = 1.0 + 0.0*I; pos++; }
        for (int64_t p = 0; p < rows[i].nnz; p++) {
            int64_t j = rows[i].cols[p];
            double complex v = rows[i].vals_z[p];
            if (drop_tol > 0.0 && cabs(v) <= drop_tol) continue;
            if ( upper && j >= i) { indices[pos] = j; data[pos] = v; pos++; }
            if (!upper && j <  i) { indices[pos] = j; data[pos] = v; pos++; }
        }
    }

    *out_indptr = indptr; *out_indices = indices; *out_data = data;
    return 0;
}

static int assemble_csr_d(SparseRow * restrict rows, int64_t n, int upper,
                           double drop_tol,
                           int64_t **out_indptr, int64_t **out_indices,
                           double **out_data)
{
    int64_t *indptr = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    if (!indptr) return -1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        int64_t cnt = upper ? 0 : 1;
        for (int64_t p = 0; p < rows[i].nnz; p++) {
            int64_t j = rows[i].cols[p];
            double v = rows[i].vals_d[p];
            if (drop_tol > 0.0 && fabs(v) <= drop_tol) continue;
            if ( upper && j >= i) cnt++;
            if (!upper && j <  i) cnt++;
        }
        indptr[i + 1] = cnt;
    }
    indptr[0] = 0;
    for (int64_t i = 0; i < n; i++) indptr[i+1] += indptr[i];
    int64_t total = indptr[n];

    int64_t *indices = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    double  *data    = (double  *)malloc((size_t)total * sizeof(double));
    if (!indices || !data) {
        free(indptr); free(indices); free(data); return -1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        int64_t pos = indptr[i];
        if (!upper) { indices[pos] = i; data[pos] = 1.0; pos++; }
        for (int64_t p = 0; p < rows[i].nnz; p++) {
            int64_t j = rows[i].cols[p];
            double v = rows[i].vals_d[p];
            if (drop_tol > 0.0 && fabs(v) <= drop_tol) continue;
            if ( upper && j >= i) { indices[pos] = j; data[pos] = v; pos++; }
            if (!upper && j <  i) { indices[pos] = j; data[pos] = v; pos++; }
        }
    }

    *out_indptr = indptr; *out_indices = indices; *out_data = data;
    return 0;
}

/* =========================================================================
 * Wrap a malloc'd C buffer as an owning 1-D numpy array.
 * NPY_ARRAY_OWNDATA transfers the malloc to numpy's GC.
 * ========================================================================= */
static PyObject *wrap_array(void *data, npy_intp len, int typenum)
{
    npy_intp dims[1] = { len };
    PyObject *arr = PyArray_SimpleNewFromData(1, dims, typenum, data);
    if (!arr) return NULL;
    PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

/* =========================================================================
 * Core factorisation driver.
 * ========================================================================= */
/* Pivot strategy selectors (must match values in sparse_lu_full_pivot.py). */
#define FPLUS_PIVOT_FULL    0
#define FPLUS_PIVOT_PARTIAL 1
#define FPLUS_PIVOT_ROOK    2

static PyObject *
run_factorisation(int64_t n,
                  const int64_t * restrict in_indptr,
                  const int64_t * restrict in_indices,
                  const void    * restrict in_data,
                  double drop_tol, int is_complex, int verbose,
                  int pivot_strategy)
{
    SparseRow *M   = NULL, *Lm  = NULL;
    int64_t   *p   = NULL, *q   = NULL;
    int64_t   *tmp_cols = NULL;
    void      *tmp_vals = NULL;
#ifdef HAVE_BLAS
    /* Second values buffer for the BLAS-path axpy.  Only allocated when
     * HAVE_BLAS is defined; the merge-path axpy ignores it.              */
    void      *tmp_vals2 = NULL;
#endif

    /* Verbose-mode timing and progress. */
    double t_start = (verbose >= 1) ? wall_time() : 0.0;
    /* Print every max(1, n/20) steps when verbose >= 2, capped at 1000. */
    int64_t print_stride = (verbose >= 2) ? (n / 20 + 1) : 0;
    if (print_stride < 1)    print_stride = 1;
    if (print_stride > 1000) print_stride = 1000;

    const char *strat_name = (pivot_strategy == FPLUS_PIVOT_FULL)    ? "full"
                            : (pivot_strategy == FPLUS_PIVOT_PARTIAL) ? "partial"
                            : (pivot_strategy == FPLUS_PIVOT_ROOK)    ? "rook"
                            : "unknown";

    if (verbose >= 1) {
        int64_t in_nnz = in_indptr[n];
        printf("[FPLUS] LU factorisation start: n=%ld, nnz(A)=%ld, %s, "
               "drop_tol=%.1e, pivot=%s\n",
               (long)n, (long)in_nnz,
               is_complex ? "complex128" : "float64", drop_tol, strat_name);
#ifdef _OPENMP
        printf("[FPLUS]   OpenMP enabled, %d threads available\n",
               omp_get_max_threads());
#else
        printf("[FPLUS]   serial build (no OpenMP)\n");
#endif
        fflush(stdout);
    }

    /* Output buffers (transferred to numpy on success). */
    int64_t        *L_indptr = NULL, *L_indices = NULL;
    int64_t        *U_indptr = NULL, *U_indices = NULL;
    double complex *L_data_z = NULL, *U_data_z  = NULL;
    double         *L_data_d = NULL, *U_data_d  = NULL;

    /* Allocate working structures. */
    M  = (SparseRow *)calloc((size_t)n, sizeof(SparseRow));
    Lm = (SparseRow *)calloc((size_t)n, sizeof(SparseRow));
    p  = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    q  = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!M || !Lm || !p || !q) goto oom;

    /* Scratch buffers: 2n covers worst-case merge output before col-k removal. */
    tmp_cols = (int64_t *)malloc((size_t)2 * (size_t)n * sizeof(int64_t));
    tmp_vals = is_complex
        ? (void *)malloc((size_t)2 * (size_t)n * sizeof(double complex))
        : (void *)malloc((size_t)2 * (size_t)n * sizeof(double));
    if (!tmp_cols || !tmp_vals) goto oom;
#ifdef HAVE_BLAS
    /* Second values buffer for BLAS axpy path.  Only the BLAS path uses it,
     * but allocating once here amortises the cost across all axpy calls.  */
    tmp_vals2 = is_complex
        ? (void *)malloc((size_t)2 * (size_t)n * sizeof(double complex))
        : (void *)malloc((size_t)2 * (size_t)n * sizeof(double));
    if (!tmp_vals2) goto oom;
#endif

    for (int64_t i = 0; i < n; i++) { p[i] = i; q[i] = i; }

    /* Load CSR input into SparseRow array. */
    for (int64_t i = 0; i < n; i++) {
        int64_t rnnz = in_indptr[i+1] - in_indptr[i];
        int64_t cap  = rnnz + INITIAL_SLACK;
        if (cap < INITIAL_SLACK) cap = INITIAL_SLACK;
        if (sr_alloc(&M[i],  cap, is_complex) != 0) goto oom;
        if (sr_alloc(&Lm[i], INITIAL_SLACK, is_complex) != 0) goto oom;

        M[i].nnz = rnnz;
        memcpy(M[i].cols, in_indices + in_indptr[i],
               (size_t)rnnz * sizeof(int64_t));
        if (is_complex)
            memcpy(M[i].vals_z,
                   (const double complex *)in_data + in_indptr[i],
                   (size_t)rnnz * sizeof(double complex));
        else
            memcpy(M[i].vals_d,
                   (const double *)in_data + in_indptr[i],
                   (size_t)rnnz * sizeof(double));
    }

    /* ---- Main elimination loop (sequential; inner ops use OpenMP) ---- */
    /* Relative pivot threshold: declare singular when the best available
     * pivot is negligibly small relative to the largest entry in the
     * original (equilibrated) matrix.  After Ruiz equilibration all entries
     * start with magnitude ≈ 1, so max_entry_k0 ≈ 1 and the threshold
     * catches near-singular matrices that produce near-zero pivots from
     * floating-point cancellation (e.g. [[1,2],[2,4]] after equilibration
     * becomes [[1,1],[1,1]], whose step-1 pivot is ~eps rather than 0).
     * sqrt(DBL_EPSILON) ≈ 1.5e-8 is the standard LAPACK choice.         */
    double pivot_threshold = 0.0;   /* set after step 0 */
    double max_entry_k0    = 0.0;

    for (int64_t k = 0; k < n; k++) {

        /* 1. Pivot search — dispatched on pivot_strategy.
         *    - FULL:    scan entire M[k:,k:] for max |entry|; row + col swap.
         *    - PARTIAL: scan column k for max |entry|; row swap only.
         *    - ROOK:    alternate column/row scans until rook condition met.
         */
        int64_t pi = -1, pj = -1;
        double  best_abs_k;
        int found;
        if (is_complex) {
            double complex pv_z;
            switch (pivot_strategy) {
              case FPLUS_PIVOT_PARTIAL:
                pj = k;     /* No column swap for partial pivoting. */
                found = find_partial_pivot_z(M, k, n, &pi, &pv_z);
                break;
              case FPLUS_PIVOT_ROOK:
                found = find_rook_pivot_z(M, k, n, &pi, &pj, &pv_z);
                break;
              case FPLUS_PIVOT_FULL:
              default:
                found = find_full_pivot_z(M, k, n, &pi, &pj, &pv_z);
                break;
            }
            if (found < 0) goto oom;   /* OOM in thread-local alloc */
            best_abs_k = cabs(pv_z);
        } else {
            double pv_d;
            switch (pivot_strategy) {
              case FPLUS_PIVOT_PARTIAL:
                pj = k;
                found = find_partial_pivot_d(M, k, n, &pi, &pv_d);
                break;
              case FPLUS_PIVOT_ROOK:
                found = find_rook_pivot_d(M, k, n, &pi, &pj, &pv_d);
                break;
              case FPLUS_PIVOT_FULL:
              default:
                found = find_full_pivot_d(M, k, n, &pi, &pj, &pv_d);
                break;
            }
            if (found < 0) goto oom;
            best_abs_k = fabs(pv_d);
        }

        if (k == 0) {
            max_entry_k0   = best_abs_k;
            pivot_threshold = sqrt(2.2204460492503131e-16) * max_entry_k0;
        }

        if (!found || best_abs_k < pivot_threshold) {
            /* PyErr_Format does not support %lld or floating-point specifiers.
             * Pre-format the numeric values into a char buffer via snprintf,
             * then pass the buffer as a plain %s string.                     */
            char _msg[256];
            snprintf(_msg, sizeof(_msg),
                "Near-zero or zero pivot %.6e at step %ld "
                "(threshold %.6e). Matrix is singular.",
                best_abs_k, (long)k, pivot_threshold);
            PyErr_SetString(PyExc_RuntimeError, _msg);
            goto fail;
        }

        /* Verbose progress (verbose >= 2): print every print_stride steps,
         * plus the very first and last step.  Reports current pivot magnitude
         * and a running estimate of L+U fill-in (sum of M and Lm row nnz). */
        if (verbose >= 2 && (k % print_stride == 0 || k == n - 1)) {
            int64_t cur_nnz = 0;
            for (int64_t r = 0; r < n; r++) cur_nnz += M[r].nnz + Lm[r].nnz;
            double pct     = 100.0 * (double)(k + 1) / (double)n;
            double elapsed = wall_time() - t_start;
            printf("[FPLUS]   [%5.1f%%] step %ld/%ld  pivot=%.3e  "
                   "L+U nnz=%ld  elapsed=%.2fs\n",
                   pct, (long)(k + 1), (long)n, best_abs_k,
                   (long)cur_nnz, elapsed);
            fflush(stdout);
        }

        /* 2. Row swap k <-> pi — O(1), serial. */
        if (pi != k) {
            sr_swap(&M[pi],  &M[k]);
            sr_swap(&Lm[pi], &Lm[k]);
            int64_t tp = p[k]; p[k] = p[pi]; p[pi] = tp;
        }

        /* 3. Column swap k <-> pj — O(n * avg_nnz), parallel.
         *    Skipped for partial pivoting (pj is always k there).
         *    For rook pivoting, swaps only occur when the pivot column
         *    differs from k.  Preserves the COLAMD ordering far better
         *    than full pivoting since the swap region is restricted.   */
        if (pj != k) {
            if (is_complex) col_swap_z(M, n, k, pj);
            else            col_swap_d(M, n, k, pj);
            int64_t tq = q[k]; q[k] = q[pj]; q[pj] = tq;
        }

        /* 4. Locate pivot value (now at M[k][k]). */
        int64_t pp = bsearch_i64(M[k].cols, k, M[k].nnz);
        if (pp < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "Pivot entry missing at step %ld after swap.", (long)k);
            goto fail;
        }

        /* 5. Eliminate column k in rows k+1..n-1 — parallel.
         *
         * Each row M[i] is written independently.  M[k] is read-only here.
         * sr_ensure_cap reallocs M[i]->cols/vals; different threads touch
         * different rows so no data race occurs.
         *
         * OpenMP scratch buffers: allocated once per thread before the
         * parallel region, reused across all rows within the region, and
         * freed after the region ends.  This avoids millions of
         * malloc/free pairs (one per row per step) in the old approach.
         *
         * OpenMP error handling: if row_axpy returns -1 (OOM) or
         * sr_insert fails, we set a shared flag; remaining iterations
         * skip work.  The error is raised after the parallel block.
         */
        int elim_oom = 0;

#ifdef _OPENMP
        /* Allocate per-thread scratch buffers once. */
        int nthreads_elim = omp_get_max_threads();
        int64_t **omp_tcols = (int64_t **)calloc(
                (size_t)nthreads_elim, sizeof(int64_t *));
        void    **omp_tvals = (void **)calloc(
                (size_t)nthreads_elim, sizeof(void *));
#ifdef HAVE_BLAS
        void    **omp_tvals2 = (void **)calloc(
                (size_t)nthreads_elim, sizeof(void *));
        if (!omp_tcols || !omp_tvals || !omp_tvals2) {
            free(omp_tcols); free(omp_tvals); free(omp_tvals2);
            goto oom;
        }
#else
        if (!omp_tcols || !omp_tvals) {
            free(omp_tcols); free(omp_tvals);
            goto oom;
        }
#endif
        {
            size_t col_sz = (size_t)2 * (size_t)n * sizeof(int64_t);
            size_t val_sz = is_complex
                ? (size_t)2 * (size_t)n * sizeof(double complex)
                : (size_t)2 * (size_t)n * sizeof(double);
            for (int t = 0; t < nthreads_elim; t++) {
                omp_tcols[t] = (int64_t *)malloc(col_sz);
                omp_tvals[t] = malloc(val_sz);
                if (!omp_tcols[t] || !omp_tvals[t]) { elim_oom = 1; break; }
#ifdef HAVE_BLAS
                omp_tvals2[t] = malloc(val_sz);
                if (!omp_tvals2[t]) { elim_oom = 1; break; }
#endif
            }
        }
        if (elim_oom) {
            for (int t = 0; t < nthreads_elim; t++) {
                free(omp_tcols[t]); free(omp_tvals[t]);
#ifdef HAVE_BLAS
                free(omp_tvals2[t]);
#endif
            }
            free(omp_tcols); free(omp_tvals);
#ifdef HAVE_BLAS
            free(omp_tvals2);
#endif
            goto oom;
        }

#pragma omp parallel for schedule(dynamic,16) shared(elim_oom)
#endif
        for (int64_t i = k + 1; i < n; i++) {
#ifdef _OPENMP
            if (elim_oom) continue;   /* skip remaining work on error */
#endif
            int64_t pos = bsearch_i64(M[i].cols, k, M[i].nnz);
            if (pos < 0) continue;

            if (is_complex) {
                double complex pivot_val = M[k].vals_z[pp];
                double complex Mik       = M[i].vals_z[pos];
                double complex m         = Mik / pivot_val;
                if (drop_tol > 0.0 && cabs(m) <= drop_tol) {
                    sr_delete_z(&M[i], pos);
                    continue;
                }
#ifdef _OPENMP
                int tid = omp_get_thread_num();
                int64_t        *t_cols = omp_tcols[tid];
                double complex *t_vals = (double complex *)omp_tvals[tid];
#ifdef HAVE_BLAS
                double complex *t_vals2 = (double complex *)omp_tvals2[tid];
#endif
#else
                int64_t        *t_cols = tmp_cols;
                double complex *t_vals = (double complex *)tmp_vals;
#ifdef HAVE_BLAS
                double complex *t_vals2 = (double complex *)tmp_vals2;
#endif
#endif
                if (sr_insert_z(&Lm[i], k, m) != 0) {
#ifdef _OPENMP
#pragma omp atomic write
                    elim_oom = 1;
#endif
                    continue;
                }
                if (row_axpy_z(M, i, k, -m, drop_tol, t_cols, t_vals
#ifdef HAVE_BLAS
                                , t_vals2
#endif
                                ) != 0) {
#ifdef _OPENMP
#pragma omp atomic write
                    elim_oom = 1;
#endif
                    continue;
                }
            } else {  /* real */
                double pivot_val = M[k].vals_d[pp];
                double Mik       = M[i].vals_d[pos];
                double m         = Mik / pivot_val;
                if (drop_tol > 0.0 && fabs(m) <= drop_tol) {
                    sr_delete_d(&M[i], pos);
                    continue;
                }
#ifdef _OPENMP
                int tid = omp_get_thread_num();
                int64_t *t_cols = omp_tcols[tid];
                double  *t_vals = (double *)omp_tvals[tid];
#ifdef HAVE_BLAS
                double  *t_vals2 = (double *)omp_tvals2[tid];
#endif
#else
                int64_t *t_cols = tmp_cols;
                double  *t_vals = (double *)tmp_vals;
#ifdef HAVE_BLAS
                double  *t_vals2 = (double *)tmp_vals2;
#endif
#endif
                if (sr_insert_d(&Lm[i], k, m) != 0) {
#ifdef _OPENMP
#pragma omp atomic write
                    elim_oom = 1;
#endif
                    continue;
                }
                if (row_axpy_d(M, i, k, -m, drop_tol, t_cols, t_vals
#ifdef HAVE_BLAS
                                , t_vals2
#endif
                                ) != 0) {
#ifdef _OPENMP
#pragma omp atomic write
                    elim_oom = 1;
#endif
                    continue;
                }
            }
        } /* end parallel elimination */

#ifdef _OPENMP
        /* Free per-thread scratch buffers. */
        for (int t = 0; t < nthreads_elim; t++) {
            free(omp_tcols[t]); free(omp_tvals[t]);
#ifdef HAVE_BLAS
            free(omp_tvals2[t]);
#endif
        }
        free(omp_tcols); free(omp_tvals);
#ifdef HAVE_BLAS
        free(omp_tvals2);
#endif
#endif

        if (elim_oom) goto oom;
    } /* end k-loop */

    /* Assemble CSR output. */
    int rc = 0;
    if (is_complex) {
        rc  = assemble_csr_z(Lm, n, 0, drop_tol,
                             &L_indptr, &L_indices, &L_data_z);
        rc |= assemble_csr_z(M,  n, 1, drop_tol,
                             &U_indptr, &U_indices, &U_data_z);
    } else {
        rc  = assemble_csr_d(Lm, n, 0, drop_tol,
                             &L_indptr, &L_indices, &L_data_d);
        rc |= assemble_csr_d(M,  n, 1, drop_tol,
                             &U_indptr, &U_indices, &U_data_d);
    }
    if (rc) goto oom;

    free_rows(M,  n); M  = NULL;
    free_rows(Lm, n); Lm = NULL;
    free(tmp_cols); tmp_cols = NULL;
    free(tmp_vals); tmp_vals = NULL;
#ifdef HAVE_BLAS
    free(tmp_vals2); tmp_vals2 = NULL;
#endif

    if (verbose >= 1) {
        int64_t l_nnz = L_indptr[n];
        int64_t u_nnz = U_indptr[n];
        double  total_elapsed = wall_time() - t_start;
        printf("[FPLUS] LU factorisation complete: nnz(L)=%ld, nnz(U)=%ld, "
               "total nnz=%ld, time=%.3fs\n",
               (long)l_nnz, (long)u_nnz, (long)(l_nnz + u_nnz),
               total_elapsed);
        fflush(stdout);
    }

    /* Wrap output buffers as numpy arrays.
     * After each successful wrap_array, NULL the corresponding raw pointer
     * so the goto-fail cleanup path doesn't double-free (numpy now owns
     * the buffer via NPY_ARRAY_OWNDATA).  If any wrap fails, we Py_XDECREF
     * the successfully-wrapped arrays (which frees their buffers) and then
     * goto fail to clean up any remaining raw pointers.                   */
    int64_t L_nnz = L_indptr[n], U_nnz = U_indptr[n];
    int typenum = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;

    PyObject *L_ip  = wrap_array(L_indptr,  (npy_intp)(n+1),  NPY_INT64);
    if (L_ip) L_indptr = NULL;
    PyObject *L_idx = wrap_array(L_indices, (npy_intp)L_nnz,  NPY_INT64);
    if (L_idx) L_indices = NULL;
    PyObject *L_dat = wrap_array(is_complex ? (void*)L_data_z
                                            : (void*)L_data_d,
                                  (npy_intp)L_nnz, typenum);
    if (L_dat) { if (is_complex) L_data_z = NULL; else L_data_d = NULL; }
    PyObject *U_ip  = wrap_array(U_indptr,  (npy_intp)(n+1),  NPY_INT64);
    if (U_ip) U_indptr = NULL;
    PyObject *U_idx = wrap_array(U_indices, (npy_intp)U_nnz,  NPY_INT64);
    if (U_idx) U_indices = NULL;
    PyObject *U_dat = wrap_array(is_complex ? (void*)U_data_z
                                            : (void*)U_data_d,
                                  (npy_intp)U_nnz, typenum);
    if (U_dat) { if (is_complex) U_data_z = NULL; else U_data_d = NULL; }
    PyObject *p_arr = wrap_array(p, (npy_intp)n, NPY_INT64);
    if (p_arr) p = NULL;
    PyObject *q_arr = wrap_array(q, (npy_intp)n, NPY_INT64);
    if (q_arr) q = NULL;

    if (!L_ip || !L_idx || !L_dat || !U_ip || !U_idx || !U_dat
             || !p_arr || !q_arr) {
        Py_XDECREF(L_ip);  Py_XDECREF(L_idx); Py_XDECREF(L_dat);
        Py_XDECREF(U_ip);  Py_XDECREF(U_idx); Py_XDECREF(U_dat);
        Py_XDECREF(p_arr); Py_XDECREF(q_arr);
        goto oom;
    }

    PyObject *result = Py_BuildValue(
        "{sOsOsOsOsOsOsOsO}",
        "L_data",    L_dat, "L_indices", L_idx, "L_indptr",  L_ip,
        "U_data",    U_dat, "U_indices", U_idx, "U_indptr",  U_ip,
        "p", p_arr, "q", q_arr);
    Py_XDECREF(L_ip);  Py_XDECREF(L_idx); Py_XDECREF(L_dat);
    Py_XDECREF(U_ip);  Py_XDECREF(U_idx); Py_XDECREF(U_dat);
    Py_XDECREF(p_arr); Py_XDECREF(q_arr);
    return result;

oom:
    PyErr_NoMemory();
fail:
    free_rows(M, n); free_rows(Lm, n);
    free(p); free(q); free(tmp_cols); free(tmp_vals);
#ifdef HAVE_BLAS
    free(tmp_vals2);
#endif
    free(L_indptr);  free(L_indices);
    free(U_indptr);  free(U_indices);
    free(L_data_z);  free(U_data_z);
    free(L_data_d);  free(U_data_d);
    return NULL;
}

/* =========================================================================
 * Python-facing entry point.
 * Accepts int32 or int64 index arrays; converts to int64 internally.
 * ========================================================================= */
static PyObject *
py_lu_factor(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    double drop_tol;
    int is_complex;
    int verbose = 0;          /* optional, default 0 = silent */
    int pivot_strategy = FPLUS_PIVOT_PARTIAL;  /* optional, default partial */

    if (!PyArg_ParseTuple(args, "OOOndi|ii",
                          &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &drop_tol, &is_complex,
                          &verbose, &pivot_strategy))
        return NULL;
    int64_t n = (int64_t)n_sst;

    /* Validate pivot_strategy. */
    if (pivot_strategy != FPLUS_PIVOT_FULL    &&
        pivot_strategy != FPLUS_PIVOT_PARTIAL &&
        pivot_strategy != FPLUS_PIVOT_ROOK) {
        PyErr_Format(PyExc_ValueError,
            "pivot_strategy must be 0 (full), 1 (partial), or 2 (rook); "
            "got %d", pivot_strategy);
        return NULL;
    }

    /* Accept data as complex128 or float64. */
    int data_type = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(
            data_obj, data_type, NPY_ARRAY_IN_ARRAY);
    if (!data_arr) return NULL;

    /* Accept indices/indptr as int32 or int64; cast to int64. */
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
            indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(
            indptr_obj,  NPY_INT64, NPY_ARRAY_IN_ARRAY);

    if (!idx_arr || !ptr_arr) {
        Py_DECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr);
        return NULL;
    }

    const void    *data    = PyArray_DATA(data_arr);
    const int64_t *indices = (const int64_t *)PyArray_DATA(idx_arr);
    const int64_t *indptr  = (const int64_t *)PyArray_DATA(ptr_arr);

    PyObject *result = run_factorisation(n, indptr, indices, data,
                                         drop_tol, is_complex, verbose,
                                         pivot_strategy);
    Py_DECREF(data_arr); Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    return result;
}

/* =========================================================================
 * Ruiz iterative L-infinity equilibration
 * =========================================================================
 *
 * Finds real diagonal scaling vectors Dr (length n) and Dc (length n) such
 * that the equilibrated matrix  B = diag(Dr) * A * diag(Dc)  has all row
 * and column L-inf norms close to 1.
 *
 * Algorithm (Ruiz 2001, "A scaling algorithm to equilibrate both rows and
 * columns norms in matrices"):
 *
 *   Repeat until convergence:
 *     For each row i:    Dr[i] *= 1 / sqrt(max_j |B[i,j]|)
 *     For each col j:    Dc[j] *= 1 / sqrt(max_i |B[i,j]|)
 *     Update B accordingly.
 *
 * Convergence criterion: max(|log(row_max)|, |log(col_max)|) < tol,
 * i.e. all row and column maxima are within factor e^tol of 1.
 *
 * Properties:
 *   - Works identically for real and complex matrices (uses |a_ij|).
 *   - Dr and Dc are always real even when A is complex.
 *   - Typically converges in 5-15 iterations.
 *   - Cost: O(nnz + n) per iteration; total O(10 * nnz) — negligible
 *     compared to the O(n^2) col_swap cost of the factorisation.
 *   - Zero rows/columns indicate structural singularity and are flagged.
 *
 * Why this is worth doing:
 *   Even though full pivoting bounds the growth factor, the pivot selection
 *   criterion (choose largest |entry|) is scale-dependent.  A row with all
 *   entries 1e-10 will never contribute a pivot even if it is numerically
 *   important.  After equilibration all rows and columns are comparable in
 *   magnitude, so the pivot search reflects genuine numerical importance
 *   rather than accidental scaling.  The condition number of B is also
 *   minimised (or near-minimised) by this procedure, which directly reduces
 *   the rounding errors in the triangular solve step.
 *
 * Integration with the factorisation:
 *   sparse_lu_full_pivot() calls ruiz_equilibrate() when equilibrate=True,
 *   passes the scaled matrix to run_factorisation(), and stores Dr and Dc
 *   in the returned object.  lu_solve() automatically applies:
 *       b_scaled = Dr * b
 *       y        = U \ (L \ b_scaled)      [triangular solves]
 *       x        = Dc * y
 *
 * Python call:
 *   ruiz_equilibrate(data, indices, indptr, n, max_iter, tol)
 *   -> dict {"Dr": float64[n], "Dc": float64[n], "data": dtype[nnz]}
 *
 * The returned "data" array is the scaled CSR data (indices/indptr unchanged).
 * ========================================================================= */
static PyObject *
py_ruiz_equilibrate(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    int max_iter, is_complex;
    double tol;
    int verbose = 0;   /* optional, default 0 = silent */

    if (!PyArg_ParseTuple(args, "OOOnidi|i",
                          &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &max_iter, &tol, &is_complex, &verbose))
        return NULL;

    int64_t n = (int64_t)n_sst;

    /* Accept int32 or int64 index arrays. */
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
            indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(
            indptr_obj,  NPY_INT64, NPY_ARRAY_IN_ARRAY);
    int data_type = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    /* Data is copied so we can modify it. */
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(
            data_obj, data_type, NPY_ARRAY_ENSURECOPY | NPY_ARRAY_IN_ARRAY);

    if (!idx_arr || !ptr_arr || !data_arr) {
        Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    const int64_t *indices = (const int64_t *)PyArray_DATA(idx_arr);
    const int64_t *indptr  = (const int64_t *)PyArray_DATA(ptr_arr);
    int64_t nnz = indptr[n];

    /* Allocate Dr, Dc (real, length n regardless of complex/real). */
    double *Dr = (double *)malloc((size_t)n * sizeof(double));
    double *Dc = (double *)malloc((size_t)n * sizeof(double));
    double *row_max = (double *)malloc((size_t)n * sizeof(double));
    double *col_max = (double *)malloc((size_t)n * sizeof(double));
    if (!Dr || !Dc || !row_max || !col_max) {
        Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr);
        free(Dr); free(Dc); free(row_max); free(col_max);
        return PyErr_NoMemory();
    }
    for (int64_t i = 0; i < n; i++) { Dr[i] = 1.0; Dc[i] = 1.0; }

    /* Verbose-mode timing and summary tracking. */
    double t_ruiz_start = (verbose >= 1) ? wall_time() : 0.0;
    int    iters_used   = 0;
    double final_res    = 0.0;
    if (verbose >= 1) {
        printf("[FPLUS] Ruiz equilibration start: n=%ld, nnz=%ld, "
               "max_iter=%d, tol=%.1e\n",
               (long)n, (long)nnz, max_iter, tol);
        fflush(stdout);
    }

    /* ------------------------------------------------------------------ */
    /* Main Ruiz loop: per iteration a row pass (the row L-inf maxima), a  */
    /* row-scaling pass that also gathers the column maxima of the scaled */
    /* entries, and a column-scaling pass.  Maxima are taken on squared    */
    /* moduli, one square root per line.  The residual is the deviation    */
    /* the iteration corrects, max |log| of the row and column maxima it   */
    /* found; it halves per iteration.                                     */
    /* ------------------------------------------------------------------ */
    for (int iter = 0; iter < max_iter; iter++) {
        for (int64_t i = 0; i < n; i++) row_max[i] = 0.0;
        for (int64_t j = 0; j < n; j++) col_max[j] = 0.0;
        if (is_complex) {
            double complex *zd = (double complex *)PyArray_DATA(data_arr);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < n; i++) {
                double rm = 0.0;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    double re = creal(zd[p]), im = cimag(zd[p]);
                    double av = re * re + im * im;
                    if (av > rm) rm = av;
                }
                row_max[i] = rm;
            }
        } else {
            double *dd = (double *)PyArray_DATA(data_arr);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < n; i++) {
                double rm = 0.0;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    double av = dd[p] * dd[p];
                    if (av > rm) rm = av;
                }
                row_max[i] = rm;
            }
        }
        for (int64_t i = 0; i < n; i++) {
            if (row_max[i] == 0.0) {
                Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr);
                free(Dr); free(Dc); free(row_max); free(col_max);
                PyErr_Format(PyExc_RuntimeError,
                    "Row %ld of the matrix is entirely zero (structural "
                    "singularity detected during equilibration).", (long)i);
                return NULL;
            }
        }
        double res = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double v = 0.5 * fabs(log(row_max[i]));   /* |log| of the modulus */
            if (v > res) res = v;
        }
        /* row scaling by 1/sqrt(row max), gathering the column maxima */
        if (is_complex) {
            double complex *zd = (double complex *)PyArray_DATA(data_arr);
            for (int64_t i = 0; i < n; i++) {
                double dr = 1.0 / sqrt(sqrt(row_max[i]));
                Dr[i] *= dr;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    zd[p] *= dr;
                    double re = creal(zd[p]), im = cimag(zd[p]);
                    double av = re * re + im * im;
                    int64_t j = indices[p];
                    if (av > col_max[j]) col_max[j] = av;
                }
            }
        } else {
            double *dd = (double *)PyArray_DATA(data_arr);
            for (int64_t i = 0; i < n; i++) {
                double dr = 1.0 / sqrt(sqrt(row_max[i]));
                Dr[i] *= dr;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    dd[p] *= dr;
                    double av = dd[p] * dd[p];
                    int64_t j = indices[p];
                    if (av > col_max[j]) col_max[j] = av;
                }
            }
        }
        for (int64_t j = 0; j < n; j++) {
            if (col_max[j] == 0.0) {
                Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr);
                free(Dr); free(Dc); free(row_max); free(col_max);
                PyErr_Format(PyExc_RuntimeError,
                    "Column %ld of the matrix is entirely zero (structural "
                    "singularity detected during equilibration).", (long)j);
                return NULL;
            }
        }
        /* column scaling by 1/sqrt(column max); col_max becomes the scale */
        for (int64_t j = 0; j < n; j++) {
            double v = 0.5 * fabs(log(col_max[j]));
            if (v > res) res = v;
            col_max[j] = 1.0 / sqrt(sqrt(col_max[j]));
            Dc[j] *= col_max[j];
        }
        if (is_complex) {
            double complex *zd = (double complex *)PyArray_DATA(data_arr);
            for (int64_t p = 0; p < nnz; p++) zd[p] *= col_max[indices[p]];
        } else {
            double *dd = (double *)PyArray_DATA(data_arr);
            for (int64_t p = 0; p < nnz; p++) dd[p] *= col_max[indices[p]];
        }
        iters_used = iter + 1;
        final_res  = res;
        if (verbose >= 2) {
            printf("[FPLUS]   iter %3d  residual=%.3e\n", iter + 1, res);
            fflush(stdout);
        }
        if (res < tol) break;
    } /* end Ruiz iterations */

    free(row_max); free(col_max);

    if (verbose >= 1) {
        /* Compute Dr/Dc range for the summary (informative for diagnosing
         * matrices with extreme scale separation).                       */
        double Dr_min = Dr[0], Dr_max = Dr[0];
        double Dc_min = Dc[0], Dc_max = Dc[0];
        for (int64_t i = 1; i < n; i++) {
            if (Dr[i] < Dr_min) Dr_min = Dr[i];
            if (Dr[i] > Dr_max) Dr_max = Dr[i];
            if (Dc[i] < Dc_min) Dc_min = Dc[i];
            if (Dc[i] > Dc_max) Dc_max = Dc[i];
        }
        double t_elapsed = wall_time() - t_ruiz_start;
        const char *status = (final_res < tol) ? "converged" : "max_iter reached";
        printf("[FPLUS] Ruiz %s: %d iters, final res=%.3e, "
               "Dr in [%.2e, %.2e], Dc in [%.2e, %.2e], time=%.3fs\n",
               status, iters_used, final_res,
               Dr_min, Dr_max, Dc_min, Dc_max, t_elapsed);
        fflush(stdout);
    }

    /* Wrap Dr and Dc as numpy float64 arrays (they own the malloc'd memory). */
    PyObject *Dr_arr = wrap_array(Dr, (npy_intp)n, NPY_FLOAT64);
    PyObject *Dc_arr = wrap_array(Dc, (npy_intp)n, NPY_FLOAT64);
    /* data_arr already holds the scaled data as a numpy array. */

    if (!Dr_arr || !Dc_arr) {
        Py_XDECREF(Dr_arr); Py_XDECREF(Dc_arr);
        Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr);
        return PyErr_NoMemory();
    }

    PyObject *result = Py_BuildValue("{sOsOsO}",
        "Dr",   Dr_arr,
        "Dc",   Dc_arr,
        "data", (PyObject *)data_arr);
    Py_XDECREF(Dr_arr); Py_XDECREF(Dc_arr);
    Py_DECREF(idx_arr); Py_DECREF(ptr_arr); Py_DECREF(data_arr);
    return result;
}

/* growable (index, value) arrays of the U factor */
static int gp_grow(int64_t **idx, void **val, int64_t *cap, int64_t need, int is_complex)
{
    if (need <= *cap) return 0;
    int64_t ncap = *cap * 2 > need ? *cap * 2 : need;
    int64_t *ni = (int64_t *)realloc(*idx, (size_t)ncap * sizeof(int64_t));
    if (!ni) return -1;
    *idx = ni;
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    void *nv = realloc(*val, (size_t)ncap * vs);
    if (!nv) return -1;
    *val = nv;
    *cap = ncap;
    return 0;
}

/* =========================================================================
 * Supernodal left-looking LU with partial pivoting.
 *
 * Columns whose L patterns nest (column c's non-pivotal rows are column
 * c-1's minus the pivot row of c) form a supernode, stored as one dense
 * column-major block over the union of its rows, the pivot rows of the
 * supernode's columns first in column order, the remaining rows after.
 * Column j is computed by the sparse triangular solve x = L^-1 A[:, j]:
 * a depth-first search over the supernodes gives the ones reached, in
 * topological order, each with the first of its columns reached (the
 * columns from there to its last form the U segment); the numeric sweep
 * scatters A[:, j] into a dense accumulator and applies each reached
 * supernode as a dense unit-lower solve on its segment followed by a
 * dense rectangle update of the rows below, both contiguous in memory.
 * The pivot is the entry of largest modulus among the rows not yet
 * pivotal; the column then joins the open supernode when its pattern
 * nests, or opens a new one.  The search reads pruned row lists
 * (Eisenstat-Liu): once the pivot row of column j lies in the rows of a
 * supernode whose segment column j reached, the rows of that supernode not
 * yet pivotal are reachable through column j, so its list is partitioned
 * and the search reads the pivotal part from then on.  Rows carry their
 * original index during the factorisation (pinv maps a row to its pivot
 * step) and are renumbered to pivot order on output, which is CSC with
 * A[p, :] = L U and q the identity.  L keeps its full pattern, zeros
 * included: the pruning argument needs it.  The singularity test is a
 * pivot below sqrt(eps) times the largest modulus of the column's own
 * entries in A.
 * ========================================================================= */

typedef struct {
    int64_t first;      /* first column */
    int64_t ncols;      /* columns stored so far */
    int64_t nrows;      /* rows of the block: ncols pivot rows, then the rest */
    int64_t cap;        /* column capacity of the block */
    int64_t *rows;      /* original row index per block row */
    void *blk;          /* dense column-major nrows x ncols */
    int64_t *dfs_rows;  /* the rest rows, partitioned by pruning (closed supernodes) */
    int64_t dfs_end;    /* end of the pivotal part of dfs_rows, -1 when unpruned */
} Supernode;

typedef struct {
    int64_t n;
    int is_complex;
    Supernode *sn;      /* one slot per column at most */
    int64_t nsn;        /* supernodes opened */
    int64_t *col_sn;    /* column -> supernode */
    int64_t *pinv, *p;
    int64_t *rowpos;    /* block row position of a row in the open supernode, else -1 */
    int64_t *wrow;      /* row visit stamp */
    int64_t *wsn;       /* supernode visit stamp */
    int64_t *segfirst;  /* first column reached per supernode, this column */
    int64_t *xi;        /* reached supernodes in topological order at xi[top..n) */
    int64_t *nonpiv;    /* non-pivotal rows of the pattern */
    int64_t *pstack, *head;
    int64_t *Up, *Ui;  void *Ux;  int64_t nnzU, capU;
    void *x, *y;        /* accumulator and the dense work vector */
} SnWork;

static inline int64_t sn_rest_end(const SnWork *W, int64_t s, int64_t open)
{
    const Supernode *S = &W->sn[s];
    if (s == open) return S->nrows - S->ncols; /* the open block: no pruning */
    return S->dfs_end >= 0 ? S->dfs_end : S->nrows - S->ncols;
}

static inline const int64_t *sn_rest_rows(const SnWork *W, int64_t s, int64_t open)
{
    const Supernode *S = &W->sn[s];
    return s == open ? S->rows + S->ncols : S->dfs_rows;
}

/* Depth-first search from supernode `root`, whose column `col` was
 * reached; finished supernodes are pushed on xi[--top].  Non-pivotal rows
 * met on the way are stamped and listed in nonpiv.  Returns the new top. */
static int64_t sn_dfs(int64_t root, int64_t col, int64_t top, SnWork *W, int64_t stamp,
                      int64_t open, int64_t *n_nonpiv)
{
    int64_t *pstack = W->pstack, *head = W->head, *wsn = W->wsn, *wrow = W->wrow;
    int64_t *segfirst = W->segfirst, *pinv = W->pinv, *col_sn = W->col_sn;
    int64_t sp = 0;
    pstack[0] = root;
    wsn[root] = stamp;
    segfirst[root] = col;
    head[root] = 0;
    while (sp >= 0) {
        int64_t s = pstack[sp];
        const int64_t *rows = sn_rest_rows(W, s, open);
        int64_t end = sn_rest_end(W, s, open);
        int done = 1;
        for (int64_t q = head[s]; q < end; q++) {
            int64_t i = rows[q];
            int64_t k = pinv[i];
            if (k < 0) {
                if (wrow[i] != stamp) { wrow[i] = stamp; W->nonpiv[(*n_nonpiv)++] = i; }
                continue;
            }
            int64_t t = col_sn[k];
            if (wsn[t] == stamp) {
                if (k < segfirst[t]) segfirst[t] = k;
                continue;
            }
            wsn[t] = stamp;
            segfirst[t] = k;
            head[s] = q + 1;
            head[t] = 0;
            pstack[++sp] = t;
            done = 0;
            break;
        }
        if (done) {
            sp--;
            W->xi[--top] = s;
        }
    }
    return top;
}

static int sn_grow_cols(Supernode *S, int64_t need, int is_complex)
{
    if (need <= S->cap) return 0;
    int64_t ncap = S->cap * 2 > need ? S->cap * 2 : need;
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    void *nb = realloc(S->blk, (size_t)ncap * (size_t)S->nrows * vs);
    if (!nb) return -1;
    S->blk = nb;
    S->cap = ncap;
    return 0;
}

#ifdef HAVE_BLAS
#  define SN_BLAS(m, nrest) ((m) * (nrest) >= SN_BLAS_MIN)
#else
#  define SN_BLAS(m, nrest) 0
#endif

#define SN_FACTOR_BODY(NAME, T, ABSF, TRSV, GEMV)                                       \
static int NAME(SnWork *W, const int64_t *Ap, const int64_t *Ai, const T *Ax, int verbose) \
{                                                                                       \
    const int64_t n = W->n;                                                             \
    const T ONE = 1, ZERO = 0; (void)ONE; (void)ZERO;                                   \
    const int is_complex = W->is_complex;                                               \
    int64_t *pinv = W->pinv, *p = W->p, *wrow = W->wrow, *xi = W->xi;                   \
    int64_t *rowpos = W->rowpos, *nonpiv = W->nonpiv, *col_sn = W->col_sn;              \
    T *x = (T *)W->x, *y = (T *)W->y;                                                   \
    double pivot_threshold;                                                             \
    int64_t open = -1;                                                                  \
    for (int64_t j = 0; j < n; j++) {                                                   \
        const int64_t stamp = j + 1;                                                    \
        int64_t top = n, n_nonpiv = 0;                                                  \
        /* symbolic: the reached supernodes in topological order */                    \
        for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) {                                   \
            int64_t i = Ai[q], k = pinv[i];                                             \
            if (k < 0) {                                                                \
                if (wrow[i] != stamp) { wrow[i] = stamp; nonpiv[n_nonpiv++] = i; }      \
                continue;                                                               \
            }                                                                           \
            int64_t s = col_sn[k];                                                      \
            if (W->wsn[s] == stamp) { if (k < W->segfirst[s]) W->segfirst[s] = k; continue; } \
            top = sn_dfs(s, k, top, W, stamp, open, &n_nonpiv);                         \
        }                                                                               \
        /* numeric: scatter A[:, j] (noting its largest modulus), apply the reached */ \
        /* supernodes */                                                                \
        double colmax = 0.0;                                                            \
        for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) {                                   \
            x[Ai[q]] += Ax[q];                                                          \
            double a = ABSF(Ax[q]);                                                     \
            if (a > colmax) colmax = a;                                                 \
        }                                                                               \
        for (int64_t t = top; t < n; t++) {                                             \
            Supernode *S = &W->sn[xi[t]];                                               \
            const int64_t r1 = S->nrows, w = S->ncols, kf = W->segfirst[xi[t]] - S->first; \
            const T *B = (const T *)S->blk;                                             \
            const int64_t *rows = S->rows;                                              \
            int64_t nrest = r1 - w, m = w - kf;                                         \
            const int64_t *rest = rows + w;                                             \
            if (SN_BLAS(m, nrest)) {                                                    \
                /* x_seg = L_seg^-1 x_seg, then y = B_rest x_seg, with BLAS */          \
                T *xs = y + nrest;                                                      \
                for (int64_t cc = kf; cc < w; cc++) xs[cc - kf] = x[rows[cc]];          \
                TRSV(102, 122, 111, 132, (int)m, B + kf * r1 + kf, (int)r1, xs, 1);     \
                GEMV(102, 111, (int)nrest, (int)m, &ONE, B + kf * r1 + w, (int)r1, xs, 1, &ZERO, y, 1); \
                for (int64_t cc = kf; cc < w; cc++) x[rows[cc]] = xs[cc - kf];          \
            } else {                                                                    \
                for (int64_t r = 0; r < nrest; r++) y[r] = 0;                           \
                for (int64_t cc = kf; cc < w; cc++) {                                   \
                    T xc = x[rows[cc]];                                                 \
                    if (xc == 0) continue;                                              \
                    const T *Bc = B + cc * r1;                                          \
                    for (int64_t rr = cc + 1; rr < w; rr++) x[rows[rr]] -= xc * Bc[rr]; \
                    const T *Br = Bc + w;                                               \
                    for (int64_t r = 0; r < nrest; r++) y[r] += Br[r] * xc;             \
                }                                                                       \
            }                                                                           \
            for (int64_t r = 0; r < nrest; r++) x[rest[r]] -= y[r];                     \
        }                                                                               \
        /* pivot: the largest modulus among the rows not yet pivotal */                 \
        int64_t pivrow = -1;                                                            \
        double best = -1.0;                                                             \
        for (int64_t s = 0; s < n_nonpiv; s++) {                                        \
            double a = ABSF(x[nonpiv[s]]);                                              \
            if (a > best) { best = a; pivrow = nonpiv[s]; }                             \
        }                                                                               \
        pivot_threshold = sqrt(2.2204460492503131e-16) * colmax;                       \
        if (pivrow < 0 || best <= 0.0 || best < pivot_threshold) {                      \
            char _msg[256];                                                             \
            snprintf(_msg, sizeof(_msg),                                                \
                "Near-zero or zero pivot %.6e at step %ld (threshold %.6e). "           \
                "Matrix is singular.", best > 0 ? best : 0.0, (long)j, pivot_threshold); \
            PyErr_SetString(PyExc_RuntimeError, _msg);                                  \
            return -2;                                                                  \
        }                                                                               \
        T piv = x[pivrow];                                                              \
        /* U[:, j]: the segments of the reached supernodes, then the diagonal */       \
        {                                                                               \
            int64_t need = 1;                                                           \
            for (int64_t t = top; t < n; t++)                                           \
                need += W->sn[xi[t]].first + W->sn[xi[t]].ncols - W->segfirst[xi[t]];   \
            if (gp_grow(&W->Ui, &W->Ux, &W->capU, W->nnzU + need, is_complex)) return -1; \
            int64_t *Ui = W->Ui; T *Ux = (T *)W->Ux;                                    \
            int64_t u = W->nnzU;                                                        \
            for (int64_t t = top; t < n; t++) {                                         \
                Supernode *S = &W->sn[xi[t]];                                           \
                for (int64_t cc = W->segfirst[xi[t]] - S->first; cc < S->ncols; cc++) { \
                    Ui[u] = S->first + cc; Ux[u] = x[S->rows[cc]]; u++;                 \
                }                                                                       \
            }                                                                           \
            Ui[u] = j; Ux[u] = piv; u++;                                                \
            W->nnzU = u; W->Up[j + 1] = u;                                              \
        }                                                                               \
        /* L[:, j]: join the open supernode when the pattern nests, else open one */    \
        int join = 0;                                                                   \
        if (open >= 0) {                                                                \
            Supernode *S = &W->sn[open];                                                \
            if (n_nonpiv == S->nrows - S->ncols && rowpos[pivrow] >= S->ncols) {        \
                join = 1;                                                               \
                for (int64_t s = 0; s < n_nonpiv; s++)                                  \
                    if (rowpos[nonpiv[s]] < S->ncols) { join = 0; break; }              \
            }                                                                           \
        }                                                                               \
        if (join) {                                                                     \
            Supernode *S = &W->sn[open];                                                \
            const int64_t r1 = S->nrows, w = S->ncols;                                  \
            if (sn_grow_cols(S, w + 1, is_complex)) return -1;                          \
            T *B = (T *)S->blk;                                                         \
            /* the pivot row moves to block row w */                                    \
            int64_t pp = rowpos[pivrow];                                                \
            if (pp != w) {                                                              \
                int64_t other = S->rows[w];                                             \
                S->rows[w] = pivrow; S->rows[pp] = other;                               \
                rowpos[pivrow] = w; rowpos[other] = pp;                                 \
                for (int64_t cc = 0; cc < w; cc++) {                                    \
                    T tv = B[cc * r1 + w]; B[cc * r1 + w] = B[cc * r1 + pp]; B[cc * r1 + pp] = tv; \
                }                                                                       \
            }                                                                           \
            T *Bc = B + w * r1;                                                         \
            for (int64_t rr = 0; rr < w; rr++) Bc[rr] = x[S->rows[rr]];                 \
            Bc[w] = 1;                                                                  \
            for (int64_t rr = w + 1; rr < r1; rr++) Bc[rr] = x[S->rows[rr]] / piv;      \
            S->ncols = w + 1;                                                           \
            col_sn[j] = open;                                                           \
        } else {                                                                        \
            if (open >= 0) {                                                            \
                /* close the open supernode: its rest rows become the search list */    \
                Supernode *S = &W->sn[open];                                            \
                int64_t nrest = S->nrows - S->ncols;                                    \
                S->dfs_rows = (int64_t *)malloc((size_t)(nrest > 0 ? nrest : 1) * sizeof(int64_t)); \
                if (!S->dfs_rows) return -1;                                            \
                memcpy(S->dfs_rows, S->rows + S->ncols, (size_t)nrest * sizeof(int64_t)); \
                for (int64_t rr = 0; rr < S->nrows; rr++) rowpos[S->rows[rr]] = -1;     \
            }                                                                           \
            int64_t s = W->nsn++;                                                       \
            Supernode *S = &W->sn[s];                                                   \
            S->first = j; S->ncols = 1; S->nrows = n_nonpiv; S->cap = 4;                \
            S->dfs_rows = NULL; S->dfs_end = -1;                                        \
            S->rows = (int64_t *)malloc((size_t)S->nrows * sizeof(int64_t));            \
            S->blk = malloc((size_t)S->cap * (size_t)S->nrows * (is_complex ? sizeof(double complex) : sizeof(double))); \
            if (!S->rows || !S->blk) return -1;                                         \
            T *B = (T *)S->blk;                                                         \
            S->rows[0] = pivrow; rowpos[pivrow] = 0; B[0] = 1;                          \
            int64_t rr = 1;                                                             \
            for (int64_t t = 0; t < n_nonpiv; t++) {                                    \
                int64_t i = nonpiv[t];                                                  \
                if (i == pivrow) continue;                                              \
                S->rows[rr] = i; rowpos[i] = rr; B[rr] = x[i] / piv; rr++;              \
            }                                                                           \
            col_sn[j] = s;                                                              \
            open = s;                                                                   \
        }                                                                               \
        pinv[pivrow] = j; p[j] = pivrow;                                                \
        /* clear the accumulator on the pattern */                                      \
        for (int64_t t = 0; t < n_nonpiv; t++) x[nonpiv[t]] = 0;                        \
        for (int64_t t = top; t < n; t++) {                                             \
            Supernode *S = &W->sn[xi[t]];                                               \
            for (int64_t cc = W->segfirst[xi[t]] - S->first; cc < S->ncols; cc++) x[S->rows[cc]] = 0; \
        }                                                                               \
        x[pivrow] = 0;                                                                  \
        /* prune the closed supernodes whose segments this column reached */            \
        for (int64_t t = top; t < n; t++) {                                             \
            int64_t s = xi[t];                                                          \
            Supernode *S = &W->sn[s];                                                   \
            if (s == open || S->dfs_end >= 0) continue;                                 \
            int64_t nrest = S->nrows - S->ncols;                                        \
            int64_t *rows = S->dfs_rows;                                                \
            int found = 0;                                                              \
            for (int64_t r = 0; r < nrest; r++) if (rows[r] == pivrow) { found = 1; break; } \
            if (!found) continue;                                                       \
            int64_t ph = 0, pt = nrest;                                                 \
            while (ph < pt) {                                                           \
                if (pinv[rows[ph]] >= 0) { ph++; continue; }                            \
                pt--;                                                                   \
                int64_t ti = rows[ph]; rows[ph] = rows[pt]; rows[pt] = ti;              \
            }                                                                           \
            S->dfs_end = pt;                                                            \
        }                                                                               \
        if (verbose >= 2 && (j % (n / 20 + 1) == 0 || j == n - 1))                      \
            printf("[FPLUS]   [%5.1f%%] column %ld/%ld  pivot=%.3e  supernodes=%ld\n",  \
                   100.0 * (double)(j + 1) / (double)n, (long)(j + 1), (long)n, best,   \
                   (long)W->nsn);                                                       \
    }                                                                                   \
    return 0;                                                                           \
}

#ifdef HAVE_BLAS
/* dgemv takes alpha and beta by value, zgemv by pointer; one calling form */
static inline void sn_dgemv(int order, int trans, int M, int N, const double *alpha,
                            const double *A, int lda, const double *X, int incX,
                            const double *beta, double *Y, int incY)
{
    cblas_dgemv(order, trans, M, N, *alpha, A, lda, X, incX, *beta, Y, incY);
}
SN_FACTOR_BODY(sn_factor_z, double complex, cabs, cblas_ztrsv, cblas_zgemv)
SN_FACTOR_BODY(sn_factor_d, double, fabs, cblas_dtrsv, sn_dgemv)
#else
SN_FACTOR_BODY(sn_factor_z, double complex, cabs, (void), (void))
SN_FACTOR_BODY(sn_factor_d, double, fabs, (void), (void))
#endif

static PyObject *
run_factorisation_sn(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax,
                     int is_complex, int verbose)
{
    double t_start = wall_time();
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    int64_t nnzA = Ap[n];
    SnWork W;
    memset(&W, 0, sizeof(W));
    W.n = n; W.is_complex = is_complex;
    W.sn = (Supernode *)calloc((size_t)n, sizeof(Supernode));
    W.col_sn = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.pinv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.p = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.rowpos = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.wrow = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    W.wsn = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    W.segfirst = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.xi = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.nonpiv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.pstack = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.head = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.capU = 4 * nnzA + n + 16;
    W.Up = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    W.Ui = (int64_t *)malloc((size_t)W.capU * sizeof(int64_t));
    W.Ux = malloc((size_t)W.capU * vs);
    W.x = calloc((size_t)n, vs);
    W.y = malloc((size_t)(n > 0 ? 2 * n : 1) * vs);
    PyObject *result = NULL;
    int64_t *Lp = NULL, *Li = NULL, *q = NULL;
    void *Lx = NULL;
    if (!W.sn || !W.col_sn || !W.pinv || !W.p || !W.rowpos || !W.wrow || !W.wsn || !W.segfirst
            || !W.xi || !W.nonpiv || !W.pstack || !W.head || !W.Up || !W.Ui || !W.Ux || !W.x || !W.y) {
        PyErr_NoMemory();
        goto done;
    }
    for (int64_t i = 0; i < n; i++) { W.pinv[i] = -1; W.rowpos[i] = -1; }
    W.Up[0] = 0;
    if (verbose >= 1) {
        printf("[FPLUS] LU factorisation start: n=%ld, nnz(A)=%ld, %s, supernodal left-looking, pivot=partial\n",
               (long)n, (long)nnzA, is_complex ? "complex128" : "float64");
        fflush(stdout);
    }
    int rc = is_complex
        ? sn_factor_z(&W, Ap, Ai, (const double complex *)Ax, verbose)
        : sn_factor_d(&W, Ap, Ai, (const double *)Ax, verbose);
    if (rc == -1) { PyErr_NoMemory(); goto done; }
    if (rc != 0) goto done;

    /* L in CSC, pivot-order rows, the unit diagonal first in every column */
    {
        int64_t nnzL = 0;
        for (int64_t s = 0; s < W.nsn; s++) {
            Supernode *S = &W.sn[s];
            for (int64_t cc = 0; cc < S->ncols; cc++) nnzL += S->nrows - cc;
        }
        Lp = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
        Li = (int64_t *)malloc((size_t)nnzL * sizeof(int64_t));
        Lx = malloc((size_t)nnzL * vs);
        q = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        if (!Lp || !Li || !Lx || !q) { PyErr_NoMemory(); goto done; }
        int64_t o = 0;
        for (int64_t s = 0; s < W.nsn; s++) {
            Supernode *S = &W.sn[s];
            for (int64_t cc = 0; cc < S->ncols; cc++) {
                int64_t j = S->first + cc;
                Lp[j] = o;
                Li[o] = j;
                if (is_complex) ((double complex *)Lx)[o] = 1.0; else ((double *)Lx)[o] = 1.0;
                o++;
                for (int64_t rr = cc + 1; rr < S->nrows; rr++) {
                    Li[o] = W.pinv[S->rows[rr]];
                    if (is_complex) ((double complex *)Lx)[o] = ((double complex *)S->blk)[cc * S->nrows + rr];
                    else            ((double *)Lx)[o] = ((double *)S->blk)[cc * S->nrows + rr];
                    o++;
                }
            }
        }
        Lp[n] = o;
        for (int64_t i = 0; i < n; i++) q[i] = i;
        int64_t *nui = (int64_t *)realloc(W.Ui, (size_t)(W.nnzU > 0 ? W.nnzU : 1) * sizeof(int64_t));
        void *nux = realloc(W.Ux, (size_t)(W.nnzU > 0 ? W.nnzU : 1) * vs);
        if (nui) W.Ui = nui;
        if (nux) W.Ux = nux;
        if (verbose >= 1) {
            printf("[FPLUS] LU factorisation complete: nnz(L)=%ld, nnz(U)=%ld, total nnz=%ld, "
                   "supernodes=%ld, time=%.3fs\n", (long)nnzL, (long)W.nnzU,
                   (long)(nnzL + W.nnzU), (long)W.nsn, wall_time() - t_start);
            fflush(stdout);
        }
        int vt = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
        PyObject *L_data = wrap_array(Lx, nnzL, vt);
        PyObject *L_idx  = wrap_array(Li, nnzL, NPY_INT64);
        PyObject *L_ptr  = wrap_array(Lp, n + 1, NPY_INT64);
        PyObject *U_data = wrap_array(W.Ux, W.nnzU, vt);
        PyObject *U_idx  = wrap_array(W.Ui, W.nnzU, NPY_INT64);
        PyObject *U_ptr  = wrap_array(W.Up, n + 1, NPY_INT64);
        PyObject *p_arr  = wrap_array(W.p, n, NPY_INT64);
        PyObject *q_arr  = wrap_array(q, n, NPY_INT64);
        if (!L_data || !L_idx || !L_ptr || !U_data || !U_idx || !U_ptr || !p_arr || !q_arr) {
            Py_XDECREF(L_data); Py_XDECREF(L_idx); Py_XDECREF(L_ptr);
            Py_XDECREF(U_data); Py_XDECREF(U_idx); Py_XDECREF(U_ptr);
            Py_XDECREF(p_arr); Py_XDECREF(q_arr);
            goto done;
        }
        Lx = NULL; Li = NULL; Lp = NULL; q = NULL;
        W.Ux = NULL; W.Ui = NULL; W.Up = NULL; W.p = NULL;
        result = Py_BuildValue("{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:s}",
                               "L_data", L_data, "L_indices", L_idx, "L_indptr", L_ptr,
                               "U_data", U_data, "U_indices", U_idx, "U_indptr", U_ptr,
                               "p", p_arr, "q", q_arr, "format", "csc");
    }
done:
    if (W.sn)
        for (int64_t s = 0; s < W.nsn; s++) { free(W.sn[s].rows); free(W.sn[s].blk); free(W.sn[s].dfs_rows); }
    free(W.sn); free(W.col_sn); free(W.pinv); free(W.p); free(W.rowpos); free(W.wrow); free(W.wsn);
    free(W.segfirst); free(W.xi); free(W.nonpiv); free(W.pstack); free(W.head);
    free(W.Up); free(W.Ui); free(W.Ux); free(W.x); free(W.y);
    free(Lp); free(Li); free(Lx); free(q);
    return result;
}

static PyObject *
py_lu_factor_sn(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    Py_ssize_t n_sst;
    int is_complex, verbose = 0;
    if (!PyArg_ParseTuple(args, "OOOni|i", &data_obj, &indices_obj, &indptr_obj,
                          &n_sst, &is_complex, &verbose))
        return NULL;
    int64_t n = (int64_t)n_sst;
    int data_type = is_complex ? NPY_COMPLEX128 : NPY_FLOAT64;
    PyArrayObject *data_arr = (PyArrayObject *)PyArray_FROM_OTF(data_obj, data_type, NPY_ARRAY_IN_ARRAY);
    if (!data_arr) return NULL;
    PyArrayObject *idx_arr = (PyArrayObject *)PyArray_FROM_OTF(indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ptr_arr = (PyArrayObject *)PyArray_FROM_OTF(indptr_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!idx_arr || !ptr_arr) {
        Py_DECREF(data_arr); Py_XDECREF(idx_arr); Py_XDECREF(ptr_arr);
        return NULL;
    }
    PyObject *result = run_factorisation_sn(
        n, (const int64_t *)PyArray_DATA(ptr_arr), (const int64_t *)PyArray_DATA(idx_arr),
        PyArray_DATA(data_arr), is_complex, verbose);
    Py_DECREF(data_arr); Py_DECREF(idx_arr); Py_DECREF(ptr_arr);
    return result;
}


/* =========================================================================
 * Triangular solves on the CSC factors: x = U^-1 L^-1 b (trans 0),
 * x = L^-T U^-T b (trans 1) or x = L^-H U^-H b (trans 2), in place on a
 * copy of b.  L is unit lower (an explicit unit diagonal is skipped), U is
 * upper with its diagonal stored.
 * ========================================================================= */

#define TRI_SOLVE_BODY(NAME, T, CONJ)                                                       \
static int NAME(int64_t n, const int64_t *Lp, const int64_t *Li, const T *Lx,              \
                const int64_t *Up, const int64_t *Ui, const T *Ux, T *x, int trans)         \
{                                                                                           \
    if (trans == 0) {                                                                       \
        for (int64_t j = 0; j < n; j++) {                                                   \
            T xj = x[j];                                                                    \
            if (xj == 0) continue;                                                          \
            for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) {                                   \
                int64_t i = Li[q];                                                          \
                if (i != j) x[i] -= xj * Lx[q];                                             \
            }                                                                               \
        }                                                                                   \
        for (int64_t j = n - 1; j >= 0; j--) {                                              \
            T d = 0; int have = 0;                                                          \
            for (int64_t q = Up[j]; q < Up[j + 1]; q++)                                     \
                if (Ui[q] == j) { d = Ux[q]; have = 1; break; }                             \
            if (!have || d == 0) return -1;                                                 \
            T xj = x[j] / d;                                                                \
            x[j] = xj;                                                                      \
            if (xj == 0) continue;                                                          \
            for (int64_t q = Up[j]; q < Up[j + 1]; q++) {                                   \
                int64_t i = Ui[q];                                                          \
                if (i < j) x[i] -= xj * Ux[q];                                              \
            }                                                                               \
        }                                                                                   \
        return 0;                                                                           \
    }                                                                                       \
    int cj = (trans == 2);                                                                  \
    for (int64_t j = 0; j < n; j++) {                                                       \
        T s = x[j], d = 0; int have = 0;                                                    \
        for (int64_t q = Up[j]; q < Up[j + 1]; q++) {                                       \
            int64_t i = Ui[q];                                                              \
            T v = cj ? CONJ(Ux[q]) : Ux[q];                                                 \
            if (i < j) s -= v * x[i];                                                       \
            else if (i == j) { d = v; have = 1; }                                           \
        }                                                                                   \
        if (!have || d == 0) return -1;                                                     \
        x[j] = s / d;                                                                       \
    }                                                                                       \
    for (int64_t j = n - 1; j >= 0; j--) {                                                  \
        T s = x[j];                                                                         \
        for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) {                                       \
            int64_t i = Li[q];                                                              \
            if (i > j) s -= (cj ? CONJ(Lx[q]) : Lx[q]) * x[i];                              \
        }                                                                                   \
        x[j] = s;                                                                           \
    }                                                                                       \
    return 0;                                                                               \
}

static inline double conj_d(double v) { return v; }
TRI_SOLVE_BODY(tri_solve_z, double complex, conj)
TRI_SOLVE_BODY(tri_solve_d, double, conj_d)

static PyObject *
py_lu_solve(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *Lp_o, *Li_o, *Lx_o, *Up_o, *Ui_o, *Ux_o, *b_o;
    int trans, is_complex;
    if (!PyArg_ParseTuple(args, "OOOOOOOii", &Lp_o, &Li_o, &Lx_o, &Up_o, &Ui_o, &Ux_o, &b_o,
                          &trans, &is_complex))
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
        int rc = is_complex
            ? tri_solve_z(n, (int64_t *)PyArray_DATA(Lp), (int64_t *)PyArray_DATA(Li),
                          (double complex *)PyArray_DATA(Lx), (int64_t *)PyArray_DATA(Up),
                          (int64_t *)PyArray_DATA(Ui), (double complex *)PyArray_DATA(Ux),
                          (double complex *)PyArray_DATA(x), trans)
            : tri_solve_d(n, (int64_t *)PyArray_DATA(Lp), (int64_t *)PyArray_DATA(Li),
                          (double *)PyArray_DATA(Lx), (int64_t *)PyArray_DATA(Up),
                          (int64_t *)PyArray_DATA(Ui), (double *)PyArray_DATA(Ux),
                          (double *)PyArray_DATA(x), trans);
        if (rc != 0) {
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


/* =========================================================================
 * Module definition
 * ========================================================================= */
static PyMethodDef SparseLUMethods[] = {
    {"lu_factor", py_lu_factor, METH_VARARGS,
     "lu_factor(data, indices, indptr, n, drop_tol, is_complex) -> dict\n"
     "\n"
     "Sparse LU with full pivoting.  Accepts int32 or int64 index arrays.\n"
     "Returns dict with keys: L_data, L_indices (int64), L_indptr (int64),\n"
     "                        U_data, U_indices (int64), U_indptr (int64),\n"
     "                        p (int64), q (int64).\n"
    },
    {"ruiz_equilibrate", py_ruiz_equilibrate, METH_VARARGS,
     "ruiz_equilibrate(data, indices, indptr, n, max_iter, tol, is_complex)"
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
