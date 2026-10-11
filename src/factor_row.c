/* factor_row.c -- row-oriented sparse LU with full / partial / rook pivoting (CSR in, CSR factors out). */
#include "fplus_internal.h"


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
        r->cols = NULL; r->vals_z = NULL; r->vals_d = NULL; r->cap = 0;   /* the caller's cleanup frees again (audit 2026-10-10) */
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
            tmp_cols[U]    = ~ci;            /* untouched by this update: passes through unfiltered */
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
        tmp_cols[U]    = ~ri->cols[pi];
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

    /* Pass 3: filter the computed entries (drop exact zeros and, for ILU, |v| <= drop_tol), strip column k, and write
     * back to ri.  Entries of row i that this update did not touch (marked ~col in pass 1) pass through unchanged, as in
     * the merge path and the Python reference: drop_tol applies to the values an update computes, so the factors do not
     * depend on which path a row's size selects. */
    int64_t final_nnz = 0;
    for (int64_t p = 0; p < U; p++) {
        int64_t c = tmp_cols[p];
        double complex v = tmp_vals_i[p];
        if (c < 0) {
            c = ~c;                          /* entry of row i only: kept as is, as in the merge path */
        } else {
            if (c == k) continue;
            double av = cabs(v);
            if (av == 0.0) continue;
            if (drop_tol > 0.0 && av <= drop_tol) continue;
        }
        tmp_cols[final_nnz]   = c;
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
            tmp_cols[U]    = ~ci;            /* untouched by this update: passes through unfiltered */
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
        tmp_cols[U]    = ~ri->cols[pi];
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
        int64_t c = tmp_cols[p];
        double v = tmp_vals_i[p];
        if (c < 0) {
            c = ~c;                          /* entry of row i only: kept as is, as in the merge path */
        } else {
            if (c == k) continue;
            double av = fabs(v);
            if (av == 0.0) continue;
            if (drop_tol > 0.0 && av <= drop_tol) continue;
        }
        tmp_cols[final_nnz]   = c;
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
            if (drop_tol > 0.0 && j != i && cabs(v) <= drop_tol) continue;   /* never drop the diagonal */
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
            if (drop_tol > 0.0 && j != i && cabs(v) <= drop_tol) continue;   /* never drop the diagonal */
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
            if (drop_tol > 0.0 && j != i && fabs(v) <= drop_tol) continue;   /* never drop the diagonal */
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
            if (drop_tol > 0.0 && j != i && fabs(v) <= drop_tol) continue;   /* never drop the diagonal */
            if ( upper && j >= i) { indices[pos] = j; data[pos] = v; pos++; }
            if (!upper && j <  i) { indices[pos] = j; data[pos] = v; pos++; }
        }
    }

    *out_indptr = indptr; *out_indices = indices; *out_data = data;
    return 0;
}

/* =========================================================================
 * Core factorisation driver.
 * ========================================================================= */
/* Pivot strategy selectors FPLUS_PIVOT_* (fplus.h) must match the values in sparse_lu_full_pivot.py. */

int fplus_factor_row(int64_t n,
                     const int64_t * restrict in_indptr,
                     const int64_t * restrict in_indices,
                     const void    * restrict in_data,
                     double drop_tol, int is_complex, int verbose,
                     int pivot_strategy, double pivot_rel,
                     fplus_factors *out, char *err, size_t errlen)
{
    int ret = FPLUS_ERR_NOMEM;
    memset(out, 0, sizeof(*out));
    if (pivot_strategy != FPLUS_PIVOT_FULL && pivot_strategy != FPLUS_PIVOT_PARTIAL && pivot_strategy != FPLUS_PIVOT_ROOK) {
        fplus_errf(err, errlen, "pivot_strategy must be 0 (full), 1 (partial), or 2 (rook); got %d", pivot_strategy);
        return FPLUS_ERR_BADARG;
    }
    if (!(pivot_rel >= 0.0)) { fplus_errf(err, errlen, "pivot_rel must be >= 0"); return FPLUS_ERR_BADARG; }
    if (!(drop_tol >= 0.0)) { fplus_errf(err, errlen, "drop_tol must be >= 0 (got %g)", drop_tol); return FPLUS_ERR_BADARG; }
    { int vrc = fplus_check_compressed(n, in_indptr, in_indices, 1, err, errlen);   /* the row kernel needs sorted, unique rows */
      if (vrc == FPLUS_OK) vrc = fplus_check_finite(in_indptr[n], in_data, is_complex, err, errlen);
      if (vrc != FPLUS_OK) return vrc; }
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
            /* relative to the largest |a_ij| of the input (for full pivoting the step-0 pivot itself); the step-0 pivot
               of partial / rook pivoting is only a column maximum (audit 2026-10-10) */
            max_entry_k0 = best_abs_k;
            for (int64_t q = 0; q < in_indptr[n]; q++) {
                double a = is_complex ? cabs(((const double complex *)in_data)[q]) : fabs(((const double *)in_data)[q]);
                if (a > max_entry_k0) max_entry_k0 = a;
            }
            pivot_threshold = pivot_rel * max_entry_k0;
        }

        if (!found || best_abs_k < pivot_threshold) {
            fplus_errf(err, errlen,
                "Near-zero or zero pivot %.6e at step %ld "
                "(threshold %.6e). Matrix is singular.",
                best_abs_k, (long)k, pivot_threshold);
            ret = FPLUS_ERR_SINGULAR;
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
            fplus_errf(err, errlen,
                "Pivot entry missing at step %ld after swap.", (long)k);
            ret = FPLUS_ERR_INTERNAL;
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

    /* hand the buffers to the caller */
    out->n = n; out->is_complex = is_complex; out->csc = 0;
    out->Lp = L_indptr; out->Li = L_indices; out->Lx = is_complex ? (void *)L_data_z : (void *)L_data_d;
    out->Up = U_indptr; out->Ui = U_indices; out->Ux = is_complex ? (void *)U_data_z : (void *)U_data_d;
    out->p = p; out->q = q;
    return FPLUS_OK;

oom:
    ret = FPLUS_ERR_NOMEM;
    fplus_errf(err, errlen, "out of memory");
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
    return ret;
}

