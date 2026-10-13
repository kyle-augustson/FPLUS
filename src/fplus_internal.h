/* fplus_internal.h -- shared by the FPLUS core sources (not installed). */
#ifndef FPLUS_INTERNAL_H
#define FPLUS_INTERNAL_H

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE   /* posix_memalign, madvise under -std=c99 */
#endif
#include "fplus.h"
#include <stdarg.h>

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
extern void cblas_dtrsm(int order, int side, int uplo, int transa, int diag, int M, int N,
                        double alpha, const double *A, int lda, double *B, int ldb);
extern void cblas_dgemm(int order, int transa, int transb, int M, int N, int K, double alpha,
                        const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
extern void cblas_ztrsm(int order, int side, int uplo, int transa, int diag, int M, int N,
                        const void *alpha, const void *A, int lda, void *B, int ldb);
extern void cblas_zgemm(int order, int transa, int transb, int M, int N, int K, const void *alpha,
                        const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc);
extern void cblas_zgemv(int order, int trans, int M, int N, const void *alpha,
                        const void *A, int lda, const void *X, int incX,
                        const void *beta, void *Y, int incY);
#  ifndef SN_BLAS_MIN
#    define SN_BLAS_MIN 256   /* segment columns x rest rows below which the loops are used */
#  endif
#endif
/* OpenMP regions only when more than one thread may run them.  An inactive region still costs a runtime call, and Cray's
 * OpenMP runtime (libsci_cray_mp) aborts the process when a region is opened from a thread it did not create (FPLUS called
 * from several Python threads), even with one thread. */
static inline int fplus_omp_active(void)
{
#ifdef _OPENMP
    return omp_get_max_threads() > 1;
#else
    return 0;
#endif
}

#ifndef SN_PAR_MIN
#  define SN_PAR_MIN 16384   /* entries below which the gather / scatter / L-store loops stay serial (OpenMP) */
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

/* Extra capacity added to each row on first allocation (row kernel). */
#define INITIAL_SLACK 4

/* format an error message into err (when err is not NULL) */
static inline void fplus_errf(char *err, size_t errlen, const char *fmt, ...)
{
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* A zeroed allocation for large, randomly accessed workspaces: 2 MB-aligned and advised for transparent huge pages on
 * Linux (fewer TLB misses -- COLAMD on a 7e7-entry pattern: 5.7 -> 3.8 s; numpy does the same for its large arrays),
 * calloc elsewhere.  Released with free(). */
#if defined(__linux__)
#  include <sys/mman.h>
#endif
static inline void *fplus_calloc_huge(size_t count, size_t size)
{
    size_t bytes = count * size;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const size_t huge = (size_t)2 << 20;
    if (bytes >= huge) {
        void *p = NULL;
        size_t rounded = (bytes + huge - 1) / huge * huge;
        if (posix_memalign(&p, huge, rounded) != 0) return NULL;
        madvise(p, rounded, MADV_HUGEPAGE);
        memset(p, 0, bytes);
        return p;
    }
#endif
    return calloc(count > 0 ? count : 1, size);
}

/* Structure check of a compressed matrix (CSC: ptr over columns, idx = rows; CSR the same with roles swapped): n >= 0,
 * ptr[0] == 0, ptr non-decreasing, every index in [0, n); canonical: indices strictly increasing within each line (sorted,
 * no duplicates).  O(nnz).  Malformed input would otherwise index out of bounds (audit 2026-10-10). */
static inline int fplus_check_compressed(int64_t n, const int64_t *ptr, const int64_t *idx, int canonical, char *err, size_t errlen)
{
    if (n < 0) { fplus_errf(err, errlen, "n must be >= 0 (got %ld)", (long)n); return FPLUS_ERR_BADARG; }
    if (ptr[0] != 0) { fplus_errf(err, errlen, "indptr[0] must be 0 (got %ld)", (long)ptr[0]); return FPLUS_ERR_BADARG; }
    for (int64_t j = 0; j < n; j++) {
        if (ptr[j + 1] < ptr[j]) { fplus_errf(err, errlen, "indptr decreases at %ld", (long)j); return FPLUS_ERR_BADARG; }
        for (int64_t q = ptr[j]; q < ptr[j + 1]; q++) {
            if (idx[q] < 0 || idx[q] >= n) { fplus_errf(err, errlen, "index %ld out of range [0, %ld) in line %ld", (long)idx[q], (long)n, (long)j); return FPLUS_ERR_BADARG; }
            if (canonical && q > ptr[j] && idx[q] <= idx[q - 1]) {
                fplus_errf(err, errlen, "indices not strictly increasing in line %ld (canonical form required: sorted, no duplicates)", (long)j);
                return FPLUS_ERR_BADARG;
            }
        }
    }
    return FPLUS_OK;
}

/* every value finite (NaN / Inf would pass the pivot search unnoticed and contaminate the factors) */
static inline int fplus_check_finite(int64_t nnz, const void *x, int is_complex, char *err, size_t errlen)
{
    const double *d = (const double *)x;
    const int64_t m = is_complex ? 2 * nnz : nnz;
    for (int64_t k = 0; k < m; k++)
        if (!isfinite(d[k])) { fplus_errf(err, errlen, "non-finite value in entry %ld", (long)(is_complex ? k / 2 : k)); return FPLUS_ERR_BADARG; }
    return FPLUS_OK;
}

/* p is a permutation of 0..n-1 (work: n bytes, zeroed by the caller) */
static inline int fplus_check_perm(int64_t n, const int64_t *p, unsigned char *seen, const char *name, char *err, size_t errlen)
{
    for (int64_t i = 0; i < n; i++) {
        if (p[i] < 0 || p[i] >= n || seen[p[i]]) { fplus_errf(err, errlen, "%s is not a permutation of 0..n-1 (entry %ld)", name, (long)i); return FPLUS_ERR_BADARG; }
        seen[p[i]] = 1;
    }
    return FPLUS_OK;
}

#endif /* FPLUS_INTERNAL_H */
