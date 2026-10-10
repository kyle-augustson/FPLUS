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

#endif /* FPLUS_INTERNAL_H */
