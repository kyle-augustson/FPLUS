/* fplus.h -- FPLUS sparse LU, the C interface (phase 1 of docs/C_LIBRARY_PLAN.md).
 *
 * Index arrays are int64; values are double or double complex (C99), selected by is_complex.  Every entry point returns
 * FPLUS_OK or a negative FPLUS_ERR_* code, with a message in err (when err is not NULL).  There is no global state outside
 * profiling builds (-DFPLUS_PROFILE), so separate calls may run in separate threads.
 */
#ifndef FPLUS_H
#define FPLUS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FPLUS_OK = 0,
    FPLUS_ERR_NOMEM = -1,      /* allocation failed */
    FPLUS_ERR_SINGULAR = -2,   /* no admissible pivot, an all-zero row / column, or a zero diagonal in a solve */
    FPLUS_ERR_BADARG = -3,     /* invalid argument */
    FPLUS_ERR_GPU = -4,        /* the optional GPU path failed */
    FPLUS_ERR_INTERNAL = -5
};

/* pivot strategies of the row kernel (the supernodal kernel is partial pivoting) */
#define FPLUS_PIVOT_FULL    0
#define FPLUS_PIVOT_PARTIAL 1
#define FPLUS_PIVOT_ROOK    2

/* default relative singular-pivot threshold: sqrt(DBL_EPSILON), the LAPACK choice */
#define FPLUS_PIVOT_REL_DEFAULT 1.4901161193847656e-08

/* L and U with their permutations, malloc'd; P A Q = L U with p = row order, q = column order.  csc: compressed columns
 * (supernodal kernel) or rows (row kernel).  fplus_factors_free releases what is still owned (a caller may take
 * ownership of an array and set its pointer to NULL). */
typedef struct {
    int64_t n;
    int is_complex;
    int csc;
    int64_t *Lp, *Li; void *Lx;
    int64_t *Up, *Ui; void *Ux;
    int64_t *p, *q;
    int64_t nsupernodes;
} fplus_factors;

void fplus_factors_free(fplus_factors *f);

/* Supernodal left-looking LU with partial pivoting of the CSC matrix (Ap, Ai, Ax).  relax_rows: relaxed supernode joins
 * (rows of pattern mismatch allowed); panel: columns per panel (complex only; > 1 batches the updates); use_gpu: cuBLAS
 * zgemm for updates above gpu_min_flops (dlopen; complex, panel > 1). */
int fplus_factor_sn(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex, int verbose,
                    int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                    fplus_factors *out, char *err, size_t errlen);

/* Row-oriented LU of the CSR matrix (indptr, indices, data) with full, partial or rook pivoting and an optional drop
 * tolerance; L and U row-compressed. */
int fplus_factor_row(int64_t n, const int64_t *indptr, const int64_t *indices, const void *data, double drop_tol,
                     int is_complex, int verbose, int pivot_strategy, double pivot_rel,
                     fplus_factors *out, char *err, size_t errlen);

/* Ruiz L-inf equilibration in place: data (CSR, or CSC when is_csc) becomes diag(Dr) A diag(Dc); Dr, Dc (length n) are
 * filled.  The CSC route is bit-identical to the CSR route on the same matrix. */
int fplus_ruiz(int64_t n, const int64_t *indptr, const int64_t *indices, void *data, int is_complex, int is_csc,
               int max_iter, double tol, int verbose, double *Dr, double *Dc, char *err, size_t errlen);

/* Triangular solves on CSC factors in place on x: trans 0 solves L U x = b, 1 (L U)^T x = b, 2 (L U)^H x = b.
 * L unit lower (an explicit unit diagonal is skipped), U upper with its diagonal stored. */
int fplus_tri_solve(int64_t n, const int64_t *Lp, const int64_t *Li, const void *Lx, const int64_t *Up, const int64_t *Ui,
                    const void *Ux, void *x, int trans, int is_complex);

/* Postorder of the column elimination tree of the m x n CSC pattern (SuperLU's sp_coletree + TreePostorder):
 * post[k] = the column placed k-th. */
int fplus_coletree_postorder(int64_t m, int64_t n, const int64_t *Ap, const int64_t *Ai, int64_t *post);

#ifdef __cplusplus
}
#endif
#endif /* FPLUS_H */
