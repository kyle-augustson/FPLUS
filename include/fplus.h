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
    int64_t n_pivots_replaced;     /* incomplete LU: pivots replaced at a breakdown (see fplus_factor) */
} fplus_factors;

void fplus_factors_free(fplus_factors *f);

/* Supernodal left-looking LU with partial pivoting of the CSC matrix (Ap, Ai, Ax).  relax_rows: relaxed supernode joins
 * (rows of pattern mismatch allowed; < 0: automatic, by the share of explicit zeros, CHOLMOD-style); panel: columns per panel
 * (> 1 batches the updates, real and complex; < 0: from the thread count); use_gpu: cuBLAS zgemm for updates above
 * gpu_min_flops (dlopen; complex, panel > 1). */
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
 * L unit lower (an explicit unit diagonal is skipped), U upper with its diagonal stored.  The factors are trusted (they come
 * from fplus_factor_*): no structure check here, it would cost a pass over L and U per solve. */
int fplus_tri_solve(int64_t n, const int64_t *Lp, const int64_t *Li, const void *Lx, const int64_t *Up, const int64_t *Ui,
                    const void *Ux, void *x, int trans, int is_complex);

/* ---- solve plans: blocked triangular solves, threaded inside large supernodes, several right-hand sides at once ----
 * A plan is built once per factorization from the CSC patterns (fplus_factor_* output, or any CSC factors with the unit
 * diagonal stored first in every L column); fplus_tri_solve_plan then solves for nrhs right-hand sides stored column-major
 * in x (leading dimension ldx >= n), in place, with the same factor arrays.  The solves with trans 0 are bit-identical to
 * fplus_tri_solve at any thread count; the transposed solves agree with it to round-off.  Factors without the supernodal
 * layout fall back to fplus_tri_solve per right-hand side. */
typedef struct fplus_solve_plan fplus_solve_plan;
int fplus_solve_plan_create(int64_t n, const int64_t *Lp, const int64_t *Li, const int64_t *Up, const int64_t *Ui,
                            fplus_solve_plan **plan, char *err, size_t errlen);
int fplus_tri_solve_plan(const fplus_solve_plan *plan, int64_t n, const int64_t *Lp, const int64_t *Li, const void *Lx,
                         const int64_t *Up, const int64_t *Ui, const void *Ux, void *x, int64_t nrhs, int64_t ldx,
                         int trans, int is_complex);
int64_t fplus_solve_plan_nsupernodes(const fplus_solve_plan *plan);   /* L supernodes found, -1 for the fallback */
void fplus_solve_plan_free(fplus_solve_plan *plan);

/* Postorder of the column elimination tree of the m x n CSC pattern (SuperLU's sp_coletree + TreePostorder):
 * post[k] = the column placed k-th. */
int fplus_coletree_postorder(int64_t m, int64_t n, const int64_t *Ap, const int64_t *Ai, int64_t *post);

/* ---- symbolic analysis: the static orders of one sparsity pattern, reusable for any values on it ---- */
typedef struct fplus_symbolic fplus_symbolic;

/* Dense-line pinning (a row / column is dense with more than max(tau_density n, tau_min_nnz) entries, or as given by the
 * 0/1 masks when not NULL), COLAMD on the remaining pattern, and (postorder) the column-etree postorder.  Ap, Ai: the
 * n x n pattern in canonical CSC (rows ascending in each column, no duplicates).  Dense rows go last in the row order,
 * dense columns last in the column order. */
int fplus_analyze(int64_t n, const int64_t *Ap, const int64_t *Ai, int pin_dense, double tau_density, int64_t tau_min_nnz,
                  const unsigned char *tau_row_mask, const unsigned char *tau_col_mask, int postorder,
                  fplus_symbolic **sym, char *err, size_t errlen);
/* fplus_analyze with a choice of the fill-reducing column order (fplus_analyze: FPLUS_ORDER_COLAMD).  FPLUS_ORDER_AMD_ATA:
 * minimum degree (SuiteSparse AMD) on the pattern of A^T A without the dense lines.  FPLUS_ORDER_AUTO: for small sparse
 * matrices (n <= 4096, nnz(A) <= 24 n) minimum degree when its Cholesky factor of A^T A is at least 5% smaller than COLAMD's,
 * COLAMD otherwise. */
#define FPLUS_ORDER_COLAMD  0
#define FPLUS_ORDER_AMD_ATA 1
#define FPLUS_ORDER_AUTO    2
int fplus_analyze_ord(int64_t n, const int64_t *Ap, const int64_t *Ai, int pin_dense, double tau_density, int64_t tau_min_nnz,
                      const unsigned char *tau_row_mask, const unsigned char *tau_col_mask, int postorder, int ordering,
                      fplus_symbolic **sym, char *err, size_t errlen);
void fplus_symbolic_free(fplus_symbolic *sym);
int64_t fplus_symbolic_n(const fplus_symbolic *sym);
int64_t fplus_symbolic_nnz(const fplus_symbolic *sym);
const int64_t *fplus_symbolic_row_perm(const fplus_symbolic *sym);   /* A[row_perm, :][:, col_perm] is factored */
const int64_t *fplus_symbolic_col_perm(const fplus_symbolic *sym);
int64_t fplus_symbolic_n_dense_rows(const fplus_symbolic *sym);
int64_t fplus_symbolic_n_dense_cols(const fplus_symbolic *sym);

/* The supernodal factorization of A[row_perm, :][:, col_perm] (A canonical CSC): the permuted matrix, Ruiz (when
 * equilibrate; Dr, Dc of length n filled, in the permuted order), then fplus_factor_sn.  The factors and p, q refer to
 * the permuted matrix. */
int fplus_factor_sn_perm(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                         const int64_t *row_perm, const int64_t *col_perm,
                         int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                         int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                         fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen);
/* the same with the orders of a symbolic analysis (the pattern must be the analysed one) */
int fplus_factor_sn_sym(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                        int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                        int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                        fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen);

/* The row-kernel factorization of A[row_perm, :][:, col_perm] (A canonical CSC): the permuted matrix, Ruiz (when
 * equilibrate), then fplus_factor_row with drop_tol and pivot_strategy; the factors are returned column-compressed (csc = 1)
 * for fplus_tri_solve. */
int fplus_factor_row_perm(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                          const int64_t *row_perm, const int64_t *col_perm,
                          int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                          double drop_tol, int pivot_strategy, double pivot_rel,
                          fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen);
int fplus_factor_row_sym(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                         int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                         double drop_tol, int pivot_strategy, double pivot_rel,
                         fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen);

/* ---- the high-level interface: options, analyze, factor, solve ---- */
typedef struct {
    int     pin_dense;          /* dense rows / columns last (default 1) */
    double  tau_density;        /* a line is dense with more than max(tau_density n, tau_min_nnz) entries (0.5) */
    int64_t tau_min_nnz;        /* (10) */
    int     postorder;          /* column-etree postorder after COLAMD (1) */
    int     equilibrate;        /* Ruiz (1) */
    int     ruiz_max_iter;      /* (20) */
    double  ruiz_tol;           /* (1e-2) */
    double  pivot_rel;          /* singular when the best pivot < pivot_rel * the column's max (FPLUS_PIVOT_REL_DEFAULT; 0: exact zeros only) */
    int     pivot_strategy;     /* FPLUS_PIVOT_PARTIAL (default), _ROOK or _FULL */
    double  drop_tol;           /* 0 (default): exact LU.  > 0: incomplete LU (ILUT-style threshold dropping), see below */
    int64_t relax_rows;         /* relaxed supernodes (32; < 0 automatic) */
    int64_t panel;              /* panel width (512; < 0 automatic: 1 when n <= 4096 and nnz(A) <= 24 n, else from the thread count) */
    int     use_gpu;            /* cuBLAS updates (0) */
    double  gpu_min_flops;      /* (5e7) */
    int     verbose;            /* (0) */
} fplus_options;

void fplus_default_options(fplus_options *opt);

/* fplus_analyze with the options' pinning / postorder settings */
int fplus_analyze_opts(int64_t n, const int64_t *Ap, const int64_t *Ai, const fplus_options *opt,
                       fplus_symbolic **sym, char *err, size_t errlen);

/* The factorization of the analysed pattern with values Ax (canonical CSC, the analysed pattern): diag(Dr) A[p,:][:,q] diag(Dc)
 * = L U, with p, q, Dr, Dc in the original coordinates.  Partial pivoting with drop_tol 0 runs the supernodal kernel;
 * rook / full pivoting or drop_tol > 0 the row kernel.
 *
 * Incomplete LU (drop_tol > 0): the factorization of the equilibrated, ordered matrix drops, as they arise, every multiplier
 * and every updated or fill value with |v| <= drop_tol, and, when a row becomes a pivot row (its final form, row k of U),
 * its off-pivot entries with |v| <= drop_tol before it updates the rows below.  The pivots (U's diagonal) are never dropped.
 * With Ruiz equilibration (the default) the entries are O(1), so drop_tol acts as a relative threshold.  The factors then
 * satisfy L U = diag(Dr) A[p,:][:,q] diag(Dc) - R with R made of the dropped entries; fplus_solve applies (L U)^-1, a
 * preconditioner for an iterative solver rather than a solve.  When the exact factorization's trailing pivots are at or below
 * drop_tol, dropping can leave no pivot at all (a breakdown): the pivot is then replaced by drop_tol * max|a_ij| as in
 * SuperLU's ILU, and the factors count these replacements (fplus_numeric_factors(num)->n_pivots_replaced).  With drop_tol 0
 * a zero pivot is FPLUS_ERR_SINGULAR. */
typedef struct fplus_numeric fplus_numeric;
int fplus_factor(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                 const fplus_options *opt, fplus_numeric **num, char *err, size_t errlen);

/* Solve in place for nrhs right-hand sides stored column-major in b (leading dimension ldb >= n), of the factor's type:
 * trans 0: A x = b, 1: A^T x = b, 2: A^H x = b. */
int fplus_solve(const fplus_numeric *num, void *b, int64_t nrhs, int64_t ldb, int trans, char *err, size_t errlen);

const fplus_factors *fplus_numeric_factors(const fplus_numeric *num);   /* L, U of the scaled, permuted matrix */
const int64_t *fplus_numeric_p(const fplus_numeric *num);              /* row order: A[p, :] */
const int64_t *fplus_numeric_q(const fplus_numeric *num);              /* column order: A[:, q] */
const double *fplus_numeric_Dr(const fplus_numeric *num);              /* row scaling, original row index */
const double *fplus_numeric_Dc(const fplus_numeric *num);              /* column scaling, original column index */
void fplus_numeric_free(fplus_numeric *num);

#ifdef __cplusplus
}
#endif
#endif /* FPLUS_H */
