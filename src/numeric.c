/* numeric.c -- the high-level C interface: options, analyze with options, factor (composed orders and scalings), solve.
 * The composition and the solves follow the Python layer (sparse_lu_full_pivot.py: Step 3 of sparse_lu_full_pivot, SparseLU's
 * _solve_single / _solve_transposed) operation for operation. */
#include "fplus_internal.h"

struct fplus_numeric {
    int64_t n;
    int is_complex;
    fplus_factors f;            /* L, U of diag(Dr) A[p,:][:,q] diag(Dc) */
    fplus_solve_plan *plan;     /* the blocked solves on f */
    int64_t *p, *q;             /* composed with the static orders: original coordinates */
    double *Dr, *Dc;            /* original coordinates */
};

void fplus_default_options(fplus_options *o)
{
    o->pin_dense = 1; o->tau_density = 0.5; o->tau_min_nnz = 10; o->postorder = 1;
    o->equilibrate = 1; o->ruiz_max_iter = 20; o->ruiz_tol = 1e-2;
    o->pivot_rel = FPLUS_PIVOT_REL_DEFAULT; o->pivot_strategy = FPLUS_PIVOT_PARTIAL; o->drop_tol = 0.0;
    o->relax_rows = 32; o->panel = 512;
    o->use_gpu = 0; o->gpu_min_flops = 5e7; o->verbose = 0;
}

int fplus_analyze_opts(int64_t n, const int64_t *Ap, const int64_t *Ai, const fplus_options *opt,
                       fplus_symbolic **sym, char *err, size_t errlen)
{
    fplus_options d; if (!opt) { fplus_default_options(&d); opt = &d; }
    return fplus_analyze(n, Ap, Ai, opt->pin_dense, opt->tau_density, opt->tau_min_nnz, NULL, NULL, opt->postorder, sym, err, errlen);
}

void fplus_numeric_free(fplus_numeric *num)
{
    if (!num) return;
    fplus_factors_free(&num->f);
    fplus_solve_plan_free(num->plan);
    free(num->p); free(num->q); free(num->Dr); free(num->Dc); free(num);
}

int fplus_factor(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                 const fplus_options *opt, fplus_numeric **out, char *err, size_t errlen)
{
    *out = NULL;
    fplus_options d; if (!opt) { fplus_default_options(&d); opt = &d; }
    const int64_t n = fplus_symbolic_n(sym);
    const size_t nn = (size_t)(n > 0 ? n : 1);
    fplus_numeric *num = (fplus_numeric *)calloc(1, sizeof(fplus_numeric));
    double *Drp = (double *)malloc(nn * sizeof(double)), *Dcp = (double *)malloc(nn * sizeof(double));
    if (!num || !Drp || !Dcp) { free(num); free(Drp); free(Dcp); fplus_errf(err, errlen, "out of memory"); return FPLUS_ERR_NOMEM; }
    num->n = n; num->is_complex = is_complex;
    int rc;
    if (opt->pivot_strategy != FPLUS_PIVOT_PARTIAL && opt->pivot_strategy != FPLUS_PIVOT_ROOK && opt->pivot_strategy != FPLUS_PIVOT_FULL) {
        fplus_errf(err, errlen, "pivot_strategy must be FPLUS_PIVOT_PARTIAL, _ROOK or _FULL (got %d)", opt->pivot_strategy);
        rc = FPLUS_ERR_BADARG;
    } else if (!(opt->drop_tol >= 0.0)) {
        fplus_errf(err, errlen, "drop_tol must be >= 0 (got %g)", opt->drop_tol);
        rc = FPLUS_ERR_BADARG;
    } else if (opt->pivot_strategy == FPLUS_PIVOT_PARTIAL && opt->drop_tol == 0.0) {
        rc = fplus_factor_sn_sym(sym, Ap, Ai, Ax, is_complex, opt->equilibrate, opt->ruiz_max_iter, opt->ruiz_tol, opt->verbose,
                                 opt->relax_rows, opt->panel, opt->use_gpu, opt->gpu_min_flops, opt->pivot_rel, &num->f,
                                 Drp, Dcp, err, errlen);
    } else {
        rc = fplus_factor_row_sym(sym, Ap, Ai, Ax, is_complex, opt->equilibrate, opt->ruiz_max_iter, opt->ruiz_tol, opt->verbose,
                                  opt->drop_tol, opt->pivot_strategy, opt->pivot_rel, &num->f, Drp, Dcp, err, errlen);
    }
    if (rc != FPLUS_OK) { free(Drp); free(Dcp); fplus_numeric_free(num); return rc; }
    num->p = (int64_t *)malloc(nn * sizeof(int64_t)); num->q = (int64_t *)malloc(nn * sizeof(int64_t));
    num->Dr = (double *)malloc(nn * sizeof(double)); num->Dc = (double *)malloc(nn * sizeof(double));
    if (!num->p || !num->q || !num->Dr || !num->Dc) {
        free(Drp); free(Dcp); fplus_numeric_free(num); fplus_errf(err, errlen, "out of memory"); return FPLUS_ERR_NOMEM;
    }
    const int64_t *rp = fplus_symbolic_row_perm(sym), *cp = fplus_symbolic_col_perm(sym);
    for (int64_t k = 0; k < n; k++) {
        num->p[k] = rp[num->f.p[k]];         /* p_final = row_perm[p_lu] */
        num->q[k] = cp[num->f.q[k]];         /* q_final = col_perm[q_lu] */
        num->Dr[rp[k]] = Drp[k];             /* Dr_final = Dr[inv_row_perm] */
        num->Dc[cp[k]] = Dcp[k];             /* Dc_final = Dc[inv_col_perm] */
    }
    free(Drp); free(Dcp);
    rc = fplus_solve_plan_create(n, num->f.Lp, num->f.Li, num->f.Up, num->f.Ui, &num->plan, err, errlen);
    if (rc != FPLUS_OK) { fplus_numeric_free(num); return rc; }
    *out = num;
    return FPLUS_OK;
}

/* the right-hand sides go through the plan in groups: scaled and permuted into w (n x g), solved together, scattered back */
#define SOLVE_GROUP 32

int fplus_solve(const fplus_numeric *num, void *b, int64_t nrhs, int64_t ldb, int trans, char *err, size_t errlen)
{
    const int64_t n = num->n;
    if (trans < 0 || trans > 2 || nrhs < 0 || ldb < n) { fplus_errf(err, errlen, "fplus_solve: bad trans, nrhs or ldb"); return FPLUS_ERR_BADARG; }
    const fplus_factors *f = &num->f;
    const int64_t *p = num->p, *q = num->q;
    const double *Dr = num->Dr, *Dc = num->Dc;
    const size_t vs = num->is_complex ? sizeof(double complex) : sizeof(double);
    const int64_t g = nrhs < SOLVE_GROUP ? nrhs : SOLVE_GROUP;
    void *w = malloc((size_t)(n > 0 ? n : 1) * (size_t)(g > 0 ? g : 1) * vs);
    if (!w) { fplus_errf(err, errlen, "out of memory"); return FPLUS_ERR_NOMEM; }
    int rc = FPLUS_OK;
    for (int64_t r0 = 0; r0 < nrhs && rc == FPLUS_OK; r0 += g) {
        const int64_t m = nrhs - r0 < g ? nrhs - r0 : g;
        for (int64_t r = 0; r < m; r++) {
            if (num->is_complex) {
                const double complex *x = (const double complex *)b + (r0 + r) * ldb; double complex *ww = (double complex *)w + r * n;
                if (trans == 0) for (int64_t k = 0; k < n; k++) ww[k] = Dr[p[k]] * x[p[k]];   /* b_perm = Dr[p] * b[p] */
                else            for (int64_t k = 0; k < n; k++) ww[k] = Dc[q[k]] * x[q[k]];   /* rhs = Dc[q] * b[q] */
            } else {
                const double *x = (const double *)b + (r0 + r) * ldb; double *ww = (double *)w + r * n;
                if (trans == 0) for (int64_t k = 0; k < n; k++) ww[k] = Dr[p[k]] * x[p[k]];
                else            for (int64_t k = 0; k < n; k++) ww[k] = Dc[q[k]] * x[q[k]];
            }
        }
        rc = fplus_tri_solve_plan(num->plan, n, f->Lp, f->Li, f->Lx, f->Up, f->Ui, f->Ux, w, m, n, trans, num->is_complex);
        if (rc != FPLUS_OK) break;
        for (int64_t r = 0; r < m; r++) {
            if (num->is_complex) {
                double complex *x = (double complex *)b + (r0 + r) * ldb; const double complex *ww = (const double complex *)w + r * n;
                if (trans == 0) for (int64_t k = 0; k < n; k++) x[q[k]] = Dc[q[k]] * ww[k];   /* x[q] = Dc[q] * z */
                else            for (int64_t k = 0; k < n; k++) x[p[k]] = Dr[p[k]] * ww[k];   /* x[p] = Dr[p] * w */
            } else {
                double *x = (double *)b + (r0 + r) * ldb; const double *ww = (const double *)w + r * n;
                if (trans == 0) for (int64_t k = 0; k < n; k++) x[q[k]] = Dc[q[k]] * ww[k];
                else            for (int64_t k = 0; k < n; k++) x[p[k]] = Dr[p[k]] * ww[k];
            }
        }
    }
    free(w);
    if (rc == FPLUS_ERR_SINGULAR) fplus_errf(err, errlen, "fplus_solve: zero or missing diagonal in U");
    else if (rc != FPLUS_OK) fplus_errf(err, errlen, "fplus_solve: error %d", rc);
    return rc;
}

const fplus_factors *fplus_numeric_factors(const fplus_numeric *num) { return &num->f; }
const int64_t *fplus_numeric_p(const fplus_numeric *num) { return num->p; }
const int64_t *fplus_numeric_q(const fplus_numeric *num) { return num->q; }
const double *fplus_numeric_Dr(const fplus_numeric *num) { return num->Dr; }
const double *fplus_numeric_Dc(const fplus_numeric *num) { return num->Dc; }
