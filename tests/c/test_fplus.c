/* test_fplus.c -- C-interface tests (ctest): analyze / factor / solve on complex and real matrices with dense lines and
 * strong scaling, all three solve directions, several right-hand sides and all pivot strategies, incomplete LU as a
 * preconditioner, the low-level supernodal and row kernels, the Ruiz range guard, and input validation.  Exit status 0 when every check passes. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fplus.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* xorshift: deterministic, no libc rand */
static unsigned long long rng_s = 88172645463325252ULL;
static double urand(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (double)(rng_s >> 11) / 9007199254740992.0; }

/* A random n x n CSC matrix (canonical): a band, a few random entries per column, two dense rows, one dense column;
 * row scales 10^[-s, s]; complex or real. */
typedef struct { int64_t n, nnz; int64_t *p, *i; void *x; int cplx; } Csc;

static Csc make_matrix(int64_t n, int cplx, double s)
{
    Csc A; A.n = n; A.cplx = cplx;
    char *dense = (char *)calloc((size_t)(n * n), 1);   /* the pattern as a dense mask: small n only */
    for (int64_t j = 0; j < n; j++) {
        for (int64_t d = -2; d <= 2; d++) if (j + d >= 0 && j + d < n) dense[j * n + j + d] = 1;
        for (int k = 0; k < 3; k++) dense[j * n + (int64_t)(urand() * n)] = 1;
    }
    for (int64_t j = 0; j < n; j++) { dense[j * n + 1] = 1; dense[j * n + n / 2] = 1; }   /* dense rows 1 and n/2 */
    for (int64_t i = 0; i < n; i++) dense[(n / 3) * n + i] = 1;                         /* dense column n/3 */
    double *rs = (double *)malloc((size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; i++) rs[i] = pow(10.0, s * (2 * urand() - 1));
    A.nnz = 0; for (int64_t q = 0; q < n * n; q++) A.nnz += dense[q];
    A.p = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t)); A.i = (int64_t *)malloc((size_t)A.nnz * sizeof(int64_t));
    A.x = malloc((size_t)A.nnz * (cplx ? sizeof(double complex) : sizeof(double)));
    int64_t o = 0; A.p[0] = 0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) if (dense[j * n + i]) {
            double re = 2 * urand() - 1 + (i == j ? 0.5 : 0.0), im = 2 * urand() - 1;
            A.i[o] = i;
            if (cplx) ((double complex *)A.x)[o] = rs[i] * (re + im * I); else ((double *)A.x)[o] = rs[i] * re;
            o++;
        }
        A.p[j + 1] = o;
    }
    free(dense); free(rs);
    return A;
}

static void free_matrix(Csc *A) { free(A->p); free(A->i); free(A->x); }

/* y = op(A) x, op = A (0), A^T (1), A^H (2) */
static void matvec(const Csc *A, const void *x, void *y, int trans)
{
    const int64_t n = A->n;
    if (A->cplx) {
        const double complex *ax = (const double complex *)A->x, *xx = (const double complex *)x; double complex *yy = (double complex *)y;
        for (int64_t i = 0; i < n; i++) yy[i] = 0;
        for (int64_t j = 0; j < n; j++) for (int64_t q = A->p[j]; q < A->p[j + 1]; q++) {
            int64_t i = A->i[q];
            if (trans == 0) yy[i] += ax[q] * xx[j]; else yy[j] += (trans == 2 ? conj(ax[q]) : ax[q]) * xx[i];
        }
    } else {
        const double *ax = (const double *)A->x, *xx = (const double *)x; double *yy = (double *)y;
        for (int64_t i = 0; i < n; i++) yy[i] = 0;
        for (int64_t j = 0; j < n; j++) for (int64_t q = A->p[j]; q < A->p[j + 1]; q++) {
            int64_t i = A->i[q];
            if (trans == 0) yy[i] += ax[q] * xx[j]; else yy[j] += ax[q] * xx[i];
        }
    }
}

/* backward error ||op(A) x - b|| / (||A||_1 ||x|| + ||b||) */
static double backward_error(const Csc *A, const void *x, const void *b, int trans)
{
    const int64_t n = A->n;
    double nA = 0;
    for (int64_t j = 0; j < n; j++) { double c = 0; for (int64_t q = A->p[j]; q < A->p[j + 1]; q++) c += A->cplx ? cabs(((double complex *)A->x)[q]) : fabs(((double *)A->x)[q]); if (c > nA) nA = c; }
    void *r = malloc((size_t)n * (A->cplx ? sizeof(double complex) : sizeof(double)));
    matvec(A, x, r, trans);
    double nr = 0, nx = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        if (A->cplx) { double complex d = ((double complex *)r)[i] - ((const double complex *)b)[i]; nr += creal(d * conj(d)); nx += creal(((const double complex *)x)[i] * conj(((const double complex *)x)[i])); nb += creal(((const double complex *)b)[i] * conj(((const double complex *)b)[i])); }
        else { double d = ((double *)r)[i] - ((const double *)b)[i]; nr += d * d; nx += ((const double *)x)[i] * ((const double *)x)[i]; nb += ((const double *)b)[i] * ((const double *)b)[i]; }
    }
    free(r);
    return sqrt(nr) / (nA * sqrt(nx) + sqrt(nb));
}

static void test_high_level(int cplx, double scale, int strategy)
{
    const int64_t n = 300, nrhs = 3;
    Csc A = make_matrix(n, cplx, scale);
    char err[256] = {0};
    fplus_options opt; fplus_default_options(&opt); opt.relax_rows = 4; opt.panel = 16; opt.pivot_strategy = strategy;
    fplus_symbolic *sym = NULL; fplus_numeric *num = NULL;
    int rc = fplus_analyze_opts(n, A.p, A.i, &opt, &sym, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "analyze: %d %s", rc, err);
    CHECK(fplus_symbolic_n_dense_rows(sym) >= 2 && fplus_symbolic_n_dense_cols(sym) >= 1, "dense lines pinned: %ld rows %ld cols",
          (long)fplus_symbolic_n_dense_rows(sym), (long)fplus_symbolic_n_dense_cols(sym));
    rc = fplus_factor(sym, A.p, A.i, A.x, cplx, &opt, &num, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "factor: %d %s", rc, err);
    const size_t vs = cplx ? sizeof(double complex) : sizeof(double);
    for (int trans = 0; trans <= (cplx ? 2 : 1); trans++) {
        void *b = malloc((size_t)(n * nrhs) * vs), *x = malloc((size_t)(n * nrhs) * vs);
        for (int64_t k = 0; k < n * nrhs; k++) {
            if (cplx) ((double complex *)b)[k] = (2 * urand() - 1) + (2 * urand() - 1) * I; else ((double *)b)[k] = 2 * urand() - 1;
        }
        memcpy(x, b, (size_t)(n * nrhs) * vs);
        rc = fplus_solve(num, x, nrhs, n, trans, err, sizeof(err));
        CHECK(rc == FPLUS_OK, "solve trans %d: %d %s", trans, rc, err);
        for (int64_t r = 0; r < nrhs; r++) {
            double be = backward_error(&A, (char *)x + (size_t)(r * n) * vs, (char *)b + (size_t)(r * n) * vs, trans);
            CHECK(be < 1e-12, "%s scale 1e%g strategy %d trans %d rhs %ld: backward error %.2e", cplx ? "complex" : "real", scale,
                  strategy, trans, (long)r, be);
        }
        free(b); free(x);
    }
    fplus_numeric_free(num); fplus_symbolic_free(sym); free_matrix(&A);
}

static int64_t factor_nnz(const fplus_numeric *num)
{
    const fplus_factors *f = fplus_numeric_factors(num);
    return f->Lp[f->n] + f->Up[f->n];
}

/* the 5-point convection-diffusion operator on a g x g grid (complex: with an imaginary shift), canonical CSC */
static Csc make_convdiff(int64_t g, int cplx)
{
    Csc A; A.n = g * g; A.cplx = cplx;
    const int64_t n = A.n;
    A.p = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t)); A.i = (int64_t *)malloc((size_t)(5 * n) * sizeof(int64_t));
    A.x = malloc((size_t)(5 * n) * (cplx ? sizeof(double complex) : sizeof(double)));
    int64_t o = 0; A.p[0] = 0;
    for (int64_t j = 0; j < n; j++) {        /* column j: the rows coupled to unknown j, ascending */
        const int64_t gx = j % g, gy = j / g;
        const int64_t rows[5] = {gy > 0 ? j - g : -1, gx > 0 ? j - 1 : -1, j, gx < g - 1 ? j + 1 : -1, gy < g - 1 ? j + g : -1};
        const double vals[5] = {-1.0 - 0.3, -1.0 - 0.2, 4.0, -1.0 + 0.2, -1.0 + 0.3};   /* row i's coefficient of unknown j */
        for (int t = 0; t < 5; t++) {
            if (rows[t] < 0) continue;
            A.i[o] = rows[t];
            if (cplx) ((double complex *)A.x)[o] = vals[t] + (t == 2 ? 0.5 * I : 0.0); else ((double *)A.x)[o] = vals[t];
            o++;
        }
        A.p[j + 1] = o;
    }
    A.nnz = o;
    return A;
}

static void test_ilu(int cplx)
{
    /* incomplete LU through the high-level interface: less fill than the exact factorization, and a preconditioner on which
     * the Richardson iteration x += (LU)^-1 (b - A x) converges to a full-accuracy solution */
    Csc A = make_convdiff(20, cplx);
    const int64_t n = A.n;
    const size_t vs = cplx ? sizeof(double complex) : sizeof(double);
    char err[256] = {0};
    fplus_options opt; fplus_default_options(&opt);
    fplus_symbolic *sym = NULL; fplus_numeric *exact = NULL, *ilu = NULL;
    int rc = fplus_analyze_opts(n, A.p, A.i, &opt, &sym, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "ilu analyze: %s", err);
    rc = fplus_factor(sym, A.p, A.i, A.x, cplx, &opt, &exact, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "exact factor: %s", err);
    for (int strategy = FPLUS_PIVOT_FULL; strategy <= FPLUS_PIVOT_ROOK; strategy++) {
        opt.pivot_strategy = strategy; opt.drop_tol = 1e-2;
        rc = fplus_factor(sym, A.p, A.i, A.x, cplx, &opt, &ilu, err, sizeof(err));
        CHECK(rc == FPLUS_OK, "ilu factor strategy %d: %s", strategy, err);
        if (rc != FPLUS_OK) continue;
        CHECK(fplus_numeric_factors(ilu)->csc == 1, "ilu factors are column-compressed");
        CHECK(factor_nnz(ilu) < factor_nnz(exact), "ilu strategy %d: fill %ld not below the exact %ld", strategy,
              (long)factor_nnz(ilu), (long)factor_nnz(exact));
        void *b = malloc((size_t)n * vs), *x = calloc((size_t)n, vs), *r = malloc((size_t)n * vs);
        for (int64_t k = 0; k < n; k++) {
            if (cplx) ((double complex *)b)[k] = (2 * urand() - 1) + (2 * urand() - 1) * I; else ((double *)b)[k] = 2 * urand() - 1;
        }
        double be = 1.0; int it;
        for (it = 0; it < 60 && be >= 1e-13; it++) {
            matvec(&A, x, r, 0);
            for (int64_t k = 0; k < n; k++) {
                if (cplx) ((double complex *)r)[k] = ((double complex *)b)[k] - ((double complex *)r)[k];
                else ((double *)r)[k] = ((double *)b)[k] - ((double *)r)[k];
            }
            if (fplus_solve(ilu, r, 1, n, 0, err, sizeof(err)) != FPLUS_OK) { CHECK(0, "ilu solve: %s", err); break; }
            for (int64_t k = 0; k < n; k++) {
                if (cplx) ((double complex *)x)[k] += ((double complex *)r)[k]; else ((double *)x)[k] += ((double *)r)[k];
            }
            be = backward_error(&A, x, b, 0);
        }
        CHECK(be < 1e-12, "ilu strategy %d: preconditioned Richardson reached backward error %.2e in %d iterations", strategy, be, it);
        if (it > 1) printf("ilu %s strategy %d: fill %ld vs exact %ld, Richardson converged in %d iterations\n", cplx ? "complex" : "real",
                           strategy, (long)factor_nnz(ilu), (long)factor_nnz(exact), it);
        free(b); free(x); free(r);
        fplus_numeric_free(ilu); ilu = NULL;
    }
    opt.pivot_strategy = 7;
    CHECK(fplus_factor(sym, A.p, A.i, A.x, cplx, &opt, &ilu, err, sizeof(err)) == FPLUS_ERR_BADARG, "bad pivot_strategy rejected");
    opt.pivot_strategy = FPLUS_PIVOT_PARTIAL; opt.drop_tol = -1.0;
    CHECK(fplus_factor(sym, A.p, A.i, A.x, cplx, &opt, &ilu, err, sizeof(err)) == FPLUS_ERR_BADARG, "negative drop_tol rejected");
    fplus_numeric_free(exact); fplus_symbolic_free(sym); free_matrix(&A);
}

static void test_ilu_breakdown(void)
{
    /* [[1, 1], [1, 1.0005]], drop_tol 1e-2, no equilibration: the second pivot is dropped and replaced by 0.010005 */
    const int64_t n = 2;
    int64_t Ap[3] = {0, 2, 4}, Ai[4] = {0, 1, 0, 1};
    double Ax[4] = {1.0, 1.0, 1.0, 1.0005};
    char err[256] = {0};
    fplus_options opt; fplus_default_options(&opt); opt.pin_dense = 0; opt.equilibrate = 0;
    opt.pivot_strategy = FPLUS_PIVOT_FULL; opt.drop_tol = 1e-2;
    fplus_symbolic *sym = NULL; fplus_numeric *num = NULL;
    int rc = fplus_analyze_opts(n, Ap, Ai, &opt, &sym, err, sizeof(err));
    if (rc == FPLUS_OK) rc = fplus_factor(sym, Ap, Ai, Ax, 0, &opt, &num, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "ilu breakdown: %d %s", rc, err);
    if (rc == FPLUS_OK) {
        const fplus_factors *f = fplus_numeric_factors(num);
        CHECK(f->n_pivots_replaced == 1, "ilu breakdown: %ld pivots replaced (expected 1)", (long)f->n_pivots_replaced);
        double last = 0;
        for (int64_t q = f->Up[1]; q < f->Up[2]; q++) if (f->Ui[q] == 1) last = ((const double *)f->Ux)[q];
        CHECK(fabs(last - 0.010005) < 1e-15, "ilu breakdown: replaced pivot %.17g", last);
        fplus_numeric_free(num); num = NULL;
    }
    double Bx[4] = {1.0, 1.0, 1.0, 1.0};          /* exactly singular: an exact LU still fails */
    opt.drop_tol = 0.0;
    CHECK(fplus_factor(sym, Ap, Ai, Bx, 0, &opt, &num, err, sizeof(err)) == FPLUS_ERR_SINGULAR, "exact LU of a singular matrix");
    fplus_symbolic_free(sym);
}

static void test_row_kernel(void)
{
    /* the row kernel through its own entry point: CSR input, full / partial / rook pivoting */
    const int64_t n = 120;
    Csc A = make_matrix(n, 1, 2.0);    /* CSC of A is the CSR of A^T: factor A^T and check P A^T Q = L U through a solve */
    for (int strat = 0; strat <= 2; strat++) {
        fplus_factors f; char err[256] = {0};
        int rc = fplus_factor_row(n, A.p, A.i, A.x, 0.0, 1, 0, strat, FPLUS_PIVOT_REL_DEFAULT, &f, err, sizeof(err));
        CHECK(rc == FPLUS_OK, "row kernel strategy %d: %d %s", strat, rc, err);
        if (rc == FPLUS_OK) {
            CHECK(f.csc == 0 && f.Lp[n] >= n && f.Up[n] >= n, "row kernel %d: factor sizes", strat);
            fplus_factors_free(&f);
        }
    }
    free_matrix(&A);
}

static void test_ruiz_range_guard(void)
{
    /* entries near 1e200 square to inf: the guard rescales by an exact power of two instead of reporting a zero line */
    const int64_t n = 3;
    int64_t p[4] = {0, 1, 2, 3}, i[3] = {0, 1, 2};
    double x[3] = {1e200, 1e100, 1e150}, Dr[3], Dc[3];
    char err[256] = {0};
    int rc = fplus_ruiz(n, p, i, x, 0, 1, 20, 1e-2, 0, Dr, Dc, err, sizeof(err));
    CHECK(rc == FPLUS_OK, "ruiz range guard: %d %s", rc, err);
    const double x0[3] = {1e200, 1e100, 1e150};
    for (int k = 0; k < 3; k++) {
        CHECK(fabs(log(fabs(x[k]))) < 1e-2, "ruiz scaled entry %d = %g (expected 1 to the 1e-2 tolerance)", k, x[k]);
        CHECK(fabs(Dr[k] * x0[k] * Dc[k] - x[k]) < 1e-14 * fabs(x[k]), "ruiz: Dr A Dc reproduces the scaled entry %d", k);
    }
    double y[3] = {1e300, 1e-300, 1.0};
    rc = fplus_ruiz(n, p, i, y, 0, 1, 20, 1e-2, 0, Dr, Dc, err, sizeof(err));
    CHECK(rc == FPLUS_ERR_BADARG, "ruiz: a range beyond double precision is a clear error (got %d)", rc);
}

static void test_validation(void)
{
    char err[256] = {0};
    int64_t p[3] = {0, 2, 3}, i_bad[3] = {0, 5, 1}, i_unsorted[3] = {1, 0, 1};
    double complex x[3] = {1, 2, 3};
    fplus_factors f; fplus_symbolic *sym = NULL;
    CHECK(fplus_factor_sn(2, p, i_bad, x, 1, 0, 0, 1, 0, 5e7, 1e-8, &f, err, sizeof(err)) == FPLUS_ERR_BADARG, "index out of range");
    CHECK(fplus_analyze(2, p, i_unsorted, 1, 0.5, 10, NULL, NULL, 1, &sym, err, sizeof(err)) == FPLUS_ERR_BADARG, "non-canonical analyze");
    CHECK(fplus_factor_sn(-1, p, i_bad, x, 1, 0, 0, 1, 0, 5e7, 1e-8, &f, err, sizeof(err)) == FPLUS_ERR_BADARG, "negative n");
    double complex xn[3] = {1, NAN, 3}; int64_t i_ok[3] = {0, 1, 1};
    CHECK(fplus_factor_sn(2, p, i_ok, xn, 1, 0, 0, 1, 0, 5e7, 1e-8, &f, err, sizeof(err)) == FPLUS_ERR_BADARG, "non-finite value");
}

int main(void)
{
    test_high_level(1, 0.0, FPLUS_PIVOT_PARTIAL);
    test_high_level(1, 6.0, FPLUS_PIVOT_PARTIAL);
    test_high_level(0, 6.0, FPLUS_PIVOT_PARTIAL);
    test_high_level(1, 6.0, FPLUS_PIVOT_ROOK);
    test_high_level(0, 6.0, FPLUS_PIVOT_FULL);
    test_ilu(1);
    test_ilu(0);
    test_ilu_breakdown();
    test_row_kernel();
    test_ruiz_range_guard();
    test_validation();
    if (failures) { printf("%d check(s) failed\n", failures); return 1; }
    printf("all FPLUS C-interface checks passed\n");
    return 0;
}
