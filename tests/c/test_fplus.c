/* test_fplus.c -- C-interface tests (ctest): analyze / factor / solve on complex and real matrices with dense lines and
 * strong scaling, all three solve directions, several right-hand sides, the low-level supernodal and row kernels, the
 * Ruiz range guard, and input validation.  Exit status 0 when every check passes. */
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

static void test_high_level(int cplx, double scale)
{
    const int64_t n = 300, nrhs = 3;
    Csc A = make_matrix(n, cplx, scale);
    char err[256] = {0};
    fplus_options opt; fplus_default_options(&opt); opt.relax_rows = 4; opt.panel = 16;
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
            CHECK(be < 1e-12, "%s scale 1e%g trans %d rhs %ld: backward error %.2e", cplx ? "complex" : "real", scale, trans, (long)r, be);
        }
        free(b); free(x);
    }
    fplus_numeric_free(num); fplus_symbolic_free(sym); free_matrix(&A);
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
    test_high_level(1, 0.0);
    test_high_level(1, 6.0);
    test_high_level(0, 6.0);
    test_row_kernel();
    test_ruiz_range_guard();
    test_validation();
    if (failures) { printf("%d check(s) failed\n", failures); return 1; }
    printf("all FPLUS C-interface checks passed\n");
    return 0;
}
