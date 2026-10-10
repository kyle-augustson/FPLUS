/* pipeline.c -- the supernodal path from a symbolic analysis: apply the static orders, equilibrate, factor.
 * The C port of the Python layer's _permute_csc + _lu_factor_c; the factors are bit-identical to that route. */
#include "fplus_internal.h"

typedef struct { int64_t r, src; } RowSrc;

static int rowsrc_cmp(const void *a, const void *b)
{
    int64_t x = ((const RowSrc *)a)->r, y = ((const RowSrc *)b)->r;
    return (x > y) - (x < y);
}

/* B = A[row_perm, :][:, col_perm] in canonical CSC (rows ascending within each column), values gathered */
static int permute_csc(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                       const int64_t *row_perm, const int64_t *col_perm, int64_t **Bp_out, int64_t **Bi_out, void **Bx_out)
{
    const size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    const int64_t nnz = Ap[n];
    int64_t *Bp = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t *Bi = (int64_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int64_t));
    void *Bx = malloc((size_t)(nnz > 0 ? nnz : 1) * vs);
    int64_t *inv = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    if (!Bp || !Bi || !Bx || !inv) { free(Bp); free(Bi); free(Bx); free(inv); return FPLUS_ERR_NOMEM; }
    int identity = 1;
    for (int64_t i = 0; i < n; i++) { inv[row_perm[i]] = i; if (row_perm[i] != i) identity = 0; }
    Bp[0] = 0;
    for (int64_t jj = 0; jj < n; jj++) { int64_t c = col_perm[jj]; Bp[jj + 1] = Bp[jj] + (Ap[c + 1] - Ap[c]); }
    int bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) reduction(|:bad)
#endif
    for (int64_t jj = 0; jj < n; jj++) {
        const int64_t c = col_perm[jj], a0 = Ap[c], len = Ap[c + 1] - a0, b0 = Bp[jj];
        if (identity) {
            memcpy(Bi + b0, Ai + a0, (size_t)len * sizeof(int64_t));
            memcpy((char *)Bx + (size_t)b0 * vs, (const char *)Ax + (size_t)a0 * vs, (size_t)len * vs);
            continue;
        }
        RowSrc *rs = (RowSrc *)malloc((size_t)(len > 0 ? len : 1) * sizeof(RowSrc));
        if (!rs) { bad = 1; continue; }
        for (int64_t k = 0; k < len; k++) { rs[k].r = inv[Ai[a0 + k]]; rs[k].src = a0 + k; }
        qsort(rs, (size_t)len, sizeof(RowSrc), rowsrc_cmp);   /* the row indices are distinct: the order is unique */
        for (int64_t k = 0; k < len; k++) {
            Bi[b0 + k] = rs[k].r;
            memcpy((char *)Bx + (size_t)(b0 + k) * vs, (const char *)Ax + (size_t)rs[k].src * vs, vs);
        }
        free(rs);
    }
    free(inv);
    if (bad) { free(Bp); free(Bi); free(Bx); return FPLUS_ERR_NOMEM; }
    *Bp_out = Bp; *Bi_out = Bi; *Bx_out = Bx;
    return FPLUS_OK;
}

int fplus_factor_sn_perm(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                         const int64_t *row_perm, const int64_t *col_perm,
                         int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                         int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                         fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen)
{
    memset(out, 0, sizeof(*out));
    {   /* canonical input and true permutations: the gather below indexes with both (audit 2026-10-10) */
        int vrc = fplus_check_compressed(n, Ap, Ai, 1, err, errlen);
        unsigned char *seen = vrc == FPLUS_OK ? (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1) : NULL;
        if (vrc == FPLUS_OK && !seen) { fplus_errf(err, errlen, "out of memory"); return FPLUS_ERR_NOMEM; }
        if (vrc == FPLUS_OK) vrc = fplus_check_perm(n, row_perm, seen, "row_perm", err, errlen);
        if (vrc == FPLUS_OK) { memset(seen, 0, (size_t)(n > 0 ? n : 1)); vrc = fplus_check_perm(n, col_perm, seen, "col_perm", err, errlen); }
        free(seen);
        if (vrc != FPLUS_OK) return vrc;
    }
    int64_t *Bp = NULL, *Bi = NULL; void *Bx = NULL;
    int rc = permute_csc(n, Ap, Ai, Ax, is_complex, row_perm, col_perm, &Bp, &Bi, &Bx);
    if (rc != FPLUS_OK) { fplus_errf(err, errlen, "out of memory"); return rc; }
    if (equilibrate)
        rc = fplus_ruiz(n, Bp, Bi, Bx, is_complex, 1, ruiz_max_iter, ruiz_tol, verbose, Dr, Dc, err, errlen);
    else
        for (int64_t i = 0; i < n; i++) { Dr[i] = 1.0; Dc[i] = 1.0; }
    if (rc == FPLUS_OK)
        rc = fplus_factor_sn(n, Bp, Bi, Bx, is_complex, verbose, relax_rows, panel, use_gpu, gpu_min_flops, pivot_rel, out, err, errlen);
    free(Bp); free(Bi); free(Bx);
    return rc;
}

int fplus_factor_sn_sym(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                        int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                        int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                        fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen)
{
    const int64_t n = fplus_symbolic_n(sym);
    if (Ap[n] != fplus_symbolic_nnz(sym)) {
        memset(out, 0, sizeof(*out));
        fplus_errf(err, errlen, "the matrix has %ld nonzeros, the symbolic analysis %ld", (long)Ap[n], (long)fplus_symbolic_nnz(sym));
        return FPLUS_ERR_BADARG;
    }
    return fplus_factor_sn_perm(n, Ap, Ai, Ax, is_complex, fplus_symbolic_row_perm(sym), fplus_symbolic_col_perm(sym),
                                equilibrate, ruiz_max_iter, ruiz_tol, verbose, relax_rows, panel, use_gpu, gpu_min_flops,
                                pivot_rel, out, Dr, Dc, err, errlen);
}
