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
    if (fplus_omp_active() && n >= 20000) {
        /* many threads, a large matrix: each column sorted on its own (rows distinct: the order is unique) */
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
    } else {
        /* one thread (or a small matrix): two counting-sort passes, O(nnz) and no allocation per column -- the columns in
           col_perm order with their rows renumbered, transposed (rows ascending by construction), transposed back.  The
           same arrays as the per-column sort (the rows in a column are distinct). */
        int64_t *Tp = (int64_t *)calloc((size_t)(n + 1), sizeof(int64_t));
        int64_t *Tj = (int64_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int64_t));
        int64_t *Ts = (int64_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int64_t));
        int64_t *fill = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
        if (!Tp || !Tj || !Ts || !fill) { free(Tp); free(Tj); free(Ts); free(fill); bad = 1; }
        else {
            for (int64_t q = 0; q < nnz; q++) Tp[inv[Ai[q]] + 1]++;
            for (int64_t i = 0; i < n; i++) Tp[i + 1] += Tp[i];
            memcpy(fill, Tp, (size_t)(n + 1) * sizeof(int64_t));
            for (int64_t jj = 0; jj < n; jj++) {       /* row-major: (new row) -> (new column jj, source position) */
                const int64_t c = col_perm[jj];
                for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) { int64_t t = fill[inv[Ai[q]]]++; Tj[t] = jj; Ts[t] = q; }
            }
            memcpy(fill, Bp, (size_t)(n + 1) * sizeof(int64_t));
            for (int64_t r = 0; r < n; r++)          /* back to columns, rows ascending */
                for (int64_t t = Tp[r]; t < Tp[r + 1]; t++) {
                    int64_t o = fill[Tj[t]]++;
                    Bi[o] = r;
                    memcpy((char *)Bx + (size_t)o * vs, (const char *)Ax + (size_t)Ts[t] * vs, vs);
                }
            free(Tp); free(Tj); free(Ts); free(fill);
        }
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

/* ---- the row-kernel path (rook / full pivoting, or ILU): the same static orders and equilibration, then the CSR kernel;
 * the factors come back in CSC for the triangular solves (the Python layer's _lu_factor_c + SparseLU's tocsc). ---- */

/* the transpose of a compressed n x n pattern with values: CSC <-> CSR of the same matrix, indices ascending */
static int compressed_transpose(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, size_t vs,
                                int64_t **Tp_out, int64_t **Ti_out, void **Tx_out)
{
    const int64_t nnz = Ap[n];
    int64_t *Tp = (int64_t *)calloc((size_t)(n + 1), sizeof(int64_t));
    int64_t *Ti = (int64_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int64_t));
    void *Tx = malloc((size_t)(nnz > 0 ? nnz : 1) * vs);
    int64_t *next = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    if (!Tp || !Ti || !Tx || !next) { free(Tp); free(Ti); free(Tx); free(next); return FPLUS_ERR_NOMEM; }
    for (int64_t q = 0; q < nnz; q++) Tp[Ai[q] + 1]++;
    for (int64_t i = 0; i < n; i++) Tp[i + 1] += Tp[i];
    memcpy(next, Tp, (size_t)n * sizeof(int64_t));
    for (int64_t j = 0; j < n; j++)            /* j ascending: the transposed indices come out sorted */
        for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) {
            int64_t d = next[Ai[q]]++;
            Ti[d] = j;
            memcpy((char *)Tx + (size_t)d * vs, (const char *)Ax + (size_t)q * vs, vs);
        }
    free(next);
    *Tp_out = Tp; *Ti_out = Ti; *Tx_out = Tx;
    return FPLUS_OK;
}

/* row-compressed factors -> column-compressed, in place */
static int factors_to_csc(fplus_factors *f)
{
    if (f->csc) return FPLUS_OK;
    const size_t vs = f->is_complex ? sizeof(double complex) : sizeof(double);
    int64_t *Lp, *Li, *Up, *Ui; void *Lx, *Ux;
    int rc = compressed_transpose(f->n, f->Lp, f->Li, f->Lx, vs, &Lp, &Li, &Lx);
    if (rc != FPLUS_OK) return rc;
    rc = compressed_transpose(f->n, f->Up, f->Ui, f->Ux, vs, &Up, &Ui, &Ux);
    if (rc != FPLUS_OK) { free(Lp); free(Li); free(Lx); return rc; }
    free(f->Lp); free(f->Li); free(f->Lx); free(f->Up); free(f->Ui); free(f->Ux);
    f->Lp = Lp; f->Li = Li; f->Lx = Lx; f->Up = Up; f->Ui = Ui; f->Ux = Ux; f->csc = 1;
    return FPLUS_OK;
}

int fplus_factor_row_perm(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                          const int64_t *row_perm, const int64_t *col_perm,
                          int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                          double drop_tol, int pivot_strategy, double pivot_rel,
                          fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen)
{
    memset(out, 0, sizeof(*out));
    {
        int vrc = fplus_check_compressed(n, Ap, Ai, 1, err, errlen);
        unsigned char *seen = vrc == FPLUS_OK ? (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1) : NULL;
        if (vrc == FPLUS_OK && !seen) { fplus_errf(err, errlen, "out of memory"); return FPLUS_ERR_NOMEM; }
        if (vrc == FPLUS_OK) vrc = fplus_check_perm(n, row_perm, seen, "row_perm", err, errlen);
        if (vrc == FPLUS_OK) { memset(seen, 0, (size_t)(n > 0 ? n : 1)); vrc = fplus_check_perm(n, col_perm, seen, "col_perm", err, errlen); }
        free(seen);
        if (vrc != FPLUS_OK) return vrc;
    }
    const size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    int64_t *Bp = NULL, *Bi = NULL; void *Bx = NULL;
    int rc = permute_csc(n, Ap, Ai, Ax, is_complex, row_perm, col_perm, &Bp, &Bi, &Bx);
    if (rc != FPLUS_OK) { fplus_errf(err, errlen, "out of memory"); return rc; }
    if (equilibrate)
        rc = fplus_ruiz(n, Bp, Bi, Bx, is_complex, 1, ruiz_max_iter, ruiz_tol, verbose, Dr, Dc, err, errlen);
    else
        for (int64_t i = 0; i < n; i++) { Dr[i] = 1.0; Dc[i] = 1.0; }
    int64_t *Rp = NULL, *Ri = NULL; void *Rx = NULL;
    if (rc == FPLUS_OK) {
        rc = compressed_transpose(n, Bp, Bi, Bx, vs, &Rp, &Ri, &Rx);
        if (rc != FPLUS_OK) fplus_errf(err, errlen, "out of memory");
    }
    free(Bp); free(Bi); free(Bx);
    if (rc == FPLUS_OK)
        rc = fplus_factor_row(n, Rp, Ri, Rx, drop_tol, is_complex, verbose, pivot_strategy, pivot_rel, out, err, errlen);
    free(Rp); free(Ri); free(Rx);
    if (rc == FPLUS_OK && (rc = factors_to_csc(out)) != FPLUS_OK) {
        fplus_factors_free(out);
        fplus_errf(err, errlen, "out of memory");
    }
    return rc;
}

int fplus_factor_row_sym(const fplus_symbolic *sym, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex,
                         int equilibrate, int ruiz_max_iter, double ruiz_tol, int verbose,
                         double drop_tol, int pivot_strategy, double pivot_rel,
                         fplus_factors *out, double *Dr, double *Dc, char *err, size_t errlen)
{
    const int64_t n = fplus_symbolic_n(sym);
    if (Ap[n] != fplus_symbolic_nnz(sym)) {
        memset(out, 0, sizeof(*out));
        fplus_errf(err, errlen, "the matrix has %ld nonzeros, the symbolic analysis %ld", (long)Ap[n], (long)fplus_symbolic_nnz(sym));
        return FPLUS_ERR_BADARG;
    }
    return fplus_factor_row_perm(n, Ap, Ai, Ax, is_complex, fplus_symbolic_row_perm(sym), fplus_symbolic_col_perm(sym),
                                 equilibrate, ruiz_max_iter, ruiz_tol, verbose, drop_tol, pivot_strategy, pivot_rel,
                                 out, Dr, Dc, err, errlen);
}
