/* order.c -- orderings: the column elimination tree postorder. */
#include "fplus_internal.h"

/* =========================================================================
 * Column elimination tree (the etree of A^T A, without forming it) and its
 * postorder -- SuperLU's sp_coletree + TreePostorder.  Postordering the
 * fill-reducing column order keeps the same elimination tree (so the same
 * fill bound) and makes the columns of each subtree contiguous, which is
 * what lets left-looking supernodal LU find wide supernodes.
 * Input: CSC (indptr, indices) of the m x n column-permuted matrix.
 * Output: post[k] = the column placed k-th.
 * ========================================================================= */
static int64_t ds_find(int64_t *pp, int64_t i)
{
    int64_t r = i;
    while (pp[r] != r) r = pp[r];
    while (pp[i] != r) { int64_t nx = pp[i]; pp[i] = r; i = nx; }   /* path compression */
    return r;
}

int fplus_coletree_postorder(int64_t m, int64_t n, const int64_t *Ap, const int64_t *Ai, int64_t *post)
{
    int64_t *firstcol = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    int64_t *parent = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t *pp = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *root = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *head = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t *next = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *stack = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    if (!firstcol || !parent || !pp || !root || !head || !next || !stack) {
        free(firstcol); free(parent); free(pp); free(root); free(head); free(next); free(stack);
        return FPLUS_ERR_NOMEM;
    }
    for (int64_t r = 0; r < m; r++) firstcol[r] = n;
    for (int64_t col = 0; col < n; col++)
        for (int64_t q = Ap[col]; q < Ap[col + 1]; q++) { int64_t r = Ai[q]; if (r >= 0 && r < m && col < firstcol[r]) firstcol[r] = col; }
    /* Liu's algorithm on the row-merged graph: each row is represented by its first column */
    for (int64_t col = 0; col < n; col++) {
        pp[col] = col; int64_t cset = col; root[cset] = col; parent[col] = n;
        for (int64_t q = Ap[col]; q < Ap[col + 1]; q++) {
            int64_t r = Ai[q]; if (r < 0 || r >= m) continue;
            int64_t rc = firstcol[r]; if (rc >= col) continue;
            int64_t rset = ds_find(pp, rc), rroot = root[rset];
            if (rroot != col) {
                parent[rroot] = col;
                pp[rset] = cset;            /* link: rset joins cset (cset stays the representative) */
                root[cset] = col;
            }
        }
    }
    /* children lists (in increasing column order), then a non-recursive DFS postorder from the virtual root n */
    for (int64_t v = 0; v <= n; v++) head[v] = -1;
    for (int64_t v = n - 1; v >= 0; v--) { int64_t p = parent[v]; next[v] = head[p]; head[p] = v; }
    int64_t k = 0, top = 0; stack[0] = n;
    while (top >= 0) {
        int64_t v = stack[top], c = head[v];
        if (c >= 0) { head[v] = next[c]; stack[++top] = c; }
        else { top--; if (v < n) post[k++] = v; }
    }
    free(firstcol); free(parent); free(pp); free(root); free(head); free(next); free(stack);
    return k == n ? FPLUS_OK : FPLUS_ERR_INTERNAL;
}

/* =========================================================================
 * fplus_analyze: the static row / column orders (dense-line pinning, COLAMD on the interior pattern, the column-etree
 * postorder) -- the C port of the Python layer's _colamd_permutations_uncached, operation for operation, so the orders
 * are identical to it.  The symbolic object holds the orders only (no permuted pattern: at 3.6e8 nonzeros that would
 * be gigabytes); fplus_factor_sn_sym applies them.
 * ========================================================================= */
#include <colamd.h>

struct fplus_symbolic {
    int64_t n, nnz;
    int64_t *row_perm, *col_perm;      /* A[row_perm, :][:, col_perm] is factored */
    int64_t n_tau_rows, n_tau_cols;
};

void fplus_symbolic_free(fplus_symbolic *sym)
{
    if (!sym) return;
    free(sym->row_perm); free(sym->col_perm); free(sym);
}

int64_t fplus_symbolic_n(const fplus_symbolic *sym) { return sym->n; }
int64_t fplus_symbolic_nnz(const fplus_symbolic *sym) { return sym->nnz; }
const int64_t *fplus_symbolic_row_perm(const fplus_symbolic *sym) { return sym->row_perm; }
const int64_t *fplus_symbolic_col_perm(const fplus_symbolic *sym) { return sym->col_perm; }
int64_t fplus_symbolic_n_dense_rows(const fplus_symbolic *sym) { return sym->n_tau_rows; }
int64_t fplus_symbolic_n_dense_cols(const fplus_symbolic *sym) { return sym->n_tau_cols; }

int fplus_analyze(int64_t n, const int64_t *Ap, const int64_t *Ai, int pin_dense, double tau_density, int64_t tau_min_nnz,
                  const unsigned char *tau_row_mask, const unsigned char *tau_col_mask, int postorder,
                  fplus_symbolic **sym_out, char *err, size_t errlen)
{
    *sym_out = NULL;
    { int vrc = fplus_check_compressed(n, Ap, Ai, 1, err, errlen); if (vrc != FPLUS_OK) return vrc; }
    const int64_t nnz = Ap[n];
    int ret = FPLUS_ERR_NOMEM;
    fplus_symbolic *sym = (fplus_symbolic *)calloc(1, sizeof(fplus_symbolic));
    unsigned char *trow = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    unsigned char *tcol = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    int64_t *cnt = (int64_t *)calloc((size_t)(n > 0 ? n : 1), sizeof(int64_t));
    int64_t *Ain = NULL, *pin = NULL, *Awork = NULL, *pwork = NULL, *icols = NULL, *rmap = NULL, *Bp = NULL, *Bi = NULL, *post = NULL;
    if (!sym || !trow || !tcol || !cnt) goto fail;
    sym->n = n; sym->nnz = nnz;
    sym->row_perm = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    sym->col_perm = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    if (!sym->row_perm || !sym->col_perm) goto fail;

    /* dense rows / columns: more than max(tau_density n, tau_min_nnz) entries (or the given masks) */
    if (pin_dense) {
        const double thr = (double)tau_density * (double)n > (double)tau_min_nnz ? (double)tau_density * (double)n : (double)tau_min_nnz;
        if (tau_row_mask) memcpy(trow, tau_row_mask, (size_t)n);
        else {
            for (int64_t q = 0; q < nnz; q++) cnt[Ai[q]]++;
            for (int64_t i = 0; i < n; i++) trow[i] = (double)cnt[i] > thr;
        }
        if (tau_col_mask) memcpy(tcol, tau_col_mask, (size_t)n);
        else for (int64_t j = 0; j < n; j++) tcol[j] = (double)(Ap[j + 1] - Ap[j]) > thr;
    }
    int64_t ntr = 0, ntc = 0, any_tau;
    for (int64_t i = 0; i < n; i++) { ntr += trow[i] != 0; ntc += tcol[i] != 0; }
    any_tau = ntr > 0 || ntc > 0;
    sym->n_tau_rows = ntr; sym->n_tau_cols = ntc;

    /* row order: the interior rows, then the dense rows, each ascending */
    { int64_t k = 0; for (int64_t i = 0; i < n; i++) if (!trow[i]) sym->row_perm[k++] = i;
      for (int64_t i = 0; i < n; i++) if (trow[i]) sym->row_perm[k++] = i; }

    /* COLAMD on the pattern without the dense lines */
    int64_t snnz = nnz;
    const int64_t *sp = Ap, *si = Ai;
    if (any_tau) {
        pin = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
        Ain = (int64_t *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int64_t));
        if (!pin || !Ain) goto fail;
        int64_t o = 0; pin[0] = 0;
        for (int64_t j = 0; j < n; j++) {
            if (!tcol[j]) for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) if (!trow[Ai[q]]) Ain[o++] = Ai[q];
            pin[j + 1] = o;
        }
        snnz = o; sp = pin; si = Ain;
    }
    {
        size_t Alen = colamd_l_recommended(snnz, n, n);
        if (Alen == 0) { fplus_errf(err, errlen, "COLAMD: colamd_l_recommended failed (nnz %ld)", (long)snnz); ret = FPLUS_ERR_INTERNAL; goto fail; }
        Awork = (int64_t *)fplus_calloc_huge(Alen, sizeof(int64_t));   /* COLAMD's workspace: huge pages (fplus_internal.h) */
        pwork = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
        if (!Awork || !pwork) goto fail;
        memcpy(Awork, si, (size_t)snnz * sizeof(int64_t));
        memcpy(pwork, sp, (size_t)(n + 1) * sizeof(int64_t));
        double knobs[COLAMD_KNOBS]; int64_t stats[COLAMD_STATS];
        colamd_l_set_defaults(knobs);
        if (colamd_l(n, n, (int64_t)Alen, Awork, pwork, knobs, stats) != 1) {
            fplus_errf(err, errlen, "COLAMD failed (status %ld)", (long)stats[COLAMD_STATUS]); ret = FPLUS_ERR_INTERNAL; goto fail;
        }
    }
    int64_t *colamd_perm = pwork;   /* pwork[0..n) */

    /* postorder of the column etree of the interior pattern (interior rows renumbered ascending, columns in COLAMD order) */
    if (postorder) {
        int64_t ni = 0, mi = 0;
        icols = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        rmap = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        if (!icols || !rmap) goto fail;
        for (int64_t k = 0; k < n; k++) if (!tcol[colamd_perm[k]]) icols[ni++] = colamd_perm[k];
        for (int64_t i = 0; i < n; i++) rmap[i] = trow[i] ? -1 : mi++;
        int64_t bnz = 0;
        for (int64_t jj = 0; jj < ni; jj++) { int64_t c = icols[jj]; for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) bnz += rmap[Ai[q]] >= 0; }
        Bp = (int64_t *)malloc((size_t)(ni + 1) * sizeof(int64_t));
        Bi = (int64_t *)malloc((size_t)(bnz > 0 ? bnz : 1) * sizeof(int64_t));
        post = (int64_t *)malloc((size_t)(ni > 0 ? ni : 1) * sizeof(int64_t));
        if (!Bp || !Bi || !post) goto fail;
        int64_t o = 0; Bp[0] = 0;
        for (int64_t jj = 0; jj < ni; jj++) {
            int64_t c = icols[jj];
            for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) if (rmap[Ai[q]] >= 0) Bi[o++] = rmap[Ai[q]];
            Bp[jj + 1] = o;
        }
        int rc = fplus_coletree_postorder(mi, ni, Bp, Bi, post);
        if (rc != FPLUS_OK) { ret = rc; fplus_errf(err, errlen, "column etree postorder failed"); goto fail; }
        /* the postordered interior columns, then the dense columns in COLAMD order */
        int64_t k = 0;
        for (int64_t jj = 0; jj < ni; jj++) sym->col_perm[k++] = icols[post[jj]];
        for (int64_t q = 0; q < n; q++) if (tcol[colamd_perm[q]]) sym->col_perm[k++] = colamd_perm[q];
    } else {
        /* the interior columns in COLAMD order, then the dense columns */
        int64_t k = 0;
        for (int64_t q = 0; q < n; q++) if (!tcol[colamd_perm[q]]) sym->col_perm[k++] = colamd_perm[q];
        for (int64_t q = 0; q < n; q++) if (tcol[colamd_perm[q]]) sym->col_perm[k++] = colamd_perm[q];
    }
    *sym_out = sym; sym = NULL; ret = FPLUS_OK;
fail:
    if (ret == FPLUS_ERR_NOMEM) fplus_errf(err, errlen, "out of memory");
    fplus_symbolic_free(sym);
    free(trow); free(tcol); free(cnt); free(Ain); free(pin); free(Awork); free(pwork); free(icols); free(rmap); free(Bp); free(Bi); free(post);
    return ret;
}
