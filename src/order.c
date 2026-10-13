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
#include <amd.h>

/* --- the alternative ordering (minimum degree on A^T A) and the fill count that chooses between the two ---------------- */

/* the column etree of the m x n CSC pattern C (Liu's algorithm, as fplus_coletree_postorder): parent[j], -1 for a root;
 * and its postorder, children in increasing order */
static int coletree_parent_post(int64_t m, int64_t n, const int64_t *Cp, const int64_t *Ci, int64_t *parent, int64_t *post)
{
    int64_t *firstcol = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    int64_t *pp = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *root = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *head = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t *next = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *stack = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int rc = FPLUS_ERR_NOMEM;
    if (!firstcol || !pp || !root || !head || !next || !stack) goto out;
    for (int64_t r = 0; r < m; r++) firstcol[r] = n;
    for (int64_t c = 0; c < n; c++) for (int64_t q = Cp[c]; q < Cp[c + 1]; q++) if (c < firstcol[Ci[q]]) firstcol[Ci[q]] = c;
    for (int64_t c = 0; c < n; c++) {
        pp[c] = c; root[c] = c; parent[c] = -1;
        for (int64_t q = Cp[c]; q < Cp[c + 1]; q++) {
            int64_t rc_ = firstcol[Ci[q]]; if (rc_ >= c) continue;
            int64_t rset = ds_find(pp, rc_), rroot = root[rset];
            if (rroot != c) { parent[rroot] = c; pp[rset] = c; root[c] = c; }
        }
    }
    for (int64_t v = 0; v <= n; v++) head[v] = -1;
    for (int64_t v = n - 1; v >= 0; v--) { int64_t p = parent[v] < 0 ? n : parent[v]; next[v] = head[p]; head[p] = v; }
    { int64_t k = 0, top = 0; stack[0] = n;
      while (top >= 0) { int64_t v = stack[top], c = head[v];
                         if (c >= 0) { head[v] = next[c]; stack[++top] = c; } else { top--; if (v < n) post[k++] = v; } } }
    rc = FPLUS_OK;
out:
    free(firstcol); free(pp); free(root); free(head); free(next); free(stack);
    return rc;
}

/* CSparse's cs_leaf */
static int64_t ata_leaf(int64_t i, int64_t j, const int64_t *first, int64_t *maxfirst, int64_t *prevleaf, int64_t *ancestor, int *jleaf)
{
    *jleaf = 0;
    if (i <= j || first[j] <= maxfirst[i]) return -1;
    maxfirst[i] = first[j];
    int64_t jprev = prevleaf[i]; prevleaf[i] = j;
    *jleaf = jprev == -1 ? 1 : 2;
    if (*jleaf == 1) return i;
    int64_t q = jprev;
    while (q != ancestor[q]) q = ancestor[q];
    for (int64_t s = jprev, sp; s != q; s = sp) { sp = ancestor[s]; ancestor[s] = q; }
    return q;
}

/* nnz of the Cholesky factor of C^T C (C: m x n CSC, columns in the order to evaluate), without forming C^T C: CSparse's
 * cs_counts with ata = 1 (Gilbert, Ng, Peyton).  The bound on the LU fill with partial pivoting (George, Ng) that COLAMD and
 * minimum degree on A^T A both try to make small.  -1 on failure. */
static double chol_ata_nnz(int64_t m, int64_t n, const int64_t *Cp, const int64_t *Ci)
{
    double total = -1;
    int64_t *parent = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), *post = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *Rp = calloc((size_t)(m + 1), sizeof(int64_t)), *Ri = malloc((size_t)(Cp[n] > 0 ? Cp[n] : 1) * sizeof(int64_t));
    int64_t *cnt = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), *anc = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *maxf = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), *prevl = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *first = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), *pinv = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *head = malloc((size_t)(n + 1) * sizeof(int64_t)), *next = malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    int64_t *fill = malloc((size_t)(m + 1) * sizeof(int64_t));
    if (!parent || !post || !Rp || !Ri || !cnt || !anc || !maxf || !prevl || !first || !pinv || !head || !next || !fill) goto out;
    if (coletree_parent_post(m, n, Cp, Ci, parent, post) != FPLUS_OK) goto out;
    /* the rows of C (its transpose, CSC) */
    for (int64_t q = 0; q < Cp[n]; q++) Rp[Ci[q] + 1]++;
    for (int64_t r = 0; r < m; r++) Rp[r + 1] += Rp[r];
    memcpy(fill, Rp, (size_t)(m + 1) * sizeof(int64_t));
    for (int64_t c = 0; c < n; c++) for (int64_t q = Cp[c]; q < Cp[c + 1]; q++) Ri[fill[Ci[q]]++] = c;
    for (int64_t j = 0; j < n; j++) { anc[j] = j; maxf[j] = -1; prevl[j] = -1; first[j] = -1; }
    for (int64_t k = 0; k < n; k++) {
        int64_t j = post[k];
        cnt[j] = first[j] == -1 ? 1 : 0;
        for (; j != -1 && first[j] == -1; j = parent[j]) first[j] = k;
    }
    for (int64_t k = 0; k < n; k++) pinv[post[k]] = k;
    for (int64_t k = 0; k <= n; k++) head[k] = -1;
    for (int64_t r = 0; r < m; r++) {        /* row r in the list of its earliest column in postorder */
        int64_t k = n;
        for (int64_t q = Rp[r]; q < Rp[r + 1]; q++) if (pinv[Ri[q]] < k) k = pinv[Ri[q]];
        next[r] = head[k]; head[k] = r;
    }
    for (int64_t k = 0; k < n; k++) {
        int64_t j = post[k];
        if (parent[j] != -1) cnt[parent[j]]--;
        for (int64_t J = head[k]; J != -1; J = next[J])
            for (int64_t q = Rp[J]; q < Rp[J + 1]; q++) {
                int jleaf; int64_t qq = ata_leaf(Ri[q], j, first, maxf, prevl, anc, &jleaf);
                if (jleaf >= 1) cnt[j]++;
                if (jleaf == 2) cnt[qq]--;
            }
        if (parent[j] != -1) anc[j] = parent[j];
    }
    for (int64_t j = 0; j < n; j++) if (parent[j] != -1) cnt[parent[j]] += cnt[j];
    total = 0; for (int64_t j = 0; j < n; j++) total += (double)cnt[j];
out:
    free(parent); free(post); free(Rp); free(Ri); free(cnt); free(anc); free(maxf); free(prevl); free(first); free(pinv); free(head); free(next); free(fill);
    return total;
}

/* the interior pattern (m_i rows renumbered ascending, the interior columns) with its columns in the order of perm (all n
 * columns; the dense ones skipped): Cp / Ci allocated here */
static int interior_in_order(int64_t n, const int64_t *Ap, const int64_t *Ai, const unsigned char *trow, const unsigned char *tcol,
                             const int64_t *perm, int64_t *mi_out, int64_t *ni_out, int64_t **Cp_out, int64_t **Ci_out)
{
    int64_t *rmap = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), mi = 0, ni = 0, nz = 0;
    if (!rmap) return FPLUS_ERR_NOMEM;
    for (int64_t i = 0; i < n; i++) rmap[i] = trow[i] ? -1 : mi++;
    for (int64_t k = 0; k < n; k++) { int64_t c = perm[k]; if (tcol[c]) continue; ni++; for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) nz += rmap[Ai[q]] >= 0; }
    int64_t *Cp = malloc((size_t)(ni + 1) * sizeof(int64_t)), *Ci = malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int64_t));
    if (!Cp || !Ci) { free(rmap); free(Cp); free(Ci); return FPLUS_ERR_NOMEM; }
    int64_t o = 0, jj = 0; Cp[0] = 0;
    for (int64_t k = 0; k < n; k++) { int64_t c = perm[k]; if (tcol[c]) continue;
        for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) if (rmap[Ai[q]] >= 0) Ci[o++] = rmap[Ai[q]];
        Cp[++jj] = o; }
    free(rmap);
    *mi_out = mi; *ni_out = ni; *Cp_out = Cp; *Ci_out = Ci;
    return FPLUS_OK;
}

/* minimum degree (AMD) on the pattern of C^T C, C the interior pattern in natural column order: the interior columns in
 * AMD order, then the dense columns, in perm (length n).  FPLUS_ERR_INTERNAL when C^T C would be too large to form. */
static int amd_ata_order(int64_t n, const int64_t *Ap, const int64_t *Ai, const unsigned char *trow, const unsigned char *tcol,
                         int64_t *perm, char *err, size_t errlen)
{
    int64_t *nat = malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), mi, ni, *Cp = NULL, *Ci = NULL;
    if (!nat) return FPLUS_ERR_NOMEM;
    for (int64_t k = 0; k < n; k++) nat[k] = k;
    int rc = interior_in_order(n, Ap, Ai, trow, tcol, nat, &mi, &ni, &Cp, &Ci);
    int64_t *icol = nat;                              /* reused: interior column k -> its index in A */
    if (rc != FPLUS_OK) { free(nat); return rc; }
    { int64_t jj = 0; for (int64_t c = 0; c < n; c++) if (!tcol[c]) icol[jj++] = c; }
    int64_t *Rp = calloc((size_t)(mi + 1), sizeof(int64_t)), *Ri = malloc((size_t)(Cp[ni] > 0 ? Cp[ni] : 1) * sizeof(int64_t));
    int64_t *mark = malloc((size_t)(ni > 0 ? ni : 1) * sizeof(int64_t)), *Bp = malloc((size_t)(ni + 1) * sizeof(int64_t)), *Bi = NULL, *P = NULL;
    rc = FPLUS_ERR_NOMEM;
    if (!Rp || !Ri || !mark || !Bp) goto out;
    for (int64_t q = 0; q < Cp[ni]; q++) Rp[Ci[q] + 1]++;
    for (int64_t r = 0; r < mi; r++) Rp[r + 1] += Rp[r];
    { int64_t *f = malloc((size_t)(mi + 1) * sizeof(int64_t)); if (!f) goto out; memcpy(f, Rp, (size_t)(mi + 1) * sizeof(int64_t));
      for (int64_t c = 0; c < ni; c++) for (int64_t q = Cp[c]; q < Cp[c + 1]; q++) Ri[f[Ci[q]]++] = c; free(f); }
    double work = 0; for (int64_t r = 0; r < mi; r++) work += (double)(Rp[r + 1] - Rp[r]) * (double)(Rp[r + 1] - Rp[r]);
    if (work > 64.0 * (double)(Cp[ni] + ni) + 1e6) { rc = FPLUS_ERR_INTERNAL; fplus_errf(err, errlen, "A^T A too dense for the AMD ordering"); goto out; }
    Bi = malloc((size_t)(work > 1 ? work : 1) * sizeof(int64_t));
    P = malloc((size_t)(ni > 0 ? ni : 1) * sizeof(int64_t));
    if (!Bi || !P) goto out;
    for (int64_t c = 0; c < ni; c++) mark[c] = -1;
    int64_t o = 0; Bp[0] = 0;
    for (int64_t c = 0; c < ni; c++) {              /* column c of C^T C: the columns sharing a row with c, without c */
        mark[c] = c;
        for (int64_t q = Cp[c]; q < Cp[c + 1]; q++) { int64_t r = Ci[q];
            for (int64_t t = Rp[r]; t < Rp[r + 1]; t++) { int64_t k = Ri[t]; if (mark[k] != c) { mark[k] = c; Bi[o++] = k; } } }
        Bp[c + 1] = o;
    }
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_l_defaults(Control);
    int st = amd_l_order(ni, Bp, Bi, P, Control, Info);
    if (st != AMD_OK && st != AMD_OK_BUT_JUMBLED) { rc = FPLUS_ERR_INTERNAL; fplus_errf(err, errlen, "AMD failed (status %d)", st); goto out; }
    { int64_t k = 0; for (int64_t t = 0; t < ni; t++) perm[k++] = icol[P[t]];
      for (int64_t c = 0; c < n; c++) if (tcol[c]) perm[k++] = c; }
    rc = FPLUS_OK;
out:
    free(nat); free(Cp); free(Ci); free(Rp); free(Ri); free(mark); free(Bp); free(Bi); free(P);
    return rc;
}


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
    return fplus_analyze_ord(n, Ap, Ai, pin_dense, tau_density, tau_min_nnz, tau_row_mask, tau_col_mask, postorder,
                             FPLUS_ORDER_COLAMD, sym_out, err, errlen);
}

int fplus_analyze_ord(int64_t n, const int64_t *Ap, const int64_t *Ai, int pin_dense, double tau_density, int64_t tau_min_nnz,
                      const unsigned char *tau_row_mask, const unsigned char *tau_col_mask, int postorder, int ordering,
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
    /* FPLUS_ORDER_AMD_ATA: minimum degree on A^T A instead.  FPLUS_ORDER_AUTO: for small sparse matrices (n <= 4096,
       nnz <= 24 n, as the factor's automatic settings) both, minimum degree when its Cholesky factor of A^T A is at least
       5% smaller, COLAMD otherwise.  The bound predicts the LU fill only loosely once pivoting moves rows: at 5% it picked
       minimum degree on all 24 Rayleigh SBDF3 pencils as A^T (bound 0.82-0.95, LU fill 0.86-0.94 of COLAMD's) and kept COLAMD
       where minimum degree filled more (a radial pencil: bound 0.96, LU fill 1.06-1.26).  Computed once per pattern (the
       ordering cache). */
    int64_t *amd_perm = NULL;
    if (ordering == FPLUS_ORDER_AMD_ATA || (ordering == FPLUS_ORDER_AUTO && n <= 4096 && nnz <= 24 * n)) {
        amd_perm = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        if (!amd_perm) goto fail;
        char aerr[128] = {0};
        int arc = amd_ata_order(n, Ap, Ai, trow, tcol, amd_perm, aerr, sizeof(aerr));
        if (arc == FPLUS_ERR_NOMEM) { free(amd_perm); goto fail; }
        if (arc != FPLUS_OK) {
            free(amd_perm); amd_perm = NULL;
            if (ordering == FPLUS_ORDER_AMD_ATA) { fplus_errf(err, errlen, "%s", aerr); ret = arc; goto fail; }
        }
    }
    if (amd_perm && ordering == FPLUS_ORDER_AUTO) {
        int64_t mi, ni, *Cp = NULL, *Ci = NULL; double fc = -1, fa = -1;
        if (interior_in_order(n, Ap, Ai, trow, tcol, colamd_perm, &mi, &ni, &Cp, &Ci) == FPLUS_OK) { fc = chol_ata_nnz(mi, ni, Cp, Ci); free(Cp); free(Ci); }
        if (interior_in_order(n, Ap, Ai, trow, tcol, amd_perm, &mi, &ni, &Cp, &Ci) == FPLUS_OK) { fa = chol_ata_nnz(mi, ni, Cp, Ci); free(Cp); free(Ci); }
        if (!(fc > 0 && fa > 0 && fa < 0.95 * fc)) { free(amd_perm); amd_perm = NULL; }
    }
    if (amd_perm) { memcpy(colamd_perm, amd_perm, (size_t)n * sizeof(int64_t)); free(amd_perm); }

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
