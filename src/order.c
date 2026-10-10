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
