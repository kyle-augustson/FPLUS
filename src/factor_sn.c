/* factor_sn.c -- supernodal left-looking sparse LU with partial pivoting (CSC in, CSC factors out); relaxed
 * supernodes, panels, OpenMP, the optional cuBLAS update path. */
#include "fplus_internal.h"

/* growable (index, value) arrays of the U factor */
static int gp_grow(int64_t **idx, void **val, int64_t *cap, int64_t need, int is_complex)
{
    if (need <= *cap) return 0;
    int64_t ncap = *cap * 2 > need ? *cap * 2 : need;
    int64_t *ni = (int64_t *)realloc(*idx, (size_t)ncap * sizeof(int64_t));
    if (!ni) return -1;
    *idx = ni;
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    void *nv = realloc(*val, (size_t)ncap * vs);
    if (!nv) return -1;
    *val = nv;
    *cap = ncap;
    return 0;
}

/* =========================================================================
 * Supernodal left-looking LU with partial pivoting.
 *
 * Columns whose L patterns nest (column c's non-pivotal rows are column
 * c-1's minus the pivot row of c) form a supernode, stored as one dense
 * column-major block over the union of its rows, the pivot rows of the
 * supernode's columns first in column order, the remaining rows after.
 * Column j is computed by the sparse triangular solve x = L^-1 A[:, j]:
 * a depth-first search over the supernodes gives the ones reached, in
 * topological order, each with the first of its columns reached (the
 * columns from there to its last form the U segment); the numeric sweep
 * scatters A[:, j] into a dense accumulator and applies each reached
 * supernode as a dense unit-lower solve on its segment followed by a
 * dense rectangle update of the rows below, both contiguous in memory.
 * The pivot is the entry of largest modulus among the rows not yet
 * pivotal; the column then joins the open supernode when its pattern
 * nests, or opens a new one.  The search reads pruned row lists
 * (Eisenstat-Liu): once the pivot row of column j lies in the rows of a
 * supernode whose segment column j reached, the rows of that supernode not
 * yet pivotal are reachable through column j, so its list is partitioned
 * and the search reads the pivotal part from then on.  Rows carry their
 * original index during the factorisation (pinv maps a row to its pivot
 * step) and are renumbered to pivot order on output, which is CSC with
 * A[p, :] = L U and q the identity.  L keeps its full pattern, zeros
 * included: the pruning argument needs it.  The singularity test is a
 * pivot below sqrt(eps) times the largest modulus of the column's own
 * entries in A.
 * ========================================================================= */

typedef struct { int64_t row, rr; } RowOrd;     /* output row index, block row */

static int roword_cmp(const void *a, const void *b)
{
    int64_t x = ((const RowOrd *)a)->row, y = ((const RowOrd *)b)->row;
    return (x > y) - (x < y);
}

typedef struct {
    int64_t first;      /* first column */
    int64_t ncols;      /* columns stored so far */
    int64_t nrows;      /* rows of the block: ncols pivot rows, then the rest */
    int64_t ld;         /* leading dimension of blk and capacity of rows (>= nrows; relaxed joins append rows in place) */
    int64_t cap;        /* column capacity of the block */
    int64_t *rows;      /* original row index per block row */
    void *blk;          /* dense column-major nrows x ncols */
    int64_t *dfs_rows;  /* the rest rows, partitioned by pruning (closed supernodes) */
    int64_t dfs_end;    /* end of the pivotal part of dfs_rows, -1 when unpruned */
    double zeros;       /* explicit zeros stored by relaxed joins */
} Supernode;

typedef struct {
    int64_t n;
    int is_complex;
    Supernode *sn;      /* one slot per column at most */
    int64_t nsn;        /* supernodes opened */
    int64_t *col_sn;    /* column -> supernode */
    int64_t *pinv, *p;
    int64_t *rowpos;    /* block row position of a row in the open supernode, else -1 */
    int64_t *wrow;      /* row visit stamp */
    int64_t *wsn;       /* supernode visit stamp */
    int64_t *segfirst;  /* first column reached per supernode, this column */
    int64_t *xi;        /* reached supernodes in topological order at xi[top..n) */
    int64_t *nonpiv;    /* non-pivotal rows of the pattern */
    int64_t *miss;      /* rows of column j missing from the open supernode (relaxed join) */
    int64_t relax_rows; /* relaxed join: at most this many rows of pattern mismatch (0 = exact supernodes) */
    int relax_auto;     /* relax_rows < 0: joins by the share of explicit zeros (sn_relax_ok) */
    int64_t rz_w[3];    /* auto rule: width bounds */
    double rz_z[3];     /* auto rule: zero shares allowed up to each width bound */
    int64_t rz_cap;     /* auto rule: at most this many rows of mismatch per join */
    double pivot_rel;   /* singular when the best pivot < pivot_rel * the column's max (default sqrt(DBL_EPSILON)) */
    char *err; size_t errlen;   /* error message buffer (the kernels' failure exits) */
    int64_t *pstack, *head;
    int64_t *Up, *Ui;  void *Ux;  int64_t nnzU, capU;
    void *x, *y;        /* accumulator and the dense work vector */
} SnWork;

static inline int64_t sn_rest_end(const SnWork *W, int64_t s, int64_t open)
{
    const Supernode *S = &W->sn[s];
    if (s == open) return S->nrows - S->ncols; /* the open block: no pruning */
    return S->dfs_end >= 0 ? S->dfs_end : S->nrows - S->ncols;
}

static inline const int64_t *sn_rest_rows(const SnWork *W, int64_t s, int64_t open)
{
    const Supernode *S = &W->sn[s];
    return s == open ? S->rows + S->ncols : S->dfs_rows;
}

/* Depth-first search from supernode `root`, whose column `col` was
 * reached; finished supernodes are pushed on xi[--top].  Non-pivotal rows
 * met on the way are stamped and listed in nonpiv.  Returns the new top. */
static int64_t sn_dfs(int64_t root, int64_t col, int64_t top, SnWork *W, int64_t stamp,
                      int64_t open, int64_t *n_nonpiv)
{
    int64_t *pstack = W->pstack, *head = W->head, *wsn = W->wsn, *wrow = W->wrow;
    int64_t *segfirst = W->segfirst, *pinv = W->pinv, *col_sn = W->col_sn;
    int64_t sp = 0;
    pstack[0] = root;
    wsn[root] = stamp;
    segfirst[root] = col;
    head[root] = 0;
    while (sp >= 0) {
        int64_t s = pstack[sp];
        const int64_t *rows = sn_rest_rows(W, s, open);
        int64_t end = sn_rest_end(W, s, open);
        int done = 1;
        for (int64_t q = head[s]; q < end; q++) {
            int64_t i = rows[q];
            int64_t k = pinv[i];
            if (k < 0) {
                if (wrow[i] != stamp) { wrow[i] = stamp; W->nonpiv[(*n_nonpiv)++] = i; }
                continue;
            }
            int64_t t = col_sn[k];
            if (wsn[t] == stamp) {
                if (k < segfirst[t]) segfirst[t] = k;
                continue;
            }
            wsn[t] = stamp;
            segfirst[t] = k;
            head[s] = q + 1;
            head[t] = 0;
            pstack[++sp] = t;
            done = 0;
            break;
        }
        if (done) {
            sp--;
            W->xi[--top] = s;
        }
    }
    return top;
}

/* Relaxed join of column j into the open supernode S, whose pattern differs from the column's by n_miss rows missing from S and
 * `extra` rows of S absent from the column.  relax_rows >= 0: the fixed rule, mismatch <= relax_rows.  relax_rows < 0 (auto): the
 * amalgamation rule of CHOLMOD's supernodal analysis applied on the fly -- join while the share of explicit zeros in the grown block
 * stays below a bound that tightens with the width: any share up to width rz_w[0], rz_z[0] up to rz_w[1], rz_z[1] up to rz_w[2],
 * rz_z[2] beyond; and never more than rz_cap rows of mismatch in one join.  Narrow supernodes of very sparse matrices then stay
 * nearly exact (few zeros, no wasted flops), while the wide supernodes of the dense-row meridional pencils still absorb the
 * near-nested columns that make the BLAS updates efficient. */
static inline int sn_relax_ok(const SnWork *W, const Supernode *S, int64_t n_miss, int64_t extra)
{
    const int64_t mm = n_miss + extra;
    if (!W->relax_auto) return mm <= W->relax_rows;
    if (mm > W->rz_cap) return 0;
    const int64_t w = S->ncols + 1, r = S->nrows + n_miss;
    if (w <= W->rz_w[0]) return 1;
    const double z = S->zeros + (double)n_miss * (double)S->ncols + (double)extra, tot = (double)w * (double)r;
    const double zmax = w <= W->rz_w[1] ? W->rz_z[0] : (w <= W->rz_w[2] ? W->rz_z[1] : W->rz_z[2]);
    return z <= zmax * tot;
}

/* append k rows to the open supernode: zeros in every stored column (a relaxed join) */
static int sn_add_rows(Supernode *S, const int64_t *rows_new, int64_t k, int64_t *rowpos, int is_complex)
{
    const size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    const int64_t r1 = S->nrows, nr = r1 + k;
    if (nr > S->ld) {
        /* grow the leading dimension geometrically: amortized O(1) copies per added row */
        const int64_t nld = nr + nr / 4 + 4;
        int64_t *nrw = (int64_t *)realloc(S->rows, (size_t)nld * sizeof(int64_t));
        if (!nrw) return -1;
        S->rows = nrw;
        char *nb = (char *)malloc((size_t)S->cap * (size_t)nld * vs);
        if (!nb) return -1;
        const char *ob = (const char *)S->blk;
        for (int64_t c = 0; c < S->ncols; c++)
            memcpy(nb + (size_t)c * nld * vs, ob + (size_t)c * S->ld * vs, (size_t)r1 * vs);
        free(S->blk); S->blk = nb; S->ld = nld;
    }
    char *b = (char *)S->blk;
    for (int64_t c = 0; c < S->ncols; c++) memset(b + ((size_t)c * S->ld + r1) * vs, 0, (size_t)k * vs);
    for (int64_t i = 0; i < k; i++) { S->rows[r1 + i] = rows_new[i]; rowpos[rows_new[i]] = r1 + i; }
    S->nrows = nr;
    return 0;
}

static int sn_grow_cols(Supernode *S, int64_t need, int is_complex)
{
    if (need <= S->cap) return 0;
    int64_t ncap = S->cap * 2 > need ? S->cap * 2 : need;
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    void *nb = realloc(S->blk, (size_t)ncap * (size_t)S->ld * vs);
    if (!nb) return -1;
    S->blk = nb;
    S->cap = ncap;
    return 0;
}

#ifdef HAVE_BLAS
#  define SN_BLAS(m, nrest) ((m) * (nrest) >= SN_BLAS_MIN)
#else
#  define SN_BLAS(m, nrest) 0
#endif

/* ---- profiling (compile with -DFPLUS_PROFILE; no effect otherwise) ---- */
#ifdef FPLUS_PROFILE
#define NPH 8
static const char *PH_NAME[NPH] = {"symbolic DFS", "scatter A[:,j]", "updates (BLAS path)", "updates (loop path)", "pivot search",
                                   "U store", "L store / supernode", "clear + prune"};
static struct { double t[NPH]; double fl_blas, fl_loop, by_blas, by_loop, fl_w[24], fl_reuse; int64_t n_upd, n_upd_blas, mm[24], mm_rows[24];
                int64_t prev_sn_stamp_dummy; int64_t n_relax; double zeros; } SNP;
static int64_t *SNP_last;   /* last column each supernode updated */
static inline int ilog2_i64(int64_t v) { int b = 0; while (v > 1 && b < 23) { v >>= 1; b++; } return b; }
#define PROF_DECL double _tl = wall_time(); double _tu;
#define PROF_MARK(k) do { double _t = wall_time(); SNP.t[k] += _t - _tl; _tl = _t; } while (0)
#define PROF_UPD_BEGIN() do { _tu = wall_time(); } while (0)
#define PROF_UPD_END(blas, m, nrest, w, s, j) do { double _t = wall_time(); double _fl = 8.0 * ((double)(m) * ((m) - 1) / 2 + (double)(m) * (nrest)); \
        double _by = 16.0 * ((double)(m) * ((m) + 1) / 2 + (double)(m) * (nrest));                                                          \
        if (blas) { SNP.t[2] += _t - _tu; SNP.fl_blas += _fl; SNP.by_blas += _by; SNP.n_upd_blas++; }                                     \
        else { SNP.t[3] += _t - _tu; SNP.fl_loop += _fl; SNP.by_loop += _by; }                                                            \
        SNP.n_upd++; SNP.fl_w[ilog2_i64(w)] += _fl; if (SNP_last[s] == (j) - 1) SNP.fl_reuse += _fl; SNP_last[s] = (j); _tl = _t; } while (0)
#define PROF_MISMATCH(missing, extra) do { int64_t _d = (missing) + (extra); SNP.mm[ilog2_i64(_d + 1)]++; } while (0)
#define PROF_RELAX(nz_) do { SNP.n_relax++; SNP.zeros += (double)(nz_); } while (0)
#else
#define PROF_RELAX(nz_) do {} while (0)
#define PROF_DECL
#define PROF_MARK(k) do {} while (0)
#define PROF_UPD_BEGIN() do {} while (0)
#define PROF_UPD_END(blas, m, nrest, w, s, j) do {} while (0)
#define PROF_MISMATCH(missing, extra) do {} while (0)
#endif

#define SN_FACTOR_BODY(NAME, T, ABSF, TRSV, GEMV)                                       \
static int NAME(SnWork *W, const int64_t *Ap, const int64_t *Ai, const T *Ax, int verbose) \
{                                                                                       \
    const int64_t n = W->n;                                                             \
    const T ONE = 1, ZERO = 0; (void)ONE; (void)ZERO;                                   \
    const int is_complex = W->is_complex;                                               \
    int64_t *pinv = W->pinv, *p = W->p, *wrow = W->wrow, *xi = W->xi;                   \
    int64_t *rowpos = W->rowpos, *nonpiv = W->nonpiv, *col_sn = W->col_sn;              \
    T *x = (T *)W->x, *y = (T *)W->y;                                                   \
    double pivot_threshold;                                                             \
    int64_t open = -1;                                                                  \
    for (int64_t j = 0; j < n; j++) {                                                   \
        PROF_DECL                                                                       \
        const int64_t stamp = j + 1;                                                    \
        int64_t top = n, n_nonpiv = 0;                                                  \
        /* symbolic: the reached supernodes in topological order */                    \
        for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) {                                   \
            int64_t i = Ai[q], k = pinv[i];                                             \
            if (k < 0) {                                                                \
                if (wrow[i] != stamp) { wrow[i] = stamp; nonpiv[n_nonpiv++] = i; }      \
                continue;                                                               \
            }                                                                           \
            int64_t s = col_sn[k];                                                      \
            if (W->wsn[s] == stamp) { if (k < W->segfirst[s]) W->segfirst[s] = k; continue; } \
            top = sn_dfs(s, k, top, W, stamp, open, &n_nonpiv);                         \
        }                                                                               \
        PROF_MARK(0);                                                                   \
        /* numeric: scatter A[:, j] (noting its largest modulus), apply the reached */ \
        /* supernodes */                                                                \
        double colmax = 0.0;                                                            \
        for (int64_t q = Ap[j]; q < Ap[j + 1]; q++) {                                   \
            x[Ai[q]] += Ax[q];                                                          \
            double a = ABSF(Ax[q]);                                                     \
            if (a > colmax) colmax = a;                                                 \
        }                                                                               \
        PROF_MARK(1);                                                                   \
        for (int64_t t = top; t < n; t++) {                                             \
            PROF_UPD_BEGIN();                                                           \
            Supernode *S = &W->sn[xi[t]];                                               \
            const int64_t r1 = S->nrows, ld = S->ld, w = S->ncols, kf = W->segfirst[xi[t]] - S->first; \
            const T *B = (const T *)S->blk;                                             \
            const int64_t *rows = S->rows;                                              \
            int64_t nrest = r1 - w, m = w - kf;                                         \
            const int64_t *rest = rows + w;                                             \
            if (SN_BLAS(m, nrest)) {                                                    \
                /* x_seg = L_seg^-1 x_seg, then y = B_rest x_seg, with BLAS */          \
                T *xs = y + nrest;                                                      \
                for (int64_t cc = kf; cc < w; cc++) xs[cc - kf] = x[rows[cc]];          \
                TRSV(102, 122, 111, 132, (int)m, B + kf * ld + kf, (int)ld, xs, 1);     \
                GEMV(102, 111, (int)nrest, (int)m, &ONE, B + kf * ld + w, (int)ld, xs, 1, &ZERO, y, 1); \
                for (int64_t cc = kf; cc < w; cc++) x[rows[cc]] = xs[cc - kf];          \
            } else {                                                                    \
                for (int64_t r = 0; r < nrest; r++) y[r] = 0;                           \
                for (int64_t cc = kf; cc < w; cc++) {                                   \
                    T xc = x[rows[cc]];                                                 \
                    if (xc == 0) continue;                                              \
                    const T *Bc = B + cc * ld;                                          \
                    for (int64_t rr = cc + 1; rr < w; rr++) x[rows[rr]] -= xc * Bc[rr]; \
                    const T *Br = Bc + w;                                               \
                    for (int64_t r = 0; r < nrest; r++) y[r] += Br[r] * xc;             \
                }                                                                       \
            }                                                                           \
            for (int64_t r = 0; r < nrest; r++) x[rest[r]] -= y[r];                     \
            PROF_UPD_END(SN_BLAS(m, nrest), m, nrest, w, xi[t], j);                     \
        }                                                                               \
        /* pivot: the largest modulus among the rows not yet pivotal */                 \
        int64_t pivrow = -1;                                                            \
        double best = -1.0;                                                             \
        for (int64_t s = 0; s < n_nonpiv; s++) {                                        \
            double a = ABSF(x[nonpiv[s]]);                                              \
            if (a > best) { best = a; pivrow = nonpiv[s]; }                             \
        }                                                                               \
        pivot_threshold = W->pivot_rel * colmax;                                     \
        if (pivrow < 0 || best <= 0.0 || best < pivot_threshold) {                      \
            char _msg[256];                                                             \
            snprintf(_msg, sizeof(_msg),                                                \
                "Near-zero or zero pivot %.6e at step %ld (threshold %.6e). "           \
                "Matrix is singular.", best > 0 ? best : 0.0, (long)j, pivot_threshold); \
            fplus_errf(W->err, W->errlen, "%s", _msg);                                  \
            return -2;                                                                  \
        }                                                                               \
        T piv = x[pivrow];                                                              \
        PROF_MARK(4);                                                                   \
        /* U[:, j]: the segments of the reached supernodes, then the diagonal */       \
        {                                                                               \
            int64_t need = 1;                                                           \
            for (int64_t t = top; t < n; t++)                                           \
                need += W->sn[xi[t]].first + W->sn[xi[t]].ncols - W->segfirst[xi[t]];   \
            if (gp_grow(&W->Ui, &W->Ux, &W->capU, W->nnzU + need, is_complex)) return -1; \
            int64_t *Ui = W->Ui; T *Ux = (T *)W->Ux;                                    \
            int64_t u = W->nnzU;                                                        \
            for (int64_t t = top; t < n; t++) {                                         \
                Supernode *S = &W->sn[xi[t]];                                           \
                for (int64_t cc = W->segfirst[xi[t]] - S->first; cc < S->ncols; cc++) { \
                    Ui[u] = S->first + cc; Ux[u] = x[S->rows[cc]]; u++;                 \
                }                                                                       \
            }                                                                           \
            Ui[u] = j; Ux[u] = piv; u++;                                                \
            W->nnzU = u; W->Up[j + 1] = u;                                              \
        }                                                                               \
        PROF_MARK(5);                                                                   \
        /* L[:, j]: join the open supernode when the pattern nests, else open one */    \
        int join = 0;                                                                   \
        int64_t n_miss = 0;                                                             \
        if (open >= 0) {                                                                \
            Supernode *S = &W->sn[open];                                                \
            int64_t in_S = 0;                                                           \
            for (int64_t s = 0; s < n_nonpiv; s++) {                                    \
                if (rowpos[nonpiv[s]] >= S->ncols) in_S++;                              \
                else W->miss[n_miss++] = nonpiv[s];                                     \
            }                                                                           \
            int64_t extra = (S->nrows - S->ncols) - in_S;                               \
            if (n_miss + extra == 0) join = 1;                                          \
            else if (sn_relax_ok(W, S, n_miss, extra)) join = 2;                        \
            if (join == 2) {                                                            \
                S->zeros += (double)n_miss * (double)S->ncols + (double)extra;          \
                if (n_miss > 0 && sn_add_rows(S, W->miss, n_miss, rowpos, is_complex)) return -1; \
                PROF_RELAX(n_miss * S->ncols + extra);                                  \
            }                                                                           \
        }                                                                               \
        if (join) {                                                                     \
            Supernode *S = &W->sn[open];                                                \
            const int64_t r1 = S->nrows, ld = S->ld, w = S->ncols;                      \
            if (sn_grow_cols(S, w + 1, is_complex)) return -1;                          \
            T *B = (T *)S->blk;                                                         \
            /* the pivot row moves to block row w */                                    \
            int64_t pp = rowpos[pivrow];                                                \
            if (pp != w) {                                                              \
                int64_t other = S->rows[w];                                             \
                S->rows[w] = pivrow; S->rows[pp] = other;                               \
                rowpos[pivrow] = w; rowpos[other] = pp;                                 \
                for (int64_t cc = 0; cc < w; cc++) {                                    \
                    T tv = B[cc * ld + w]; B[cc * ld + w] = B[cc * ld + pp]; B[cc * ld + pp] = tv; \
                }                                                                       \
            }                                                                           \
            T *Bc = B + w * ld;                                                         \
            for (int64_t rr = 0; rr < w; rr++) Bc[rr] = x[S->rows[rr]];                 \
            Bc[w] = 1;                                                                  \
            for (int64_t rr = w + 1; rr < r1; rr++) Bc[rr] = x[S->rows[rr]] / piv;      \
            S->ncols = w + 1;                                                           \
            col_sn[j] = open;                                                           \
        } else {                                                                        \
            if (open >= 0) {                                                            \
                /* close the open supernode: its rest rows become the search list */    \
                Supernode *S = &W->sn[open];                                            \
                { int64_t _in = 0; for (int64_t s_ = 0; s_ < n_nonpiv; s_++) if (rowpos[nonpiv[s_]] >= S->ncols) _in++; \
                  PROF_MISMATCH(n_nonpiv - _in, (S->nrows - S->ncols) - _in); (void)_in; }  \
                int64_t nrest = S->nrows - S->ncols;                                    \
                S->dfs_rows = (int64_t *)malloc((size_t)(nrest > 0 ? nrest : 1) * sizeof(int64_t)); \
                if (!S->dfs_rows) return -1;                                            \
                memcpy(S->dfs_rows, S->rows + S->ncols, (size_t)nrest * sizeof(int64_t)); \
                for (int64_t rr = 0; rr < S->nrows; rr++) rowpos[S->rows[rr]] = -1;     \
            }                                                                           \
            int64_t s = W->nsn++;                                                       \
            Supernode *S = &W->sn[s];                                                   \
            S->first = j; S->ncols = 1; S->nrows = n_nonpiv; S->ld = n_nonpiv; S->cap = 4; S->zeros = 0; \
            S->dfs_rows = NULL; S->dfs_end = -1;                                        \
            S->rows = (int64_t *)malloc((size_t)S->nrows * sizeof(int64_t));            \
            S->blk = malloc((size_t)S->cap * (size_t)S->ld * (is_complex ? sizeof(double complex) : sizeof(double))); \
            if (!S->rows || !S->blk) return -1;                                         \
            T *B = (T *)S->blk;                                                         \
            S->rows[0] = pivrow; rowpos[pivrow] = 0; B[0] = 1;                          \
            int64_t rr = 1;                                                             \
            for (int64_t t = 0; t < n_nonpiv; t++) {                                    \
                int64_t i = nonpiv[t];                                                  \
                if (i == pivrow) continue;                                              \
                S->rows[rr] = i; rowpos[i] = rr; B[rr] = x[i] / piv; rr++;              \
            }                                                                           \
            col_sn[j] = s;                                                              \
            open = s;                                                                   \
        }                                                                               \
        pinv[pivrow] = j; p[j] = pivrow;                                                \
        PROF_MARK(6);                                                                   \
        /* clear the accumulator on the pattern */                                      \
        for (int64_t t = 0; t < n_nonpiv; t++) x[nonpiv[t]] = 0;                        \
        for (int64_t t = top; t < n; t++) {                                             \
            Supernode *S = &W->sn[xi[t]];                                               \
            for (int64_t cc = W->segfirst[xi[t]] - S->first; cc < S->ncols; cc++) x[S->rows[cc]] = 0; \
        }                                                                               \
        x[pivrow] = 0;                                                                  \
        /* prune the closed supernodes whose segments this column reached */            \
        for (int64_t t = top; t < n; t++) {                                             \
            int64_t s = xi[t];                                                          \
            Supernode *S = &W->sn[s];                                                   \
            if (s == open || S->dfs_end >= 0) continue;                                 \
            int64_t nrest = S->nrows - S->ncols;                                        \
            int64_t *rows = S->dfs_rows;                                                \
            int found = 0;                                                              \
            for (int64_t r = 0; r < nrest; r++) if (rows[r] == pivrow) { found = 1; break; } \
            if (!found) continue;                                                       \
            int64_t ph = 0, pt = nrest;                                                 \
            while (ph < pt) {                                                           \
                if (pinv[rows[ph]] >= 0) { ph++; continue; }                            \
                pt--;                                                                   \
                int64_t ti = rows[ph]; rows[ph] = rows[pt]; rows[pt] = ti;              \
            }                                                                           \
            S->dfs_end = pt;                                                            \
        }                                                                               \
        PROF_MARK(7);                                                                   \
        if (verbose >= 2 && (j % (n / 20 + 1) == 0 || j == n - 1))                      \
            printf("[FPLUS]   [%5.1f%%] column %ld/%ld  pivot=%.3e  supernodes=%ld\n",  \
                   100.0 * (double)(j + 1) / (double)n, (long)(j + 1), (long)n, best,   \
                   (long)W->nsn);                                                       \
    }                                                                                   \
    return 0;                                                                           \
}

#ifdef HAVE_BLAS
/* dgemv takes alpha and beta by value, zgemv by pointer; one calling form */
static inline void sn_dgemv(int order, int trans, int M, int N, const double *alpha,
                            const double *A, int lda, const double *X, int incX,
                            const double *beta, double *Y, int incY)
{
    cblas_dgemv(order, trans, M, N, *alpha, A, lda, X, incX, *beta, Y, incY);
}
SN_FACTOR_BODY(sn_factor_z, double complex, cabs, cblas_ztrsv, cblas_zgemv)
SN_FACTOR_BODY(sn_factor_d, double, fabs, cblas_dtrsv, sn_dgemv)
#else
SN_FACTOR_BODY(sn_factor_z, double complex, cabs, (void), (void))
SN_FACTOR_BODY(sn_factor_d, double, fabs, (void), (void))
#endif



#ifdef HAVE_BLAS
/* -------------------------------------------------------------------------
 * Optional GPU path for the panel updates (cuBLAS loaded at run time with
 * dlopen; nothing CUDA at build time).  Only on devices that read pageable
 * host memory directly (cudaDevAttrPageableMemoryAccess, e.g. coherent
 * CPU-GPU memory over NVLink-C2C): the zgemm of a panel update runs on the GPU on the host-
 * resident supernode block and work buffers, the small ztrsm stays on the
 * CPU.  Library names: FPLUS_CUBLAS_LIB / FPLUS_CUDART_LIB, else
 * libcublas.so.13 / libcudart.so.13, then .so.12.
 * ------------------------------------------------------------------------- */
#include <dlfcn.h>
typedef int (*cublasCreate_t)(void **);
typedef int (*cublasZgemm_t)(void *, int, int, int, int, int, const void *, const void *, int, const void *, int,
                             const void *, void *, int);
typedef int (*cudaSync_t)(void);
typedef int (*cudaAttr_t)(int *, int, int);
typedef struct {
    int ok; void *handle; cublasZgemm_t zgemm; cudaSync_t sync; int (*destroy)(void *);
    double min_flops, t, flops; int64_t calls;
} GpuCtx;

static void *dlopen_first(const char *env, const char *a, const char *b)
{
    const char *e = getenv(env); void *h = NULL;
    if (e) h = dlopen(e, RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen(a, RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen(b, RTLD_NOW | RTLD_GLOBAL);
    return h;
}

static void gpu_init(GpuCtx *G, double min_flops, int verbose)
{
    memset(G, 0, sizeof(*G)); G->min_flops = min_flops;
    void *rt = dlopen_first("FPLUS_CUDART_LIB", "libcudart.so.13", "libcudart.so.12");
    void *cb = dlopen_first("FPLUS_CUBLAS_LIB", "libcublas.so.13", "libcublas.so.12");
    if (!rt || !cb) { if (verbose >= 1) printf("[FPLUS] gpu: CUDA runtime / cuBLAS not found (%s); CPU only\n", dlerror()); return; }
    cudaAttr_t attr = (cudaAttr_t)dlsym(rt, "cudaDeviceGetAttribute");
    G->sync = (cudaSync_t)dlsym(rt, "cudaDeviceSynchronize");
    cublasCreate_t create = (cublasCreate_t)dlsym(cb, "cublasCreate_v2");
    G->zgemm = (cublasZgemm_t)dlsym(cb, "cublasZgemm_v2");
    G->destroy = (int (*)(void *))dlsym(cb, "cublasDestroy_v2");
    if (!attr || !G->sync || !create || !G->zgemm) { if (verbose >= 1) printf("[FPLUS] gpu: symbols missing; CPU only\n"); return; }
    int pageable = 0;
    if (attr(&pageable, 88 /* cudaDevAttrPageableMemoryAccess */, 0) != 0 || !pageable) {
        if (verbose >= 1) printf("[FPLUS] gpu: device 0 cannot read pageable host memory; CPU only\n");
        return;
    }
    if (create(&G->handle) != 0) { if (verbose >= 1) printf("[FPLUS] gpu: cublasCreate failed; CPU only\n"); return; }
    G->ok = 1;
    if (verbose >= 1) printf("[FPLUS] gpu: cuBLAS zgemm for panel updates above %.2e flops (pageable host memory)\n", min_flops);
}
#endif

#ifdef HAVE_BLAS
/* growable int64 array */
typedef struct { int64_t *a; int64_t n, cap; } I64V;
static int i64v_push(I64V *v, int64_t x)
{
    if (v->n == v->cap) {
        int64_t nc = v->cap ? 2 * v->cap : 1024;
        int64_t *na = (int64_t *)realloc(v->a, (size_t)nc * sizeof(int64_t));
        if (!na) return -1;
        v->a = na; v->cap = nc;
    }
    v->a[v->n++] = x;
    return 0;
}

/* -------------------------------------------------------------------------
 * Supernode-panel variant of sn_factor_z / sn_factor_d (panel width P > 1; src/factor_sn_panel.inc).
 * Per panel [j0, j1): (1) for each column, the search over the supernodes
 * complete before j0 (its reached list in topological order, segment
 * starts, the rows non-pivotal at j0) and the scatter of A[:, c] into the
 * column's own accumulator; (2) one search from all those supernodes gives
 * a common topological order, and each supernode is applied to every panel
 * column that reached it at once (gather, ztrsm, zgemm, scatter); (3) per
 * column in order: a search from its pattern rows that became pivotal
 * inside the panel, the updates of those (within-panel) supernode segments
 * one column at a time, then pivot, U and L exactly as in sn_factor_z.
 * A supernode complete before j0 never reads a row pivoted inside the
 * panel, so all pre-panel updates first and the within-panel ones after is
 * a valid order; the open supernode at j0 contributes its columns < j0 in
 * (2) and its columns >= j0 in (3) (a block split of its triangular solve).
 * ------------------------------------------------------------------------- */
/* pointer-alpha calling forms of the real level-3 / level-2 BLAS (the complex ones take alpha and beta by pointer) */
static inline void sn_dtrsm(int order, int side, int uplo, int trans, int diag, int M, int N, const double *alpha,
                            const double *A, int lda, double *B, int ldb)
{
    cblas_dtrsm(order, side, uplo, trans, diag, M, N, *alpha, A, lda, B, ldb);
}
static inline void sn_dgemm(int order, int ta, int tb, int M, int N, int K, const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta, double *C, int ldc)
{
    cblas_dgemm(order, ta, tb, M, N, K, *alpha, A, lda, B, ldb, *beta, C, ldc);
}
#define PANEL_NAME sn_factor_panel_z
#define PANEL_T double complex
#define PANEL_IS_COMPLEX 1
#define PANEL_ABS cabs
#define PANEL_TRSM cblas_ztrsm
#define PANEL_GEMM cblas_zgemm
#define PANEL_TRSV cblas_ztrsv
#define PANEL_GEMV cblas_zgemv
#define PANEL_GPU 1
#define PANEL_FLOPF 8.0
#include "factor_sn_panel.inc"
#undef PANEL_NAME
#undef PANEL_T
#undef PANEL_IS_COMPLEX
#undef PANEL_ABS
#undef PANEL_TRSM
#undef PANEL_GEMM
#undef PANEL_TRSV
#undef PANEL_GEMV
#undef PANEL_GPU
#undef PANEL_FLOPF
#define PANEL_NAME sn_factor_panel_d
#define PANEL_T double
#define PANEL_IS_COMPLEX 0
#define PANEL_ABS fabs
#define PANEL_TRSM sn_dtrsm
#define PANEL_GEMM sn_dgemm
#define PANEL_TRSV cblas_dtrsv
#define PANEL_GEMV sn_dgemv
#define PANEL_GPU 0
#define PANEL_FLOPF 2.0
#include "factor_sn_panel.inc"
#undef PANEL_NAME
#undef PANEL_T
#undef PANEL_IS_COMPLEX
#undef PANEL_ABS
#undef PANEL_TRSM
#undef PANEL_GEMM
#undef PANEL_TRSV
#undef PANEL_GEMV
#undef PANEL_GPU
#undef PANEL_FLOPF
#endif

int fplus_factor_sn(int64_t n, const int64_t *Ap, const int64_t *Ai, const void *Ax, int is_complex, int verbose,
                    int64_t relax_rows, int64_t panel, int use_gpu, double gpu_min_flops, double pivot_rel,
                    fplus_factors *out, char *err, size_t errlen)
{
    memset(out, 0, sizeof(*out));
    if (!(pivot_rel >= 0.0)) { fplus_errf(err, errlen, "pivot_rel must be >= 0"); return FPLUS_ERR_BADARG; }
    { int vrc = fplus_check_compressed(n, Ap, Ai, 0, err, errlen);
      if (vrc == FPLUS_OK) vrc = fplus_check_finite(Ap[n], Ax, is_complex, err, errlen);
      if (vrc != FPLUS_OK) return vrc; }
    int ret = FPLUS_ERR_NOMEM;
    double t_start = wall_time();
    size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    int64_t nnzA = Ap[n];
    SnWork W;
    memset(&W, 0, sizeof(W));
    W.n = n; W.is_complex = is_complex;
    W.sn = (Supernode *)calloc((size_t)n, sizeof(Supernode));
    W.col_sn = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.pinv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.p = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.rowpos = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.wrow = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    W.wsn = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    W.segfirst = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.xi = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.nonpiv = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.miss = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.relax_rows = relax_rows > 0 ? relax_rows : 0;
    W.relax_auto = relax_rows < 0;
    W.rz_w[0] = 4; W.rz_w[1] = 16; W.rz_w[2] = 48; W.rz_z[0] = 0.8; W.rz_z[1] = 0.1; W.rz_z[2] = 0.05; W.rz_cap = 32;
    { const char *e = getenv("FPLUS_RELAX_AUTO");      /* "w0,w1,w2,z0,z1,z2,cap": tuning only */
      if (e) { long a0, a1, a2, a6; double b0, b1, b2;
               if (sscanf(e, "%ld,%ld,%ld,%lf,%lf,%lf,%ld", &a0, &a1, &a2, &b0, &b1, &b2, &a6) == 7) {
                   W.rz_w[0] = a0; W.rz_w[1] = a1; W.rz_w[2] = a2; W.rz_z[0] = b0; W.rz_z[1] = b1; W.rz_z[2] = b2; W.rz_cap = a6; } } }
    W.pivot_rel = pivot_rel;
    W.err = err; W.errlen = errlen;
    W.pstack = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.head = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    W.capU = 4 * nnzA + n + 16;
    W.Up = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    W.Ui = (int64_t *)malloc((size_t)W.capU * sizeof(int64_t));
    W.Ux = malloc((size_t)W.capU * vs);
    W.x = calloc((size_t)n, vs);
    W.y = malloc((size_t)(n > 0 ? 2 * n : 1) * vs);
    int64_t *Lp = NULL, *Li = NULL, *q = NULL;
    void *Lx = NULL;
    if (!W.sn || !W.col_sn || !W.pinv || !W.p || !W.rowpos || !W.wrow || !W.wsn || !W.segfirst
            || !W.xi || !W.nonpiv || !W.miss || !W.pstack || !W.head || !W.Up || !W.Ui || !W.Ux || !W.x || !W.y) {
        fplus_errf(err, errlen, "out of memory");
        goto done;
    }
    for (int64_t i = 0; i < n; i++) { W.pinv[i] = -1; W.rowpos[i] = -1; }
    W.Up[0] = 0;
    if (verbose >= 1) {
        printf("[FPLUS] LU factorisation start: n=%ld, nnz(A)=%ld, %s, supernodal left-looking, pivot=partial, relax_rows=%ld, panel=%ld\n",
               (long)n, (long)nnzA, is_complex ? "complex128" : "float64", (long)relax_rows, (long)panel);
        fflush(stdout);
    }
#ifdef FPLUS_PROFILE
    memset(&SNP, 0, sizeof(SNP)); SNP_last = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) SNP_last[i] = -5;
#endif
    /* panel < 0 (auto): the width from the thread count.  One thread: 64 (2026-10-12, 1 thread, n 1.2k-14k, real and complex:
       64-128 within 2 percent of the best width, 512 up to 1.3x slower); more threads keep more columns in flight: base x threads/2,
       at most cap (512 at 16 threads: the 48 x 144 production setting of FINDINGS 63.126).  FPLUS_PANEL_AUTO="base,cap": tuning. */
    if (panel < 0) {
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        int64_t base = 64, cap = 512;
        { const char *e = getenv("FPLUS_PANEL_AUTO"); long a, b; if (e && sscanf(e, "%ld,%ld", &a, &b) == 2 && a > 0 && b > 0) { base = a; cap = b; } }
        int64_t P = base * (nt > 2 ? nt / 2 : 1);
        panel = P < cap ? P : cap;
    }
    /* a panel wider than n is one panel of all n columns: cap it (the workspaces scale with it; audit 2026-10-10) */
    const int64_t panel_w = panel > 1 ? (panel < (n > 1 ? n : 1) ? panel : (n > 1 ? n : 1)) : 1;
    int rc;
#ifdef HAVE_BLAS
    GpuCtx G; memset(&G, 0, sizeof(G));
    if (use_gpu && is_complex && panel_w > 1) gpu_init(&G, gpu_min_flops, verbose);
    if (!is_complex && panel_w > 1) {
        rc = sn_factor_panel_d(&W, Ap, Ai, (const double *)Ax, verbose, panel_w, NULL);
    }
    else if (is_complex && panel_w > 1) {
        rc = sn_factor_panel_z(&W, Ap, Ai, (const double complex *)Ax, verbose, panel_w, G.ok ? &G : NULL);
        if (G.ok && verbose >= 1)
            printf("[FPLUS] gpu: %ld zgemm calls, %.3e flops, %.2f s (%.1f GF/s including synchronization)\n",
                   (long)G.calls, G.flops, G.t, G.t > 0 ? G.flops / G.t / 1e9 : 0.0);
        if (G.ok && G.destroy) G.destroy(G.handle);   /* one handle per factorization: release it (audit 2026-10-10) */
    }
    else
#endif
    rc = is_complex
        ? sn_factor_z(&W, Ap, Ai, (const double complex *)Ax, verbose)
        : sn_factor_d(&W, Ap, Ai, (const double *)Ax, verbose);
#ifdef FPLUS_PROFILE
    if (rc == 0) {
        double tt = 0; for (int k = 0; k < NPH; k++) tt += SNP.t[k];
        printf("[PROF] phases (s, share of %.1f s):\n", tt);
        for (int k = 0; k < NPH; k++) printf("[PROF]   %-22s %9.2f  %5.1f%%\n", PH_NAME[k], SNP.t[k], 100 * SNP.t[k] / tt);
        printf("[PROF] updates %ld (BLAS path %ld); flops BLAS %.3e loop %.3e; GF/s BLAS %.2f loop %.2f; GB/s BLAS %.2f loop %.2f\n",
               (long)SNP.n_upd, (long)SNP.n_upd_blas, SNP.fl_blas, SNP.fl_loop, SNP.fl_blas / SNP.t[2] / 1e9, SNP.fl_loop / SNP.t[3] / 1e9,
               SNP.by_blas / SNP.t[2] / 1e9, SNP.by_loop / SNP.t[3] / 1e9);
        double ft = SNP.fl_blas + SNP.fl_loop;
        printf("[PROF] panel reuse: %.1f%% of update flops come from a supernode that also updated the previous column\n", 100 * SNP.fl_reuse / ft);
        printf("[PROF] update flops by updating-supernode width (2^k):\n");
        for (int k = 0; k < 24; k++) if (SNP.fl_w[k] > 0) printf("[PROF]   w %8ld-%-8ld %5.1f%%\n", 1L << k, (2L << k) - 1, 100 * SNP.fl_w[k] / ft);
        /* final supernode widths */
        int64_t cw[24] = {0}, cc[24] = {0}; double nl[24] = {0}, nlt = 0;
        for (int64_t s = 0; s < W.nsn; s++) { Supernode *S = &W.sn[s]; int b = ilog2_i64(S->ncols); cw[b]++; cc[b] += S->ncols;
            double z = (double)S->ncols * S->nrows - (double)S->ncols * (S->ncols - 1) / 2; nl[b] += z; nlt += z; }
        printf("[PROF] supernodes %ld, mean width %.2f; by width (2^k): count, columns, share of nnz(L)\n", (long)W.nsn, (double)n / W.nsn);
        for (int k = 0; k < 24; k++) if (cw[k]) printf("[PROF]   w %8ld-%-8ld %9ld %9ld %5.1f%%\n", 1L << k, (2L << k) - 1, (long)cw[k], (long)cc[k], 100 * nl[k] / nlt);
        printf("[PROF] relaxed joins %ld (relax_rows %ld), explicit zeros added %.3e\n", (long)SNP.n_relax, (long)W.relax_rows, SNP.zeros);
        printf("[PROF] non-joins by pattern mismatch (missing + extra rows, bucket 2^k of d+1):\n");
        for (int k = 0; k < 24; k++) if (SNP.mm[k]) printf("[PROF]   d %8ld-%-8ld %9ld\n", (1L << k) - 1, (2L << k) - 2, (long)SNP.mm[k]);
        fflush(stdout);
    }
    free(SNP_last);
#endif
    if (rc == -1) { ret = FPLUS_ERR_NOMEM; fplus_errf(err, errlen, "out of memory"); goto done; }
    if (rc != 0) { ret = rc; goto done; }

    /* L in CSC, pivot-order rows, the unit diagonal first in every column */
    {
        int64_t nnzL = 0;
        for (int64_t s = 0; s < W.nsn; s++) {
            Supernode *S = &W.sn[s];
            for (int64_t cc = 0; cc < S->ncols; cc++) nnzL += S->nrows - cc;
        }
        Lp = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
        Li = (int64_t *)malloc((size_t)nnzL * sizeof(int64_t));
        Lx = malloc((size_t)nnzL * vs);
        q = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        if (!Lp || !Li || !Lx || !q) { ret = FPLUS_ERR_NOMEM; fplus_errf(err, errlen, "out of memory"); goto done; }
        /* the rows below a supernode's pivot rows go out in ascending (pivot-order) row index, the same order in each of
         * its columns: the blocked solves (solve_plan.c) then split them across threads without two threads writing one
         * cache line of the solution */
        int64_t maxrest = 0;
        for (int64_t s = 0; s < W.nsn; s++) if (W.sn[s].nrows - W.sn[s].ncols > maxrest) maxrest = W.sn[s].nrows - W.sn[s].ncols;
        RowOrd *ord = (RowOrd *)malloc((size_t)(maxrest > 0 ? maxrest : 1) * sizeof(RowOrd));
        if (!ord) { ret = FPLUS_ERR_NOMEM; fplus_errf(err, errlen, "out of memory"); goto done; }
        int64_t o = 0;
        for (int64_t s = 0; s < W.nsn; s++) {
            Supernode *S = &W.sn[s];
            const int64_t nrest = S->nrows - S->ncols;
            for (int64_t t = 0; t < nrest; t++) { ord[t].row = W.pinv[S->rows[S->ncols + t]]; ord[t].rr = S->ncols + t; }
            qsort(ord, (size_t)nrest, sizeof(RowOrd), roword_cmp);
            for (int64_t cc = 0; cc < S->ncols; cc++) {
                int64_t j = S->first + cc;
                Lp[j] = o;
                Li[o] = j;
                if (is_complex) ((double complex *)Lx)[o] = 1.0; else ((double *)Lx)[o] = 1.0;
                o++;
                for (int64_t rr = cc + 1; rr < S->ncols; rr++) {      /* the pivot rows of the supernode: ascending already */
                    Li[o] = W.pinv[S->rows[rr]];
                    if (is_complex) ((double complex *)Lx)[o] = ((double complex *)S->blk)[cc * S->ld + rr];
                    else            ((double *)Lx)[o] = ((double *)S->blk)[cc * S->ld + rr];
                    o++;
                }
                for (int64_t t = 0; t < nrest; t++) {
                    const int64_t rr = ord[t].rr;
                    Li[o] = ord[t].row;
                    if (is_complex) ((double complex *)Lx)[o] = ((double complex *)S->blk)[cc * S->ld + rr];
                    else            ((double *)Lx)[o] = ((double *)S->blk)[cc * S->ld + rr];
                    o++;
                }
            }
        }
        free(ord);
        Lp[n] = o;
        for (int64_t i = 0; i < n; i++) q[i] = i;
        int64_t *nui = (int64_t *)realloc(W.Ui, (size_t)(W.nnzU > 0 ? W.nnzU : 1) * sizeof(int64_t));
        void *nux = realloc(W.Ux, (size_t)(W.nnzU > 0 ? W.nnzU : 1) * vs);
        if (nui) W.Ui = nui;
        if (nux) W.Ux = nux;
        if (verbose >= 1) {
            printf("[FPLUS] LU factorisation complete: nnz(L)=%ld, nnz(U)=%ld, total nnz=%ld, "
                   "supernodes=%ld, time=%.3fs\n", (long)nnzL, (long)W.nnzU,
                   (long)(nnzL + W.nnzU), (long)W.nsn, wall_time() - t_start);
            fflush(stdout);
        }
        /* hand the buffers to the caller */
        out->n = n; out->is_complex = is_complex; out->csc = 1; out->nsupernodes = W.nsn;
        out->Lp = Lp; out->Li = Li; out->Lx = Lx;
        out->Up = W.Up; out->Ui = W.Ui; out->Ux = W.Ux;
        out->p = W.p; out->q = q;
        Lx = NULL; Li = NULL; Lp = NULL; q = NULL;
        W.Ux = NULL; W.Ui = NULL; W.Up = NULL; W.p = NULL;
        ret = FPLUS_OK;
    }
done:
    if (W.sn)
        for (int64_t s = 0; s < W.nsn; s++) { free(W.sn[s].rows); free(W.sn[s].blk); free(W.sn[s].dfs_rows); }
    free(W.sn); free(W.col_sn); free(W.pinv); free(W.p); free(W.rowpos); free(W.wrow); free(W.wsn);
    free(W.segfirst); free(W.xi); free(W.nonpiv); free(W.miss); free(W.pstack); free(W.head);
    free(W.Up); free(W.Ui); free(W.Ux); free(W.x); free(W.y);
    free(Lp); free(Li); free(Lx); free(q);
    return ret;
}

void fplus_factors_free(fplus_factors *f)
{
    if (!f) return;
    free(f->Lp); free(f->Li); free(f->Lx); free(f->Up); free(f->Ui); free(f->Ux); free(f->p); free(f->q);
    f->Lp = f->Li = f->Up = f->Ui = f->p = f->q = NULL; f->Lx = f->Ux = NULL;
}

