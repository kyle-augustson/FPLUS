/* solve_plan.c -- blocked triangular solves on the CSC factors, threaded inside large supernodes, for one or several
 * right-hand sides.
 *
 * A plan, built once per factorization from the patterns alone, records the supernodes of L (consecutive columns whose
 * rows below the diagonal block coincide: column c of supernode [c0, c1) holds c, c+1 .. c1-1, then the shared rows R)
 * and the row segments of each U column (runs of consecutive rows above the diagonal).  The sweeps go through each
 * supernode in column blocks.  A block's update of the rows outside it is split across threads by target row: every
 * row is owned by one thread and receives its updates in the column order of the plain sweeps (fplus_tri_solve), so
 * the solves with A (trans 0) are bit-identical to fplus_tri_solve at any thread count.  The transposed solves gather
 * dot products; a block's columns compute the part of their dot products outside the block in parallel, which groups
 * the L^T sums differently from the plain sweep (deterministic, equal to round-off).
 *
 * Several right-hand sides share each read of the factors: with the solves memory-bound, k right-hand sides cost much
 * less than k solves. */
#include "fplus_internal.h"

#define PLAN_BLOCK   256        /* columns per block */
#define PLAN_PAR_MIN 32768      /* (entries x right-hand sides) below which a block update stays on one thread */
#define PLAN_CHUNK   1024       /* target rows per parallel task */

#ifdef _OPENMP
#define PLAN_OMP(x) _Pragma(x)
#else
#define PLAN_OMP(x)
#endif

/* -DPLAN_PROFILE: wall time and entries per phase of the forward / backward sweeps, printed after each solve with A
 * (global state: one solve at a time).  0 plain-sweep supernodes of L, 1 L triangles, 2 L updates threaded, 3 L updates
 * on one thread, 4 U triangles, 5 U updates threaded, 6 U updates on one thread */
#if defined(PLAN_PROFILE) && defined(_OPENMP)
#include <stdio.h>
static double plan_t[7], plan_e[7];
#define PT_START(v) const double v = omp_get_wtime()
#define PT_ADD(v, k, ent) do { plan_t[k] += omp_get_wtime() - (v); plan_e[k] += (double)(ent); } while (0)
#else
#define PT_START(v) do {} while (0)
#define PT_ADD(v, k, ent) do {} while (0)
#endif

struct fplus_solve_plan {
    int64_t n, nnzL, nnzU;
    int64_t block, chunk_min;   /* columns per block; minimum target rows per parallel task (FPLUS_PLAN_BLOCK / _CHUNK_MIN) */
    int generic;                /* the L pattern lacks the layout (diagonal first in every column): plain sweeps */
    int64_t nsn, *sn;           /* L supernode s: columns sn[s] .. sn[s+1]-1 */
    unsigned char *sn_par;      /* 1: the rows R are distinct and below the supernode (blocked, threaded); 0: plain sweeps */
    int64_t *useg_ptr;          /* U column j: segments useg_ptr[j] .. useg_ptr[j+1]-1 in storage order, rows < j only */
    int64_t *useg_row, *useg_len, *useg_off;
    int64_t *useg_srt;          /* per column, its segment indices sorted by first row (absolute indices into useg_*) */
    int64_t *udiag;             /* position of U[j, j] in Ui / Ux, -1 when missing */
};

void fplus_solve_plan_free(fplus_solve_plan *pl)
{
    if (!pl) return;
    free(pl->sn); free(pl->sn_par); free(pl->useg_ptr); free(pl->useg_row); free(pl->useg_len); free(pl->useg_off);
    free(pl->useg_srt); free(pl->udiag); free(pl);
}

int64_t fplus_solve_plan_nsupernodes(const fplus_solve_plan *pl) { return pl->generic ? -1 : pl->nsn; }

typedef struct { int64_t row, idx; } SegKey;

static int segkey_cmp(const void *a, const void *b)
{
    int64_t x = ((const SegKey *)a)->row, y = ((const SegKey *)b)->row;
    return (x > y) - (x < y);
}

int fplus_solve_plan_create(int64_t n, const int64_t *Lp, const int64_t *Li, const int64_t *Up, const int64_t *Ui,
                            fplus_solve_plan **out, char *err, size_t errlen)
{
    *out = NULL;
    {   /* the sweeps index x with these: in range, monotone pointers */
        int vrc = fplus_check_compressed(n, Lp, Li, 0, err, errlen);
        if (vrc == FPLUS_OK) vrc = fplus_check_compressed(n, Up, Ui, 0, err, errlen);
        if (vrc != FPLUS_OK) return vrc;
    }
    fplus_solve_plan *pl = (fplus_solve_plan *)calloc(1, sizeof(*pl));
    if (!pl) goto oom;
    pl->n = n; pl->nnzL = Lp[n]; pl->nnzU = Up[n];
    {   /* tuning knobs for experiments; the defaults otherwise */
        const char *e = getenv("FPLUS_PLAN_BLOCK"), *c = getenv("FPLUS_PLAN_CHUNK_MIN");
        pl->block = e && atol(e) > 0 ? atol(e) : PLAN_BLOCK;
        pl->chunk_min = c && atol(c) > 0 ? atol(c) : PLAN_CHUNK / 8;
    }
    for (int64_t j = 0; j < n; j++)
        if (Lp[j] >= Lp[j + 1] || Li[Lp[j]] != j) { pl->generic = 1; *out = pl; return FPLUS_OK; }

    /* L supernodes: column j+1 continues column j's when column j is (j, j+1, rest) and column j+1 is (j+1, rest) */
    const size_t nn = (size_t)(n > 0 ? n : 1);
    pl->sn = (int64_t *)malloc((nn + 1) * sizeof(int64_t));
    pl->sn_par = (unsigned char *)malloc(nn);
    int64_t *stamp = (int64_t *)malloc(nn * sizeof(int64_t));
    if (!pl->sn || !pl->sn_par || !stamp) { free(stamp); goto oom; }
    int64_t ns = 0;
    pl->sn[0] = 0;
    for (int64_t j = 0; j < n; j++) {
        int join = 0;
        if (j + 1 < n) {
            const int64_t a = Lp[j], la = Lp[j + 1] - a, b = Lp[j + 1], lb = Lp[j + 2] - b;
            join = la >= 2 && lb == la - 1 && Li[a + 1] == j + 1
                   && (la == 2 || memcmp(Li + a + 2, Li + b + 1, (size_t)(la - 2) * sizeof(int64_t)) == 0);
        }
        if (!join) pl->sn[++ns] = j + 1;
    }
    pl->nsn = ns;
    for (int64_t i = 0; i < n; i++) stamp[i] = -1;
    for (int64_t s = 0; s < ns; s++) {     /* threaded updates need R distinct and below the supernode */
        const int64_t c1 = pl->sn[s + 1], q0 = Lp[c1 - 1] + 1, q1 = Lp[c1];
        unsigned char par = 1;
        for (int64_t q = q0; q < q1; q++) {
            const int64_t r = Li[q];
            if (r < c1 || stamp[r] == s) { par = 0; break; }
            stamp[r] = s;
        }
        pl->sn_par[s] = par;
    }
    free(stamp);

    /* U: the diagonal, and the runs of consecutive rows < j in storage order */
    pl->udiag = (int64_t *)malloc(nn * sizeof(int64_t));
    pl->useg_ptr = (int64_t *)malloc((nn + 1) * sizeof(int64_t));
    if (!pl->udiag || !pl->useg_ptr) goto oom;
    int64_t nseg = 0;
    for (int64_t j = 0; j < n; j++) {
        pl->useg_ptr[j] = nseg;
        pl->udiag[j] = -1;
        int64_t prev = -2;                 /* row of the previous entry when it was < j, else -2 */
        for (int64_t q = Up[j]; q < Up[j + 1]; q++) {
            const int64_t i = Ui[q];
            if (i == j && pl->udiag[j] < 0) pl->udiag[j] = q;
            if (i >= j) { prev = -2; continue; }
            if (i != prev + 1) nseg++;
            prev = i;
        }
    }
    pl->useg_ptr[n] = nseg;
    const size_t sz = (size_t)(nseg > 0 ? nseg : 1) * sizeof(int64_t);
    pl->useg_row = (int64_t *)malloc(sz); pl->useg_len = (int64_t *)malloc(sz);
    pl->useg_off = (int64_t *)malloc(sz); pl->useg_srt = (int64_t *)malloc(sz);
    if (!pl->useg_row || !pl->useg_len || !pl->useg_off || !pl->useg_srt) goto oom;
    int64_t maxseg = 0;
    for (int64_t j = 0; j < n; j++) {
        int64_t k = pl->useg_ptr[j] - 1, prev = -2;
        for (int64_t q = Up[j]; q < Up[j + 1]; q++) {
            const int64_t i = Ui[q];
            if (i >= j) { prev = -2; continue; }
            if (i != prev + 1) { k++; pl->useg_row[k] = i; pl->useg_len[k] = 0; pl->useg_off[k] = q; }
            pl->useg_len[k]++;
            prev = i;
        }
        if (pl->useg_ptr[j + 1] - pl->useg_ptr[j] > maxseg) maxseg = pl->useg_ptr[j + 1] - pl->useg_ptr[j];
    }
    SegKey *key = (SegKey *)malloc((size_t)(maxseg > 0 ? maxseg : 1) * sizeof(SegKey));
    if (!key) goto oom;
    for (int64_t j = 0; j < n; j++) {     /* sorted by row; overlapping segments (a repeated row) fall back to plain sweeps */
        const int64_t a = pl->useg_ptr[j], m = pl->useg_ptr[j + 1] - a;
        for (int64_t t = 0; t < m; t++) { key[t].row = pl->useg_row[a + t]; key[t].idx = a + t; }
        qsort(key, (size_t)m, sizeof(SegKey), segkey_cmp);
        for (int64_t t = 0; t < m; t++) {
            pl->useg_srt[a + t] = key[t].idx;
            if (t > 0 && key[t].row < key[t - 1].row + pl->useg_len[key[t - 1].idx]) pl->generic = 1;
        }
    }
    free(key);
    *out = pl;
    return FPLUS_OK;
oom:
    fplus_solve_plan_free(pl);
    fplus_errf(err, errlen, "out of memory");
    return FPLUS_ERR_NOMEM;
}

/* y -= a * b.  The complex product written out (the textbook formula, as C99's complex multiply computes it without its NaN
 * recovery): the loops vectorize, where the C99 product's NaN branch keeps them scalar (4x slower per entry on current x86-64 server CPUs with gcc 11).
 * The solve files are compiled without floating-point contraction (CMakeLists.txt, setup.py), so every operation rounds as written:
 * the same bits as fplus_tri_solve's C99 products, on any compiler or flags.  Finite factors and right-hand sides only (no NaN
 * recovery; the factorizations reject non-finite input). */
static inline void fms_z(double complex *y, double complex a, double complex b)
{
    const double ar = creal(a), ai = cimag(a), br = creal(b), bi = cimag(b);
    double *yd = (double *)y;
    yd[0] = yd[0] - (ar * br - ai * bi);
    yd[1] = yd[1] - (ar * bi + ai * br);
}
#define FMS_z(y, a, b) fms_z(&(y), (a), (b))
#define FMS_d(y, a, b) ((y) = (y) - (a) * (b))

static inline int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t max_i64(int64_t a, int64_t b) { return a > b ? a : b; }

/* threads available to a block update (1: every update takes the serial path, one chunk, no OpenMP region) */
static inline int plan_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* rows per parallel task: about 4 tasks per thread, within [PLAN_CHUNK / 8, PLAN_CHUNK] */
static inline int64_t plan_chunk(int64_t rows, int64_t cmin)
{
#ifdef _OPENMP
    const int64_t T = omp_get_max_threads();
#else
    const int64_t T = 1;
#endif
    const int64_t c = (rows + 4 * T - 1) / (4 * T);
    const int64_t cmax = cmin > PLAN_CHUNK ? cmin : PLAN_CHUNK;
    return c < cmin ? cmin : (c > cmax ? cmax : c);
}

#define PLAN_KERNELS(SUF, T, CJ)                                                                                         \
/* L y = b: supernodes ascending, column blocks ascending */                                                             \
static void fwdL_##SUF(const fplus_solve_plan *pl, const int64_t *Lp, const int64_t *Li, const T *Lx, T *x,             \
                       int64_t nrhs, int64_t ldx)                                                                        \
{                                                                                                                        \
    for (int64_t s = 0; s < pl->nsn; s++) {                                                                              \
        const int64_t c0 = pl->sn[s], c1 = pl->sn[s + 1];                                                                \
        if (!pl->sn_par[s]) {                  /* the plain column sweep of fplus_tri_solve */                           \
            PT_START(t0);                                                                                                \
            for (int64_t j = c0; j < c1; j++)                                                                            \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    T *xr = x + r * ldx; const T xj = xr[j];                                                             \
                    if (xj == 0) continue;                                                                               \
                    for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) { const int64_t i = Li[q]; if (i != j) FMS_##SUF(xr[i], xj, Lx[q]); } \
                }                                                                                                        \
            PT_ADD(t0, 0, Lp[c1] - Lp[c0]);                                                                              \
            continue;                                                                                                    \
        }                                                                                                                \
        const int64_t *R = Li + Lp[c1 - 1] + 1, m = Lp[c1] - Lp[c1 - 1] - 1;                                            \
        for (int64_t k0 = c0; k0 < c1; k0 += pl->block) {                                                               \
            const int64_t k1 = min_i64(k0 + pl->block, c1);                                                             \
            PT_START(t1);                                                                                                \
            for (int64_t c = k0; c < k1; c++) {   /* the block's triangle: rows c+1 .. k1-1 at Lp[c]+1 .. */             \
                const T *v = Lx + Lp[c] + 1;                                                                             \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    T *xr = x + r * ldx; const T xc = xr[c];                                                             \
                    if (xc == 0) continue;                                                                               \
                    for (int64_t i = c + 1; i < k1; i++) FMS_##SUF(xr[i], xc, v[i - c - 1]);                                     \
                }                                                                                                        \
            }                                                                                                            \
            PT_ADD(t1, 1, (k1 - k0) * (k1 - k0 - 1) / 2);                                                                \
            /* the rows below the block: pivot rows k1 .. c1-1, then R; position p of column c at Lp[c] + (k1 - c) + p */ \
            const int64_t nt = c1 - k1, nb = nt + m;                                                                     \
            if (nb == 0) continue;                                                                                       \
            PT_START(t2);                                                                                                \
            const int64_t cs0 = plan_chunk(nb, pl->chunk_min), nch0 = (nb + cs0 - 1) / cs0;                                                      \
            const int par = plan_threads() > 1 && nch0 > 1 && nb * (k1 - k0) * nrhs >= PLAN_PAR_MIN;  /* serial: one chunk (same per-row order) */ \
            const int64_t cs = par ? cs0 : nb, nch = par ? nch0 : 1;                                            \
            if (par) { PLAN_OMP("omp parallel for schedule(dynamic, 1)") for (int64_t ch = 0; ch < nch; ch++) {                                                                       \
                const int64_t p0 = ch * cs, p1 = min_i64(p0 + cs, nb);                                   \
                const int64_t pt = min_i64(p1, nt), pr = max_i64(p0, nt);                                                \
                for (int64_t c = k0; c < k1; c++) {                                                                      \
                    const T *v = Lx + Lp[c] + (k1 - c);                                                                  \
                    for (int64_t r = 0; r < nrhs; r++) {                                                                 \
                        T *xr = x + r * ldx; const T xc = xr[c];                                                         \
                        if (xc == 0) continue;                                                                           \
                        for (int64_t p = p0; p < pt; p++) FMS_##SUF(xr[k1 + p], xc, v[p]);                                       \
                        for (int64_t p = pr; p < p1; p++) FMS_##SUF(xr[R[p - nt]], xc, v[p]);                                    \
                    }                                                                                                    \
                }                                                                                                        \
            } } else { for (int64_t ch = 0; ch < nch; ch++) {                                                                       \
                const int64_t p0 = ch * cs, p1 = min_i64(p0 + cs, nb);                                   \
                const int64_t pt = min_i64(p1, nt), pr = max_i64(p0, nt);                                                \
                for (int64_t c = k0; c < k1; c++) {                                                                      \
                    const T *v = Lx + Lp[c] + (k1 - c);                                                                  \
                    for (int64_t r = 0; r < nrhs; r++) {                                                                 \
                        T *xr = x + r * ldx; const T xc = xr[c];                                                         \
                        if (xc == 0) continue;                                                                           \
                        for (int64_t p = p0; p < pt; p++) FMS_##SUF(xr[k1 + p], xc, v[p]);                                       \
                        for (int64_t p = pr; p < p1; p++) FMS_##SUF(xr[R[p - nt]], xc, v[p]);                                    \
                    }                                                                                                    \
                }                                                                                                        \
            } }                                                                                                            \
            PT_ADD(t2, par ? 2 : 3, nb * (k1 - k0));                                                                     \
        }                                                                                                                \
    }                                                                                                                    \
}                                                                                                                        \
                                                                                                                         \
/* U x = y: supernodes descending, column blocks descending */                                                           \
static void bwdU_##SUF(const fplus_solve_plan *pl, const T *Ux, T *x, int64_t nrhs, int64_t ldx)                         \
{                                                                                                                        \
    const int64_t *sp = pl->useg_ptr, *srow = pl->useg_row, *slen = pl->useg_len, *soff = pl->useg_off, *srt = pl->useg_srt; \
    for (int64_t s = pl->nsn - 1; s >= 0; s--) {                                                                         \
        const int64_t c0 = pl->sn[s], c1 = pl->sn[s + 1];                                                                \
        for (int64_t k1 = c1; k1 > c0; k1 -= pl->block) {                                                               \
            const int64_t k0 = max_i64(k1 - pl->block, c0);                                                             \
            int64_t lo_row = k0, work = 0;                                                                               \
            PT_START(t4);                                                                                                \
            for (int64_t j = k1 - 1; j >= k0; j--) {  /* the block's triangle: x_j /= U_jj, then the rows k0 .. j-1 */  \
                const T d = Ux[pl->udiag[j]];                                                                            \
                for (int64_t r = 0; r < nrhs; r++) x[j + r * ldx] = x[j + r * ldx] / d;                                  \
                for (int64_t k = sp[j]; k < sp[j + 1]; k++) {                                                            \
                    const int64_t a0 = max_i64(srow[k], k0), a1 = min_i64(srow[k] + slen[k], j);                         \
                    if (srow[k] < lo_row) lo_row = srow[k];                                                              \
                    work += slen[k];                                                                                     \
                    if (a0 >= a1) continue;                                                                              \
                    const T *v = Ux + soff[k] + (a0 - srow[k]);                                                          \
                    for (int64_t r = 0; r < nrhs; r++) {                                                                 \
                        T *xr = x + r * ldx; const T xj = xr[j];                                                         \
                        if (xj == 0) continue;                                                                           \
                        for (int64_t i = a0; i < a1; i++) FMS_##SUF(xr[i], xj, v[i - a0]);                                       \
                    }                                                                                                    \
                }                                                                                                        \
            }                                                                                                            \
            PT_ADD(t4, 4, (k1 - k0) * (k1 - k0 - 1) / 2);                                                                \
            if (lo_row >= k0) continue;                                                                                  \
            PT_START(t5);                                                                                                \
            /* the rows above the block, lo_row .. k0-1, by row range; each range finds its segments by bisection */     \
            const int64_t span = k0 - lo_row, cs0 = plan_chunk(span, pl->chunk_min), nch0 = (span + cs0 - 1) / cs0;                                \
            const int par = plan_threads() > 1 && nch0 > 1 && work * nrhs >= PLAN_PAR_MIN;  /* serial: one chunk (same per-row order) */ \
            const int64_t cs = par ? cs0 : span, nch = par ? nch0 : 1;                                                      \
            if (par) { PLAN_OMP("omp parallel for schedule(dynamic, 1)") for (int64_t ch = 0; ch < nch; ch++) {                                                                       \
                const int64_t r0 = lo_row + ch * cs, r1 = min_i64(r0 + cs, k0);                          \
                for (int64_t j = k1 - 1; j >= k0; j--) {                                                                 \
                    int64_t lo = sp[j], hi = sp[j + 1];                                                                  \
                    const int64_t end = hi;                                                                              \
                    while (lo < hi) {                 /* first segment (by row) ending after r0 */                      \
                        const int64_t mid = lo + (hi - lo) / 2, k = srt[mid];                                            \
                        if (srow[k] + slen[k] <= r0) lo = mid + 1; else hi = mid;                                        \
                    }                                                                                                    \
                    for (int64_t t = lo; t < end; t++) {                                                                 \
                        const int64_t k = srt[t];                                                                        \
                        if (srow[k] >= r1) break;                                                                        \
                        const int64_t a0 = max_i64(srow[k], r0), a1 = min_i64(srow[k] + slen[k], r1);                    \
                        const T *v = Ux + soff[k] + (a0 - srow[k]);                                                      \
                        for (int64_t r = 0; r < nrhs; r++) {                                                             \
                            T *xr = x + r * ldx; const T xj = xr[j];                                                     \
                            if (xj == 0) continue;                                                                       \
                            for (int64_t i = a0; i < a1; i++) FMS_##SUF(xr[i], xj, v[i - a0]);                                   \
                        }                                                                                                \
                    }                                                                                                    \
                }                                                                                                        \
            } } else { for (int64_t ch = 0; ch < nch; ch++) {                                                                       \
                const int64_t r0 = lo_row + ch * cs, r1 = min_i64(r0 + cs, k0);                          \
                for (int64_t j = k1 - 1; j >= k0; j--) {                                                                 \
                    int64_t lo = sp[j], hi = sp[j + 1];                                                                  \
                    const int64_t end = hi;                                                                              \
                    while (lo < hi) {                 /* first segment (by row) ending after r0 */                      \
                        const int64_t mid = lo + (hi - lo) / 2, k = srt[mid];                                            \
                        if (srow[k] + slen[k] <= r0) lo = mid + 1; else hi = mid;                                        \
                    }                                                                                                    \
                    for (int64_t t = lo; t < end; t++) {                                                                 \
                        const int64_t k = srt[t];                                                                        \
                        if (srow[k] >= r1) break;                                                                        \
                        const int64_t a0 = max_i64(srow[k], r0), a1 = min_i64(srow[k] + slen[k], r1);                    \
                        const T *v = Ux + soff[k] + (a0 - srow[k]);                                                      \
                        for (int64_t r = 0; r < nrhs; r++) {                                                             \
                            T *xr = x + r * ldx; const T xj = xr[j];                                                     \
                            if (xj == 0) continue;                                                                       \
                            for (int64_t i = a0; i < a1; i++) FMS_##SUF(xr[i], xj, v[i - a0]);                                   \
                        }                                                                                                \
                    }                                                                                                    \
                }                                                                                                        \
            } }                                                                                                            \
            PT_ADD(t5, par ? 5 : 6, work);                                                                               \
        }                                                                                                                \
    }                                                                                                                    \
}                                                                                                                        \
                                                                                                                         \
/* U^T y = b (U^H when cj): column blocks ascending; the parts of the block's dot products over rows < k0 in parallel,   \
 * in storage order, then the rows in the block */                                                                       \
static void fwdUt_##SUF(const fplus_solve_plan *pl, const T *Ux, T *x, int64_t nrhs, int64_t ldx, int cj, T *part)      \
{                                                                                                                        \
    const int64_t *sp = pl->useg_ptr, *srow = pl->useg_row, *slen = pl->useg_len, *soff = pl->useg_off;                  \
    for (int64_t s = 0; s < pl->nsn; s++) {                                                                              \
        const int64_t c0 = pl->sn[s], c1 = pl->sn[s + 1];                                                                \
        for (int64_t k0 = c0; k0 < c1; k0 += pl->block) {                                                               \
            const int64_t k1 = min_i64(k0 + pl->block, c1);                                                             \
            int64_t work = 0;                                                                                            \
            for (int64_t k = sp[k0]; k < sp[k1]; k++) work += slen[k];                                                   \
            const int par = plan_threads() > 1 && (k1 - k0) > 1 && work * nrhs >= PLAN_PAR_MIN; (void)par;                                                \
            if (par) { PLAN_OMP("omp parallel for schedule(dynamic, 4)") for (int64_t j = k0; j < k1; j++)                                                                            \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    const T *xr = x + r * ldx;                                                                           \
                    T acc = xr[j];                                                                                       \
                    for (int64_t k = sp[j]; k < sp[j + 1]; k++) {                                                        \
                        const int64_t a1 = min_i64(srow[k] + slen[k], k0);                                               \
                        const T *v = Ux + soff[k];                                                                       \
                        if (cj) { for (int64_t i = srow[k]; i < a1; i++) FMS_##SUF(acc, CJ(v[i - srow[k]]), xr[i]); }            \
                        else    { for (int64_t i = srow[k]; i < a1; i++) FMS_##SUF(acc, v[i - srow[k]], xr[i]); }                \
                    }                                                                                                    \
                    part[(j - k0) * nrhs + r] = acc;                                                                     \
                } } else { for (int64_t j = k0; j < k1; j++)                                                                            \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    const T *xr = x + r * ldx;                                                                           \
                    T acc = xr[j];                                                                                       \
                    for (int64_t k = sp[j]; k < sp[j + 1]; k++) {                                                        \
                        const int64_t a1 = min_i64(srow[k] + slen[k], k0);                                               \
                        const T *v = Ux + soff[k];                                                                       \
                        if (cj) { for (int64_t i = srow[k]; i < a1; i++) FMS_##SUF(acc, CJ(v[i - srow[k]]), xr[i]); }            \
                        else    { for (int64_t i = srow[k]; i < a1; i++) FMS_##SUF(acc, v[i - srow[k]], xr[i]); }                \
                    }                                                                                                    \
                    part[(j - k0) * nrhs + r] = acc;                                                                     \
                } }                                                                                                        \
            for (int64_t j = k0; j < k1; j++) {                                                                          \
                const T d = cj ? CJ(Ux[pl->udiag[j]]) : Ux[pl->udiag[j]];                                                \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    T *xr = x + r * ldx;                                                                                 \
                    T acc = part[(j - k0) * nrhs + r];                                                                   \
                    for (int64_t k = sp[j]; k < sp[j + 1]; k++) {                                                        \
                        const int64_t a0 = max_i64(srow[k], k0), a1 = srow[k] + slen[k];                                 \
                        const T *v = Ux + soff[k];                                                                       \
                        if (cj) { for (int64_t i = a0; i < a1; i++) FMS_##SUF(acc, CJ(v[i - srow[k]]), xr[i]); }                 \
                        else    { for (int64_t i = a0; i < a1; i++) FMS_##SUF(acc, v[i - srow[k]], xr[i]); }                     \
                    }                                                                                                    \
                    xr[j] = acc / d;                                                                                     \
                }                                                                                                        \
            }                                                                                                            \
        }                                                                                                                \
    }                                                                                                                    \
}                                                                                                                        \
                                                                                                                         \
/* L^T x = y (L^H when cj): column blocks descending; the parts of the block's dot products over the rows below it in   \
 * parallel, then the rows in the block */                                                                               \
static void bwdLt_##SUF(const fplus_solve_plan *pl, const int64_t *Lp, const int64_t *Li, const T *Lx, T *x,            \
                        int64_t nrhs, int64_t ldx, int cj, T *part)                                                      \
{                                                                                                                        \
    for (int64_t s = pl->nsn - 1; s >= 0; s--) {                                                                         \
        const int64_t c0 = pl->sn[s], c1 = pl->sn[s + 1];                                                                \
        if (!pl->sn_par[s]) {                  /* the plain column sweep of fplus_tri_solve */                           \
            for (int64_t j = c1 - 1; j >= c0; j--)                                                                       \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    T *xr = x + r * ldx; T acc = xr[j];                                                                  \
                    for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) {                                                        \
                        const int64_t i = Li[q];                                                                         \
                        if (i > j) FMS_##SUF(acc, (cj ? CJ(Lx[q]) : Lx[q]), xr[i]);                                              \
                    }                                                                                                    \
                    xr[j] = acc;                                                                                         \
                }                                                                                                        \
            continue;                                                                                                    \
        }                                                                                                                \
        const int64_t *R = Li + Lp[c1 - 1] + 1, m = Lp[c1] - Lp[c1 - 1] - 1;                                            \
        for (int64_t k1 = c1; k1 > c0; k1 -= pl->block) {                                                               \
            const int64_t k0 = max_i64(k1 - pl->block, c0), nt = c1 - k1, nb = nt + m;                                  \
            const int par = plan_threads() > 1 && (k1 - k0) > 1 && nb * (k1 - k0) * nrhs >= PLAN_PAR_MIN; (void)par;                                      \
            if (par) { PLAN_OMP("omp parallel for schedule(dynamic, 4)") for (int64_t j = k0; j < k1; j++) {    /* rows below the block: k1 .. c1-1, then R (all final) */            \
                const T *v = Lx + Lp[j] + (k1 - j);                                                                      \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    const T *xr = x + r * ldx;                                                                           \
                    T acc = 0;                                                                                           \
                    if (cj) {                                                                                            \
                        for (int64_t p = 0; p < nt; p++) FMS_##SUF(acc, CJ(v[p]), xr[k1 + p]);                                   \
                        for (int64_t p = nt; p < nb; p++) FMS_##SUF(acc, CJ(v[p]), xr[R[p - nt]]);                               \
                    } else {                                                                                             \
                        for (int64_t p = 0; p < nt; p++) FMS_##SUF(acc, v[p], xr[k1 + p]);                                       \
                        for (int64_t p = nt; p < nb; p++) FMS_##SUF(acc, v[p], xr[R[p - nt]]);                                   \
                    }                                                                                                    \
                    part[(j - k0) * nrhs + r] = acc;                                                                     \
                }                                                                                                        \
            } } else { for (int64_t j = k0; j < k1; j++) {    /* rows below the block: k1 .. c1-1, then R (all final) */            \
                const T *v = Lx + Lp[j] + (k1 - j);                                                                      \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    const T *xr = x + r * ldx;                                                                           \
                    T acc = 0;                                                                                           \
                    if (cj) {                                                                                            \
                        for (int64_t p = 0; p < nt; p++) FMS_##SUF(acc, CJ(v[p]), xr[k1 + p]);                                   \
                        for (int64_t p = nt; p < nb; p++) FMS_##SUF(acc, CJ(v[p]), xr[R[p - nt]]);                               \
                    } else {                                                                                             \
                        for (int64_t p = 0; p < nt; p++) FMS_##SUF(acc, v[p], xr[k1 + p]);                                       \
                        for (int64_t p = nt; p < nb; p++) FMS_##SUF(acc, v[p], xr[R[p - nt]]);                                   \
                    }                                                                                                    \
                    part[(j - k0) * nrhs + r] = acc;                                                                     \
                }                                                                                                        \
            } }                                                                                                            \
            for (int64_t j = k1 - 1; j >= k0; j--) {   /* rows j+1 .. k1-1 at Lp[j]+1 .. */                              \
                const T *v = Lx + Lp[j] + 1;                                                                             \
                for (int64_t r = 0; r < nrhs; r++) {                                                                     \
                    T *xr = x + r * ldx;                                                                                 \
                    T acc = xr[j];                                                                                       \
                    if (cj) { for (int64_t i = j + 1; i < k1; i++) FMS_##SUF(acc, CJ(v[i - j - 1]), xr[i]); }                    \
                    else    { for (int64_t i = j + 1; i < k1; i++) FMS_##SUF(acc, v[i - j - 1], xr[i]); }                        \
                    xr[j] = acc + part[(j - k0) * nrhs + r];                                                             \
                }                                                                                                        \
            }                                                                                                            \
        }                                                                                                                \
    }                                                                                                                    \
}

static inline double conj_plan_d(double v) { return v; }
PLAN_KERNELS(z, double complex, conj)
PLAN_KERNELS(d, double, conj_plan_d)

int fplus_tri_solve_plan(const fplus_solve_plan *pl, int64_t n, const int64_t *Lp, const int64_t *Li, const void *Lx,
                         const int64_t *Up, const int64_t *Ui, const void *Ux, void *x, int64_t nrhs, int64_t ldx,
                         int trans, int is_complex)
{
    if (trans < 0 || trans > 2 || n < 0 || nrhs < 0 || ldx < n || n != pl->n || Lp[n] != pl->nnzL || Up[n] != pl->nnzU)
        return FPLUS_ERR_BADARG;
    const size_t vs = is_complex ? sizeof(double complex) : sizeof(double);
    if (pl->generic) {                     /* plain sweeps, one right-hand side at a time */
        for (int64_t r = 0; r < nrhs; r++) {
            int rc = fplus_tri_solve(n, Lp, Li, Lx, Up, Ui, Ux, (char *)x + (size_t)(r * ldx) * vs, trans, is_complex);
            if (rc != FPLUS_OK) return rc;
        }
        return FPLUS_OK;
    }
    for (int64_t j = 0; j < n; j++) {      /* a zero or missing pivot fails before x is touched */
        const int64_t q = pl->udiag[j];
        if (q < 0) return FPLUS_ERR_SINGULAR;
        if (is_complex ? ((const double complex *)Ux)[q] == 0 : ((const double *)Ux)[q] == 0) return FPLUS_ERR_SINGULAR;
    }
    if (nrhs == 0 || n == 0) return FPLUS_OK;
    if (trans == 0) {
        if (is_complex) { fwdL_z(pl, Lp, Li, (const double complex *)Lx, (double complex *)x, nrhs, ldx);
                          bwdU_z(pl, (const double complex *)Ux, (double complex *)x, nrhs, ldx); }
        else            { fwdL_d(pl, Lp, Li, (const double *)Lx, (double *)x, nrhs, ldx);
                          bwdU_d(pl, (const double *)Ux, (double *)x, nrhs, ldx); }
#if defined(PLAN_PROFILE) && defined(_OPENMP)
        {
            static const char *nm[7] = {"L plain-sweep supernodes", "L triangles", "L updates, threaded", "L updates, one thread",
                                        "U triangles + segment scan", "U updates, threaded", "U updates, one thread"};
            fprintf(stderr, "[plan] %d threads, %ld supernodes:\n", omp_get_max_threads(), (long)pl->nsn);
            for (int k = 0; k < 7; k++) {
                fprintf(stderr, "[plan]   %-28s %8.4f s  %10.3e entries\n", nm[k], plan_t[k], plan_e[k]);
                plan_t[k] = 0; plan_e[k] = 0;
            }
        }
#endif
        return FPLUS_OK;
    }
    void *part = malloc((size_t)(pl->block * nrhs) * vs);
    if (!part) return FPLUS_ERR_NOMEM;
    const int cj = trans == 2;
    if (is_complex) { fwdUt_z(pl, (const double complex *)Ux, (double complex *)x, nrhs, ldx, cj, (double complex *)part);
                      bwdLt_z(pl, Lp, Li, (const double complex *)Lx, (double complex *)x, nrhs, ldx, cj, (double complex *)part); }
    else            { fwdUt_d(pl, (const double *)Ux, (double *)x, nrhs, ldx, 0, (double *)part);
                      bwdLt_d(pl, Lp, Li, (const double *)Lx, (double *)x, nrhs, ldx, 0, (double *)part); }
    free(part);
    return FPLUS_OK;
}
