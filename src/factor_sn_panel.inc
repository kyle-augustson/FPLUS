/* factor_sn_panel.inc -- the supernode-panel kernel, instantiated by factor_sn.c for double complex (sn_factor_panel_z, with the
 * optional GPU zgemm) and double (sn_factor_panel_d).  Parameters: PANEL_NAME, PANEL_T, PANEL_IS_COMPLEX, PANEL_ABS, PANEL_TRSM,
 * PANEL_GEMM, PANEL_TRSV, PANEL_GEMV (pointer-alpha calling form), PANEL_GPU (0 / 1), PANEL_FLOPF (flops per multiply-add). */
static int PANEL_NAME(SnWork *W, const int64_t *Ap, const int64_t *Ai, const PANEL_T *Ax,
                             int verbose, int64_t P, GpuCtx *G)
{
    typedef PANEL_T T;
    const int64_t n = W->n;
    const T ONE = 1, ZERO = 0, MONE = -1;
    const int is_complex = PANEL_IS_COMPLEX;
    int64_t *pinv = W->pinv, *p = W->p, *rowpos = W->rowpos, *col_sn = W->col_sn;
    T *y = (T *)W->y;
    int64_t open = -1;
    int rc = -1;
    /* OpenMP regions only with more than one thread: an inactive region (if clause false) still costs a runtime call per
       supernode update and per L column, ~a third of the factor time on small matrices at one thread (2026-10-12) */
    int par = 0;
#ifdef _OPENMP
    par = omp_get_max_threads() > 1;
#endif
    T *X = (T *)calloc((size_t)P * (size_t)n, sizeof(T));
    double *colmax = (double *)malloc((size_t)P * sizeof(double));
    int64_t *ext_off = (int64_t *)malloc((size_t)(P + 1) * sizeof(int64_t));
    int64_t *np_off = (int64_t *)malloc((size_t)(P + 1) * sizeof(int64_t));
    int64_t *uidx = (int64_t *)malloc((size_t)n * sizeof(int64_t));   /* supernode -> union position, stamped by ustamp */
    int64_t *ustamp = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    I64V ext_s = {0}, ext_kf = {0}, np = {0}, un = {0}, ucnt = {0}, ucol = {0}, ukf = {0};
    T *XS = NULL, *YB = NULL; int64_t capXS = 0, capYB = 0;
    if (!X || !colmax || !ext_off || !np_off || !uidx || !ustamp) goto out;
    for (int64_t j0 = 0; j0 < n; j0 += P) {
        const int64_t j1 = (j0 + P < n) ? j0 + P : n, np_ = j1 - j0;
        const int64_t open0 = open, we_open = open >= 0 ? W->sn[open].ncols : 0;
        ext_s.n = ext_kf.n = np.n = 0;
        /* (1) per-column search over the pre-panel supernodes, scatter */
        for (int64_t ci = 0; ci < np_; ci++) {
            const int64_t c = j0 + ci, stamp = c + 1;
            int64_t top = n, n_nonpiv = 0;
            T *x = X + (size_t)ci * n;
            for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) {
                int64_t i = Ai[q], k = pinv[i];
                if (k < 0) {
                    if (W->wrow[i] != stamp) { W->wrow[i] = stamp; W->nonpiv[n_nonpiv++] = i; }
                    continue;
                }
                int64_t sn = col_sn[k];
                if (W->wsn[sn] == stamp) { if (k < W->segfirst[sn]) W->segfirst[sn] = k; continue; }
                top = sn_dfs(sn, k, top, W, stamp, open, &n_nonpiv);
            }
            double cm = 0.0;
            for (int64_t q = Ap[c]; q < Ap[c + 1]; q++) { x[Ai[q]] += Ax[q]; double a = PANEL_ABS(Ax[q]); if (a > cm) cm = a; }
            colmax[ci] = cm;
            ext_off[ci] = ext_s.n; np_off[ci] = np.n;
            for (int64_t t = top; t < n; t++) {
                if (i64v_push(&ext_s, W->xi[t]) || i64v_push(&ext_kf, W->segfirst[W->xi[t]])) goto out;
            }
            for (int64_t t = 0; t < n_nonpiv; t++) if (i64v_push(&np, W->nonpiv[t])) goto out;
        }
        ext_off[np_] = ext_s.n; np_off[np_] = np.n;
        /* (2) union order and the batched updates */
        {
            const int64_t us = 2 * n + 1 + j0;
            int64_t top = n, dummy = 0;
            for (int64_t q = 0; q < ext_s.n; q++) {
                int64_t sn = ext_s.a[q];
                if (W->wsn[sn] == us) continue;
                top = sn_dfs(sn, ext_kf.a[q], top, W, us, open, &dummy);
                dummy = 0;
            }
            un.n = 0;
            for (int64_t t = top; t < n; t++) { if (i64v_push(&un, W->xi[t])) goto out; uidx[W->xi[t]] = un.n - 1; ustamp[W->xi[t]] = us; }
            /* columns per union entry (counting sort) */
            ucnt.n = 0;
            for (int64_t u = 0; u <= un.n; u++) if (i64v_push(&ucnt, 0)) goto out;
            for (int64_t q = 0; q < ext_s.n; q++) ucnt.a[uidx[ext_s.a[q]] + 1]++;
            for (int64_t u = 0; u < un.n; u++) ucnt.a[u + 1] += ucnt.a[u];
            ucol.n = ukf.n = 0;
            for (int64_t q = 0; q < ext_s.n; q++) { if (i64v_push(&ucol, 0) || i64v_push(&ukf, 0)) goto out; }
            {
                int64_t *fill = (int64_t *)malloc((size_t)(un.n + 1) * sizeof(int64_t));
                if (!fill) goto out;
                memcpy(fill, ucnt.a, (size_t)(un.n + 1) * sizeof(int64_t));
                for (int64_t ci = 0; ci < np_; ci++)
                    for (int64_t q = ext_off[ci]; q < ext_off[ci + 1]; q++) {
                        int64_t u = uidx[ext_s.a[q]], pos = fill[u]++;
                        ucol.a[pos] = ci; ukf.a[pos] = ext_kf.a[q];
                    }
                free(fill);
            }
#ifdef FPLUS_PROFILE
            double _tb = wall_time();
#endif
            for (int64_t u = 0; u < un.n; u++) {
                const int64_t sn = un.a[u], c0 = ucnt.a[u], c1 = ucnt.a[u + 1], pc = c1 - c0;
                if (pc == 0) continue;
                Supernode *S = &W->sn[sn];
                const int64_t r1 = S->nrows, ld = S->ld, we = (sn == open0) ? we_open : S->ncols;
                int64_t kfa = we;
                for (int64_t q = c0; q < c1; q++) { int64_t k = ukf.a[q] - S->first; if (k < kfa) kfa = k; }
                const int64_t m = we - kfa, nrest = r1 - we;
                if (m <= 0) continue;
                const T *B = (const T *)S->blk;
                const int64_t *rows = S->rows;
                if (capXS < m * pc) { free(XS); capXS = 2 * m * pc; XS = (T *)malloc((size_t)capXS * sizeof(T)); if (!XS) goto out; }
                if (capYB < nrest * pc) { free(YB); capYB = 2 * nrest * pc + 1; YB = (T *)malloc((size_t)capYB * sizeof(T)); if (!YB) goto out; }
                /* gather and scatter: each panel column has its own accumulator column, so the columns are independent */
#define PANEL_GATHER_LOOP                                                                                        \
                for (int64_t q = c0; q < c1; q++) {                                                                      \
                    const T *x = X + (size_t)ucol.a[q] * n; T *xs = XS + (q - c0) * m;                                   \
                    for (int64_t cc = 0; cc < m; cc++) xs[cc] = x[rows[kfa + cc]];                                       \
                }
                if (par && pc > 1 && pc * m >= SN_PAR_MIN) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    PANEL_GATHER_LOOP
                } else {
                    PANEL_GATHER_LOOP
                }
#undef PANEL_GATHER_LOOP
                PANEL_TRSM(102, 141, 122, 111, 132, (int)m, (int)pc, &ONE, B + kfa * ld + kfa, (int)ld, XS, (int)m);
                if (nrest > 0) {
                    const double gfl = PANEL_FLOPF * (double)nrest * (double)pc * (double)m;
                    if (PANEL_GPU && G && G->ok && gfl >= G->min_flops) {
                        double tg = wall_time();
                        int st = G->zgemm(G->handle, 0, 0, (int)nrest, (int)pc, (int)m, &ONE, B + kfa * ld + we, (int)ld, XS, (int)m, &ZERO, YB, (int)nrest);
                        int se = st == 0 ? G->sync() : 0;
                        if (st != 0 || se != 0) {
                            if (st != 0) fplus_errf(W->err, W->errlen, "cuBLAS zgemm failed (status %d)", st);
                            else fplus_errf(W->err, W->errlen, "CUDA device synchronize failed (error %d)", se);
                            rc = FPLUS_ERR_GPU; goto out;
                        }
                        G->t += wall_time() - tg; G->flops += gfl; G->calls++;
                    } else
                        PANEL_GEMM(102, 111, 111, (int)nrest, (int)pc, (int)m, &ONE, B + kfa * ld + we, (int)ld, XS, (int)m, &ZERO, YB, (int)nrest);
                }
#define PANEL_SCATTER_LOOP                                                                                       \
                for (int64_t q = c0; q < c1; q++) {                                                                      \
                    T *x = X + (size_t)ucol.a[q] * n; const T *xs = XS + (q - c0) * m, *yb = YB + (q - c0) * nrest;      \
                    for (int64_t cc = 0; cc < m; cc++) x[rows[kfa + cc]] = xs[cc];                                       \
                    for (int64_t r = 0; r < nrest; r++) x[rows[we + r]] -= yb[r];                                        \
                }
                if (par && pc > 1 && pc * (m + nrest) >= SN_PAR_MIN) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    PANEL_SCATTER_LOOP
                } else {
                    PANEL_SCATTER_LOOP
                }
#undef PANEL_SCATTER_LOOP
#ifdef FPLUS_PROFILE
                SNP.fl_blas += PANEL_FLOPF * pc * ((double)m * (m - 1) / 2 + (double)m * nrest); SNP.n_upd_blas++; SNP.n_upd++;
                SNP.fl_w[ilog2_i64(S->ncols)] += PANEL_FLOPF * pc * ((double)m * (m - 1) / 2 + (double)m * nrest);
#endif
            }
#ifdef FPLUS_PROFILE
            SNP.t[2] += wall_time() - _tb;
#endif
        }
        (void)MONE;
        /* (3) per column: within-panel updates, pivot, U, L */
        for (int64_t ci = 0; ci < np_; ci++) {
            const int64_t j = j0 + ci, stamp2 = n + 1 + j;
            T *x = X + (size_t)ci * n;
#ifdef FPLUS_PROFILE
            double _t0 = wall_time();
#endif
            int64_t n_nonpiv = 0, top = n;
            for (int64_t q = np_off[ci]; q < np_off[ci + 1]; q++) { int64_t i = np.a[q]; W->nonpiv[n_nonpiv++] = i; W->wrow[i] = stamp2; }
            const int64_t n_np0 = n_nonpiv;
            /* the supernode open at the panel start: a column that reached its pre-panel columns reaches its whole tail, as in
               the column-at-a-time kernel -- including the columns it gained inside the panel and the rows those relaxed joins
               appended after this column's pre-panel search.  Without this, such rows can be missing from the column's pattern,
               and pruning (which assumes the pruning column holds every non-pivotal row of the pruned supernode) then loses them
               for good: missing fill, a wrong factorization (relax_rows > 0, panel > 1; found on the scipy-splu COLAMD ordering). */
            if (open0 >= 0 && W->sn[open0].ncols > we_open && W->wsn[open0] != stamp2) {
                for (int64_t q = ext_off[ci]; q < ext_off[ci + 1]; q++)
                    if (ext_s.a[q] == open0) { top = sn_dfs(open0, W->sn[open0].first + we_open, top, W, stamp2, open, &n_nonpiv); break; }
            }
            for (int64_t q = 0; q < n_np0; q++) {
                int64_t i = W->nonpiv[q], k = pinv[i];
                if (k < 0) continue;                        /* still non-pivotal */
                int64_t sn = col_sn[k];
                if (W->wsn[sn] == stamp2) { if (k < W->segfirst[sn]) W->segfirst[sn] = k; continue; }
                top = sn_dfs(sn, k, top, W, stamp2, open, &n_nonpiv);
            }
#ifdef FPLUS_PROFILE
            double _t1 = wall_time(); SNP.t[0] += _t1 - _t0;
#endif
            for (int64_t t = top; t < n; t++) {
                Supernode *S = &W->sn[W->xi[t]];
                const int64_t r1 = S->nrows, ld = S->ld, w = S->ncols, kf = W->segfirst[W->xi[t]] - S->first;
                const T *B = (const T *)S->blk;
                const int64_t *rows = S->rows;
                int64_t nrest = r1 - w, m = w - kf;
                const int64_t *rest = rows + w;
                if (m * nrest >= SN_BLAS_MIN) {
                    T *xs = y + nrest;
                    for (int64_t cc = kf; cc < w; cc++) xs[cc - kf] = x[rows[cc]];
                    PANEL_TRSV(102, 122, 111, 132, (int)m, B + kf * ld + kf, (int)ld, xs, 1);
                    PANEL_GEMV(102, 111, (int)nrest, (int)m, &ONE, B + kf * ld + w, (int)ld, xs, 1, &ZERO, y, 1);
                    for (int64_t cc = kf; cc < w; cc++) x[rows[cc]] = xs[cc - kf];
                } else {
                    for (int64_t r = 0; r < nrest; r++) y[r] = 0;
                    for (int64_t cc = kf; cc < w; cc++) {
                        T xc = x[rows[cc]];
                        if (xc == 0) continue;
                        const T *Bc = B + cc * ld;
                        for (int64_t rr = cc + 1; rr < w; rr++) x[rows[rr]] -= xc * Bc[rr];
                        const T *Br = Bc + w;
                        for (int64_t r = 0; r < nrest; r++) y[r] += Br[r] * xc;
                    }
                }
                for (int64_t r = 0; r < nrest; r++) x[rest[r]] -= y[r];
#ifdef FPLUS_PROFILE
                SNP.fl_loop += PANEL_FLOPF * ((double)m * (m - 1) / 2 + (double)m * nrest); SNP.n_upd++;
#endif
            }
#ifdef FPLUS_PROFILE
            double _t2 = wall_time(); SNP.t[3] += _t2 - _t1;
#endif
            /* the rows written: clear list = every pattern row (pivotal or not) */
            const int64_t n_all = n_nonpiv;
            /* pivot among the rows not yet pivotal */
            int64_t pivrow = -1; double best = -1.0;
            for (int64_t q = 0; q < n_all; q++) {
                int64_t i = W->nonpiv[q];
                if (pinv[i] >= 0) continue;
                double a = PANEL_ABS(x[i]);
                if (a > best) { best = a; pivrow = i; }
            }
            double pivot_threshold = W->pivot_rel * colmax[ci];
            if (pivrow < 0 || best <= 0.0 || best < pivot_threshold) {
                char _msg[256];
                snprintf(_msg, sizeof(_msg), "Near-zero or zero pivot %.6e at step %ld (threshold %.6e). Matrix is singular.",
                         best > 0 ? best : 0.0, (long)j, pivot_threshold);
                fplus_errf(W->err, W->errlen, "%s", _msg);
                rc = -2; goto out;
            }
            T piv = x[pivrow];
#ifdef FPLUS_PROFILE
            double _t3 = wall_time(); SNP.t[4] += _t3 - _t2;
#endif
            /* U[:, j]: the pre-panel segments (ending at the panel-start width), the within-panel segments, the diagonal */
            {
                int64_t need = 1;
                for (int64_t q = ext_off[ci]; q < ext_off[ci + 1]; q++) {
                    int64_t sn = ext_s.a[q]; Supernode *S = &W->sn[sn];
                    int64_t we = (sn == open0) ? we_open : S->ncols;
                    need += S->first + we - ext_kf.a[q];
                }
                for (int64_t t = top; t < n; t++) need += W->sn[W->xi[t]].first + W->sn[W->xi[t]].ncols - W->segfirst[W->xi[t]];
                if (gp_grow(&W->Ui, &W->Ux, &W->capU, W->nnzU + need, is_complex)) goto out;
                int64_t *Ui = W->Ui; T *Ux = (T *)W->Ux;
                int64_t u = W->nnzU;
                for (int64_t q = ext_off[ci]; q < ext_off[ci + 1]; q++) {
                    int64_t sn = ext_s.a[q]; Supernode *S = &W->sn[sn];
                    int64_t we = (sn == open0) ? we_open : S->ncols;
                    for (int64_t cc = ext_kf.a[q] - S->first; cc < we; cc++) { Ui[u] = S->first + cc; Ux[u] = x[S->rows[cc]]; u++; }
                }
                for (int64_t t = top; t < n; t++) {
                    Supernode *S = &W->sn[W->xi[t]];
                    for (int64_t cc = W->segfirst[W->xi[t]] - S->first; cc < S->ncols; cc++) { Ui[u] = S->first + cc; Ux[u] = x[S->rows[cc]]; u++; }
                }
                Ui[u] = j; Ux[u] = piv; u++;
                W->nnzU = u; W->Up[j + 1] = u;
            }
#ifdef FPLUS_PROFILE
            double _t4 = wall_time(); SNP.t[5] += _t4 - _t3;
#endif
            /* the non-pivotal rows only, for the L logic */
            int64_t n_nonpiv_l = 0;
            for (int64_t q = 0; q < n_all; q++) { int64_t i = W->nonpiv[q]; if (pinv[i] < 0) W->miss[n_nonpiv_l++] = i; }
            /* (W->miss doubles as the list of non-pivotal rows here; the relaxed join below takes its missing rows from W->nonpiv) */
            /* L[:, j]: join the open supernode when the pattern nests (or nearly: relaxed), else open one */
            int join = 0; int64_t n_miss = 0;
            int64_t *npl = (int64_t *)malloc((size_t)(n_nonpiv_l > 0 ? n_nonpiv_l : 1) * sizeof(int64_t));
            if (!npl) goto out;
            memcpy(npl, W->miss, (size_t)n_nonpiv_l * sizeof(int64_t));
            if (open >= 0) {
                Supernode *S = &W->sn[open];
                int64_t in_S = 0;
                for (int64_t q = 0; q < n_nonpiv_l; q++) {
                    if (rowpos[npl[q]] >= S->ncols) in_S++;
                    else W->miss[n_miss++] = npl[q];
                }
                int64_t extra = (S->nrows - S->ncols) - in_S;
                if (n_miss + extra == 0) join = 1;
                else if (sn_relax_ok(W, S, n_miss, extra)) join = 2;
                if (join == 2) {
                    S->zeros += (double)n_miss * (double)S->ncols + (double)extra;
                    if (n_miss > 0 && sn_add_rows(S, W->miss, n_miss, rowpos, is_complex)) { free(npl); goto out; }
                    PROF_RELAX(n_miss * S->ncols + extra);
                }
            }
            if (join) {
                Supernode *S = &W->sn[open];
                const int64_t r1 = S->nrows, ld = S->ld, w = S->ncols;
                if (sn_grow_cols(S, w + 1, is_complex)) { free(npl); goto out; }
                T *B = (T *)S->blk;
                int64_t pp = rowpos[pivrow];
                if (pp != w) {
                    int64_t other = S->rows[w];
                    S->rows[w] = pivrow; S->rows[pp] = other;
                    rowpos[pivrow] = w; rowpos[other] = pp;
                    for (int64_t cc = 0; cc < w; cc++) { T tv = B[cc * ld + w]; B[cc * ld + w] = B[cc * ld + pp]; B[cc * ld + pp] = tv; }
                }
                T *Bc = B + w * ld;
                for (int64_t rr = 0; rr < w; rr++) Bc[rr] = x[S->rows[rr]];
                Bc[w] = 1;
                /* the L column: one division per row, independent rows */
                if (par && r1 - w - 1 >= SN_PAR_MIN) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int64_t rr = w + 1; rr < r1; rr++) Bc[rr] = x[S->rows[rr]] / piv;
                } else {
                    for (int64_t rr = w + 1; rr < r1; rr++) Bc[rr] = x[S->rows[rr]] / piv;
                }
                S->ncols = w + 1;
                col_sn[j] = open;
            } else {
                if (open >= 0) {
                    Supernode *S = &W->sn[open];
                    int64_t nrest = S->nrows - S->ncols;
                    S->dfs_rows = (int64_t *)malloc((size_t)(nrest > 0 ? nrest : 1) * sizeof(int64_t));
                    if (!S->dfs_rows) { free(npl); goto out; }
                    memcpy(S->dfs_rows, S->rows + S->ncols, (size_t)nrest * sizeof(int64_t));
                    for (int64_t rr = 0; rr < S->nrows; rr++) rowpos[S->rows[rr]] = -1;
                }
                int64_t sn = W->nsn++;
                Supernode *S = &W->sn[sn];
                S->first = j; S->ncols = 1; S->nrows = n_nonpiv_l; S->ld = n_nonpiv_l; S->cap = 4; S->zeros = 0;
                S->dfs_rows = NULL; S->dfs_end = -1;
                S->rows = (int64_t *)malloc((size_t)S->nrows * sizeof(int64_t));
                S->blk = malloc((size_t)S->cap * (size_t)S->ld * sizeof(T));
                if (!S->rows || !S->blk) { free(npl); goto out; }
                T *B = (T *)S->blk;
                S->rows[0] = pivrow; rowpos[pivrow] = 0; B[0] = 1;
                int64_t rr = 1;
                for (int64_t t = 0; t < n_nonpiv_l; t++) {
                    int64_t i = npl[t];
                    if (i == pivrow) continue;
                    S->rows[rr] = i; rowpos[i] = rr; B[rr] = x[i] / piv; rr++;
                }
                col_sn[j] = sn;
                open = sn;
            }
            free(npl);
            pinv[pivrow] = j; p[j] = pivrow;
#ifdef FPLUS_PROFILE
            double _t5 = wall_time(); SNP.t[6] += _t5 - _t4;
#endif
            /* clear the accumulator: every pattern row and every segment row */
            for (int64_t q = 0; q < n_all; q++) x[W->nonpiv[q]] = 0;
            for (int64_t q = ext_off[ci]; q < ext_off[ci + 1]; q++) {
                int64_t sn = ext_s.a[q]; Supernode *S = &W->sn[sn];
                int64_t we = (sn == open0) ? we_open : S->ncols;
                for (int64_t cc = ext_kf.a[q] - S->first; cc < we; cc++) x[S->rows[cc]] = 0;
            }
            for (int64_t t = top; t < n; t++) {
                Supernode *S = &W->sn[W->xi[t]];
                for (int64_t cc = W->segfirst[W->xi[t]] - S->first; cc < S->ncols; cc++) x[S->rows[cc]] = 0;
            }
            x[pivrow] = 0;
            /* prune the closed supernodes whose segments this column reached (pre-panel and within-panel) */
            for (int pass = 0; pass < 2; pass++) {
                int64_t qa = pass == 0 ? ext_off[ci] : top, qb = pass == 0 ? ext_off[ci + 1] : n;
                for (int64_t q = qa; q < qb; q++) {
                    int64_t sn = pass == 0 ? ext_s.a[q] : W->xi[q];
                    Supernode *S = &W->sn[sn];
                    if (sn == open || S->dfs_end >= 0 || !S->dfs_rows) continue;
                    int64_t nrest = S->nrows - S->ncols;
                    int64_t *rows = S->dfs_rows;
                    int found = 0;
                    for (int64_t r = 0; r < nrest; r++) if (rows[r] == pivrow) { found = 1; break; }
                    if (!found) continue;
                    int64_t ph = 0, pt = nrest;
                    while (ph < pt) {
                        if (pinv[rows[ph]] >= 0) { ph++; continue; }
                        pt--;
                        int64_t ti = rows[ph]; rows[ph] = rows[pt]; rows[pt] = ti;
                    }
                    S->dfs_end = pt;
                }
            }
#ifdef FPLUS_PROFILE
            SNP.t[7] += wall_time() - _t5;
#endif
        }
        if (verbose >= 2 && ((j0 / P) % (n / (20 * P) + 1) == 0))
            printf("[FPLUS]   [%5.1f%%] column %ld/%ld  supernodes=%ld (panel %ld)\n", 100.0 * (double)j1 / (double)n, (long)j1, (long)n, (long)W->nsn, (long)P);
    }
    rc = 0;
out:
    free(X); free(colmax); free(ext_off); free(np_off); free(uidx); free(ustamp);
    free(ext_s.a); free(ext_kf.a); free(np.a); free(un.a); free(ucnt.a); free(ucol.a); free(ukf.a); free(XS); free(YB);
    return rc;
}
