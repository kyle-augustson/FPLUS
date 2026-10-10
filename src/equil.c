/* equil.c -- Ruiz L-inf equilibration (CSR or CSC). */
#include "fplus_internal.h"

/* =========================================================================
 * Ruiz iterative L-infinity equilibration
 * =========================================================================
 *
 * Finds real diagonal scaling vectors Dr (length n) and Dc (length n) such
 * that the equilibrated matrix  B = diag(Dr) * A * diag(Dc)  has all row
 * and column L-inf norms close to 1.
 *
 * Algorithm (Ruiz 2001, "A scaling algorithm to equilibrate both rows and
 * columns norms in matrices"):
 *
 *   Repeat until convergence:
 *     For each row i:    Dr[i] *= 1 / sqrt(max_j |B[i,j]|)
 *     For each col j:    Dc[j] *= 1 / sqrt(max_i |B[i,j]|)
 *     Update B accordingly.
 *
 * Convergence criterion: max(|log(row_max)|, |log(col_max)|) < tol,
 * i.e. all row and column maxima are within factor e^tol of 1.
 *
 * Properties:
 *   - Works identically for real and complex matrices (uses |a_ij|).
 *   - Dr and Dc are always real even when A is complex.
 *   - Typically converges in 5-15 iterations.
 *   - Cost: O(nnz + n) per iteration; total O(10 * nnz) — negligible
 *     compared to the O(n^2) col_swap cost of the factorisation.
 *   - Zero rows/columns indicate structural singularity and are flagged.
 *
 * Why this is worth doing:
 *   Even though full pivoting bounds the growth factor, the pivot selection
 *   criterion (choose largest |entry|) is scale-dependent.  A row with all
 *   entries 1e-10 will never contribute a pivot even if it is numerically
 *   important.  After equilibration all rows and columns are comparable in
 *   magnitude, so the pivot search reflects genuine numerical importance
 *   rather than accidental scaling.  The condition number of B is also
 *   minimised (or near-minimised) by this procedure, which directly reduces
 *   the rounding errors in the triangular solve step.
 *
 * Integration with the factorisation:
 *   sparse_lu_full_pivot() calls ruiz_equilibrate() when equilibrate=True,
 *   passes the scaled matrix to run_factorisation(), and stores Dr and Dc
 *   in the returned object.  lu_solve() automatically applies:
 *       b_scaled = Dr * b
 *       y        = U \ (L \ b_scaled)      [triangular solves]
 *       x        = Dc * y
 *
 * Python call:
 *   ruiz_equilibrate(data, indices, indptr, n, max_iter, tol)
 *   -> dict {"Dr": float64[n], "Dc": float64[n], "data": dtype[nnz]}
 *
 * The returned "data" array is the scaled CSR data (indices/indptr unchanged).
 * ========================================================================= */
/* One Ruiz iteration on CSC arrays: the arithmetic of the CSR loop in py_ruiz_equilibrate (row maxima of the squared
 * moduli, row scaling by 1/sqrt(row max), column maxima of the scaled entries, column scaling), so Dr, Dc and the
 * scaled data agree bit for bit with the CSR route on the same matrix (maxima do not depend on traversal order and
 * the scalings are elementwise).  Column-oriented passes run in parallel.  row_max / col_max are workspace.
 * Returns 0, or 1 + i for an all-zero row i, or -(1 + j) for an all-zero column j. */
static int64_t ruiz_csc_iter(void *data, const int64_t *indices, const int64_t *indptr, int64_t n, int is_complex,
                             double *Dr, double *Dc, double *row_max, double *col_max, double *res_out)
{
    for (int64_t i = 0; i < n; i++) row_max[i] = 0.0;
    if (is_complex) {
        const double complex *zd = (const double complex *)data;
        for (int64_t j = 0; j < n; j++)
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) {
                double re = creal(zd[p]), im = cimag(zd[p]);
                double av = re * re + im * im;
                if (av > row_max[indices[p]]) row_max[indices[p]] = av;
            }
    } else {
        const double *dd = (const double *)data;
        for (int64_t j = 0; j < n; j++)
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) {
                double av = dd[p] * dd[p];
                if (av > row_max[indices[p]]) row_max[indices[p]] = av;
            }
    }
    for (int64_t i = 0; i < n; i++)
        if (row_max[i] == 0.0) return 1 + i;
    double res = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double v = 0.5 * fabs(log(row_max[i]));
        if (v > res) res = v;
    }
    /* row_max becomes the row scale */
    for (int64_t i = 0; i < n; i++) {
        double dr = 1.0 / sqrt(sqrt(row_max[i]));
        Dr[i] *= dr;
        row_max[i] = dr;
    }
    if (is_complex) {
        double complex *zd = (double complex *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n; j++) {
            double cm = 0.0;
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) {
                zd[p] *= row_max[indices[p]];
                double re = creal(zd[p]), im = cimag(zd[p]);
                double av = re * re + im * im;
                if (av > cm) cm = av;
            }
            col_max[j] = cm;
        }
    } else {
        double *dd = (double *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n; j++) {
            double cm = 0.0;
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) {
                dd[p] *= row_max[indices[p]];
                double av = dd[p] * dd[p];
                if (av > cm) cm = av;
            }
            col_max[j] = cm;
        }
    }
    for (int64_t j = 0; j < n; j++)
        if (col_max[j] == 0.0) return -(1 + j);
    for (int64_t j = 0; j < n; j++) {
        double v = 0.5 * fabs(log(col_max[j]));
        if (v > res) res = v;
        col_max[j] = 1.0 / sqrt(sqrt(col_max[j]));
        Dc[j] *= col_max[j];
    }
    if (is_complex) {
        double complex *zd = (double complex *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n; j++)
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) zd[p] *= col_max[j];
    } else {
        double *dd = (double *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n; j++)
            for (int64_t p = indptr[j]; p < indptr[j+1]; p++) dd[p] *= col_max[j];
    }
    *res_out = res;
    return 0;
}


int fplus_ruiz(int64_t n, const int64_t *indptr, const int64_t *indices, void *data, int is_complex, int is_csc,
               int max_iter, double tol, int verbose, double *Dr, double *Dc, char *err, size_t errlen)
{
    int64_t nnz = indptr[n];
    double *row_max = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *col_max = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!row_max || !col_max) {
        free(row_max); free(col_max);
        fplus_errf(err, errlen, "out of memory");
        return FPLUS_ERR_NOMEM;
    }
    for (int64_t i = 0; i < n; i++) { Dr[i] = 1.0; Dc[i] = 1.0; }

    /* Verbose-mode timing and summary tracking. */
    double t_ruiz_start = (verbose >= 1) ? wall_time() : 0.0;
    int    iters_used   = 0;
    double final_res    = 0.0;
    if (verbose >= 1) {
        printf("[FPLUS] Ruiz equilibration start: n=%ld, nnz=%ld, "
               "max_iter=%d, tol=%.1e\n",
               (long)n, (long)nnz, max_iter, tol);
        fflush(stdout);
    }

    /* ------------------------------------------------------------------ */
    /* Main Ruiz loop: per iteration a row pass (the row L-inf maxima), a  */
    /* row-scaling pass that also gathers the column maxima of the scaled */
    /* entries, and a column-scaling pass.  Maxima are taken on squared    */
    /* moduli, one square root per line.  The residual is the deviation    */
    /* the iteration corrects, max |log| of the row and column maxima it   */
    /* found; it halves per iteration.                                     */
    /* ------------------------------------------------------------------ */
    for (int iter = 0; iter < max_iter; iter++) {
        double res = 0.0;
        if (is_csc) {
            int64_t bad = ruiz_csc_iter(data, indices, indptr, n, is_complex, Dr, Dc, row_max, col_max, &res);
            if (bad != 0) {
                free(row_max); free(col_max);
                if (bad > 0)
                    fplus_errf(err, errlen,
                        "Row %ld of the matrix is entirely zero (structural "
                        "singularity detected during equilibration).", (long)(bad - 1));
                else
                    fplus_errf(err, errlen,
                        "Column %ld of the matrix is entirely zero (structural "
                        "singularity detected during equilibration).", (long)(-bad - 1));
                return FPLUS_ERR_SINGULAR;
            }
            goto ruiz_iter_done;
        }
        for (int64_t i = 0; i < n; i++) row_max[i] = 0.0;
        for (int64_t j = 0; j < n; j++) col_max[j] = 0.0;
        if (is_complex) {
            double complex *zd = (double complex *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < n; i++) {
                double rm = 0.0;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    double re = creal(zd[p]), im = cimag(zd[p]);
                    double av = re * re + im * im;
                    if (av > rm) rm = av;
                }
                row_max[i] = rm;
            }
        } else {
            double *dd = (double *)data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < n; i++) {
                double rm = 0.0;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    double av = dd[p] * dd[p];
                    if (av > rm) rm = av;
                }
                row_max[i] = rm;
            }
        }
        for (int64_t i = 0; i < n; i++) {
            if (row_max[i] == 0.0) {
                free(row_max); free(col_max);
                fplus_errf(err, errlen,
                    "Row %ld of the matrix is entirely zero (structural "
                    "singularity detected during equilibration).", (long)i);
                return FPLUS_ERR_SINGULAR;
            }
        }
        for (int64_t i = 0; i < n; i++) {
            double v = 0.5 * fabs(log(row_max[i]));   /* |log| of the modulus */
            if (v > res) res = v;
        }
        /* row scaling by 1/sqrt(row max), gathering the column maxima */
        if (is_complex) {
            double complex *zd = (double complex *)data;
            for (int64_t i = 0; i < n; i++) {
                double dr = 1.0 / sqrt(sqrt(row_max[i]));
                Dr[i] *= dr;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    zd[p] *= dr;
                    double re = creal(zd[p]), im = cimag(zd[p]);
                    double av = re * re + im * im;
                    int64_t j = indices[p];
                    if (av > col_max[j]) col_max[j] = av;
                }
            }
        } else {
            double *dd = (double *)data;
            for (int64_t i = 0; i < n; i++) {
                double dr = 1.0 / sqrt(sqrt(row_max[i]));
                Dr[i] *= dr;
                for (int64_t p = indptr[i]; p < indptr[i+1]; p++) {
                    dd[p] *= dr;
                    double av = dd[p] * dd[p];
                    int64_t j = indices[p];
                    if (av > col_max[j]) col_max[j] = av;
                }
            }
        }
        for (int64_t j = 0; j < n; j++) {
            if (col_max[j] == 0.0) {
                free(row_max); free(col_max);
                fplus_errf(err, errlen,
                    "Column %ld of the matrix is entirely zero (structural "
                    "singularity detected during equilibration).", (long)j);
                return FPLUS_ERR_SINGULAR;
            }
        }
        /* column scaling by 1/sqrt(column max); col_max becomes the scale */
        for (int64_t j = 0; j < n; j++) {
            double v = 0.5 * fabs(log(col_max[j]));
            if (v > res) res = v;
            col_max[j] = 1.0 / sqrt(sqrt(col_max[j]));
            Dc[j] *= col_max[j];
        }
        if (is_complex) {
            double complex *zd = (double complex *)data;
            for (int64_t p = 0; p < nnz; p++) zd[p] *= col_max[indices[p]];
        } else {
            double *dd = (double *)data;
            for (int64_t p = 0; p < nnz; p++) dd[p] *= col_max[indices[p]];
        }
ruiz_iter_done:
        iters_used = iter + 1;
        final_res  = res;
        if (verbose >= 2) {
            printf("[FPLUS]   iter %3d  residual=%.3e\n", iter + 1, res);
            fflush(stdout);
        }
        if (res < tol) break;
    } /* end Ruiz iterations */

    free(row_max); free(col_max);

    if (verbose >= 1) {
        /* Compute Dr/Dc range for the summary (informative for diagnosing
         * matrices with extreme scale separation).                       */
        double Dr_min = Dr[0], Dr_max = Dr[0];
        double Dc_min = Dc[0], Dc_max = Dc[0];
        for (int64_t i = 1; i < n; i++) {
            if (Dr[i] < Dr_min) Dr_min = Dr[i];
            if (Dr[i] > Dr_max) Dr_max = Dr[i];
            if (Dc[i] < Dc_min) Dc_min = Dc[i];
            if (Dc[i] > Dc_max) Dc_max = Dc[i];
        }
        double t_elapsed = wall_time() - t_ruiz_start;
        const char *status = (final_res < tol) ? "converged" : "max_iter reached";
        printf("[FPLUS] Ruiz %s: %d iters, final res=%.3e, "
               "Dr in [%.2e, %.2e], Dc in [%.2e, %.2e], time=%.3fs\n",
               status, iters_used, final_res,
               Dr_min, Dr_max, Dc_min, Dc_max, t_elapsed);
        fflush(stdout);
    }

    return FPLUS_OK;
}
