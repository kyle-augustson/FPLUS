/* solve.c -- triangular solves on CSC factors. */
#include "fplus_internal.h"

/* =========================================================================
 * Triangular solves on the CSC factors: x = U^-1 L^-1 b (trans 0),
 * x = L^-T U^-T b (trans 1) or x = L^-H U^-H b (trans 2), in place on a
 * copy of b.  L is unit lower (an explicit unit diagonal is skipped), U is
 * upper with its diagonal stored.
 * ========================================================================= */

#define TRI_SOLVE_BODY(NAME, T, CONJ)                                                       \
static int NAME(int64_t n, const int64_t *Lp, const int64_t *Li, const T *Lx,              \
                const int64_t *Up, const int64_t *Ui, const T *Ux, T *x, int trans)         \
{                                                                                           \
    if (trans == 0) {                                                                       \
        for (int64_t j = 0; j < n; j++) {                                                   \
            T xj = x[j];                                                                    \
            if (xj == 0) continue;                                                          \
            for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) {                                   \
                int64_t i = Li[q];                                                          \
                if (i != j) x[i] -= xj * Lx[q];                                             \
            }                                                                               \
        }                                                                                   \
        for (int64_t j = n - 1; j >= 0; j--) {                                              \
            T d = 0; int have = 0;                                                          \
            for (int64_t q = Up[j]; q < Up[j + 1]; q++)                                     \
                if (Ui[q] == j) { d = Ux[q]; have = 1; break; }                             \
            if (!have || d == 0) return -1;                                                 \
            T xj = x[j] / d;                                                                \
            x[j] = xj;                                                                      \
            if (xj == 0) continue;                                                          \
            for (int64_t q = Up[j]; q < Up[j + 1]; q++) {                                   \
                int64_t i = Ui[q];                                                          \
                if (i < j) x[i] -= xj * Ux[q];                                              \
            }                                                                               \
        }                                                                                   \
        return 0;                                                                           \
    }                                                                                       \
    int cj = (trans == 2);                                                                  \
    for (int64_t j = 0; j < n; j++) {                                                       \
        T s = x[j], d = 0; int have = 0;                                                    \
        for (int64_t q = Up[j]; q < Up[j + 1]; q++) {                                       \
            int64_t i = Ui[q];                                                              \
            T v = cj ? CONJ(Ux[q]) : Ux[q];                                                 \
            if (i < j) s -= v * x[i];                                                       \
            else if (i == j) { d = v; have = 1; }                                           \
        }                                                                                   \
        if (!have || d == 0) return -1;                                                     \
        x[j] = s / d;                                                                       \
    }                                                                                       \
    for (int64_t j = n - 1; j >= 0; j--) {                                                  \
        T s = x[j];                                                                         \
        for (int64_t q = Lp[j]; q < Lp[j + 1]; q++) {                                       \
            int64_t i = Li[q];                                                              \
            if (i > j) s -= (cj ? CONJ(Lx[q]) : Lx[q]) * x[i];                              \
        }                                                                                   \
        x[j] = s;                                                                           \
    }                                                                                       \
    return 0;                                                                               \
}

static inline double conj_d(double v) { return v; }
TRI_SOLVE_BODY(tri_solve_z, double complex, conj)
TRI_SOLVE_BODY(tri_solve_d, double, conj_d)

int fplus_tri_solve(int64_t n, const int64_t *Lp, const int64_t *Li, const void *Lx, const int64_t *Up, const int64_t *Ui,
                    const void *Ux, void *x, int trans, int is_complex)
{
    if (trans < 0 || trans > 2 || n < 0) return FPLUS_ERR_BADARG;
    int rc = is_complex
        ? tri_solve_z(n, Lp, Li, (const double complex *)Lx, Up, Ui, (const double complex *)Ux, (double complex *)x, trans)
        : tri_solve_d(n, Lp, Li, (const double *)Lx, Up, Ui, (const double *)Ux, (double *)x, trans);
    return rc == 0 ? FPLUS_OK : FPLUS_ERR_SINGULAR;
}
