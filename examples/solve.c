/* solve.c -- the FPLUS C interface in a few lines: factor a small complex sparse matrix once, solve with it, and solve the
 * conjugate-transposed system with the same factors.
 *
 *     cmake -S . -B build && cmake --build build && ./build/fplus_example
 */
#include <complex.h>
#include <stdio.h>
#include "fplus.h"

int main(void)
{
    /*      | 4    1    0  |
     *  A = | 1i   3    1  |   in canonical CSC (columns, rows ascending in each)
     *      | 0    2    5  |                                                       */
    int64_t n = 3;
    int64_t Ap[] = {0, 2, 5, 7};
    int64_t Ai[] = {0, 1, 0, 1, 2, 1, 2};
    double complex Ax[] = {4, 1 * I, 1, 3, 2, 1, 5};
    char err[256];

    fplus_options opt;
    fplus_default_options(&opt);                     /* production defaults: relax 32, panel 512, COLAMD, postorder, Ruiz */

    fplus_symbolic *sym = NULL;                      /* the ordering: reusable for any values on this pattern */
    if (fplus_analyze_opts(n, Ap, Ai, &opt, &sym, err, sizeof(err)) != FPLUS_OK) { printf("analyze: %s\n", err); return 1; }

    fplus_numeric *num = NULL;
    if (fplus_factor(sym, Ap, Ai, Ax, 1, &opt, &num, err, sizeof(err)) != FPLUS_OK) { printf("factor: %s\n", err); return 1; }

    double complex b[] = {5, 4 + 1 * I, 7};          /* A x = b has the solution x = (1, 1, 1) */
    if (fplus_solve(num, b, 1, n, 0, err, sizeof(err)) != FPLUS_OK) { printf("solve: %s\n", err); return 1; }
    printf("A x = b:    x = (%.3f%+.3fi, %.3f%+.3fi, %.3f%+.3fi)\n", creal(b[0]), cimag(b[0]), creal(b[1]), cimag(b[1]), creal(b[2]), cimag(b[2]));

    double complex c[] = {4 - 1 * I, 6, 6};          /* A^H y = c has the solution y = (1, 1, 1) */
    if (fplus_solve(num, c, 1, n, 2, err, sizeof(err)) != FPLUS_OK) { printf("solve: %s\n", err); return 1; }
    printf("A^H y = c:  y = (%.3f%+.3fi, %.3f%+.3fi, %.3f%+.3fi)\n", creal(c[0]), cimag(c[0]), creal(c[1]), cimag(c[1]), creal(c[2]), cimag(c[2]));

    fplus_numeric_free(num);
    fplus_symbolic_free(sym);
    return 0;
}
