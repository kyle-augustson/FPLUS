# FPLUS — Full-Pivot LU with Scaling

A sparse LU factorization library with selectable pivoting (partial / rook /
full), Ruiz equilibration, and direct-COLAMD fill-reducing ordering, designed
for ill-conditioned shift-invert eigenvalue problems where SuperLU's threshold
partial pivoting discards physically important small entries.

Key features:
- **int64 indices throughout** (no 2^31 ceiling on total nonzeros)
- **Three pivot strategies**: `partial` (default, preserves COLAMD ordering),
  `rook` (alternates row/column scans), `full` (scans entire active submatrix)
- **Ruiz L-inf equilibration** before factorization (handles 17+ orders of
  magnitude in entry scale)
- **Direct COLAMD via SuiteSparse** (no SuperLU dependency, int64-safe)
- **Tau row/column pinning**: dense boundary-condition lines moved to edges
  before COLAMD analysis
- **OpenMP-parallelized C extension** (build with `SPARSE_LU_OPENMP=1`)
- **Optional BLAS axpy** for dense rows (build with `SPARSE_LU_BLAS=1`)
- **Drop-in interface** for `scipy.sparse.linalg.splu` via the `SparseLU` class

## The C library

The factorization is a C library (`include/fplus.h`, `src/`) that the Python
module wraps.  It builds with CMake and needs SuiteSparse COLAMD, a BLAS with
the CBLAS interface (optional), and OpenMP (optional):

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX    # or wherever SuiteSparse lives
cmake --build build -j 4
ctest --test-dir build                                    # the C tests
./build/fplus_example                                     # examples/solve.c
cmake --install build --prefix /some/prefix               # libfplus.so + fplus.h
```

CMake options:

| option | default | |
|---|---|---|
| `FPLUS_OPENMP` | `ON` | OpenMP parallel loops |
| `FPLUS_BLAS` | `ON` | BLAS kernels (CBLAS interface) |
| `FPLUS_BLAS_LIBRARIES` | search | the BLAS link line, e.g. `-lopenblas` or a library path |
| `FPLUS_MARCH` | `native` | `-march` value; set `x86-64-v3` (or empty) for a build that must run on other CPUs |
| `FPLUS_PROFILE` | `OFF` | phase timers in the supernodal kernel |
| `FPLUS_BUILD_TESTS`, `FPLUS_BUILD_EXAMPLES` | `ON` | |

The high-level interface: options, then analyze (the ordering, reusable for
any values on the same pattern), factor, and solve with `A`, `A^T` or `A^H`:

```c
fplus_options opt;  fplus_default_options(&opt);   /* relax 32, panel 512, COLAMD, Ruiz */
fplus_symbolic *sym;  fplus_analyze_opts(n, Ap, Ai, &opt, &sym, err, sizeof err);
fplus_numeric  *num;  fplus_factor(sym, Ap, Ai, Ax, /*complex*/ 1, &opt, &num, err, sizeof err);
fplus_solve(num, b, nrhs, ldb, /*0: A, 1: A^T, 2: A^H*/ 0, err, sizeof err);
fplus_numeric_free(num);  fplus_symbolic_free(sym);
```

Matrices are canonical CSC (rows ascending in each column, no duplicates) with
int64 indices; values are `double` or `double complex`.  Every call returns
`FPLUS_OK` or an error code with a message in `err`.  The lower-level kernels
(the supernodal and row factorizations, Ruiz, the triangular solves) are in
the same header.

## Installing the Python package

```bash
pip install .            # scikit-build-core drives the CMake build
```

`CMAKE_PREFIX_PATH` (or `CMAKE_ARGS="-DFPLUS_BLAS_LIBRARIES=..."`) points the
build at SuiteSparse and BLAS as above.

## Building the C extension in place (setup.py)

From this directory (`dedalus/libraries/FPLUS/`):

```bash
# Single-threaded baseline:
python setup.py build_ext --inplace

# With OpenMP parallelism (recommended):
SPARSE_LU_OPENMP=1 python setup.py build_ext --inplace

# With OpenMP and BLAS axpy (helpful when factorization fill-in is large):
SPARSE_LU_OPENMP=1 SPARSE_LU_BLAS=1 python setup.py build_ext --inplace

# Override BLAS library (default tries -lopenblas):
SPARSE_LU_OPENMP=1 SPARSE_LU_BLAS=1 BLAS_LIBS="-lblas" \
    python setup.py build_ext --inplace

# Override BLAS-axpy threshold (default 32; rows with union >= 2*threshold
# use the BLAS path):
SPARSE_LU_OPENMP=1 SPARSE_LU_BLAS=1 \
    CFLAGS="-O3 -march=native -DBLAS_AXPY_THRESHOLD=64" \
    python setup.py build_ext --inplace
```

This produces `_sparse_lu_c.cpython-XYZ-*.so` in this directory. Once built,
the FPLUS solvers are automatically available in Dedalus via the matsolver
registry (no further configuration required).

If the C extension is not built, the library falls back to a pure-Python
implementation that is correct but much slower (and only supports
`pivot_strategy='full'`).

## Optional: install SuiteSparse COLAMD

The COLAMD-based variants prefer to call SuiteSparse's `colamd_l` directly
via ctypes for int64 support. On Ubuntu/Debian:

```bash
sudo apt-get install libsuitesparse-dev
```

If `libcolamd.so` is not available, the library falls back to extracting the
COLAMD ordering through scipy's SuperLU interface, which works but is subject
to int32 limits on the structural matrix size.

## Optional: install BLAS development headers

For the BLAS axpy path, install development headers for any BLAS library:

```bash
# Ubuntu/Debian, OpenBLAS (recommended for performance):
sudo apt-get install libopenblas-dev

# Or the reference (Netlib) BLAS:
sudo apt-get install libblas-dev

# Or Intel MKL via conda:
conda install -c conda-forge mkl-devel
```

The reference BLAS provides correct results but is not vectorized; OpenBLAS
or MKL provides AVX-512 vectorized axpy and shows a much larger speedup on
matrices with significant fill-in.

## Use from Python

### Direct use of `SparseLU` (drop-in for `scipy.sparse.linalg.splu`):

```python
from dedalus.libraries.FPLUS import SparseLU

# Default: partial pivoting + COLAMD + Ruiz equilibration
lu = SparseLU(A, colamd=True, pin_dense_rows=True)
x = lu.solve(b)
xT = lu.solve(b, trans='T')   # solve A^T x = b
xH = lu.solve(b, trans='H')   # solve A^H x = b

# Different pivot strategies:
lu = SparseLU(A, pivot_strategy='partial', colamd=True)  # default
lu = SparseLU(A, pivot_strategy='rook',    colamd=True)
lu = SparseLU(A, pivot_strategy='full',    colamd=True)  # full pivoting

# Verbose output (0=silent, 1=summaries, 2=per-step progress):
lu = SparseLU(A, colamd=True, verbose=1)
```

### Through the Dedalus matsolver interface:

```python
from dedalus.libraries.matsolvers import matsolvers

# Recommended for Dedalus EVPs (partial pivoting):
matsolver = matsolvers['fpluscolamdtaufactorized']

# Or explicitly select pivot strategy:
matsolver = matsolvers['fpluscolamdrookfactorized']
matsolver = matsolvers['fpluscolamdfullfactorized']

solver = matsolver(A)
x = solver.solve(b)
```

### As a Dedalus default (in `dedalus.cfg`):

```ini
[linear algebra]
# Default sparse matrix factorizer for repeated solves:
MATRIX_FACTORIZER = FPLUSColamdTauFactorized
# Optional FPLUS verbosity (0=silent, 1=summaries, 2=per-step progress):
VERBOSE_FACTORIZATION = 0
```

## Incomplete LU (preconditioner)

`drop_tol > 0` gives a threshold incomplete LU in place of the exact
factorization, with any pivot strategy, from Python
(`SparseLU(A, drop_tol=1e-3)`) or C (`opt.drop_tol = 1e-3`):

- on the equilibrated, ordered matrix, multipliers and updated or fill values
  with `|v| <= drop_tol` are dropped as they arise, and a pivot row's small
  off-pivot entries are dropped before it updates the rows below (ILUT's
  drop-before-use); pivots are never dropped.  After Ruiz equilibration the
  entries are O(1), so `drop_tol` is effectively relative;
- `solve` then applies `(LU)^-1`: use it as the preconditioner of an
  iterative solver (e.g. `scipy.sparse.linalg.gmres(A, b, M=LinearOperator(A.shape, lu.solve))`);
- if the exact factorization's trailing pivots are at or below `drop_tol`,
  dropping can leave no pivot; the pivot is then replaced by
  `drop_tol * max|a_ij|` (as SuperLU's ILU does), with a RuntimeWarning in
  Python and the count in `n_pivots_replaced` in C.  Choose `drop_tol` below
  the matrix's small pivots for a useful preconditioner.

## Pivot strategy guide

For typical Dedalus EVP problems, use **partial pivoting** (default). It
preserves the COLAMD column ordering through factorization, giving
fill-in comparable to scipy's `splu`. Combined with Ruiz equilibration,
it provides numerical robustness comparable to full pivoting at a tiny
fraction of the cost.

Use **rook pivoting** when partial pivoting fails to give acceptable accuracy
on a problem with strong asymmetric scale variation, but full pivoting is
too expensive.

Use **full pivoting** only when you specifically need the strongest
numerical guarantee and can afford the dramatic fill-in cost. Full pivoting's
column swaps destroy the COLAMD ordering and can produce factors orders of
magnitude denser than partial pivoting on banded matrices.

## Testing

```bash
python -m pytest test_sparse_lu_full_pivot.py test_audit_2026_10_10.py
ctest --test-dir build          # the C tests, after a CMake build
```

54 unit tests covering factorization residual, solve accuracy, COLAMD
with tau pinning, sparse triangular solves, edge cases, and the `SparseLU`
class (including transpose / conjugate-transpose solves).

## Files

- `include/fplus.h`          — the C interface
- `src/`                     — the C library: supernodal and row kernels, Ruiz, solves, ordering, the high-level interface
- `_sparse_lu_c.c`           — the CPython extension (wraps the library; releases the GIL)
- `sparse_lu_full_pivot.py`  — Python wrapper module
- `CMakeLists.txt`, `pyproject.toml` — the CMake build and the Python package
- `setup.py`                 — in-place extension build
- `__init__.py`              — package re-exports
- `tests/c/`, `examples/`    — C tests (ctest) and an example program
- `test_sparse_lu_full_pivot.py`, `test_audit_2026_10_10.py` — Python unit tests

## License

GPL-3.0-or-later (see `LICENSE`).  This copy is vendored into SPDEE; the
library is developed in its own repository.
