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

## Building the C extension

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
python test_sparse_lu_full_pivot.py
```

54 unit tests covering factorization residual, solve accuracy, COLAMD
with tau pinning, sparse triangular solves, edge cases, and the `SparseLU`
class (including transpose / conjugate-transpose solves).

## Files

- `_sparse_lu_c.c`           — C extension source (LU + Ruiz + BLAS axpy)
- `sparse_lu_full_pivot.py`  — Python wrapper module
- `setup.py`                 — C extension build script
- `__init__.py`              — package re-exports
- `test_sparse_lu_full_pivot.py` — unit tests

## License

GPL-3.0-or-later (see `LICENSE`).  This copy is vendored into SPDEE; the
library is developed in its own repository.
