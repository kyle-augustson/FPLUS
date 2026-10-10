# FPLUS as a C library with a Python interface

FPLUS today is a Python package backed by one C extension, `_sparse_lu_c.c`. Its numerical kernels are already C: Ruiz equilibration, the
supernodal left-looking kernel, the row kernel for full/rook pivoting, the triangular solves and the column-etree postorder. Ordering and
permutation are Python (COLAMD via ctypes, dense-line pinning, the ordering cache), and the kernels report results and errors through the
Python C API. This plan turns FPLUS into a standalone C library, `libfplus`, with the Python package as a thin wrapper over it.

Decisions: the existing repository is restructured, built with CMake; Python calls the library through a hand-written CPython extension;
COLAMD comes from the system SuiteSparse and is linked at build time.

## Layout

```
include/fplus.h          public API
src/order.c              dense-line pinning, COLAMD (linked colamd_l), column-etree postorder, permutation
src/equil.c              Ruiz equilibration (CSC)
src/factor_sn.c          supernodal left-looking kernel (relaxed supernodes, panels, OpenMP)
src/factor_row.c         row kernel (partial / rook / full pivoting, drop tolerance)
src/solve.c              triangular solves (N, T, H), multiple right-hand sides
src/gpu.c                optional cuBLAS / hipBLAS update path (dlopen, as now)
src/fplus_internal.h
python/fplus/            package: SparseLU, sparse_lu_full_pivot (same signatures as today), pure-Python fallback
python/_fplus_ext.c      CPython extension over libfplus; releases the GIL in analyze / factor / solve
tests/                   C tests (ctest) and the existing pytest suite with its fixtures
CMakeLists.txt           BLAS vendor (OpenBLAS, MKL, Cray libsci, NVPL), OpenMP, SuiteSparse COLAMD, optional GPU
pyproject.toml           scikit-build-core, building the extension against the same CMake
```

## API sketch

```c
typedef struct fplus_symbolic fplus_symbolic;   /* ordering of one sparsity pattern; reusable across values */
typedef struct fplus_numeric  fplus_numeric;    /* L, U, p, q, Dr, Dc */

typedef struct {
    int     pin_dense;                /* move dense rows / columns last (default 1) */
    double  dense_fraction;           /* a line is dense above this fraction of n and dense_min_nnz entries */
    int64_t dense_min_nnz;
    int     postorder;                /* postorder the column etree after COLAMD (default 1) */
    int     equilibrate;              /* Ruiz (default 1) */
    int     pivot;                    /* FPLUS_PIVOT_PARTIAL (supernodal), _ROOK, _FULL (row kernel) */
    double  pivot_rel;                /* singular when the best pivot < pivot_rel * column max (default sqrt(eps)) */
    int64_t relax_rows, panel;        /* supernodal kernel (defaults 32, 512) */
    double  drop_tol;                 /* row kernel only */
    int     nthreads;                 /* 0: OpenMP default */
    int     gpu; double gpu_min_flops;
    int     verbose;
} fplus_options;

void fplus_default_options(fplus_options *opt);
int  fplus_analyze  (int64_t n, const int64_t *colptr, const int64_t *rowind, const fplus_options *opt,
                     fplus_symbolic **sym, char *err, size_t errlen);
int  fplus_factor_z (const fplus_symbolic *sym, const double _Complex *values, const fplus_options *opt,
                     fplus_numeric **num, char *err, size_t errlen);
int  fplus_factor_d (const fplus_symbolic *sym, const double *values, const fplus_options *opt,
                     fplus_numeric **num, char *err, size_t errlen);
int  fplus_solve_z  (const fplus_numeric *num, double _Complex *b, int64_t nrhs, int trans);   /* in place; trans N / T / H */
int  fplus_solve_d  (const fplus_numeric *num, double *b, int64_t nrhs, int trans);
int  fplus_get_factors(const fplus_numeric *num, ...);   /* L, U (CSC), p, q, Dr, Dc, nnz, supernodes */
void fplus_free_symbolic(fplus_symbolic *sym);
void fplus_free_numeric (fplus_numeric *num);
```

The input is CSC with int64 indices; values follow the pattern given to `fplus_analyze`. The return codes are FPLUS_OK, _SINGULAR,
_NOMEM, _BADARG and _GPU, with a message in `err`. There is no global state, so separate handles can be used from separate threads.

## Phases

Each phase keeps the factors bit-identical to the current kernel on the OpenBLAS reference pencils (L, U, p, q, Dr, Dc; the 25,972-row
complex check), and the pytest suite passing.

1. **Separate the C core from the Python API.** Split `_sparse_lu_c.c` into `src/`. The kernels return status codes and a message
   instead of raising, and fill plain structs instead of Python dicts. The existing extension becomes a wrapper over them; the build
   stays setup.py for this phase. **Done:** `include/fplus.h` and `src/{factor_row,factor_sn,equil,solve,order}.c` have no Python API
   calls, and `_sparse_lu_c.c` is the wrapper. The GIL release moved up from phase 3: two factorizations from two Python threads run
   1.8x faster than one after the other. Factors are bit-identical to 266c55e on the 25,972-row pencil, and pytest passes 61/61.
2. **Ordering in C.** Port dense-line pinning, the COLAMD call (linked `colamd_l`), the postorder and the permutation; add
   `fplus_analyze`. **Done:**
   - `fplus_analyze` and the symbolic handle (`src/order.c`), and `fplus_factor_sn_perm` / `_sym` (`src/pipeline.c`: permute in
     parallel, Ruiz on CSC, the supernodal kernel). COLAMD is linked at build time; `setup.py` fails without it.
   - The symbolic object keeps only the orders: a permuted pattern plus value map would be ~6 GB at 3.6e8 nonzeros. So the
     existing Python LRU cache of orders stays, and the C pipeline re-applies them.
   - COLAMD's 1.2 GB workspace is allocated with transparent huge pages (`fplus_calloc_huge`): 5.7 -> 3.8 s, the same as numpy's
     allocation in the old ctypes route.
   - Checks: bit-identical to the Python route in 160 randomized cases (118 with pinned dense rows / columns, explicit masks,
     postorder on / off, pinning off), and to 266c55e on the 25,972-row pencil. `FPLUS_PY_ORDERING=1` forces the old route.
3. **CMake and packaging.** CMake for the library (BLAS vendor, OpenMP, SuiteSparse, GPU) and scikit-build-core for the extension.
   Release the GIL around analyze / factor / solve. Add C tests and an example program.
4. **Afterwards.** Multiple right-hand sides in the solves, GPU-resident updates (cuBLAS / hipBLAS), HIP on the MI300X.
