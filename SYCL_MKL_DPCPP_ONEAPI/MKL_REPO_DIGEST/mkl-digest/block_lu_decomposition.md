# Block LU Decomposition

## Domain & Purpose
oneMKL domain: `BLAS` (`oneapi::mkl::blas`) and `LAPACK` (`oneapi::mkl::lapack`) through the SYCL API (`"oneapi/mkl.hpp"`), plus host-side CBLAS/LAPACKE calls in the residual checks. Two small programs: `factor` computes an LU factorization with partial pivoting of a general block tridiagonal matrix; `solve` reuses that factorization to solve `A*X = F` with multiple right-hand sides. Pointer-based USM is used so single oneMKL routines operate on submatrices of the original arrays.

## Problem & Math
Coefficient matrix, `N` block rows, blocks `NB` by `NB`:

```
(D_1  C_1                          )
(B_1  D_2  C_2                     )
(     B_2  D_3  C_3                )
(           .........              )
(              B_N-2 D_N-1  C_N-1  )
(                    B_N-1  D_N    )
```

`A = L*U` with elimination and partial pivoting; `L` is a product of permutation and unit lower bidiagonal block matrices, `U` upper triangular with nonzeroes only in the main block diagonal and first two block superdiagonals. Factorization is a block version of LAPACK `DGTTRF`.

Each outer step `k` forms the `2*NB` by `3*NB` submatrix `[D_k C_k 0 ; B_k D_k+1 C_k+1]` and partially factors it (`ptldgetrf`); the last two block rows are handled by a single `2*NB` by `2*NB` `getrf`. Solving is forward substitution (unit lower triangular `trsm` + `gemm` update) followed by backward substitution (non-unit upper triangular `trsm` + `gemm` updates), with the recorded pivots applied as row swaps on `F`.

Residual checks (host side): `||A - L*U||_F/||A||_F` (`resid1`) and `max_(i=1,...,nrhs){||ax(i)-f(i)||/||f(i)||}` (`resid2`).

## oneMKL Routines Used

Setup shared by every call: `using namespace oneapi;`, header `#include "oneapi/mkl.hpp"`, and a `sycl::queue` variable named `queue`.

### BLAS — `mkl::blas::copy` (returns an event)
Used as strided column copies to/from the scratch submatrix, one column at a time:

```cpp
auto event1 = mkl::blas::copy(queue, nb,   &D(0,(k)*nb + j), 1,   &A(0,     j),1);
auto event2 = mkl::blas::copy(queue, nb,  &DL(0,(k)*nb + j), 1,   &A(nb,    j),1);
auto event3 = mkl::blas::copy(queue, nb, &DU1(0,(k)*nb + j), 1,   &A(0,  nb+j),1);
event1.wait_and_throw();
```

Argument order as called: `(queue, count, x, incx, y, incy)`.

### BLAS — `mkl::blas::swap`
```cpp
auto event1 = mkl::blas::swap(queue, nrhs, &F(k*nb+i, 0), ldf, &F(k*nb+IPIV(i,k)-1, 0), ldf);
event1.wait_and_throw();
```
Argument order as called: `(queue, n, x, incx, y, incy)`. `incx`/`incy` here are leading dimensions (`ldf`) because the swap applies to one row across all RHS columns. `ptldgetrf` also calls the same routine with `lda` as the stride to swap two rows of the `2*NB` by `3*NB` scratch block.

### BLAS — `mkl::blas::trsm`
```cpp
auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, nb, nrhs, 1.0, &D(0, k*nb), nb, &F(k*nb, 0), ldf);
auto event3 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, nb, nrhs, 1.0, &D(0, (n-1)*nb), nb, &F((n-1)*nb, 0), ldf, {event2});
```
Argument order as called: `(queue, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb[, deps])`. The first call carries no dependency list; the second takes `{event2}`. Enum values observed: `mkl::side::left`, `mkl::uplo::lower`, `mkl::uplo::upper`, `mkl::transpose::nontrans`, `mkl::diag::unit`, `mkl::diag::nonunit`.

### BLAS — `mkl::blas::gemm`
```cpp
auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, k, n-k, 1.0, &A(0,0), lda, &A(0,k), lda);
auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, m-k, n-k, k, -1.0, &A(k,0), lda, &A(0,k), lda, 1.0, &A(k,k), lda, {event1});
event2.wait_and_throw();
```
Argument order as called: `(queue, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc[, deps])`. Only `mkl::transpose::nontrans` appears in this sample. The `trsm` line is included because the trailing `{event1}` on the `gemm` is the dependency event list that orders it after that `trsm`.

### LAPACK — `mkl::lapack::getrf_scratchpad_size<double>` and `mkl::lapack::getrf`
```cpp
std::int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, 2*nb, 2*nb, lda);
double* scratchpad = sycl::malloc_shared<double>(scratchpad_size, device, context);
if (!scratchpad) {
    info = -1000;
    goto cleanup;
}
auto event1 = mkl::lapack::getrf(queue, 2*nb, 2*nb, &A(0,0), lda, &IPIV(0,n-2), scratchpad, scratchpad_size );
event1.wait_and_throw();
sycl::free(scratchpad, context);
```
`getrf_scratchpad_size<double>(queue, m, n, lda)` returns `std::int64_t`. `getrf(queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size)` returns an event; call it, then `wait_and_throw()`.

### LAPACK — `mkl::lapack::exception`
```cpp
} catch(mkl::lapack::exception const& e) {
    // Handle LAPACK related exceptions happened during synchronous call
    std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
    if (e.info() > 0) {
    // INFO is equal to the 'global' index of the element u_ii of the factor
    // U which is equal to zero
        info = e.info() + (n-2)*nb;
    }
    return info;
}
```

### Host-side CBLAS / LAPACKE (not SYCL, not device-queued)
`LAPACKE_dlange(MKL_COL_MAJOR, 'F', m, n, a, lda)`, `LAPACKE_dlarnv(idist, iseed, n, x)` called as `LAPACKE_dlarnv(2, iseed.data(), n*nb*nb, d.data())`, `cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`, `cblas_dcopy(n, x, incx, y, incy)`, `cblas_dswap(n, x, incx, y, incy)`, `cblas_dnrm2(n, x, incx)`. Matrix-layout enum values observed: `MKL_COL_MAJOR`, `CblasColMajor`, `CblasNoTrans`.

## Key Code Patterns

Device/queue setup: default device via `sycl::default_selector_v`, no in-order property — the queue is created as

```cpp
sycl::device device{sycl::default_selector_v};
sycl::queue queue(device, error_handler);
sycl::context context = queue.get_context();
```

where `error_handler` is a `[&] (sycl::exception_list exceptions)` lambda that rethrows each exception and catches `mkl::lapack::exception` (recording `info = e.info()`) and `sycl::exception` separately.

Two device gates before any allocation, verbatim (from `factor.cpp`; `solve.cpp` repeats them):

```cpp
if (device.is_gpu() && device.get_platform().get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cerr << "This sample requires Level Zero when running on GPUs." << std::endl;
    std::cerr << "Please check your system configuration." << std::endl;
    return 0;
}

if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
    std::cerr << "This sample uses double precision, which is not supported" << std::endl;
    std::cerr << "by the selected device. Quitting." << std::endl;
    return 0;
}
```

USM allocation through the standard allocator alias, so `std::vector` storage lives in shared USM:

```cpp
template<typename T>
using allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
allocator_t<double> allocator_d(context, device);
std::vector<double, allocator_t<double>> d(nb* n*nb, allocator_d);
```

Raw shared USM for the block scratchpad inside `dgeblttrf` / `ptldgetrf`:

```cpp
const int64_t lda = 2*nb;
double* a = sycl::malloc_shared<double>(lda * 3*nb, device, context);
...
sycl::free(a, context);
```

Submatrix addressing is done with host lambdas capturing the base pointer by value and indexing column-major with a fixed leading dimension, e.g. `[=,&d](int64_t i, int64_t j) -> double& { return d[i + j*nb]; }`. Passing `&D(0, k*nb)` hands oneMKL a pointer to the start of block column `k`.

Ordering is expressed with events, not with an in-order queue: each BLAS call returns an event; dependent calls receive the predecessor in a trailing `{event1}` list, and `.wait_and_throw()` is called before the host reads/writes the arrays. Example of a two-call dependency chain: `trsm` -> `gemm(..., {event1})` -> `event2.wait_and_throw()`.

Zeroing the padding column of the `2*NB` by `3*NB` scratch block uses a raw SYCL kernel submitted to the same queue:

```cpp
queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(nb), [=] (sycl::id<1> it) {
                const int64_t i = it[0];
                a[i + (2*nb + j)*lda] = 0.0;
                });
        });
queue.wait();
```

Synchronous error surface: `factor.cpp` checks `info` returned by `dgeblttrf` and prints `"DGEBLTTRF returned nonzero INFOi = " << info`; `solve.cpp` prints `-info << "-th parameter in call of dgeblttrs has illegal value"`. Both `main`s first reject a build whose `MKL_INT` is not 64-bit:

```cpp
if (sizeof(MKL_INT) != sizeof(int64_t)) {
    std::cerr << "MKL_INT not 64bit" << std::endl;
    return -1;
}
```

Host-side pivoting is applied manually on the host arrays, mirroring what the `getrf` calls did inside the block: loops over `if (IPIV(i,k) != i+1) cblas_dswap(...)`, and in `resid1` the host-side reconstruction of `L`/`U` re-applies `ipiv` in reverse order (`for (int64_t i = nb -1; i >= 0; i--)`).

## Build & Run

GNU/Linux (`GNUmakefile`); `MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="blas,lapack"`:

```
make                 # builds factor and solve, then runs ./factor and ./solve
icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)
clean: -rm -f factor solve genxir
```

Compilation units: `factor` from `factor.cpp dgeblttrf.cpp auxi.cpp`; `solve` from `solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp`.

Windows (NMAKE `makefile`); `DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,lapack" /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib`:

```
nmake                # builds factor.exe and solve.exe, then runs both
icx-cl -fsycl factor.cpp dgeblttrf.cpp auxi.cpp /Fefactor.exe $(DPCPP_OPTS)
icx-cl -fsycl solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp /Fesolve.exe $(DPCPP_OPTS)
nmake clean          # del /q factor.exe factor.exp factor.lib solve.exe solve.exp solve.lib
```

Device selection: the sample runs on the default SYCL device; set `ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"` (README). On GPUs the sample requires the Level Zero backend, checked at runtime.

Program parameters, hard-coded: `factor.cpp` uses `n = 200`, `nb = 20`; `solve.cpp` uses `n = 200`, `nb = 20`, `nrhs = 10`, `ldf = nb*n`, `iseed = {1, 4, 23, 77}` (`factor.cpp` uses `{9, 41, 11, 3}`).

Expected output lines (README example; note `solve.cpp` itself prints `to RHS vectors' norms.` with an apostrophe, while the README example writes `to RHS vectors norms.`):

```
Testing the accuracy of LU factorization with pivoting
of randomly generated block tridiagonal matrix
by calculating norm of the residual matrix.
||A - LU||_F/||A||_F = 3.65246e-16

Testing accuracy of solution of linear equations system
with randomly generated block tridiagonal coefficient
matrix by calculating ratios of residuals
to RHS vectors norms.
max_(i=1,...,nrhs){||ax(i)-f(i)||/||f(i)||} = 6.88457e-13
```

## Gotchas & Invariants
- ILP64 is mandatory: the program aborts with `"MKL_INT not 64bit"` unless `-DMKL_ILP64` + `-qmkl-ilp64` (or `/DMKL_ILP64` + `/Qmkl-ilp64`) are used. All block indices are `int64_t`. The random seed arrays differ: `factor.cpp` declares `std::vector<MKL_INT> iseed = {9, 41, 11, 3}` and passes `iseed.data()`; `solve.cpp` declares `std::vector<int64_t> iseed = {1, 4, 23, 77}` and passes `reinterpret_cast<MKL_INT*>(iseed.data())` to `LAPACKE_dlarnv`.
- All block arrays are column-major with leading dimension exactly `nb`, laid out block column by block column: `D` is `nb * (n*nb)`, `DL`/`DU1` are `nb * ((n-1)*nb)`, `DU2` is `nb * ((n-2)*nb)`, `IPIV` is `nb * n`. `DU2` holds only `N-2` blocks and the outer loop bound is `k < n-2`, so for `n < 3` `du2` is zero-sized and the loop body never runs; the source validates only `n <= 0` and `nb <= 0`, not `n >= 3`.
- `ptldgetrf` uses `lda = 2*nb` for the `2*NB` by `3*NB` scratch block and factors only the first `k = nb` columns; it is only entered with `k < min(m,n)`. Its `ipiv` has dimension `min(M,K)`.
- Pivots in `IPIV` are "local" block indices in the range `1..2*NB`; the global row index is `IPIV(I,K) + (K-1)*NB`. Comparing `IPIV(i,n-1) != i+1 + nb` in `dgeblttrs` (rather than `i+1`) is the consequence for the final factorized block.
- Scratchpad must be queried and allocated per `getrf` call, sized by `mkl::lapack::getrf_scratchpad_size<double>(queue, m, n, lda)`; it is `sycl::malloc_shared<double>` and must be `sycl::free(scratchpad, context)`-ed. The sample does not free the scratchpad on the `getrf`-throws path inside `ptldgetrf` (nor on the corresponding path in `dgeblttrf`'s final `getrf`).
- `info` conventions differ by function: `dgeblttrf` returns `0` on success, `-1000` when a memory buffer could not be allocated, `-i` for the i-th illegal argument (`-1` for `n`, `-2` for `nb`), and a positive value when `U(i,i)` is exactly zero — from the `ptldgetrf` call it returns `info + k*nb` (global index) and from the final `getrf` catch it returns `e.info() + (n-2)*nb`. `dgeblttrs` returns `0` or `-i` (`-1` n, `-2` nb, `-3` nrhs, `-10` ldf). `ptldgetrf` returns `0`, `-1000` (scratchpad not allocated), `-i` (`-1` m, `-2` n, `-3` k, `-5` lda) or `i > 0`.
- `LDF >= N*NB` is validated (`info = -10` otherwise); in `solve.cpp` `ldf = nb*n`. `resid2` declares `LDX >= N*NB` and is called with both `ldx` and `ldb` equal to `ldf`.
- The queue is out-of-order; correctness depends on the explicit event chain and on `.wait_and_throw()` before subsequent host or device use. Dropping the `{event}` dependency arguments, or the `wait_and_throw`, produces races rather than compile errors. Note also that the small zeroing kernel is followed by a full `queue.wait()` inside the `j` loop.
- Asynchronous errors are delivered only through the queue's error handler (`sycl::exception_list`); it catches `mkl::lapack::exception` (using `e.info()`) and `sycl::exception` separately.
- `sycl::info::device::double_fp_config` must be non-empty; a device without `double` support exits with status 0 rather than failing loudly. GPU runs on a non-Level-Zero backend also exit with status 0.
- The last `2*NB` by `2*NB` `getrf` writes its pivots into the last two block columns of `IPIV` (`&IPIV(0,n-2)`), i.e. `ipiv` is overwritten in place with `2*nb` entries there.

## Explicit gaps
- The sample calls `mkl::blas::gemm` and `mkl::blas::trsm` through the un-prefixed `mkl::blas::` namespace; no layout namespace is named at any call site. The files never state which layout this resolves to; the column-major interpretation is only inferable from the pointer arithmetic and leading dimensions, not asserted by the source.
- No explicit template-argument form of `mkl::blas::gemm`/`mkl::blas::trsm` appears anywhere; the exact overload set and template parameter list are not established by these files.
- The full parameter list and return semantics of `mkl::blas::copy` and `mkl::blas::swap` are only visible through call sites; their declared prototype is not shown (in particular, whether the trailing arguments could also include a dependency list is not demonstrated).
- `dgeblttrf` guards only `n <= 0` and `nb <= 0`. The behavior for `n == 1` or `n == 2` (where `DU2` has zero length, the `k < n-2` loop body never runs, and `resid1` still touches `D(0, (n-2)*nb + j)`) is not analyzed by the source.
- The source does not document the required minimum oneMKL version, the value of `MKLROOT`/environment setup, or any CMake target (`GNUmakefile` and NMAKE `makefile` are the only build inputs present; the directory contains no `CMakeLists.txt`).
- The meaning of `du2` for `n == 2`, and whether `du2` may legitimately be a null pointer in that case, is not stated.
- The `#undef A / D / DL / DU1 / DU2 / IPIV` lines before `ptldgetrf` are unexplained; the source gives no reason these names could be macros.

## Source Map
- `README.md` — purpose, routines named (`gemm`, `trsm`, `getrf`), build/run instructions, `ONEAPI_DEVICE_SELECTOR`, sample output.
- `makefile` — NMAKE/Windows build: `icx-cl -fsycl`, `DPCPP_OPTS`, `factor.exe`/`solve.exe` targets, `clean`.
- `GNUmakefile` — GNU/Linux build: `icpx`, `MKL_COPTS`, `factor`/`solve` targets, `clean`, `.PHONY`.
- `dgeblttrf.cpp` — `dgeblttrf` (block tridiagonal LU with pivoting) and `ptldgetrf` (partial LU of a rectangular matrix); uses `mkl::blas::copy/gemm/trsm/swap` and `mkl::lapack::getrf`/`getrf_scratchpad_size<double>`.
- `dgeblttrs.cpp` — `dgeblttrs`: forward/backward substitution solving `A*X = F` from the `dgeblttrf` factors, applying `IPIV` row swaps to `F`.
- `factor.cpp` — driver: random fill via `LAPACKE_dlarnv`, calls `dgeblttrf`, reports `||A - LU||_F/||A||_F`.
- `solve.cpp` — driver: random fill, calls `dgeblttrf` then `dgeblttrs`, reports `max_(i=1,...,nrhs){||ax(i)-f(i)||/||f(i)||}`.
- `auxi.cpp` — host-side `resid1` and `resid2` residual/norm checks using `LAPACKE_dlange`, `cblas_dgemm`, `cblas_dcopy`, `cblas_dswap`, `cblas_dnrm2`.
