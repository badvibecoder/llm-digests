# Block Cholesky Decomposition

## Domain & Purpose

oneMKL domains: **BLAS** (`gemm`, `syrk`, `trsm`) and **LAPACK** (`potrf`, plus `potrf_scratchpad_size`) through the oneMKL SYCL API (`oneapi/mkl.hpp`); host-side CBLAS/LAPACKE is used only for data generation and residual checking and its declarations come from `mkl.h`.

Two small applications: `factor.cpp` generates a symmetric positive-definite block tridiagonal matrix and Cholesky-factors it with `L*L^t`; `solve.cpp` factors the same matrix and then solves `A*X=F` with multiple RHS. Both use USM (shared) allocation throughout so individual oneMKL routines operate on submatrices of the original arrays, and they express inter-routine dependencies with SYCL events.

## Problem & Math

Matrix (block tridiagonal, symmetric positive definite), `N` block rows, block size `NB`:

```
D_1  B_1^t
B_1  D_2   B_2^t
     B_2  D_3   B_3^t
        .     .      .
            .     .      .
              B_N-2  D_N-1   B_N-1^t
                     B_N-1    D_N
```

Factorization `A = L*L^t` with lower block-bidiagonal `L` (`L_j` diagonal blocks, `C_j` sub-diagonal blocks, `C_j` in the math = array `B` in the storage). This is a block version of LAPACK `DPTTRF`/`DPTTRS`.

Per block step (`k = 0 .. n-2`), with `D_k` already factored:

```
trsm:  C_k  := C_k * (L_k^t)^-1
syrk:  D_k+1 := D_k+1 - C_k * C_k^t          (lower triangle only)
potrf: D_k+1 := chol(D_k+1)                  (lower)
```

Solve (`dpbltrs`) is two triangular sweeps; forward `L*Y=F`, then backward `L^t*X=Y`:

```
Y_1 := L_1^-1 * F_1 ;  Y_k+1 := L_k+1^-1 * (F_k+1 - C_k * Y_k)
X_N := L_N^-T * Y_N ;  X_k  := L_k^-T  * (Y_k - C_k^t * X_k+1)
```

Accuracy tests: factor checks `||A-L*L^t||_F/||A||_F <= 5*EPS`; solve checks `max_(i=1,...,NRHS){||A*X(i)-F(i)||/||F(i)||} <= 10*EPS`, with `EPS = LAPACKE_dlamch('E')`.

## oneMKL Routines Used

Headers used across the sample: `#include <sycl/sycl.hpp>` and `#include "oneapi/mkl.hpp"` in `dpbltrf.cpp`, `dpbltrs.cpp`, `factor.cpp`, and `solve.cpp`; `#include "mkl.h"` in `auxi.cpp` only. Those four SYCL files each have `using namespace oneapi;`, so the SYCL calls appear as `mkl::...`; `auxi.cpp` has no `using namespace oneapi;` and makes no oneMKL SYCL calls.

Calls are written with **no `row_major`/`column_major` namespace qualifier** — `mkl::blas::trsm`, `mkl::blas::syrk`, `mkl::blas::gemm` — and all array accessors are column-major (`d[i + j*ldd]`).

**Scratchpad query (LAPACK), verbatim:**

```cpp
std::int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue, mkl::uplo::lower, nb, ldd);
```

Explicit template argument `<double>`; arguments in order `(queue, uplo, n, lda)` where `n = nb` (one block) and `lda = ldd`.

**Cholesky factorization (LAPACK), verbatim (first block, and inside the loop):**

```cpp
auto event1 = mkl::lapack::potrf(queue, mkl::uplo::lower, nb, d, ldd, scratchpad, scratchpad_size );
...
auto event1 = mkl::lapack::potrf(queue, mkl::uplo::lower, nb, &D(0,(k+1)*nb), ldd, scratchpad, scratchpad_size );
```

Argument order as used: `(queue, uplo, n, a, lda, scratchpad, scratchpad_size)`. The returned event is stored in `auto event1` and consumed synchronously via `event1.wait_and_throw()`.

**Rank-k update (BLAS), verbatim:**

```cpp
auto event2 = mkl::blas::syrk(queue, mkl::uplo::lower, mkl::transpose::nontrans, nb, nb,
        -1.0, &B(0,k*nb), ldb, 1.0, &D(0,(k+1)*nb), ldd, {event1});
```

Argument order as used: `(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies)`. `uplo::lower` + `transpose::nontrans` with `alpha = -1.0`, `beta = 1.0` gives `C := C - A*A^t` on the lower triangle. The trailing `{event1}` is the dependency-event list.

**Triangular solve (BLAS), right side + transpose, verbatim:**

```cpp
auto event1 = mkl::blas::trsm(queue, mkl::side::right, mkl::uplo::lower, mkl::transpose::trans,
        mkl::diag::nonunit, nb, nb, 1.0, &D(0,k*nb), ldd, &B(0,k*nb), ldb);
```

Argument order as used: `(queue, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)`. Updates `B(:,k)` (size `nb x nb`) in place.

**Triangular solve (BLAS), left side** — from `dpbltrs.cpp`, verbatim:

```cpp
auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, d, ldd, f, ldf);
...
auto event3 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::trans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, (n-1)*nb), ldd, &F((n-1)*nb, 0), ldf);
```

**General matrix multiply (BLAS), with dependency event, verbatim:**

```cpp
auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &B(0, k*nb), ldb, &F(k*nb, 0), ldf, 1.0, &F((k+1)*nb, 0), ldf);
```

Argument order as used: `(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies)`. The backward sweep uses `mkl::transpose::trans, mkl::transpose::nontrans`.

**Enum values actually used:** `mkl::uplo::lower`, `mkl::side::left`, `mkl::side::right`, `mkl::transpose::nontrans`, `mkl::transpose::trans`, `mkl::diag::nonunit`. No other oneMKL enum value appears in these files.

**Host-side (non-SYCL) routines used for verification only:** `LAPACKE_dlange(MKL_COL_MAJOR, 'F', m, n, a, lda)`, `LAPACKE_dlarnv(2, iseed.data(), count, ptr)`, `LAPACKE_dlamch('E')`, `cblas_dcopy`, `cblas_dtrmm(CblasColMajor, CblasLeft|CblasRight, CblasLower, CblasNoTrans|CblasTrans, CblasNonUnit, m, n, alpha, A, lda, B, ldb)`, `cblas_dgemm(CblasColMajor, ...)`, `cblas_dnrm2(n, x, incx)`.

## Key Code Patterns

**Device + queue + async error handler.** The handler is passed as the second `sycl::queue` constructor argument and rethrows to discriminate LAPACK from generic SYCL errors:

```cpp
auto error_handler = [&] (sycl::exception_list exceptions) {
    for (auto const& e : exceptions) {
        try { std::rethrow_exception(e); }
        catch(mkl::lapack::exception const& e) { info = e.info(); /* ... */ }
        catch(sycl::exception const& e) { info = -1; /* ... */ }
    }
};
sycl::device device{sycl::default_selector_v};
sycl::queue queue(device, error_handler);
sycl::context context = queue.get_context();
```

**Double-precision capability gate** (returns 0, i.e. success-like exit, when unsupported):

```cpp
if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
    std::cerr << "The sample uses double precision, which is not supported" << std::endl;
    std::cerr << "by the selected device. Quitting." << std::endl;
    return 0;
}
```

**USM shared allocation via an allocator type, not raw `malloc_shared`** (for the arrays passed to oneMKL):

```cpp
template<typename T>
using allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
allocator_t<double> allocator_d(context, device);
std::vector<double, allocator_t<double>>  d(nb * n*nb,     allocator_d);
```

**Scratchpad: query size, allocate shared, free after the synchronous wait.** The size is in *elements* (`* sizeof(double)`); allocation failure is reported as `info = -1000` and jumps to a `cleanup:` label:

```cpp
std::int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue, mkl::uplo::lower, nb, ldd);
double* scratchpad = static_cast<double*>(sycl::malloc_shared(scratchpad_size * sizeof(double), device, context));
if (scratchpad_size != 0 && !scratchpad) { info = -1000; goto cleanup; }
auto event1 = mkl::lapack::potrf(queue, mkl::uplo::lower, nb, d, ldd, scratchpad, scratchpad_size );
event1.wait_and_throw();
sycl::free(scratchpad, context);
```

**Ordering model.** In `dpbltrf`: the first `potrf` event is waited on; in the loop the `trsm` event is *not* waited on directly but is passed to `syrk` as the trailing dependency list `{event1}`, and `event2.wait_and_throw()` (the `syrk` event) is what orders the result before the next `potrf`; each in-loop `potrf` is then waited on before its scratchpad is freed. In `dpbltrs`: the two outer `trsm` calls (`event1.wait_and_throw()`) and each loop `trsm` (`event3.wait_and_throw()`) are waited on, while each loop `gemm` event orders the following `trsm` only through the dependency list `{event2}`. The queue is constructed without the in-order property, and every dependency above is expressed explicitly through waits or dependency lists.

**Submatrix addressing.** Pointers are built by offsetting into the packed block array; the leading-dimension argument passed from the drivers stays `nb`, the block stride, so block `k` occupies columns `k*nb .. k*nb+nb-1`:

```cpp
auto D = [=](int64_t i, int64_t j) -> double& { return d[(i) + (j)*ldd]; };
auto B = [=](int64_t i, int64_t j) -> double& { return b[(i) + (j)*ldb]; };
```

`dpbltrf` then passes `&D(0,k*nb), ldd, &B(0,k*nb), ldb` as the per-block arguments.

**LAPACK exception handling inside `dpbltrf`** maps a per-block positive `info` to a global element index:

```cpp
} catch(mkl::lapack::exception const& e) {
    std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
    if (e.info() > 0) { info = e.info() + (k+1)*nb; }
    return info;
}
```

**Argument validation codes** (checked before any oneMKL call; `dpbltrf` uses `-1,-2,-4,-6` for `n, nb, ldd, ldb`; `dpbltrs` uses `-1,-2,-3,-5,-7,-9` for `n, nrhs, nb, ldd, ldb, ldf`). Return code `> 0` from `dpbltrf` means the leading minor of that order is not positive definite.

**64-bit integer gate** at program start, before any allocation:

```cpp
if (sizeof(MKL_INT) != sizeof(int64_t)) { std::cerr << "MKL_INT not 64bit" << std::endl; return -1; }
```

## Build & Run

There is no CMake file among the sources read; build comes from `GNUmakefile` (GNU make) and `makefile` (NMAKE).

Linux (`GNUmakefile`), verbatim flags and rules:

```
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="blas,lapack"

factor: factor.cpp dpbltrf.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)
```

Targets: `all` (depends on `factor solve`, then runs `./factor` and `./solve`), `factor`, `solve`, `clean` (`-rm -f factor solve`); `.PHONY: all clean`. So a bare `make` builds **and** runs both programs.

Windows (`makefile`, NMAKE + `icx-cl`), verbatim:

```
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,lapack" /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

factor.exe: factor.cpp dpbltrf.cpp auxi.cpp
	icx-cl -fsycl factor.cpp dpbltrf.cpp auxi.cpp /Fefactor.exe $(DPCPP_OPTS)
```

`all` runs `.\factor.exe` then `.\solve.exe`; `clean` deletes `factor.exe factor.exp factor.lib solve.exe solve.exp solve.lib`. `MKLROOT` must be set (it is referenced by the Windows options).

Environment: the README requires sourcing `setvars` from the oneAPI root, e.g. `. /opt/intel/oneapi/setvars.sh` (system-wide) or `. ~/intel/oneapi/setvars.sh` (private). Device selection uses the default SYCL device; `ONEAPI_DEVICE_SELECTOR` may be set to `"*:cpu"` or `"*:gpu"`.

Program parameters (hardcoded, no CLI arguments): `factor` uses `n = 200`, `nb = 20`; `solve` uses `n = 200`, `nb = 20`, `nrhs = 10`, `ldf = nb*n`. `iseed` is a 4-element `std::vector<MKL_INT>` (`{1, 2, 33, 15}` in factor, `{1, 2, 3, 19}` in solve).

Expected output (literal strings from the sample/README):

```
Matrix size = 200
Block  size = 20
Matrices are being generated.
Call Cholesky factorization
Cholesky factorization succeeded.
Testing the residual
||A-L*L^t||_F/||A||_F <= 5*EPS...
passed
```

```
Call solving the system of linear equations
Solution succeeded.
The system is solved. Testing the residual
max_(i=1,...,NRHS){||A*X(i)-F(i)||/||F(i)||} <= 10*EPS
passed
```

## Gotchas & Invariants

- **Everything is column-major and block-packed.** `d` holds `N` diagonal blocks in consecutive groups of `NB` columns; `b` holds the `N-1` sub-diagonal blocks the same way. The accessor lambda in each driver hardcodes the stride (`d[i + j*nb]`), and `ldd = ldb = nb` is what is actually passed, so changing `nb` layout assumptions breaks the offsets.
- **Leading-dimension minimums are enforced by the sample, not by the libraries here:** `ldd >= nb`, `ldb >= nb` (checked in `dpbltrf` and `dpbltrs`), and `ldf >= nb*n` (checked in `dpbltrs`). `dpbltrf` does **not** validate an `ldf` (it has none).
- **`potrf` needs only one triangle and leaves the other untouched.** The diagonal blocks are stored lower-only; the code comments warn explicitly: "upper triangles of diagonal blocks are not zeroed on exit" and "upper triangles of D_j are not assumed to be zeroed." Any code that consumes `d` as a full symmetric block must zero/mirror the upper triangle itself.
- **`syrk` is called with `alpha = -1.0, beta = 1.0`** — it updates `D_{k+1}` in place; the diagonal block must already hold the input matrix and `uplo::lower` means only the lower half is updated.
- **Sub-block argument sizing:** `syrk` is called with `n = k = nb` (block rank), and `trsm`/`gemm` with `n = nrhs` for the solve sweeps — not the global `N*NB` size. The sources do not state what would happen if global dimensions were passed instead.
- **Scratchpad is queried per `potrf` call, inside the loop**, with `nb`/`ldd`, allocated on `device`/`context` explicitly, then freed after `event1.wait_and_throw()`. A `scratchpad_size` of 0 is allowed (the null check is guarded by `scratchpad_size != 0`).
- **`info = -1000` is a sample-private allocation-failure code**, not a oneMKL or LAPACK status; both drivers propagate it as a generic failure.
- **Positive `info` from `potrf` means not positive definite.** The loop handler in `dpbltrf` rebases it by `+(k+1)*nb` so it reads as a global element index; the first block's handler stores `e.info()` unmodified, which is the same index for offset 0. `dpbltrs` never produces positive `info`; it only validates arguments.
- **Synchronous waiting is mandatory before reusing `d`, `b`, or `f`.** In `dpbltrf` the `syrk` result is consumed by the next `potrf` only because `event2.wait_and_throw()` intervenes.
- **64-bit interface is required:** `-DMKL_ILP64` / `-qmkl-ilp64` / `/Qmkl-ilp64` plus `MKL_INT == int64_t`, otherwise the programs print `MKL_INT not 64bit` and return -1.
- **`-fsycl-device-code-split=per_kernel`** is present in both build systems; the SYCL implementations of BLAS/LAPACK are requested by `-qmkl-sycl-impl="blas,lapack"`.
- **Double precision is required**; devices lacking `double_fp_config` cause an early `return 0` (which looks like success at the shell).
- **Discrepancy between the two drivers:** the mirror-of-lower-triangle copy loop uses `cblas_dcopy(nb-j, &D(j+1, k*nb+j), 1, &D(j, k*nb+j+1), nb);` in `factor.cpp` but `cblas_dcopy(nb-j-1, ...)` with the same pointers in `solve.cpp`. The sources do not state which count is intended.
- `factor.cpp` declares `MKL_INT n = 200; MKL_INT nb = 20;` while `solve.cpp` declares the same names as `int64_t`; both are consistent only under the ILP64 build the makefiles request.

## Explicit gaps

- No `CMakeLists.txt` is among the files read; the only build information available is `GNUmakefile` and `makefile`. Any CMake target name, option, or `find_package` invocation for this sample is not established here.
- `factor.cpp` and `solve.cpp` call `LAPACKE_dlarnv`, `LAPACKE_dlamch`, and `cblas_dcopy` but include `oneapi/mkl.hpp` and never include `mkl.h`; these files do not show where those host-side declarations come from.
- Exact oneMKL overload sets are not visible: the sample always calls `mkl::blas::<fn>` / `mkl::lapack::potrf` with no `row_major`/`column_major` qualifier and with `double`/`std::int64_t` arguments, and always passes a one-element trailing event list (`{event1}`, `{event2}`). Which parameter combinations those overloads accept in general (e.g. whether the event-list argument is required vs. defaulted, or which `transpose`/`uplo` combinations are legal) is not shown.
- The semantics of `LAPACKE_dlarnv`'s first argument (`2`) are not documented in these files; only the literal call is known.
- `test_res`/`test_res1` are not oneMKL routines — they are sample-local functions in `auxi.cpp`; their declared `INFO` outputs are documented in comments but the actual functions return `double` and never return or check `INFO`.
- `test_res1` calls `cblas_dnrm2(nb*n, ...)` to compute RHS norms while the final residual norm uses `cblas_dnrm2(n*nb, ...)`; both are equal, but the inconsistency is unexplained.
- No performance/timing, stream, host-task, or in-order-queue variant is shown; whether an in-order queue would work here is not established by the source.
- The README output block contains `...` placeholder lines between messages; the exact full stdout ordering beyond the quoted strings is not fully pinned down.

## Source Map

- `README.md` — purpose, key routines (`gemm`, `syrk`, `trsm`, `potrf`), build instructions, `ONEAPI_DEVICE_SELECTOR`, example output.
- `GNUmakefile` — Linux `icpx` build: `-DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl="blas,lapack"`, `-fsycl -fsycl-device-code-split=per_kernel`; targets `all`, `factor`, `solve`, `clean`.
- `makefile` — Windows NMAKE build with `icx-cl`, `/Qmkl-ilp64`, `/Qmkl-sycl-impl="blas,lapack"`, `$(MKLROOT)`.
- `dpbltrf.cpp` — `dpbltrf(sycl::queue queue, int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb)`: LAPACK `potrf_scratchpad_size<double>`/`potrf` + BLAS `trsm`/`syrk` block Cholesky factorization loop, argument validation, scratchpad lifecycle, event dependency `{event1}`.
- `dpbltrs.cpp` — `dpbltrs(sycl::queue queue, int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf)`: forward/backward `trsm`+`gemm` sweeps for `A*X=F`.
- `factor.cpp` — driver: queue/error handler/double-FP gate, USM allocator vectors, random matrix generation with `LAPACKE_dlarnv`/`cblas_dcopy`, calls `dpbltrf`, calls `test_res`, prints `5*EPS` residual verdict.
- `solve.cpp` — driver: same setup plus RHS generation, `dpbltrf` then `dpbltrs`, calls `test_res1`, prints `10*EPS` residual verdict.
- `auxi.cpp` — host-only `test_res` and `test_res1` reference implementations using `LAPACKE_dlange`, `LAPACKE_dlamch`, `cblas_dcopy`, `cblas_dtrmm`, `cblas_dgemm`, `cblas_dnrm2`; not oneMKL SYCL code.
