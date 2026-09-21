# Sparse Conjugate Gradient

## Domain & Purpose
oneMKL **sparse linear algebra** (`oneapi::mkl::sparse::`) for CSR matrix handle setup, `optimize_*` analysis, sparse `gemv` and `trsv`; plus **BLAS Level-1 USM** (`oneapi::mkl::blas::`) vector routines. Solves a sparse, symmetric `A*x = b` by preconditioned conjugate gradient (PCG) with a symmetric Gauss-Seidel preconditioner, using USM allocations and a SYCL queue built with no explicit in-order property. Two variants ship: `sparse_cg.cpp` (alpha/beta on host) and `sparse_cg2.cpp` (alpha/beta kept on device).

## Problem & Math
Verbatim from the source comment blocks:

```
x_0 initial guess
r_0 = b - A*x_0
k = 0
while (||r_k|| / ||r_0|| > relTol and k < maxIter )
    solve M*z_k = r_k for z_k
    if (k == 0)
        p_1 = z_0
    else
        beta_k = dot(r_k, z_k) / dot(r_{k-1}, z_{k-1})
        p_{k+1} = z_k + beta_k * p_k
    end if
    Ap_{k+1} = A*p_{k+1}
    alpha_{k+1} = (r_k, z_k) / (p_{k+1}, Ap_{k+1})
    x_{k+1} = x_k + alpha_{k+1} * p_{k+1}
    r_{k+1} = r_k - alpha_{k+1} * Ap_{k+1}
    if (||r_k|| < absTol) break with convergence
    k=k+1
end
```
`where A = L+D+L^T is in CSR format and the preconditioner is M = (D+L)*D^{-1}*(D+L^T)`.
The preconditioner application is the three-step comment in `precon_gauss_seidel`:
```
t = inv(D+L) * r;   // forward triangular solve
t = D*t             // diagonal mv
z = inv(D+U) * t    // backward triangular solve
```
Test problem: 27-point finite-difference stencil for a 3D Laplacian, `size = 16`, `n = size*size*size = 4096`, `nnzUB = 27 * n`; diagonal overwritten with `52.0` to make the matrix diagonally dominant. `maxIter = 500`, `relTol = 1.0e-5`, `absTol = 5.0e-4`.

## oneMKL Routines Used
Header: `#include "oneapi/mkl.hpp"` plus `using namespace oneapi;`. All calls below are quoted as they appear in `sparse_cg.cpp` / `sparse_cg2.cpp`; no explicit template arguments are written at any call site.

Sparse matrix handle lifecycle:
```cpp
oneapi::mkl::sparse::matrix_handle_t A = nullptr;
oneapi::mkl::sparse::init_matrix_handle(&A);
oneapi::mkl::sparse::release_matrix_handle(q, &A, {}).wait();  // normal path
oneapi::mkl::sparse::release_matrix_handle(q, &A).wait();      // catch path (no deps arg)
```

CSR descriptor — **two overloads selected by a preprocessor macro**:
```cpp
#if (INTEL_MKL_VERSION < 20250300)
        auto ev_set = oneapi::mkl::sparse::set_csr_data(q, A, n, n,
                oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {});
#else
        auto ev_set = oneapi::mkl::sparse::set_csr_data(q, A, n, n, nnz,
                oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {});
#endif
```
Argument order (new overload): `(queue, handle, nrows, ncols, nnz, index_base, row_ptr, col_ind, values, deps)`. The pre-20250300 overload omits `nnz`; the source does not otherwise describe it. Both are called with `index_base::zero` and a zero-indexed `ia_d` of length `n+1`, `ja_d`/`a_d` of length `nnz`.

Host-side (no queue, no event) properties, called after `set_csr_data`:
```cpp
oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::symmetric);
oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::sorted);
```

Optimize (analysis) phase, chained by dependency events:
```cpp
auto ev_optSvL = oneapi::mkl::sparse::optimize_trsv(q,
        oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, A, {ev_set});
auto ev_optSvU = oneapi::mkl::sparse::optimize_trsv(q,
        oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, A, {ev_optSvL});
auto ev_optGemv = oneapi::mkl::sparse::optimize_gemv(q,
        oneapi::mkl::transpose::nontrans, A, {ev_optSvU});
```
`optimize_trsv` order: `(queue, uplo, transpose, diag, handle, deps)`. `optimize_gemv` order: `(queue, transpose, handle, deps)`.

Triangular solve (preconditioner), where `deps` is the caller's event vector:
```cpp
oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, dataType(1.0) /* alpha */, A, r, t, deps);
oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, dataType(1.0) /* alpha */, A, t, z, {ev_diagmv});
```
`trsv` order: `(queue, uplo, transpose, diag, alpha, handle, x_in, y_out, deps)`. Note the scalar `alpha` sits **between** `diag` and the handle.

Sparse matrix-vector multiply, `y := alpha*op(A)*x + beta*y`:
```cpp
auto ev_r = oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, A,
                              x_d, 0.0, r_d, {ev_optGemv}); // r := A * x
auto ev_Ap = oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans,
        1.0, A, p_d, 0.0, t_d, {ev_p});
```
`gemv` order: `(queue, transpose, alpha, handle, x, beta, y, deps)`.

BLAS Level-1 (all in `oneapi::mkl::blas::`, no `row_major`/`column_major` qualifier):
```cpp
oneapi::mkl::blas::axpby(q, n, 1.0, b_d, 1, -1.0, r_d, 1, {ev_r}); // r := 1 * b + -1 * r
oneapi::mkl::blas::axpby(q, n, 1.0, z_d, 1, rTz / oldrTz, p_d, 1, {ev_rtz}); // p := 1*z + beta*p
oneapi::mkl::blas::nrm2(q, n, r_d, 1, normr_d, {ev_r});
oneapi::mkl::blas::dot(q, n, r_d, 1, z_d, 1, rtz_d, {ev_r, ev_z});
oneapi::mkl::blas::copy(q, n, z_d, 1, p_d, 1, {ev_z, ev_rtz});
oneapi::mkl::blas::axpy(q, n, rTz / pAp, p_d, 1, x_d, 1, {});
oneapi::mkl::blas::axpy(q, n, -rTz / pAp, t_d, 1, r_d, 1, {});
```
Order: `axpby(queue, n, alpha, x, incx, beta, y, incy, deps)` computes `y := alpha*x + beta*y`; `axpy(queue, n, alpha, x, incx, y, incy, deps)` computes `y := alpha*x + y`; `copy(queue, n, x, incx, y, incy, deps)`; `dot(queue, n, x, incx, y, incy, result, deps)`; `nrm2(queue, n, x, incx, result, deps)`. **`dot` and `nrm2` write their scalar result into device USM memory** (`rtz_d`, `pAp_d`, `normr_d`), returning an event — they are not host-returning.

Enums used verbatim: `oneapi::mkl::index_base::zero`, `oneapi::mkl::transpose::nontrans`, `oneapi::mkl::uplo::lower`, `oneapi::mkl::uplo::upper`, `oneapi::mkl::diag::nonunit`, `oneapi::mkl::sparse::property::symmetric`, `oneapi::mkl::sparse::property::sorted`.

## Key Code Patterns
**Queue + device.** `sycl::device my_dev{sycl::default_selector_v};`, then a queue with an async handler:
```cpp
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try { std::rethrow_exception(e); }
        catch (sycl::exception const &e) { std::cout << "Caught asynchronous SYCL exception during sparse CG:\n" << e.what() << std::endl; }
    }
};
sycl::queue q(dev, exception_handler);
```
No in-order queue property is requested; ordering is expressed entirely by `deps` event lists.

**USM allocation.** `sycl::malloc_host<T>(count, q)` for staging and `sycl::malloc_device<T>(count, q)` for operands; every allocation is null-checked and throws `std::runtime_error("Failed to allocate host side USM memory")` / `"...device side USM memory"`. Scratch scalars are **padded to cache-line width and scattered**. `sparse_cg.cpp`:
```cpp
const intType width = 8; // width * sizeof(dataType) >= cacheline size (64 Bytes)
dataType *temp_d = sycl::malloc_device<dataType>(3*width, q);
dataType *temp_h = sycl::malloc_host<dataType>(3*width, q);
dataType *normr_h  = temp_h;
dataType *rtz_h    = temp_h+1*width;
dataType *pAp_h    = temp_h+2*width;
dataType *normr_d  = temp_d;
dataType *rtz_d    = temp_d+1*width;
dataType *pAp_d    = temp_d+2*width;
```
`sparse_cg2.cpp` uses a `4*width` block and adds `oldrtz_d`:
```cpp
dataType *temp_d = sycl::malloc_device<dataType>(4*width, q);
dataType *temp_h = sycl::malloc_host<dataType>(4*width, q);
dataType *normr_h  = temp_h;
dataType *normr_d  = temp_d;
dataType *rtz_d    = temp_d+1*width;
dataType *oldrtz_d = temp_d+2*width;
dataType *pAp_d    = temp_d+3*width;
```

**Host↔device transfer.** `q.copy(ia_h, ia_d, n+1).wait();` … `q.copy(rtz_d, rtz_h, 1, {ev_rtz}).wait();` — result scalars are pulled back with an event dependency and an explicit `.wait()`, and only one element is copied per iteration.

**Variant 1 vs variant 2 (the whole point of the two files).** `sparse_cg.cpp` reads `rTz`, `pAp`, `oldrTz` back to host and passes host scalars into BLAS (`rTz / oldrTz`, `rTz / pAp`); each loop iteration contains three explicit `q.copy(...).wait()` host synchronizations (rtz, pAp, normr). `sparse_cg2.cpp` replaces these with two custom device-side kernels so no scalar round-trip is needed:
```cpp
// y = alpha * x + beta1 / beta2 * y      (beta1, beta2 are device arrays, scalar in [0])
sycl::event axpby2(sycl::queue q, const intType n, const dataType alpha, const dataType *x,
                   const dataType *beta1_d, const dataType *beta2_d, dataType *y,
                   const std::vector<sycl::event> &deps = {});
// y = scale * alpha1 / alpha2 * x + y
sycl::event axpy3(sycl::queue q, const intType n, const dataType scale,
                  const dataType *alpha1_d, const dataType *alpha2_d,
                  const dataType *x, dataType *y,
                  const std::vector<sycl::event> &deps = {});
```
called as `axpby2(q, n, dataType(1.0), z_d, rtz_d, oldrtz_d, p_d, {ev_rtz})`,
`axpy3(q, n, dataType(1.0), rtz_d, pAp_d, p_d, x_d, {ev_pAp})`,
`axpy3(q, n, dataType(-1.0), rtz_d, pAp_d, t_d, r_d, {ev_pAp})`. `axpby2`/`axpy3` are **sample-local kernels, not oneMKL routines**.

**Dependency chaining.** `ev_set` → `optimize_trsv(lower)` → `optimize_trsv(upper)` → `optimize_gemv` → `gemv` → `axpby` → `nrm2`; inside the loop `ev_r` → `precon_gauss_seidel` → `dot` → `axpby`/`copy` → `gemv` → `dot` → `axpy` → `nrm2` → next `ev_r`. Inside `precon_gauss_seidel` the chain is `trsv(L)` → custom `diagonal_mv` → `trsv(U)`, and the returned event is `ev_trsvU`. Some calls pass an **empty** deps list (the two `axpy` calls in `sparse_cg.cpp` pass `{}`); the first `trsv` inside `precon_gauss_seidel` forwards the caller's `deps` vector and the second passes `{ev_diagmv}`.

**Exception handling.** The sparse section is wrapped in `try { ... } catch (sycl::exception const &e) { ... } catch (std::exception const &e) { ... }`; both handlers print, `q.wait()`, call `release_matrix_handle(q, &A).wait()`, and `return 1`. The outer `main` repeats the same two catches. Asynchronous failures are routed to the queue's `exception_handler`.

**Double-precision gating.** `if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0)` runs the `double` instantiation only when the device reports double support.

## Build & Run
Linux (`GNUmakefile`): `make` (equivalently `make all` or `make run`) builds and **runs** both binaries. The exact variable definitions and recipes:
```
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="blas,sparse"
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel
sparse_cg: sparse_cg.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
sparse_cg2: sparse_cg2.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
```
`run` target executes `./sparse_cg` then `./sparse_cg2`; `clean` is `-rm -f sparse_cg sparse_cg2 genxir`. Note `-DMKL_ILP64` + `-qmkl-ilp64`: the sample is compiled against the ILP64 oneMKL interface, while the sources instantiate with `std::int32_t` index type.

Windows (`makefile`, NMAKE): `nmake` builds and runs `.\sparse_cg` then `.\sparse_cg2`:
```
SYCL_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,sparse" /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib
icx-cl -fsycl sparse_cg.cpp /Fesparse_cg.exe $(SYCL_OPTS)
icx-cl -fsycl sparse_cg2.cpp /Fesparse_cg2.exe $(SYCL_OPTS)
```
Environment: source the oneAPI `setvars` script first (`. /opt/intel/oneapi/setvars.sh` for a system-wide install, `. ~/intel/oneapi/setvars.sh` for a private install). Device selection: `ONEAPI_DEVICE_SELECTOR="*:cpu"` or `"*:gpu"`; otherwise the code uses `sycl::default_selector_v`.

Expected output (from README, on an Intel(R) Data Center GPU Max 1550): banner `# Sparse Preconditioned Conjugate Gradient Solver with USM` (and `... with USM 2`), `Running tests on <device>.`, then per precision `sparse PCG parameters:` with `A size: (4096, 4096)`, `Preconditioner = Symmetric Gauss-Seidel`, `max iterations = 500`, `relative tolerance limit = 1e-05`, `absolute tolerance limit = 0.0005`, six lines of `relative norm of residual on <k> iteration: <value>`, and the success line `Preconditioned CG process has successfully converged in absolute error in 6 steps with` followed by `relative error ||r||_2 / ||r_0||_2 = 1.86945e-05 > 1e-05` and `absolute error ||r||_2 = 0.000149556 < 0.0005`. Both precisions print near-identical residuals. `run_sparse_pcg_example` returns 0 only when its local `good` flag was set, but `main` ignores both call results and returns 0 (it returns 1 only if a `sycl::exception` or `std::exception` escapes).

## Gotchas & Invariants
- `set_csr_data`'s argument list is **version-dependent**: the `nnz` parameter exists only for `INTEL_MKL_VERSION >= 20250300`. The source switches between the two forms with `#if`; copy that `#if` verbatim rather than picking one form.
- `index_base::zero` is required by the data: `const intType nnz = ia_h[n]; // assumes zero indexing` and `generate_sparse_matrix` writes `ia[0] = index` with `index = 0`.
- Order matters: `init_matrix_handle` → `set_csr_data` → `set_matrix_property` → `optimize_trsv`/`optimize_gemv` → `trsv`/`gemv`. The two `set_matrix_property` calls state `symmetric` and `sorted`; the source does not describe what either property changes.
- `optimize_trsv` must be called separately for `uplo::lower` and `uplo::upper`; the two calls are chained (`ev_optSvU` depends on `ev_optSvL`).
- `trsv` places the scalar `alpha` after `diag` and before the matrix handle — easy to mis-order against `gemv`, which places `alpha` after `transpose`.
- The sample relies on the diagonal being stored explicitly in the CSR rows: it derives `d_d`/`invd_d` by scanning each row for `ja_d[i] == row`. Rows without a stored diagonal would leave those entries uninitialized (`modify_diagonal` only rewrites an existing entry, it never inserts one).
- `modify_diagonal` asserts `new_diagVal != dataType(0.0)`; the sample uses `dataType(52.0)` to enforce diagonal dominance.
- `dot`, `nrm2` write their result to a **device** pointer; reading it back needs an explicit `q.copy(...).wait()` (three such synchronization points per loop iteration in `sparse_cg.cpp`, plus one before the loop). The sample's `nrm2` result is then passed through `std::sqrt` on the host (`normr = std::sqrt(normr_h[0]);`), and the loop's residual-norm call is issued on `z_d` while its comment reads `// temp_d = ||r_{k+1}||^2`.
- Scratch scalars are spread `width` (`= 8`) elements apart so each occupies its own cache line; the first post-`optimize` transfer copies `12` elements (`q.copy(normr_d, normr_h, 12, {ev_normr})`) while loop transfers copy `1`.
- Every USM allocation must be `sycl::free`d on the queue after `q.wait()`; `temp_h`/`temp_d` sizes must match the alias offsets (`3*width` vs `4*width` differ between the two files).
- `release_matrix_handle` has two forms in use: `(q, &A, {}).wait()` and `(q, &A).wait()`. Both appear; do not assume a single overload.
- Loop termination is `while (normr / normr_0 > relTol && k < maxIter)` plus a `break` on `normr <= absTol`; the reported "converged" branch tests `normr < absTol` and the relative branch `normr / normr_0 <= relTol`.

## Explicit gaps
- The source never states whether the device-side result of `oneapi::mkl::blas::nrm2` is the 2-norm or its square; the sample applies `std::sqrt` to it and labels the buffer `||r||^2` in a comment. Treat the exact semantic of that return value as unestablished by these files.
- No workspace/scratchpad *query* API is used anywhere (no buffer-size query call, no `optimize_*` scratchpad size request). Only manual USM scratch (`temp_d`) appears, so the reader cannot learn oneMKL workspace-query idioms from this sample.
- `optimize_gemv` is only ever called with `transpose::nontrans`; the behavior for other transpose values is not exercised.
- The meaning/requirement of the `property::symmetric` and `property::sorted` flags is asserted only by use; no doc text or observable effect is in these files.
- `precon_jacobi` and `precon_none` are defined but their call sites are commented out; the sample never validates them at runtime.
- The sample directory contains no `CMakeLists.txt` (only `GNUmakefile`, `makefile`, the sources, `utils.hpp`, `README.md`, `License.txt`), so no CMake build path is documented here.
- The `dot`/`nrm2`/`axpby`/`axpy`/`copy` calls carry no explicit template arguments in the source; the exact overload resolution rule (how the scalar type is deduced from the USM pointer type) is not shown.
- The significance of the literal `12` in the first `q.copy(normr_d, normr_h, 12, ...)` is not explained by the source.
- Both files use `sycl::default_selector_v`; no code path constructs a queue with an explicit in-order property, so in-order-vs-out-of-order oneMKL queue requirements cannot be inferred here.
- The source builds with `-DMKL_ILP64`/`-qmkl-ilp64` yet instantiates its index type as `std::int32_t`; the files do not explain what that combination implies for the oneMKL interface.

## Source Map
- `sparse_cg.cpp` — PCG with host-side alpha/beta; all oneMKL sparse + BLAS calls, handle lifecycle, optimize chaining, `std::sqrt`-based convergence reporting.
- `sparse_cg2.cpp` — same algorithm with device-side alpha/beta via local `axpby2`/`axpy3` kernels; adds `oldrtz_d` and a `4*width` scratch block.
- `utils.hpp` — `rand_scalar<>`, `set_fp_value<>` (real and `std::complex<fp>` overloads), `generate_sparse_matrix<>` (27-point 3D Laplacian stencil in 3-array CSR).
- `GNUmakefile` — Linux `icpx -fsycl` build, `-DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl="blas,sparse" -fsycl-device-code-split=per_kernel`; targets `run`/`all`/`clean`.
- `makefile` — Windows NMAKE build with `icx-cl`, `/Qmkl-*` flags, `OpenCL.lib`; targets `run`/`all`/`clean`/`pseudo`.
- `README.md` — purpose, two-stage optimize/sparse workflow note, build instructions, `ONEAPI_DEVICE_SELECTOR` usage, and the reference program output for both binaries.
