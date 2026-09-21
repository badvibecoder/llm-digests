# Fourier Correlation

## Domain & Purpose

oneMKL domains touched: **DFT** (`oneapi/mkl/dft.hpp`), **RNG** (`oneapi/mkl/rng.hpp`), **VM** (`oneapi/mkl/vm.hpp`), **BLAS** (`oneapi/mkl/blas.hpp`). One sample computes the periodic cross-correlation of two real 1D signals (buffer and USM variants) and one of two 2D images (USM) on the SYCL device (RNG fill, norms, DFTs, component-wise product, plus a naive reference kernel), then compares the DFT result against the naive SYCL kernel in a host-side verification loop.

## Problem & Math

Find the integer shift(s) maximizing the cross-correlation of two real periodic signals.

1D: `corr[s] = \sum_{j = 0}^{N-1} sig1[j] * sig2[(j - s + N) mod N]`, `0 <= s < N`.

2D: `corr(s, p) = \sum_{j=0}^{n_rows-1} \sum_{k=0}^{n_cols-1} img1(j,k) * img2((j-s+n_rows) mod n_rows, (k-p+n_cols) mod n_cols)`.

Both are evaluated via `corr = (1 / prod(n_i)) * iDFT(DFT(u) * CONJ(DFT(v)))` (`CONJ` is the complex conjugate, `*` the component-wise product), and reported as a normalized score `rho = (max_s c_s / prod(n_i) - avg_u*avg_v) / (sigma_u*sigma_v)`.

## oneMKL Routines Used

RNG engine construction (1D variants; `Q` first, `seed` second):

```cpp
oneapi::mkl::rng::mcg31m1 engine(Q, seed);  // Initialize RNG engine
oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
    rng_distribution(-0.00005f, 0.00005f);
```

RNG generation — USM form takes a raw pointer, buffer form takes `sycl::buffer`; both take `N` as the 3rd argument:

```cpp
auto evt1 = oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1);  // float* sig1
oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1);              // sycl::buffer<float> sig1
```

BLAS L2 norm — the USM variant captures the returned event in `evt1`, the buffer variant discards it; `incx` is passed literally as `1`:

```cpp
evt1 = oneapi::mkl::blas::nrm2(Q, N, sig1, 1, norm_sig1, {evt});  // norm_sig1 is float* (USM, length 1)
oneapi::mkl::blas::nrm2(Q, N, sig1, 1, temp);                     // temp is sycl::buffer<float>{1}
```

DFT descriptor — template parameters are `<precision, domain>`, constructor arg is the transform size (1D: scalar `N`; 2D: brace-init `{n_rows, n_cols}`):

```cpp
oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                             oneapi::mkl::dft::domain::REAL> desc(N);
oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                             oneapi::mkl::dft::domain::REAL> desc({n_rows, n_cols});
```

Config + commit — `BACKWARD_SCALE` is the only config param used anywhere in the sample:

```cpp
desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f / N);        // 1D
desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f / num_elem); // 2D, num_elem = n_rows*n_cols
desc.commit(Q);
```

Forward/backward transforms — in-place, data-only overload and data + dependency-events overload both used; USM form passes `float*`, buffer form passes `sycl::buffer<float>`:

```cpp
evt1 = oneapi::mkl::dft::compute_forward(desc, sig1, {evt1});  // USM + deps, returns sycl::event
oneapi::mkl::dft::compute_forward(desc, sig1);                 // buffer, no deps
oneapi::mkl::dft::compute_backward(desc, corr, {evt}).wait();  // USM + deps
oneapi::mkl::dft::compute_backward(desc, corr);                // buffer, no deps
```

VM component-wise `a * conj(b)`; the count passed is `N / 2 + 1`, i.e. the number of complex coefficients in the backward domain (not the `N` real values):

```cpp
evt = oneapi::mkl::vm::mulbyconj(
    Q, N / 2 + 1,
    reinterpret_cast<std::complex<float>*>(sig1),
    reinterpret_cast<std::complex<float>*>(sig2),
    reinterpret_cast<std::complex<float>*>(corr), {evt1, evt2});
```

In the buffer variant the inputs are the complex views returned by `reinterpret<std::complex<float>, 1>` over the float buffers (see below), and no dependency list is passed: `oneapi::mkl::vm::mulbyconj(Q, N / 2 + 1, sig1_cplx, sig2_cplx, corr_cplx);`

Argument order summary (as used): `generate(distr, engine, count, out)`, `nrm2(queue, n, x, incx, result[, deps])`, `compute_forward(desc, inout[, deps])`, `compute_backward(desc, inout[, deps])`, `mulbyconj(queue, n, a, b, y[, deps])`.

## Key Code Patterns

**Queue setup** — default selector, no properties, out-of-order (default) in all three programs:

```cpp
sycl::queue Q(sycl::default_selector_v);
std::cout << "Running on: "
          << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
```

**USM allocation sized for both domains.** The real forward domain needs `N` floats; the complex backward domain needs `N/2+1` complex values, i.e. `2*(N/2+1)` floats — which the source comments note equals `N+1` for odd `N` and `N+2` for even `N` — so the complex size dominates, and one allocation serves both in-place:

```cpp
auto sig1 = sycl::malloc_shared<float>(2 * (N / 2 + 1), Q);
auto corr = sycl::malloc_shared<float>(2 * (N / 2 + 1), Q);
auto naive_corr = sycl::malloc_shared<float>(N, Q);
```

**USM accessibility checks before launching kernels** (guards naive path and host printing):

```cpp
template <typename T>
static bool is_device_accessible(const T* x, const sycl::queue& Q) {
    sycl::usm::alloc alloc_type = sycl::get_pointer_type(x, Q.get_context());
    return (alloc_type == sycl::usm::alloc::shared
            || alloc_type == sycl::usm::alloc::device);
}
```

`fcorr_2d_usm.cpp` also defines `is_host_accessible`, identical but accepting `sycl::usm::alloc::host` instead of `device`.

**Event chaining / ordering.** The queue is created with no properties (`sycl::queue Q(sycl::default_selector_v);`). On the USM paths the `nrm2`, `compute_forward`, `compute_backward`, and `mulbyconj` calls receive predecessor events as an initializer list of `sycl::event`; the two `rng::generate` calls in the USM variant pass no dependency list, and the buffer variant passes none at all, relying on `sycl::buffer` accessor dependencies. Final results are forced to the host with `.wait()` on the last event, or by `get_host_access` for buffers.

**Buffer reinterpret for the DFT output domain** (buffer variant, replaces the `reinterpret_cast` of the USM variant):

```cpp
auto sig1_cplx = sig1.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
auto sig2_cplx = sig2.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
auto corr_cplx = corr.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
```

**2D padded column stride.** For in-place real-to-complex and complex-to-real transforms the row start must coincide in both domains, so the forward-domain column stride is double the complex column count:

```cpp
const unsigned col_stride_fwd_domain = 2 * (n_cols / 2 + 1);
// col_stride_fwd_domain / 2 is to be used in backward (complex) domain
```

The descriptor is created with `{n_rows, n_cols}` and — per the source comment — `desc operates in-place with the strides used herein by default`; **no** `set_value` for strides or input/output strides is called anywhere in the sample.

**Exception handling**: whole `main` body wrapped in `try { ... } catch (std::exception const &e) { std::cerr << "An error occurred: " << e.what() << std::endl; return EXIT_FAILURE; }`. Helpers throw `std::invalid_argument` on bad allocation size/accessibility.

**Norm reuse trick**: after the real-to-complex transform, `sig1[0]` / `img1[0]` is the unnormalized DC coefficient (the forward transform is unscaled, so it holds the sum of the original real data), so the mean is read back as `sig1[0] / N` (2D: `img1[0] / num_elem`) without a separate reduction.

## Build & Run

Linux/GNU (`GNUmakefile`) — `make` (default target `run_all`) builds and immediately runs all three programs:

```
DPCPP_OPTS = -DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl="blas,dft,rng,vm"

fcorr_1d_buff: fcorr_1d_buffers.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
fcorr_1d_usm: fcorr_1d_usm.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
fcorr_2d_usm: fcorr_2d_usm.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
```

Targets: `run_all` (default and `all`), `clean` (`-rm -f fcorr_1d_buff fcorr_1d_usm fcorr_2d_usm`); `.PHONY: run_all clean all`. Run lines from `run_all`: `./fcorr_1d_buff 4096`, `./fcorr_1d_usm 4096`, `./fcorr_2d_usm` (2D takes optional rows/cols argv).

Windows/NMAKE (`makefile`) — run `nmake`, clean with `nmake clean` (`del /q fcorr_1d_buff.exe fcorr_1d_usm.exe fcorr_2d_usm.exe`, plus `.exp` files):

```
DPCPP_OPTS=-DMKL_ILP64 -I"%MKLROOT%\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,dft,rng,vm" OpenCL.lib /EHsc
icx-cl -fsycl fcorr_1d_usm.cpp /Fefcorr_1d_usm.exe $(DPCPP_OPTS)
```

Run lines from the NMAKE `run_all`: `.\fcorr_1d_buff 1024`, `.\fcorr_1d_usm 1024`, `.\fcorr_2d_usm`. Targets: `run_all` (default and `all`), `clean`, `pseudo` (`run_all clean all`).

Environment: source `setvars` (`/opt/intel/oneapi/setvars.sh` or `~/intel/oneapi/setvars.sh`) first. Device selection via `ONEAPI_DEVICE_SELECTOR="*:cpu"` or `"*:gpu"`; default is the default SYCL device. Sample requires oneMKL 2024.1 or newer.

Expected output (1D, `4096`): two lines — `Right-shift the second signal 2048 elements to get a maximum, normalized correlation score of 1 (treating the signals as periodic).` and `Max difference between naive and Fourier-based calculations : 2.38419e-07 (verification threshold: 6.66459e-06).` The 2D program additionally prints the two images and `Shift the second signal by translation vector (3, 4) to get a maximum, normalized correlation score of 1 ...` for the default 8x8 case. Exit code 0 = verification passed, 1 = tolerance exceeded, `EXIT_FAILURE` = exception.

## Gotchas & Invariants

- **Allocation size drives correctness of the in-place real transforms.** Buffers/allocations must hold `max(N, 2*(N/2+1)) = 2*(N/2+1)` floats (1D) or `n_rows * (n_cols/2+1) * 2` floats (2D). Allocating only `N` floats would leave no room for the complex backward domain.
- **`mulbyconj` count is `N/2+1` (1D) / `n_rows*(n_cols/2+1)` (2D)** — the number of complex coefficients, not `N` real elements.
- **Default DFT scaling is unit.** Without `config_param::BACKWARD_SCALE = 1.0f/N` (2D: `1.0f/num_elem`) the result is off by a factor of the transform size. `commit(Q)` must be called after `set_value` and before any `compute_forward`/`compute_backward`.
- **`mulbyconj` computes `a * conj(b)`**, so the second argument (the `v` signal / second image) is conjugated; swapping arguments changes the sign/direction of the recovered shift.
- **2D stride invariant**: `col_stride_fwd_domain = 2 * (n_cols / 2 + 1)`; complex rows are indexed with `col_stride_fwd_domain / 2`. Using `n_cols` as a row stride would break the in-place real/complex aliasing requirement stated in the source comment.
- **Buffers variant reinterprets the *whole* allocation** as complex values, `reinterpret<std::complex<float>, 1>(N / 2 + 1)`; the DFT still operates on the underlying float count.
- **naive vs DFT sizes differ**: `naive_corr` (1D) is only `N` floats, and only indices `0..N-1` are compared; the remaining `1` (odd `N`) or `2` (even `N`) floats of `corr` beyond index `N-1` are ignored.
- **Input constraints enforced at runtime**: 1D `N >= 8` and `N <= INT_MAX`; 2D `n_rows >= 6`, `n_cols >= 7` (else `std::invalid_argument`). These come from the hard-coded unit-element patterns, not from oneMKL.
- **Callback/exception style**: no asynchronous error handler is passed to any queue; errors surface as C++ exceptions from oneMKL calls and `sycl::malloc_shared` results are not null-checked.
- USM and buffer overloads of the same oneMKL function are **not interchangeable**: `generate`, `nrm2`, `compute_forward`, `compute_backward` accept either a container (`sycl::buffer`) or a USM pointer; dependency-event lists are passed only on the USM paths in this sample.
- `sycl::free` is called on every USM allocation at the end (1D USM: six frees; 2D: four frees) before `return`.

## Explicit gaps

- The only `set_value` call in any file is `oneapi::mkl::dft::config_param::BACKWARD_SCALE`; no stride, placement, transform-count, or workspace configuration is present. Therefore this sample does **not** establish that the default descriptor layout is generally sufficient, nor how to configure a non-default or out-of-place layout, nor whether/how a scratchpad workspace query is performed.
- The exact overload set and C++ signature text of `oneapi::mkl::dft::compute_forward` / `compute_backward` (buffer container vs USM pointer, with or without a dependency-event list) is only implied by these call sites; no oneMKL header declaring them was read.
- `oneapi::mkl::blas::nrm2` buffer overload: whether the result buffer must be size 1 and whether the call waits are not specified in the source (the sample reads it back with `get_host_access(sycl::read_only)`, which imposes a dependency).
- The 2D program computes its Frobenius norm with a hand-written SYCL reduction (`sycl::reduction(temp, 0.0f, sycl::plus<float>())`) and never calls `oneapi::mkl::blas::nrm2`.
- No numeric data on DFT algorithm selection or on double-precision variants; only `oneapi::mkl::dft::precision::SINGLE` + `oneapi::mkl::dft::domain::REAL` appears.
- Memory-type assumptions are asserted at runtime, but no statement is made about which USM types oneMKL DFT accepts in general.
- No 2D `sycl::buffer`/accessor variant is given, so the sample does not establish whether the 2D descriptor plus `std::complex` buffer reinterpretation works the same way as the 1D buffer case.
- The sample documents `N >= 8` / `n_rows >= 6` / `n_cols >= 7` but not the full set of constraints oneMKL itself imposes on real-transform sizes.

## Source Map

- `fcorr_1d_usm.cpp` — 1D correlation with USM allocations, RNG fill, `nrm2`, `mulbyconj`, real DFT descriptor with `BACKWARD_SCALE`, naive SYCL kernel, host verification and normalized-score report.
- `fcorr_1d_buffers.cpp` — same 1D algorithm using `sycl::buffer`/accessors and `reinterpret<std::complex<float>, 1>` instead of USM pointers/`reinterpret_cast`.
- `fcorr_2d_usm.cpp` — 2D USM correlation with padded `col_stride_fwd_domain = 2*(n_cols/2+1)`, 2D `{n_rows, n_cols}` descriptor, SYCL reduction Frobenius norm, image printing, translation-vector report.
- `README.md` — math definition, algorithm description, output examples, build/run instructions, `ONEAPI_DEVICE_SELECTOR` device selection, oneMKL 2024.1+ requirement.
- `GNUmakefile` — Linux `icpx -fsycl` build rules, `DPCPP_OPTS`, `run_all`/`clean` targets.
- `makefile` — Windows NMAKE rules, `icx-cl -fsycl` flags, `run_all`/`clean`/`pseudo` targets.
