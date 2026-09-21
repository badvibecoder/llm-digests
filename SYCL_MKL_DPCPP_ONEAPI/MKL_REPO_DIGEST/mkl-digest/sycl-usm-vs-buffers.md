# USM vs Buffer Memory Models: Cross-Sample Patterns

Scope note: the requested directory `input/oneMKL-samples/sycl-usm-vs-buffers/` does not
exist in this workspace. The nine named files were found at
`input/oneMKL-samples/<sample>/<file>` (paths in Source Map). Everything below is
verbatim from those files plus the per-sample `GNUmakefile`/`makefile`/`README.md`
(and `sparse_conjugate_gradient/utils.hpp`, which `sparse_cg.cpp` includes), except
where a statement is explicitly marked as an inference or listed under 'Explicit gaps'.

## Domain & Purpose
Cross-cutting chapter over oneMKL `rng`, `stats`, `dft`, `vm`, `blas` and `sparse`.
Three of the five sample directories pair the same algorithm twice — a SYCL
buffer/accessor body and a USM (unified shared memory) body — namely
`monte_carlo_pi`, `student_t_test` and `fourier_correlation`. By contrast
`sparse_conjugate_gradient` ships a single USM body (`sparse_cg.cpp`), and both
files in `random_sampling_without_replacement` are USM (see 'Explicit gaps'). Where
a real pair exists, the differences isolate the memory model, the call overload
chosen, and the synchronization mechanism.

## Problem & Math
- Monte Carlo Pi: generate `n_points * 2` uniform floats, count pairs with
  `sycl::length(r) <= 1.0f`, `estimated_pi = n_under_curve / n_points * 4.0`.
- Lottery (partial Fisher-Yates): for each of `num_exp` experiments, produce `m`
  unique values from `1..n`; `j = i + (size_t)(rng_buf[id*m+i] * (float)(n-i))`,
  then `std::swap(local_buf[i], local_buf[j])` for `i` in `0..m-1`.
- Student's t-test: `|mean - expected_mean| * sqrt(n) / sqrt(variance) < threshold`
  with `threshold = 1.95996f`; two-sample case uses pooled
  `(1/n1 + 1/n2) * ((n1-1)^2*variance1 + (n2-1)^2*variance2) / (n1+n2-2)` when the
  variances are within a factor of two, else `sqrt(variance1 + variance2)`.
- Sparse PCG: solve `A*x = b` with `r_0 = b - A*x_0`, `M*z_k = r_k`,
  `M = (D+L)*D^-1*(D+L^T)`, `beta = dot(r_k,z_k)/dot(r_{k-1},z_{k-1})`,
  `alpha = dot(r_k,z_k)/dot(p_{k+1},A*p_{k+1})`; stop on `||r||/||r_0|| > relTol`
  and `k < maxIter`, break when `||r|| < absTol`.
- Fourier correlation: `corr = (1/N) * iDFT(DFT(sig1) * CONJ(DFT(sig2)))`, compared
  against naive `corr[s] = sum_j sig1[j]*sig2[(j-s+N) mod N]`.

## oneMKL Routines Used
RNG engines/distributions — constructed exactly as:
```cpp
oneapi::mkl::rng::philox4x32x10 engine(q, seed);   // seed=7777 (mc_pi), 777 (lottery)
oneapi::mkl::rng::mcg31m1 engine(Q, seed);         // fcorr, seed = time(NULL)
oneapi::mkl::rng::default_engine engine(q, seed);  // t_test, seed=7777 (int)
oneapi::mkl::rng::uniform distr;                   // defaults: float, a=0.0f, b=1.0f
oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
    rng_distribution(-0.00005f, 0.00005f);
oneapi::mkl::rng::gaussian<fp_type> distribution(mean, std_dev);
```
`generate` call sites as written (argument order is always
`(distribution, engine, count, output)` in every sample):
```cpp
auto event = mkl::rng::generate(distr, engine, n_points * 2, rng_ptr);  // USM float*, event captured
mkl::rng::generate(distr, engine, n_points * 2, rng_buf);              // mc_pi.cpp sycl::buffer, return discarded
oneapi::mkl::rng::generate(distribution, engine, n_points, rng_arr0);  // USM float*, return discarded
```
Only the USM-result calls capture an `sycl::event`; the declared return type of the
buffer overload is never spelled out in these files, so no signature is claimed.

Statistics (the `t_test.cpp` buffer body and the `t_test_usm.cpp` USM body otherwise
differ only in the output argument kind and the read-back mechanism):
```cpp
auto dataset =
    oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(1, n, r);
oneapi::mkl::stats::mean(q, dataset, mean);
oneapi::mkl::stats::central_moment(q, mean, dataset, variance);
```
`make_dataset<layout::row_major>(dimension_count, observation_count, data)`; `mean`
takes `(queue, dataset, mean_out)`; `central_moment` takes
`(queue, mean_in, dataset, moment_out)` and is called with the mean, so the value
produced is the variance. In the buffer body `mean`/`variance` are
`sycl::buffer<RealType, 1>` objects read back via `sycl::host_accessor`; in the USM
body they are `sycl::malloc_shared<RealType>(1, q)` scalars.

DFT (identical descriptor in both 1D variants):
```cpp
oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                             oneapi::mkl::dft::domain::REAL> desc(N);
desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f / N);
desc.commit(Q);
evt1 = oneapi::mkl::dft::compute_forward(desc, sig1, {evt1});   // USM, event captured
oneapi::mkl::dft::compute_forward(desc, sig1);                  // buffer, return discarded
oneapi::mkl::dft::compute_backward(desc, corr, {evt}).wait();   // USM
oneapi::mkl::dft::compute_backward(desc, corr);                 // buffer
```
VM, BLAS, sparse (verbatim call sites):
```cpp
oneapi::mkl::vm::mulbyconj(Q, N / 2 + 1, sig1_cplx, sig2_cplx, corr_cplx);  // buffer variant
oneapi::mkl::blas::nrm2(Q, N, sig1, 1, norm_sig1, {evt});          // USM, event captured
oneapi::mkl::blas::nrm2(Q, N, sig1, 1, temp);                      // buffer variant, return discarded
oneapi::mkl::blas::axpby(q, n, 1.0, b_d, 1, -1.0, r_d, 1, {ev_r}); // y := a*x + b*y
oneapi::mkl::blas::dot(q, n, r_d, 1, z_d, 1, rtz_d, {ev_r, ev_z});
oneapi::mkl::blas::copy(q, n, z_d, 1, p_d, 1, {ev_z, ev_rtz});
oneapi::mkl::blas::axpy(q, n, rTz / pAp, p_d, 1, x_d, 1, {});
oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, dataType(1.0) /* alpha */, A, r, t, deps);
oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, A,
        x_d, 0.0, r_d, {ev_optGemv});                               // y := alpha*A*x + beta*y
```
The USM `mulbyconj` call site passes an explicit dependency list and casts its
pointers inline: `oneapi::mkl::vm::mulbyconj(Q, N / 2 + 1,
reinterpret_cast<std::complex<float>*>(sig1), ... , {evt1, evt2})`.

## Key Code Patterns
Queue creation, by sample (the selector is always `sycl::default_selector_v`):
```cpp
auto exception_handler = [&](sycl::exception_list exceptions) { /* rethrow, terminate */ };
sycl::queue q(sycl::default_selector_v, exception_handler);   // mc_pi, lottery, t_test
sycl::queue Q(sycl::default_selector_v);                      // fcorr (no handler)
sycl::device my_dev{sycl::default_selector_v};                // sparse_cg main
sycl::queue q(dev, exception_handler);                        // sparse_cg, dev = const sycl::device& parameter
```
USM allocation vs buffer construction:
```cpp
float* rng_ptr = sycl::malloc_shared<float>(n_points * 2, q);   // shared w/ host
float* rng_buf = sycl::malloc_device<float>(m * num_exp, q);    // device only
size_t *n_under_curve = sycl::malloc_host<size_t>(1, q);        // host
sycl::buffer<float, 1> rng_buf(n_points * 2);
sycl::buffer<size_t> count_buf{ &n_under_curve, 1 };            // wraps host scalar
```
Reduction — the operand and the dependency channel both change with the model:
```cpp
auto reductor = sycl::reduction(n_under_curve, size_t(0), std::plus<size_t>{});
q.parallel_for(sycl::range<1>(n_points / count_per_thread), event, reductor, [...]);
...
auto reductor = sycl::reduction(count_buf, h, size_t(0), std::plus<size_t>());
h.parallel_for(sycl::range<1>(n_points / count_per_thread), reductor, [...]);
```
Device-side data read in the USM body needs an explicit global pointer; the buffer
body uses the accessor itself:
```cpp
r.load(i + item.get_id(0) * count_per_thread, sycl::global_ptr<float>(rng_ptr));
r.load(i + item.get_id(0) * count_per_thread,
       rng_acc.template get_multi_ptr<sycl::access::decorated::yes>());
```
Synchronization: only the USM form of `mkl::rng::generate` captures the returned
event in these samples, and the USM code waits on it (`event.wait_and_throw()`,
`mc_pi_usm` instead passes the event as a dependency of `q.parallel_for`); the buffer
body (`mc_pi.cpp`) obtains the generated data through an accessor, so ordering comes
from the buffer dependency. `sparse_cg` threads explicit event lists
(`{ev_trsvL}`, `{ev_p}`, `{ev_r, ev_z}`) and inserts host synchronization points:
```cpp
q.copy(normr_d, normr_h, 12, {ev_normr}).wait();
normr = std::sqrt(normr_h[0]);
```
Sparse handle lifecycle and the two-stage optimize/execute pattern:
```cpp
oneapi::mkl::sparse::matrix_handle_t A = nullptr;
oneapi::mkl::sparse::init_matrix_handle(&A);
auto ev_set = oneapi::mkl::sparse::set_csr_data(q, A, n, n, nnz,
        oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {});   // #else branch, INTEL_MKL_VERSION >= 20250300
oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::symmetric);
oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::sorted);
auto ev_optSvL = oneapi::mkl::sparse::optimize_trsv(q,
        oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, A, {ev_set});
auto ev_optSvU = oneapi::mkl::sparse::optimize_trsv(q,
        oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, A, {ev_optSvL});
auto ev_optGemv = oneapi::mkl::sparse::optimize_gemv(q,
        oneapi::mkl::transpose::nontrans, A, {ev_optSvU});
oneapi::mkl::sparse::release_matrix_handle(q, &A, {}).wait();
```
Buffer reinterpret for the complex backward domain, and its USM analogue:
```cpp
auto sig1_cplx = sig1.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
reinterpret_cast<std::complex<float>*>(sig1)   // USM
```
USM pointer-kind check used before launching a kernel over USM pointers:
```cpp
sycl::usm::alloc alloc_type = sycl::get_pointer_type(x, Q.get_context());
return (alloc_type == sycl::usm::alloc::shared
        || alloc_type == sycl::usm::alloc::device);
```

## Build & Run
Every sample: source the oneAPI environment first (`. /opt/intel/oneapi/setvars.sh`
for a system-wide install or `. ~/intel/oneapi/setvars.sh` for a private one), then
`make` in the sample directory (`nmake` on Windows). Toolchain is `icpx` on Linux,
`icx-cl` on Windows. The flag sets are **not** shared across samples; verbatim from
each Linux `GNUmakefile`:
```
monte_carlo_pi:
  MKL_COPTS  = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng
  DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations $(MKL_LIBS)
random_sampling_without_replacement:
  MKL_COPTS  = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng
  DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations
               (same as monte_carlo_pi except that $(MKL_LIBS) is omitted)
student_t_test:
  MKL_COPTS  = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="stats,rng"
  DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations
sparse_conjugate_gradient:
  MKL_COPTS  = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="blas,sparse"
  DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel   (no -fno-sycl-early-optimizations)
fourier_correlation:
  DPCPP_OPTS = -DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl="blas,dft,rng,vm"   (no MKL_COPTS variable)
```
The compile rule is the same everywhere: `icpx $< -fsycl -o $@ $(DPCPP_OPTS)`.
Windows `makefile` flags, also per sample (note `sparse_conjugate_gradient` names its
variable `SYCL_OPTS`, not `DPCPP_OPTS`):
```
monte_carlo_pi, random_sampling_without_replacement (DPCPP_OPTS):
  /I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=rng /DMKL_ILP64 /EHsc
  -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib
student_t_test (DPCPP_OPTS): the same, with /Qmkl-sycl-impl="stats,rng"
sparse_conjugate_gradient (SYCL_OPTS):
  /I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,sparse" /EHsc
  -fsycl-device-code-split=per_kernel OpenCL.lib        (no /DMKL_ILP64, no -fno-sycl-early-optimizations)
fourier_correlation (DPCPP_OPTS):
  -DMKL_ILP64 -I"%MKLROOT%\include" /Qmkl-ilp64 /Qmkl-sycl-impl="blas,dft,rng,vm" OpenCL.lib /EHsc
  (no -fsycl-device-code-split, no -fno-sycl-early-optimizations)
```
Binaries and targets for the nine files: `mc_pi`, `mc_pi_usm`; `lottery`,
`lottery_usm`; `t_test`, `t_test_usm`; `fcorr_1d_buff` (built from
`fcorr_1d_buffers.cpp`), `fcorr_1d_usm`; `sparse_cg`. The default target is `run`
for every sample except `fourier_correlation`, whose default is `run_all`. `run` /
`run_all` build and then execute their prerequisites — which also include sibling
programs outside this chapter: `mc_pi_device_api`, `lottery_device_api`,
`sparse_cg2` and `fcorr_2d_usm`. `make clean` removes the generated binaries.
`run_all` for `fourier_correlation` passes `4096` to `fcorr_1d_buff` and
`fcorr_1d_usm` (the Windows `makefile` passes `1024`).
Device selection (stated in all five sample READMEs): the samples use the default
SYCL device, and `ONEAPI_DEVICE_SELECTOR` can be set to `"*:cpu"` or `"*:gpu"` to
override it. Optional CLI args: mc_pi and t_test take `n_points` (t_test also
`mean`, `std_dev`), lottery takes `m n num_exp`, `fcorr_1d_*` take `N` (default 32
in the source; `make run_all` passes 4096). Expected output markers: `TEST PASSED` /
`TEST FAILED` (mc_pi, lottery, t_test); `Running on: <device name>` plus
`Max difference between naive and Fourier-based calculations : ...` (fcorr);
`Preconditioned CG process has successfully converged ...` (sparse_cg).

## Gotchas & Invariants
- `count_per_thread = 32` in both mc_pi bodies and the launch is
  `range<1>(n_points / count_per_thread)`. Deriving from those two facts: the kernels
  read exactly `(n_points / 32) * 32` values, so an `n_points` that is not a multiple
  of 32 leaves the trailing points unexamined; the source never validates this.
- `fcorr_1d_*` requires `N >= 8` (explicit `throw std::invalid_argument`) and sizes
  all signal/corr allocations as `2 * (N / 2 + 1)` floats to hold the
  `(N/2 + 1)` complex values of the backward domain.
- Forward DFT is unscaled; the inverse must be scaled by setting
  `config_param::BACKWARD_SCALE` to `1.0f / N` **before** `desc.commit(Q)`.
- `oneapi::mkl::vm::mulbyconj` operates on `N / 2 + 1` complex elements, not `N`:
  the forward real-to-complex transform stores `N/2 + 1` coefficients.
- `oneapi::mkl::sparse::set_csr_data` argument list is version-gated by
  `#if (INTEL_MKL_VERSION < 20250300)`: the older form is
  `set_csr_data(q, A, n, n, oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {})`
  and the newer (`#else`) form inserts `nnz` before `index_base`; the chapter's
  verbatim excerpt above is the `#else` form. Indexing throughout `sparse_cg`
  assumes zero-based rows (`nnz = ia_h[n]; // assumes zero indexing`).
- `A` must be released on every path. Both `catch` blocks in `sparse_cg` call
  `q.wait()` then `oneapi::mkl::sparse::release_matrix_handle(q, &A).wait()`.
- The Gauss-Seidel preconditioner assumes the diagonal exists; `modify_diagonal`
  asserts `new_diagVal != dataType(0.0)` and `sparse_cg` calls it with
  `dataType(52.0)` "to make the matrix diagonally dominant" before solving (the
  stencil built by `generate_sparse_matrix` in `utils.hpp` has diagonal entries
  `fp(26.0)` and off-diagonal entries `fp(-1.0)`).
- `sparse_cg` convergence test is on `normr`, and the `nrm2` at the end of the loop
  is taken over `z_d` (label comment says `||r_{k+1}||^2`); the loop's host scalar
  copies (`q.copy(..., {ev}).wait()`) are real synchronization points that serialize
  the iteration.
- `t_test` divides by `std::sqrt(variance[0])` and the source contains no guard
  against `variance[0] == 0`; the samples do not establish what happens if the
  generated data has zero variance.
- `lottery`/`mc_pi` guard `m == 0 || n == 0 || num_exp == 0 || m > n` and fall back
  to defaults; `mc_pi`/`t_test` treat `n_points == 0` as "use default".
- `sycl::buffer<size_t> count_buf{ &n_under_curve, 1 }` and later reads of
  `n_under_curve` after the enclosing scope closes are only valid because the buffer
  destructor (end of scope) copies back to host.

## Explicit gaps
- The named directory `input/oneMKL-samples/sycl-usm-vs-buffers/` and its
  `mc_pi.cpp`, `mc_pi_usm.cpp`, `lottery.cpp`, `lottery_usm.cpp`, `t_test.cpp`,
  `t_test_usm.cpp`, `sparse_cg.cpp`, `fcorr_1d_usm.cpp`, `fcorr_1d_buffers.cpp`
  do not exist under that path; the identically named files used here live in
  per-sample directories.
- `lottery.cpp` is labelled "Buffer Api" in its banner but its body is the same USM
  implementation as `lottery_usm.cpp` (the two files differ only in that banner
  string and one extra `Results with Host API:` print in the USM file): it uses
  `sycl::malloc_device`/`sycl::malloc_shared` and no `sycl::buffer`. The source
  therefore does **not** establish a buffer-based lottery implementation, and its
  banner text is not a reliable indicator.
- Only three of the five directories actually provide a buffer body / USM body pair
  (monte_carlo_pi, student_t_test, fourier_correlation); `sparse_conjugate_gradient`
  provides only `sparse_cg.cpp` and `random_sampling_without_replacement` provides
  two USM files, so no cross-memory-model comparison can be drawn for those two from
  the assigned files.
- Sibling sources exist in the same directories but were outside the nine assigned
  files and are not covered here: `mc_pi_device_api.cpp`, `lottery_device_api.cpp`,
  `sparse_cg2.cpp`, `fcorr_2d_usm.cpp`.
- Buffer overloads are only established for `generate`, `mean`, `central_moment`,
  `nrm2`, `compute_forward`, `compute_backward`, `mulbyconj`. The USM overloads of
  `stats::mean` / `stats::central_moment` are the ones exercised in `t_test_usm.cpp`;
  no other overload sets were observed.
- Exact declared signatures/template parameter lists for `mean`, `central_moment`,
  `stats::make_dataset`, `optimize_trsv`, `optimize_gemv`, `set_matrix_property`,
  `init_matrix_handle`, `release_matrix_handle` and `vm::mulbyconj` are not present
  in these files; only call sites with deduced template arguments are visible, so
  no complete signature is claimed. In particular the template parameter of
  `mulbyconj` is never written, and the return-type difference between the USM and
  buffer forms of `generate`/`nrm2` is inferred only from whether the sample
  captures the result.
- The semantics of the `INTEL_MKL_VERSION < 20250300` branch (which overload is
  correct for which oneMKL release) are not explained in the sample.
- `fcorr_1d_usm.cpp` declares `std::uint32_t seed = static_cast<std::uint64_t>(time(NULL));`
  while `fcorr_1d_buffers.cpp` declares `std::uint64_t seed = ...`; the samples
  disagree, so the seed type expected by `oneapi::mkl::rng::mcg31m1` is not
  established by them.
- `nnzUB = 27 * n` is only the allocation upper bound for the 27-point stencil; the
  value actually used (`nnz = ia_h[n]`) is produced by `generate_sparse_matrix` in
  `sparse_conjugate_gradient/utils.hpp`, whose body is available: it writes
  `ia[current_row + 1] = nnz + index` with `index = 0`. The samples do not state
  whether `nnz` can be strictly less than `nnzUB` here.
- No CMake build files were present in these sample directories; only
  `GNUmakefile` (GNU make) and `makefile` (NMAKE) were used.
- `sparse_cg` calls `set_matrix_property(A, ...::symmetric)` and `...::sorted`
  after `set_csr_data`; whether these must precede the `optimize_*` calls in
  general (vs. only in this sample) is not stated.

## Source Map
- `input/oneMKL-samples/monte_carlo_pi/mc_pi.cpp` — buffer-body Monte Carlo Pi (RNG + SYCL reduction over accessor).
- `input/oneMKL-samples/monte_carlo_pi/mc_pi_usm.cpp` — USM-body Monte Carlo Pi (shared `float*` + shared USM reduction).
- `input/oneMKL-samples/random_sampling_without_replacement/lottery.cpp` — banner says "Buffer Api"; body is USM/USM-paged partial Fisher-Yates.
- `input/oneMKL-samples/random_sampling_without_replacement/lottery_usm.cpp` — same USM body as lottery.cpp (banner + one extra print differ); device USM RNG buffer + `local_accessor` shuffle.
- `input/oneMKL-samples/student_t_test/t_test.cpp` — buffer-body t-test using buffers + `host_accessor` for mean/variance.
- `input/oneMKL-samples/student_t_test/t_test_usm.cpp` — USM-body t-test using `malloc_shared` scalars for mean/variance.
- `input/oneMKL-samples/sparse_conjugate_gradient/sparse_cg.cpp` — sparse `gemv`/`trsv` PCG with symmetric Gauss-Seidel preconditioner, BLAS vector ops, host-side alpha/beta.
- `input/oneMKL-samples/sparse_conjugate_gradient/utils.hpp` — included by `sparse_cg.cpp`; defines `set_fp_value` and the 27-point-stencil CSR builder `generate_sparse_matrix`.
- `input/oneMKL-samples/fourier_correlation/fcorr_1d_usm.cpp` — 1D Fourier correlation with USM arrays, event chaining, `vm::mulbyconj`, in-place DFT.
- `input/oneMKL-samples/fourier_correlation/fcorr_1d_buffers.cpp` — same algorithm with `sycl::buffer` + `reinterpret<std::complex<float>,1>`.
- `input/oneMKL-samples/{monte_carlo_pi,random_sampling_without_replacement,student_t_test,sparse_conjugate_gradient,fourier_correlation}/{GNUmakefile,makefile,README.md}` — build flags, targets, device selector and expected output lines quoted above.
