# oneMKL Digest — Master Index

Compressed, machine-ingestible reference distilled from the **Intel oneAPI oneMKL samples** repository (`input/oneMKL-samples/`, a checkout of `oneapi-src/oneMKL-samples`). Written to give a future LLM session accurate oneMKL/SYCL context with minimal hallucination.

**Scope:** technical knowledge only — oneMKL API surface as actually used, SYCL integration mechanics, build/link conventions, and failure modes. Legal text, licensing, authorship, and marketing exposition are deliberately excluded.

## How to use this digest

1. Read this file for orientation, build conventions, and cross-cutting rules.
2. Open only the chapter(s) for the domain you need (table below).
3. Treat every chapter's **`## Explicit gaps`** section as binding: it lists what the samples do **not** establish. Do not fill those gaps from memory — the samples never declared those signatures, and guessing there is exactly the failure this digest exists to prevent.
4. Chapters quote call sites, not headers. oneMKL's full declared prototypes are *not* in the sample corpus, so signature details beyond the observed call forms are flagged, not asserted.

## Chapter index

| # | Chapter | oneMKL domain(s) | Sample dir |
|---|---------|------------------|------------|
| 1 | [american_options](mkl-digest/american_options.md) | RNG (host **and** device API), FP64 gating, CMake | `american_options/` |
| 2 | [monte_carlo_european_opt](mkl-digest/monte_carlo_european_opt.md) | RNG (host + device, 3 engines) | `monte_carlo_european_opt/` |
| 3 | [monte_carlo_pi](mkl-digest/monte_carlo_pi.md) | RNG (USM, buffer, device API) | `monte_carlo_pi/` |
| 4 | [random_sampling_without_replacement](mkl-digest/random_sampling_without_replacement.md) | RNG (USM, buffer, device API) | `random_sampling_without_replacement/` |
| 5 | [black_scholes](mkl-digest/black_scholes.md) | RNG | `black_scholes/` |
| 6 | [binomial](mkl-digest/binomial.md) | RNG | `binomial/` |
| 7 | [student_t_test](mkl-digest/student_t_test.md) | Stats + RNG | `student_t_test/` |
| 8 | [matrix_mul_mkl](mkl-digest/matrix_mul_mkl.md) | BLAS (`gemm`) | `matrix_mul_mkl/` |
| 9 | [fourier_correlation](mkl-digest/fourier_correlation.md) | DFT + BLAS + VM + RNG | `fourier_correlation/` |
| 10 | [computed_tomography](mkl-digest/computed_tomography.md) | DFT (real transforms) | `computed_tomography/` |
| 11 | [sparse_conjugate_gradient](mkl-digest/sparse_conjugate_gradient.md) | Sparse + BLAS | `sparse_conjugate_gradient/` |
| 12 | [batched_linear_solver](mkl-digest/batched_linear_solver.md) | LAPACK batched (Fortran + OpenMP offload) | `batched_linear_solver/` |
| 13 | [block_cholesky_decomposition](mkl-digest/block_cholesky_decomposition.md) | BLAS + LAPACK (`potrf`) | `block_cholesky_decomposition/` |
| 14 | [block_lu_decomposition](mkl-digest/block_lu_decomposition.md) | BLAS + LAPACK (`getrf`) | `block_lu_decomposition/` |
| 15 | [sycl-usm-vs-buffers](mkl-digest/sycl-usm-vs-buffers.md) | cross-cutting: USM vs buffer memory models | multiple |
| 16 | [device-api-patterns](mkl-digest/device-api-patterns.md) | cross-cutting: RNG device API in-kernel | multiple |

Chapters 1–14 are per-sample. Chapters 15–16 are cross-cutting syntheses of the same sources and are the best starting point for "how do I call oneMKL from SYCL" questions.

## Find a routine

All routine names below are verbatim as they appear in the samples.

**RNG — host API** (`oneapi::mkl::rng`), include `<oneapi/mkl.hpp>`:

| Routine | Where used |
|---|---|
| `oneapi::mkl::rng::generate` | all RNG samples; 4-arg form `(distribution, engine, count, ptr)` |
| `oneapi::mkl::rng::uniform<T>(a, b)` | monte_carlo_pi, monte_carlo_european_opt, random_sampling_without_replacement, black_scholes, binomial, fourier_correlation |
| `oneapi::mkl::rng::gaussian<T>(mean, stddev)` | student_t_test, monte_carlo_european_opt, american_options |
| `oneapi::mkl::rng::philox4x32x10` | 5 samples (the most common engine) |
| `oneapi::mkl::rng::mrg32k3a` | monte_carlo_european_opt, american_options |
| `oneapi::mkl::rng::mcg59` | monte_carlo_european_opt |
| `oneapi::mkl::rng::mcg31m1` | fourier_correlation |
| `oneapi::mkl::rng::default_engine` | student_t_test |

**RNG — device API** (`oneapi::mkl::rng::device`), include `<oneapi/mkl/rng/device.hpp>` **in addition to** `<oneapi/mkl.hpp>`:

| Routine | Where used |
|---|---|
| `oneapi::mkl::rng::device::generate` | monte_carlo_pi, random_sampling_without_replacement, american_options, device-api-patterns |
| `oneapi::mkl::rng::device::philox4x32x10<VEC_SIZE>` | monte_carlo_pi, random_sampling_without_replacement, monte_carlo_european_opt |
| `oneapi::mkl::rng::device::mrg32k3a` | monte_carlo_european_opt, american_options |
| `oneapi::mkl::rng::device::mcg59` | monte_carlo_european_opt |
| `oneapi::mkl::rng::device::uniform<T>` | monte_carlo_pi, random_sampling_without_replacement |
| `oneapi::mkl::rng::device::gaussian<T>` | monte_carlo_european_opt, american_options |

**Stats** (`oneapi::mkl::stats`):

| Routine | Where used |
|---|---|
| `oneapi::mkl::stats::make_dataset` | student_t_test |
| `oneapi::mkl::stats::mean` | student_t_test |
| `oneapi::mkl::stats::central_moment` | student_t_test |

**BLAS** (`oneapi::mkl::blas`) and LAPACK (`oneapi::mkl::lapack`):

| Routine | Where used |
|---|---|
| `oneapi::mkl::blas::gemm` | matrix_mul_mkl, block Cholesky, block LU |
| `oneapi::mkl::blas::trsm`, `syrk` | block Cholesky |
| `oneapi::mkl::blas::trsm` | block LU |
| `oneapi::mkl::blas::copy`, `swap` | block LU |
| `oneapi::mkl::blas::axpy`, `axpby`, `dot`, `nrm2` | sparse_conjugate_gradient |
| `oneapi::mkl::blas::nrm2` | fourier_correlation |
| `oneapi::mkl::lapack::getrf`, `oneapi::mkl::lapack::getrf_scratchpad_size` | block LU |
| `oneapi::mkl::lapack::potrf`, `oneapi::mkl::lapack::potrf_scratchpad_size` | block Cholesky |

**DFT** (`oneapi::mkl::dft`) and **VM** (`oneapi::mkl::vm`):

| Routine | Where used |
|---|---|
| `oneapi::mkl::dft::descriptor` | computed_tomography, fourier_correlation |
| `oneapi::mkl::dft::descriptor::set_value`, `::commit` | computed_tomography |
| `oneapi::mkl::dft::compute_forward`, `compute_backward` | computed_tomography, fourier_correlation |
| `oneapi::mkl::vm::mulbyconj` | fourier_correlation |

**Sparse** (`oneapi::mkl::sparse`) — see [sparse_conjugate_gradient](mkl-digest/sparse_conjugate_gradient.md) for the full handle lifecycle:

| Routine | Role |
|---|---|
| `oneapi::mkl::sparse::init_matrix_handle` | create the matrix handle |
| `oneapi::mkl::sparse::set_csr_data` | attach CSR arrays (indexing, rows, cols, ptr, indices, values) |
| `oneapi::mkl::sparse::set_matrix_property` | e.g. `property::symmetric`, `property::sorted` |
| `oneapi::mkl::sparse::optimize_trsv`, `optimize_gemv` | per-operation optimization before the compute call |
| `oneapi::mkl::sparse::trsv`, `gemv` | triangular solve / matrix-vector product |
| `oneapi::mkl::sparse::release_matrix_handle` | teardown |

**LAPACK batched (Fortran interface)** — `dgetrf_batch_strided`, `sgetrf_batch_strided`, `dgetrs_batch_strided`, `sgetrs_batch_strided`, `dgemv`, `sgemv`. See [batched_linear_solver](mkl-digest/batched_linear_solver.md).

## Build conventions

**Environment is a prerequisite for every sample.** Source the oneAPI environment first: system-wide `. /opt/intel/oneapi/setvars.sh`, private install `. ~/intel/oneapi/setvars.sh`. Windows builds that reference `$(MKLROOT)` require it to be set.

**Domain selection via the compiler driver.** Most samples do not name oneMKL libraries; they pass a domain list to the DPC++ driver, which is the reusable convention:

| Sample | `-qmkl-sycl-impl=` value |
|---|---|
| black_scholes, binomial, monte_carlo_pi, random_sampling_without_replacement, monte_carlo_european_opt | `rng` |
| matrix_mul_mkl | `blas` |
| computed_tomography | `dft` |
| batched_linear_solver | `lapack` |
| fourier_correlation | `"blas,dft,rng,vm"` |
| sparse_conjugate_gradient | `"blas,sparse"` |
| student_t_test | `"stats,rng"` |
| block_lu_decomposition, block_cholesky_decomposition | `"blas,lapack"` |
| american_options | *(none — CMake names libraries directly)* |

- Combine multiple domains in one quoted, comma-separated list.
- Windows uses the `/Q` form (`/Qmkl-sycl-impl=rng`) and `OpenCL.lib` on the link line.
- **ILP64 is selected everywhere**: `-DMKL_ILP64 -qmkl-ilp64` on Linux, `/DMKL_ILP64 /Qmkl-ilp64` on Windows. Pass both the define and the driver flag, as every sample does.
- Compiler is `icpx` on Linux, `icx`/`icx-cl` on Windows. All C++ samples need `-fsycl` (Windows `-fsycl` plus `/EHsc`).
- Common extras: `-fsycl-device-code-split=per_kernel` (sparse_conjugate_gradient, student_t_test, block samples; Windows matrix_mul_mkl), `-fno-sycl-early-optimizations` (student_t_test).

**Build systems.** Each sample ships `GNUmakefile` (Linux GNU make, default target `run`) and `makefile` (Windows NMAKE, default target `run`). `american_options` is the sole exception: **CMake only** (`cmake -D USE_DEVICE_API=1 ..`), and the only sample where the link line is spelled out as libraries (`sycl mkl_sycl_rng mkl_intel_ilp64 mkl_sequential mkl_core`).

**Device selection is external to the binary.** Samples run on the default SYCL device and expect `ONEAPI_DEVICE_SELECTOR` to choose CPU vs GPU — `"*:cpu"` / `"*:gpu"` in 14 READMEs; `american_options` uses the OpenCL-style `opencl:cpu`. Never hard-code a device.

**Per-sample run commands:** see each chapter's `## Build & Run`.

## FP64 handling — two established patterns

Every numerically serious sample checks for double support; none silently degrades:

1. **Probe then dispatch** (binomial, black_scholes, monte_carlo_european_opt): construct a throwaway queue purely to query the aspect, then instantiate a templated implementation for `double` or fall back to `float` with a printed warning — `test_queue.get_device().has(sycl::aspect::fp64)`; or query the config list — `D.get_info<sycl::info::device::double_fp_config>().size() != 0` (matrix_mul_mkl) / `.empty()` (block samples).
2. **Assert and fail** (american_options): `internal::has_capability_or_fail(stream->get_device(), sycl::aspect::fp64)` throws when absent.

computed_tomography takes a third route, selecting the device through the queue itself: `sycl::queue(sycl::aspect_selector({sycl::aspect::fp64}))`.

## Memory model: USM vs buffers

See [sycl-usm-vs-buffers](mkl-digest/sycl-usm-vs-buffers.md) for the full comparison. The essentials:

- **USM dominates.** Samples favor `sycl::malloc_shared` / `sycl::malloc_device`, sometimes via `sycl::usm_allocator<T, sycl::usm::alloc::shared>` so a `std::vector` is device-accessible. USM memory is passed to oneMKL routines as raw pointers.
- **Buffers appear as parallel variants** in monte_carlo_pi, student_t_test, and fourier_correlation (`fcorr_1d_buffers.cpp`), where oneMKL has buffer overloads. Not every routine demonstrated with USM has a buffer variant shown.
- **Free what you allocate, with the same queue/context**: `sycl::free(ptr, queue)`. Samples that heap-allocate the queue pair it with `delete`.
- **A named pitfall:** `random_sampling_without_replacement/lottery.cpp` prints a "Buffer Api" banner but its body is USM code with no `sycl::buffer` — do not trust that banner as evidence of a buffer implementation.

## Cross-cutting rules and pitfalls

Verified against the sources; each is expanded in the linked chapter.

1. **Queues are out-of-order.** Samples never request `sycl::property::queue::in_order()`. Correctness and timing therefore depend on explicit synchronization — `sycl::event::wait({e1, e2, e3})`, `queue.wait_and_throw()`, `.wait()` on a copy event, or buffer-scope exit. When timing a loop, the wait must be inside the timed region or the measurement is meaningless.
2. **Warm up before timing.** The recurring pattern is one or more untimed iterations to absorb JIT/kernel-preparation cost, then timed iterations. matrix_mul_mkl even estimates the call count needed to keep the device busy for ~1 s and subtracts the tare measurement.
3. **RNG engine state advances across successive `generate` calls.** Reusing one engine for several arrays (black_scholes, binomial) yields different data than regenerating with a fresh engine; reordering the calls changes the results.
4. **Device-API stream offsets must be per-work-item.** The second engine constructor argument partitions the random stream (`id * count`, `id * ITEMS_PER_WORK_ITEM * VEC_SIZE * block_n`, `path * num_timesteps`). Getting it wrong silently correlates streams rather than failing.
5. **Leading dimensions are padded, not the raw matrix size.** matrix_mul_mkl's `nice_ld<T>` rounds up to 512-byte boundaries plus a 256-byte offset, so buffers must be allocated as `ld*K`, not `M*K`. Passing raw dimensions as leading dimensions breaks the layout silently.
6. **Division/divisibility constraints are hard invariants**, often enforced only by arithmetic: binomial asserts `block_size * wg_size == num_steps`; monte_carlo_european_opt derives `global_size` assuming `num_options` is divisible by `ITEMS_PER_WORK_ITEM` and `block_n = path_length / (local_size * VEC_SIZE)`; black_scholes derives the global range from `opt_n / block_size`.
7. **Local-accessor sizes carry off-by-one slack.** binomial's `slm_call` is `wg_size + 1` because the exchange reads `slm_call[local_id + 1]`; shrinking it reads garbage.
8. **`this` cannot be captured in a kernel lambda** — samples copy members to locals first, then capture by value.
9. **Sparse handles have an explicit lifecycle**: init → set data → set properties → `optimize_*` → compute → release. Skipping `optimize_trsv`/`optimize_gemv`, or releasing too early, is the classic misuse.
10. **DFT descriptors need explicit configuration and `commit()`** before `compute_forward`/`compute_backward`; `set_value` (e.g. `BACKWARD_SCALE`) must be applied to the descriptor first.
11. **Pass-through link/define gaps exist in the corpus and are flagged, not fixed**: e.g. `MKL_LIBS` is referenced but never defined in some GNUmakefiles, and `DLL_EXPORT` is used but never defined in binomial. Where a chapter says a flag or symbol is undefined, treat the sample as needing repair, not as documentation of a required macro.
12. **README output blocks are illustrative and hardware-dependent.** Several READMEs disagree with the current source's printed strings or compute call/put results the source never validates. Prefer strings quoted from source over README samples; chapters note the discrepancies they found.

## Coverage and known gaps

- **Digested:** all 15 sample directories under `input/oneMKL-samples/` — 14 per-sample chapters plus `american_options`, and 2 cross-cutting chapters that re-synthesize the same sources from a memory-model and device-API angle. Repository-level READMEs were read for orientation.
- **Excluded by design:** `LICENSE`, per-sample `License.txt`, `third-party-programs.txt`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.github/`, `.git/`, and the `computed_tomography/input.bmp` test image.
- **Not covered by any chapter:** `american_options/src/longstaff_schwartz_svd_2.cu` (the pre-migration CUDA source — a migration reference, not oneMKL/SYCL API surface), and the sibling sources noted as outside their chapter's file set in the cross-cutting chapters (`sparse_cg2.cpp`, `fcorr_2d_usm.cpp`).
- **Systemic gap across every chapter:** the corpus contains sample *call sites*, not oneMKL headers. No chapter can therefore assert a full declared prototype, template parameter list, or overload set. Chapters say so explicitly. If you need exact signatures, consult the oneMKL documentation or headers rather than extrapolating from this digest.
