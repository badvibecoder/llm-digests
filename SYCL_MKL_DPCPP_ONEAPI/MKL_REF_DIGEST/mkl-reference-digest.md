# oneMKL DPC++ Reference Digest — Master Index

Compressed, machine-ingestible reference distilled from the official **Intel oneMKL — Data Parallel C++ Developer Reference (2026.0)**, a 1,215-page PDF (`input/onemkl_developer-reference-dpcpp_2026.0-772045-916342.pdf`).

This is the **API reference** companion to the samples digest at `output/mkl-digest.md`. Use this one for *what a routine is called, what it computes, and what its arguments mean*; use the samples digest for *worked, compiling usage patterns*.

**Scope:** technical reference only — routine namespaces, signatures, include files, parameters, batch/USM/buffer variants, mathematical definitions, scratchpad sizing, and gotchas. Marketing text, legal notices, and product-performance boilerplate are excluded.

## How to use this digest

1. Read this file first for the cross-cutting conventions and the routine index.
2. Open exactly one domain chapter for the routine you need.
3. **All 16 chapters carry an `## Explicit gaps` section.** Those list what the source does not state. Treat them as binding — do not fill them from memory.
4. Chapters use the source's own vocabulary verbatim. Identifiers are never normalized.

### Structural facts worth knowing before you read a chapter

- **Precision/integer selection is a template + `-DMKL_ILP64` concern.** The reference assumes `std::int64_t` dimensions throughout. Build with `-DMKL_ILP64` and the `mkl_intel_ilp64` interface, or dimensions will silently mismatch.
- **Host/buffer and USM duality is universal.** Nearly every runtime routine has a **buffer version** (SYCL `sycl::buffer`, returns `void`) and a **USM version** (raw pointers, returns `sycl::event`). Chapters show both signatures where both exist.
- **Dependency chaining uses `const std::vector<sycl::event> &dependencies = {}`** on USM forms. Buffer forms get their ordering from the SYCL buffer/accessor model instead.
- **Almost every nontrivial routine needs a scratchpad.** A `*_scratchpad_size` query returns the workspace element count, which you allocate yourself and pass as the `scratchpad` argument.
- **Formulas come from images.** In the source PDF, every mathematical definition is a vector/raster image that text extraction drops entirely. They were transcribed here by rendering and reading them. Each formula is labelled with its source page, e.g. `gemm (p145)`. Coverage: **182 of 185 formula pages**, plus the two RNG stream-splitting diagrams described inline in the RNG chapter. The 3 remaining pages (1007, 1008, 1058) are covered narratively in `rng.md`.

## Chapter index

| # | Chapter | Domain | Source pages |
|---|---------|--------|--------------|
| 1 | [foundations](mkl-reference-digest/foundations.md) | Data types, matrix storage, scalar args, error handling, known limitations | 10–29 |
| 2 | [blas-level1](mkl-reference-digest/blas-level1.md) | BLAS vector-vector | 30–72 |
| 3 | [blas-level2](mkl-reference-digest/blas-level2.md) | BLAS matrix-vector | 72–144 |
| 4 | [blas-level3](mkl-reference-digest/blas-level3.md) | BLAS matrix-matrix + `gemm` | 144–192 |
| 5 | [blas-like-extensions](mkl-reference-digest/blas-like-extensions.md) | Batched GEMM/TRSM, `gemm_bias`, `omatcopy`/`imatcopy`/`omatadd` | 192–290 |
| 6 | [compute-modes-reproducibility](mkl-reference-digest/compute-modes-reproducibility.md) | Compute modes, numerical reproducibility | 290–294 |
| 7 | [sparse-blas](mkl-reference-digest/sparse-blas.md) | Sparse handle contract, CSR/COO/CSC/BSR, sparse routines | 294–393 |
| 8 | [lapack-part1](mkl-reference-digest/lapack-part1.md) | LAPACK: factorization, linear equations, SVD | 393–546 |
| 9 | [lapack-part2](mkl-reference-digest/lapack-part2.md) | LAPACK: eigenvalues, orthogonal/unitary, triangular | 546–702 |
| 10 | [vm-part1](mkl-reference-digest/vm-part1.md) | Vector math: arithmetic, power and log functions | 702–840 |
| 11 | [vm-part2](mkl-reference-digest/vm-part2.md) | Vector math: trigonometry, special functions, modes/status | 840–987 |
| 12 | [rng](mkl-reference-digest/rng.md) | RNG engines, service routines, host + device distributions | 987–1096 |
| 13 | [summary-statistics](mkl-reference-digest/summary-statistics.md) | Dataset, moments, variation, skewness, kurtosis, min/max | 1096–1131 |
| 14 | [dft](mkl-reference-digest/dft.md) | DFT descriptors, configuration, forward/backward, distributed | 1131–1183 |
| 15 | [data-fitting](mkl-reference-digest/data-fitting.md) | Spline class, construction, interpolation | 1183–1197 |
| 16 | [device-support-matrix](mkl-reference-digest/device-support-matrix.md) | Appendix A: which routines run on which devices | 1197–1215 |

**Domain boundaries are explicit.** Because the source was chunked at ~900 lines, a few chapters open or close with routines belonging to a neighbouring domain — these are labelled as boundary sections rather than smuggled in silently. Examples: `vm-part1` opens with LAPACK `unmqr`/`unmrq`/`unmtr` entries; `dft.md` closes with `stats::skewness`/`kurtosis`/`min`/`max`/`min_max`; `summary-statistics` closes with RNG/DFT pointers. When a routine is not in the chapter you expect, check the adjacent chapters' boundary sections.

## Routine index

Routine names exactly as documented. `B` = buffer version, `U` = USM version; most routines have both.

### BLAS — [level 1](mkl-reference-digest/blas-level1.md), [level 2](mkl-reference-digest/blas-level2.md), [level 3](mkl-reference-digest/blas-level3.md)

- **Level 1 (16):** `asum` `axpy` `copy` `dot` `sdsdot` `dotc` `dotu` `nrm2` `rot` `rotg` `rotm` `rotmg` `scal` `swap` `iamax` `iamin`
- **Level 2 (25):** `gbmv` `gemv` `ger` `gerc` `geru` `hbmv` `hemv` `her` `her2` `hpmv` `hpr` `hpr2` `sbmv` `spmv` `spr` `spr2` `symv` `syr` `syr2` `tbmv` `tbsv` `tpmv` `tpsv` `trmv` `trsv`
- **Level 3 (11):** `gemm` `hemm` `her2k` `herk` `symm` `syr2k` `syrk` `trmm` `trsm` (plus `trmv`/`trsv` covered at their level-2 entries)

### BLAS-like extensions — [chapter](mkl-reference-digest/blas-like-extensions.md)

`axpby` `axpy_batch` `copy_batch` `dgmm_batch` `gemm_batch` `gemm_bias` `gemmt` `gemv_batch` `syrk_batch` `trsm_batch` `omatcopy` `imatcopy` `omatadd` `omatcopy_batch` `imatcopy_batch` `omatadd_batch`

### Sparse BLAS — [chapter](mkl-reference-digest/sparse-blas.md)

- **Handle lifecycle:** `init_matrix_handle` `release_matrix_handle`
- **Data setters:** `set_csr_data` `set_csc_data` `set_coo_data` `set_bsr_data` `set_matrix_property`
- **Optimize (per-op, before compute):** `optimize_gemv` `optimize_trmv` `optimize_trsv` `optimize_gemm` `optimize_trsm`
- **Compute:** `gemv` `gemvdot` `symv` `trmv` `trsv` `gemm` `trsm`
- **Multi-stage (analyze → get_nnz → compute, for size-varying output):** `omatadd`, `omatconvert` (each with `*_buffer_size`, `*_analyze`, `*_get_nnz`), plus `init_omatadd_descr`/`release_omatadd_descr`, `init_omatconvert_descr`/`release_omatconvert_descr`
- **Matrix-matrix descriptors:** `init_matmat_descr` `set_matmat_data` `get_matmat_data` `release_matmat_descr` `matmat` `matmatd`
- **Other:** `omatcopy` `sort_matrix` `update_diagonal_values`

### LAPACK — [part 1](mkl-reference-digest/lapack-part1.md), [part 2](mkl-reference-digest/lapack-part2.md)

Almost every name has a paired `<name>_scratchpad_size`, and many have `_batch` (buffer strided / USM strided / group) variants.

- **Factorization / equations:** `getrf` `getrfnp` `getri` `getrs` `getrsnp_batch` `gesv` `gels` `geqrf` `gerqf` `gebrd` `gesvd` `gesvda_batch` `geinv_batch`
- **Symmetric/Hermitian:** `potrf` `potri` `potrs` `hetrf` `sytrf` `hetrd` `sytrd` `heevd` `heevx` `syevd` `syevx` `hegvd` `hegvx` `sygvd` `sygvx`
- **Orthogonal/unitary generation and application:** `orgqr` `orgbr` `orgtr` `ormqr` `ormrq` `ormtr` `ungqr` `ungbr` `ungtr` `unmqr` `unmrq` `unmtr`
- **Triangular:** `trtri` `trtrs`

### Vector Math — [part 1](mkl-reference-digest/vm-part1.md), [part 2](mkl-reference-digest/vm-part2.md)

- **Arithmetic:** `add` `sub` `sqr` `mul` `mulbyconj` `conj` `abs` `arg` `linearfrac` `fmod` `remainder` `inv` `div` `hypot`
- **Power / exponential / log:** `sqrt` `invsqrt` `cbrt` `invcbrt` `pow2o3` `pow3o2` `pow` `powx` `powr` `exp` `exp2` `exp10` `expm1` `ln` `log2` `log10` `log1p` `logb`
- **Trigonometric:** `cos` `sin` `sincos` `cis` `tan` `acos` `asin` `atan` `atan2` `cospi` `sinpi` `tanpi` `acospi` `asinpi` `atanpi` `atan2pi` `cosd` `sind` `tand`
- **Hyperbolic:** `cosh` `sinh` `tanh` `acosh` `asinh` `atanh`
- **Error / special functions:** `erf` `erfc` `erfcx` `cdfnorm` `erfinv` `erfcinv` `cdfnorminv` `lgamma` `tgamma` `expint1` `i0` `i1` `j0` `j1` `jn` `y0` `y1` `yn`
- **Rounding / classification:** `floor` `ceil` `trunc` `round` `nearbyint` `rint` `modf` `frac` `copysign` `nextafter` `fdim` `fmax` `fmin` `maxmag` `minmag`
- **Mode / status control:** `set_mode` `get_mode` `set_status` `get_status` `clear_status` `create_error_handler`

### RNG — [chapter](mkl-reference-digest/rng.md)

- **Host engines:** `mrg32k3a` `philox4x32x10` `mcg31m1` `mcg59` `r250` `wichmann_hill` `mt19937` `sfmt19937` `mt2203` `ars5` `sobol` `niederreiter` `nondeterministic`
- **Device engines:** `mrg32k3a` `philox4x32x10` `mcg31m1` `mcg59` `pcg64_dxsm`
- **Service routines:** `leapfrog` `skip_ahead` `save_state` `load_state` `get_state_size`; helpers `engine_descriptor` `engine_accessor` `count_engine_adaptor`
- **Generate:** `generate` (host), `device::generate` / `device::generate_single`
- **Host distributions:** `uniform` (continuous) `gaussian` `exponential` `laplace` `weibull` `cauchy` `rayleigh` `lognormal` `gumbel` `gamma` `beta` `chi_square` `gaussian_mv` `uniform` (discrete) `uniform_bits` `bits` `bernoulli` `geometric` `binomial` `hypergeometric` `poisson` `poisson_v` `negative_binomial` `multinomial`
- **Device distributions:** `uniform` (continuous) `gaussian` `lognormal` `exponential` `uniform` (discrete) `bits` `uniform_bits` `poisson` `bernoulli` `geometric` `beta` `gamma`

### Summary Statistics — [chapter](mkl-reference-digest/summary-statistics.md)

`dataset` `make_dataset` `raw_sum` `central_sum` `raw_moment` `central_moment` `mean` `variation` `skewness` `kurtosis` `min` `max` `min_max` — most with a "with user-provided mean" overload.

### DFT — [chapter](mkl-reference-digest/dft.md)

`precision` `domain` `config_param` `config_value` `descriptor` (`set_value` `get_value` `set_workspace` `commit`) `compute_forward` `compute_backward`; experimental distributed variants under `oneapi::mkl::experimental::dft::` (`distributed_config_param` `distributed_config_value` `distributed_descriptor` `distributed_descriptor::set_value` `get_value` `commit` `compute_forward` `compute_backward`).

### Data Fitting — [chapter](mkl-reference-digest/data-fitting.md)

`spline` class template (`set_partitions` `set_function_values` `set_coefficients` `set_internal_conditions` `set_boundary_conditions` `is_initialized` `get_required_coeffs_size` `construct`) `linear_spline::default_type` `cubic_spline::hermite` `interpolate`; hints `partition_hint` `function_hint` `coefficient_hint` `site_hint` `interpolate_hint`; enums `derivatives` `bc_type`.

## Cross-cutting rules

Verified across the chapters; each has more detail in its domain chapter.

1. **Scratchpad is caller-allocated.** Compute `<routine>_scratchpad_size(queue, ...)` with the same shape arguments you will pass to the routine, allocate that many elements, pass as `scratchpad`. Omitting or under-sizing it is the most common oneMKL-LAPACK error.
2. **Buffer vs USM changes the return type.** Buffer versions return `void` and synchronize via the SYCL buffer model; USM versions return `sycl::event`. Only USM forms take a `dependencies` vector.
3. **Integer type discipline.** `std::int64_t` for dimensions/leading dimensions/offsets under ILP64; the sparse domain additionally documents its own supported index-integer types per format.
4. **Leading dimensions are never optional.** Every dense routine takes `lda`/`ldb`/`ldc`; a leading dimension smaller than the minimum implied by the layout is invalid.
5. **Sparse handles are a contract, not an object.** `init_matrix_handle` → `set_<format>_data` (+ `set_matrix_property`) → `optimize_*` → compute → `release_matrix_handle`. `release_matrix_handle` is asynchronous and returns a `sycl::event`; the handle pointer must remain valid until it completes.
6. **Multi-stage routines need a size query first.** `omatadd`/`omatconvert` and the `matmat` family require analyze/buffer-size calls before compute, because the output nonzero count is not known in advance.
7. **DFT requires explicit configuration then `commit()`.** Set `config_param` values on the descriptor, call `commit()`, then `compute_forward`/`compute_backward`. Use `set_workspace` when supplying externally allocated workspace.
8. **Errors surface as exceptions.** The reference defers to the "Error Handling" section (see [foundations](mkl-reference-digest/foundations.md)) — oneMKL throws standard SYCL/oneMKL exception types rather than returning status codes.
9. **RNG engines own the queue (host) or take seed+offset (device).** Host `generate` submits into the engine's queue; the device API takes no queue and uses explicit `seed`/`offset`. `leapfrog` interleaves streams, `skip_ahead` takes contiguous blocks.

## Coverage and known gaps

- **Digested:** all 1,215 pages — foundations (p10–29), BLAS (p30–293), sparse (p294–392), LAPACK (p393–701), VM (p702–986), RNG (p987–1095), summary statistics (p1096–1130), DFT (p1131–1182), data fitting (p1183–1196), and Appendix A + functionality tables (p1197–1215). **429 distinct routine sections.**
- **Excluded by design:** table of contents, bibliography, and "Notices and Disclaimers".
- **Formula fidelity:** 182/185 formula pages transcribed from images; the remaining three are described narratively in `rng.md`. Where a source image contained an obvious typo, the chapters say so and reproduce it verbatim rather than silently correcting it.
- **Systemic gap:** the reference documents *declared* signatures, which is stronger than the samples digest, but it does **not** include compilable end-to-end programs, CMake setup, or compiler flags. For build/link lines and runnable patterns, use `output/mkl-digest.md` (the samples digest). The two digests are complementary halves of the same knowledge base.
