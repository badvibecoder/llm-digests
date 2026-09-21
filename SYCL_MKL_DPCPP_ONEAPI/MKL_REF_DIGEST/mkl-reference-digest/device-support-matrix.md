# Device Support Matrix and Functionality Tables

This chapter covers Appendix A of the oneMKL - Data Parallel C++ Developer Reference (2026.0), "oneMKL Functionality Device Support Matrix": which oneMKL SYCL API families and named routines are available on CPU versus Intel GPU, plus the Data Fitting component tables that the appendix references. It is organized by domain (BLAS, LAPACK, DFT, Sparse BLAS, Sparse Solvers, RNG, Vector Math, Data Fitting, Summary Statistics) with one `###` subsection per domain family and one table row per named routine; the only entries in these pages that carry a `Syntax` signature are the Data Fitting component APIs, which are given in full. Everything in these tables is an availability statement only — the appendix states **no** signatures, argument lists, precision templates, or error conditions for the routines it lists.

## Overview

- **What the matrix asserts.** Rows are functionality entries (a routine, a routine family, or a named operation); columns are **CPU** and **Intel GPU**. Cell values are `Yes`, `No`, `All` (BLAS only, meaning the whole level is supported), `None` (Sparse BLAS tables, meaning that variant is unsupported for that format), or `CPU/GPU` / `CPU` / `GPU` (Sparse BLAS tables, giving which devices that specific variant supports). There are no partial, emulated, or "deprecated" qualifiers.
- **Only two device columns exist.** The matrix covers CPU and Intel GPU. It makes no statement about any other SYCL device or vendor; absence of a column is not a claim about that hardware. No precision-by-precision availability is given except for Vector Math (single/double/complex) and the RNG distribution list (which names precision in parentheses for some entries).
- **Naming conventions in the tables.** The BLAS level names (`BLAS Level 1`, `BLAS Level 2`, `BLAS Level 3`, `BLAS-like Extensions`) are the source's exact labels. LAPACK families use the LAPACK brace shorthand `{or,un}gqr`, `{sy,he}ev`, etc.; the source does **not** expand these braces, so they must be read as "the family of routines matching that name pattern". Sparse BLAS names are listed under a column explicitly headed "SYCL API name with `oneapi::mkl::` namespace", so e.g. the table's `sparse::gemv()` means `oneapi::mkl::sparse::gemv`; the appendix never gives its arguments.
- **No host/device distinction, no buffer/USM distinction.** The tables distinguish only device *type*. Nothing on these pages says whether a listed functionality is available in a host API, a buffer API, or a USM API.
- **Only one cross-cutting note about signatures** appears: for LAPACK, "All of the DPC++ LAPACK computational routines have a corresponding `*_scratchpad_size` function for calculating the required amount of scratchpad space." No such note is made for any other domain on these pages.
- **Version notes are rare.** Only the Data Fitting tables carry them: `default linear` and `hermite` are "Yes (since oneMKL 2022.1)" on both CPU and Intel GPU; nothing else in the appendix is version-annotated.
- **Source span.** The chunks behind this chapter are pages 1184-1215. Pages 1184-1198 are the tail of the Data Fitting component chapter (Common Terms, Hints, Splines, Interpolate Function, Examples, Bibliography) and are included here only because they carry the mathematical formula images for these pages; Appendix A itself begins on page 1199. Pages 1210-1215 are "Notices and Disclaimers", third-party licences, and trademark text and are deliberately excluded.

## Routines

### BLAS Functionality

| Functionality | CPU | Intel GPU |
|---|---|---|
| BLAS Level 1 | All | All |
| BLAS Level 2 | All | All |
| BLAS Level 3 | All | All |
| BLAS-like Extensions | All | All |

`All` is the literal cell value: the whole functionality group is supported on that device. No individual BLAS routine names appear in this table.

### LAPACK Functionality

NOTE (verbatim): All of the DPC++ LAPACK computational routines have a corresponding `*_scratchpad_size` function for calculating the required amount of scratchpad space.

The source splits the list into several consecutive tables; the group boundaries are reproduced below. No group is titled in the extracted text.

Group 1 — LU:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `getrf` | Yes | Yes |
| `getrs` | Yes | Yes |
| `getri` | Yes | Yes |

Group 2 — Cholesky:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `potrf` | Yes | Yes |
| `potrs` | Yes | Yes |
| `potri` | Yes | Yes |

Group 3 — QR:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `geqrf` | Yes | Yes |
| `{or,un}gqr` | Yes | Yes |
| `{or,un}mqr` | Yes | Yes |
| `gerqf` | Yes | **No** |
| `{or,un}mrq` | Yes | **No** |

Group 4 — triangular:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `trtrs` | Yes | Yes |
| `{sy,he}trf` | Yes | **No** |

Group 5 — symmetric/Hermitian eigenproblems:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `{sy,he}ev` | Yes | Yes |
| `{sy,he}evd` | Yes | Yes |
| `{sy,he}evx` | Yes | Yes |
| `{sy,he}trd` | Yes | Yes |
| `{or,un}gtr` | Yes | **No** |
| `{or,un}mtr` | Yes | **No** |
| `steqr` | Yes | Yes |

Group 6 — generalized eigenproblems:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `{sy,he}gvd` | Yes | Yes |
| `{sy,he}gvx` | Yes | Yes |

Group 7 — SVD:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `gesvd` | Yes | Yes |
| `gebrd` | Yes | Yes |
| `{or,un}gbr` | Yes | **No** |

Group 8 — batched:

| Functionality | CPU | Intel GPU |
|---|---|---|
| `getrf_batch` | Yes | Yes |
| `getrs_batch` | Yes | Yes |
| `getri_batch` | Yes | Yes |
| `potrf_batch` | Yes | Yes |
| `potrs_batch` | Yes | Yes |
| `geqrf_batch` | Yes | Yes |
| `{or,un}gqr_batch` | Yes | Yes |

Group 9 — catch-all:

| Functionality | CPU | Intel GPU |
|---|---|---|
| Other LAPACK Routines | No | No |

### DFT Functionality

| Functionality | CPU | Intel GPU |
|---|---|---|
| 1D Complex-to-Complex FFT transformations | Yes | Yes |
| 2D Complex-to-Complex FFT transformations | Yes | Yes |
| 3D Complex-to-Complex FFT transformations | Yes | Yes |
| 4D Complex-to-Complex FFT transformations | No | No |
| 5D Complex-to-Complex FFT transformations | No | No |
| 6D Complex-to-Complex FFT transformations | No | No |
| 7D Complex-to-Complex FFT transformations | No | No |
| 1D Real-to-Complex FFT transformations | Yes | Yes |
| 2D Real-to-Complex FFT transformations | Yes | Yes |
| 3D Real-to-Complex FFT transformations | Yes | Yes |
| 4D Real-to-Complex FFT transformations | No | No |
| 5D Real-to-Complex FFT transformations | No | No |
| 6D Real-to-Complex FFT transformations | No | No |
| 7D Real-to-Complex FFT transformations | No | No |

The table lists only Complex-to-Complex and Real-to-Complex; no Complex-to-Real or Real-to-Real rows appear on these pages. No individual transform constructor/descriptor names are given.

### Sparse BLAS SYCL Functionality

The appendix gives Sparse BLAS as four tables (Level 1, Level 2, Level 3, Auxiliary Functions) keyed by matrix format: **CSR**, **COO**, **CSC**, **BSR**.

Operation-description legend used by the tables (the mathematical symbols themselves are images and did **not** survive text extraction, so only their roles are recoverable): scalar values, dense vectors, sparse vectors, dense matrices, sparse matrices, and the identity matrix. The tables use:

- an upper/lower selector for the operation, appearing verbatim as `uplo::lower` and `uplo::upper`.
- a diagonal selector, appearing verbatim as `diag::nonunit` and `diag::unit`. For triangular operations, `diag::nonunit` "uses [the diagonal] as provided in the data" and `diag::unit` "ignores [the diagonal] and instead uses the identity".
- a transpose selector that the source introduces only as "the [blank] operation represents one of three \"transpose\" operations", its three bullets each reading "when using [blank]"; **all three operation names and enum value identifiers are lost** — see Explicit gaps.

#### Sparse BLAS Level 1

Verbatim: "There are currently no SYCL Sparse BLAS APIs in oneMKL supporting sparse vectors or Sparse Level 1 operations."

#### Sparse BLAS Level 2 (sparse matrix and vectors)

`SpMV` is a merged cell in the source covering the `sparse::gemv()`, `sparse::symv()` and `sparse::trmv()` rows. In the extracted text the `sparse::gemv_dot()` row is printed split across lines as `sparse::gemv` / `dot()`, so that name is reconstructed from the two fragments.

| Common Name | SYCL API name (`oneapi::mkl::` namespace) | Operation | CSR support | COO support | CSC support | BSR support |
|---|---|---|---|---|---|---|
| SpMV | `sparse::gemv()` | (not in extracted text) | CPU/GPU | CPU/GPU | CPU/GPU | CPU/GPU |
| SpMV | `sparse::symv()` + `uplo::lower` | (not in extracted text) | CPU/GPU | None | None | None |
| SpMV | `sparse::symv()` + `uplo::upper` | (not in extracted text) | CPU/GPU | None | None | None |
| SpMV | `sparse::trmv()` + `uplo::lower` + `diag::nonunit` | (not in extracted text) | CPU/GPU | CPU | None | None |
| SpMV | `sparse::trmv()` + `uplo::lower` + `diag::unit` | (not in extracted text) | CPU/GPU | CPU | None | None |
| SpMV | `sparse::trmv()` + `uplo::upper` + `diag::nonunit` | (not in extracted text) | CPU/GPU | CPU | None | None |
| SpMV | `sparse::trmv()` + `uplo::upper` + `diag::unit` | (not in extracted text) | CPU/GPU | CPU | None | None |
| SpMV + dot fusion | `sparse::gemv_dot()` | (not in extracted text) | CPU/GPU | CPU | None | None |
| SpSV or SpTRSV | `sparse::trsv()` + `uplo::lower` + `diag::nonunit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSV or SpTRSV | `sparse::trsv()` + `uplo::lower` + `diag::unit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSV or SpTRSV | `sparse::trsv()` + `uplo::upper` + `diag::nonunit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSV or SpTRSV | `sparse::trsv()` + `uplo::upper` + `diag::unit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |

#### Sparse BLAS Level 3 (sparse/dense matrices)

| Common Name | SYCL API name (`oneapi::mkl::` namespace) | Operation | CSR support | COO support | CSC support | BSR support |
|---|---|---|---|---|---|---|
| SpMM (Dn <- Sp * Dn) | `sparse::gemm()` | (not in extracted text) | CPU/GPU | CPU/GPU | None | None |
| SpSM or SpTRSM | `sparse::trsm()` + `uplo::lower` + `diag::nonunit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSM or SpTRSM | `sparse::trsm()` + `uplo::lower` + `diag::unit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSM or SpTRSM | `sparse::trsm()` + `uplo::upper` + `diag::nonunit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpSM or SpTRSM | `sparse::trsm()` + `uplo::upper` + `diag::unit` | "Solve for :" (operands lost) | CPU/GPU | CPU | None | None |
| SpGEAM (Sp <- Sp + Sp) | `sparse::omatadd()` | (not in extracted text) | CPU/GPU | None | None | None |
| SpGEMM (Sp <- Sp * Sp) | `sparse::matmat()` | (not in extracted text) | CPU/GPU | None | None | None |
| SpGEMMD (Dn <- Sp * Sp) | `sparse::matmatd()` | (not in extracted text) | **GPU** | None | None | None |

#### Auxiliary Functions

Operations that can manipulate the sparse matrix handle.

| Common Name | SYCL API name (`oneapi::mkl::` namespace) | Operation | CSR support | COO support | CSC support | BSR support |
|---|---|---|---|---|---|---|
| Sparse Copy or Sparse Matrix Transpose | `sparse::omatcopy()` | (not in extracted text) | CPU/GPU | CPU/GPU | None | None |
| Sparse Matrix Conversion | `sparse::omatconvert()` | conversion where the two operands are represented using **different sparse matrix formats** | CPU/GPU | CPU/GPU | None | None |
| Sparse Matrix Sort | `sparse::sort_matrix()` | sort by natural ordering for the sparse matrix format | CPU/GPU | None (see note below) | None | None |
| Update Matrix Main Diagonal | `sparse::update_diagonal_data()` | update in-place with new provided diagonal values | **GPU** | None | None | None |

The COO entry for `sparse::sort_matrix()` is annotated "None (see note below)", but the note itself is not present in the extracted text.

### Sparse Solvers Functionality

| Functionality | CPU | Intel GPU |
|---|---|---|
| Sparse Cholesky Factorization | No | No |
| Sparse LU Factorization | No | No |
| Sparse QR factorization | No | No |
| Hermitian/Symmetric Eigensolver on intervals for Sparse Matrices | No | No |
| Extremal Eigensolvers for Sparse Matrix | No | No |
| Poisson Solver | No | No |
| Trust Region Solver | No | No |

### Random Number Generators (RNG) Functionality

#### Engines

| Functionality | CPU | Intel GPU |
|---|---|---|
| `MRG32K3A` | Yes | Yes |
| `MT2203` | Yes | Yes |
| `MT19937` | Yes | Yes |
| `PHILOX4X32X10` | Yes | Yes |
| `SOBOL` | Yes | Yes |
| `ARS5` | Yes | **No** |
| `MCG59` | Yes | Yes |
| `NIEDERR` | Yes | **No** |
| `MCG31` | Yes | Yes |
| `WH` | Yes | **No** |
| `SFMT19937` | Yes | **No** |
| `R250` | Yes | **No** |
| `NONDETERM` | Yes | **No** |
| `DABSTRACT` | No | No |
| `SABSTRACT` | No | No |
| `SABSTRACT` | No | No |

The last three rows are transcribed exactly as extracted: `DABSTRACT`, `SABSTRACT`, `SABSTRACT` — the same name appears twice with identical `No`/`No` cells. This looks like an extraction or source artifact (see Explicit gaps). These names are spelled with the case shown; the appendix gives no namespace qualification for them, nor their template parameters.

#### Distributions

| Functionality | CPU | Intel GPU |
|---|---|---|
| Uniform (single/double/integer) | Yes | Yes |
| UniformBits32 UniformBits64 | Yes | Yes |
| Lognormal (single/double) | Yes | Yes |
| Gaussian (single/double) | Yes | Yes |
| Poisson | Yes | Yes |
| UniformBits | Yes | Yes |
| Bernoulli | Yes | Yes |
| Beta (single/double) | Yes | **No** |
| Binomial | Yes | **No** |
| ChiSquare (single/double) | Yes | **No** |
| Exponential (single/double) | Yes | Yes |
| Gamma (single/double) | Yes | **No** |
| Geometric | Yes | Yes |
| Gumbel (single/double) | Yes | Yes |
| Hyper Geometric | Yes | **No** |
| Laplace (single/double) | Yes | Yes |
| Multinomial | Yes | **No** |
| Negative Binomial | Yes | **No** |
| PoissonV | Yes | **No** |
| Rayleigh (single/double) | Yes | Yes |
| Weibull (single/double) | Yes | Yes |
| Cauchy (single/double) | Yes | Yes |
| GaussianMV (single/double) | Yes | **No** |

`UniformBits32` and `UniformBits64` are a single merged row in the source (`UniformBits32 UniformBits64` | Yes | Yes) and are reproduced as-is. Distribution names are given verbatim; the appendix shows no `<T>` template forms here.

### Vector Math (VM) Functionality

| Functionality | CPU | Intel GPU |
|---|---|---|
| Vector Math Functions, Single Precision | Yes | Yes |
| Vector Math Functions, Double Precision | Yes | Yes |
| Vector Math Functions, Single Precision Complex | Yes | **No** |
| Vector Math Functions, Double Precision Complex | Yes | **No** |

Additional verbatim note on this page: "OpenMP* offload to the GPU is implemented in the Linux* OS, but not in the Windows* OS. The Windows OS implementation will be available in a future release." This note is not attached to a specific row in the extracted text; it appears immediately after the VM table.

No individual VM function names appear in the appendix.

### Data Fitting Functionality

Two tables, then the API surface from the Data Fitting chapter that these chunks also carry.

#### Splines, Spline Type

| Functionality | CPU | Intel GPU |
|---|---|---|
| default linear | Yes (since oneMKL 2022.1) | Yes (since oneMKL 2022.1) |
| default quadratic | No | No |
| default cubic | No | No |
| subbotin | No | No |
| natural | No | No |
| hermite | Yes (since oneMKL 2022.1) | Yes (since oneMKL 2022.1) |
| akima | No | No |
| bessel | No | No |
| hyman | No | No |
| lookup interpolant | No | No |
| cr stepwise const interpolant | No | No |
| cl stepwise const interpolant | No | No |

#### Computation Routines

| Functionality | CPU | Intel GPU |
|---|---|---|
| Construct1D | Yes (since oneMKL 2022.1) | Yes (since oneMKL 2022.1) |
| Interpolate1D | Yes (since oneMKL 2022.1) | Yes (since oneMKL 2022.1) |
| Interpolate1DEx | No | No |
| Integrate1D | No | No |
| Integrate1DEx | No | No |
| SearchCells1D | No | No |
| SearchCells1DEx | No | No |
| InterpolationCallBack | No | No |
| IntegrateCallBack | No | No |
| SearchCellsCallBack | No | No |

#### Data Fitting component API (names/signatures from pages 1184-1198)

The appendix's Data Fitting rows correspond to the DPC++ Data Fitting component API whose text precedes the appendix in the same source pages. Header and namespace, verbatim:

```cpp
#include<oneapi/mkl/experimental/data_fitting.hpp>
oneapi::mkl::experimental::data_fitiing
```

The namespace is printed with three `i`s (`data_fitiing`) in the header/namespace blocks, while the worked example writes `oneapi::mkl::experimental::data_fitting`. Both spellings are reproduced here; the discrepancy is in the source.

##### spline

Class template; the common API for all spline types. Purpose: hold user-supplied partition/function-value/coefficient memory and construct spline coefficients on a SYCL device.

```cpp
template <
  typename FpType,
  typename SplineType,
  int Dimensions = 1>
class spline {
public:
    using value_type = FpType;
    using spline_type = SplineType;
    spline(
      const sycl::queue& q,
      std::int64_t ny = 1,
      bool were_coeffs_computed = false);
    spline(
      const sycl::device& dev,
      const sycl::context& ctx,
      std::int64_t ny = 1,
      bool were_coeffs_computed = false);
    ~spline();
    spline(const spline<FpType, SplineType, Dimensions>& other) = delete;
    spline(spline<FpType, SplineType, Dimensions>&& other) = delete;
    spline<FpType, SplineType, Dimensions>& operator=(
        const spline<FpType, SplineType, Dimensions>& other) = delete;
    spline<FpType, SplineType, Dimensions>& operator=(
        spline<FpType, SplineType, Dimensions>&& other) = delete;
    spline& set_partitions(
      FpType* input_data,
      std::int64_t nx,
      partition_hint PartitionHint = partition_hint::non_uniform);
    spline& set_function_values(
      FpType* input_data,
      function_hint FunctionHint = storage_hint::row_major);
    spline& set_coefficients(
      FpType* data,
      coefficient_hint CoeffHint = storage_hint::row_major);
    bool is_initialized() const;
    std::int64_t get_required_coeffs_size() const;
    sycl::event construct(const std::vector<sycl::event>& dependencies = {});
};
```

An instance of `spline<T, ST, N>` creates the N-dimensional spline that operates with the `T` data type; `ST` is the spline type. `T` can only be `float` or `double`.

Constructors:

- `spline(const sycl::queue& q, std::int64_t ny = 1, bool were_coeffs_computed = false)` — create an object with the `q` SYCL queue and `ny` number of functions. If spline coefficients were already computed, pass `true` as the third parameter. `were_coeffs_computed == false` by default, which means `construct` must be called to compute spline coefficients.
- `spline(const sycl::device& dev, const sycl::context& ctx, std::int64_t ny = 1, bool were_coeffs_computed = false)` — same, using the `dev` SYCL device and `ctx` context.

Member functions:

- `spline& set_partitions(FpType* input_data, std::int64_t nx, partition_hint PartitionHint = partition_hint::non_uniform)` — set partition values from `input_data` with `nx` partition values. Default layout is `non_uniform`. If `uniform` is specified, `nx` must equal 2 and `input_data` must contain only 2 values: the left and the right borders of the partition; otherwise behavior is undefined. If the `input_data` layout does not satisfy `PartitionHint`, behavior is undefined. Returns a reference to the spline object.
- `spline& set_function_values(FpType* input_data, function_hint FunctionHint = storage_hint::row_major)` — set function values from `input_data`; the number of function values must equal `ny * nx` elements. Default layout is `row_major`. If the layout does not satisfy `FunctionHint`, behavior is undefined. Returns a reference to the spline object.
- `spline& set_coefficients(FpType* data, coefficient_hint CoeffHint = storage_hint::row_major)` — set coefficients from `data`; the number of coefficients in memory must equal the return value of `get_required_coeffs_size()`. Default layout is `row_major`. If the layout does not satisfy `CoeffHint`, behavior is undefined. If `were_coeffs_computed == false`, `data` will be rewritten during the `construct` call. Returns a reference to the spline object.
- `bool is_initialized() const` — returns `true` if all required data are set (for example, partitions, function values, coefficients).
- `std::int64_t get_required_coeffs_size() const` — returns the amount of memory required for coefficients storage.
- `sycl::event construct(const std::vector<sycl::event>& dependencies = {})` — constructs the spline (calculates spline coefficients if `were_coeffs_computed == false`). Submits a SYCL kernel and returns the SYCL event to wait on to ensure computation is complete. `dependencies` is a list of SYCL events to wait for before starting computations.

Two further member functions must be called for spline types that require internal conditions and boundary conditions:

- `spline& set_internal_conditions(FpType* input_data)` — set internal conditions from `input_data`; the number of internal condition values must equal `nx - 2`. Returns a reference to the spline object.
- `spline& set_boundary_conditions(bc_type BCType = bc_type::free_end, FpType input_value = {})` — set the `input_value` boundary condition corresponding to `BCType`. Default value for `input_value` is empty since some boundary conditions do not require a value. Returns a reference to the spline object.

NOTES (verbatim): copy/move constructor and copy/move assignment operators are deleted since the spline class is just a wrapper over memory that users provide. Memory management responsibility is on the user's side.

##### linear_spline::default_type

Linear spline is a spline whose degree is equal to 1.

```cpp
namespace linear_spline {
  struct default_type {};
}
```

Usage: `spline<float, linear_spline::default_type> val(/*SYCL queue object*/q, /*number of spline functions*/ny);`

##### cubic_spline::hermite

Cubic splines are splines whose degree is equal to 3. "There are a lot of different types of cubic splines: Hermite, natural, Akima, Bessel. However, the current version of DPC++ API supports only one type: Hermite."

```cpp
namespace cubic_spline {
  struct hermite {};
}
```

Usage: `spline<float, cubic_spline::hermite> val(/*SYCL queue object*/q, /*number of spline functions*/ny);`

##### interpolate

Purpose: perform computations of function and derivative values at interpolation sites. If the sites do not belong to the interpolation interval `[a, b]`, the library uses the interpolant coefficients computed for the first interval for sites to the left of `a`, and the interpolant coefficients computed for the last interval for sites to the right of `b`. The interpolation algorithm depends on the interpolant's type (for example, for a cubic spline, evaluation of a third-order polynomial obtains function values).

```cpp
template <typename Interpolant>
sycl::event interpolate(
    Interpolant& interpolant,
    typename Interpolant::fp_type* sites,
    std::int64_t n_sites,
    typename Interpolant::fp_type* results,
    const std::vector<sycl::event>& dependencies,
    interpolate_hint ResultHint = interpolate_hint::funcs_sites_ders,
    site_hint SiteHint = site_hint::non_uniform); // (1)
template <typename Interpolant>
sycl::event interpolate(
    Interpolant& interpolant,
    typename Interpolant::fp_type* sites,
    std::int64_t n_sites,
    typename Interpolant::fp_type* results,
    std::bitset<32> der_indicator,
    const std::vector<sycl::event>& dependencies = {},
    interpolate_hint ResultHint = interpolate_hint::funcs_sites_ders,
    site_hint SiteHint = site_hint::non_uniform); // (2)
template <typename Interpolant>
sycl::event interpolate(
    sycl::queue& q,
    const Interpolant& interpolant,
    typename Interpolant::fp_type* sites,
    std::int64_t n_sites,
    typename Interpolant::fp_type* results,
    const std::vector<sycl::event>& dependencies,
    interpolate_hint ResultHint = interpolate_hint::funcs_sites_ders,
    site_hint SiteHint = site_hint::non_uniform); // (3)
template <typename Interpolant>
sycl::event interpolate(
    sycl::queue& q,
    const Interpolant& interpolant,
    typename Interpolant::fp_type* sites,
    std::int64_t n_sites,
    typename Interpolant::fp_type* results,
    std::bitset<32> der_indicator,
    const std::vector<sycl::event>& dependencies = {},
    interpolate_hint ResultHint = interpolate_hint::funcs_sites_ders,
    site_hint SiteHint = site_hint::non_uniform); // (4)
```

Overload meanings, verbatim numbering:

1. Performs computations of function values only using the SYCL queue associated with `interpolant`.
2. Performs computations of certain derivatives (function values is considered as a zero derivative) which are indicated in `der_indicator` (each bit corresponds to certain derivative starting from lower bit) using the SYCL queue associated with `interpolant`.
3. Performs computations of function values only using `q` as an input argument that should be created from the same context and device as the SYCL queue associated with `interpolant`.
4. Performs computations of certain derivatives (function values is considered as a zero derivative) which are indicated in `der_indicator` (each bit corresponds to certain derivative starting from lower bit) using `q` as an input argument that should be created from the same context and device as the SYCL queue associated with `interpolant`.

For all functions, `SiteHint` and `ResultHint` specify the layout of sites and results respectively. If the results layout does not satisfy `ResultHint` and/or the sites layout does not satisfy `SiteHint`, behavior is undefined. Returns the SYCL event of the submitted task.

##### Data Fitting hint and boundary-condition enums

```cpp
enum class partition_hint {
  non_uniform,
  quasi_uniform,
  uniform
};

enum class function_hint {
  row_major,
  col_major
};

enum class coefficient_hint {
  row_major
};

enum class site_hint {
  non_uniform,
  uniform,
  sorted
};

enum class interpolate_hint {
  funcs_sites_ders,
  funcs_ders_sites,
  sites_funcs_ders,
  sites_ders_funcs
};

enum class derivatives {
  zero,
  first,
  second,
  third
};

enum class bc_type {
  free_end,
  first_left_der,
  first_right_der,
  second_left_der,
  second_right_der,
  periodic
};
```

Hints, as described in the source:

- **partition_hint**: supported values are `non_uniform`, `quasi_uniform`, `uniform`.
- **function_hint**: function values are stored in a one-dimensional array with `nx * ny` elements; both row-major and column-major layouts are possible, and the supported hints are `row_major` and `col_major`.
- **coefficient_hint**: for a cubic spline, 4 coefficients per interpolation interval and function value are required, stored in a one-dimensional array with `4 * (nx - 1) * ny` elements; the only supported value is `row_major` (row-major and column-major layouts are both described in the prose, but only `row_major` is listed as supported).
- **site_hint**: supported values are `non_uniform`, `uniform`, `sorted`.
- **interpolate_hint**: with `d` derivatives (including interpolation values) to be calculated, the memory for results is `nsite * ny * d` elements. Six layouts are described (functions-sites-derivatives, functions-derivatives-sites, sites-functions-derivatives, sites-derivatives-functions, derivatives-functions-sites, derivatives-sites-functions) but only four are listed as supported, matching the four enum values: `funcs_sites_ders`, `funcs_ders_sites`, `sites_funcs_ders`, `sites_ders_funcs`.
- **derivatives**: selects which derivative orders are computed by `interpolate`: interpolation values only, first derivative only, second derivative only, third derivative only. `operator|` is overloaded to create combinations, e.g. `std::bitset<32> bit_mask = derivatives::zero | derivatives::first | derivatives::third;`
- **bc_type**: boundary conditions supported are free end (no value required), periodic, first derivative, and second derivative. NOTE (verbatim): first derivative and second derivative types must be set on the left and on the right borders; free end doesn't require any values to be set.

##### Worked example (verbatim from the source, p1196-1197)

```cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/experimental/data_fitting.hpp>

constexpr std::int64_t nx = 10'000;
constexpr std::int64_t nsites = 150'000;
int main (int argc, char ** argv) {
    sycl::queue q;
    sycl::usm_allocator<double, sycl::usm::alloc::shared> alloc(q);
    // Allocate memory for spline parameters
    std::vector<double, decltype(alloc)> partitions(nx, alloc);
    std::vector<double, decltype(alloc)> functions(nx, alloc);
    std::vector<double, decltype(alloc)> coeffs(2 * (nx - 1), alloc);
    std::vector<double, decltype(alloc)> sites(nsites, alloc);
    std::vector<double, decltype(alloc)> results(nsites, alloc);
    // Fill parameters with valid data
    for (std::int64_t i = 0; i < nx; ++i) {
        partitions[i] = 0.1 * i;
        functions[i] = i * i;
    }
    for (std::int64_t i = 0; i < nsites; ++i) {
        sites[i] = (0.1 * nx * i) / nsites);
    }
    namespace df = oneapi::mkl::experimental::data_fitting;
    // Set parameters to spline
    df::spline<double, df::linear_spline::default_type> spl(q);
    spl.set_partitions(partitions.data(), nx)
       .set_coefficients(coeffs.data())
       .set_function_values(functions.data());
    // Construct spline
    auto event = spl.construct();
    event = df::interpolate(spl, sites.data(), nsites, results.data(), { event });
    event.wait();
    std::cout << "done" << std::endl;
    return 0;
}
```

The example is reproduced exactly, including the unmatched `)` in `sites[i] = (0.1 * nx * i) / nsites);` and the unqualified `set_partitions`/`set_coefficients`/`set_function_values` call chain (no `df::` prefix on the member calls, which is correct for member functions).

### Summary Statistics (SS) Functionality

| Functionality | CPU | Intel GPU |
|---|---|---|
| min | Yes | No |
| max | Yes | No |
| raw sum | Yes | No |
| 2nd order raw sum | Yes | No |
| 3rd order raw sum | Yes | No |
| 4th order raw sum | Yes | No |
| 2nd order central sum | Yes | No |
| 3rd order central sum | Yes | No |
| 4th order central sum | Yes | No |
| mean | Yes | No |
| 2nd order raw moment | Yes | No |
| 3rd order raw moment | Yes | No |
| 4th order raw moment | Yes | No |
| 2nd order central moment | Yes | No |
| 3rd order central moment | Yes | No |
| 4th order central moment | Yes | No |
| kurtosis | Yes | No |
| skewness | Yes | No |
| variation coefficient | Yes | No |
| covariance matrix | No | No |
| correlation matrix | No | No |
| cross-product matrix | No | No |
| pooled covariance matrix | No | No |
| pooled mean | No | No |
| group covariance matrix | No | No |
| group mean | No | No |
| quantiles | No | No |
| order statistics | No | No |
| robust covariance matrix | No | No |
| ouliers detection | No | No |
| partial covariance matrix | No | No |
| partial covariance matrix | No | No |
| missing values | No | No |
| parameterized correlation matrix | No | No |
| stream quantiles | No | No |
| mean absolute deviation | No | No |
| median absolute deviation | No | No |
| sorted observations | No | No |

In the source, `raw sum` is a merged cell covering the 2nd/3rd/4th order raw sums (all yes on CPU, no on GPU), `mean` is a merged cell covering mean and the 2nd/3rd/4th order raw moments, and `2nd order central sum` / `2nd order central moment` are merged cells covering their 2nd/3rd/4th order rows; the flattened extraction flattens this, so the merged grouping is noted here. `ouliers detection` and the duplicated `partial covariance matrix` row are reproduced verbatim from the source.

## Formulas

All formulas below are transcribed from the formula images listed for pages 1184, 1192, 1193 and 1194 (labels are the red `pNNNN#i` tags in each image). They belong to the Data Fitting component (spline glossary, linear spline, cubic Hermite spline), not to the appendix tables.

Glossary (p1184):

- `spline (p1184#0): P(x) = c_1 + c_2 (x - x_i) + c_3 (x - x_i)^2 + ... + c_{k-1} (x - x_i)^k.`
- `spline (p1184#1): P_i(x) = c_{1,i} + c_{2,i} (x - x_i) + c_{3,i} (x - x_i)^2 + ... + c_{k-1,i} (x - x_i)^k.`
  - As printed, the final coefficient subscript reads `k-1` while its exponent reads `k`; no other reading is supported by the pixels.

Linear spline (p1192, continuing on p1193):

- `linear_spline::default_type (p1192#0): P_i(x) = c_{1,i} + c_{2,i} (x - x_i),`
- `linear_spline::default_type (p1192#1): x ∈ [x_i, x_{i+1}),`
- `linear_spline::default_type (p1192#2): c_{1,i} = f(x_i),`
  - The subscripts on the last formula sit at the bottom edge of the image and are partly clipped but legible.
- `linear_spline::default_type (p1193#0): c_{2,i} = [x_i, x_{i+1}] f,`
- `linear_spline::default_type (p1193#1): i = 1, ..., n - 1.`

Cubic Hermite spline (p1193):

- `cubic_spline::hermite (p1193#2): P_i(x) = c_{1,i} + c_{2,i} (x - x_i) + c_{3,i} (x - x_i)^2 + c_{4,i} (x - x_i)^3,`
- `cubic_spline::hermite (p1193#3): x ∈ [x_i, x_{i+1}),`

Cubic Hermite spline coefficients (p1194):

- `cubic_spline::hermite (p1194#0): i = 1, ..., n - 1.`
- `cubic_spline::hermite (p1194#1): c_{1,i} = f(x_i),`
- `cubic_spline::hermite (p1194#2): c_{2,i} = s_i,`
- `cubic_spline::hermite (p1194#3): c_{3,i} = ([x_i, x_{i+1}] f - s_i) / (Δx_i) - c_{4,i} (Δx_i),`
- `cubic_spline::hermite (p1194#4): c_{4,i} = (s_i + s_{i+1} - 2 [x_i, x_{i+1}] f) / (Δx_i)^2,`
- `cubic_spline::hermite (p1194#5, tag clipped): s_i = f^{(1)}(x_i).`

Total: 15 formula images transcribed from the four formula PNGs. Notation used by the source elsewhere: `[x_i, x_{i+1}] f` is the first divided difference of `f`; the Mathematical Notation table defines the k-th divided difference as the leading coefficient of the order-`k+1` polynomial that agrees with `f` at the points, but its displayed formulas did not survive extraction. `Δx_i` is used in the Hermite coefficient formulas without being defined in the extracted text.

## Conventions & Gotchas

- **Availability only, never signatures.** Every routine listed in Appendix A with a "Yes"/"All"/"CPU/GPU" cell has *no* syntax, argument list, template parameter, default argument, or error condition stated in these pages. Do not infer a signature from the matrix; use the chapter for the routine's own domain.
- **Scope qualification.** Sparse BLAS names in the tables are prefixed implicitly: the column heading is "SYCL API name with `oneapi::mkl::` namespace", so `sparse::gemv()` denotes `oneapi::mkl::sparse::gemv`. The RNG, VM, DFT, LAPACK, and SS tables give only bare functionality labels.
- **Brace families are not expanded.** `{or,un}gqr`, `{or,un}mqr`, `{or,un}mrq`, `{or,un}gtr`, `{or,un}mtr`, `{or,un}gbr`, `{sy,he}trf`, `{sy,he}ev`, `{sy,he}evd`, `{sy,he}evx`, `{sy,he}trd`, `{sy,he}gvd`, `{sy,he}gvx`, and the `*_batch` forms must be read as family patterns exactly as the appendix writes them.
- **`All` vs `Yes`.** `All` appears only in the BLAS table and states that the whole level (not a subset) is supported; everywhere else a binary `Yes`/`No` is used.
- **CPU/GPU cells in Sparse BLAS** are per-variant: the same `sparse::` function with different `uplo`/`diag` selectors can differ (e.g. `sparse::trsv()` is `CPU/GPU` on CSR but only `CPU` on COO; `sparse::matmatd()` and `sparse::update_diagonal_data()` are GPU-only).
- **Format coverage is narrow.** CSC and BSR appear as `None` for every Level 2, Level 3, and Auxiliary row on these pages; only CSR and COO (and, for `sparse::gemv()`, all four) carry support.
- **Sparse Level 1 does not exist** in the SYCL Sparse BLAS API.
- **Sparse Solvers are entirely unavailable** on both CPU and Intel GPU in this release — all seven rows are `No`.
- **Summary Statistics splits at the GPU boundary:** everything up through `variation coefficient` is CPU-only-and-supported, GPU-unsupported; from `covariance matrix` onward nothing is supported on either device.
- **DFT dimensionality limit:** 1D-3D complex-to-complex and real-to-complex are supported on both devices; 4D-7D are unsupported on both.
- **Out-of-place vs in-place** cannot be read from the appendix for sparse operations. The only in-place statement on these pages is for Data Fitting coefficients: `set_coefficients` data "will be rewritten during the `construct` call" when `were_coeffs_computed == false`.
- **Scratchpad sizing** is mentioned only for LAPACK (`*_scratchpad_size`), and only as a note; no sizes, formulas, or per-routine names are listed.
- **Dependency events (Data Fitting).** `construct` takes `const std::vector<sycl::event>& dependencies = {}` and returns a `sycl::event`; every `interpolate` overload returns a `sycl::event`. Overloads (1) and (3) have no default for `dependencies` (it is a required argument), while overloads (2) and (4) default it to `{}`.
- **Queue consistency (Data Fitting).** For `interpolate` overloads (3) and (4), `q` "should be created from the same context and device as the SYCL queue associated with `interpolant`".
- **Undefined behavior, not exceptions.** The Data Fitting API signals misuse by undefined behavior: mismatched `PartitionHint`, `FunctionHint`, `CoeffHint`, `SiteHint`, or `ResultHint` layouts; `nx != 2` with `uniform` partitions. No exception list or error-code table appears on these pages.
- **Integer sizes.** The Data Fitting API uses `std::int64_t` for `nx`, `ny` and `n_sites`, and `std::bitset<32>` for derivative masks. No integer-type requirements are stated for the appendix's routines.
- **Layout element counts** (Data Fitting): function values `nx * ny`; cubic coefficients `4 * (nx - 1) * ny`; interpolation results `nsite * ny * d`; internal conditions `nx - 2`.
- **Default-argument oddity (Data Fitting).** `set_function_values` and `set_coefficients` declare a parameter of type `function_hint` / `coefficient_hint` but the default expression is written `storage_hint::row_major`. Transcribed verbatim; `storage_hint` is defined nowhere in the extracted text, so treat this as a documentation defect rather than an API requirement.
- **Memory ownership (Data Fitting).** `spline` is a wrapper over memory the user provides; copy/move construction and assignment are deleted, and `spline` is not copyable or movable.
- **Uniform partition shortcut.** With `partition_hint::uniform`, pass `nx == 2` and exactly two values, the left and right borders.

## Explicit gaps

- **All inline mathematics on pages 1184-1198 was lost.** The Glossary, the "Mathematical Notation in the Data Fitting Component" table (partition, uniform partition, quasi-uniform partition, vector-valued function of dimension p, k-order derivative, agreement/p multiplicity, k-th divided difference), the function-value/coefficient/interpolation-result layout definitions, and the free-end boundary-condition formula appear as blank where symbols should be. Only the four formula PNGs listed above survived.
- **Sparse BLAS symbol legend is empty.** The roles (scalars, dense vectors, sparse vectors, dense matrices, sparse matrices, identity matrix) survive, but none of the symbol names do.
- **The transpose-selector enum values in the Sparse BLAS legend are missing** — the text reads "when using ." three times with the identifier blank. Do not infer the three operation names or their enum value identifiers on the strength of this source.
- **Operation descriptions for the Sparse BLAS rows are missing.** Level 2 and Level 3 cells that say "Solve for :" have lost their operands, and the `Operation` cells for `sparse::gemv()`, `sparse::symv()`, `sparse::trmv()`, `sparse::gemv_dot()`, `sparse::gemm()`, `sparse::omatadd()`, `sparse::matmat()`, `sparse::matmatd()`, and `sparse::omatcopy()` contain no text at all.
- **The COO note for `sparse::sort_matrix()` is absent** although the cell reads "None (see note below)".
- **The two operands of `sparse::omatconvert()` are unnamed** ("where and are represented using different sparse matrix formats").
- **No signatures anywhere in the appendix.** LAPACK, DFT, Sparse BLAS, Sparse Solvers, RNG, VM, Data Fitting and SS rows are availability labels only.
- **RNG table anomalies.** `SABSTRACT` appears twice with identical cells; whether the second entry was a different name (e.g. another abstract engine) or a duplicated row cannot be determined from the extracted text. `UniformBits32` and `UniformBits64` share one merged row. Engine names carry no template parameters or namespace qualification.
- **No precision breakdown** beyond the VM table and the parenthesised precisions in the RNG distribution names.
- **No device coverage beyond CPU and Intel GPU**, and no statement about what "Intel GPU" means in terms of specific hardware generations.
- **`Δx_i` is used but never defined** in the extracted text.
- **The p1194#5 tag is clipped** at the bottom edge of the image; the formula itself (`s_i = f^{(1)}(x_i)`) is legible.
- **`storage_hint` is referenced in two default arguments but never declared** in the extracted text.
- **Namespace spelling conflict**: `oneapi::mkl::experimental::data_fitiing` (namespace/header blocks) versus `oneapi::mkl::experimental::data_fitting` (worked example). Both are reproduced; the source does not resolve which is correct.
- **Pages 1210-1215** (Notices and Disclaimers, third-party licences) were excluded as legal text.
