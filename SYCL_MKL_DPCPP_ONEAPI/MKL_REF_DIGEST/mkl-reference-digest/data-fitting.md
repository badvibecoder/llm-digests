# Data Fitting

oneMKL's **experimental** Data Fitting component provides spline-based interpolation: spline
construction (linear, cubic, quadratic etc. in principle; only linear and cubic Hermite are
implemented here), cell-search internally, and approximation of functions, function derivatives, or
integrals. The domain is organized as (1) common terms — glossary, mathematical notation, and the
"hint" enumerations that describe memory layouts; (2) the `spline` class template that the caller
fills with user-owned USM memory and then constructs; (3) the free function `interpolate` that
evaluates function values and derivative values at interpolation sites; plus error handling and a
device support matrix. Everything is driven by raw `FpType*` pointers into USM memory that the user
allocates and owns; the library exposes no buffer-based API.

Scope note: the immediately preceding pages of the source (1169–1182: `distributed_descriptor`,
`compute_forward`, `compute_backward` in `oneapi::mkl::experimental::dft`) belong to the distributed
DFT domain, not to Data Fitting, and are not documented in this chapter.

## Overview

**Status / stability.** "APIs are experimental. It means that no API or ABI backward compatibility
are guaranteed." APIs are based on SYCL USM (the source prints "Unfied Shared Memory") input
"datas".

**Header and namespace.** Header File: `#include<oneapi/mkl/experimental/data_fitting.hpp>`.
Namespace as printed in the Splines, Linear Spline, Cubic Splines and Interpolate Function sections:
`oneapi::mkl::experimental::data_fitiing` (three `i`s — reproduced verbatim as printed). The Examples
page instead writes, verbatim, `namespace df = oneapi::mkl::experimental::data_fitting;` (correct
spelling). Both spellings occur in the source; treat this as a documentation defect and check the
installed header for the real name.

**Model.** A `spline<FpType, SplineType, Dimensions>` object is a thin wrapper over memory the *user*
provides:
- `set_partitions` gives the partition points `x_1..x_n` (`nx` values),
- `set_function_values` gives the `ny * nx` function values,
- `set_coefficients` gives storage of size `get_required_coeffs_size()`; coefficients are computed by
  `construct` when `were_coeffs_computed == false`,
- `construct` submits a SYCL kernel that computes coefficients and returns a `sycl::event`,
- `interpolate` evaluates values/derivatives at sites into a user array.
Copy/move construction and assignment are deleted "since the spline class is just a wrapper over
memory that users provide. Memory management responsibility is on user's side."

**Precision.** `FpType` ("`T`" in the source's prose) "can only be float or double".

**Hints (memory-layout contracts).** The caller declares the layout of each array; if the actual
layout does not satisfy the hint, behavior is undefined (stated for partitions, function values and
coefficients; same wording pattern for sites/results in `interpolate`).

**Error handling model (p1183–1184).** The DPC++ error handling model supports two types of errors:
1. Synchronous errors cause the DPC++ host runtime libraries to throw exceptions.
2. Asynchronous errors may only be processed in a user-supplied error handler associated with a SYCL
   queue.
For routines, handling all errors, synchronous or asynchronous, is the responsibility of the caller.
Exceptions are thrown explicitly by algorithms in the following scenarios: input parameters are
unexpected; provided SYCL device is not supported; spline is not fully initialized. Exceptions thrown
by runtime libraries at the host CPU, including DPC++ synchronous exceptions, are passed through to
the caller. DPC++ asynchronous errors are not handled.

**Device support (Appendix A, p1208–1209 — "Data Fitting Functionality" tables).**

| Splines, Spline Type | CPU | Intel GPU |
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

The same Appendix section also lists, separately from the spline types, the Data Fitting computation
and callback functionalities — `Construct1D` and `Interpolate1D` (Yes on CPU and Intel GPU, since oneMKL
2022.1), and `Interpolate1DEx`, `Integrate1D`, `Integrate1DEx`, `SearchCells1D`, `SearchCells1DEx`,
`InterpolationCallBack`, `IntegrateCallBack`, `SearchCellsCallBack` (all No on both devices). The
appendix gives no signatures for them, and they are not part of the C++ spline API documented here.

The routine pages themselves state no further device restriction; the only device-related statement is
the error-handling rule that an exception is thrown when "provided SYCL device is not supported".

**Glossary / model of a spline (p1184).** Assume we need to interpolate a function `f(x)` on the
`[a, b)` interval using splines. Break `[a, b)` into sub-intervals by `n` points (called partition
points, or simply partition); function values at these points are also given. A spline has degree `k`
if it can be expressed by the polynomial shown in Formulas. Splines are constructed on the
sub-intervals, so for each sub-interval there is one polynomial `P_i(x)`; the `c` values are called
spline coefficients. The function is interpolated at points of `[a, b)`; such points are called
interpolation sites, or simply sites. "Sites might or migtn't equals to partition points" (verbatim).

**Mathematical notation (p1185) — concept list only.** The source's notation table defines: Partition,
Uniform partition, Quasi-uniform partition, Vector-valued function of dimension `p` being fit, a
`k`-order derivative of `f(x)` at point `t`, `p` agrees with `f` at given points, multiplicity-`m`
repetition equality, the `k`-th divided difference of `f` at points (the leading coefficient of the
polynomial of order `k+1` that agrees with `f` at those points), and two special divided-difference
cases. The corresponding equations were formula images that did **not** survive extraction; see
Explicit gaps.

## Routines

### spline (class template)

Common API for all spline types. An instance of `spline<T, ST, N>` creates the `N`-dimensional spline
that operates with the `T` data type; `ST` is a type of spline; `T` can only be `float` or `double`.

Syntax:

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

Include Files: `#include<oneapi/mkl/experimental/data_fitting.hpp>`.

Constructors:
- `spline(const sycl::queue& q, std::int64_t ny = 1, bool were_coeffs_computed = false)` — create an
  object with the `q` SYCL queue and `ny` number of functions. If spline coefficients were already
  computed, provide `true` as the 3-rd parameter. `were_coeffs_computed == false` by default, meaning
  `construct` must be called to compute spline coefficients.
- `spline(const sycl::device& dev, const sycl::context& ctx, std::int64_t ny = 1, bool were_coeffs_computed = false)`
  — create an object using the `dev` SYCL device, the `ctx` context and `ny` number of functions;
  otherwise identical semantics.

Destructor: `~spline()`.

Restrictions: copy/move constructor and copy/move assignment operators are deleted. Users own the
memory the spline wraps.

NOTE: the printed default arguments `storage_hint::row_major` for `set_function_values` and
`set_coefficients` name a `storage_hint` that is not declared anywhere in this section — a source
defect; the intended type is the respective `function_hint` / `coefficient_hint`.

### spline::set_partitions

Set partition values from the `input_data` memory pointer, `nx` partition values.

```cpp
spline& set_partitions(
  FpType* input_data,
  std::int64_t nx,
  partition_hint PartitionHint = partition_hint::non_uniform);
```

Inputs: `input_data` — pointer to the `nx` partition values; `nx` — number of partition values;
`PartitionHint` — layout hint for the partition data (default `non_uniform`).
Returns a reference to the spline object for which partitions are set.

Restrictions: if `uniform` is specified, `nx` must equal `2` and `input_data` must contain only 2
values — the left and the right borders of the partition. Otherwise, behavior is undefined. If the
`input_data` layout doesn't satisfy `PartitionHint`, behavior is undefined. Example given by the
source for `uniform`: let the partition be the uniform one; `input_data` must contain only 2 values
`1, n` (the accompanying formula is missing from extraction).

### spline::set_function_values

Set function values from the `input_data` memory pointer.

```cpp
spline& set_function_values(
  FpType* input_data,
  function_hint FunctionHint = storage_hint::row_major);
```

Inputs: `input_data`; `FunctionHint` — layout hint (default `row_major`).
Number of function values must equal `ny * nx` elements.
Returns a reference to the spline object for which function values are set.
Restriction: if the `input_data` layout doesn't satisfy `FunctionHint`, behavior is undefined.

### spline::set_coefficients

Set coefficients from the `data` memory pointer.

```cpp
spline& set_coefficients(
  FpType* data,
  coefficient_hint CoeffHint = storage_hint::row_major);
```

Inputs: `data`; `CoeffHint` — layout hint (default `row_major`).
Number of coefficients in the memory must equal the return value of `get_required_coeffs_size()`.
If `were_coeffs_computed == false`, `data` will be rewritten during the `construct` call.
Returns a reference to the spline object for which coefficients are set.
Restriction: if the `data` layout doesn't satisfy `CoeffHint`, behavior is undefined.

### spline::set_internal_conditions

```cpp
spline& set_internal_conditions(
  FpType* input_data);
```

Set internal conditions specified by the `input_data` memory pointer. Number of internal condition
values must equal `nx - 2`. Returns a reference to the spline object for which internal conditions are
set. "There are some splines that requires internal conditions and boundary conditions to be set. For
such spline types, the following member functions must be called." (i.e. `set_internal_conditions` and
`set_boundary_conditions`).

### spline::set_boundary_conditions

```cpp
spline& set_boundary_conditions(
  bc_type BCType = bc_type::free_end,
  FpType input_value = {});
```

Set the `input_value` boundary condition corresponding to `BCType`. Default value for `input_value`
is empty since some boundary conditions don't require a value to be provided. Returns a reference to
the spline object for which the boundary condition value is set.

### spline::is_initialized

```cpp
bool is_initialized() const;
```

Returns `true` if all required data are set (for example, partitions, function values, coefficients).

### spline::get_required_coeffs_size

```cpp
std::int64_t get_required_coeffs_size() const;
```

Returns amount of memory that is required for coefficients storage.

### spline::construct

```cpp
sycl::event construct(const std::vector<sycl::event>& dependencies = {});
```

Constructs the spline (calculates spline coefficients if `were_coeffs_computed == false`). The
function submits a SYCL kernel and returns the SYCL event to wait on to ensure computation is
complete. `dependencies` is a list of SYCL events to wait for before starting computations (default:
empty).

### linear_spline::default_type

Linear spline is a spline whose degree is equal to 1 (see Formulas). Header File:
`#include<oneapi/mkl/experimental/data_fitting.hpp>`; Namespace as printed:
`oneapi::mkl::experimental::data_fitiing`.

```cpp
namespace linear_spline {
  struct default_type {};
}
```

Example (verbatim):

```cpp
spline<float, linear_spline::default_type> val(
  /*SYCL queue object*/q,
  /*number of spline functions*/ny
);
```

### cubic_spline::hermite

Cubic splines are splines whose degree is equal to 3. "There are a lot of different types of cubic
splines: Hermite, natural, Akima, Bessel. However, the current version of DPC++ API supports only one
type: Hermite." Header File: `#include<oneapi/mkl/experimental/data_fitting.hpp>`; Namespace as
printed: `oneapi::mkl::experimental::data_fitiing`.

```cpp
namespace cubic_spline {
  struct hermite {};
}
```

Example (verbatim):

```cpp
spline<float, cubic_spline::hermite> val(
  /*SYCL queue object*/q,
  /*number of spline functions*/ny
);
```

Supported boundary conditions for the Hermite spline: Free end, Periodic, First derivative, Second
Derivative.

### interpolate

Computes function and derivative values at interpolation sites. If the sites do not belong to the
interpolation interval `[a, b]`, the library uses (a) the interpolant whose coefficients were computed
for the interval identified by a formula image for computations at sites to the left of `a`, and (b)
the interpolant whose coefficients were computed for the interval identified by a formula image for
computations at sites to the right of `b` (both interval identifiers are missing from extraction; see
Explicit gaps). The interpolation algorithm depends on the interpolant's type (e.g., for cubic spline
interpolation, evaluation of a third-order polynomial is performed to obtain function values).
Header File: `#include<oneapi/mkl/experimental/data_fitting.hpp>`. Namespace as printed:
`oneapi::mkl::experimental::data_fitiing`.

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

Overload semantics, exactly as numbered in the source:
1. Performs computations of function values only using the SYCL queue associated with `interpolant`.
2. Performs computations of certain derivatives (function values is considered as a zero derivative)
   which are indicated in `der_indicator` (each bit corresponds to certain derivative starting from
   lower bit) using the SYCL queue associated with `interpolant`.
3. Performs computations of function values only using `q` as an input argument that should be created
   from the same context and device as the SYCL queue associated with `interpolant`.
4. Performs computations of certain derivatives (function values is considered as a zero derivative)
   which are indicated in `der_indicator` (each bit corresponds to certain derivative starting from
   lower bit) using `q` as an input argument that should be created from the same context and device
   as the SYCL queue associated with `interpolant`.

For all functions users can provide `SiteHint` and `ResultHint` to specify the layout of sites and
results respectively. If the results layout doesn't satisfy `ResultHint` and/or the sites layout
doesn't satisfy `SiteHint`, behavior is undefined. Returns the SYCL event of the submitted task.

NOTE: `interpolate` is templated on `typename Interpolant` and takes `typename Interpolant::fp_type*`
for `sites` and `results`, but the `spline` class definition in this section declares only
`using value_type = FpType;` and `using spline_type = SplineType;`. `fp_type` is not defined anywhere
in the extracted text.

Overload note: overloads (1) and (3) have **no** default argument for `dependencies`; overloads (2)
and (4) default it to `{}`.

### Example: linear spline + interpolate (p1196–1197)

Reproduced verbatim from the Examples section, including the extra closing parenthesis the source
prints on the `sites[i] = ...` line:

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

Observations a reader can rely on: the linear spline coefficient array is allocated as
`2 * (nx - 1)` elements; partitions/function values/coefficients are chained through the returned
references; `construct` and `interpolate` are serialized by passing the `sycl::event` from the
previous call into `dependencies`; results are `nsites` values for the default
`interpolate_hint::funcs_sites_ders` with one function and one derivative.

### partition_hint

Partition hints. Supported: Non-uniform; Quasi-uniform; Uniform.

```cpp
enum class partition_hint {
  non_uniform,
  quasi_uniform,
  uniform
};
```

### function_hint

Layout of the function-values array (one-dimensional array with `nx * ny` elements). Supported:
Row major; Column major.

```cpp
enum class function_hint {
  row_major,
  col_major
};
```

### coefficient_hint

Coefficients are stored in a one-dimensional array. For a cubic spline it requires 4 coefficients per
each interpolation interval and function value, i.e. `4 * (nx - 1) * ny` elements. Supported: row
major only.

```cpp
enum class coefficient_hint {
  row_major
};
```

### site_hint

Sites hints. Supported: Non-uniform; Uniform; Sorted.

```cpp
enum class site_hint {
  non_uniform,
  uniform,
  sorted
};
```

### interpolate_hint

Layout of interpolation results. With `ny` functions, `nsite` sites and `d` derivatives (including
interpolation values) the result memory size is `nsite * ny * d` elements. Six layouts are described
in prose — functions-sites-derivatives, functions-derivatives-sites, sites-functions-derivatives,
sites-derivatives-functions, derivatives-functions-sites, derivatives-sites-functions — but only the
first four are supported, matching the enumerators:

```cpp
enum class interpolate_hint {
  funcs_sites_ders,
  funcs_ders_sites,
  sites_funcs_ders,
  sites_ders_funcs
};
```

### derivatives

Derivative-order selector for `interpolate`: just compute interpolation values; compute first
derivative of the spline polynomial only; compute second derivative of the spline polynomial only;
compute third derivative of the spline polynomial only.

```cpp
enum class derivatives {
  zero,
  first,
  second,
  third
};
```

`operator|` is overloaded to create combinations of derivative orders to be computed by `interpolate`.
Example (verbatim), for interpolation values plus 1-st and 3-rd derivatives:

```cpp
std::bitset<32> bit_mask = derivatives::zero | derivatives::first | derivatives::third;
```

### bc_type

Boundary condition types for splines that require them. Supported: Free end; Periodic; First
derivative; Second Derivative.

```cpp
enum class bc_type {
  free_end,
  first_left_der,
  first_right_der,
  second_left_der,
  second_right_der,
  periodic
};
```

NOTE: (1) First derivative and second derivative types must be set on the left and on the right
borders. (2) Free end doesn't require any values to be set.

## Formulas

Spline polynomial, general degree (p1184):
- `P(x) = c_1 + c_2 (x - x_i) + c_3 (x - x_i)^2 + ... + c_{k-1} (x - x_i)^k`
- `P_i(x) = c_{1,i} + c_{2,i} (x - x_i) + c_{3,i} (x - x_i)^2 + ... + c_{k-1,i} (x - x_i)^k`

Linear spline (p1192, p1193):
- `P_i(x) = c_{1,i} + c_{2,i} (x - x_i),`
- `x ∈ [x_i, x_{i+1}),`
- `c_{1,i} = f(x_i),`
- `c_{2,i} = [x_i, x_{i+1}] f,`   (first-order divided difference)
- `i = 1, ..., n - 1.`

Cubic spline (p1193, p1194):
- `P_i(x) = c_{1,i} + c_{2,i} (x - x_i) + c_{3,i} (x - x_i)^2 + c_{4,i} (x - x_i)^3,`
- `x ∈ [x_i, x_{i+1}),`
- `i = 1, ..., n - 1.`

Hermite spline coefficients (p1194) — "Coefficients of Hermite spline are calculated using the
following formulas" (here `Δx_i` is the interval width as printed; `[x_i, x_{i+1}] f` is the
first-order divided difference and `f^{(1)}(x_i)` is the first derivative of `f` at `x_i`):
- `c_{1,i} = f(x_i),`
- `c_{2,i} = s_i,`
- `c_{3,i} = ([x_i, x_{i+1}] f - s_i) / (Δx_i) - c_{4,i} (Δx_i),`
- `c_{4,i} = (s_i + s_{i+1} - 2 [x_i, x_{i+1}] f) / (Δx_i)^2,`
- `s_i = f^{(1)}(x_i).`

## Conventions & Gotchas

- **Indexing / partition model.** `n` partition points define `n - 1` sub-intervals; coefficients are
  per interval, with `i = 1, ..., n - 1`. Polynomial `P_i` applies on `x ∈ [x_i, x_{i+1})` (left-closed,
  right-open).
- **Integer types.** `nx`, `ny`, `n_sites` and `get_required_coeffs_size()` are `std::int64_t`.
- **Precision.** `FpType` can only be `float` or `double`.
- **Memory model.** All data is raw USM memory (`FpType*`) supplied by the caller; there is no buffer
  API for Data Fitting, and the spline object does not own or free the arrays. Copy/move construction
  and assignment are deleted for exactly this reason.
- **Coefficient sizing.** Always allocate exactly `get_required_coeffs_size()` values for
  `set_coefficients`; the example allocates `2 * (nx - 1)` for a linear (degree-1) spline, consistent
  with `(degree + 1) * (nx - 1) * ny`. The prose statement `4 * (nx - 1) * ny` is given for the cubic
  case. Passing a different size is not permitted by the `set_coefficients` contract.
- **`were_coeffs_computed`.** `false` (default) means coefficients are computed by `construct` and the
  user buffer passed to `set_coefficients` is rewritten during `construct`; `true` means the user
  already computed them.
- **Layout hints are unchecked contracts.** If partition / function-value / coefficient / site /
  result layout does not satisfy the corresponding hint, behavior is undefined (the source states no
  diagnostic).
- **Uniform partition special case.** With `partition_hint::uniform`, `nx` must equal `2` and the
  array must hold only the left and right borders; otherwise behavior is undefined.
- **Dependencies / asynchrony.** Both `construct` and `interpolate` take
  `const std::vector<sycl::event>& dependencies`; `construct` defaults to `{}`, `interpolate`
  overloads (2) and (4) default to `{}` while (1) and (3) have no default. Both return a `sycl::event`
  to track completion. Overloads (3) and (4) take a `sycl::queue& q` that "should be created from the
  same context and device as the SYCL queue associated with `interpolant`".
- **Derivative selection.** Function value counts as the zero derivative; derivative orders are
  combined with the overloaded `operator|` on `derivatives` into a `std::bitset<32>` passed as
  `der_indicator`, "each bit corresponds to certain derivative starting from lower bit".
- **Boundary conditions.** First/second derivative conditions must be set on **both** the left and the
  right borders; `bc_type::free_end` requires no value (`input_value` defaults to `{}`).
- **Internal conditions.** `set_internal_conditions` expects exactly `nx - 2` values.
- **Initialization check.** `is_initialized()` returns `true` only when all required data are set;
  algorithms throw when the spline is not fully initialized.
- **No workspace/scratchpad argument.** Neither the constructors, `construct`, nor `interpolate`
  take a scratchpad; no scratchpad sizing rule is stated.
- **Namespace spelling defect.** The Splines/Syntax pages print `oneapi::mkl::experimental::data_fitiing`;
  the example prints `oneapi::mkl::experimental::data_fitting`. Verify against the installed header.
- **Default-argument defect.** `set_function_values`/`set_coefficients` print
  `storage_hint::row_major`, a type that is not declared in the documented API.
- **`fp_type` defect.** `interpolate` requires `typename Interpolant::fp_type`; the documented
  `spline` class exposes `value_type` and `spline_type`.
- **Supported splines.** Only `linear_spline::default_type` and `cubic_spline::hermite` are supported
  in this API version (device matrix: CPU and Intel GPU, since oneMKL 2022.1); quadratic, Subbotin,
  natural, Akima, Bessel, Hyman, lookup interpolant, cr stepwise const interpolant and cl stepwise
  const interpolant are "No" on both.
- **Out-of-range sites.** Sites left of `a` / right of `b` are handled with the coefficients of an end
  interval rather than by rejecting the site.
- **Errors.** Explicit throws occur for unexpected input parameters, an unsupported SYCL device, and a
  not-fully-initialized spline; host-side runtime exceptions (including DPC++ synchronous exceptions)
  pass through; DPC++ asynchronous errors are not handled by the library (use a SYCL queue error
  handler).

## Explicit gaps

- The equations in the "Mathematical Notation in the Data Fitting Component" table (p1185) did not
  survive extraction and were not captured as formula images: the partition definitions, uniform and
  quasi-uniform partition conditions and constant `C`, the vector-valued function definition, the
  `k`-order derivative notation, the agreement/multiplicity equalities, the `k`-th divided difference
  definition, and the two "in particular" cases are all absent. Only the concept column survives.
- The index-mapping formulas for row-major and column-major function values (p1186) and for the six
  interpolation-result layouts (p1187) are missing (the layout names and total element counts survive).
- The uniform-partition example on p1190 references a partition formula that is missing, so the exact
  meaning of "input_data must contain only 2 values: 1, n" cannot be confirmed from the formulas.
- The formula for the "Free end" boundary condition (printed as "Free end ()" on p1188 and p1194) is
  missing.
- In the Interpolate section (p1195) the two interval identifiers used for sites to the left of `a`
  and to the right of `b` are formula images that are missing, so the exact extrapolation intervals
  cannot be stated.
- The source gives no default value for `site_hint` semantics beyond the signature default
  `site_hint::non_uniform`; the `sorted`/`uniform` site behavior is not otherwise specified.
- No exception type list specific to `interpolate` or to `construct` is given; only the general error
  handling model and the three explicit throw scenarios are stated.
- `Interpolant::fp_type` is used by `interpolate` but never defined in the documented `spline` class
  (which defines `value_type` / `spline_type`).
- Pages 1169–1182 (`distributed_descriptor` and the distributed DFT `compute_forward` /
  `compute_backward`) appear in the same source chunks but belong to the DFT domain and are not part
  of this chapter.
