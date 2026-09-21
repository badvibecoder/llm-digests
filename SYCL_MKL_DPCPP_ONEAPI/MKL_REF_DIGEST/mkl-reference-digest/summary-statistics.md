# Summary Statistics

oneMKL Summary Statistics provides Data Parallel C++ interfaces that compute basic statistical
estimates for single- and double-precision multi-dimensional datasets (pages 1096-1131 of the
2026.0 *oneMKL - Data Parallel C++ Developer Reference*). The domain is one data container,
`oneapi::mkl::stats::dataset` (built with the `oneapi::mkl::stats::make_dataset` service function),
plus free-function routines that each read that dataset and write per-dimension results into
caller-provided buffers or USM arrays. Every routine has a Buffer API form (returns `void`, takes
`sycl::buffer`) and a USM API form (returns `sycl::event`, takes raw pointers plus a dependencies
vector). The routines cover raw/central sums and moments up to fourth order, variation coefficient,
skewness and excess kurtosis, and minimum/maximum; several have an overload taking a user-provided
mean.

## Overview

**Namespace / header.** Everything in this domain is in namespace `oneapi::mkl::stats`, declared in
header `oneapi/mkl/stats.hpp`.

**Model.** A multi-dimensional dataset is a matrix of observations: `n_dims` dimensions (variables,
`p` in the formulas) by `n_observations` observations (`n` in the formulas). Struct member
`observations` holds that matrix, row-major or column-major (`oneapi::mkl::stats::layout::row_major`
or `oneapi::mkl::stats::layout::col_major`). Optional `weights` (length `n_observations`,
non-negative; if not specified each observation is assigned weight 1) and optional `indices`
(length `n_dims`, `std::int64_t`; components to process; if not specified all components of the
vector are processed) complete the dataset.

**Usage model (as stated).** (1) Create and initialize the object for the dataset
(`make_dataset` is the documented helper). (2) Call the summary statistics routine to calculate the
appropriate estimate. The documented example computes `mean` for a 3-dimensional dataset, once with
the Buffer API and once with the USM API; both use
`oneapi::mkl::stats::make_dataset<mkl::stats::layout::row_major>(n_dims, n_observations, x)` then
`oneapi::mkl::stats::mean(queue, dataset, mean)`. Additional examples are stated to be in
`${MKL}/share/doc/mkl/examples/sycl/stats/source`. Raw-pointer USM is said to be usable via
`sycl::malloc_shared`/`malloc_device`.

**Precision support.** Single and double precision only: `Type` is `float` or `double`. The
buffer-based dataset uses `sycl::buffer<float,1>` / `sycl::buffer<double,1>`; the USM-based dataset
uses `float*` / `double*`.

**Device support (verbatim).** "All Data Parallel C++ routines of oneMKL summary statistics support
the CPU and GPU devices." The two device types listed are CPU device (computations on a CPU using
OpenCL(TM)) and GPU device (computations on a GPU using OpenCL(TM) or Level Zero).

**Computation methods.** Every routine takes `method Method = method::fast`; documented values are
`oneapi::mkl::stats::method::fast` and `oneapi::mkl::stats::method::one_pass`. `one_pass` is listed
only for `raw_sum`, `central_sum`, `raw_moment`, `central_moment`, `mean`, `variation`, `skewness`,
`kurtosis`; the "with User-provided Mean" overloads and `min`/`max`/`min_max` list only
`method::fast`. The reference pages assigned here do not explain the numerical difference between
`fast` and `one_pass`.

**Estimates provided (Definitions list, verbatim).** Raw and central sums/moments up to the fourth
order; variation coefficient; skewness and excess kurtosis (further referred to as a kurtosis);
minimum and maximum.

## Routines

Applies to every routine in this section:
- Include Files: `oneapi/mkl/stats.hpp`.
- Template parameters: `oneapi::mkl::stats::method Method = oneapi::mkl::stats::method::fast` (the
  Syntax blocks write it unqualified as `method Method = method::fast`); `typename Type`;
  `layout ObservationsLayout`.
- Input parameters common to all: `queue` (`sycl::queue&`) — a valid `sycl::queue`, the routine
  submits kernels in it; `data` (`const dataset<ObservationsLayout, Type*>&` in the parameter
  tables; `const dataset<ObservationsLayout, sycl::buffer<Type, 1>>&` in the Buffer signatures) —
  the dataset used for estimates computation.
- Optional input for every USM API: `dependencies` (`const std::vector<sycl::event>&`) — list of
  events to wait for before starting computation, if any.
- Every USM API returns `event` (`sycl::event`): "Function returns event after submitting task in
  `sycl::queue`."
- Output arrays are of length `n_dims`. Buffer outputs default to `{0}` and USM outputs default to
  `nullptr` unless noted.

### dataset

Consolidates the information of a multi-dimensional dataset (`n_dims`, `n_observations`, layout,
observations, optional weights, optional indices).

Primary template declaration, verbatim from the source (malformed exactly as printed):

```cpp
namespace oneapi::mkl::stats {
  template<layout ObservationsLayout = layout::row_major,
  typename Type = float*>
namespace oneapi::mkl::stats {
  struct dataset {}
}
```

Buffer API specialization:

```cpp
namespace oneapi::mkl::stats {
  template<layout ObservationsLayout, typename Type>struct dataset<ObservationsLayout,
sycl::buffer<Type, 1>> {
     explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_,
           sycl::buffer<Type, 1> observations_, sycl::buffer<Type, 1> weights_ = {0},
           sycl::buffer<std::int64_t, 1> indices_ = {0}) : n_dims(n_dims_),
n_observations(n_observations_), observations(observations_),
           weights(weights_), indices(indices_);
 std::int64_t n_dims;
 std::int64_t n_observations;
 sycl::buffer<Type, 1> observations;
 sycl::buffer<Type, 1> weights = {0};
 sycl::buffer<std::int64_t, 1> indices = {0};
 static constexpr layout layout = ObservationsLayout;
 };
}
```

USM API specialization:

```cpp
namespace oneapi::mkl::stats {
 template< layout ObservationsLayout, typename Type>
 struct dataset<ObservationsLayout, Type*> {
      explicit dataset(std::int64_t n_dims_, std::int64_t n_observations_, Type* observations_,
           Type* weights_ = nullptr, std::int64_t* indices_ = nullptr) :
           n_dims(n_dims_), n_observations(n_observations_),
           observations(observations_),
           weights(weights_), indices(indices_);
 std::int64_t n_dims;
 std::int64_t n_observations;
 Type* observations;
 Type* weights = nullptr;
 std::int64_t* indices = nullptr;
 static constexpr layout layout = ObservationsLayout;
 };
}
```

Template parameters / members: `DataType` — type of dataset, may be `sycl::buffer<float,1>`,
`sycl::buffer<double,1>` (buffer-based) or `float*`, `double*` (USM-based).
`oneapi::mkl::stats::layout ObservationsLayout` — layout of the observations matrix, values
`oneapi::mkl::stats::layout::row_major`, `oneapi::mkl::stats::layout::col_major`. Members:
`n_dims` (`std::int64_t`) the number of dimensions (variables); `n_observations`
(`std::int64_t`) the number of observations; `layout` (`oneapi::mkl::stats::layout`) row or column
major; `observations` (`sycl::buffer<Type, 1>` / `Type*`) matrix of observations; `weights`
(`sycl::buffer<Type, 1>` / `Type*`) array of weights of size `n_observations`, elements are
non-negative numbers, default weight 1 per observation; `indices`
(`sycl::buffer<std::int64_t, 1>` / `std::int64_t*`) array of vector components that will be
processed, size `n_dims`, default all components.

### oneapi::mkl::stats::make_dataset

Entry point to create a dataset from the provided parameters.

Buffer API:

```cpp
template<layout ObservationsLayout = layout::row_major,
  typename Type> dataset<ObservationsLayout,
  sycl::buffer<Type, 1>> make_dataset(std::int64_t n_dims,
  std::int64_t n_observations,
  sycl::buffer<Type, 1> observations,
  sycl::buffer<Type, 1> weights = {0},
  sycl::buffer<std::int64_t, 1> indices = {0});
```

USM API:

```cpp
template<layout ObservationsLayout = layout::row_major,
  typename Type> dataset<ObservationsLayout,
  Type*> make_dataset(std::int64_t n_dims,
  std::int64_t n_observations,
  Type* observations, Type* weights = nullptr,
  std::int64_t* indices = nullptr);
```

Template parameter `DataType` — may be `float`, `double`; `ObservationsLayout` defaults to
`layout::row_major`. Inputs: `n_dims`, `n_observations` (`std::int64_t`), `observations` (matrix of
observations); optional `weights` (non-negative, size `n_observations`, default 1 each) and
`indices` (size `n_dims`, default all components of the random vector are processed). Returns
`oneapi::mkl::stats::dataset<ObservationsLayout, sycl::buffer<Type, 1>>` (Buffer API) or
`oneapi::mkl::stats::dataset<ObservationsLayout, Type*>` (USM API).

### oneapi::mkl::stats::raw_sum

Entry point to compute arrays of raw sums up to the 4th order.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void raw_sum(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> sum,
    sycl::buffer<Type, 1> raw_sum_2 = {0},
    sycl::buffer<Type, 1> raw_sum_3 = {0},
    sycl::buffer<Type, 1> raw_sum_4 = {0});
```

USM API, transcribed exactly as printed (the source erroneously repeats the `max` signature here;
see Explicit gaps):

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event max(
    sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* max,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `oneapi::mkl::stats::method::fast`, `oneapi::mkl::stats::method::one_pass`.
Outputs: `sum` (mandatory) — output array of sum; `raw_sum_2`, `raw_sum_3`, `raw_sum_4` (optional) —
raw sums of the 2nd, 3rd, 4th order.

### oneapi::mkl::stats::central_sum

Entry point to compute arrays of central sums up to the 4th order.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void central_sum(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> central_sum_2,
    sycl::buffer<Type, 1> central_sum_3 = {0},
    sycl::buffer<Type, 1> central_sum_4 = {0});
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event central_sum(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* central_sum_2,
    Type* central_sum_3 = nullptr,
    Type* central_sum_4 = nullptr,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Outputs: `central_sum_2` (mandatory — the only
one with no default, although the source Output Parameters tables label it "(optional)"),
`central_sum_3` (optional), `central_sum_4` (optional).

### oneapi::mkl::stats::central_sum with User-provided Mean

Entry point to compute arrays of central sums up to the 4th order with mean provided by the user.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void central_sum(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> central_sum_2,
    sycl::buffer<Type, 1> central_sum_3 = {0},
    sycl::buffer<Type, 1> central_sum_4 = {0});
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event central_sum(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* central_sum_2,
    Type* central_sum_3 = nullptr,
    Type* central_sum_4 = nullptr,
    const std::vector<sycl::event> &dependencies = {});
```

(the USM block above is printed with a trailing `;;` in the source text). Only
`oneapi::mkl::stats::method::fast` is listed for this overload. `mean` (`sycl::buffer<Type,1>` /
`Type*`) is the buffer/pointer to the array of mean values provided by the user (the USM parameter
table describes it as "Pointer to the output array of mean values provided by the user" although it
is the user-supplied mean). Outputs `central_sum_2`, `central_sum_3`, `central_sum_4` are printed
with no default values in this overload.

### oneapi::mkl::stats::raw_moment

Entry point to compute arrays of raw moments up to 4th order.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void raw_moment(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> mean,
    sycl::buffer<Type, 1> raw_moment_2 = {0},
    sycl::buffer<Type, 1> raw_moment_3 = {0},
    sycl::buffer<Type, 1> raw_moment_4 = {0});
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event raw_moment(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* mean,
    Type* raw_moment_2 = nullptr,
    Type* raw_moment_3 = nullptr,
    Type* raw_moment_4 = nullptr,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Outputs: `mean` (mandatory) — output array of
mean values; `raw_moment_2`, `raw_moment_3`, `raw_moment_4` (optional) — raw moments of the 2nd,
3rd, 4th order.

### oneapi::mkl::stats::central_moment

Entry point to compute arrays of central moments up to 4th order.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void central_moment(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> central_moment_2,
    sycl::buffer<Type, 1> central_moment_3 = {0},
    sycl::buffer<Type, 1> central_moment_4 = {0});
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event central_moment(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* central_moment_2,
    Type* central_moment_3 = nullptr,
    Type* central_moment_4 = nullptr,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Outputs: `central_moment_2` (mandatory),
`central_moment_3`, `central_moment_4` (optional).

### oneapi::mkl::stats::central_moment with User-provided Mean

Entry point to compute arrays of central moments up to the 4th order with mean provided by the
user.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void central_moment(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> central_moment_2,
    sycl::buffer<Type, 1> central_moment_3 = {0},
    sycl::buffer<Type, 1> central_moment_4 = {0});
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event central_moment(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* central_moment_2,
    Type* central_moment_3 = nullptr,
    Type* central_moment_4 = nullptr,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed for this overload. `mean`
(`sycl::buffer<Type,1>` / `Type*`) — array of mean values provided by the user. The three
`central_moment_*` outputs are printed with no default values in this overload.

### oneapi::mkl::stats::mean

Entry point to compute the arrays of mean values.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void mean(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> mean);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event mean(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* mean,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Output `mean` (`sycl::buffer<Type, 1>` /
`Type*`) — output array of mean values.

### oneapi::mkl::stats::variation

Entry point to compute the arrays of variation coefficients.

Buffer API, transcribed exactly as printed (the last line is `sycl::buffer<Type, 1> variation;` —
a `)` is missing; the page image also prints `sycl::buffer<Type, 1> variation;` with no closing
parenthesis):

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void variation(sycl::queue& queue,
    const dataset< ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> variation;
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event variation(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* variation,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Output `variation` (`sycl::buffer<Type, 1>` /
`Type*`) — output array of variation coefficients.

### oneapi::mkl::stats::variation with User-provided Mean

Entry point to compute the arrays of variation coefficients with the mean provided by the user.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void variation(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> variaion);
```

(the third parameter is spelled `variaion` in the source text; the page image shows `variation`)

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event variation(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* variation,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed for this overload. `mean`
(`sycl::buffer<Type,1>` / `Type*`) — array of mean values provided by the user. Output `variation` —
output array of variation coefficients.

### oneapi::mkl::stats::skewness

Entry point to compute the arrays of skewness values.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void skewness(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> skewness);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event skewness(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* skewness,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Output `skewness` (`sycl::buffer<Type, 1>` /
`Type*`) — output array of skewness values.

### oneapi::mkl::stats::skewness with User-provided Mean

Entry point to compute the arrays of skewness values with a mean provided by the user.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void skewness(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> skewness);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event skewness(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* skewness,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed for this overload. `mean`
(`sycl::buffer<Type,1>` / `Type*`) — array of mean values provided by the user. Buffer output
`skewness` (`sycl::buffer<Type, 1>`) — output array of skewness values. (The USM output table in
the source is misprinted: it names the parameter `variation` with description "Pointer to the
output array of variation coefficients".)

### oneapi::mkl::stats::kurtosis

Entry point to compute the array of kurtosis values (excess kurtosis, see the formula).

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void kurtosis(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> kurtosis);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event kurtosis(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* kurtosis,
    const std::vector<sycl::event> &dependencies = {});
```

Method values: `method::fast`, `method::one_pass`. Output `kurtosis` (`sycl::buffer<Type, 1>` /
`Type*`) — output array of kurtosis values.

### oneapi::mkl::stats::kurtosis with User-provided Mean

Entry point to compute the array of kurtosis values with a mean provided by the user.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void kurtosis(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> kurtosis);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event kurtosis(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* kurtosis,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed for this overload. `mean`
(`sycl::buffer<Type,1>` / `Type*`) — array of mean values provided by the user. Output `kurtosis`
(`sycl::buffer<Type, 1>` / `Type*`) — output array of kurtosis values.

### oneapi::mkl::stats::min

Entry point to compute the array of minimum values.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void min(
    sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> min);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event min(
    sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* min,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed. Output `min` (`sycl::buffer<Type, 1>` / `Type*`)
— output array of minimum values.

### oneapi::mkl::stats::max

Entry point to compute the array of maximum values.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void max(
    sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> max);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event max(
    sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* max,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed. Output `max` (`sycl::buffer<Type, 1>` / `Type*`)
— output array of maximum values.

### oneapi::mkl::stats::min_max

Entry point to compute the array of minimum and maximum values simultaneously.

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
void min_max(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> min, sycl::buffer<Type, 1> max);
```

```cpp
template<method Method = method::fast, typename Type, layout ObservationsLayout>
sycl::event min_max(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* min, Type* max,
    const std::vector<sycl::event> &dependencies = {});
```

Only `oneapi::mkl::stats::method::fast` is listed. Outputs: `min` (`sycl::buffer<Type, 1>` /
`Type*`) — output array of minimum values; `max` (`sycl::buffer<Type, 1>` / `Type*`) — output array
of maximum values.

### Edge overlap from adjacent domains (not Summary Statistics)

The source pages assigned to this chapter begin mid-way through the Random Number Generators
chapter (p1091-p1096) and end at the start of the Fourier Transform Functions chapter
(p1131-p1133). The names below are listed only so nothing from the assigned pages is lost; their
authoritative chapters are the RNG and Fourier Transform chapters.

The three RNG names below are declared in namespace `oneapi::mkl::rng::device`, header
`oneapi/mkl/rng/device.hpp`, as `template<typename RealType, typename Method> class ...` /
`template <typename Engine> class count_engine_adaptor` (full class syntax is the RNG chapter's
material; key members quoted below). The Fourier Transform overlap (p1131-p1133) contributes only
the DFT definition transcribed under Formulas below; its API names (`oneapi::mkl::dft::descriptor`,
`compute_forward`, `compute_backward`, and the scoped enumerations `config_param`, `precision`,
`domain`, `config_value`) are the `dft` chapter's material.

- `beta` — "Generates Beta distributed random numbers." `method_type`/`result_type` typedefs;
  default ctor `beta() : beta((RealType)1.0, (RealType)1.0, (RealType)0.0, (RealType)1.0){}`;
  `explicit beta(RealType p, RealType q, RealType a, RealType b)`; `explicit beta(const param_type& pt)`;
  accessors `p()`, `q()`, `a()`, `b()`, `param()`; `void param(const param_type& pt)`;
  `std::size_t count_rejected_numbers() const`. Template parameters `typename Type` (values `float`,
  `double`) and `typename Method` (values
  `oneapi::mkl::rng::device::beta_method::by_default`,
  `oneapi::mkl::rng::device::beta_method::cja`,
  `oneapi::mkl::rng::device::beta_method::cja_accurate`). Inputs: `p` shape p, `q` shape q, `a`
  displacement, `b` scalefactor. `count_rejected_numbers()` returns the amount of random numbers
  rejected during the last `generate` call; 0 if no generate calls.
- `gamma` — "Generates Gamma distributed random numbers." Default ctor
  `gamma() : gamma((RealType)1.0, (RealType)0.0, (RealType)1.0){}`;
  `explicit gamma(RealType alpha, RealType a, RealType beta)`;
  `explicit gamma(const param_type& pt)`; accessors `alpha()`, `a()`, `beta()`, `param()`;
  `void param(const param_type& pt)`; `std::size_t count_rejected_numbers() const`. Template
  parameters `typename Type` (`float`, `double`) and `typename Method` (values
  `oneapi::mkl::rng::device::gamma_method::by_default`,
  `oneapi::mkl::rng::device::gamma_method::marsaglia`,
  `oneapi::mkl::rng::device::gamma_method::marsaglia_accurate`). Inputs: `alpha` shape, `a`
  displacement a, `beta` scalefactor.
- `count_engine_adaptor` — random number engine adaptor that counts how many times random 32 bits
  were taken from the engine during the `generate` call; especially useful for distributions based
  on acceptance-rejection algorithms, such as beta, gamma. Members:
  `static constexpr std::int32_t vec_size = Engine::vec_size;`
  `explicit count_engine_adaptor(const Engine& engine);`
  `explicit count_engine_adaptor(Engine&& engine);`
  `template <typename... Params> count_engine_adaptor(Params... params);` (constructs an engine
  inside from a pack of input parameters); `std::int64_t get_count() const;` (the "Getters" prose
  instead says `std::size_t get_count() const` — the source is inconsistent); `const Engine& base() const;`
  (returns the underlying random engine).

## Formulas

Notation as printed: `p` = number of dimensions/variables, `n` = number of observations,
`omega_j` = weight of observation `j`, `x_{ij}` = observation `j` of dimension `i`, `i = 1,...,p`.

- `raw_sum` (p1102): `S^k(X) = (S_1^k(X), ..., S_p^k(X))`, where `S_i^k(X) = sum_{j=1}^{n} omega_j * x_{ij}^k`; where `i = 1,...,p`, `k = 1,2,3,4`.
- `central_sum` (p1104): `S^k(X) = (S_1^k(X), ..., S_p^k(X))`, where `S_i^k(X) = sum_{j=1}^{n} omega_j * (x_{ij} - S_i(X))^k`; where `S_i(X) = sum_{j=1}^{n} omega_j * x_{ij}`, `i = 1,...,p`, `k = 2,3,4`.
- `central_sum` with User-provided Mean (p1106): same expression as `central_sum`, `S_i^k(X) = sum_{j=1}^{n} omega_j * (x_{ij} - S_i(X))^k`, with `S_i(X)` supplied by the user; where `S_i(X) = sum_{j=1}^{n} omega_j * x_{ij}`, `i = 1,...,p`, `k = 2,3,4`.
- `raw_moment` (p1108): `R^k(X) = (R_1^k(X), ..., R_p^k(X))`, where `R_i^k(X) = (1/W) * sum_{j=1}^{n} omega_j * x_{ij}^k`; where `W = sum_{j=1}^{n} omega_j`, and the index clause is printed as `j = 1,...,p; k = 1,2,3,4` (the trailing index letter looks like `j` in the page image although `i` is the dimension index used everywhere else — treated as a source typo, not corrected).
- `central_moment` (p1110): `C^k(X) = (C_1^k(X), ..., C_p^k(X))`, where `C_i^k(X) = (1/W) * sum_{j=1}^{n} omega_j * (x_{ij} - M_i(X))^k`; where `W = sum_{j=1}^{n} omega_j`, `i = 1,...,p`, `k = 2,3,4`.
- `central_moment` with User-provided Mean (p1112): identical expression `C_i^k(X) = (1/W) * sum_{j=1}^{n} omega_j * (x_{ij} - M_i(X))^k`, with `M_i(X)` supplied by the user; where `W = sum_{j=1}^{n} omega_j`, `i = 1,...,p`, `k = 2,3,4`.
- `mean` (p1114): `M(X) = (M_1(X), ..., M_p(X))`, where `M_i(X) = (1/W) * sum_{j=1}^{n} omega_j * x_{ij}`; where `W = sum_{j=1}^{n} omega_j`, `i = 1,...,p`.
- `variation` (p1116): `VC(X) = (VC_1(X), ..., VC_p(X))`, `VC_i(X) = V_i^{0.5}(X) / M_i(X)`; where `i = 1,...,p`.
- `variation` with User-provided Mean (p1118): `VC(X) = (VC_1(X), ..., VC_p(X))`, `VC_i(X) = V_i^{0.5}(X) / M_i(X)` (same expression as `variation`); where `i = 1,...,p`.
- `skewness` (p1119): `Gamma(X) = (Gamma_1(X), ..., Gamma_p(X))`, `Gamma_i(X) = C_i^{(3)}(X) / V_i^{1.5}(X)`; where `i = 1,...,p`. (The source uses the Greek capital gamma, transliterated here as `Gamma`.)
- `skewness` with User-provided Mean (p1121): `Gamma(X) = (Gamma_1(X), ..., Gamma_p(X))`, `Gamma_i(X) = C_i^{(3)}(X) / V_i^{1.5}(X)` (same expression as `skewness`); where `i = 1,...,p`.
- `kurtosis` (p1123): `B(X) = (B_1(X), ..., B_p(X))`, `B_i(X) = C_i^{(4)}(X) / V_i^{2}(X) - 3`; where `i = 1,...,p`.
- `kurtosis` with User-provided Mean (p1124): `B(X) = (B_1(X), ..., B_p(X))`, `B_i(X) = C_i^{(4)}(X) / V_i^{2}(X) - 3` (same expression as `kurtosis`); where `i = 1,...,p`.
- `min` (p1126): `min(X) = (min_1(X), ..., min_p(X))`; where `i = 1,...,p`.
- `max` (p1128): `max(X) = (max_1(X), ..., max_p(X))`; where `i = 1,...,p`.
- `min_max` (p1129): prints both `max(X) = (max_1(X), ..., max_p(X))` ("Vector of maximum values") and `min(X) = (min_1(X), ..., min_p(X))` ("Vector of minimum values"); where `i = 1,...,p`.

Edge overlap — Random Number Generators (p1092-p1093):

- `beta` probability density function (p1092): `f_{p,q,alpha,beta}(x) = { 1 / (B(p,q) * beta^{p+q-1}) * (x - a)^{p-1} * (beta + alpha - x)^{q-1}, alpha <= x < alpha + beta;  0, x < alpha, x >= alpha + beta }`. Here `B` is the complete beta function; `alpha` is the displacement and `beta` the scale parameter in the reference's Greek notation, corresponding to constructor arguments `a` and `b`.
- `beta` cumulative distribution function (p1092): `F_{a,b}(x) = { 0, x < alpha;  integral_{alpha}^{x} 1/(B(p,q) * beta^{p+q-1}) * (y - alpha)^{p-1} * (beta + alpha - y)^{q-1} dy, alpha <= x < alpha + beta, x in R;  1, x >= alpha + beta }` (transcribed as printed, including the mixed `a,b` / `alpha,beta` subscripts in the source).
- `gamma` probability distribution (p1093): `f_{a,alpha,beta}(x) = { 1/(Gamma(alpha) * beta^{alpha}) * (x - a)^{alpha-1} * e^{-(x-a)/beta}, x >= a;  0, x < a }`.
- `gamma` cumulative distribution function (p1093): `F_{a,alpha,beta}(x) = { integral_{a}^{x} 1/(Gamma(alpha) * beta^{alpha}) * (y - a)^{alpha-1} * e^{-(y-a)/beta} dy, x >= a;  0, x < a }`.

Edge overlap — Fourier Transform Functions (p1131):

- DFT definition (p1131): `z^m_{k_1,k_2,...,k_d} = sigma_delta * sum_{j_d=0}^{n_d-1} ... sum_{j_2=0}^{n_2-1} sum_{j_1=0}^{n_1-1} z^m_{j_1,j_2,...,j_d} * exp[ delta * 2*pi*i * ( sum_{l=1}^{d} j_l*k_l / n_l ) ]`, for all `m` in `{0,1,...,M-1}`, where `i` is the imaginary unit, `delta` selects the direction of the DFT and `sigma_delta` is the scaling factor associated with that direction.

## Conventions & Gotchas

- **Namespace / header.** All routines here are `oneapi::mkl::stats`, header `oneapi/mkl/stats.hpp`.
- **Buffer API vs USM API.** Buffer API returns `void`; USM API returns `sycl::event`. Only the USM API has the trailing `const std::vector<sycl::event> &dependencies = {}` ("List of events to wait for before starting computation, if any"); there is no dependencies parameter in any Buffer API signature here.
- **Default arguments.** Buffer optional outputs default to `{0}` (empty buffer); USM optional outputs default to `nullptr`. Passing the default means that order is not computed.
- **Template parameter order is fixed:** `template<method Method = method::fast, typename Type, layout ObservationsLayout>`.
- **Mandatory vs optional outputs.** Mandatory (no default): `raw_sum`'s `sum`, `central_sum`'s `central_sum_2`, `raw_moment`'s `mean`, `central_moment`'s `central_moment_2`, and the single outputs of `mean`, `variation`, `skewness`, `kurtosis`, `min`, `max`, `min_max`. Optional: `raw_sum_2/3/4`, `central_sum_3/4`, `raw_moment_2/3/4`, `central_moment_3/4`. The "with User-provided Mean" overloads print no defaults for their outputs.
- **Output sizing.** One value per processed dimension, so every output array has length `n_dims` (`p`); `min_max` writes two such arrays.
- **Weights.** Size `n_observations`, elements non-negative; absent means each observation is weighted 1. `W = sum_{j=1}^{n} omega_j` appears in the `mean`, `raw_moment`, `central_moment` formulas but not in the raw/central-sum formulas.
- **indices.** `sycl::buffer<std::int64_t, 1>` / `std::int64_t*`, size `n_dims`; selects which vector components are processed; absent means all are processed.
- **Layout.** `ObservationsLayout` is `oneapi::mkl::stats::layout::row_major` or `layout::col_major`; `make_dataset` and the primary `dataset` template default it to `layout::row_major`.
- **Integer types.** `n_dims`, `n_observations` and the elements of `indices` are `std::int64_t`; the value type `Type` is `float` or `double` only.
- **Method availability.** `oneapi::mkl::stats::method::one_pass` is offered for `raw_sum`, `central_sum`, `raw_moment`, `central_moment`, `mean`, `variation`, `skewness`, `kurtosis` (no user-mean overloads, and not for `min`/`max`/`min_max`).
- **No scratchpad/workspace.** No signature in this domain takes a scratchpad or workspace argument, and no workspace-sizing requirement is stated.
- **In-place operation.** Outputs are separate arrays/buffers; `data` is always passed `const`-qualified. No in-place variant is documented.
- **Error handling.** No error conditions, exception types, or status codes are stated on these pages (there is no Errors/Exceptions subsection for this domain).
- **Source errata when copying from the PDF:** `raw_sum` USM API is printed with the `max` signature (`sycl::event max(..., Type* max, ...)`); `variation` Buffer API line ends `sycl::buffer<Type, 1> variation;` (the closing `)` is missing in the text layer and the page image alike); `variation` with User-provided Mean Buffer API spells the parameter `variaion`; `central_sum` with User-provided Mean USM API ends `...dependencies = {});;` (double semicolon); `skewness` with User-provided Mean USM output table names the parameter `variation`; `raw_moment`'s "Where:" index clause appears to use `j` where `i` is meant; `count_engine_adaptor`'s getter type differs between syntax block and prose.

## Explicit gaps

- The mathematical definitions ("Vector of ..." and "Where:") for `raw_sum`, `central_sum`, `central_sum` with User-provided Mean, `raw_moment`, `central_moment`, `central_moment` with User-provided Mean, `mean`, `min`, `max`, and `min_max` did **not** survive text extraction and have **no** cropped formula image (pages 1102, 1104, 1106, 1108, 1110, 1112, 1114, 1115, 1126, 1128, 1129 have no entry in the formula manifest). The formulas above were transcribed from the full-page raster images of those pages; the text layer never contained them.
- The `variation` / `skewness` / `kurtosis` "Where:" clauses define only `i = 1,...,p`. The symbols `V_i(X)`, `M_i(X)`, `C_i^{(3)}(X)`, `C_i^{(4)}(X)` used in those formulas are **not** defined on those pages (they must be read as the corresponding central-moment / mean estimates documented in this chapter).
- The `beta` page (p1091) parameter constraints (the "where ..., ..., ..., ..." inequalities and the definition of the complete beta function `B`) were inline formulas on a page with no formula image and are absent from the extracted text.
- The USM API signature of `oneapi::mkl::stats::raw_sum` is **not stated** in the source: the printed block duplicates the `max` signature, so the intended parameter list is not transcribed here (it is not invented).
- The usage-model example code is printed with apparent typos (`for(int i = 0; i < n_dims, i++)`, and the USM example passes `mean` where the declaration names `mean_buf`); not corrected here.
- A separate TOC section "Summary Statistics Functionality" exists at page 1209, outside the pages assigned to this chapter, and is not covered.
- No default value, device restriction, or error condition beyond the descriptions above is stated for `weights`/`indices`; where the source is silent this digest is silent.
