# Fourier Transform Functions

This domain covers `oneapi::mkl::dft` — the DPC++/SYCL interface for Discrete Fourier Transforms (DFTs) — plus the experimentally-namespaced distributed (multi-process, MPI-based) DFT interface `oneapi::mkl::experimental::dft`. It is organized around a descriptor object that captures the whole transform configuration, must be committed to a `sycl::queue`, and is then passed to `compute_forward` / `compute_backward`. Sources are pages 1131–1183 of the oneMKL - Data Parallel C++ Developer Reference (2026.0). The chunks that carry this chapter also begin with the tail of the Statistics domain (p1121–1131) and end with the opening of the Data Fitting domain (p1183–1184); those boundary items are recorded briefly at the ends of the `Routines` and `Formulas` sections so no named routine or transcribed formula is lost.

## Overview

- Single-process DFT: namespace `oneapi::mkl::dft`, header `oneapi/mkl/dft.hpp`.
- Distributed DFT: namespace `oneapi::mkl::experimental::dft`, header `oneapi/mkl/experimental/distributed_dft.hpp`.

Definitions (from p1131):

- Let `S` be a set of `M` finite `d`-dimensional discrete sequences `z^m` of lengths `(n_1, n_2, ..., n_d)`, `m` in `{0, 1, ..., M-1}`. The entries of the `m`-th sequence are denoted `z^m_{j_1,...,j_d}`, where the integer indices `j_l` satisfy `0 <= j_l <= n_l - 1` for `l` in `{1, ..., d}`.
- The DFT of `z` is the finite `d`-dimensional discrete sequence `zhat` with entries given by the formula transcribed in `## Formulas` (`DFT (p1131)`).
- `delta` determines one of the two "directions" of the DFT and `sigma_delta` is a scaling factor associated with that direction. `delta` defines the "forward DFT" (forward scaling factor) and the other sign defines the "backward DFT" (backward scaling factor). The explicit numeric values of `delta` and `sigma_delta` were set as inline math that did not survive text extraction (see `## Explicit gaps`).
- The calculation of `M` identically-defined DFTs for several data sets is a "batched DFT"; `M` is the "batch size".

Forward / backward domain:

- The domain of the input (resp. output) sequences for a forward (resp. backward) DFT is the "forward domain"; the output (resp. input) domain of a forward (resp. backward) DFT is the "backward domain".
- Forward domains are either the set of complex `d`-dimensional sequences ("complex forward domain") or the set of real `d`-dimensional sequences ("real forward domain"). Transforms are correspondingly "complex DFTs" or "real DFTs".
- Regardless of forward-domain type, backward-domain sequences are always complex, with constraints for real transforms.

Elementary range of non-redundant entries and real-DFT constraints (p1132):

- For real DFTs the backward-domain data is conjugate-symmetric and roughly half the entries are redundant. oneMKL expects/produces data sequences such that, for backward-domain data of a real DFT, the last index runs only over the elementary non-redundant set (conventionally `k_d = 0 ... n_d/2`, matching the `lengths[rank-1]/2 + 1` packing used in the supplied usage example).
- Additional constraints on backward-domain real-DFT entries: the imaginary part must be zero for any entry whose last index equals `n_d/2`, and pairs of entries symmetric about `n_d/2` must be complex conjugates of one another. The exact index expressions were inline math lost in extraction.
- NOTE: the behavior of oneMKL is undefined for a real backward DFT if the input data does not satisfy those constraints; it is the user's responsibility to guarantee them.

Four-step usage model (uncommitted-user-workspace case, p1133):

1. Create a descriptor with e.g. `descriptor<prec, dom> desc(lengths);`. On creation all configuration settings take default values.
2. Optionally adjust configuration with `set_value` (as many times as needed); query with `get_value` (defaults are returned unless a parameter was set).
3. `desc.commit(queue)` — freezes configuration and binds the descriptor to a device/queue.
4. Call `compute_forward` / `compute_backward` as many times as needed with the committed descriptor and device-accessible data.

Defaults at construction: in-place calculation, unbatched (`M = 1`), unscaled, with forward domain, precision and length(s) set at construction.

General conventions:

- The DFT functions assume the **Cartesian representation** of complex data (complex numbers defined by real and imaginary parts). oneMKL provides Vector Mathematical Functions for conversion to/from polar representation.
- Available precisions: `precision::SINGLE` (FP32) and `precision::DOUBLE` (FP64). Some GPU devices do not support double-precision descriptors; verify the target device supports FP64 before using a double-precision descriptor.
- The DPC++ interface supports CPU and Intel GPU devices with limitations documented per routine below.
- Data containers may be SYCL buffers (`sycl::buffer<T, 1>`) or device-accessible USM allocations (`T*`). Distributed DFT supports USM only.
- Complex storage in the default configuration is `config_value::COMPLEX_COMPLEX` (i.e. `std::complex<fp_type>`); the alternative `config_value::REAL_REAL` split storage is documented but **not implemented via the DPC++ interface of oneMKL**.
- Performance note from the source: for repeated identically-configured DFT computations, create/configure/commit descriptors outside the hot path; only `compute_forward` / `compute_backward` should be invoked in performance-critical sections.

## Routines

### oneapi::mkl::dft::precision

Scoped enumeration identifying the floating-point format considered by a descriptor. Declared in `oneapi/mkl/dft.hpp`.

```
enum class precision {
  SINGLE = DFTI_SINGLE,
  DOUBLE = DFTI_DOUBLE
};
```

`precision::SINGLE` = single-precision (FP32); `precision::DOUBLE` = double-precision (FP64). For any descriptor object this value is bound to its type (non-type template parameter) and immutable.

### oneapi::mkl::dft::domain

Scoped enumeration identifying the type of forward domain of a descriptor.

```
enum class domain {
   REAL    = DFTI_REAL,
   COMPLEX = DFTI_COMPLEX
};
```

`domain::REAL` = real forward domain; `domain::COMPLEX` = complex forward domain. Bound to the descriptor's template argument and immutable.

### oneapi::mkl::dft::config_param

Scoped enumeration representing all configuration parameters of a DFT descriptor.

```
enum class config_param {
   FORWARD_DOMAIN,
   DIMENSION,
   LENGTHS,
   PRECISION,
   FORWARD_SCALE,
   BACKWARD_SCALE,
   NUMBER_OF_TRANSFORMS,
   COMPLEX_STORAGE,
   PLACEMENT,
   FWD_DISTANCE,
   BWD_DISTANCE,
   WORKSPACE,
   COMMIT_STATUS,
   THREAD_LIMIT,
   DESTROY_INPUT,
   WORKSPACE_ESTIMATE_BYTES,
   WORKSPACE_BYTES,
   FWD_STRIDES,
   BWD_STRIDES,
   WORKSPACE_PLACEMENT,        // alias for WORKSPACE
   WORKSPACE_EXTERNAL_BYTES    // alias for WORKSPACE_BYTES
};
```

Meanings: `FORWARD_DOMAIN` type of forward domain; `DIMENSION` dimension `d`; `LENGTHS` length(s) `n_l`; `PRECISION` floating-point format; `FORWARD_SCALE` scaling factor of the forward DFT; `BACKWARD_SCALE` scaling factor of the backward DFT; `NUMBER_OF_TRANSFORMS` batch size `M`; `PLACEMENT` in-place or not; `COMPLEX_STORAGE` elementary data type considered in either domain by a complex descriptor (irrelevant for real descriptors); `FWD_STRIDES` / `BWD_STRIDES` index offset and strides for the forward/backward data container; `FWD_DISTANCE` / `BWD_DISTANCE` distance between successive sequences; `WORKSPACE` (alias `WORKSPACE_PLACEMENT`) user- vs oneMKL-managed workspace; `WORKSPACE_BYTES` (alias `WORKSPACE_EXTERNAL_BYTES`) exact workspace size in bytes; `WORKSPACE_ESTIMATE_BYTES` conservative workspace size estimate; `DESTROY_INPUT` whether input may be overwritten for out-of-place operations; `THREAD_LIMIT` CPU thread limit (irrelevant for GPU-committed objects); `COMMIT_STATUS` committed or not.

**Read-only (not writable) parameters:** `DIMENSION`, `FORWARD_DOMAIN`, `LENGTHS`, `PRECISION`, `COMMIT_STATUS`, `WORKSPACE_ESTIMATE_BYTES`, `WORKSPACE_BYTES`, `WORKSPACE_EXTERNAL_BYTES`.

### oneapi::mkl::dft::config_value

Scoped enumeration for configuration values not representable by `precision`, `domain`, or a primitive/derived type.

```
enum class config_value {
   COMMITTED,
   UNCOMMITTED,
   COMPLEX_COMPLEX,
   REAL_REAL,
   INPLACE,
   NOT_INPLACE,
   WORKSPACE_AUTOMATIC,        // alias for WORKSPACE_INTERNAL
   ALLOW,
   AVOID,
   WORKSPACE_INTERNAL,
   WORKSPACE_EXTERNAL
};
```

- `COMMITTED` / `UNCOMMITTED` → `config_param::COMMIT_STATUS`.
- `COMPLEX_COMPLEX` / `REAL_REAL` → `config_param::COMPLEX_STORAGE`.
- `INPLACE` / `NOT_INPLACE` → `config_param::PLACEMENT`.
- `WORKSPACE_AUTOMATIC` / `WORKSPACE_INTERNAL` / `WORKSPACE_EXTERNAL` → `config_param::WORKSPACE` (or alias `WORKSPACE_PLACEMENT`); `WORKSPACE_AUTOMATIC` and `WORKSPACE_INTERNAL` are equivalent and interchangeable.
- `ALLOW` / `AVOID` → `config_param::DESTROY_INPUT`.

### oneapi::mkl::dft::descriptor

Purpose: class template whose successfully-committed instances fully specify and configure a DFT computation. Declared in `oneapi/mkl/dft.hpp` as `template <precision prec, domain dom> class descriptor;`.

Template parameters (in order): a value of type `precision`; a value of type `domain`. Instances are called "single-/double-precision descriptors" and "complex/real descriptors" accordingly.

Constructors, assignment operators, destructor (p1139):

```
namespace oneapi::mkl::dft {
  template <precision prec, domain dom>
  class descriptor {
  public:
    // parameterized constructors:
    descriptor(std::vector<std::int64_t> dimensions);
    descriptor(std::int64_t length);
    // destructor:
    ~descriptor();
    // unsupported copy constructor and assignment operator:
    descriptor(const descriptor&) = delete;
    descriptor& operator=(const descriptor&) = delete;
    // move constructor and assignment operator:
    descriptor(descriptor&&);
    descriptor& operator=(descriptor&&);
  }
}
```

Input parameter for a 1-D DFT: `length`, `std::int64_t`, the length of the one-dimensional DFT. Input parameter for any descriptor: `lengths`, `std::vector<std::int64_t>`, vector of size `d` containing `n_1, ..., n_d` in that order. (The declaration spells the vector parameter `dimensions`; the parameter table names it `lengths`.)

Copy construction and assignment are **not yet supported**. Move construction/assignment are supported; a moved-from descriptor must be reinitialized (e.g. via assignment) before reuse — improper reuse may throw `oneapi::mkl::uninitialized`. The destructor frees all resources.

NOTES / restrictions: parameterized constructors allocate data structures and default-configure the object but do no significant computational work; that happens at `commit`. The constructors may throw `std::runtime_error` if [condition was inline math lost in extraction — appears to concern the supplied length(s)] or if construction fails to allocate required resources.

### oneapi::mkl::dft::descriptor::set_value

Purpose: assign a configuration value to any writable configuration parameter.

```
namespace oneapi::mkl::dft {
  template <precision prec, domain dom>
  class descriptor {
    using real_scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
  public:
    // for integer-valued parameters:
    void set_value(config_param param, std::int64_t value);
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    void set_value(config_param param, T value) {
        set_value(param, static_cast<std::int64_t>(value));
    }
    // for real-valued parameters:
    void set_value(config_param param, real_scalar_t value);
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void set_value(config_param param, T value) {
        set_value(param, static_cast<real_scalar_t>(value));
    }
    // for vector-valued parameters:
    void set_value(config_param param, const std::vector<std::int64_t>& value);
    // for other parameters:
    void set_value(config_param param, config_value value);
  }
}
```

Accepted parameters, accepted values, defaults: `FWD_DISTANCE` — all integers, default `0`; `BWD_DISTANCE` — all integers, default `0`; `NUMBER_OF_TRANSFORMS` — positive integers, default `1`; `THREAD_LIMIT` — non-negative integers, default `0`; `FORWARD_SCALE` — all floating-point values, default `1.0`; `BACKWARD_SCALE` — all floating-point values, default `1.0`; `FWD_STRIDES` / `BWD_STRIDES` — `std::vector<std::int64_t>` of size `d+1`, defaults documented in "Configuring data layouts" (not in these pages); `COMPLEX_STORAGE` — `config_value::COMPLEX_COMPLEX` or `config_value::REAL_REAL`, default `config_value::COMPLEX_COMPLEX`; `PLACEMENT` — `config_value::INPLACE` or `config_value::NOT_INPLACE`, default `config_value::INPLACE`; `WORKSPACE` (or `WORKSPACE_PLACEMENT`) — `config_value::WORKSPACE_AUTOMATIC` (≡ `WORKSPACE_INTERNAL`) or `config_value::WORKSPACE_EXTERNAL`, default `config_value::WORKSPACE_AUTOMATIC`; `DESTROY_INPUT` — `config_value::AVOID` or `config_value::ALLOW`, default `config_value::AVOID`. All parameter names are prefixed `config_param::`.

NOTES:

- The integer overloads may also set `FORWARD_SCALE` / `BACKWARD_SCALE` provided the integer is representable as a `double` without loss of accuracy; otherwise `oneapi::mkl::invalid_argument` is thrown. Setting an integer scaling factor representable as `double` but not as `float` for a single-precision descriptor is unrecommended (truncated, possibly incorrect results).
- Integer types other than `std::int64_t` may be used but must lie in the representable range of `std::int64_t` (validation happens after the cast).
- The vector overload may also configure the integer-valued parameters above as long as the vector contains a single unique accepted integer value.
- Real-valued values not representable as `float` (resp. `double`) without loss are unrecommended for single- (resp. double-) precision descriptors.
- Configuration values may be accepted at `set_value` time yet found invalid later, at commit time; validity requires analysis of all configuration values.
- A `set_value` call that changes the configuration of a **committed** object **uncommits** it: `config_param::COMMIT_STATUS` changes from `config_value::COMMITTED` to `config_value::UNCOMMITTED`. Avoid `set_value` after `commit`.

Exceptions: `std::runtime_error` if an issue is found with the calling object; `oneapi::mkl::invalid_argument` if the parameter being set is not writable, is rejected (e.g. inconsistent with the type of value used), or the value is rejected for that parameter; `oneapi::mkl::uninitialized` if the calling object has been moved-from and not reinitialized.

### oneapi::mkl::dft::descriptor::get_value

Purpose: query the configuration value associated with a configuration parameter. The calling object is left unchanged.

```
namespace oneapi::mkl::dft {
  template <precision prec, domain dom>
  class descriptor {
    using real_scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
  public:
    // for the type of forward domain:
    void get_value(config_param param, domain* value_ptr) const;
    // for the floating-point format:
    void get_value(config_param param, precision* value_ptr) const;
    // for integer-valued parameters:
    void get_value(config_param param, std::int64_t* value_ptr) const;
    // for real-valued parameters:
    void get_value(config_param param, real_scalar_t* value_ptr) const;
    // for vector-valued parameters:
    void get_value(config_param param, std::vector<std::int64_t>* value_ptr) const;
    // for other parameters:
    void get_value(config_param param, config_value* value_ptr) const;
  }
}
```

Behavior:

- `domain*`: `config_param::FORWARD_DOMAIN` → `value_ptr[0]` (assigned `dom`).
- `precision*`: `config_param::PRECISION` → `value_ptr[0]` (assigned `prec`).
- `std::int64_t*`: `config_param::DIMENSION`, `NUMBER_OF_TRANSFORMS`, `PLACEMENT`, `FWD_DISTANCE`, `BWD_DISTANCE`, `THREAD_LIMIT`, `WORKSPACE_ESTIMATE_BYTES`, `WORKSPACE_BYTES`, `WORKSPACE_EXTERNAL_BYTES` → `value_ptr[0]`. `FORWARD_SCALE` and `BACKWARD_SCALE` → `value_ptr[0]` (see exceptions). `LENGTHS` when the transform is one-dimensional → `value_ptr[0]` (assigned `n_1`); the source's condition text was inline math lost in extraction.
- `real_scalar_t*`: `config_param::FORWARD_SCALE` and `config_param::BACKWARD_SCALE` → `value_ptr[0]`.
- `std::vector<std::int64_t>*` (with `std::int64_t* data = value_ptr->data();`): `config_param::LENGTHS` → `data[0], data[1], ..., data[d-1]` (required vector size `d`); `config_param::FWD_STRIDES` and `BWD_STRIDES` → `data[0], data[1], ..., data[d]`; all integer-valued configuration parameters → `data[0]`.
- `config_value*`: `config_param::COMPLEX_STORAGE`, `PLACEMENT`, `WORKSPACE`, `WORKSPACE_PLACEMENT`, `COMMIT_STATUS`, `DESTROY_INPUT` → `value_ptr[0]`.

Exceptions:

- `std::runtime_error` if an issue is found with the calling object.
- `oneapi::mkl::uninitialized` if the object is uncommitted yet queried about a parameter that requires commitment (e.g. `WORKSPACE_BYTES`, `WORKSPACE_EXTERNAL_BYTES`), or if the object was moved-from and not reinitialized.
- `oneapi::mkl::unimplemented` if the queried parameter corresponds to a feature not implemented for the calling object (e.g. `WORKSPACE_BYTES`/`WORKSPACE_EXTERNAL_BYTES` for a CPU-committed descriptor).
- `oneapi::mkl::invalid_argument` if the second argument is `nullptr`; the queried parameter is rejected (e.g. inconsistent with the pointer type); the size of the vector pointed to by `value_ptr` is not as required; or the configuration value cannot be safely/accurately converted to the requested type (e.g. querying a scaling factor that happens to be [value lost] with the integer-valued overload).

### oneapi::mkl::dft::descriptor::commit

Purpose: mark the configuration complete and bind the descriptor to a queue/device.

```
namespace oneapi::mkl::dft {
  template <precision prec, domain dom>
  class descriptor {
  public:
    void commit(sycl::queue &user_queue);
  }
}
```

Input parameter: `user_queue`, `sycl::queue` — queue to which DFT computations are enqueued by the calling object when later used in compute functions. The queue defines the targeted computation device.

- Triggers initialization work (pre-computing data, exploring factorizations, assessing algorithm suitability) for enqueueing the DFT.
- On success `COMMIT_STATUS` becomes `config_value::COMMITTED`; descriptors **must** be committed to be used in any compute function.
- Changing any configuration setting afterwards leaves the object uncommitted (`config_value::UNCOMMITTED`); avoid `set_value` after `commit`.
- Exceptions: `oneapi::mkl::unimplemented` (configuration not supported yet); `oneapi::mkl::uninitialized` (moved-from and not reinitialized); `std::runtime_error` (configuration found inconsistent).

### oneapi::mkl::dft::descriptor::set_workspace

Purpose: supply a user-allocated (externally-allocated) workspace for a committed descriptor.

```
namespace oneapi::mkl::dft {
  template <precision prec, domain dom>
  class descriptor {
    using real_scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
  public:
    template<typename data_type>
    void set_workspace(sycl::buffer<data_type, 1> &workspace);
    template<typename data_type>
    void set_workspace(data_type* workspace);
  }
}
```

- Workspace = additional memory the object may require (intermediate results, pre-computed data). Default (`config_value::WORKSPACE_AUTOMATIC` / `WORKSPACE_INTERNAL`) = internally allocated at commit and owned by the object; `config_value::WORKSPACE_EXTERNAL` = the user must provide a device-accessible allocation after commit with `set_workspace`, in the same form (SYCL buffer vs USM) as the input/output data containers used later in compute functions.
- Minimal required size in bytes = value of `config_param::WORKSPACE_BYTES` (alias `WORKSPACE_EXTERNAL_BYTES`) after commit; for an uncommitted object a conservative estimate is `config_param::WORKSPACE_ESTIMATE_BYTES`. Both are read-only, queried with the integer-valued overload of `get_value`.
- Calling `set_workspace` on an object with an internal workspace frees the internal one, uses the user one, and sets `WORKSPACE`/`WORKSPACE_PLACEMENT` to `config_value::WORKSPACE_EXTERNAL`; the object remains committed.
- An externally-allocated workspace must not be used for any other purpose between providing it and completion of any compute function using the object (especially for USM, since internal workspace dependencies may otherwise be ignored).
- Partial support: externally-allocated workspaces are supported on **GPU devices only**. `WORKSPACE_ESTIMATE_BYTES` implicitly assumes the object will be committed to a GPU-targeting queue (even if already committed to a CPU device).
- Template parameter `data_type`: the only specializations available in oneMKL are `float` and `double`, regardless of the calling object's type. Input parameter `workspace`: (i) `sycl::buffer<data_type, 1>` or (ii) `data_type*` device-accessible USM allocation.
- Exceptions: `oneapi::mkl::invalid_argument` if the workspace is a `sycl::buffer` that is a sub-buffer; `std::runtime_error` if the object is not committed, if a buffer workspace is smaller than required, if `nullptr` is passed while required size is nonzero, if a USM allocation is not device-accessible, or if an error occurred during the workspace-setting operations; `oneapi::mkl::uninitialized` if the object was moved-from and not reinitialized.
- NOTE: for a USM allocation the size is **not** verified by oneMKL (assumed large enough).

### oneapi::mkl::dft::compute_forward / oneapi::mkl::dft::compute_backward

Purpose: enqueue the forward (resp. backward) DFT defined by a committed descriptor.

```
namespace oneapi::mkl::dft {
  /*----------------------- for in-place transforms -----------------------*/
  // using SYCL buffers
  template<typename descriptor_type, typename data_type>
  void compute_forward(descriptor_type &desc,
                       sycl::buffer<data_type, 1> &inout);
  template<typename descriptor_type, typename data_type>
  void compute_backward(descriptor_type &desc,
                        sycl::buffer<data_type, 1> &inout);
  // using device-accessible USM allocations
  template <typename descriptor_type, typename data_type>
  sycl::event compute_forward(descriptor_type &desc,
                              data_type *inout,
                              const std::vector<sycl::event> &dependencies = {});
  template <typename descriptor_type, typename data_type>
  sycl::event compute_backward(descriptor_type &desc,
                               data_type *inout,
                               const std::vector<sycl::event> &dependencies = {});
  /*--------------------- for out-of-place transforms ---------------------*/
  // using SYCL buffers
  template<typename descriptor_type, typename input_type, typename output_type>
  void compute_forward(descriptor_type &desc,
                       sycl::buffer<input_type, 1> &in,
                       sycl::buffer<output_type, 1> &out);
  template<typename descriptor_type, typename input_type, typename output_type>
  void compute_backward(descriptor_type &desc,
                        sycl::buffer<input_type, 1> &in,
                        sycl::buffer<output_type, 1> &out);
  // using device-accessible USM allocations
  template<typename descriptor_type, typename input_type, typename output_type>
  sycl::event compute_forward(descriptor_type &desc,
                              input_type *in,
                              output_type *out,
                              const std::vector<sycl::event> &dependencies = {});
  template<typename descriptor_type, typename input_type, typename output_type>
  sycl::event compute_backward(descriptor_type &desc,
                               input_type *in,
                               output_type *out,
                               const std::vector<sycl::event> &dependencies = {});
}
```

Description: accepts a committed `desc` plus one data argument (in-place) or two data arguments (out-of-place), and enqueues the computation of the transform defined by `desc` (forward or backward) to the queue `desc` was committed to. Buffer overloads return `void`; USM overloads return a `sycl::event`.

Template parameters:

- In-place: `descriptor_type` (type of `desc`), `data_type` (elementary data type of `inout`).
- Out-of-place: `descriptor_type`, `input_type`, `output_type`.

Available specializations for in-place (`compute_forward` and `compute_backward` alike), as `(descriptor_type, data_type)` pairs: `(descriptor<precision::SINGLE, domain::REAL>, float)`, `(descriptor<precision::SINGLE, domain::REAL>, std::complex<float>)`, `(descriptor<precision::SINGLE, domain::COMPLEX>, float)`, `(descriptor<precision::SINGLE, domain::COMPLEX>, std::complex<float>)`, `(descriptor<precision::DOUBLE, domain::REAL>, double)`, `(descriptor<precision::DOUBLE, domain::REAL>, std::complex<double>)`, `(descriptor<precision::DOUBLE, domain::COMPLEX>, double)`, `(descriptor<precision::DOUBLE, domain::COMPLEX>, std::complex<double>)`.

Available specializations for out-of-place, as `(function template, descriptor_type, input_type, output_type)`: both functions with `(descriptor<precision::SINGLE, domain::REAL>, float, float)`; both with `(descriptor<precision::SINGLE, domain::REAL>, std::complex<float>, std::complex<float>)`; both with `(descriptor<precision::SINGLE, domain::COMPLEX>, float, float)`; both with `(descriptor<precision::SINGLE, domain::COMPLEX>, std::complex<float>, std::complex<float>)`; `compute_forward` with `(descriptor<precision::SINGLE, domain::REAL>, float, std::complex<float>)`; `compute_backward` with `(descriptor<precision::SINGLE, domain::REAL>, std::complex<float>, float)`; both with `(descriptor<precision::DOUBLE, domain::REAL>, double, double)`; both with `(descriptor<precision::DOUBLE, domain::REAL>, std::complex<double>, std::complex<double>)`; both with `(descriptor<precision::DOUBLE, domain::COMPLEX>, double, double)`; both with `(descriptor<precision::DOUBLE, domain::COMPLEX>, std::complex<double>, std::complex<double>)`; `compute_forward` with `(descriptor<precision::DOUBLE, domain::REAL>, double, std::complex<double>)`; `compute_backward` with `(descriptor<precision::DOUBLE, domain::REAL>, std::complex<double>, double)`.

Input parameters and returned value: `desc` (`descriptor_type`) — committed instance of a specialization of the descriptor class template, defining the DFT to compute; `inout` (in-place; `sycl::buffer<data_type, 1>` or `data_type*`) — data-containing argument for the DFT (input data on input, output results on output); `in` (out-of-place; `sycl::buffer<input_type, 1>` or `input_type*`) — input data; `out` (out-of-place; `sycl::buffer<output_type, 1>` or `output_type*`) — output data; `dependencies` (`std::vector<sycl::event>`) — vector of dependencies to be honored by the enqueued DFT computation (only with USM allocation arguments).

NOTES / warnings:

- Successive computations using the same descriptor object **must be serialized** to guarantee correct results: wait on the commit queue, use an in-order queue, reuse the same buffer for any argument (buffer case), or serialize with the returned event plus input dependencies (USM case). When possible, configuring a batched transform delivers better performance than successive computations.
- In-place functions require `config_value::INPLACE` for `config_param::PLACEMENT`; out-of-place functions require `config_value::NOT_INPLACE`. Otherwise behavior is **undefined**.
- For out-of-place operations `in` and `out` must not share common elements.
- Exceptions (`std::runtime_error`): an issue is found with `desc`; `desc` is uncommitted; `desc` uses an externally-allocated workspace yet none was provided, or the form of data arguments differs from the form of the provided workspace; a failure is detected when computing the DFT. With SYCL buffers, some code paths do not support sub-buffers and may throw `std::invalid_argument`. `oneapi::mkl::uninitialized` if `desc` has been moved-from and not reinitialized.

### oneapi::mkl::experimental::dft::distributed_config_param

Purpose: scoped enumeration of distributed-DFT-specific configuration parameters (p1168–1169).

```
namespace oneapi::mkl::experimental::dft {
  enum class distirbuted_config_param {
     fwd_divided_dimension,
     bwd_divided_dimension,
     fwd_distribution,
     bwd_distribution,
     fwd_local_data_size_bytes,
     bwd_local_data_size_bytes
  };
}
```

The listing above is copied verbatim, including the spelling `distirbuted_config_param` (the prose in the same page calls it `distributed_config_param`; see `## Explicit gaps`). Meanings: `fwd_divided_dimension` / `bwd_divided_dimension` — which dimension to slab decompose for the forward/backward domain; `fwd_distribution` / `bwd_distribution` — a custom decomposition is to be used for the forward/backward domain; `fwd_local_data_size_bytes` / `bwd_local_data_size_bytes` — exact size in bytes to allocate for the forward/backward domain within that specific process.

**Read-only:** `fwd_local_data_size_bytes`, `bwd_local_data_size_bytes`.

Declared in `oneapi/mkl/experimental/distributed_dft.hpp`.

### oneapi::mkl::experimental::dft::distributed_config_value

Purpose: scoped enumeration for distributed-DFT configuration values. The prose (p1166) declares that the interface "contains the scoped enumerations `oneapi::mkl::experimental::dft::distributed_config_param` and `oneapi::mkl::experimental::dft::distributed_config_value`", and that the distributed interface re-uses the scoped enumerations defined in `oneapi::mkl::dft`. No member listing for `distributed_config_value` appears anywhere in these pages (see `## Explicit gaps`).

### oneapi::mkl::experimental::dft::distributed_descriptor

Purpose: class template for a global DFT distributed across MPI processes, one SYCL GPU device per process. Declared in `oneapi/mkl/experimental/distributed_dft.hpp` as `template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom> class distributed_descriptor;`.

Template parameters (in order): a value of type `oneapi::mkl::dft::precision`; a value of type `oneapi::mkl::dft::domain`.

Constructors, destructor (p1170–1171):

```
namespace oneapi::mkl::experimental::dft {
  template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
  class distributed_descriptor {
  public:
    // parameterized constructors:
    distributed_descriptor(MPI_Comm Comm, std::vector<std::int64_t> dimensions);
    // destructor
    ~distributed_descriptor();
    // unsupported copy constructor and assignment operator:
    distributed_descriptor(const distributed_descriptor&) = delete;
    distributed_descriptor& operator=(const distributed_descriptor&) = delete;
    // unsupported move constructor and assignment operator:
    distributed_descriptor(distributed_descriptor&&) = delete;
    distributed_descriptor& operator=(distributed_descriptor&&) = delete;
  }
}
```

Input parameters: `Comm`, `MPI_Comm` — the MPI communicator containing the processes among which the DFT problem will be distributed; `lengths`, `std::vector<std::int64_t>` — vector of size `d` containing `n_1, ..., n_d` in that order (lengths of the global DFT). Defaults on construction mirror the single-process descriptor: in-place, unbatched (`M = 1`), unscaled global DFT of the forward domain, precision and length(s) set at construction.

Restrictions: copy construction/assignment are not supported; move construction/assignment are **not** supported (unlike the single-process descriptor). The destructor frees all resources allocated for and by objects of that class within the respective MPI process.

Exceptions (`oneapi::mkl::exception`): [two lost inline-math conditions — see `## Explicit gaps`]; the dimensions have length less than the number of processes; any MPI-related error occurs; the construction of the `distributed_descriptor` object fails to allocate its required resources.

### oneapi::mkl::experimental::dft::distributed_descriptor::set_value

Purpose: set global DFT configuration (re-using `oneapi::mkl::dft::config_param` values) or distributed-specific configuration.

```
namespace oneapi::mkl::experimental::dft {
  template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
  class distributed_descriptor {
    using real_scalar_t = std::conditional_t<prec == oneapi::mkl::dft::precision::DOUBLE, double, float>;
  public:
    void set_value(oneapi::mkl::dft::config_param, oneapi::mkl::dft::config_value);
    void set_value(oneapi::mkl::dft::config_param, std::int64_t);
    void set_value(distributed_config_param, std::int64_t);
    void set_value(oneapi::mkl::dft::config_param, const std::vector<std::int64_t>&);
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    void set_value(distributed_config_param param, T value) {
        set_value(param, static_cast<std::int64_t>(value));
    }
    void set_value(oneapi::mkl::dft::config_param, real_scalar_t);
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    void set_value(oneapi::mkl::dft::config_param param, T value) {
        set_value(param, static_cast<std::int64_t>(value));
    }
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void set_value(oneapi::mkl::dft::config_param param, T value) {
        set_value(param, static_cast<real_scalar_t>(value));
    }
    void set_value(distributed_config_param param,
                   const std::vector<std::int64_t> &lower_bound,
                   const std::vector<std::int64_t> &upper_bound,
                   const std::vector<std::int64_t> &strides);
  }
}
```

- Global transform configuration (forward/backward scale, etc.) uses the `oneapi::mkl::dft::config_param` overloads and must be set **uniformly across all processes**.
- Deprecated functionality is not supported by the distributed DFT; setting any deprecated parameter throws `oneapi::mkl::invalid_argument`.
- For `oneapi::mkl::dft::FWD_STRIDES` / `BWD_STRIDES` the global layout must conform to the default layout; for real transforms, padded forward-domain data is supported for both in-place and out-of-place, and out-of-place additionally supports packed forward-domain strides.
- A configuration change on a committed `distributed_descriptor` uncommits it (`COMMIT_STATUS`: `COMMITTED` → `UNCOMMITTED`).

Slab distribution — `set_value(distributed_config_param param, std::int64_t value)`:

| Parameter | Accepted values | Default |
|---|---|---|
| `distributed_config_param::fwd_divided_dimension` | `0` to `rank-1` | `0` |
| `distributed_config_param::bwd_divided_dimension` | `0` to `rank-1` | `1` |

Custom distribution — `set_value(distributed_config_param param, const std::vector<std::int64_t> &lower_bound, const std::vector<std::int64_t> &upper_bound, const std::vector<std::int64_t> &strides)`; `param` must be `distributed_config_param::fwd_distribution` or `distributed_config_param::bwd_distribution` and selects the domain for the custom decomposition. `lower_bound`, `upper_bound` and `strides` are each a `std::vector<std::int64_t>` object of size `d` (the rank): lower-corner and upper-corner of the portion of the global array owned by the current process, and the local data layout in memory for the forward or backward domain respectively; **strides must be in decreasing order and positive**.

NOTE: custom distribution requires bounds to be set for **both** forward and backward domains, otherwise `oneapi::mkl::exception` is thrown during commit. Setting custom bounds in either domain overrides any previously set slab distribution; setting slab distribution after custom bounds does **not** override the custom bounds.

Exceptions: `std::runtime_error` if an issue is found with the calling object; `oneapi::mkl::invalid_argument` if the parameter is not writable, is rejected (e.g. inconsistent value type), or the value is rejected for that parameter; `oneapi::mkl::unimplemented` if the parameter being set is not yet implemented.

### oneapi::mkl::experimental::dft::distributed_descriptor::get_value

Purpose: query global or distributed configuration of a `distributed_descriptor`.

```
namespace oneapi::mkl::experimental::dft {
  template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
  class distributed_descriptor {
    using real_scalar_t = std::conditional_t<prec == oneapi::mkl::dft::precision::DOUBLE, double, float>;
  public:
    // for the type of forward domain:
    void get_value(oneapi::mkl::dft::config_param, oneapi::mkl::dft::domain*) const;
    // for the floating-point format:
    void get_value(oneapi::mkl::dft::config_param, oneapi::mkl::dft::precision*) const;
    // for integer-valued parameters:
    void get_value(oneapi::mkl::dft::config_param, std::int64_t*) const;
    void get_value(distributed_config_param, std::int64_t*) const;
    // for vector-valued parameters:
    void get_value(oneapi::mkl::dft::config_param, std::vector<std::int64_t>*) const;
    // for real-valued parameters:
    void get_value(oneapi::mkl::dft::config_param, real_scalar_t*) const;
    // for custom distribution:
    void get_value(distributed_config_param param,
                   std::vector<std::int64_t> *lower_bound,
                   std::vector<std::int64_t> *upper_bound,
                   std::vector<std::int64_t> *strides) const;
    // for other parameters:
    void get_value(oneapi::mkl::dft::config_param, oneapi::mkl::dft::config_value*) const;
  }
}
```

- Integer-valued overload (`distributed_config_param`, `std::int64_t*`): `value_ptr[0]` for `fwd_divided_dimension`, `bwd_divided_dimension`, `fwd_local_data_size_bytes`, `bwd_local_data_size_bytes`.
- `fwd_local_data_size_bytes` / `bwd_local_data_size_bytes` give the process-specific number of bytes for which device-accessible memory must be allocated and initialized with input data. Query **after** commit (only then is the exact size known); the allocation may exceed the size deduced from the data distribution alone.
- Custom-distribution overload (`lower_bound`, `upper_bound`, `strides`) queries the per-process bounds; each `std::vector<std::int64_t>*` must have length equal to the rank of the transform. Querying custom distribution without having set it throws `oneapi::mkl::invalid_argument`. Deprecated parameters are unsupported; querying them throws. A querying function leaves the object unchanged.
- Exceptions: `std::runtime_error` (issue with the calling object); `oneapi::mkl::uninitialized` (uncommitted yet queried about a parameter requiring commit); `oneapi::mkl::unimplemented` (feature not implemented); `oneapi::mkl::invalid_argument` (pointer arguments are `nullptr`; parameter rejected, e.g. inconsistent pointer type; vector size not as required; value not safely/accurately convertible to the desired type).

### oneapi::mkl::experimental::dft::distributed_descriptor::commit

```
namespace oneapi::mkl::experimental::dft {
  template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
  class distributed_descriptor {
  public:
    void commit(sycl::queue &user_queue);
  }
}
```

Input parameter: `user_queue`, `sycl::queue` — queue to which local DFT computations are enqueued thereafter; it is mapped to a physical device by MPI.

- On success `COMMIT_STATUS` is `config_value::COMMITTED`. All processes must successfully commit their `distributed_descriptor` object to obtain correct results.
- Changing any configuration setting afterwards uncommits the object; avoid `set_value` after `commit`.
- Exceptions: `oneapi::mkl::unimplemented` if the configuration is not supported yet (e.g. non-default slab distribution is used); `oneapi::mkl::exception` if the device is not an Intel® Data Center GPU Max Series; the environment variable `I_MPI_OFFLOAD` is not set to `1`; the SYCL backend is not Level Zero; default packed layouts are not used for the global array; batching is used (setting `oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS > 1`); or allocation/initialization of resources required for the `distributed_descriptor` object fails.

### oneapi::mkl::experimental::dft::compute_forward / oneapi::mkl::experimental::dft::compute_backward

Purpose: enqueue the distributed forward (resp. backward) global DFT defined by a committed `distributed_descriptor`.

```
namespace oneapi::mkl::experimental::dft {
  /*----------------------- for in-place transforms -----------------------*/
  template <typename descriptor_type, typename data_type>
  sycl::event compute_forward(descriptor_type &desc,
                              data_type *inout,
                              const std::vector<sycl::event> &dependencies = {});
  template <typename descriptor_type, typename data_type>
  sycl::event compute_backward(descriptor_type &desc,
                               data_type *inout,
                               const std::vector<sycl::event> &dependencies = {});
  /*--------------------- for out-of-place transforms ---------------------*/
  template<typename descriptor_type, typename input_type, typename output_type>
  sycl::event compute_forward(descriptor_type &desc,
                              input_type *in,
                              output_type *out,
                              const std::vector<sycl::event> &dependencies = {});
  template<typename descriptor_type, typename input_type, typename output_type>
  sycl::event compute_backward(descriptor_type &desc,
                               input_type *in,
                               output_type *out,
                               const std::vector<sycl::event> &dependencies = {});
}
```

All four overloads return `sycl::event`. Each process takes its local chunk of input, computes the DFT on the data it holds, exchanges intermediate results with MPI, and produces the final local output chunk. Successive global computations require serialization across processes (wait on the commit queue, use an in-order queue, or use the returned `sycl::event` and input dependencies).

Available in-place specializations (`compute_forward` and `compute_backward`): `distributed_descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>` with `float` or `std::complex<float>`; `...<SINGLE, COMPLEX>` with `float` or `std::complex<float>`; `...<DOUBLE, REAL>` with `double` or `std::complex<double>`; `...<DOUBLE, COMPLEX>` with `double` or `std::complex<double>`.

Available out-of-place specializations, mirroring the single-process table: `(SINGLE, REAL)` with `float→float` and `std::complex<float>→std::complex<float>`; `(SINGLE, COMPLEX)` with `float→float` and `std::complex<float>→std::complex<float>`; `compute_forward` `(SINGLE, REAL)` `float→std::complex<float>`; `compute_backward` `(SINGLE, REAL)` `std::complex<float>→float`; `(DOUBLE, REAL)` with `double→double` and `std::complex<double>→std::complex<double>`; `(DOUBLE, COMPLEX)` with `double→double` and `std::complex<double>→std::complex<double>`; `compute_forward` `(DOUBLE, REAL)` `double→std::complex<double>`; `compute_backward` `(DOUBLE, REAL)` `std::complex<double>→double`.

Input parameters:

| Name | Supported types | Description |
|---|---|---|
| `desc` | `descriptor_type` | committed instance of a specialization of the `distributed_descriptor` class template, defining the global DFT to compute |
| `inout` (in-place) | `data_type*` | data-containing argument (local input data on input, local output results on output) |
| `in` (out-of-place) | `input_type*` | data-containing argument for the local input data |
| `out` (out-of-place) | `output_type*` | data-containing argument for the local output data |
| `dependencies` | `std::vector<sycl::event>` | vector of dependencies to be honored by the enqueued DFT computation |

Warnings / NOTES:

- In-place functions require `oneapi::mkl::dft::config_value::INPLACE` for `oneapi::mkl::dft::config_param::PLACEMENT`; out-of-place functions require `NOT_INPLACE`; all processes must be configured uniformly. Behavior is undefined otherwise.
- For out-of-place operations `in` and `out` must not share common elements.
- Sizes of `in` and `out` must be greater than or equal to the queried sizes after commit.
- The distributed DFT interface does **not** support SYCL buffers.
- Exceptions (`oneapi::mkl::exception`): an issue is found with `desc`; `desc` is uncommitted; a failure is detected when computing the DFT; an MPI issue is detected.

### Boundary routines from the Statistics domain (source chunk p1121–p1131)

These five `oneapi::mkl::stats` routines appear at the head of the source chunks but belong to the Statistics domain (a different digest chapter); recorded compactly so no name is lost. All: `#include <oneapi/mkl/stats.hpp>`, `template<method Method = method::fast, typename Type, layout ObservationsLayout>`, a `const dataset<ObservationsLayout, ...>& data` argument, buffer form returning `void`, USM form returning `sycl::event`, and USM-only `const std::vector<sycl::event> &dependencies = {}`. Full names: `oneapi::mkl::stats::skewness`, `oneapi::mkl::stats::kurtosis`, `oneapi::mkl::stats::min`, `oneapi::mkl::stats::max`, `oneapi::mkl::stats::min_max`.

```
// skewness with user-provided mean -- Method: method::fast only.
// (Parameter table names the USM output `variation`; signature names it
//  `skewness` -- reference inconsistency, reproduced verbatim.)
void skewness(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> skewness);
sycl::event skewness(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* skewness,
    const std::vector<sycl::event> &dependencies = {});

// kurtosis -- Method: method::fast or method::one_pass
void kurtosis(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> kurtosis);
sycl::event kurtosis(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* kurtosis,
    const std::vector<sycl::event> &dependencies = {});

// kurtosis with user-provided mean -- Method: method::fast only
void kurtosis(sycl::queue& queue,
    sycl::buffer<Type, 1> mean,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> kurtosis);
sycl::event kurtosis(sycl::queue& queue,
    Type* mean,
    const dataset<ObservationsLayout, Type*>& data,
    Type* kurtosis,
    const std::vector<sycl::event> &dependencies = {});

// min -- Method: method::fast only
void min(
    sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> min);
sycl::event min(
    sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* min,
    const std::vector<sycl::event> &dependencies = {});

// max -- Method: method::fast only
void max(
    sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> max);
sycl::event max(
    sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* max,
    const std::vector<sycl::event> &dependencies = {});

// min_max -- Method: method::fast only
void min_max(sycl::queue& queue,
    const dataset<ObservationsLayout, sycl::buffer<Type, 1>>& data,
    sycl::buffer<Type, 1> min, sycl::buffer<Type, 1> max);
sycl::event min_max(sycl::queue& queue,
    const dataset<ObservationsLayout, Type*>& data,
    Type* min, Type* max,
    const std::vector<sycl::event> &dependencies = {});
```

## Formulas

- **DFT (p1131)** — transform definition for a `d`-dimensional sequence `z^m` of lengths `n_1, ..., n_d`, `m` in `{0, 1, ..., M-1}`; transcribed verbatim from the image: `ẑ^m_{k_1,k_2,…,k_d} = σ_δ Σ_{j_d=0}^{n_d-1} … Σ_{j_2=0}^{n_2-1} Σ_{j_1=0}^{n_1-1} z^m_{j_1,j_2,…,j_d} exp[ δ 2πi ( Σ_{ℓ=1}^{d} j_ℓ k_ℓ / n_ℓ ) ], ∀m ∈ {0,1,…,M−1}`. ASCII: `zhat^m_{k_1,...,k_d} = sigma_delta * SUM_{j_d=0}^{n_d-1} ... SUM_{j_2=0}^{n_2-1} SUM_{j_1=0}^{n_1-1} z^m_{j_1,...,j_d} * exp[ delta * 2*pi*i * ( SUM_{l=1}^{d} j_l*k_l / n_l ) ]` for all `m` in `{0,1,...,M-1}`. `delta` selects direction; `sigma_delta` is the associated scaling factor (forward scale for the forward DFT, backward scale for the backward DFT).
- **skewness with user-provided mean (p1121)** — vector of skewness values: `Gamma(X) = ( Gamma_1(X), ..., Gamma_p(X) ),  Gamma_i(X) = C_i^(3)(X) / V_i^1.5(X)`.
- **kurtosis (p1123)** — vector of kurtosis values: `B(X) = ( B_1(X), ..., B_p(X) ),  B_i(X) = C_i^(4)(X) / V_i^2(X) - 3`.
- **kurtosis with user-provided mean (p1124)** — vector of kurtosis values; the image is identical to p1123: `B(X) = ( B_1(X), ..., B_p(X) ),  B_i(X) = C_i^(4)(X) / V_i^2(X) - 3`.
- **Spline interpolant polynomials (p1184, Data Fitting domain)** — two formulas on one image, transcribed verbatim: `P(x) = c_1 + c_2 (x - x_i) + c_3 (x - x_i)^2 + ... + c_{k-1} (x - x_i)^k .` and `P_i(x) = c_{1,i} + c_{2,i} (x - x_i) + c_{3,i} (x - x_i)^2 + ... + c_{k-1,i} (x - x_i)^k .` These belong to the Data Fitting domain (p1183–p1184, the chapter boundary); recorded here only because the source chunk lists them.

No formula images were provided for pages 1133–1183 (all recipe/configuration pages of the DFT domain have text-only content).

## Conventions & Gotchas

- **Indexing / data layout.** A non-redundant entry (or its real/imaginary part) is stored at an index of the user data container defined by `offset + sum_l (k_l - k_l^min) * stride_l` (+ distance * sequence number for batched data). The exact closed-form index expression, and the default stride values, were inline math that did not survive text extraction (see `## Explicit gaps`); the prose does state: offsets and generalized strides `(stride_0, stride_1, ..., stride_d)` define locations within each `d`-dimensional data sequence, and `distance` separates successive sequences of a batch. All are counted in number of elements of the relevant implicitly-assumed elementary data type, and all elements accessed at a given index must belong to the same block allocation.
- **Default layout (from the supplied usage examples).** Freshly created descriptors are configured for unbatched, in-place transforms with a unit stride along the last dimension and no offset. For a real descriptor interpreting backward-domain data as complex (the default), the backward last-dimension extent is `n_d/2 + 1`, and forward data is minimally padded to match in-place layout requirements. Recommended explicit strides for out-of-place real transforms (from the example): `fwd_strides[rank] = bwd_strides[rank] = 1`; `fwd_strides[rank-1] = lengths[rank-1]`, `bwd_strides[rank-1] = lengths[rank-1]/2 + 1`; for `dim = 2; dim < rank; dim++`: `fwd_strides[rank-dim] = fwd_strides[rank-dim+1] * lengths[rank-dim]` and likewise for `bwd_strides`; offsets `fwd_strides[0] = bwd_strides[0] = 0`. For batched transforms both `FWD_DISTANCE` and `BWD_DISTANCE` must be set explicitly, since their default `0` would break layout requirements for `M > 1`; a typical distance is the product of the lengths with the last-dimension factor adjusted for real/in-place packing.
- **Implicitly-assumed elementary data type.** Complex descriptor with `config_value::COMPLEX_COMPLEX` (default): `std::complex<fp_type>` in both domains. Complex descriptor with `config_value::REAL_REAL`: `fp_type` in both domains, but **not implemented** via the DPC++ interface. Real descriptor: `fp_type` in the forward domain, `std::complex<fp_type>` in the backward domain. Here `fp_type` is `float` for single-precision and `double` for double-precision descriptors.
- **Layout requirements.** Distances and strides must give non-negative index values for all in-range tuples, and every index must correspond to a unique tuple. Negative strides with a large positive offset are **not enabled yet (unimplemented)**.
- **In-place consistency.** For in-place transforms, descriptors expecting the same type in both domains (e.g. complex descriptors) must use the same offset, stride(s) and distance in both domains. For real descriptors expecting complex backward data (the default), the memory addresses of the leading entries along the last dimension must be identical in forward and backward domains, which forces padding in the forward domain when unit strides are used (the default, recommended usage). The explicit conditions are inline math lost in extraction.
- **GPU-specific layout limits** (p1154–1155): the rank of the transform must be no greater than [value lost]; the offset values must be [value lost]; further restrictions apply to batched 2D and 3D real transforms; real descriptors require real forward-domain data and complex (non-redundant entries) backward-domain data. A GPU-committed descriptor may overwrite parts of the output block allocation that are irrelevant to the computed DFT (e.g. padding between successive sequences).
- **In-place vs out-of-place must match `PLACEMENT`.** Calling an in-place compute overload on a descriptor configured `NOT_INPLACE` (or vice versa) is undefined behavior. For out-of-place calls `in` and `out` must not share elements.
- **Serialization.** Successive computations with one descriptor must be serialized: wait on the commit queue, use an in-order queue, reuse the same buffer argument, or chain the returned event and the `dependencies` vector. The `dependencies` parameter exists only on the USM overloads; buffer overloads return `void`.
- **Sub-buffers.** Some buffer code paths do not support sub-buffers and may throw `std::invalid_argument`; `set_workspace` explicitly rejects a `sycl::buffer` that is a sub-buffer with `oneapi::mkl::invalid_argument`.
- **Commit lifecycle.** Configuration is frozen for computation at `commit`. Any `set_value` on a committed object silently uncommits it (`COMMIT_STATUS` → `UNCOMMITTED`) and invalidates compute readiness; re-`commit` before computing.
- **Moved-from descriptors.** After move construction/assignment a descriptor must be reinitialized before reuse; otherwise `oneapi::mkl::uninitialized` may be thrown. (Distributed descriptors cannot be moved at all.)
- **Integer type requirement.** Configuration integer values are validated after conversion to `std::int64_t`; other integral types are accepted but must be representable as `std::int64_t`.
- **Precision support.** `precision::SINGLE` (FP32) and `precision::DOUBLE` (FP64). Some GPU devices do not support double-precision descriptors — verify device FP64 support first.
- **Thread limit.** `THREAD_LIMIT` is irrelevant for GPU-committed descriptors and for oneMKL used in sequential mode (where it is permanently fixed and `set_value` attempts are ignored). After commit a descriptor may report a *lower* value than was set.
- **`DESTROY_INPUT`.** Relevant only for out-of-place descriptors (ignored for in-place). Default `config_value::AVOID`; allowing overwrite (`ALLOW`) can significantly reduce required workspace and improve performance, and is particularly relevant for multidimensional, out-of-place, real, backward DFTs.
- **Workspace.** Default internal/automatic; only GPU-committed descriptors support external workspaces; `data_type` is limited to `float`/`double`; USM workspace size is not validated.
- **Exceptions.** Common types: `std::runtime_error`, `oneapi::mkl::invalid_argument`, `oneapi::mkl::uninitialized`, `oneapi::mkl::unimplemented`; the distributed DFT additionally throws `oneapi::mkl::exception`. Compute functions throw `std::runtime_error` (not `oneapi::mkl::exception`) for the single-process interface.
- **Distributed-DFT environment.** Requires Intel MPI (`I_MPI_OFFLOAD=1`), Level Zero backend, Intel® Data Center GPU Max Series devices, the same MPI communicator in all processes, 2D or 3D transforms with each dimension length `>=` number of processes, default packed global layouts, and no batching. All global-transform configuration must be set uniformly across processes (custom distribution being the exception). Batching: the text describes how batches would be divided (first `b % p` processes perform `[b/p] + 1` transforms, the rest `b/p`), but also states batching is **not implemented** and that a `oneapi::mkl::exception` is thrown at commit time if `batch != 1` — the reference contradicts itself here; trust the limitation statement.
- **Built-in distributed decompositions.** 2D array `[Y][X]` over `p` processes: along `Y`, the first `Y % p` processes own `(Y/p + 1)*X` elements and the rest `(Y/p)*X`; along `X`, the first `X % p` own `(X/p + 1)*Y` and the rest `(X/p)*Y`. 3D array `[Z][Y][X]` over `p` processes: along `Z`, first `Z % p` own `(Z/p + 1)*Y*X`, rest `(Z/p)*Y*X`; along `Y`, first `Y % p` own `(Y/p + 1)*X*Z`, rest `(Y/p)*X*Z`; along `X`, first `X % p` own `(X/p + 1)*Y*Z`, rest `(X/p)*Y*Z`. Slab decomposition does not allow strides leading to a non-packed global layout. For real forward domain, in-place requires padded data; out-of-place may be padded or packed.

## Explicit gaps

- **Inline math lost in extraction (DFT pages).** The PDF renders many symbols as inline images that the text extractor dropped, so the following are absent from the source text and are **not** reconstructed here: the numeric values/convention for `delta` and `sigma_delta` (forward vs backward sign and scale); the conjugate-symmetry relation `zhat_{...} = conj(...)`; the exact elementary non-redundant index set; the closed-form data-layout index expression; the default `FWD_STRIDES`/`BWD_STRIDES` values (only "documented in the page specific to configuring data layouts", which is outside these chunks); the descriptor-constructor `std::runtime_error` condition; the GPU rank limit, offset restriction and 2D/3D real-transform layout conditions; the in-place real-descriptor consistency conditions; the `get_value(..., std::int64_t*)` `LENGTHS` condition; and the "scaling factor that happens to be [value]" example in the `get_value` exception list.
- **`distributed_config_value`** is named in prose as one of the enumerations the distributed interface declares, but no member listing, underlying values, or usage is given anywhere in these pages.
- **Enum spelling discrepancy.** The prose calls the distributed parameter enumeration `oneapi::mkl::experimental::dft::distributed_config_param`, while the code listing declares `enum class distirbuted_config_param` (transposed letters). Both spellings are reproduced verbatim; the correct identifier to use is not stated.
- **`distributed_descriptor` constructor exception conditions** are two lost inline-math expressions (the rendered text shows only "if [math] or [math]").
- **Class-name ambiguity in prose.** p1168 refers to `oneapi::mkl::experimental::dft::distributed_dft` (for `set_value` with bounds/strides) although the described class is `distributed_descriptor`; the reference uses both names.
- **Inconsistent parameter naming.** For `skewness` with user-provided mean, the USM signature names the output parameter `skewness` while the parameter table names it `variation` and describes it as "variation coefficients"; for the buffer form, `mean` is listed as a `sycl::buffer` input yet the signature takes it by value as a non-const buffer.
- **Boundary content not part of this domain.** The Statistics routines `skewness`, `kurtosis`, `min`, `max`, `min_max` (p1121–p1131) and the Data Fitting spline polynomials (p1183–p1184) are recorded here only because the assigned chunks include them; their full documentation belongs to their own digest chapters.
