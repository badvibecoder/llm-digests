# Compute Modes and Numerical Reproducibility

This chapter covers two cross-cutting controls on BLAS results in oneMKL: **alternate compute
modes** (`oneapi::mkl::blas::compute_mode`), which trade numerical properties/accuracy for
performance, and **conditional numerical reproducibility (CNR)**, which trades run-to-run speed
freedom for bitwise reproducibility. Neither topic introduces new computational routines; they
change how existing level-3 BLAS routines and level-3 extensions execute. The assigned source pages
also open the BLAS-like batch-extension routines (`imatcopy_batch`, `omatadd_batch`, pp. 277–290) and
the Sparse BLAS domain (pp. 294–304); those boundary pages are covered at the end of `## Routines`,
`## Formulas`, and `## Conventions & Gotchas` so no routine name on the assigned pages is dropped,
with a note that their own chapters hold the full treatment.

## Overview

**Where this sits.** The reference introduces both topics immediately after the BLAS-like extension
routines and before "Sparse BLAS Functionality" (p294). Neither section defines a callable routine of
its own: compute modes are an enum plus selection mechanisms, and CNR is a policy whose API is not
shown on these pages.

### Alternate compute modes

Verbatim model (p290): "BLAS level-3 routines and extensions support alternate compute modes, which
can provide increased performance in exchange for different numerical properties or reduced
accuracy." A list of one or more allowed modes can be specified:

1. **at compile time, per source file** — define the `MKL_BLAS_COMPUTE_MODE` macro before including
   any oneMKL header file. "This macro must be set to an expression of type
   `oneapi::mkl::blas::compute_mode`."
2. **per call** — an optional `oneapi::mkl::blas::compute_mode` argument appended to the parameter
   list (position rules below).
3. **at runtime** — the `MKL_BLAS_COMPUTE_MODE` environment variable.

Selection and fallback: "oneMKL will automatically select an appropriate implementation from this
list, taking into account routine parameters and hardware characteristics. In case none of the
allowed alternate modes are supported by the given routine on the selected device, or if none of the
allowed alternate modes are expected to improve performance, oneMKL will automatically fall back to a
standard implementation."

Availability in the current oneMKL release (verbatim NOTE, p290): "Whether a particular mode
provides additional performance will depend on a number of factors, including routine type, matrix
sizes, transpose parameters, and hardware configuration. In the current oneMKL release, alternate
implementations are available for `gemm`, `gemmt`, `syrk`, and `syr2k` on selected hardware for real
single precision, and `gemm` only for complex single precision."

Defaults: "By default, oneMKL does not enable any alternate compute modes." The library describes the
environment variable as intended "for quickly evaluating whether alternate compute modes provide
performance benefits and acceptable accuracy", after which settings "can be permanently applied
within the application using the per-call or per-source-file APIs."

Combining modes: enum values are combined with `|` (OR); environment-variable settings are combined
with commas:

```cpp
using oneapi::mkl::blas;
auto mode_settings = compute_mode::float_to_bf16x2 | compute_mode::float_to_tf32; /* allow
either of these two modes */
```

```bash
set MKL_BLAS_COMPUTE_MODE=FLOAT_TO_BF16X2,FLOAT_TO_TF32
```

### Mode settings table (verbatim values)

Enum values live in `oneapi::mkl::blas::compute_mode`. The extracted table breaks several enum
identifiers across line/page boundaries; the spellings below use the environment-variable names plus
the confirmed code examples elsewhere in the same section (see Explicit gaps).

| `compute_mode` enum value | Environment variable setting | Description (verbatim) |
|---|---|---|
| `compute_mode::float_to_bf16` | `FLOAT_TO_BF16` | Convert single-precision inputs to bfloat16 format internally; output is accumulated in single precision. Accuracy will be reduced, with possibly much higher performance. |
| `compute_mode::float_to_bf16x2` | `FLOAT_TO_BF16X2` | Convert each single-precision input value to a sum of two bfloat16 values internally; output is accumulated in single precision. Expected accuracy is between that of standard single precision arithmetic and bfloat16 arithmetic. Note: infinite inputs may produce unexpected NaNs in the output. |
| `compute_mode::float_to_bf16x3` | `FLOAT_TO_BF16X3` | Convert each single-precision input value to a sum of three bfloat16 values internally; output is accumulated in single precision. Expected accuracy is comparable to standard single precision arithmetic in most cases. Note: infinite inputs may produce unexpected NaNs in the output. |
| `compute_mode::float_to_tf32` | `FLOAT_TO_TF32` | Convert each single-precision input value to tf32 format internally; output is accumulated in single precision. Expected accuracy is between `float_to_bf16` and `float_to_bf16x2`. |
| `compute_mode::complex_3m` | `COMPLEX_3M` | Reduce the four real multiplications in a standard complex multiplication to three real multiplications. Expected accuracy is comparable to standard arithmetic in most cases. |
| `compute_mode::any` | `ANY` | Allow any alternate compute mode. |
| `compute_mode::standard` | `STANDARD` | Do not allow any alternate compute modes. |
| `compute_mode::prefer_alternate` | `PREFER_ALTERNATE` | Used in conjunction with one or more alternate modes. Prefer an alternate compute mode over the standard implementation whenever available, even if performance may be reduced. |
| `compute_mode::force_alternate` | `FORCE_ALTERNATE` | Used in conjunction with one or more alternate modes. Never use the standard implementation; if no allowed alternate implementation is available, an exception will be thrown. |

A tenth enumerator, `compute_mode::unset`, appears as the **default value of the `mode` parameter**
in routine syntax blocks (verified for `gemm`, p146/p149 — see `### gemm` below). It is not given a
row or a description in the Mode Settings table.

### Per-call mode settings

Verbatim (p292): "BLAS level-3 routines and extensions support an optional
`oneapi::mkl::blas::compute_mode` argument to specify a desired mode setting, at the end of the
parameter list. For USM APIs, the `compute_mode` argument goes before the list of input
dependencies, if any; either argument may be omitted."

The reference's own examples (p292, `syrk`; reproduced with the source's line wrapping):

```cpp
using oneapi::mkl::blas;
sycl::buffer<float, 1> a_buffer, c_buffer;
float *a_ptr, *c_ptr;
/* ... */
// Buffer API
syrk(my_queue, n, k, uplo, trans, alpha, a_buffer, lda, beta, c_buffer, ldc,
compute_mode::float_to_bf16);
// Buffer API, forcing float_to_bf16 mode
syrk(my_queue, n, k, uplo, trans, alpha, a_buffer, lda, beta, c_buffer, ldc,
compute_mode::float_to_bf16 | compute_mode::force_alternate);
// USM API, without dependencies
syrk(my_queue, n, k, uplo, trans, alpha, a_ptr, lda, beta, c_ptr, ldc,
compute_mode::float_to_bf16);
// USM API, with dependencies
syrk(my_queue, n, k, uplo, trans, alpha, a_ptr, lda, beta, c_ptr, ldc,
compute_mode::float_to_bf16, {event1, event2});
// USM API, dependencies but no special compute_mode settings
syrk(my_queue, n, k, uplo, trans, alpha, a_ptr, lda, beta, c_ptr, ldc, {event1, event2});
```

### Per-source-file mode settings

Verbatim (p293): "You can provide default mode settings for all calls within a source file by
defining the `MKL_BLAS_COMPUTE_MODE` macro before including any oneMKL header files." Also verbatim:
"`compute_mode` parameters passed to a oneMKL routine take precedence over the default setting".

```cpp
#define MKL_BLAS_COMPUTE_MODE oneapi::mkl::blas::compute_mode::complex_3m
#include <oneapi/mkl.hpp>
void my_function() {
    /* ... */
    // 3M mode will be allowed by default:
    gemm(my_queue, m, n, k, trans_a, trans_b, alpha, a, lda, b, ldb, beta, c, ldc);
    // Provided compute_mode overrides the default 3M mode.
    gemm(my_queue, m, n, k, trans_a, trans_b, alpha, a, lda, b, ldb, beta, c, ldc,
compute_mode::standard);
}
```

(Both calls above appear in the reference as separate listings of the same file; they are shown
together here to contrast the default with the per-call override.)

### Runtime mode settings

Verbatim (p293): "The `MKL_BLAS_COMPUTE_MODE` environment variable allows you to set default
application-wide alternate compute mode settings, as a quick method for evaluating alternate compute
modes at runtime." Precedence is stated explicitly: "Any per-call or per-source-file mode settings
take precedence over the `MKL_BLAS_COMPUTE_MODE` environment variable. One result of this is that
compiling an application with
`-DMKL_BLAS_COMPUTE_MODE=oneapi::mkl::blas::compute_mode::standard` effectively disables the
environment variable."

```cpp
// my_application.cpp
#include <oneapi/mkl.hpp>
int main() {
    /* ... */
    // Call to gemm without a compute_mode argument:
    oneapi::mkl::blas::gemm(my_queue, /* ... */);
}
```

```bash
export MKL_BLAS_COMPUTE_MODE=FLOAT_TO_BF16X3
my_application      # gemm may use bf16x3 arithmetic.
```

### Checking which mode is used

Verbatim (p294): "On GPU, when oneMKL's verbose mode is enabled, information on the compute mode(s)
enabled and used for each call is provided in the verbose log." The example given (both
`float_to_bf16` and `float_to_tf32` enabled; `float_to_bf16` selected):

```
MKL_VERBOSE oneapi::mkl::blas::column_major::gemm[float](0x7ffd39046350,...,float_to_bf16|
float_to_tf32) mode:float_to_bf16 host:nan device:nan GPU0
```

"Verbose mode can be enabled by setting the `MKL_VERBOSE` environment variable to 1, or via the
`mkl_verbose` API." Cross-reference: "Using oneMKL Verbose Mode" in the *Intel(R) oneAPI Math Kernel
Library Developer Guide*.

### Numerical reproducibility (CNR)

Verbatim model (p294): "Executing a BLAS routine with the same inputs may result in slightly
different results from run to run. This is due, fundamentally, to the limited precision of
floating-point arithmetic. In particular, the results of a summation will depend on the order of
operations, because floating-point addition is not associative. Because of this nonassociative
behavior, some algorithms used by oneMKL BLAS routines may produce slightly different results
depending on the ordering of internal operations."

Verbatim guarantees and scope: "In some cases, it may be desirable to ensure deterministic
(run-to-run bitwise reproducible) results. oneMKL supports conditional numerical reproducibility
(CNR) modes that provide bitwise reproducibility guarantees under certain circumstances. CNR mode is
available on CPU for all BLAS DPC++ APIs, and on GPU for level-3 routines and level-3 extensions."
"More information on CNR mode may be found in the oneMKL Developer Guide (Windows, Linux)." No CNR
function name, macro, signature, or enum appears in the assigned extracted text.

### Device vs host, precision, memory model

- **CNR:** CPU — all BLAS DPC++ APIs; GPU — level-3 routines and level-3 extensions. No other
  restriction is stated.
- **Verbose reporting (modes):** "When running on GPU, oneMKL's verbose output indicates which mode is
  used for each call, whether the standard mode or one of the alternate modes."
- **Alternate implementations exist for:** real single precision (`gemm`, `gemmt`, `syrk`, `syr2k`)
  and complex single precision (`gemm` only) — implementation availability, not a restriction on use.
- **Memory model:** modes apply to both buffer and USM APIs; the only API-shape difference is
  parameter position (`compute_mode` before `dependencies` for USM; buffer APIs have no
  `dependencies`). No buffer/USM semantic, alignment, or scratchpad change is stated.

### Scope of the assigned pages

Pages 277–290 are the tail of the BLAS-like extensions (`imatcopy_batch`, `omatadd_batch`); p290–294
are Compute Modes and Numerical Reproducibility; p294–304 open Sparse BLAS (overview, routine tables,
matrix-handle contract, supported types, CSR format). Boundary-page material is indexed below.

## Routines

The routines that compute modes actually change behavior for are `gemm`, `gemmt`, `syrk`, and
`syr2k` (p290). The enum used by every per-call parameter is `oneapi::mkl::blas::compute_mode`
(tabulated in Overview). Full parameter documentation for the level-3 routines is in the BLAS Level
3 chapter of this digest; entries here are restricted to compute-mode-relevant facts. The two
batch-extension routines from the assigned pages are documented in full, followed by an index of the
Sparse BLAS names that appear only in tables on these pages.

### gemm

General matrix-matrix product; one of the routines with alternate compute-mode implementations (real
and complex single precision). Syntax reproduced here only to pin down the exact position and default
of the per-call `compute_mode` parameter — full parameter documentation is in the BLAS Level 3
chapter. Buffer version (p146): the same parameter names in the same order as the USM listing below
through `std::int64_t ldc`, but with buffer-API types (`Ts alpha`, `sycl::buffer<Ta,1> &a`,
`Ts beta`, `sycl::buffer<Tc,1> &c`), then the final parameter
`compute_mode mode = compute_mode::unset)` (the reference prints it as the last line of the
signature, before the closing brace).

USM version (p149, verbatim; `column_major` shown, `row_major` differs only in the namespace):

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event gemm(sycl::queue &queue,
                     oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb,
                     std::int64_t m,
                     std::int64_t n,
                     std::int64_t k,
                     oneapi::mkl::value_or_pointer<Ts> alpha,
                     const Ta *a,
                     std::int64_t lda,
                     const Tb *b,
                     std::int64_t ldb,
                     oneapi::mkl::value_or_pointer<Ts> beta,
                     Tc *c,
                     std::int64_t ldc,
                     compute_mode mode = compute_mode::unset,
                     const std::vector<sycl::event> &dependencies = {})
}
```

Parameter: `mode` — "Optional. Compute mode settings. See Compute Modes for more details." Output:
`c` is "overwritten by `alpha * op(A)*op(B) + beta * C`"; NOTE: "If `beta = 0`, matrix `C` does not
need to be initialized before calling `gemm`."

### gemmt

Named on p290 as having alternate compute-mode implementations on selected hardware for real single
precision. Only the name appears (inside the compute-mode NOTE); no syntax or parameter list is on
these pages. Its per-call `compute_mode` argument, when present, is appended at the end of the
parameter list (p292). See the BLAS Level 3 chapter for its signature.

### syrk

Symmetric rank-k update; named on p290 for real single-precision alternate implementations. Its only
material here is the set of example calls in the Overview showing `compute_mode` placement (buffer
and USM, with and without `dependencies`); the mode argument is optional and, for USM, precedes
`dependencies`.

### syr2k

Symmetric rank-2k update; named on p290 for real single-precision alternate implementations. No
syntax on these pages; same per-call `compute_mode` placement rule as `syrk`.

### imatcopy_batch

Computes a group of in-place scaled matrix transpose or copy operations using general matrices.
Precisions: `T` = `float`, `double`, `std::complex<float>`, `std::complex<double>`. It is described
as "similar to the `imatcopy` routines, but the `imatcopy_batch` routines perform their operations
with groups of matrices. The groups contain matrices with the same parameters."

Operation (strided API, p278 and p282):

```
for i = 0 … batch_size – 1
    AB is a matrix at offset i * stride in ab
    AB = alpha * op(AB)
end for
where:
  op(X) is one of op(X) = X, op(X) = X', or op(X) = conjg(X')
  alpha is a scalar
  AB is a matrix to be transformed in place
```

Operation (USM group API, p280):

```
idx = 0
for i = 0 … group_count – 1
    m,n, alpha, lda, ldb and group_size at position i in their respective arrays
    for j = 0 … group_size – 1
        AB is a matrix at position idx in AB
        AB = alpha * op(AB)
        idx := idx + 1
    end for
end for
```

"For the strided API, the single buffer `AB` contains all the matrices to be transformed in place.
The locations of the individual matrices within the buffer are given by stride lengths, while the
number of matrices is given by the `batch_size` parameter."

Include Files: not stated on these pages (the per-source-file example elsewhere in this same range
uses `#include <oneapi/mkl.hpp>`).

**Buffer version — "Buffer version of imatcopy_batch supports only strided API."** The `column_major`
and `row_major` signatures are identical apart from the namespace (`row_major` repeated verbatim in
the source; p278–279):

```cpp
namespace oneapi::mkl::blas::column_major {
    void imatcopy_batch(sycl::queue &queue,
                        transpose trans,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        sycl::buffer<T, 1> &ab,
                        std::int64_t lda,
                        std::int64_t ldb,
                        std::int64_t stride,
                        std::int64_t batch_size);
}
```

**USM version — supports group API and strided API.** "The type `Ti` of integer pointers in the
group API may be either `std::int64_t` or `std::int32_t`." Group API (p280, verbatim; `row_major`
differs only in the namespace):

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event imatcopy_batch(sycl::queue &queue,
                               const transpose *trans,
                               const Ti *m,
                               const Ti *n,
                               const T *alpha, T **ab,
                               const Ti *lda,
                               const Ti *ldb,
                               std::int64_t group_count,
                               const Ti *groupsize,
                               const std::vector<sycl::event> &dependencies = {});
}
```

Strided API (p282, verbatim — the source prints two identical blocks, both headed
`column_major`; see Explicit gaps):

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event imatcopy_batch(sycl::queue &queue,
                               transpose trans,
                               std::int64_t m,
                               std::int64_t n,
                               oneapi::mkl::value_or_pointer<T> alpha,
                               T *ab,
                               std::int64_t lda,
                               std::int64_t ldb,
                               std::int64_t stride,
                               std::int64_t batch_size,
                               const std::vector<sycl::event> &dependencies = {});
}
```

Input parameters (shared by both APIs unless noted):

- `queue` — the queue where the routine should be executed.
- `trans` — (buffer/strided) "Specifies `op(AB)`, the transposition operation applied to the
  matrices `AB`." (group API) array of size `group_count`; "Each element `i` in the array specifies
  `op(AB)` the transposition operation applied to the matrices `AB`."
- `m`, `n` — number of rows/columns for each matrix `AB` on input. Must be at least 0. Group API:
  arrays of `group_count` integers; `m[i]`/`n[i]` for `AB[i]`; each entry must be at least zero.
- `alpha` — "Scaling factor for the matrix transpose or copy operation." (USM: "See Scalar Arguments
  for more information on the `value_or_pointer` data type.") Group API: array of size
  `group_count` containing scaling factors.
- `ab` — buffer/array holding the matrices `AB`. "Must have size at least `stride*batch_size`."
  Group API: "Array of size `total_batch_count`, holding pointers to arrays used to store `AB`
  matrices."
- `lda` — leading dimension of the `AB` matrices on input: column major at least `m`; row major at
  least `n`. "Must be positive." Group API: array of `group_count` integers with the same bound per
  group.
- `ldb` — leading dimension of the matrices `AB` on output. "Must be positive." Bounds (verbatim
  table, p279 for buffer/strided and p281 for group with `[i]` subscripts):

| | `trans = transpose::nontrans` | `trans = transpose::trans` or `transpose::conjtrans` |
|---|---|---|
| Column major | Must be at least `m` | Must be at least `n` |
| Row major | Must be at least `n` | Must be at least `m` |

- `stride` (strided API) — "Stride between the different `AB` matrices. It must be at least
  `max(ldb,lda)*max(ka, kb)`, where: `ka` is `m` if column major layout is used or `n` if row major
  layout is used; `kb` is `n` if column major layout is used and `AB` is not transposed, or `m`
  otherwise."
- `batch_size` (strided API) — "Specifies the number of matrices to transpose or copy. Must be at
  least zero."
- `group_count` (group API) — "Number of groups. Must be at least 0."
- `group_size` (group API) — "Array of size `group_count`. The element `group_size[i]` is the number
  of matrices in the group `i`. Each element in `group_size` must be at least 0."
- `dependencies` — "List of events to wait for before starting computation, if any. If omitted,
  defaults to no dependencies."

Output: `ab` — (buffer/strided) "Output buffer, overwritten by `batch_size` matrix multiply
operations of the form `alpha*op(AB)`." (group) "Output array of pointers to `AB` matrices,
overwritten by `total_batch_count` matrix transpose or copy operations of the form `alpha*op(AB)`."
Return value (USM): "Output event to wait on to ensure computation is complete."

### omatadd_batch

Computes a group of out-of-place scaled matrix additions using general matrices. Precisions: `float`,
`double`, `std::complex<float>`, `std::complex<double>`. Described as "similar to the `omatadd`
routines, but the `omatadd_batch` routines perform matrix operations with a group of matrices."

Operation (p284 buffer, p287 USM): "The matrices are always in a strided format for
`omatadd_batch`" (buffer wording); the USM wording on p287 drops the suffix and reads "The matrices
are always in a strided format for `omatadd`."

```
for i = 0 … batch_size – 1
    A is a matrix at offset i * stride_a in a
    B is a matrix at offset i * stride_b in b
    C is a matrix at offset i * stride_c in c
    C = alpha * op(A) + beta * op(B)
end for
where:
  op(X) is one of op(X) = X, op(X) = X', or op(X) = conjg(X')
  alpha and beta are scalars
  A, B and C are matrices
```

Overlap rule (verbatim, both versions): "In general, the `a`, `b`, and `c` buffers must not overlap
in memory, with the exception of the following in-place operations: `a` and `c` may point to the same
memory if `op(A)` is non-transpose and all the `A` matrices within `a` have the same parameters as
all the respective `C` matrices within `c`; `b` and `c` may point to the same memory if `op(B)` is
non-transpose and all the `B` matrices within `b` have the same parameters as all the respective `C`
matrices within `c`."

Include Files: not stated on these pages.

**Buffer version (strided API only)** (p284–285, verbatim; both namespaces shown in the source with
identical bodies apart from the namespace):

```cpp
namespace oneapi::mkl::blas::column_major {
    void omatadd_batch(sycl::queue &queue,
                       transpose transa,
                       transpose transb,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T, 1> &a,
                       std::int64_t lda,
                       std::int64_t stride_a,
                       T beta,
                       sycl::buffer<T, 1> &b,
                       std::int64_t ldb,
                       std::int64_t stride_b,
                       sycl::buffer<T, 1> &c,
                       std::int64_t ldc,
                       std::int64_t stride_c,
                       std::int64_t batch_size);
}
```

**USM version (strided API)** (p287–288, verbatim):

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event omatadd_batch(sycl::queue &queue,
                              transpose transa,
                              transpose transb,
                              std::int64_t m,
                              std::int64_t n,
                              oneapi::mkl::value_or_pointer<T> alpha,
                              const T *a,
                              std::int64_t lda,
                              std::int64_t stride_a,
                              oneapi::mkl::value_or_pointer<T> beta,
                              const T *b,
                              std::int64_t ldb,
                              std::int64_t stride_b,
                              T *c,
                              std::int64_t ldc,
                              std::int64_t stride_c,
                              std::int64_t batch_size,
                              const std::vector<sycl::event> &dependencies = {});
}
```

Input parameters:

- `queue` — the queue where the routine should be executed.
- `transa` / `transb` — "Specifies `op(A)`, the transposition operation applied to the matrices
  `A`." / "... the matrices `B`."
- `m`, `n` — "Number of rows for the result matrix `C`. Must be at least zero." / "Number of columns
  for the result matrix `C`. Must be at least zero."
- `alpha` — scaling factor for the matrices `A` (USM adds "See Scalar Arguments for more information
  on the `value_or_pointer` data type.").
- `a` — "Buffer holding the input matrices `A`. If `alpha` is zero, `a` is never accessed. Otherwise
  it must have size at least `stride_a*batch_size`." (USM: "... never accessed and may be a null
  pointer.")
- `lda` — leading dimension of the `A` matrices; must be positive and satisfy:

| | `transa = transpose::nontrans` | `transa = transpose::trans` or `transpose::conjtrans` |
|---|---|---|
| Column major | Must be at least `m` | Must be at least `n` |
| Row major | Must be at least `n` | Must be at least `m` |

- `stride_a` — stride between the different `A` matrices; must be positive and satisfy:

| | `transa = transpose::nontrans` | `transa = transpose::trans` or `transpose::conjtrans` |
|---|---|---|
| Column major | Must be at least `lda*n` | Must be at least `lda*m` |
| Row major | Must be at least `lda*m` | Must be at least `lda*n` |

- `beta` — scaling factor for the matrices `B` (same `value_or_pointer` note for USM).
- `b` — as `a`, with `stride_b*batch_size` and `beta`.
- `ldb` — as `lda` with `transb` and `B` (same `m`/`n` bounds table).
- `stride_b` — as `stride_a` with `transb` and `ldb` (same `ldb*n` / `ldb*m` bounds table).
- `ldc` — "Leading dimension of the `C` matrices. If matrices are stored using column major layout,
  `ldc` must be at least `m`. If matrices are stored using row major layout, `ldc` must be at least
  `n`. Must be positive."
- `stride_c` — "Stride between the different `C` matrices. If matrices are stored using column major
  layout, `stride_c` must be at least `ldc*n`. If matrices are stored using row major layout,
  `stride_c` must be at least `ldc*m`."
- `batch_size` — "Specifies the number of input and output matrices to add. Must be at least zero."
- `dependencies` — "List of events to wait for before starting computation, if any. If omitted,
  defaults to no dependencies." (The buffer version's input-parameter list also prints this entry
  even though its signature has no `dependencies` parameter — see Explicit gaps.)

Output: `c` — "Output buffer/array, overwritten by `batch_size` matrix addition operations of the
form `alpha*op(A) + beta*op(B)`. Must have size at least `stride_c*batch_size`." Return value (USM):
"Output event to wait on to ensure computation is complete."

### Sparse BLAS names on the assigned pages (index only)

Pages 294–304 introduce the Sparse BLAS domain and the CSR format; they contain **routine tables and
format rules, but no signatures** for any sparse routine. All sparse routines support
`<DATA_TYPE>` ∈ {`float`, `double`, `std::complex<float>`, `std::complex<double>`} with integer
types `<INT_TYPE>` ∈ {`std::int32_t`, `std::int64_t`} (p300). The full API is documented in the
Sparse BLAS chapters of this digest; names are listed here so nothing from the assigned pages is
dropped.

Data structures and objects (p295): `sparse::matrix_handle_t` (pointer to an opaque sparse matrix
object), `sparse::property` (user-provided matrix data guarantees, such as sorted input data),
`sparse::omatadd_alg`, `sparse::omatadd_descr_t`, `sparse::matrix_view_descr`,
`sparse::matmat_request`, `sparse::matmat_descr_t`, `sparse::omatconvert_alg`,
`sparse::omatconvert_descr_t`.

| Group | Routines |
|---|---|
| State management (p296–297) | `sparse::init_matrix_handle`, `sparse::release_matrix_handle`, `sparse::set_csr_data`, `sparse::set_csc_data`, `sparse::set_coo_data`, `sparse::set_bsr_data`, `sparse::set_matrix_property`, `sparse::init_omatadd_descr`, `sparse::release_omatadd_descr`, `sparse::init_matmat_descr`, `sparse::set_matmat_data`, `sparse::get_matmat_data`, `sparse::release_matmat_descr`, `sparse::init_omatconvert_descr`, `sparse::release_omatconvert_descr` |
| Analysis / inspector / optimize (p297) | `sparse::optimize_gemv`, `sparse::optimize_trmv`, `sparse::optimize_trsv`, `sparse::optimize_gemm`, `sparse::optimize_trsm` |
| Execution (p297–298) | `sparse::gemv`, `sparse::gemvdot`, `sparse::symv`, `sparse::trmv`, `sparse::trsv`, `sparse::gemm`, `sparse::trsm`, `sparse::omatadd`, `sparse::matmat`, `sparse::matmatd` |
| Helper (p298) | `sparse::omatcopy`, `sparse::omatconvert`, `sparse::sort_matrix`, `sparse::update_diagonal_values` |

Descriptions verbatim from the tables — execution: `sparse::gemv` "General sparse matrix-dense
vector product"; `sparse::gemvdot` the same "with fused dot product"; `sparse::symv` "Symmetric
sparse matrix-dense vector product"; `sparse::trmv` "Triangular sparse matrix-dense vector product";
`sparse::trsv` "Triangular solve of sparse matrix against a dense vector."; `sparse::gemm` "General
sparse matrix-dense matrix product"; `sparse::trsm` "Triangular solve of sparse matrix against a
dense matrix."; `sparse::omatadd` "General sparse matrix-sparse matrix addition"; `sparse::matmat`
"General sparse matrix-sparse matrix product"; `sparse::matmatd` "Product of two sparse matrices with
a dense output matrix". Helper: `sparse::omatcopy` "General sparse matrix out-of-place
copy/transposition into a new matrix handle"; `sparse::omatconvert` "General sparse matrix
out-of-place conversion into a new matrix handle"; `sparse::sort_matrix` "General sparse matrix sort
of matrix format in matrix handle"; `sparse::update_diagonal_values` "Change values of the main
diagonal entries in matrix handle". Each `sparse::optimize_*` routine "Perform[s] internal
optimizations for the `sparse::<name>` operation."

### Compressed Sparse Row (CSR) format on the assigned pages

CSR ("sometimes called 3-array CSR or CSR3") is "represented by scalar sizes (`nrows`, `ncols`,
`nnz`), as well as three data arrays: `row_ptr`, `col_ind` and `values`, and the `index_base`
parameter" (p301). Element definitions (verbatim, p301–302):

- `nrows`, `ncols` — number of rows / columns in the sparse matrix.
- `nnz` — "Number of stored elements (sometimes called number of non-zeros) in the sparse matrix."
- `index` — "Parameter that is used to specify whether the matrix has zero or one-based indexing."
- `values` — "An array that contains the `nnz` stored element values of the sparse matrix stored row
  by row."
- `col_ind` — "An integer array of `nnz` column indices for the stored (sometimes called non-zero)
  elements stored in the `values` array, such that `col_ind[i]` is the column number (using zero- or
  one-based indexing) of the element of the sparse matrix stored in `values[i]`."
- `row_ptr` — "An integer array of size equal to `nrows + 1`. Element `j` of this integer array gives
  the position of the element in the `values` array that is first non-zero element in a row `j` of
  `A`. Note that this position is equal to `row_ptr[j] - index`. The last element of the `row_ptr`
  array (`row_ptr[nrows]`) stores the sum of `nnz` and `index`. That is, `nnz = row_ptr[nrows] -
  index`."

Sortedness (p302): "The 3-array CSR format that oneMKL supports has sorted rows by definition.
However, the column indices within each row may or may not be sorted (in ascending order) leading to
a possibility of unsorted data." State table: **Unsorted** — "Column indices within each row may not
be ordered" — appropriate property "None (default)"; **Sorted** — "Sorted by rows and sorted by
columns within each row" — `sparse::property::sorted`. "oneMKL assumes CSR matrix handles to be
unsorted by default. If the CSR format arrays are known to be in sorted form ..., then users are
strongly encouraged to provide that information to oneMKL through the `sparse::set_matrix_property()`
API to possibly allow better algorithm choices and optimizations ... This is particularly important
for APIs like triangular solve or sparse matrix addition."

## Formulas

Formulas transcribed from the formula images extracted from the PDF (formulas in the source are
images and did not survive text extraction). Each is labelled with its routine/section and PDF page.
The operation loops that survived as text are reproduced inside the routine subsections above:
`imatcopy_batch` (p278 buffer-strided / p282 USM-strided) `AB = alpha * op(AB)`, `imatcopy_batch`
(p280 USM-group) the same per group element, and `omatadd_batch` (p284 buffer / p287 USM)
`C = alpha * op(A) + beta * op(B)`, all with `op(X) ∈ {X, X', conjg(X')}`.

- **imatcopy_batch (p280), USM group API:** `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`
  (image tag `p280#0`; the prose sentence introducing it, "The total number of entries in `ab` is
  given by:", is truncated in the extracted text — the expression comes from the image).
- **Numerical Reproducibility (p294), summation:** `a_0 + a_1 + ... + a_n` (image tag `p294#0`) —
  the expression whose value "will depend on the order of operations, because floating-point
  addition is not associative".
- **CSR Case 1 (p302), sorted square matrix, zero-based indexing:** `A = [[1.0, 0.0, 2.0], [0.0,
  -1.0, 4.0], [3.0, 0.0, 0.0]]` (image tag `p302#0`; see the `row_ptr`/`col_ind`/`values` arrays in
  Conventions).
- **CSR Case 2 (p303), sorted rectangular matrix with one-based indexing and an empty row:** `A =
  [[1.0, 0.0, 2.0, 0.0, 0.0], [0.0, -1.0, 4.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0,
  0.0, 1.0, 0.0]]` (image tag `p303#0`).
- **CSR Case 3 (p304), unsorted rectangular matrix, zero-based indexing:** `A = [[1.0, 0.0, 2.0,
  0.0, 0.0], [0.0, -1.0, 4.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0]]`
  (image tag `p304#0`).

## Conventions & Gotchas

- **Where the compute-mode argument goes.** Buffer API: `compute_mode mode = compute_mode::unset`
  is the last parameter. USM API: `compute_mode mode = compute_mode::unset` immediately precedes
  `const std::vector<sycl::event> &dependencies = {}`. "either argument may be omitted" (p292).
  Omitting the mode argument does **not** mean "no modes allowed by default" in the sense of the
  table — the default is the separate enumerator `compute_mode::unset`.
- **Precedence and combining.** per-call/per-source-file settings > `MKL_BLAS_COMPUTE_MODE`
  environment variable, and within a source file a `compute_mode` parameter overrides the `#define
  MKL_BLAS_COMPUTE_MODE` default; `-DMKL_BLAS_COMPUTE_MODE=...::standard` on the compile line
  "effectively disables the environment variable." Enum values are OR'ed with `|` in C++
  (`compute_mode::float_to_bf16x2 | compute_mode::float_to_tf32`); environment-variable names are
  comma-separated and uppercase (`FLOAT_TO_BF16X2,FLOAT_TO_TF32`). The macro "must be set ... before
  including any oneMKL header files" and must be "an expression of type
  `oneapi::mkl::blas::compute_mode`".
- **`force_alternate` can throw.** "Never use the standard implementation; if no allowed alternate
  implementation is available, an exception will be thrown." No exception type is named on these
  pages. `prefer_alternate` does not throw; it only prefers an alternate implementation and accepts
  reduced performance. `compute_mode::any` allows any alternate mode; `compute_mode::standard`
  disallows all alternate modes; `prefer_alternate` and `force_alternate` are modifiers "used in
  conjunction with one or more alternate modes", not standalone modes.
- **Accuracy notes are mode-specific.** `float_to_bf16`: reduced accuracy, possibly much higher
  performance. `float_to_bf16x2`: between standard single precision and bfloat16. `float_to_bf16x3`
  and `complex_3m`: comparable to standard in most cases. `float_to_bf16x2` and `float_to_bf16x3`
  warn: "infinite inputs may produce unexpected NaNs in the output."
- **CNR scope.** Bitwise run-to-run reproducibility is conditional: CPU — all BLAS DPC++ APIs; GPU —
  level-3 routines and level-3 extensions. The source gives no CNR API in this range; consult the
  oneMKL Developer Guide.
- **`imatcopy_batch` is in-place and its API coverage differs by memory form.** The same matrix
  storage is read and overwritten; `lda` describes the input leading dimension and `ldb` the output
  leading dimension, and the two have different lower bounds depending on `trans` and the layout
  namespace. The strided form additionally requires `stride >= max(ldb,lda)*max(ka, kb)` with `ka` =
  `m` (column major) or `n` (row major) and `kb` = `n` (column major, `AB` not transposed) or `m`
  otherwise. Buffer: strided API only. USM: group API and strided API, where the group API's integer
  pointer type `Ti` "may be either `std::int64_t` or `std::int32_t`".
- **`omatadd_batch` is out-of-place but has two sanctioned in-place aliases.** `a`/`c` may alias only
  when `op(A)` is non-transpose and the `A` matrices have the same parameters as the corresponding
  `C` matrices; `b`/`c` likewise for `B`. All other overlap between `a`, `b`, and `c` is disallowed.
  If `alpha` is zero, `a` is never accessed and (USM) may be a null pointer; if `beta` is zero, `b`
  is never accessed and (USM) may be a null pointer. `stride_a`/`stride_b` bounds are `ld*{n|m}`
  depending on `trans*` and layout; `stride_c` is at least `ldc*n` (column major) or `ldc*m` (row
  major); buffer sizes are `stride_a*batch_size`, `stride_b*batch_size`, `stride_c*batch_size`.
- **Return/event convention.** USM versions of both batch routines return `sycl::event` ("Output
  event to wait on to ensure computation is complete."); buffer versions return `void` and take no
  `dependencies`. For USM, pass `dependencies` to chain work.
- **Matrix-layout namespace selects sizes, not just wrappers.**
  `oneapi::mkl::blas::column_major` vs `row_major` flips every `ld*` bound and stride bound; the
  same code with the wrong namespace silently computes wrong results.
- **Headers.** No "Include Files" block survived extraction for any routine on these pages. The only
  header named in this source range is `#include <oneapi/mkl.hpp>` in the per-source-file
  `MKL_BLAS_COMPUTE_MODE` example. Do not invent per-routine headers.
- **Sparse: repeated indices are undefined behavior.** Verbatim (p301): matrices with "so-called
  'repeated' indices where the actual value of the non-zero is considered as a sum of its values
  where the non-zero row and column indices are repeated ... is currently undefined behavior with no
  guarantees of correctness. It is user-responsibility to ensure that non-zero values are not
  repeated and are appropriately 'compressed' into a single non-zero value."
- **Sparse: "non-zero" means structural non-zero.** "All references of 'non-zeros' in the oneMKL
  Sparse BLAS domain mean structural non-zeros that are stored in the matrix (which includes
  explicitly stored zero values)."
- **Sparse handle contract (p299–300).** `sparse::matrix_handle_t` is a "view of User's matrix data
  arrays with an opaque state attached to it"; it is populated by `sparse::set_<format>_data(q,
  handle, /*user data*/)` and **persists outside individual library calls**. The User owns the data
  arrays, must not modify them while attached, and may dispose of them only after all uses and the
  matrix handle release have finished. The Library may store its own data in the handle (not
  accessible to the User) and "agrees to not modify the user-provided data arrays ... unless through
  a library API that specifically states it may change the data (such as `sparse::sort_matrix()` or
  `sparse::omatcopy()`)". Device USM pointers bind the handle to the associated `sycl::context`/
  `sycl::device`; shared/host USM or `sycl::buffer` bind it to the context. "The
  `sparse::matrix_handle_t` object is not currently considered thread-safe, so it must be used
  serially on the host." Reusing the same arrays across handles, or reading them in other kernels
  concurrently, is "recommended ... not" to do although possible with care, and "may affect the
  performance."
- **CSR index base.** `row_ptr[j] - index` is the position in `values` of the first stored element of
  row `j`; `nnz = row_ptr[nrows] - index`. `row_ptr` has `nrows + 1` entries; `col_ind` and `values`
  have `nnz` entries and "must be consistent with each other."
- **CSR sortedness is opt-in.** CSR handles are assumed unsorted by default; declare
  `sparse::property::sorted` via `sparse::set_matrix_property()` only when all column indices within
  each row are ascending. It matters most for triangular solve and sparse matrix addition.
- **CSR worked examples (p302–304).** Case 1 (zero-based, 3×3, `nnz` = 5): `index` = 0, `row_ptr` =
  {0, 2, 4, 5}, `col_ind` = {0, 2, 1, 2, 0}, `values` = {1.0, 2.0, -1.0, 4.0, 3.0}. Case 2
  (one-based, 4×5 with empty row 3, `nnz` = 7): `index` = 1, `row_ptr` = {1, 3, 6, 6, 8}, `col_ind` =
  {1, 3, 2, 3, 5, 1, 4}, `values` = {1.0, 2.0, -1.0, 4.0, 1.0, 3.0, 1.0}. Case 3 (zero-based, 4×5
  unsorted within rows, `nnz` = 10): `index` = 0, `row_ptr` = {0, 2, 5, 9, 10}, `col_ind` = {0, 2, 4,
  1, 2, 1, 2, 0, 3, 0}, `values` = {1.0, 2.0, 1.0, -1.0, 4.0, 2.0, 3.0, 1.0, 4.0, 3.0}.
- **Sparse matrix formats supported in this domain (p301):** `Compressed Sparse Row (CSR)`,
  `Coordinate (COO)`, `Compressed Sparse Column (CSC)`, `Block Compressed Sparse Row (BSR)`. Only CSR
  is described on the assigned pages; `sparse::set_bsr_data` is the only mention of BSR here
  ("Supports square or rectangular blocks with in row or column major layout").

## Explicit gaps

- **No CNR API.** The Numerical Reproducibility section names no function, macro, environment
  variable, or enumerator; it only says CNR modes exist, states the CPU/GPU availability scope, and
  points to the oneMKL Developer Guide (Windows, Linux). Everything a caller would need to *enable*
  CNR is absent from the assigned text.
- **`compute_mode::unset` is undocumented.** It appears only as the default value of the `mode`
  parameter in routine syntax blocks (observed in `gemm`, p146/p149). The Mode Settings table gives
  no row or description for it, and no precedence rule for `unset` versus an explicit mode is stated.
- **Enum identifiers in the mode table are broken by line/page breaks in the extraction:**
  `compute_mode::float_to_bf1` + `6`, `compute_mode::float_to_bf1` + `6x2`,
  `compute_mode::float_to_bf1` + `6x3`, `compute_mode::float_to_tf3` + `2`,
  `compute_mode::prefer_alter` + `nate`, `compute_mode::force_altern` + `ate`. They are reconstructed
  above from the uppercase environment-variable names in the same table plus the example code that
  uses them unbroken (`compute_mode::float_to_bf16x2`, `compute_mode::float_to_tf32`,
  `compute_mode::complex_3m`, `compute_mode::force_alternate`).
- **No signatures for `gemmt`, `syrk`, `syr2k`** on the assigned pages; only their names inside the
  compute-mode NOTE (plus `syrk` example calls). Their full syntax is outside this range.
- **No sparse routine signatures.** Pages 294–304 give only names, data types, and one-line
  descriptions; `sparse::init_matrix_handle` is the first sparse routine with its own section, and it
  begins on p315 (outside this range).
- **No error/exception semantics for compute modes** beyond the statement that `force_alternate`
  throws "an exception" when no allowed alternate implementation is available. No exception type or
  error code is given here.
- **`imatcopy_batch` USM strided API.** The extracted Syntax block prints two identical blocks, both
  under `namespace oneapi::mkl::blas::column_major { ... }`; no `row_major` USM strided block appears
  in the text. This may be a source typo or an extraction duplication — it is reproduced verbatim
  rather than corrected, and the `row_major` USM strided signature is therefore not transcribed.
- **`omatadd_batch` buffer version** lists `dependencies` ("List of events to wait for before
  starting computation...") among its Input Parameters although the printed buffer signature has no
  `dependencies` parameter and returns `void`.
- **`imatcopy_batch` group-API parameter name mismatch.** The group-API signature declares
  `const Ti *groupsize` while the Input Parameters text calls the same argument `group_size`; both
  spellings are reproduced here as they appear in the source.
- **`total_batch_count` prose is truncated** in the sourced text for `imatcopy_batch`'s group API
  ("The total number of entries in `ab` is given by:"); the definition was recovered from the formula
  image `p280#0` (`total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`).
- **COO, CSC, and BSR formats** are named on p301 but described only in sections beginning at p304 and
  beyond, which are outside the assigned pages.
