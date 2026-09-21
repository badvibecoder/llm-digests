# BLAS-like Extensions and Batched Routines

This chapter covers the oneMKL DPC++ routines that extend the standard BLAS set: the vector/scalar
extension `axpby`, the "omat"/"imat" matrix copy-add family, and the *batched* routines that
perform many independent vector-vector, vector-matrix, or matrix-matrix operations in one call. It
also covers the triangular routines `trmm`/`trsm` that precede the extensions section in the
source, plus the Compute Modes and numerical-reproducibility rules applying to all BLAS level-3
routines and extensions. Every routine exists in two parallel namespaces (`column_major` and
`row_major`) and, except where noted, as a Buffer version and a USM version.

## Overview

**Namespaces.** All routines here are declared in both `oneapi::mkl::blas::column_major` and
`oneapi::mkl::blas::row_major`. The overloads share a name and (except where noted per routine) a
parameter list; the namespace selects the memory layout used to interpret leading dimensions and
`m`/`n`.

**Buffer vs USM.** Buffer versions return `void` and take `sycl::buffer<T,1>&` arguments. USM
versions take raw pointers and return `sycl::event` ("Output event to wait on to ensure computation
is complete."). USM scalars are `oneapi::mkl::value_or_pointer<T>` (see Scalar Arguments in the
source); Buffer versions take the scalar by value (`T alpha`, or `float` for `gemm_bias`).

**Optional arguments.** `compute_mode mode = compute_mode::unset` ("Optional. Compute mode
settings.") and `const std::vector<sycl::event> &dependencies = {}` ("Optional. List of events to
wait for before starting computation, if any. If omitted, defaults to no dependencies."). "mode and
dependencies may be omitted independently; it is not necessary to specify mode in order to provide
dependencies." In USM signatures `mode` precedes `dependencies`.

**Batched API model.** Two shapes:
* *Strided API* — one buffer/array holds all items; item `i` starts at offset `i * stride...`;
  sizes and increments are uniform; the count is `batch_size`.
* *Group API* (USM only) — arrays of pointers plus per-group parameter arrays; group `i` holds
  `group_size[i]` items; the count is
  `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`.
  "The type `Ti` of integer pointers in the group API may be either `std::int64_t` or
  `std::int32_t`."
* Buffer batch versions support **only** the strided API. `omatadd_batch` is strided-only in both
  its Buffer and USM versions (no group API).

**Blas-like extensions table (pages 193-194).** `float`, `double`, `std::complex<float>`,
`std::complex<double>` are supported by `axpby`, `axpy_batch`, `copy_batch`, `dgmm_batch`, `gemmt`,
`gemv_batch`, `syrk_batch`, `trsm_batch`, `omatcopy`, `imatcopy`, `omatadd`, `omatcopy_batch`,
`imatcopy_batch`, `omatadd_batch`, and (for `trmm`/`trsm`) as listed below. `gemm_batch` supports
"std::int8_t, oneapi::mkl::bfloat16, sycl::half, float, double, std::complex<float>,
std::complex<double>, mixed". `gemm_bias` supports "mixed std::int8_t, std::uint8_t, and
std::int32_t".

**Compute Modes** (pages 290-294). "BLAS level-3 routines and extensions support alternate compute
modes, which can provide increased performance in exchange for different numerical properties or
reduced accuracy." Modes are OR'ed with `|` per call or per source file
(`compute_mode::float_to_bf16x2 | compute_mode::float_to_tf32`), or set application-wide with commas
in `MKL_BLAS_COMPUTE_MODE` (`set MKL_BLAS_COMPUTE_MODE=FLOAT_TO_BF16X2,FLOAT_TO_TF32`). Per-call
and per-source-file settings take precedence over the environment variable; a per-source-file
default comes from `#define MKL_BLAS_COMPUTE_MODE oneapi::mkl::blas::compute_mode::<mode>` before any
oneMKL header.

| enum value | env var setting | effect |
|---|---|---|
| `compute_mode::float_to_bf16` | `FLOAT_TO_BF16` | single-precision inputs converted to bfloat16 internally; output accumulated in single precision; reduced accuracy, possibly much higher performance |
| `compute_mode::float_to_bf16x2` | `FLOAT_TO_BF16X2` | each input as a sum of two bfloat16 values; accumulation in single precision; accuracy between standard single and bfloat16; infinite inputs may produce unexpected NaNs |
| `compute_mode::float_to_bf16x3` | `FLOAT_TO_BF16X3` | each input as a sum of three bfloat16 values; accumulation in single precision; accuracy comparable to standard single precision in most cases; infinite inputs may produce unexpected NaNs |
| `compute_mode::float_to_tf32` | `FLOAT_TO_TF32` | tf32 internally; accumulation in single precision; accuracy between `float_to_bf16` and `float_to_bf16x2` |
| `compute_mode::complex_3m` | `COMPLEX_3M` | reduce the four real multiplications in a standard complex multiplication to three |
| `compute_mode::any` | `ANY` | allow any alternate compute mode |
| `compute_mode::standard` | `STANDARD` | do not allow any alternate compute modes |
| `compute_mode::prefer_alternate` | `PREFER_ALTERNATE` | used with one or more alternate modes; prefer an alternate over the standard implementation whenever available, even if performance may be reduced |
| `compute_mode::force_alternate` | `FORCE_ALTERNATE` | used with one or more alternate modes; never use the standard implementation; if no allowed alternate implementation is available, an exception will be thrown |

"By default, oneMKL does not enable any alternate compute modes." "In the current oneMKL release,
alternate implementations are available for `gemm`, `gemmt`, `syrk`, and `syr2k` on selected
hardware for real single precision, and `gemm` only for complex single precision." If no allowed
mode is supported or expected to help, oneMKL falls back to a standard implementation. On GPU,
verbose mode (`MKL_VERBOSE=1` or the `mkl_verbose` API) logs the enabled/selected mode, e.g.
`... mode:float_to_bf16 host:nan device:nan GPU0`.

**Numerical reproducibility** (page 294). "Executing a BLAS routine with the same inputs may result
in slightly different results from run to run… the results of a summation will depend on the order
of operations, because floating-point addition is not associative." CNR modes provide bitwise
reproducibility: "CNR mode is available on CPU for all BLAS DPC++ APIs, and on GPU for level-3
routines and level-3 extensions." Details are in the oneMKL Developer Guide.

**Include files.** The extracted sections for these routines contain no "Include Files" block; the
only include shown in this material is `#include <oneapi/mkl.hpp>` (Compute Modes examples).

## Routines

Unless stated otherwise: `queue` is "The queue where the routine will be executed" for
`trmm`/`trsm`/`omatcopy`/`imatcopy`/`omatadd` and "should be executed" for the other extensions;
`m`, `n`, `k` are referenced dimensions ("Must be at least zero" where
stated); `alpha`/`beta` are scalars; `lda`/`ldb`/`ldc` are leading dimensions and "Must be positive".
Where a leading dimension has a four-case bound, the order is always: **nontrans column major,
trans/conjtran column major, nontrans row major, trans row major** (per-routine lists below give the
bound explicitly).

### trmm

"Computes a matrix-matrix product where one input matrix is triangular and other matrix is
general." `op(A)` is `op(A) = A`, `op(A) = A^T`, or `op(A) = A^H`; `A` is `m x m` or `n x n`
triangular; `B` and `C` are `m x n`. `left_right` = `side::left` or `side::right`; both in-place and
out-of-place operations exist.

```cpp
// Buffer, in-place (p178; row_major has the identical parameter list)
void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a,
          std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, compute_mode mode = compute_mode::unset)
// Buffer, out-of-place (p180; row_major identical): as above plus "T beta," and "sycl::buffer<T,1> &c,
//   std::int64_t ldc," inserted after ldb (before mode)
// USM, in-place (p181): sycl::event trmm(sycl::queue&, side, uplo, transpose, diag, m, n,
//   oneapi::mkl::value_or_pointer<T> alpha, const T *a, lda, T *b, ldb,
//   compute_mode mode = compute_mode::unset, const std::vector<sycl::event> &dependencies = {})
// USM, out-of-place (p183): ... oneapi::mkl::value_or_pointer<T> alpha, const T *a, lda, const T *b, ldb,
//   oneapi::mkl::value_or_pointer<T> beta, T *c, ldc, mode, dependencies
```

`left_right` "Specifies whether matrix A is on the left side or right side of the multiplication.";
`upper_lower` upper/lower triangular; `trans` is `op(A)`; `unit_diag` unit triangular or not; `a` at
least `lda * m` (left) or `lda * n` (right); `lda` at least `m` (left) or `n` (right); `b` at least
`ldb * n` column major or `ldb * m` row major. Output: `b` overwritten by `alpha * op(A) * B`
(left) or `alpha * B * op(A)` (right); out-of-place `c` by `alpha * op(A) * B + beta * C` (left) or
`alpha * B * op(A) + beta * C` (right). NOTE: "If `alpha = 0`, matrix `B` is set to zero, and `A`
and `B` do not need to be initialized at entry."

### trsm

"Solves a triangular matrix equation (forward or backward solve)": `op(A) * X = alpha * B` or
`X * op(A) = alpha * B`. `A` is `m x m` or `n x n` triangular; `B`, `X`, `C` are `m x n`. "For the
in-place operation, the matrix B is overwritten by solution matrix X, while for the out-of-place
operation, B remains untouched and the solution is added to a scaled C matrix."

```cpp
// Buffer, in-place (p186). Verbatim difference: column_major names the transpose argument "transa";
// row_major names it "trans".
void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
          oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a,
          std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, compute_mode mode = compute_mode::unset)
// Buffer, out-of-place (p187): as above (still transa / trans by namespace) plus "T beta," and
//   "sycl::buffer<T,1> &c, std::int64_t ldc," after ldb
// USM, in-place (p189): sycl::event trsm(sycl::queue&, side, uplo, transpose trans, diag, m, n,
//   oneapi::mkl::value_or_pointer<T> alpha, const T *a, lda, T *b, ldb,
//   compute_mode mode = compute_mode::unset, const std::vector<sycl::event> &dependencies = {})
// USM, out-of-place (p191): ... const T *b, ldb, oneapi::mkl::value_or_pointer<T> beta, T *c, ldc, mode,
//   dependencies; both namespaces use the name trans
```

Parameters mirror `trmm`. Output in-place: `b` = solution matrix `X`; out-of-place: `c` =
"solution matrix X + beta * C". NOTE: "If `alpha = 0`, matrix `B` is set to zero, and `A` and `B`
do not need to be initialized before calling `trsm`." (No `beta = 0` shortcut note exists for the
solve.)

### axpby

"Computes a vector-scalar product added to a scaled-vector": `y <- beta * y + alpha * x`. No `mode`
parameter in either version.

```cpp
// Buffer (p194)
void axpby(sycl::queue &queue, std::int64_t n, T alpha, sycl::buffer<T,1> &x, std::int64_t incx,
           T beta, sycl::buffer<T,1> &y, std::int64_t incy);
// USM (p195). Verbatim: dependency type is "event", not "sycl::event".
sycl::event axpby(sycl::queue &queue, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                  const T *x, std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y,
                  std::int64_t incy, const std::vector<event> &dependencies = {});
```

`n` "Number of elements in vectors x and y."; `x`/`y` size "at least `1 + (n - 1)*abs(incx))`" /
"`1 + (n - 1)*abs(incy))`"; `incx`/`incy` "Stride between two consecutive elements" of x/y. Output
`y`: "Buffer/Array holding updated vector y."

### axpy_batch

"Computes a group of axpy operations… Each axpy operation adds a scalar-vector product to a vector":
`Y = alpha * X + Y`. No `mode` parameter. Buffer version supports only strided API.

```cpp
// Buffer, strided only (p197)
void axpy_batch(sycl::queue &queue, std::int64_t n, T alpha, sycl::buffer<T, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<T, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size)
// USM, group API (p199)
sycl::event axpy_batch(sycl::queue &queue, const Ti *n, const T *alpha, const T **x, const Ti *incx, T **y,
                       const Ti *incy, std::int64_t group_count, const Ti *group_size,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p200)
sycl::event axpy_batch(sycl::queue &queue, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                       const T *x, std::int64_t incx, std::int64_t stridex, T *y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies = {})
```

`x` at least `batch_size * stridex`; `incx`/`incy` "Must not be zero."; `stridex` "Must be at least
zero."; `stridey` "Must be at least `(1 + (n-1)*abs(incy))`"; `batch_size` "Number of axpy
computations to perform. Must be at least zero." Group: `n[i]`/`incx[i]`/`incy[i]`/`alpha[i]` per
group; `group_size[i]` "the number of axpy operations in group i. Each element in group_size must be
at least zero."; per-group X/Y arrays at least `(1 + (n[i] – 1)*abs(incx[i]))` /
`(1 + (n[i] – 1)*abs(incy[i]))`. Output `y`: overwritten by `batch_size`/`total_batch_count` axpy
operations of the form `alpha * X + Y`.

### copy_batch

"Computes a group of copy operations", `Y = X`; same strided/group structure as `axpy_batch`
without `alpha`, and no `mode` parameter. Buffer version supports only strided API.

```cpp
// Buffer, strided only (p202)
void copy_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<T, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<T, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size)
// USM, group API (p203)
sycl::event copy_batch(sycl::queue &queue, const Ti *n, const T **x, const Ti *incx, T **y, const Ti *incy,
                       std::int64_t group_count, const Ti *group_size,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p205)
sycl::event copy_batch(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, std::int64_t stridex,
                       T *y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event> &dependencies = {})
```

`n`, `incx`, `incy`, `stridex`, `stridey`, `batch_size`, `group_count`, `group_size` carry the same
requirements as `axpy_batch`. Output `y`: "overwritten by `batch_size`/`total_batch_count` copy
operations." Example (source): `share/doc/mkl/examples/sycl/blas/source/copy_batch_usm.cpp`.

### dgmm_batch

"Computes a group of (diagonal matrix-matrix product (dgmm) operations"; `C = diag(X) * A` if
`left_right == side::left`, else `C = A * diag(X)`. "The diagonal matrices are stored as dense
vectors." No `mode` parameter. Buffer version supports only strided API. The Buffer and USM strided
variants print the row count as `std::inte64_t` (sic); the group API uses `const Ti *m`.

```cpp
// Buffer, strided only (p207)
void dgmm_batch(sycl::queue &queue, oneapi::mkl::side left_right, std::inte64_t m, std::int64_t n,
                sycl::buffer<T,1> &a, std::int64_t lda, std::int64_t stridea, sycl::buffer<T,1> &x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<T,1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size);
// USM, group API (p209)
sycl::event dgmm_batch(sycl::queue &queue, const oneapi::mkl::side *left_right, const Ti *m, const Ti *n,
                       const T **a, const Ti *lda, const T **x, const Ti *incx, T **c, const Ti *ldc,
                       std::int64_t group_count, const Ti *group_size,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p211): std::inte64_t m, std::int64_t n, const T *a, lda, stridea, const T *x, incx,
//   stridex, T *c, ldc, stridec, batch_size, const std::vector<sycl::event> &dependencies = {}
```

`a` at least `lda * k + stridea * (batch_size - 1)` where `k` is `n` column major or `m` row major;
`lda` at least `m` column major / `n` row major; `stridea` at least zero; `x` at least
`(1 + (len - 1)*abs(incx)) + stridex * (batch_size - 1)` where `len` is `n` if the diagonal matrix
is on the right or `m` otherwise; `stridex` at least zero; `c` at least `batch_size * stridec`;
`ldc` at least `m` column major / `n` row major; `stridec` at least `ldc * n` column major or
`ldc * m` row major; `batch_size` "Number of dgmm computations to perform. Must be at least zero."
Group: `incx[i]` "must be positive"; `lda[i]`/`ldc[i]` positive and at least `m[i]` (column major)
or `n[i]` (row major); `x[i]` at least `(1 + len[i] – 1)*abs(incx[i]))`; `a[i]` at least `lda[i] *
n[i]` column major or `lda[i] * m[i]` row major; `c[i]` at least `ldc[i] * n[i]` column major or
`ldc[i] * m[i]` row major. Output `c`: overwritten by `batch_size`/`total_batch_count` dgmm
operations.

### gemm_batch

"Computes groups of matrix-matrix product with general matrices":
`C = alpha * op(A) * op(B) + beta * C`, `op(X)` one of `op(X) = X`, `op(X) = X^T`, `op(X) = X^H`;
`op(A)` is `m x k`, `op(B)` is `k x n`, `C` is `m x n`. Precisions as (`Ta`,`Tb`,`Tc`,`Ts`):
(`sycl::half`,`sycl::half`,`sycl::half`,`sycl::half`); (`sycl::half`,`sycl::half`,`float`,`float`);
(`oneapi::mkl::bfloat16`,`oneapi::mkl::bfloat16`,`oneapi::mkl::bfloat16`,`float`);
(`oneapi::mkl::bfloat16`,`oneapi::mkl::bfloat16`,`float`,`float`);
(`std::int8_t`,`std::int8_t`,`std::int32_t`,`float`); (`std::int8_t`,`std::int8_t`,`float`,`float`);
(`float`,`float`,`float`,`float`); (`double`,`double`,`double`,`double`); `std::complex<float>` and
`std::complex<double>` for all four.

```cpp
// Buffer, strided only (p213)
void gemm_batch(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, Ts alpha, sycl::buffer<Ta,1> &a,
                std::int64_t lda, std::int64_t stridea, sycl::buffer<Tb,1> &b, std::int64_t ldb,
                std::int64_t strideb, Ts beta, sycl::buffer<Tc,1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size, compute_mode mode = compute_mode::unset)
// USM, group API, pointer overload (p217; row_major identical)
sycl::event gemm_batch(sycl::queue &queue, const oneapi::mkl::transpose *transa,
                       const oneapi::mkl::transpose *transb, const Ti *m, const Ti *n, const Ti *k,
                       const Ts *alpha, const Ta **a, const Ti *lda, const Tb **b, const Ti *ldb,
                       const Ts *beta, Tc **c, const Ti *ldc, std::int64_t group_count,
                       const Ti *group_size, compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
// USM, group API, span overload (p217). Span sizes may be 1, group_count, or total_batch_count for
// every parameter except the output matrices, whose span "must be the total batch size".
sycl::event gemm_batch(sycl::queue &queue, const sycl::span<oneapi::mkl::transpose> &transa,
                       const sycl::span<oneapi::mkl::transpose> &transb, const sycl::span<std::int64_t> &m,
                       const sycl::span<std::int64_t> &n, const sycl::span<std::int64_t> &k,
                       const sycl::span<Ts> &alpha, const sycl::span<const Ta*> &a,
                       const sycl::span<std::int64_t> &lda, const sycl::span<const Tb*> &b,
                       const sycl::span<std::int64_t> &ldb, const sycl::span<Ts> &beta,
                       sycl::span<Tc*> &c, const sycl::span<std::int64_t> &ldc, size_t group_count,
                       const sycl::span<size_t> &group_sizes, compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p221): oneapi::mkl::value_or_pointer<Ts> alpha/beta, const Ta *a, const Tb *b,
//   Tc *c, lda/stridea/ldb/strideb/ldc/stridec/batch_size, compute_mode mode = compute_mode::unset,
//   const std::vector<sycl::event> &dependencies = {}
```

Sizes: `a` at least `stridea * batch_size`; `stridea` at least `lda*k` (nontrans column major),
`lda*m` (trans column major), `lda*m` (nontrans row major), `lda*k` (trans row major). `b` at least
`strideb * batch_size`; `strideb` at least `ldb*n`, `ldb*k`, `ldb*k`, `ldb*n` in the same four
cases. `c` at least `stridec * batch_size`; `stridec` at least `ldc*n` column major or `ldc*m` row
major. `lda` at least `m`/`k`/`k`/`m`; `ldb` at least `k`/`n`/`n`/`k`; `ldc` at least `m` column
major or `n` row major. Group API `a[i]`, `b[i]`, `c[i]` use the same table with `[i]` subscripts.
Output `c`: overwritten by `batch_size`/`total_batch_count` gemm operations of the form
`alpha * op(A) * op(B) + beta * C`. NOTE: "If `beta = 0`, matrices `C` do not need to be initialized
before calling `gemm_batch`." Example: `share/doc/mkl/examples/sycl/blas/source/gemm_batch.cpp`.

### gemm_bias

"Computes a matrix-matrix product using general integer matrices with bias"/"with biases/offsets".
`A_offset` is `m x k` with every element `ao`, `B_offset` is `k x n` with every element `bo`,
`C_offset` is `m x n` defined by the `co` buffer. Precisions (`Ta`,`Tb`):
(`std::uint8_t`,`std::uint8_t`), (`std::int8_t`,`std::uint8_t`), (`std::uint8_t`,`std::int8_t`),
(`std::int8_t`,`std::int8_t`); `C` is `std::int32_t`, scalars `float`.

```cpp
// Buffer (p225; row_major identical)
void gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               sycl::buffer<Ta,1> &a, std::int64_t lda, Ta ao, sycl::buffer<Tb,1> &b, std::int64_t ldb,
               Tb bo, float beta, sycl::buffer<std::int32_t,1> &c, std::int64_t ldc,
               sycl::buffer<std::int32_t,1> &co, compute_mode mode = compute_mode::unset)
// USM (p228; row_major identical)
sycl::event gemm_bias(sycl::queue &queue, oneapi::mkl::transpose transa,
                      oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                      std::int64_t n, std::int64_t k, oneapi::mkl::value_or_pointer<float> alpha,
                      const Ta *a, std::int64_t lda, Ta ao, const Tb *b, std::int64_t ldb, Tb bo,
                      oneapi::mkl::value_or_pointer<float> beta, std::int32_t *c, std::int64_t ldc,
                      const std::int32_t *co, compute_mode mode = compute_mode::unset,
                      const std::vector<sycl::event> &dependencies = {})
```

`offsetc` "Specifies the form of `C_offset` used in the matrix multiplication."; `ao`/`bo` "the
scalar offset value for matrix A/B."; `co` — "Buffer/Pointer holding the offset values for matrix C.
If `offset_type = offset::fix`, size of co array must be at least 1. If `offset_type =
offset::col`, size of co array must be at least `max(1,m)`. If `offset_type = offset::row`, size of
co array must be at least `max(1,n)`." (The text says `offset_type`; the parameter is `offsetc`.)
`a` at least `lda*k`/`lda*m`/`lda*m`/`lda*k`; `lda` at least `m`/`k`/`k`/`m`; `b` at least
`ldb*n`/`ldb*k`/`ldb*k`/`ldb*n`; `ldb` at least `k`/`n`/`n`/`k` (four-case order as in Overview);
`c` at least `ldc*n` column major or `ldc*m` row major; `ldc` at least `m` column major or `n` row
major. Output `c`: overwritten by
`alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`. NOTE: "If `beta = 0`,
matrix `C` does not need to be initialized before calling `gemm_bias`."

### gemmt

"Computes a matrix-matrix product with general matrices, but updates only the upper or lower
triangular part of the result matrix." `op(A)` is `n x k`, `op(B)` is `k x n`, `C` is `n x n`.

```cpp
// Buffer (p232; row_major identical)
void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, T alpha,
           sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &b, std::int64_t ldb, T beta,
           sycl::buffer<T,1> &c, std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM (p234; row_major identical)
sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                  oneapi::mkl::value_or_pointer<T> alpha, const T* a, std::int64_t lda, const T* b,
                  std::int64_t ldb, oneapi::mkl::value_or_pointer<T> beta, T* c, std::int64_t ldc,
                  compute_mode mode = compute_mode::unset,
                  const std::vector<sycl::event> &dependencies = {})
```

`upper_lower` "Specifies whether matrix C is upper or lower triangular."; `n` "Number of rows of
matrix op(A) and matrix C."; `k` "Number of columns of matrix op(A) and rows of matrix op(B).";
`a` at least `lda*k`/`lda*n`/`lda*n`/`lda*k`; `lda` at least `n`/`k`/`k`/`n`; `b` at least
`ldb*n`/`ldb*k`/`ldb*k`/`ldb*n`; `ldb` at least `k`/`n`/`n`/`k` (four-case order); `c` at least
`ldc*n` column major or `ldc*m` row major, `ldc` at least `m` column major or `n` row major (the
source `c`/`ldc` tables say "C is m x n", though `gemmt` has no `m` parameter — see Explicit gaps).
Output `c`:
"overwritten by upper or lower triangular part of `alpha * op(A)*op(B) + beta * C`." NOTE: "If
`beta = 0`, matrix `C` does not need to be initialized before calling `gemmt`."

### gemv_batch

"Computes a group of gemv operations… Each gemv operations perform a scalar-matrix-vector product
and add the result to a scalar-vector product": `Y = alpha * op(A) * X + beta * Y`. No `mode`
parameter in any variant; Buffer version supports only strided API; the USM strided variant has
**neither** `mode` nor `dependencies`.

```cpp
// Buffer, strided only (p238)
void gemv_batch(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                T alpha, sycl::buffer<T,1> &a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<T,1> &x, std::int64_t incx, std::int64_t stridex, T beta,
                sycl::buffer<T,1> &y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size)
// USM, group API (p240)
sycl::event gemv_batch(sycl::queue &queue, const oneapi::mkl::transpose *trans, const Ti *m, const Ti *n,
                       const T *alpha, const T **a, const Ti *lda, const T **x, const Ti *incx,
                       const T *beta, T **y, const Ti *incy, std::int64_t group_count,
                       const Ti *group_size, const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p242) — no mode, no dependencies
sycl::event gemv_batch(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                       oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                       std::int64_t stridea, const T *x, std::int64_t incx, std::int64_t stridex,
                       oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size)
```

`a` at least `stridea * batch_size`; `lda` "Must be positive and at least `m` if column major layout
or at least `n` if row major layout is used."; `stridea` at least zero; `x`/`y` at least
`stridex * batch_size` / `stridey * batch_size`; `incx`/`incy` "Must not be zero."; `stridex` at
least zero; `stridey` "Must be at least `(1 + (m - 1)*abs(incy))` if layout is column major or
`(1 + (n - 1)*abs(incy))` if row major layout is used."; `batch_size` "Number of gemv computations
to perform. Must be at least zero." Output `y`: overwritten by
`batch_size`/`total_batch_count` gemv operations of the form `alpha * op(A) * X + beta * Y`.
Example: `share/doc/mkl/examples/sycl/blas/source/gemv_batch_usm.cpp`.

### syrk_batch

"Computes a group of syrk operations… Each syrk operation performs a rank-k update with general
matrices": `C = alpha * op(A) * op(A)^T + beta * C`, `op(A)` is `n x k`, `C` is `n x n` symmetric.
Restriction (verbatim in the `trans` descriptions): "Conjugation is never performed even if
`trans = transpose::conjtrans`." Buffer version supports only strided API.

```cpp
// Buffer, strided only (p244)
void syrk_batch(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
                std::int64_t stridea, T beta, sycl::buffer<T,1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size, compute_mode mode = compute_mode::unset)
// USM, group API (p247)
sycl::event syrk_batch(sycl::queue &queue, const oneapi::mkl::uplo *upper_lower,
                       const oneapi::mkl::transpose *trans, const Ti *n, const Ti *k, const T *alpha,
                       const T **a, const Ti *lda, const T *beta, T **c, const Ti *ldc,
                       std::int64_t group_count, const Ti *group_size,
                       compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p249)
sycl::event syrk_batch(sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                       oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                       std::int64_t stridea, oneapi::mkl::value_or_pointer<T> beta, T *c,
                       std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                       compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
```

`n` "Number of rows and columns of matrices C. Must be at least zero."; `k` "Number of columns of
matrices op(A). Must be at least zero."; `a` at least `stridea * batch_size`; `lda` at least
`n`/`k`/`k`/`n`; `stridea` at least `lda*k`/`lda*n`/`lda*n`/`lda*k` (four-case order); `c` at least
`stridec * batch_size`; `ldc` "Must be positive and at least `n`."; `stridec` "Must be least
`ldc * n`." (sic); `batch_size` "Specifies the number of matrix multiply operations to perform."
(no "at least zero" sentence in this routine's text). Group: `c[i]` at least `ldc[i] * n[i]`.
Output `c`: overwritten by `batch_size`/`total_batch_count` syrk operations of the form
`alpha * op(A) * op(A)^T + beta * C`.

### trsm_batch

"Computes a group of trsm operations… Each trsm solves an equation of the form `op(A) * X = alpha *
B` or `X * op(A) = alpha * B`." `A` is `m x m` or `n x n` triangular; `B` and `X` are `m x n`
general. "On return, matrix B is overwritten by solution matrix X." No out-of-place variant; Buffer
version supports only strided API.

```cpp
// Buffer, strided only (p252)
void trsm_batch(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                T alpha, sycl::buffer<T,1> &a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<T,1> &b, std::int64_t ldb, std::int64_t strideb, std::int64_t batch_size,
                compute_mode mode = compute_mode::unset)
// USM, group API (p254)
sycl::event trsm_batch(sycl::queue &queue, const oneapi::mkl::side *left_right,
                       const oneapi::mkl::uplo *upper_lower, const oneapi::mkl::transpose *trans,
                       const oneapi::mkl::diag *unit_diag, const Ti *m, const Ti *n, const T *alpha,
                       const T **a, const Ti *lda, T **b, const Ti *ldb, std::int64_t group_count,
                       const Ti *group_size, compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API (p257)
sycl::event trsm_batch(sycl::queue &queue, oneapi::mkl::side left_right,
                       oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                       oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                       std::int64_t stridea, T *b, std::int64_t ldb, std::int64_t strideb,
                       std::int64_t batch_size, compute_mode mode = compute_mode::unset,
                       const std::vector<sycl::event> &dependencies = {})
```

`a` at least `stridea * batch_size` ("input matricees A", sic); `lda` "Must be at least `m` if
`left_right = side::left` or at least `n` if `left_right = side::right`. Must be positive."; `b` at
least `strideb * batch_size`; `ldb` at least `m` column major or `n` row major, positive;
`batch_size` "Specifies number of triangular linear systems to solve." Group: `lda[i]`/`ldb[i]`
bounded as above with `[i]`, "All entries must be positive." Output `b`: overwritten by
`batch_size`/`total_batch_count` solution matrices X. NOTE (verbatim, including the doubled
period): "If `alpha = 0`, matrices `B` are set to zero, and `A` and `B` do not need to be
initialized before calling `trsm_batch..`"

### omatcopy

"Computes an out-of-place scaled matrix transpose or copy operation using a general matrix":
`B <- alpha * op(A)`, `op(X)` one of `op(X) = X`, `op(X) = X^T`, `op(X) = X^H`; `A` is `m x n`;
"B is m x n matrix if op is non-transpose and an n x m matrix otherwise." No `mode` parameter.

```cpp
// Buffer (p259; both namespaces identical)
void omatcopy(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
              T alpha, sycl::buffer<T, 1> &a, std::int64_t lda, sycl::buffer<T, 1> &b, std::int64_t ldb);
// USM arrays (p260; both namespaces identical)
sycl::event omatcopy(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                     oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, T *b,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies = {});
```

`a` "Must have size at least `lda * n` for column-major and at least `lda * m` for row-major.";
`lda` at least `m` column major / `n` row major, positive; `ldb` positive and at least `m`/`n`
(nontrans/trans column major) or `n`/`m` (nontrans/trans row major). Output `b`: "overwritten by
`alpha * op(A)`" — size at least `ldb*n` (nontrans column major), `ldb*m` (transposed column
major), `ldb*m` (nontrans row major), `ldb*n` (transposed row major). Return: "Output event to wait
for to ensure computation is complete."

### imatcopy

"Computes an in-place scaled matrix transpose or copy operation using a general matrix":
`AB <- alpha * op(AB)`, `op(X)` as above; "AB is m x n on input." No `mode` parameter.

```cpp
// Buffer (p263; both namespaces identical)
void imatcopy(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
              T alpha, sycl::buffer<T, 1> &ab, std::int64_t lda, std::int64_t ldb);
// USM (p264; both namespaces identical)
sycl::event imatcopy(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                     oneapi::mkl::value_or_pointer<T> alpha, T *ab, std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies = {});
```

`ab` size: nontrans column major at least `max(lda, ldb) * n`; transposed column major at least
`max(lda, ldb) * max(m, n)`; nontrans row major at least `max(lda, ldb) * m`; transposed row major
at least `max(lda, ldb) * max(m, n)`. `lda` (input) at least `m` column major / `n` row major,
positive; `ldb` (output) positive and at least `m`/`n` (nontrans/trans column major) or `n`/`m`
(nontrans/trans row major). Output: `ab` "overwritten by `alpha * op(AB)`."

### omatadd

"Computes a sum of two general matrices, with optional transposes":
`C <- alpha * op(A) + beta * op(B)`; `op(A)` `m x n`, `op(B)` `m x n`, `C` `m x n`. No `mode`
parameter. Overlap: "In general, A, B, and C must not overlap in memory, with the exception of the
following in-place operations: A and C may point to the same memory if `op(A)` is non-transpose and
`lda = ldc`; B and C may point to the same memory if `op(B)` is non-transpose and `ldb = ldc`."

```cpp
// Buffer (p266; both namespaces identical)
void omatadd(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
             std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T, 1> &a, std::int64_t lda, T beta,
             sycl::buffer<T, 1> &b, std::int64_t ldb, sycl::buffer<T, 1> &c, std::int64_t ldc)
// USM (p268; both namespaces identical)
sycl::event omatadd(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                    std::int64_t m, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                    std::int64_t lda, oneapi::mkl::value_or_pointer<T> beta, const T *b, std::int64_t ldb,
                    T *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});
```

`a` — buffer: "If alpha is zero, this buffer is never accessed. Otherwise" at least `lda * n`
(nontrans column major), `lda * m` (trans column major), `lda * m` (nontrans row major), `lda * n`
(trans row major); USM: "If alpha is zero, a is never accessed and may be a null pointer." `b`
obeys the same table with `beta`, `ldb`, `transb`. `lda`/`ldb` at least `m`/`n` (nontrans/trans
column major) or `n`/`m` (nontrans/trans row major), positive; `ldc` at least `m` column major /
`n` row major, positive. Output `c`: "overwritten by `alpha * op(A) + beta * op(B)`"; size at least
`ldc * n` column major or `ldc * m` row major.

### omatcopy_batch

"Computes a group of out-of-place scaled matrix transpose or copy operations using general
matrices": `B = alpha * op(A)`, with `op(X)` one of `op(X) = X`, `op(X) = X'`,
`op(X) = conjg(X')` as printed in the batch sections. Buffer version supports only strided API.

```cpp
// Buffer, strided only (p272; both namespaces identical)
void omatcopy_batch(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, T alpha,
                    sycl::buffer<T, 1> &a, std::int64_t lda, std::int64_t stride_a,
                    sycl::buffer<T, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                    std::int64_t batch_size);
// USM, group API (p274). NOTE the parameter is named "groupsize".
sycl::event omatcopy_batch(sycl::queue &queue, const transpose *trans, const Ti *m, const Ti *n,
                           const T *alpha, const T **a, const Ti *lda, T **b, const Ti *ldb,
                           std::int64_t group_count, const Ti *groupsize,
                           const std::vector<sycl::event> &dependencies = {});
// USM, strided API (p276; both namespaces identical)
sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                           std::int64_t stride_a, T *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size, const std::vector<sycl::event> &dependencies = {});
```

`a` at least `stride_a*batch_size`; `lda` at least `m` column major / `n` row major, positive;
`stride_a` at least `lda*n` column major or `lda*m` row major; `ldb` positive and at least `m`/`n`
(nontrans/trans column major) or `n`/`m` (nontrans/trans row major); `stride_b` positive and at
least `ldb*n`/`ldb*m` (nontrans/trans column major) or `ldb*m`/`ldb*n` (nontrans/trans row major);
`batch_size` "Specifies the number of matrices to transpose or copy. Must be at least zero." Group:
`m[i]`/`n[i]` "must be at least zero"; group-`i` `A` array at least `lda[i] * n[i]` column major or
`lda[i]*m[i]` row major; `lda[i]`/`ldb[i]` bounded as above, positive; `group_count` "Must be at
least 0"; `group_size[i]`/`groupsize[i]` "must be at least 0". Output `b`: "overwritten by
`batch_size`/`total_batch_count` matrix transpose or copy operations of the form `alpha*op(A)`";
per-group `B` sizes: column major `ldb[i] * n[i]` if B is not transposed or `ldb[i]*m[i]` if
transposed; row major `ldb[i] * m[i]` if not transposed or `ldb[i]*n[i]` if transposed.

### imatcopy_batch

"Computes a group of in-place scaled matrix transpose or copy operations using general matrices":
`AB = alpha * op(AB)`, `op(X)` as in `omatcopy_batch`; "AB is a matrix to be transformed in place."
Buffer version supports only strided API.

```cpp
// Buffer, strided only (p278)
void imatcopy_batch(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, T alpha,
                    sycl::buffer<T, 1> &ab, std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                    std::int64_t batch_size);
// USM, group API (p280). NOTE the parameter is named "groupsize".
sycl::event imatcopy_batch(sycl::queue &queue, const transpose *trans, const Ti *m, const Ti *n,
                           const T *alpha, T **ab, const Ti *lda, const Ti *ldb,
                           std::int64_t group_count, const Ti *groupsize,
                           const std::vector<sycl::event> &dependencies = {});
// USM, strided API (p282). The source prints "namespace oneapi::mkl::blas::column_major" twice; the
// second, identical block is the row_major overload mislabelled in the extraction.
sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           oneapi::mkl::value_or_pointer<T> alpha, T *ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies = {});
```

`ab` at least `stride*batch_size`; `lda` (input) at least `m` column major / `n` row major,
positive; `ldb` (output) positive and at least `m`/`n` (nontrans/trans column major) or `n`/`m`
(nontrans/trans row major); `stride` "It must be at least `max(ldb,lda)*max(ka, kb)`, where: `ka` is
m if column major layout is used or n if row major layout is used; `kb` is n if column major layout
is used and AB is not transposed, or m otherwise"; `batch_size` at least zero. Group: `m[i]`/`n[i]`
at least zero; `lda[i]`/`ldb[i]` as above; `group_count` "Must be at least 0"; `group_size[i]`/
`groupsize[i]` "must be at least 0". Output `ab`: "overwritten by `batch_size`/`total_batch_count`
matrix transpose or copy operations of the form `alpha*op(AB)`." (The buffer output sentence says
"matrix multiply operations of the form `alpha*op(AB)`" — sic.)

### omatadd_batch

"Computes a group of out-of-place scaled matrix additions using general matrices":
`C = alpha * op(A) + beta * op(B)`, `op(X)` one of `op(X) = X`, `op(X) = X'`, `op(X) = conjg(X')`.
"The matrices are always in a strided format for omatadd_batch." No group API in either version.
Overlap: "In general, the a, b, and c buffers must not overlap in memory, with the exception of the
following in-place operations: a and c may point to the same memory if `op(A)` is non-transpose and
all the A matrices within a have the same parameters as all the respective C matrices within c; b
and c may point to the same memory if `op(B)` is non-transpose and all the B matrices within b have
the same parameters as all the respective C matrices within c."

```cpp
// Buffer, strided (p284; both namespaces identical)
void omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
                   T alpha, sycl::buffer<T, 1> &a, std::int64_t lda, std::int64_t stride_a, T beta,
                   sycl::buffer<T, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                   sycl::buffer<T, 1> &c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);
// USM, strided (p287; both namespaces identical)
sycl::event omatadd_batch(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                          std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                          std::int64_t lda, std::int64_t stride_a,
                          oneapi::mkl::value_or_pointer<T> beta, const T *b, std::int64_t ldb,
                          std::int64_t stride_b, T *c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event> &dependencies = {});
```

`a` — "If alpha is zero, a is never accessed and may be a null pointer. Otherwise it must have size
at least `stride_a*batch_size`."; `lda` positive and at least `m`/`n` (nontrans/trans column major)
or `n`/`m` (nontrans/trans row major); `stride_a` positive and at least `lda*n`/`lda*m`
(nontrans/trans column major) or `lda*m`/`lda*n` (nontrans/trans row major); `b`/`beta`/`ldb`/
`stride_b` obey the same table with `transb`; `ldc` at least `m` column major / `n` row major,
positive; `stride_c` at least `ldc*n` column major or `ldc*m` row major; `batch_size` "Specifies
the number of input and output matrices to add. Must be at least zero." Output `c`: "overwritten by
`batch_size` matrix addition operations of the form `alpha*op(A) + beta*op(B)`. Must have size at
least `stride_c*batch_size`."

## Formulas

From formula images (tag `pNNNN#i` = PDF page, index) and from operation loops that survived as
text where no image exists.

* **trmm** (p177#0, in-place `side::left`): `B <- alpha * op(A) * B`
* **trmm** (p178#0, in-place `side::right`): `B <- alpha * B * op(A)`
* **trmm** (p178#1, out-of-place `side::left`): `C <- alpha * op(A) * B + beta * C`
* **trmm** (p178#2, out-of-place `side::right`): `C <- alpha * B * op(A) + beta * C`
* **trsm** (p185#0, in-place `side::left`): `op(A) * X = alpha * B`
* **trsm** (p185#1, in-place `side::right`): `X * op(A) = alpha * B`
* **trsm** (p185#2, out-of-place `side::left`): `op(A) * X = alpha * B, C <- X + beta * C`
* **trsm** (p185#3, out-of-place `side::right`): `X * op(A) = alpha * B, C <- X + beta * C`
* **axpby** (p194#0): `y <- beta * y + alpha * x`
* **axpy_batch** (p197/p200 strided; text loop): `for i = 0 … batch_size – 1: X and Y are vectors at offset i * stridex and i * stridey in x and y; Y = alpha * X + Y`
* **axpy_batch** (p198 group; text loop): `idx = 0; for i = 0 … group_count – 1: for j = 0 … group_size – 1: X and Y are vectors at x[idx] and y[idx]; Y = alpha[i] * X + Y; idx = idx + 1`
* **copy_batch** (p202/p204 strided, p203 group; text loops): `Y = X` for the vector at offsets `i * stridex`, `i * stridey` (strided) or at `x[idx]`, `y[idx]` (group)
* **dgmm_batch** (p206/p210 strided, p208 group; text loops): `C = diag(X) * A` if `left_right[idx] == side::left`, else `C = A * diag(X)`
* **gemm_batch** (p213/p221 strided, p216 group; text loops): `C = alpha * op(A) * op(B) + beta * C` (group form with `alpha[i]`, `beta[i]`); `op(A)` is `m x k`, `op(B)` is `k x n`, `C` is `m x n`
* **gemm_bias** (no formula image extracted; from the Output Parameters text): `C <- alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`, where `A_offset` is `m x k` with every element `ao`, `B_offset` is `k x n` with every element `bo`, `C_offset` is defined by the `co` buffer
* **gemmt** (p231#0): `C <- alpha * op(A) * op(B) + beta * C`, applied to the "upper or lower triangular part" of `C` only
* **gemv_batch** (p237/p242 strided, p239 group; text loops): `Y = alpha * op(A) * X + beta * Y` (group form with `alpha[i]`, `beta[i]`)
* **syrk_batch** (p244/p249 strided, p246 group; text loops): `C = alpha * op(A) * op(A)^T + beta * C` (group form with `alpha[i]`, `beta[i]`); `op(A)` is `n x k`, `C` is `n x n`
* **trsm_batch** (p252/p256 strided, p254 group; text loops): `if (left_right == side::left) compute X such that op(A) * X = alpha * B; else compute X such that X * op(A) = alpha * B; B = X` (group form with `alpha[i]`)
* **omatcopy** (p259#0): `B <- alpha * op(A)`
* **imatcopy** (p262#0): `AB <- alpha * op(AB)`
* **omatadd** (p265#0): `C <- alpha * op(A) + beta * op(B)`
* **omatcopy_batch** (p272/p276 strided, p274 group; text loops): `B = alpha * op(A)` per batch item
* **imatcopy_batch** (p278/p282 strided, p280 group; text loops): `AB = alpha * op(AB)` per batch item
* **omatadd_batch** (p284/p287 strided; text loop): `C = alpha * op(A) + beta * op(B)` per batch item
* **total_batch_count** (p198#0, p203#0, p209#0, p217#0, p240#0, p247#0, p254#0, p274#0, p280#0 — nine distinct images, all identical): `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`
* **summation illustration** (p294#0, Numerical Reproducibility): `a_0 + a_1 + ... + a_n` — the example of a summation whose result depends on operation order because floating-point addition is not associative
* **Sparse-format images** (p302#0, p303#0, p304#0): CSR example matrices `A` from the Sparse BLAS section — a `3 x 3` matrix with `nnz = 5`; a `4 x 5` matrix with `nnz = 7` containing an empty row (`row_ptr` = `1 3 6 6 8`); a `4 x 5` matrix with `nnz = 10`. Headers: `nrows`, `ncols`, `nnz`, `index`, `row_ptr`, `col_ind`, `values`. Outside this chapter's domain.

## Conventions & Gotchas

* **Two namespaces, one name.** Always qualify. Column-major and row-major builds need *different*
  `lda`/`ldb`/`ldc` for the same mathematical matrices (see each routine's bounds above).
* **Verbatim identifier traps.** `trsm` Buffer overloads name the transpose argument `transa` in
  `column_major` but `trans` in `row_major` (USM uses `trans` in both). `axpby`'s USM dependency
  vector is typed `const std::vector<event>&`, not `sycl::event`. `omatcopy_batch` /
  `imatcopy_batch` group APIs name the group-size array `groupsize`, not `group_size`.
  `dgmm_batch` prints the row count as `std::inte64_t`. `gemm_bias`'s text refers to `offset_type`
  while the parameter is `offsetc`. `omatcopy_batch`, `imatcopy_batch`, and `omatadd_batch`
  signatures use unqualified `transpose` (`transa`/`transb`) rather than `oneapi::mkl::transpose`.
* **Buffer vs USM.** Buffer versions return `void`; USM versions return `sycl::event`
  ("Output event to wait on to ensure computation is complete", or "…wait for…" for
  `omatcopy`/`imatcopy`/`omatadd`).
* **Scalars.** Buffer versions take `T alpha`/`T beta` by value (`float` for `gemm_bias`); USM
  versions take `oneapi::mkl::value_or_pointer<T>` (or `...<float>` for `gemm_bias`).
* **Optional arguments.** Defaults are `compute_mode mode = compute_mode::unset` and
  `const std::vector<sycl::event> &dependencies = {}`; they may be omitted independently; `mode`
  precedes `dependencies` in USM. Routines with **no** `mode` parameter: `axpby`, `axpy_batch`,
  `copy_batch`, `dgmm_batch`, `gemv_batch`, `omatcopy`, `imatcopy`, `omatadd`, `omatcopy_batch`,
  `imatcopy_batch`, `omatadd_batch`. USM strided `gemv_batch` has no `dependencies` either.
* **No scratchpad.** No extracted page mentions a scratchpad buffer or scratchpad-size query for
  these routines; do not assume one exists.
* **Group API arithmetic.** `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`; `Ti` is
  `std::int64_t` or `std::int32_t`; pointer arrays have size `total_batch_count`; per-group scalar
  arrays (`n`, `m`, `k`, `alpha`, `beta`, `lda`, `ldb`, `ldc`, `incx`, `incy`, `group_size`) have
  size `group_count`.
* **Span overloads.** Only `gemm_batch` documents `sycl::span` group inputs. Span size 1 = reused
  for all computations; `group_count` = one value per group; `total_batch_count` = one per
  computation; for output matrices "size of the span must be the total batch size". The span
  overload takes `size_t group_count` and `const sycl::span<size_t> &group_sizes`.
* **Strides.** Element and matrix strides are "at least zero" unless stated; matrix strides must
  cover a full matrix (`stridea >= lda*k`, etc.). `incx`/`incy` "Must not be zero."
  `dgmm_batch`'s `stridec` must be at least `ldc*n` column major or `ldc*m` row major.
* **Size formulas.** Vector length `1 + (n - 1)*abs(inc)`; packed batch buffer `batch_size * stride`
  (for `dgmm_batch`'s `a`: `lda * k + stridea * (batch_size - 1)`).
* **Zero-scalar shortcuts.** `alpha = 0`: `trmm` sets `B` to zero; `trsm`/`trsm_batch` set `B` to
  zero; inputs need not be initialized. `beta = 0`: `gemm_batch`, `gemm_bias`, `gemmt` do not
  require `C` to be initialized. `omatadd`/`omatadd_batch`: a zero `alpha`/`beta` means the
  corresponding input is never accessed and (USM) may be a null pointer.
* **Overlap.** `omatadd`: `A`/`C` may alias if `op(A)` is non-transpose and `lda = ldc`; `B`/`C` if
  `op(B)` is non-transpose and `ldb = ldc`. `omatadd_batch`: `a`/`c` may alias if `op(A)` is
  non-transpose and the corresponding `A` and `C` matrices have the same parameters; likewise
  `b`/`c`. No other aliasing is documented as safe.
* **In-place vs out-of-place.** `trmm`/`trsm`: both. `trsm_batch`: in-place only (`B = X`).
  `imatcopy`/`imatcopy_batch`: in-place (`AB = alpha*op(AB)`).
  `omatcopy`/`omatcopy_batch`/`omatadd`/`omatadd_batch`: out-of-place. `dgmm_batch`, `gemm_batch`,
  `gemm_bias`, `gemmt`, `gemv_batch`, `syrk_batch`: output argument overwritten in place (`C`, or
  `Y` for `gemv_batch`).
* **Conjugation.** `syrk_batch`: "Conjugation is never performed even if
  `trans = transpose::conjtrans`." The batch extension text writes `op(X) = X'` or
  `op(X) = conjg(X')`; the non-batch routines use `transpose::nontrans` / `transpose::trans` /
  `transpose::conjtrans`.
* **Compute modes.** Alternate modes are opt-in; `prefer_alternate`/`force_alternate` are modifiers
  used together with alternate modes. `force_alternate`: "Never use the standard implementation; if
  no allowed alternate implementation is available, an exception will be thrown." Absent that, if no
  allowed mode is supported or expected to help, oneMKL falls back to the standard implementation.
* **Reproducibility.** Same inputs may give slightly different results run to run (floating-point
  summation is order-dependent); CNR mode gives bitwise reproducibility on CPU for all BLAS DPC++
  APIs and on GPU for level-3 routines and level-3 extensions.
* **Error handling.** Apart from `compute_mode::force_alternate` throwing, these pages define no
  error codes or exception types; parameter constraints are stated as "Must be…" requirements only.

## Explicit gaps

* **No Include Files blocks** for any routine in this chapter; only `#include <oneapi/mkl.hpp>`
  appears (Compute Modes examples).
* **`gemm_bias` formula image missing.** The p224 "operation is defined as:" line has no image in
  the extraction; the formula above comes from the Output Parameters sentence.
* **`compute_mode::unset` is not defined** in these pages; it appears only as the default value.
* **Ambiguous compute-mode routine list.** The source sentence reads "alternate implementations are
  available for gemm, gemmt, syrk, and syr2k…". `syr2k` is not part of this chapter and no
  compute-mode details for it appear here; whether the printed name is accurate is not stated.
* **`gemmt` size tables reference an undefined `m`.** The `c`/`ldc` tables say "C is m x n matrix"
  and require `ldc >= m` column major, but `gemmt` has no `m` parameter (its `C` is `n x n`); the
  intended bound is presumably `n`. The tables are reproduced as printed.
* **`imatcopy_batch` USM strided signature mislabelled.** The second namespace block repeats
  `oneapi::mkl::blas::column_major`; the row-major overload is not distinctly shown.
* **Unexplained omissions.** `trsm_batch` USM strided has `mode` + `dependencies`, but `gemv_batch`
  USM strided has neither; no rationale is given in the extracted text.
* **No scratchpad, error code, or device-restriction statements** for these routines beyond the
  coarse GPU/CPU notes on CNR and compute-mode availability.
* **Sparse BLAS is out of scope.** Pages 294-304 of the source (`sparse::matrix_handle_t`,
  CSR/CSC/COO/BSR formats, `sparse::*` routines, and the CSR example matrices in images p302-p304)
  belong to the separate Sparse BLAS domain; this chapter only records that the p294 summation
  image illustrates non-associative floating-point summation.
