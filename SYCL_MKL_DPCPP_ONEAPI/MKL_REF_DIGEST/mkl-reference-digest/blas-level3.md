# BLAS Level 3 Routines

BLAS Level 3 covers matrix-matrix operations: general, symmetric, Hermitian, and triangular matrix
products, rank-k / rank-2k updates, and triangular solves. Every routine is exposed in two layout
namespaces — `oneapi::mkl::blas::column_major` and `oneapi::mkl::blas::row_major` — and in two
memory forms: a **buffer** version taking `sycl::buffer<T,1>` arguments and returning `void`, and a
**USM** version taking raw pointers, `oneapi::mkl::value_or_pointer<T>` scalars, a `dependencies`
event list, and returning a `sycl::event`. The chapter gives the routine-group table, one subsection
per Level 3 routine (`gemm`, `hemm`, `herk`, `her2k`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`), the
Level-2 boundary routines carried on the same pages (`trmv`, `trsv`), and the BLAS-like extension
routines that begin at the end of the assigned range.

> Signatures are reproduced token-for-token from the reference — parameter names, types, order, and
> default arguments. Only line wrapping is compacted for density (the reference prints one parameter
> per line); where a fence says `row_major` differs from `column_major` only by namespace, that is
> literally so.

## Overview

**Operation shapes.**

- product + accumulate: `C <- alpha * op(A) * op(B) + beta * C` — `gemm`, `hemm`, `symm`
- rank-k / rank-2k update: `C <- alpha * op(A) * op(A)^H + beta * C` and variants — `herk`,
  `her2k`, `syrk`, `syr2k`
- triangular product / solve, each with an **in-place** and an **out-of-place** API — `trmm`, `trsm`

**Routine groups and data types** (verbatim from the reference table, p144):

| Routine Group | Data Types | Description |
|---|---|---|
| gemm | std::int8_t, oneapi::mkl::bfloat16, sycl::half, float, double, std::complex<float>, std::complex<double>, mixed | Computes a matrix-matrix product with general matrices. |
| hemm | std::complex<float>, std::complex<double> | Computes a matrix-matrix product where one input matrix is Hermitian and one is general. |
| herk | std::complex<float>, std::complex<double> | Performs a Hermitian rank-k update. |
| her2k | std::complex<float>, std::complex<double> | Performs a Hermitian rank-2k update. |
| symm | float, double, std::complex<float>, std::complex<double> | Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general. |
| syrk | float, double, std::complex<float>, std::complex<double> | Performs a symmetric rank-k update. |
| syr2k | float, double, std::complex<float>, std::complex<double> | Performs a symmetric rank-2k update. |
| trmm | float, double, std::complex<float>, std::complex<double> | Computes a matrix-matrix product where one input matrix is triangular and one input matrix is general. |
| trsm | float, double, std::complex<float>, std::complex<double> | Solves a triangular matrix equation (forward or backward solve). |

The reference adds: "The BLAS functions are blocked where possible to restructure the code in a way
that increases the localization of data reference, enhances cache memory use, and reduces the
dependency on the memory bus. The code is distributed across the processors to maximize
parallelism." No device/host restriction is stated for Level 3; the `queue` selects the device.

**Types and enums named in the signatures.** `oneapi::mkl::uplo upper_lower`;
`oneapi::mkl::transpose trans` (values named inline: `transpose::nontrans`, `transpose::trans`,
`transpose::conjtrans`); `oneapi::mkl::diag unit_diag`; `oneapi::mkl::side left_right` (values
`side::left`, `side::right`); `compute_mode mode = compute_mode::unset`. Full enumerator lists are
deferred to a "Data Types" section ("See Data Types for more details") and the scalar wrapper to a
"Scalar Arguments" section; neither is on the assigned pages.

**Memory models and optional parameters.** Buffer: `void f(sycl::queue&, ..., sycl::buffer<T,1>&
...)` — no `dependencies`, no return event; results go into the output buffer. USM:
`sycl::event f(sycl::queue&, ..., T* ..., const std::vector<sycl::event> &dependencies = {})` —
`alpha`/`beta` become `oneapi::mkl::value_or_pointer<T>` (or `value_or_pointer<Treal>` where the
scalar type differs from the matrix type, as in `herk`/`her2k`). The reference states: "mode and
dependencies may be omitted independently; it is not necessary to specify mode in order to provide
dependencies." USM returns "Output event to wait on to ensure computation is complete."

**Signature asymmetries.** For every routine except the buffer version of `trsm`, the
`column_major` and `row_major` overloads have identical parameter lists (only the namespace
differs). The buffer `trsm` overloads name the transpose parameter `transa` in `column_major` but
`trans` in `row_major`; the USM `trsm` overloads use `trans` in both. In the extracted text every
buffer version except `axpby` ends with a closing brace and no trailing semicolon, while `axpby`
(buffer and USM) prints a trailing semicolon and an extra indented closing brace.

## Routines

### trmv

Computes a matrix-vector product using a triangular matrix. *(Level 2 operation carried on the
assigned pages; its full entry belongs to the Level 2 chapter.)* Precisions: `float`, `double`,
`std::complex<float>`, `std::complex<double>`. `op(A)` is `A`, `A^T`, or `A^H`; `A` is `n x n` unit
or non-unit, upper or lower triangular; `x` has length `n`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const T *a, std::int64_t lda, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

Both are declared in `namespace oneapi::mkl::blas::column_major` and identically in
`namespace oneapi::mkl::blas::row_major`. Inputs: `n` at least zero; `a` size `>= lda * n`; `lda`
"Must be at least n and positive"; `x` size `>= (1 + (n - 1)*abs(incx))`; `incx` must not be zero.
Output: `x` holds the updated vector.

### trsv

Solves a system of linear equations whose coefficients are in a triangular matrix. *(Level 2
operation carried on the assigned pages.)* Precisions: `float`, `double`, `std::complex<float>`,
`std::complex<double>`. `op(A)` is `A`, `A^T`, or `A^H`; `A` is `n x n` unit or non-unit, upper or
lower triangular; `b` and `x` are vectors of length `n`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const T *a, std::int64_t lda, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

Both are declared in `namespace oneapi::mkl::blas::column_major` and identically in
`namespace oneapi::mkl::blas::row_major`. Input `x` is "the n-element right-hand side vector b",
size `>= (1 + (n - 1)*abs(incx))`. Output `x` is the solution vector. `lda >= n` and positive;
`incx` must not be zero.

### gemm

Computes a matrix-matrix product with general matrices: `C <- alpha * op(A) * op(B) + beta * C`,
where `op(X)` is `X`, `X^T`, or `X^H`; `op(A)` is `m x k`, `op(B)` is `k x n`, `C` is `m x n`.

Precision combinations (`Ta` = A matrix, `Tb` = B matrix, `Tc` = C matrix, `Ts` = alpha/beta),
verbatim:

| Ta | Tb | Tc | Ts |
|---|---|---|---|
| sycl::half | sycl::half | sycl::half | sycl::half |
| sycl::half | sycl::half | float | float |
| oneapi::mkl::bfloat16 | oneapi::mkl::bfloat16 | oneapi::mkl::bfloat16 | float |
| oneapi::mkl::bfloat16 | oneapi::mkl::bfloat16 | float | float |
| std::int8_t | std::int8_t | std::int32_t | float |
| std::int8_t | std::int8_t | float | float |
| float | float | float | float |
| double | double | double | double |
| std::complex<float> | std::complex<float> | std::complex<float> | std::complex<float> |
| std::complex<double> | std::complex<double> | std::complex<double> | std::complex<double> |

```cpp
// Buffer (column_major shown; row_major overload is identical)
void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, Ts alpha, sycl::buffer<Ta,1> &a,
          std::int64_t lda, sycl::buffer<Tb,1> &b, std::int64_t ldb, Ts beta,
          sycl::buffer<Tc,1> &c, std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM
sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k,
                 oneapi::mkl::value_or_pointer<Ts> alpha, const Ta *a, std::int64_t lda,
                 const Tb *b, std::int64_t ldb, oneapi::mkl::value_or_pointer<Ts> beta, Tc *c,
                 std::int64_t ldc, compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

Both are declared in `namespace oneapi::mkl::blas::column_major` and identically in
`namespace oneapi::mkl::blas::row_major`. Inputs: `transa` specifies `op(A)`; `transb` specifies
`op(B)`; `m` = rows of `op(A)` and `C`, at least zero; `n` = columns of `op(B)` and `C`, at least
zero; `k` = columns of `op(A)` and rows of `op(B)`, at least zero; `alpha` = scaling factor for
matrix-matrix product; `beta` = scaling factor for matrix C; `mode` = "Optional. Compute mode
settings. See Compute Modes for more details."

`a` size / `lda` bound — nontrans vs (`transa = transpose::trans or trans = transpose::conjtrans`):

- column major: A is `m x k`, `a >= lda * k`, `lda >= m` | A is `k x m`, `a >= lda * m`, `lda >= k`
- row major: A is `m x k`, `a >= lda * m`, `lda >= k` | A is `k x m`, `a >= lda * k`, `lda >= m`

`b` size / `ldb` bound — nontrans vs (`transb = transpose::trans or trans = transpose::conjtrans`):

- column major: B is `k x n`, `b >= ldb * n`, `ldb >= k` | B is `n x k`, `b >= ldb * k`, `ldb >= n`
- row major: B is `k x n`, `b >= ldb * k`, `ldb >= n` | B is `n x k`, `b >= ldb * n`, `ldb >= k`

`c` / `ldc`: column major — `C` is `m x n`, size `>= ldc * n`, `ldc >= m`; row major — size
`>= ldc * m`, `ldc >= n`; `ldc` must be positive. All leading dimensions must be positive.

Output: `c` overwritten by `alpha * op(A)*op(B) + beta * C`. NOTE (verbatim): "If beta = 0, matrix C
does not need to be initialized before calling gemm." Examples named by the text: buffer version at
`share/doc/mkl/examples/sycl/blas/source/gemm.cpp`, USM version at
`share/doc/mkl/examples/sycl/blas/source/gemm_usm.cpp`.

### hemm

Computes a matrix-matrix product where one input matrix is hermitian and one is general.
`left_right` determines whether the hermitian matrix, A, is on the left (`side::left`) or right
(`side::right`). Precisions (`T`): `std::complex<float>`, `std::complex<double>`. `alpha` and `beta`
are scalars; `A` is either `m x m` or `n x n` hermitian; `B` and `C` are `m x n`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c,
          std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM
sycl::event hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                 const T *a, std::int64_t lda, const T *b, std::int64_t ldb,
                 oneapi::mkl::value_or_pointer<T> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

`m` = rows of B and C (at least zero); `n` = columns of B and C (at least zero). `a` size
`>= lda * m` if `left_right = side::left` or `lda * n` if `side::right`; `lda` at least `m` (left)
or `n` (right), positive. `b` size `>= ldb * n` (column major) or `>= ldb * m` (row major); `ldb` at
least `m` (column major) or `n` (row major), positive. `c` size `>= ldc * n` (column major) or
`>= ldc * m` (row major); `ldc` at least `m` (column major) or `n` (row major), positive. For `hemm`
(and `symm`) `upper_lower` is documented as "whether matrix A is upper or lower triangular" — it
selects the stored triangle of the symmetric/Hermitian matrix. Output: `c` overwritten by
`alpha * A * B + beta * C` if `left_right = side::left` or `alpha * B * A + beta * C` if
`side::right`. NOTE: "If beta = 0, matrix C does not need to be initialized before calling hemm."

### her2k

Performs a hermitian rank-2k update of an `n x n` hermitian matrix C by general matrices A and B.
Precisions: `T` ∈ {`std::complex<float>`, `std::complex<double>`}, `Treal` ∈ {`float`, `double`}.
`trans` supports only `transpose::nontrans` and `transpose::conjtrans`; `alpha` is a complex scalar,
`beta` a real scalar; the inner dimension of both matrix multiplications is `k`. Shapes: with
`nontrans`, A is `n x k` and B is `k x n`; with `conjtrans`, A is `k x n` and B is `n x k`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
           sycl::buffer<T,1> &b, std::int64_t ldb, Treal beta, sycl::buffer<T,1> &c,
           std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM
sycl::event her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                  oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                  const T *b, std::int64_t ldb, oneapi::mkl::value_or_pointer<Treal> beta, T *c,
                  std::int64_t ldc, compute_mode mode = compute_mode::unset,
                  const std::vector<sycl::event> &dependencies = {})
```

`upper_lower` = whether matrix C is upper or lower triangular; `n` = rows and columns of C (at least
zero); `k` = inner dimension of the matrix multiplications (at least zero); `c` size `>= ldc * n`;
`ldc` "positive and at least n".

- a / lda — column major: A `n x k`, `a >= lda * k`, `lda >= n` | A `k x n`, `a >= lda * n`,
  `lda >= k`; row major: A `n x k`, `a >= lda * n`, `lda >= k` | A `k x n`, `a >= lda * k`,
  `lda >= n`
- b / ldb — column major: B `k x n`, `b >= ldb * n`, `ldb >= k` | B `n x k`, `b >= ldb * k`,
  `ldb >= n`; row major: B `k x n`, `b >= ldb * k`, `ldb >= n` | B `n x k`, `b >= ldb * n`,
  `ldb >= k`
- all leading dimensions positive. Output: `c` overwritten by the updated C matrix.

### herk

Performs a hermitian rank-k update of an `n x n` hermitian matrix C by a general matrix A. `op(X)`
is `X` or `X^H`; `alpha` and `beta` are real scalars; `op(A)` is `n x k` general. Precisions: `T` ∈
{`std::complex<float>`, `std::complex<double>`}, `Treal` ∈ {`float`, `double`}. `trans` supports only
`transpose::nontrans` and `transpose::conjtrans`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, Treal alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          Treal beta, sycl::buffer<T,1> &c, std::int64_t ldc,
          compute_mode mode = compute_mode::unset)
// USM
sycl::event herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                 oneapi::mkl::value_or_pointer<Treal> alpha, const T *a, std::int64_t lda,
                 oneapi::mkl::value_or_pointer<Treal> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

`upper_lower` = whether matrix C is upper or lower triangular; `n` = rows and columns of C (at least
zero); `k` = "Number of columns of matrix op(A)" (at least zero); `c` size `>= ldc * n`; `ldc`
positive and at least `n`. `a`/`lda` (nontrans vs conjtrans): column major — A `n x k`,
`a >= lda * k`, `lda >= n` | A `k x n`, `a >= lda * n`, `lda >= k`; row major — A `n x k`,
`a >= lda * n`, `lda >= k` | A `k x n`, `a >= lda * k`, `lda >= n`; leading dimensions positive.

Output: `c` overwritten by `alpha * op(A) * op(A)^H + beta * C`. **The imaginary parts of the
diagonal elements are set to zero.**

### symm

Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.
`left_right` determines whether the symmetric matrix A is on the left (`side::left`) or on the right
(`side::right`). Precisions (`T`): `float`, `double`, `std::complex<float>`,
`std::complex<double>`. `A` is either `m x m` or `n x n` symmetric; B and C are `m x n`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c,
          std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM
sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                 const T *a, std::int64_t lda, const T *b, std::int64_t ldb,
                 oneapi::mkl::value_or_pointer<T> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

`m` = rows of B and C (at least zero); `n` = columns of B and C (at least zero). Size and
leading-dimension rules are identical to `hemm`: `a >= lda * m` (left) / `lda * n` (right), `lda` at
least `m` (left) / `n` (right) and positive; `b >= ldb * n` (column major) / `ldb * m` (row major)
with `ldb` at least `m` (column major) / `n` (row major); `c >= ldc * n` (column major) /
`ldc * m` (row major) with `ldc` at least `m` (column major) / `n` (row major); all leading
dimensions positive. Output: `c` overwritten by `alpha * A * B + beta * C` if
`left_right = side::left` or `alpha * B * A + beta * C` if `left_right = side::right`. NOTE: "If
beta = 0, matrix C does not need to be initialized before calling symm."

### syr2k

Performs a symmetric rank-2k update of an `n x n` symmetric matrix C by general matrices A and B.
**"Conjugation is never performed even if trans = transpose::conjtrans."** Precisions (`T`):
`float`, `double`, `std::complex<float>`, `std::complex<double>`. `alpha` and `beta` are scalars; C
is symmetric; the inner dimension of both matrix multiplications is `k`. Shapes: `nontrans` → A and
B are `n x k`; `trans` or `conjtrans` → A and B are `k x n`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
           sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c,
           std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM
sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                  oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                  const T *b, std::int64_t ldb, oneapi::mkl::value_or_pointer<T> beta, T *c,
                  std::int64_t ldc, compute_mode mode = compute_mode::unset,
                  const std::vector<sycl::event> &dependencies = {})
```

`upper_lower` = whether matrix C is upper or lower triangular; `n` = rows and columns of C (at least
zero); `k` = inner dimension of the matrix multiplications (at least zero). `alpha` is described as
"Complex scaling factor for the rank-2k update" and `beta` as "Real scaling factor for matrix C",
though both are typed `T` in the signature.

- a / lda — column major: A `n x k`, `a >= lda * k`, `lda >= n` | A `k x n`, `a >= lda * n`,
  `lda >= k`; row major: A `n x k`, `a >= lda * n`, `lda >= k` | A `k x n`, `a >= lda * k`,
  `lda >= n`
- b / ldb — column major: B `n x k`, `b >= ldb * k`, `ldb >= n` | B `k x n`, `b >= ldb * n`,
  `ldb >= k`; row major: B `n x k`, `b >= ldb * n`, `ldb >= k` | B `k x n`, `b >= ldb * k`,
  `ldb >= n`
- `c` size `>= ldc * n`; `ldc` positive and at least `n`; all leading dimensions positive. Output:
  `c` overwritten by the updated C matrix.

### syrk

Performs a symmetric rank-k update of an `n x n` symmetric matrix C by a general matrix A. `op(X)`
is `X` or `X^T`; `alpha` and `beta` are scalars; `op(A)` is `n x k` general. **"Conjugation is never
performed even if trans = transpose::conjtrans."** Precisions (`T`): `float`, `double`,
`std::complex<float>`, `std::complex<double>`.

```cpp
// Buffer (column_major shown; row_major overload is identical)
void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          T beta, sycl::buffer<T,1> &c, std::int64_t ldc,
          compute_mode mode = compute_mode::unset)
// USM
sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda,
                 oneapi::mkl::value_or_pointer<T> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

`upper_lower` = whether matrix C is upper or lower triangular; `n` = rows and columns of C (at least
zero); `k` = "Number of columns of matrix op(A)" (at least zero). `a`/`lda` use the same table as
`syr2k` with the `trans`/`conjtrans` cases unified: nontrans — A `n x k`, `a >= lda * k` (column
major) / `>= lda * n` (row major), `lda >= n` (column major) / `>= k` (row major); transposed — A
`k x n`, `a >= lda * n` (column major) / `>= lda * k` (row major), `lda >= k` (column major) /
`>= n` (row major). `c` size `>= ldc * n`; `ldc` positive and at least `n`. Output: `c` overwritten
by `alpha * op(A) * op(A)^T + beta * C`.

### trmm

Computes a matrix-matrix product where one input matrix is triangular and other matrix is general.
`left_right` determines whether the triangular matrix, A, is on the left (`side::left`) or right
(`side::right`). There are two operations available, an in-place operation and an out-of-place
operation. `op(A)` is `A`, `A^T`, or `A^H`; `A` is either `m x m` or `n x n` triangular; B is `m x n`
general; C is `m x n` general. Precisions (`T`): `float`, `double`, `std::complex<float>`,
`std::complex<double>`.

```cpp
// Buffer, in-place (column_major shown; row_major overload is identical)
void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, compute_mode mode = compute_mode::unset)
// Buffer, out-of-place
void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c,
          std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM, in-place
sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                 std::int64_t lda, T *b, std::int64_t ldb,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
// USM, out-of-place
sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                 std::int64_t lda, const T *b, std::int64_t ldb,
                 oneapi::mkl::value_or_pointer<T> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

All four are declared in `namespace oneapi::mkl::blas::column_major` and identically in
`namespace oneapi::mkl::blas::row_major`. Parameters (shared by in-place and out-of-place): queue —
"The queue where the routine will be executed" (trmm/trsm say "will be", other routines say "should
be"); `left_right`, `upper_lower`, `trans`, `unit_diag` per Data Types; `m` = rows of matrix B, at
least zero; `n` = columns of matrix B, at least zero; `alpha` = scaling factor for matrix-matrix
product. Out-of-place also documents `beta` = scaling factor for matrix C.

- `a` size `>= lda * m` (`side::left`) or `lda * n` (`side::right`); `lda` at least `m` (left) or
  `n` (right), positive.
- `b` size `>= ldb * n` (column major) or `ldb * m` (row major); `ldb` at least `m` (column major)
  or `n` (row major), positive.
- `c` (out-of-place) size `>= ldc * n` (column major) or `ldc * m` (row major); `ldc` at least `m`
  (column major) or `n` (row major), positive.

Outputs: in-place — `b` overwritten by `alpha * op(A) * B` if `left_right = side::left` or
`alpha * B * op(A)` if `left_right = side::right`. NOTE: "If alpha = 0, matrix B is set to zero, and
A and B do not need to be initialized at entry." Out-of-place — `c` overwritten by
`alpha * op(A) * B + beta * C` if `side::left` or `alpha * B * op(A) + beta * C` if `side::right`.

### trsm

Solves a triangular matrix equation (forward or backward solve). `left_right` determines whether the
triangular matrix, A, is on the left (`side::left`) or right (`side::right`). `op(A)` is `A`,
`A^T`, or `A^H`; `A` is either `m x m` or `n x n` triangular; B, X, and C are `m x n` general
matrices. Precisions (`T`): `float`, `double`, `std::complex<float>`, `std::complex<double>`. "For
the in-place operation, the matrix B is overwritten by solution matrix X, while for the out-of-place
operation, B remains untouched and the solution is added to a scaled C matrix."

```cpp
// Buffer, in-place; column_major. The row_major overload is identical except that the parameter
// named `transa` here is named `trans` there, and the namespace is oneapi::mkl::blas::row_major.
void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, compute_mode mode = compute_mode::unset)
// Buffer, out-of-place; column_major (row_major: same, `trans` in place of `transa`)
void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &b, std::int64_t ldb, T beta, sycl::buffer<T,1> &c,
          std::int64_t ldc, compute_mode mode = compute_mode::unset)
// USM, in-place; column_major (row_major identical; both use `trans`)
sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                 std::int64_t lda, T *b, std::int64_t ldb,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
// USM, out-of-place; column_major (row_major identical)
sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                 std::int64_t lda, const T *b, std::int64_t ldb,
                 oneapi::mkl::value_or_pointer<T> beta, T *c, std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {})
```

Parameters match `trmm` in every shape rule, with `alpha` described as "Scaling factor for the
solution" rather than for the product: `m` = rows of B (at least zero); `n` = columns of B (at least
zero); `a >= lda * m` (left) / `lda * n` (right), `lda` at least `m` (left) / `n` (right), positive;
`b >= ldb * n` (column major) / `ldb * m` (row major), `ldb` at least `m` (column major) / `n` (row
major), positive; out-of-place `beta` scales C, `c >= ldc * n` (column major) / `ldc * m` (row
major), `ldc` at least `m` (column major) / `n` (row major), positive.

Outputs: in-place — `b` overwritten by solution matrix X. NOTE: "If alpha = 0, matrix B is set to
zero, and A and B do not need to be initialized before calling trsm." Out-of-place — `c` overwritten
by solution matrix `X + beta * C`.

### axpby

Computes a vector-scalar product added to a scaled-vector: two scalar-vector products added. `x`
and `y` are vectors of `n` elements; `alpha` and `beta` are scalars. Precisions (`T`): `float`,
`double`, `std::complex<float>`, `std::complex<double>`. First of the reference's BLAS-like
extension routines (that table begins on p193).

```cpp
// Buffer (column_major shown; row_major overload is identical) — source prints the trailing
// semicolon and brace exactly as shown
void axpby(sycl::queue &queue, std::int64_t n, T alpha, sycl::buffer<T,1> &x,
           std::int64_t incx, T beta, sycl::buffer<T,1> &y, std::int64_t incy);
// USM — note the dependency element type is written `event`, not `sycl::event`
sycl::event axpby(sycl::queue &queue, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                  const T *x, std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y,
                  std::int64_t incy, const std::vector<event> &dependencies = {});
```

Both are declared in `namespace oneapi::mkl::blas::column_major` and identically in
`namespace oneapi::mkl::blas::row_major`. Inputs: `n` = number of elements in vectors x and y;
`alpha`; `x` size at least `1 + (n - 1)*abs(incx)`; `incx` = stride between two consecutive
elements of vector x; `beta`; `y` size at least `1 + (n - 1)*abs(incy)`; `incy` = stride between two
consecutive elements of vector y. No non-zero stride requirement is stated for axpby. Output: `y`
holds the updated vector.

### axpy_batch

Computes a group of axpy operations — "batched versions of axpy, performing multiple axpy
operations in a single call. Each axpy operation adds a scalar-vector product to a vector."
Precisions (`T`): `float`, `double`, `std::complex<float>`, `std::complex<double>`. **The buffer
version supports only the strided API; the USM version supports both group and strided APIs.** For
the group API, "The type Ti of integer pointers in the group API may be either `std::int64_t` or
`std::int32_t`." Strided semantics: `for i = 0 … batch_size – 1 { X and Y are vectors at offset
i * stridex and i * stridey in x and y; Y = alpha * X + Y }`.

```cpp
// Buffer (strided only); column_major (row_major identical)
void axpy_batch(sycl::queue &queue, std::int64_t n, T alpha, sycl::buffer<T, 1> &x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<T, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size)
// USM, group API
sycl::event axpy_batch(sycl::queue &queue, const Ti *n, const T *alpha, const T **x,
                       const Ti *incx, T **y, const Ti *incy, std::int64_t group_count,
                       const Ti *group_size, const std::vector<sycl::event> &dependencies = {})
// USM, strided API
sycl::event axpy_batch(sycl::queue &queue, std::int64_t n,
                       oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx,
                       std::int64_t stridex, T *y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies = {})
```

Strided parameters: `n` = number of elements in vectors X and Y; `alpha`; `x` size
`>= batch_size * stridex`; `incx` = stride between two consecutive elements of X vectors, must not
be zero; `stridex` = stride between two consecutive X vectors, at least zero; `y` size
`>= batch_size * stridey`; `incy` must not be zero; `stridey` at least `(1 + (n-1)*abs(incy))`;
`batch_size` = number of axpy computations to perform, at least zero. Output: `y` overwritten by
`batch_size` axpy operations of the form `alpha * X + Y`.

Group semantics: `idx = 0; for i = 0 … group_count – 1 { for j = 0 … group_size – 1 { X and Y are
vectors at x[idx] and y[idx]; Y = alpha[i] * X + Y; idx = idx + 1 } }`. Group parameters: `n` =
array of `group_count` integers, `n[i]` = elements in X and Y for every vector in group i; `alpha` =
array of `group_count` scalar elements; `x` = array of pointers to input vectors X with size
`total_batch_count`, group-i X array at least `(1 + (n[i] – 1)*abs(incx[i]))`; `incx` = array of
`group_count` integers, must not be zero; `y` = array of pointers to input/output vectors Y with
size `total_batch_count`, group-i Y array at least `(1 + (n[i] – 1)*abs(incy[i]))`; `incy` = array
of `group_count` integers, must not be zero; `group_count` at least zero; `group_size` = array of
`group_count` integers, each at least zero. Output: `y` overwritten by `total_batch_count` axpy
operations of the form `alpha * X + Y`.

### copy_batch

Computes a group of copy operations — "batched versions of copy, performing multiple copy operations
in a single call. Each copy operation copies one vector to another." Precisions (`T`): `float`,
`double`, `std::complex<float>`, `std::complex<double>`. **Buffer version supports only the strided
API; USM version supports both group and strided APIs.** `Ti` may be `std::int64_t` or
`std::int32_t`. Strided semantics: `for i = 0 … batch_size – 1 { X and Y are vectors at offset
i * stridex and i * stridey in x and y; Y = X }`.

```cpp
// Buffer (strided only); column_major (row_major identical)
void copy_batch(sycl::queue &queue, std::int64_t n, sycl::buffer<T, 1> &x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<T, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size)
// USM, group API
sycl::event copy_batch(sycl::queue &queue, const Ti *n, const T **x, const Ti *incx, T **y,
                       const Ti *incy, std::int64_t group_count, const Ti *group_size,
                       const std::vector<sycl::event> &dependencies = {})
// USM, strided API
sycl::event copy_batch(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx,
                       std::int64_t stridex, T *y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event> &dependencies = {})
```

Parameter meanings and restrictions are the same as `axpy_batch` minus `alpha`/`beta`; the group
operation is `Y = X` instead of `Y = alpha[i] * X + Y`. Output: `y` overwritten by `batch_size`
(strided) / `total_batch_count` (group) copy operations. Example named by the text: USM
`copy_batch` at `share/doc/mkl/examples/sycl/blas/source/copy_batch_usm.cpp`.

### Further BLAS-like extension routines (names, types, descriptions only)

Listed in the extension table on the assigned pages; their syntax lies beyond the range. All are
float, double, std::complex<float>, std::complex<double> except where noted: `dgmm_batch` (group of
diagonal matrix-matrix product); `gemm_batch` (also std::int8_t, oneapi::mkl::bfloat16, sycl::half,
and mixed; groups of matrix-matrix product with general matrices); `gemm_bias` (mixed std::int8_t,
std::uint8_t, and std::int32_t; matrix-matrix product with general matrices and mixed precision);
`gemmt` (matrix-matrix product with general matrices, but updates only the upper/lower triangular
part of the output matrix); `gemv_batch` (group of matrix-vector product using general matrices);
`syrk_batch` (rank-k updates on a group of symmetric matrices by a group of general matrices);
`trsm_batch` (triangular matrix equation for a group of matrices); `omatcopy` (out-of-place matrix
copy or transposition); `imatcopy` (in-place matrix copy or transposition); `omatadd` (sum of two
general matrices, with optional transposes); `omatcopy_batch` (groups of out-of-place matrix copies
or transpositions); `imatcopy_batch` (groups of in-place matrix copies or transpositions);
`omatadd_batch` (groups of matrix additions).

## Formulas

Transcribed from the formula images extracted from the PDF (formulas are images in the source and do
not survive text extraction). Each is labelled with its routine and PDF page.

- **trmv (p138):** `x <- op(A) * x`
- **trsv (p141):** `op(A) * x = b`
- **gemm (p145):** `C <- alpha * op(A) * op(B) + beta * C`
- **hemm (p151):** `if left_right = side::left:  C <- alpha * A * B + beta * C`
- **hemm (p151):** `if left_right = side::right: C <- alpha * B * A + beta * C`
- **her2k (p155), `trans = transpose::nontrans`:** `C <- alpha * A * B^H + conj(alpha) * B * A^H + beta * C`
  (A is n x k, B is k x n)
- **her2k (p155), `trans = transpose::conjtrans`:** `C <- alpha * A^H * B + conj(alpha) * B^H * A + beta * C`
  (A is k x n, B is n x k) — the `p0155` crop holds only the first expression; this one was recovered
  from the p155 page image.
- **herk (p160):** `C <- alpha * op(A) * op(A)^H + beta * C`
- **symm (p164):** `if left_right = side::left:  C <- alpha * A * B + beta * C`
- **symm (p164):** `if left_right = side::right: C <- alpha * B * A + beta * C`
- **syr2k (p168):** `if trans = transpose::nontrans: C <- alpha * (A * B^T + B * A^T) + beta * C`
- **syr2k (p168):** `if trans = transpose::trans:   C <- alpha * (A^T * B + B^T * A) + beta * C`
- **syrk (p173):** `C <- alpha * op(A) * op(A)^T + beta * C`
- **trmm (p177), in-place, left_right = side::left:** `B <- alpha * op(A) * B`
- **trmm (p178), in-place, left_right = side::right:** `B <- alpha * B * op(A)`
- **trmm (p178), out-of-place, left_right = side::left:** `C <- alpha * op(A) * B + beta * C`
- **trmm (p178), out-of-place, left_right = side::right:** `C <- alpha * B * op(A) + beta * C`
- **trsm (p185), in-place, left_right = side::left:** `op(A) * X = alpha * B`
- **trsm (p185), in-place, left_right = side::right:** `X * op(A) = alpha * B`
- **trsm (p185), out-of-place, left_right = side::left:** `op(A) * X = alpha * B,  C <- X + beta * C`
- **trsm (p185), out-of-place, left_right = side::right:** `X * op(A) = alpha * B,  C <- X + beta * C`
- **axpby (p194):** `y <- beta * y + alpha * x`
- **axpy_batch (p198), group API:** `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`
- **copy_batch (p203), group API:** `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`

Formula crops transcribed (15 images, 23 tagged expressions, plus 1 expression recovered from the p155
page image — 24 expressions total): p0138, p0141, p0145, p0151 (2), p0155 (1 crop + 1 page), p0160,
p0164 (2), p0168 (2), p0173, p0177, p0178 (3), p0185 (4), p0194, p0198, p0203.

## Conventions & Gotchas

- **Header / include files: not stated on the assigned pages.** The extracted blocks go straight
  from the description to "Syntax" with no "Include Files" table. Do not invent a header.
- **The namespace selects the memory layout, not just a wrapper.** Switching between
  `oneapi::mkl::blas::column_major` and `oneapi::mkl::blas::row_major` flips every leading-dimension
  bound and every required array size; a wrong choice is a silent wrong-answer bug, not a compile
  error.
- **Leading dimensions, gemm.** Column major: `lda >= m` (A nontrans) or `>= k` (A trans/conjtrans);
  `ldb >= k` (B nontrans) or `>= n` (B trans/conjtrans); `ldc >= m`. Row major: `lda >= k` (A
  nontrans) or `>= m` (A trans/conjtrans); `ldb >= n` (B nontrans) or `>= k` (B trans/conjtrans);
  `ldc >= n`. All leading dimensions must be positive, and the bound follows the *stored* shape of
  the matrix, not `op(...)`.
- **Rank-k / rank-2k routines (herk, her2k, syrk, syr2k):** `c` size `>= ldc * n` and `ldc` must be
  "positive and at least n" in both layout namespaces; the reference gives no row-major variant of
  the `ldc` bound for these routines. Their `trans` support differs: `herk`/`her2k` accept only
  `transpose::nontrans` and `transpose::conjtrans`; `syrk`/`syr2k` accept `transpose::conjtrans`
  but never conjugate — "Conjugation is never performed even if trans = transpose::conjtrans", so
  `conjtrans` behaves as `trans`.
- **Hermitian diagonals.** `herk` documents that "the imaginary parts of the diagonal elements are set
  to zero"; on the assigned pages this is stated only for `herk` — the `her2k` output section says only
  that C is overwritten by the updated matrix.
- **`beta = 0` shortcut** (gemm, hemm, symm): matrix C does not need to be initialized. For
  `trmm`/`trsm` the analogous shortcut is `alpha = 0`: matrix B is set to zero and neither A nor B
  needs to be initialized at entry — the zeroing write still happens.
- **In-place vs out-of-place (trmm, trsm).** In-place overwrites `b` and has no `beta`/`c`/`ldc`.
  Out-of-place reads `b`, writes `c`, and adds `beta`, `c`, `ldc`. `trsm` out-of-place computes
  `C <- X + beta * C`: `alpha` is not applied to the `beta * C` term.
- **Scalar wrapping is USM-only.** USM takes `oneapi::mkl::value_or_pointer<T>` (or
  `value_or_pointer<Treal>`) for `alpha`/`beta`; buffer versions take plain `T`/`Treal` by value.
- **Optional parameters.** USM signatures default `compute_mode mode = compute_mode::unset` and
  `const std::vector<sycl::event> &dependencies = {}`; "mode and dependencies may be omitted
  independently; it is not necessary to specify mode in order to provide dependencies."
- **Return / error model.** Buffer versions return `void`; USM versions return a `sycl::event`
  ("Output event to wait on to ensure computation is complete"). No exception types, error codes, or
  exception behavior are documented on the assigned pages.
- **Integer widths.** All size/stride/leading-dimension parameters are `std::int64_t`. Only the
  batch group API generalizes to `Ti` ("may be either `std::int64_t` or `std::int32_t`"), used for
  the `n`, `incx`, `incy`, and `group_size` arrays, while `group_count` remains `std::int64_t`.
- **Batch strided vs group storage.** Strided APIs take one contiguous `x`/`y` with per-element
  increments `incx`/`incy` and per-vector strides `stridex`/`stridey`; buffer `axpy_batch` and
  `copy_batch` are strided-only. Group APIs take arrays of pointers (`const T **x`, `T **y`) with
  per-group `n`, `incx`, `incy` and a `group_size` array; the vector count is
  `total_batch_count = sum_{i=0}^{group_count-1} group_size[i]`.
- **Extraction artifacts (not API facts).** The extracted text wraps `transpose::conjtrans` as
  `conjtran` + `s` and `oneapi::mkl::bfloat16` as `bfloat` + `16`; the real identifiers are
  `transpose::conjtrans` and `oneapi::mkl::bfloat16`. In the gemm size/layout tables the column
  header alternates between `transa`/`transb` and a truncated `trans` (e.g. "transa =
  transpose::trans or trans = transpose::conjtrans"), and the USM table renames the headers to "A not
  transposed" / "A transposed" and "B not transposed" / "B transposed". A few words run together in
  the extract ("batch_sizeaxpy", "batch_sizecopy", "total_batch_countaxpy"), which are `batch_size`
  / `total_batch_count` followed by the next word.

## Explicit gaps

- **No "Include Files" / header information** for any routine on the assigned pages; whether an
  umbrella header or per-routine headers are required is not stated there.
- **No exception/error semantics** for Level 3 routines on the assigned pages (no error codes, no
  documented exception behavior).
- **No device restrictions** are stated on the assigned pages (no GPU-only or CPU-only notes).
- **`compute_mode` enumerators are not listed**; the text only refers to "Compute Modes".
- **Enum definitions** for `oneapi::mkl::uplo`, `oneapi::mkl::transpose`, `oneapi::mkl::diag`, and
  `oneapi::mkl::side` — beyond the values named inline (`transpose::nontrans`, `transpose::trans`,
  `transpose::conjtrans`, `side::left`, `side::right`, `compute_mode::unset`) — are deferred to a
  "Data Types" section outside the assigned pages.
- **`value_or_pointer<T>` semantics** are deferred to a "Scalar Arguments" section outside the
  assigned pages.
- **The her2k formula *crop* is incomplete.** The extracted crop `p0155.png` (one tagged expression)
  contains only the `trans = transpose::nontrans` case; the `conjtrans` expression was transcribed from
  the p155 page image instead (see Formulas).
- **BLAS-like extension routines beyond `axpby`, `axpy_batch`, and `copy_batch`** are given by name,
  data types, and description only; their syntax lies outside the assigned pages.
- **The group-API sentence "Total number of vectors in x and y are given by:" is truncated** in the
  prose for both `axpy_batch` and `copy_batch`; the defining expression was taken from the formula
  images, not the running text.
- **"Matrix Storage"** is referenced repeatedly for buffer/array size conventions but is not
  reproduced on the assigned pages.
