# BLAS Level 2 Routines

BLAS Level 2 covers matrix-vector operations: general/symmetric/Hermitian/triangular
matrix-vector products (`*mv`), triangular solves (`*sv`), and rank-1/rank-2 updates
(`*r`, `*r2`). The domain is organised by matrix shape and storage: general (`ge*`), general
band (`gb*`), symmetric (`sy*`), symmetric band (`sb*`), symmetric packed (`sp*`),
Hermitian (`he*`/`hb*`/`hp*`), triangular (`tr*`), triangular band (`tb*`), triangular
packed (`tp*`). Every routine exists in a **buffer** flavour (`sycl::buffer` arguments,
`void` return) and a **USM** flavour (raw pointers, `sycl::event` return), each provided
twice — once in `oneapi::mkl::blas::column_major` and once in
`oneapi::mkl::blas::row_major`.

## Overview

**Routine groups and data types (source table):**

| Routine group | Data types (T) | Description |
|---|---|---|
| `gbmv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a general band matrix |
| `gemv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a general matrix |
| `ger` | float, double | Rank-1 update of a general matrix |
| `gerc` | std::complex<float>, std::complex<double> | Rank-1 update of a conjugated general matrix |
| `geru` | std::complex<float>, std::complex<double> | Rank-1 update of a general matrix, unconjugated |
| `hbmv` | std::complex<float>, std::complex<double> | Matrix-vector product using a Hermitian band matrix |
| `hemv` | std::complex<float>, std::complex<double> | Matrix-vector product using a Hermitian matrix |
| `her` | std::complex<float>, std::complex<double>; `Treal` = float, double | Rank-1 update of a Hermitian matrix |
| `her2` | std::complex<float>, std::complex<double> | Rank-2 update of a Hermitian matrix |
| `hpmv` | std::complex<float>, std::complex<double> | Matrix-vector product using a Hermitian packed matrix |
| `hpr` | std::complex<float>, std::complex<double>; `Treal` = float, double | Rank-1 update of a Hermitian packed matrix |
| `hpr2` | std::complex<float>, std::complex<double> | Rank-2 update of a Hermitian packed matrix |
| `sbmv` | float, double | Matrix-vector product using a symmetric band matrix |
| `spmv` | float, double | Matrix-vector product using a symmetric packed matrix |
| `spr` | float, double | Rank-1 update of a symmetric packed matrix |
| `spr2` | float, double | Rank-2 update of a symmetric packed matrix |
| `symv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a symmetric matrix |
| `syr` | float, double, std::complex<float>, std::complex<double> | Rank-1 update of a symmetric matrix |
| `syr2` | float, double, std::complex<float>, std::complex<double> | Rank-2 update of a symmetric matrix |
| `tbmv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a triangular band matrix |
| `tbsv` | float, double, std::complex<float>, std::complex<double> | Solution of a linear system of equations with a triangular band matrix |
| `tpmv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a triangular packed matrix |
| `tpsv` | float, double, std::complex<float>, std::complex<double> | Solution of a linear system of equations with a triangular packed matrix |
| `trmv` | float, double, std::complex<float>, std::complex<double> | Matrix-vector product using a triangular matrix |
| `trsv` | float, double, std::complex<float>, std::complex<double> | Solution of a linear system of equations with a triangular matrix |

**Buffer vs USM model.** Buffer versions return `void` and pass all data as
`sycl::buffer<T,1>&` or by value (`alpha`, `beta`, sizes, strides). USM versions return
`sycl::event` ("Output event to wait on to ensure computation is complete"), take `const T*`
for inputs and `T*` for outputs, and take scalars as `oneapi::mkl::value_or_pointer<T>`
(`value_or_pointer<Treal>` for `her`/`hpr`; no Level 2 routine uses the `value_or_pointer<Ts>`
form seen on the Level 1 and Level 3 pages). "See Scalar Arguments
for more information on the `value_or_pointer` data type."

**Dependencies.** Only USM versions accept
`const std::vector<sycl::event> &dependencies = {}` as the last parameter: "List of events
to wait for before starting computation, if any. If omitted, defaults to no dependencies."
Buffer versions have no such parameter.

**Layout.** Namespace choice selects layout; it is not a runtime argument. The
`column_major` and `row_major` blocks are textually identical apart from the namespace, but
required array sizes and minimum `ld*` values differ (given per routine below).

**`op(A)`.** `trans` (Level 2; `transa`/`transb` occur only on the Level 3 `gemm` pages) is
`oneapi::mkl::transpose` and selects `op(A) ∈ {A, A^T, A^H}`. `A^H` (`transpose::conjtran`)
is available wherever `std::complex` types are supported. `upper_lower` is `oneapi::mkl::uplo`; `unit_diag` is
`oneapi::mkl::diag`.

**Scratchpad.** No scratchpad argument appears anywhere in the Level 2 signatures in this
source, and no scratchpad size is defined.

## Routines

All signatures below are the `column_major` blocks transcribed verbatim; the `row_major`
block for each routine is signature-identical. Include files are not printed on these pages
(the text only cross-references "See Data Types", "See Matrix Storage", "See Scalar
Arguments").

### gbmv

Matrix-vector product with a general band matrix.

```cpp
// Buffer — note the source prints `queue`, not `sycl::queue`
void gbmv(queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, T alpha, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &x, std::int64_t incx, T beta, sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event gbmv(queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, oneapi::mkl::value_or_pointer<T> alpha, const T *a,
                 std::int64_t lda, const T *x, std::int64_t incx,
                 oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `m`, `n` rows/cols of A, `kl` sub-diagonals, `ku` super-diagonals; each at least zero.
- `a`: size ≥ `lda * n` (column major) or ≥ `lda * m` (row major); `lda ≥ (kl + ku + 1)` and positive.
- `x`: `len` = `n` if A not transposed, `m` if transposed; size ≥ `(1 + (len - 1)*abs(incx))`; `incx ≠ 0`.
- `y`: `len` = `m` if A not transposed, `n` if transposed; size ≥ `(1 + (len - 1)*abs(incy))`. If `beta = 0`, y need not be initialized.
- **`incy` carries no "must not be zero" statement for `gbmv`** (unlike every other routine here). Output: `y`.

### gemv

Matrix-vector product using a general matrix.

```cpp
// Buffer
void gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx,
          T beta, sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event gemv(sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, const T *x,
                 std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `m`, `n` ≥ 0. `a`: size ≥ `lda * n` (column major) or ≥ `lda * m` (row major).
- `lda`: positive and ≥ `m` (column major) or ≥ `n` (row major).
- `x`: `len` = `n` if A not transposed, `m` if transposed; `incx ≠ 0`. `y`: `len` = `m` if A not transposed, `n` if transposed; `incy ≠ 0`.
- `beta = 0` ⇒ y need not be initialized. Output: `y`.
- Example: `share/doc/mkl/examples/sycl/blas/source/gemv.cpp`.

### ger

Rank-1 update of a general matrix (real only).

```cpp
// Buffer
void ger(sycl::queue &queue, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &x,
         std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &a,
         std::int64_t lda)
// USM
sycl::event ger(sycl::queue &queue, std::int64_t m, std::int64_t n,
                oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                std::int64_t incy, T *a, std::int64_t lda,
                const std::vector<sycl::event> &dependencies = {})
```

- `m`, `n` ≥ 0. `x` has length m (size ≥ `(1 + (m - 1)*abs(incx))`), `y` length n (size ≥ `(1 + (n - 1)*abs(incy))`); `incx ≠ 0`, `incy ≠ 0`.
- `a`: size ≥ `lda * n` (column major) or ≥ `lda * m` (row major); `lda` positive and ≥ `m` (column major) or ≥ `n` (row major).
- **Matrix `a` is the last argument** (unlike `gemv`). Output: `a`.

### gerc

Rank-1 update (conjugated) of a general complex matrix; T = `std::complex<float>`,
`std::complex<double>`.

```cpp
// Buffer
void gerc(sycl::queue &queue, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &x,
          std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &a,
          std::int64_t lda)
// USM
sycl::event gerc(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies = {})
```

Parameter meanings and constraints identical to `ger`; the update conjugates `y`. Output: `a`.

### geru

Rank-1 update (unconjugated) of a general complex matrix; T = `std::complex<float>`,
`std::complex<double>`.

```cpp
// Buffer
void geru(sycl::queue &queue, std::int64_t m, std::int64_t n, T alpha, sycl::buffer<T,1> &x,
          std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &a,
          std::int64_t lda)
// USM
sycl::event geru(sycl::queue &queue, std::int64_t m, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies = {})
```

Constraints as for `ger`. Source prose says "The geru routines routines compute..." and
describes `y` as "input/output vector y", but only `a` is listed as an output. Output: `a`.

### hbmv

Matrix-vector product using a hermitian band matrix (source spelling: "hermitian").

```cpp
// Buffer
void hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx,
          T beta, sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event hbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, const T *x,
                 std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `n` rows/cols ≥ 0; `k` super-diagonals ≥ 0.
- `a`: size ≥ `lda * n`; `lda ≥ (k + 1)` and positive.
- `x`, `y`: size ≥ `(1 + (n - 1)*abs(incx))` / `(1 + (n - 1)*abs(incy))`; both strides ≠ 0.
- `beta = 0` ⇒ y need not be initialized. Output: `y`.

### hemv

Matrix-vector product using a hermitian matrix.

```cpp
// Buffer
void hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx, T beta,
          sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event hemv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, const T *x,
                 std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`/`y` strides ≠ 0.
- `beta = 0` ⇒ y need not be initialized. Output: `y`.

### her

Rank-1 update of a hermitian matrix. `T` is complex; `alpha` has the separate real type
`Treal`.

```cpp
// Buffer
void her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, Treal alpha,
         sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &a, std::int64_t lda)
// USM
sycl::event her(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                oneapi::mkl::value_or_pointer<Treal> alpha, const T *x, std::int64_t incx, T *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`.
- Output `a`: updated upper triangular part if `upper_lower=upper`, else updated lower triangular part.
- "If alpha is zero, A matrix is unchanged, otherwise imaginary parts of the diagonal elements are set to zero."

### her2

Rank-2 update of a hermitian matrix. `alpha` is complex `T` (no `Treal`).

```cpp
// Buffer
void her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy,
          sycl::buffer<T,1> &a, std::int64_t lda)
// USM
sycl::event her2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies = {})
```

- Source says `n` is "Number of columns of matrix A. Must be at least zero."
- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`/`y` strides ≠ 0.
- Output `a` and the alpha-zero / diagonal-imaginary note as for `her`.

### hpmv

Matrix-vector product using a hermitian packed matrix. **No `lda`.**

```cpp
// Buffer
void hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &a, sycl::buffer<T,1> &x, std::int64_t incx, T beta,
          sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event hpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, const T *x, std::int64_t incx,
                 oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `(n*(n+1))/2`. "The imaginary parts of the diagonal elements need not be set and are assumed to be zero."
- `x`/`y` strides ≠ 0. `beta = 0` ⇒ y need not be initialized. Output: `y`.

### hpr

Rank-1 update of a hermitian packed matrix. `alpha` is `Treal`. **No `lda`.**

```cpp
// Buffer
void hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, Treal alpha,
         sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &a)
// USM
sycl::event hpr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                oneapi::mkl::value_or_pointer<Treal> alpha, const T *x, std::int64_t incx, T *a,
                const std::vector<sycl::event> &dependencies = {})
```

- `a`: source states size ≥ `(n*(n-1))/2` (verbatim; inconsistent with the `(n*(n+1))/2` printed for `hpmv`/`spmv`/`tpmv`/`tpsv`). Imaginary parts of diagonal elements need not be set, assumed zero.
- `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`. Output `a` (upper/lower per `upper_lower`), with the alpha-zero / diagonal-imaginary note as for `her`.

### hpr2

Rank-2 update of a hermitian packed matrix. **No `lda`.**

```cpp
// Buffer
void hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy,
          sycl::buffer<T,1> &a)
// USM
sycl::event hpr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `(n*(n-1))/2` (verbatim); imaginary parts of diagonal elements need not be set, assumed zero.
- `x`/`y` strides ≠ 0. Output `a` with the same notes as `hpr`.

### sbmv

Matrix-vector product with a symmetric band matrix (real only).

```cpp
// Buffer
void sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          T alpha, sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx,
          T beta, sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event sbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, const T *x,
                 std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `k` super-diagonals ≥ 0. `a`: size ≥ `lda * n`; `lda ≥ (k + 1)` and positive.
- `x`/`y` strides ≠ 0. `beta = 0` ⇒ y need not be initialized. Output: `y`.

### spmv

Matrix-vector product with a symmetric packed matrix (real only). **No `lda`.**

```cpp
// Buffer
void spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &a, sycl::buffer<T,1> &x, std::int64_t incx, T beta,
          sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event spmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, const T *x, std::int64_t incx,
                 oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `(n*(n+1))/2`. `x`/`y` strides ≠ 0. `beta = 0` ⇒ y need not be initialized. Output: `y`.

### spr

Rank-1 update of a symmetric packed matrix. **No `lda`.**

```cpp
// Buffer — `std::std::int64_t n` is a source typo, reproduced verbatim
void spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::std::int64_t n, T alpha,
         sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &a)
// USM
sycl::event spr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, T *a,
                const std::vector<sycl::event> &dependencies = {})
```

- The buffer signature prints `std::std::int64_t n` in both `column_major` and `row_major`; the USM signature prints `std::int64_t n`.
- `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`. `a`: source states size ≥ `(n*(n-n))/2` (verbatim; printed as `n-n`).
- Output: `a` — updated upper triangular part if `upper_lower=upper`, else updated lower triangular part.

### spr2

Rank-2 update of a symmetric packed matrix. **No `lda`.**

```cpp
// Buffer
void spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy,
          sycl::buffer<T,1> &a)
// USM
sycl::event spr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `(n*(n-1))/2`. `x`/`y` strides ≠ 0.
- Output: `a` — updated upper triangular part if `upper_lower=upper` or updated lower triangular part if `upper_lower=lower`.

### symv

Matrix-vector product for a symmetric matrix (T includes complex types).

```cpp
// Buffer
void symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &a, std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx, T beta,
          sycl::buffer<T,1> &y, std::int64_t incy)
// USM
sycl::event symv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *a, std::int64_t lda, const T *x,
                 std::int64_t incx, oneapi::mkl::value_or_pointer<T> beta, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`/`y` strides ≠ 0.
- `beta = 0` ⇒ y need not be initialized. Output: `y`.

### syr

Rank-1 update of a symmetric matrix. T includes complex types, but the update uses the
**unconjugated** transpose.

```cpp
// Buffer
void syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
         sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &a, std::int64_t lda)
// USM
sycl::event syr(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, T *a,
                std::int64_t lda, const std::vector<sycl::event> &dependencies = {})
```

- Source says `n` is "Number of columns of matrix A. Must be at least zero."
- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`: `incx ≠ 0`.
- Output: `a` — upper triangular part if `upper_lower=upper`, otherwise lower triangular part. No alpha-zero/diagonal note is given for `syr`.

### syr2

Rank-2 update of a symmetric matrix (T includes complex types; no conjugation).

```cpp
// Buffer
void syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, T alpha,
          sycl::buffer<T,1> &x, std::int64_t incx, sycl::buffer<T,1> &y, std::int64_t incy,
          sycl::buffer<T,1> &a, std::int64_t lda)
// USM
sycl::event syr2(sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 oneapi::mkl::value_or_pointer<T> alpha, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *a, std::int64_t lda,
                 const std::vector<sycl::event> &dependencies = {})
```

- Source says `n` is "Number of columns of matrix A. Must be at least zero."
- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`/`y` strides ≠ 0.
- Output: `a` (upper/lower per `upper_lower`).

### tbmv

Matrix-vector product using a triangular band matrix.

```cpp
// Buffer
void tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a,
          std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event tbmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const T *a,
                 std::int64_t lda, T *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies = {})
```

- `unit_diag` — whether A is unit triangular or not.
- `n`: "Numbers of rows and columns of matrix A. Must be at least zero." `k`: sub/super-diagonals ≥ 0.
- `a`: size ≥ `lda * n`; `lda ≥ (k + 1)` and positive. `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`.
- Output: `x` (updated) — **in place**, `x` is both input and output.

### tbsv

Solves a linear system whose coefficients are in a triangular band matrix.

```cpp
// Buffer
void tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a,
          std::int64_t lda, sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event tbsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const T *a,
                 std::int64_t lda, T *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ (k + 1)` and positive. `x` is the n-element right-hand side vector; `incx ≠ 0`.
- Output: `x` — solution vector x.

### tpmv

Matrix-vector product using a triangular packed matrix. **No `lda`, no `k`.**

```cpp
// Buffer
void tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<T,1> &a, sycl::buffer<T,1> &x,
          std::int64_t incx)
// USM
sycl::event tpmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const T *a, T *x, std::int64_t incx,
                 const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `(n*(n+1))/2`. `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`. Output: `x`.

### tpsv

Solves a linear system whose coefficients are in a triangular packed matrix.

```cpp
// Buffer
void tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<T,1> &a,
          sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event tpsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const T *a, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

- **Quirk:** both signatures declare `std::int64_t k`, but the parameter text that follows lists only `queue`, `upper_lower`, `trans`, `unit_diag`, `n`, `a`, `x`, `incx` — **`k` is never described**, and no packing dimension argument is documented (A is packed, so there is no `lda`).
- `a`: size ≥ `(n*(n+1))/2`. `x`: the n-element right-hand side vector `b`, size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`.
- Output: `x` — solution vector x.

### trmv

Matrix-vector product using a triangular matrix.

```cpp
// Buffer
void trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const T *a, std::int64_t lda, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x`: size ≥ `(1 + (n - 1)*abs(incx))`; `incx ≠ 0`.
- Output: `x` (updated vector x).
- The description calls A an "n x n unit or non-unit, upper or lower triangular **band** matrix" — the word "band" is in the source text even though `trmv` takes no `k`.

### trsv

Solves a linear system whose coefficients are in a triangular matrix.

```cpp
// Buffer
void trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<T,1> &a, std::int64_t lda,
          sycl::buffer<T,1> &x, std::int64_t incx)
// USM
sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const T *a, std::int64_t lda, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

- `a`: size ≥ `lda * n`; `lda ≥ n` and positive. `x` is the n-element right-hand side vector `b`; `incx ≠ 0`.
- Output: `x` — solution vector x.

## Formulas

Transcribed from the formula images (the red tag `p<page>#<i>` gives the PDF page, e.g.
`p74#0`). `op(X)` is `X`, `X^T`, or `X^H` per `trans`.

Matrix-vector products and solves:

- `gbmv` (p74): `y <- alpha * op(A) * x + beta * y`
- `gemv` (p77): `y <- alpha * op(A) * x + beta * y`
- `hbmv` (p89): `y <- alpha * A * x + beta * y`
- `hemv` (p92): `y <- alpha * A * x + beta * y`
- `hpmv` (p100): `y <- alpha * A * x + beta * y`
- `sbmv` (p108): `y <- alpha * A * x + beta * y`
- `spmv` (p111): `y <- alpha * A * x + beta * y`
- `symv` (p119): `y <- alpha * A * x + beta * y`
- `tbmv` (p128): `x <- op(A) * x`
- `tpmv` (p133): `x <- op(A) * x`
- `trmv` (p138): `x <- op(A) * x`
- `tbsv` (p130): `op(A) * x = b`
- `tpsv` (p136): `op(A) * x = b`
- `trsv` (p141): `op(A) * x = b`

Rank-1 and rank-2 updates:

- `ger` (p80): `A <- alpha * x * y^T + A`
- `gerc` (p83): `A <- alpha * x * y^H + A`
- `geru` (p86): `A <- alpha * x * y^T + A`
- `her` (p95): `A <- alpha * x * x^H + A`
- `her2` (p97): `A <- alpha * x * y^H + conjg(alpha) * y * x^H + A`
- `hpr` (p103): `A <- alpha * x * x^H + A`
- `hpr2` (p106): `A <- alpha * x * y^H + conjg(alpha) * y * x^H + A`
- `spr` (p114): `A <- alpha * x * x^T + A`
- `spr2` (p117): `A <- alpha * x * y^T + alpha * y * x^T + A`
- `syr` (p122): `A <- alpha * x * x^T + A`
- `syr2` (p125): `A <- alpha * x * y^T + alpha * y * x^T + A`

Notation as printed in the images: `^T` and `^H` are literal superscripts, and
`her2`/`hpr2` literally print the function call `conjg(alpha)` for the conjugate of the
scalar. For `geru` the image shows `^T` (the same glyph as `ger`), while the surrounding
text calls the update "unconjugated". The images give no summation/index form for Level 2;
the closed forms above are the complete definitions printed.

## Conventions & Gotchas

- **Two namespaces = two layouts.** There is no runtime layout flag. The `column_major` and
  `row_major` blocks are signature-identical for every Level 2 routine, but required array
  sizes and minimum leading dimensions differ: with `transpose::nontrans` the array size is
  `lda * (columns of the stored matrix)`; with `transpose::trans`/`conjtran` it is
  `lda * (rows of the stored matrix)`. `lda` is at least the stored matrix's row count for
  column major and at least its column count for row major.
- **`lda` minima.** `gemv`/`ger`/`gerc`/`geru`: `≥ m` (column major) or `≥ n` (row major),
  positive. `gbmv`: `≥ (kl + ku + 1)`. `hbmv`/`sbmv`/`tbmv`/`tbsv`: `≥ (k + 1)`.
  `hemv`/`her`/`her2`/`symv`/`syr`/`syr2`/`trmv`/`trsv`: `≥ n`, positive. Packed routines
  (`hpmv`, `hpr`, `hpr2`, `spmv`, `spr`, `spr2`, `tpmv`, `tpsv`) take **no `lda`**.
- **Vector length versus transpose.** `gbmv`/`gemv`: `len(x)` = `n` if A is not transposed,
  `m` if transposed; `len(y)` = `m` if A is not transposed, `n` if transposed.
- **Strides.** Every size formula uses `abs(incx)`/`abs(incy)`, so negative strides are
  meaningful, yet every documented stride says "Must not be zero." The single exception on
  these pages is **`gbmv`'s `incy`**, which carries no such restriction.
- **`beta = 0` shortcut.** Stated for `gbmv`, `gemv`, `hbmv`, `hemv`, `hpmv`, `sbmv`,
  `spmv`, `symv`: "If beta = 0, vector y does not need to be initialized before calling".
  Not applicable to `beta`-less update routines.
- **In-place results.** `tbmv`, `tpmv`, `trmv` overwrite `x` with `op(A)*x`; `tbsv`,
  `tpsv`, `trsv` overwrite `x` (holding `b` on input) with the solution; update routines
  overwrite `a`.
- **Argument order differs by family.** `gemv` is
  `(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)`. `ger`/`gerc`/`geru` are
  `(m, n, alpha, x, incx, y, incy, a, lda)` — matrix last. `uplo`-taking `*mv` are
  `(upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)`; packed `*pmv`/`*psv` drop `lda`.
- **Hermitian diagonal handling.** `her`, `her2`, `hpr`, `hpr2`: "If alpha is zero, A matrix
  is unchanged, otherwise imaginary parts of the diagonal elements are set to zero."
  `hpmv` inputs: "The imaginary parts of the diagonal elements need not be
  set and are assumed to be zero"; `hpr`/`hpr2` inputs: "do not need to be set and are
  assumed to be zero".
- **`uplo` writes only the stored half.** "Updated upper triangular part ... if
  `upper_lower=upper` or updated lower triangular part ... if `upper_lower=lower`." The
  source prose describes `upper_lower` as "whether matrix A is upper or lower triangular"
  even for Hermitian/symmetric routines — a wording quirk; the enum is `oneapi::mkl::uplo`.
- **Complex symmetric vs Hermitian.** `syr`/`syr2` accept `std::complex` but use plain
  (unconjugated) transposes; `her`/`her2`/`hpr`/`hpr2` use `^H` and `conjg(alpha)`.
- **`Treal` template parameter.** `her` and `hpr` take `Treal alpha` (real) while
  vector/matrix are `T` (complex); USM form `oneapi::mkl::value_or_pointer<Treal>`.
- **`unit_diag`.** Only the triangular routines (`tbmv`, `tbsv`, `tpmv`, `tpsv`, `trmv`,
  `trsv`) take `oneapi::mkl::diag unit_diag` — "Specifies whether matrix A is unit
  triangular or not."
- **Integer type.** All sizes/strides/leading dimensions are `std::int64_t` (except the
  `spr` buffer typo `std::std::int64_t`). Constraint text: "Must be at least zero" for `m`,
  `n`, `kl`, `ku`, `k`; "Must be positive" plus the layout minimum for `lda`.
- **Return values.** Buffer versions return `void`; USM versions return the output
  `sycl::event`, with `dependencies` always the last parameter defaulting to `{}`. (Level 3
  `gemm` additionally notes that `mode` and `dependencies` may be omitted independently; no
  Level 2 routine has a `mode` parameter.)
- **Errors/exceptions.** These pages state value constraints ("Must be at least zero",
  "Must be positive", "Must not be zero") but print **no error codes, exception types, or
  device restrictions** for any Level 2 routine.
- **Examples.** Only `gemv` names an example file:
  `share/doc/mkl/examples/sycl/blas/source/gemv.cpp`.
- **Include files** are not printed on any Level 2 page in this source.

## Explicit gaps

- The chunk headers also list formula images for BLAS **Level 1** routines
  (`p0057`–`p0070`: `rotm`, `rotmg`, `scal`, `sdsdot`, `swap`) and for BLAS **Level 3**
  `gemm` (`p0145`, `p0151`). These are outside this chapter's domain and are not
  transcribed as Level 2 formulas; sampled images confirmed their membership (`p0057` =
  modified Givens rotation of (x_i, y_i) by H; `p0065` = the `H` flag matrices plus
  `x <- alpha * x`; `p0068` = `result = sb + sum_{i=1..n} x_i y_i`; `p0070` = the swap of x
  and y; `p0145` = `C <- alpha * op(A) * op(B) + beta * C`).
- No `#include` lists appear on the Level 2 pages; the text only cross-references "See Data
  Types", "See Matrix Storage", and "See Scalar Arguments".
- No error/exception table, no device or queue restriction, and no scratchpad sizing is
  stated for any Level 2 routine on these pages.
- `tpsv`'s `std::int64_t k` is declared in both syntax blocks but never described.
- Verbatim source defects preserved rather than corrected: `spr` buffer signature prints
  `std::std::int64_t n`; `spr` states the `a` size as `(n*(n-n))/2`; `hpr`/`hpr2` state the
  `a` size as `(n*(n-1))/2`; `gbmv` prints `queue &queue` instead of
  `sycl::queue &queue`; `her2`/`syr`/`syr2` print "Number of columns of
  matrix A" where other square routines say "Number of rows and columns"; `tbmv` prints
  "Numbers of rows and columns" (plural) where the other square routines are singular.
- Odd prose noted but not corrected: `geru`'s "The geru routines routines compute...", the
  `rotm` USM page running the labels `y`/`param` together as `yparam`, and `trmv` calling A
  a triangular "band" matrix while taking no `k`.
