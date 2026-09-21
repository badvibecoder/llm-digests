# oneMKL Foundations: Data Types, Matrix Storage, Error Handling

This chapter covers what applies to *all* oneMKL DPC++ (SYCL) domains: namespace/header layout, the
scoped enumerations that replace Fortran character options, the `value_or_pointer` scalar wrapper, the
buffer-vs-USM memory model, Fortran-style matrix and vector storage layouts, scalar-argument
dereferencing semantics, and the C++ exception hierarchy. It then documents the BLAS Level 1 routines
present in the extracted page range (`asum`, `axpy`, `copy`, and the `dot` formula) with their exact
buffer and USM syntax. Read the storage-layout section before writing index arithmetic: leading
dimension and band-offset conventions are the most common source of silent out-of-bounds errors.

## Overview

### Headers and namespaces

| Header | Declares | Namespace |
|---|---|---|
| `oneapi/mkl/types.hpp` | enums, `value_or_pointer` | `oneapi::mkl` |
| `oneapi/mkl/blas.hpp` | BLAS routines | `oneapi::mkl::blas` |
| `oneapi/mkl/lapack.hpp` | LAPACK routines | `oneapi::mkl::lapack` |
| `oneapi/mkl/spblas.hpp` | Sparse BLAS routines + helpers | `oneapi::mkl::sparse` |

`oneapi/mkl/types.hpp` is included automatically when you include `oneapi/mkl/blas.hpp` or
`oneapi/mkl/lapack.hpp`. All oneMKL DPC++ routines and non-domain-specific data types are inside the
`oneapi::mkl` namespace; the VM `slice` type is likewise in `oneapi::mkl`, though the extracted text
does not name the header that declares it.

BLAS routines exist in three namespaces: `oneapi::mkl::blas` (column major assumed by default),
`oneapi::mkl::blas::column_major` (explicit column major), and `oneapi::mkl::blas::row_major`
(required for row major layout). LAPACK routines are in `oneapi::mkl::lapack`. Currently, LAPACK DPC++
APIs **do not support matrices stored using row major layout**; the documented workaround is to
transpose row major input matrices, call the DPC++ LAPACK routines, then transpose outputs back.

### Differences from standard BLAS/LAPACK

* **Naming.** DPC++ APIs are overloaded on precision: standard BLAS has `sgemm`/`dgemm`/`cgemm`/`zgemm`,
  while DPC++ has the single entry point `gemm` accepting `float`, `double`, `half`, `bfloat16`,
  `std::complex<float>`, and `std::complex<double>`.
* **References.** All DPC++ objects (buffers and queues) are passed by reference, not by pointer;
  other parameters are typically passed by value.
* **Queues.** Every DPC++ BLAS and LAPACK routine takes a leading `sycl::queue&` parameter where
  computational tasks are submitted; the queue may target a CPU or a GPU device.
* **Scalars.** Scalar inputs are passed by value for all BLAS functions (buffer API); the USM API uses
  `value_or_pointer<T>` (below).
* **Complex numbers.** Complex values use C++ `std::complex`; `MKL_Complex8` is replaced by
  `std::complex<float>`. A double-precision complex vector has type
  `sycl::buffer<std::complex<double>,1>`.
* **Character options.** Fortran characters (`'N'`, `'T'`, `'U'`, `'L'`, `'R'`, ...) are replaced by
  scoped enums for type safety.
* **Return values.** Routines that return a scalar (`dot`, `nrm2`, `asum`, `iamax`) instead take an
  extra argument at the **end** of the argument list, written when the computation completes; the
  return type is `void` for the buffer API and `sycl::event` for the USM API.

Equivalent call (column major; `A` is `m x k`, `B` is `k x n`, `C` is `m x n`):

```cpp
dgemm("N", "T", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);   // standard BLAS
blas::gemm(Q, transpose::N, transpose::T, m, n, k, 2.0, A, lda, B, ldb, 3.0, C, ldc);  // DPC++
```

### Buffer API vs USM API

* **Buffer API.** Vector and matrix inputs are DPC++ buffer types. Currently all buffers must be
  **one-dimensional**; use DPC++'s `buffer::reinterpret()` member function to convert a
  higher-dimensional buffer to a one-dimensional one.
* **USM API.** Vector and matrix inputs are pointers of the appropriate type, but they must point to
  memory allocated by a DPC++ USM allocation routine (`sycl::malloc_host`, `sycl::malloc_shared`, or
  `sycl::malloc_device`). Memory allocated with the usual `malloc` or `new` routines **cannot** be
  used for vector/matrix arguments (scalar pointers are the exception).

Example: `gemv` takes a matrix `A` and vectors `x`, `y`. For real double precision each parameter is
`double*` in standard BLAS, `sycl::buffer<double,1>&` with the DPC++ buffer API, and `double*` with
the DPC++ USM API (with the device-accessible allocation restriction above).

### Data types: scoped enumerations

Each value has two exactly-equivalent names — a single-character name (the traditional BLAS/LAPACK
character) and a longer descriptive name — usable interchangeably.

**`transpose`** — whether an input matrix should be transposed and/or conjugated:

| Short | Long | Description |
|---|---|---|
| `transpose::N` | `transpose::nontrans` | Do not transpose or conjugate the matrix. |
| `transpose::T` | `transpose::trans` | Transpose the matrix. |
| `transpose::C` | `transpose::conjtrans` | Perform Hermitian transpose (transpose and conjugate). Only applicable to complex matrices. |

**`uplo`** — whether the lower or upper triangle of a triangular, symmetric, or Hermitian matrix is
accessed:

| Short | Long | Description |
|---|---|---|
| `uplo::U` | `uplo::upper` | Access the upper triangle of the matrix. |
| `uplo::L` | `uplo::lower` | Access the lower triangle of the matrix. |

In both cases, elements that are not in the selected triangle are not accessed or updated.

**`diag`** — values on the diagonal of a triangular matrix:

| Short | Long | Description |
|---|---|---|
| `diag::N` | `diag::nonunit` | The matrix is not unit triangular. The diagonal entries are stored with the matrix data. |
| `diag::U` | `diag::unit` | The matrix is unit triangular (the diagonal entries are all 1s). The diagonal entries in the matrix data are not accessed. |

**`side`** — order of multiplication when one matrix has a special form (triangular, symmetric, or
Hermitian):

| Short | Long | Description |
|---|---|---|
| `side::L` | `side::left` | The special form matrix is on the left in the multiplication. |
| `side::R` | `side::right` | The special form matrix is on the right in the multiplication. |

**`offset`** — whether the offset to apply to an output matrix is a fix, column, or row offset:

| Short | Long | Description |
|---|---|---|
| `offset::F` | `offset::fix` | The offset is fix; all inputs in the `C_offset` matrix have the same value, given by the first element in the `co` array. |
| `offset::C` | `offset::column` | Column offset; all columns in the `C_offset` matrix are the same and given by the elements in the `co` array. |
| `offset::R` | `offset::row` | Row offset; all rows in the `C_offset` matrix are the same and given by the elements in the `co` array. |

### `oneapi::mkl::value_or_pointer<T>`

Declared in `oneapi/mkl/types.hpp`, namespace `oneapi::mkl`. The USM version of oneMKL BLAS routines
uses this templated wrapper so that either a value or a pointer may be passed for parameters that
represent a single fixed value (not a vector or matrix), usually named `alpha` or `beta`.

* Users should **not** explicitly name or construct this type; values and pointers are implicitly
  converted when a oneMKL function is called. Scalar and pointer arguments may be mixed in one call.
* It has two constructors: one converting a value of type `T` (or anything convertible to `T`), and
  one converting a pointer to `T`.
* Example forms accepted for `alpha`/`beta` in a USM `gemv` call:
  `oneapi::mkl::blas::column_major::gemv(queue, trans, m, n, alpha_ptr, lda, x, incx, beta_ptr, y, incy)`,
  `... gemv(queue, trans, m, n, 2, lda, x, incx, 2.7, y, incy)`, and
  `... gemv(queue, trans, m, n, alpha_ptr, lda, x, incx, 2.7, y, incy)`.

### `oneapi::mkl::slice` (Vector Math)

A `slice` type in namespace `oneapi::mkl`, used by the DPC++ VM Strided APIs. Slices accept positive,
zero, and negative strides for forward, static, and backward traversals respectively.

```cpp
slice()                                                     // default == slice(0, 0, 0)
slice(std::size_t start, std::size_t size, std::int64_t stride)
slice(const slice& other)                                   // copy constructor
```

* `slice(1, 5, 2)` selects indices 1, 3, 5, 7, 9; `slice(0, 5, 0)` selects index 0 repeated five
  times; `slice(9, 4, -3)` selects indices 9, 6, 3, 0.
* A slice is invalid if it produces negative indices. A slice of size 0 selects no element. If a slice
  may cause out-of-bounds memory accesses, behavior is undefined.
* Behavior with invalid or non-equal slices is configurable via oneMKL VM mode values (see
  `set_mode`). In **`slice_cyclic` mode, zero-sized slices are invalid**.

### Precision and device support

Each routine is templated on precision and carries a table of supported precisions. Types appearing
across the API: `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `std::complex<float>`,
`std::complex<double>`, plus "mixed" combinations where a routine's table says so.

* **CPU device:** computations on a CPU using OpenCL™. **GPU device:** computations on a GPU using
  OpenCL™ or Level Zero.
* In the current release of oneMKL BLAS for DPC++, **all standard Level 1, Level 2, and Level 3 BLAS
  routines and the BLAS extensions support CPU and GPU devices**. For LAPACK, the reference notes that
  on GPU some or all of the computations may be performed on the CPU, and each routine documents its
  supported devices. Sparse BLAS routines support CPU and GPU with the Compressed Sparse Row (CSR)
  format unless otherwise noted, with limited COO/CSC/BSR support specified per API.

### Matrix and vector storage

`lda` is the leading dimension: the distance between consecutive columns (column major) or
consecutive rows (row major).

* **General matrix.** `A` is `m x n` with leading dimension `lda`, stored in an array `a` of size at
  least `lda * n` (column major) or at least `lda * m` (row major). The leading `m` by `n` part of `a`
  must contain `A`. For column (respectively, row) major layout the elements of each column
  (respectively, row) are contiguous, while the elements of each row (respectively, column) are at
  distance `lda` from the same elements in the previous row (respectively, column).
* **Triangular matrix.** `A` is `n x n` with leading dimension `lda`, in an array of size at least
  `lda * n`. With `upper_lower = uplo::upper` the leading `n` by `n` upper triangular part of `a` must
  contain the upper triangular part of `A` (the strictly lower triangular part is not referenced);
  with `upper_lower = uplo::lower` the corresponding lower triangular parts apply.
* **General band matrix.** `A` is `m x n` with `kl` sub-diagonals, `ku` super-diagonals and leading
  dimension `lda`, in an array of size at least `lda * n` (column major) or `lda * m` (row major).
  The leading `(kl + ku + 1)` by `n` (respectively, `m`) part of `a` must contain `A`, supplied
  column-by-column (respectively, row-by-row), with the **main diagonal in row `ku`** (respectively,
  **column `kl`**) of the array (0-based indexing), the first super-diagonal starting at position 1
  (respectively, 0) in row `(ku - 1)` (respectively, column `(kl + 1)`), the first sub-diagonal
  starting at position 0 (respectively, 1) in row `(ku + 1)` (respectively, column `(kl - 1)`), and so
  on. Array elements not corresponding to band elements (such as the top left `ku`-by-`ku` triangle)
  are not referenced.
* **Triangular band matrix.** `A` is `n x n` with `k` sub/super-diagonals and leading dimension `lda`,
  in an array of size at least `lda * n`, with the leading `(k + 1)` by `n` part holding the band.
  For `uplo::upper`: column major puts the main diagonal in row `(k)` and the first super-diagonal
  starting at position 1 in row `(k - 1)`; row major puts the main diagonal in **column 0** and the
  first super-diagonal starting at position 0 in column 1. For `uplo::lower`: column major puts the
  main diagonal in row 0 and the first sub-diagonal starting at position 0 in row 1. Elements not
  corresponding to the triangular band (upper: the bottom left `k` by `k` triangle) are not
  referenced. (The source text says "upper triangular band part" in the `uplo::lower` paragraph; that
  is a typo in the reference.)
* **Packed triangular matrix.** `A` is `n x n`, represented as a one-dimensional array `a` of size at
  least `(n*(n + 1))/2`, with all elements of the upper or lower part stored contiguously. For
  `uplo::upper`, column major packs column by column so `a[0]` is `A11`, `a[1]` and `a[2]` are `A12`
  and `A22`, and so on. For `uplo::lower`, with column (respectively, row) major layout the array
  holds the lower triangular part packed sequentially column by column (respectively, row by row), so
  `a[0]` is `A11` and `a[1]` and `a[2]` are `A21` and `A31`.
* **Vector.** A vector `X` of `n` elements with increment `incx` is represented as a one-dimensional
  array `x` of size at least `(1 + (n - 1) * abs(incx))`.

Program segments transferring a band matrix from conventional full matrix storage (variable `matrix`,
leading dimension `ldm`) to band storage (variable `a`, leading dimension `lda`), verbatim:

```cpp
// general band, column major            // general band, row major
for (j = 0; j < n; j++) {                for (i = 0; i < m; i++) {
    k = ku - j;                              k = kl - i;
    for (i = max(0, j - ku);                 for (j = max(0, i - kl);
         i < min(m, j + kl + 1); i++) {          j < min(n, i + ku + 1); j++) {
        a[(k + i) + j * lda] =                   a[(k + j) + i * lda] =
            matrix[i + j * ldm];                     matrix[j + i * ldm];
    }                                        }
}                                        }
// triangular band, upper, column major // triangular band, upper, row major
for (j = 0; j < n; j++) {                for (i = 0; i < n; i++) {
    m = k - j;                               m = -i;
    for (i = max(0, j - k); i <= j; i++) {   for (j = i; j < min(n, i + k + 1); j++) {
        a[(m + i) + j * lda] =                   a[(m + j) + i * lda] =
            matrix[i + j * ldm];                     matrix[j + i * ldm];
    }                                        }
}                                        }
// triangular band, lower, column major // triangular band, lower, row major
for (j = 0; j < n; j++) {                for (i = 0; i < n; i++) {
    m = -j;                                  m = k - i;
    for (i = j; i < min(n, j + k + 1); i++) { for (j = max(0, i - k); j <= i; j++) {
        a[(m + i) + j * lda] =                   a[(m + j) + i * lda] =
            matrix[i + j * ldm];                     matrix[j + i * ldm];
    }                                        }
}                                        }
```

### Scalar arguments and pointer dereferencing

The USM version of oneMKL BLAS routines accepts either a scalar (for example `float`) or a pointer
(`float*`) for parameters representing a single fixed value (usually named `alpha` or `beta`).
Pointers for scalar parameters may be SYCL-managed pointers to device or host memory (created with
`sycl::malloc_device`, `sycl::malloc_shared`, or `sycl::malloc_host`), **or raw pointers created with
`malloc` or `new`** — unlike vector/matrix arguments, which must be USM-allocated.

* **USM-managed pointer:** dereferenced at kernel launch, after the dependencies passed to the
  function have been resolved; the value may therefore be assigned asynchronously in another event
  passed as a dependency to the routine.
* **Raw pointer (`malloc`/`new`):** dereferenced at the function call, so it must be valid when the
  function is called, and it may not be assigned asynchronously.

Internally oneMKL may behave slightly differently depending on whether the underlying data is a value
or a pointer, and whether that pointer targets host-side or device-side memory, but this is
transparent to users.

### Error handling

oneMKL error handling relies on C++ exceptions, propagated at the point of the function call and
caught with standard C++ mechanisms. `oneapi::mkl::exception` is the base class, inherited from
`std::exception`; all other oneMKL exception classes derive from it, and all oneMKL routines throw
exceptions inherited from this base class.

| Exception class | Description |
|---|---|
| `oneapi::mkl::unsupported_device` | Reports a problem when the routine is not supported on a specific device |
| `oneapi::mkl::host_bad_alloc` | Reports a problem that occurred during memory allocation on the host |
| `oneapi::mkl::device_bad_alloc` | Reports a problem that occurred during memory allocation on a specific device |
| `oneapi::mkl::unimplemented` | Reports a problem when a specific routine has not been implemented for the specified parameters |
| `oneapi::mkl::invalid_argument` | Reports problem when arguments to the routine were rejected |
| `oneapi::mkl::uninitialized` | Reports problem when a handle (descriptor) has not been initialized |
| `oneapi::mkl::computation_error` | Reports any computation error that occurred inside the oneMKL routine |
| `oneapi::mkl::batch_error` | Reports errors that occurred inside batch oneMKL routines |

### Known limitations

* **Discard Events:** using oneMKL DPC++ routines with queues created with the `discard_events`
  property is not supported and may lead to unexpected exceptions being thrown.
* **SYCL Graph:** recording oneMKL DPC++ routines with the SYCL Graph extension is not supported;
  unexpected behavior may occur.

## Routines

BLAS routines are grouped as Level 1 (vector-vector), Level 2 (matrix-vector), Level 3
(matrix-matrix), and BLAS-like Extensions. This chapter documents the BLAS Level 1 routines present in
the extracted page range. For each routine the `oneapi::mkl::blas::column_major` and
`oneapi::mkl::blas::row_major` variants are textually identical apart from the namespace, so both are
shown. The source prints these syntax blocks **without a trailing semicolon** after the closing brace.

### asum

Computes the sum of magnitudes of the vector elements.

**Syntax (Buffer Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    void asum(sycl::queue &queue,
              std::int64_t n,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<Tres,1> &result)
}
namespace oneapi::mkl::blas::row_major {
    void asum(sycl::queue &queue,
              std::int64_t n,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<Tres,1> &result)
}
```

**Syntax (USM Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event asum(sycl::queue &queue,
                     std::int64_t n,
                     const T *x,
                     std::int64_t incx,
                     Tres *result,
                     const std::vector<sycl::event> &dependencies = {})
}
namespace oneapi::mkl::blas::row_major {
    sycl::event asum(sycl::queue &queue,
                     std::int64_t n,
                     const T *x,
                     std::int64_t incx,
                     Tres *result,
                     const std::vector<sycl::event> &dependencies = {})
}
```

**Include Files:** `oneapi/mkl/blas.hpp` (which pulls in `oneapi/mkl/types.hpp`).

**Supported precisions**

| `T` | `Tres` |
|---|---|
| `float` | `float` |
| `double` | `double` |
| `std::complex<float>` | `float` |
| `std::complex<double>` | `double` |

**Input parameters:** `queue` (queue where the routine should be executed); `n` (number of elements in
vector `x`); `x` (buffer holding / pointer to input vector `x`, size at least
`(1 + (n - 1)*abs(incx))`); `incx` (stride of vector `x`); `dependencies` (USM only, list of events to
wait for before starting computation, if any — if omitted, defaults to no dependencies).

**Output parameters:** `result` — buffer (buffer API) or pointer (USM API) where the scalar result is
stored.

**Return values (USM):** output event to wait on to ensure computation is complete.

**Notes:** `Tres` is the *real* type matching `T`. The routine sums the magnitudes of a real vector's
elements, or the magnitudes of the real and imaginary parts of a complex vector's elements.

### axpy

Computes a vector-scalar product and adds the result to a vector.

**Syntax (Buffer Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    void axpy(sycl::queue &queue,
              std::int64_t n,
              T alpha,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<T,1> &y,
              std::int64_t incy)
}
namespace oneapi::mkl::blas::row_major {
    void axpy(sycl::queue &queue,
              std::int64_t n,
              T alpha,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<T,1> &y,
              std::int64_t incy)
}
```

**Syntax (USM Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event axpy(sycl::queue &queue,
                     std::int64_t n,
                     oneapi::mkl::value_or_pointer<T> alpha,
                     const T *x,
                     std::int64_t incx,
                     T *y,
                     std::int64_t incy,
                     const std::vector<sycl::event> &dependencies = {})
}
namespace oneapi::mkl::blas::row_major {
    sycl::event axpy(sycl::queue &queue,
                     std::int64_t n,
                     oneapi::mkl::value_or_pointer<T> alpha,
                     const T *x,
                     std::int64_t incx,
                     T *y,
                     std::int64_t incy,
                     const std::vector<sycl::event> &dependencies = {})
}
```

**Include Files:** `oneapi/mkl/blas.hpp`.

**Supported precisions:** `T` is `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`,
`std::complex<float>`, or `std::complex<double>`.

**Input parameters:** `queue`; `n` (number of elements in vector `x`); `alpha` (the scalar `alpha`,
type `T` in the buffer API and `oneapi::mkl::value_or_pointer<T>` in the USM API — see Scalar
Arguments); `x` (input vector, size at least `(1 + (n - 1)*abs(incx))`); `incx` (stride of `x`); `y`
(input vector, size at least `(1 + (n - 1)*abs(incy))`); `incy` (stride of `y`); `dependencies` (USM
only, defaults to no dependencies).

**Output parameters:** `y` — buffer holding / pointer to the updated vector `y`.

**Return values (USM):** output event to wait on to ensure computation is complete.

**Examples:** `share/doc/mkl/examples/sycl/blas/source/axpy.cpp` in the oneMKL installation directory.

### copy

Copies a vector to another vector.

**Syntax (Buffer Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    void copy(sycl::queue &queue,
              std::int64_t n,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<T,1> &y,
              std::int64_t incy)
}
namespace oneapi::mkl::blas::row_major {
    void copy(sycl::queue &queue,
              std::int64_t n,
              sycl::buffer<T,1> &x,
              std::int64_t incx,
              sycl::buffer<T,1> &y,
              std::int64_t incy)
}
```

**Syntax (USM Version)**

```cpp
namespace oneapi::mkl::blas::column_major {
    sycl::event copy(sycl::queue &queue,
                     std::int64_t n,
                     const T *x,
                     std::int64_t incx,
                     T *y,
                     std::int64_t incy,
                     const std::vector<sycl::event> &dependencies = {})
}
namespace oneapi::mkl::blas::row_major {
    sycl::event copy(sycl::queue &queue,
                     std::int64_t n,
                     const T *x,
                     std::int64_t incx,
                     T *y,
                     std::int64_t incy,
                     const std::vector<sycl::event> &dependencies = {})
}
```

**Include Files:** `oneapi/mkl/blas.hpp`.

**Supported precisions:** `T` is `float`, `double`, `std::complex<float>`, or `std::complex<double>`
(no `sycl::half` / `bfloat16` for `copy`).

**Input parameters:** `queue`; `n` (number of elements in vector `x`); `x` (input vector, size at least
`(1 + (n - 1)*abs(incx))`); `incx` (stride of `x`); `incy` (stride of `y`); `dependencies` (USM only,
defaults to no dependencies).

**Output parameters:** `y` — buffer holding / pointer to the updated vector `y`.

**Return values (USM):** output event to wait on to ensure computation is complete.

### dot

Dot product of two vectors. Level 1 routine group table data types: `sycl::half`,
`oneapi::mkl::bfloat16`, `float`, `double`, `mixed float and double`. Formula on `p38#0`.
**The syntax block, parameter descriptions, and precision table for `dot` lie beyond the extracted
page range and are not reproduced here**; as a scalar-returning routine it follows the documented
pattern of an extra result argument at the end of the argument list, with return type `void` (buffer
API) or `sycl::event` (USM API).

### dotc

Dot product conjugated. Level 1 group table data types: `std::complex<float>`, `std::complex<double>`. Detailed syntax and precision table are outside the extracted page range.

### dotu

Dot product unconjugated. Level 1 group table data types: `std::complex<float>`, `std::complex<double>`. Detailed syntax is outside the extracted page range.

### iamax

Index of the maximum absolute value element of a vector. Level 1 group table data types: `float`, `double`, `std::complex<float>`, `std::complex<double>`. Named in the general return-value rule as a trailing-result routine. Detailed syntax is outside the extracted page range.

### iamin

Index of the minimum absolute value element of a vector. Level 1 group table data types: `float`, `double`, `std::complex<float>`, `std::complex<double>`. Detailed syntax is outside the extracted page range.

### nrm2

Vector 2-norm (Euclidean norm). Level 1 group table data types: `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `mixed float and std::complex<float>`, `mixed double and std::complex<double>`. Named in the general return-value rule as a trailing-result routine. Detailed syntax is outside the extracted page range.

### rot

Plane rotation of points. Level 1 group table data types: `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `mixed float and std::complex<float>`, `mixed double and std::complex<double>`. Detailed syntax is outside the extracted page range.

### rotg

Generate Givens rotation of points. Level 1 group table data types: `float`, `double`, `std::complex<float>`, `std::complex<double>`. Detailed syntax is outside the extracted page range.

### rotm

Modified Givens plane rotation of points. Level 1 group table data types: `float`, `double`. Detailed syntax is outside the extracted page range.

### rotmg

Generate modified Givens plane rotation of points. Level 1 group table data types: `float`, `double`. Detailed syntax is outside the extracted page range.

### scal

Vector-scalar product. Level 1 group table data types: `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `std::complex<float>`, `std::complex<double>`, `mixed float and std::complex<float>`, `mixed double and std::complex<double>`. Detailed syntax is outside the extracted page range.

### sdsdot

Dot product with double precision. Level 1 group table data types: `mixed float and double`. Detailed syntax is outside the extracted page range.

### swap

Vector-vector swap. Level 1 group table data types: `float`, `double`, `std::complex<float>`, `std::complex<double>`. Detailed syntax is outside the extracted page range.

## Formulas

Transcribed from the formula images; each is tagged with the red label printed in the source image
(`pNNNN#i` = PDF page NNNN, object index i on that page). Subscripts are written linearly (`A_ij`), and
`*` marks array positions that are not referenced.

### Matrix and vector storage (pages 20-27)

* **p20#0** — general `m x n` matrix `A`:
  `A = [ A_11 A_12 A_13 ... A_1n ; A_21 A_22 A_23 ... A_2n ; A_31 A_32 A_33 ... A_3n ; ... ; A_m1 A_m2 A_m3 ... A_mn ]`
* **p20#1** — general matrix, **column major** array (each group bracketed `lda`, total `lda x n`):
  `a = [ A_11, A_21, A_31, ..., A_m1, *, ..., *, A_12, A_22, A_32, ..., A_m2, *, ..., *, ..., A_1n, A_2n, A_3n, ..., A_mn, *, ..., * ]`
* **p20#2** — general matrix, **row major** array (each group bracketed `lda`, total `m x lda`):
  `a = [ A_11, A_12, A_13, ..., A_1n, *, ..., *, A_21, A_22, A_23, ..., A_2n, *, ..., *, ..., A_m1, A_m2, A_m3, ..., A_mn, *, ..., * ]`
* **p21#0** — upper triangular `n x n` matrix `A` (entries strictly below the diagonal shown as `*`).
* **p21#1** — upper triangular, **column major** array (groups bracketed `lda`, total `lda x n`):
  `a = [ A_11, *, ..., *, A_12, A_22, *, ..., *, ..., A_1n, A_2n, A_3n, ..., A_nn, *, ..., * ]`
* **p21#2** — upper triangular, **row major** array (groups bracketed `lda`, total `lda x n`):
  `a = [ A_11, A_12, A_13, ..., A_1n, *, ..., *, A_22, A_23, ..., A_2n, *, ..., *, ..., *, ..., *, A_nn, *, ..., * ]`
* **p21#3** — lower triangular `n x n` matrix `A` (entries strictly above the diagonal shown as `*`).
* **p22#0** — lower triangular, **column major** array (groups bracketed `lda`, total `lda x n`):
  `a = [ A_11, A_21, A_31, ..., A_n1, *, ..., *, A_22, A_32, ..., A_n2, *, ..., *, ..., *, ..., *, A_nn, *, ..., * ]`
* **p22#1** — lower triangular, **row major** array (groups bracketed `lda`, total `lda x n`):
  `a = [ A_11, *, ..., *, A_21, A_22, *, ..., *, ..., A_n1, A_n2, A_n3, ..., A_nn, *, ..., * ]`
* **p22#2** — general band matrix `A` (`m x n`, `kl` sub-diagonals, `ku` super-diagonals), showing the
  banded sparsity pattern: `A_11 A_12 A_13 ... A_1,ku+1`, `A_21 A_22 A_23 A_24 ... A_2,ku+2`,
  `A_31 A_32 A_33 A_34 A_35 ... A_3,ku+3`, continuing to `A_m,m-kl ... A_m,m-2 A_m,m-1 A_mn`.
* **p22#3** — general band, **column major** array; brace labels under the groups read `ku`, `ku-1`,
  `lda`, `max(0,kl-n+1)`, total `lda x n`:
  `a = [ *, ..., *, A_11, A_12, ..., A_min(kl+1,m),1, *, ..., *, *, ..., *, A_max(1,2-ku),2, ..., A_min(kl+2,m),2, *, ..., *, ..., *, ..., *, A_max(1,...) ]`
  (the tail of this array formula is truncated in the page image; the leading `ku` / `ku-1` padding
  groups and the first two element groups are legible, but the final element is cut off).
* **p23#0** — general band, **row major** array; brace labels read `kl`, `kl-1`, `lda`,
  `max(0,kl-m+1)`, total `lda x m`:
  `a = [ *, ..., *, A_11, A_12, ..., A_1,min(ku+1,n), *, ..., *, *, ..., *, A_2,max(1,2-kl), ..., A_2,min(ku+2,n), *, ..., *, ..., *, ..., *, A_m,max(...) ]`
  (trailing term truncated in the page image).
* **p24#0** — upper triangular band `n x n` matrix `A`.
* **p24#1** — upper triangular band array as printed (brace labels `ku`, `ku-1`, `lda`,
  `max(0,k-n+1)`, total `lda x n`):
  `a = [ *, ..., *, A_11, *, ..., *, *, ..., *, A_max(1,2-k),2, ..., A_2,2, *, ..., *, ..., *, ..., *, A_max(1,n-k),n, ..., A_n,n, *, ..., * ]`
* **p24#2** — second upper triangular band array as printed (brace labels `lda`, total `lda x n`):
  `a = [ A_11, A_21, ..., A_min(k+1,n),1, *, ..., *, A_2,2, ..., A_min(k+2,n),2, *, ..., *, ..., A_n,n, *, ..., * ]`
  (see Explicit gaps: this element order is column-order and conflicts with the page 23 prose that
  puts the main diagonal in column 0 for row major).
* **p25#0** — lower triangular band `n x n` matrix `A`.
* **p25#1** — lower triangular band array (brace labels `lda`, total `lda x n`):
  `a = [ A_11, A_21, ..., A_min(k+1,n),1, *, ..., *, A_2,2, ..., A_min(k+2,n),2, *, ..., *, ..., A_n,n, *, ..., * ]`
* **p25#2** — second lower triangular band array (brace labels `k`, `k-1`, `max(0,k-n+1)`, `lda`,
  total `lda x n`):
  `a = [ *, ..., *, A_11, *, ..., *, *, ..., *, A_max(1,2-k),2, ..., A_2,2, *, ..., *, ..., *, ..., *, A_max(1,n-k),n, ..., A_n,n, *, ..., * ]`
* **p26#0** — upper triangular packed `n x n` matrix `A`.
* **p26#1** — upper triangular, packed array (column-by-column order of the upper triangle):
  `a = [ A_11, A_12, A_22, A_13, A_23, A_33, ..., A_(n-1),n, A_nn ]`
* **p27#0** — lower triangular packed `n x n` matrix `A`.
* **p27#1** — lower triangular, packed array (row-by-row order of the lower triangle):
  `a = [ A_11, A_21, A_22, A_31, A_32, A_33, ..., A_n,(n-1), A_nn ]`
* **p27#2** — vector definition: `X = ( X_1, X_2, X_3, ..., X_n )`
* **p27#3** — vector storage when `incx > 0` (braces `incx`, total `1 + (n-1) x incx`):
  `x = [ X_1, *, ..., *, X_2, *, ..., *, ..., X_(n-1), *, ..., *, X_n ]`
* **p27#4** — vector storage when `incx < 0` (braces `-incx`, total `1 + (1-n) x incx`):
  `x = [ X_n, *, ..., *, X_(n-1), *, ..., *, ..., X_2, *, ..., *, X_1 ]`

### Exception hierarchy (page 29)

* **p29#0** — a diagram, not a mathematical formula. Reading downward: `std::exception` ->
  `oneMKL base exception` = `oneapi::mkl::exception` -> `oneMKL problem-specific exceptions`, of which
  the diagram labels `oneapi::mkl::unsupported_device`, `oneapi::mkl::host_bad_alloc`, and an ellipsis
  (`...`) covering the remaining classes in the exception table above.

### Level 1 operations (pages 32-38)

* **asum (p32#0):** `result <- sum_{i=1}^{n} ( |Re(X_i)| + |Im(X_i)| )`
  (the source text introduces the vector as `x` with `n` elements while the formula uses `X_i`).
* **axpy (p34#0):** `y <- alpha * x + y`
* **copy (p36#0):** `y <- x`
* **dot (p38#0):** `result = sum_{i=1}^{n} X_i Y_i`

## Conventions & Gotchas

* **Namespace is part of the call.** `oneapi::mkl::blas::gemm` is column major by default; the
  `row_major` namespace must be selected for row major BLAS. LAPACK DPC++ has no row major support at
  all — transpose in and out manually.
* **Header.** BLAS declarations live in `oneapi/mkl/blas.hpp`; including it also provides
  `oneapi/mkl/types.hpp`. The reference omits `sycl::` "for brevity" in prose, but signatures use
  `sycl::queue &queue` and `sycl::buffer<T,1> &`.
* **Buffers must be 1-D.** Convert higher-dimensional buffers with `buffer::reinterpret()`.
* **USM allocation rule.** Vector/matrix pointer arguments must come from `sycl::malloc_host`,
  `sycl::malloc_shared`, or `sycl::malloc_device` — never from `malloc`/`new`. Scalar `alpha`/`beta`
  pointers *may* be raw `malloc`/`new` pointers.
* **Scalar dereference timing.** USM-managed scalar pointers are read at kernel launch after
  dependency resolution (safe to fill asynchronously in a dependency event); raw pointers are read at
  the call and must already be valid.
* **Leading dimensions.** `lda` is the stride between consecutive columns (column major) or rows (row
  major): general matrix needs `lda * n` (column major) / `lda * m` (row major); triangular matrix
  needs `lda * n`; general band needs `lda * n` / `lda * m` with the leading `(kl + ku + 1) x n`
  (or `x m`) part holding `A`; triangular band needs `lda * n` with the leading `(k + 1) x n` part
  holding the band.
* **Vector sizes and increments.** Any vector of `n` elements with increment `inc` needs at least
  `1 + (n - 1)*abs(inc)` elements. `incx`/`incy` may be negative for backward traversal (see the
  `p27#4` formula); the reference states no minimum-magnitude restriction, documents a zero stride
  only for VM `slice` (not for `incx`/`incy`), and states nothing about `n = 0`.
* **In-place vs out-of-place.** `axpy` updates `y` in place (the same object is input and output).
  For the routines in this page range the reference states no aliasing restriction between `x` and
  `y`.
* **Asynchrony and events.** The buffer API returns `void` (completion expressed through buffer
  dependencies); the USM API returns `sycl::event`, with `dependencies = {}` the default in every USM
  signature shown. Scalar-returning routines take a trailing `result` buffer/pointer instead of a C++
  return value.
* **Integer types in signatures.** Sizes and strides are `std::int64_t` at the API boundary
  (`n`, `incx`, `incy`).
* **Short vs long enum names** (`transpose::N`/`transpose::nontrans`, `uplo::U`/`uplo::upper`,
  `diag::N`/`diag::nonunit`, `side::L`/`side::left`, `offset::F`/`offset::fix`) are exactly equivalent;
  either form may be used.
* **Untouched triangles.** For `uplo::upper` / `uplo::lower`, elements outside the selected triangle
  are not accessed or updated; with `diag::unit` the stored diagonal entries are not accessed at all.
* **Error handling.** Wrap calls in `try`/`catch (const oneapi::mkl::exception &)`; all oneMKL
  exceptions derive from `oneapi::mkl::exception`, which derives from `std::exception`. Rejected
  arguments surface as `oneapi::mkl::invalid_argument`, unsupported devices as
  `oneapi::mkl::unsupported_device`, and unimplemented parameter combinations as
  `oneapi::mkl::unimplemented`.
* **Do not use** queues created with the `discard_events` property, and do not record oneMKL routines
  into SYCL Graph recordings.
* **Device support.** BLAS Level 1/2/3 and BLAS-like extensions support CPU and GPU.

## Explicit gaps

* **Page alignment of formula images vs extracted text.** The `dot` formula is tagged `p38#0`, but the
  extracted prose for page 38 still covers `copy` (USM); the formula-image page numbering and the
  prose page markers diverge by roughly two pages from page 38 onward. The formula is unambiguously
  the dot product (`sum X_i Y_i`) by content.
* **Level 1 routines without full detail.** `dot`, `dotc`, `dotu`, `iamax`, `iamin`, `nrm2`, `rot`,
  `rotg`, `rotm`, `rotmg`, `scal`, `sdsdot`, and `swap` are covered here by name, supported data types,
  and a one-line description only: their syntax blocks, parameter tables, and per-routine precision
  tables fall outside the extracted pages (they begin after page 38). The `dot` formula is included.
* **Internal inconsistency in the triangular band row-major formula.** On page 24 (the `uplo::upper`
  section) the second array formula (`p24#2`) reads `a = [ A_11, A_21, ..., A_min(k+1,n),1, ... ]` — a
  column-ordered list of sub-diagonal entries — whereas the page 23 prose states that in row major the
  main diagonal is in column 0 of the array and the first super-diagonal starts at position 0 in
  column 1. The rendered formula therefore appears to contradict the prose (and duplicates the
  lower-triangular column-major array shown on page 25). Both are transcribed verbatim above rather
  than reconciled.
* **Brace labels `ku` vs `k`.** On `p24#1` the leading padding braces are labelled `ku` and `ku-1`,
  while the triangular-band prose (and `p25#2`) use a single parameter `k`. The reference does not
  explain the difference.
* **Packed triangular arrays.** Pages 26 and 27 each list both a "For column major layout" and a "For
  row major layout" bullet, but only one packed-array formula image is present per page (`p26#1` is
  column-by-column order for the upper triangle; `p27#1` is row-by-row order for the lower triangle).
  The second array formula per page is not present in the extracted formula layer.
* **Truncated band-array formulas.** The tail of the general band arrays `p22#3` (column major) and
  `p23#0` (row major) is cut off in the page images, so the final element (`A_max(1,...)` in `p22#3`,
  `A_m,max(...)` in `p23#0`) is not fully legible; the leading padding groups (`ku`/`ku-1` and
  `kl`/`kl-1`), the `max(...)` brace labels, and the earlier element groups are transcribed as printed.
* **Not stated anywhere in this range:** default values for `lda`, `incx`, `incy`, or `n` (the
  reference gives none — all are required arguments); aliasing/in-place restrictions for `asum`,
  `axpy`, or `copy`; behavior of Level 1 routines for `n = 0` or for vectors shorter than
  `1 + (n - 1)*abs(incx)`; and per-routine device restrictions for Level 1 routines (the general
  statement is only that all Level 1/2/3 BLAS routines and BLAS extensions support CPU and GPU).
* **No formula image was blank or unreadable**; `p29#0` is a class-hierarchy diagram rather than a
  formula.
