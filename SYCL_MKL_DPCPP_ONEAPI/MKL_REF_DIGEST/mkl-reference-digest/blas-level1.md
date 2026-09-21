# BLAS Level 1 Routines

This chapter covers the oneMKL Data Parallel C++ (DPC++) BLAS Level 1 routine groups — vector-vector
operations: reductions (`asum`, `dot`, `dotc`, `dotu`, `sdsdot`, `nrm2`, `iamax`, `iamin`), vector
updates (`axpy`, `copy`, `scal`, `swap`), and plane rotations (`rot`, `rotg`, `rotm`, `rotmg`). Each
routine exists as a **Buffer** version (`sycl::buffer` arguments, returns `void`) and a **USM** version
(raw pointers, returns `sycl::event`), in both `oneapi::mkl::blas::column_major` and
`oneapi::mkl::blas::row_major`. The domain-wide material (data types, scalar wrappers, storage, errors)
is included because it applies to all BLAS routines.

## Overview

### API shape

- Namespaces: `oneapi::mkl::blas::column_major::<routine>` and `oneapi::mkl::blas::row_major::<routine>`.
  For every Level 1 routine the source prints **identical** signatures in both namespaces (only
  whitespace differs), so each routine below shows the `column_major` form once.
- Headers: the source states `oneapi/mkl/types.hpp` "is included automatically when you include
  `oneapi/mkl/blas.hpp` or `oneapi/mkl/lapack.hpp`". It contains **no** per-routine "Include Files"
  block.
- Buffer form returns `void`; results land in `sycl::buffer`. USM form returns `sycl::event`, described
  as the "Output event to wait on to ensure computation is complete", and takes a trailing
  `const std::vector<sycl::event> &dependencies = {}` ("List of events to wait for before starting
  computation, if any. If omitted, defaults to no dependencies.").

### BLAS/LAPACK enumeration data types (`oneapi/mkl/types.hpp`)

"Each enumeration value comes with two names: A single-character name (the traditional BLAS/LAPACK
character) and a longer, descriptive name. The two names are exactly equivalent and may be used
interchangeably."

| Type | Values (short / long) | Meaning |
|---|---|---|
| `transpose` | `transpose::N` / `transpose::nontrans` | Do not transpose or conjugate |
| | `transpose::T` / `transpose::trans` | Transpose the matrix |
| | `transpose::C` / `transpose::conjtrans` | Hermitian transpose; "Only applicable to complex matrices" |
| `uplo` | `uplo::U` / `uplo::upper` | Access the upper triangle |
| | `uplo::L` / `uplo::lower` | Access the lower triangle |
| `diag` | `diag::N` / `diag::nonunit` | Not unit triangular; diagonal stored with the matrix data |
| | `diag::U` / `diag::unit` | Unit triangular; "The diagonal entries in the matrix data are not accessed" |
| `side` | `side::L` / `side::left` | Special-form matrix on the left |
| | `side::R` / `side::right` | Special-form matrix on the right |
| `offset` | `offset::F` / `offset::fix` | All entries of `C_offset` equal the first element of `co` |
| | `offset::C` / `offset::column` | Columns of `C_offset` equal, given by `co` |
| | `offset::R` / `offset::row` | Rows of `C_offset` equal, given by `co` |

`uplo`: "elements that are not in the selected triangle are not accessed or updated." `offset` is
unused by Level 1. `index_base` (used by `iamax`/`iamin`) is `oneapi::mkl::index_base::zero`
(zero-based, C-style, "indices start at 0") or `oneapi::mkl::index_base::one` (one-based,
Fortran-style, "indices start at 1"). The `slice` type (VM, not BLAS) provides `slice()` — "Default
constructor equivalent to `slice(0, 0, 0)`" — `slice(std::size_t start, std::size_t size,
std::int64_t stride)`, and `slice(const slice& other)`; strides may be positive, zero, or negative.

### Scalar arguments (`value_or_pointer<T>`)

USM routines accept "either a scalar (for example `float`) or pointer (`float*`) for parameters that
represent a single fixed value (not a vector or matrix). These parameters are often named `alpha` or
`beta` in BLAS." Mixing is legal, e.g. `gemv(queue, trans, m, n, alpha_ptr, lda, x, incx, 2.7, y,
incy)`. The mechanism is "a templated `value_or_pointer<T>` wrapper" in `oneapi::mkl::`, defined in
`oneapi/mkl/types.hpp`. "In general, users should not explicitly use this type in their code. There is
no need to construct an object of type `value_or_pointer` in order to use the oneMKL functions that
include it in their function signatures." Values/pointers are implicitly converted.

**Dereferencing timing (critical):** "For a USM-managed pointer, it is dereferenced at kernel launch
after the dependencies passed to the function have been resolved, so the value may be assigned
asynchronously in another event passed as a dependency to the routine. A raw pointer (such as those
allocated with `malloc` or `new`) is dereferenced at the function call, so it must be valid when the
function is called. In this case the data must be valid when the function is called and it may not be
assigned asynchronously." Scalar pointers "may be SYCL-managed pointers to either device or host
memory (for example pointers created with `sycl::malloc_device`, `sycl::malloc_shared`, or
`sycl::malloc_host`), or they may be raw pointers created with `malloc` or `new`."

Level 1 routines whose USM signature takes `value_or_pointer`: `axpy` (`alpha`), `rot` (`c`, `s`),
`rotmg` (`y1`), `scal` (`alpha`). Buffer versions of `axpy`, `rot`, `scal` take plain values.

### Vector and matrix storage

A vector `X` of `n` elements with increment `incx` "is represented as a one dimensional array `x` of
size at least `(1 + (n - 1) * abs(incx))`"; every Level 1 routine repeats this for its vector
arguments. Negative increments reverse traversal (see p. 27 figures in Formulas).

Matrix rules (quoted for completeness; used by Level 2/3): "These are the same formats used in
traditional Fortran BLAS/LAPACK."

- **General** `m` x `n`, leading dimension `lda`, size at least `lda * n` (column major) or `lda * m`
  (row major); elements of each column (row) are contiguous, elements of each row (column) are `lda`
  apart.
- **Triangular** `n` x `n`, size at least `lda * n`; for `uplo::upper` the leading `n` by `n` upper
  triangular part holds `A` and "the strictly lower triangular part of the array `a` is not
  referenced"; for `uplo::lower` the converse.
- **Band** with `kl` sub-diagonals and `ku` super-diagonals, size at least `lda * n` (column major) or
  `lda * m` (row major); the leading `(kl + ku + 1)` by `n` (respectively `m`) part holds `A`, "with
  the main diagonal of the matrix in row `ku` (respectively, column `kl`) of the array (0-based
  indexing), the first super-diagonal starting at position 1 (respectively, 0) in row (`ku - 1`)
  (respectively, column (`kl + 1`)), the first sub-diagonal starting at position 0 (respectively, 1)
  in row (`ku + 1`) (respectively, column (`kl - 1`))"; other entries "are not referenced".
- **Triangular band**, size at least `lda * n`; `uplo::upper` stores the upper band in the leading
  `(k + 1)` by `n` part with the main diagonal in row `k` (column major) or column 0 (row major);
  `uplo::lower` puts the main diagonal in row 0.
- **Packed triangular**, size at least `(n*(n + 1))/2`, selected triangle stored contiguously.

### Error handling

"oneMKL error handling relies on the mechanism of C++ exceptions. Should errors occur, they are
propagated at the point of a function call where they are caught using standard C++ error handling
mechanisms." `oneapi::mkl::exception` is the base class, derived from `std::exception`; "All other
oneMKL exception classes are derived from this base class."

| Exception class | Description (verbatim) |
|---|---|
| `oneapi::mkl::unsupported_device` | "Reports a problem when the routine is not supported on a specific device" |
| `oneapi::mkl::host_bad_alloc` | "Reports a problem that occurred during memory allocation on the host" |
| `oneapi::mkl::device_bad_alloc` | "Reports a problem that occurred during memory allocation on a specific device" |
| `oneapi::mkl::unimplemented` | "Reports a problem when a specific routine has not been implemented for the specified parameters" |
| `oneapi::mkl::invalid_argument` | "Reports problem when arguments to the routine were rejected" |
| `oneapi::mkl::uninitialized` | "Reports problem when a handle (descriptor) has not been initialized" |
| `oneapi::mkl::computation_error` | "Reports any computation error that occurred inside the oneMKL routine" |
| `oneapi::mkl::batch_error` | "Reports errors that occurred inside batch oneMKL routines" |

### Known limitations

"oneMKL DPC++ routines currently may not work with the following SYCL extensions."

- **Discard Events**: "Using oneMKL DPC++ routines with queues created with the `discard_events`
  property is not supported and may lead to unexpected exceptions being thrown."
- **SYCL Graph**: "Recording oneMKL DPC++ routines with the SYCL Graph extension is not supported.
  Unexpected behavior may occur."

### Level 1 groups and data types

| Group | Data types | Description |
|---|---|---|
| `asum` | `float`, `double`, mixed `float` and `std::complex<float>`, mixed `double` and `std::complex<double>` | Sum of vector magnitudes |
| `axpy` | `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `std::complex<float>`, `std::complex<double>` | Scalar-vector product |
| `copy` | `float`, `double`, `std::complex<float>`, `std::complex<double>` | Copy vector |
| `dot` | `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, mixed `float` and `double` | Dot product |
| `sdsdot` | mixed `float` and `double` | Dot product with double precision |
| `dotc` | `std::complex<float>`, `std::complex<double>` | Dot product conjugated |
| `dotu` | `std::complex<float>`, `std::complex<double>` | Dot product unconjugated |
| `nrm2` | `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, mixed `float` and `std::complex<float>`, mixed `double` and `std::complex<double>` | Vector 2-norm (Euclidean norm) |
| `rot` | `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, mixed `float` and `std::complex<float>`, mixed `double` and `std::complex<double>` | Plane rotation of points |
| `rotg` | `float`, `double`, `std::complex<float>`, `std::complex<double>` | Generate Givens rotation of points |
| `rotm` | `float`, `double` | Modified Givens plane rotation of points |
| `rotmg` | `float`, `double` | Generate modified Givens plane rotation of points |
| `scal` | `sycl::half`, `oneapi::mkl::bfloat16`, `float`, `double`, `std::complex<float>`, `std::complex<double>`, mixed `float` and `std::complex<float>`, mixed `double` and `std::complex<double>` | Vector-scalar product |
| `swap` | `float`, `double`, `std::complex<float>`, `std::complex<double>` | Vector-vector swap |
| `iamax` | `float`, `double`, `std::complex<float>`, `std::complex<double>` | Index of the maximum absolute value element |
| `iamin` | `float`, `double`, `std::complex<float>`, `std::complex<double>` | Index of the minimum absolute value element |

## Routines

Each block below gives the Buffer form followed by the USM form of one routine in namespace
`oneapi::mkl::blas::column_major`; the `row_major` namespace prints the same declaration (only
whitespace differs). Parameter order, types, and default arguments are copied from the source.

### asum

"Computes the sum of magnitudes of the vector elements": "the sum of the magnitudes of elements of a
real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector."
Precisions: `T`/`Tres` = `float`/`float`, `double`/`double`, `std::complex<float>`/`float`,
`std::complex<double>`/`double`.

```cpp
void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<Tres,1> &result)

sycl::event asum(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, Tres *result,
                 const std::vector<sycl::event> &dependencies = {})
```

- `queue` — "The queue where the routine should be executed."
- `n` — "Number of elements in vector x."
- `x` — input vector; "Size of the buffer must be at least (1 + (n - 1)*abs(incx))" (USM: "Size of the
  array holding vector x must be least (1 + (n - 1)*abs(incx))").
- `incx` — "Stride of vector x."
- `result` (output) — "Buffer where the scalar result is stored." / "Pointer to where the scalar
  result is stored."
- `dependencies` — USM only; defaults to no dependencies. No NOTES or error conditions are stated.

### axpy

"Computes a vector-scalar product and adds the result to a vector." Precisions (`T`): `sycl::half`,
`oneapi::mkl::bfloat16`, `float`, `double`, `std::complex<float>`, `std::complex<double>`.

```cpp
void axpy(sycl::queue &queue, std::int64_t n, T alpha, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy)

sycl::event axpy(sycl::queue &queue, std::int64_t n, oneapi::mkl::value_or_pointer<T> alpha,
                 const T *x, std::int64_t incx, T *y, std::int64_t incy,
                 const std::vector<sycl::event> &dependencies = {})
```

- `alpha` — "Specifies scalar alpha." (USM: "Specifies the scalar alpha. See Scalar Arguments for more
  information on the `value_or_pointer` data type.")
- `x`, `y` — input vectors; sizes at least `(1 + (n - 1)*abs(incx))` and `(1 + (n - 1)*abs(incy))`.
- `y` (output) — "Buffer holding updated vector y." / "Pointer to updated vector y."
- Example: `share/doc/mkl/examples/sycl/blas/source/axpy.cpp`. No other notes stated.

### copy

"Copies a vector to another vector." Precisions (`T`): `float`, `double`, `std::complex<float>`,
`std::complex<double>`.

```cpp
void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy)

sycl::event copy(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, T *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies = {})
```

- `x` input, `y` output ("Buffer holding updated vector y." / "Pointer to updated vector y.").
- Sizes as above for `x`/`y`. Aliasing/in-place behavior is **not** stated.

### dot

"Computes the dot product of two real vectors." Precisions (`T`/`Tres`): `sycl::half`/`sycl::half`,
`oneapi::mkl::bfloat16`/`oneapi::mkl::bfloat16`, `float`/`float`, `double`/`double`, `float`/`double`.
"NOTE For mixed precision version (inputs are `float` while result is `double`), dot product is
computed with double precision."

```cpp
void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
         sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<Tres,1> &result)

sycl::event dot(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, const T *y,
                std::int64_t incy, Tres *result,
                const std::vector<sycl::event> &dependencies = {})
```

- `n` — "Number of elements in vectors x and y."
- `result` (output) — "Buffer where the result (a scalar) will be stored." / "Pointer to where the
  result (a scalar) will be stored." No error conditions stated.

### sdsdot

"Computes a vector-vector dot product with double precision." No precision table is given; the
signature is `float`-typed (group table says mixed `float` and `double`).

```cpp
void sdsdot(sycl::queue &queue, std::int64_t n, float sb, sycl::buffer<float,1> &x,
            std::int64_t incx, sycl::buffer<float,1> &y, std::int64_t incy,
            sycl::buffer<float,1> &result)

sycl::event sdsdot(sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
                   const float *y, std::int64_t incy, float *result,
                   const std::vector<sycl::event> &dependencies = {})
```

- `sb` — "Single precision scalar to be added to the dot product."
- `result` (output) — "Buffer where the result (a scalar) will be stored. **If `n < 0` the result is
  `sb`.**"
- Buffer `y` size text reads `abs(incxy)` verbatim; USM says `abs(incy)`.

### dotc

"Computes the dot product of two complex vectors, conjugating the first vector." Precision (`T`, also
the result type): `std::complex<float>`, `std::complex<double>`.

```cpp
void dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &result)

sycl::event dotc(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *result,
                 const std::vector<sycl::event> &dependencies = {})
```

- `n` — "The number of elements in vectors x and y."
- `x`/`y` buffers "must be at least (1 + (n - 1)*abs(incx))" / `(1 + (n - 1)*abs(incy))`.
- `result` (output) — "The buffer where the result (a scalar) is stored." / "The pointer to where the
  result (a scalar) is stored."

### dotu

"Computes the dot product of two complex vectors" (unconjugated). Precision (`T`, also result type):
`std::complex<float>`, `std::complex<double>`.

```cpp
void dotu(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &result)

sycl::event dotu(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, const T *y,
                 std::int64_t incy, T *result,
                 const std::vector<sycl::event> &dependencies = {})
```

- `n` — "Number of elements in vectors x and y."
- `y` buffer "must have size at least (1 + (n - 1)*abs(incy))".
- `result` (output) — "Buffer where the result (a scalar) is stored." / "Pointer to where the result
  (a scalar) is stored."

### nrm2

"Computes the Euclidean norm of a vector." Precisions (`T`/`Tres`): `sycl::half`/`sycl::half`,
`oneapi::mkl::bfloat16`/`oneapi::mkl::bfloat16`, `float`/`float`, `double`/`double`,
`std::complex<float>`/`float`, `std::complex<double>`/`double`.

```cpp
void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<Tres,1> &result)

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx, Tres *result,
                 const std::vector<sycl::event> &dependencies = {})
```

- `x` input, size at least `(1 + (n - 1)*abs(incx))`; `incx` stride.
- `result` (output) — "Buffer where the Euclidean norm of the vector x is stored." / "Pointer to where
  the Euclidean norm of the vector x is stored."

### rot

"Performs rotation of points in the plane." "Given two vectors x and y of n elements, the rot
routines compute four scalar-vector products and update the input vectors with the sum of two of
these scalar-vector products." Precisions (`T`, `Tc`, `Ts`): `sycl::half` x3;
`oneapi::mkl::bfloat16` x3; `float` x3; `double` x3;
`std::complex<float>`/`float`/`std::complex<float>`;
`std::complex<double>`/`double`/`std::complex<double>`; `std::complex<float>`/`float`/`float`;
`std::complex<double>`/`double`/`double`.

```cpp
void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
         sycl::buffer<T,1> &y, std::int64_t incy, Tc c, Ts s)

sycl::event rot(sycl::queue &queue, std::int64_t n, T *x, std::int64_t incx, T *y,
                std::int64_t incy, oneapi::mkl::value_or_pointer<Tc> c,
                oneapi::mkl::value_or_pointer<Ts> s,
                const std::vector<sycl::event> &dependencies = {})
```

- `c`, `s` — "Scaling factor." (USM: see Scalar Arguments for `value_or_pointer`.)
- `x`, `y` are both updated ("Buffer holding updated buffer x." / "Pointer to updated vector x.");
  in the USM form they are non-const `T *`. No error conditions stated.

### rotg

"Computes the parameters for a Givens rotation." "Given the Cartesian coordinates (a, b) of a point,
the rotg routines return the parameters c, s, r, and z associated with the Givens rotation. The
parameters c and s define a unitary matrix such that:" `[c s; -s c]*[a; b] = [r; 0]`. "The parameter z
is defined such that if |a| > |b|, z is s; otherwise if c is not 0 z is 1/c; otherwise z is 1."
Precisions (`T`/`Tc`): `float`/`float`, `double`/`double`, `std::complex<float>`/`float`,
`std::complex<double>`/`double`.

```cpp
void rotg(sycl::queue &queue, sycl::buffer<T,1> &a, sycl::buffer<T,1> &b, sycl::buffer<Tc,1> &c,
          sycl::buffer<T,1> &s)

sycl::event rotg(sycl::queue &queue, T *a, T *b, Tc *c, T *s,
                 const std::vector<sycl::event> &dependencies = {})
```

- Inputs: `a` — "Buffer holding x-coordinate of the point." / "Pointer to x-coordinate of the point.";
  `b` — "Buffer holding y-coordinate of the point."
- Outputs: `a` — "parameter r"; `b` — "parameter z"; `c` — "parameter c"; `s` — "parameter s"
  ("associated with the Givens rotation" in each case). `a`, `b`, `s` are in/out; `c` is output.

### rotm

"Performs modified Givens rotation of points in the plane." "Given two vectors x and y, each vector
element of these vectors is replaced as follows: for i from 1 to n, where H is a modified Givens
transformation matrix." Precision (`T`): `float`, `double`.

```cpp
void rotm(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy, sycl::buffer<T,1> &param)

sycl::event rotm(sycl::queue &queue, std::int64_t n, T *x, std::int64_t incx, T *y,
                 std::int64_t incy, const T *param,
                 const std::vector<sycl::event> &dependencies = {})
```

- `param` — "Buffer holding an array of size 5. The elements of the `param` array are: `param[0]`
  contains a switch, `flag`. The other array elements `param[1-4]` contain the components of the
  modified Givens transformation matrix `H`: `h11`, `h21`, `h12`, and `h22`, respectively."
- `flag` cases (figures in Formulas): `-1.0` → full `H`; `0.0` → `[1.0 h12; h21 1.0]`; `1.0` →
  `[h11 1.0; -1.0 h22]`; `-2.0` → identity. "In the last three cases, the matrix entries of 1.0,
  -1.0, 0.0 are assumed based on the value of `flag` and are not required to be set in the `param`
  vector."
- `x`, `y` updated in place. No error conditions stated.

### rotmg

"Computes the parameters for a modified Givens rotation." "Given Cartesian coordinates (x1, y1) of an
input vector, the rotmg routines compute the components of a modified Givens transformation matrix H
that zeros the y-component of the resulting vector." Precision (`T`): `float`, `double`.

```cpp
void rotmg(sycl::queue &queue, sycl::buffer<T,1> &d1, sycl::buffer<T,1> &d2, sycl::buffer<T,1> &x1,
           sycl::buffer<T,1>  y1, sycl::buffer<T,1> &param)

sycl::event rotmg(sycl::queue &queue, T *d1, T *d2, T *x1,
                  oneapi::mkl::value_or_pointer<T> y1, T *param,
                  const std::vector<sycl::event> &dependencies = {})
```

- Inputs: `d1` — "Buffer holding the scaling factor for x-coordinate of the input vector."; `d2` —
  "Buffer holding the scaling factor for y-coordinate of the input vector."; `x1` — "Buffer holding
  x-coordinate of the input vector."; `y1` — "Scalar specifying y-coordinate of the input vector."
  (USM adds "See Scalar Arguments for more information on the `value_or_pointer` data type.").
- Outputs: `d1` — "Buffer holding the first diagonal element of the updated matrix."; `d2` — "Buffer
  holding the second diagonal element of the updated matrix."; `x1` — "Buffer holding x-coordinate of
  the rotated vector before scaling"; `param` — "Buffer holding an array of size 5" with `param[0]` the
  switch `flag` and `param[1-4]` "the components of the array `H`: h11, h21, h12, and h22,
  respectively."
- `y1` is **not** listed as an output. The Buffer signature prints `sycl::buffer<T,1>  y1` with no `&`
  in both namespaces (see Explicit gaps).

### scal

"Computes the product of a vector by a scalar." Precisions (`T`/`Ts`): `sycl::half`/`sycl::half`,
`oneapi::mkl::bfloat16`/`oneapi::mkl::bfloat16`, `float`/`float`, `double`/`double`,
`std::complex<float>`/`std::complex<float>`, `std::complex<double>`/`std::complex<double>`,
`std::complex<float>`/`float`, `std::complex<double>`/`double`.

```cpp
void scal(sycl::queue &queue, std::int64_t n, Ts alpha, sycl::buffer<T,1> &x, std::int64_t incx)

sycl::event scal(sycl::queue &queue, std::int64_t n, oneapi::mkl::value_or_pointer<Ts> alpha, T *x,
                 std::int64_t incx, const std::vector<sycl::event> &dependencies = {})
```

- `alpha` — "Specifies the scalar alpha" (USM: see Scalar Arguments).
- `x` (output) — "Buffer holding updated buffer x." / "Pointer to updated array x."; size at least
  `(1 + (n - 1)*abs(incx))`.

### swap

"Swaps a vector with another vector." "Given two vectors of n elements, x and y, the swap routines
return vectors y and x swapped, each replacing the other." Precisions (`T`): `float`, `double`,
`std::complex<float>`, `std::complex<double>`.

```cpp
void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
          sycl::buffer<T,1> &y, std::int64_t incy)

sycl::event swap(sycl::queue &queue, std::int64_t n, T *x, std::int64_t incx, T *y,
                 std::int64_t incy, const std::vector<sycl::event> &dependencies = {})
```

- Outputs: `x` — "Buffer holding updated buffer x, that is, the input vector y."; `y` — "Buffer holding
  updated buffer y, that is, the input vector x." (USM: "Pointer to updated array x, that is, the input
  vector y." / "Pointer to updated array y, that is, the input vector x.").
- `n` — "Number of elements in vectors x and y."

### iamax

"Finds the index of the element with the largest absolute value in a vector." "The iamax routines
return an index i such that x[i] has the maximum absolute value of all elements in vector x (real
variants), or such that (|Re(x[i])| + |Im(x[i])|) is maximal (complex variants)."

`Tresult` is `std::int32_t` or `std::int64_t`; `T` is `float`, `double`, `std::complex<float>`,
`std::complex<double>`.

**NOTES (verbatim):** "The index is zero-based if `base` is set to `index_base::zero` (default) or
one-based if it is set to `index_base::one`. If either `n` or `incx` is not positive, the routine
returns 0, regardless of the base of the index selected. If more than one vector element is found with
the same largest absolute value, the index of the first one encountered is returned. If the vector
contains NaN values, then the routine returns the index of the first NaN."

```cpp
void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<T, 1> &x, std::int64_t incx,
           sycl::buffer<Tresult, 1> &result, index_base base = index_base::zero)

sycl::event iamax(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx,
                  Tresult *result, index_base base = index_base::zero,
                  const std::vector<sycl::event> &dependencies = {})
```

- `base` — "Indicates how the output value is indexed. If omitted, defaults to zero-based indexing."
  Values `oneapi::mkl::index_base::zero` / `oneapi::mkl::index_base::one`.
- `result` (output) — "The buffer where the index i of the maximal element is stored."
- Example: `share/doc/mkl/examples/sycl/blas/source/iamax.cpp`.

### iamin

"Finds the index of the element with the smallest absolute value in a vector". "The iamin routines
return an index i such that x[i] has the minimum absolute value of all elements in vector x (real
variants), or such that (|Re(x[i])| + |Im(x[i])|) is minimal (complex variants)."

`Tresult` is `std::int32_t` or `std::int64_t`; `T` is `float`, `double`, `std::complex<float>`,
`std::complex<double>`.

**NOTES (verbatim):** "The index is zero-based if `base` is set to `index_base::zero` (default) or
one-based if it is set to `index_base::one`. If either `n` or `incx` is not positive, the routine
returns 0, regardless of the base of the index selected. If more than one vector element is found with
the same smallest absolute value, the index of the first one encountered is returned. If the vector
contains NaN values, then the routine returns the index of the first NaN."

```cpp
void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<T,1> &x, std::int64_t incx,
           sycl::buffer<Tresult,1> &result, index_base base = index_base::zero)

sycl::event iamin(sycl::queue &queue, std::int64_t n, const T *x, std::int64_t incx,
                  Tresult *result, index_base base = index_base::zero,
                  const std::vector<sycl::event> &dependencies = {})
```

- `base`, `result` as for `iamax` ("Buffer where the index i of the minimum element is stored." /
  "Pointer to where the index i of the minimum element is stored.").

## Formulas

Transcribed from the source formula images; the red tag `pNNNN#i` gives the PDF page and index.
Storage-layout items are from "Matrix Storage" (pp. 20-27); the rest are the per-routine "The
operation is defined as:" figures.

### Storage layouts

- general matrix (p20#0): `A = [A11 A12 A13 ... A1n; A21 A22 A23 ... A2n; A31 A32 A33 ... A3n; ...; Am1 Am2 Am3 ... Amn]`
- general, column major (p20#1): `a = [A11, A21, ..., Am1, *, ..., * | A12, A22, ..., Am2, *, ..., * | ... | A1n, A2n, ..., Amn, *, ..., *]`, blocks of length `lda`, total `lda x n`
- general, row major (p20#2): `a = [A11, A12, ..., A1n, *, ..., * | A21, A22, ..., A2n, *, ..., * | ... | Am1, Am2, ..., Amn, *, ..., *]`, total `m x lda`
- upper triangular (p21#0): `A = [A11 A12 A13 ... A1n; * A22 A23 ... A2n; * * A33 ... A3n; ...; * * * ... Ann]`
- upper triangular, column major (p21#1): `a = [A11, *, ..., * | A12, A22, *, ..., * | ... | A1n, A2n, A3n, ..., Ann, *, ..., *]`, `lda x n`
- upper triangular, row major (p21#2): `a = [A11, A12, A13, ..., A1n, *, ..., * | A22, A23, ..., A2n, *, ..., * | ... | *, *, ..., Ann, *, ..., *]`, `lda x n`
- lower triangular (p21#3): `A = [A11 * * ... *; A21 A22 * ... *; A31 A32 A33 ... *; ...; An1 An2 An3 ... Ann]`
- lower triangular, column major (p22#0): `a = [A11, A21, A31, ..., An1, *, ..., * | *, A22, A32, ..., An2, *, ..., * | ... | *, ..., *, Ann, *, ..., *]`, `lda x n`
- lower triangular, row major (p22#1): `a = [A11, *, ..., * | A21, A22, *, ..., * | ... | An1, An2, An3, ..., Ann, *, ..., *]`, `lda x n`
- general band matrix (p22#2): `A` (m x n, kl sub-diagonals, ku super-diagonals): row 1 `A11 A12 A13 ... A1,ku+1 * ... *`; row 2 `A21 A22 A23 A24 ... A2,ku+2 * ... *`; row 3 `A31 A32 A33 A34 A35 ... A3,ku+3 * ... *`; ... ; `A_{n-ku,n}`, `A_{m-2,n}`, `A_{m-1,n}`, `A_{m,n}`
- general band, column major (p22#3): `a = [*, ..., *(ku), A11, A12, ..., A_min(kl+1,m),1, *, ..., *, *, ..., *(ku-1), A_max(1,2-ku),2, ..., A_min(kl+2,m),2, *, ..., *, ..., *, ..., *]`; brace labels observed `ku`, `ku-1`, `max(0,ku-n+1)`; total `lda x n`
- general band, row major (p23#0): `a = [*, ..., *(kl), A11, A12, ..., A_1,min(ku+1,n), *, ..., *, *, ..., *(kl-1), A_2,max(1,2-kl), ..., A_2,min(ku+2,n), *, ..., *, ..., *]`; brace labels observed `kl`, `kl-1`, `max(0,kl-m+1)`; total `lda x m`
- upper triangular band matrix (p24#0): `A = [A11 A12 A13 ... A_1,k+1 * ... *; * A22 A23 A24 ... A_2,k+2 * ... *; * * A33 A34 A35 ... A_3,k+3 * ... *; ...; ... A_{n-k,n}; ... A_{n-2,n}; A_{n-1,n}; A_{n,n}]`
- upper triangular band, array figure 1 (p24#1): `a = [*, ..., *(k), A11, *, ..., *, *, ..., *(k-1), A_max(1,2-k),2, ..., A2,2, *, ..., *, ..., *, ..., *(max(0,k-n+1)), A_max(1,n-k),n, ..., An,n, *, ..., *]`, `lda x n`
- upper triangular band, array figure 2 (p24#2): `a = [A11, A21, ..., A_min(k+1,n),1, *, ..., *, A2,2, ..., A_min(k+2,n),2, *, ..., *, ..., An,n, *, ..., *]`, `lda x n`
- lower triangular band matrix (p25#0): `A = [A11 * ... *; A21 A22 * ... *; A31 A32 A33 ... *; ...; A_{k+1,1} ...; * A_{k+2,2} ...; ...; *, ..., *, A_{n,n-k}, ..., A_{n,n-2}, A_{n,n-1}, A_{n,n}]`
- lower triangular band, array figure 1 (p25#1): `a = [A11, A21, ..., A_min(k+1,n),1, *, ..., *, A2,2, ..., A_min(k+2,n),2, *, ..., *, ..., An,n, *, ..., *]`, `lda x n`
- lower triangular band, array figure 2 (p25#2): `a = [*, ..., *(k), A11, *, ..., *, *, ..., *(k-1), A_max(1,2-k),2, ..., A2,2, *, ..., *, ..., *, ..., *(max(0,k-n+1)), A_max(1,n-k),n, ..., An,n, *, ..., *]`, `lda x n`
- upper packed triangular (p26#0): `A = [A11 A12 A13 ... A1n; * A22 A23 ... A2n; * * A33 ... A3n; ...; * * * ... Ann]`
- upper packed array (p26#1): `a = [A11, A12, A22, A13, A23, A33, ..., A_(n-1),n, A_nn]`
- lower packed triangular (p27#0): `A = [A11 * * ... *; A21 A22 * ... *; A31 A32 A33 ... *; ...; An1 An2 An3 ... Ann]`
- lower packed array (p27#1): `a = [A11, A21, A22, A31, A32, A33, ..., A_n,(n-1), A_nn]`
- vector (p27#2): `X = (X1, X2, X3, ..., Xn)`
- vector, `incx > 0` (p27#3): `x = [X1, *, ..., *(incx), X2, *, ..., *, ..., X_(n-1), *, ..., *, Xn]`, length `1 + (n-1) x incx`
- vector, `incx < 0` (p27#4): `x = [Xn, *, ..., *(|incx|), X_(n-1), *, ..., *, ..., X2, *, ..., *, X1]`, length `1 + (1-n) x incx`

### Routine operations

- `asum` (p32): `result <- sum_{i=1..n} ( |Re(x_i)| + |Im(x_i)| )`
- `axpy` (p34): `y <- alpha * x + y`
- `copy` (p36): `y <- x`
- `dot` (p38): `result = sum_{i=1..n} x_i * y_i`
- `dotc` (p41): `result = sum_{i=1..n} conj(x_i) * y_i` (conjugate bar drawn over `x_i`)
- `dotu` (p43): `result = sum_{i=1..n} x_i * y_i`
- `nrm2` (p50): `result = ||x||`
- `rot`, real `s` (p52#0): `[x; y] <- [ c*x + s*y ; -s*x + c*y ]`
- `rot`, complex `s` (p52#1): `[x; y] <- [ c*x + s*y ; -conj(s)*x + c*y ]`
- `rotg` (p55): `[ c  s ; -s  c ] * [ a ; b ] = [ r ; 0 ]`
- `rotm` (p57): `[ x_i ; y_i ] <- H * [ x_i ; y_i ]`, for i from 1 to n
- `rotm`, `flag = -1.0` (p58#0): `H = [ h11  h12 ; h21  h22 ]`
- `rotm`, `flag = 0.0` (p58#1): `H = [ 1.0  h12 ; h21  1.0 ]`
- `rotm`, `flag = 1.0` (p58#2): `H = [ h11  1.0 ; -1.0  h22 ]`
- `rotm`, `flag = -2.0` (p59#0): `H = [ 1.0  0.0 ; 0.0  1.0 ]`
- `rotm` USM, `flag = -1.0` (p60#0): `H = [ h11  h12 ; h21  h22 ]`
- `rotm` USM, `flag = 0.0` (p60#1): `H = [ 1.0  h12 ; h21  1.0 ]`
- `rotm` USM, `flag = 1.0` (p60#2): `H = [ h11  1.0 ; -1.0  h22 ]`
- `rotmg` (p61#0): `H = [ 1.0  0.0 ; 0.0  1.0 ]`
- `rotmg` (p61#1): `[ x1 ; 0 ] = H * [ x1  sqrt(d1) ; y1  sqrt(d2) ]` (transcribed exactly as drawn: right factor is a 2x2 array, left column `x1`,`y1`, right column `sqrt(d1)`,`sqrt(d2)`)
- `rotmg`, `flag = -1.0` (p62#0): `H = [ h11  h12 ; h21  h22 ]`
- `rotmg`, `flag = 0.0` (p63#0): `H = [ 1.0  h12 ; h21  1.0 ]`
- `rotmg`, `flag = 1.0` (p63#1): `H = [ h11  1.0 ; -1.9  h22 ]` (printed `-1.9`; prose says the assumed entries are "1.0, -1.0, and 0.0" — see Explicit gaps)
- `rotmg`, `flag = -2.0` (p63#2): `H = [ 1.0  0.0 ; 0.0  1.0 ]`
- `rotmg` USM, `flag = -1.0` (p64#0): `H = [ h11  h12 ; h21  h22 ]`
- `rotmg` USM, `flag = 0.0` (p65#0): `H = [ 1.0  h12 ; h21  1.0 ]`
- `rotmg` USM, `flag = 1.0` (p65#1): `H = [ h11  1.0 ; -1.0  h22 ]`
- `rotmg` USM, `flag = -2.0` (p65#2): `H = [ 1.0  0.0 ; 0.0  1.0 ]`
- `scal` (p65#3): `x <- alpha * x`
- `sdsdot` (p68): `result = sb + sum_{i=1..n} x_i * y_i`
- `swap` (p70): `[ y ; x ] <- [ x ; y ]`

## Conventions & Gotchas

- **Buffer vs USM**: Buffer entry points return `void` and take `sycl::buffer<T,1> &`; USM entry points
  return `sycl::event` and take `const T *` inputs, `T *` outputs, plus trailing
  `const std::vector<sycl::event> &dependencies = {}`. Keep the returned event if later work depends
  on the result.
- **Scalar dereferencing**: USM-managed scalar pointers are dereferenced "at kernel launch after the
  dependencies passed to the function have been resolved"; raw host pointers (`malloc`/`new`) are
  dereferenced at the call and "may not be assigned asynchronously".
- **Vector sizing**: each vector must have at least `(1 + (n - 1)*abs(incx))` elements; a negative
  stride walks backwards (`x[0]` is `Xn`, p. 27 figures).
- **Integer types**: `n`, `incx`, `incy` are `std::int64_t`; `iamax`/`iamin`'s `Tresult` is
  `std::int32_t` or `std::int64_t`. `lda`/`ldm` appear only in the storage rules, untyped there.
- **Index base**: `iamax`/`iamin` default to `index_base::zero`; `index_base::one` gives Fortran-style
  indices. Non-positive `n` or `incx` yields index 0 "regardless of the base".
- **Ties/NaN**: `iamax`/`iamin` return the first maximal/minimal element on ties, and "the index of the
  first NaN" when NaNs are present.
- **`sdsdot` with `n < 0`**: `result` is `sb`.
- **`rotm`/`rotmg` `param`**: size 5; `param[0]` = `flag`, then `h11`, `h21`, `h12`, `h22` (column
  order, not row order). For `flag` = `0.0`, `1.0`, `-2.0` the implied `1.0`/`-1.0`/`0.0` entries "are
  not required to be set"; only `flag = -1.0` uses all four supplied components.
- **Mixed precision**: `dot` with `float` inputs and `double` result computes in double precision;
  `scal` supports complex `x` with real `Ts`; `asum`/`nrm2` return the real type for complex inputs.
- **`rot` in place**: both `x` and `y` are updated (non-const `T *` in the USM form).
- **Band transfer loops** (verbatim; general band, `kl`/`ku`):
  column major `for (j = 0; j < n; j++) { k = ku – j; for (i = max(0, j – ku); i < min(m, j + kl + 1); i++) { a[(k + i) + j * lda] = matrix[i + j * ldm]; } }`;
  row major `for (i = 0; i < m; i++) { k = kl – i; for (j = max(0, i – kl); j < min(n, i + ku + 1); j++) { a[(k + j) + i * lda] = matrix[j + i * ldm]; } }`.
- **Upper triangular band transfer loops** (verbatim):
  column major `for (j = 0; j < n; j++) { m = k – j; for (i = max(0, j – k); i <= j; i++) { a[(m + i) + j * lda] = matrix[i + j * ldm]; } }`;
  row major `for (i = 0; i < n; i++) { m = –i; for (j = i; j < min(n, i + k + 1); j++) { a[(m + j) + i * lda] = matrix[j + i * ldm]; } }`.
- **Lower triangular band transfer loops** (verbatim):
  column major `for (j = 0; j < n; j++) { m = –j; for (i = j; i < min(n, j + k + 1); i++) { a[(m + i) + j * lda] = matrix[i + j * ldm]; } }`;
  row major `for (i = 0; i < n; i++) { m = k – i; for (j = max(0, i – k); j <= i; j++) { a[(m + j) + i * lda] = matrix[j + i * ldm]; } }`.
- **Enum spelling**: short and long enumerator names are interchangeable (`uplo::U` == `uplo::upper`,
  `transpose::C` == `transpose::conjtrans`, ...).
- **Unsupported SYCL features**: `discard_events` queues, SYCL Graph recording.
- **Exceptions**: all oneMKL errors derive from `oneapi::mkl::exception` (from `std::exception`);
  `invalid_argument` signals rejected arguments.

## Explicit gaps

- **No per-routine "Include Files" blocks** in the extracted text; only the statement that
  `oneapi/mkl/types.hpp` is included by `oneapi/mkl/blas.hpp`.
- **No per-routine NOTES/error text** for `asum`, `axpy`, `copy`, `dot`, `sdsdot`, `dotc`, `dotu`,
  `nrm2`, `rot`, `rotg`, `rotm`, `rotmg`, `scal`, `swap` (only `iamax`/`iamin` have NOTEs). The source
  does not state whether `copy`/`swap` permit aliasing, nor any device restrictions for these routines.
- **`asum` mixed precision**: the group table lists "mixed float"/"mixed double", but the routine's
  `T`/`Tres` table has no `float`→`double` row; the intended combination is not stated.
- **`sdsdot`**: no precision table; Buffer `y` size text reads `abs(incxy)` (verbatim artifact).
- **`rotmg` Buffer signature** prints `sycl::buffer<T,1>  y1` without `&` while the parameter text
  calls `y1` a "Scalar" and omits it from Output Parameters — the source is internally inconsistent.
- **p61#0** (`H = [1.0 0.0; 0.0 1.0]`) appears on the `rotmg` Description page with no explanation of
  its role there.
- **`rotmg` `flag = 1.0` on p63** is printed `H = [ h11  1.0 ; -1.9  h22 ]`, while the prose says the
  assumed entries are "1.0, -1.0, and 0.0" and the same figure reads `-1.0` on pp. 58, 60, 65.
  Transcribed as printed; almost certainly a source typo.
- **Triangular-band storage figures (pp. 24-25)** both print the same two array listings (one with `k`
  leading unreferenced entries, one beginning `A11, A21, ...`) under the "column major"/"row major"
  bullets; the figure-to-layout mapping could not be established. Use the transfer loops and the text
  rules (main diagonal in row `k` for column major, column 0 for row major under `uplo::upper`).
- **Lower packed triangular figure (p27#1)** shows `a = [A11, A21, A22, A31, A32, A33, ...]` (row-wise
  grouping), while the text describes column-by-column packing with "`a[1]` and `a[2]` contain `A21`
  and `A31`". Figure and text disagree for the column-major case; the text is the explicit rule.
- **`offset`** is documented in Data Types but is not used by any Level 1 routine, and no consuming
  routine is named in the extracted text.
- Level 2 groups listed at pp. 72-73 (`gbmv`, `gemv`, `ger`, `gerc`, `geru`, `hbmv`, `hemv`, `her`,
  `her2`, `hpmv`, `hpr`, `hpr2`, `sbmv`, `spmv`, `spr`, `spr2`, `symv`, `syr`, `syr2`, `tbmv`,
  `tbsv`, `tpmv`, `tpsv`, `trmv`, `trsv`) are out of scope for this chapter.
