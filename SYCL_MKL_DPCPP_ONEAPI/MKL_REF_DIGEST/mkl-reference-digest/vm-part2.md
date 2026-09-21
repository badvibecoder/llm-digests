# Vector Math, Part 2: Trigonometry, Special Functions, Modes

This chapter continues the oneMKL Vector Math (VM) domain: the trigonometric families (pi-scaled, degree
and hyperbolic), the special functions (error functions, gamma, exponential integral, Bessel), the rounding
family, the VM service functions (`set_mode`/`get_mode`/`set_status`/`get_status`/`clear_status`/
`create_error_handler`) and the miscellaneous pair-wise VM functions. Coverage is source pages 827-987.
Every routine has a Buffer API and a USM API form; each entry gives the Buffer `n`-form signature and a
clause describing the other three overloads, which differ only in slice parameters and the `depends` vector.

## Overview

### Domain model
- Every math routine is in `namespace oneapi::mkl::vm` and returns `sycl::event` ("Computation end event"); the six service functions (`set_mode`/`get_mode`/`set_status`/`get_status`/`clear_status`/`create_error_handler`) instead return `uint64_t`, `uint8_t`, or `error_handler<T>`.
- All are element-wise over 1-D vectors; `n` is `std::int64_t` and "Specifies the number of elements to be calculated."
- Buffer API takes `sycl::buffer<T> &`; USM API takes `T const *` for inputs (`T *` for outputs). `a` = 1st input vector, `b` (where present) = 2nd input vector, `y` = output vector.
- Slice overloads replace `std::int64_t n` with `oneapi::mkl::slice sa,` and insert `oneapi::mkl::slice sy,` (and `sb`, `sz`) before `mode`: "Slice selector for <x>. See Data Types for a description of the oneMKL slice type." A slice overload has no `n`; extent handling is controlled by the `mode` slice bits.
- `depends` (USM only) is inserted immediately before `mode`: "Vector of dependent events (to wait for input data to be ready). This is an optional parameter. The default is an empty vector."
- `mode` = `oneapi::mkl::vm::mode::not_defined` by default: "Overrides the global VM mode setting for this function call. See set_mode function for possible values and their description."
- `errhandler` = `oneapi::mkl::vm::error_handler<T>` defaulting to `{}`: "Sets local error handling mode for this function call. See the create_error_handler function for arguments and their descriptions. This is an optional parameter. The local error handler is disabled by default." Support is per-routine (see each entry).
- The source page range contains **no Include Files sections**; no header path can be quoted. The only oneMKL include directive present in these chunks is `#include "oneapi/mkl/rng.hpp"`, in the out-of-scope RNG example (which also includes `<iostream>`, `<vector>`, and `<sycl/sycl.hpp>`).

### Signature shapes (verbatim tokens; source line breaks collapsed to one line)
Shape U — unary, `errhandler` present (used by `cospi`, `sinpi`, `tanpi`, `acospi`, `asinpi`, `cosd`, `sind`, `tand`, `cosh`, `sinh`, `acosh`, `atanh`, `erfc`, `erfcx`, `cdfnorm`, `erfinv`, `erfcinv`, `cdfnorminv`, `lgamma`, `tgamma`, `expint1`, `i0`, `i1`, `j0`, `j1`, `y0`, `y1`):
```cpp
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, T const * a, T * y, std::vector<sycl::event> const & depends = {}, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
Shape U0 — unary, no `errhandler` (`atanpi`, `tanh`, `asinh`, `erf`, `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `rint`, `frac`):
```cpp
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, T const * a, T * y, std::vector<sycl::event> const & depends = {}, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Shape B — binary `a`,`b`→`y`, no `errhandler` (`atan2`, `atan2pi`, `copysign`, `fdim`, `fmax`, `fmin`, `maxmag`, `minmag`):
```cpp
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
sycl::event NAME(sycl::queue & exec_queue, std::int64_t n, T const * a, T const * b, T * y, std::vector<sycl::event> const & depends = {}, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Shape B+E — binary with `errhandler`: same as Shape B plus `, oneapi::mkl::vm::error_handler<T> errhandler = {}` after `mode` (only `nextafter`).
Shape S — unary with scalar order `T b` between `a` and `y`, with `errhandler` (only `jn`, `yn`).
Shape U2 — unary with a second output vector, no `errhandler` (only `modf`): the Buffer `n`-form inserts `sycl::buffer<T> & z` after `y` (slice form inserts `oneapi::mkl::slice sz` before `mode`; USM `n`-form inserts `T * z`).

### Precision and device support
Default table, stated for `atan2`, `cospi`, `sinpi`, `tanpi`, `acospi`, `asinpi`, `atanpi`, `atan2pi`, `cosd`, `sind`, `tand`, `cosh`, `sinh`, `tanh`, `acosh`, `asinh`, `atanh`, `erf`, `erfc`, `erfcx`, `cdfnorm`, `erfinv`, `erfcinv`, `cdfnorminv`, `jn`, `yn`, `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `rint`, `modf`, `frac`, `copysign`, `nextafter`, `fdim`, `fmax`, `fmin`, `maxmag`, `minmag`:

| T | Devices supported |
|---|---|
| `sycl::half` | GPU |
| `_Float16` | CPU |
| `float` | CPU and GPU |
| `double` | CPU and GPU |

**CPU-only for `float`/`double`** — `lgamma`, `tgamma`, `expint1`, `i0`, `i1`, `j0`, `j1`, `y0`, `y1`:
`sycl::half` GPU, `_Float16` CPU, `float` CPU, `double` CPU.

## Routines

### atan2
Element-wise four-quadrant arctangent of `a[i] / b[i]`. Shape B.
```cpp
sycl::event atan2(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Slice form: `sycl::event atan2(sycl::queue & exec_queue, sycl::buffer<T> & a, oneapi::mkl::slice sa, sycl::buffer<T> & b, oneapi::mkl::slice sb, sycl::buffer<T> & y, oneapi::mkl::slice sy, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);`
No `errhandler`. Full Argument 1/Argument 2/Result/Error Code table is given for all sign, zero, `QNAN`, `SNAN` combinations. "The atan2(a,b) function does not generate any errors."

### cospi
Element-wise cosine of vector elements multiplied by pi: `cos(pi*a)`. Shape U.
```cpp
sycl::event cospi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1`, `-0 -> +1`, `n + 0.5` (any integer n where representable) `-> +0`; `±inf -> QNAN status::errdom`; `QNAN -> QNAN`; `SNAN -> QNAN`. Fast path `abs(ai) <= 2^22` single / `2^51` double.

### sinpi
Element-wise sine of vector elements multiplied by pi: `sin(pi*a)`. Shape U.
```cpp
sycl::event sinpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> -0`, `+n` positive integer `-> +0`, `-n` negative integer `-> -0`; `±inf -> QNAN status::errdom`; `QNAN -> QNAN`; `SNAN -> QNAN`. Fast path `abs(ai) <= 2^22` single / `2^51` double.

### tanpi
Element-wise tangent of vector elements multiplied by pi: `tan(pi*a)`. Shape U.
```cpp
sycl::event tanpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> +0`, `n` even integer `-> pi*copysign(0.0, n)`, `n` odd integer `-> pi*copysign(0.0, -n)`, `n + 0.5` (n even, representable) `-> +inf`, `n + 0.5` (n odd, representable) `-> -inf`; `±inf -> QNAN status::errdom`. Source defines: "The `copysign(x, y)` function returns the first vector argument x with the sign changed to match that of the second argument y." Fast path `abs(ai) <= 2^13` single / `2^67` double.

### acospi
Element-wise arc cosine of vector elements, divided by pi: `acos(a)/pi`. Shape U.
```cpp
sycl::event acospi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1/2`, `-0 -> +1/2`, `+1 -> +0`, `-1 -> +1`; `|a| > 1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`. USM output description: "Pointer y to the output vector of size n."

### asinpi
Element-wise arcsine of vector elements, divided by pi: `asin(a)/pi`. Shape U.
```cpp
sycl::event asinpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> -0`, `+1 -> +1/2`, `-1 -> -1/2`; `|a| > 1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`.

### atanpi
Element-wise arctangent of vector elements, divided by pi: `atan(a)/pi`. Shape U0 — **no `errhandler`**.
```cpp
sycl::event atanpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `+inf -> +1/2`, `-inf -> -1/2`; `QNAN -> QNAN`; `SNAN -> QNAN`. "The atanpi function does not generate any errors."

### atan2pi
Four-quadrant arctangent of the ratios of corresponding elements of two vectors, divided by pi. Shape B.
```cpp
sycl::event atan2pi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Slice form analogous to `atan2` (`sa`, `sb`, `sy`). No `errhandler`. "the function computers the four-quadrant arctangent of ai/bi, with the result divided by pi" [sic]. Full Argument 1/Argument 2/Result table given. "The atan2pi(a,b) function does not generate any errors."

### cosd
Element-wise cosine of vector elements expressed in degrees: `cos(pi*a/180)`. Shape U.
```cpp
sycl::event cosd(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1`, `-0 -> +1`; `±inf -> QNAN status::errdom`. Fast path `abs(ai) <= 2^24` single / `2^52` double.

### sind
Element-wise sine of vector elements expressed in degrees: `sin(pi*a/180)`. Shape U.
```cpp
sycl::event sind(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> -0`; `±inf -> QNAN status::errdom`. Fast path `abs(ai) <= 2^24` single / `2^52` double.

### tand
Element-wise tangent of vector elements expressed in degrees: `tan(pi*x/180)`. Shape U.
```cpp
sycl::event tand(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1`, `-0 -> +1`; `±inf -> QNAN status::errdom`. Fast path `abs(ai) <= 2^38` single / `2^67` double.

### cosh
Element-wise hyperbolic cosine of vector elements. Shape U.
```cpp
sycl::event cosh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
Threshold limitations on input parameters: half `-Log(FLT16_MAX)-Log2 < a[i] < Log(FLT16_MAX)+Log2`; single `-Log(FLT_MAX)-Log2 < a[i] < Log(FLT_MAX)+Log2`; double `-Log(DBL_MAX)-Log2 < a[i] < Log(DBL_MAX)+Log2`. Values `+0 -> +1`, `-0 -> +1`, `X > overflow -> status::overflow`, `X < -overflow -> status::overflow`. NOTE: "The complex cosh(a) function sets the VM Error Status to `status::overflow` in the case of overflow, that is, when RE(a), IM(a) are finite non-zero numbers, but the real or imaginary part of the exact result is so large that it does not meet the target precision." Identities `cosh(CONJ(a))=CONJ(cosh(a))`, `cosh(-a)=cosh(a)`.

### sinh
Element-wise hyperbolic sine of vector elements. Shape U.
```cpp
sycl::event sinh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
Thresholds use `Log(2)`: half `-Log(FLT16_MAX)-Log(2) < a[i] < Log(FLT16_MAX)+Log(2)`; single `-Log(FLT_MAX)-Log(2) < a[i] < Log(FLT_MAX)+Log(2)`; double `-Log(DBL_MAX)-Log(2) < a[i] < Log(DBL_MAX)+Log(2)`. Values `+0 -> +0`, `-0 -> -0`, `a > overflow -> +inf status::overflow`, `a < -overflow -> -inf status::overflow`. Identities `sinh(CONJ(a))=CONJ(sinh(a))`, `sinh(-a)=-sinh(a)`.

### tanh
Element-wise hyperbolic tangent of vector elements. Shape U0 — **no `errhandler`**.
```cpp
sycl::event tanh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Slice-buffer overload printed as `sycl::event tanh(sycl::queue & exec_queue, sycl::buffer<T> & a, oneapi::mkl::slice sa, sycl::buffer<T> & y, oneapi::mkl::slice sy, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);`. Values `+0 -> +0`, `-0 -> -0`, `+inf -> +1`, `-inf -> -1`; `QNAN -> QNAN`, `SNAN -> QNAN`. Identities `tanh(CONJ(a))=CONJ(tanh(a))`, `tanh(-a)=-tanh(a)`. "The tanh(a) function does not generate any errors."

### acosh
Element-wise inverse hyperbolic cosine (nonnegative) of vector elements. Shape U.
```cpp
sycl::event acosh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+1 -> +0`; `a < +1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`; `QNAN -> QNAN`, `SNAN -> QNAN`. NOTE `acosh(CONJ(a))=CONJ(acosh(a))`.

### asinh
Element-wise inverse hyperbolic sine of vector elements. Shape U0 — **no `errhandler`**.
```cpp
sycl::event asinh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `±inf -> ±inf` (table columns `RE(a)`, `i*IM(a)`); `QNAN -> QNAN`, `SNAN -> QNAN`. Identities `asinh(CONJ(a))=CONJ(asinh(a))`, `asinh(-a)=-asinh(a)`. "The asinh(a) function does not generate any errors."

### atanh
Element-wise inverse hyperbolic tangent of vector elements. Shape U.
```cpp
sycl::event atanh(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+1 -> status::sing`, `-1 -> status::sing`; `|a| > 1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`. NOTE (mangled in source): "atanh(±1±i*0)=±:math:`\infty`±i*0, and status::sing error is generated"; `atanh(CONJ(a))=CONJ(atanh(a))`, `atanh(-a)=-atanh(a)`.

### erf
Element-wise error function of vector elements. Shape U0 — **no `errhandler`** (unlike its siblings).
```cpp
sycl::event erf(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+inf -> +1`, `-inf -> -1`, `QNAN -> QNAN`, `SNAN -> QNAN`. Definition and useful relations in `## Formulas`; the source also shows the figure "erf Family Functions Relationship" (not reproduced).

### erfc
Element-wise complementary error function of vector elements. Shape U.
```cpp
sycl::event erfc(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`a > underflow -> +0 status::underflow`; `+inf -> +0`; `-inf -> +2`; `QNAN -> QNAN`, `SNAN -> QNAN`. USM `errhandler` description references "the ref:create_error_handler function". Source figure "erfc Family Functions Relationship" (not reproduced).

### erfcx
Element-wise scaled complementary error function of vector elements. Shape U.
```cpp
sycl::event erfcx(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+inf -> +0`, `-inf -> +inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. Outputs: "The buffer y containing the output vector of size n." / "Pointer y to the output vector of size n."

### cdfnorm
Element-wise cumulative normal distribution function of vector elements. Shape U.
```cpp
sycl::event cdfnorm(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`a < underflow -> +0 status::underflow`; `+inf -> +1`; `-inf -> +0`; `QNAN -> QNAN`, `SNAN -> QNAN`. Source figure "cdfnorm Family Functions Relationship" (not reproduced).

### erfinv
Element-wise inverse error function of vector elements: `y = erf-1(a)`. Shape U.
```cpp
sycl::event erfinv(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> -0`; `+1 -> +inf status::sing`, `-1 -> -inf status::sing`; `|a| > 1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`. Source figure "erfinv Family Functions Relationship" (not reproduced).

### erfcinv
Element-wise inverse complementary error function of vector elements. Shape U.
```cpp
sycl::event erfcinv(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+1 -> +0`; `+2 -> -inf status::sing`; `-0 -> +inf status::sing`; `+0 -> +inf status::sing`; `a < -0 -> QNAN status::errdom`; `a > +2 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`. Outputs described as "The buffer y containing the output vector of size n." / "Pointer y to the output vector of size n." Source figure "erfcinv Family Functions Relationship" (not reproduced).

### cdfnorminv
Element-wise inverse cumulative normal distribution function of vector elements. Shape U.
```cpp
sycl::event cdfnorminv(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0.5 -> +0`; `+1 -> +inf status::sing`; `-0 -> -inf status::sing`; `+0 -> -inf status::sing`; `a < -0 -> QNAN status::errdom`; `a > +1 -> QNAN status::errdom`; `±inf -> QNAN status::errdom`. Source figure "cdfnorminv Family Functions Relationship" (not reproduced).

### lgamma
Element-wise natural logarithm of the absolute value of the gamma function of vector elements. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event lgamma(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
"Precision overflow thresholds for the lgamma function are beyond the scope of this document. If the result does not meet the target precision, the function sets the VM Error Status to `status::overflow`." Values: `+1 -> +0`, `+2 -> +0`, `+0 -> +inf status::sing`, `-0 -> +inf status::sing`, negative integer `-> +inf status::sing`, `±inf -> +inf`, `a > overflow -> +inf status::overflow`.

### tgamma
Element-wise gamma function of vector elements. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event tgamma(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
"Precision overflow thresholds for the tgamma function are beyond the scope of this document. If the result does not meet the target precision, the function raises sets the VM Error Status to `status::sing`." Values: `+0 -> +inf status::sing`, `-0 -> -inf status::sing`, negative integer `-> QNAN status::errdom`, `-inf -> QNAN status::errdom`, `+inf -> +inf`, `a > overflow -> +inf status::sing`. Slice parameter text here reads "See Data Types for more details."

### expint1
Element-wise exponential integral of vector elements. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event expint1(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
"For positive real values x, this can be written as" `E_1(x)` (see `## Formulas`); "For negative real values x, the result is defined as NAN." Values: `x < +0 -> QNAN status::errdom`, `+0 -> +inf status::sing`, `-0 -> +inf status::sing`, `+inf -> +0`, `-inf -> QNAN status::errdom`.

### i0
Element-wise regular modified cylindrical Bessel function of order 0. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event i0(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1`, `-0 -> +1`, `+inf -> +inf`, `-inf -> +inf`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### i1
Element-wise regular modified cylindrical Bessel function of order 1. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event i1(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> -0`, `+inf -> +inf`, `-inf -> -inf`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### j0
Element-wise Bessel function of the first kind of order 0. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event j0(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +1`, `-0 -> +1`, `±inf -> 0`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### j1
Element-wise Bessel function of the first kind of order 1. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event j1(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> +0`, `-0 -> +0`, `±inf -> 0`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### jn
Bessel function of the first kind of given scalar order `b` for elements of vector `a`. Shape S.
```cpp
namespace oneapi::mkl::vm {
sycl::event jn(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, T b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Buffer slice form: `sycl::event jn(sycl::queue & exec_queue, sycl::buffer<T> & a, oneapi::mkl::slice sa, T b, sycl::buffer<T> & y, oneapi::mkl::slice sy, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});`
USM n-form: `sycl::event jn(sycl::queue & exec_queue, std::int64_t n, T const * a, T b, T * y, std::vector<sycl::event> const & depends = {}, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});`
`b` = "Fixed value of power." Precision/devices: default table (`float`/`double` on CPU **and GPU**). Values `+0 -> +0`, `-0 -> +0`, `±inf -> 0`.

### y0
Element-wise Bessel function of the second kind of order 0. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event y0(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> -inf status::sing`, `-0 -> -inf status::sing`, `+inf -> 0`, `-inf -> 0`, `x < 0` (including `-inf`) `-> SNAN status::errdom`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### y1
Element-wise Bessel function of the second kind of order 1. Shape U. **CPU-only for `float`/`double`.**
```cpp
sycl::event y1(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
`+0 -> -inf status::sing`, `-0 -> -inf status::sing`, `±inf -> 0`, `x < 0` (including `-inf`) `-> SNAN status::errdom`, `QNAN -> QNAN`, `SNAN -> QNAN`.

### yn
Bessel function of the second kind of given scalar order `b` for elements of vector `a`. Shape S (same overload set as `jn`).
```cpp
namespace oneapi::mkl::vm {
sycl::event yn(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, T b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`b` = "Fixed value of power." Precision/devices: default table (`float`/`double` on CPU **and GPU**). Values `+0 -> -inf status::sing`, `-0 -> -inf status::sing`, `±inf -> 0`, `x < 0` (including `-inf`) `-> SNAN status::errdom`.

### floor
Element-wise rounding of vector elements to the nearest integer towards minus infinity. Shape U0 — **no `errhandler`**.
```cpp
sycl::event floor(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `+inf -> +inf`, `-inf -> -inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The floor function does not generate any errors."

### ceil
Element-wise rounding of vector elements to the nearest integer towards plus infinity. Shape U0 — **no `errhandler`**.
```cpp
sycl::event ceil(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `±inf -> ±inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The ceil function does not generate any errors."

### trunc
Element-wise rounding of vector elements to the nearest integer towards zero. Shape U0 — **no `errhandler`**.
```cpp
sycl::event trunc(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `±inf -> ±inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The trunc function does not generate any errors."

### round
Element-wise rounding of vector elements to the nearest integer, breaking ties away from zero. Shape U0 — **no `errhandler`**.
```cpp
sycl::event round(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
"Input elements that are halfway between two consecutive integers are always rounded away from zero regardless of the rounding mode." `+0 -> +0`, `-0 -> -0`, `±inf -> ±inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The round(a) function does not generate any errors." In the extracted text the Buffer API block for `round` lacks its opening `namespace oneapi::mkl::vm {` and the USM blocks use `namespace oneapi::mkl::vm{`.

### nearbyint
Element-wise rounding of vector elements to the nearest integer according to the current rounding mode. Shape U0 — **no `errhandler`**.
```cpp
sycl::event nearbyint(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `±inf -> ±inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The nearbyint function does not generate any errors."

### rint
Element-wise rounding of vector elements to the nearest integer (floating-point result) using the current rounding mode. Shape U0 — **no `errhandler`**.
```cpp
sycl::event rint(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Rounding-mode examples from the source: `f(0.5) = 0` for round to nearest, round toward zero or minus infinity; `f(0.5) = 1` for plus infinity; `f(-1.5) = -2` for round to nearest or minus infinity; `f(-1.5) = -1` for round toward zero or plus infinity. `±0 -> ±0`, `±inf -> ±inf`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The rint function does not generate any errors."

### modf
Element-wise truncated integral and remaining fractional parts of vector elements. Shape U2 — two outputs, **no `errhandler`**.
```cpp
namespace oneapi::mkl::vm {
sycl::event modf(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, sycl::buffer<T> & z, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
Buffer slice form: `sycl::event modf(sycl::queue & exec_queue, sycl::buffer<T> & a, oneapi::mkl::slice sa, sycl::buffer<T> & y, oneapi::mkl::slice sy, sycl::buffer<T> & z, oneapi::mkl::slice sz, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);`
USM n-form: `sycl::event modf(sycl::queue & exec_queue, std::int64_t n, T const * a, T * y, T * z, std::vector<sycl::event> const & depends = {}, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);`
Outputs: `y` = "output vector for truncated integer values", `z` = "output vector for remaining fraction parts"; `sz` = slice selector for `z`. Values `+0 -> +0,+0`; `-0 -> -0,-0`; `+inf -> +inf,+0`; `-inf -> -inf,-0`; `SNAN -> QNAN,QNAN`; `QNAN -> QNAN,QNAN`. "The modf function does not generate any errors."

### frac
Element-wise signed fractional part of vector elements. Shape U0 — **no `errhandler`**.
```cpp
sycl::event frac(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
`+0 -> +0`, `-0 -> -0`, `+inf -> +0`, `-inf -> -0`, `QNAN -> QNAN`, `SNAN -> QNAN`. "The frac function does not generate any errors."

### set_mode
Sets a new mode for VM functions according to `new_mode` and returns the previous VM mode. "set_mode supports the following devices: CPU and GPU."
```cpp
uint64_t set_mode(queue& exec_queue, uint64_t   new_mode )
```
No `namespace` qualifier is printed for this signature. `new_mode` = "Specifies the VM mode to be set."; return value `old_mode` = "Specifies the former VM mode." The change has a global effect on all VM functions within a queue and can be overridden locally by a routine's `mode` parameter. Values are combined with bitwise OR (`|`):

| Value | Description |
|---|---|
| `oneapi::mkl::vm::mode::ha` | High accuracy versions of VM functions. (DEFAULT) |
| `oneapi::mkl::vm::mode::la` | Low accuracy versions of VM functions. |
| `oneapi::mkl::vm::mode::ep` | Enhanced performance accuracy versions of VM functions. |
| `oneapi::mkl::vm::mode::badarg_exception` | Throw a `oneapi::mkl::invalid_argument` exception on invalid arguments. (DEFAULT) |
| `oneapi::mkl::vm::mode::badarg_quiet` | Invalid arguments quietly make the call a "no-op". The VM status is set to `vm::status::empty_computation`. |
| `oneapi::mkl::vm::mode::slice_normal` | Non-equal slice sizes are considered invalid. (DEFAULT) |
| `oneapi::mkl::vm::mode::slice_minimum` | The minimum of all slice sizes defines the number of evaluations. |
| `oneapi::mkl::vm::mode::slice_cyclic` | The output slice(s) size defines the number of evaluations. Input slices wrap around from the start. |
| `oneapi::mkl::vm::mode::not_defined` | VM mode not defined. This has no effect. |

Default if no VM mode is defined or `mode::not_defined` is set: `(mode::badarg_exception | mode::slice_normal | mode::ha)`. Examples: `oldmode = set_mode (exec_queue , mode::la);` and `oldmode = set_mode (exec_queue , mode::ep | mode::ftzdazon);` — `mode::ftzdazon` appears only here and is not defined in the mode tables.

### get_mode
Gets the VM mode for a queue. "get_mode supports the following devices: CPU and GPU."
```cpp
uint64_t get_mode( queue& exec_queue )
```
Return value `old_mode` (source label) = "Specifies the global VM mode"; it is a bitwise OR of the same `mode::` values listed under `set_mode`. Examples: `accm = get_mode (exec_queue) & mode::accuracy_mask;` and `denm = get_mode (exec_queue) & mode::ftzdaz_mask;` — `mode::accuracy_mask` and `mode::ftzdaz_mask` appear only here and are not defined in the tables.

### set_status
Sets the global VM Status to `new_status` and returns the previous VM Status. "set_status supports the following devices: CPU and GPU."
```cpp
uint8_t set_status (queue& exec_queue,uint_8   new_status )
```
The second parameter type is printed as `uint_8` (verbatim; `uint8_t` is intended). "The global VM Status is a single value and it accumulates via bitwise OR ( | ) all errors that happen inside VM functions." Status values: `status::success` "VM function execution completed successfully"; `status::not_defined` "VM status not defined"; `status::accuracy_warning` "VM function execution completed successfully in a different accuracy mode"; `status::errdom` "Values are out of a range of definition producing invalid (QNaN) result"; `status::sing` "Values cause divide-by-zero (singularity) errors and produce and invalid (QNaN or Inf) result"; `status::overflow` "An overflow happened during the calculation process"; `status::underflow` "An underflow happened during the calculation process". Examples: `uint8_t olderr = set_status (exec_queue, status::success);` then `if (olderr & status::errdom) {...}` and `if (olderr & status::sing) {...}`.

### get_status
Gets the VM Status. "get_status supports the following devices: CPU and GPU."
```cpp
uint8_t get_status (queue& exec_queue )
```
Return value `status` = "Specifies the VM status." Same status value table as `set_status`. Examples: `uint8_t err = get_status (exec_queue);` then `if (err & status::errdom) {...}`, `if (err & status::sing) {...}`.

### clear_status
Sets the VM Status to `status::success` and returns the previous VM Status. "clear_status supports the following devices: CPU and GPU."
```cpp
namespace oneapi::mkl::vm {
  uint8_t clear_status
  (queue& exec_queue )
}
```
Return value `old_status` = "Specifies the former VM status." Example: `uint8_t olderr = clear_status (exec_queue);`.

### create_error_handler
Creates the local VM Error Handler for a function.
```cpp
// Buffer API:
namespace oneapi::mkl::vm {
  error_handler<T> create_error_handler(
  buffer<uint8_t, 1> & errarray,
  int64_t length = 1,
  uint8_t errstatus = status::not_defined,
  T fixup = 0.0,
  bool copysign = false )
}
// USM API:
namespace oneapi::mkl::vm {
  error_handler<T> create_error_handler(
  uint8_t* errarray,
  int64_t length = 1,
  uint8_t errstatus = status::not_defined,
  T fixup = 0.0,
  bool copysign = false )
}
```
Precision/devices: `sycl::half` GPU, `_Float16` CPU, `float` CPU and GPU, `double` CPU and GPU. Three modes: **Single status mode** — all errors are written into one status value; at execution end it is either un-changed (no errors) or contains accumulated (bitwise-OR-merged) statuses; "Set the array pointer to any status object and the length equals 1 to enable this mode." **Multiple status mode** — "error statuses are saved as an array by indices where they happen"; only error statuses are written (success statuses are not); the user must allocate **and initialize** the array before execution with the same size as the argument and result vectors, set `errarray` to it and `length` to the vector size. **Fixup mode** — results for arguments that caused a specific error status are overwritten by a user-defined value; set the desired `errstatus` and `fixup`; "If the `copysign` is set to true then fixup value's sign set to the same sign of the argument which caused the `errstatus` – a suitable option for symmetric math functions." Notes: "You must allocate and initialize array errarray before calling VM functions in multiple status error handling mode. The array should be large enough to contain n error codes, where n is the same as inputoutput vector size for the VM function." If no arguments are passed, "the empty object is created with all of three error handling modes disabled. In this case, the VM math functions set the global error status only."
Input parameters: `errarray` "Array to store error statuses (should be a buffer for buffer API)"; `length` "Length of the errarray. This is an optional argument, default value is 1"; the third parameter is named `errstatus` in the signature but described in the parameter list as `errcode` — "Error status to fixup results. This is an optional argument, default value is `status::not_defined`"; `fixup` "Fixup value for results. This is an optional argument, default value is 0.0."; `copysign` "Flag for setting the fixup value's sign the same as the argument's. This is an optional argument, default value false." Output: "Specifies the error handler object to be created." Usage models shown (USM API), each with and without `create_error_handler` (brace-init as the handler argument): `error_handler<float> handler = vm::create_error_handler (st);` / `vm::sin(exec_queue, 1000, a, r, {st });`; `vm::create_error_handler (st, 1000)` / `vm::inv(exec_queue, 1000, a, r, {st, 1000});`; `vm::create_error_handler (nullptr, 0, status::errdom, fixup, true)` with `fixup = 1.0` / `vm::erfinv(exec_queue, 1000, a, r, { nullptr, 0, status::errdom, fixup, true });`; `vm::create_error_handler (st, 1, status::overflow, fixup)` with `fixup = 1e38` / `vm::exp(exec_queue, 1000, a, r, {st, 1, status::overflow, fixup});`; `vm::create_error_handler (st, 1000, status::errdom, fixup)` / `vm::acospi(exec_queue, 1000, a, r,{ st, 1000, status::errdom, fixup});`. No local error handling mode: `vm::pow(exec_queue, n, a, b, r);` then `uint8_t err = vm::get_status (exec_queue);` — "Only global accumulated error status err is set."

### copysign
Element-wise copy of vector `a` elements with the sign of vector `b` elements. Shape B — **no `errhandler`**.
```cpp
sycl::event copysign(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Slice form: `sycl::event copysign(sycl::queue & exec_queue, sycl::buffer<T> & a, oneapi::mkl::slice sa, sycl::buffer<T> & b, oneapi::mkl::slice sb, sycl::buffer<T> & y, oneapi::mkl::slice sy, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);`
Result: any value with positive `b` -> `+any value`; any value with negative `b` -> `-any value`. "The copysign(a,b) function does not generate any errors."

### nextafter
Element-wise next representable floating-point value of vector `a` elements in the direction of vector `b` elements. Shape B+E.
```cpp
sycl::event nextafter(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
```
Error codes: "Input vector argument element is finite and the corresponding result vector element value is infinite" -> `status::overflow`; "Result vector element value is subnormal or zero, and different from the corresponding input vector argument element" -> `status::underflow`. "Even though underflow or overflow can occur, the returned value is independent of the current rounding direction mode."

### fdim
Element-wise positive difference between vector `a` elements and vector `b` elements. Shape B — **no `errhandler`**.
```cpp
sycl::event fdim(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
"returns a vector containing the differences of the corresponding elements of the first and second vector arguments if the first element is larger, and +0 otherwise." `any, QNAN -> QNAN`; `any, SNAN -> QNAN`; `QNAN, any -> QNAN`; `SNAN, any -> QNAN`. "The fdim(a,b) function does not generate any errors."

### fmax
Element-wise maximum of each pair of vector `a` elements and vector `b` elements. Shape B — **no `errhandler`**.
```cpp
sycl::event fmax(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
"if a < b fmax(a,b) returns b, otherwise fmax(a,b) returns a." `a not NAN, NAN -> a`; `NAN, b not NAN -> b`; `NAN, NAN -> NAN`. "The fmax(a,b) function does not generate any errors."

### fmin
Element-wise minimum of each pair of vector `a` elements and vector `b` elements. Shape B — **no `errhandler`**.
```cpp
sycl::event fmin(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
"if a > b fmin(a,b) returns b, otherwise fmin(a,b) returns a." `a not NAN, NAN -> a`; `NAN, b not NAN -> b`; `NAN, NAN -> NAN`. "The fmin(a,b) function does not generate any errors."

### maxmag
Element-wise maximum of each pair of vector `a` elements and vector `b` elements in terms of magnitude. Shape B — **no `errhandler`**.
```cpp
sycl::event maxmag(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Rules: "If |a| > |b| maxmag(a,b) returns a, otherwise maxmag(a,b) returns b. If |b| > |a| maxmag(a,b) returns b, otherwise maxmag(a,b) returns a. Otherwise maxmag(a,b) behaves like fmax." `a not NAN, NAN -> a`; `NAN, b not NAN -> b`; `NAN, NAN -> NAN`. "The maxmag(a,b) function does not generate any errors."

### minmag
Element-wise minimum of each pair of vector `a` elements and vector `b` elements in terms of magnitude. Shape B — **no `errhandler`**.
```cpp
sycl::event minmag(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
```
Rules: "If |a| < |b| minmag(a,b) returns a, otherwise minmag(a,b) returns b. If |b| < |a| minmag(a,b) returns b, otherwise minmag(a,b) returns a. Otherwise minmag behaves like fmin." `a not NAN, NAN -> a`; `NAN, b not NAN -> b`; `NAN, NAN -> NAN`. "The minmag(a,b) function does not generate any errors."

## Formulas
Transcribed from the formula images (tag `pNNNN#i` = PDF page NNNN, index i on that page).
- **erf (p880#0, p894#0, p899#0):** `erf(x) = 2 / sqrt(pi) * integral_0^x e^(-t^2) dt`
- **erfc relation (p880#1, p884#1):** `erfc(x) = 1 - erf(x)` (the p885#1 and p891#1 regions are the relationship diagram, not this formula)
- **Phi (p880#2, p884#2, p894#2):** `Phi(x) = (1/2) * erf(x / sqrt(2))`
- **Phi as integral (p880#3, p884#3, p895#0):** `Phi(x) = -1 / sqrt(2*pi) * integral_0^x exp(-t^2 / 2) dt` (image renders the exponent with a mangled `frac` glyph: `exp(-frac{t^2}{2})dt`; intended exponent `-t^2/2`)
- **Phi inverse (p881#0, p884#4, p895#1):** `Phi^(-1)(x) = sqrt(2) * erf^(-1)(2x - 1)`
- **erf + erfc (p881#2, p885#1, p891#2):** `erf(x) + erfc(x) = 1` (image prints `erf(x) + erfc = 1`)
- **cdfnorm (p881#3, p885#2, p891#0, p891#3):** `cdfnorm(x) = (1/2) * (1 + erf(x / sqrt(2))) = 1 - (1/2) * erfc(x / sqrt(2))` (printed as `1 - frac12erfc(x/sqrt(2))`)
- **erfc (p884#0):** `erfc(x) = 2 / sqrt(pi) * integral_x^inf e^(-t^2) dt`
- **erfcx (p888#0):** `erfcx(x) = exp(x^2) * erfc(x)`
- **CdfNorm (p890#0):** `CdfNorm(x) = 1 / sqrt(2*pi) * integral_(-inf)^x e^(-t^2) dt`
- **inverse error/complementary relation (p894#1):** `erf^(-1)(x) = erfc^(-1)(1 - x)`
- **erfinv (p898#1):** `erfinv(x) = erf^(-1)(x)`
- **erfcinv (p898#0, p895#3, p899#2, p903#1):** `erfcinv(x) = erfinv(1 - x)`
- **cdfnorminv (p895#4, p899#3, p902#1, p903#2):** `cdfnorminv(x) = sqrt(2) * erfinv(2x - 1) = sqrt(2) * erfcinv(2 - 2x)`
- **CdfNormInv (p902#0):** `CdfNormInv(x) = CdfNorm^(-1)(x),`
- **expint1 (p912#0):** `E_1(x) = integral_x^inf (e^(-t) / t) dt = integral_1^inf (e^(-x*t) / t) dt`
- **floor (p936#0):** `y_i = floor(a_i)` (image shows floor brackets around `a_i`)
- **ceil (p938#0):** `y_i = ceil(a_i)` (image shows ceiling brackets around `a_i`)
- **trunc (p941#0):** `a_i >= 0: y_i = floor(a_i)` ; `a_i < 0: y_i = ceil(a_i)`
- **modf (p951#0, p952#0):** `a_i >= 0: y_i = floor(a_i), z_i = a_i - floor(a_i)` ; `a_i < 0: y_i = ceil(a_i), z_i = a_i - ceil(a_i)`
- **frac (p955#0, p955#1):** `y_i = a_i - floor(a_i)` for `a_i >= 0` ; `y_i = a_i - ceil(a_i)` for `a_i < 0` (both p955 regions print the two-branch expression identically and both branches show floor brackets; see gap below)

Images that are diagrams, not formulas (not transcribed): "erf Family Functions Relationship" (p881), "erfc Family Functions Relationship" (p885), "cdfnorm Family Functions Relationship" (p891), "erfinv Family Functions Relationship" (p895), "erfcinv Family Functions Relationship" (p899), "cdfnorminv Family Functions Relationship" (p903), "RNG Manual Offload Structure" (p989).

## Conventions & Gotchas
- **Buffer↔USM mapping is mechanical**: USM replaces `sycl::buffer<T> &` with `T const *` (inputs) / `T *` (outputs) and inserts `std::vector<sycl::event> const & depends = {}` immediately before `mode`. `mode` and `errhandler` defaults are identical in both APIs. The USM slice overloads have no `n`.
- **Slice overloads have no `n`**; the number of evaluations comes from slice extents plus the `mode::slice_normal` (DEFAULT; non-equal slice sizes invalid) / `mode::slice_minimum` / `mode::slice_cyclic` bits. `slice_cyclic`: "The output slice(s) size defines the number of evaluations. Input slices wrap around from the start."
- **Invalid slice arguments** follow `mode::badarg_exception` (throws `oneapi::mkl::invalid_argument`; DEFAULT) or `mode::badarg_quiet` (call becomes a "no-op" and VM status is set to `vm::status::empty_computation`).
- **`mode` is per-queue global and overridable per call**: `set_mode` returns the previous mode, `get_mode` returns the current one, and the effective default is `(mode::badarg_exception | mode::slice_normal | mode::ha)`. `mode::not_defined` on a call means no override.
- **Two error channels**: global status (`get_status`/`set_status`/`clear_status`; a single `uint8_t` accumulated by bitwise OR across VM calls on the queue) and the local `error_handler<T>` passed to routines that support it. With no local handler, "the VM math functions set the global error status only."
- **`errhandler` support is per-routine, not uniform.** Present: `cospi`, `sinpi`, `tanpi`, `acospi`, `asinpi`, `cosd`, `sind`, `tand`, `cosh`, `sinh`, `acosh`, `atanh`, `erfc`, `erfcx`, `cdfnorm`, `erfinv`, `erfcinv`, `cdfnorminv`, `lgamma`, `tgamma`, `expint1`, `i0`, `i1`, `j0`, `j1`, `jn`, `y0`, `y1`, `yn`, `nextafter`. Absent: `atan2`, `atan2pi`, `atanpi`, `tanh`, `asinh`, `erf`, `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `rint`, `modf`, `frac`, `copysign`, `fdim`, `fmax`, `fmin`, `maxmag`, `minmag`. **`erf` is the notable oddity**: it takes `mode` but no `errhandler`, while `erfc`, `erfcx`, `cdfnorm`, `erfinv`, `erfcinv`, `cdfnorminv` all take one.
- **Round-trip identities**: `erf(x) + erfc(x) = 1`; `cdfnorm(x) = (1/2)(1 + erf(x/sqrt(2))) = 1 - (1/2) erfc(x/sqrt(2))`; `erfcinv(x) = erfinv(1 - x)`; `erf^(-1)(x) = erfc^(-1)(1 - x)`; `cdfnorminv(x) = sqrt(2) erfinv(2x - 1) = sqrt(2) erfcinv(2 - 2x)`; `Phi^(-1)(x) = sqrt(2) erf^(-1)(2x - 1)`.
- **Fast computational path thresholds** (performance only): `cospi`/`sinpi` `|a| <= 2^22` single, `2^51` double; `tanpi` `2^13` / `2^67`; `cosd`/`sind` `2^24` / `2^52`; `tand` `2^38` / `2^67`. Outside these, the source recommends VM Enhanced Performance (EP) functions, "fast on the entire function domain. However, these functions provide lower accuracy."
- **`cosh`/`sinh` overflow thresholds** are stated per precision in terms of `Log(FLT16_MAX)`, `Log(FLT_MAX)`, `Log(DBL_MAX)` with `Log2` (`cosh`) or `Log(2)` (`sinh`).
- **`status::sing` vs `status::errdom`**: `status::sing` covers divide-by-zero/singularity (`atanh(±1)`, `tgamma(±0)`, `lgamma(0 and negative integers)`, `y0`/`y1`/`yn(±0)`, `expint1(±0)`, `erfinv(±1)`, `erfcinv(0 or 2)`, `cdfnorminv(0 or 1)`); `status::errdom` covers out-of-domain arguments (`acospi`/`asinpi` `|a|>1`, `acosh(a<1)`, `atanh(|a|>1)`, `tgamma(negative integer)`, `erfinv(|a|>1)`, `erfcinv(a<-0 or a>2)`, `cdfnorminv(a<-0 or a>1)`, `expint1(x<0)`, `y0`/`y1`/`yn(a<0)`, and `±inf` for the pi/degree trig routines).
- **CPU-only for `float` and `double`**: `lgamma`, `tgamma`, `expint1`, `i0`, `i1`, `j0`, `j1`, `y0`, `y1`. `jn` and `yn` **do** support `float`/`double` on CPU and GPU.
- **`sycl::half` / `_Float16`** are supported by every routine listed above (`sycl::half` GPU, `_Float16` CPU).
- **`modf` is the only two-output routine here**: `y` = truncated integral part, `z` = remaining fraction part, each with its own buffer/pointer and slice (`sy`/`sz`).
- **`jn`/`yn` take a scalar order**, named `b`, typed `T`, "Fixed value of power", positioned between `a` (or `sa`) and `y` — not a second vector.
- **Service-function signatures are not fully namespaced in the printed Syntax**: `set_mode`, `get_mode`, `set_status`, `get_status` are printed as free functions taking `queue&` (not `sycl::queue &`); only `clear_status` and `create_error_handler` are printed inside `namespace oneapi::mkl::vm`. Enumerators are written qualified as `oneapi::mkl::vm::mode::...` and `vm::status::...` / `status::...`.
- **Examples** are named per routine under `share/doc/mkl/examples/sycl/vml/source/v<routine>.cpp` (e.g. `vatan2.cpp`, `vcospi.cpp`, `vsinpi.cpp`, `vtanpi.cpp`, `vtand.cpp`, `vcopysign.cpp`, `vnextafter.cpp`, `vfrac.cpp`).
- `copysign` here is the VM routine `oneapi::mkl::vm::copysign(a, b, y)`, not `sycl::copysign`; it takes two input vectors and an explicit output.

## Explicit gaps
- **Include Files**: no Include Files section exists in the extracted text for pages 827-987, so no header path can be quoted verbatim. The only oneMKL include directive present is `#include "oneapi/mkl/rng.hpp"` in the out-of-scope RNG example (alongside `<iostream>`, `<vector>`, `<sycl/sycl.hpp>`).
- **No fast-computational-path threshold is stated** for `acospi`, `asinpi`, `atanpi`, `atan2pi`, `erf`, `erfc`, `erfcx`, `cdfnorm`, `erfinv`, `erfcinv`, `cdfnorminv`, `lgamma`, `tgamma`, `expint1`, `i0`, `i1`, `j0`, `j1`, `y0`, `y1`, `jn`, `yn`, `acosh`, `asinh`, `atanh`, `cosd`-siblings other than those listed above, `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `rint`, `modf`, `frac`, `copysign`, `nextafter`, `fdim`, `fmax`, `fmin`, `maxmag`, `minmag`.
- **`frac` p955 images** (two identical regions, `#0` and `#1`): both branches render with floor brackets, so the `a_i < 0` branch prints `a_i - floor(a_i)` rather than the `a_i - ceil(a_i)` implied by the `-inf -> -0` value table; the first branch's condition renders as `⌊a_i⌋a_i, >= 0` (the condition's `a_i` collides with the closing bracket) and the second as `a < 0` (subscript missing).
- **`atanh` NOTE** in the extracted text is mangled: `atanh(±1±i*0)=±:math:`\infty`±i*0` — the limit value is infinity but the source rendering is broken.
- **`Phi` integrals (p880#3, p884#3, p895#0)** contain a mangled `frac` glyph in the exponent (`exp(-frac{t^2}{2})`); transcribed as `exp(-t^2/2)` and flagged.
- **`expint1`**: the prose says "For negative real values x, the result is defined as NAN" while the result table lists `QNAN` with `status::errdom`; both are reproduced.
- **`set_status` signature** prints the second parameter type as `uint_8` (not a standard type name); reproduced verbatim, `uint8_t` intended.
- **`round` Buffer API block** lacks its opening `namespace oneapi::mkl::vm {` line in the extracted text; treat the routine as `oneapi::mkl::vm::round` like its siblings.
- **`mode::ftzdazon`, `mode::accuracy_mask`, `mode::ftzdaz_mask`** appear only inside `set_mode`/`get_mode` examples and are not defined in the mode value tables in this page range.
- **`status::empty_computation`** is referenced only in the `mode::badarg_quiet` description (`vm::status::empty_computation`) and is not listed in any status value table.
- **`create_error_handler` parameter-name mismatch**: the signature names the third parameter `errstatus`; the Input Parameters list calls it `errcode`. Both spellings reproduced.
- **Out-of-scope material present in the same chunks**: page 987 onward begins the **Random Number Generators** domain — the PRNG definition `(S, mu, f, U, g)`, the categories (Engines / Transformation classes / Generate function / Service routines), the captioned "RNG Manual Offload Structure" figure (p989; its contents are an image and are not in the extracted text), and the oneMKL RNG Usage Model example, whose only concrete RNG identifiers in this page range are `oneapi::mkl::rng::philox4x32x10`, `oneapi::mkl::rng::gaussian` (`gaussian_method::icdf`), and `oneapi::mkl::rng::generate`, plus the prose names `skip_ahead` and `leapfrog`. These are not VM routines and are deliberately not documented in this chapter.
