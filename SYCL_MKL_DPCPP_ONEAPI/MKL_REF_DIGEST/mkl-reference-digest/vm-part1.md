# Vector Math, Part 1: Arithmetic and Power/Log Functions

This chapter covers the oneMKL Vector Mathematics (VM) domain in `namespace oneapi::mkl::vm`, header
`oneapi/mkl/vm.hpp`: the VM model (accuracy `mode`, error handling, special-value notation, buffer/USM/slice
overload families) and, in document order, the Arithmetic Functions (p709–741), Power and Root Functions
(p741–778), Exponential and Logarithmic Functions (p778–804), and the Trigonometric Functions inside the
extracted page range (p804–840), plus the complete VM function-name index from the p704–709 summary table.
Pages 691–702 are the tail of the preceding LAPACK chapter and are reproduced only as a flagged boundary
section. No formula images exist for pages 691–840 (every chunk header says `(none)`; `formula-manifest.json`
has zero entries in the range), so all formulas below were transcribed from inline text; the π symbol was
lost by text extraction and is flagged where it matters.

## Overview

**Namespace, header, model.** All VM routines are in `namespace oneapi::mkl::vm`; "The oneMKL interfaces are
given in: `oneapi/mkl/vm.hpp`" (p703). Every VM routine takes `sycl::queue & exec_queue` first and returns
`sycl::event`, the computation end event. A VM function takes an input vector, computes its function
element-wise, and returns the results in an output vector — except `sincos` (two outputs `y`, `z`) and `cis`
(real input, complex output). There is no scratchpad parameter and no cross-element coupling. "All the VM
mathematical functions can perform in-place operations, where the input and output arrays are at the same
memory locations. For VM mathematical functions with positive increment indexing, in-place operations are
supported only when the input and output increments have the same value" (p703).

**Four overload families per routine** (illustrated with `add`):
```cpp
// (1) Buffer API, count form
namespace oneapi::mkl::vm {
sycl::event add(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
// (2) Buffer API, slice form: `std::int64_t n` is removed and `oneapi::mkl::slice sa` / `sb` / `sy`
//     follow each buffer; `mode` / `errhandler` unchanged.
// (3) USM API, count form: raw pointers instead of buffers and
//     `std::vector<sycl::event> const & depends = {}` inserted immediately before `mode`.
// (4) USM API, slice form: pointers + slices + `depends`.
```
Each routine below prints signature (1) verbatim. Explicitly documented exceptions to that shape:
**no `errhandler` parameter** in any overload of `sqr`, `conj`, `abs`, `arg`, `cbrt`, `pow2o3`, `hypot`,
`atan`, `atan2` (their signatures end at `mode`); **extra scalar `T b`**: `powx`; **four extra scalars**
`T scalea, T shifta, T scaleb, T shiftb`: `linearfrac`; **second output** `sycl::buffer<T> & z` (and
`oneapi::mkl::slice sz` / `T * z`): `sincos`.

**Common parameter meanings (verbatim wording).** `exec_queue` — "The queue where the routine will be
executed." `n` — "Specifies the number of elements to be calculated." `a` — "The buffer containing the input
vector" (two-vector routines: "the 1st input vector"). `b` — "The buffer containing the 2nd input vector."
`sa`, `sb`, `sy`, `sz` — "Slice selector for a/b/y/z. See Data Types for a description of the oneMKL slice
type." `y` — "The buffer containing the output vector" (USM: "Pointer to the output vector"). `mode` —
"Overrides the global VM mode setting for this function call. See set_mode function for possible values and
their description. This is an optional parameter. The default value is `mode::not_defined`." `errhandler` —
"Sets local error handling mode for this function call. See the create_error_handler function for arguments
and their descriptions. This is an optional parameter. The local error handler is disabled by default."
`depends` — "Vector of dependent events (to wait for input data to be ready). This is an optional parameter.
The default is an empty vector." Return value — "Computation end event."

**Special Value Notations (p703).** `z, z1, z2, …` denote complex numbers; `i`, with `i^2 = -1`, is the
imaginary unit; `x, X, x1, x2, …` denote real parts; `y, Y, y1, y2, …` denote imaginary parts; `X` and `Y`
represent any finite positive IEEE-754 floating point values, if not stated otherwise; Quiet NaN and
signaling NaN are `QNAN` and `SNAN`; IEEE-754 positive infinities or floating-point numbers take a `+` sign
before `X`/`Y`, negative ones a `-` sign. The tables show the result for argument `z` at the intersection of
the `RE(z)` column and the `i*IM(z)` row; if the function raises an exception on `z`, the lower part of the
cell shows the raised exception and the VM Error Status; an empty cell means the argument is normal and the
result is defined mathematically.

**Input-range thresholds (p703).** "The input range of parameters is equal to the mathematical range of the
input data type, unless the function description specifies input threshold values, which mark off the
precision overflow": `FLT16_MAX` = maximum representable in half precision real; `FLT_MAX` = maximum in
single precision real; `DBL_MAX` = maximum in double precision real.

**Data-type letters and device support.** The p704–709 summary table labels types `h, s, d, c, z`, but
**those abbreviations are never defined on these pages** (not stated; from context: half, single, double,
complex<float>, complex<double>). Where routines spell out C++ types, support is uniform: `sycl::half` → GPU;
`_Float16` → CPU; `float` → CPU and GPU; `double` → CPU and GPU; `std::complex<float>` → CPU and GPU;
`std::complex<double>` → CPU and GPU. `cis` uses a two-column table `T` / `R`: `float` / `std::complex<float>`
and `double` / `std::complex<double>`, both CPU and GPU. No other device restriction is stated.

**Errors and modes.** `oneapi::mkl::vm::mode` global default `mode::not_defined`; accuracy modes named HA
(High Accuracy), LA (Low Accuracy), EP (Enhanced Performance); status values seen: `status::sing`,
`status::errdom`, `status::overflow`, `status::underflow`, `status::accuracy_warning`; the literal token
`INVALID` appears in the `hypot` error-code column (p775). "Overflow in a complex function occurs (supported
in the HA/LA accuracy modes only) … the function returns `+` in that part of the result, and sets the VM
Error Status to `status::overflow` (overriding any possible `status::accuracy_warning` status)."

## Routines

### Boundary: LAPACK entries at the head of the range (not VM)

Tail of the previous `oneapi::mkl::lapack` chapter, included only because these pages are in range; full
context belongs to the LAPACK chapter. All throw `mkl::lapack::exception` (`info = -i` => i-th parameter
illegal; if `info` equals the passed scratchpad size and `detail()` is non-zero the scratchpad is too small,
required size from `detail()`), and `scratchpad_size` must be >= the value returned by the matching
`*_scratchpad_size` function.

```cpp
// unmqr_scratchpad_size (p691-692, tail) — element count for unmqr (buffer or USM)
template<typename T>
int64_t unmqr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc)

// unmrq (p692-694), buffer form
void unmrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)

// unmrq (USM Version) (p694-696)
sycl::event unmrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})

// unmrq_scratchpad_size (p696-697) — extraction merges `int64_t k,` + `int64_t lda` (see gaps)
template<typename T>
int64_t unmrq_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc)

// unmtr (p697-699), buffer form; the USM version (p699-701) has the same parameter list with
// `const T *a`, `const T *tau`, `T *c`, `T *scratchpad`, and a trailing `const std::vector<sycl::event> &events = {}`
void unmtr(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans, int64_t m, int64_t n,
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)

// unmtr_scratchpad_size (p701-702) — no `m` parameter as extracted (see gaps)
template<typename T>
int64_t unmtr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans,
  int64_t n, int64_t lda, int64_t ldc)
```

`unmqr_scratchpad_size` inputs: `side` = `mkl::side::left` => Q or QT applied to C from the left,
`mkl::side::right` => from the right; `trans` = `mkl::transpose::trans` => multiply C by Q, `nontrans` =>
by QT; `m` rows of A (`0 <= m`); `n` columns of A (`0 <= n <= m`); `k` elementary reflectors defining Q
(`0 <= k <= n`); `lda`, `ldc` leading dimensions. `unmrq` multiplies a complex m-by-n C by Q or QH (buffer
text) / Q or QT (USM text), Q being the unitary matrix of the RQ factorization from `gerqf`
(`Q = H(1)H* H(2)H*...H(k)H`), forming `Q*C`, `QH*C`, `C*Q`, or `C*QH` over C; `T` = `std::complex<float>`,
`std::complex<double>`, **CPU only**; `a` size >= `lda*k`, `c` size >= `ldc*n`; USM returns the output
event. `unmtr` multiplies complex C by Q or QH where Q comes from `hetrd` (`A = Q*T*QH`), same product set
over C, same `T` and CPU-only restriction; `r = m` if `side = side::left`, `r = n` if `side = side::right`;
`uplo` must be `uplo::upper` or `uplo::lower` and match what was given to `hetrd`; `m >= 0`, `n >= 0`,
`lda >= max(1, r)`, tau dimension >= `max(1, r-1)`, `ldc >= max(1, m)`; USM returns the output event.

### add (p709–712)
Element-wise addition of vectors a and b. Include `oneapi/mkl/vm.hpp`.
```cpp
namespace oneapi::mkl::vm {
sycl::event add(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Complex special values per the p709 formula; complex overflow per p709. Example `share/doc/mkl/examples/sycl/vml/source/vadd.cpp`.

### sub (p712–715)
Element-wise subtraction of vector b from vector a.
```cpp
namespace oneapi::mkl::vm {
sycl::event sub(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Same `T`/device list as `add`; special-value table p712–713; complex overflow as for `add`. Example `.../vsub.cpp`.

### sqr (p715–718)
Element-wise squaring of vector a.
```cpp
namespace oneapi::mkl::vm {
sycl::event sqr(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`** in any overload. `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. "The sqr function does not generate any errors." Example `.../vsqr.cpp`.

### mul (p718–721)
Element-wise multiplication of vectors a and b.
```cpp
namespace oneapi::mkl::vm {
sycl::event mul(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Special-value table p718–719 (includes `QNAN`/non-SNAN pairs); complex overflow as for `add`. Example `.../vmul.cpp`.

### mulbyconj (p721–724)
Element-wise multiplication of vector a elements by the complex conjugate of vector b elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event mulbyconj(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `std::complex<float>`, `std::complex<double>` (CPU and GPU) only. Example `.../vmulbyconj.cpp`.

### conj (p724–726)
Element-wise conjugation of vector a.
```cpp
namespace oneapi::mkl::vm {
sycl::event conj(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `std::complex<float>`, `std::complex<double>` (CPU and GPU). "No special values are specified. The conj function does not generate any errors." Example `.../vconj.cpp`.

### abs (p726–729)
Absolute value of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event abs(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Complex special values per `abs(a) = hypot(RE(a),IM(a))`; does not generate errors. Example `.../vabs.cpp`.

### arg (p729–731)
Argument of complex vector elements, in `[-pi, pi]`.
```cpp
namespace oneapi::mkl::vm {
sycl::event arg(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `std::complex<float>`, `std::complex<double>` (CPU and GPU). Special values per `arg(a)=atan2(IM(a),RE(a))`; does not generate errors. Example `.../varg.cpp`.

### linearfrac (p731–735)
Element-wise linear fractional transformation of vectors a and b with scalar parameters.
```cpp
namespace oneapi::mkl::vm {
sycl::event linearfrac(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b,
      T scalea, T shifta, T scaleb, T shiftb, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Scalars sit after `b`/`sb` and before `y`/`sy` in all four overloads. `scalea`/`scaleb` — "Constant value for scaling multipliers of vector a/b"; `shifta`/`shiftb` — "Constant value for shifting addend of vector a/b". `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. **Implemented in the EP accuracy mode only, therefore no special values are defined**; in HA or LA mode it sets `status::accuracy_warning`; correctness is guaranteed only within the p732 thresholds, otherwise the behavior is unspecified. Thresholds relax to no limitation on `a[i]`/`shifta` if `scalea=0`, and on `b[i]`/`shiftb` if `scaleb=0`. Example `.../vllinearfrac.cpp`.

### fmod (p735–738)
Element-wise remainder of a divided by b that has the same sign as the a elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event fmod(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p736 (table verified against the page image): `a` not NAN with `±0` b ⇒ NAN, `status::sing`; `±` a with `b` not NAN ⇒ NAN, `status::sing`; `±0` a with `b≠0` (not NAN) ⇒ `±0`; finite `a` with `±` b ⇒ `a`; NAN propagates. `y` is "the output vector of size n". Example `.../vfmod.cpp`.

### remainder (p738–741)
Element-wise remainder of a divided by b, nearest to zero.
```cpp
namespace oneapi::mkl::vm {
sycl::event remainder(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. `n` is the integer nearest to `ai/bi`; if two integers are equally close, `n` is the even one; if `n` is zero it has the same sign as `ai`. Special values p739 (table verified against the page image): `a` not NAN with `±0` b ⇒ NAN, `status::errdom`; `±` a with `b` not NAN ⇒ NAN; `±0` a with `b≠0` (not NAN) ⇒ `±0`; finite `a` with `±` b ⇒ `a`; NAN propagates. Example `.../vremainder.cpp`.

### inv (p741–744)
Element-wise multiplicative inverse (reciprocal) of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event inv(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p742: `+0` ⇒ `+`, `status::sing`; `-0` ⇒ `-`, `status::sing`; `+` ⇒ `+0`; `-` ⇒ `-0`; `QNAN`/`SNAN` ⇒ `QNAN`. Example `.../vinv.cpp`.

### div (p744–747)
Element-wise division of vector a elements by vector b elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event div(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Special values p744–745: `X>+0` with `+0` ⇒ `+` (`status::sing`); `X>+0`/`-0` ⇒ `-` (`status::sing`); `X<+0` with `+0` ⇒ `-` (`status::sing`); `X<+0`/`-0` ⇒ `+` (`status::sing`); `+0`/`+0` and `-0`/`-0` ⇒ `QNAN` (`status::sing`); `X>+0`/`+` ⇒ `+0`; `X>+0`/`-` ⇒ `-0`; `+`/`+` and `-`/`-` ⇒ `QNAN`; NAN propagates. Complex special values per the p745 formula; complex overflow returns `+` in the affected part and sets `status::overflow`. Example `.../vdiv.cpp`.

### sqrt (p747–750)
Element-wise square root of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event sqrt(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double` (p749 table). Special values p748 (table verified against the page image): `a < +0` ⇒ `QNAN` `status::errdom`; `+0` ⇒ `+0`; `-0` ⇒ `-0`; `-` ⇒ `QNAN` `status::errdom`; `+` ⇒ `+`; `QNAN`/`SNAN` ⇒ `QNAN`; plus a complex-argument block. Identity `Sqrt(CONJ(z)) = CONJ(Sqrt(z))`. Example `.../vsqrt.cpp`.

### invsqrt (p750–753)
Element-wise inverse square root of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event invsqrt(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p751: `a < +0` ⇒ `QNAN` `status::errdom`; `+0` ⇒ `+` `status::sing`; `-0` ⇒ `-` `status::sing`; `-` ⇒ `QNAN` `status::errdom`; `+` ⇒ `+0`; `QNAN`/`SNAN` ⇒ `QNAN`. Example `.../vinvsqrt.cpp`.

### cbrt (p753–755)
Element-wise cube root of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event cbrt(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`, `-0`⇒`-0`, `+`⇒`+`, `-`⇒`-`, `QNAN`⇒`QNAN`, `SNAN`⇒`QNAN`. Example `.../vcbrt.cpp`.

### invcbrt (p756–758)
Element-wise inverse cube root of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event invcbrt(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p756: `+0` ⇒ `+` `status::sing`; `-0` ⇒ `-` `status::sing`; `+` ⇒ `+0`; `-` ⇒ `-0`; `QNAN`/`SNAN` ⇒ `QNAN`. Example `.../vinvcbrt.cpp`.

### pow2o3 (p758–761)
Element-wise cube root of the square of each vector element.
```cpp
namespace oneapi::mkl::vm {
sycl::event pow2o3(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`, `-0`⇒`+0`, `+`⇒`+`, `-`⇒`+`, `QNAN`⇒`QNAN`, `SNAN`⇒`QNAN`. Example `.../vpow203.cpp`.

### pow3o2 (p761–763)
Element-wise square root of the cube of each vector element.
```cpp
namespace oneapi::mkl::vm {
sycl::event pow3o2(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Thresholds on input p761. Special values: `a < +0` ⇒ `QNAN` `status::errdom`; `+0`⇒`+0`; `-0`⇒`-0`; `-` ⇒ `QNAN` `status::errdom`; `+`⇒`+`; `QNAN`/`SNAN`⇒`QNAN`. Example `.../vpow3o2.cpp`.

### pow (p764–768)
Element-wise exponentiation of vector a elements raised to the power of vector b elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event pow(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Range: if `a[i]` is positive then `b[i]` may be arbitrary; for negative `a[i]`, `b[i]` must be an integer (positive or negative). Complex `pow` has no input range limitations. The p764–766 special-value table covers `±0` with negative/positive odd, even and non-integer exponents, `±1` base, `b = 0`, `|a|<1`, `|a|>1`, and NaN cases, with `status::errdom` for invalid combinations. Real overflow (HA/LA only) returns `+` and sets `status::overflow`. **The complex double precision versions are implemented in the EP accuracy mode only; in HA or LA mode they set `status::accuracy_warning`.** Example `.../vpow.cpp`.

### powx (p768–771)
Element-wise exponentiation of vector a elements raised to the constant scalar power `b`.
```cpp
namespace oneapi::mkl::vm {
sycl::event powx(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, T b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`b` = "Fixed value of power." Range rules as for `pow` (positive `a[i]` ⇒ `b` arbitrary; negative `a[i]` ⇒ `b` must be an integer; complex has no input range limitations). "Special values and VM Error Status treatment are the same as for the pow function." `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>`. Example `.../vpowx.cpp`.

### powr (p771–775)
Element-wise `a` to the power `b` where all elements of a are non-negative.
```cpp
namespace oneapi::mkl::vm {
sycl::event powr(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. `ai ≥ 0`. Input thresholds p772. "Special values and VM Error Status treatment for v?Powr function are the same as for pow, unless otherwise indicated in this table" (p772 adds `a < 0` ⇒ `NAN` with `status::errdom`, and further cases). Example `.../vpowr.cpp`.

### hypot (p775–778)
Element-wise square root of the sum of the squares of a and b elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event hypot(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p775: `+0`/`+0`⇒`+0`; `-0`/`-0`⇒`+0`; `+` with any value ⇒ `+`; `SNAN` with any value ⇒ `QNAN`, error code `INVALID`; `any value`/`SNAN` ⇒ `QNAN`, `INVALID`; `QNAN` propagates. Thresholds per data type. "The hypot(a,b) function does not generate any errors." Example `.../vhypot.cpp`.

### exp (p778–781)
Element-wise natural (base-e) exponential of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event exp(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Precision/device table as extracted (p780): `sycl::half` (GPU), `_Float16` (CPU), `float` (CPU and GPU), `double` (CPU and GPU); **complex rows do not appear there**, although the p705 index lists `exp` for `h, s, d, c, z` and the notes describe complex `exp(z)`. Input threshold per data type. Special values: `+0`⇒`+1`; `-0`⇒`+1`; `a > overflow` ⇒ `+`, `status::overflow`; `a < underflow` ⇒ `+0`, `status::overflow` (as printed in the p778 table); `+`⇒`+`; `-`⇒`+0`; NaN propagates. "The complex exp(z) function sets the VM Error Status to `status::overflow` in the case of overflow, that is, when both RE(z) and IM(z) are finite non-zero numbers, but the real or imaginary part of the exact result is so large that it does not meet the target precision." Example `.../vexp.cpp`.

### exp2 (p781–784)
Element-wise base-2 exponential of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event exp2(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Input threshold per data type. Special values: `+0`⇒`+1`; `-0`⇒`+1`; overflow ⇒ `+`, `status::overflow`; underflow ⇒ `+0`, `status::underflow`; `+`⇒`+`; `-`⇒`+0`; NaN propagates. Example `.../vexp2.cpp`.

### exp10 (p784–787)
Element-wise base-10 exponential of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event exp10(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Input threshold per data type. Special values p785: `+0`⇒`+1`; `-0`⇒`+1`; overflow ⇒ `+`, `status::overflow`; underflow ⇒ `+0`, `status::underflow`; `+`⇒`+`; `-`⇒`+0`; NaN propagates. Example `.../vexp10.cpp`.

### expm1 (p787–790)
Element-wise base-e exponential of vector elements decreased by 1.
```cpp
namespace oneapi::mkl::vm {
sycl::event expm1(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Input threshold per data type (p788). Special values: `+0`⇒`+1`; `-0`⇒`+1` (both as printed in the p787 table; mathematically the results should be `+0` and `-0`); `a > overflow` ⇒ `+`, `status::overflow`; `+`⇒`+`; `-`⇒`-0`; NaN propagates. Example `.../vexpm1.cpp`.

### ln (p790–793)
Element-wise natural logarithm of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event ln(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>` (p792 table). Special values p790: `+1`⇒`+0`; `a < +0` ⇒ `QNAN` `status::errdom`; `+0` ⇒ `-` `status::sing`; `-0` ⇒ `-` `status::sing`; `-` ⇒ `QNAN` `status::errdom`; `+`⇒`+`; NaN propagates. Complex table headers exist (p790–791) but cells did not survive extraction. Example `.../vln.cpp`.

### log2 (p793–796)
Element-wise base-2 logarithm of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event log2(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p793: `+1`⇒`+0`; `a < +0` ⇒ `QNAN` `status::errdom`; `+0` ⇒ `-` `status::sing`; `-0` ⇒ `-` `status::sing`; `-` ⇒ `QNAN` `status::errdom`; `+`⇒`+`; NaN propagates. Example `.../vlog2.cpp`.

### log10 (p796–799)
Element-wise base-10 logarithm of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event log10(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`, `std::complex<float>`, `std::complex<double>` (p798 table). Special values p796 (table verified against the page image): `+1`⇒`+0`; `a < +0` ⇒ `QNAN` `status::errdom`; `+0` ⇒ `-` `status::sing`; `-0` ⇒ `-` `status::sing`; `-` ⇒ `QNAN` `status::errdom`; `+`⇒`+`; `QNAN`/`SNAN`⇒`QNAN`. Complex headers only; cells lost. Example `.../vlog10.cpp`.

### log1p (p799–802)
Element-wise natural logarithm of 1 plus vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event log1p(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values p799: `-1` ⇒ `-` `status::sing`; `a < -1` ⇒ `QNAN` `status::errdom`; `+0`⇒`+0`; `-0`⇒`-0`; `-` ⇒ `QNAN` `status::errdom`; `+`⇒`+`; NaN propagates. Example `.../vlog1p.cpp`.

### logb (p802–804)
Element-wise exponents of vector elements: for each `ai`, "the integral part of `log2|ai|`".
```cpp
namespace oneapi::mkl::vm {
sycl::event logb(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. "The returned value is exact and is independent of the current rounding direction mode." Special values p802: `+0` ⇒ `+` `status::errdom`; `-0` ⇒ `-` `status::errdom`; `-`⇒`+`; `+`⇒`+`; `QNAN`/`SNAN`⇒`QNAN`. Example `.../vlogb.cpp`.

### cos (p804–808)
Element-wise cosine of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event cos(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T` per p806–807: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`; the p705 index also lists `c, z` and "Specifications for special values of the complex functions are defined according to the following formula `Cos(z) = Cosh(i*z)`". Fast computational path `abs(a[i]) ≤ 2^13` (single) / `≤ 2^16` (double); avoid arguments outside it in HA/LA, or use EP functions. Special values: `+0`⇒`+1`; `-0`⇒`+1`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Example `.../vcos.cpp`.

### sin (p808–810)
Element-wise sine of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event sin(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T` per p809: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`; index lists `h, s, d, c, z`, complex per `Sin(z) = -i*Sinh(i*z)`. Fast path `abs(a[i]) ≤ 2^13` (single) / `≤ 2^16` (double) with the same HA/LA-vs-EP guidance. Special values: `+0`⇒`+0`; `-0`⇒`-0`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Example `.../vsin.cpp`.

### sincos (p810–813)
Element-wise sine and cosine of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event sincos(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, sycl::buffer<T> & z, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Slice-buffer form adds `sa`, `sy`, `sz`; USM forms use `T const * a, T * y, T * z` and add `sa`, `sy`, `sz` in the slice variant. **Two outputs:** `y` = "the buffer containing the output sine vector"; `z` = "the buffer containing the output cosine vector". `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Fast path as for `cos`/`sin`. Special values p811: `+0`⇒(`+0`,`+1`); `-0`⇒(`-0`,`+1`); `+`/`-`⇒(`QNAN`,`QNAN`) `status::errdom`; NaN propagates. Example `.../vsincos.cpp`.

### cis (p813–816)
Element-wise complex exponential of real vector elements (cosine and sine combined into a complex value).
```cpp
namespace oneapi::mkl::vm {
sycl::event cis(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
Precision table columns `T` / `R`: `float` ⇒ `std::complex<float>`, `double` ⇒ `std::complex<double>`, both CPU and GPU. Special values p814: `+0` ⇒ `+1+i·0`; `-0` ⇒ `+1-i·0`; `+`/`-` ⇒ `QNAN+i·QNAN` `status::errdom`; `QNAN` ⇒ `QNAN+i·QNAN`; `SNAN` ⇒ `QNAN+i·QNAN`. Example `.../vcis.cpp`.

### tan (p816–819)
Element-wise tangent of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event tan(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T` per p818: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`; index lists `h, s, d, c, z`, complex per `Tan(z) = -i*Tanh(i*z)`. Fast path as for `cos`/`sin`. Special values: `+0`⇒`+0`; `-0`⇒`-0`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Example `.../vtan.cpp`.

### acos (p819–822)
Element-wise arccosine of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event acos(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T` per p821: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`; index lists `h, s, d, c, z` and the complex identity `acos(CONJ(a))=CONJ(acos(a))`. Special values: `+0`⇒`pi/2`; `-0`⇒`pi/2`; `+1`⇒`+0`; `-1`⇒`pi`; `|a| > 1` ⇒ `QNAN` `status::errdom`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Complex table cells largely lost (p819–820). Example `.../vacos.cpp`.

### asin (p822–825)
Element-wise arcsine of vector elements.
```cpp
namespace oneapi::mkl::vm {
sycl::event asin(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`; `-0`⇒`-0`; `+1`⇒`pi/2`; `-1`⇒`-pi/2`; `|a| > 1` ⇒ `QNAN` `status::errdom`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Complex per `asin(a) = -i*asinh(i*z)`. Example `.../vasin.cpp`.

### atan (p825–827)
Element-wise arctangent of vector elements in `[-pi/2, pi/2]`.
```cpp
namespace oneapi::mkl::vm {
sycl::event atan(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`; `-0`⇒`-0`; `+`⇒`pi/2`; `-`⇒`-pi/2`; NaN propagates. "The atan function does not generate any errors." Complex per `atan(a) = -i*atanh(i*a)`. Example `.../vatan.cpp`.

### atan2 (p827–831)
Element-wise four-quadrant arctangent of `a[i] / b[i]`.
```cpp
namespace oneapi::mkl::vm {
sycl::event atan2(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & b, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined);
}
```
**No `errhandler`.** `T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. The p828–829 table is organized by argument signs and returns multiples of `pi/4`, `pi/2`, `pi`, with `a > +0` / `b > +0` rows for `QNAN` and `SNAN`. "The atan2(a,b) function does not generate any errors." Example `.../vatan2.cpp`.

### cospi (p831–834)
Element-wise cosine of vector elements multiplied by pi.
```cpp
namespace oneapi::mkl::vm {
sycl::event cospi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+1`; `-0`⇒`+1`; `n + 0.5` (any integer `n` where `n + 0.5` is representable) ⇒ `+0`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Fast path `abs(ai) ≤ 2^22` (single) / `≤ 2^51` (double). Example `.../vcospi.cpp`.

### sinpi (p834–837)
Element-wise sine of vector elements multiplied by pi.
```cpp
namespace oneapi::mkl::vm {
sycl::event sinpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`; `-0`⇒`-0`; `+n` positive integer ⇒ `+0`; `-n` negative integer ⇒ `-0`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. Fast path `abs(ai) ≤ 2^22` (single) / `≤ 2^51` (double). Example `.../vsinpi.cpp`.

### tanpi (p837–840)
Element-wise tangent of vector elements multiplied by pi.
```cpp
namespace oneapi::mkl::vm {
sycl::event tanpi(sycl::queue & exec_queue, std::int64_t n, sycl::buffer<T> & a, sycl::buffer<T> & y, oneapi::mkl::vm::mode mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<T> errhandler = {});
}
```
`T`: `sycl::half` (GPU), `_Float16` (CPU), `float`, `double`. Special values: `+0`⇒`+0`; `-0`⇒`+0`; `n` even integer ⇒ `*copysign(0.0, n)` (leading `pi*` lost in extraction); `n` odd integer ⇒ `*copysign(0.0, -n)`; `n + 0.5` for even representable `n` ⇒ `+`, for odd representable `n` ⇒ `-`; `+`/`-` ⇒ `QNAN` `status::errdom`; NaN propagates. `copysign(x, y)` "returns the first vector argument x with the sign changed to match that of the second argument y." Fast path `abs(ai) ≤ 2^13` (single) / `≤ 2^67` (double), as printed on p838. Example `.../vtanpi.cpp`.

### Complete VM function-name index (p704–709 summary table)

Names, data-type letters, and descriptions verbatim from the summary table. Routines with a full section
above are still listed here so no routine name is dropped. Routines whose full entries fall outside this
page range are grouped at the end of the table.

| Function | Data Types | Description |
|---|---|---|
| `add` | h, s, d, c, z | Adds vector elements |
| `sub` | h, s, d, c, z | Subtracts vector elements |
| `sqr` | h, s, d | Squares vector elements |
| `mul` | h, s, d, c, z | Multiplies vector elements |
| `mulbyconj` | c, z | Multiplies elements of one vector by conjugated elements of the second vector |
| `conj` | c, z | Conjugates vector elements |
| `abs` | h, s, d, c, z | Computes the absolute value of vector elements |
| `arg` | c, z | Computes the argument of vector elements |
| `linearfrac` | h, s, d | Performs linear fraction transformation of vectors |
| `fmod` | h, s, d | Performs element by element computation of the modulus function of vector a with respect to vector b |
| `remainder` | h, s, d | Performs element by element computation of the remainder function on the elements of vector a and the corresponding elements of vector b |
| `inv` | h, s, d | Inverts vector elements |
| `div` | h, s, d, c, z | Divides elements of one vector by elements of the second vector |
| `sqrt` | h, s, d, c, z | Computes the square root of vector elements |
| `invsqrt` | h, s, d | Computes the inverse square root of vector elements |
| `cbrt` | h, s, d | Computes the cube root of vector elements |
| `invcbrt` | h, s, d | Computes the inverse cube root of vector elements |
| `pow2o3` | h, s, d | Computes the cube root of the square of each vector element |
| `pow3o2` | h, s, d | Computes the square root of the cube of each vector element |
| `pow` | h, s, d, c, z | Raises each vector element to the specified power |
| `powx` | h, s, d, c, z | Raises each vector element to the constant power |
| `powr` | h, s, d | Computes a to the power b for elements of two vectors, where the elements of vector argument a are all non-negative |
| `hypot` | h, s, d | Computes the square root of sum of squares |
| `exp` | h, s, d, c, z | Computes the base e exponential of vector elements |
| `exp2` | h, s, d | Computes the base 2 exponential of vector elements |
| `exp10` | h, s, d | Computes the base 10 exponential of vector elements |
| `expm1` | h, s, d | Computes the base e exponential of vector elements decreased by 1 |
| `ln` | h, s, d, c, z | Computes the natural logarithm of vector elements |
| `log2` | h, s, d | Computes the base 2 logarithm of vector elements |
| `log10` | h, s, d, c, z | Computes the base 10 logarithm of vector elements |
| `log1p` | h, s, d | Computes the natural logarithm of vector elements that are increased by 1 |
| `logb` | h, s, d | Computes the exponents of the elements of input vector a |
| `cos` | h, s, d, c, z | Computes the cosine of vector elements |
| `sin` | h, s, d, c, z | Computes the sine of vector elements |
| `sincos` | h, s, d | Computes the sine and cosine of vector elements |
| `cis` | c, z | Computes the complex exponent of vector elements (cosine and sine combined to complex value) |
| `tan` | h, s, d, c, z | Computes the tangent of vector elements |
| `acos` | h, s, d, c, z | Computes the inverse cosine of vector elements |
| `asin` | h, s, d, c, z | Computes the inverse sine of vector elements |
| `atan` | h, s, d, c, z | Computes the inverse tangent of vector elements |
| `atan2` | h, s, d | Computes the four-quadrant inverse tangent of ratios of the elements of two vectors |
| `cospi` | h, s, d | Computes the cosine of vector elements multiplied by pi |
| `sinpi` | h, s, d | Computes the sine of vector elements multiplied by pi |
| `tanpi` | h, s, d | Computes the tangent of vector elements multiplied by pi |

Routines outside this page range, from the same p704–709 table (names and data types verbatim; descriptions
compressed here, full entries in later parts):

- **Trigonometric, continued (h, s, d):** `acospi` inverse cosine divided by pi; `asinpi` inverse sine divided by pi; `atanpi` inverse tangent divided by pi; `atan2pi` four-quadrant inverse tangent of the ratios of the corresponding elements of two vectors divided by pi; `cosd` cosine multiplied by pi/180; `sind` sine multiplied by pi/180 (printed as `sin` in the extracted table); `tand` tangent multiplied by pi/180.
- **Hyperbolic (h, s, d, c, z):** `cosh` hyperbolic cosine; `sinh` hyperbolic sine; `tanh` hyperbolic tangent; `acosh` inverse hyperbolic cosine; `asinh` inverse hyperbolic sine; `atanh` inverse hyperbolic tangent.
- **Special (h, s, d):** `erf` error function value; `erfc` complementary error function value; `erfcx` scaled complementary error function value; `cdfnorm` cumulative normal distribution function value; `erfinv` inverse error function value; `erfcinv` inverse complementary error function value; `cdfnorminv` inverse cumulative normal distribution function value; `lgamma` natural logarithm for the absolute value of the gamma function; `tgamma` gamma function; `expint1` exponential integral; `i0`/`i1` regular modified cylindrical Bessel function of order 0/1; `j0`/`j1` Bessel function of the first kind of order 0/1; `jn` Bessel function of the first kind of a given order; `y0`/`y1` Bessel function of the second kind of order 0/1; `yn` Bessel function of the second kind of a given order.
- **Rounding (h, s, d):** `floor` rounds towards minus infinity; `ceil` rounds towards plus infinity; `trunc` rounds towards zero infinity; `round` rounds to nearest integer; `nearbyint` rounds according to current mode; `rint` rounds according to current mode and raising inexact result exception; `modf` computes the integer and fractional parts; `frac` computes the fractional part.
- **Miscellaneous (h, s, d):** `copysign` returns elements of one argument with signs changed to match other argument elements; `nextafter` returns the next representable floating-point values following one vector's elements toward the corresponding elements of another; `fdim` returns the differences of corresponding elements if the first is larger and +0 otherwise; `fmax`/`fmin` return the larger/smaller of each pair of elements; `maxmag`/`minmag` return the element with the larger/smaller magnitude between each pair of elements.

Referenced by the VM preamble but defined outside this page range (names only, no signature here):
`set_mode`, `create_error_handler`, and the oneMKL `slice` type. Examples directory named in the
introduction: `${MKL}/share/doc/mkl/examples/sycl/vml/source`.

## Formulas

- Special-value notation (p703): `CONJ(x+i*y) = x-i*y`; `CIS(y) = cos(y)+i*sin(y)`; `i^2 = -1`.
- `add` (p709): `add(x1+i*y1, x2+i*y2) = (x1+x2) + i*(y1+y2)`.
- `sub` (p713): `sub(x1+i*y1, x2+i*y2) = (x1-x2) + i*(y1-y2)`.
- `mul` (p719): `mul(x1+i*y1, x2+i*y2) = (x1*x2-y1*y2) + i*(x1*y2+y1*x2)`.
- `abs` (p727): `abs(a) = hypot(RE(a), IM(a))`.
- `arg` (p729): `arg(a) = atan2(IM(a), RE(a))`; result in `[-pi, pi]`.
- `linearfrac` (p732): `y[i] = (scalea*a[i] + shifta) / (scaleb*b[i] + shiftb)`, `i = 1, 2, ... n`.
- `linearfrac` thresholds (p732): `2^(EMIN/2) <= |scalea| <= 2^((EMAX-2)/2)`; `2^(EMIN/2) <= |scaleb| <= 2^((EMAX-2)/2)`; `|shifta| <= 2^(EMAX-2)`; `|shiftb| <= 2^(EMAX-2)`; `2^(EMIN/2) <= a[i] <= 2^((EMAX-2)/2)`; `2^(EMIN/2) <= b[i] <= 2^((EMAX-2)/2)`; `a[i] != -(shifta/scalea)*(1-delta1)` with `|delta1| <= 2^(1-(p-1)/2)`; `b[i] != -(shiftb/scaleb)*(1-delta2)` with `|delta2| <= 2^(1-(p-1)/2)`. Single: `EMIN = -126, EMAX = 127, p = 24`. Double: `EMIN = -1022, EMAX = 1023, p = 53`.
- `fmod` (p735): `ai - bi*trunc(ai/bi)`; generally `fmod(ai, bi) = ai - n*bi` for some integer `n` such that, if `bi` is nonzero, the result has the same sign as `ai` and magnitude less than `|bi|`.
- `remainder` (p738): as printed, "compute the values of n such that `n = ai - n*bi` where `n` is the integer nearest to the exact value of `ai/bi`"; ties go to the even integer; if `n` is zero it has the same sign as `ai`. (Left-hand symbol almost certainly the remainder `r`, lost in extraction — see gaps.)
- `div` complex (p745): `Div(x1+i*y1, x2+i*y2) = (x1+i*y1)*(x2-i*y2)/(x2*x2+y2*y2)`.
- `pow3o2` thresholds (p761): half `|ai| < (FLT16_MAX)^(2/3)`; single `|ai| < (FLT_MAX)^(2/3)`; double `|ai| < (DBL_MAX)^(2/3)`.
- `powr` (p771): `ai >= 0`. Thresholds (p772): half `ai < (FLT16_MAX)^(1/|b_i|)`; single `ai < (FLT_MAX)^(1/|b_i|)`; double `ai < (DBL_MAX)^(1/|b_i|)`.
- `pow` real range (p764): `a[i] > 0` ⇒ `b[i]` arbitrary; `a[i] < 0` ⇒ `b[i]` must be an integer.
- `hypot` thresholds (p775): half `abs(a[i]) < sqrt(FLT16_MAX)`, `abs(b[i]) < sqrt(FLT16_MAX)`; single `abs(a[i]) < sqrt(FLT_MAX)`, `abs(b[i]) < sqrt(FLT_MAX)`; double `abs(a[i]) < sqrt(DBL_MAX)`, `abs(b[i]) < sqrt(DBL_MAX)`.
- `exp` threshold (p778): half `a[i] < Log(FLT16_MAX)`; single `a[i] < Log(FLT_MAX)`; double `a[i] < Log(DBL_MAX)`.
- `exp2` threshold (p781): half `ai < log2(FLT16_MAX)`; single `ai < log2(FLT_MAX)`; double `ai < log2(DBL_MAX)`.
- `exp10` threshold (p784): half `ai < log10(FLT16_MAX)`; single `ai < log10(FLT_MAX)`; double `ai < log10(DBL_MAX)`.
- `expm1` threshold (p787): half `a[i] < Log(FLT16_MAX)`; single `a[i] < Log(FLT_MAX)`; double `a[i] < Log(DBL_MAX)`.
- `sqrt` symmetry (p748): `Sqrt(CONJ(z)) = CONJ(Sqrt(z))`.
- `logb` (p802): for each `ai`, the result is the integral part of `log2|ai|`; exact and independent of the current rounding direction mode.
- `cos` (p805): complex `Cos(z) = Cosh(i*z)`; fast path `abs(a[i]) <= 2^13` single, `<= 2^16` double.
- `sin` (p808): complex `Sin(z) = -i*Sinh(i*z)`; fast path `abs(a[i]) <= 2^13` single, `<= 2^16` double.
- `sincos` fast path (p811): `abs(a[i]) <= 2^13` single, `<= 2^16` double.
- `tan` (p817): complex `Tan(z) = -i*Tanh(i*z)`; fast path `abs(a[i]) <= 2^13` single, `<= 2^16` double.
- `acos` (p820): complex symmetry `acos(CONJ(a)) = CONJ(acos(a))`; real `+1 -> 0`, `-1 -> pi`.
- `asin` (p823): complex `asin(a) = -i*asinh(i*z)`; real `+1 -> pi/2`, `-1 -> -pi/2`.
- `atan` (p825): complex `atan(a) = -i*atanh(i*a)`; real `+ -> pi/2`, `- -> -pi/2`.
- `cospi` (p831–832): computes `cos(pi*a)`; fast path `abs(ai) <= 2^22` single, `<= 2^51` double.
- `sinpi` (p834–835): computes `sin(pi*a)`; fast path `abs(ai) <= 2^22` single, `<= 2^51` double.
- `tanpi` (p837–838): computes `tan(pi*a)`; fast path `abs(ai) <= 2^13` single, `<= 2^67` double.
- `cis` (p814): `+0 -> +1 + i*0`; `-0 -> +1 - i*0`; `+/- -> QNAN + i*QNAN` with `status::errdom`.

## Conventions & Gotchas

- **Namespace/header/return.** All VM routines: `oneapi::mkl::vm`, header `oneapi/mkl/vm.hpp`, return
  `sycl::event`; wait on it or pass it into a later `depends` vector.
- **Four overloads, always.** Buffer count (`std::int64_t n`), buffer slice (slices replace `n`), USM count
  (raw pointers + `std::vector<sycl::event> const & depends = {}`), USM slice (pointers + slices +
  `depends`). The slice/USM forms exist even where only the buffer count form is printed above.
- **In-place.** Allowed for all VM mathematical functions; with positive increment indexing it requires the
  input and output increments to be equal (p703).
- **`mode`.** Default `oneapi::mkl::vm::mode::not_defined` = use the global setting from `set_mode`; the
  per-call value overrides it. Accuracy modes named here: HA, LA, EP.
- **`errhandler`.** Defaults to `{}`; "the local error handler is disabled by default". **Absent from the
  documented syntax of** `sqr`, `conj`, `abs`, `arg`, `cbrt`, `pow2o3`, `hypot`, `atan`, `atan2`.
- **`depends`.** USM only, default empty vector; sits immediately before `mode` (after `sy`/`sz` in the
  slice forms). Not present in the buffer API.
- **Precision/device.** `sycl::half` GPU-only; `_Float16` CPU-only; `float`, `double`, `std::complex<float>`,
  `std::complex<double>` CPU and GPU. LAPACK-tail `unmrq`/`unmtr` are CPU-only.
- **Special values.** Result read at the intersection of the `RE(z)` column and the `i*IM(z)` row; an empty
  cell means normal/mathematically defined. Status vocabulary: `status::sing`, `status::errdom`,
  `status::overflow`, `status::underflow`, `status::accuracy_warning`, and literal `INVALID` (hypot).
- **Complex overflow.** HA/LA only; the affected part becomes `+` and status becomes `status::overflow`,
  overriding any `status::accuracy_warning`.
- **Accuracy-mode-only.** `linearfrac` is EP-only (HA/LA ⇒ `status::accuracy_warning`, no special values
  defined). Complex **double precision** `pow` is EP-only (HA/LA ⇒ `status::accuracy_warning`).
- **Fast computational path.** `cos`/`sin`/`sincos`/`tan`: `abs(a[i]) <= 2^13` single, `<= 2^16` double.
  `cospi`/`sinpi`: `<= 2^22` single, `<= 2^51` double. `tanpi`: `<= 2^13` single, `<= 2^67` double.
  Outside the path in HA/LA, use EP functions (fast over the whole domain, lower accuracy).
- **Thresholds.** Where an input-threshold table exists (`linearfrac`, `pow3o2`, `powr`, `hypot`, `exp`,
  `exp2`, `exp10`, `expm1`), exceeding it marks the precision overflow; for `linearfrac` the source states
  correctness is guaranteed only inside the thresholds, otherwise behavior is unspecified.
- **Scalars by value of type `T`.** `powx`: `T b` (fixed power). `linearfrac`: `T scalea, T shifta,
  T scaleb, T shiftb`, positioned between input vector(s)/slices and output vector/slice.
- **Two outputs.** `sincos` writes sine to `y`, cosine to `z`. `cis` maps real `a` to complex `y` and its
  table has two type columns (`T` real input, `R` complex output).
- **LAPACK tail.** `oneapi::mkl::lapack`; throws `mkl::lapack::exception` (illegal argument at `info = -i`;
  undersized scratchpad when `info` equals the passed scratchpad size and `detail()` is non-zero, required
  size from `detail()`); `scratchpad_size` must be at least the value returned by the matching
  `*_scratchpad_size` function.

## Explicit gaps

- **No formula images exist for pages 691–840.** Every chunk header reports
  `# formula image files (transcribe these): (none)` and `formula-manifest.json` has zero entries for pages
  691–840; all formulas above came from inline text and are keyed to the `@@@PAGE n@@@` markers.
- The π symbol was dropped by text extraction. Hence `cospi`/`sinpi`/`tanpi`/`acospi`/`asinpi`/`atanpi`/
  `atan2pi` descriptions read "multiplied by " / "divided by " with no symbol, `cosd`/`tand` read
  "multiplied by /180", and the summary table prints `sin` where the section list (p804–805) gives `sind`.
- `remainder`'s definition is garbled: printed as `n = ai - n*bi` where the left side is almost certainly
  the remainder. Transcribed as printed; do not implement literally.
- `tanpi` special values print `*copysign(0.0, n)` with the leading `pi*` factor missing.
- `expm1`'s p787 table prints `+0 ⇒ +1` and `-0 ⇒ +1`; transcribed as printed (flagged inline). Those two
  entries are a source typo, not the mathematical values.
- Special-value rows whose first column was a `±` infinity were flattened by extraction for `fmod`
  (p736), `remainder` (p739), `sqrt` (p748) and `log10` (p796); these four tables were re-read from the
  page images, so their `±0`/`±`/`NAN` rows above are the printed rows, not the flattened text.
- `unmrq_scratchpad_size`'s extracted signature merges two lines into `int64_t kint64_t lda,`; intended:
  `int64_t k, int64_t lda`.
- `unmtr_scratchpad_size`'s extracted signature has **no `m` parameter** although its prose defines
  `r = m` for `side::left`.
- Complex special-value table cells for `ln` (p790–791), `log10` (p796–797) and `acos` (p819–820) did not
  survive extraction (headers only); `sqrt` (p748) and `exp` (p778–779) complex tables are fragmentary.
- `exp`'s precision/device table (p780) lists only real types while the p705 index lists `exp` for
  `h, s, d, c, z` and the page carries a complex-overflow note, so complex `exp` device support is not
  stated here. The same index-vs-table discrepancy applies to `cos`, `sin`, `tan`, `acos`, `asin`, `atan`.
- The abbreviations `h, s, d, c, z` in the summary table are **not defined** on these pages.
- `set_mode`, `create_error_handler`, the `oneMKL slice` type, and the accuracy-mode enumeration values are
  referenced but defined outside this page range.
- Header names are stated only for VM (`oneapi/mkl/vm.hpp`); the LAPACK-tail routines' header is not stated.
- No default values are stated for `n`, for the LAPACK leading dimensions, or for slice strides; no device
  restriction beyond the precision/device tables is stated.
