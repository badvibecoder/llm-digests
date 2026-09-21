---
chunk: 13-math-library-and-device-lib
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 814-947
covers: Optimization reports (-qopt-report family); the Intel oneAPI DPC++/C++ Compiler Math Library (mathimf.h, all function categories, C99 macros, errno/rounding rules); the SYCL* device library (basic arithmetic, simple math, integer ops, rounding-mode arithmetic, type casting incl. half/bfloat16, half-precision arithmetic and comparisons, and IMF transcendental math with accuracy tables and special values).
---

# Optimization Reports, Compiler Math Library, and SYCL* Device Library

> **Scope.** This chunk covers compiler optimization reports, and the two numerical libraries
> shipped with the compiler: the Compiler Math Library (`mathimf.h`, host-side, with C99 `_Complex`
> and `errno` semantics) and the SYCL* device library (device-side simple math, integer/bit
> operations, rounding-mode arithmetic, type casts for `half`/`bfloat16`, and the Intel Math
> Functions (IMF) transcendental library, `#include <sycl/ext/intel/math.hpp>`). A reader can answer
> questions about report option spellings/levels, exact math function names and signatures, `errno`
> conditions, IMF accuracy flavors and ULP values, and documented special-value results.

## Key facts

- Optimization reports: `-qopt-report` (Linux*) / `/Qopt-report` (Windows*). Levels `n` = **0–3**;
  no value → **medium** detail; higher = more detail. Related options: `-qopt-report-file` /
  `/Qopt-report-file`, `-qopt-report-stdout` / `/Qopt-report-stdout`, `-qopt-report-phase` /
  `/Qopt-report-phase`, `-qopt-report-names` / `/Qopt-report-names`.
- Many interprocedural optimizations (IPO) run at compile time **or** link time, so reports differ;
  optimization level and profile-guided optimization (PGO) tools also change results.
- Compiler Math Library declarations live in `mathimf.h`; if the compiler links, the math library is
  used **by default**. C99 `_Complex` needs `[Q]std=c99`; `_Complex` is C-only, **not** C++.
- Linux math libraries: `libimf.a` (default static), `libimf.so` (default shared). Windows:
  `libm.lib` (default static), `libmmt.lib` (`/MT`, multi-threaded static), `libmmd.lib` (`/MD`,
  dynamic), `libmmdd.lib` (`/MDd`, dynamic debug), `libmmds.lib` (static, compiled with `/MD`).
- Math errors: Domain error (EDOM) `errno = 33`; Range error (ERANGE) `errno = 34`.
  `-fmath-errno` is **disabled by default** and required for full `errno` support; `-ffast-math`
  (disabled by default) overrides `-fmath-errno`.
- Round-to-nearest-even is strongly recommended for transcendental functions at default
  optimization or higher. `-fp-model strict` (Linux) / `/fp: strict` (Windows) warns the compiler
  not to assume default floating-point environment settings.
- 64-bit decimal transcendentals rely on binary double extended precision and need x87 in 80-bit
  precision; on Windows use `/Qpc80`. `/Qlong-double` is required for 80-bit `long double`
  (otherwise `long double` maps to `double`); `/Qstd=c99` supports `_Complex`.
- FP16 Math Functions require compiler **2021.4 or higher** and a next-generation Intel® Xeon®
  Scalable processor, code name **Sapphire Rapids**.
- SYCL* device library provides basic arithmetic/simple math plus Intel Math Functions (IMF). IMF
  runs on SYCL devices (GPU, CPU, and accelerators) and mostly complies with ISO C99, SYCL,
  OpenCL™, and IEEE754 for computed outputs and IEEE754-special-value processing.
- IMF accuracy flavors: **default** (compliant to the best of OpenCL/SYCL/CUDA requirements),
  **ha** High accuracy (ULP ≤ 1.0), **la** Low accuracy (ULP ≤ 4.0), **ep** Enhanced performance
  (~half the mantissa bits correct). All functions provide default; a subset adds `ha`, `la`, `ep`,
  selected via nested namespaces `sycl::ext::intel::math::ha` / `::la` / `::ep`.
- oneAPI uses the OpenCL Specification to determine ULP accuracy for OpenCL math functions; accuracy
  differs per device, and CPU-optimized OpenCL code may not work on a GPU device.

## Quick reference tables

Optimization report options:

| Linux | Windows | Purpose | Values / default |
|---|---|---|---|
| `-qopt-report[=n]` | `/Qopt-report[:n]` | Enable report with level of detail | `n` = 0–3; no value → medium; higher = more detail |
| `-qopt-report-file=file` | `/Qopt-report-file:file` | Write report to `file` | No path → current directory |
| `-qopt-report-stdout` | `/Qopt-report-stdout` | Report to stdout | = `-qopt-report-file=stdout` / `/Qopt-report-file:stdout` |
| `-qopt-report-phase[=list]` | `/Qopt-report-phase[:list]` | Comma-separated phases to report | No phase → **all**; list phases with `[Q]opt-report-help` |
| `-qopt-report-names` | `/Qopt-report-names` | Mangled vs demangled names | Default **demangled**; usable without `-qopt-report` |

Counts / pages: Compiler Math Library (`mathimf.h`, 7 categories) pp. 821–825; SYCL* basic
arithmetic + simple math (12) pp. 855–857; simple integer math (8) pp. 857–858; basic integer
arithmetic operations (21) pp. 858–861; arithmetic with rounding mode (56, `f*` + `d*`) pp. 861–868;
type casting floating-point (71) pp. 868–876; `half` (54) pp. 876–883; `bfloat16` (59) pp. 883–891;
half-precision arithmetic (23) pp. 891–894; half-precision comparison (37) pp. 894–900; IMF
transcendental math (`sycl/ext/intel/math.hpp`, 85) pp. 900–947.

## Code size options on page 814 (tail of preceding topic)

- **Avoid References to Compiler-specific Libraries** — the compiler will not assume the presence
  of compiler-specific libraries and generates only calls appearing in the source. Advantage:
  smaller binaries; disadvantage: possible performance loss if library code was in hotspots, and
  some optimizations are suppressed. Linux: option `ffreestanding`; Windows: option
  `Qfreestanding-` (not available for SYCL). Implies `fno-builtin`; override with `fbuiltin`. Can
  sometimes **increase** binary size.
- **Use Interprocedural Optimization (IPO)** — may reduce code size via dead code elimination and
  suppression of code for functions always inlined or proven never called. Linux: option `ipo`;
  Windows: option `Qipo`. Not recommended if you plan to ship object files as part of a final
  product. Binary size can increase depending on code/application.

## Optimization Reports

`-qopt-report` (Linux*) / `/Qopt-report` (Windows*) generate optimization reports with different
levels of detail. Related options under **Optimization Report Options** specify the phase, direct
output to a file instead of stderr, and choose mangled vs demangled function/method names. Certain
options let you request report generation during the compile step or the link step.

`-qopt-report-names` / `/Qopt-report-names`: not specified → **demangled** is the default;
`mangled` adds encoding (decoration), appropriate for matching annotations with the **assembly**
listing; `demangled` adds no encoding, appropriate for matching the **source** listing. Using this
option means you do not have to specify `-qopt-report` (Linux) or `/Qopt-report` (Windows).

See also: `qopt-report`, `qopt-report-file`, `qopt-report-names`, `qopt-report-phase`,
`qopt-report-stdout` compiler options.

## Compiler Math Library

A highly optimized, very accurate math library for scientific/graphic and other floating-point-heavy
programs. Use `[Q]std=c99` to support C99 `_Complex`. Many routines are more optimized for Intel®
microprocessors than for non-Intel microprocessors. `mathimf.h` includes prototypes. Documented
caveats:

- Intel's `math.h` is compatible with the GCC Math Library `libm` but does **not** cause the GCC
  Math Library to be linked. The source can be built with `gcc` or `icx`. `mathimf.h` contains
  additional functions found only in the math library; source using those can only be built with the
  compiler and libraries.
- `long double` functions (`expl`, `logl`, ...) are **ABI incompatible** with the Microsoft
  libraries. The Intel compiler/libraries support the 80-bit `long double` (see `Qlong-double`). For
  maximum compatibility use `math.h` or `mathimf.h` with the math library.

**Linux libraries** (linked library depends on compilation/linkage options): `libimf.a` (default
static), `libimf.so` (default shared). The libraries contain performance-optimized implementations
for various Intel platforms; the best implementation for the underlying hardware is selected at
runtime. Library dispatch of multi-threaded code may lead to apparent data races detected by
analysis tools; as long as threads run on cores with the same CPUID, these data races are harmless.

**Windows libraries:**

| Library | Option | Description |
|---|---|---|
| `libm.lib` | | Default static math library |
| `libmmt.lib` | `/MT` | Multi-threaded static math library |
| `libmmd.lib` | `/MD` | Dynamically linked math library |
| `libmmdd.lib` | `/MDd` | Dynamically linked debug math library |
| `libmmds.lib` | | Static version compiled with `/MD` |

**oneAPI and OpenCL™ Considerations.** oneAPI uses the OpenCL Specification to determine ULP
accuracy for OpenCL mathematical functions; details and single/double precision tables are in the
Khronos OpenCL Specification section **Relative Error as ULPs**. Accuracy levels differ by device:
the specification sets a maximum ULP error (where applicable), but individual devices may be more
accurate. CPU-optimized OpenCL implementations may not work on a GPU device.

See also: Math Function List; `Qlong-double`, `MD`, `MT`, `std`/`Qstd`.

### Use the Compiler Math Library

Include `mathimf.h`; if the compiler is used for linking, the math library is used by default.

**Use Real Functions** — after compiling and running, the program should display a sine value of x.
Linux `real_math.c` (compile with `icx real_math.c`):

```c
// real_math.c
#include <stdio.h>
#include <mathimf.h>

int main() {
 float fp32bits;
 double fp64bits;
 long double fp80bits;
 long double pi_by_four = 3.141592653589793238/4.0;

 // pi/4 radians is about 45 degrees
 fp32bits = (float) pi_by_four; // float approximation to pi/4
 fp64bits = (double) pi_by_four; // double approximation to pi/4
 fp80bits = pi_by_four; // long double (extended) approximation to pi/4

 // The sin(pi/4) is known to be 1/sqrt(2) or approximately .7071067
 printf("When x = %8.8f, sinf(x) = %8.8f \n", fp32bits, sinf(fp32bits));
 printf("When x = %16.16f, sin(x) = %16.16f \n", fp64bits, sin(fp64bits));
 printf("When x = %20.20Lf, sinl(x) = %20.20Lf \n", fp80bits, sinl(fp80bits));

 return 0;
}
```

Windows uses the same `real_math.c` with these documented differences: `/Qlong-double` is required
because without it `long double` types are mapped to doubles; `printf()` does not support printing
long doubles on Microsoft Windows, so `fp80bits` is cast to `double` in the `sinl` call
(`printf("When x = %20.20f, sinl(x) = %20.20f \n", (double) fp80bits, (double) sinl(fp80bits));`).
Because the program includes `long double`, use `/Qlong-double` and `/Qpc80`:

```bash
icx /Qlong-double /Qpc80 real_math.c
```

**Use Complex Functions** — after compiling and running you should get:

```text
When z = 1.0000000 + 0.7853982 i, cexpf(z) = 1.9221154 + 1.9221156 i

When z = 1.000000000000 + 0.785398163397 i, cexp(z) = 1.922115514080 + 1.922115514080 i
```

Linux and Windows `complex_math.c`:

```c
// complex_math.c
#include <stdio.h>
#include <complex.h>

int main() {
  float _Complex c32in,c32out;
  double _Complex c64in,c64out;
  double pi_by_four= 3.141592653589793238/4.0;
  c64in = 1.0 + I * pi_by_four;

// Create the double precision complex number 1 + (pi/4)         i
// where I is the imaginary unit.
  c32in = (float _Complex) c64in;

// Create the float complex value from the double complex value.
  c64out = cexp(c64in);
  c32out = cexpf(c32in);

// Call the complex exponential,
// cexp(z) = cexp(x+iy) = e^ (x + i y) = e^x (cos(y) + i sin(y))
 printf("When z = %7.7f + %7.7f i, cexpf(z) = %7.7f + %7.7f i \n"
 ,crealf(c32in),cimagf(c32in),crealf(c32out),cimagf(c32out));
 printf("When z = %12.12f + %12.12f i, cexp(z) = %12.12f + %12.12f i \n"
 ,creal(c64in),cimag(c64in),creal(c64out),cimagf(c64out));

    return 0;
}
```

Because the program includes `_Complex`, include `[Q]std=c99`:

```bash
# Linux
icx -std=c99 complex_math.c
# Windows
icx /Qstd=c99 complex_math.c
```

`_Complex` data types are supported in C but not in C++ programs.

### Exception Conditions

Calling a math function with argument(s) that may produce undefined results assigns an error number
to `errno`. Errors are usually domain errors or range errors:

- **Domain errors** — arguments outside the function domain. `acos` is defined only for arguments
  between -1 and +1 inclusive; `acos(-2)` or `acos(3)` gives a domain error with return value
  `QNaN`.
- **Range errors** — a mathematically valid argument gives a value exceeding the representable range
  of the floating-point type. `exp(1000)` gives a range error with return value `INF`.

Values assigned to `errno`: Domain error (EDOM) `errno = 33`; Range error (ERANGE) `errno = 34`.

```c
// errno.c
#include <errno.h>
#include <mathimf.h>
#include <stdio.h>

int main(void) {
  double neg_one=-1.0;
  double zero=0.0;

// The natural log of a negative number is considered a domain error - EDOM
  printf("log(%e) = %e and errno(EDOM) = %d \n",neg_one,log(neg_one),errno);

// The natural log of zero is considered a range error - ERANGE
  printf("log(%e) = %e and errno(ERANGE) = %d \n",zero,log(zero),errno);
}
```

`-fmath-errno` (disabled by default) is required for full `errno` support; without it the compiler
may optimize by avoiding `errno` writes. Not recommended for performance unless the application
requires it. `-ffast-math` (disabled by default) overrides `-fmath-errno`. A corresponding `errno`
value is listed with each function below where applicable.

### Other Considerations

- Some math functions are inlined automatically; which ones varies and may depend on vectorization
  or processor-specific options. Disable automatic inline expansion of all functions with
  `-fno-builtin` (Linux) or `/Oi-` (Windows).
- Use default rounding mode (round-to-nearest-even) when calling transcendental functions at default
  optimization or higher. Faster implementations are validated under round-to-nearest-even; other
  rounding modes may reduce their accuracy or set unexpected floating-point status flags. Avoid with
  `-fp-model strict` (Linux) or `/fp: strict` (Windows).
- **64-bit decimal transcendentals** rely on binary double extended precision arithmetic; ensure the
  x87 unit operates in 80-bit precision (64-bit binary significands). Where the default x87
  precision is not 80 bits (such as Windows), set it with `/Qpc80`.
- A change of default precision control or rounding mode may affect results of some math functions.
- Important options for data types on Intel® 64 Windows: `/Qlong-double` for `long double` (80-bit
  floating-point) — without it compilation succeeds but `long double` maps to `double`;
  `/Qstd=c99` for `_Complex`.

See also: `fbuiltin`/`Oi`, `fmath-errno`, `fp-model`/`fp`, `Qlong-double`, `std`/`Qstd`; Tuning
Performance.

## Math Function List

FP16 Math Functions require compiler 2021.4 or higher and a next-generation Intel® Xeon® Scalable
processor, code name Sapphire Rapids.

**Variant legend.** `d` = `double name(double x)`; `ld` = `long double namel(long double x)`;
`f` = `float namef(float x)`; `f16` = `_Float16 namef16(_Float16 x)`. Two-argument functions repeat
the type; the base return type is that of `d` unless stated. **A name with no bracketed variant list
has all four of `d`/`ld`/`f`/`f16`; brackets list exactly the documented variants** (an absent tag is
not documented). `pi` = π.

**Literal documented names** (base + variant suffix; the bracketed entries above are exactly these):

- Trigonometric: `acos` `acosl` `acosf` `acosf16`, `acosd` `acosdl` `acosdf` `acosdf16`, `acospi` `acospif` `acospif16`, `asin` `asinl` `asinf` `asinf16`, `asind` `asindl` `asindf` `asindf16`, `asinpi` `asinpif` `asinpif16`, `atan` `atanl` `atanf` `atanf16`, `atan2` `atan2l` `atan2f` `atan2f16`, `atan2pi` `atan2pif` `atan2pif16`, `atand` `atandl` `atandf` `atandf16`, `atan2d` `atan2dl` `atan2df` `atan2df16`, `atand2` `atand2l` `atand2f` `atand2f16`, `atanpi` `atanpif` `atanpif16`, `cos` `cosl` `cosf` `cosf16`, `cosd` `cosdl` `cosdf` `cosdf16`, `cospi` `cospif` `cospif16`, `cot` `cotl` `cotf` `cotf16`, `cotd` `cotdl` `cotdf` `cotdf16`, `sin` `sinl` `sinf` `sinf16`, `sincos` `sincosl` `sincosf` `sincosf16`, `sincosd` `sincosdl` `sincosdf` `sincosdf16`, `sind` `sindl` `sindf` `sindf16`, `sinpi` `sinpif` `sinpif16`, `tan` `tanl` `tanf` `tanf16`, `tand` `tandl` `tandf` `tandf16`, `tanpi` `tanpif` `tanpif16`
- Hyperbolic: `acosh` `acoshl` `acoshf` `acoshf16`, `asinh` `asinhl` `asinhf` `asinhf16`, `atanh` `atanhl` `atanhf` `atanhf16`, `cosh` `coshl` `coshf` `coshf16`, `sinh` `sinhl` `sinhf` `sinhf16`, `sinhcosh` `sinhcoshl` `sinhcoshf` `sinhcoshf16`, `tanh` `tanhl` `tanhf` `tanhf16`
- Exponential/Logarithmic: `cbrt` `cbrtl` `cbrtf` `cbrtf16`, `exp` `expl` `expf` `expf16`, `exp10` `exp10l` `exp10f` `exp10f16`, `exp2` `exp2l` `exp2f` `exp2f16`, `expm1` `expm1l` `expm1f` `expm1f16`, `frexp` `frexpl` `frexpf` `frexpf16`, `hypot` `hypotl` `hypotf` `hypotf16`, `invsqrt` `invsqrtl` `invsqrtf` `invsqrtf16`, `ilogb` `ilogbl` `ilogbf` `ilogbf16`, `ldexp` `ldexpl` `ldexpf` `ldexpf16`, `log` `logl` `logf` `logf16`, `log10` `log10l` `log10f` `log10f16`, `log1p` `log1pl` `log1pf` `log1pf16`, `log2` `log2l` `log2f` `log2f16`, `logb` `logbl` `logbf` `logbf16`, `pow` `powl` `powf` `powf16`, `pow2o3` `pow2o3f` `pow2o3f16`, `pow3o2` `pow3o2f` `pow3o2f16`, `powr` `powrf` `powrf16`, `scalb` `scalbl` `scalbf` `scalbf16`, `scalbln` `scalblnl` `scalblnf` `scalblnf16`, `scalbn` `scalbnl` `scalbnf` `scalbnf16`, `sqrt` `sqrtl` `sqrtf` `sqrtf16`
- Special: `annuity` `annuityl` `annuityf` `annuityf16`, `cdfnorm` `cdfnormf` `cdfnormf16`, `cdfnorminv` `cdfnorminvf` `cdfnorminvf16`, `compound` `compoundl` `compoundf` `compoundf16`, `erf` `erfl` `erff` `erff16`, `erfc` `erfcl` `erfcf` `erfcf16`, `erfcx` `erfcxf`, `erfcinv` `erfcinvf` `erfcinvf16`, `erfinv` `erfinvl` `erfinvf` `erfinvf16`, `gamma` `gammal` `gammaf` `gammaf16`, `gamma_r` `gammal_r` `gammaf_r` `gammaf16_r`, `j0` `j0l` `j0f` `j0f16`, `j1` `j1l` `j1f` `j1f16`, `jn` `jnl` `jnf` `jnf16`, `lgamma` `lgammal` `lgammaf` `lgammaf16`, `lgamma_r` `lgammal_r` `lgammaf_r` `lgammaf16_r`, `tgamma` `tgammal` `tgammaf` `tgammaf16`, `y0` `y0l` `y0f` `y0f16`, `y1` `y1l` `y1f` `y1f16`, `yn` `ynl` `ynf` `ynf16`
- Nearest Integer/Remainder: `ceil` `ceill` `ceilf` `ceilf16`, `floor` `floorl` `floorf` `floorf16`, `llrint` `llrintl` `llrintf` `llrintf16`, `llround` `llroundl` `llroundf` `llroundf16`, `lrint` `lrintl` `lrintf` `lrintf16`, `lround` `lroundl` `lroundf` `lroundf16`, `modf` `modfl` `modff` `modff16`, `nearbyint` `nearbyintl` `nearbyintf` `nearbyintf16`, `rint` `rintl` `rintf` `rintf16`, `round` `roundl` `roundf` `roundf16`, `trunc` `truncl` `truncf` `truncf16`, `fmod` `fmodl` `fmodf` `fmodf16`, `remainder` `remainderl` `remainderf` `remainderf16`, `remquo` `remquol` `remquof` `remquof16`
- Miscellaneous: `copysign` `copysignl` `copysignf` `copysignf16`, `fabs` `fabsl` `fabsf` `fabsf16`, `fdim` `fdiml` `fdimf` `fdimf16`, `finite` `finitel` `finitef` `finitef16`, `fma` `fmal` `fmaf` `fmaf16`, `fmax` `fmaxl` `fmaxf` `fmaxf16`, `fmin` `fminl` `fminf` `fminf16`, `fpclassify` `fpclassifyl` `fpclassifyf` `fpclassifyf16`, `isfinite` `isfinitel` `isfinitef` `isfinitef16`, `isgreater` `isgreaterl` `isgreaterf` `isgreaterf16`, `isgreaterequal` `isgreaterequall` `isgreaterequalf` `isgreaterequalf16`, `isinf` `isinfl` `isinff` `isinff16`, `isless` `islessl` `islessf` `islessf16`, `islessequal` `islessequall` `islessequalf` `islessequalf16`, `islessgreater` `islessgreaterl` `islessgreaterf` `islessgreaterf16`, `isnan` `isnanl` `isnanf` `isnanf16`, `isnormal` `isnormall` `isnormalf` `isnormalf16`, `isunordered` `isunorderedl` `isunorderedf` `isunorderedf16`, `maxmag` `maxmagf` `maxmagf16`, `minmag` `minmagf` `minmagf16`, `nan` `nanl` `nanf` `nanf16`, `nextafter` `nextafterl` `nextafterf` `nextafterf16`, `nexttoward` `nexttowardl` `nexttowardf` `nexttowardf16`, `signbit` `signbitl` `signbitf`, `significand` `significandl` `significandf` `significandf16`

### Trigonometric Functions

- `acos` — principal inverse cosine, [0,pi] rad, x in [-1,1]; EDOM |x|>1.
- `acosd` — inverse cosine, [0,180] **degrees**, x in [-1,1]; EDOM |x|>1.
- `acospi` [d,f,f16] — inverse cosine divided by pi, [0,1], x in [-1,1]; EDOM |x|>1.
- `asin` — principal inverse sine, [-pi/2,+pi/2] rad, x in [-1,1]; EDOM |x|>1.
- `asind` — inverse sine, [-90,90] degrees, x in [-1,1]; EDOM |x|>1.
- `asinpi` [d,f,f16] — inverse sine divided by pi, [-1/2,1/2] (source says degrees), x in [-1,1];
  EDOM |x| > 1 divided by pi.
- `atan` — principal inverse tangent, [-pi/2,+pi/2] rad.
- `atan2` — inverse tangent of y/x, [-pi,+pi] rad; args (y,x); EDOM x=0 and y=0.
- `atan2pi` [d,f,f16] — inverse tangent of y/x divided by pi, [-1,+1]; args (y,x); EDOM x=0, y=0.
- `atand` — inverse tangent, [-90,90] degrees.
- `atan2d` — inverse tangent of y/x, [-180,+180] degrees; args (x,y); EDOM x=0, y=0.
- `atand2` — inverse tangent of y/x, [-180,+180] degrees; args (x,y); EDOM x=0, y=0.
- `atanpi` [d,f,f16] — inverse tangent divided by pi, [-1/2,+1/2].
- `cos` — cosine of x measured in radians.
- `cosd` — cosine of x measured in degrees.
- `cospi` [d,f,f16] — cosine of x multiplied by pi, cos(x*pi).
- `cot` — cotangent in radians; ERANGE for overflow conditions at x=0.
- `cotd` — cotangent in degrees; ERANGE for overflow conditions at x=0.
- `sin` — sine of x measured in radians.
- `sincos` — both sine and cosine in radians: `void sincos(double x, double *sinval, double *cosval); void sincosl(long double x, long double *sinval, long double *cosval); void sincosf(float x, float *sinval, float *cosval); void sincosf16(_Float16 x, _Float16 *sinval, _Float16 *cosval);`
- `sincosd` — both sine and cosine in degrees: `void sincosd(double x, double *sinval, double *cosval);` plus `sincosdl`, `sincosdf`, `sincosdf16` with matching types.
- `sind` — sine of x measured in degrees.
- `sinpi` [d,f,f16] — sine of x multiplied by pi, sin(x*pi).
- `tan` — tangent of x measured in radians.
- `tand` — tangent of x measured in degrees; ERANGE for overflow conditions.
- `tanpi` [d,f,f16] — tangent of x multiplied by pi, tan(x*pi).

### Hyperbolic Functions

- `acosh` — inverse hyperbolic cosine; EDOM x<1.
- `asinh` — inverse hyperbolic sine.
- `atanh` — inverse hyperbolic tangent; EDOM |x|>1; ERANGE x=1.
- `cosh` — hyperbolic cosine, (e^x + e^-x)/2; ERANGE overflow.
- `sinh` — hyperbolic sine, (e^x - e^-x)/2; ERANGE overflow.
- `sinhcosh` — both hyperbolic sine and cosine: `void sinhcosh(double x, double *sinval, double *cosval);` plus `sinhcoshl`, `sinhcoshf`, `sinhcoshf16`; ERANGE overflow.
- `tanh` — hyperbolic tangent, (e^x - e^-x)/(e^x + e^-x).

### Exponential Functions

- `cbrt` — cube root of x.
- `exp` — e raised to the x power, e^x; ERANGE underflow and overflow.
- `exp10` — 10 raised to the x power, 10^x; ERANGE underflow and overflow.
- `exp2` — 2 raised to the x power, 2^x; ERANGE underflow and overflow.
- `expm1` — e^x - 1; ERANGE overflow.
- `frexp` — converts x into a signed normalized fraction in [1/2,1) times an integral power of two;
  returns the fraction, stores the integer exponent at `exp`:
  `double frexp(double x, int *exp); long double frexpl(long double x, int *exp); float frexpf(float x, int *exp); _Float16 frexpf16(_Float16 x, int *exp);`
- `hypot` — sqrt(x^2 + y^2); ERANGE overflow.
- `ilogb` — exponent of x base two as a signed `int`; ERANGE x=0:
  `int ilogb(double x); int ilogbl(long double x); int ilogbf(float x); int ilogbf16(_Float16 x);`
- `invsqrt` — inverse square root.
- `ldexp` — x*2^exp for integer `exp`; ERANGE underflow and overflow:
  `double ldexp(double x, int exp); long double ldexpl(long double x, int exp); float ldexpf(float x, int exp); _Float16 ldexpf16(_Float16 x, int exp);`
- `log` — natural log, ln(x); EDOM x<0; ERANGE x=0.
- `log10` — base-10 log, log10(x); EDOM x<0; ERANGE x=0.
- `log1p` — natural log of (x+1); EDOM x<-1; ERANGE x=-1.
- `log2` — base-2 log, log2(x); EDOM x<0; ERANGE x=0.
- `logb` — signed exponent of x; EDOM x=0.
- `pow` — x^y; EDOM x=0 and y<0; EDOM x<0 and y a non-integer; ERANGE overflow and underflow.
- `pow2o3` [d,f,f16] — cube root of x squared, cbrt(x^2).
- `pow3o2` [d,f,f16] — square root of the cube of x, sqrt(x^3); EDOM x<0; ERANGE overflow/underflow.
- `powr` [d,f,f16] — x^y where x >= 0; EDOM x<0; ERANGE overflow and underflow.
- `scalb` — x*2^y for floating-point y; ERANGE underflow and overflow.
- `scalbn` — x*2^n for integer n; ERANGE underflow and overflow:
  `double scalbn(double x, int n); long double scalbnl(long double x, int n); float scalbnf(float x, int n); _Float16 scalbnf16(_Float16 x, int n);`
- `scalbln` — x*2^n for long integer n; ERANGE underflow and overflow:
  `double scalbln(double x, long int n); long double scalblnl(long double x, long int n); float scalblnf(float x, long int n); _Float16 scalblnf16(_Float16 x, long int n);`
- `sqrt` — correctly rounded square root; EDOM x<0.

### Special Functions

- `annuity` — present value factor for an annuity, (1 - (1+x)^(-y))/x, x rate, y period; ERANGE
  underflow and overflow.
- `cdfnorm` [d,f,f16] — cumulative normal distribution function value.
- `cdfnorminv` [d,f,f16] — inverse cumulative normal distribution function value; EDOM for finite
  or infinite (x > 1) || (x < 0); ERANGE for x = 0 or x = 1.
- `compound` — compound interest factor, (1+x)^y, x rate, y period; ERANGE underflow/overflow.
- `erf` — error function value.
- `erfc` — complementary error function value; ERANGE underflow.
- `erfcx` [d,f] — scaled complementary error function value; ERANGE overflow.
- `erfcinv` [d,f,f16] — inverse complementary error function of x; EDOM for finite or infinite
  (x > 2) || (x < 0).
- `erfinv` — inverse error function of x; EDOM for finite or infinite |x| > 1.
- `gamma` — logarithm of the absolute value of gamma; ERANGE overflow when x is a negative integer.
- `gamma_r` — as `gamma`, with the sign of the gamma function returned in the integer `signgam`:
  `double gamma_r(double x, int *signgam); long double gammal_r(long double x, int *signgam); float gammaf_r(float x, int *signgam); _Float16 gammaf16_r(_Float16 x, int *signgam);`
- `j0` — Bessel function (first kind) of x, order 0. `j1` — same, order 1.
- `jn` — Bessel function (first kind) of x, order n:
  `double jn(int n, double x); long double jnl(int n, long double x); float jnf(int n, float x); _Float16 jnf16(int n, _Float16 x);`
- `lgamma` — logarithm of the absolute value of gamma; ERANGE overflow, x=0 or negative integers.
- `lgamma_r` — as `lgamma` plus sign in `signgam`; ERANGE overflow, x=0 or negative integers:
  `double lgamma_r(double x, int *signgam); long double lgammal_r(long double x, int *signgam); float lgammaf_r(float x, int *signgam); _Float16 lgammaf16_r(_Float16 x, int *signgam);`
- `tgamma` — gamma function of x; EDOM x=0 or negative integers; ERANGE overflow.
- `y0` — Bessel function (second kind) of x, order 0; EDOM x<=0. `y1` — same, order 1; EDOM x<=0.
- `yn` — Bessel function (second kind) of x, order n; EDOM x<=0:
  `double yn(int n, double x); long double ynl(int n, long double x); float ynf(int n, float x); _Float16 ynf16(int n, _Float16 x);`

### Nearest Integer Functions

- `ceil` — smallest integral value not less than x, as a floating-point number.
- `floor` — largest integral value not greater than x, as a floating-point value.
- `llrint` — rounded integer per current rounding direction, as `long long int`; ERANGE for values
  too large: `long long int llrint(double x); long long int llrintl(long double x); long long int llrintf(float x); long long int llrintf16(_Float16 x);`
- `llround` — rounded integer as `long long int`; ERANGE for values too large:
  `long long int llround(double x); long long int llroundl(long double x); long long int llroundf(float x); long long int llroundf16(_Float16 x);`
- `lrint` — rounded integer per current rounding direction, as `long int`; ERANGE for values too
  large: `long int lrint(double x); long int lrintl(long double x); long int lrintf(float x); long int lrintf16(_Float16 x);`
- `lround` — rounded integer as `long int`; **halfway cases rounded away from zero**; ERANGE for
  values too large: `long int lround(double x); long int lroundl(long double x); long int lroundf(float x); long int lroundf16(_Float16 x);`
- `modf` — returns the signed fractional part of x and stores the integral part at `*iptr` as a
  floating-point number:
  `double modf(double x, double *iptr); long double modfl(long double x, long double *iptr); float modff(float x, float *iptr); _Float16 modff16(_Float16 x, _Float16 *iptr);`
- `nearbyint` — rounded integral value as a floating-point number, current rounding direction.
- `rint` — rounded integral value as a floating-point number, current rounding direction.
- `round` — nearest integral value as a floating-point number; halfway cases rounded away from zero.
- `trunc` — truncated integral value as a floating-point number.

### Remainder Functions

- `fmod` — x-n*y for integer n such that, if y is nonzero, the result has the same sign as x and
  magnitude less than the magnitude of y; EDOM y=0.
- `remainder` — x REM y as required by the IEEE standard; EDOM y=0.
- `remquo` — x REM y; stores at `quo` a value whose sign is the sign of x/y and whose magnitude is
  congruent modulo 2^n of the integral quotient of x/y. **N is implementation-defined; for all
  systems N is equal to 31.** EDOM y=0:
  `double remquo(double x, double y, int *quo); long double remquol(long double x, long double y, int *quo); float remquof(float x, float y, int *quo); _Float16 remquof16(_Float16 x, _Float16 y, int *quo);`

### Miscellaneous Functions

- `copysign` — magnitude of x with the sign of y.
- `fabs` — absolute value of x.
- `fdim` — positive difference, x-y (for x > y) or +0 (for x <= y); ERANGE overflow.
- `finite` — 1 if x is not a NaN or +/- infinity, otherwise 0:
  `int finite(double x); int finitel(long double x); int finitef(float x); int finitef16(_Float16 x);`
- `fma` — returns (x*y)+z:
  `double fma(double x, double y, double z); long double fmal(long double x, long double y, long double z); float fmaf(float x, float y, float z); _Float16 fmaf16(_Float16 x, _Float16 y, _Float16 z);`
- `fmax` — maximum numeric value of its arguments. `fmin` — minimum numeric value.
- `fpclassify` — value of the number classification macro; possible values **0 (NaN), 1 (Infinity),
  2 (Zero), 3 (Subnormal), 4 (Finite)**:
  `int fpclassify(double x); int fpclassifyl(long double x); int fpclassifyf(float x); int fpclassifyf16(_Float16 x);`
- `isfinite` — 1 if x is not a NaN or +/- infinity, otherwise 0, as `int`.
- `isgreater` — 1 if x > y; does **not** raise the invalid floating-point exception.
- `isgreaterequal` — 1 if x >= y; does not raise the invalid floating-point exception.
- `isinf` — non-zero iff the argument has an infinite value.
- `isless` — 1 if x < y; does not raise the invalid floating-point exception.
- `islessequal` — 1 if x <= y; does not raise the invalid floating-point exception.
- `islessgreater` — 1 if x is less than or greater than y; does not raise the invalid
  floating-point exception.
- `isnan` — non-zero iff x has a NaN value. `isnormal` — non-zero iff x is normal.
- `isunordered` — 1 if either x or y is a NaN; does not raise the invalid floating-point exception.
- `maxmag` [d,f,f16] — larger magnitude of x,y: |x|>|y| returns x; |y|>|x| returns y; otherwise
  behaves like `fmax(x,y)`. `double maxmag(double x, double y); float maxmagf(float x, float y); _Float16 maxmagf16(_Float16 x, _Float16 y);`
- `minmag` [d,f,f16] — smaller magnitude of x,y, otherwise behaves like `fmin(x,y)`;
  `double minmag(double x, double y); float minmagf(float x, float y);` (source's third line prints
  `_Float16 maxmagf16(...)` [sic: should be `minmagf16`]).
- `nan` — quiet NaN with content indicated through `tagp`:
  `double nan(const char *tagp); long double nanl(const char *tagp); float nanf(const char *tagp); _Float16 nanf16(const char *tagp);`
- `nextafter` — next representable value in the specified format after x toward y; ERANGE overflow
  and underflow.
- `nexttoward` — next representable value after x toward y; if x equals y returns y converted to the
  function type; **use `Qlong-double` on Windows for accurate results**; ERANGE overflow/underflow:
  `double nexttoward(double x, long double y); long double nexttowardl(long double x, long double y); float nexttowardf(float x, long double y); _Float16 nexttowardf16(_Float16 x, long double y);`
- `signbit` [d,ld,f] — non-zero iff the sign of x is negative; `int signbit(double x); int signbitl(long double x); int signbitf(float x);` (no `f16` documented).
- `significand` — significand of x in [1,2). For x equal to zero, NaN, or +/- infinity, the
  original x is returned.

### Complex Functions

Documented for `_Complex` types (requires `[Q]std=c99`; C only, not C++). Each has a
`long double _Complex` (`...l`) and a `float _Complex` (`...f`) variant unless noted; the base
variant is `double ...(double _Complex z);`.

- `cabs` — complex absolute value of z; returns `double`/`long double`/`float`:
  `double cabs(double _Complex z); long double cabsl(long double _Complex z); float cabsf(float _Complex z);`
- `cacos` — complex inverse cosine of z (also `cacosl`, `cacosf`).
- `cacosh` — complex inverse hyperbolic cosine of z (also `cacoshl`, `cacoshf`).
- `carg` — argument in the interval [-pi, +pi]; returns real:
  `double carg(double _Complex z); long double cargl(long double _Complex z); float cargf(float _Complex z);`
- `casin` — complex inverse sine of z (also `casinl`, `casinf`).
- `casinh` — complex inverse hyperbolic sine of z (also `casinhl`, `casinhf`).
- `catan` — complex inverse tangent of z (also `catanl`, `catanf`).
- `catanh` — complex inverse hyperbolic tangent of z (also `catanhl`, `catanhf`).
- `ccos` — complex cosine of z (also `ccosl`, `ccosf`).
- `ccosh` — complex hyperbolic cosine of z (also `ccoshl`, `ccoshf`).
- `cexp` — e^z (e raised to the power z) (also `cexpl`, `cexpf`).
- `cexp2` — 2^z (2 raised to the power z); source description says "The cexp function returns 2z"
  [sic] (also `cexp2l`, `cexp2f`).
- `cexp10` — 10^z (10 raised to the power z); in the body but absent from the summary list table
  (also `cexp10l`, `cexp10f`).
- `cimag` — imaginary part of z; returns real:
  `double cimag(double _Complex z); long double cimagl(long double _Complex z); float cimagf(float _Complex z);`
- `cis` — cosine and sine as a complex value, z measured in radians:
  `double _Complex cis(double x); long double _Complex cisl(long double z); float _Complex cisf(float z);`
  (source mixes parameter names x/z [sic]).
- `cisd` — cosine and sine as a complex value, z measured in degrees:
  `double _Complex cisd(double x); long double _Complex cisdl(long double z); float _Complex cisdf(float z);`
  In the body but absent from the summary list table.
- `clog` — complex natural logarithm of z (also `clogl`, `clogf`).
- `clog2` — complex logarithm base 2 of z (also `clog2l`, `clog2f`); in the body but absent from the
  summary list table.
- `clog10` — complex logarithm base 10 of z (also `clog10l`, `clog10f`).
- `conj` — complex conjugate of z, reversing the sign of its imaginary part (also `conjl`, `conjf`).
- `cpow` — complex power function, x^y:
  `double _Complex cpow(double _Complex x, double _Complex y);` plus `cpowl`, `cpowf` with matching types.
- `cproj` — projection of z onto the Riemann sphere (also `cprojl`, `cprojf`).
- `creal` — real part of z:
  `double creal(double _Complex z); long double creall(long double _Complex z); float crealf(float _Complex z);`
- `csin` — complex sine of z (also `csinl`, `csinf`).
- `csinh` — complex hyperbolic sine of z (also `csinhl`, `csinhf`).
- `csqrt` — complex square root of z (also `csqrtl`, `csqrtf`).
- `ctan` — complex tangent of z (also `ctanl`, `ctanf`).
- `ctanh` — complex hyperbolic tangent of z (also `ctanhl`, `ctanhf`).

## C99 Macros

`mathimf.h` and the math library support: `int fpclassify(x)` · `int isfinite(x)` ·
`int isgreater(x, y)` · `int isgreaterequal(x, y)` · `int isinf(x)` · `int isless(x, y)` ·
`int islessequal(x, y)` · `int islessgreater(x, y)` · `int isnan(x)` · `int isnormal(x)` ·
`int isunordered(x, y)` · `int signbit(x)`. See also: Miscellaneous Functions.

## SYCL* Device Library

The compiler provides a SYCL* device library with Intel Math Functions (IMF), in addition to basic
arithmetic operations and simple math functions.

### Basic Arithmetic Operations and Simple Math Functions

- `abs` — absolute value of integer x for `int`; these are declared in `<stdlib.h>` in normal CPU
  programming and the compiler supports them in SYCL device code. `int abs(int x)`
- `labs` — absolute value for `long`. `long labs(long x)`
- `llabs` — absolute value for `long long`. `long long llabs(long long x)`
- `cpolar` — double complex number with magnitude `rho` and phase angle `theta` (the C99 `<complex.h>`
  complex arithmetic functions are provided in SYCL device code):
  `double __complex__ __devicelib_cpolar(double rho, double theta)`
- `cpolarf` — float complex number with magnitude `rho` and phase angle `theta`.
  `float __complex__ cpolarf(float rho, float theta)`
- `div` — quotient and remainder of `int` x divided by `int` y; `struct div_t { int quot; int rem; };`
  `div_t div(int x, int y)`
- `ldiv` — same for `long`; `struct ldiv_t { long quot; long rem; };` `ldiv_t ldiv(long x, long y)`
- `lldiv` — same for `long long`; `struct lldiv_t { long long quot; long long rem; };`
  `lldiv_t lldiv(long long x, long long y)`
- `__divdc3` — double complex quotient of (a + ib) / (c + id). During compilation the compiler may
  insert references to these two functions and the linker resolves the undefined reference later.
  `double __complex__ __divdc3(double __a, double __b, double __c, double __d)`
- `__divsc3` — float complex quotient of (a + ib) / (c + id), same compiler-inserted reference
  behavior. `float __complex__ __divsc3(float __a, float __b, float __c, float __d)`
- `__muldc3` — double complex product of (a + ib) * (c + id).
  `double __complex__ __muldc3(double __a, double __b, double __c, double __d)`
- `__mulsc3` — float complex product of (a + ib) * (c + id).
  `float __complex__ __mulsc3(float __a, float __b, float __c, float __d)`

### Simple Integer Math Functions

`max` — max of two `int`: `int max(int x, int y)` · `llmax` — max of two `long long`:
`long long llmax(long long x, long long y)` · `umax` — max of two `unsigned int`:
`unsigned umax(unsigned x, unsigned y)` · `ullmax` — max of two `unsigned long long`:
`unsigned long long ullmax(unsigned long long x, unsigned long long y)` · `min` — min of two `int`
(source signature `int min(int x, int x)` [sic]) · `llmin` — min of two `long long`:
`long long llmin(long long x, long long y)` · `umin` — min of two `unsigned int`:
`unsigned umin(unsigned x, unsigned y)` · `ullmin` — min of two `unsigned long long`:
`unsigned long long ullmin(unsigned long long x, unsigned long long y)`.

### Basic Integer Arithmetic Operations

- `brev` — reverses bit order of a 32-bit unsigned int. `unsigned brev(unsigned x)`
- `brevll` — reverses bit order of a 64-bit unsigned long long.
  `unsigned long long brevll(unsigned long long x)`
- `byte_perm` — 32-bit unsigned int whose bytes are selected from 2 inputs by a selector value:

  ```c
  uint64_t y_x = ((uint64_t)y << 32) | x;
  s0 = z & 0x7;
  s1 = (z >> 4) & 0x7;
  s2 = (z >> 8) & 0x7;
  s3 = (z >> 12) & 0x7;
  ```

  `res` is an unsigned int with bit representation:

  ```c
  res[bit_7...bit_0] = y_x[s0];
  res[bit_15...bit_8] = y_x[s1];
  res[bit_23...bit_16] = y_x[s2];
  res[bit_31...bit_24] = y_x[s3];
  ```

  `unsigned int byte_perm(unsigned int x, unsigned int y, unsigned int z)`
- `clz` — consecutive high-order zero bits in a 32-bit integer. `int clz(int x)`
- `clzll` — consecutive high-order zero bits in a 64-bit integer. `int clzll(long long x)`
- `ffs` — position of the least significant bit set to 1 in a 32-bit integer. `int ffs(int x)`
- `ffsll` — position of the least significant bit set to 1 in a 64-bit integer.
  `int ffsll(long long x)`
- `hadd` — `(x + y) >> 1` for a signed integer, avoiding overflow in the intermediate sum.
  `int hadd(int x, int y)`
- `rhadd` — `(x + y + 1) >> 1` for a signed integer, avoiding overflow. `int rhadd(int x, int y)`
- `uhadd` — `(x + y) >> 1` for an unsigned integer, avoiding overflow.
  `unsigned int uhadd(unsigned int x, unsigned int y)`
- `urhadd` — `(x + y + 1) >> 1` for an unsigned integer, avoiding overflow.
  `unsigned int urhadd(unsigned int x, unsigned int y)`
- `mul24` — multiply two 24-bit values x and y; x and y are 32-bit signed integers but only the low
  24 bits are used, the high order 8 bits are ignored. `int mul24(int x, int y)`
- `umul24` — as `mul24` for 32-bit unsigned integers.
  `unsigned int umul24(unsigned int x, unsigned int y)`
- `mul64hi` — 128-bit product x * y of two 64-bit signed integers; returns the most significant
  64 bits. `long long mul64hi(long long x, long long y)`
- `umul64hi` — 128-bit product of two 64-bit unsigned integers; returns the most significant 64
  bits. `unsigned long long umul64hi(unsigned long long x, unsigned long long)` (source omits the
  second parameter name [sic]).
- `mulhi` — 64-bit product x * y of two 32-bit signed integers; returns the most significant 32
  bits. `int mulhi(int x, int y)` (source description says "most significant 32-bit of the 128-bit
  product" [sic]).
- `umulhi` — 64-bit product of two 32-bit unsigned integers; returns the most significant 32 bits.
  `unsigned int umulhi(unsigned int x, unsigned int y)`
- `popc` — number of bits set to 1 in a 32-bit integer. `int popc(unsigned int x)`
- `popcll` — number of bits set to 1 in a 64-bit integer. `int popcll(unsigned long long x)`
- `sad` — `|x - y| + z`. `unsigned int sad(int x, int y, unsigned int z)`
- `usad` — `|x - y| + z`. `unsigned int usad(unsigned int x, unsigned int y, unsigned int z)`

### Simple Arithmetic Operations with Rounding Mode

Suffixes: `_rd` round-down, `_rn` round-to-nearest-even, `_ru` round-up, `_rz` round-towards-zero.

Single precision (`float`): `fAdds_<sfx>(float x, float y)` add · `fsub_<sfx>(float x, float y)`
subtract · `fdiv_<sfx>(float x, float y)` divide · `fmul_<sfx>(float x, float y)` multiply ·
`fmaf_<sfx>(float x, float y, float z)` returns x * y + z · `frcp_<sfx>(float x)` reciprocal ·
`fsqrt_<sfx>(float x)` square root.

Double precision (`double`): `dAdds_<sfx>(double x, double y)` add · `dsub_<sfx>(double x, double y)`
subtract · `ddiv_<sfx>(double x, double y)` divide · `dmul_<sfx>(double x, double y)` multiply ·
`fma_<sfx>(double x, double y, double z)` returns x * y + z · `drcp_<sfx>(double x)` reciprocal ·
`dsqrt_<sfx>(double x)` square root.

Full documented name set: `fAdds_rd` `fAdds_rn` `fAdds_ru` `fAdds_rz` `fsub_rd` `fsub_rn` `fsub_ru`
`fsub_rz` `fdiv_rd` `fdiv_rn` `fdiv_ru` `fdiv_rz` `fmul_rd` `fmul_rn` `fmul_ru` `fmul_rz` `fmaf_rd`
`fmaf_rn` `fmaf_ru` `fmaf_rz` `frcp_rd` `frcp_rn` `frcp_ru` `frcp_rz` `fsqrt_rd` `fsqrt_rn`
`fsqrt_ru` `fsqrt_rz` `dAdds_rd` `dAdds_rn` `dAdds_ru` `dAdds_rz` `dsub_rd` `dsub_rn` `dsub_ru`
`dsub_rz` `ddiv_rd` `ddiv_rn` `ddiv_ru` `ddiv_rz` `dmul_rd` `dmul_rn` `dmul_ru` `dmul_rz` `fma_rd`
`fma_rn` `fma_ru` `fma_rz` `drcp_rd` `drcp_rn` `drcp_ru` `drcp_rz` `dsqrt_rd` `dsqrt_rn` `dsqrt_ru`
`dsqrt_rz`.

### Type Casting Functions for Floating-Point Numbers

All four suffixes documented for each family (`_rd`/`_rn`/`_ru`/`_rz` = round-down /
round-to-nearest-even / round-up / round-towards-zero):

| Family (`_rd`,`_rn`,`_ru`,`_rz`) | Signature / meaning |
|---|---|
| `double2float_*` | `float double2float_<sfx>(double x)` — double to float |
| `double2int_*` | `int double2int_<sfx>(double x)` — double to signed int |
| `double2ll_*` | `long long double2ll_<sfx>(double x)` — double to 64-bit signed int |
| `double2uint_*` | `unsigned int double2uint_<sfx>(double x)` — double to unsigned int |
| `double2ull_*` | `unsigned long long double2ull_<sfx>(double x)` — double to unsigned 64-bit int |
| `float2int_*` | `int float2int_<sfx>(float x)` — float to signed int |
| `float2ll_*` | `long long float2ll_<sfx>(float x)` — float to signed 64-bit int |
| `float2uint_*` | `unsigned int float2uint_<sfx>(float x)` — float to unsigned int |
| `float2ull_*` | `unsigned long long float2ull_<sfx>(float x)` — float to unsigned 64-bit int |
| `int2float_*` | `float int2float_<sfx>(int x)` — signed int to float |
| `ll2double_*` | `double ll2double_<sfx>(long long x)` — signed 64-bit int to double |
| `ll2float_*` | `float ll2float_<sfx>(long long x)` — signed 64-bit int to float |
| `uint2float_*` | `float uint2float_<sfx>(unsigned int x)` — unsigned int to float |
| `ull2float_*` | `float ull2float_<sfx>(unsigned long long x)` — unsigned 64-bit int to float |
| `ull2double_*` | `double ull2double_<sfx>(unsigned long long x)` — unsigned 64-bit int to double |

Literal documented names for these 15 families: `double2float_rd` `double2float_rn` `double2float_ru` `double2float_rz` `double2int_rd` `double2int_rn` `double2int_ru` `double2int_rz` `double2ll_rd` `double2ll_rn` `double2ll_ru` `double2ll_rz` `double2uint_rd` `double2uint_rn` `double2uint_ru` `double2uint_rz` `double2ull_rd` `double2ull_rn` `double2ull_ru` `double2ull_rz` `float2int_rd` `float2int_rn` `float2int_ru` `float2int_rz` `float2ll_rd` `float2ll_rn` `float2ll_ru` `float2ll_rz` `float2uint_rd` `float2uint_rn` `float2uint_ru` `float2uint_rz` `float2ull_rd` `float2ull_rn` `float2ull_ru` `float2ull_rz` `int2float_rd` `int2float_rn` `int2float_ru` `int2float_rz` `ll2double_rd` `ll2double_rn` `ll2double_ru` `ll2double_rz` `ll2float_rd` `ll2float_rn` `ll2float_ru` `ll2float_rz` `uint2float_rd` `uint2float_rn` `uint2float_ru` `uint2float_rz` `ull2float_rd` `ull2float_rn` `ull2float_ru` `ull2float_rz` `ull2double_rd` `ull2double_rn` `ull2double_ru` `ull2double_rz`.

Single-variant: `double int2double_rn(int x)` and `double uint2double_rn(unsigned int x)`
(round-to-nearest-even only).

Bit reinterpretation (no rounding mode): `int double2hiint(double x)` (high 32 bits of a double as
signed int) · `int double2loint(double x)` (low 32 bits) · `long long double_as_longlong(double x)`
(bits of a double as signed 64-bit int) · `int float_as_int(float x)` · `unsigned int
float_as_uint(float x)`; `uint_as_float(unsigned int x)` returns `float` · `int_as_float(int x)`
returns `float` · `longlong_as_double(long long x)` returns `double` · `double hiloint2double(int hi,
int log)` (high and low 32-bit integer values as a double; source parameter name `log` [sic: likely
`lo`]).

### Type Casting Functions for Half-Precision Types

Suffixes `_rd`/`_rn`/`_ru`/`_rz` as above; `sycl::half` is the half-precision type.

| Family | Signature |
|---|---|
| `double2half` | `sycl::half double2half(double x)` (round-to-nearest-even only) |
| `float2half_*` | `sycl::half float2half_<sfx>(float x)` |
| `half2float` | `float half2float(sycl::half x)` |
| `half2int_*` | `int half2int_<sfx>(sycl::half x)` (source prints a stray `a` [sic]) |
| `half2ll_*` | `long long half2ll_<sfx>(sycl::half x)` |
| `half2short_*` | `short half2short_<sfx>(sycl::half x)` |
| `half2uint_*` | `unsigned int half2uint_<sfx>(sycl::half x)` |
| `half2ull_*` | `unsigned long long half2ull_<sfx>(sycl::half x)` |
| `half2ushort_*` | `unsigned short half2ushort_<sfx>(sycl::half x)` |
| `int2half_*` | `sycl::half int2half_<sfx>(int x)` |
| `ll2half_*` | `sycl::half ll2half_<sfx>(long long x)` |
| `short2half_*` | `sycl::half short2half_<sfx>(short x)` |
| `uint2half_*` | `sycl::half uint2half_<sfx>(unsigned int x)` |
| `ull2half_*` | `sycl::half ull2half_<sfx>(unsigned long long x)` |
| `ushort2half_*` | `sycl::half ushort2half_<sfx>(unsigned short)` |

Literal documented names: `float2half_rd` `float2half_rn` `float2half_ru` `float2half_rz` `half2int_rd` `half2int_rn` `half2int_ru` `half2int_rz` `half2ll_rd` `half2ll_rn` `half2ll_ru` `half2ll_rz` `half2short_rd` `half2short_rn` `half2short_ru` `half2short_rz` `half2uint_rd` `half2uint_rn` `half2uint_ru` `half2uint_rz` `half2ull_rd` `half2ull_rn` `half2ull_ru` `half2ull_rz` `half2ushort_rd` `half2ushort_rn` `half2ushort_ru` `half2ushort_rz` `int2half_rd` `int2half_rn` `int2half_ru` `int2half_rz` `ll2half_rd` `ll2half_rn` `ll2half_ru` `ll2half_rz` `short2half_rd` `short2half_rn` `short2half_ru` `short2half_rz` `uint2half_rd` `uint2half_rn` `uint2half_ru` `uint2half_rz` `ull2half_rd` `ull2half_rn` `ull2half_ru` `ull2half_rz` `ushort2half_rd` `ushort2half_rn` `ushort2half_ru` `ushort2half_rz`.

### Type Casting Functions for bfloat16 Type

`sycl::ext::oneapi::bfloat16` is the bfloat16 type.

| Family | Signature |
|---|---|
| `bfloat162float` | `float bfloat162float(sycl::ext::oneapi::bfloat16)` |
| `bfloat162int_*` | `int bfloat162int_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `bfloat162ll_*` | `long long bfloat162ll_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `bfloat162short_*` | `short bfloat162short_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `bfloat162uint_*` | `unsigned int bfloat162uint_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `bfloat162ull_*` | `unsigned long long bfloat162ull_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `bfloat162ushort_*` | `unsigned short bfloat162ushort_<sfx>(sycl::ext::oneapi::bfloat16)` |
| `double2bfloat16` | `sycl::ext::oneapi::bfloat16 double2bfloat16(double x)` (round-to-nearest-even) |
| `float2bfloat16` | `sycl::ext::oneapi::bfloat16 float2bfloat16(float x)` (round-to-nearest-even) |
| `float2bfloat16_*` | `sycl::ext::oneapi::bfloat16 float2bfloat16_<sfx>(float x)` |
| `int2bfloat16_*` | `sycl::ext::oneapi::bfloat16 int2bfloat16_<sfx>(int x)` |
| `ll2bfloat16_*` | `sycl::ext::oneapi::bfloat16 ll2bfloat16_<sfx>(long long x)` |
| `short2bfloat16_*` | `sycl::ext::oneapi::bfloat16 short2bfloat16_<sfx>(short x)` |
| `uint2bfloat16_*` | `sycl::ext::oneapi::bfloat16 uint2bfloat16_<sfx>(unsigned int x)` |
| `ull2bfloat16_*` | `sycl::ext::oneapi::bfloat16 ull2bfloat16_<sfx>(unsigned long long x)` |
| `ushort2bfloat16_*` | `sycl::ext::oneapi::bfloat16 ushort2bfloat16_<sfx>(unsigned short x)` |

Literal documented names: `bfloat162int_rd` `bfloat162int_rn` `bfloat162int_ru` `bfloat162int_rz` `bfloat162ll_rd` `bfloat162ll_rn` `bfloat162ll_ru` `bfloat162ll_rz` `bfloat162short_rd` `bfloat162short_rn` `bfloat162short_ru` `bfloat162short_rz` `bfloat162uint_rd` `bfloat162uint_rn` `bfloat162uint_ru` `bfloat162uint_rz` `bfloat162ull_rd` `bfloat162ull_rn` `bfloat162ull_ru` `bfloat162ull_rz` `bfloat162ushort_rd` `bfloat162ushort_rn` `bfloat162ushort_ru` `bfloat162ushort_rz` `float2bfloat16_rd` `float2bfloat16_rn` `float2bfloat16_ru` `float2bfloat16_rz` `int2bfloat16_rd` `int2bfloat16_rn` `int2bfloat16_ru` `int2bfloat16_rz` `ll2bfloat16_rd` `ll2bfloat16_rn` `ll2bfloat16_ru` `ll2bfloat16_rz` `short2bfloat16_rd` `short2bfloat16_rn` `short2bfloat16_ru` `short2bfloat16_rz` `uint2bfloat16_rd` `uint2bfloat16_rn` `uint2bfloat16_ru` `uint2bfloat16_rz` `ull2bfloat16_rd` `ull2bfloat16_rn` `ull2bfloat16_ru` `ull2bfloat16_rz` `ushort2bfloat16_rd` `ushort2bfloat16_rn` `ushort2bfloat16_ru` `ushort2bfloat16_rz`.

Bit reinterpretation: `unsigned short bfloat16_as_ushort(sycl::ext::oneapi::bfloat16 x)`;
`short bfloat16_as_short(sycl::ext::oneapi::bfloat16 x)`;
`sycl::ext::oneapi::bfloat16 short_as_bfloat16(short x)`;
`sycl::ext::oneapi::bfloat16 ushort_as_bfloat16(unsigned short x)`.

### Simple Half-Precision Arithmetic Math Functions

Scalar `sycl::half` (two `sycl::half` args, returns `sycl::half` unless noted): `hadd` addition ·
`hadd_sat` addition with saturation to [0.0, 1.0] · `hfma` fused multiply-add · `hfma_sat` fused
multiply-add with saturation to [0.0, 1.0] · `hfma_relu` fused multiply-add with negative result
clamped to 0 · `hmul` multiplication · `hmul_sat` multiplication with saturation to [0.0, 1.0] ·
`hdiv` division · `hsub` subtraction · `hsub_sat` subtraction with saturation to [0.0, 1.0] ·
`hneg` negates the input, `sycl::half hneg(sycl::half x)`.

Element-wise `sycl::half2` (two `sycl::half2` args return `sycl::half2` unless noted): `hadd2`
addition · `hadd2_sat` addition with saturation to [0.0, 1.0] · `hfma2` fused multiply-add ·
`hfma2_sat` fused multiply-add with saturation to [0.0, 1.0] · `hfma2_relu` fused multiply-add with
negative results clamped to 0 · `hmul2` multiplication · `hmul2_sat` multiplication with saturation
to [0.0, 1.0] · `h2div` division · `hneg2` negates each element, `sycl::half2 hneg2(sycl::half2 x)` ·
`hsub2` subtraction · `hsub2_sat` subtraction with saturation to [0.0, 1.0] · `hcmadd` complex
multiply-add, `sycl::half2 hcmadd(sycl::half2 x, sycl::half2 y, sycl::half2 z)`.

`hcmadd` with inputs x, y, z each including two half precision elements x0, x1, y0, y1, z0, z1:

```cpp
sycl::half c0 = x0 * y0 - x1 * y1 + z0;
sycl::half c1 = x0 * y1 + x1 * y0 + z1;
return sycl::half2{c0, c1};
```

### Half-Precision Comparison Functions

Scalar forms return `bool`; signature `bool <name>(sycl::half x, sycl::half y)` unless noted:
`hisnan` — `bool hisnan(sycl::half x)`; true if the input is NAN · `heq` / `hne` — inputs equal /
not equal · `hge` / `hgt` / `hle` / `hlt` — x >= y / x > y / x <= y / x < y · `hequ`, `hgeu`,
`hgtu`, `hleu`, `hltu`, `hneu` — same comparisons, but **if inputs include NAN the final result is
true**.

`sycl::half2` forms returning `bool` (`bool <name>(sycl::half2 x, sycl::half2 y)`; each input has
x0, x1 and y0, y1): `hbeq2`, `hbne2`, `hbge2`, `hbgt2`, `hble2`, `hblt2` compute
`bool b0 = (x0 <op> y0);` `bool b1 = (x1 <op> y1);` and return true only when both b0 and b1 are
true. `hbequ2`, `hbneu2`, `hbgeu2`, `hbgtu2`, `hbleu2`, `hbltu2` add: if either x0 or y0 is NAN, b0
is true and if either y0 or y1 is NAN, b1 is true [sic: source repeats `y0`]; returns true only when
both are true.

`sycl::half2` forms returning `sycl::half2` (`sycl::half2 <name>(sycl::half2 x, sycl::half2 y)`):
`heq2`, `hne2`, `hge2`, `hgt2`, `hle2`, `hlt2` compute
`sycl::half b0 = (x0 <op> y0) ? 1.0 : 0.0;` `sycl::half b1 = (x1 <op> y1) ? 1.0 : 0.0;` and return
`sycl::half2{b0, b1}`. `hequ2`, `hneu2`, `hgeu2`, `hgtu2`, `hleu2`, `hltu2` add: if either x0 or y0
is NAN, b0 is 1.0 and if either y0 or y1 is NAN, b1 is 1.0 [sic: source repeats `y0`].

## IMF Transcendental Math Functions

The Intel Math Functions (IMF) Device Library is a set of standard math functions implemented for
execution on SYCL devices (GPU, CPU, and accelerators). Most IMF functions comply with ISO C99,
SYCL, OpenCL™, and IEEE754 in terms of computed outputs and IEEE754-special values processing.
Interfaces are available through:

```cpp
#include <sycl/ext/intel/math.hpp>
```

### Accuracy

- **default**: default accuracy compliant to the best of OpenCL/SYCL/CUDA requirements.
- **ha**: High accuracy (ULP not greater than 1.0).
- **la**: Low accuracy (ULP not greater than 4.0).
- **ep**: Enhanced performance (approximately half of the mantissa bits are correct).

All functions provide default accuracy; a subset has additional `ha`, `la`, and `ep` flavors. The
extended versions use nested namespaces, for example:

```cpp
float sycl::ext::intel::math::acos ( float x );     // default accuracy
float sycl::ext::intel::math::ha::acos ( float x ); // ha (High Accuracy)
float sycl::ext::intel::math::la::acos ( float x ); // la (Low Accuracy)
float sycl::ext::intel::math::ep::acos ( float x ); // ep (Enhanced Performance)
```

Accuracy is measured in ULPs on uniformly distributed random inputs along commonly used
function-specific work intervals, plus: values with random mantissa and all possible exponent
fields; corner cases (sub-normals, largest normal values, etc.); IEEE754-special numbers (zeroes,
Inf(A)'s, NaN's, etc.).

**Accuracy table (ULPs).** Each cell lists `default/ha/la/ep`; `-` means that flavor is not
documented for that precision. Extracted from a PDF-broken table (pp. 901–904); rows with fewer
values than columns were placed positionally `[unclear in source]`. The source column header prints
`fp16 (sylc::half)` [sic: `sycl::half`].

| function | fp64 (double) | fp32 (float) | fp16 |
|---|---|---|---|
| acos | 0.79/0.79/2.27/4.0E+07 | 3.0/0.78/3.0/525.0 | - |
| asin | 0.72/0.72/2.61/4.1E+07 | 3.73/0.69/3.73/535.0 | - |
| atan | 0.65/0.65/2.14/2.2E+07 | 0.87/0.87/3.05/2.2E+03 | - |
| atan2 | 0.76/0.76/2.31/2.2E+07 | 2.65/0.87/2.65/436 | - |
| acosh | 1.37/0.89/1.37/- | 1.39/0.86/1.39/1.6E+03 | - |
| asinh | 1.6/0.62/1.6/- | 1.58/0.68/1.58/1.6E+03 | - |
| atanh | 2.12/0.65/2.12/- | 1.85/0.56/1.85/1.5E+03 | - |
| ceil | 0.0 | 0.0 | - |
| cbrt | 0.73 | 0.79 | - |
| copysign | 0.0 | 0.0 | - |
| cdfnorm*** | 1.0 | 1.12 | - |
| cdfnorminv*** | 2.0 | 3.46 | - |
| cos | 0.85/0.85/3.23/6.1E+07 | 1.79/0.64/1.79/2.5E+03 | 1.43 |
| cosh | 0.75/0.75/1.42/- | 1.99/0.56/1.99/380.0 | - |
| cospi | 1.0 | 1.78 | - |
| erf | 0.82/0.82/2.07/7.03 | 0.90/0.90/2.16/6.33 | - |
| erfc | 2.92/0.75/2.92/- | 2.72/0.76/2.72 | - |
| erfcinv | 1.0 | 3.15 | - |
| erfcx | 2.0 | 2.34 | - |
| erfinv | 1.41 | 1.0 | - |
| exp10 | 1.0/0.51/1.00/2.8E+07 | 0.93/0.93 | - |
| exp2 | 0.71/0.71/1.07/6.0E+04 | 0.68/0.68 | 1.66 |
| exp | 0.92/0.92/1.25/1.7E+07 | 0.82/0.82 | 1.61/0.83/1.61 `[unclear]` |
| expm1 | 0.75/0.75/1.76/1.1E+07 | 0.74/0.74/1.69/328.0 | - |
| fdim | 0.0 | 0.0 | - |
| floor | 0.0 | 0.0 | - |
| fmod | 0.0 | 0.0 | - |
| frexp | 0.0 | 0.0 | - |
| hypot | 1.12/0.85/1.12/- | 0.96/0.5/0.96 | - |
| cyl_bessel_i0 | 1.36 | 5.21 | - |
| cyl_bessel_i1 | 2.77 | 5.69 | - |
| j0 | 3.81 | 2.78 | - |
| j1 | 3.01 | 2.38 | - |
| jn | 2.7E+03 | 8.0E+01 | - |
| lgamma | 3.52 | 2.99 | - |
| ilogb | 0.0 | 0.0 | - |
| isfinite | 0.0 | 0.0 | - |
| isinf | 0.0 | 0.0 | - |
| isnan | 0.0 | 0.0 | - |
| ldexp | 0.0 | 0.0 | - |
| llrint | 0.0 | 0.0 | - |
| llround | 0.0 | 0.0 | - |
| log | 0.5/0.5/1.35/4.0E+07 | 0.94/0.94/1.14/1.5E+03 | 0.59 |
| log10 | 0.5/0.5/1.9/- | 1.58/0.72/1.58/989.0 | 0.58 |
| log1p | 0.77/0.77/1.6/- | 0.55/0.55/1.73/1.6E+03 | - |
| log2 | 0.5/0.5/1.58/- | 0.71/0.71/1.93/889.0 | 0.6 |
| logb | 0.0 | 0.0 | - |
| lrint | 0.0 | 0.0 | - |
| lround | 0.0 | 0.0 | - |
| modf | 0.0 | 0.0 | - |
| nan | 0.0 | 0.0 | - |
| nearbyint | 0.0 | 0.0 | - |
| nextafter | 0.0 | 0.0 | - |
| norm | 1.31 | 1.46 | - |
| norm3d | 0.5 | 1.04 | - |
| norm4d | 0.5 | 1.09 | - |
| pow | 0.98/0.85/0.98/- | 1.05/0.78/1.05/1.8E+03 | - |
| powi | 1.48 | 18.4 | - |
| rcbrt | 0.53 | 0.85 | - |
| remainder | 0.0 | 0.0 | - |
| remquo | 0.0 | 0.0 | - |
| rhypot | 0.75 | 1.36 | - |
| rint | 0.0 | 0.0 | - |
| rnorm | 2.2 | 1.66 | - |
| rnorm3d | 0.74 | 1.24 | - |
| rnorm4d | 0.75 | 1.26 | - |
| round | 0.0 | 0.0 | - |
| saturate | - | 0.0 | - |
| scalbn | 0.0 | 0.0 | - |
| signbit | 0.0 | 0.0 | - |
| sin | 0.85/0.85/3.15/6.1E+07 | 1.96/0.65/1.96/2.5E+03 | 1.88 |
| sincos | 1.49/0.85/1.49/2.8E+07 | 2.38/0.86/2.38 | - |
| sincospi | 2.0 | 1.78 | - |
| sinh | 1.74/0.79/1.74/- | 1.34/0.68/1.34/1.1E+03 | - |
| sinpi | 1.0 | 1.78 | - |
| tan | 0.52/0.52/3.01/5.2E+07 | 3.88/0.76/3.88 | - |
| tanh | 0.65/0.65/2.11/- | 0.57/0.57/1.36/1.5E+03 | - |
| tgamma | 9.06 | 3.01 | - |
| trunc | 0.0 | 0.0 | - |
| y0 | 5.47 | 3.2 | - |
| y1 | 3.64 | 4.86 | - |
| yn | 2.0E+03 | 145.0 | - |

Notes: the accuracy of the inlined `inv`, `sqrt` and `rsqrt` is defined by the OpenCL™/SYCL
standards and may be affected by the `-f[no-]fast-math` compiler switch. The ULP ranges come from
random sampling over a large number of data points, so the actual ULP value might be higher for
specific argument values. `cdfnorm` and `cdfnorminv` have CUDA-specific aliases `normcdf` and
`normcdfinv`, mapped to the same computation kernels (`***` in the table).

### IMF Device Library Usage Example

Power function calling example:

```cpp
#include <sycl/ext/intel/math.hpp>
...
sycl::queue{}.submit([&](sycl::handler& h) {
    sycl::accessor out{a, h};
    h.parallel_for(r, [=](sycl::item<1> idx) {
      out[idx] = sycl::ext::intel::math::pow(x, y);
    });
  });
```

`pow.cpp` prompts for two numbers and performs the power calculation with SYCL parallel computing;
the computed result and the expected reference result are printed for comparison:

```cpp
/* file: pow.cpp */
#include <iostream>
#include <sycl/ext/intel/math.hpp>
#include <sycl/sycl.hpp>

/**
 * Number of inputs
 */
constexpr int num = 1;

/**
 * @brief Entry point of the program.
 *
 * This function calculates the power of two numbers using the SYCL framework.
 * It prompts user to enter two numbers, and then performs the power calculation
 * using SYCL parallel computing. The calculated results are printed to the console.
 * The expected result is also printed for comparison.
 *
 * @return 0 indicating successful execution of the program.
 */
int main() {
   float x, y;
   auto r = sycl::range{num};
   sycl::buffer<float> a{r};

    std::cout << "Enter x and y: ";
    std::cin >> x >> y;

    sycl::queue{}.submit([&](sycl::handler& h) {
      sycl::accessor out{a, h};
      h.parallel_for(r, [=](sycl::item<1> idx) {
        out[idx] = sycl::ext::intel::math::pow(x, y);
      });
    });

    sycl::host_accessor result{a};

    for (int i = 0; i < num; ++i) {
      std::cout << "Computed pow(" << x << ", " << y << ") = " << result[i]
                << "\n";
    }

    std::cout << "Expected pow(" << x << ", " << y << ") = " << std::pow(x, y)
              << "\n";

    return 0;
}
```

Compile and run with the SYCL compiler set up in the environment:

```bash
make exe
make run
make clean
```

Makefile example:

```make
# file: makefile

default: all
exe:     pow.exe
all:     run

SHELL := /bin/bash

pow.exe: pow.cpp
    icx -fsycl ./pow.cpp -o ./pow.exe

run: pow.exe
    ./pow.exe

clean:
    @rm -f ./pow.exe
```

### IMF Device Library Function List (names by type)

Trigonometric `acos` `asin` `atan` `atan2` `cos` `cospi` `sin` `sincos` `sincospi` `sinpi` `tan`;
hyperbolic `acosh` `asinh` `atanh` `cosh` `sinh` `tanh`; exponential `exp` `exp10` `exp2` `expm1`;
logarithmic `ilogb` `log` `log10` `log1p` `log2` `logb`; power `cbrt` `hypot` `inv` `norm` `norm3d`
`norm4d` `pow` `powi` `rcbrt` `rhypot` `rnorm` `rnorm3d` `rnorm4d` `rsqrt` `sqrt`; special `cdfnorm`
`cdfnorminv` `cyl_bessel_i0` `cyl_bessel_i1` `erf` `erfc` `erfcinv` `erfcx` `erfinv` `j0` `j1` `jn`
`lgamma` `tgamma` `y0` `y1` `yn`; rounding `ceil` `floor` `llrint` `llround` `lrint` `lround`
`nearbyint` `rint` `round` `trunc`; miscellaneous `copysign` `fdim` `fmod` `frexp` `isfinite`
`isinf` `isnan` `ldexp` `modf` `nan` `nextafter` `remainder` `remquo` `saturate` `scalbn` `signbit`.

**IMF signature convention.** All functions are in `sycl::ext::intel::math`; `ha`/`la`/`ep` flavors
are the same name in nested namespaces. **Unless a row states otherwise, the function provides
`float f(float x)` and `double f(double x)` returning the same type and documents all of
`def/ha/la/ep`.** Rows note deviations (`acc` = documented accuracy flavors). `SV` = special values
(argument → result); `S/QNAN` = signaling/quiet NaN, `∞` = infinity.

### IMF Device Library Trigonometric Functions

- `acos` — principal inverse cosine, [0, π] rad, x in [-1, 1]. SV: -1→+π; ±0→+π/2; +1→+0;
  |x| > 1→QNAN; ±∞→QNAN.
- `asin` — principal inverse sine, [-π/2, +π/2] rad, x in [-1, 1]. SV: -1→-π/2; ±0→±0; +1→+π/2;
  |x| > 1→QNAN; ±∞→QNAN.
- `atan` — principal inverse tangent, [-π/2, +π/2] rad. SV: ±0→±0; ±∞→±π/2.
- `atan2` — `atan2(y, x)` principal inverse tangent of y/x, [-π, +π] rad. `acc` (source's LA block
  prints `ep::atan2` [sic]). SV: -∞,-∞→-3*π/2; -∞,x<+0→-π/2; -∞,-0→-π/2; -∞,+0→-π/2; -∞,x>+0→-π/2;
  -∞,+∞→-π/4; y<+0,-∞→-π; y<+0,-0→-π/2; y<+0,+0→-π/2; y<+0,+∞→-0; -0,-∞→-π; -0,x<+0→-π; -0,-0→-π;
  -0,+0→-0; -0,x>+0→-0; -0,+∞→-0; +0,-∞→+π; +0,x<+0→+π; +0,-0→+π; +0,+0→+0; +0,x>+0→+0;
  +0,+∞→+0; y>+0,-∞→+π; y>+0,-0→+π/2; y>+0,+0→+π/2; y>+0,+∞→+0; +∞,-∞→+3*π/4; +∞,x<+0→+π/2;
  +∞,-0→+π/2; +∞,+0→+π/2; +∞,x>+0→+π/2; +∞,+∞→+π/4; any y,S/QNAN→QNAN; S/QNAN,any x→QNAN.
  (`-3*π/2`, `+3*π/4` as printed [sic].)
- `cos` — cosine of x in radians. `acc`, plus `def` for `sycl::half`. SV: ±0→+1; ±∞→QNAN.
- `cospi` — cos(x·π). `acc def` only (f,d). SV: ±0→+1; n + 0.5→+0; ±∞→QNAN; n is any integer
  number where n + 0.5 is representable. Identity: `cospi(x) = cos(x·PI)`.
- `sin` — sine of x in radians. `acc`, plus `def` for `sycl::half`. SV: ±0→±0; ±∞→QNAN.
- `sincos` — `sincos(x, &s, &c)`: s=sin(x), c=cos(x), radians. Default:
  `float sincos(float x, float* s, float* c)`; `double sincos(double x, double* s, double* c)`.
  HA/LA documented as `sincos(float x)` / `sincos(double x)` [sic: out-pointer params omitted]; EP:
  `double` only. SV: ±0→s=±0,c=+1; ±∞→s=QNAN,c=QNAN.
- `sincospi` — `sincospi(x, &s, &c)`: s=sin(x·π), c=cos(x·π). `acc def` only (f,d). Default:
  `float sincospi(float x, float* s, float* c)`; `double sincospi(double x, double* s, double* c)`.
  SV: ±0→s=±0,c=+1; ±n (any integer n)→s=±0,c=cospi(x); n + 0.5 (any integer n where n + 0.5 is
  representable)→s=sinpi(x),c=+0; ±∞→s=QNAN,c=QNAN.
- `sinpi` — sin(x·π). `acc def` only (f,d). SV: ±0→±0; ±n*→±0; ±∞→QNAN.
  Identity: `sinpi(x) = sin(x·PI)`.
- `tan` — tangent of x in radians. `acc` except `ep` is `double` only. SV: ±0→±0; ±∞→QNAN.
  Identity: `tan(x) = sin(x) / cos(x)`.

### IMF Device Library Hyperbolic Functions

Each function's `ep` flavor is `float` only; all others `def/ha/la` (f,d).

- `acosh` — inverse hyperbolic cosine. SV: +1→+0; x<+1→QNAN; -∞→QNAN; +∞→+∞.
  Identity: `acosh(x) = log( x + sqrt(x^2 - 1) )`.
- `asinh` — inverse hyperbolic sine. SV: ±0→±0; ±∞→±∞.
  Identity: `asinh(x) = log( x + sqrt(x^2 + 1) )`.
- `atanh` — inverse hyperbolic tangent. SV: ±1→±∞; |x|>1→QNAN; ±∞→QNAN.
  Identity: `atanh(x) = 0.5 · log( (1 + x)/(1 - x) )`.
- `cosh` — hyperbolic cosine. SV: ±0→+1; x>+OVFL→+∞; x<-OVFL→+∞; ±∞→+∞. OVFL is the overflow
  threshold, `OVFL = log(MAX)+log(2)`, MAX the maximum floating point normal number for the
  precision. Identity: `cosh(x) = ( exp(x) + exp(-x) ) / 2`.
- `sinh` — hyperbolic sine. SV: ±0→±0; x>+OVFL→+∞; x<-OVFL→-∞; ±∞→±∞;
  `OVFL = log(MAX)+log(2)`. Identity: `sinh(x) = ( exp(x) - exp(-x) ) / 2`.
- `tanh` — hyperbolic tangent. SV: ±0→±0; ±∞→±1.
  Identity: `tanh(x) = ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )`.

### IMF Device Library Exponential Functions

- `exp` — e (Euler's number ~2.7182818) raised to x: e^x. `acc def/HA` for `sycl::half`, `float`,
  `double`; `LA/EP` for `sycl::half`, `double`. SV: -∞→+0; x<-OVFL→+0; ±0→+1; x>+OVFL→+∞; +∞→+∞.
  `OVFL` is the overflow/underflow threshold, `OVFL = log(MAX)`. Identities:
  `exp(x) = 2 · tanh( x/2 )/(1 - tanh( x/2 )) + 1`; `exp(x) = cosh(x) + sinh(x)`;
  `exp(x) = exp2( log2(e)·x )`; `exp(x) = exp10( log10(e)·x )` (source prints a trailing `>` on the
  last two [sic]).
- `exp10` — 10^x. `acc def/HA` (f,d); `LA/EP` (d). SV: -∞→+0; x<-OVFL→+0; ±0→+1; x>+OVFL→+∞;
  +∞→+∞; `OVFL = log10(MAX)`. Identities: `exp10(x) = exp( log(10)·x )`;
  `exp10(x) = exp2( log2(10)·x )`.
- `exp2` — 2^x. `acc def` for `sycl::half`, `float`, `double`; `HA` (f,d); `LA/EP` (d).
  SV: -∞→+0; x<-OVFL→+0; ±0→+1; x>+OVFL→+∞; +∞→+∞; `OVFL = log2(MAX)`.
  Identities: `exp2(x) = exp( log(2)·x )`; `exp2(x) = exp10( log10(2)·x )`.
- `expm1` — e^x-1; more accurate than `exp(x)-1.0` for small x and produces no underflows; useful
  for financial calculations such as small daily interest rates, where `(1+x)^n-1` is computed as
  `expm1( n·log1p(x) )`. SV: -∞→-1; ±0→+0; x>+OVFL→+∞; +∞→+∞; `OVFL = log(MAX)`.
  Identities: `expm1(x) = exp(x) - 1`; `expm1(x) = 2 · tanh( x/2 )/(1 - tanh( x/2 ))`.

### IMF Device Library Logarithmic Functions

- `ilogb` — unbiased base-2 exponent of x as a signed integer. `acc def` only; signatures
  `int ilogb(float x)`, `int ilogb(double x)`. SV: ±0→INT_MIN; ±∞→INT_MAX; S/QNAN→INT_MIN;
  INT_MIN/INT_MAX are the minimum and maximum signed integer values.
- `log` — natural logarithm. `acc`, plus `def` for `sycl::half`. SV: -∞→QNAN; x<+0→QNAN; ±0→-∞;
  +∞→+∞. Identities: `log(x) = log2(x) / log2(e)`; `log(x) = log10(x) / log10(e)`.
- `log10` — base-10 logarithm. `acc`, plus `def` for `sycl::half`, except `ep` is `float` only.
  SV: -∞→QNAN; x<+0→QNAN; ±0→-∞; +∞→+∞. Identities (one line in source):
  `log10(x) = log2(x) / log2(10)`; `log10(x) = log(x) / log(10)`.
- `log1p` — natural logarithm of x+1; more accurate than `log(x + 1)` if x is close to zero, when
  the (x + 1) result can be rounded to 1 in the target precision; particularly useful for financial
  calculations such as small daily interest rates using `(1+x)^n-1` computed as
  `expm1( n·log1p(x) )`. `acc` except `ep` is `float` only. SV: -∞→QNAN; x<-1→QNAN; -1→-∞; ±0→±0;
  +∞→+∞. Identity: `log1p(x) = log(x+1), excepts |x| close to zero` [sic].
- `log2` — base-2 logarithm. `acc`, plus `def` for `sycl::half`, except `ep` is `float` only.
  SV: -∞→QNAN; x<+0→QNAN; ±0→-∞; +∞→+∞. Identities: `log2(x) = log10(x) / log10(2)`;
  `log2(x) = log(x) / log(2)`.
- `logb` — unbiased radix-independent signed exponent of x as a floating-point value.
  `acc def` only (f,d). SV: -∞→+∞; ±0→-∞; +∞→+∞.

### IMF Device Library Power Functions

- `cbrt` — **inlined**; cube root x^(1/3). Not equivalent to `pow(x, 1.0/3.0)` because the rational
  number 1/3 is not equal to the floating point number 1.0/3.0; `cbrt(x)` usually gives a more
  accurate result than `pow(x, 1.0/3.0)`. `acc def` only (f,d). SV: ±∞→±∞; ±0→±0.
- `hypot` — `hypot(y, x)` = sqrt( x^2 + y^2 ); length of the hypotenuse of a right triangle with
  sides x and y without undue overflow or underflow. `acc` except no `ep`. SV: ±0,±0→+0;
  ±∞,any y→+∞; any x,±∞→+∞.
- `inv` — **inlined**; 1/x. `acc def` only (f,d). SV: ±0→±∞; ±∞→±0. Special-values processing rules
  of inlined `inv` may be affected by the `-f[no-]fast-math` compiler switch.
- `norm` — `norm(dim, x)` = sqrt( x[0]^2 + x[1]^2 + ... + x[dim-1]^2 ); dim-D vector length
  without undue overflow or underflow. `acc def` only; `float norm(int dim, float* x)`,
  `double norm(int dim, double* x)`. SV: any x[i] = ±∞→+∞.
- `norm3d` — `norm3d(x, y, z)` = sqrt( x^2 + y^2 + z^2 ). `acc def` only (f,d). SV: any x,y,z = ±∞→+∞.
- `norm4d` — `norm4d(w, x, y, z)` = 4-D vector length. Source prints the formula as
  `sqrt( x^2 + x^2 + y^2 + z^2 )` [sic: likely w^2 + x^2 + y^2 + z^2]. `acc def` only.
  SV: any w,x,y,z = ±∞→+∞.
- `pow` — `pow(x, y)` = x^y. `acc` except `ep` is `float` only. SV: -∞,+∞→+∞; -∞,-∞→+0;
  -∞,neg. odd int→-0; -∞,neg. even int→+0; -∞,neg. non-int→+0; -∞,pos. odd int→-∞;
  -∞,pos. even int→+∞; -∞,pos. non-int→+∞; -∞,±0→+1; -1,±∞→+1; x<+0,non-int→QNAN;
  -0,neg. non-int→+∞; ±0,neg. odd int→±∞; ±0,neg. even int→+∞; ±0,pos. even int→+0;
  ±0,pos. non-int→+0; ±0,+∞→+0; ±0,pos. odd int→±0; ±0,-∞→+∞; ±0,-0→+1; ±0,+0→+1; |x|<1,-∞→+∞;
  |x|<1,+∞→+0; +1,any y→+1; +1,±0→+1; +1,±∞→+1; +1,S/QNAN→+1; |x|>1,-∞→+0; |x|>1,+∞→+∞;
  +∞,±0→+1; +∞,y<+0→+0; +∞,y>+0→+∞; +∞,-∞→+0; +∞,+∞→+∞; any x,+0→+1; any x,-0→+1;
  S/QNAN,+0→+1; S/QNAN,-0→+1. Identity: `pow(x, y) = exp( y·log(x) ), in low accuracy and possible
  special values processing incompliance`.
- `powi` — `powi(x, n)` = x^n for integer n; `float powi(float x, int n)`,
  `double powi(double x, int n)`. `acc def` only. SV: -∞,neg. odd int→-0; -∞,neg. even int→+0;
  -∞,pos. odd int→-∞; -∞,pos. even int→+∞; -∞,0→+1; ±0,neg. odd int→±∞; ±0,neg. even int→+∞;
  ±0,pos. even int→+0; ±0,pos. odd int→±0; ±0,0→+1; +1,any n→+1; +1,0→+1; +∞,0→+1; +∞,n<0→+0;
  +∞,n>0→+∞; any x,0→+1; S/QNAN,0→+1.
- `rcbrt` — **inlined**; inverse cube root x^(-1/3). Not equivalent to `pow(x, -1.0/3.0)` because
  the rational number -1/3 is not equal to the floating point number -1.0/3.0; usually more
  accurate. `acc def` only (f,d). SV: ±0→±∞; ±∞→±0.
- `rhypot` — `rhypot(y, x)` = 1 / sqrt( x^2 + y^2 ), without undue overflow or underflow.
  `acc def` only (f,d). SV: ±0,±0→+∞; ±∞,any y→+0; any x,±∞→+0.
- `rnorm` — `rnorm(dim, x)` = 1 / sqrt( x[0]^2 + x[1]^2 + ... + x[dim-1]^2 ); `acc def` only;
  `float rnorm(int dim, float* x)`, `double rnorm(int dim, double* x)`. SV: any x[i] = ±∞→+0.
- `rnorm3d` — `rnorm3d(x, y, z)` = 1 / sqrt( x^2 + y^2 + z^2 ). `acc def` only (f,d).
  SV: any x,y,z = ±∞→+0.
- `rnorm4d` — `rnorm4d(w, x, y, z)` = 1 / sqrt( w^2 + x^2 + y^2 + z^2 ). `acc def` only (f,d).
  SV: any w,x,y,z = ±∞→+0.
- `rsqrt` — **inlined**; inverse square root x^(-1/2). `acc def` only (f,d). SV: -∞→QNAN; x<+0→QNAN;
  ±0→±∞; +∞→+0. Special-values processing rules of inlined `rsqrt` may be affected by the
  `-f[no-]fast-math` compiler switch.
- `sqrt` — **inlined**; square root x^(1/2). `acc def` only (f,d). SV: -∞→QNAN; x<+0→QNAN; ±0→±0;
  +∞→+∞. Special-values processing rules of inlined `sqrt` may be affected by `-f[no-]fast-math`.

### IMF Device Library Special Functions

Each `acc def` only unless noted.

- `cdfnorm` — cumulative normal distribution function value; useful for Monte Carlo applications in
  computational finance. SV: -∞→+0; x<UDFL→+0; +∞→+1. UDFL is the underflow threshold,
  `UDFL = ~ -14.2` for float, `~ -38.5` for double. Identities:
  `cdfnorm(x) = 1/2 · ( 1 + erf( x/sqrt(2) ) )`; `cdfnorm(x) = 1 - 1/2 · erfc( x/sqrt(2) )`.
  CUDA alias `normcdf` (the source's p.931 CUDA block prints `cdfnorm` again [sic]).
- `cdfnorminv` — inverse cumulative normal distribution function value; useful for Monte Carlo
  applications in computational finance. SV: -∞→QNAN; x<-0→QNAN; ±0→-∞; +0.5→+0; +1→+∞; x>+1→QNAN;
  +∞→QNAN. Identities: `cdfnorminv(x) = sqrt(2) · erfinv( 2·x - 1 )`;
  `cdfnorminv(x) = sqrt(2) · erfcinv( 2 - 2·x )`. CUDA naming:
  `float sycl::ext::intel::math::normcdfinv ( float x );`
  `double sycl::ext::intel::math::normcdfinv ( double x );`
- `cyl_bessel_i0` — regular modified cylindrical Bessel function, order 0. SV: -∞→+∞; ±0→+1;
  +∞→+∞.
- `cyl_bessel_i1` — regular modified cylindrical Bessel function, order 1. SV: -∞→-∞; ±0→±0; +∞→+∞.
- `erf` — error function value. `acc` (all four). SV: -∞→-1; +∞→+1. Identity: `erf(x) = 1 - erfc(x)`.
- `erfc` — complementary error function 1 - erf(x); useful when erf(x) is close to 1 to obtain
  greater accuracy. `acc` except no `ep`. SV: -∞→+2; x>UDFL→+0; +∞→+0. UDFL is the underflow
  threshold, `UDFL = ~9.2` for float, `~26.6` for double. Identity: `erfc(x) = 1 - erf(x)`.
- `erfcinv` — inverse complementary error function erfinv(1-x); useful for Monte Carlo applications
  in computational finance (closely related to the normal cumulative distribution); use it to
  replace expressions containing `erfinv(1-x)` for greater accuracy when x is close to 1.
  SV: -∞→QNAN; x<-0→QNAN; ±0→+∞; +1→+0; +2→-∞; x>+0→QNAN [sic: source likely x>+2]; +∞→QNAN.
  Identity: `erfcinv(x) = erfinv(1-x)`.
- `erfcx` — scaled complementary error function exp(x^2)·erfc(x); occurs in diffusion problems of
  physics and chemistry and avoids underflow/overflow errors from computing exp(x^2) and erfc(x)
  separately. SV: -∞→+∞; +∞→+0. Identity: `erfcx(x) ~ (1/sqrt(π))/x, for large x`.
- `erfinv` — inverse error function value; useful for Monte Carlo applications in computational
  finance (closely related to the normal cumulative distribution). SV: -∞→QNAN; ±0→±0; ±1→±∞;
  |x|>1→QNAN; +∞→QNAN. Identity: `erfinv(x) = erfcinv(1-x)`.
- `j0` — Bessel function of the first kind, order 0. SV: -∞→+0; ±0→+1; +∞→+0.
- `j1` — Bessel function of the first kind, order 1. SV: -∞→+0; ±0→±0; +∞→+0.
- `jn` — `jn(n, x)` Bessel function of the first kind, order n; `float jn(int n, float x)`,
  `double jn(int n, double x)`. SV: any n,-∞→+0; 0,±0→+1; n>=1,±0→±0; any n,+∞→+0.
- `lgamma` — natural logarithm of the absolute value of gamma, `log( |tgamma(x)| )`. SV: -∞→+∞;
  neg, int→+∞; ±0→+∞; +1→+0; +2→+0; +∞→+∞. Identity:
  `lgamma(x) = log(|tgamma(x)|), less accurate and impacted by overflows in tgamma`.
- `tgamma` — gamma function value. SV: -∞→QNAN; neg, int→QNAN; ±0→±∞; x>OVFL→+∞; +∞→+∞. OVFL is
  the overflow threshold, `OVFL = ~ 35.1` for float, `~ 171.7` for double.
- `y0` — Bessel function of the second kind, order 0. SV: -∞→QNAN; x<+0→QNAN; ±0→-∞; +∞→+0.
- `y1` — Bessel function of the second kind, order 1. SV (source table header prints `Result y0(x)`
  [sic]): -∞→QNAN; x<+0→QNAN; ±0→-∞; +∞→+0.
- `yn` — `yn(n, x)` Bessel function of the second kind, order n; `float yn(int n, float x)`,
  `double yn(int n, double x)`. SV: any n,-∞→QNAN; any n,x<+0→QNAN; any n,±0→-∞; any n,+∞→+0.

### IMF Device Library Rounding Functions

All `acc def` only (f,d).

- `ceil` — integer value rounded towards plus infinity. SV: ±∞→±∞; -1.5→-1; ±0→+0; +1.5→+2.
- `floor` — integer value rounded towards minus infinity. SV: ±∞→±∞; -1.5→-2; ±0→+0; +1.5→+1.
- `llrint` — integer value in the executing device's current rounding mode, returned as
  `long long int`. Linux: technically the same as `lrint(x)` since `long int` is the same 64-bit
  data type as `long long int`; Windows: different because `long int` is 32-bit while
  `long long int` is 64-bit (source prints `ling long int` for the double overload [sic]).
  SV (rounding mode set to nearest integer): ±∞→±∞; -3.5→-4; -2.5→-2; -1.5→-2; -0.5→-0; ±0→+0;
  +0.5→+0; +1.5→+2; +2.5→+2; +3.5→+4.
- `llround` — value rounded to the nearest integer, returned as `long long int`; input elements
  halfway between two consecutive integers are always rounded **away from zero** regardless of the
  rounding mode. Linux: technically the same as `lround(x)`; Windows: different (long int 32-bit vs
  long long int 64-bit). SV: ±∞→±∞; -1.5→-2; ±0→+0; +1.5→+2.
- `lrint` — integer value in the current device rounding mode, returned as `long int`. Linux:
  technically the same as `llrint(x)`; Windows: different. SV (nearest-integer mode): ±∞→±∞;
  -3.5→-4; -2.5→-2; -1.5→-2; -0.5→-0; ±0→+0; +0.5→+0; +1.5→+2; +2.5→+2; +3.5→+4.
- `lround` — value rounded to the nearest integer, returned as `long int`; halfway cases always
  rounded **away from zero** regardless of rounding mode. Linux: technically the same as
  `llround(x)`; Windows: different. SV: ±∞→±∞; -1.5→-2; ±0→+0; +1.5→+2.
- `nearbyint` — integer value in the current device rounding mode. Per the standard `nearbyint(x)`
  never raises an `FE_INEXACT` exception while `rint(x)` does in exceptional cases; floating-point
  exceptions are not supported on IMF platforms, so technically `nearbyint(x)` is the same as
  `rint(x)` implementation. SV (nearest-integer mode): ±∞→±∞; -3.5→-4; -2.5→-2; -1.5→-2; -0.5→-0;
  ±0→+0; +0.5→+0; +1.5→+2; +2.5→+2; +3.5→+4.
- `rint` — integer value in the current device rounding mode. SV (nearest-integer mode): ±∞→±∞;
  -3.5→-4; -2.5→-2; -1.5→-2; -0.5→-0; ±0→+0; +0.5→+0; +1.5→+2; +2.5→+2; +3.5→+4.
- `round` — value rounded to the nearest integer; halfway cases always rounded **away from zero**
  regardless of rounding mode. SV: ±∞→±∞; -1.5→-2; ±0→+0; +1.5→+2.
- `trunc` — integer value rounded towards zero. SV: ±∞→±∞; -1.5→-1; ±0→+0; +1.5→+1.

### IMF Device Library Miscellaneous Functions

All `acc def` only (f,d) unless noted.

- `copysign` — `copysign(x, y)` creates a value with the given x magnitude and the copied sign of a
  second value y.
- `fdim` — `fdim(x, y)` positive difference x - y for x > y, or zero for x <= y. SV (source table
  header prints `Result fdim(y,x)` [sic]): any x,S/QNAN→QNAN; S/QNAN,any y→QNAN.
- `fmod` — `fmod(x, y)` = x - n·y for integer n; if y is nonzero the result has the same sign as x
  and magnitude less than the magnitude of y. Similar to `remainder` except that it rounds the
  internal quotient n towards zero to an integer instead of to the nearest integer. SV (header
  prints `Result fmod(y,x)` [sic]): x not S/QNAN,±0→QNAN; ±∞,y not S/QNAN→QNAN;
  ±0,y not 0 or S/QNAN→±0; x finite,±∞→x; S/QNAN,any→QNAN; any,S/QNAN→QNAN.
- `frexp` — `frexp(x, exp)` extracts a mantissa and exponent: converts x into a signed normalized
  fraction in (1/2, 1) multiplied by an integral power of two; the fraction is returned and the
  integer exponent stored at `exp`. `float frexp(float x, int* exp)`,
  `double frexp(double x, int* exp)`. SV: ±0→return ±0,exp 0; ±∞→return ±∞,exp undef;
  S/QNAN→return QNAN,exp undef.
- `isfinite` — determines if x has a finite value (normal, subnormal or zero, but not infinite or
  NaN). Returns `int`. SV: x finite→+1; ±∞ or S/QNAN→+0.
- `isinf` — determines if x is positive or negative infinity. Returns `int`. SV: ±∞→+1;
  not ±∞→+0.
- `isnan` — determines if x is a NaN value. Returns `int`. SV: S/QNAN→+1; not S/QNAN→+0.
- `ldexp` — `ldexp(x, exp)` = x·2^exp; on binary systems (where `FLT_RADIX` is 2) equivalent to
  `scalbn`. `float ldexp(float x, int exp)`, `double ldexp(double x, int exp)`.
  SV: ±0,any exp→±0; ±∞,any exp→±∞; any x,exp 0→x.
- `modf` — `modf(x, intptr)` splits x into fractional and integer parts, each with the same sign as
  x; the signed fractional portion is returned and the integer portion stored as a floating-point
  value at `intptr`. `float modf(float x, float* intptr)`, `double modf(double x, double* intptr)`.
  SV: ±0→return ±0,intptr ±0; ±∞→return ±0,intptr ±∞.
- `nan` — `nan(tagp)` returns a representation of a quiet NaN; `tagp` selects one of the possible
  representations. `float nanf(const char* x)`; `double nan(const char* x)`.
- `nextafter` — `nextafter(x, y)` returns the next representable floating-point value after x in the
  direction of y; if y is greater than x it returns the smallest representable number greater than
  x. The returned value is **independent of the execution device's current rounding mode**.
  SV (header prints `Result nextafter(y,x)` [sic]): -0,+0→+0; +0,-0→-0.
- `remainder` — `remainder(x, y)` returns x REM y per IEEE754: x - n·y where n is the integer
  nearest to the exact value of x / y; if two integers are equally close, n is the even one; if n is
  zero it has the same sign as x. Similar to `fmod` except that it rounds n to the nearest integer
  instead of towards zero. SV (header prints `Result remainder(y,x)` [sic]): x not S/QNAN,±0→QNAN;
  ±∞,y not S/QNAN→QNAN; ±0,y not 0 or S/QNAN→±0; x finite,±∞→x; S/QNAN,any→QNAN; any,S/QNAN→QNAN.
- `remquo` — `remquo(x, y, quo)` computes the floating-point remainder like `remainder` and returns
  part of a quotient through the `quo` pointer. `quo` has the same sign as and may not be the exact
  quotient, but agrees with the exact quotient in the low order 3 bits, sufficient to determine the
  octant of the result within a period. `float remquo(float x, float y, int* quo)`,
  `double remquo(double x, double y, int* quo)`. SV (header prints `Return remquo(y,x,quo)` [sic]):
  x not S/QNAN,±0→return QNAN,quo undef; ±∞,y not S/QNAN→QNAN,undef; ±0,y not 0 or S/QNAN→±0,0;
  x finite,±∞→x,0; S/QNAN,any→QNAN,undef; any,S/QNAN→QNAN,undef.
- `saturate` — `saturate(x)` clamps x to the [+0.0, 1.0] range. `float saturate(float x)` (float
  only). SV: x<+0→+0; x>+1→+1; 0<=x<=1→x; S/QNAN→+0.
- `scalbn` — `scalbn(x, exp)` = x·FLT_RADIX^exp; on binary systems (where `FLT_RADIX` is 2)
  equivalent to `ldexp`. `float scalbn(float x, int exp)`, `double scalbn(double x, int exp)`.
  SV: ±0,any exp→±0; ±∞,any exp→±∞; any x,exp 0→x.
- `signbit` — `signbit(x)` detects the sign bit of both finite and infinite values including
  zeroes, infinities, and NaNs. Returns `int`. SV: x neg.→+1; x pos.→+0.

## Figures

The source page range 814–947 contains none of the figures listed in the figure catalog, so no
figure is embedded.

## Gotchas & failure modes

- **`-qopt-report` level vs names.** `-qopt-report-names` works without `-qopt-report`, but to get
  names in a report you must also set the report level. Mangled names match the **assembly** listing;
  demangled names match the **source** listing; demangled is the default.
- **Report differences are expected.** IPO can run at compile or link time, and optimization level /
  PGO change results — the same source can produce different reports.
- **`errno` is not populated by default.** Without `-fmath-errno` the compiler may avoid `errno`
  writes entirely, and `-ffast-math` overrides `-fmath-errno`. Do not rely on EDOM (33) / ERANGE
  (34) if either optimization is in play.
- **Windows `long double` silently becomes `double`.** Compilation succeeds without `/Qlong-double`,
  but 64-bit decimal transcendentals lose the extended precision they rely on. Pair `/Qlong-double`
  with `/Qpc80` to keep x87 at 80-bit precision. Intel `long double` math functions are ABI
  incompatible with Microsoft libraries.
- **`_Complex` needs `[Q]std=c99` and is C-only.** C++ programs cannot use `_Complex` data types.
- **Non-default rounding modes.** Fast transcendental implementations are validated under
  round-to-nearest-even; other modes may reduce accuracy or set unexpected floating-point status
  flags. Use `-fp-model strict` (Linux) / `/fp: strict` (Windows).
- **Inlining changes which functions are optimized.** Automatic inlining of math functions varies
  with vectorization and processor-specific options; `-fno-builtin` (Linux) / `/Oi-` (Windows)
  disables automatic inline expansion. `Qfreestanding-` implies `fno-builtin` (override with
  `fbuiltin`) and is not available for SYCL.
- **Code-size options can backfire.** `ffreestanding`/`Qfreestanding-` may increase binary size and
  suppress optimizations; `ipo`/`Qipo` may increase size and is not recommended if you ship object
  files.
- **FP16 functions are hardware- and version-gated.** Compiler 2021.4+ and a next-generation Intel®
  Xeon® Scalable processor (Sapphire Rapids) are required.
- **IMF accuracy is a performance/accuracy trade.** `ha` (ULP <= 1.0), `la` (ULP <= 4.0), and `ep`
  (~half the mantissa bits correct) are opt-in nested namespaces; the ULP figures are random-sampling
  ranges, so a specific argument can be worse. `inv`, `sqrt`, `rsqrt` accuracy instead comes from
  OpenCL/SYCL standards and is affected by `-f[no-]fast-math`.
- **`-f[no-]fast-math` changes special-value behavior** of the inlined `inv`, `rsqrt`, and `sqrt`.
- **IMF `llrint`/`lrint` and `llround`/`lround` differ by platform.** Linux has 64-bit `long int`, so
  the pairs are technically the same; on Windows `long int` is 32-bit and `long long int` is 64-bit.
- **`nearbyint` vs `rint`.** `nearbyint` never raises `FE_INEXACT` while `rint` does in exceptional
  cases, but floating-point exceptions are unsupported on IMF platforms, so the implementations are
  technically the same.
- **CUDA aliases.** `cdfnorm`/`cdfnorminv` have aliases `normcdf`/`normcdfinv` mapped to the same
  computation kernels.
- **Complex-family documentation gaps.** `cisd`, `cexp10`, and `clog2` appear in the body but are
  missing from the summary Complex Functions list table; `erfcx`, `cpolar`, `cpolarf` and `cabs`
  have fewer precision variants than neighbors — do not assume a variant exists.
- **Source typos — flagged, not silently fixed:** `min` (`int min(int x, int x)`), `minmag`'s third
  signature printed as `maxmagf16`, `umul64hi`'s unnamed second parameter, `hiloint2double`'s `log`
  parameter, `atan2`'s `-3*π/2` / `+3*π/4`, `erfcinv`'s `x > +0`→QNAN row, `y1`/`yn` SV table
  headers, the `sylc::half` column header, `pow`'s trailing `>` identities, and `-0` results in the
  `-3.5`-style rounding tables.

## Source map

- Code size options; Optimization Reports and report options — pp. 814–815
- Compiler Math Library overview, Linux/Windows libraries, oneAPI/OpenCL ULP considerations; Use the Compiler Math Library (real/complex examples); Exception Conditions (`errno`); Other Considerations — pp. 816–821
- Math Function List; Trigonometric Functions — pp. 821–831
- Hyperbolic Functions; Exponential Functions — pp. 831–837
- Special Functions; Nearest Integer Functions; Remainder Functions — pp. 837–845
- Miscellaneous Functions; Complex Functions; C99 Macros — pp. 845–855
- SYCL* Device Library: basic arithmetic/simple math, simple integer math, basic integer arithmetic operations, rounding-mode arithmetic — pp. 855–868
- Type casting for floating-point numbers; half-precision types; bfloat16 — pp. 868–891
- Simple half-precision arithmetic; half-precision comparison functions — pp. 891–900
- IMF Transcendental Math Functions; Accuracy + ULP table; Usage Example (`pow.cpp`, makefile); Function List — pp. 900–908
- IMF Trigonometric — pp. 908–914; Hyperbolic — pp. 914–917; Exponential — pp. 918–920
- IMF Logarithmic — pp. 920–924; Power — pp. 924–930
- IMF Special — pp. 930–938; Rounding — pp. 938–942; Miscellaneous — pp. 942–947
