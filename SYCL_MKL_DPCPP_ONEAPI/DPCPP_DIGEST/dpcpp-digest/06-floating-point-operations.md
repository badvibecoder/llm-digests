---
chunk: 06-floating-point-operations
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0
source_pages: 390-397
covers: FP tradeoffs, -fp-model//fp, -fimf-* accuracy options, math-routine dispatch, denormals, FP environment, FTZ/DAZ, FP tuning
---

# Floating-Point Operations

> **Scope.** FP accuracy/reproducibility/performance tradeoffs; options controlling them
> (`-fp-model`/`/fp`, `-fimf-*`, `-ffast-math`, `-fmath-errno`, `[Q]ftz`); denormals, FP environment,
> FTZ/DAZ, FP tuning (DG pp. 390–397).

## Key facts

- Objectives: accuracy; reproducibility/portability (across runs, build options, compilers, platforms,
  architectures); performance. Default: performance.
- Default model `fp-model=fast` → lower-accuracy math library functions (host: vectorized calls only;
  target: varies). GPU + `fast`: OpenCL™-conformant, SVML <=4 ULP ≡ `-fimf-precision=medium`.
- Cross-device bitwise identity only for correctly-rounded functions (<= 0.5 ULP).
- FP optimizations (constant evaluation, hoisting loop invariants, reordering) change rounding; not
  strict ANSI/ISO C/C++.

## Option / API quick table

| name | purpose, values, default | notes |
|---|---|---|
| `-fp-model` (Linux*) / `/fp` (Windows*) | FP optimizations + compiler rules; `precise`, `strict`, `fast`; **default `fast`** | `-fp-model precise`, `-fp-model fast`, `-fp-model=fast`, `-fp-model=strict`, `-fp-model:strict`, `/fp:precise`, `/fp:fast`, `/fp:strict` |
| `-fimf-max-error` / `[Q]imf-max-error` | max ULP relative error, all/named math functions; `-fimf-max-error=2`; `-fimf-max-error=10:sin -fimf-max-error=4` | **host only** |
| `-fimf-accuracy-bits` | accuracy bits, all/named functions; `-fimf-accuracy-bits=12:sin` | more precise than low/medium/high |
| `-fimf-precision` / `--fimf-precision` | `=medium` ≡ <=4 ULP SVML | `--fimf-precision` = high-accuracy **host** code |
| `-f[no-]approx-func` | high-accuracy FP options on **device** code | host counterpart `--fimf-precision` |
| `-fmath-errno` | full `errno` support; **disabled by default** | required for full errno; avoid for perf |
| `-ffast-math` (Windows/Linux) | fast math; **disabled by default**; overrides `-fmath-errno` | disables NaN/INF + errno too |
| `[Q]ftz` | flush denormal results to zero; sets FTZ+DAZ flags | negatives `-no-ftz` (Linux*), `/Qftz-` (Windows*); main program only |
| `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` | set FTZ manually | macro, prototype `xmmintrin.h` |
| `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)` | set DAZ manually | macro, prototype `pmmintrin.h` |
| `STDC_FENV_ACCESS` pragma / pragma `fenv_access` | enable FP environment access; off by default | or `-fp-model=strict` / `/fp:strict` |

**See Also (source):** `fimf-max-error`, `Qimf-max-error`; `fp-model`, `fp`; `fmath-errno`; `ftz`, `Qftz`.

**p. 390 `/`-option list fragment:** `/showIncludes`, `/TC`, `/Tc<source file>`, `/TP`,
`/Tp<source file>`, `/u`, `/U<name>`, `/vd<n>`, `/vmg`, `/vmv`, `/W<n>`, `/Wall`, `/WX`, `/X`, `/Y-`,
`/Yc[<file>]`, `/Yu[<file>]`, `/Z7`, `/Zc:<arg1>[, <arg2>]`, `/Zg`, `/Zi`, `/ZI`, `/Zl`, `/Zp[<n>]`,
`/Zs`.

## Programming Tradeoffs in Floating-Point Applications

Accuracy vs reproducibility/portability vs performance — balance per goal. `-fp-model`/`/fp` and
fast-but-low-accuracy `[Q]imf-max-error` (**host only**) also change which math routines run.

## Use Floating-Point Options

`fast` (default, `fp-model=fast`) = lower-accuracy math library functions; `fp-model=precise` =
high-accuracy host scalar and SVML implementations, target devices SYCL*-conforming (between `fast`
and host).

`float t0, t1, t2; ... t0=t1+t2+4.0f+0.1f;` with `-fp-model precise` (Linux) / `/fp:precise`
(Windows) preserves source semantics:
```text
movss     xmm0, _t1
addss     xmm0, _t2
addss     xmm0, DWORD PTR _Cnst4.0
addss     xmm0, DWORD PTR _Cnst0.1
movss     DWORD PTR _t0, xmm0
```
`-fp-model fast` (Linux) / `/fp:fast` (Windows) uses Intel® SSE and pre-computes `4.0f + 0.1f`:
```text
movss    xmm0, DWORD PTR _Cnst4.1
addss    xmm0, DWORD PTR _t1
addss    xmm0, DWORD PTR _t2
movss    DWORD PTR _t0, xmm0
```
Fast form: less accurate (intermediate rounding) and non-reproducible (reorders to pre-compute). With
`-ffast-math` (Windows/Linux), `/fp:fast` (Windows) or `-fp-model=fast` (Linux), **the compiler decides
operation ordering**; it may vary by context/compiler.

## Tune Compilation Accuracy

Dedicated options set math-function accuracy (all or selected functions, finer than low/medium/high)
and expose bundled Intel math library perf/accuracy tradeoffs.
```bash
-fimf-max-error=2                        # 2 ULP relative error for all single, double, long double, quad precision functions
-fimf-accuracy-bits=12:sin               # twelve bits of accuracy for sin
-fimf-max-error=10:sin -fimf-max-error=4 # 10 ULP for sin, 4 ULP for other math functions called in the source file
```
Windows default max-error from `/fp`: sets `max-error=4.0` in `/fp:fast`, else `max-error=0.6`. Host
high accuracy: `--fimf-precision`; device: `-f[no-]approx-func`; Windows accurate SYCL: `/fp:precise`.
**NOTE:** OpenCL™ gives per-operation max ULP variance guidelines (e.g. cosine <= 4 ULP); default fast
FP operations likely exceed them.

## Dispatching of Math Routines

`libm`/`svml` calls become direct CPU-specific calls when the tuning-target CPU is specified **and**
the compilation ISA is not narrower than that CPU's ISA. Cross-device bitwise identity holds only for
correctly-rounded functions (<= 0.5 ULP); high-accuracy options reduce, not eliminate, differences
(errors compound; less accurate implementations amplify them; 4-ULP `cos()` need not agree).

## Floating-Point Optimizations

Constant evaluation, hoisting loop invariants, and reordering can change FP behavior; some conflict
with strict ANSI/ISO C/C++. `-fp-model` (Linux*)/`/fp` (Windows*) rules: **value safety** (SAFE refuses
`x/x`→`1.0`, x may be 0/NaN; default **UNSAFE**); **FP contractions** (on = FMA for multiply+add; off =
separate ops with rounding); **FP environment access** and **precise FP exceptions** (both off by
default; `-fp-model:strict` Linux / `/fp:strict` Windows).

| `-fp-model` keyword | Description |
|---|---|
| `precise` | Enables value-safe optimizations on floating-point data . |
| `strict` | Enables precise, disables contractions, and enables pragma stdc fenv_access. |
| `fast` | Enables more aggressive optimizations on floating-point data. |

| Keyword | Value Safety | Math errno Support | FP Contractions | FP Environment Access | Precise FP Exceptions |
|---|---|---|---|---|---|
| `precise` | Safe | Enabled | Sets `fp-contract=on` | No | No |
| `strict` | Safe | Enabled | No | Yes | Yes |
| `fast=1` (default) | Unsafe | Disabled | Sets `fp-contract=fast` | No | No |
| `fast=2` | Very unsafe | Disabled | Sets `fp-contract=fast` | No | No |

Many `libirc`/`libm`/`libsvml` routines are more Intel-optimized than for non-Intel. `-fmath-errno`
(off by default) is required for full errno support; `-ffast-math` (off by default) overrides it.

## Denormal Numbers

Normalized = non-zero exponent (incl. bias) + non-zero MSB of mantissa. Smallest normalized
single-precision value > 0: about `1.1754943-38` [sic: source garbled exponent]. Smaller values need a
zero exponent and leading mantissa zeros → precision loss: **denormals** (newer specifications:
**subnormal numbers**), which use hardware/OS resources and cost **hundreds of clock cycles**. Avoid:
scale into normalized range; use higher-precision/wider-range type; flush to zero (hardware does so in
most cases).

## Floating-Point Environment

Registers for rounding mode controls, exception masks, flush-to-zero (FTZ) controls, exception status
flags, other FP features; exception mask bits decide which conditions raise exceptions. The OS
generally sets defaults; modify via inline assembly, compiler built-ins, library functions, or
command-line options. **Access off by default**; enable `STDC_FENV_ACCESS` = ON or
`-fp-model=strict`. Changes affect **runtime results only**, never compile-time pre-computed values;
for strict reproducibility also use `-fp-model strict` (Linux*)/`/fp:strict` (Windows*) or pragma
`fenv_access`.

## Set the FTZ and DAZ Flags

Intel® x86: the flush-to-zero (FTZ) and denormals-are-zero (DAZ) flags in **MXCSR** control FP
calculations; Intel® SSE and Intel® AVX instructions (scalar and vector) are accelerated when they are
enabled. `[Q]ftz` flushes denormal results to zero in gradual underflow mode; applied to the main
program it sets the FTZ+DAZ hardware flags, while negatives (`-no-ftz` Linux*, `/Qftz-` Windows*)
leave them as they are.

| Flag | When ON, the compiler... | When OFF, the compiler... |
|---|---|---|
| FTZ | ...sets denormal results from FP calculations to zero. | ...does not change the denormal results. |
| DAZ | ...treats denormal values used as input to FP instructions as zero. | ...does not change the denormal instruction inputs. |

- FTZ applies only to Intel® SSE/AVX instructions (x86-generated denormals unaffected); DAZ+FTZ are
  not ISO/IEC/IEEE 60559-compatible.
- Perf options only: only runtime-generated denormals flush. By default the compiler inserts FTZ/DAZ
  setup code in the main routine; the negative form prevents that.
- Effective only when the **main program** is compiled; sets process FTZ/DAZ mode, inherited by later
  threads. With default `fast`, **every optimization option O level except O0 sets `[Q]ftz`**.

```c
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)         /* FTZ; prototype in xmmintrin.h */
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON) /* DAZ; prototype in pmmintrin.h */
```

## Tuning Performance

Guidelines: Floating-Point Array Operations in a Loop Body; Reduce the Impact of Denormal Exceptions;
Avoid Mixed Data Type Arithmetic Expressions; Use Efficient Data Types.

### Floating-Point Array Operations in a Loop Body

- Supported arithmetic in `float`/`double` loop bodies (typically arrays): addition, subtraction,
  multiplication, division, negation, square root, MAX, MIN, SIN, COS. `precise`/`strict` + math-`errno`
  → less likely to vectorize.
- Mixed single-precision and double writes in one loop → different vector length → less likely to
  auto-vectorize; avoid mixed types if it fails.

**NOTE:** `__m64`/`__m128`/`__m256` are not vectorizable; no function calls in the loop body; Intel®
SSE/AVX intrinsics (e.g. `mm_add_ps`) are not allowed.

### Reduce the Impact of Denormal Exceptions

Denormals need hardware/OS intervention → performance loss. Also: multiply by a large scalar, compute
in normal space, then scale back down. Converting `float`→`double` can slow the program (more
storage/load-store time; lower Intel® SSE/AVX throughput) and may require changing non-generic library
calls (e.g. `cos()` vs `cosf()`). Denormals are often safely treatable as zero; use FTZ per target.

### Avoid Mixed Data Type Arithmetic Expressions

Avoid mixing integer and floating-point (`float`, `double`, `long double`) data: all-floating-point
(or all-integer) expressions eliminate fixed/floating-point conversions. With `I`, `J` both `int`,
writing `2.0` as `2` avoids the conversion.
```c
int I, J;      /* Inefficient: */
  I = J / 2.0;
```
```c
int I, J;      /* Efficient: */
  I = J / 2;
```

### Use Efficient Data Types

Most to least efficient: `char`, `short`, `int`, `long`, `long long`, `float`, `double`,
`long double`. **NOTE:** avoid mixing integer and floating-point data in an arithmetic expression.
Integer types (`int`, `int long`, etc.) in loops can improve floating point performance: convert to
integer, process, convert back.

## Gotchas & failure modes

- `-ffast-math` (default off) overrides `-fmath-errno`; disables NaN/INF and errno support.
- `-fp-model=fast` (default): non-reproducible ordering; value-safety default **UNSAFE**;
  `precise`→`fp-contract=on`, `fast=1`/`fast=2`→`fp-contract=fast`, `strict`→none.
- FP environment access / precise exceptions need `-fp-model=strict`/`/fp:strict` (or
  `STDC_FENV_ACCESS` ON / pragma `fenv_access`); environment changes never affect compile-time values.
- `[Q]ftz` is process-wide from the **main program** (runtime denormals only); default `fast` already
  sets it at every O level except O0. `[Q]imf-max-error` is **host only**.

## Source map

- pp. 390–393 — `/`-option list fragment; intro; Programming Tradeoffs; Use Floating-Point Options; Tune Compilation Accuracy; Dispatching; Floating-Point Optimizations
- pp. 394–397 — Denormal Numbers; Floating-Point Environment; Set the FTZ and DAZ Flags; Tuning Performance
