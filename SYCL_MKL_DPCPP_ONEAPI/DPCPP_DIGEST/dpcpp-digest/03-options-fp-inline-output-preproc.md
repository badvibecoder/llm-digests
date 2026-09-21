---
chunk: 03-options-fp-inline-output-preproc
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 228-294
covers: Floating-Point Options (ffp-*, fimf-*, fma, fp-model, fp-speculation, ftz, pc), Inlining Options, Output/Debug/Precompiled-Header Options, Preprocessor Options
---

# Floating-Point, Inlining, Output/Debug/PCH, and Preprocessor Options

> **Scope.** Reference for four option families of the Intel® oneAPI DPC++/C++ Compiler:
> floating-point math/accuracy, inlining, output/debug/precompiled-header, and preprocessor options.
> Gives exact Linux (`-…`) / Windows (`/…`) spellings, legal values, and defaults, so a reader can
> answer "what is the `-fp-model` default?", "which options conflict?", "which are host-only?"
> without guessing.

## Key facts

- **Host-only** (source note: *"only applies to host compilation. When offloading is enabled, it
  does not impact device-specific compilation"*): `fimf-absolute-error`, `fimf-accuracy-bits`,
  `fimf-arch-consistency`, `fimf-domain-exclusion`, `fimf-max-error`, `fimf-precision`,
  `fimf-use-svml`, `Fa`, `fasm-blocks`, `Fe`, `Fo`, `Fp`, `fverbose-asm`, `gsplit-dwarf`, `Y-`,
  `Yc`, `Yu`, `TP`.
- `-ffp-accuracy` is **incompatible** with `fimf-*`/`Qimf-*` (error reported); CPU Ahead of Time
  (AOT) SYCL/OpenMP only.
- Maximum relative error is set by `-fimf-precision`, `-fimf-max-error`, or `-fimf-accuracy-bits`
  -- **last on the command line wins**; if none, by `-fp-model` (Linux) / `/fp` (Windows).
- Defaults: `-fp-model=fast` (`/fp:fast`), `-ffp-contract=fast`, `-fma`/`/Qfma`,
  `-fp-speculation=fast` (to `safe` at `-O0`), `-ftz`/`/Qftz`, `-fimf-precision=medium`,
  `-fno-inline`, `finline-functions` ON except at `-O0`, `-debug none` / `/debug:none`.
- Platform splits: `-pc80` (Linux) vs `/Qpc64` (Windows); `-g` implies `-O0` and
  `-fno-omit-frame-pointer` unless `-O2`+ is explicit; `-S` turns on `-fverbose-asm`.
- **Deprecated:** `/debug:partial` (absent from the IDE); `/TP` (may be removed).
- Pages 228-294 contain no catalogued figure: no figure section.

## Option quick table

Every option in this chunk, long tail included.

| name | values / default |
|---|---|
| `-ffp-accuracy` / `/Qfp-accuracy` | `high`=1 ulp, `medium`=4 ulp, `low`=11-bit single (~8192 ulp)/26-bit double, `sycl`(OpenCL spec), `cuda`(CUDA std); default OFF (host: `-fp-model`/`/fp`; device: device-dependent) |
| `-ffp-contract` | `fast`/`on`/`off`; default `fast`, `off` under `-fp-model=strict`. Linux only |
| `-fimf-absolute-error` / `/Qimf-absolute-error` | positive FP number `[:funclist]`; default `0`. Host-only |
| `-fimf-accuracy-bits` / `/Qimf-accuracy-bits` | positive FP number `[:funclist]`; default `-fimf-precision=medium`; ulps = 2^(p-1-bits). Host-only |
| `-fimf-arch-consistency` / `/Qimf-arch-consistency` | `true`/`false` `[:funclist]`; default `false`; one binary only. Host-only |
| `-fimf-domain-exclusion` / `/Qimf-domain-exclusion` | `classlist[:funclist]`; default `0`; extremes 1, nans 2, infinities 4, denormals 8, zeros 16; none 0, all 31, common 15. Host-only |
| `-fimf-max-error` / `/Qimf-max-error` | positive FP number `[:funclist]`; default `-fimf-precision=medium`. Host-only |
| `-fimf-precision` / `/Qimf-precision` | `high`(=max-error 1.0), `medium`(=max-error 4), `low`(accuracy-bits 11 single/26 double); default `medium`; `value` optional. Host-only |
| `-fimf-use-svml` / `/Qimf-use-svml` | `true`/`false` `[:funclist]`; default `false`. Host-only |
| `-fma` / `-no-fma`, `/Qfma` / `/Qfma-` | default `-fma`/`/Qfma`; negative form under `fp-model strict`. No effect below CORE-AVX2 |
| `-fp-model` / `/fp` | `precise`, `fast[=1\|2]`, `consistent`, `strict`; default `fast`; `fast=1` (default) supports NaN/inf, `fast=2` does not |
| `-fp-speculation` / `/Qfp-speculation` | `fast`/`safe`/`strict`; default `fast` (also with optimizations), `safe` at `-O0` |
| `-ftz` / `-no-ftz`, `/Qftz` / `/Qftz-` | default `-ftz`/`/Qftz`; every `O` level except `O0` sets it; sets FTZ/DAZ process-wide |
| `-pcn` / `/Qpcn` | `32`=24 bits, `64`=53 bits, `80`=64 bits; default `-pc80` Linux, `/Qpc64` Windows |
| `-fgnu89-inline` | default OFF |
| `-finline` / `-fno-inline` | default `-fno-inline` |
| `-finline-functions` / `-fno-inline-functions` | default ON; OFF at `-O0` |
| `-inline-forceinline` / `/Qinline-forceinline` | default OFF |
| `-c` / `/c` | default OFF |
| `-debug[=keyword]` (Linux*) | `none`, `full`\|`all`, `minimal`, `[no]emit-column`, `extended`, `[no]parallel`; default `-debug none` |
| `/debug[:keyword]` (Windows*) | `none`, `full`\|`all`, `minimal`, `partial`, `[no]expr-source-pos`, `[no]inline-debug-info`; default `/debug:none` (cmd line/release IDE), `/debug:all` (debug IDE) |
| `/Fa[filename\|dir]` | default OFF; Linux None; alt `-S`/`/S`. Host-only |
| `-fasm-blocks` | default OFF (GNU*-style allowed); alt `-use-msasm`. Host-only |
| `/Fe[[:]filename\|dir]` | default first source name + `.exe`; alt `-o`. Host-only |
| `/Fo[[:]filename\|dir]` | default first source name + `.obj`; Linux `o`. Host-only |
| `/Fp{filename\|dir}` | default OFF; Linux None. Host-only |
| `-fsystem-debug` / `-fno-system-debug` | default `-fsystem-debug`; same spelling both OS |
| `-fverbose-asm` / `-fno-verbose-asm` | default `-fno-verbose-asm`; `-S` sets it. Host-only |
| `-g[n]` | `0`/`1`/`2`/`3`; default `-g`/`-g2`; DWARF Version 4; turns off `-O2` |
| `-gdwarf-n` | `2`/`3`/`4`/`5`; default OFF, DWARF 4 if `-g` |
| `-grecord-gcc-switches` | default OFF; appends to `DW_AT_producer` |
| `-gsplit-dwarf` | default OFF; binutils-2.24+, gdb-7.6.1+. Host-only |
| `-o filename` | default OFF; meaning depends on `-c`/`-S`/`-P`; Windows `/Fe` |
| `-S` / `/S` | default OFF; `.s` Linux, `.asm` Windows; Windows alt `/Fa` |
| `-use-msasm` | default OFF; alt `-fasm-blocks` |
| `/Y-` | default OFF. Host-only |
| `/Yc[filename]` | default OFF; name = source + `.pch`; single source file; not with `/Yu`. Host-only |
| `/Yu[filename]` | default OFF; name = source + `.pch`; not with `/Yc`. Host-only |
| `/Zi`, `/Z7`, `/ZI` | default OFF; `/ZI` = synonym for `/Zi`; Linux `-g` |
| `-Bdir` | default OFF (`PATH`); becomes `-I/dir/include`, `-L/dir` |
| `-C` / `/C` | default OFF; comments after directives not preserved |
| `-Dname[=value]` / `/Dname[=value]` | default OFF; no value ⇒ `1`; `#define` |
| `-dD` / `/QdD` | default OFF; requires `E` |
| `-dM` / `/QdM` | default OFF; requires `E` |
| `-E` / `/E` | default OFF; output has `#line` directives |
| `-EP` / `/EP` | default OFF; with `P`, writes a file |
| `/FIfilename` | default OFF; Linux None |
| `-H` / `/QH` | default OFF |
| `-Idir` / `/Idir` | default OFF; repeat per directory |
| `-idirafterdir` | default OFF; after `-I` |
| `-imacros filename` | default OFF |
| `-iprefix prefix` | default OFF; used with `-iwithprefix` |
| `-iquote dir` | default OFF |
| `-isystemdir` | default OFF; after `-I`, before standard dirs |
| `-iwithprefixdir` | default OFF |
| `-iwithprefixbeforedir` | default OFF |
| `-M` / `/QM` | default OFF |
| `-MD` / `/QMD` | default OFF; preprocess + compile |
| `-MFfilename` / `/QMFfilename` | default OFF; requires `/QM`/`/QMM` |
| `-MG` / `/QMG` | default OFF; like `/QM` |
| `-MM` / `/QMM` | default OFF; like `/QM` |
| `-MMD` / `/QMMD` | default OFF; like `/QMD` |
| `-MQtarget` / `/QMQtarget` | default OFF; like `-MT`, quotes Make chars |
| `-MTtarget` / `/QMTtarget` | default OFF |
| `-nostdinc++` | default OFF |
| `-P` / `/P` | default OFF; Linux `.i` file, no `#line` |
| `/TP` | default OFF; **deprecated**; alt `-x c++`/`/Tp`. Host-only |
| `-Uname` / `/Uname` | default OFF; `#undef`; Windows `/u` undefines all |
| `-undef` | default OFF |
| `-X` / `/X` | default OFF; Linux `-noinclude` alias |

## Floating-Point Options

Alphabetical; Linux `-...` / Windows `/Q...` unless Windows is `None`. No IDE equivalents. Shared
`fimf-*`/`Qimf-*` conventions: `funclist` = optional comma-separated math functions; precision
variants (`sin` vs `sinf`) are *different* functions and both must be named. Divide symbols `/f`
single-, `/` double-, `/l` extended-, `/q` quad-precision. Numeric format
`[digits] [.digits] [ { e | E }[sign]digits]`. No names means all functions/precisions; one name
means only that precision (`sinf`/`sin`/`sinl`). LIBM and SVML routines are more highly optimized
for Intel(R) microprocessors than for non-Intel microprocessors.

### ffp-accuracy, Qfp-accuracy

`-ffp-accuracy=value` / `/Qfp-accuracy:value` -- required accuracy for FP operations and library
calls (host code and, if used, OpenMP/SYCL device code). Default OFF: host accuracy from
`-fp-model` (Linux*) / `/fp` (Windows*); offload devices device-dependent. **Incompatible with
`fimf-*`/`Qimf-*`** (error reported). CPU AOT SYCL/OpenMP only.

### ffp-contract

`-ffp-contract=keyword`; Windows `None` -- when fused FP operations such as FMA may be formed
(fused operations may be more precise than separate ones). `fast` across statements, `on` within
one statement, `off` never; default `fast`, but `off` if `-fp-model=strict`.

### fimf-absolute-error, Qimf-absolute-error

`-fimf-absolute-error=value[:funclist]` / `/Qimf-absolute-error:value[:funclist]` -- max absolute
error of math results (`value` = positive FP number). Errors may exceed the relative-error
(`max-error`) setting if `absolute-error` is <= `value`; default `0` (function bound by the
relative-error setting). Only functions able to return zero (`log`, `sin`, `asin`); a result must
have relative error < `max-error` or absolute error < `absolute-error`. Faster, may lose accuracy.
E.g. `-fimf-absolute-error=0.00001:sin,sinf`, `-fimf-absolute-error=0.00001:/`,
`-fimf-absolute-error=0.00001:sqrtf` (Windows `/Qimf-absolute-error:...`).

### fimf-accuracy-bits, Qimf-accuracy-bits

`-fimf-accuracy-bits=bits[:funclist]` / `/Qimf-accuracy-bits:bits[:funclist]` -- relative error of
math results including division and square root; `bits` = positive FP number of correct bits.
Default `-fimf-precision=medium` / `/Qimf-precision:medium`. `ulps = 2p-1-bits`, `p` =
target-format mantissa bits (24 single, 53 double, 64 long double). E.g.
`-fimf-accuracy-bits=10.0:/f`; more below.

### fimf-arch-consistency, Qimf-arch-consistency

`-fimf-arch-consistency=value[:funclist]` / `/Qimf-arch-consistency:value[:funclist]` -- consistent
math results across microarchitectural implementations of the same architecture (e.g. Intel(R) 64);
`value` = `true`/`false`, default `false` (some functions differ slightly between implementations
of the same architecture). Guaranteed for one binary only, not across architectures. May cost
performance but is bit-wise consistent on all Intel(R) and compatible non-Intel processors
regardless of micro-architecture; may not be between architectures. E.g.
`-fimf-arch-consistency=true:sin`, `-fimf-arch-consistency=false:sqrtf`.

### fimf-domain-exclusion, Qimf-domain-exclusion

`-fimf-domain-exclusion=classlist[:funclist]` / `/Qimf-domain-exclusion:classlist[:funclist]` --
input-arguments domain on which math functions must give correct results (the program is correct if
the listed functions are not standard conforming on the excluded classes). Each classlist element is
a power of two and the attribute is their logical OR (bitmask); specify the integer value. Excluded
values may give unexpected results.

| class | meaning | value |
|---|---|---|
| `extremes` | values not within the usual domain of arguments for a given function | 1 |
| `nans` | "x=Nan" | 2 |
| `infinities` | "x=infinities" | 4 |
| `denormals` | "x=denormal" | 8 |
| `zeros` | "x=0" | 16 |
| `none` | none excluded; specify `0` (`-fimf-domain-exclusion=0` / `/Qimf-domain-exclusion:0`) | 0 |
| `all` | all excluded; specify `31` (`-fimf-domain-exclusion=31` / `/Qimf-domain-exclusion:31`) | 31 |
| `common` | = `extremes,nans,infinities,denormals`; specify `15` (1+2+4+8), as in `-fimf-domain-exclusion=15` / `/Qimf-domain-exclusion:15` | 15 |
| other combinations | bitwise OR of the used values | -- |

Default `0` (default heuristics). More exclusions means faster code sequences (affects performance
and accuracy). `funclist` forms `-fimf-domain-exclusion=4:sin,sinf`, `=4`, `=5:/,powf`,
`=23:log,logf,/,sin,cosf` (Windows `/Qimf-domain-exclusion:...`); with no `funclist`, all functions
are restricted.

**Worked `exp2f` example (single precision).** `y = exp2f(x)`, accuracy 1.014 ulp, instructions 4
(2 without fix-up). The 2-instruction sequence, without fix-up (it mishandles NaNs):

```text
vcvtfxpntps2dq    zmm1 {k1}, zmm0, 0x50         // zmm1 <-- rndToInt(2^24 * x)
vexp223ps         zmm1 {k1}, zmm1               // zmm1 <-- exp2(x)
```

Required fix-up for correct NaN handling:

```text
vpxord            zmm2, zmm2, zmm2             // zmm2 <-- 0
vfixupnanps       zmm1 {k1}, zmm0, zmm2 {aaaa} // zmm1 <-- QNaN(x) if x is NaN <F>
```

Options generating the **2-instruction** sequence:

```text
-fimf-domain-exclusion=2:exp2f      <- NaNs excluded (2 = NaNs)
-fimf-domain-exclusion=6:exp2f      <- +infinities (4; 2+4=6)
-fimf-domain-exclusion=7:exp2f      <- +extremes (1; 2+4+1=7)
-fimf-domain-exclusion=15:exp2f     <- +denormals (8; 2+4+1+8=15)
```

With `vfixupnanps` included, all arguments including NaN are handled; options generating the
**4-instruction** sequence:

```text
-fimf-domain-exclusion=1:exp2f      <- only extremes (1)
-fimf-domain-exclusion=4:exp2f      <- only infinities (4)
-fimf-domain-exclusion=8:exp2f      <- only denormals (8)
-fimf-domain-exclusion=13:exp2f     <- extremes+infinities+denormals (1+4+8=13)
```

### fimf-max-error, Qimf-max-error

`-fimf-max-error=ulps[:funclist]` / `/Qimf-max-error:ulps[:funclist]` -- max relative error of math
results including division and square root (`ulps` = positive FP number). Default
`-fimf-precision=medium` / `/Qimf-precision:medium`. Affects performance and accuracy. E.g.
`-fimf-max-error=4.0:sin,sinf`, `-fimf-max-error=4.0:/`, `-fimf-max-error=4.0:sin`,
`-fimf-max-error=4.0:sqrtf`.

### fimf-precision, Qimf-precision

`-fimf-precision[=value[:funclist]]` / `/Qimf-precision[:value[:funclist]]` -- accuracy level for
choosing math functions (`value` optional). `high` = `max-error = 1.0`; `medium` = `max-error = 4`,
the default when the option is given without `value`; `low` = `accuracy-bits = 11` for
single-precision and `26` for double-precision functions (`max-error` = `-fimf-max-error`,
`accuracy-bits` = `-fimf-accuracy-bits`; Windows `/Qimf-max-error`, `/Qimf-accuracy-bits`). Default
`medium`; lower precision can improve performance, higher precision may reduce it. E.g.
`-fimf-precision=high:sin,sinf`, `-fimf-precision=low:/`, `-fimf-precision=low:/f`,
`-fimf-precision=low:sin`, `-fimf-precision=high:sqrtf`.

### fimf-use-svml, Qimf-use-svml

`-fimf-use-svml=value[:funclist]` / `/Qimf-use-svml:value[:funclist]` -- use SVML instead of the
Intel(R) oneAPI DPC++/C++ Compiler Math Library (LIBM); default `false` (other options may still let
the compiler choose LIBM or SVML). With `=true` the SVML variant is influenced by
`-fimf-precision` and `-fp-model` (Windows `/Qimf-precision`, `/fp`); no effect on functions in
LIBM but not SVML. E.g. `-fimf-use-svml=true:sin,sinf` -- printed in the source as
`-fimf-use-svmlt=true:sin,sinf` [sic: source garbled] -- or `/Qimf-use-svml:true:sin,sinf`;
`-fimf-use-svml=false:sqrtf` / `/Qimf-use-svml:false:sqrtf`. **Caveats:** with value-safe
`-fp-model`/`/fp` settings such as `precise` it slightly decreases accuracy (even high-accuracy SVML
functions are slightly less accurate than the LIBM ones); SVML functions might not accurately raise
FP exceptions, do not maintain `errno`, and work correctly only in round-to-nearest-even rounding
mode. It can significantly improve performance by enabling efficient vectorization of loops
containing math library calls. Be cautious about FP-exception trapping, since SVML functions may
raise unexpected exceptions.

### fma, Qfma

`-fma` / `-no-fma`; `/Qfma` / `/Qfma-` -- generate FMA instructions if they exist on the target
processor. Default `-fma`/`/Qfma`; if `-fp-model strict` (Linux*) or `/fp:strict` (Windows*) is
given without an explicit `-fma`/`/Qfma`, the default is `-no-fma`/`/Qfma-`. `[Q]fma` lets the
compiler combine multiply and add; the negative form requires separate multiply and add
instructions with intermediate rounding. No effect unless `CORE-AVX2` or higher is set via `[Q]x`,
`-march` (Linux), or `/arch` (Windows).

### fp-model, fp

`-fp-model=keyword` / `/fp:keyword` -- controls the semantics of floating-point calculations;
default `-fp-model=fast` / `/fp:fast`.

| option | Description |
|---|---|
| `-fp-model=precise` / `/fp:precise` | Value-safe optimizations only; disables optimizations that can change the result, required for strict ANSI conformance. Reproducible for serial code, including compiler-vectorized/auto-parallelized code, but may slow performance; not value safety or run-to-run reproducibility for other parallel code. FP reductions in OpenMP* code can be made run-to-run reproducible for a fixed number of threads via `KMP_DETERMINISTIC_REDUCTION`. Assumes the default FP environment; you may not modify it. |
| `-fp-model=fast[=1\|2]` / `/fp:fast[=1\|2]` | More aggressive optimizations: faster, but may affect accuracy or reproducibility. `fast` = `fast=1`. `fast=1` (default) recognizes and supports NaN and infinite values; `fast=2` does not -- none will be used or produced. |
| `-fp-model=consistent` / `/fp:consistent` | Disables non-value-safe optimizations on FP data, disables contraction (FMA), and selects math functions that produce consistent results across different microarchitectural implementations of the same architecture. |
| `-fp-model=strict` / `/fp:strict` | Enables `precise`, disables contractions, and enables pragma `stdc fenv_access`; the strictest model. Does not assume the default FP environment; you may modify it. |

The FP environment is a collection of registers controlling FP machine instructions and indicating
current FP status; it can include rounding-mode controls, exception masks, flush-to-zero controls,
exception status flags, and other FP features. These options set `max-error` when none of
`-fimf-accuracy-bits`, `-fimf-max-error`, `-fimf-precision` (Windows `/Qimf-*`) is specified:
`-fp-model=fast` (`/fp:fast`) sets `-fimf-precision=medium` (`/Qimf-precision:medium`);
`-fp-model=precise` (`/fp:precise`) implies `-fimf-precision=high` (`/Qimf-precision:high`) [sic:
source runs the two clauses together]. **NOTE (Microsoft* Visual Studio):** a new Microsoft* Visual
C++ project sets `/fp:precise` by default, improving consistency by disabling optimizations that
may reduce performance; restore `/fp:fast` via the IDE project property Floating Point Model =
`Fast`.

### fp-speculation, Qfp-speculation

`-fp-speculation=mode` / `/Qfp-speculation:mode` -- `fast` speculates; `safe` disables speculation
if it may cause a floating-point exception; `strict` disables speculation on FP operations. Default
`fast` (also with optimizations enabled); with `-O0` it becomes `safe` (Linux) /
`/Qfp-speculation:safe` (Windows). Disabling speculation may prevent vectorization of some loops
containing conditionals.

### ftz, Qftz

`-ftz` / `-no-ftz`; `/Qftz` / `/Qftz-` -- flush denormal results to zero. Default `-ftz`/`/Qftz`;
every optimization `O` level except `O0` sets `[Q]ftz`. Flushes denormals in gradual-underflow mode
(no effect during compile-time optimization); may help performance if denormal values are not
critical. Sets/resets the FTZ and DAZ flags: FTZ ON makes denormal results zero, FTZ OFF leaves
them; DAZ ON treats denormal inputs as zero, DAZ OFF leaves them. Intel(R) 64 systems have both.
`-no-ftz` / `/Qftz-` prevents the compiler inserting code that might set FTZ/DAZ. Effective only
when the main program is compiled: it sets FTZ/DAZ for the process, so the initial thread and
threads it later creates operate in FTZ/DAZ mode; specify `-no-ftz`/`/Qftz-` to restore numerical
behavior while keeping `O3`. **NOTE:** a performance option -- it does not guarantee all denormals
in a program are flushed, only runtime-generated ones.

### pc, Qpc

`-pcn` / `/Qpcn` -- floating-point significand precision: `32` = 24 bits (single precision), `64` =
53 bits (double precision), `80` = 64 bits (extended precision). Default `-pc80` (Linux*: 64 bits) /
`/Qpc64` (Windows*: 53 bits). Iterative operations like division and square root can run faster at
lower precision. A change of the default precision control or rounding mode (e.g. `[Q]pc32`, or user
intervention) may affect results returned by some mathematical functions.

## Inlining Options

Alphabetical; no IDE equivalents.

- **`-fgnu89-inline`** (Linux; Windows `None`) -- C89 semantics for inline functions in C99 mode.
  Default OFF.
- **`-finline` / `-fno-inline`** (Linux; Windows `None`) -- inline `__inline` functions and perform
  C++ inlining. Default `-fno-inline`.
- **`-finline-functions` / `-fno-inline-functions`** (Linux; Windows `None`) -- function inlining
  for single file compilation (inline expansion for calls to functions defined in the current
  source file, by compiler heuristic). Default `-finline-functions` (interprocedural optimizations
  occur); OFF if `-O0`.
- **`-inline-forceinline` / `/Qinline-forceinline`** -- treat inline routines as `forceinline`.
  Default OFF (default inline-expansion heuristics). **Caution:** makes the compiler inline
  aggressively, so it may run out of memory and terminate with an "out of memory" message.

## Output, Debug, and Precompiled Header Options

Alphabetical; IDE equivalents omitted. "Host-only" per Key facts.

- **`-c` / `/c`** -- object only, no link (compilation stops after the object file). Default OFF.
  Emits an object file per C/C++/preprocessed source file; also assembles an assembler file.
- **`/Fa[filename|dir]`** (Windows; Linux `None`) -- assembly listing file; `dir` may include
  `filename`. Default OFF. Alt `-S`/`/S`. Host-only.
- **`-fasm-blocks`** (Linux; Windows `None`) -- allow Microsoft* MASM-style inline assembly blocks
  and entire functions, not GNU*-style. Default OFF (GNU*-style allowed). Alt `-use-msasm`.
  Host-only.
- **`-use-msasm`** (Linux; Windows `None`) -- same as `-fasm-blocks`. Default OFF. Alt
  `-fasm-blocks`.
- **`/Fe[[:]filename|dir]`** (Windows; Linux `None`) -- name of the built program or dynamic-link
  library. Default: first source file name + `.exe` (`file.f` -> `file.exe`). Alt `-o`. Host-only.
  Example below.
- **`/Fo[[:]filename|dir]`** (Windows; Linux: see `o`) -- object file name. Default: first source
  file name + `.obj`. Host-only.
- **`/Fp{filename|dir}`** (Windows; Linux `None`) -- alternate PCH path/file name. Default OFF (no
  PCH created or used unless told to). Host-only.
- **`-fsystem-debug` / `-fno-system-debug`** (same spelling on both OS) -- debug information for
  system-header declarations. Default `-fsystem-debug`; `-fno-system-debug` shrinks debug info from
  Linux `-g` or an MSVC debug option such as `/Z7`. Examples below.
- **`-fverbose-asm` / `-fno-verbose-asm`** (Linux; Windows `None`) -- assembly listing with compiler
  comments (options and version information). Default `-fno-verbose-asm`; `-S` sets
  `-fverbose-asm`, and `-fno-verbose-asm` suppresses it. Host-only.
- **`-g[n]`** (Linux; Windows: see `Zi`, `Z7`, `ZI`) -- debug level `0` none, `1` minimal for stack
  traces, `2` complete (same as `-g`), `3` extra useful for some tools. Default `-g`/`-g2`. Grows
  the object file; never generated in assemblable files. Turns off `-O2` and defaults to `-O0`
  unless `-O2`+ is explicit in the same command line; `-g` or `-O0` sets
  `-fno-omit-frame-pointer`. For C++ on Linux, `-debug inline-debug-info` is enabled by default
  with `-O2`+ and `-g`. `-g` uses DWARF Version 4; older analysis tools may need `-gdwarf-2`. Alt:
  Windows `/Zi`, `/Z7`, `/ZI`.
- **`-gdwarf-n`** (Linux; Windows `None`) -- DWARF Version format; `n` = `2`/`3`/`4`/`5` gives
  DWARF Version 2/3/4/5. Default OFF; DWARF Version 4 with `-g`. Older tools may need
  `-gdwarf-2`.
- **`-grecord-gcc-switches`** (Linux; Windows `None`) -- append the command line options used to
  invoke the compiler to the `DW_AT_producer` attribute in DWARF debugging information,
  whitespace-separated from each other and from the compiler version. Default OFF.
- **`-gsplit-dwarf`** (both OS) -- split DWARF debug information between the generated object
  (`.o`) and a new DWARF object (`.dwo`). Default OFF. The `.dwo` is not used by the linker, so the
  linker processes less debug information and the executable is smaller. Requires binutils-2.24 or
  higher; debug the result with gdb-7.6.1 or higher. Tools without split-DWARF support behave as
  though the debug information is absent. Host-only.
- **`-o filename`** (Linux; Windows: see `Fo`) -- output file name (space before `filename`
  optional). Default OFF. With `-c`/`-S`/`-P` it names the object file / assembly listing /
  preprocessor file; otherwise the executable. Alt: Windows `/Fe`.
- **`-S` / `/S`** -- assembly file only, no link. Default OFF. Linux* suffix `.s`, Windows* `.asm`.
  Alt: Windows `/Fa`.
- **`/Y-`** (Windows; Linux `None`) -- ignore all other precompiled header files. Default OFF (the
  compiler recognizes PCH files when certain options are specified). Host-only.
- **`/Yc[filename]`** (Windows; Linux `None`) -- create a precompiled header (PCH); `filename` = a
  C/C++ header included via `#include`. Default OFF. Single source file only. With `filename`,
  creates the PCH from headers up to and including it; without it, compiles all code to the end of
  the source file or to a `hdrstop`. `/Fp` names the PCH, else the header name, else the source file
  name, with extension `.pch`. Cannot be combined with `/Yu`. Host-only.
- **`/Yu[filename]`** (Windows; Linux `None`) -- use a precompiled header (PCH); `filename` = a
  C/C++ header included via `#include`. Default OFF. Multiple source files allowed when all use the
  same `.pch`. Code before the header is treated as precompiled: the compiler skips to just beyond
  the associated `#include`, uses the PCH code, then compiles all code after `filename`; all text
  including declarations before that `#include` is ignored. Without `filename` the PCH name derives
  from the source file name; `/Fp` selects the PCH instead. Cannot be combined with `/Yc`. Naming
  examples below. Host-only.
- **`/Zi`, `/Z7`, `/ZI`** (Windows; Linux: see `g`) -- full debug information in an object (`.obj`)
  file or a project database (PDB). Default OFF. `/Z7` puts symbolic debugging information in the
  `.obj` for the debugger (no `.pdb`). `/ZI` is a synonym for `/Zi`. `/Zi` uses a PDB, with type
  information in the `.pdb` rather than the `.obj`, so object files are smaller than with `/Z7`.
  `/Zi` makes two PDBs: the compiler's `project.pdb`, or `vcx0.pdb` when compiling a file without a
  project (`x` = major Visual C++ version, e.g. `vc140.pdb`), holding all per-object debugging
  information in the project makefile directory and renameable with `/Fd`; and the linker's
  `executablename.pdb` for the `.exe`, in the `debug` subdirectory, with full debug information
  including function prototypes. Both allow incremental updates; the linker embeds the `.pdb` path
  in the `.exe`/`.dll`. Never in assemblable files. Turns off `/O2` and defaults to `/Od` unless
  `/O2`+ is explicit. Alt: Linux `-g`.

### debug (Linux*)

`-debug[=keyword]` (Windows `None`) -- enable/disable debugging information. Keywords: `none`
(disable); `full`/`all` (complete; same as `-debug` alone); `minimal` (line number information);
`[no]emit-column` (column number information); `extended` (complete information plus keyword values
`semantic-stepping` and `variable-locations`, and column numbers -- more powerful than `full`/`all`);
`[no]parallel` (parallel debug code instrumentation for thread data sharing and reentrant call
detection; Linux only, and `-qopenmp` must be set for shared data and reentrancy detection). Default
`-debug none`. Enabling debugging disables optimization by default; use `-debug` with an
optimization level option for both (source reads "`-O3`, `-O2` or `-O3`" [sic]). Keywords
`inline-debug-info`, `variable-locations`, `extended` can be combined; conflicting keywords: last on
the command line wins. On Linux* systems debuggers read debug information from executable images,
so it is written to object files and added by the linker. Alt: for `-debug full`/`all`/`-debug` --
Linux `-g`; Windows `/debug:full`, `/debug:all`, `/debug`.

### debug (Windows*)

`/debug[:keyword]` (Linux `None`) -- enable/disable debugging information; passed to the linker.
Keywords: `none` (disable); `full`/`all` (complete -- symbol table information for full symbolic
debugging of unoptimized code plus global symbol information for linking; same as `/debug` alone);
`minimal` (line number information); `partial` (**deprecated**, not in the IDE -- global symbol
table information for linking, but not local symbol table information for debugging);
`[no]expr-source-pos` (source position information at expression-level granularity);
`[no]inline-debug-info` (enhanced debug information for inlined code; by default inlined symbols are
associated with the caller, this associates them with the source of the called function). Default
`/debug:none` on the command line and for a release IDE configuration; `/debug:all` for a debug IDE
configuration. Enabling debugging disables optimization by default; use `/debug` with an
optimization level option for both (`/O3`, `/O2` or `/O3` [sic]). Conflicting keywords: last on the
command line wins. Alt: for `/debug:all` or `/debug` -- Windows `/Zi`.

## Preprocessor Options

Alphabetical; IDE equivalents omitted.

- **`-Bdir`** (Linux; Windows `None`) -- directory used to find include files, libraries, and
  executables (a separator is appended to `dir` if needed). Default OFF (uses `PATH`). As a prefix:
  includes become `-I/dir/include`, added to the front of preprocessor includes; libraries become
  `-L/dir`, added before the standard `-L` inclusions and system libraries; if `dir` contains a tool
  name such as `ld` or `as`, that tool replaces the default. Includes are looked for in
  `dir /include`, libraries in `dir`. On Linux*, `GCC_EXEC_PREFIX` gives the same behavior.
- **`-C` / `/C`** -- keep comments in preprocessed source output. Default OFF. Comments following
  preprocessing directives are not preserved. Example below.
- **`-Dname[=value]` / `/Dname[=value]`** -- define a macro with an optional integer or
  double-quoted string value (e.g. `Dname=string`); equivalent to `#define`. With no value, `name`
  is defined as "1". Default OFF (only default symbols or macros defined). Example below.
- **`-dD` / `/QdD`** -- like `-dM` but outputs `#define` directives in preprocessed source. Default
  OFF. Requires `E`.
- **`-dM` / `/QdM`** -- output macro definitions in effect after preprocessing. Default OFF.
  Requires `E`.
- **`-E` / `/E`** -- preprocessor output to stdout; compilation stops when files are preprocessed.
  Default OFF (preprocessed source files go to the compiler). The output contains `#line`
  directives used by the compiler to determine source file and line number. Example below.
- **`-EP` / `/EP`** -- preprocessor output to stdout, omitting `#line` directives. Default OFF.
  With option `P`, results go to a file instead of stdout. Example below.
- **`/FIfilename`** (Windows; Linux `None`) -- include the named file as the header file, before
  the first line of the primary source file. Default OFF (default header files used).
- **`-H` / `/QH`** -- display the include file order and continue compilation. Default OFF.
- **`-Idir` / `/Idir`** -- additional include search directory. Default OFF. Repeat per directory.
- **`-idirafterdir`** (Linux; Windows `None`) -- add `dir` to the second include search path (after
  `-I`). Default OFF.
- **`-imacros filename`** (Linux; Windows `None`) -- a header included in front of the other headers
  in the translation unit. Default OFF.
- **`-iprefix prefix`** (Linux; Windows `None`) -- prefix for referencing directories containing
  header files; used with `-iwithprefix`. Default OFF.
- **`-iquote dir`** (Linux; Windows `None`) -- add `dir` to the front of the include search path for
  files included with quotes but not brackets. Default OFF.
- **`-isystemdir`** (Linux; Windows `None`) -- add `dir` to the system include path; searched after
  all `-I` directories but before the standard system directories. Default OFF. Linux-only, for gcc
  compatibility.
- **`-iwithprefixdir`** (Linux; Windows `None`) -- append `dir` to the `-iprefix` prefix and put it
  at the end of the include directories. Default OFF.
- **`-iwithprefixbeforedir`** (Linux; Windows `None`) -- like `-iwithprefix`, but placed with `-I`
  command-line include directories. Default OFF.
- **`-nostdinc++`** (Linux; Windows `None`) -- do not search the C++ standard directories for
  header files, but do search the other standard directories. Default OFF.
- **`-P` / `/P`** -- stop compilation and write results to a file. Default OFF. Stops after C/C++
  source files are preprocessed, using the compiler's default naming conventions. On Linux the
  output goes to a `.i` file instead of stdout and, unlike `-E`, has no `#line` directives; the
  default name is the source file prefix with `.i`, changeable with `-o`. Alt: Linux `-F`.
- **`/TP`** (Windows; Linux `None`) -- process all source or unrecognized file types as C++ source
  files. Default OFF (default rules decide). **Deprecated**; may be removed in a future release.
  Replacement for `Kc++` is `-x c++`; for `/TP` it is `/Tp<file>`. Alt: Linux `-x c++`, Windows
  `/Tp`. Host-only.
- **`-Uname` / `/Uname`** -- undefine any definition in effect for the named macro; equivalent to
  `#undef`. Default OFF. On Windows, `/u` undefines all previously defined preprocessor values.
  Undefining an ANSI C macro emits an error:

```text
invalid macro undefinition: <name of macro>
```

- **`-undef`** (Linux; Windows `None`) -- disable all predefined macros. Default OFF (defined macros
  are in effect until undefined).
- **`-X` / `/X`** -- remove standard directories from the include search path. Default OFF. On
  Linux*, `-X` (or `-noinclude`) prevents searching `/usr/include` for files specified in an
  `INCLUDE` statement.

### Dependency-generation options

All default OFF; Linux `-...` / Windows `/Q...`.

| option | Behavior |
|---|---|
| `-M` / `/QM` | Makefile dependency lines per source file, based on its `#include` lines. |
| `-MD` / `/QMD` | Preprocess and compile, generating an output file containing dependency information ending with extension `.d`. |
| `-MFfilename` / `/QMFfilename` | Makefile dependency information in a file. You must also specify `/QM` or `/QMM`. |
| `-MG` / `/QMG` | Like `/QM` but treats missing header files as generated files. |
| `-MM` / `/QMM` | Like `/QM` but excludes system header files. |
| `-MMD` / `/QMMD` | Preprocess and compile a file, then generate an output file (extension `.d`) with dependency information; like `/QMD` but excludes system header files. |
| `-MQtarget` / `/QMQtarget` | Change the default target rule for dependency generation (`target`); like `-MT`/`/QMT` but quotes special Make characters. |
| `-MTtarget` / `/QMTtarget` | Change the default target rule for dependency generation (`target`). |

## Code examples

`fimf-accuracy-bits` (Linux, then Windows):

```bash
-fimf-accuracy-bits=23:sinf,cosf,logf
-fimf-accuracy-bits=52:sqrt,/,trunc
-fimf-accuracy-bits=10:powf

/Qimf-accuracy-bits:23:sinf,cosf,logf
/Qimf-accuracy-bits:52:sqrt,/,trunc
/Qimf-accuracy-bits:10:powf
```

`Fe` — produces `outfile.exe`; without `/Fe` the result would be `file1.exe`:

```bash
prompt> icx /Feoutfile.exe file1.obj file2.cpp file3.cpp
```

`fsystem-debug`:

```bash
icpx -fsycl -fsystem-debug test.cpp
icpx -fsycl -g -fno-system-debug test.cpp
```

`Yc` — `/Fp` names the PCH file, here `precomp.pch`:

```bash
icx /c /Ycheader.h /Fpprecomp foo.cpp
icx /c /Yc /Fpprecomp foo.cpp
```

`Yc` — header-based and source-based names:

```bash
icx /c /Ycheader.h foo.cpp      # PCH name is header.pch
icx /c /Yc foo.cpp              # PCH name is foo.pch
```

`Yu`:

```bash
icx /c /Yuheader.h bar.cpp      # PCH used is header.pch
icx /Yu bar.cpp                 # PCH used is bar.pch
icx /Yu /Fpprecomp bar.cpp      # PCH used is precomp.pch
```

`C` — preserve comments in `prog1.i`:

```bash
icpx -C -P prog1.cpp prog2.cpp
icx  /C /P prog1.cpp prog2.cpp
```

`D` — define macro `SIZE` = 100; with no value the macro defaults to 1:

```bash
icpx -DSIZE=100 prog1.cpp
icx  /DSIZE=100 prog1.cpp
```

`E` — preprocess two source files to stdout:

```bash
icpx -E prog1.cpp prog2.cpp
icx  /E prog1.cpp prog2.cpp
```

`EP` — preprocess to stdout omitting `#line` directives:

```bash
icpx -EP prog1.cpp prog2.cpp
icx  /EP prog1.cpp prog2.cpp
```

`U`:

```bash
icx  /Uia64 prog1.cpp     # Windows
icpx -Uia64 prog1.cpp     # Linux
```

The `fimf-domain-exclusion` `exp2f` instruction sequences are in that option's section above.



## Gotchas & failure modes

- **`-ffp-accuracy` + `fimf-*`/`Qimf-*` is a hard error**; `-ffp-accuracy` is CPU AOT SYCL/OpenMP
  only.
- **The three relative-error options override each other silently** -- last on the command line
  wins, so ordering matters when flags come from different build files.
- **`-fp-model=strict`/`consistent` change other defaults**: `strict` flips `-ffp-contract` to
  `off` and `[Q]fma` to its negative form unless FMA is named explicitly -- the same source is
  slower and less fused than under the `fast` default.
- **`-fp-model=fast=2` gives up NaN/infinity handling** (none used or produced): code relying on
  NaN propagation or infinity arithmetic is "compiles but wrong".
- **`-fp-model=precise`/`strict` do not give reproducibility for parallel code**, only serial code
  (including compiler-vectorized/auto-parallelized code). For OpenMP* FP reductions use
  `KMP_DETERMINISTIC_REDUCTION` with a fixed thread count.
- **Visual Studio C++ projects default to `/fp:precise`, not `/fp:fast`**, so IDE and command-line
  builds of the same source can differ; set the IDE Floating Point Model property to `Fast`.
- **`-O0` changes several defaults at once**: `-fp-speculation` becomes `safe`, `finline-functions`
  becomes OFF, and no `O` level sets `[Q]ftz`.
- **`-ftz` is process-wide and partial**: set only when the main program is compiled, it applies to
  the whole process, and it flushes only runtime-generated denormals. Use `-no-ftz`/`/Qftz-` to keep
  `O3` while restoring denormal behavior.
- **`-fimf-use-svml=true` under value-safe `-fp-model`/`/fp` slightly decreases accuracy**, may not
  raise FP exceptions accurately, does not maintain `errno`, and works only in round-to-nearest-even
  mode -- do not combine it with FP-exception trapping.
- **`-fimf-domain-exclusion` intentionally produces unexpected results on excluded values**; the
  setting must be the integer bitmask (`=2:exp2f` selects a shorter sequence that does not handle
  NaN).
- **Debug flags disable optimization**: `-g` turns off `-O2`, defaults to `-O0`, and sets
  `-fno-omit-frame-pointer`; `/Zi`/`/Z7`/`/ZI` turn off `/O2` and default to `/Od`; enabling
  `-debug`/`/debug` disables optimization unless an `-O`/`/O` level is also passed.
- **Debug-info traps**: `-g` emits DWARF Version 4 (old tools may need `-gdwarf-2`);
  `-gsplit-dwarf` needs binutils-2.24+ and gdb-7.6.1+ and is invisible to tools without split-DWARF
  support; debug information never goes into assemblable files.
- **PCH options are mutually exclusive**: `/Yc` cannot be used in the same compilation as `/Yu`;
  `/Yc` is single-source-file only; `/Yu` requires all source files to share one `.pch`; `/Y-`
  ignores all other PCH files; `/debug:partial` is deprecated.
- **`-E` vs `-EP` vs `-P`**: `-E`/`-EP` write to stdout (`-EP` omits `#line`); `-P` writes a file
  named from the source prefix with `.i` on Linux and omits `#line`; `-EP` plus `P` writes to a
  file. `-o` names the output only with `-c`, `-S`, or `-P`; otherwise it names the executable.
- **Undefining an ANSI C macro with `-U` is an error** (`invalid macro undefinition: <name of
  macro>`); on Windows, `/u` undefines all previously defined preprocessor values.
- **Include-path surgery is easy to get wrong**: `-X`/`-noinclude` on Linux stops the compiler
  searching `/usr/include`, breaking `INCLUDE`-statement lookups; `-nostdinc++` drops only the C++
  standard directories; `-isystem` is searched after all `-I` directories but before the standard
  system directories; `-idirafter` is a second search path after `-I`.
- **Host-only options do nothing for device compilation**: `-Fa`, PCH flags, `gsplit-dwarf`, `TP`,
  and any `fimf-*`/`Qimf-*` option will not change device-specific code generation.

## Source map

- Floating-Point Options (section intro, all `ffp-*`, `fimf-*`, `fma`, `fp-model`, `fp-speculation`,
  `ftz`, `pc`; `fimf-domain-exclusion` `exp2f` example on p. 238) -- pp. 228-251
- Inlining Options (`fgnu89-inline`, `finline`, `finline-functions`, `inline-forceinline`) -- pp. 252-254
- Output, Debug, and Precompiled Header Options (`c` through `Zi`/`Z7`/`ZI`, incl. both `debug`
  variants) -- pp. 254-274
- Preprocessor Options (`B` through `X`, incl. dependency-generation options) -- pp. 274-294
- Code examples gathered from pp. 232, 260, 263, 271, 273, 276, 277, 279, 280, 293
