---
chunk: 04-options-lang-data-diagnostics
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 295-344
covers: Component control (Qoption); language (ansi–Zp); data (fcommon–Qlong-double); compiler diagnostic (qunknown-option-as-warning–Wwrite-strings); compatibility (gcc-toolchain)
---

# Component Control, Language, Data, and Compiler Diagnostic Options

> **Scope.** Reference for DPC++/C++ compiler options in the Component Control, Language, Data, and
> Compiler Diagnostic sections, plus the first Compatibility option. Each option's exact
> Linux/Windows spelling, default, legal values, platform restriction, and documented caveats.

## Key facts

- An option with no form for an OS is listed `None`; `(L)`/`(W)` = Linux-only / Windows-only,
  unmarked = both. Host-only options (no effect on device compilation under offloading) are marked
  `host only` in the table.
- Deprecated: `/Zg` (may be removed in a future release; **no replacement option**).
- Windows `/Q...` negated form appends `-` (`/Qkeep-static-consts-`,
  `/Qmaintain-32-byte-stack-align-`, `/Qzero-initialized-in-bss-`); `/GS` negates as `/GS-`.
- Page 295 opens with the tail of an option from the preceding section (name absent here): used with
  the `I` option to stop the compiler searching the default include path and point it at an alternate
  path; Alternate Options Linux `-nostdinc`, Windows `None`; See Also `I` compiler option.

## Option quick table

Unmarked options exist on both platforms as `-X` / `/X` and `-Xno-…` / `/Xno-…`. Default after `;`.

### Component control and compatibility

| option | purpose; default |
|---|---|
| `-Qoption,string,options` (Linux), `/Qoption,string,options` (Windows) | pass options to a named tool; OFF. `string` = tool, `options` = comma-separated options for it; list below; host only. |
| `--gcc-toolchain=dir` (L) | base toolchain location; Windows `None`; no Default documented. |

### Language Options

| option | purpose; default |
|---|---|
| `-ansi` (L) | gcc `-ansi` compatibility and ANSI conformance level; OFF (GNU C++ preferred over ANSI C); `-strict-ansi` for strict. |
| `-fno-gnu-keywords` (L) | do not recognize `typeof` as a keyword; OFF (recognized). |
| `-fno-operator-names` (L) | disable standard operator names; OFF (supported). |
| `-fno-rtti` (L) | disable runtime type information (RTTI); OFF (enabled). |
| `-fpermissive` (L) | allow non-conformant code; OFF (not allowed). |
| `-fshort-enums` (L) | allocate as many bytes as needed for enumerated types; OFF (default bytes). |
| `-fsyntax-only` (L), `/Zs` (W) | syntax check only, no object file; OFF (normal compilation). |
| `-funsigned-char` (L), `/J` (W) | default char unsigned; OFF (signed); sets `_CHAR_UNSIGNED = 1`. |
| `-std=val` (L), `/Qstd:val`, `/std:val` (W) | conform to a language standard; `c++17 or c17`; values below. |
| `-strict-ansi` (L) | strict ANSI conformance dialect; OFF; sets `fmath-errno`; host only. |
| `/vdn` (W) | hidden vtordisp members in C++ objects; `0` suppress, `1` when necessary, `2` all virtual base classes with virtual functions; `/vd1`. `2` recommended for a destructor-only virtual function or `dynamic_cast` on a partially constructed object; MS VC++* `/vdn` compatibility. |
| `/vmg` (W) | general representation for pointers to members; OFF (default rules). |
| `-x type` (L) | subsequent source files recognized as `type`; OFF. `type` ∈ `c++`, `c++-header`, `c++-cpp-output`, `c`, `c-header`, `cpp-output`, `assembler`, `assembler-with-cpp`, `none`; `none` reverts to file extension. |
| `/Zc:arg` (W) | ANSI C conformance per language feature; varies. Negative form opts out (e.g. `/Zc:threadSafeInit-`); table below. |
| `/Zg` (W) | generate function prototypes; OFF; **deprecated**, no replacement; host only. |
| `-Zp[n]` (L), `/Zp[n]` (W) | structure byte-boundary alignment; `n` ∈ `1`,`2`,`4`,`8`,`16`; `Zp16` (boundary 16 or natural alignment); omitting `n` gives `Zp16`. |

### Data Options

| option | purpose; default |
|---|---|
| `-fcommon`, `-fno-common` (L) | common symbols as global definitions; `-fcommon` (no). `-fno-common` makes them global: one module only or link fails with multiple defined symbols. `int i;` is a common symbol treated as an external reference; if no other unit defines it globally the linker allocates memory. |
| `-fkeep-static-consts`, `-fno-keep-static-consts` (L), `/Qkeep-static-consts`, `/Qkeep-static-consts-` (W) | preserve unreferenced variables; default negative (discarded unless `-O0` L / `/Od` W). Negated form saves static-data memory when optimizing. |
| `-fmaintain-32-byte-stack-align`, `-fno-maintain-32-byte-stack-align` (L), `/Qmaintain-32-byte-stack-align`, `/Qmaintain-32-byte-stack-align-` (W) | realign stack to 32-byte when uncertain for external-linkage functions, retain for others; OFF. Not with Clang `-mstack-alignment`/`-mstackrealign`; host only. |
| `-fmath-errno`, `-fno-math-errno` (L) | `errno` testable after standard math calls; `-fno-math-errno` (not tested). `-fmath-errno` restricts optimization (math functions side-effecting); `-fno-math-errno` often faster, safe with IEEE exceptions. |
| `-fpack-struct` (L) | pack structure members; OFF. May be unusable with standard (system) C/C++ libraries; alternate `-Zp1`. |
| `-fpic`, `-fno-pic` (L) | position-independent code; `-fno-pic`. `-fpic` = full symbol preemption, required for shared objects; also `-fPIC`. |
| `-fpie` (L) | position-independent code for executables only; OFF; better optimization of some symbol references; needs `-pie` on the link line; also `-fPIE`. |
| `-fstack-protector[-keyword]`, `-fno-stack-protector[-keyword]` (L) | stack overflow checks (storing more data in a stack variable than allocated): `-fstack-protector-strong` = routines with any type of buffer, `-fstack-protector-all` = every routine, no keyword = string buffer; defaults `-fno-stack-protector`, `-fno-stack-protector-strong`, `-fno-stack-protector-all`. gcc compatibility (gcc/glibc if available, else Intel). |
| `-fstack-security-check`, `-fno-stack-security-check` (L) | detect buffer overruns overwriting the return address; default negative. Always Intel; gcc version `fstack-protector`; Windows alternate `/GS`; host only. |
| `-fvisibility=…` family (L) | symbol visibility (5 options, listed under `fvisibility` below); `-fvisibility=default`. Last wins; Clang options supported. |
| `-fzero-initialized-in-bss`, `-fno-zero-initialized-in-bss` (L), `/Qzero-initialized-in-bss`, `/Qzero-initialized-in-bss-` (W) | zero-initialized vars in DATA; default negative → BSS (saves space); Description also says negative → DATA. `[sic: conflict]` |
| `-ftls-model=local-exec` (L), `/GA` (W) | faster TLS access; OFF. Main .EXE faster `__declspec(thread)` access; .DLLs may error; Clang option name. |
| `/Gs[n]` (W) | stack-check threshold in bytes; `/Gs` = >4KB (4096 bytes), also when `n` omitted; inserts `__chkstk()` in the prologue. Host only. |
| `/GS`, `/GS-` (W) | full stack security checking; `/GS-` (no detection); Microsoft compatibility; C++ Linux alternate `-fstack-security-check`. |
| `-mcmodel=mem_model` (L) | memory model; default `-mcmodel=small` on Intel® 64. `-mcmodel=small`/`-mcmodel=medium`/`-mcmodel=large`; affects code size/performance; table below; not for SYCL. |
| `/Qlong-double` (W) | `long double` 64 (OFF) → 80 bits; alignment 16 bytes so size 16 bytes with only lower 10 bytes valid; caveats below. |

### Compiler Diagnostic Options

All rows exist on both platforms (`-X` / `/X`, negations `-Wno-…` / `/Wno-…`); exceptions marked.

| option | purpose; default |
|---|---|
| `-qunknown-option-as-warning`, `-qno-unknown-option-as-warning` (L) | unknown Linux options → warning not error; `-qno-unknown-option-as-warning` (errors). Host only. |
| `-w` (L), `/w` (W) | disable all warnings; OFF. Windows alternate `/W0`. |
| `/Wn` (W) | level: `0` errors only; `1` warnings+errors; `2` plus extra warnings on Linux (=1 on Windows); `3` remarks+warnings+errors (production); `4` level 3 + informational warnings/remarks (W only); `5` all remarks/warnings/errors (W only); `n=1`. |
| `-Wabi`, `-Wno-abi` | warn if code is not C++ ABI compliant; `Wno-abi`. |
| `-Wall` (L), `/Wall` (W) | many warnings+errors; OFF. Windows `/W4`; Linux similar to gcc `-Wall` (all errors, some warnings). |
| `-Wcheck-unicode-security`, `-Wno-check-unicode-security` | Unicode vulnerability checking: bi-directional formatting codes, zero-width chars in strings/identifiers, homoglyphs; `Wno-check-unicode-security`; host only. |
| `-Wcomment`, `-Wno-comment` | warn when `/*` appears inside a `/* */` comment; `Wno-comment`. |
| `-Wdeprecated`, `-Wno-deprecated` | warn for deprecated C++ headers; `Wdeprecated`; no effect in C mode; defines `__DEPRECATED`. |
| `-Werror` (L), `/WX` (W) | all warnings → errors; OFF. |
| `-Werror-all` | all warnings and enabled remarks → errors; OFF. |
| `-Wextra-tokens`, `-Wno-extra-tokens` | warn about extra tokens at the end of preprocessor directives; `Wno-extra-tokens`. |
| `-Wformat`, `-Wno-format` | check arguments of `printf`, `scanf`, and so forth; `Wno-format`. |
| `-Wformat-security`, `-Wno-format-security` | warn on insecure format-function use (format string not a literal, no format arguments); `Wno-format-security`. |
| `-Wmain`, `-Wno-main` | warn if the return type of `main` is unexpected; `Wno-main`. |
| `-Wmissing-declarations`, `-Wno-missing-declarations` | warn for global functions/variables without prior declaration; `Wno-missing-declarations`. |
| `-Wmissing-prototypes`, `-Wno-missing-prototypes` | warn for missing prototypes (global functions defined without a previous declaration); `Wno-missing-prototypes`. |
| `-Wpointer-arith`, `-Wno-pointer-arith` | warn for questionable pointer arithmetic; `Wno-pointer-arith`. |
| `-Wreorder` | warn when member initializer order differs from execution order; OFF. C++ only. |
| `-Wreturn-type`, `-Wno-return-type` | warn on missing return type, `return expr` in a `void` function, or reaching the closing brace of a non-void function; ON for the last condition only. |
| `-Wshadow`, `-Wno-shadow` | warn when a declaration hides a previous one; `Wno-shadow`. Same as `-ww1599`. |
| `-Wsign-compare`, `-Wno-sign-compare` | warn when signed/unsigned comparison may be incorrect after conversion to unsigned; `Wno-sign-compare`; Linux gcc compatibility. |
| `-Wstrict-aliasing`, `-Wno-strict-aliasing` | warn for code that might violate the optimizer's strict aliasing rules; `Wno-strict-aliasing`; `-fstrict-aliasing` and `-Ofast` also enable strict aliasing. |
| `-Wstrict-prototypes`, `-Wno-strict-prototypes` | warn for functions declared/defined without specified argument types; `Wno-strict-prototypes`. |
| `-Wtrigraphs`, `-Wno-trigraphs` | warn about trigraphs that might change program meaning; `Wno-trigraphs`. |
| `-Wuninitialized`, `-Wno-uninitialized` | warn if a variable is used before being initialized; `Wno-uninitialized`. |
| `-Wunknown-pragmas`, `-Wno-unknown-pragmas` | warn if an unknown `#pragma` directive is used; `Wunknown-pragmas`. |
| `-Wunused-function`, `-Wno-unused-function` | warn if a declared function is not used; `Wno-unused-function`. |
| `-Wunused-variable`, `-Wno-unused-variable` | warn if a local or non-constant static variable is unused after declaration; `Wno-unused-variable`. |
| `-Wwrite-strings` | diagnostic if `const char *` is converted to (non-const) `char *`; OFF. |

## Component Control Options

### Qoption

Linux `-Qoption,string,options`; Windows `/Qoption,string,options`. Passes options to a specified
tool: `string` is the tool name, `options` are comma-separated valid options for that tool (some
tools require them within quotation marks `" "`). If an argument contains a space or tab, enclose
the whole argument in quotation marks; separate multiple arguments with commas. Default OFF: no
options are passed to tools.

`string` values: `cpp` (preprocessor), `c` (the Intel® oneAPI DPC++/C++ Compiler), `asm`
(assembler), `link` (linker). Windows also `masm` (Microsoft assembler). Linux also `as`
(assembler), `gas` (GNU assembler), `ld` (loader), `gld` (GNU loader), `lib` (additional library),
`crt` (the crt%.o files linked into executables to contain the place to start execution). Host only.

## Language Options

### std, Qstd

Linux `-std=val`; Windows `/Qstd:val`, `/std:val` (Microsoft* compatibility). `/Qstd` supports more
values than `/std`; `/std` complies with values permitted by Microsoft `/std`.

Both platforms: `c++2b` (Working Draft for ISO C++ 2023 DIS standard; Windows `/Qstd` only),
`c++20` (2020 ISO C++ DIS), `c++17` (2017 ISO C++ with amendments), `c++14` (2014 ISO C++ with
amendments), `c18` and `c17` (both the 2017 ISO C standard; Windows `c18` only for `/Qstd`; `c17`
also `iso9899:2017`, `c18` also `iso9899:2018`), `c11` (2011 ISO C standard; also `iso9899:2011`).

Windows `/std` only: `c++latest` — all currently implemented compiler and standard library features
proposed for the next draft standard plus some in-progress and experimental features.

Linux only — C++: `c++11` (2011 ISO C++ with amendments), `c++98`, `c++03` (1998 ISO C++ with
amendments). C: `c2x` (ISO C2x Working Draft), `c99` (1999 ISO C; also `iso9899:1999`), `c90`,
`c89` (1990 ISO C; also `iso9899:1990`). GNU variants: `gnu++2b` (ISO C++ 2023 DIS Working Draft),
`gnu++20` (2020 ISO C++ DIS), `gnu++17`, `gnu++14`, `gnu++11`, `gnu++98`, `gnu++03` (matching ISO
C++ standard with amendments), `gnu2x` (ISO C2x Working Draft), `gnu18`, `gnu17` (2017 ISO C),
`gnu11` (2011 ISO C), `gnu99` (1999 ISO C), `gnu90`, `gnu89` (1990 ISO C) — all plus GNU
extensions.

Default `c++17 or c17`: C++ conforms to the 2017 ISO C++ standard; C conforms to the 2017 ISO C
standard.

### Zc

Windows `/Zc:arg`; Linux `None`. ANSI C standard conformance per language feature, with
Microsoft*-compatible settings. Default varies; specify the negative form to leave a non-default
(e.g. `/Zc:threadSafeInit-`).

| `/Zc` setting | description |
|---|---|
| `alignedNew[-]` | C++17 aligned allocation functions (default for C++17); off via `/Zc:alignedNew-` |
| `char8_t[-]` | `char8_t` from C++2a; off via `/Zc:char8_t-` (default) |
| `cplusplus[-]` | `__cplusplus` reports the supported standard; off via `/Zc:cplusplus-` (default) |
| `dllexportInlines[-]` | dllexport/dllimport inline member functions of dllexport/import classes (default); off via `/Zc:dllexportInlines-` |
| `sizedDealloc[-]` | C++14 sized global deallocation functions (default); off via `/Zc:sizedDealloc-` |
| `strictStrings[-]` | const qualification for string literals; off via `/Zc:strictStrings-` (default) |
| `threadSafeInit[-]` | thread-safe initialization of local statics (default); off via `/Zc:threadSafeInit-` |
| `trigraphs[-]` | trigraph character sequences; off via `/Zc:trigraphs-` (default) |
| `twoPhase[-]` | two-phase name lookup in templates; off via `/Zc:twoPhase-` (default) |

### x (type option)

Linux `-x type`; Windows `None`. Files found subsequent to `-x type` are recognized as `type`:
`c++` (C++ source file), `c++-header` (C++ header file), `c++-cpp-output` (C++ pre-processed file),
`c` (C source file), `c-header` (C header file), `cpp-output` (C pre-processed file), `assembler`
(Assembly file), `assembler-with-cpp` (Assembly file needing preprocessing), `none` (disables
recognition, reverts to file extension). Default OFF: type unchanged. Example below.

## Data Options

### fvisibility

Linux `-fvisibility=arg`, `-fvisibility-global-new-delete-hidden`, `-fvisibility-inlines-hidden`,
`-f[no]visibility-inlines-hidden-static-local-var`, `-fvisibility-ms-compat`; Windows `None`.

| `arg` | meaning |
|---|---|
| `default` | visible outside this shared object; referenceable and preemptable by a same-named definition in another component |
| `hidden` | not visible outside this shared object; not directly referenceable |
| `internal` | same as `hidden` |
| `protected` | seen by the dynamic linker but always resolves to an object in this shared object; referenceable but not overridable; not supported on all targets |

| option | description |
|---|---|
| `-fvisibility=arg` | visibility for all global declarations (`arg` = `hidden`, `internal`, `default`, `protected`) |
| `-fvisibility-global-new-delete-hidden` | hidden visibility for global C++ `operator new`/`delete` declarations |
| `-fvisibility-inlines-hidden` | hidden visibility by default for inline C++ member functions |
| `-fvisibility-inlines-hidden-static-local-var` / `-fno-visibility-inlines-hidden-static-local-var` | with `-fvisibility-inlines-hidden`, static variables in inline C++ member functions also get hidden visibility; disable with the `-fno-` form |
| `-fvisibility-ms-compat` | default visibility for global types, hidden for global functions and variables |

If specified more than once, the last takes precedence. Precedence greatest to least visibility:
`default`, `protected`, `hidden`.

### mcmodel

Linux `-mcmodel=mem_model`; Windows `None`. Default `small` on Intel® 64 architecture.

| `mem_model` | meaning |
|---|---|
| `-mcmodel=small` | code and data restricted to the first 2GB of address space; all accesses can use Instruction Pointer (IP)-relative addressing |
| `-mcmodel=medium` | code restricted to the first 2GB, no data restriction; code accesses IP-relative, data accesses absolute |
| `-mcmodel=large` | no restriction on code or data; all accesses absolute |

Global and static data with total size smaller than 2GB: `-mcmodel=small` is sufficient; larger than
2GB requires `-mcmodel=medium` or `-mcmodel=large`. Allocating more than 2GB works with any setting.
IP-relative addressing needs only 32 bits vs 64 for absolute and is somewhat faster, so
`-mcmodel=small` has the least performance impact. NOTE: does not apply to SYCL.
`-mcmodel=medium`/`-mcmodel=large` set `-shared-intel`; specifying `-static-intel` while either is
set displays an error. Example below. See Also `shared-intel`, `fpic`.

### Qlong-double

Windows `/Qlong-double`; Linux `None`. Changes the `long double` default size from 64 bits (OFF) to
80 bits. Alignment is 16 bytes and size must be a multiple of alignment, so the Windows* size is
also 16 bytes; only the lower 10 bytes (80 bits) have valid data.

- Source using double extended precision floating-point types (FP80) must be carefully segregated
  from source not written to support them; code assuming a size/layout for FP80 may fail at compile
  time, link time, or runtime.
- Microsoft* C Standard Library and Microsoft* C++ Standard Template Library do not support FP80
  datatypes; check with your library vendor whether they support FP80 formats.
- The Microsoft* compiler and Microsoft*-provided library routines (such as `printf` or
  `long double` math functions) do not support 80-bit floating-point values and should not be called
  from code compiled with `Qlong-double`.
- Starting with Microsoft Visual Studio 2019 version 16.10, `/std:c++latest` with `/Qlong-double`
  may cause compilation errors when `<complex>`, `<xutility>`, or `<cmath>` is included directly or
  indirectly.

## Compiler Diagnostic Options

`/Wn` and `Wall` can override each other, as can `Wall`, `-wn`, and `/Wn`; the last option on the
command line takes precedence (Windows `/W4` = `/Wall`).

## Compatibility Options

### gcc-toolchain

Linux OS `--gcc-toolchain=dir`; Windows OS `None`. Lets you specify the location of the base
toolchain; `dir` is that location. (Extract ends here; no Default, Description, or Alternate Options
in range.)

## Code examples

### x (type option)

`file1.c99` (C) and `file2.cplusplus` (C++) are not recognized extensions; `file3.c` (C) and
`file4.cpp` (C++) are recognized.

```bash
icpx -x c file1.c99 -x c++ file2.cplusplus -x none file3.c file4.cpp
```

### mcmodel

```bash
icx -shared-intel -mcmodel=medium -o prog prog.c
```

(This content does not apply to SYCL.)

### qunknown-option-as-warning

```bash
icpx -qunknown-option-as-warning -fbad-opt file.cpp
```

```text
icpx: warning: unknown argument ignored: '-fbad-opt' [-Wunknown-argument]
```

### Qlong-double with /std:c++latest

```cpp
#include <iostream>
#include <complex>

int main()
{long double ld2 = 1256789.98765432106L;int iNan = isnan(ld2);std::cout << "Hello World!\n"; }
```

```bash
ksh-3.2$ icx -c -EHsc -GR        -std:c++latest /Qlong-double /MD        test1.cpp
```

```text
test1.cpp
.../include\xutility(5971,24): error: no matching function for call to '_Bit_cast'
    const auto _Bits = _Bit_cast<_Uint_type>(_Xx);
.../include\xutility(6014,12): note: in instantiation of 'std::_Float_abs_bits<long double, 0>'
.../include\cmath(1239,31): note: in instantiation of 'std::_Is_finite<long double, 0>'
.../include\cmath(1324,12): note: in instantiation of 'std::_Common_lerp<long double>'
.../include\xutility(66,36): note: candidate template ignored: requirement
    'conjunction_v<std::integral_constant<bool, false>,
    std::is_trivially_copyable<unsigned long long>,
    std::is_trivially_copyable<long double>>' was not satisfied [with _To = _Uint_type,
    _From = long double]
```

## Gotchas & failure modes

- **Default-on diagnostics:** `Wdeprecated`, `Wunknown-pragmas`, and (one condition) `Wreturn-type`
  are on by default; Linux `-Wall` is not all warnings (`-w2`/`-w3` are); `/W4` equals `/Wall` only
  on Windows; `-Werror`/`/WX` makes warnings errors and `-Werror-all` also promotes enabled remarks.
- **Stack:** `-fstack-protector` may use gcc/glibc while `-fstack-security-check` always uses
  Intel's; `/GS-` (default) means no detection; `-fmaintain-32-byte-stack-align` conflicts with
  Clang `-mstack-alignment`/`-mstackrealign`.
- **Link/conflict:** `-fpic` required for shared objects; `-fpie` needs `-pie` on the link line;
  `-mcmodel=medium`/`large` set `-shared-intel`, so `-static-intel` with either errors; `/GA` on
  .DLLs may error; `-fzero-initialized-in-bss` Default/Description disagree (BSS vs DATA).
- **Silent no-ops:** host-only options do nothing to device code under offloading; unknown Linux
  options are errors unless `-qunknown-option-as-warning`.
- **Positional:** `-x` applies only to files found *subsequent* to it; `-x none` restores
  file-extension recognition. The top of p. 295 belongs to an earlier option (Linux alternate
  `-nostdinc`), not to `Qoption`.

## Source map

- p. 295 fragment; Component Control Options (`Qoption`) — pp. 295–296
- Language Options (`ansi` through `Zp`) — pp. 296–309
- Data Options (`fcommon` through `Qlong-double`) — pp. 309–324
- Compiler Diagnostic Options (`qunknown-option-as-warning` through `Wwrite-strings`) — pp. 324–344
- Compatibility Options (`gcc-toolchain`) — p. 344
