---
chunk: 14-compat-portability-notices
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 948-976
covers: Standards conformance (C/C++, SYCL*, OpenMP*, IEEE 754-2008), GCC* compatibility and interoperability, Microsoft* compatibility, porting from Microsoft Visual C++* and GCC*, configuration/response files, bundled Intel libraries, technically relevant notices, and the compiler-option + topic index
---

# Compatibility, Portability, Standards Conformance, and Porting

> **Scope.** Which language standards the Intel® oneAPI DPC++/C++ Compiler claims and at what level; whether GCC*- or Microsoft Visual C++*-built objects and libraries interoperate; how to port MSVC* or GCC* makefiles to `icx`/`icpx`; default optimization levels; configuration/response files and the `ICXCFG`/`ICPXCFG` overrides; bundled Intel math/`libirc` libraries.

## Key facts

- **Defaults:** C++17 and C17; `-std` selects others. C/C++ support comes via the **Clang front end** (matching Clang version in the release notes). Levels — C++23/C++20 **Partial**, C++17/C++14/C++11 **Full**, C++98 **Full (except for export)**; C23/C17/C11 **Partial**, C99 **Full** (source prints ISO/IEC 9899:2018 for both C23 and C17 [sic]).
- **SYCL 2020 conformant** · **most of OpenMP* 5.0 and 5.1** · Intel® IEEE 754-2008 Binary Floating-point Conformance Library for **binary32**/**binary64**.
- **GCC*:** C objects **binary compatible** with GCC and the C/C++ language library; shared macros `__GNUC__`, `__GNUG__`, `__GNUC_MINOR__`, `__GNUC_PATCHLEVEL__`; `--gcc-toolchain` picks a non-default GCC/G++.
- **Microsoft*:** fully source- and binary-compatible (native code only) with MSVC; Visual Studio* 2019/2022 projects; `/GS` security checks.
- **Optimization:** `-O2`/`/O2` is in the Intel compiler's default invocation; GCC/MSVC default to none (`-O0`/`/Od`).
- **IPO dummy objects:** `link`→`lld-link` + `lib`→`llvm-lib` (Windows*), `ld`→`lld-link` + `ar`→`llvm-ar` (Linux*, source wording), or link with the Intel compiler.
- **Config:** `..\bin\icx.cfg` (Windows) or `icx.cfg`/`icpx.cfg`; `ICXCFG`/`ICPXCFG` name a private `.cfg` [sic: source prints "ICPXCFGenvironment variable"]; a custom config makes the system file(s) ignored.
- Calling Interfaces / Default accuracy: `int sycl::ext::intel::math::signbit ( float x );` · `int sycl::ext::intel::math::signbit ( double x );`

## Standards Conformance

### C/C++ Standards

| Standard | Intel Feature Support |
|---|---|
| C++23 standard (ISO/IEC 14882:2023) | Partial support |
| C++20 standard (ISO/IEC 14882:2020) | Partial support |
| C++17 standard (ISO/IEC 14882:2017) | Full support |
| C++14 standard (ISO/IEC 14882:2014) | Full support |
| C++11 standard (ISO/IEC 14882:2011) | Full support |
| C++98 standard (ISO/IEC 14882:1998) | Full support (except for export) |
| C23 standard (ISO/IEC 9899:2018) | Partial support |
| C17 standard (ISO/IEC 9899:2018) | Partial support |
| C11 standard (ISO/IEC 9899:2011) | Partial support |
| C99 standard (ISO/IEC 9899:1999) | Full support |

Default standards: C++17 and C17; `-std` selects others. The source says to "visit ISO Standards and search for the specific standard you are interested in, such as ISO/IEC 14882:2020."

### SYCL Standards

The Intel® oneAPI DPC++ Compiler supports the SYCL 2020 Specification and is SYCL 2020 conformant. SYCL is based on C++; the compiler headers include some C++ standard headers, so current C/C++ restrictions relating to library headers also apply to SYCL headers.

### Tested Standard C++ APIs within SYCL Kernels

A set of standard **C++17** APIs, besides host-code support, has been tested for use in SYCL kernels; full list in "Tested Standard C++ APIs".

### OpenMP Standards

Supports most of the OpenMP Application Programming Interface versions **5.0** and **5.1**; see OpenMP Features for status.

### IEEE 754-2008 Standard for Floating-Point Formats

The Intel® IEEE 754-2008 Binary Floating-point Conformance Library conforms to IEEE 754-2008 for the **binary32** and **binary64** binary floating-point interchange formats.

### Additional Language and Standards Information

C: http://www.open-std.org/jtc1/sc22/wg14/ · C++: http://www.isocpp.org/ · OpenMP: http://www.openmp.org/ · SYCL: https://www.khronos.org/sycl/

## GCC Compatibility and Interoperability

### GCC Compatibility

- Compatible with **most GCC versions**; System Requirements lists compatible ones.
- **C object files are binary compatible with GCC and the C/C++ language library**; either compiler can pass objects to the linker.
- **NOTE:** with an Intel product whose compiler has a Clang front end you can also use `icx` or `icpx`.
- Many GNU language extensions are supported. **Statement expressions** are supported, but prohibited inside them: dynamically-initialized local static variables; local non-POD class definitions; try/catch; variable length arrays. **Branching out of a statement expression and statement expressions in constructor initializers are not allowed. Variable-length arrays are no longer allowed in statement expressions.**
- **GCC-style inline ASM** is supported if the assembler code uses AT&T* System V/386 syntax.

### GCC Interoperability

Compilers are interoperable if objects/libraries from both link and the executable runs successfully; this compiler is highly compatible with the GNU compilers. Shared predefined macros: `__GNUC__`, `__GNUG__`, `__GNUC_MINOR__`, `__GNUC_PATCHLEVEL__`. **Caution:** not defining them yields different paths through system header files — poorly tested or otherwise incompatible.

### How the Compiler Uses GCC

Uses the system's GNU tools (GNU header files including `stdio.h`, GNU linker and libraries), so it must match the installed GCC or G++* version. **By default the version is determined from the `PATH` environment variable.** `--gcc-toolchain` sets the base-toolchain location for a non-default GCC/G++ (legacy version for compatibility with third-party libraries, or a later version than the default); the driver uses it to extract header and library locations.

### Compatibility with Open Source Tools

Improved support: **GNU Libtool** (generic shared-library support for package developers), **Valgrind** (debugging/profiling executables on x86 processors), **GNU Automake** (generates `Makefile.ins` from `Makefile.am`).

## Microsoft Compatibility

- **Fully source- and binary-compatible (native code only) with Microsoft Visual C++ (MSVC)**; Intel-built binaries can be debugged from within Microsoft Visual Studio.
- Security checks use **`/GS`**; in the Visual Studio IDE control it via **C/C++ > Code Generation > Security Check**.
- **Microsoft Visual Studio Integration:** compatible with Visual Studio **2019** and **2022** projects.

### Unsupported Features

**Project types:** `.NET`-based CLR C++ project types (specific types vary by Visual Studio version), e.g. **CLR Class Library**, **CLR Console App**, **CLR Empty Project**.

**Major features:** COM Attributes; C++ Accelerated Massive Parallelism (C++ AMP); Managed extensions for C++ (new pragmas, keywords, and command-line options); Event handling (new keywords); select keywords `__abstract`, `__box`, `__delegate`, `__gc`, `__identifier`, `__nogc`, `__pin`, `__property`, `__sealed`, `__try_cast`, `__w64`.

**Preprocessor features:** `#import` directive changes for attributed code; `#using` directive; managed, unmanaged pragmas; `_MANAGED` macro; `runtime_checks` pragma.

### Mix Managed and Unmanaged Code

With managed extensions to C++ in Microsoft Visual Studio .NET you can use the compiler for non-managed code for better performance; keep managed keywords out of non-managed code. (Source points to the Microsoft article "An Overview of Managed/Unmanaged Code Interoperability".)

### Precompiled Header Support

- Intel-generated PCH information is **not compatible** with Microsoft Visual Studio Compiler PCH information.
- **PCH generation and use in the same translation unit is not supported.**

### Compilation and Execution Differences

#### Inlining Functions Marked for dllimport

The Intel compiler attempts to inline functions marked `dllimport`; Microsoft does not. Calls or variables used inside a `dllimport` routine must therefore be available at link time, or the result is an unresolved symbol.

```cpp
// header.h
#ifndef _HEADER_H
#define _HEADER_H
namespace Foo_NS {
        class Foo2 {
        public:
                Foo2(){};
                ~Foo2();
                static int test(int m_i);
        };
}
#endif
```

```cpp
// bug.cpp
#include "header.h"
struct Foo2 {
   static void test();
};
struct __declspec(dllimport) Foo
{
   void getI() { Foo2::test(); };
};
struct C {
  virtual void test();
};
void C::test() { Foo p;      p->getI(); }
int main() {
   return 0;
}
```

#### Enum Bit-Field Signedness

MSVC **always considers enum bit fields signed**, even if not all enum values fit. The Intel compiler considers them **unsigned unless the enum type has at least one enum constant with a negative value**, and warns if the bit field has too few bits to represent all enum values.

## Port from Microsoft Visual C++* to the Intel® oneAPI DPC++/C++ Compiler

From the Windows* command line, modify your makefile to invoke the Intel compiler instead of Microsoft Visual C++. The Visual Studio integration converts Visual C++ projects (**Microsoft Visual Studio 2022** and **2019**). Many of the same options, macros, and environment variables are supported.

### Modify makefiles for Microsoft Applications

Change the compiler variable and review `CPPFLAGS`. Sample Microsoft makefile:

```make
PROGRAM = area.exe
CPPSOURCES = area_main.cpp area_functions.cpp
CPPOBJECTS = area_main.obj area_functions.obj
CPPFLAGS = /RTC1 /EHsc
CPP = cl
$(PROGRAM): $(CPPOBJECTS)
   link.exe /out:$@ $(CPPOBJECTS)
area_main.obj: area_main.cpp area_headers.h
area_functions.obj: area_functions.cpp area_headers.h
clean:   del   $(CPPOBJECTS) $(PROGRAM)
```

Modified makefile: identical except **`CPP = icx`**; set the environment before running `nmake`. Output:

```text
          icx /RTC1 /EHsc   /c area_main.cpp area_functions.cpp

Intel(R) Compiler for applications running on IA-64
Copyright (C) 1985-2006 Intel Corporation. All rights reserved.

area_main.cpp
area_functions.cpp
         link.exe /out:area.exe area_main.obj area_functions.obj
```

### Use IPO in makefiles

IPO by default generates dummy object files containing interprocedural information; linking or creating static libraries with them requires specific LLVM-provided tools. **Replace `link` with `lld-link` and `lib` with `llvm-lib`** — e.g. `CPPFLAGS = /RTC1 /EHsc /Qipo` with link rule `lld-link.exe /out:$@ $(CPPOBJECTS)`.

### Other Considerations

**Set the Environment** — `setvars.bat` sets the proper environment (see "Specifying the Location of Compiler Components").

**Use Optimization** — `O2` is part of the default invocation (the compiler assumes you want performance on Intel® architecture); Microsoft defaults to no optimization (`Od`/`O0`).

| Option | Intel® oneAPI DPC++/C++ Compiler | Microsoft Compiler |
|---|---|---|
| `/Od` | Turns off all optimization. Same as `O0`. | Default. Turns off all optimization. |
| `/O1` | Decreases code size with some increase in speed. | Optimizes code for minimum size. |
| `/O2` | Default. Favors speed optimization with some increase in code size. Intrinsics, loop unrolling, and inlining are performed. | Optimizes code for maximum speed. |
| `/O3` | Enables `-O2` optimizations plus more aggressive optimizations, such as prefetching, scalar replacement, and loop and memory access transformations. | Not supported. |

See Also: `O` compiler option.

**Modify Your Configuration** — configuration-file options apply to every compilation; response-file options only where added on the command line. Move build-wide options to `..\bin\icx.cfg`; **`ICXCFG`** names your own `.cfg` (e.g. `\my_code\my_config.cfg`). **A different configuration file makes the `icx.cfg` system file ignored.**

**Use the Intel Libraries** — additional libraries implement many common functions, some with **CPU dispatch** (different code may execute on different processors): Intel® oneAPI DPC++/C++ Compiler (libm), Short Vector Math Library (svml_disp), libirc and others, linked in by default when references are generated; `sin`/`memset` may be inlined. `libm` includes functions not in the standard math library (**NOTE:** you cannot call the math library with the Microsoft Compiler). Vectorization may turn `libm` calls into short-vector `svml_disp` calls (more efficient; sometimes slightly less precise). `libirc` provides optimized `memcpy`/`memset` and similar, with auto-generated calls and loop transformation. Many `libimf`, SVML, and `libirc` routines favor Intel® microprocessors.

## Port from GCC* to the Intel® oneAPI DPC++/C++ Compiler

| Language | Intel® Compiler | GCC Compiler |
|---|---|---|
| C | `icx` | `gcc` |
| C++ | `icpx` | `g++` |

**NOTE:** unless otherwise indicated, "gcc" refers to both GCC and G++* compilers from the GCC.

### Advantages to Using the Intel® oneAPI DPC++/C++ Compiler

Porting can be as easy as modifying the makefile to invoke the Intel compiler instead of `gcc`, and typically improves performance (especially on Intel processors; possibly on non-Intel too). It provides options optimized for the latest Intel® architecture processors; PGO tools similar to `gprof`; HLO; IPO; Intel intrinsics inlining instructions including Intel® Streaming SIMD Extensions and Intel® Advanced Vector Extensions; and the Intel® oneAPI DPC++/C++ Compiler Math Library. Compatibility with `gcc` brings **binary compatibility** (no need to re-build `gcc` libraries) and supports many of the same options, macros, and environment variables.

### Equivalent Macros

Compatible with the predefined GNU* macros; see "GNU Predefined Macros" for the list.

### Modify makefiles for GCC Applications

Change the GCC compiler variable and review `CFLAGS`. Sample GCC makefile:

```make
CC = gcc
CFLAGS = -O2 -std=c99
all: area_app
area_app: area_main.o area_functions.o
    $(CC) area_main.o area_functions.o -o area
area_main.o: area_main.c
    $(CC) -c $(CFLAGS) area_main.c
area_functions.o: area_functions.c
    $(CC) -c -fno-asm $(CFLAGS) area_functions.c
clean: rm -rf *o area
```

Modified: `CC = icpx`, `CFLAGS = -std=c99`, retaining `-fno-asm` because **it is not supported with the Intel compiler**. Sources using unsupported features (options, language extensions, macros, pragmas, and so on) can be compiled separately with GCC. Because Intel defaults to `O2` and GCC to `O0`, GCC is told to compile at `O2`:

```make
CC = icpx
GCC = gcc
CFLAGS = -std=c99
all: area_app
area_app: area_main.o area_functions.o
    $(CC) area_main.o area_functions.o -o area
area_main.o: area_main.c
    $(CC) -c $(CFLAGS) area_main.c
area_functions.o: area_functions.c
    $(GCC) -c -O2 -fno-asm $(CFLAGS) area_functions.c
clean: rm -rf *o area
```

`area_functions.c` includes GCC-unique features. Example `make` output:

```text
icpx -c -std=c99 area_main.c
gcc -c -O2 -fno-asm -std=c99 area_functions.c
icpx area_main.o area_functions.o -o area
```

### Use IPO in makefiles

IPO dummy object files with interprocedural information need special LLVM-provided tools to link or archive. **Replace `ld` with `lld-link` and `ar` with `llvm-ar`** [sic: source uses `lld-link` in this Linux* example], or use the Intel compiler to link:

```make
CC =icpx
CFLAGS = -std=c99 -ipo
all: area_app
area_app: area_main.o area_functions.o
    $(CC) area_main.o area_functions.o -o area
area_main.o: area_main.c
    $(CC) -c $(CFLAGS) area_main.c
area_functions.o: area_functions.c
    $(CC) -c $(CFLAGS) area_functions.c
clean: rm -rf *o area
```

### Other Considerations

**Set the Environment** — environment variables locate compiler binaries, libraries, man pages, and license files; they differ from GCC's and **are not set by default after installation**. Set `PATH`, `LD_LIBRARY_PATH` (runtime `*.so` lookup for generated executables), `MANPATH`, e.g. by `source setvars.sh`. **NOTE:** no conflict with GCC — both work in one shell.

**Use Optimization** — `O2` is in the default invocation; GCC defaults to `O0`.

| Option | Intel® oneAPI DPC++/C++ Compiler | GCC |
|---|---|---|
| `-O0` | Turns off optimization. | Default. Turns off optimization. |
| `-O1` | Decreases code size with some increase in speed. | Decreases code size with some increase in speed. |
| `-O2` | Default. Favors speed optimization with some increase in code size. Same as option `O`. Intrinsics, loop unrolling, and inlining are performed. | Optimizes for speed as long as there is not an increase in code size. Loop unrolling and function inlining, for example, are not performed. |
| `-O3` | Enables option `O2` optimizations plus more aggressive optimizations, such as prefetching, scalar replacement, and loop and memory access transformations. | Optimizes for speed while generating larger code size. Includes option `O2` optimizations plus loop unrolling and inlining. |

**Target Intel® Processors** — many processor-targeting options are shared; Intel adds options using processor-specific instruction scheduling for the latest Intel® processors.

**Modify Your Configuration** — configuration-file options apply to every compilation, response-file options only where placed. Move build-wide options to `icx.cfg`/`icpx.cfg`; **`ICXCFG` or `ICPXCFG`** names your own `.cfg` (e.g. `/my_code/my_config.cfg`). **A different configuration file makes the system files ignored.**

**Use the Intel Libraries** — **libimf** (Math Library), **libsvml** (Short Vector Math Library), **libirc** and others, linked in by default. `libimf` includes functions not in the standard math library (**NOTE:** you cannot call the math library with GCC). Vectorization may turn `libimf` calls into short-vector `libsvml` calls (more efficient; sometimes slightly less precise). `libirc` provides the optimized string/memory functions described above. All three favor Intel® microprocessors.

> **NOTE:** The Intel Compiler Math Libraries contain performance-optimized implementations for various Intel platforms; by default the best implementation for the underlying hardware is selected at runtime. Library dispatch of multi-threaded code may lead to apparent data races detectable by analysis tools, but while threads run on cores with the same CPUID these are harmless and not a cause for concern.

## Compiler option index (source pp. 963–966)

All option names in the guide's index,in the source's Linux*/`-` and Windows*/`/` spellings; platform-restricted entries annotated. Index page references omitted for compression.

```text
-ansi,/arch,-ax,-B,-C,-c,/C,/c,-D,/D,-daal,-dD,-debug,/debug,-device-math-lib,/device-math-lib,-dM,-dryrun,-dumpmachine,
-dumpversion,-E,/E,/EH,-EP,/EP,/F,-Fa,/Fa,-fasm-blocks,-fast,/fast,-fasynchronous-unwind-tables,-fbuiltin,-fcf-protection,-fcommon,
-fdata-sections,/Fe,-fexceptions,-ffp-accuracy,-ffp-contract,-ffreestanding,-ffunction-sections,-fgnu89-inline,/FI,-fimf-absolute-error,
-fimf-accuracy-bits,-fimf-arch-consistency,-fimf-domain-exclusion,-fimf-max-error,-fimf-precision,-fimf-use-svml,-finline,-finline-functions,
-fiopenmp,/fixed,-fjump-tables,-fkeep-static-consts,-flink-huge-device-code (Linux* only),-flto,-fma,-fmaintain-32-byte-stack-align,
-fmath-errno,-fno-asynchronous-unwind-tables,-fno-exceptions,-fno-gnu-keywords,-fno-operator-names,-fno-rtti,-fno-sycl-libspirv,
-fno-system-debug,/Fo,-foffload-fp32-prec-div,-foffload-fp32-prec-sqrt,-fomit-frame-pointer,-fopenmp,-fopenmp-concurrent-host-device-compile,
-fopenmp-declare-target-scalar-defaultmap,-fopenmp-device-code-split,-fopenmp-device-lib,-fopenmp-device-link,-fopenmp-max-parallel-link-jobs,
-fopenmp-offload-mandatory,-fopenmp-target-buffers,-fopenmp-target-default-sub-group-size,-fopenmp-target-loopopt,-fopenmp-target-simd,
-fopenmp-target-teams-default-vla-alloc-mode,-fopenmp-targets,-foptimize-sibling-calls,-fortlib,-fp,/Fp,/fp,-fp-model,-fp-speculation,
-fpack-struct,-fpermissive,-fpic,-fpie (Linux* only),-fpreview-breaking-changes,-fprofile-dwo-dir,/fprofile-dwo-dir,-fprofile-ml-use,
/fprofile-ml-use,-fprofile-sample-generate,/fprofile-sample-generate,-fprofile-sample-use,/fprofile-sample-use,-fshort-enums,-fstack-protector,
-fstack-protector-all,-fstack-protector-strong,-fstack-security-check,-fsycl,-fsycl-add-default-spec-consts-image,
-fsycl-allow-device-dependencies,-fsycl-allow-device-image-dependencies,-fsycl-dead-args-optimization,-fsycl-device-code-split,-fsycl-device-lib,
-fsycl-device-obj,-fsycl-device-only,-fsycl-early-optimizations,-fsycl-enable-function-pointers,-fsycl-esimd-force-stateless-mem,
-fsycl-explicit-simd,-fsycl-force-target,-fsycl-fp64-conv-emu,-fsycl-help,-fsycl-host-compiler,-fsycl-host-compiler-options,
-fsycl-id-queries-fit-in-int,-fsycl-instrument-device-code,-fsycl-link,-fsycl-max-parallel-link-jobs,-fsycl-optimize-non-user-code,
-fsycl-pstl-offload,/fsycl-pstl-offload,-fsycl-rdc,-fsycl-targets,-fsycl-unnamed-lambda,-fsycl-use-bitcode,-fsyntax-only,-fsystem-debug,
-ftarget-compile-fast,-ftarget-export-symbols,-ftarget-register-alloc-mode,-ftz,-funroll-loops,-funsigned-char,-fuse-ld,
-fvec-allow-scalar-stores,-fvec-non-loop-argument-load,-fvec-peel-loops,-fvec-remainder-loops,-fvec-with-mask,-fverbose-asm,-fvisibility,
-fzero-initialized-in-bss,-g,-g0,-g1,-g2,-g3,/GA,--gcc-toolchain (Linux* only),/Gd,-gdwarf-2,-gdwarf-3,-gdwarf-4,/GF,/GR,
-grecord-gcc-switches (Linux* only),/GS,/Gs,-gsplit-dwarf (Linux* only),/guard,/Gv,/Gw,/GX,/Gy,-H,-help,/help,-I,/I,-idirafter,-imacros,
-inline-forceinline,-ipo,-ipp,-ipp-link,-iprefix,-iquote,-isystem,-iwithprefix,-iwithprefixbefore,/J,-L,-l,/LD,/link,-M,-m,-m64,
-m80387,-march,-masm (Linux* only),-mauto-arch,-mbranches-within-32B-boundaries,-mcmodel (Linux* only),-mcpu,-MD,/MD,-MF,-MG,
-mintrinsic-promote,-MM,-MMD,-mno-gather,-mno-scatter,-momit-leaf-frame-pointer,/MP,-MQ,-MT,/MT,-mtune,-no-intel-lib,-nodefaultlibs,
-nolib-inline,-nolibsycl,/nologo,-nostartfiles,-nostdinc++,-nostdlib,-O,-o,/O,/Od,-Ofast,/Oi,/openmp,-Os,/Os,/Ot,/Ox,-P,/P,-pc,
-pie,-pthread,-qactypes,/Qactypes,/Qauto-arch,/Qax,/Qbranches-within-32B-boundaries,-qcf-protection,/Qcf-protection,-qdaal,/Qdaal,/QdD,
/QdM,/Qfma,/Qfp-accuracy,/Qfp-speculation,/Qfreestanding,/Qftz,/Qgather-,/QH,/Qimf-absolute-error,/Qimf-accuracy-bits,/Qimf-arch-consistency,
/Qimf-domain-exclusion,/Qimf-max-error,/Qimf-precision,/Qimf-use-svml,/Qinline-forceinline,/Qintrinsic-promote,/Qiopenmp,/Qipo,-qipp,/Qipp,
/Qipp-link,/Qkeep-static-consts,/Qlong-double,/QM,/Qm,/Qm64,/Qmaintain-32-byte-stack-align,/QMD,/QMF,/QMG,-qmkl,/Qmkl,-qmkl-ilp64,
/Qmkl-ilp64,-qmkl-sycl-impl,/Qmkl-sycl-impl,/QMM,/QMMD,/QMQ,/QMT,/Qno-builtin-name,/Qno-intel-lib,-qopenmp (using in apps),-qopenmp,
/Qopenmp (using in apps),/Qopenmp-concurrent-host-device-compile,/Qopenmp-declare-target-scalar-defaultmap,/Qopenmp-device-code-split,
/Qopenmp-device-link,-qopenmp-link,/Qopenmp-max-parallel-link-jobs,/Qopenmp-offload-mandatory,-qopenmp-simd,/Qopenmp-simd,-qopenmp-stubs,
/Qopenmp-stubs,/Qopenmp-target-buffers,/Qopenmp-target-default-sub-group-size,/Qopenmp-target-loopopt,/Qopenmp-target-simd,/Qopenmp-targets,
-qopt-assume-no-loop-carried-dep,/Qopt-assume-no-loop-carried-dep,-qopt-dword-index-for-array-of-structs,/Qopt-dword-index-for-array-of-structs,
-qopt-dynamic-align,/Qopt-dynamic-align,-qopt-for-throughput,/Qopt-for-throughput,-qopt-mem-layout-trans,/Qopt-mem-layout-trans,
-qopt-multiple-gather-scatter-by-shuffles,/Qopt-multiple-gather-scatter-by-shuffles,-qopt-prefetch,/Qopt-prefetch,-qopt-prefetch-distance,
/Qopt-prefetch-distance,-qopt-prefetch-loads-only,/Qopt-prefetch-loads-only,-qopt-report,/Qopt-report,-qopt-report-file,/Qopt-report-file,
-qopt-report-names,/Qopt-report-names,-qopt-report-phase,/Qopt-report-phase,-qopt-report-stdout,/Qopt-report-stdout,-qopt-streaming-stores,
/Qopt-streaming-stores,-qopt-zmm-usage,/Qopt-zmm-usage,-Qoption,/Qoption,/Qpc,/Qregcall,/Qsave-temps,/Qscatter-,/Qstd,
/Qtarget-register-alloc-mode,-qtbb,/Qtbb,-qunknown-option-as-warning (Linux* only),/Qunroll,/Quse-intel-optimized-headers,/Qvec,
/Qvec-non-loop-argument-load,/Qvec-peel-loops,/Qvec-remainder-loops,/Qvec-threshold,/Qvec-with-mask,/Qvec_assume_index_overflow,/Qvecabi,/Qx,
/QxHost,/Qzero-initialized-in-bss,-regcall,-S,/S,-save-temps,-shared,-shared (Linux* only),-shared-intel,-shared-libgcc (Linux* only),
/showIncludes,-sox,-static (Linux* only),-static-intel,-static-libgcc (Linux* only),-static-libstdc++ (Linux* only),-std,/std,-strict-ansi,
--sysroot (Linux* only),-T (Linux* only),-tbb,/TC,/Tc,/TP,/Tp,/tune,-U,-u,/U,-undef,-unroll,-use-intel-optimized-headers,-use-msasm,-v,
/vd,-vec,-vec-threshold,-vec_assume_index_overflow,-vecabi,--version,/vmg,/vmv,-w,/W,/w,-Wa,-Wabi,/Wabi,-Wall,/Wall,
-Wcheck-unicode-security,/Wcheck-unicode-security,-Wcomment,/Wcomment,-Wdeprecated,/Wdeprecated,-Werror,-Werror-all,/Werror-all,
-Wextra-tokens,/Wextra-tokens,-Wformat,/Wformat,-Wformat-security,/Wformat-security,-Wl,-Wmain,/Wmain,-Wmissing-declarations,
/Wmissing-declarations,-Wmissing-prototypes,/Wmissing-prototypes,-Wno-sycl-strict,-Wp,-Wpointer-arith,/Wpointer-arith,-Wreorder,/Wreorder,
-Wreturn-type,/Wreturn-type,-Wshadow,-Wsign-compare,/Wsign-compare,-Wstrict-aliasing,/Wstrict-aliasing,-Wstrict-prototypes,/Wstrict-prototypes,
-Wtrigraphs,/Wtrigraphs,-Wuninitialized,/Wuninitialized,-Wunknown-pragmas,-Wunused-function,-Wunused-variable,/Wunused-variable,
-Wwrite-strings,/Wwrite-strings,/WX,-X,-x,/X,-x (type),-xHost,-Xlinker,-Xopenmp-target,-Xs,-Xsycl-target,/Y-,/Yc,/Yu,/Z7,/Zc,/Zg,/ZI,
/Zi,/Zl,-Zp,/Zp
```

## Topic index (source pp. 966–976)

Condensed from the guide's index; named entities preserved, generic pointers compressed. Page numbers are the guide's.

- **Pragmas** — overview 595, Intel-specific 596, gcc*-compatible 620, HP*-compatible 620, Microsoft*-compatible 620; `block_loop` (factor/level 597); `noblock_loop` 597; `distribute_point` 598; `forceinline` (recursive 600); `inline` (recursive 600); `noinline` 600; `ivdep` 602; `loop_count` (avg/max/min/n 603); `nofusion` 604; `novector` 604; `omp target variant dispatch` 605; `noprefetch` (var 607); `prefetch` (distance/hint/var 607); `nounroll` 608; `unroll` (n 608); `nounroll_and_jam` 609; `unroll_and_jam` (n 609); `simd` 781; `vector` 611; `fenv_access` 394.
- **Attributes / `__declspec`** — `align`/`align_value`/`aligned` 399; `code_align` 402; `const` 402; `cpu_dispatch` 403; `cpu_specific` 403; `target` 405; `__assume_aligned` 795; `__regcall` 49.
- **OpenMP*** — support/using 666; supported pragmas summary 614; `omp.h` 728; header files 680,720; contexts 724; context selectors 725 (scoring/matching 727); advanced issues, C/C++ interoperability, Fortran and C/C++ interoperability, debugging, performance — all 728; environment variables 702; `KMP_AFFINITY` (modifier/offset/permute/type 702); `KMP_LIBRARY` 696; `KMP_TOPOLOGY_METHOD` 702; `OMP_STACKSIZE` 666; `openmp_version` 680,720; linked-runtime-library option 208; OpenMP* API option 206; sequential-mode option 210; compatibility/legacy/support libraries 696; library file names 696; Intel extension routines 690; runtime routines 680,690,720; SIMD-enabled functions 783; load balancing 671; thread affinity/topology maps 702; parallel processing thread model 668; shared scalars 732; worker thread 696.
- **Asynchronous I/O** (intro/library 535, template class 550) — `async_class` 550, `thread_control` 550; methods `clear_queue()` 553, `get_error_operation_id()` 552, `get_last_error()` 552, `get_last_operation_id()` 551, `get_status()` 551, `resume_queue()` 553, `stop_queue()` 552, `wait()` 551; functions `aio_cancel()` 545, `aio_error()` 542, `aio_fsync()` 544, `aio_read()` 536, `aio_return()` 542, `aio_suspend()` 540, `aio_write()` 536, `lio_listio()` 546; `errno` macro 548; Error Handling 548; examples 537–547; Usage Example 553.
- **Math library functions** (using/code examples 817) — Trigonometric 825 `acos acosd asin asind atan atan2 atand atand2 cos cosd cot cotd sin sincos sincosd sind tan tand`; Hyperbolic 831 `acosh asinh atanh cosh sinh sinhcosh tanh`; Exponential 832 `cbrt exp exp10 exp2 expm1 frexp hypot ilogb ldexp log log10 log1p log2 logb pow scalb scalbn sqrt`; Special 837 `annuity compound erf erfc gamma gamma_r j0 j1 jn lgamma lgamma_r tgamma y0 y1 yn`; Nearest Integer 841 `ceil floor llrint llround lrint lround modf nearbyint rint round trunc`; Remainder 844 `fmod remainder remquo`; Miscellaneous 845 `copysign fabs fdim finite fma fmax fmin nextafter`; Complex 850 `cabs cacos cacosh carg casin casinh catan catanh ccos ccosh cexp cexp10 cimag cis clog clog2 conj cpow cproj creal csin csinh csqrt ctan ctanh`; C99 macros 854 `fpclassify isfinite isgreater isgreaterequal isinf isless islessequal islessgreater isnan isnormal isunordered signbit`. Options: input-argument domain 235, consistent results 234, level of accuracy 241.
- **IEEE 754-2008 Binary Floating-point Conformance Library** (using 555) — general-computational 563 `add binary32_to_binary64 binary64_to_binary32 div fma from_hexstring from_int32 from_int64 from_string from_uint32 from_uint64 mul sqrt sub to_hexstring to_string to_int{32,64}_{ceil,floor,int,rnint,rninta,xceil,xfloor,xint,xrnint,xrninta} to_uint{32,64}_{ceil,floor,int,rnint,rninta,xceil,xfloor,xint,xrnint,xrninta}`; homogeneous 561 `ilogb maxnum maxnum_mag minnum minnum_mag next_down next_up rem round_integral_exact round_integral_nearest_away round_integral_nearest_even round_integral_negative round_integral_positive round_integral_zero scalbn`; non-computational 574 `class defaultMode getBinaryRoundingDirection is754version1985 is754version2008 isCanonical isFinite isInfinite isNaN isNormal isSignaling isSignMinus isSubnormal isZero lowerFlags radix raiseFlags restoreFlags restoreModes saveFlags setBinaryRoundingDirectionsaveModes testFlags testSavedFlags totalOrder totalOrderMag`; quiet 569 `copy copysign negate`; comparisons 570 `quiet_equal quiet_greater quiet_greater_equal quiet_greater_unordered quiet_less quiet_less_equal quiet_less_unordered quiet_not_equal quiet_not_greater quiet_not_less quiet_ordered quiet_unordered signaling_equal signaling_greater signaling_greater_equal signaling_greater_unordered signaling_less signaling_less_ unordered [sic] signaling_less_equal signaling_not_equal signaling_not_greater signaling_not_less`.
- **SDLT** — `aos1d_container` 421,424,429,433,436,438–440,465,468,470,471; `soa1d_container` 427; `aos1d_container::accessor`/`soa1d_container::accessor` 441,444,445,448,450,451,453; `::const_accessor` 452; accessors 441,451; indexes 464; number representation 459; proxy objects 456; `Proxy` 457,458; `aligned_offset` 463; `fixed_offset` 464; `linear_index` 465; `min_val` 471; `access_by` 432; `SDLT_DEBUG` 470; `SDLT_INLINE` 470; sdlt layout namespace 435; example programs 472,479.
- **Other indexed topics** — SIMD class libraries 480–533 (operator groups, Quick reference 486–525, example 532, `valarray` 533); HLO 803; IPO 804–807 (`xiar`/`xild`/`xilib`/`xilibtool`/`xilink`); `LIB`/`LD_LIBRARY_PATH` 409; `libqkmalloc` 417; `libistrconv` 579–581; Intel® Performance Libraries 35 (IPP, MKL, TBB); Level Zero 745,749; ICV 730; Xeon Phi™ coprocessor 678; MPI 36; Open Source tools 949; vectorization 763–781; `-fp-model`/`/fp` 393,394; DAZ/FTZ 395; MXCSR 395; IEEE 754-2008 398; NaN/normalized/signed infinity/signed zer [sic] 398; FP semantics 246; FMA 245; `errno` after math calls 311; Visual Studio* 24–48.

## Gotchas & failure modes

- **C++20/C++23 and C are partial** (C++98 full support excludes `export`); beyond C++17/C17 use `-std` and expect missing features. **Do not undefine `__GNUC__`/`__GNUG__`/`__GNUC_MINOR__`/`__GNUC_PATCHLEVEL__`** — system headers then take alternate, possibly poorly tested/incompatible paths.
- **Statement expressions** forbid dynamically-initialized local statics, local non-POD class definitions, try/catch, and variable length arrays; no branching out of one; not allowed in constructor initializers; VLAs are "no longer allowed" there. **GCC-style inline ASM** needs AT&T* System V/386 syntax.
- **Optimization defaults differ** (Intel `-O2`/`/O2` vs GCC/MSVC `-O0`/`/Od`); compile GCC-only files with `-O2` as the source does. **`-fno-asm` is unsupported by the Intel compiler** — keep it only for GCC.
- **IPO objects are not ordinary objects**: use `lld-link`/`llvm-lib` (Windows*) or `lld-link`/`llvm-ar` (Linux*, source wording), or link with the Intel compiler, else linking/archiving fails.
- **PCH**: Intel PCH ≠ MSVC PCH, and generation plus use in one translation unit is unsupported. **`dllimport` inlining** can produce unresolved symbols. **Enum bit-field signedness differs**: MSVC always signed; Intel unsigned unless the enum has a negative constant (warns on too-narrow fields).
- **Unsupported MSVC surface**: CLR/.NET projects, COM Attributes, C++ AMP, managed extensions, event keywords (`__abstract`, `__box`, `__delegate`, `__gc`, `__identifier`, `__nogc`, `__pin`, `__property`, `__sealed`, `__try_cast`, `__w64`), `#using`, `_MANAGED`, managed/unmanaged pragmas, `runtime_checks`.
- **Config precedence**: a custom `ICXCFG`/`ICPXCFG` file makes the system `icx.cfg`/`icpx.cfg` ignored; config-file options apply to all compilations, response-file options only where placed. **Environment is not set automatically**: source `setvars.sh` (or `setvars.bat`) for `PATH`, `LD_LIBRARY_PATH`, `MANPATH`; the math library cannot be called from the Microsoft compiler or GCC.
- **Precision/portability**: `svml_disp`/`libsvml` may be slightly less precise than `libm`/`libimf`; `libimf`/SVML/`libirc` favor Intel® microprocessors; multi-threaded library dispatch can look like data races but is harmless when threads share the same CPUID.

## Source map

- Standards Conformance — pp. 948–949
- GCC Compatibility and Interoperability — pp. 949–950
- Microsoft Compatibility — pp. 950–952
- Port from Microsoft Visual C++* — pp. 952–957
- Port from GCC* — pp. 957–962
- Notices and Disclaimers — p. 962 (legal text not reproduced)
- Index — pp. 963–976 (option entries pp. 963–966; topic entries pp. 966–976)
