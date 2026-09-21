---
chunk: 01-options-list-and-optimization
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 52-114
covers: __regcall register/stack placement; option conventions, general command-line rules and description anatomy; the complete alphabetical option list; Optimization Options; Advanced Optimization Options
---

# Compiler Options: Conventions, Alphabetical List, Optimization and Advanced Optimization Options

> **Scope.** Answers what a given oneAPI DPC++/C++ compiler option does, its Linux `-`/Windows `/`
> syntax, its default and legal values, and which options are deprecated or experimental. Contains one
> row per documented option, full entries for the Optimization and Advanced Optimization groups, the
> command-line rules, and `__regcall` register placement.

## Key facts

- Clang compiler options are supported but **not documented** here; check `-help` on the command line (else see the Clang documentation).
- Intel vs open-source Clang defaults: `-fp-model=fast` vs `-fp-model=precise`; `-O2` vs `-O0`; `-fveclib=SVML` vs no default set for `-fveclib`.
- MSVC-compatible `/` syntax requires the appropriate compiler driver (see "Invoke the Compiler"). New options are announced in the Release Notes.
- Doc name shortcuts: `Fa` = Linux `-Fa` / Windows `/Fa`; `[Q]ipo` = `-ipo` / `/Qipo`; `[q or Q]opt-report` = `-qopt-report` / `/Qopt-report`.
- Options may be **case sensitive**: `c` prevents linking, `C` places comments in preprocessed output.
- `/Od`, `/O1`, `/O2`, `/O3`, `/Ox` are mutually exclusive; the last specified wins. Generally the last of enabling/disabling forms wins, and the last `O` option wins.
- Default optimization level is `O2`; on Linux, if `-g` is specified the default is `-O0` unless `-O2` (or higher) is also explicitly on the command line.
- Host-compilation-only (no effect on device-specific compilation when offloading): `fast`, `Ofast`, `nolib-inline`, `ffreestanding`, `ipp-link`, `mno-gather`, `mno-scatter`, `qopt-prefetch-loads-only`, `vec`, `vec-threshold`.
- Deprecated/removal-tracked: `fsycl-allow-device-dependencies` (replacement `fsycl-allow-device-image-dependencies`), `fsycl-explicit-simd`, `m64`/`Qm64`, `TP`, `Zg`; deprecated alternates `-daal` (for `qdaal`), `-tbb` (for `qtbb`).
- No figure occurs in this page range.

## __regcall placement in registers or on the stack (p52)

After the data-type and structured-data-type classifications, `__regcall` parameters and return values are either put into registers (Available Registers) or placed in memory:

1. Each chunk (eight bytes on systems based on Intel® 64 architecture) of a value of **Data Type** is assigned a register class. If enough registers from **Available Registers** are available, the whole value is passed in registers; otherwise the value is passed using the stack.
2. If the classification were to use one or more register classes, the registers of these classes from the table in **Available Registers** are used, in the order given there.
3. If no more registers are available in one of the required register classes, the whole value is put on the stack.

Registers that preserve their values across a `__regcall` call, as long as they were not used for passing a parameter or returning a value:

| Register Class/ABI | Intel® 64 for Linux | Intel® 64 for Windows |
|---|---|---|
| GPR | R12 - R15, RBX, RBP, RSP | R12 - R15, RBX, RBP, RSP |
| FP | None | None |
| MMX | None | None |
| XMM | XMM8 - XMM15 | XMM8 - XMM15 |
| YMM | XMM8 - XMM15 | XMM8 - XMM15 |
| ZMM | XMM8 - XMM15 | XMM8 - XMM15 |

All other registers do not preserve their values across this call. [sic: the source repeats `XMM8 - XMM15` in the YMM and ZMM rows.]

## Conventions Used for Compiler Options

Name shortcuts: **no initial `-`/`/`** = same name on both OSes (`Fa` → `-Fa` / `/Fa`); **`[Q]name`** = Windows adds `Q` (`[Q]ipo` → `-ipo` / `/Qipo`); **`[q or Q]name`** = Linux `q`, Windows `Q` (`[q or Q]opt-report` → `-qopt-report` / `/Qopt-report`). More dissimilar names are shown in full.

| Syntax notation | Meaning |
|---|---|
| `/option` or `-option` | `/` = Windows, `-` = Linux (Linux `-help`, Windows `/help`). If an option exists on all supported OSes, no slash/dash appears in the general description, only in the syntax. |
| `/option:argument` or `-option=argument` | Requires an argument (parameter). |
| `/option:keyword` or `-option=keyword` | Requires one of the keyword values. |
| `/option[:keyword]` or `-option[=keyword]` | Usable alone or with an optional keyword. |
| `option[n]`, `option[:n]`, `option[=n]` | Usable alone or with an optional value, e.g. `-unroll[=n]` — `n` may be omitted or a valid value given. |
| `option[-]` | Trailing hyphen disables: `/Qglobal_hoist-` disables `/Qglobal_hoist`. |
| `[no]option` or `[no-]option` | `no`/`no-` preceding disables: `-global_hoist` enables, `-no-global_hoist` disables; some place `no` later (`-fno-inline` disables `-finline`). |

## General Rules for Compiler Options

- Case matters and can change meaning (`c` = no link; `C` = comments in preprocessed output).
- Options on the command line apply to all files named on that command line.
- Arguments can be file names, strings, letters, or numbers; a string containing spaces must be quoted.
- Options may appear in any order; the command line both compiles and links unless certain options say otherwise.
- Some names may be abbreviated to as many characters as are needed to identify them uniquely.
- Certain options accept one or more keyword arguments (e.g. architecture option `x`); to specify multiple keywords you typically repeat the option.
- To disable an option, use its negative form if one exists; if enabling and disabling forms both appear, the **last on the command line takes precedence**.
- Options remain in effect for the whole compilation unless overridden by a compiler `#pragma`.
- **Linux**: cannot combine options after one dash — `-Ec` is incorrect, `-E -c` is correct. **Windows**: `/Ec` is incorrect, `/E /c` is correct.
- All compiler options must precede `/link` options, if any.
- A comma can sometimes separate keywords: `ifx /warn:usage,declarations test.f90` is valid.
- Optimization options can be disabled by putting `/Od` last (it is in the mutually exclusive group `/Od`, `/O1`, `/O2`, `/O3`, `/Ox`).

## What Appears in the Compiler Option Descriptions

Grouped by functional category, alphabetical within each. Each has: primary name + short description; **Syntax** (Linux and Windows; `None` if invalid on an OS); **Arguments** (`None` if none); **Default**; **Description**; **IDE Equivalent** (Intel® IDE Property Pages; the Windows IDE is Microsoft Visual Studio .NET); **Alternate Options (does not apply to SYCL)** — synonyms, `None` if none; some alternates are deprecated, and many options have an older valid spelling using underscores (`_`) instead of hyphens (`-`). Some add **Example(s)** and **See Also**.

## Alphabetical Option List

One row per documented option, in source order. `-` = no value/default and no special note. Linux/Windows spellings follow the Conventions above. Options detailed in the two sections below are marked `▸`.

| name | purpose | key values / default / notes |
|---|---|---|
| ansi | GCC `ansi` language compatibility. | - |
| arch | Targetable features/instruction sets. | - |
| ax, Qax | Feature-specific auto-dispatch paths for Intel® CPUs. | - |
| B | Include/library/executable search directory. | - |
| C | Comments in preprocessed output. | - |
| c | Object only; no link. | - |
| D | Define a macro (optional value). | - |
| dD, QdD | Like `-dM`, but emits `#define` directives. | - |
| debug (Linux*) | Debug info generation on/off. | Linux* |
| debug (Windows*) | Debug info generation on/off. | Windows* |
| device-math-lib | Certain device libraries on/off. | - |
| dM, QdM | Macro definitions in effect after preprocessing. | - |
| dryrun | Show driver tool commands without executing. | - |
| dumpmachine | Target machine and OS configuration. | - |
| dumpversion | Compiler version number. | - |
| E | Preprocessor output to stdout. | - |
| EH | Exception-handling model. | - |
| EP | Preprocessor output to stdout without `#line`. | - |
| F (Windows) | Program stack reserve amount. | Windows |
| Fa | Assembly listing file. | - |
| fasm-blocks | Asm blocks/whole functions within C/C++. | - |
| fast ▸ | Maximize speed across the entire program. | OFF; Linux `-ipo -O3 -static -fp-model fast`; Windows `/O3 /Qipo /fp:fast`; host-only |
| fasynchronous-unwind-tables | Unwind precision at instruction vs call boundary. | - |
| fbuiltin, Oi ▸ | Inline expansion of intrinsic functions on/off. | ON |
| fcf-protection, Qcf-protection | CET protection against certain exploit attacks. | preliminary CET support |
| fcommon | Common symbols treated as global definitions. | - |
| fdata-sections, Gw | Per-data-item COMDAT sections. | - |
| Fe | Built program / dynamic-link library name. | - |
| fexceptions | Exception handling table generation. | - |
| ffp-accuracy | Required FP accuracy for FP ops and library calls. | - |
| ffp-contract | When fused FP ops (e.g. FMA) may be formed. | - |
| ffreestanding, Qfreestanding ▸ | Freestanding-environment compilation. | OFF |
| ffunction-sections, Gy | Per-function COMDAT sections. | - |
| fgnu89-inline | C89 inline semantics in C99 mode. | - |
| fimf-absolute-error, Qimf-absolute-error | Max absolute error for math library results. | - |
| fimf-accuracy-bits, Qimf-accuracy-bits | Relative error for math results incl. division/sqrt. | - |
| fimf-arch-consistency, Qimf-arch-consistency | Consistent math results across microarchitectures. | - |
| fimf-domain-exclusion, Qimf-domain-exclusion | Input-argument domain where math functions must be correct. | - |
| fimf-max-error, Qimf-max-error | Max relative error for math results incl. division/sqrt. | - |
| fimf-precision, Qimf-precision | Accuracy level used to select math functions. | - |
| fimf-use-svml, Qimf-use-svml | Use SVML instead of LIBM. | - |
| finline | Inline `__inline` functions; perform C++ inlining. | - |
| finline-functions | Function inlining, single-file compilation. | - |
| fiopenmp, Qiopenmp | OpenMP* recognition + multithreaded generation. | alternate to `-qopenmp` / `/Qopenmp` |
| FI | Include a file as the header file. | - |
| fixed | Program loadable only at its preferred base address. | - |
| fjump-tables ▸ | Jump tables for switch statements. | `-fjump-tables` |
| fkeep-static-consts, Qkeep-static-consts | Preserve allocation of unreferenced variables. | - |
| flink-huge-device-code | Device code later in the binary; avoids 32-bit PC-relative relocations >2GB. | - |
| flto | Whole-program link time optimization (LTO). | - |
| fma, Qfma | FMA instructions when present on the target processor. | - |
| fmaintain-32-byte-stack-align, Qmaintain-32-byte-stack-align | 32-byte stack alignment for external/other functions. | - |
| fmath-errno | `errno` testable after standard math library calls. | - |
| fno-gnu-keywords | Do not treat `typeof` as a keyword. | - |
| fno-operator-names | Disable standard operator names. | - |
| fno-rtti | Disable runtime type information (RTTI). | - |
| fno-sycl-libspirv | Disable the libspirv (SPIR-V* tools library) check. | - |
| Fo | Object file name. | - |
| foffload-fp32-prec-div | Enforce correct rounding; SYCL*/offload languages do not follow IEEE 754 precision (e.g. PyTorch*). | - |
| foffload-fp32-prec-sqrt | Enforce correct rounding; SYCL*/offload languages do not follow IEEE 754 precision (e.g. PyTorch*). | - |
| fomit-frame-pointer | EBP as general-purpose register in optimizations. | - |
| fopenmp | OpenMP recognition + multithreading via LLVM front-end lowering. | Less performant than `-fiopenmp` (Intel backend lowering); **does not support offloading to GPUs** |
| fopenmp-concurrent-host-device-compile, Qopenmp-concurrent-host-device-compile | Parallel host and target compilation for OpenMP offload. | experimental |
| fopenmp-declare-target-scalar-defaultmap, Qopenmp-declare-target-scalar-defaultmap | Implicit data-mapping/sharing rules for a scalar in a target pragma. | - |
| fopenmp-device-code-split, Qopenmp-device-code-split | Parallel SPIR-V kernel compilation, OpenMP offload AOT. | - |
| fopenmp-device-lib | Device libraries for an OpenMP target on/off. | - |
| fopenmp-device-link | Device link during compilation instead of the link step. | Produces device binaries in the fat object |
| fopenmp-max-parallel-link-jobs, Qopenmp-max-parallel-link-jobs | Max parallel actions during device linking. | - |
| fopenmp-offload-mandatory, Qopenmp-offload-mandatory | Generate only a device version of OpenMP target regions. | - |
| fopenmp-target-buffers, Qopenmp-target-buffers | Workaround for incorrect SPIR-V code when a target object is larger than 4GB. | - |
| fopenmp-target-default-sub-group-size, Qopenmp-target-default-sub-group-size | Default sub-group size for SPMD kernels, SPIR64-based devices. | - |
| fopenmp-target-loopopt, Qopenmp-target-loopopt | Loop optimizer + auto-vectorization for OpenMP device compilation at O2+. | - |
| fopenmp-target-simd, Qopenmp-target-simd | OpenMP SIMD loop vectorization for device compilation at O2+. | - |
| fopenmp-target-teams-default-vla-alloc-mode, Qopenmp-target-teams-default-vla-alloc-mode | Default VLA/assumed-size local-copy allocation on teams/distribute privatization clauses. | - |
| fopenmp-targets, Qopenmp-targets | Offload to a specified GPU target when OpenMP features are enabled. | - |
| foptimize-sibling-calls ▸ | Optimize tail recursive calls. | `-foptimize-sibling-calls` |
| fortlib | Link Fortran libraries (mixed-language C/C++). | - |
| Fp | Alternate precompiled-header path or file name. | - |
| fpack-struct | Pack structure members together. | - |
| fpermissive | Allow non-conformant code. | - |
| fpic | Position-independent code. | - |
| fpie | PIC that can be linked only into executables. | - |
| fp-model, fp | Floating-point calculation semantics. | Intel `-fp-model=fast` vs Clang `-fp-model=precise` |
| fpreview-breaking-changes | Give up backward compatibility; enable next-major breaking changes. | - |
| fprofile-dwo-dir | `.dwo` file directory with `-fprofile-sample-generate` + `-gsplit-dwarf`. | experimental |
| fprofile-ml-use | Pre-trained ML model predicts branch probabilities for PGO. | - |
| fprofile-sample-generate | Generate information and adjust optimization for Hardware Profile-Guided Optimization (HWPGO). | - |
| fprofile-sample-use | Use HWPGO information. | experimental |
| fp-speculation, Qfp-speculation | Floating-point speculation mode. | - |
| fshort-enums | Allocate as many bytes as needed for enumerated types. | - |
| fstack-protector | Stack overflow security checks for certain/all routines. | - |
| fstack-security-check | Buffer-overrun detection code. | - |
| fsycl | Compile as a SYCL* program, not plain C++11. | - |
| fsycl-add-default-spec-consts-image | Device-image copies with default specialization-constant values. | - |
| fsycl-allow-device-dependencies | Device-image dependencies when splitting device code. | **deprecated**; replacement `fsycl-allow-device-image-dependencies` |
| fsycl-allow-device-image-dependencies | Device-image dependencies when splitting device code. | - |
| fsycl-dead-args-optimization | Eliminate SYCL dead kernel arguments. | - |
| fsycl-device-code-split | SYCL device code module assembly. | - |
| fsycl-device-lib | Device libraries for a SYCL target on/off. | - |
| fsycl-device-obj | Device-code format in the resulting object. | experimental |
| fsycl-device-only | Device-only binary. | - |
| fsycl-early-optimizations | LLVM optimizations before SPIR-V generation. | - |
| fsycl-enable-function-pointers | Function pointers and virtual functions for SYCL kernels/device functions. | experimental |
| fsycl-esimd-force-stateless-mem | Stateless memory accesses within ESIMD kernels on the target device. | experimental |
| fsycl-explicit-simd | Experimental Explicit SIMD SYCL extension. | **deprecated**, may be removed |
| fsycl-force-target | Force the specified target triple device when extracting device code. | - |
| fsycl-fp64-conv-emu | fp64 partial emulation for conversion-only fp64 kernels. | Requires an Intel GPU supporting fp64 partial emulation |
| fsycl-help | Device compiler backend help information. | - |
| fsycl-host-compiler | Host compiler for the overall offloading compilation. | - |
| fsycl-host-compiler-options | Pass options to the `fsycl-host-compiler` compiler. | - |
| fsycl-id-queries-fit-in-int | Assume SYCL ID queries fit within MAX_INT. | - |
| fsycl-instrument-device-code | Link Instrumentation and Tracing Technology (ITT) device libraries for VTune™. | - |
| fsycl-link | Partial link of device binaries. | - |
| fsycl-max-parallel-link-jobs | Spawn up to N processes to link SYCL applications. | experimental |
| fsycl-optimize-non-user-code | Optimize SYCL framework utilities; leave kernel code unoptimized for debugging. | - |
| fsycl-pstl-offload | Offload C++ standard parallel algorithms to a SYCL device. | - |
| fsycl-rdc | Relocatable device code during SYCL offload target compilation. | - |
| fsycl-remove-unused-external-funcs | Remove unused `SYCL_EXTERNAL` functions in SYCL device compilation. | - |
| fsycl-targets | Code generation for specified device targets. | - |
| fsycl-unnamed-lambda | Unnamed SYCL lambda kernels. | - |
| fsycl-use-bitcode | Device code as LLVM Intermediate Representation (IR) bitcode in fat objects. | - |
| fsyntax-only, Zs | Syntax check only. | - |
| fsystem-debug | Debug information for system-header declarations. | - |
| ftarget-compile-fast | Faster compilation, less optimal target code. | experimental |
| ftarget-export-symbols | Expose exported target-library symbols to other modules. | - |
| ftarget-register-alloc-mode | Register allocation mode for specific hardware. | - |
| ftz, Qftz | Flush denormal results to zero. | - |
| funsigned-char, J | Default character type unsigned. | - |
| fuse-ld | Non-default linker (`ld` on Linux, `link` on Windows). | - |
| fvec-allow-scalar-stores, Qvec-allow-scalar-stores ▸ | Ensure vectorization of an explicit SIMD loop. | `-fno-vec-allow-scalar-stores` / `/Qvec-allow-scalar-stores-` |
| fvec-non-loop-argument-load, Qvec-non-loop-argument-load ▸ | Non-loop vectorizer load combining. | `-fvec-non-loop-argument-load` / `/Qvec-non-loop-argument-load` |
| fvec-peel-loops, Qvec-peel-loops ▸ | Peel loop vectorization. | `-fno-vec-peel-loops` / `/Qvec-peel-loops-` |
| fvec-remainder-loops, Qvec-remainder-loops ▸ | Remainder loop vectorization. | `-fno-vec-remainder-loops` / `/Qvec-remainder-loops-` |
| fvec-with-mask, Qvec-with-mask ▸ | Short trip-count loop vectorization with masking. | `-fno-vec-with-mask` / `/Qvec-with-mask-` |
| fverbose-asm | Assembly listing with compiler comments, options, version. | - |
| fvisibility | Default visibility for global symbols or symbols in declarations/functions/variables. | - |
| fzero-initialized-in-bss, Qzero-initialized-in-bss | Zero-initialized variables in the DATA section. | - |
| g | Debug information level in the object file. | - |
| GA | Faster access to certain thread-local storage (TLS) variables. | - |
| gcc-toolchain | Base toolchain location. | - |
| Gd | `__cdecl` as the default calling convention. | - |
| gdwarf | DWARF Version format for debug information. | - |
| GF ▸ | Read-only string-pooling optimization. | OFF [sic: source default text says "Read/write string-pooling optimization is enabled" — unclear in source] |
| GR | C++ Runtime Type Information (RTTI) on/off. | - |
| grecord-gcc-switches | Invocation options appended to `DW_AT_producer` in DWARF debug info. | - |
| GS | Buffer-overrun detection code. | - |
| Gs | Stack-check routine call threshold. | - |
| gsplit-dwarf | Separate object file containing DWARF debug information. | - |
| guard | Control flow protection mechanisms. | - |
| Gv | `__vectorcall` for vector type arguments. | - |
| H, QH | Display include file order and continue compilation. | - |
| help | Supported compiler options, alphabetical. | - |
| I | Additional include file search directory. | - |
| idirafter | Second include file search path. | - |
| imacros | Header included in front of all other headers in the translation unit. | - |
| inline-forceinline, Qinline-forceinline | Treat inline routines as forceinline. | - |
| ipo, Qipo | Interprocedural optimization between files. | - |
| ipp-link, Qipp-link ▸ | Static/dynamic threaded Intel® IPP runtime library link. | `dynamic` (static if Linux `-static`); requires `[Q]ipp`; host-only |
| iprefix | Prefix for referencing directories containing header files. | - |
| iquote | Front of the include search path for quote-included files. | - |
| isystem | Start of the system include path. | - |
| iwithprefix | Append to the `-iprefix` prefix; end of the include directories. | - |
| iwithprefixbefore | As `-iwithprefix`, but placed with `-I` include directories. | - |
| l | Search a named library when linking. | - |
| L | Search libraries in a directory before the standard directories. | - |
| LD | Link as a dynamic-link (DLL) library. | - |
| link | Pass user options directly to the linker at compile time. | - |
| m | Targetable features, including ISA. | - |
| M, QM | Makefile dependency lines per source file. | - |
| m64, Qm64 | Code for a specific architecture. | Legacy, **deprecated**, will be removed |
| m80387 | Whether x87 instructions may be used. | - |
| march | Code for processors supporting certain features. | - |
| masm | Assembler output file in a selected dialect. | - |
| mauto-arch, Qauto-arch | Auto-dispatch paths for x86 architecture processors. | - |
| mbranches-within-32B-boundaries, Qbranches-within-32B-boundaries | 32-byte branch/fused-branch alignment. | - |
| mcmodel | Memory model for code generation and data storage. | - |
| MD | Multithreaded, dynamic-link runtime library references. | - |
| MD, QMD | Preprocess and compile to a dependency file ending in `.d`. | - |
| MF, QMF | Makefile dependency information in a file. | - |
| MG, QMG | Makefile dependency lines per source file. | - |
| mintrinsic-promote, Qintrinsic-promote | Auto-promote architecture for intrinsics needing a CPU feature. | - |
| MM, QMM | Makefile dependency lines per source file. | - |
| MMD, QMMD | Dependency information output file. | - |
| mno-gather, Qgather- ▸ | Disable gather instructions in auto-vectorization. | OFF; host-only |
| mno-scatter, Qscatter- ▸ | Disable scatter instructions in auto-vectorization. | OFF; host-only |
| momit-leaf-frame-pointer | Frame pointer omission/retention in leaf functions. | - |
| MP | Multiple processes for compiling many source files at once. | - |
| MQ, QMQ | Default target rule for dependency generation. | - |
| MT | Multithreaded, static runtime library references. | - |
| MT, QMT | Default target rule for dependency generation. | - |
| mtune, tune | Per-processor optimization without extended instruction sets (unlike `-march`). | - |
| nodefaultlibs | No standard libraries when linking. | - |
| no-intel-lib, Qno-intel-lib | Disable linking specified/all Intel® libraries. | - |
| nolib-inline ▸ | Disable standard library/intrinsic inline expansion. | OFF; host-only |
| nolibsycl | Disable SYCL runtime library linking. | - |
| nologo | No compiler version display. | - |
| nostartfiles | No standard startup files when linking. | - |
| nostdinc++ | Skip C++ standard header directories (other standard dirs still searched). | - |
| nostdlib | No standard libraries and startup files when linking. | - |
| O ▸ | Application code optimization. | `O2`; with `-g` on Linux the default is `-O0` unless `-O2`+ is explicit |
| o | Output file name. | - |
| Od ▸ | Disable all optimizations. | OFF; Windows only |
| Ofast ▸ | Aggressive options to improve application speed. | OFF; host-only |
| Os ▸ | Size-neutral optimizations; smaller than `O2`. | OFF (but `Os` is default if `O1`) |
| Ot ▸ | All speed optimizations. | `/Ot` (`Od` disables all; `O1` makes `Os` default); Windows only |
| Ox ▸ | Maximum optimizations. | OFF; Windows only |
| P | Stop compilation and write results to a file. | - |
| pc, Qpc | Floating-point significand precision control. | - |
| pie | PIC that will be linked into an executable. | - |
| pthread | pthreads library for multithreading support. | - |
| qactypes, Qactypes ▸ | AC data type headers + libraries for CPU compilations. | OFF |
| qdaal, Qdaal ▸ | Link certain Intel® oneAPI Data Analytics Library (oneDAL) libraries. | OFF; alternate `-daal` deprecated |
| qipp, Qipp ▸ | Link some/all Intel® Integrated Performance Primitives (Intel® IPP) libraries. | OFF |
| Qlong-double | Default `long double` data type size. | - |
| qmkl, Qmkl ▸ | Link certain Intel® oneAPI Math Kernel Library (oneMKL) libraries. | OFF; on Windows must be at compile time |
| qmkl-ilp64, Qmkl-ilp64 ▸ | Link the ILP64-specific oneMKL version. | OFF; on Windows must be at compile time |
| qmkl-sycl-impl, Qmkl-sycl-impl ▸ | Link specific oneMKL SYCL libraries. | OFF |
| qopenmp, Qopenmp | OpenMP recognition + multithreaded generation. | Alternate name for `-fiopenmp` / `/Qiopenmp` |
| qopenmp-link | Static/dynamic OpenMP runtime library link. | - |
| qopenmp-simd, Qopenmp-simd | OpenMP SIMD compilation on/off. | - |
| qopenmp-stubs, Qopenmp-stubs | Compile OpenMP programs in sequential mode. | - |
| Qoption | Pass options to a specified tool. | - |
| qopt-assume-no-loop-carried-dep, Qopt-assume-no-loop-carried-dep ▸ | Level of loop performance tuning. | `=0` |
| qopt-dword-index-for-array-of-structs, Qopt-dword-index-for-array-of-structs ▸ | Dword indexes for arrays of structs up to a size. | OFF |
| qopt-dynamic-align, Qopt-dynamic-align ▸ | Dynamic data alignment optimizations. | disabled |
| qopt-for-throughput, Qopt-for-throughput ▸ | Throughput optimization for single-job vs multi-job mode. | OFF |
| qopt-mem-layout-trans, Qopt-mem-layout-trans ▸ | Compiler memory layout transformation level. | `=0` |
| qopt-multiple-gather-scatter-by-shuffles, Qopt-multiple-gather-scatter-by-shuffles ▸ | Adjacent gather/scatter vector reference optimization by shuffles. | varies |
| qopt-prefetch, Qopt-prefetch ▸ | Prefetch insertion optimization. | varies |
| qopt-prefetch-distance, Qopt-prefetch-distance ▸ | Prefetch distance for compiler-generated prefetches in loops. | OFF |
| qopt-prefetch-loads-only, Qopt-prefetch-loads-only ▸ | Prefetch loads only, ignore stores. | OFF; host-only |
| qopt-report, Qopt-report | YAML file of optimization transformation information. | - |
| qopt-report-file, Qopt-report-file | Report output to file, stderr, or stdout. | - |
| qopt-report-names, Qopt-report-names | Mangled/unmangled names in the report. | - |
| qopt-report-phase, Qopt-report-phase | Optimizer phases for which reports are generated. | - |
| qopt-report-stdout, Qopt-report-stdout | Send the report to stdout. | - |
| qopt-streaming-stores, Qopt-streaming-stores ▸ | Streaming stores for optimization. | `=auto` |
| qopt-zmm-usage, Qopt-zmm-usage ▸ | zmm registers usage level. | `low` with `[Q]xCORE-AVX512`; `high` with `[Q]xCOMMON-AVX512` |
| qtbb, Qtbb ▸ | Link Intel® oneAPI Threading Building Blocks (oneTBB) libraries. | OFF; alternate `-tbb` deprecated |
| qunknown-option-as-warning | Unknown Linux compiler options warn instead of error. | - |
| regcall, Qregcall | `__regcall` convention for functions not specifying one. | - |
| S | Assembly file only; no link. | - |
| save-temps, Qsave-temps | Save intermediate compilation files. | - |
| shared | Dynamic shared object instead of an executable. | - |
| shared-intel | Intel-provided libraries linked dynamically. | - |
| shared-libgcc | GNU libgcc linked dynamically. | - |
| showIncludes | Display a list of the include files. | - |
| sox | Save compilation options in the executable file. | - |
| static | No linking with shared libraries. | - |
| static-intel | Intel-provided libraries linked statically. | - |
| static-libgcc | GNU libgcc linked statically. | - |
| static-libstdc++ | GNU libstdc++ linked statically. | - |
| std, Qstd | Conform to a specific language standard. | - |
| strict-ansi | Strict ANSI conformance dialect. | - |
| sysroot | Root directory for headers and libraries. | - |
| T | Read link commands from a file. | - |
| TC | All source/unrecognized file types as C. | - |
| Tc | One file as C. | - |
| TP | All source/unrecognized file types as C++. | **deprecated**, may be removed |
| Tp | One file as C++. | - |
| U | Undefine the specified macro's definition. | - |
| u (Linux) | The specified symbol is undefined. | Linux |
| undef | Disable all predefined macros. | - |
| unroll, Qunroll ▸ | Maximum loop unroll count. | `-unroll` / `/Qunroll` |
| use-intel-optimized-headers, Quse-intel-optimized-headers ▸ | Add the performance headers directory to the include path search list. | disabled |
| use-msasm | Asm blocks/whole functions within C/C++. | - |
| v | Display and execute driver tool commands. | - |
| vd | Hidden vtordisp members in C++ objects. | - |
| vec, Qvec ▸ | Vectorization on/off. | `-vec` / `/Qvec` (enabled at `O2`+); host-only |
| vec-threshold, Qvec-threshold ▸ | Loop vectorization threshold. | `100`; host-only |
| vecabi, Qvecabi ▸ | Vector function application binary interface (ABI). | `gcc` |
| version | GCC-style version information. | - |
| vmg | Pointer-to-member general representation. | - |
| vmv | Pointers to members of any inheritance type. | - |
| w | Disable all warnings. | - |
| W | Diagnostic message level. | - |
| Wa | Pass options to the assembler. | - |
| Wabi | Warn if generated code is not C++ ABI compliant. | - |
| Wall | Enable warning and error diagnostics. | - |
| Wcheck-unicode-security | Source checking for Unicode vulnerabilities. | - |
| Wcomment | Warn on `/*` inside a `/* */` comment. | - |
| Wdeprecated | Warn for deprecated C++ headers. | - |
| Werror, WX | All warnings become errors. | - |
| Werror-all | All warnings and enabled remarks become errors. | - |
| Wextra-tokens | Warn on extra tokens at the end of preprocessor directives. | - |
| Wformat | Argument checking for `printf`, `scanf`, etc. | - |
| Wformat-security | Warn on security-risky format function use. | - |
| Wl | Pass options to the linker. | - |
| Wmain | Warn if the return type of `main` is not expected. | - |
| Wmissing-declarations | Warn on global functions/variables without prior declaration. | - |
| Wmissing-prototypes | Warn on missing prototypes. | - |
| Wno-sycl-strict | Disable strict SYCL compatibility warnings. | - |
| Wp | Pass options to the preprocessor. | - |
| Wpointer-arith | Warn on questionable pointer arithmetic. | - |
| Wreorder | Warn when member initializer order differs from execution order. | - |
| Wreturn-type | Warn on missing return type, `return expr;` in a void function, or falling off a non-void function. | - |
| Wshadow | Warn when a declaration hides a previous declaration. | - |
| Wsign-compare | Warn on signed/unsigned comparison hazards. | - |
| Wstrict-aliasing | Warn on possible strict-aliasing rule violations. | - |
| Wstrict-prototypes | Warn for functions without specified argument types. | - |
| Wtrigraphs | Warn on meaning-changing trigraphs. | - |
| Wuninitialized | Warn on use before initialization. | - |
| Wunknown-pragmas | Warn on unknown `#pragma`. | - |
| Wunused-function | Warn on an unused declared function. | - |
| Wunused-variable | Warn on an unused local or non-constant static variable. | - |
| Wwrite-strings | Diagnose `const char *` → `char *` conversion. | - |
| X | Remove standard directories from the include search path. | - |
| x (type option) | Following source files recognized as a particular type. | - |
| x, Qx | Targetable processor features, instruction sets, optimizations. | - |
| xHost, QxHost | Instructions for the host's highest available instruction set. | - |
| Xlinker | Pass a linker option to the linker. | - |
| Xopenmp-target | Pass options to a specified tool in the OpenMP target device tool chain. | - |
| Xs | Pass options to the backend tool. | - |
| Xsycl-target | Pass options to a specified tool in the SYCL target device tool chain. | - |
| Y- | Ignore all other precompiled header files. | - |
| Yc | Create a precompiled header file. | - |
| Yu | Use a precompiled header file. | - |
| Zc | ANSI C conformance for certain language features. | - |
| Zg | Generate function prototypes. | **deprecated**, may be removed |
| Zi, Z7, ZI | Full debug information in an object (`.obj`) file or project database (PDB). | - |
| Zl | Omit library names from the object file. | - |
| Zp | Structure alignment on byte boundaries. | - |
## Optimization Options

`▸` entries, alphabetical, with arguments, defaults, semantics, caveats.

### fast
`-fast` / `/fast`; None; OFF. Linux sets `-ipo, -O3, -static, -fp-model fast`; Windows sets `/O3, /Qipo, /fp:fast`. Aggressive: the executable may not run on processor types other than the compile host, so understand the options it enables. Host-only.

### fbuiltin, Oi
`-fbuiltin[-name]` / `-fno-builtin[-name]` (Linux), `/Oi[-]` / `/Qno-builtin-name` (Windows); ON. `name` = one or more intrinsic functions (comma-separated); the named negative form disables expansion for those functions, and `-fno-builtin` / `/Oi-` without `name` disables it for all. Affected built-ins: gcc* docs "built-in functions" for `-fbuiltin`; Microsoft* Visual C/C++* docs "/Oi" for `/Oi`.

### foptimize-sibling-calls
`-foptimize-sibling-calls` / `-fno-optimize-sibling-calls` (Linux); Windows None; default positive. Converts tail recursion into loops: the recursive call becomes a GOTO to the function start and its value is only returned, not used in another expression, so the function becomes a loop and the stack space used is not modified.

### GF
Linux None; Windows `/GF` / `/GF-`; OFF. Read only string-pooling optimization. [sic: the source default text reads "Read/write string-pooling optimization is enabled", conflicting with OFF — unclear in source.]

### nolib-inline
`-nolib-inline` (Linux); Windows None; OFF (the compiler inlines many standard library and intrinsic functions). Disables that expansion, preventing the unexpected results it can cause. Host-only.

### O
`-O[n]` / `/O[n]`; `n` = 1, 2, or 3, and 0 on Linux*. Default `O2` (code speed); on Linux*, with `-g` the default is `-O0` unless `-O2` (or higher) is also explicitly on the command line.

| Option | Description |
|---|---|
| `O` (Linux*) | Same as `O2`. |
| `O0` (Linux) | Disables all optimizations. May set other options, determined by the compiler depending on OS and architecture; may change from release to release. |
| `O1` | Speed optimizations, disabling some that increase code size and affect speed. To limit code size it disables code-duplicating optimizations such as automatic function inlining, loop unrolling, and function cloning. May set other options per OS/architecture; may change per release. May improve performance for applications with very large code size, many branches, and execution time not dominated by code within loops. |
| `O2` | Speed optimizations; generally recommended. Vectorization is enabled at `O2` and higher. Also enables inlining of intrinsics and intra-file interprocedural optimization (inlining, constant propagation, forward substitution, routine attribute propagation, variable address-taken analysis, dead static function elimination, removal of unreferenced variables), plus for performance gain: constant propagation, copy propagation, dead-code elimination, global register allocation, global instruction scheduling and control speculation, loop unrolling, optimized code selection, partial redundancy elimination, strength reduction/induction variable simplification, variable renaming, exception handling optimizations, tail recursions, peephole optimizations, structure assignment lowering and optimizations, dead store elimination. May set other (especially code-speed) options per OS/architecture; may change per release. **This content does not apply to SYCL.** On Linux, `-debug inline-debug-info` is enabled by default when compiling with `-O2` or higher and `-g`. Many shared-library routines are more highly optimized for Intel® microprocessors than for non-Intel microprocessors. **NOTE**: on Windows, Microsoft Visual Studio C++ supports `/O2` as its highest optimization level; Intel compilers add optimizations under `/O3`. |
| `O3` | `O2` plus more aggressive loop transformations such as Fusion, Block-Unroll-and-Jam, and collapsing IF statements. May set other options per OS/architecture; may change per release. May not give higher performance unless loop and memory access transformations take place; may slow code in some cases compared to `O2`. Recommended for applications with loops that heavily use floating-point calculations and process large data sets. Shared-library routines more highly optimized for Intel® microprocessors. |

The last `O` option on the command line takes precedence. Alternate: `O0` = Linux None / Windows `/Od`.

### Od
Linux None; Windows `/Od`; OFF (the compiler performs default optimizations). Disables all optimizations; usable for selective cases such as `/Od` with `/Ob1` (all optimizations disabled, inlining enabled). Alternate: Linux `-O0`.

### Ofast
Linux `-Ofast`; Windows None; OFF. Improves application speed; provided for compatibility with gcc. Host-only.

### Os
`-Os` / `/Os`; OFF (optimizations are made for code speed), but `Os` is the default if `O1` is specified. Enables optimizations that do not increase code size; produces smaller code than `O2`; disables some size-increasing optimizations that give only a small speed benefit; favors size reduction over maximum performance.

### Ot
Linux None; Windows `/Ot`; default `/Ot` (code speed; `Od` disables all optimizations; `O1` makes `Os` the default). Enables all speed optimizations.

### Ox
Linux None; Windows `/Ox`; OFF. Maximum optimizations by combining `/Oi` and `/Ot`.

## Advanced Optimization Options

Shared link mechanics: for `qactypes`, `qdaal`, `qipp`, `qmkl`, `qmkl-ilp64` and `qtbb`, Linux requires the driver to add library names explicitly (specifying the option performs the link and pulls in dependent libraries); Windows adds directives to the compiled code that the linker reads without further driver input, so no separate link command is needed. `qmkl` and `qmkl-ilp64` must be specified at compile time on Windows.

### ffreestanding, Qfreestanding
`-ffreestanding` / `/Qfreestanding`; OFF. No standard library assumed and startup need not be at `main` (the C/C++ freestanding definition; e.g. an OS kernel); no compiler-specific libraries assumed — only calls appearing in the source code are generated. Host-only.

### fjump-tables
`-fjump-tables` / `-fno-jump-tables` (Linux); Windows None; default positive. The negative form suppresses jump tables unconditionally, independent of generated-code performance, and stops the compiler creating switch statements internally as a result of optimizations. Use it with `-fpic` where the jump table relocation cannot be resolved.

### fvec-allow-scalar-stores, Qvec-allow-scalar-stores
`-fvec-allow-scalar-stores` / `-fno-vec-allow-scalar-stores` (Linux), `/Qvec-allow-scalar-stores` / `/Qvec-allow-scalar-stores-` (Windows); default negative. Ensures vectorization of an explicit simd loop (e.g. `#pragma omp simd`) even when it stores to a scalar not marked `private`/`lastprivate`/`reduction`. OpenMP* requires those markers; violating that can generate incorrect code, especially if the scalar is reused in the loop. Use only if the loop is known safe (e.g. after an incorrect error message).

```bash
icpx -O3 -c -fvec-allow-scalar-stores simd.cpp -o simd.o   # Linux
icx /O3 /c /Qvec-allow-scalar-stores simd.cpp -o simd.o    # Windows
```

### fvec-non-loop-argument-load, Qvec-non-loop-argument-load
`-fvec-non-loop-argument-load` / `-fno-vec-non-loop-argument-load` (Linux), `/Qvec-non-loop-argument-load` / `/Qvec-non-loop-argument-load-` (Windows); default positive (loads combined when profitable). Disabled, loads from a parameter address very near the function's beginning are not combined — in rare cases avoiding processor stalls when the caller recently modified one of those locations. No effect on the loop vectorizer.

```bash
icpx -O2 -c -fno-vec-non-loop-argument-load test.cpp -o test.o  # Linux
icx -O2 -c /Qvec-non-loop-argument-load- test.cpp -o test.o    # Windows
```

### fvec-peel-loops, Qvec-peel-loops
`-fvec-peel-loops` / `-fno-vec-peel-loops` (Linux), `/Qvec-peel-loops` / `/Qvec-peel-loops-` (Windows); off. Vectorizes peeling loops created during loop vectorization (a peel loop improves memory-reference alignment in the main vectorized loop). Needs masked mode (`-fvec-with-mask` / `/Qvec-with-mask`); cannot be enforced — the cost model decides.

### fvec-remainder-loops, Qvec-remainder-loops
`-fvec-remainder-loops` / `-fno-vec-remainder-loops` (Linux), `/Qvec-remainder-loops` / `/Qvec-remainder-loops-` (Windows); off. Vectorizes the remainder loop created for the vectorized main loop; the cost model picks vector factor and mode. Enforce with `#pragma vector vecremainder` on the loop.

### fvec-with-mask, Qvec-with-mask
`-fvec-with-mask` / `-fno-vec-with-mask` (Linux), `/Qvec-with-mask` / `/Qvec-with-mask-` (Windows); off. Vectorization mode for loops with small compile-time-known iteration counts (peel and remainder loops also qualify). Vector factor = lowest power of two greater than the known (maximum) iteration count; usually one iteration with most operations masked.

### ipp-link, Qipp-link
`-ipp-link[=lib]` / `/Qipp-link[:lib]`; `lib` = `static` or `dynamic` threaded Intel® IPP runtime libraries; default `dynamic` (static if Linux `-static`). Requires `[Q]ipp`. Host-only.

### mno-gather, Qgather-
`-mno-gather` / `/Qgather-`; OFF. Disables gather instructions in auto-vectorization. Host-only.

```bash
icx -c -mno-gather t.c                       # Linux
icpx -c -mno-gather -mno-scatter t.cpp       # Linux
icx /c /Qgather- t.c                         # Windows
icx /c /Qgather- /Qscatter- t.cpp            # Windows
```

### mno-scatter, Qscatter-
`-mno-scatter` / `/Qscatter-`; OFF. Disables scatter instructions in auto-vectorization. Host-only.

```bash
icx -c -mno-gather -mno-scatter t.cpp        # Linux
icx /c /Qgather- /Qscatter- t.cpp            # Windows
```

### qactypes, Qactypes
`-qactypes` / `-qno-actypes` (Linux), `/Qactypes` / `/Qactypes-` (Windows); OFF (no Algorithmic C (AC) header search or library link, so AC data types are unusable). Adds the AC data type folder to header searches and links AC data types libraries for CPU compilations. AC types (arbitrary precision integers, fixed precision integers, arbitrary precision floating point) are built on `_ExtInt`. Dynamic linking only — static AC library linking is impossible. May impact target compilations.

```bash
icpx -qactypes file.cpp       # Linux
icx /Qactypes file.cpp        # Windows
```

### qdaal, Qdaal
`-qdaal[=lib]` / `/Qdaal[:lib]`; `lib` = `parallel` (threaded; default if no `lib`) or `sequential` (non-threaded) Intel® oneAPI Data Analytics Library (oneDAL) libraries; OFF. Linux includes the associated oneDAL headers. Alternate: Linux `-daal` (deprecated).

### qipp, Qipp
`-qipp[=lib]` / `/Qipp[:lib]`; `lib` = `common` (main libraries set; default), `crypto` (Intel® Cryptography Primitives Library), `nonpic` (Linux* only; no position-independent code), `nonpic_crypto` (Linux only; cryptography library without PIC); OFF. Links some or all Intel® Integrated Performance Primitives (Intel® IPP) libraries and their headers; `[Q]ipp-link` picks static/dynamic threaded runtimes.

### qmkl, Qmkl
`-qmkl[=lib]` / `/Qmkl[:lib]`; `lib` = `parallel` (threaded; default), `sequential`, `cluster` (cluster-specific plus sequential) oneMKL libraries; OFF. Linux dynamic default (static: `-qmkl -static-intel`); Windows static default (dynamic: `/Qmkl /MD`). With `-qmkl` and `-qmkl-ilp64`, the rightmost wins. `[q or Q]mkl` + `-fsycl` links the combined oneMKL* SYCL library (specific library: also `[q or Q]mkl-sycl-impl`). `[q or Q]mkl` or `=parallel` + `[Q]tbb` → standard threaded oneMKL; adding `[q or Q]openmp` → OpenMP* threaded.

### qmkl-ilp64, Qmkl-ilp64
`-qmkl-ilp64[=lib]` / `/Qmkl-ilp64[:lib]`; `lib` and defaults as `qmkl`; OFF. Rightmost of `-qmkl-ilp64` / `-qmkl` wins. Linux dynamic default (static: `-qmkl-ilp64 -static-intel`); Windows static default (dynamic: `/Qmkl-ilp64 /MD`).

### qmkl-sycl-impl, Qmkl-sycl-impl
`-qmkl-sycl-impl=arg[, arg,...]` / `/Qmkl-sycl-impl:arg[, arg,...]`; `arg` = `blas` (BLAS), `dft` (Discrete Fourier Transform (DFT)), `lapack` (LAPACK), `rng` (Random Number Generator (RNG)), `sparse` (Sparse BLAS), `stats` (Summary Statistic), `vm` (Vector Mathematics (VM)) — each the corresponding oneMKL SYCL library; OFF (must be specified to link a specific oneMKL SYCL library). Not supported for static linking; requires `-fsycl` + `-qmkl` (`/Qmkl`) or a diagnostic warning appears. Available SYCL drivers: see Invoke the Compiler.

```bash
icpx -fsycl -qmkl -qmkl-sycl-impl=blas file.cpp    # Linux
icx /fsycl /Qmkl /Qmkl-sycl-impl:blas file.cpp     # Windows
# each of these emits a diagnostic warning on Linux:
icpx -qmkl -qmkl-sycl-impl=blas file.cpp
icpx -fsycl -qmkl-sycl-impl=blas file.cpp
```

### qopt-assume-no-loop-carried-dep, Qopt-assume-no-loop-carried-dep
`-qopt-assume-no-loop-carried-dep[=n]` / `/Qopt-assume-no-loop-carried-dep[=n]`; `n` = `0` (no assumption; default if omitted), `1` (innermost loops; default when the option is used without `n`), `2` (all loop levels); default `=0`. For C/C++ code where pointers and arguments could alias: levels 1-2 vectorize more loops or enable loop transformations. Applied to all loops in the file, not to code outside loops.

The first loop is not vectorized because of data dependency; level 1 assumes no data dependence occurs and allows vectorization:

```c
void sub (float *A, float *B, int* M ) {
 for (int i =0; i< 10000 ; i++) {
     A[i] += B[M[i]] + 1;
   }
 }
```

The matrix multiply kernel is not optimized because of dependency in all loop nests; level 2 gives transformations such as blocking, unroll and jam, and vectorization:

```c
void matmul(double *a, double *b, double *c) {
 int i, j, k;
 int n = 1024;
 for (i = 0; i < 1024; i++) {
    for (j = 0; j < 1024; j++) {
       for (k = 0; k < 1024; k++) {
         c[i * n + j] += a[i * n + k] * b[k * n + j];
       }
      }
    }
 }
```

### qopt-dword-index-for-array-of-structs, Qopt-dword-index-for-array-of-structs
`-qopt-dword-index-for-array-of-structs[=val]` / `/Qopt-dword-index-for-array-of-structs[:value]`; `val` = 16 or 32, defaulting to 16 bytes when omitted; OFF (dword indexes only when the compiler can determine it is safe). Relevant arrays should contain no more than `INT_MAX / sizeof(element)` elements. Example: `icx -qopt-dword-index-for-array-of-structs t.c`.

### qopt-dynamic-align, Qopt-dynamic-align
`-qopt-dynamic-align` / `-qno-opt-dynamic-align` (Linux), `/Qopt-dynamic-align` / `/Qopt-dynamic-align-` (Windows); default negative (no alignment-dependent code: no data-location-based optimizations, results depend on the data values). Positive: conditional optimizations on dynamic input alignment, which may give different bitwise results for aligned vs unaligned data with the same values; speeds some vectorized code (especially long trip counts) at the cost of code size and compile time, while disabling improves bitwise reproducibility.

### qopt-for-throughput, Qopt-for-throughput
`-qopt-for-throughput=value` / `/Qopt-for-throughput:value`; `value` = `"multi-job"` or `"single-job"`; OFF. Single- vs multi-job memory tuning differs (e.g. the loop-tiling and prefetching cost models; more memory is available to a single job).

### qopt-mem-layout-trans, Qopt-mem-layout-trans
`-qopt-mem-layout-trans[=n]` / `-qno-opt-mem-layout-trans` (Linux), `/Qopt-mem-layout-trans[:n]` / `/Qopt-mem-layout-trans-` (Windows); `n` = `0` none (same as the negative forms), `1` basic, `2` more (same as the option with no argument), `3` more, including copy-in/copy-out of structures for a code region — only for systems with more than 4GB of physical memory per core, `4` more aggressive — only for systems with more than 4GB per core; default `=0`. Can improve cache reuse and locality.

### qopt-multiple-gather-scatter-by-shuffles, Qopt-multiple-gather-scatter-by-shuffles
`-qopt-multiple-gather-scatter-by-shuffles` / `-qno-opt-multiple-gather-scatter-by-shuffles` (Linux), `/Qopt-multiple-gather-scatter-by-shuffles` / `/Qopt-multiple-gather-scatter-by-shuffles-` (Windows); default varies (heuristics). A tuning hint for multiple adjacent gather/scatter type vector memory references, generating better shuffle-based sequences. Affected by `[Q]x`, `-march` (Linux*), `/arch` (Windows*).

### qopt-prefetch, Qopt-prefetch
`-qopt-prefetch[=n]` / `-qno-opt-prefetch` (Linux), `/Qopt-prefetch[:n]` / `/Qopt-prefetch-` (Windows); `n` = `0` disable (same as the negative form) or `1`-`5` levels; omitted `n` gives `-qopt-prefetch=2` / `/Qopt-prefetch:2`; lower values prefetch less. Default varies: disabled with the negative form or `O0`/`O1` (explicit or implicit); `=2` at `O2` and above. Reduces cache misses by hinting when data should be loaded into cache; enabled at higher optimization levels.

### qopt-prefetch-distance, Qopt-prefetch-distance
`-qopt-prefetch-distance=n` / `/Qopt-prefetch-distance:n`; `n` = distance in (possibly vectorized) iterations, non-negative `>=0`; `n = 0` turns off all compiler-issued prefetches from memory to L1; OFF (heuristics). If the loop is vectorized, the unit is vectorized iterations. Applies to memory-to-L1 prefetches (for example the `vprefetch0` instruction). Ignored with `-qopt-prefetch=0`/`-qno-opt-prefetch` (Linux) or `/Qopt-prefetch=0`/`/Qopt-prefetch-` (Windows).

```text
-qopt-prefetch-distance=24   # 24 iterations for memory-to-L1 prefetches
-qopt-prefetch-distance=0    # turn off all compiler-inserted memory-to-L1 prefetches in loops
-qopt-prefetch-distance=16   # 16 iterations for memory-to-L1 prefetches
```

### qopt-prefetch-loads-only, Qopt-prefetch-loads-only
`-qopt-prefetch-loads-only` / `/Qopt-prefetch-loads-only`; OFF (both loads and stores are prefetched). Prefetch loads only, ignoring stores. Ignored with prefetching off as above. Host-only.

### qopt-streaming-stores, Qopt-streaming-stores
`-qopt-streaming-stores=keyword` / `-qno-opt-streaming-stores` (Linux), `/Qopt-streaming-stores:keyword` / `/Qopt-streaming-stores-` (Windows); `keyword` = `always` (assumes a memory-bound application; you must insert memory barriers/fences for correct ordering within and across threads), `never` (normal stores; same as the negative forms), `auto` (the compiler decides); default `=auto`. Uses non-temporal-buffer instructions, minimizing memory hierarchy pollution. Example fences a streaming-store loop with a `_mm_sfence()` call just after it:

```c
void simple1(double * restrict a, double * restrict b, double * restrict c, double *d, int n)
{
    int i, j;

#pragma omp parallel for
      for (j=0; j<n; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
        }

      _mm_sfence(); // OR _mm_mfence();

#pragma omp parallel for
    for (i=0; i<n; i++)
        a[i] = a[i] + c[i]*b[i];
}
```

### qopt-zmm-usage, Qopt-zmm-usage
`-qopt-zmm-usage=keyword` / `/Qopt-zmm-usage:keyword`; `keyword` = `low` (avoid zmm unless the gain is provable) or `high` (unrestricted zmm code); default varies: `low` with `[Q]xCORE-AVX512`, `high` with `[Q]xCOMMON-AVX512`. Targets Intel® processors on the Intel® microarchitecture formerly code-named Skylake. Neither is predictably better (xmm/ymm-heavy code may improve with zmm; some zmm code gains little or loses) — try both. Ignored unless an Intel® AVX-512-enabling option such as `[Q]xCORE-AVX512` or `[Q]xCOMMON-AVX512` is specified. No effect on `pragma omp simd simdlen(n)` loops or CORE-AVX512-specific vector-spec functions.

### qtbb, Qtbb
`-qtbb` / `/Qtbb`; OFF. Links Intel® oneAPI Threading Building Blocks (oneTBB) libraries and headers. Alternate: Linux `-tbb` (deprecated).

### unroll, Qunroll
`-unroll[=n]` / `/Qunroll[:n]`; `n` = maximum unroll count, `0` disables loop enrolling [sic: source wording, means unrolling]. Default `-unroll` / `/Qunroll` (heuristics); omitted `n` lets the optimizer decide. Alternate: Linux `-funroll-loops`.

### use-intel-optimized-headers, Quse-intel-optimized-headers
`-use-intel-optimized-headers` / `/Quse-intel-optimized-headers`; default `-no-use-intel-optimized-headers` / `/Quse-intel-optimized-headers-`. Adds the performance headers directory to the include path search list; appropriate libraries are linked in as needed.

### vec, Qvec
`-vec` / `-no-vec` (Linux), `/Qvec` / `/Qvec-` (Windows); default `-vec` / `/Qvec` (enabled if `O2` or higher is in effect). Enables vectorization at default optimization levels for Intel® and non-Intel microprocessors; vectorization may call library routines giving additional gain on Intel microprocessors. Host-only.

### vec-assume-index-overflow, Qvec-assume-index-overflow
`-vec-assume-index-overflow` / `-no-vec-assume-index-overflow` (Linux), `/Qvec-assume-index-overflow` / `/Qvec-assume-index-overflow-` (Windows); default less conservative mode (the option toggles between modes). Set, assumes integer-index overflow may occur, disabling additional load/store optimization.

```bash
icpx -O2 -c -fno-vec-assume-index-overflow test.cpp -o test.o   # Linux
icpx -O2 -c -fvec-assume-index-overflow test.cpp -o test.o      # Linux
```

### vec-threshold, Qvec-threshold
`-vec-threshold[n]` / `/Qvec-threshold[[:]n]`; `n` = 0 through 100: `0` always vectorize regardless of computation work volume; `100` only when profitable vector-level parallel execution is almost certain; 1-99 = percentage probability of profitable speed-up (e.g. `n=50` → only at 50% probability); default `-vec-threshold100` / `/Qvec-threshold100`, also the default if `n` is unspecified. For loops whose work volume is unknown at compile time (usually an unknown trip count); the heuristic balances thread-creation overhead against available work. Host-only.

### vecabi, Qvecabi
`-vecabi=keyword` / `/Qvecabi:keyword`; `keyword` = `cmdtarget` (extended set of vector functions, variants for all targets from `[Q]x` and/or `[Q]ax`, no source change) or `gcc` (gcc vector function ABI); default `gcc`. All files defining or using vector functions must use `-vecabi=cmdtarget` (`/Qvecabi:cmdtarget`) identically, with identical `[Q]x`/`[Q]ax` values, or link-time/runtime errors occur; link errors also occur against modules with vector function definitions that cannot be recompiled. With `cmdtarget`, versions are created by copying each vector specification and changing the target processor; the count follows `[Q]x`/`[Q]ax`.

## Code examples

Every documented example in this range (also shown inline above), grouped for grep-ability.

```bash
icpx -O3 -c -fvec-allow-scalar-stores simd.cpp -o simd.o     # Linux
icx /O3 /c /Qvec-allow-scalar-stores simd.cpp -o simd.o      # Windows
icpx -O2 -c -fno-vec-non-loop-argument-load test.cpp -o test.o   # Linux
icx -O2 -c /Qvec-non-loop-argument-load- test.cpp -o test.o      # Windows
icx -c -mno-gather t.c                                       # Linux
icpx -c -mno-gather -mno-scatter t.cpp                       # Linux
icx /c /Qgather- t.c                                         # Windows
icx /c /Qgather- /Qscatter- t.cpp                            # Windows
icx -c -mno-gather -mno-scatter t.cpp                        # Linux
icx /c /Qgather- /Qscatter- t.cpp                            # Windows
icpx -qactypes file.cpp                                      # Linux
icx /Qactypes file.cpp                                       # Windows
icpx -fsycl -qmkl -qmkl-sycl-impl=blas file.cpp              # Linux
icx /fsycl /Qmkl /Qmkl-sycl-impl:blas file.cpp               # Windows
icpx -qmkl -qmkl-sycl-impl=blas file.cpp                     # Linux: diagnostic warning (no -fsycl)
icpx -fsycl -qmkl-sycl-impl=blas file.cpp                    # Linux: diagnostic warning (no -qmkl)
icx -qopt-dword-index-for-array-of-structs t.c
-qopt-prefetch-distance=24
-qopt-prefetch-distance=0
-qopt-prefetch-distance=16
icpx -O2 -c -fno-vec-assume-index-overflow test.cpp -o test.o    # Linux
icpx -O2 -c -fvec-assume-index-overflow test.cpp -o test.o       # Linux
ifx /warn:usage,declarations test.f90                        # comma-separated keywords are valid
```

C/C++ examples: `sub(float *A, float *B, int *M)` and `matmul(double *a, double *b, double *c)` for `[q or Q]opt-assume-no-loop-carried-dep` levels 1 and 2, and `simple1(double * restrict a, double * restrict b, double * restrict c, double *d, int n)` for `[q or Q]opt-streaming-stores=always` with `_mm_sfence(); // OR _mm_mfence();` — full source in the entries above.

## Gotchas & failure modes

- **`-O0` default with `-g` on Linux** unless `-O2`+ is explicitly on the command line; an implied `-O2` is not enough.
- **`/Od`, `/O1`, `/O2`, `/O3`, `/Ox` are mutually exclusive** (last wins); `/Od` last disables optimization options, and `/Od` + `/Ob1` = no optimization but inlining.
- **`-fast` portability**: sets `-ipo -O3 -static -fp-model fast` (so it also switches the FP model away from the Intel/Clang documented defaults) and the executable may not run on processor types other than the compile host.
- **Host-only options do nothing for device code** under offloading: `fast`, `Ofast`, `nolib-inline`, `ffreestanding`, `ipp-link`, `mno-gather`, `mno-scatter`, `qopt-prefetch-loads-only`, `vec`, `vec-threshold`.
- **`-fopenmp` ≠ `-fiopenmp`/`-qopenmp`**: front-end (LLVM community) lowering, less performant than `-fiopenmp`'s backend lowering, and **no GPU offloading**.
- **`qopt-streaming-stores=always` shifts memory-ordering duty to you**: insert barriers/fences (e.g. `_mm_sfence()` or `_mm_mfence()`).
- **`fvec-allow-scalar-stores` can generate incorrect code** if a scalar stored in an explicit simd loop is not `private`/`lastprivate`/`reduction` and is reused in the loop; use only when the loop is safe (e.g. after an incorrect error message).
- **Peel/remainder vectorization cannot be forced** (cost model); `-fvec-peel-loops` also needs masked mode (`-fvec-with-mask`). Only remainder vectorization can be enforced, via `#pragma vector vecremainder`.
- **`-fno-jump-tables` + `-fpic`** for objects loaded where the jump table relocation cannot be resolved; it also stops the compiler creating switch statements internally during optimization.
- **Prefetch options are silently ignored** when prefetching is off (`-qopt-prefetch=0` / `-qno-opt-prefetch`; Windows `/Qopt-prefetch=0` / `/Qopt-prefetch-`): `-qopt-prefetch-distance`, `-qopt-prefetch-loads-only`. `-qopt-prefetch` itself is disabled at `O0`/`O1` and level 2 at `O2`+.
- **`-qopt-zmm-usage` is ignored without an Intel® AVX-512 option** (`[Q]xCORE-AVX512`, `[Q]xCOMMON-AVX512`), which also sets its `low`/`high` default; no effect on `pragma omp simd simdlen(n)` loops or CORE-AVX512-specific vector-spec functions.
- **`-vecabi=cmdtarget` consistency**: all files defining/using vector functions need identical use and identical `[Q]x`/`[Q]ax` values, or link-time/runtime errors occur; linking against non-recompilable vector-function modules can also fail to link. Default `gcc`.
- **`-qopt-dynamic-align` reproducibility**: enabled dynamic alignment may give different bitwise results for aligned vs unaligned data with the same values; the default is disabled.
- **`-qopt-mem-layout-trans` levels 3 and 4** only for systems with more than 4GB of physical memory per core; **`-qopt-dword-index-for-array-of-structs`** allows at most `INT_MAX / sizeof(element)` elements and defaults `val` to 16 bytes; **`-qopt-assume-no-loop-carried-dep`** applies file-wide and assumes no loop-carried dependencies where they may exist.
- **oneMKL link asymmetry**: Linux dynamic by default (`-qmkl`; static needs `-qmkl -static-intel`; `-qmkl-ilp64` likewise), Windows static by default (`/Qmkl`; dynamic needs `/Qmkl /MD`; `/Qmkl-ilp64` likewise); with both `-qmkl` and `-qmkl-ilp64` the rightmost wins; on Windows oneMKL options must be at compile time; Windows adds linker directives while Linux requires explicit library names.
- **oneMKL SYCL**: `[q or Q]mkl` + `-fsycl` links the combined oneMKL* SYCL library; a specific library needs `[q or Q]mkl-sycl-impl`, unsupported for static linking and warning if `-fsycl` or `[q or Q]mkl` is missing.
- **oneMKL threading**: `[q or Q]mkl` (or `=parallel`) + `[Q]tbb` → standard threaded oneMKL; adding `[q or Q]openmp` → OpenMP* threaded oneMKL.
- **Performance-library companions**: `[Q]ipp-link` needs `[Q]ipp`; `qactypes` cannot link statically (dynamic default) and on Linux `-qactypes` must be used at link time; `-qactypes` may impact target compilations.
- **Deprecated options that will break future builds**: `fsycl-allow-device-dependencies` (use `fsycl-allow-device-image-dependencies`), `fsycl-explicit-simd`, `m64`/`Qm64`, `TP`, `Zg`; deprecated alternates `-daal` (use `-qdaal`) and `-tbb` (use `-qtbb`).
- **`-fpie` output** can only be linked into executables; **`-unroll=0`** disables unrolling (an omitted `n` leaves it to heuristics); **`-vec-threshold`** `0` always vectorizes and `100` (default) only when almost certainly profitable.
- **`-O2` docs note "This content does not apply to SYCL"** for its optimization list; on Linux `-debug inline-debug-info` is auto-enabled with `-O2`+ and `-g`.

## Source map

- `__regcall` register/stack placement and preserved registers — p. 52
- Compiler Options intro, Intel-vs-Clang defaults, MSVC `/` syntax note — pp. 52–53
- Conventions Used for Compiler Options — pp. 53–54
- Alphabetical Option List — pp. 54–68
- General Rules for Compiler Options — pp. 68–69
- How We Refer to Compiler Option Names in Descriptions — p. 69 (same shortcuts as Conventions)
- What Appears in the Compiler Option Descriptions — p. 70
- Optimization Options (`fast`, `fbuiltin`/`Oi`, `foptimize-sibling-calls`, `GF`, `nolib-inline`, `O`, `Od`, `Ofast`, `Os`, `Ot`, `Ox`) — pp. 70–80
- Advanced Optimization Options (`ffreestanding`/`Qfreestanding` … `vecabi`/`Qvecabi`) — pp. 81–114
- Code examples — pp. 83, 84, 88, 89, 90, 96, 97, 98–99, 105, 107–108, 113