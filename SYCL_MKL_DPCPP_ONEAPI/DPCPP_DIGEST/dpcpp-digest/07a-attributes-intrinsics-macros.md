---
chunk: 07a-attributes-intrinsics-macros
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 398-594
covers: Attributes (align, align_value, allow_cpu_features, code_align, const, cpu_dispatch/cpu_specific, target); Intrinsics; the Libraries section (creating/using/redistributing libraries, redistributable-library table, libqkmalloc, SIMD Data Layout Templates, Intel C++ Class Libraries, Asynchronous I/O, IEEE 754 library, Numeric String Conversion); predefined Macros
---

# Attributes, Intrinsics, Libraries, and Macros (Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference)

> **Scope.** How to attach attributes to C/C++ declarations (`align`, `align_value`, `allow_cpu_features`, `code_align`, `const`, `cpu_dispatch`/`cpu_specific`, `target`), when and how intrinsics may be used, how the compiler's libraries are created, linked, managed and redistributed (including the library reference sections that fall in this page range), and the full set of predefined macros. A reader can use this chunk to answer "which attribute syntax and values are legal on Linux vs Windows", "which library is pulled in by an option", "which predefined macro identifies the compiler", and "what the default is".

## Key facts

- Attributes may be written three ways: GNU `__attribute__((...))`, Microsoft `__declspec(...)`, and C++11 standardized `[[...]]`. The C2x attribute syntax is consistent with the C++11 standard.
- Some attributes are available for both Intel® and non-Intel microprocessors but may perform additional optimizations for Intel microprocessors.
- Intrinsics are assembly-coded functions; with this compiler they **can be used only on the host**, are expanded inline, and require the `immintrin.h` header plus `__attribute__((target(<required target>)))` on the functions that use them.
- `cpu_dispatch`/`cpu_specific` manual processor dispatch is for Intel processors based on Intel® 64 architecture; on an unsupported non-Intel processor you get an `"invalid option"` error at compile time.
- The libraries the compiler links depend on the options used (`-shared-intel`/`-static-intel`, `-qopenmp`/`/Qopenmp`, `-fsycl`, `-qmkl=...`, `-qtbb`, `-qipp`, etc.); see the redistributable-library table.
- This version of the compiler uses a close approximation to IEEE Standard for Floating-point Arithmetic **IEEE 754-2008** unless otherwise stated.
- The compiler supports the ISO Standard predefined macros and additional predefined macros; **single capital letter macros are not supported**.
- Default `__cplusplus` is `201703L`; `_OPENMP` default is `202011` when `[q or Q]openmp` is specified; `__INTEL_LLVM_COMPILER` uses form `VVVVMMUU` and 2023.1 is `20230100`.
- SDLT documentation in this range is for **SDLT version 2** (C++11 template library, adds n-dimensional containers over version 1).
- Platform differences abound: many macros and options exist only on Linux or only on Windows; `__LONG_DOUBLE_SIZE__` is 80 on Linux and 64 on Windows (80 with `Qlong-double`).

## Attribute / API quick table

| Attribute | Purpose | Syntax (Windows / Linux) | Key values / defaults |
|---|---|---|---|
| `align` | Align a variable to an n-byte boundary | `__declspec(align(n))` / `__attribute__((aligned(n)))` | `n` = alignment in bytes |
| `align_value` | Pointer alignment value on a pointer typedef | `__declspec(align_value(alignment))` / `__attribute__((align_value(alignment)))` | `alignment` = 8, 16, 32, 64, 128, 256, … |
| `allow_cpu_features` | Permit intrinsics / arch-specific functionality in a function | `__declspec(allow_cpu_features(featp1[,featp2]))` / `__attribute__((allow_cpu_features(featp1[,featp2])))` | `featp1` = page-one CPUID bitmask (unsigned 64-bit), `featp2` optional page-two bitmask; use `0` for `featp1` if only page-two features wanted |
| `code_align` | Byte alignment for a loop | `[[clang::code_align(n)]]` / `__attribute__((code_align(n)))` or `[[clang::code_align(n)]]` | `n` optional, power of 2 between 1 and 4096; `1` = no alignment; default 16 bytes |
| `const` | Function depends only on its arguments and returns a value | `__declspec(const)` / `__attribute__((const))` | Equivalent to the gcc\* attribute `const`; function declarations only |
| `cpu_dispatch`, `cpu_specific` | Multiple function versions targeted at processor lists / one processor type | `__declspec(cpu_dispatch(cpuid, cpuid, ...))`, `__declspec(cpu_specific(cpuid))` / `__attribute__((cpu_dispatch(cpuid, cpuid, ...)))`, `__attribute__((cpu_specific(cpuid)))` | Long `cpuid` list below; Intel® 64-based Intel processors only |
| `target` | Make a called function or variable available on a target | `__attribute__((target(target-name)))` / `__attribute__((target(target-name)))` | `arch=skylake-avx512`, `arch=corei7`, `arch=core2`, `arch=atom`, `mmx`, `sse`, `sse2`, `sse3`, `ssse3`, `sse4.1`, `sse4.2`, `popcnt`, `aes`, `pclmul`, `avx`, `avx2`, `avx512f` |

---

## IEEE Floating-Point Operations (p. 398 continuation)

**IEEE 754-2008.** This version of the compiler uses a close approximation to the IEEE Standard for Floating-point Arithmetic, version IEEE 754-2008, unless otherwise stated. This standard is common to many microcomputer-based systems due to fast processors that implement the required characteristics. Except as noted, the description refers to both the standard and the compiler implementation.

**Special Values** supported by the Intel® oneAPI DPC++/C++ Compiler:

- **Signed Zero**: the sign of zero is the same as the sign of a nonzero number. Comparisons consider `+0` equal to `-0`. Useful in some numerical analysis algorithms; in most applications the sign of zero is invisible.
- **Denormalized Numbers** (denormals): fill the gap between the smallest positive and the smallest negative normalized number (otherwise only (+/-) 0 occurs in that interval). Extend the range of computable results by allowing gradual underflow. The Underflow status flag is set when a number loses precision and becomes a denormal.
- **Signed Infinity**: result of arithmetic in the limiting case of operands with arbitrarily large magnitude; provide a way to continue when an overflow occurs. The sign of an infinity is the sign obtained for a finite number in the same operation as the finite number approaches an infinite value. By retrieving the status flags you can differentiate an infinity resulting from an overflow from one resulting from division by zero. The compiler treats infinity as **signed by default**; the output value is `+Infinity` or `-Infinity`.
- **Not a Number (NaN)**: may result from an invalid operation, for example `0/0` and `SQRT(-1)`. In general an operation involving a NaN produces another NaN. Because the fraction of a NaN is unspecified there are many possible NaNs. The compiler treats all NaNs identically, but there are two classes:
  - **Signaling NaNs**: initial mantissa bit of 0; usually raise an invalid exception when used in an operation.
  - **Quiet NaNs**: initial mantissa bit of 1.
  - Floating-point hardware usually converts a signaling NaN into a quiet NaN during computational operations; an invalid exception is raised and the resulting value is a quiet NaN.

**Caveat:** when `fp-model fast` is used (**the default**), the compiler assumes no signed zeros, no infinite values, no NaN values, and denormal values are flushed to zero.

See also: Programming Guidelines for Vectorization; Setting the FTZ and DAZ Flags; Intel® 64 Software Developer's Manual, Volume 1: Basic Architecture.

## Attributes

Attributes are a way to provide additional information to the compiler.

### Use Attributes

The compiler supports three ways to add attributes to a program:

```c
// GNU Syntax
__attribute__((attribute_name(arguments)))

// Microsoft Syntax
__declspec(attribute_name(argument))

// C++11 Standardized Attribute Syntax (part of the C++11 language standard)
[[attribute_name(arguments)]]
[[attribute-namespace :: attribute_name(arguments)]]
```

Some attributes are available for both Intel® microprocessors and non-Intel microprocessors but they may perform additional optimizations for Intel® microprocessors than they perform for non-Intel microprocessors.

### align

Directs the compiler to align the variable to a specified boundary and a specified offset.

**Syntax**

```c
// Windows
__declspec(align(n))
// Linux
__attribute__((aligned(n)))
```

For portability on Linux OS you should use the syntax form `__attribute__((aligned(n)))`; this form is compatible with the GNU compiler.

**Arguments**

| Argument | Meaning |
|---|---|
| `n` | Specifies the alignment. The compiler will align the variable to an `n`-byte boundary. |

**Description:** this keyword directs the compiler to align the variable to an `n`-byte boundary.

### align_value

Provides the ability to add a pointer alignment value to a pointer typedef declaration.

**Syntax**

```c
// Windows:
__declspec(align_value(alignment))
// Linux:
__attribute__((align_value(alignment)))
```

**Arguments**

| Argument | Meaning |
|---|---|
| `alignment` | Specifies the alignment (8, 16, 32, 64, 128, 256,...) for what the pointer points to. |

**Description:** this keyword can be added to a pointer typedef declaration to specify the alignment value of pointers declared for that pointer type. It tells the compiler that the data referenced by the designated pointer is aligned by the indicated value, and the compiler can generate code based on that assumption.

**Caveat:** if this attribute is used incorrectly and the data is not aligned to the designated value, the behavior is **undefined**.

### allow_cpu_features

Provides the ability for a function to use intrinsic functions and architecture-specific functionality.

**Syntax**

```c
// Windows:
__declspec(allow_cpu_features(featp1[,featp2]))
// Linux:
__attribute__((allow_cpu_features(featp1[,featp2])))
```

**Arguments**

| Argument | Meaning |
|---|---|
| `featp1` | Specifies features to allow for the function. Values are integral constant expressions that evaluate to the page one bitmask of permissible features from the libirc CPUID information. The evaluated type is an `unsigned 64-bit integer`, which permits use of template-dependent code. |
| `featp2` | Optional. Specifies features to allow for the function. Values are integral constant expressions that evaluate to the page two bitmask of permissible features from the libirc CPUID information. The evaluated type is an `unsigned 64-bit integer`, which permits use of template-dependent code. If only features from page two are desired, specify `0` for `featp1`. |

**Possible `featp1` (page one) values:**

`_FEATURE_GENERIC_IA32`, `_FEATURE_FPU`, `_FEATURE_CMOV`, `_FEATURE_MMX`, `_FEATURE_FXSAVE`, `_FEATURE_SSE`, `_FEATURE_SSE2`, `_FEATURE_SSE3`, `_FEATURE_SSSE3`, `_FEATURE_SSE4_1`, `_FEATURE_SSE4_2`, `_FEATURE_MOVBE`, `_FEATURE_POPCNT`, `_FEATURE_PCLMULQDQ`, `_FEATURE_AES`, `_FEATURE_F16C`, `_FEATURE_AVX`, `_FEATURE_RDRND`, `_FEATURE_FMA`, `_FEATURE_BMI`, `_FEATURE_LZCNT`, `_FEATURE_HLE`, `_FEATURE_RTM`, `_FEATURE_AVX2`, `_FEATURE_AVX512DQ`, `_FEATURE_PTWRITE`, `_FEATURE_AVX512F`, `_FEATURE_ADX`, `_FEATURE_RDSEED`, `_FEATURE_AVX512IFMA52`, `_FEATURE_AVX512ER`, `_FEATURE_AVX512PF`, `_FEATURE_AVX512CD`, `_FEATURE_SHA`, `_FEATURE_MPX`, `_FEATURE_AVX512BW`, `_FEATURE_AVX512VL`, `_FEATURE_AVX512VBMI`, `_FEATURE_AVX512_4FMAPS`, `_FEATURE_AVX512_4VNNIW`, `_FEATURE_AVX512_VPOPCNTDQ`, `_FEATURE_AVX512_BITALG`, `_FEATURE_AVX512_VBMI2`, `_FEATURE_GFNI`, `_FEATURE_VAES`, `_FEATURE_VPCLMULQDQ`, `_FEATURE_AVX512_VNNI`, `_FEATURE_CLWB`, `_FEATURE_RDPID`, `_FEATURE_IBT`, `_FEATURE_SHSTK`, `_FEATURE_SGX`, `_FEATURE_WBNOINVD`, `_FEATURE_PCONFIG`, `_FEATURE_AXV512_VP2INTERSECT` [sic: source spells `AXV512`, not `AVX512`]

**Possible `featp2` (page two) values:**

`_FEATURE_CLDEMOTE`, `_FEATURE_MOVDIRI`, `_FEATURE_MOVDIR64B`, `_FEATURE_WAITPKG`, `_FEATURE_AVX512_Bf16`, `_FEATURE_ENQCMD`, `_FEATURE_AVX_VNNI`, `_FEATURE_AMX_TILE`, `_FEATURE_AMX_INT8`, `_FEATURE_AMX_BF16`, `_FEATURE_KL`, `_FEATURE_WIDE_KL`

**Description:** when added to a function declaration, this keyword permits the use of intrinsic functions and other architecture-specific functionality that require the listed processor features. The function is generated as if the specified features are available.

### code_align

Specifies the byte alignment for a loop.

**Syntax**

```c
// Windows
[[clang::code_align(n)]]
// Linux
__attribute__((code_align(n)))
// or
[[clang::code_align(n)]]
```

**Arguments**

| Argument | Meaning |
|---|---|
| `n` | Optional. A positive integer constant initialization expression indicating the number of bytes for the minimum desired alignment boundary. Its value must be a power of 2, between 1 and 4096, such as 1, 2, 4, 8, and so on. If you specify `1` for `n`, no alignment is performed. If you do not specify `n`, the default alignment is 16 bytes. |

**Description:** this attribute must precede the loop to be aligned. If a procedure has the `code_align(k)` attribute and another `code_align(n)` attribute precedes a loop, then both the procedure and the loop are aligned on a `max(n,k)` byte boundary.

### const

Indicates that a function has no effect other than returning a value and that it uses only its arguments to generate that return value.

**Syntax**

```c
// Windows:
__declspec(const)
// Linux:
__attribute__((const))
```

**Arguments:** None

**Description:** this keyword is equivalent to the gcc\* attribute `const` and applies to function declarations.

### cpu_dispatch, cpu_specific

Provides the ability to write one or more versions of a function that execute only on a list of targeted processors (`cpu_dispatch`). Provides the ability to declare that a version of a function is targeted at particular types of processors (`cpu_specific`).

**Syntax**

```c
// Windows:
__declspec(cpu_dispatch(cpuid, cpuid, ...))
__declspec(cpu_specific(cpuid))
// Linux:
__attribute__((cpu_dispatch(cpuid, cpuid, ...)))
__attribute__((cpu_specific(cpuid)))
```

**Arguments — `cpuid` possible values**

| `cpuid` | Processor |
|---|---|
| `arrowlake_s` | Intel® Core™ Ultra processors (formerly known as Arrow Lake) |
| `atom` | Intel® Atom™ processors with Intel® Supplemental Streaming SIMD Extensions 3 (Intel® SSSE3) |
| `atom_sse4_2` | Intel® Atom™ processors with Intel® Streaming SIMD Extensions 4.2 (Intel® SSE4.2) |
| `atom_sse4_2_movbe` | Intel® Atom™ processors with Intel® SSE4.2 with MOVBE instructions enabled |
| `broadwell` | This is a synonym for `core_5th_gen_avx` |
| `common_avx512` | Intel® processors that support Intel® Advanced Vector Extensions 512 (Intel® AVX-512) |
| `core_2_duo_sse3` | Intel® Core™ 2 Duo processors with Intel® SSE3 instructions |
| `core_2_duo_sse4_1` | Intel® Core™ 2 Duo processors with Intel® SSE4.1 instructions |
| `core_2nd_gen_avx` | 2nd generation Intel® Core™ processor family with support for Intel® AVX |
| `core_3rd_gen_avx` | 3rd generation Intel® Core™ processor family with support for Intel® AVX including the RDRND instruction |
| `core_4th_gen_avx` | 4th generation Intel® Core™ processor family with support for Intel® AVX2 including the RDRND instruction |
| `core_4th_gen_avx_tsx` | 4th generation Intel® Core™ processor family with support for Intel® AVX2 including the RDRND instruction, and support for Intel® Transactional Synchronization Extensions (Intel® TSX) |
| `core_5th_gen_avx` | 5th generation Intel® Core™ processor family with support for Intel® AVX2 including the RDSEED and Multi-Precision Add-Carry Instruction Extensions (ADX) instructions |
| `core_5th_gen_avx_tsx` | 5th generation Intel® Core™ processor family with support for Intel® AVX2 including the RDSEED and ADX instructions, and support for Intel® TSX |
| `core_aes_pclmulqdq` | Intel® Core™ processors with support for Advanced Encryption Standard (AES) instructions and carry-less multiplication instruction |
| `core_i7_sse4_2` | Intel® Core™ i7 processors with Intel® SSE4.2 instructions |
| `generic` | Other Intel processors for Intel® 64 architecture or compatible processors not provided by Intel Corporation |
| `goldmont_plus` | Intel® Atom™ processors with Goldmont Plus architecture |
| `graniterapids_d` | Intel® Xeon® 6 processors (formerly known as Granite Rapids) |
| `haswell` | This is a synonym for `core_4th_gen_avx` |
| `icelake_client` | 10th generation Intel® processors (formerly known as Ice Lake) |
| `icelake_server` | 3rd generation Intel® Xeon® scalable server processors (formerly known as Ice Lake) |
| `knl` | Intel® Xeon Phi™ x200 processors (formerly known as Knights Landing) |
| `knm` | Intel® Xeon Phi™ 72x5 processors (formerly known as Knights Mill) |
| `mic_avx512` | This is a synonym for `knl` |
| `pentium` | Intel® Pentium® processor |
| `pentium_4` | Intel® Pentium® 4 processors |
| `pentium_4_sse3` | Intel® Pentium® 4 processor with Intel® SSE3 instructions, Intel® Core™ Duo processors, Intel® Core™ Solo processors |
| `pentium_ii` | Intel® Pentium® II processors |
| `pentium_iii` | Intel® Pentium® III processors |
| `pentium_iii_no_xmm_regs` | Intel® Pentium® III processors with no XMM registers |
| `pentium_m` | Intel® Pentium® M processors |
| `pentium_mmx` | Intel® Pentium® processors with MMX™ technology |
| `pentium_pro` | Intel® Pentium® Pro processors |
| `skylake_avx512` | Intel® Xeon® Processor E3 v5 Family (formerly known as Skylake) processors with Intel® AVX-512 |

**Description:** use the `cpu_dispatch` keyword to provide a list of targeted processors, along with an empty function body/function stub. Use the `cpu_specific` keyword to declare each function version targeted at a particular type of processor.

**Caveats:** these features are available for Intel processors based on Intel® 64 architecture. They may not be available for non-Intel processors. If your non-Intel processor is not supported, you will get an `"invalid option"` error at compile-time. Applications built using the manual processor dispatch feature may be more highly optimized for Intel processors than for non-Intel processors.

### target

Specifies a target for called functions or variables.

**Syntax**

```c
// Windows:
__attribute__((target(target-name)))
// Linux:
__attribute__((target(target-name)))
```

**Arguments — `target-name` possible values:**

`arch=skylake-avx512`, `arch=corei7`, `arch=core2`, `arch=atom`, `mmx`, `sse`, `sse2`, `sse3`, `ssse3`, `sse4.1`, `sse4.2`, `popcnt`, `aes`, `pclmul`, `avx`, `avx2`, `avx512f`

**Description:** this keyword specifies that the called function or variable is also available on the target. Only functions or variables marked with this attribute are available on the target, and only these functions can be called on the target.

## Intrinsics

A detailed introduction and information about Intel intrinsics is provided in the Intel® C++ Compiler Classic Developer Guide and Reference. The Intel® Intrinsics Guide provides detailed information and a lookup tool for viewing the available Intel intrinsics.

General information:

- Intrinsics are assembly-coded functions that let you use C++ function calls and variables in place of assembly instructions.
- **Intrinsics can be used only on the host.**
- Intrinsics are expanded inline, eliminating function call overhead. Providing the same benefit as using inline assembly, intrinsics improve code readability, assist instruction scheduling, and help reduce debugging.
- Intrinsics provide access to instructions that cannot be generated using the standard constructs of the C and C++ languages.

**NOTE — to use intrinsic-based code with the Intel® oneAPI DPC++/C++ Compiler:**

1. Include the `immintrin.h` header file that comes with the intrinsic declarations.
2. Use `__attribute__((target(<required target>)))` to denote functions that are intended to be executed on specific target architectures. This provides the advantage of allowing the parts of the compilation unit that do not use intrinsics to be compiled using the default architecture, while also allowing functions that do use intrinsics to be targeted for a specific architecture. For more information about the `target` attribute, see `target`.

**Availability of Intrinsics on Intel Processors.** Not all Intel® processors support all intrinsics. For information on which intrinsics are supported on Intel® processors, visit the Product Specification, Processors page. The Processor Spec Finder tool links directly to all processor documentation and the datasheets list the features, including intrinsics, supported by each processor.

## Libraries

The Intel® oneAPI DPC++/C++ Compiler lets you use all the standard runtime libraries that are part of Microsoft Visual C++. The options described in this section can help you determine which libraries your application uses.

### Create Libraries

Libraries are an indexed collection of object files, which are included as needed in a linked program. Combining object files into a library makes it easy to distribute your code without disclosing the source and reduces the number of command-line entries needed to compile your project.

To create libraries, use the `lib.exe` tool or `xilib.exe` tool.

### Static Libraries

Executables generated using static libraries are no different than executables generated from individual source or object files. Static libraries are not required at runtime, so you do not need to include them when distributing your executable. Linking to a static library can be more efficient at compile time than linking to individual source files.

**Linux**

1. Use the `c` option to generate object files from the source files:
   ```bash
   icpx -c my_source1.cpp my_source2.cpp my_source3.cpp
   ```
2. Create the library file from the object files. With the GNU\* tool `ar`:
   ```bash
   ar rc my_lib.a my_source1.o my_source2.o my_source3.o
   ```
   If using the `flto` option during the compile step, you must use the LLVM tool `llvm-ar` to create the library file from the object files:
   ```bash
   llvm-ar rc my_lib.a my_source1.o my_source2.o my_source3.o
   ```
3. Compile and link your project with your new library:
   ```bash
   icpx main.cpp my_lib.a
   ```
   If your library file and source files are in different directories, use the `Ldir` option to indicate where your library is located:
   ```bash
   icpx -L/cpp/libs main.cpp my_lib.a
   ```

**Windows**

1. Use the `c` option to generate object files from the source files:
   ```bash
   icx -c my_source1.cpp my_source2.cpp my_source3.cpp
   ```
2. Create the library file from the object files with the `lib` command:
   ```bash
   lib -out:my_lib.lib my_source1.obj my_source2.obj my_source3.obj
   ```
   If using the `flto` option during the compile step, you must use the LLVM tool `llvm-lib`:
   ```bash
   llvm-lib -out:my_lib.lib my_source1.obj my_source2.obj my_source3.obj
   ```
3. Compile and link your project with your new library:
   ```bash
   icx main.cpp my_lib.lib
   ```
   Different directories: `icx -L/cpp/libs main.cpp my_lib.lib`

### Shared Libraries

Shared libraries, also called dynamic libraries or Dynamic Shared Objects (DSO), are linked differently than static libraries. At compile time, the linker ensures that all the necessary symbols are either linked into the executable or can be linked at runtime from the shared library. Executables compiled from shared libraries are small, but the shared libraries must be included with the executable to function correctly. When multiple programs use the same shared library, only one copy of the library is required in memory.

**Linux**

1. Use options `fPIC` and `c` to generate object files from the source files:
   ```bash
   icpx -fPIC -c my_source1.cpp my_source2.cpp my_source3.cpp
   ```
2. Use the `shared` option to create the library file from the object files:
   ```bash
   icpx -shared -o my_lib.so my_source1.o my_source2.o my_source3.o
   ```
3. Compile and link your project with your new library:
   ```bash
   icpx main.cpp my_lib.so
   ```

**Windows** — options to create libraries:

| Option | Description |
|---|---|
| `/LD`, `/LDd` | Produces a DLL. `d` indicates the debug version. |
| `/MD`, `/MDd` | Compiles and links with the dynamic, multi-thread C runtime library. `d` indicates the debug version. |
| `/MT`, `/MTd` | Compiles and links with the static, multi-thread C runtime library. `d` indicates the debug version. |
| `/Zl` | Disables embedding default libraries in object files. |

See also: Use Intel Shared Libraries; `/LD` compiler option; `/MD` compiler option; `/MT` compiler option; `/ZI` compiler option.

### Use Intel Shared Libraries

**This content does not apply for SYCL.**

By default, the Intel C++ libraries (`libirc`, `libsvml`, `libimf`, `libirng`) are linked in statically. When building a Dynamic Shared Object (DSO) using `-shared`, the Intel libraries are linked in dynamically.

**Options for Shared Libraries**

| Option | Description |
|---|---|
| `-fpic` | Use the `fpic` option when building shared libraries. It is required for the compilation of each object file included in the shared library. |
| `-shared-intel` | Use the `shared-intel` option to link Intel libraries dynamically. This has the advantage of reducing the size of the application binary, but it also requires the libraries to be on the application's target system. |
| `-shared` | Use the `shared` option to link Intel-provided libraries dynamically. The `shared` option instructs the compiler to build a DSO instead of an executable. For more details, refer to the `ld` man page documentation. |
| `-static-intel` | Use `-static-intel` to link in the Intel libraries statically. |

**Options for Dynamic Link Library (DLL) for Windows**

| Option | Description |
|---|---|
| `/LD`, `/LDd` | Creates the dynamically linked library. Produces a DLL. `d` indicates the debug version. |

See also: `fpic` compiler option; `/LD` compiler option; `shared` compiler option; `shared-intel` compiler option; `static` compiler option.

### Manage Libraries

#### Manage Libraries on Linux

During compilation, the compiler reads the `LIBRARY_PATH` environment variable for static libraries it needs to link when building the executable. At runtime, the executable will link against dynamic libraries referenced in the `LD_LIBRARY_PATH` environment variable. Add the location of your static libraries to the `LIBRARY_PATH` environment variable so that they are available for linking during compilation.

Example — compile `file.cpp` and link it with the library `lib.a` located in the `/libs` directory, using the `icpx` driver:

1. Add the directory `/libs` to `LIBRARY_PATH` from the command line with the `export` command:
   ```bash
   export LIBRARY_PATH=/libs:$LIBRARY_PATH
   ```
   Alternately, add the directory to `LIBRARY_PATH` by adding the `export` command to your startup file.
2. Compile `file.cpp` and link it with `lib.a`:
   ```bash
   icpx file.cpp lib.a
   ```

To link your library during compilation without modifying `LIBRARY_PATH`, use the `-L` option:
```bash
icpx file.cpp -L /libs lib.a
```

During compilation, the compiler passes object files to the linker in the following order:

1. Object files, from files specified on the command line, in the order they are specified (left to right)
2. Objects or libraries specified in default configuration files
3. Default Intel and system libraries

For example, the command `icpx lib1.a file.cpp lib2.a` has the link order: `lib1.a`, `file.o`, `lib2.a`, objects or libraries specified in default configuration files, default Intel and system libraries.

#### Manage Libraries on Windows

The `LIB` environment variable contains a semicolon-separated list of directories in which the Microsoft linker will search for library (`.lib`) files. The compiler does not specify library names to the linker but includes directives in the object file to specify the libraries to be linked with each object. For more information on adding library names to the response file and the configuration file, see *Use Response Files* and *Use Configuration Files*.

To specify a library name on the command line, you must first add the library's path to the `LIB` environment variable. Then you can specify the library name on the command line. For example, to compile `file.cpp` and link it with the library `mylib.lib` with the Intel® C++ Compiler:

```bash
icx file.cpp mylib.lib
```

### Compile with SYCL and Link Other Compilers

When you use the compiler and source its entire environment, then linking works correctly with other compilers if the correct path to the compiler libraries is set. This allows programs to be compiled with SYCL and then linked with other compilers (example: `gcc`). If you try to do this without sourcing the compiler environment, the linking fails with undefined references in `libsycl.so` and other internal libraries.

To resolve this, add the following paths to `LD_LIBRARY_PATH`:

```text
<install_dir>/compiler/latest/linux/compiler/lib/intel64
<install_dir>/compiler/latest/linux/lib
<install_dir>/compiler/latest/linux/lib/x64
<install_dir>/tbb/latest/lib/intel64/gcc4.8
```

### Other Considerations

The Intel Compiler Math Libraries contain performance-optimized implementations for various Intel platforms. By default, the best implementation for the underlying hardware is selected at runtime. The library dispatch of multi-threaded code may lead to apparent data races, which may be detected by certain software analysis tools. However, as long as the threads are running on cores with the same CPUID, these data races are harmless and are not a cause for concern.

### Redistribute Libraries When Deploying Applications

When you deploy your application to systems that do not have a compiler installed, you need to redistribute certain Intel® libraries which your application has a dependency on. You can address this in one of the following ways:

- **Statically link your application:** an application built with statically-linked libraries eliminates the need to distribute runtime libraries with the application executable. By linking the application to the static libraries, you are not dependent on the dynamic shared libraries.
- **Dynamically link your application:** if you build your application with dynamically linked (or shared) compiler libraries, address the following requirements: determine which shared or dynamic libraries your application needs; build your application with shared or dynamic libraries that are redistributable; pay attention to the directory where the redistributables are installed and how the OS finds them.

The redistributable library installation packages are available at: Latest Intel® oneAPI versions; Previous Intel® oneAPI and Intel® Parallel Studio XE versions.

### Shared Library Deployment

If your application relies on shared libraries distributed with Intel® oneAPI tools, you must make sure that your users have these shared libraries on their systems. Two options:

| Model | Description |
|---|---|
| Private Model | Copy the shared libraries from the Intel® oneAPI Toolkit into your application environment, and then package and deploy them with your application. Review the license and third-party files associated with the Intel® oneAPI Toolkits and/or components you have installed to determine which files you can redistribute. **Advantage:** you control your library and version choice, so you only package and deploy libraries you have tested. **Disadvantage:** end users may see multiple libraries installed on their system if multiple installed applications all use the private model; you are also responsible for updating these libraries whenever updates are required. See *Resolve Shared Library Dependencies for Private Model*. |
| Public Model | You direct your users to download and install runtime library packages provided by Intel. Your users install these packages on their system when they install your application. The runtime packages install to a fixed, accessible location, so all applications built with Intel oneAPI tools can find the libraries on which they depend. **Advantage:** one copy of each library is shared by all applications, and you can rely on updates to the runtime packages to resolve library issues independently from when you update your application. **Disadvantage:** the footprint of the runtime package is larger than the selected subset of libraries used in the private model, and your tested versions of the runtime libraries may not be the same as your end user's versions. See *Resolve Shared Library Dependencies for Public Model*. |

**NOTE:** Intel ensures that newer compiler-support libraries work with older versions of generated compiler objects, but newer versioned objects require newer versioned compiler-support libraries. If an incompatibility is introduced that causes newer compiler-support libraries not to work with older compilers, you will have sufficient warning and the library will be versioned so that deployed applications continue to work.

### Resolve Shared Library Dependencies for Private Model

1. **Determine runtime dependencies.** Use one of the following commands for each of your programs and components to list the shared libraries your application depends on:
   ```bash
   # Linux
   ldd programOrComponentName
   # Windows
   dumpbin /DEPENDENTS programOrComponentName
   ```
   **NOTE:** these commands are adequate to list dependencies for most programs. For applications that use SYCL or OpenMP offload, additionally refer to the list of offload dependencies in *Shared Library Dependencies for Device Offload*.
2. **Locate the shared libraries for redistribution.** The compiler runtime package installs the shared libraries at the following locations.
   - Runtime libraries for applications targeting CPU natively:
     - Linux — component directory layout `<oneAPI-install-dir>/compiler/<version>/`; unified directory layout `<oneAPI-install-dir>/<toolkit_version>/<install>/<version>/lib/`
     - Windows — component directory layout `C:\Program Files (x86)\Common Files\intel\Shared Libraries`; unified directory layout `<oneAPI-install-dir>\bin\share\doc\compiler\`
   - Runtime libraries for applications using SYCL or OpenMP offload:
     - Linux — component directory layout `<oneAPI-install-dir>/compiler/<version>/lib/`; unified directory layout `<oneAPI-install-dir>/<toolkit_version>/lib/`
     - Windows — component directory layout `<oneAPI-install-dir>\compiler\<version>\lib\`; unified directory layout `<oneAPI-install-dir>\<toolkit_version>\lib\`
3. **Decide where to package your dependencies.** The main requirement is the ability to resolve the dependencies during the run of the application. This can depend on the OS and on any defined search paths that are embedded in the application or resolved with environment variables.

### Resolve Shared Library Dependencies for Public Model

- Runtime packages are available from the oneAPI Standalone Components page.
- Runtime packages install to a fixed location:
  - Linux: `/opt/intel/oneapi/lib` or `/opt/intel/oneapi/<toolkit_version>/lib`
  - Windows: `C:\Program Files (x 86)\Common Files\intel\Shared Libraries`
- Set application environment variables:
  - Linux: depending on the location determined by the installed package, the dependencies can be resolved by setting the `LD_LIBRARY_PATH` environment variable or embedding the search locations via RPATH related constructs.
  - Windows: resolution of a given dll is typically done by setting the appropriate `PATH` or locating the dll in the executable location. System registration is also an option.

### Shared Library Dependencies for Device Offload

If your application uses offload, you need to:

1. Redistribute the shared libraries that your application depends on (listed as a result of step one in *Resolve Shared Library Dependencies for Private Model*).
2. Redistribute the shared libraries for each target that you are programming for.

### Compatibility in the Minor Releases of the Intel oneAPI Products

For Intel oneAPI products, each minor version of the product is compatible with the other minor version from the same release (for example, 2021). When there are breaking changes in API or ABI, the major version is increased. For example, if you tested your application with an Intel oneAPI product with a 2021.1 version, it will work with all 2021.x versions. It is not guaranteed that it will work with 2022.x or 19.x versions.

### Resolve References to Shared Libraries

If you are building an application that will be deployed to your user community and you are relying on shared libraries (`.so` shared objects on Linux, `.dll` dynamic libraries on Windows) distributed with Intel® oneAPI tools, you must make sure that your users have these shared libraries on their systems. To determine what shared libraries you depend on, use one of the following commands for each of your programs and components:

```bash
# Linux
ldd
# Windows
dumpbin /DEPENDENTS programOrComponentName
```

Once you have done this, you must choose how your users will receive these libraries.

#### Deployment Models

Two options for deploying the shared libraries from the Intel oneAPI toolkit that your application depends on:

| Model | Description |
|---|---|
| Private Model | Copy the shared libraries from the Intel oneAPI toolkit into your application environment, and then package and deploy them with your application. Review the license and third-party files associated with the Intel oneAPI toolkits and/or components you have installed to determine the files that you can redistribute. The advantage is control over library and version choice, so you only package and deploy the libraries you have tested. The disadvantage is that end users may see multiple libraries installed on their system if multiple installed applications all use the private model; you are also responsible for updating these libraries whenever updates are required. |
| Public Model | You direct your users to runtime packages provided by Intel. Your users install these packages on their system when they install your application. The runtime packages install onto a fixed location, so all applications built with Intel oneAPI tools can be used. The advantage is that one copy of each library is shared by all applications, which results in improved performance; you can rely on updates to the runtime packages to resolve library issues independently from when you update your application. The disadvantage is that the footprint of the runtime package is larger than a package from the private model, and your tested versions of the runtime libraries may not be the same as your end user's versions. |

**NOTE:** Intel ensures that newer compiler-support libraries work with older versions of generated compiler objects, but newer versioned objects require newer versioned compiler-support libraries. If an incompatibility is introduced that causes newer compiler-support libraries not to work with older compilers, you will have sufficient warning and the library will be versioned so that deployed applications continue to work.

### Additional Steps

Under either model, you must manually configure certain environment variables that are normally handled by the `oneapi-vars`, `setvars`, or `vars` scripts or module files.

For example, with the Intel® MPI Library, you must set the following environment variables during installation:

```bash
# Linux
I_MPI_ROOT=installPath FI_PROVIDER_PATH=installPath/intel64/libfabric:/usr/lib64/libfabric
# Windows
I_MPI_ROOT=installPath
```

### Redistributable Library Considerations

The Intel Compiler links to some Intel and non-Intel libraries by default; additional libraries are linked with different options. If your application links to a redistributable library, you need to ensure that those libraries are packaged with your application.

| Option (Linux / Windows) | Intel libraries (Linux / Windows) | Non-Intel libraries (Linux / Windows) |
|---|---|---|
| `default` / `default (MT)` (Linux `default (static-intel)`) | Linux: `libsvml.a`, `libirng.a`, `libimf.a`, `libirc.a`, `libirc_s.a`; Windows: `libircmt.lib`, `svml_dispmt.lib`, `libdecimal.lib`, `libmmt.lib` | Linux: `libstdc++.so` (`icpx`), `libm.so`, `libgcc_s.so`, `libgcc.so`, `libdl.so`, `libc.so`; Windows: `libcmt.lib`, `oldnames.lib` |
| `shared`/`shared-intel` / `MD` | Linux: `libsvml.so`, `libirng.so`, `libmf.so`, `libintlc.so`, `libirc_s.so`; Windows: `libircmt.lib`, `svml_dispmd.lib`, `libdecimal.lib`, `libmmd.lib` | Linux: `libstdc++.so`, `libm.so`, `libgcc.so`, `libgcc_s.so`, `libdl.so`, `libc.so`; Windows: `msvcrt.lib`, `oldnames.lib` |
| (Windows) `MTd` | `libircmt.lib`, `svml_dispmt.lib`, `libdecimal.lib`, `libmmt.lib` | `libcmtd.lib`, `oldnames.lib` |
| (Windows) `MDd` | `libircmt.lib`, `svml_dispmd.lib`, `libdecimal.lib`, `libmmdd.lib` | `msvcrtd.lib`, `oldnames.lib` |
| `fiopenmp` / `Qiopenmp` | `libiomp5.so` / `libiomp5md.lib` | Linux: `libpthread.so` |
| `fiopenmp targets=spir64` / `Qopenmp-targets=spir64` | `libiomp5.so`, `libomptarget.so` / `libiomp5md.lib`, `omptarget.lib` | — |
| `qopenmp-stubs` / `Qopenmp-stubs` | `libiompstubs5.so` / `libiompstubs5md.lib` | — |
| `fprofile-instr-generate`/`fprofile-generate` | `libclang_rt.profile.a` / `clang_rt.profile-x86_64.lib` | — |
| `fmemory-profile` | `libclang_rt.memprof.a`, `libclang_rt.memprof_cxx.a` (`icpx`) | — |
| `fortlib` | `libifcoremt.a` | Linux: `libpthread.so` |
| `fortlib` `shared`/`shared-intel` | `libifcoremt.so` | Linux: `libpthread.so` |
| `fsycl` / `fsycl` | `libsycl.so`, `libsycl-devicelib-host.so` / `sycl.lib`, `sycl-devicelib-host.lib` | — |
| `qdaal` / `Qdaal` | `libonedal_core.a`, `libonedal_thread.a`, `libtbb.a` / `tbb.lib`, `onedal_core.lib`, `onedal_thread.lib` | — |
| `qmkl=parallel` / `Qmkl:parallel` | `libmkl_intel_lp64.a`, `libmkl_intel_thread.a`, `libmkl_core.a`, `libiomp5.a` / `mkl_intel_lp64.lib`, `mkl_intel_thread.lib`, `mkl_core.lib`, `libiomp5md.lib` | Linux: `libpthread.so` |
| `qmkl=sequential` / `Qmkl:sequential` | `libmkl_intel_lp64.a`, `libmkl_intel_sequential.a`, `libmkl_core.a`, `libiomp5.a` / `mkl_intel_lp64.lib`, `mkl_intel_sequential.lib`, `mkl_core.lib`, `libiomp5md.lib` | — |
| `qtbb` / `Qtbb` | `libtbb` / `tbb.lib` | — |
| `qipp` / `Qipp` | Linux: `libippcv.a`, `libppch.a`, `libippcc.a`, `libippdc.a`, `libippe.a`, `libippi.a`, `libipps.a`, `libippvm.a`, `libippcore.a`; Windows: `ippcv.lib`, `ippch.lib`, `ippcc.lib`, `ippdc.lib`, `ipe.lib`, `ippi.lib`, `ipps.lib`, `ippvm.lib`, `ippcore.lib` | — |
| `qipp=crypto\|nonpic_crypto` / `Qipp:crypto` | `libippcp.a` / `ippcp.lib` | — |
| `qactypes` / `Qactypes` | Linux: `libdspba_mpir.a`, `libdspba_mpfr.a`, `libac_types_fixed_point_math_x86.a`, `libac_types_vpfp_library.a`; Windows: `dspba_mpir.lib`, `dspba_mpfr.lib`, `ac_types_fixed_point_math_x86.lib`, `ac_types_vpfp_library.lib` | — |

### Intel's Memory Allocator Library

Intel's `libqkmalloc` library for fast memory allocation provides a C-level interface for memory allocation optimized for performance.

You can link the `libqkmalloc` library as a shared library only on Linux and Windows platforms for Intel® 64 architecture. This library provides optimized implementation of standard allocation routines `malloc`, `calloc`, `realloc`, and `free`, and is C99 standard compliant.

**NOTE:** this library is limited to work only on Intel® processors and will redirect to standard C routines at runtime if used on non-Intel® processors.

#### Use Intel's Custom Memory Allocator Library

You can use the `libqkmalloc` library by linking directly to it or by using the `LD_PRELOAD` environment variable (**Linux only**).

To ensure that the application overrides the standard library allocation routines on Linux with `libqkmalloc`, set the environment variable `LD_PRELOAD` in the command line before the application execution. This environment variable allows you to set a library path that loads before any other library (including the C runtime library). The application uses symbols from the specified library instead of symbols from the standard library.

#### Restrictions

This library does not support threaded code such as OpenMP\* and is not thread safe. It should not be used simultaneously from multiple threads. This library should be used with large throughput workloads for the best results.

## SIMD Data Layout Templates (SDLT)

C++11 template library; containers represent arrays of Plain Old Data objects (no pointer/reference members, no virtual functions) in SIMD-friendly layouts. Standard ISO C++11, no special compiler, but uses OpenMP* SIMD extensions and `pragma ivdep`. Interfaces use generic programming (requirements on types, like the C++ STL).

Motivation: `sdlt::soa1d_container<Point3s>` plus accessors replaces `std::vector<Point3s>` (AoS), giving the vectorizer unit stride loads and alignment info:

```cpp
SDLT_PRIMITIVE(Point3s, x, y, z)
sdlt::soa1d_container<Point3s> inputDataSet(count);
auto inputData = inputDataSet.const_access();   // outputData = outputDataSet.access()
#pragma forceinline recursive
#pragma omp simd
for(int i=0; i < count; ++i) { Point3s inputElement = inputData[i]; outputData[i] /* = transform */; }
```

### Version Information

SDLT **version 2** extends version 1 with n-dimensional containers.

### Backwards Compatibility

v2 is fully backward compatible with v1 public interfaces: v1 public-API source recompiles against v2 headers unchanged; v2 APIs are in namespace `sdlt::v2`, so ABI linkage does not collide with v1 ABIs that exist only in `sdlt`, and a binary dynamically linked library using v1 can link with a program using v2 and vice versa. Limitations: SDLT containers/accessors in a library's public API (ABI) require library and caller to match versions (never mix); internal implementation is not covered (v1 internals were updated and unified with v2 parts).

### Deprecated

| Deprecated Interface | Deprecated in Version | Replaced By |
|---|---|---|
| `sdlt::fixed_offset<>` | v2 | `sdlt::fixed<>` |
| `sdlt::aligned_offset<>` | v2 | `sdlt::aligned<>` |

### Function Calls and Containers

#### Function Calls

Inline SDLT Primitives passed by value, pointer, or reference; non-inlined functions should be SIMD enabled (e.g. `#pragma omp declare simd`). A loop variable passed to a non-inlined function requires the ABI layout match the object's original, adding transformations or inhibiting vectorization, so use `#pragma forceinline recursive` (ignores compiler limits/heuristics, inlines all inlinable calls and callees) — but it can make bodies too big to optimize, in which case restructure boundaries or use non-inlined SIMD-enabled calls. "inline" alone is only a hint.

#### 1-Dimensional Containers Overview

1D containers abstract an array of objects to AOS (Array of Structures) or SOA (Structure of Arrays, SIMD friendly) and resize dynamically like `std::vector<T>`. **Import/Export Only**: the abstracted layout means no memory references to underlying data, only import/export of an object to/from an element. Copy construction is deleted (avoids C++11 lambda capture); use explicit **clone**. They provide `accessor` and `const_accessor` for SIMD loops; `std::vector` compatibility is for integration, not high performance; containers own the data, scope controls its life.

#### n-Dimensional Containers Overview

Multi-dimensional containers generalize 1D containers, separating multi-dimensional access semantics from storage logic; generic over any number of dimensions while representing data internally. They are **not resizable** and have no `std::vector`-like interface — akin to arrays (statically sized or variable length). Parameterized by primitive type, storage layout, observed shape:

```cpp
n_container<PrimitiveT, LayoutT, ExtentsT>
```

`typename PrimitiveT` = contained primitive type; `typename LayoutT` = data layout type; `typename ExtentsT` = dimensions of the container.

#### Construct an n_container

Declare the type as a `SDLT_PRIMITIVE`, choose a layout template parameter, then shape it with per-dimension extents.

| Layout | Description |
|---|---|
| `layout::soa<>` | Structure of Arrays (SOA); each data member gets its own N-dimensional array. |
| `layout::soa_per_row<>` | Structure of Arrays Per Row; each member gets its own 1-dimensional array per row; layout repeats for remaining N-1 dimensions. |
| `layout::aos_by_struct` | Array of Structures (AOS) Accessed by Struct; native AOS layout and data access. |
| `layout::aos_by_stride` | Array of Structures Accessed by Stride; native SOA access through pointers to built in types using a stride. |

Numbers/constants come in three forms, each less informative to the compiler; prefer the most precise:

| Integer Value Specification | Description |
|---|---|
| `fixed<int NumberT>` | Compile-time constant, e.g. `foo(fixed<1080>(), fixed<1920>());`. Suffix `_fixed` is an equivalent literal (`1080_fixed` == `fixed<1080>`): `foo(1080_fixed, 1920_fixed);` |
| `aligned<int AlignmentT>(number)` | Guaranteed multiple of `AlignmentT`: `foo(aligned<8>(height), aligned<128>(width));` |
| `"int"` | Arbitrary integer: `foo(width, height);` |

Shape: `n_extent_t<…>` is variadic over dimensions; generator `n_extent` builds extents with array-like syntax, most precise form preferred so alignments can be proven:

```cpp
n_extent[height][width];  n_extent[height][aligned<128>(width)];  n_extent[1080_fixed][1920_fixed];
```

Define: 2D `RGBAs` container of HD image size 1920x0180 [sic: source garbled]; runtime extents when sizes are unknown; factory `make_n_container<PrimitiveT, LayoutT>` deduces extents:

```cpp
struct RGBAs { float red, green, blue, alpha; };
SDLT_PRIMITIVE(RGBAs, red, green, blue, alpha)
typedef n_container<RGBAs, layout::soa, n_extent_t<fixed<1080>, fixed<1920>>> HdImage;   HdImage image1;
typedef n_container<RGBAs, layout::soa, n_extent_t<int, int>> Image;   Image image2(n_extent[height][width]);
auto image1 = make_n_container<RGBAs, layout::soa>(n_extent[1080_fixed][1920_fixed]);
```

Access: containers own the data; use an accessor and successive `[]` calls (like a C multi-dimensional array). `extent_d<int DimensionT>(object)` queries an extent; non-template `ca.extent_d0()`, `ca.extent_d1()` exist; fewer subscripts than rank gives a lower-rank accessor.

```cpp
auto ca = image1.const_access();  auto a = image2.access();
RGBAs pixel = ca[y][x];  a[y][x] = RGBAs(greyscale, greyscale, greyscale);
auto cay = ca[y];  RGBAs p2 = cay[x];
```

### Bounds

**Description.** `bounds_t<LowerT, UpperT>` holds the lower and upper bounds of a half-open interval, modelling a valid iteration space over one dimension (whole extent or restricted); `n_bounds_t` aggregates `bounds_t` for multi-dimensional subsections.

Creating bounds — full type (tedious) or factory `bounds` (deduces types):

```cpp
bounds_t<int, int>(start, finish);  bounds_t<fixed<0>, fixed<1920>>();
bounds(start, finish);  bounds(start, aligned<16>(finish));  bounds(0_fixed, 1920_fixed);
```

Discovering bounds — initial accessor bounds are lower `fixed<0>` and upper = the dimension's extent value/type from construction (`fixed<>`, `aligned<>`, or `int`); use `bounds_d<int DimensionT>(object)` or non-template `ca.bounds_d0()` / `ca.bounds_d1()`; `bounds_t` supports C++11 range-based for.

```cpp
for (int y = bounds_d<0>(ca).lower(); y < bounds_d<0>(ca).upper(); ++y) for (int x: ca.bounds_d1()) { /* … */ }
```

N-dimensional: `n_index_t<…>` and `n_bounds_t<…>` are variadic; `n_index` and `n_bounds` generate instances; `n_bounds_t` also builds from an `n_index_t` and `n_extent_t`:

```cpp
n_index[540][960]
n_bounds[bounds(540,1080)][bounds(960,1920)]
n_bounds(n_index[540][960], n_extent[540][960]);
```

Subsections: `section(n_bounds_t)` restricts the results of `bounds_d<int Dimension>`; `translated_to(n_index_t)` and `translated_to_zero()` re-index. Accesses are translated back to the creating accessor's lower bounds, letting a smaller container participate in a larger block-walked index space or a subsection start at the origin.

### User-Level Interface

Defined in `sdlt.h` and its associated header files.

#### SDLT Primitives

Primitives are the data worked on in SIMD and may have methods that modify their data. **Rules:** Plain Old Data (POD) — trivial copy ctor, trivial move ctor, trivial destructor, no virtual functions/bases; no reference data members; no unions; no bit fields; no `bool` types (comparison semantics not efficient in SIMD; use a 32-bit integer compared against known values like 0 or 1); data members public or `SDLT_PRIMITIVE_FRIEND` declared. **Current Limitations:** no pointer data members; no C++11 strongly typed enums (use integers); no array based data members; copy constructor and assignment operator (`=`) by individual member assignment (strongly encouraged for code generation). Refactor `class Point3d { protected: double v[3]; }` to public `double x; double y; double z;`.

#### SDLT_PRIMITIVE Macro

Containers need the structure's data layout; C++11 lacks compile time reflection, so `SDLT_PRIMITIVE` takes a struct type plus a comma separated list of its data members.

```cpp
SDLT_PRIMITIVE(STRUCT_NAME, DATA_MEMBER_1, ...)
struct UserObject { float x; float y; double acceleration; int behavior; };
SDLT_PRIMITIVE(UserObject, x, y, acceleration, behavior)
struct Point3s { float x; float y; float z; };   struct AABB { Point3s topLeft; Point3s bottomRight; };
SDLT_PRIMITIVE(Point3s, x, y, z)   SDLT_PRIMITIVE(AABB, topLeft, bottomRight)
```

Declare before use in a Container; built-ins (`float`, `double`, `int`, …) are automatically Primitives. Nested Primitives are supported but must be declared before the outer one. Structs do not derive from SDLT, so classes stay usable in non-SDLT code.

#### soa1d_container

"Structure of Arrays" one-dimensional container of Primitives. `#include <sdlt/soa1d_container.h>`

```cpp
template<typename PrimitiveT, int AlignD1OnIndexT = 0, class AllocatorT = allocator::default_alloc>
class soa1d_container;
```

**Arguments:** `typename PrimitiveT` = type each element stores; `int AlignD1OnIndexT = 0` = [Optional] index on which data access is aligned (useful for stencils); `class AllocatorT = allocator::default_alloc` = [Optional] allocator type, currently the only one supported. Dynamically sized, internally Structure of Arrays: `std::vector`-like resizing plus accessors for SIMD loops. Every constructor also takes `buffer_offset_in_cachelines buffer_offset = buffer_offset_in_cachelines(0)` (cache lines off the buffer start, managing 4k cache aliasing) and `const allocator_type & an_allocator = allocator_type()`.

| Member / signature | Description |
|---|---|
| `typedef size_t size_type;` | Type for sizes passed to container methods |
| `template <typename OffsetT = no_offset> using accessor;` | Alias to an accessor |
| `template <typename OffsetT = no_offset> using const_accessor;` | Alias to a const_accessor |
| `soa1d_container(size_type size_d1 = 0u, ...);` | Uninitialized container of `size_d1` elements |
| `soa1d_container(size_type size_d1, const PrimitiveT &a_value, ...);` | `size_d1` elements initialized with `a_value` |
| `template<typename StlAllocatorT> soa1d_container(const std::vector<PrimitiveT, StlAllocatorT> &other, ...);` | Copy of each element in `other` |
| `soa1d_container(const PrimitiveT *other_array, size_type number_of_elements, ...);` | Copy of `number_of_elements` elements from `other_array` |
| `template< typename IteratorT > soa1d_container(IteratorT a_begin, IteratorT an_end, ...);` | Elements from range `[a_begin - an_end)` |
| `soa1d_container clone() const;` | New container owning a copy of the elements |
| `void resize(size_type new_size_d1);` | Resize; new elements are unitialized [sic] |
| `accessor<> access();` | Accessor, no embedded index offset |
| `accessor<int> access(int offset);` | Accessor, integer based embedded index offset |
| `template<int IndexAlignmentT> accessor<aligned_offset<IndexAlignmentT> > access(aligned_offset<IndexAlignmentT>);` | Accessor, `aligned_offset<IndexAlignmentT>` embedded offset |
| `template<int OffsetT> accessor<fixed_offset<OffsetT> > access(fixed_offset<OffsetT>);` | Accessor, `fixed_offset<OffsetT>` embedded offset |
| `const_accessor<> const_access() const;` | Const accessor, no embedded index offset |
| `const_accessor<int> const_access(int offset) const;` | Const accessor, integer based embedded offset |
| `const_accessor<aligned_offset<IndexAlignmentT> > const_access(aligned_offset<IndexAlignmentT> offset) const;` | Const accessor, `aligned_offset<IndexAlignmentT>` embedded offset |
| `template<int OffsetT> const_accessor<fixed_offset<OffsetT> > const_access(fixed_offset<OffsetT>) const;` | Const accessor, `fixed_offset<OffsetT>` embedded offset |

**STL Compatibility.** `std::vector` subset for integration, not high performance: iterators and `operator[]` return a Proxy (no references to the object), other "const" methods return a "value_type const", iterators have no `->` but work with any STL algorithm, `resize` does not initialize new elements. Implemented: `size`, `max_size`, `capacity`, `empty`, `reserve`, `shrink_to_fit`; `assign`, `push_back`, `pop_back`, `clear`, `insert`, `emplace`, `erase`; `cbegin`, `cend`, `begin`, `end`, `crbegin`, `crend`, `rbegin`, `rend`; `operator[]`, `front() const`, `back() const`, `at() const`; `swap`, `==`, `!=`; `soa1d_container(soa1d_container&& donor)`, `soa1d_container & operator=(soa1d_container&& donor)`.

#### aos1d_container

"Array of Structures" one-dimensional container of Primitives. `#include <sdlt/aos1d_container.h>`

```cpp
template<typename PrimitiveT, AccessBy AccessByT, class AllocatorT = allocator::default_alloc>
class aos1d_container;
```

**Arguments:** `typename PrimitiveT` = type each element stores; `access_by AccessByT` = enum controlling layout access, recommend `access_by_struct` unless vectorizing fails; `class AllocatorT = allocator::default_alloc` = [Optional] allocator type, currently the only one supported. Compatible interface with `soa1d_container` while keeping an AoS layout, so changing the container type switches layouts without changing accessor/proxy code. Constructor and member signatures and the implemented `std::vector` subset correspond to `soa1d_container` (`aos1d_container` substituted, same optional `buffer_offset_in_cachelines` and allocator parameters, same `access`/`const_access` overloads, `clone`, `resize` leaving new elements unitialized [sic], same STL caveats) — except the source's `soa1d_container` iterator list repeats "begin, end" twice (and later "rbegin, rend" twice) while the `aos1d_container` list does not.

#### access_by

Controls how the memory layout is accessed. `#include <sdlt/access_by.h>`

```cpp
enum access_by { access_by_struct, access_by_stride };
```

`access_by_struct` uses structure member access, drilling down through nested structure members; an AABB of two Point3d logically expands to `AABB local; local = accessor.mData[i];`. `access_by_stride` uses pointers to built in types with a stride for the primitive size, e.g. `local.topLeft.x = *(accessor.mData + offsetof(AABB,topLeft) + offset(Point3d,x) + (sizeof(AABB)*i));` (likewise `topLeft.y`, `topLeft.z`, `topRight.x`, `topRight.y`, `topRight.z`). `access_by_struct` can give better code (wide loads plus shuffle/insert into SIMD registers) but can fail to vectorize with complex or nested primitives; `access_by_stride` has always vectorized successfully since the compiler only sees a strided array pointer. Try `access_by_stride` when vectorization fails; the choice is explicit developer policy.

#### n_container

N-dimensional container; primitive type, exact memory layout and shape come from template arguments.

```cpp
template <typename PrimitiveT, typename LayoutT, typename ExtentsT, typename AllocatorT >
class n_container;
```

| Template Argument | Description |
|---|---|
| `typename PrimitiveT` | Type stored per cell; must be declared with `SDLT_PRIMITIVE` |
| `typename LayoutT` | In-memory data layout; a class from the `layout` namespace |
| `typename ExtentsT` | Container shape; a concrete `n_extent_t` variadic template type |
| `class AllocatorT = allocator::default_alloc` | [Optional] allocator type; currently the only one supported. |

Member types: `typedef PrimitiveT primitive_type;`, `typedef PrimitiveT allocator_type;`, `typedef implementation-defined accessor;` (writes or reads cells), `typedef implementation-defined const_accessor;` (reads cells). Both constructors take `buffer_offset_in_cachelines buffer_offset = buffer_offset_in_cachelines(0)` and `const AllocatorT &an_allocator = AllocatorT()`.

| Member | Description |
|---|---|
| `n_container (const ExtentsT &a_extents, ...)` | Uninitialized container of shape `a_extents` |
| `n_container (...)` from `ExtentsT` | Uninitialized container of shape `ExtentsT`; must be default constructible — only true when made up enitrely [sic] of `fixed<NumberT>` types |
| `n_container(n_container&& donor)` | Transfers the donor's buffers/organization; donor accessors become invalid |
| `n_container & operator = (n_container&& donor)` | Frees existing buffers then transfers ownership; donor accessors invalid |
| `const ExtentsT& n_extent () const` | Container shape; also free function `extent_d<int DimenstionT>(const n_container &)` [sic] |
| `const_accessor const_access();` | Const accessor knowing the data organization, for reading cells |
| `accessor access();` | Accessor knowing the data organization, for writing or reading cells |

Friend: `std::ostream& operator << (std::ostream& output_stream, const n_container & a_container)` — appends the container's extents values to `a_output_stream`.

### Layouts

#### sdlt::layout namespace

`layout` namespace types are the `n_container` template parameter instead of distinct container types.

| Layout | Description |
|---|---|
| `template <typename AlignOnColumnIndexT=0> layout::soa` | Structure of Arrays: each Primitive data member gets its own N-dimensional array, back-to-back in a contiguous buffer; `AlignOnColumnIndexT` names the row-dimension column to cache line align (each row's `AlignOnColumnIndexT` is cache line aligned). |
| `template <typename AlignOnColumnIndexT> layout::soa_per_row` | Structure of Arrays Per Row: each data member gets its own 1-dimensional array per row (Soa1d) back to back; each row's `AlignOnColumnIndexT` is cache line aligned; Soa1d's laid out sequentially model the remaining dimensions, effectively [sic] an Array of Structures of Arrays whose array size is the row extent. Efficient when the row extent can be `fixed<NumberT>`; if the row size is unknown at compile time, add a `fixed<Number>` dimension and divide the row by it. |
| `layout::aos_by_struct` | Array of Structures Accessed by Struct: natively laid out back to back, accessed via structure/member access; nested structures drill down through members. |
| `layout::aos_by_stride` | Array of Structures Accessed by Stride: natively laid out back to back, accessed through pointers to built in types with a stride for the Primitive size; useful if `aos_by_struct` doesn't vectorize. |

The classes are empty and only specialize containers for the denoted layouts.

#### Shape

`n_extent_t` describes the container shape: the number of dimensions [and] the size of each [sic].

```cpp
template<typename... TypeListT>
class n_extent_t
```

| Template Argument | Description |
|---|---|
| `typename... TypeListT` | Comma separated types; the count is the number of dimensions, each type the size representation of that dimension. Dimension order matches C++ subscripts declaring a multi-dimensional array, leftmost to rightmost. Type must be `int`, `fixed<NumberT>`, or `aligned<AlignmentT>` for each dimension size (extent), in regular order of C++ subscripts - outer to inner. |

| Member | Description |
|---|---|
| `static constexpr int rank;` | Number of dimensions |
| `static constexpr int row_dimension = rank-1;` | Index of the last dimension, row |
| `n_extent_t()` | Default values per `TypeListT`; correctly initialized only when every type is `fixed<NumberT>`; every type must be default constructible |
| `n_extent_t(const n_extent_t &a_other)` | Construct `n_exent_t` [sic] copying each dimension size |
| `explicit n_extent_t(const TypeListT & … a_values)` | Initialize each dimension from the `a_values` list (length/types defined by `TypeListT`) |
| `template<int DimensionT> auto get() const` | `DimensionT` [sic] >=0 and < rank; extent of `DimensionT` in that 0-based `TypeListT` type |
| `template<int DimensionT> auto rightmost_dimensions() const` | `DimensionT` [sic] >=0 and <= rank; lower rank `n_extent_t` copying the righmost [sic] `DimensionT` values |
| `template<class... OtherTypeListT> bool operator == (const n_extent_t<OtherTypeListT...> a_other) const` | Equal ranks; numeric sizes only, not types |
| `template<class... OtherTypeListT> bool operator != (const n_extent_t<OtherTypeListT...> a_other) const` | Equal ranks; true if any dimension differs numerically |
| `size_t size() const` | Number of elements: `get<0>()*get<1>()*get<…>()*get<rank-1>()` |

Friend: `std::ostream& operator << (std::ostream& output_stream, const n_extent_t & a_extents)`.

#### n_extent_generator

Simpler creation of `n_extent_t` objects.

```cpp
template<typename... TypeListT> class n_extent_generator;
namespace { n_extent_generator<> n_extent; }   // generator object
```

Recursively constructing `operator[]` for `fixed<>`, `aligned<>`, and integer values, one dimension at a time, C-array-like:

```cpp
n_extent_t<int, int> ext1(height, width);   n_extent_t<int, aligned<128>> ext2(height, width);
n_extent_t<fixed<1080>, fixed<1920>> ext3(1080_fixed, 1920_fixed);
auto ext1 = n_extent[height][width];  auto ext2 = n_extent[height][aligned<128>(width)];  auto ext3 = n_extent[1080_fixed][1920_fixed];
```

Class Hierarchy: use only `n_extent_t <...>`, from which `n_extent_generator < … >` derives; the generator object `n_extent` is automatically downcast. Template argument `typename... TypeListT`: count is the generator's current dimensions; requirements as for `n_extent_t` (`int`, `fixed<NumberT>`, or `aligned<AlignmentT>`). Member type `typedef n_extent_t<TypeListT...> value_type`.

| Member | Description |
|---|---|
| `n_extent_generator ()` | `TypeListT` empty; no extents specified |
| `n_extent_generator (const n_extent_generator &a_other)` | Copy extent values |
| `n_extent_generator<TypeListT..., int> operator [] (int a_size) const` | `a_size >= 0`; append rightmost integer based extent |
| `n_extent_generator<TypeListT..., fixed<NumberT>> operator [] (fixed<NumberT> a_size) const` | `a_size >= 0`; append rightmost `fixed<NumberT>` extent |
| `n_extent_generator<TypeListT..., aligned<AlignmentT>> operator [] (aligned<AlignmentT> a_size) const` | `a_size >= 0`; append rightmost `aligned<AlignmentT>` based extent |
| `value_type value() const` | Aggregated `n_extent_t<…>` with correct types and values |

#### make_ n_container template function

Factory producing a properly-typed `n_container<…>` from the passed `n_extent_t`.

```cpp
template<typename PrimitiveT, typename LayoutT,
         typename AllocatorT = allocator::default_alloc, typename ExtentsT>
auto make_n_container(const ExtentsT &_extents) -> n_container<PrimitiveT, LayoutT, ExtentsT, AllocatorT>
```

Use with template argument deduction and C++11 `auto`: `typedef n_container<RGBAs, layout::soa, n_extent_t<int, int>> HdImage; HdImage image1(n_extent[1080][1920]);` versus `auto image1 = make_n_container<RGBAs, layout::soa>(n_extent[1080][1920]);`.

#### extent_d template function

```cpp
template<int DimensionT, typename ObjT> auto extent_d(const ObjT &a_obj)
```

Gets one dimension's extent without extracting a whole `n_extent_t<…>`. `int DimensionT` = 0 based index from the leftmost dimension, >=0 and < `ObjT::rank`. `typename ObjT` = source object; `ObT` [sic] is one of `n_container<…>`, `n_extent_t<…>`, `n_extent_generator<…>`. Returns the correctly typed extent. Example body: `int extent_z = extent_d<0>(volume); int extent_y = extent_d<1>(volume); int extent_x = extent_d<2>(volume);` (`/…` in source [sic]).

### Bounds

#### bounds_t

Half-open interval with lower and upper bounds. `#include <sdlt/bounds.h>`

```cpp
template<typename LowerT = int, typename UpperT = int>
struct bounds_t
```

Supported types include `fixed<NumberT>`, `aligned<AlignmentT>` and integer values; models a valid iteration space over one dimension (whole extent or restricted). `n_bounds_t` aggregates `bounds_t` objects for multi-demensional [sic] subsections; compatible with C++ range-based loops. `typename LowerT = int` / `typename UpperT = int` = bound types, each `int`, `fixed<NumberT>`, or `aligned<AlignmentT>`. Member types: `typedef LowerT lower_type`; `typedef UpperT upper_type`; `typedef implementation-defined iterator`.

| Member | Description |
|---|---|
| `bounds_t()` | Uninitialized bounds |
| `bounds_t(lower_type l, upper_type u)` | `(u >= l)`; half-open interval `[l, u)` |
| `bounds_t(const bounds_t & a_other)` | Bounds initialized from `a_other` |
| `template<typename OtherLowerT, typename OtherUpperT> bounds_t(const bounds_t<OtherLowerT, OtherUpperT> & a_other)` | Types must legally convert (`int` to `fixed<8>()` is illegal) |
| `void set(lower_type l, upper_type u)` | Set inclusive lower and exclusive upper bound indices |
| `void set_lower(lower_type a_lower)` | Set the inclusive lower bound index |
| `void set_upper(upper_type a_upper)` | Set the exclusive upper bound index |
| `lower_type lower() const` | Inclusive lower bound index |
| `upper_type upper() const` | Exclusive upper bound index |
| `iterator begin() const` | Index iterator for the inclusive lower bound; C++11 range loops need `begin()` & `end()` |
| `iterator end() const` | Index iterator for the exclusive upper bound |
| `auto width() const` | `upper() – lower()`; type follows the subtraction of bound types |
| `template<typename OtherLowerT, typename OtherUpperT> bool contains(const bounds_t<OtherLowerT, OtherUpperT> &a_other) const` | `(a_other.lower() >= lower() && a_other.upper() <= upper())` |
| `template<typename T> auto operator + (const T &offset) const` | `bounds(lower() + offset, upper()+offset)`; returned `bound_t` [sic] types may differ |
| `template<typename T> auto operator - (const T & offset) const` | `bounds(lower() - offset, upper()-offset)`; returned types may differ |
| `bool operator == (const bounds_t &a_other) const` | `(lower() == a_other.lower() && upper() == a_other.upper())` |
| `template<typename OtherLowerT, typename OtherUpperT> bool operator == (const bounds_t<OtherLowerT, OtherUpperT> &a_other) const` | Equality across different bound types |
| `bool operator != (const bounds_t &) const` | `(lower() != a_other.lower() \|\| upper() != a_other.upper())` |
| `template<typename OtherLowerT, typename OtherUpperT> bool operator != (const bounds_t<OtherLowerT, OtherUpperT> &a_other) const` | Inequality across different bound types (source: "with with" [sic]) |

Friend: `std::ostream& operator << (std::ostream& a_output_stream, const bounds_t &a_bounds)` — appends lower and upper values. Range-based loop: `auto ca = image_container.const_access(); for (auto y: bounds_d<0>(ca)) for (auto x: bounds_d<1>(ca)) { RGBAs pixel = ca[y][x]; /* … */ }` — the iterator yields an index value, not an object value, and is meant to index accessors.

#### sdlt::bounds Template Function

Factory for `bounds_t`; deduces `LowerT` and `UpperT`. `#include <sdlt/bounds.h>`

```cpp
template<typename LowerT, typename UpperT> auto bounds(LowerT a_lower, UpperT a_upper)
```

`LowerT`/`UpperT` (default `int`) requirements as in `bounds_t`. Returns the correctly typed `bounds_t<LowerT, UpperT>`. Example: `bounds_t<fixed<0>, aligned<16>> my_bounds1(0_fixed, aligned<16>(upper))` versus `auto my_bounds = bounds(0_fixed, aligned<16>(upper))`.

#### n_bounds_t

Valid iteration space over an N-dimensional container or its accessor as a sequence of `bounds_t` per dimension. `#include <sdlt/n_bounds.h>`

```cpp
template<typename... TypeListT> class n_bounds_t
```

Dimensions may be compile-time known (`fixed<int NumberT>`), runtime (`int`), or runtime with guaranteed alignment multiple (`aligned<int Alignment>`). On creation the `n_bounds_t` starts at `fixed<0>` for inclusive lower bounds with exclusive upper bounds matching each dimension extent; translating or sectioning an accessor changes the `n_bounds_t` it provides. `typename... TypeListT`: count is the number of dimensions, each type the bounds representation of that dimension, in C++ subscript order leftmost to rightmost; requirements: entries be `bounds_t<LowerT, UpperT>`. Member types: `typedef implementation-defined lower_type` (`n_index_t<…>` from `lower()`), `upper_type` (from `upper()`). Comparison/arithmetic members require equal ranks unless stated.

| Member | Description |
|---|---|
| `static constexpr int rank;` | Number of dimensions |
| `static constexpr int row_dimension = rank-1;` | Index of the last dimension, the row |
| `n_bounds_t()` | Default values per `bounds_t`; correctly initialized only when every `bounds_t` has `LowerT`/`UpperT` = `fixed<NumberT>`; every `bounds_t` must be default constructible |
| `n_bounds_t(const n_bounds_t &a_other)` | Copy each dimension's bounds |
| `template<int DimensionT> auto get() const` | `DimenstionT` [sic] >=0 and < rank; the `bounds_t` of `DimensionT` in that 0-based `TypeListT` type |
| `lower_type lower()` | `n_index<…>` of inclusive lower bounds: `n_index[get<0>().lower()][get<1>().lower()][get<…>().lower()][get<row_dimension>().lower()]` |
| `upper_type upper()` | `n_index<…>` of exclusive upper bounds: `n_index[get<0>().upper()][get<1>().upper()][get<…>().upper()][get<row_dimension>().upper()]` |
| `template<typename... OtherTypeListT> bool contains(n_bounds_t<OtherTypeListT...> &a_other) const` | Each dimenson [sic] fully contained: `get<0>().contains(a_other.get<0>()) && … && get<row_dimension>().contains(a_other.get<row_dimension>())` |
| `template<class... OtherTypeListT> bool operator == (const n_bounds_t<OtherTypeListT...> a_other) const` | Numeric equality of all dimensions only |
| `template<class... OtherTypeListT> bool operator != (const n_bounds_t<OtherTypeListT...> a_other) const` | True if any dimension differs numerically |
| `template<class ...OtherTypeListT> auto operator+ (const n_index_t<OtherTypeListT...> a_offset) const` | Adds each dimension's offset from `a_offset` to the bounds |
| `template<int DimensionT> auto rightmost_dimensions() const` | `DimensionT` [sic] >=0 and <= rank; lower rank copy of the righmost [sic] `DimensionT` values |
| `template<class... OtherTypeListT> auto overlay_rightmost(const n_bounds_t<OtherTypeListT...> & a_other) const` | `a_other` rank <= rank; copy with rightmost dimensions overlaid from `a_other` (ontop [sic]) |

Friend: `std::ostream& operator << (std::ostream& output_stream, const n_bounds_t & a_bounds_list)`.

#### n_bounds_generator

Simple creation of `n_bounds_t`. `#include <sdlt/n_bounds.h>`

```cpp
template<typename... TypeListT> class n_bounds_generator;
namespace { n_bounds_generator<> n_bounds; }   // generator object
```

Recursively constructing `operator[]` for `bounds_t<LowerT, UpperT>`, one dimension at a time:

```cpp
auto bounds1 = n_bounds[bounds(540_fixed, 1080_fixed)][bounds(960_fixed, 1920_fixed)];
auto bounds2 = n_bounds[bounds(540, 1080)][bounds(960, 1920)];
auto bounds1b = n_bounds(n_index[540_fixed][960_fixed], n_extent[540_fixed][960_fixed]);
auto bounds2b = n_bounds(n_index[540][960], n_extent[540][960]);
```

Equivalent explicit source forms: `n_bounds_t<bounds_t<fixed<540>, fixed<1080>>, bounds_t<fixed<960>, fixed<1920>>> bounds1(...)` and `n_bounds_t<bounds_t<int, int>, bounds_t<int, int>> bounds2(...)`. Class Hierarchy: use only `n_bounds_t<...>`, from which `n_bounds_generator<…>` derives; `n_bounds` is automatically downcast. Template argument as for `n_bounds_t` (types must be `bounds_t<LowerT, UpperT>`). Member type `typedef n_bounds_t<TypeListT...> value_type`.

| Member | Description |
|---|---|
| `n_bounds_generator()` | `TypeListT` empty; no bounds specified |
| `n_bounds_generator(const n_bounds_generator &a_other)` | Copy bounds values |
| `template<typename LowerT, typename UpperT> auto operator [] (const bounds_t<LowerT, UpperT> & a_bounds) const` | Append rightmost `bounds_t<LowerT, UpperT>` dimension; returns `n_bounds_generator<TypeListT..., bounds_t< LowerT, UpperT >>` |
| `template<class... IndexTypeListT, class... ExtentTypeListT> auto operator () ( const n_index_t<IndexTypeListT...> & a_indices, const n_extent_t<ExtentTypeListT...> & a_extents) const` | `a_indices` rank == `a_extents` rank and `TypeListT` empty; lower bounds from `a_indices`, upper = `a_indices` + `a_extents` |
| `value_type value() const` | Aggregated `n_bounds_t<…>` with correct types and values |

#### bounds_d Template Function

Gets one dimension's bounds without extracting a whole `n_bounds_t<…>`. `#include <sdlt/n_extent.h>`

```cpp
template<int DimensionT, typename ObjT> auto bounds_d(const ObjT &a_obj)
```

`int DimensionT` = 0 based index from the leftmost dimension, >=0 and < `ObjT::rank`. `typename ObjT`: `ObT` [sic] is one of `n_container<…>`, `n_bounds_t<…>`, `n_bounds_generator<…>`, `n_container<…>::accessor`, `n_container<…>::const_accessor`, or any sectioned or translated accessor. Returns the correctly typed `bounds_t<LowerT, UpperT>`. Example: `auto bounds_z = bounds_d<0>(volume); auto bounds_y = bounds_d<1>(volume); auto bounds_x = bounds_d<2>(volume);` then nested `for(auto z : bounds_z) for(auto y : bounds_y) for(auto x : bounds_x) { /* … */ }`.

### Accessors

#### soa1d_container::accessor and aos1d_container::accessor

Lightweight `[]` read/write access to elements of a `soa1d_container` or `aos1d_container`. `#include <sdlt/soa1d_container.h>` and `#include <sdlt/aos1d_container.h>`

```cpp
template <typename OffsetT> soa1d_container::accessor;   template <typename OffsetT> aos1d_container::accessor;
```

`OffsetT` is the offset type applied to each `operator[]`, from the offset passed to `soa1d_container::access(offset)`/`aos1d_container::access(offset)`. `[]` returns a proxy Element that imports/exports the Primitive's data; re-accessing yields an accessor whose `[0]` is the embedded offset's index. Lightweight, passed by value into functions or lambdas, used in place of pointers.

| Member | Description |
|---|---|
| `accessor();` | Default Constructible |
| `accessor(const accessor &);` | Copy Constructible |
| `accessor & operator = (const accessor &);` | Copy Assignable |
| `const int & get_size_d1() const;` | Number of elements in the container |
| `auto operator [] (int index_d1) const` | Proxy Element at `index_d1` (source prints "container.." [sic]) |
| `template<typename IndexT_D1> auto operator [] (const IndexT_D1 index_d1);` | `IndexT_D1` an SDLT defined or generated Index type |
| `auto reaccess(const int offset) const;` | Accessor, integer-based embedded offset |
| `template<int IndexAlignmentT> auto reaccess(aligned_offset<IndexAlignmentT> offset) const;` | Accessor, `aligned_offset<IndexAlignmentT>` embedded offset |
| `template<int fixed_offsetT> auto reaccess(fixed_offset<fixed_offsetT>) const;` | Accessor, `fixed_offset<OffsetT>` embedded offset |

#### soa1d_container::const_accessor and aos1d_container::const_accessor

Lightweight `[]` read access to elements of a `soa1d_container` or `aos1d_container`. `#include <sdlt/soa1d_container.h>` and `#include <sdlt/aos1d_container.h>`

```cpp
template <typename OffsetT> soa1d_container::const_accessor;   template <typename OffsetT> aos1d_container::const_accessor;
```

`OffsetT` is the embedded offset type applied to each `operator[]`. `[]` returns a proxy const Element exporting the Primitive's data; re-accessing yields a const_accessor whose `[0]` is the embedded offset's index. Lightweight, passed by value, used in place of const pointers.

| Member | Description |
|---|---|
| `const_accessor();` | Default Constructible |
| `const_accessor(const const_accessor &);` | Copy Constructible |
| `const_accessor & operator = (const const_accessor &);` | Copy Assignable |
| `const int & get_size_d1() const;` | Number of elements in the container |
| `auto operator [] (int index_d1) const` | Proxy ConstElement at `index_d1` (source prints "atindex_d1" [sic]) |
| `template<typename IndexT_D1> auto operator [] (const IndexT_D1 index_d1);` | `IndexT_D1` an SDLT defined or generated Index type |
| `auto reaccess(const int offset) const;` | Const accessor, integer-based embedded offset |
| `template<int IndexAlignmentT> auto reaccess(aligned_offset<IndexAlignmentT> offset) const;` | Const accessor, `aligned_offset<IndexAlignmentT>` embedded offset |
| `template<int fixed_offsetT> auto reaccess(fixed_offset<fixed_offsetT>) const;` | Const accessor, `fixed_offset<OffsetT>` embedded offset |

#### Accessor Concept

Accessors from `n_container::access()` / `n_container::const_access()` read and write `n_container` cells; `accessor_concept::section(n_bounds_t<…>)`, `accessor_concept::translated_to(n_index_t<…>)`, and `accessor_concept::translated_to_zero()` derive new ones. Index values through successive `[]` calls produce a proxy concept importing/exporting the cell's primitive data. Accessors know their valid iteration space (`bound_d<int DimensionT>(accessor)` [sic]) and may carry a translation; a section restricts that space; fewer subscripts than the rank gives a lower-rank accessor, and slicing the final dimension gives a proxy to the cell.

```cpp
auto image = make_n_container<MyStruct, layout::soa>(n_extent[128][256]);   auto acc = image.access();
acc[64][128] = in_value;  MyStruct out_value = acc[64][128];  assert(out_value == in_value);
assert(bounds_d<0>(acc) == bounds(0_fixed,128));  assert(bounds_d<1>(acc) == bounds(0_fixed,256));
auto shifted_acc = acc.translated_to(n_index[1000][2000]);
assert(bounds_d<0>(shifted_acc) == bounds(1000,1128));  assert(bounds_d<1>(shifted_acc) == bounds(2000,2256));
auto subsection_acc = a.section(n_bounds[bounds(64,96)][bounds(128,160)]);
assert(bounds_d<0>(subsection_acc) == bounds(64, 96));
assert(bounds_d<1>(subsection_acc) == bounds(128, 160);   // [sic: source garbled — unbalanced parenthesis]
auto zb_sub_acc = a.section( n_bounds[bounds(64, 96)][bounds(128, 160)] ).translated_to_zero();
assert(bounds_d<0>(zb_sub_acc) == bounds(0, 32));  assert(bounds_d<1>(zb_sub_acc) == bounds(0, 32));
auto image4d = make_n_container<MyStruct, layout::soa>(n_extent[10][20][128][256]);
auto acc4d = image4d.access();  auto acc3d = acc4d[5];  auto acc2d = acc3d[10];  auto acc1d = acc2d[64];
acc1d[128] = in_value;  assert(acc4d[5][10][64][128] == in_value);
```

| Pseudo-Signature | Description |
|---|---|
| `typedef PrimitiveT primitive_type;` | Cell data type |
| `static constexpr int rank;` | Free dimensions |
| `accessor_concept(const accessor_concept &a_other)` | Copy, exact same type |
| `template<typename IndexT> element_concept operator[] (const IndexT a_index) const` | `rank == 1`; `IndexT` is `int`, `aligned<AlignmentT>`, `fixed<NumberT>`, `linear_index`, or `simd_index<LaneCountT>`; `element_concept` proxy (read only via `const_access()`) |
| `template<typename IndexT> accessor_concept operator[] (const IndexT a_index) const` | `rank > 1`; lower-rank `accessor_concept` embeding [sic] `a_index`, fixing that dimension |
| `template<int DimensionT> auto bounds_d() const` | `DimensionT` >=0 and < rank; `bounds_t` of that dimension |
| `auto bounds_dXX() const` where XX is 0-19 | XX >=0, < rank and < 20; non templated `bounds_t` of XX |
| `template<int DimensionT> auto extent_d() const` | `DimensionT` >=0 and < rank; extent of that dimension |
| `auto extent_dXX() const` where XX is 0-19 | XX >=0, < rank and < 20; non templated extent of XX |
| `template<typename ...IndexListT> accessor_concept translated_to( n_index_t<IndexListT...> a_n_index) const` | Same rank; translation so `a_n_index` corresponds [sic] to current lower bounds |
| `template<typename ...IndexListT> accessor_concept translated_to_zero() const` | Translation so `[0]` corresponds [sic] to current lower bounds |
| `template<typename ...BoundsTypeListT> auto section(const n_bounds_t<BoundsTypeListT...> &a_n_bounds) const` | Same rank, contained in current bounds; bounds set to `a_n_bounds` (effictively [sic] a section) |

### Proxy Objects

Accessors cannot return a Primitive reference (layout abstracted), so they return a Proxy that imports/exports data: accessor `[index]` yields a Proxy, const_accessor `[index]` yields a ConstProxy. The concrete Proxy type is implementation detail; the interface is uniform. Proxy objects implement a Data Member Interface: per `value_type` data member, an access method returns a Proxy or ConstProxy for just that member, allowing drill-down instead of importing/exporting the whole Primitive. They overload (when `value_type` supports it) the usual comparison, arithmetic, logical, bitwise, compound-assignment, `++`/`--`, and dereference operators, plus `*`/`+`/`-`/`!` unary forms.

#### Proxy

Access to a specific Primitive, Primitive data member, or nested data member within a Primitive for an element in a container. A `value_type` is exported via the conversion operator or by passing the Proxy to `unproxy`, and imported with `=`; operator calls export the `value_type`, apply the operator, and for assignment import the result back into the Member returning the proxy (otherwise a result is returned). Member type: `typedef implementation-defined value_type`.

| Member | Description |
|---|---|
| `operator value_type const () const;` | Exports a copy; the constant return value prevents rvalue assignment for structs |
| `const value_type & operator = (const value_type &a_value);` | Imports `a_value`; returns the same constant `value_type` (unlike operators returning `*this`, to allow assignment chaining) |
| `Proxy & operator = (const Proxy &other);` | Exports the `other` Proxy's value and imports it; returns this Proxy obect [sic] |
| `auto name_of_values_data_member_1()const;` | Proxy for the 1st `value_type` data member (method name is that member's name) |
| `auto name_of_values_data_member_2()const;` | Proxy for the 2nd `value_type` data member |
| `auto name_of_values_data_member_...()const;` | Proxy for the ...th `value_type` data member |
| `auto name_of_values_data_member_N()const;` | Proxy for the Nth `value_type` data member |

#### ConstProxy

Access to a specific constant primitive, primitive data member, or nested data member within a primitive for an element in a container. A `value_type` is exported via the conversion operator or by passing the ConstProxy to `unproxy`; comparison/arithmetic/logical/bitwise/unary operators are overloaded when `value_type` supports them (calls export, apply, return the result). Data Member Interface as above, returning Member ConstProxy objects. Member type: `typedef implementation-defined value_type`.

| Member | Description |
|---|---|
| `operator value_type const () const;` | Exports a copy; the constant return value prevents rvalue assignment for structs |
| `auto name_of_values_data_member_1()const;` | ConstProxy for the 1st `value_type` data member |
| `auto name_of_values_data_member_2()const;` | ConstProxy for the 2nd `value_type` data member |
| `auto name_of_values_data_member_...()const;` | ConstProxy for the ...th `value_type` data member |
| `auto name_of_values_data_member_N()const;` | ConstProxy for the Nth `value_type` data member |

### Number Representation

Extents, positions inside, and bounds may be numeric `fixed`, `aligned`, or `int`; fixed is most precise and `int` least, so prefer the most precise.

#### Fixed

Numerical constant whose value [is] specified at compile time [sic].

```cpp
template <int NumberT> class fixed;
```

Compile-time offsets let the compiler keep aligned access for fixed known-aligned boundaries, detect same-cache-line accesses to avoid repeated prefetch, and skip a peel loop when the iteration-space start is a SIMD lane count multiple. Use `fixed` over `aligned` or integers where possible. The library defines its own type rather than `std::integral_constant<int>` to provide overloaded operators and avoid collisions. `int Number T` [sic] = the value represented. Members: `static constexpr int value = NumberT;`, `constexpr operator value_type() const`, `constexpr value_type operator()() const;`. Constant expression `+`, `-` (unary and binary), `*`, `/` are compile-time evaluated. `_fixed` is a C++11 user-defined literal (`1080_fixed` == `fixed<1080>`): `foo3d(fixed<1080>(), fixed<1920>());` versus `foo3d(1080_fixed, 1920_fixed);`.

> NOTE: This note does not apply to SYCL. `sdlt::fixed<NumberT>` supersedes the deprecated `sdlt::fixed_offset<OffsetT>` of SDLT v1; use `sdlt::fixed<NumberT>`, though this release aliases `sdlt::fixed_offset<OffsetT>` onto it.

#### Aligned

Integer value known at compile time to be a multiple of an `IndexAlignment`.

```cpp
template <int IndexAlignmentT> class aligned;
```

Known multiples let the compiler maintain aligned access with a SIMD loop index. Internally the value becomes a block count, `block_count = value/IndexAlignmentT;`, and `value()` is `AlignmentT*block_count`, letting the compiler prove the multiple for alignment optimizations. `int IndexAlignmentT` = stated multiple alignment, must be a power of two. Member types: `typedef int value_type`; `typedef int block_type` (the `block_count` type).

| Member | Description |
|---|---|
| `static const int index_alignment` | The `IndexAlignmentT` value |
| `aligned()` | Empty (uninitialized) object |
| `explicit aligned(value_type)` | `block_count=a_value/IndexAlignmentT` |
| `aligned(const aligned& a_other)` | Copies `block_count`; same `IndexAlignmentT` required |
| `template<int OtherAlignment> explicit aligned(const aligned& other)` | Optimized, avoiding `other.value()`; `a_other` alignment < `IndexAlignmentT` and `other.value()` a multiple of it |
| `template<int OtherAlignment> aligned(const aligned& other)` | Multiply instead of divide; `a_other` alignment > `IndexAlignmentT` |
| `static aligned from_block_count(block_type block_count)` | Direct from block count, avoiding math |
| `value_type value() const` | `aligned_block_count()*IndexAlignmentT` |
| `operator value_type()` | Conversion to `int`; returns `value()` |
| `block_type aligned_block_count() const` | Conversion to `int`; the block count |

Operations: `operator *(int)` commutative (scale value, returns `aligned<IndexAlignmentT >`); `operator *(fixed<V>)` commutative (scales `IndexAlignment` by 2^M and value by K, requires `V=2^M*K`, returns `aligned<IndexAlignmentT*(2^M)>`); `operator *(aligned<OtherAl>)` (scales by `OtherAl`, returns `aligned<IndexAlignmentT*OtherAl>`); `int operator/(fixed<IndexAlignmentT>)` = `aligned_block_count()`; `int operator/(fixed<-IndexAlignmentT>)` = `-aligned_block_count();`; `int operator/(fixed<V>)` requires `abs(V)>IndexAlignmentT && IndexAlignmentT%V==0`, returns `aligned_block_count()/(V/IndexAlignmentT)`; `int operator/(fixed<V>)` requires `abs(V) < IndexAlignmentT && V %IndexAlignmentT==0`, returns `aligned_block_count()*(IndexAlignmentT/V)`; unary `operator -()` and `operator -(const aligned &) const` (same type aligned for negated/difference values); `template<int OtherAl> aligned<?> operator -(const aligned<OtherAl>&) const` and `template<int V> aligned<?> operator -(const fixed<V> &) const` (difference, alignment result depends on the relation, value uses the lower of incoming alignments); `operator +(const aligned &)const` (same type aligned sum); `template<int OtherAl> aligned<?> operator +(const aligned<OtherAl>&) const` and `template<int V> aligned<?> operator +(const fixed<V> &) const` (sum, same rules); and `template<int OtherAl> aligned operator +=(const aligned<OtherAl> &) const`, `operator -=(const aligned<OtherAl> &) const`, `operator *=(const aligned<OtherAl> &) const`, `operator /=(const aligned<OtherAl> &) const` (increment/decrement/multiply/divide when `IndexAlignmentT` is compatible with `OtherAl`, returning same-type aligned).

> NOTE: This note does not apply to SYCL. `sdlt::aligned<>` supersedes the deprecated `sdlt::aligned_offset<>` of SDLT v1; use `sdlt::aligned<>`, though this release aliases `sdlt::aligned_offset<>` onto it.

#### int

Arbitrary integer value, usable in interfaces supporting `fixed<>` and `aligned<>`; least informative and least facilitating of compiler optimizations.

#### aligned_offset

Integer based offset whose value is a multiple of an `IndexAlignment` specified at compile time. `#include <sdlt/aligned_offset.h>`

```cpp
template<int IndexAlignmentT> class aligned_offset;
```

`aligned_offset` is a deprecated feature. Known offset multiples let the compiler maintain aligned access with a SIMD loop index; internally the offset becomes a block count `Block Count = offsetValue/IndexAlignmentT;` usable by indices. `int IndexAlignmentT` = stated index alignment.

| Member | Description |
|---|---|
| `static const int IndexAlignment = IndexAlignmentT;` | Alignment the offset is a multiple of |
| `explicit aligned_offset(const int offset)` | Construct from offset |
| `static aligned_offset from_block_count(int aligned_block_count);` | Offset value = `IndexAlignment*aligned_block_count` |
| `int aligned_block_count() const;` | Number of `IndexAlignment` blocks in the offset value |
| `int value() const;` | The offset value |

#### fixed_offset

Integer based offset whose value [is] specified at compile time [sic]. `#include <sdlt/fixed_offset.h>`

```cpp
template <int OffsetT> fixed_offset;
```

`fixed_offset` is a deprecated feature. A compile-time offset lets the compiler maintain aligned access (if the offset is aligned) and detect same-cache-line accesses, avoiding repeated prefetching; prefer it over `aligned_offset` or integer offsets. `int OffsetT` = value represented. Member: `static constexpr int value = OffsetT` — offset value known at compile [time] [sic].

### Indexes

Accessor `[]` accepts an integer based loop index, but modifications may lose the fact that it is a loop index before reaching `[]`. SDLT therefore provides classes wrapping loop indexes to capture additions/subtractions of offsets (see the Offsets section), preserving the original loop index and tracking offset arithmetic for the underlying layout. Stencils commonly need offsets during data access; for a regular linear loop use `linear_index`.

#### linear_index

Wraps an integer-based loop index iterating linearly. `#include <sdlt/linear_index.h>`

```cpp
class linear_index;
```

Wrap the loop index inside a linear loop to allow addition or subtraction of offsets. Members: `explicit linear_index(int an_index);` (from a loop index), `int value() const;` (the original loop index).

#### n_index_t

Position inside the N-dimensional container: number of dimensions and index value of each.

```cpp
template<typename... TypeListT> class n_index_t
```

Indices may be compile-time known (`fixed<int NumberT>`), runtime (`int`), or runtime with guaranteed alignment multiple (`aligned<int Alignment>`). Objects identify a cell, describe the inclusive lower bounds for `n_bounds()`, and give the position for an accessor's `translated_to()`. `typename... TypeListT`: count is the number of dimensions, each type the index representation, in C++ subscript order leftmost to rightmost; type must be `int`, `fixed<NumberT>`, or `aligned<AlignmentT>`. Comparison/arithmetic members require equal ranks unless stated.

| Member | Description |
|---|---|
| `static constexpr int rank;` | Number of dimensions |
| `static constexpr int row_dimension = rank-1;` | Index of the last dimension, row |
| `n_index_t()` | Default values for extent types; every type default constructible; correctly initialized only when every type is `fixed<NumberT>` |
| `n_index_t(const n_extent_t &a_other)` | Copy each dimension's index value |
| `explicit n_index_t(const TypeListT & … a_values)` | Initialize each dimension from `a_values` (source also prints "Returns: The last extent in its native type" [sic]) |
| `template<int DimensionT> auto get() const` | `DimenstionT` [sic] >=0 and < rank; index value of `DimensionT` in that 0-based `TypeListT` type |
| `n_index_t operator +() const` | Positive unary value per dimension (no-op); copy of the instance |
| `auto operator -() const` | Negative unary value: `n_index[-get<0>()][-get<1>()][-get<…>()][-get<row_dimension>()]` |
| `template<class... OtherTypeListT> auto operator +( const n_index_t<OtherTypeListT...> & a_other) const` | Adds each dimension's index with `a_other`'s |
| `template<class... OtherTypeListT> auto operator -( const n_index_t<OtherTypeListT...> & a_other) const` | Subtracts `a_other`'s index per dimension from this instance's |
| `template<class... OtherTypeListT> bool operator == (const n_index_t<OtherTypeListT...> a_other) const` | Numeric equality of all dimensions only |
| `template<class... OtherTypeListT> bool operator != (const n_index_t<OtherTypeListT...> a_other) const` | True if any dimension differs numerically |
| `template<int DimensionT> auto rightmost_dimensions() const` | `DimenstionT` [sic] >=0 and <= rank; lower rank copy of the righmost [sic] `DimensionT` values |
| `template<class... OtherTypeListT> auto overlay_rightmost(const n_index_t<OtherTypeListT...> & a_other) const` | `a_other` rank <= rank; copy with rightmost dimensions overlaid (ontop [sic]) |

Friend: `std::ostream& operator << (std::ostream& output_stream, const n_index_t & a_indices)`.

#### n_index_generator

Simpler creation of `n_index_t` via generator object `n_index`.

```cpp
template<typename... TypeListT> class n_index_generator;
namespace { n_index_generator<> n_index; }   // generator object
```

Recursively constructing `operator[]` for `fixed<>`, `aligned<>`, and integer values, one dimension at a time:

```cpp
n_index_t<int, int> idx1(row, col);   n_index_t<int, aligned<16>> idx2(row, aligned<16>(col));
n_index_t<fixed<540>, fixed<960>> idx3(540_fixed, 960_fixed);
auto idx1 = n_index[row][col];  auto idx2 = n_index[row][aligned<16>(col)];  auto idx3 = n_index[540_fixed][960_fixed];
```

Class Hierarchy: use only `n_index_t <...>`, from which `n_index_generator < … >` derives; `n_index` is automatically downcast. Template argument `typename... TypeListT`: count is the generator's current dimensions; requirements as for `n_index_t` (`int`, `fixed<NumberT>`, or `aligned<AlignmentT>`). Member type: `typedef n_index_t<TypeListT...> value_type`.

| Member | Description |
|---|---|
| `n_index_generator ()` | `TypeListT` empty; no indices specified |
| `n_index_generator (const n_index_generator &a_other)` | Copy index values |
| `n_index_generator<TypeListT..., int> operator [] (int a_index) const` | `a_size >= 0`; append rightmost integer based index |
| `n_index_generator<TypeListT..., fixed<NumberT>> operator [] (fixed<NumberT> a_index) const` | `a_size >= 0`; append rightmost `fixed<NumberT>` index |
| `n_index_generator<TypeListT..., aligned<AlignmentT>> operator [] (aligned<AlignmentT> a_index)` | `a_size >= 0`; append rightmost `aligned<AlignmentT>` based index |
| `value_type value() const` | `n_extent_t<…>` [sic] with correct types and values |

#### index_d template function

```cpp
template<int DimensionT, typename ObjT> auto index_d(const ObjT &a_obj)
```

Gets one dimension's index without extracting a whole `n_index_t<…>`. `int DimensionT` = 0 based index from the leftmost dimension, >=0 and < `ObjT::rank`. `typename ObjT`: `ObT` [sic] is one of `n_index_t<…>`, `n_index_generator<…>`. Returns the correctly typed index. Example: `int z = index_d<0>(a_pos); int y = index_d<1>(a_pos); int x = index_d<2>(a_pos);` (`/…` in source [sic]).

### Convenience and Correctness

Include the single header `sdlt.h` for all public features, or individual feature headers (faster builds): `#include <sdlt/sdlt.h>` instead of `#include <sdlt/primitive.h>` plus `#include <sdlt/soa1d_container.h>`. Macro `SDLT_INLINE_BLOCK` encapsulates `#pragma forceinline recursive`. SDLT trusts valid template/function parameter values (conditional checks in a SIMD loop generate extra code and inhibit vectorization via multiple exit points); to verify, define `SDLT_DEBUG=1` (`-DSDLT_DEBUG=1`). If `_DEBUG` is defined and `SDLT_DEBUG` is not defined to 0 or 1, `SDLT_DEBUG` is automatically 1; when 1, every `operator[]` is bounds checked and all addresses validated for alignment — very useful for usage bugs. Macro `__SDLT_VERSION` is predefined to `2001` for conditional compilation. `std::min`/`std::max` sometimes hurt performance, so SDLT defines `min_val` and `max_val`.

#### max_val

Returns the right value if it is greater than left, otherwise the left value. `#include <sdlt/min_max_val.h>`

```cpp
template<typename T> T max_val(const T left, const T right);
```

`typename T` = type of the left and right values. `std::min`/`std::max` create conditional control flow returning references to parameters, which may cause inefficient vector code; `max_val` returns by value instead, allowing more efficient vector code, and should inline with no overhead since most algorithms don't need a reference to the inputs. Use `sdlt::max_val` in place of `std::max` inside SIMD loops. Requires `<` defined for `T`.

#### min_val

Returns the left value if the right value is greater than left, otherwise the right value. `#include <sdlt/min_max_val.h>`

```cpp
template<typename T> T min_val(const T left, const T right);
```

`typename T` = type of the left and right values. Same rationale as `max_val`: returning by value avoids the inefficient conditional control flow of `std::min`, inlines with no overhead, and is suggested inside SIMD loops. Requires `<` defined for `T`.

### Examples

#### Efficiency with Structure of Arrays Example

AoS (non-unit stride access) versus the SDLT SoA approach:

```c
#define N 1024
typedef struct RGBs { float r; float g; float b; } RGBTy;
void main() {
    RGBTy a[N];
    #pragma omp simd
    for (int k = 0; k<N; ++k) { a[k].r = k*1.5;  a[k].g = k*2.5;  a[k].b = k*3.5; }  // non-unit stride
    // ...print a[10].r/.g/.b...
}
```

That version generates 69 AVX2 instructions, dominated by `vcvtdq2ps`/`vpaddd`/`vmulps` followed by many `vextractf128`/`vmovss`/`vextractps` stores at offsets from 12 up to 192 (%rsp,%rcx or %rsp,%rax), ending `addl $16, %edx`, `cmpl $1024, %edx`, `jb ..TOP_OF_LOOP`. The SoA version uses the container plus the proxy data member interface:

```cpp
#include <sdlt/sdlt.h>
#define N 1024
typedef struct RGBs { float r; float g; float b; } RGBTy;
SDLT_PRIMITIVE(RGBTy, r, g, b)
void main() {
    sdlt::soa1d_container<RGBTy> aContainer(N);
    auto a = aContainer.access();
    #pragma omp simd
    for (int k = 0; k<N; k++) { a[k].r() = k*1.5;  a[k].g() = k*2.5;  a[k].b() = k*3.5; }  // unit-stride
    // ...print a[10].r()/.g()/.b()...
}
```

Assemply [sic] generated: 19 instructions — `vpaddd %ymm4, %ymm3, %ymm12`, `vcvtdq2ps %ymm3, %ymm7`, `vcvtdq2ps %ymm12, %ymm10`, six `vmulps`, `vmovups` to `(%r13,%rax,4)`, `(%r15,%rax,4)`, `(%rbx,%rax,4)` plus offset 32, `vpaddd %ymm4, %ymm12, %ymm3`, `addq $16, %rax`, `cmpq $1024, %rax`, `jb ..TOP_OF_LOOP`. Both look unrolled twice; the 19-vs-69 instruction reduction comes from unit stride access, and the `soa1d_container` also aligned its allocation, gaining the architectural advantages of aligned over unaligned SIMD stores.

#### Complex SDLT Primitive Construction Example

Nested primitives with an accessor in a SIMD loop:

```cpp
#include <sdlt/sdlt.h>
#define N 1024
typedef struct XYZs { float x; float y; float z; } XYZTy;
SDLT_PRIMITIVE(XYZTy, x, y, z)
typedef struct RGBs { float r; float g; float b; XYZTy w; } RGBTy;
SDLT_PRIMITIVE(RGBs, r, g, b, w)
void main() {
    sdlt::soa1d_container<RGBTy> aContainer(N);
    auto a = aContainer.access();
    #pragma omp simd
    for (int k = 0; k<N; k++) {
        RGBTy c;  c.r = k*1.5f;  c.g = k*2.5f;  c.b = k*3.5f;
        c.w.x = k*4.5f;  c.w.y = k*5.5f;  c.w.z = k*6.5f;  a[k] = c;
    }
    const RGBTy c = a[10];   // ...printf r/g/b; printf w.x/w.y/w.z...
}
```

#### Forward Dependency Example

Structure of Arrays interacting with a forward dependency:

```cpp
#include <sdlt/primitive.h>
#include <sdlt/soa1d_container.h>
#define N 1024
typedef struct RGBs { float r; float g; float b; } RGBTy;
SDLT_PRIMITIVE(RGBTy, r, g, b)
void main() {
    sdlt::soa1d_container<RGBTy> aContainer(N);      // SOA data layout (AoS would be RGBTy a[N];)
    auto a = aContainer.access();
    #pragma omp simd
    for (int k = 0; k<N; k++) { a[k].r() = k*1.5;  a[k].g() = k*2.5;  a[k].b() = k*3.5; }
    #pragma omp simd
    for (int i = 0; i<N - 1; i++) {
        sdlt::linear_index k(i);
        a[k].r() = a[k + 1].r() + k*1.5;  a[k].g() = a[k + 1].g() + k*2.5;  a[k].b() = a[k + 1].b() + k*3.5;
    }
    // ...print a[10].r()/.g()/.b()...
}
```

#### Use of Offsets and Methods on a SDLT Primitive Example

Linearized 2d stencil with embedded offsets and methods on the primitive:

```cpp
struct RGBs {
    float red; float green; float blue;
    RGBs() {}
    RGBs(const RGBs &iOther) : red(iOther.red), green(iOther.green), blue(iOther.blue) {}
    RGBs & operator =(const RGBs &iOther) { red = iOther.red; green = iOther.green; blue = iOther.blue; return *this; }
    RGBs operator + (const RGBs &iOther) const { /* red/green/blue sums */ }
    RGBs operator * (float iScalar) const { /* scaled red/green/blue */ }
};
SDLT_PRIMITIVE(RGBs, red, green, blue)
const int StencilHaloSize = 1;  const int width = 1920;  const int height = 1080;
template<typename AccessorT> void loadImageStub(AccessorT) {}   // saveImageStub likewise
void main(void) {
    const int paddedWidth = width + 2 * StencilHaloSize;  const int paddedHeight = height + 2 * StencilHaloSize;
    int elementCount = paddedWidth*paddedHeight;
    sdlt::soa1d_container<RGBs> inputImage(elementCount);  sdlt::soa1d_container<RGBs> outputImage(elementCount);
    loadImageStub(inputImage.access());
    SDLT_INLINE_BLOCK {
        const int endOfY = StencilHaloSize + height;  const int endOfX = StencilHaloSize + width;
        for (int y = StencilHaloSize; y < endOfY; ++y) {
            auto prevRow = inputImage.const_access((y - 1)*paddedWidth);   // embedded row offsets
            auto curRow = inputImage.const_access(y*paddedWidth);
            auto nextRow = inputImage.const_access((y + 1)*paddedWidth);
            auto outputRow = outputImage.access(y*paddedWidth);
            #pragma omp simd
            for (int ix = StencilHaloSize; ix < endOfX; ++ix) {
                sdlt::linear_index x(ix);
                const RGBs color1 = curRow[x - 1];  const RGBs color2 = curRow[x];  const RGBs color3 = curRow[x + 1];
                const RGBs color4 = prevRow[x];     const RGBs color5 = nextRow[x];
                // ...compiler privatizes instances and inlines the methods...
                const RGBs sumOfColors = color1 + color2 + color3 + color4 + color5;
                const RGBs averageColor = sumOfColors*(1.0f / 5.0f);
                outputRow[x] = averageColor;
            }
        }
    }
    saveImageStub(outputImage.access());
}
```

#### RGB to YUV Conversion Example

Converts a 2D image from RGB to YUV, showing how 2D SoA `n_containers` for both images improve performance.

```cpp
#include <sdlt/sdlt.h>
using namespace sdlt;
#define WIDTH 1024
#define HEIGHT 1024
struct RGBs { float r; float g; float b; };
struct YUVs {
    float y; float u; float v;
    YUVs(){ };
    YUVs& operator=(const RGBs &tmp){
        y = 0.229f * tmp.r + 0.587f * tmp.g + 0.114f * tmp.b;
        u = -0.147f * tmp.r - 0.289f * tmp.g + 0.436f * tmp.b;
        v = 0.615 * tmp.r - 0.515f * tmp.g - 0.100 * tmp.b;
        return *this;
    }
    YUVs(const RGBs &tmp){ /* same three formulas */ }
};
SDLT_PRIMITIVE(RGBs, r, g, b)
SDLT_PRIMITIVE(YUVs, y, u, v)
int main(){
    typedef layout::soa<> LayoutT;
    n_extent_t<int, int> extents(HEIGHT, WIDTH);       // SoA N-dimensional container used in 2-D context
    typedef sdlt::n_container< RGBs, LayoutT, decltype(extents) > ContainerRGB;
    typedef sdlt::n_container< YUVs, LayoutT, decltype(extents) > ContainerYUV;
    ContainerRGB inputRGB(extents);   ContainerYUV outputYUV(extents);
    auto input = inputRGB.const_access();
    auto output = outputYUV.access();
    const auto iRGB1 = bounds_d<1>(input);   // source comments say bound_d<1>(input)
    const auto iRGB0 = bounds_d<0>(input);   // source comments say bound_d<0>(input)
    for(int y = iRGB0.lower(); y < iRGB0.upper(); y++) {
        #pragma simd
        for (int x = iRGB1.lower(); x < iRGB1.upper(); x++){
            const RGBs temp1 = input[y][x];  YUVs temp2 = temp1;  output[y][x] = temp2;
        }
    }
    return 0;
}
```

### Gotchas & failure modes

- **Non-inlined calls destroy the layout advantage** (ABI needs the original layout): inline everything (`#pragma forceinline recursive`) or mark callees `#pragma omp declare simd`; `forceinline recursive` can inflate bodies past optimizability.
- **`access_by_struct` can fail to vectorize** with complex/nested primitives; `access_by_stride` has always vectorized successfully — try it when vectorization fails.
- **Containers cannot yield references**: iterators and `operator[]` return a Proxy; no `->` on iterators (they still work with STL algorithms).
- **`resize` leaves new elements unitialized**, on either 1D container.
- **Copy construction is deleted** on 1D containers; use `clone()`.
- **Multi-dimensional containers are not resizable** and have no `std::vector`-like interface.
- **Primitive requirements are mandatory**: POD only; no pointers, unions, bit fields, arrays, or `bool` members; no strongly typed enums; public members or `SDLT_PRIMITIVE_FRIEND`; nested primitives declared before the outer one.
- **Version mixing across an ABI boundary breaks**: a library exposing SDLT containers/accessors in its public ABI must match its caller's SDLT version; internal interfaces are not covered by v1→v2 compatibility.
- **`fixed_offset<>`/`aligned_offset<>` are deprecated** in v2 for `fixed<>`/`aligned<>` (a template alias still maps them in this release); both deprecated sections say so; SYCL is exempt from those two notes.
- **`aligned<IndexAlignmentT>` requires a power-of-two alignment** and the programmer must guarantee the multiple; the compiler trusts the claim.
- **`SDLT_DEBUG` gates all validation**: when 1 (`-DSDLT_DEBUG=1`, or `_DEBUG` defined and `SDLT_DEBUG` not 0/1) every `operator[]` is bounds checked and addresses validated for alignment; the source documents validation only for that state.
- **`std::min`/`std::max` in SIMD loops** return references and can induce conditional control flow; use `sdlt::min_val`/`max_val` (require `<` for `T`).
- **`section()` requires the supplied `n_bounds` contained in the accessor's current bounds** (and same rank); `translated_to()` requires the `n_index` rank to match the accessor.
- **Chaining `[]` slicing returns a lower-rank accessor**, not an element, until the final dimension is sliced — the final slice yields a proxy embedding all previously supplied indices.

### Source map

- Overview, Motivation — pp. 418–419; Version Information, Backwards Compatibility, Deprecated — pp. 419–420
- Function Calls, 1-Dimensional Containers Overview — p. 420; n-Dimensional Containers Overview, Construct an n_container — pp. 421–423
- Bounds (overview topic) — pp. 424–425; User-Level Interface, SDLT Primitives, SDLT_PRIMITIVE Macro — pp. 425–427
- soa1d_container — pp. 427–429; aos1d_container — pp. 429–432; access_by — pp. 432–433
- n_container — pp. 433–435; Layouts / sdlt::layout namespace — pp. 435–436
- Shape (n_extent_t) — pp. 436–438; n_extent_generator — pp. 438–439; make_ n_container, extent_d — pp. 440–441
- bounds_t, sdlt::bounds — pp. 441–445; n_bounds_t — pp. 445–448; n_bounds_generator — pp. 448–450; bounds_d — pp. 450–451
- accessor/const_accessor — pp. 451–453; Accessor Concept — pp. 453–456; Proxy Objects, Proxy, ConstProxy — pp. 456–459
- Number Representation, Fixed, Aligned, int — pp. 459–463; aligned_offset — pp. 463–464; fixed_offset — p. 464
- Indexes, linear_index — pp. 464–465; n_index_t — pp. 465–468; n_index_generator — pp. 468–469; index_d — p. 470
- Convenience and Correctness — pp. 470–471; max_val — pp. 471–472; min_val — p. 472
- Examples: Efficiency with Structure of Arrays — pp. 472–475; Complex SDLT Primitive Construction — pp. 475–476
- Forward Dependency — pp. 476–477; Use of Offsets and Methods on a SDLT Primitive — pp. 477–478; RGB to YUV Conversion — pp. 479–480

## Intel® C++ Class Libraries
The Intel® C++ Class Libraries enable Single-Instruction, Multiple-Data (SIMD) operations — functions abstracted from instruction extensions on Intel® processors. SIMD exploits microprocessor architecture through parallel processing: more data throughput in fewer clock cycles, for complex, computation-intensive audio, video, and graphical data bit streams.

### Class quick table
|Class|Sign|Data type|Size|Elem|Header|Instruction set|
|---|---|---|---|---|---|---|
|I64vec1|unspec|`__m64`|64|1|ivec.h|MMX™ Technology|
|I32vec2|unspec|int|32|2|ivec.h|MMX™ Technology|
|Is32vec2|signed|int|32|2|ivec.h|MMX™ Technology|
|Iu32vec2|unsigned|int|32|2|ivec.h|MMX™ Technology|
|I16vec4|unspec|short|16|4|ivec.h|MMX™ Technology|
|Is16vec4|signed|short|16|4|ivec.h|MMX™ Technology|
|Iu16vec4|unsigned|short|16|4|ivec.h|MMX™ Technology|
|I8vec8|unspec|char|8|8|ivec.h|MMX™ Technology|
|Is8vec8|signed|char|8|8|ivec.h|MMX™ Technology|
|Iu8vec8|unsigned|char|8|8|ivec.h|MMX™ Technology|
|F32vec4|unspec|float|32|4|fvec.h|Intel® SSE|
|F32vec1|unspec|float|32|1|fvec.h|Intel® SSE|
|F64vec2|unspec|double|64|2|dvec.h|Intel® SSE2|
|I128vec1|unspec|`__m128i`|128|1|dvec.h|Intel® SSE2|
|I64vec2|unspec|long int|64|2|dvec.h|Intel® SSE2|
|I32vec4|unspec|int|32|4|dvec.h|Intel® SSE2|
|Is32vec4|signed|int|32|4|dvec.h|Intel® SSE2|
|Iu32vec4|unsigned|int|32|4|dvec.h|Intel® SSE2|
|I16vec8|unspec|int|16|8|dvec.h|Intel® SSE2|
|Is16vec8|signed|int|16|8|dvec.h|Intel® SSE2|
|Iu16vec8|unsigned|int|16|8|dvec.h|Intel® SSE2|
|I8vec16|unspec|char|8|16|dvec.h|Intel® SSE2|
|Is8vec16|signed|char|8|16|dvec.h|Intel® SSE2|
|Iu8vec16|unsigned|char|8|16|dvec.h|Intel® SSE2|
|F32vec8|unspec|float|32|8|dvec.h|Intel® AVX|
|F64vec4|unspec|double|64|4|dvec.h|Intel® AVX|
|F32vec16|unspec|float|32|16|dvec.h|Intel® AVX-512|
|F64vec8|unspec|double|64|8|dvec.h|Intel® AVX-512|
|M512vec|unspec|`__m512i`|512|1|dvec.h|Intel® AVX-512|
|I32vec16|unspec|int|32|16|dvec.h|Intel® AVX-512 Foundation|
|Is32vec16|signed|int|32|16|dvec.h|Intel® AVX-512 Foundation|
|Iu32vec16|unsigned|int|32|16|dvec.h|Intel® AVX-512 Foundation|
|I64vec8|unspec|long int|64|8|dvec.h|Intel® AVX-512 Foundation|
|Is64vec8|signed|long int|64|8|dvec.h|Intel® AVX-512 Foundation|
|Iu64vec8|unsigned|long int|64|8|dvec.h|Intel® AVX-512 Foundation|
|I16vec32|unspec|int|16|32|dvec.h|Intel® AVX-512 BWI|
|Is16vec32|signed|int|16|32|dvec.h|Intel® AVX-512 BWI|
|Iu16vec32|unsigned|int|16|32|dvec.h|Intel® AVX-512 BWI|
|I8vec64|unspec|int|8|64|dvec.h|Intel® AVX-512 BWI|
|Is8vec64|signed|int|8|64|dvec.h|Intel® AVX-512 BWI|
|Iu8vec64|unsigned|int|8|64|dvec.h|Intel® AVX-512 BWI|

`unspec` = unspecified signedness; BWI = Byte and Word Instructions. Headers are cumulative: **MMX™ Technology** → `#include <ivec.h>`; **Intel® SSE** → `#include <fvec.h>`; **Intel® SSE2**, **Intel® SSE3**, **Intel® SSE4**, **Intel® AVX** → `#include <dvec.h>` — each succeeding file includes the preceding class, so `dvec.h` alone covers all classes including SSE2, and `fvec.h` is needed only for Ivec+Fvec together. Most classes contain similar functionality for all data types and are represented by all available intrinsics; some capabilities do not translate between data types without poor performance and are excluded from individual classes.

### Key facts
- `rcp`/`rsqrt` are approximating instructions with very short latencies producing results with **at least 12 bits of accuracy**; answers may differ on non-Intel processors. `rcp_nr`/`rsqrt_nr` use software refining (Newton-Raphson) for better accuracy at minimal performance cost.
- Immediate-value intrinsics that cannot be expressed easily in classes are **not implemented**: `_mm_shuffle_ps`, `_mm_shuffle_pi16`, `_mm_extract_pi16`, `_mm_insert_pi16` (`_mm_shuffle_ps` is listed twice in the source). Shuffle intrinsics can still be mixed with class code.
- **Caution:** intermixing `M64` and `M128` data types results in unexpected behavior.
- Alignment: SSE memory operations should be on **16-byte-aligned** data and AVX on **32-byte-aligned** data whenever possible. F32vec4/F64vec2 objects are aligned by default; float arrays are not.
- Debug operations map to no compiler intrinsics and are for debugging only. Out-of-range element access/assignment aborts (diagnostic printed) only when `DEBUG` is enabled.
- Intel's `valarray` implementation requires Intel® Integrated Performance Primitives (Intel® IPP), part of the product — select Intel® IPP at install. `valarray` is **not available for SYCL**.

### SIMD Data Flow
*[SIMD Data Flow — source p. 480]*
One SIMD instruction applies an element-wise operation lane-wise to two equally wide operand vectors and produces a same-width packed result: lanes `A3 A2 A1 A0` and `B3 B2 B1 B0` produce `A3opB3 A2opB2 A1opB1 A0opB0` — four operations with one instruction, an efficiency factor of four for that instruction. Such instructions can be implemented using assembly inlining, intrinsics, or the C++ SIMD classes.

Adding four single-precision floating-point values, the three interfaces side by side (the class notation is C++-like, with fewer keystrokes and lines):

|Assembly inlining|Intrinsics|SIMD Class Libraries|
|---|---|---|
|`... __m128 a,b,c;`<br>`__asm{ movaps xmm0,b`<br>`movaps xmm1,c addps`<br>`xmm0,xmm1 movaps a,`<br>`xmm0 } ...`|`#include <xmmintrin.h> ...`<br>`__m128 a,b,c; a =`<br>`_mm_add_ps(b,c); ...`|`#include <fvec.h> ...`<br>`F32vec4 a,b,c; a = b`<br>`+c; ...`|

### C++ Classes and SIMD Operations
Scalar loop form for two 4-element vectors, then the same in one operation with an integer class:

```c
int a[4], b[4], c[4];
for (i=0; i<4; i++) /* needs four iterations */
c[i] = a[i] + b[i]; /* computes c[0], c[1], c[2], c[3] */
```

```cpp
Is16vec4 ivecA, ivecB, ivec C; /*needs one iteration*/
ivecC = ivecA + ivecB; /*computes ivecC0, ivecC1, ivecC2, ivecC3 */
```

### Ivec Class Hierarchy
*[Ivec Class Hierarchy — source p. 485]*
`M64` splits into `I64vec1`, `I32vec2`, `I16vec4`, `I8vec8`; `I16vec4` further into `Is8vec8`/`Iu8vec8`, `I32vec2` into `Is16vec4`/`Iu16vec4` and `Is32vec2`/`Iu32vec2`, `I64vec1` into `Is64vec2`/`Iu64vec2`; `M128` splits into `I128vec1`, `I64vec2`, `I32vec4`, `I16vec8`, `I8vec16` with the same signed/unsigned sub-splits — the base classes are width-parameterized and the hierarchy provides signed/unsigned conversions between narrower classes. Figure label in source: OM00834.

### Terms, Syntax, and Rules
`M64`/`M128` define the `__m64`/`__m128i` types from which the other Ivec classes derive. Intermediate (first-generation) classes: `I128vec1, I64vec1, I64vec2, I32vec2, I32vec4, I16vec4, I16vec8, I8vec8, I8vec16` (element sizes 128/64/32/16/8 bits). Second generation (signedness): `Is64vec2, Iu64vec2, Is32vec2, Iu32vec2, Is32vec4, Iu32vec4, Is16vec4, Iu16vec4, Is16vec8, Iu16vec8, Is8vec8, Iu8vec8, Is8vec16, Iu8vec16`. **Caution** Intermixing the M64 and M128 data types will result in unexpected behavior.

Class names follow `<type><signedness><bits>vec<elements>` = `{ F | I } { s | u } { 128 | 64 | 32 | 16 | 8 } vec { 16 | 8 | 4 | 2 | 1 }`: *type* `F` (floating point) or `I` (integer); *signedness* `s`/`u`, blank for Ivec intermediate classes and always blank for Fvec (no unsigned Fvec classes); *bits* per element; *elements* count. **Nearest Common Ancestor** = intermediate/parent class of two same-size classes (`Iu8vec8`+`Is8vec8` → `I8vec8`; `Iu8vec8`+`I16vec4` → `M64`). **Casting (typecast)** converts one class to another when an operation mixes data types — occasionally automatic, otherwise written explicitly. **Operator Overloading** applies a class's accepted operators to declared variables under the typecasting rules in the header files.

**Rules for Operators.** `[ operator ]` is an operator (e.g. `&`, `|`, `^`); `[ Ivec_Class ]` any Ivec class; `R, A, B` declared Ivec variables. Three conventions: `[ Ivec_Class ] R = [ Ivec_Class ] A [ operator ][ Ivec_Class ] B` (e.g. `I64vec1 R = I64vec1 A & I64vec1 B;`); `[ Ivec_Class ] R =[ operator ] ([ Ivec_Class ] A,[ Ivec_Class ] B)` (e.g. `I64vec1 R = andnot(I64vec1 A, I64vec1 B);`); `[ Ivec_Class ] R [ operator ]= [ Ivec_Class ] A` (e.g. `I64vec1 R &= I64vec1 A;`). Fvec classes use the same three with `[Fvec_Class]` (`F64vec2`/`F32vec4`/`F32vec1`): `F64vec2 R = F64vec2 A & F64vec2 B;` / `F64vec2 R = andnot(F64vec2 A, F64vec2 B);` / `F64vec2 R &= F64vec2 A;` *Explicit* below = illegal to mix types without an explicit typecast; *Automatic* = the compiler typecasts.

|Operators|Sign Typecasting|Size Typecasting|Other Requirements|
|---|---|---|---|
|Assignment|N/A|N/A|N/A|
|Logical|Automatic|Automatic (to left)|Explicit typecasting required for different types used in non-logical expressions on the right side of the assignment.|
|Addition and Subtraction|Automatic|Explicit|N/A|
|Multiplication|Automatic|Explicit|N/A|
|Shift|Automatic|Explicit|Casting required to ensure arithmetic shift.|
|Compare|Automatic|Explicit|Explicit casting required for signed classes for less-than/greater-than.|
|Conditional Select|Automatic|Explicit|Explicit casting required for signed classes for less-than/greater-than.|

**Data Declaration and Initialization** (most significant element on the left, least significant on the right):

|Operation|Class|Syntax|
|---|---|---|
|Declaration|M128|`I128vec1 A; Iu8vec16 A;`|
|Declaration|M64|`I64vec1 A; Iu8vec8 A;`|
|`__m128` init|M128|`I128vec1 A(__m128 m); Iu16vec8(__m128 m);`|
|`__m64` init|M64|`I64vec1 A(__m64 m); Iu8vec8 A(__m64 m);`|
|`__int64` init|M64|`I64vec1 A = __int64 m; Iu8vec8 A = __int64 m;`|
|`int i` init|M64|`I64vec1 A = int i; Iu8vec8 A = int i;`|
|int init|I32vec2|`I32vec2 A(int A1, int A0);` `I[s\|u]32vec2 A(…signed/unsigned int A1, A0);`|
|int init|I32vec4|`I32vec4 A(int A3, int A2, int A1, int A0);` `I[s\|u]32vec4 A(…signed/unsigned int A3, ..., A0);`|
|short init|I16vec4|`I16vec4 A(short A3, short A2, short A1, short A0);` `I[s\|u]16vec4 A(…signed/unsigned short A3, ..., A0);`|
|short init|I16vec8|`I16vec8 A(short A7, short A6, ..., short A1, short A0);` `I[s\|u]16vec8 A(signed A7, ..., A0);`|
|char init|I8vec8|`I8vec8 A(char A7, char A6, ..., char A1, char A0);` `I[s\|u]8vec8 A(…signed/unsigned char A7, ..., A0);`|
|char init|I8vec16|`I8vec16 A(char A15, ..., char A0);` `I[s\|u]8vec16 A(…signed/unsigned char A15, ..., A0);`|

Any Ivec object can be assigned to any other Ivec object; conversion on assignment is automatic:

```cpp
Is16vec4 A; Is8vec8 B; I64vec1 C;
A = B; /* assign Is8vec8 to Is16vec4 */
B = C; /* assign I64vec1 to Is8vec8 */
B = A & C; /* assign M64 result of '&' to Is8vec8 */
```

### Integer Vector Classes — operator semantics
Operator symbols, syntaxes, return types and intrinsics for every family are in **Classes Quick Reference**; the per-family rules follow. *Logical:* A and B are converted to `M64` if needed; same-class operands return the same type, differing classes the nearest common ancestor (`I32vec2 R = Is32vec2 A ^ Iu32vec2 B;`); with assignment the return type is always the pre-declared type of R and the right side may be any `I[s|u][N]vec[N] A;`.

```cpp
I64vec1 A; Is8vec8 B; Iu8vec8 C;
C = A & B; /* A and B converted to M64, result assigned to Iu8vec8 */
C = Iu8vec8(A&B)+ C; /* A&B returns M64, cast to Iu8vec8 */
```

*Addition/subtraction:* return the nearest common ancestor when right-side operands have different signs; with assignment the left operand's type wins and both operands must be the same size, else an explicit typecast is required.

```cpp
Is16vec4 A; Iu16vec4 B; I16vec4 C;
C = A + B;      /* nearest common ancestor I16vec4 */
A += B; B -= A; /* returns left-hand operand type */
Is16vec4 A,C; Iu32vec24 B;
C = A + (Is16vec4)B; /* explicit conversion */
```

*Multiplication:* accepted/returned types only `I[s|u]16vec4` or `I[s|u]16vec8`; operands must be 16 bits in size, else explicit typecast; `mul_high`/`mul_add` take `Is16vec4` data only; with assignment all operands must be 16 bytes. Mapping: `I16vec4 R` = `I[s|u]16vec4`×same; `I16vec8 R` = `I[s|u]16vec8`×same; `Is16vec4 R`/`Is16vec8` via `mul_add`; `Is32vec2 R` = `Is16vec4` `mul_high` `Is16vec4`; `Is32vec4 R` = `s16vec8` `[sic: source garbled]` `mul_high` `Is16vec8`.

```cpp
Is16vec4 A,C; Iu32vec2 B;
C = A * C;
C = A * (Is16vec4)B; /* explicit conversion */
Is16vec4 A,B,C,D;
C = mul_high(A,B); D = mul_add(A,B);
```

*Shift:* the right shift argument can be any integer or Ivec value, implicitly converted to `M64`; the left operand of `<<` can be any type except `I[s|u]8vec[8|16]`; signed types use arithmetic right shifts, unsigned and intermediate classes logical shifts; the return type is set by the first argument type (I64vec1, I32vec2, Is32vec2, Iu32vec2, I16vec4, Is16vec4, Iu16vec4).

```cpp
Is16vec4 A,C; Iu32vec2 B;
C = A; /* automatic size and sign conversion */
Is16vec4 A, C; Iu16vec4 B, R;
R = (Iu16vec4)(A & B) C; /* cast ensures logical shift, not arithmetic */
R = (Is16vec4)(A & B) C; /* cast ensures arithmetic shift, not logical */
```

*Comparison:* equality/inequality operands may have mixed signedness but must be the same size; less-than and greater-than must be the same sign and size. Overloading: `cmpeq`/`cmpne` need `I[s|u]32vec2` → I32vec2 R, `I[s|u]16vec4` → I16vec4 R, `I[s|u]8vec8` → I8vec8 R; `cmpgt`/`cmpge`/`cmplt`/`cmple` need signed `Is32vec2` → I32vec2 R, `Is16vec4` → I16vec4 R, `Is8vec8` → I8vec8 R.

```cpp
Iu8vec8 A; Is8vec8 B; I8vec8 C;
C = cmpneq(A,B); /* nearest common ancestor returned */
Iu8vec8 A, C; Is16vec4 B;
C = cmpeq(A,(Iu8vec8)B); /* cast needed for different sizes */
Iu16vec4 A; Is16vec4 B, C;
C = cmpge((Is16vec4)A,B); C = cmpgt(B,C); /* cast needed for sign/size */
```

*Conditional select:* all operands must be the same size; the return type is the nearest common ancestor of operands C and D, and A/B must be signed for greater-than/less-than forms. Overloading: `select_eq`/`select_ne` take `I[s|u]32vec2` → I32vec2 R, `I[s|u]16vec4` → I16vec4 R, `I[s|u]8vec8` → I8vec8 R; `select_gt`/`select_ge`/`select_lt`/`select_le` take signed `Is32vec2`/`Is16vec4`/`Is8vec8` for A, B, C, D. Return mapping R0…R7 (any number of elements; also applies when fewer than four return values): `R0 := (A0 op B0) ? C0 : D0;` through `R7 := (A0 op B0) ? C7 : D7;` with `op` one of `==` `!=` `>` `>=` `<` `<=`. Example `I16vec4 R = select_neq(Is16vec4, Is16vec4, Is16vec4, Iu16vec4);`

*Unpack:* `unpack_high`/`unpack_low` take A and B of the class (signed/unsigned variants too), e.g. `I64vec2 unpack_high(I64vec2 A, I64vec2 B);`, `Is8vec16 unpack_low(Is8vec16 A, Is8vec16 B);`. High-half: I64vec2/I32vec2/I32vec4 `R0=A1; R1=B1;` (I32vec4 adds `R2=A2; R3=B2;`); I16vec8/I16vec4 `R0=A2; R1=B2; R2=A3; R3=B3;`; I8vec8 `R0=A4; R1=B4; R2=A5; R3=B5; R4=A6; R5=B6; R6=A7; R7=B7;`; I8vec16 garbled (`R8=A12; R8=B12; R2=A13; R3=B13; R4=A14; R5=B14; R6=A15; R7=B15;`) `[sic: source garbled]`. Low-half: I64vec2/I32vec4 `R0=A0; R1=B0; R2=A1; R3=B1;`; I32vec2 `R0=A0; R1=B0;`; I16vec4 `R0=A0; R1=B0; R2=A1; R3=B1;`; I16vec8/I8vec8 `R0=A0; R1=B0; R2=A1; R3=B1; R4=A2; R5=B2; R6=A3; R7=B3;`; I8vec16 adds `R8=A4; R9=B4; R10=A5; R11=B5; R12=A6; R13=B6; R14=A7; R15=B7;`. Intrinsics are in the Quick Reference.

*Pack:* `Is16vec8 pack_sat(Is32vec2,Is32vec2)` eight 32-bit→eight 16-bit signed; `Is16vec4 pack_sat(Is32vec2,Is32vec2)` four 32-bit→eight 16-bit signed; `Is8vec16 pack_sat(Is16vec4,Is16vec4)` sixteen 16-bit→sixteen 8-bit signed; `Is8vec8 pack_sat(Is16vec4,Is16vec4)` eight 16-bit→eight 8-bit signed; `Iu8vec16 packu_sat(Is16vec4,Is16vec4)` and `Iu8vec8 packu_sat(Is16vec4,Is16vec4)` unsigned saturation.

*Clear MMX™ state operator:* empty the MMX™ registers and clear the MMX state; follow the guidelines for the `EMMS` instruction intrinsic. `void empty(void);` — intrinsic `_mm_empty`.

*Debug operations (Ivec):* no corresponding intrinsics. Output via `cout`, default decimal (`cout << hex << …` selects hex): `Is32vec4 A` → `"[3]:A3 [2]:A2 [1]:A1 [0]:A0"`; `Iu32vec2 A` → `"[1]:A1 [0]:A0"`; `Iu16vec8 A` → `"[7]:A7 [6]:A6 [5]:A5 [4]:A4 [3]:A3 [2]:A2 [1]:A1 [0]:A0"`; `Iu16vec4 A` → `"[3]:A3 [2]:A2 [1]:A1 [0]:A0"`; `Iu8vec16 A` → `"[15]:A15 … [0]:A0"` (all 16 lanes); `Iu8vec8 A` → `"[7]:A7 … [0]:A0"`. Element access `A[i]` returns the element type: `int` for `Is64vec2`/`Is32vec4`/`Is32vec2`, `unsigned int` for `Iu64vec2`/`Iu32vec4`/`Iu32vec2`, `short` for `Is16vec8`/`Is16vec4`, `unsigned short` for `Iu16vec8`/`Iu16vec4`, `signed char` for `Is8vec16`/`Is8vec8`, `unsigned char` for `Iu8vec16`/`Iu8vec8`; `A[i] = <typed R>` uses the same type pairs. With `DEBUG` enabled, out-of-range access or assignment prints a diagnostic and aborts.

### Integer Functions for Intel® Streaming SIMD Extensions
Requires `#include <fvec.h>`.

|Operation|Signature|Intrinsic|
|---|---|---|
|Element-wise maximum of the respective signed integer words in A and B|`Is16vec4 simd_max(Is16vec4 A, Is16vec4 B);`|`_mm_max_pi16`|
|Element-wise minimum of the respective signed integer words in A and B|`Is16vec4 simd_min(Is16vec4 A, Is16vec4 B);`|`_mm_min_pi16`|
|Element-wise maximum of the respective unsigned bytes in A and B|`Iu8vec8 simd_max(Iu8vec8 A, Iu8vec8 B);`|`_mm_max_pu8`|
|Element-wise minimum of the respective unsigned bytes in A and B|`Iu8vec8 simd_min(Iu8vec8 A, Iu8vec8 B);`|`_mm_min_pu8`|
|Create an 8-bit mask from the most significant bits of the bytes in A|`int move_mask(I8vec8 A);`|`_mm_movemask_pi8`|
|Conditionally store byte elements of A to address p; the high bit of each byte in selector B determines whether the corresponding byte in A is stored|`void mask_move(I8vec8 A, I8vec8 B, signed char *p);`|`_mm_maskmove_si64`|
|Store the data in A to address p without polluting the caches; A can be any Ivec type|`void store_nta(__m64 *p, M64 A);`|`_mm_stream_pi`|
|Element-wise average of the respective unsigned 8-bit integers in A and B|`Iu8vec8 simd_avg(Iu8vec8 A, Iu8vec8 B);`|`_mm_avg_pu8`|
|Element-wise average of the respective unsigned 16-bit integers in A and B|`Iu16vec4 simd_avg(Iu16vec4 A, Iu16vec4 B)`|`_mm_avg_pu16`|

**Conversions between Fvec and Ivec.** `int F64vec2ToInt(F64vec42 A);` `[sic: source garbled]` → `r := (int)A0;`. `F64vec2 F32vec4ToF64vec2(F32vec4 A);` → `r0 := (double)A0; r1 := (double)A1;`. `F32vec4 F64vec2ToF32vec4(F64vec2 A);` → `r0 := (float)A0; r1 := (float)A1;`. `F64vec2 InttoF64vec2(F64vec2 A, int B);` → `r0 := (double)B; r1 := A1;`. `int F32vec4ToInt(F32vec4 A);` → `r := (int)A0;`. `Is32vec2 F32vec4ToIs32vec2 (F32vec4 A);` → `r0 := (int)A0; r1 := (int)A1;`. `F32vec4 IntToF32vec4(F32vec4 A, int B);` → `r0 := (float)B; r1 := A1; r2 := A2; r3 := A3;`. `F32vec4 Is32vec2ToF32vec4(F32vec4 A, Is32vec2 B);` → `r0 := (float)B0; r1 := (float)B1; r2 := A2; r3 := A3;`. Conversion operations are intrinsics only — no classes correspond (intrinsics: `_mm_cvttsd_si32`, `_mm_cvtps_pd`, `_mm_cvtpd_ps`, `_mm_cvtsi32_sd`, `_mm_cvtt_ss2si`, `_mm_cvttps_pi32`, `_mm_cvtsi32_ss`, `_mm_cvtpi32_ps`).

### Floating-Point Vector Classes
`F64vec2`, `F32vec4`, `F32vec1`: `F64vec2 A(double x, double y);`, `F32vec4 A(float z, float y, float x, float w);`, `F32vec1 B(float w);` Packed floating-point input values are represented with the right-most value lowest. F32vec4 returns four values (R0…R3), F64vec2 two, F32vec1 the lowest only: `R0 := A0 & B0;` / `R0 := A0 andnot B0;` / `R0 &= A0;` are valid for all three classes, the R1 line additionally for F32vec4 and F64vec2 (N/A for F32vec1), R2/R3 lines for F32vec4 only (N/A elsewhere). `R3 := A3 andhot B3;` `[sic: source garbled]`.

Single-Precision Floating-Point Elements

*[Packed floating-point operand/return layout — source p. 507]*
Operands `A3 A2 A1 A0` (low value in A0) and `B3 B2 B1 B0` map lane-wise to return value `R3 R2 R1 R0`, bit range 0…127. `F32vec4` returns four packed single-precision floats (R0,R1,R2,R3); `F32vec2` returns one single-precision float (R0) `[sic: F32vec2 is not among the documented Fvec classes; source caption]`.

**Alignment.** SSE memory operations should be performed on 16-byte-aligned data whenever possible; AVX on 32-byte-aligned data. F32vec4/F64vec2 objects are properly aligned by default; floating point arrays are not automatically aligned. Use the alignment `__declspec` for 16-byte alignment:

```cpp
__declspec( align(16) ) float A[4];
```

**Conversions.** All Fvec object variables can be implicitly converted to `__m128` data types:

```cpp
__m128d mm = A & B; /* where A,B are F64vec2 object variables */
__m128 mm = A & B; /* where A,B are F32vec4 object variables */
__m128 mm = A & B; /* where A,B are F32vec1 object variables */
```

**Constructors and initialization.**

|Example|Intrinsic|Returns|
|---|---|---|
|`F64vec2 A;` `F32vec4 B;` `F32vec1 C;`|N/A|N/A|
|`F64vec2 A(__m128d mm);` `F32vec4 B(__m128 mm);` `F32vec1 C(__m128 mm);`|N/A|N/A|
|`F64vec2 A(double d0, double d1);` (`= F64vec2(double d0, double d1)` equivalent)|`_mm_set_pd`|`A0 := d0; A1 := d1;`|
|`F64vec2 A(double d0);`|`_mm_set1_pd`|`A0 := d0; A1 := d0;`|
|`F32vec4 A(float f3, float f2, float f1, float f0);` (`= F32vec4(...)` equivalent)|`_mm_set_ps`|`A0 := f0; A1 := f1; A2 := f2; A3 := f3;`|
|`F32vec4 A(float f0);`|`_mm_set1_ps`|`A0 := f0; A1 := f0; A2 := f0; A3 := f0;`|
|`F32vec4 A(double d0);`|`_mm_set1_ps(d)`|`A0 := d0; A1 := d0; A2 := d0; A3 := d0;`|
|`F32vec1 A(double d0);`|`_mm_set_ss(d)`|`A0 := d0; A1 := 0; A2 := 0; A3 := 0;`|
|`F32vec1 B(float f0);`|`_mm_set_ss`|`B0 := f0; B1 := 0; B2 := 0; B3 := 0;`|
|`F32vec1 B(int I);`|`_mm_cvtsi32_ss`|`B0 := f0; B1 := {} B2 := {} B3 := {}` `[sic: source garbled]`|

**Arithmetic operators.** Standard `+`/`+=` (`R = A + B;` `R += A;`), `-`/`-=` (`R = A - B;` `R -= A;`), `*`/`*=` (`R = A * B;` `R *= A;`), `/`/`/=` (`R = A / B;` `R /= A;`); advanced `sqrt` (`R = sqrt(A);`), `rcp`/`rcp_nr` (`R = rcp(A);` `R = rcp_nr(A);`), `rsqrt`/`rsqrt_nr` (`R = rsqrt(A);` `R = rsqrt_nr(A);`), `add_horizontal` (`float f = add_horizontal(F32vec4 A);`, `double d = add_horizontal(F64vec2 A);`). Intrinsics as F32vec4/F64vec2/F32vec1: add `_mm_add_ps`/`_mm_add_pd`/`_mm_add_ss`; sub `_mm_sub_ps`/`_mm_sub_pd`/`_mm_sub_ss`; mul `_mm_mul_ps`/`_mm_mul_pd`/`_mm_mul_ss`; div `_mm_div_ps`/`_mm_div_pd`/`_mm_div_ss`; sqrt `_mm_sqrt_ps`/`_mm_sqrt_pd`/`_mm_sqrt_ss`; rcp `_mm_rcp_ps`/`_mm_rcp_pd`/`_mm_rcp_ss`; rsqrt `_mm_rsqrt_ps`/`_mm_rsqrt_pd`/`_mm_rsqrt_ss`; `rcp_nr` `_mm_sub_ps` `_mm_add_ps` `_mm_mul_ps` `_mm_rcp_ps` / `_mm_sub_pd` `_mm_add_pd` `_mm_mul_pd` `_mm_rcp_pd` / `_mm_sub_ss` `_mm_add_ss` `_mm_mul_ss` `_mm_rcp_ss`; `rsqrt_nr` `_mm_sub_pd` `[sic: source garbled]` `_mm_mul_pd` `[sic: source garbled]` `_mm_rsqrt_ps` / `_mm_sub_pd` `_mm_mul_pd` `_mm_rsqrt_pd` / `_mm_sub_ss` `_mm_mul_ss` `_mm_rsqrt_ss`; `add_horizontal` `_mm_add_ss` `_mm_shuffle_ss` / `_mm_add_sd` `_mm_shuffle_sd`. Return mapping (`+ - * /` and `+= -= *= /=`): R0 all three classes, R1 F32vec4+F64vec2 (N/A F32vec1), R2/R3 F32vec4 only. Advanced mapping: `R0 :=` with `sqrt rcp rsqrt rcp_nr rsqrt_nr` all three; `R1 :=` F32vec4+F64vec2; `R2 :=`/`R3 :=` F32vec4 only; `f := add_horizontal (A0 + A1 + A2 + A3)` F32vec4, `d := add_horizontal (A0 + A1)` F64vec2.

**Minimum and maximum operators** (element-wise `R<i> := min/max(A<i>,B<i>)`): `F64vec2 R = simd_min(F64vec2 A, F64vec2 B)` (`_mm_min_pd`); `F32vec4 R = simd_min(F32vec4 A, F32vec4 B)` (`_mm_min_ps`); `F32vec1 R = simd_min(F32vec1 A, F32vec1 B)` with `R0 := min(A0,B0);` (`_mm_min_ss`); `F64vec2 simd_max(F64vec2 A, F64vec2 B)` (`_mm_max_pd`); `F32vec4 R = simd_man(F32vec4 A, F32vec4 B)` `[sic: source garbled]` (`_mm_max_ps`); `F32vec1 simd_max(F32vec1 A, F32vec1 B)` (`_mm_max_ss`).

**Logical operators (Fvec).** AND `&` `&=`, OR `|` `|=`, XOR `^` `^=`, and `andnot` (`R = andnot(A);`). Intrinsics F32vec4/F64vec2/F32vec1: AND `_mm_and_ps`/`_mm_and_pd`/`_mm_and_ps`; OR `_mm_or_ps`/`_mm_or_pd`/`_mm_or_ps`; XOR `_mm_xor_ps`/`_mm_xor_pd`/`_mm_xor_ps`; ANDNOT N/A/`_mm_andnot_pd`/N/A. F32vec1 uses only the lower 32 bits and has no corresponding scalar intrinsic — it accesses the lower 32 bits of the packed vector intrinsics.

**Compare operators (Fvec).** Compare single-precision floating-point values of A and B; any Fvec class comparison returns the same class compared. Mask `0xffffffff` where true, `0x00000000` where false. Operators `cmpeq`, `cmpneq`, `cmpgt`, `cmpge`, `cmpngt`, `cmpnge`, `cmplt`, `cmple`, `cmpnlt`, `cmpnle` (syntax `R = cmp<op>(A, B)`); intrinsics are `_mm_cmp<op>_` + suffix (`ps` F32vec4, `pd` F64vec2, `ss` F32vec1) for each op: `_mm_cmpeq_*`, `_mm_cmpneq_*`, `_mm_cmpgt_*`, `_mm_cmpge_*`, `_mm_cmpngt_*`, `_mm_cmpnge_*`, `_mm_cmplt_*`, `_mm_cmple_*` (source shows `_mm_cmple_pd` for F32vec1 — `[sic: source garbled]`), `_mm_cmpnlt_*`, `_mm_cmpnle_*`. Return mapping R0…R3: R0 all three, R1 F32vec4+F64vec2, R2/R3 F32vec4 only; the source's R0–R2 A-column and B-parenthesis text is garbled — `[sic: source garbled]`.

**Conditional select operators (Fvec).** Each function compares single-precision floating-point values of A and B; C and D supply the return value (C if true, D if false); any Fvec class comparison returns the same class. Operators `select_eq`, `select_neq`, `select_gt`, `select_ge`, `select_lt`, `select_le`, `select_nlt`, `select_nle` (the source also lists "Not Greater Than" as `select_gt` and "Not Greater Than or Equal To" as `select_ge` — `[sic: source garbled]`); intrinsics use the same `_mm_cmp<op>_`+suffix stems as Compare, except Less Than or Equal To, shown as `_mm_cmple_pd` for F32vec1 and `_mm_cmple_ps` for F32vec1 elsewhere (source — `[sic: source garbled]`). Return mapping: `R0:=` all three classes; `R1:=` F32vec4+F64vec2; `R2:=`/`R3:=` F32vec4 only (row text garbled in source — `[sic: source garbled]`).

**Cacheability, load/store, unpack, move mask.** `void store_nta(double *p, F64vec2 A);` (`_mm_stream_pd`) and `void store_nta(float *p, F32vec4 A);` (`_mm_stream_ps`) store non-temporally A's values and require a 16-byte aligned address. `void loadu(F64vec2 A, double *p)` (`_mm_loadu_pd`), `void storeu(float *p, F64vec2 A);` (`_mm_storeu_pd`), `void loadu(F32vec4 A, double *p)` (`_mm_loadu_ps`), `void storeu(float *p, F32vec4 A);` (`_mm_storeu_ps`) — no alignment assumption. Unpack: `F64vec2 R = unpack_low(F64vec2 A, F64vec2 B);` (`_mm_unpacklo_pd(a, b)`), `F64vec2 R = unpack_high(F64vec2 A, F64vec2 B);` (`_mm_unpackhi_pd(a, b)`), `F32vec4 R = unpack_low(F32vec4 A, F32vec4 B);` (`_mm_unpacklo_ps(a, b)`), `F32vec4 R = unpack_high(F32vec4 A F32vec4 B);` (`_mm_unpackhi_ps(a, b)`). Move mask: `int i = move_mask(F64vec2 A)` with `i := sign(a1)<<1 | sign(a0)<<0` (`_mm_movemask_pd`); `int i = move_mask(F32vec4 A)` with `i := sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)<<0` (`_mm_movemask_ps`).

**Debug operations (Fvec).** No corresponding intrinsics. `cout << F64vec2 A;` prints `"[1]:A1 [0]:A0"`; `cout << F32vec4 A;` prints `"[3]:A3 [2]:A2 [1]:A1 [0]:A0"`; `cout << F32vec1 A;` prints the lowest single-precision value. Element access `double d = F64vec2 A[int i]` permits `i` = 0 and 1; `float f = F32vec4 A[int i]` permits `i` = 0, 1, 2, 3. Element assignment `F64vec4 A[int i] = double d;` permits 0 and 1; `F32vec4 A[int i] = float f;` permits 0–3. With `DEBUG` enabled, an out-of-range index prints a diagnostic and aborts.

### Classes Quick Reference
Operators, intrinsics and implementing classes (canonical table). **N/A** = operator not implemented in that class (e.g. Andnot in F32vec4/F32vec1). Any other entry is the **suffix** substituted for `[x]`/`[y]`: `_mm_add_[x]` with `epi16` for I16vec8 means `_mm_add_epi16`.

|Symbol(s)|Intrinsic|Integer suffixes / classes|Fvec suffixes (F64vec2 / F32vec4 / F32vec1)|
|---|---|---|---|
|---|---|---|---|
|`&`, `&=`|`_mm_and_[x]`|`si128` 128-bit classes; `si64` MMX™ classes|`pd` / `ps` / `ps`|
|`|`, `|=`|`_mm_or_[x]`|same as AND|`pd` / `ps` / `ps`|
|`^`, `^=`|`_mm_xor_[x]`|same as AND|`pd` / `ps` / `ps`|
|`andnot`|`_mm_andnot_[x]`|same as AND|`pd` / N/A / N/A|
|`+`, `+=`|`_mm_add_[x]`|`epi64` I64vec2, `epi32` I32vec4, `epi16` I16vec8, `epi8` I8vec16; `pi32` I32vec2, `pi16` I16vec4, `pi8` I8vec8|`pd` / `ps` / `ss`|
|`-`, `-=`|`_mm_sub_[x]`|same as Add|`pd` / `ps` / `ss`|
|`*`, `*=`|`_mm_mullo_[x]`|only `epi16` (I16vec8), `pi16` (I16vec4); N/A elsewhere|`pd` / `ps` / `ss`|
|`/`, `/=`|`_mm_div_[x]`|N/A all integer classes|`pd` / `ps` / `ss`|
|`mul_high`|`_mm_mulhi_[x]`|only `epi16` (I16vec8), `pi16` (I16vec4)|N/A / N/A / N/A|
|`mul_add`|`_mm_madd_[x]`|only `epi16` (I16vec8), `pi16` (I16vec4)|N/A / N/A / N/A|
|`sqrt` `rcp` `rsqrt`|`_mm_sqrt_[x]`, `_mm_rcp_[x]`, `_mm_rsqrt_[x]`|N/A all integer classes|`pd` / `ps` / `ss`|
|`rcp_nr` `rsqrt_nr`|`_mm_rcp_[x]`+`_mm_add_[x]` `_mm_sub_[x]` `_mm_mul_[x]`; `_mm_rsqrt_[x]`+`_mm_sub_[x]` `_mm_mul_[x]`|N/A all integer classes|`pd` / `ps` / `ss`|
|`>>`,`>>=`|`_mm_srl_[x]` `_mm_srli_[x]` `_mm_sra__[x]` `_mm_srai_[x]`|`epi64` I64vec2, `epi32` I32vec4, `epi16` I16vec8, I8vec16 N/A, I128vec1 N/A; `_mm_sra__`/`_mm_srai_` N/A I64vec2; `si64` I64vec1, `pi32` I32vec2, `pi16` I16vec4, I8vec8 N/A; `_mm_sra__`/`_mm_srai_` N/A I64vec1|—|
|`<<`,`<<=`|`_mm_sll_[x]` `_mm_slli_[x]`|same columns as shift right (I128vec1 N/A; `si64` I64vec1, `pi32` I32vec2, `pi16` I16vec4, I8vec8 N/A)|—|
|`cmpeq` `cmpneq` `cmpgt` `cmpge` `cmplt` `cmple` `cmpngt`|`_mm_cmpeq_[x]`, `_mm_cmpgt_[x]`, `_mm_cmpge_[x]`, `_mm_cmplt_[x]`, `_mm_cmple_[x]`, `_mm_cmpngt_[x]`|`epi32` I32vec4, `epi16` I16vec8, `epi8` I8vec16; `pi32` I32vec2, `pi16` I16vec4, `pi8` I8vec8; `cmpneq`/`cmpge`/`cmple` add `_mm_andnot_[y]*` with `si128`/`si64`|`pd` / `ps` / `ss`|
|`cmpnge` `cmpnlt` `cmpnle`|`_mm_cmpnge_[x]` `_mm_cmpnlt_[x]` `_mm_cmpnle_[x]`|N/A all integer classes|`pd` / `ps` / `ss`|
|`select_eq` `select_neq` `select_gt` `select_ge` `select_lt` `select_le`|`_mm_cmpeq_[x]`/`_mm_cmpgt_[x]`/`_mm_cmpge_[x]`/`_mm_cmplt_[x]`/`_mm_cmple_[x]` plus `_mm_and_[y]`, `_mm_andnot_[y]*`, `_mm_or_[y]`|`epi32` I32vec4, `epi16` I16vec8, `epi8` I8vec16; `pi32` I32vec2, `pi16` I16vec4, `pi8` I8vec8; `[y]` = `si128`/`si64`|`pd` / `ps` / `ss`|
|`select_ngt` `select_nge` `select_nlt` `select_nle`|`_mm_cmpgt_[x]` `_mm_cmpge_[x]` `_mm_cmplt_[x]` `_mm_cmple_[x]`|N/A all integer classes|`pd` / `ps` / `ss`|
|`unpack_high`|`_mm_unpackhi_[x]`|`epi64` I64vec2, `epi32` I32vec4, `epi16` I16vec8, `epi8` I8vec16, `pi32` I32vec2; `pi16` I16vec4, `pi8` I8vec8|`pd` F64vec2, `ps` F32vec4, F32vec1 N/A|
|`unpack_low`|`_mm_unpacklo_[x]`|same as `unpack_high`|`pd` F64vec2, `ps` F32vec4, F32vec1 N/A|
|`pack_sat`|`_mm_packs_[x]`|`epi32` I32vec4, `epi16` I16vec8, `pi32` I32vec2, `pi16` I16vec4; N/A I64vec2/I8vec16|N/A|
|`packu_sat`|`_mm_packus_[x]`|`epi16` I16vec8, `pu16` I16vec4; N/A elsewhere|N/A|
|`sat_add`|`_mm_adds_[x]`|`epi16` I16vec8, `epi8` I8vec16, `pi16` I16vec4, `pi8` I8vec8; N/A I64vec2/I32vec4/I32vec2|`pd` / `ps` / `ss`|
|`sat_sub`|`_mm_subs_[x]`|same as `sat_add`|`pi16` `[sic: source garbled]` / `pi8` `[sic: source garbled]` / `pd` `[sic: source garbled]`|

\* `_mm_andnot_[y]` intrinsics do not apply to the fvec classes.

### Programming Example
Averages the elements of a twenty element floating point array using F32vec4:

```cpp
#include <fvec.h> //Intel® Streaming SIMD Extension (Intel® SSE) class definitions
//Shuffle two SP FP from a into the low two and two from b into the high two
#define SHUFFLE(a,b,i) (F32vec4)_mm_shuffle_ps(a,b,i)
#include <stdio.h>
#define SIZE 20
float result;
_MM_ALIGN16 float array[SIZE];
void Add20ArrayElements (F32vec4 *array, float *result) {
   F32vec4 vec0, vec1;
   vec0 = _mm_load_ps ((float *) array); // Load array's first four floats
   vec0 += array[1]; vec0 += array[2]; vec0 += array[3]; vec0 += array[4]; //add 5-20
   //Add the two lowers to the two raises, then add those two results together
   vec1 = SHUFFLE(vec1, vec0, 0x40);
   vec0 += vec1;
   vec1 = SHUFFLE(vec1, vec0, 0x30);
   vec0 += vec1;
   vec0 = SHUFFLE(vec0, vec0, 2);
   _mm_store_ss (result, vec0); // Store the final sum
}
int main(int argc, char *argv[]) {
   int i;
   for (i=0; i < SIZE; i++) { array[i] = (float) i; } //Initialize the array
   Add20ArrayElements ((F32vec4 *)array, &result); //add all elements
   printf ("Average of all array values = %f\n", result/20.);
   printf ("The correct answer is %f\n\n\n", 9.5);
   return 0;
}
```

### Intel's valarray Implementation
A high performance implementation of specialized one-dimensional `valarray` operations for the C++ standard STL `valarray` container (array/vector operations for high performance computing, designed to exploit parallelism), using the Intel® Integrated Performance Primitives (Intel® IPP), part of the product — select Intel® IPP when you install. It consists of a replacement header, `<valarray>`, specialized for:

|Operator|Type|
|---|---|
|`abs, acos, acosh, asin, asinh, atan, atan2, atanh, cbrt, cdfnorm, ceil, cos, cosh, erf, erfc, erfinv, exp, expm1, floor, hypot, inv, invcbrt, invsqrt, ln, log, log10, log1p, nearbyint, pow, pow2o3, pow3o2, powx, rint, round, sin, sinh, sqrt, tan, tanh, trunk`|float, double|
|`add, conj, div, mul, mulbyconj, mul, sub`|Ipp32fc, Ipp64fc|
|`addition, subtraction, division, multiplication`|float, double|
|`bitwise or, and, xor`|(all unsigned) char, short, int|
|`min, max, sum`|signed or short/signed int, float, double|

`valarray` is **not available for SYCL**. Intel's implementation allows declaring large arrays for parallel processing; improved implementation is tied up with calling the Intel® IPP libraries. Include `<valarray>`, located in `<installdir>/perf_header`:

```cpp
#include <valarray>
void test( )
{
    std::valarray<float> vi(N), va(N);
    …
    vi = vi + va; //array addition
    …
}
```

**NOTE** To use the static merged library containing all CPU-specific optimized versions of the library code, call `ippStaticInit` first, before any Intel® IPP calls — this ensures automatic dispatch to the appropriate library code for Intel® processors and the generic code for non-Intel processors at runtime. Without it, the merged library uses the generic instance. `ippStaticInit` is not needed for the dynamic version of the libraries.

**Compiling valarray Source Code.** Use `/Quse-intel-optimized-headers` (Windows) or `-use-intel-optimized-headers` (Linux) to include the required `valarray` header file and all necessary Intel® IPP library files. The merged libraries use a static library containing all CPU-specific variants of the library code.

```bash
icpx -use-intel-optimized-headers source.cpp                      # Linux one-step (Intel® 64)
icpx -use-intel-optimized-headers -c source.cpp                   # Linux separate compile;
icpx source.o -use-intel-optimized-headers -shared-intel          #   link dynamic (.so)
icpx source.o -use-intel-optimized-headers                        # Linux merged (static) link
icx /Quse-intel-optimized-headers source.cpp                      # Windows one-step
icx /Quse-intel-optimized-headers /c source.cpp                   # Windows separate compile;
icx source.obj /Quse-intel-optimized-headers                      #   link DLL (dynamic)
icx /Quse-intel-optimized-headers /Qipp-link:static /c source.cpp # Windows merged (static)
icx source.obj /Quse-intel-optimized-headers /Qipp-link:static    #   compile and link
```

### Usage Precautions
**Clear MMX™ Technology Registers.** Using both the Ivec and Fvec classes at the same time could mix MMX™ Technology instructions (called by Ivec classes) with Intel® architecture floating-point instructions (called by Fvec classes). x87 floating-point instructions exist in these Fvec functions: fvec constructors; debug functions (`cout` and element access); `rsqrt_nr`. MMX™ Technology registers are aliased on the floating-point registers, so clear the MMX™ Technology state with the `EMMS` instruction intrinsic before issuing an x87 floating-point instruction.

```cpp
ivecA = ivecA & ivecB; /* Ivec logical operation using MMX™ Technology instructions */
empty ();              /* creates a clear state */
cout << f32vec4a;      /* F32vec4 operation using x87 floating-point instructions */
```

**Caution** Failure to clear the MMX™ Technology registers can result in incorrect execution or poor performance due to an incorrect register state.

**Capabilities of C++ SIMD Classes** (four, whose interaction matters): *Computation* — vertical operator support for most arithmetic operations including shifting and saturation (`+ - * / rcp rcp_nr sqrt rsqrt rsqrt_nr`). *Horizontal Data Support* — computation across the elements of one vector rather than vertical element-by-element operations on two vectors; `add_horizontal`, `unpack_low`, `pack_sat` are examples. Shuffle intrinsics are another example of horizontal data flow — not expressed in the classes due to their immediate arguments, but mixable with other C++ functions:

```cpp
F32vec4 fveca, fvecb, fvecd;
fveca += fvecb;
fvecd = _mm_shuffle_ps(fveca,fvecb,0);
```

*Branch Compression and Elimination* — branching in SIMD architectures can be complicated and expensive; the classes eliminate branches using logical operations, max and min functions, conditional selects, and compares. For:

```c
short a[4], b[4], c[4];
for (i=0; i<4; i++)
c[i] = a[i] > b[i] ? a[i] : b[i];
```

the operation is independent of the value of `i`; for each `i` the result could be A or B depending on the actual values. A simple way of removing the branch altogether is the `select_gt` function:

```cpp
Is16vec4 a, b, c
c = select_gt(a, b, a, b)
```

*Caching Hints* — Intel® Streaming SIMD Extensions provide prefetching and streaming hints; prefetching can minimize the effects of memory latency, and streaming hints let you indicate that certain data should not be cached.

### Intel® C++ Class Libraries — gotchas
- **MMX™/x87 register aliasing.** MMX™ registers alias the floating-point registers; mixing Ivec (MMX™) and Fvec classes can interleave MMX™ and Intel® architecture floating-point instructions. Clear the MMX™ state with `EMMS` (`empty()`) before an x87 instruction — x87 instructions occur in fvec constructors, debug functions (`cout`/element access), and `rsqrt_nr` — or you get incorrect execution or poor performance.
- **Never intermix `M64` and `M128`** — unexpected behavior.
- **Integer arithmetic is partially implemented:** `*` only for `I[s|u]16vec4`/`I[s|u]16vec8`; `mul_high`/`mul_add` take `Is16vec4` data only; `/`, `sqrt`, `rcp`, `rcp_nr`, `rsqrt`, `rsqrt_nr` are N/A for every integer class (Fvec only).
- **Explicit typecasts are required** wherever the rules table says Explicit: mismatched operand sizes for add/sub/mul/shift/compare/select and signed classes for less-than/greater-than compares/selects. Only assignment and sign conversion are automatic; logical size conversion is automatic to the left.
- **Shift sign semantics follow the cast.** `A&B` returns `I16vec4`; cast to `Iu16vec4` for a logical shift or `Is16vec4` for an arithmetic shift. Signed types use arithmetic right shifts; unsigned/intermediate classes use logical shifts. `<<` cannot take `I[s|u]8vec[8|16]` as its left operand.
- **Return types are the nearest common ancestor** for differing operand classes (`Is32vec2 A ^ Iu32vec2 B` → `I32vec2 R`); same-class operands return the same type. With assignment the left operand sets size and signedness and both operands must match in size.
- **Conditional select type comes from the 3rd/4th operands** (nearest common ancestor of C and D); all four operands must be the same size, and A/B must be signed for the greater-than/less-than forms.
- **Alignment.** 16-byte for SSE memory ops and `store_nta`; 32-byte for AVX. F32vec4/F64vec2 objects are aligned by default but float arrays are not — use `__declspec(align(16)) float A[4];`. `loadu`/`storeu` make no alignment assumption.
- **Debug operations are slow and unmapped to intrinsics;** out-of-range element access/assignment aborts only when `DEBUG` is enabled.
- **Missing intrinsics:** immediate-argument intrinsics are not implemented as classes (`_mm_shuffle_ps`, `_mm_shuffle_pi16`, `_mm_extract_pi16`, `_mm_insert_pi16`), but shuffle intrinsics can be mixed with class code; conversion operations exist as intrinsics only.
- **`valarray` requires Intel® IPP**, header at `<installdir>/perf_header`, `-use-intel-optimized-headers`(Linux)/`/Quse-intel-optimized-headers`(Windows), and `ippStaticInit` first for the static merged library (not dynamic). **Not available for SYCL.**
- **Header inclusion is cumulative:** `dvec.h` covers SSE2/SSE3/SSE4/AVX; `fvec.h` covers Ivec+Fvec.

### Intel® C++ Class Libraries — source map
- Overview, SIMD Data Flow, interface comparison, C++ classes, available classes, header files — pp. 480–483
- Usage Precautions (MMX™ clearing, capability set); Ivec hierarchy; Terms and Syntax; operator rules — pp. 483–487
- Initialization; assignment; all operator families; debug, element access, unpack, pack, MMX™ state — pp. 488–505
- Intel® SSE integer functions; Fvec/Ivec conversions — pp. 505–507
- Floating-Point Vector Classes (notation, alignment, conversions, constructors, arithmetic, min/max, logical, compare, select, cacheability, debug, load/store, unpack, move mask) — pp. 507–525
- Classes Quick Reference — pp. 525–531; Programming Example — pp. 532–533; valarray — pp. 533–534

<!--SLOT:ASYNC-->

## C++ Asynchronous I/O Extensions

Intel's C/C++ asynchronous input/output (Intel's C/C++ AIO) extensions start an I/O operation and resume normal tasks immediately while it runs in parallel; implemented in `libicaio.lib` (`<install-dir>/lib`), **Windows only**.

| entry / signature | purpose | values |
|---|---|---|
| `int aio_read(struct aiocb *aiocbp)` | async read (`aio.h`) | `0` success, `-1` error (`errno`, `aio_error()`) |
| `int aio_write(struct aiocb *aiocbp)` | async write | `0` success, `-1` error |
| `int aio_suspend(const struct aiocb * const cblist[], int n, const struct timespec *timeout)` | wait for one I/O | `0`/`-1`; `EAGAIN` |
| `int aio_error(const struct aiocb *aiocbp)` | error status | `EINPROGRESS`, `ECANCELED`, `0`, error value |
| `ssize_t aio_return(struct aiocb *aiocbp)` | final return status | call **once**, after `aio_error() != EINPROGRESS` |
| `int aio_fsync(int op, struct aiocb *aiocbp)` | sync outstanding I/O on `aio_fildes` | `0`/`-1`; `O_SYNC` sample |
| `int aio_cancel(HANDLE fd, struct aiocb *aiocbp)` | cancel outstanding requests | `AIO_CANCELLED`, `AIO_NOTCANCELLED`, `AIO_ALLDONE`, `-1` |
| `int lio_listio(int mode, struct aiocb *list[], int nent, struct sigevent *sig)` | list of I/O requests | `LIO_WAIT`, `LIO_NOWAIT`; `aio_lio_opcode` = `LIO_READ`\|`LIO_WRITE`\|`LIO_NOP` |
| `async::async_class<A>`, `thread_control` (`aiostream.h`) | async STL stream I/O | 8 members; `/MT` or `/MTd` |

## Types of Intel's C/C++ Asynchronous I/O Extensions

- **Asynchronous I/O Library**: POSIX-based async I/O functions, Windows OS, C/C++; `aio.h`.
- **Asynchronous I/O Template Class**: an `asych_class` template class [sic: source garbled], Windows OS, C++; async I/O with STL streams classes; `aiostream.h`.

## Intel's C++ Asynchronous I/O Library for Windows

Like POSIX AIO for Linux, except: `struct aiocb` uses `HANDLE` (not `unsigned int`) for `aio_fildes`, and `intptr_t` for POSIX AIO types `ssize_t`/`__off_t`; `struct sigevent` supports signal notification **and** non-notification for thread call-back, but **signal notification on completion of the AIO operation is not supported** — Linux/Unix ports setting a completion handler without the handler's name in the `aiocb` struct cannot work, so set the handler's name before `aio_read`/`aio_write`:

```c
static void aio_CompletionRoutine(sigval_t sigval) { /* … code … */ }
my_aio.aio_sigevent.sigev_notify          = SIGEV_THREAD;
my_aio.aio_sigevent.sigev_notify_function = aio_CompletionRoutine;
```

> **NOTE** The POSIX AIO library and the Microsoft SDK provide similar AIO functions; POSIX allows AIO with any file, Windows only on files flagged `FILE_FLAG_OVERLAPPED`.

### aio_read / aio_write

`0` success / `-1` error (`errno`; see `aio_error()`). `aio_read` calls `"ReadFile(hFile, lpBuffer, nNumberOfBytesToRead, lpNumberOfBytesRead, NULL);"`; `aio_write` calls `"WriteFile(hFile, lpBuffer, nNumberOfBytesToWrite, lpNumberOfBytesWritten, NULL);` [sic: source garbled — closing quote missing]. `hFile` = `aiocbp->aio_fildes`, `lpBuffer` = `aiocbp->aio_buf`, count = `aiocbp->aio_nbytes`; `aio_return()` gives bytes transferred. `aiocb->aio_offset == (intptr_t)-1` starts after the last record.

### Example for aio_read and aio_write Functions

5.6 MB is written asynchronously while the main program computes (scalar multiplication of two vectors). **С-source File Executing a Scalar Multiplication** [sic: Cyrillic С]:

```c
double do_compute(double A, double B, int arr_len)
{ int i; double res = 0;
  double *xA = malloc(arr_len * sizeof(double)), *xB = malloc(arr_len * sizeof(double));
  if (!xA || !xB) abort();
  for (i = 0; i < arr_len; i++) { xA[i] = sin(A); xB[i] = cos(B); res += xA[i]*xA[i]; }
  free(xA); free(xB); return res; }
```
**Example One** (`DIM_X 123`, `DIM_Y 70000`, `aio_dat[DIM_Y]`, `aio_dat_tmp[DIM_Y]`, `fd` via `CreateFile(... FILE_ATTRIBUTE_NORMAL ...)`):

```c
my_aio.aio_fildes = fd; my_aio.aio_buf = memcpy(aio_dat_tmp, aio_dat, sizeof(aio_dat_tmp));
my_aio.aio_nbytes = sizeof(aio_dat_tmp); my_aio.aio_offset = (intptr_t)-1;
my_aio.aio_sigevent.sigev_notify = SIGEV_NONE;   if (aio_write((void*)&my_aio) == -1) abort();
```
**Example Two** (`// icx -c do_compute.c`, `// icx aio_sample2.c do_compute.obj`, `// aio_sample2.exe`; `#define WAIT { while (!aio_flg); aio_flg = 0; }`, `sigev_notify = SIGEV_THREAD`):

```c
WAIT; res = aio_write(&my_aio); WAIT; res = aio_fsync(O_SYNC, &my_aio);
aio_read(&my_aio); WAIT; res = aio_return(&my_aio); my_aio.aio_offset += my_aio.aio_nbytes;
```

### aio_suspend

`int aio_suspend(const struct aiocb * const cblist[], int n, const struct timespec *timeout);` — `cblist[]` control block on which I/O is initiated; `n` list length; `*timeout` suspend interval; returns `0`/`-1` (`errno`).

Remaining examples share `typedef struct aiocb aiocb_t;`, `fd = CreateFile("dat", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL)`, and

```c
#define IC_AIO_DATA_INIT(_aio, _fd, _dat, _len, _off)\
  {memset(&_aio, 0, sizeof(_aio)); _aio.aio_fildes = _fd; _aio.aio_buf = _dat; \
   _aio.aio_nbytes = _len; _aio.aio_offset = _off;}
```

### Example for aio_suspend Function

`// icx aio_sample3.c`, `// aio_sample3.exe`; `IC_AIO_DATA_INIT(aio[0], fd, "rec#1\n", ...)`, `IC_AIO_DATA_INIT(aio[1], fd, "rec#2\n", ... aio[0].aio_nbytes)`, two `aio_write()`:

```c
ret = aio_suspend(aio_list, 2, 0); if (ret == -1) return errno;
```
Output: `> aio_sample3.exe` → `Done`; `> type dat` → `rec#1 rec#2_`.

### aio_error / aio_return

`aio_error`: `EINPROGRESS` (not completed), `ECANCELED` (cancelled), `0` (success), or an error value via `errno` (same as `ReadFile()`/`WriteFile()`/`FlushFileBuffers()` errors). `aio_return` gives the final status; call **only once**, after `aio_error()` returns other than `EINPROGRESS`; it returns the synchronous `ReadFile()`/`WriteFile()`/`FlushFileBuffer()` value when completed, undefined when not, an error value on error (`errno`).

### Example for aio_error and aio_return Functions

`// icx aio_sample4.c`, `// aio_sample4.exe`; `char *dat = "Hello from Ex-3\n";`.

```c
IC_AIO_DATA_INIT(aio, fd, dat, strlen(dat), 0); aio_write(&aio); ret = aio_error(&aio);
if (ret == EINPROGRESS) ret = aio_suspend(aio_list, 1, NULL); else if (ret) return ret;
ret = aio_error(&aio); if (ret) return ret;   ret = aio_return(&aio);
```
Output: `> ./a.out` → `ERRNO=996 STR=Unknown error`; `> type dat` → `Hello from Ex-3`.

### aio_fsync

`int aio_fsync(int op, struct aiocb *aiocbp);` — `op` synchronization request operation type; `*aiocbp` control block. Syncs all outstanding asynchronous I/O on `aiocbp->aio_fildes`; `0`/`-1` (`errno`).

### aio_cancel

`int aio_cancel(HANDLE fd, struct aiocb *aiocbp);` — cancels outstanding asynchronous requests for `fd`; NULL `aiocbp` cancels all, else only those in that control block. Cancelled requests get return status `-1`, error status `ECANCELED`; un-cancellable control blocks unchanged; unspecified results if `aiocbp` is non-NULL and `fd` differs from the initiating descriptor. Returns `AIO_CANCELLED` (all cancelled) [sic: source garbled spelling], `AIO_NOTCANCELLED` (≥1 still being cancelled — check `aio_error`), `AIO_ALLDONE` (all completed before the call), `-1` error (`errno`).

### Example for aio_cancel Function

`// icx aio_sample5.c`, `// aio_sample5.exe`; prints `printf("AIO_CANCELED=%d AIO_NOTCANCELED=%d\n", AIO_CANCELED, AIO_NOTCANCELED);`, then `aio_write(&aio)`.

```c
ret = aio_cancel(fd, &aio);
if (ret == AIO_CANCELED) fprintf(stderr, "1 ERRNO=%d STR=%s\n", ret, strerror(ret));
else if (ret) return ret;
ret = aio_cancel(fd, &aio); if (ret == AIO_NOTCANCELED) ret = aio_suspend(aio_list, 1, NULL);
```
Output: `> aio_sample5.exe` → `AIO_CANCELED=1 AIO_NOTCANCELED=2`, `1 ERRNO=1 STR=Operation not permitted`; `> type dat` → `Hello from Ex-4`.

### lio_listio

`int lio_listio(int mode, struct aiocb *list[], int nent, struct sigevent *sig);` — one call for a list of I/O requests. `mode`: `LIO_WAIT` (return only after completing I/O) or `LIO_NOWAIT` (return as soon as I/O is queued). `*list[]` array of `aiocb` pointers (NULL ignored); `nent` element count; `*sig` notification after all I/O completes — `0` (queued signal with an application-defined value generated when an async I/O request occurs), `1` (no notification even when requests are processed), `2` (a notification function is called). `LIO_WAIT` waits for all operations, ignoring `sig`; `LIO_NOWAIT` returns immediately, notification follows `sig`. Returns `LIO_NOWAIT`: `0` queued / `-1` not queued; `LIO_WAIT`: `0` completed / `-1` not completed (`errno`). `aio_lio_opcode` selects `LIO_READ`/`LIO_WRITE`/`LIO_NOP` (`<aio.h>`).

### Example for lio_listio Function

`// icx aio_sample6.c`, `// aio_sample6.exe`.

```c
IC_AIO_DATA_INIT(aio[0], fd, "rec#1\n", strlen("rec#1\n"), 0)
IC_AIO_DATA_INIT(aio[1], fd, "rec#2\n", strlen("rec#2\n"), aio[0].aio_nbytes)
aio[0].aio_lio_opcode = aio[1].aio_lio_opcode = LIO_WRITE;
ret = lio_listio(LIO_WAIT, aio_list, 2, 0);
```
Output: `>aio_sample6.exe`; `>type dat` → `rec#1` / `rec#2`.

### Asynchronous I/O Function Errors

Windows* OS only. `errno` reports errors from request functions (`aio_read`, `aio_write`, `aio_fsync`, `lio_listio`) and control functions (`aio_cancel`, `aio_error`, `aio_return`, `aio_suspend`).

```c
struct aiocb my_aio; struct aiocb *my_aio_list[1] = {&my_aio};
int main() {
  int res; double arr[123456]; timespec_t my_t = {1, 0};
  my_aio.aio_fildes = CreateFile("dat", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
     NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  my_aio.aio_buf = (volatile char *)arr; my_aio.aio_nbytes = sizeof(arr);
  aio_write(&my_aio); do_compute(arr, 123456);
  res = aio_suspend(my_aio_list, 1, &my_t);
  if (res) { if (errno == EAGAIN) { if (aio_suspend(my_aio_list, 1, 0)) return errno; }
    else return errno; }
  CloseHandle(my_aio.aio_fildes); printf("\nPass\n"); return 0;
}
```
Includes `<stdio.h>`, `<stdlib.h>`, `<aio.h>`; zero on success, `EAGAIN` or another error value when the call ended by timeout before completion.

## Intel's C++ Asynchronous I/O Class for Windows

`async_class` performs I/O asynchronously to the main program thread, in particular with the STL streams classes, so any STL stream I/O operation can be switched to asynchronous mode with minimal code changes; defined in `aiostream.h`. Windows* OS only.

### Template Class async_class

Two classes in `namespace async`: `async_class` (template) and its `thread_control` base class.

```cpp
namespace async {
template<class A>
class async_class:
public thread_control, public A
}
```
[sic: source garbled — missing opening `{` and terminator as printed] `async_class` inherits asynchronous-execution support from the base `thread_control` class, which encapsulates control of a queue of STL stream operations. Usually add `aiostream.h` and declare the file object as an instance of `async:async_class` [sic: source garbled — single colon]; the initial stream class is the template parameter, so `<<` and `>>` execute asynchronously.

> **NOTE** `aiostream.h` includes all declarations needed for the STL stream I/O operations to add `thread_control` functionality, plus extensions: output operator `>>` and input operator `<<`. [sic: source garbled — operators swapped versus the paragraph above]

Call `wait()` to wait for completion; it is called implicitly in the object destructor if not explicitly.

### Public Interface of Template Class async_class

8 members: `get_last_operation_id()`, `wait()`, `get_status()`, `get_last_error()`, `get_error_operation_id()`, `stop_queue()`, `resume_queue()`, `clear_queue()`.

- `void get_last_operation_id(void)` — ID of last added operation; returns Nothing.
- `int wait(void)` / `int wait(unsigned int operation_id)` — stops the thread until all operations, or `operation_id`, complete; `-1` on queue error (check `get_last_error()`).
- `void get_status(unsigned int operation_id)` — status **without** stopping execution: `STATUS_WAIT`, `STATUS_COMPLETED`, `STATUS_ERROR`, `STATUS_EXECUTE`, `STATUS_BLOCKED`.
- `unsigned int get_last_error()` / `get_error_operation_id()` — error code / operation ID of the last failed operation (code equals `GetLastError()` on Windows*); stops the async queue until new requests.
- `int stop_queue()` / `int resume_queue()` — stop/resume queue execution; `0` success, `-1` error.
- `clear_queue` (documented line `void push_back_operation(class base_operation*)` [sic: source garbled — mismatched]) — clears stopped or error-interrupted queues; `0`/`-1` [sic: source garbled — documented `void`].

### Library Restrictions

The template class does not control the integrity or validity of objects during asynchronous operation; that control is the user's. For Visual Studio stability, link the C++ part of `libacaio.lib` library [sic: source garbled — named `libicaio.lib` elsewhere] with the multi-threaded `msvcrt` runtime library; use `/MT` or `/MTd`.

### Example for Using async_class Template Class

Original code sends floats to a file: `std::vector<float> v(10000);` with `std::ofstream& operator << (std::ofstream & str, std::vector<float> & vec) { /* User output actions */ ... }`, then `std::ofstream external_file(output.txt); external_file << v;`. Asynchronously:

```cpp
#include <aiostream.h>            // header for STL asynchronous IO operations
async::async_class<std::ofstream> external_file(output.txt);   // inherited ofstream type
external_file << v;   external_file.wait();   // wait completion of all asynchronous IO
```

### Performance Recommendations

It is recommended **not** to use asynchronous mode for small objects — e.g. when outputting a standard type value in a loop where other loop operations take less time than output of that value to the STL stream. The example (`#define ARR_LEN 900`) reads two matrices from files (`fA >> A`, `fB >> B`), computes `C[i][j] += A[i][k]*B[k][j]*sin((float)(k))*cos((float)(-k))*sin((float)(k+1))*cos((float)(-k-1))` and prints `fC << C[i][j] << std::endl;`. Bigger matrices also gain in parallel reading.

### Intel's C/C++ Asynchronous I/O Extensions — gotchas

- **Windows*-only** for `aio.h` (library) and `aiostream.h` (template class).
- **Signal notification on completion is unsupported** on Windows; set the handler's name (`SIGEV_THREAD` + `aio_CompletionRoutine`; Example One uses `SIGEV_NONE`) before `aio_read`/`aio_write`, or Linux/Unix ports cannot work.
- **Windows AIO works only on files flagged `FILE_FLAG_OVERLAPPED`** (POSIX: any file), yet samples use `FILE_ATTRIBUTE_NORMAL` — verify before porting [sic: source garbled conflict].
- **Call `aio_return()` only once per request**, after `aio_error()` returns other than `EINPROGRESS`.
- **All errors report through `errno`**; an `aio_suspend` timeout yields `EAGAIN`; runs print `ERRNO=996 STR=Unknown error`, `ERRNO=1 STR=Operation not permitted`.
- **Visual Studio link requirement**: `/MT` or `/MTd` (multi-threaded `msvcrt`) for the C++ part of the library.
- **A queue stops on error** (`STATUS_BLOCKED`): `get_last_error()`, `stop_queue()`, `resume_queue()`, `clear_queue()`.
- **Do not use async mode for small objects** — per-iteration small outputs can be slower; gains appear when in-loop computation overlaps.

### Intel's C/C++ Asynchronous I/O Extensions — source map

- Intro/Types/Library for Windows/`aio_read`+`aio_write`+example, `aio_suspend`+example, `aio_error`/`aio_return`+example — pp. 535–544
- `aio_fsync`, `aio_cancel`+example, `lio_listio`+example, Asynchronous I/O Function Errors — pp. 544–550
- C++ Asynchronous I/O Class/`async_class`/members/Library Restrictions, its example, Performance Recommendations — pp. 550–554

## IEEE 754-2008 Binary Floating-Point Conformance Library

`libbfp754` (header `bfp754.h`) provides all operations mandated by IEEE 754-2008 for the `binary32` and `binary64` binary floating-point interchange formats. Minimum requirements: Intel® Pentium® 4 processor and an OS supporting Intel® Streaming SIMD Extensions 2 (Intel® SSE2). Many routines are more optimized for Intel® microprocessors than for non-Intel microprocessors.

Notation: `<N>`∈{`32`,`64`} selects `binary32`/`binary64`; `<T>`/`<R>` are `float` for N=32 and `double` for N=64. `{a,b,c}` means the source documents the signature separately for each listed item.

### IEEE 754-2008 Binary Floating-Point Conformance Library and Usage

- Include `bfp754.h`; specify linker option `-lbfp754` and floating-point semantics control option `-fp-model strict`.
- **Not available for SYCL**; cannot be used with SYCL kernels.
- Rounding-direction attributes supported (all four mandated): `roundTiesToEven`, `roundTowardPositive`, `roundTowardNegative`, `roundTowardZero`. `roundTiesToAway` is not required by the standard, hence not fully supported. Default: `roundTiesToEven`.
- Exceptions supported (all mandated): invalid operation, division by zero, overflow, underflow, inexact; flags set accordingly under default exception handling. Alternate exception handling (optional in the standard) is not supported.

### Operations

| | Produce result | Produce no result |
| --- | --- | --- |
| Might signal FP exception | General-computational | Signaling-computational |
| Do not signal FP exception | Quiet-computational | Non-computational |

General-computational operations produce correctly rounded floating-point or integer results and might signal exceptions; quiet-computational produce floating-point results; signaling-computational and non-computational produce no floating-point results. General-computational operations are also distinguished by **homogenous** (same format in and out) vs **formatOf** (different formats).

**NOTE:** The standard requires all formatOf general-computational operations be computed without any loss of precision before converting to the destination format; most hardware/software instead rounds an intermediate `binary64` result and then converts to `binary32`, and this double rounding may differ from the standard under certain rounding modes. Source example: `x = 0x3ff0000010000000 = 1.000000000000000000000001_2`, `y = 0x3ca0000000000000 = 1.0_2*2^(-53)`, `x+y = 1.00000000000000000000000100000000000000000000000000001_2`; with `roundTiesToEven`, double rounding gives `1.000000000000000000000001_2` (`0x3ff0000010000000`) in binary64 then `1` (`0x3f800000`) in binary32, but the standard requires `1.00000000000000000000001_2` (`0x3f800001`) in binary32.

### Data Types

Standard format names → C99 types used in this library: `binary32`→`float`; `binary64`→`double`; `int`→`int`, `unsigned int`, `long long int`, `unsigned long long int`; `int32`→`int`; `uint32`→`unsigned int`; `int64`→`long long int`; `uint64`→`unsigned long long int`; `boolean`→`int`; `enum` (floating-point class, floating-point radix)→`int`; `logBFormat` (destination of logB, scale exponent operand of scaleB)→`int`; `decimalCharacterSequence`→`char*`; `hexCharacterSequence`→[unclear in source: no C99 type given]; `exceptionGroup` (set of exceptions as a set of booleans)→`int`; `flags`→`int`; `binaryRoundingDirection`→`int`; `modeGroup`→`int`; `void`→`void`.

### Use the Intel® IEEE 754-2008 Binary Floating-Point Conformance Library

Include `bfp754.h`. Example on Linux* OS (cannot be used with SYCL kernels):

```c
//binary.c
#include <stdio.h>
#include <bfp754.h>
int main(){
   double a64, b64;
   float c32;
   a64 = 1.000000059604644775390625;
   b64 = 1.1102230246251565404236316680908203125e-16;
   c32 = __binary32_add_binary64_binary64(a64, b64);
   printf("The addition result using the libary: %8.8f\n", c32);
   c32 = a64 + b64;
   printf("The addition result without the libary: %8.8f\n", c32);
   return 0;
}
```

```bash
icx -fp-model source -fp-model except binary.c –lbfp754
```

```text
The addition result using the libary: 1.00000012
The addition result without the libary: 1.00000000
```

### Function List

`libbfp754 name`→`IEEE standard equivalent` (general-computational equivalents are shown with the functions themselves in the section below):

- **Homogeneous General-Computational Operations Functions:** `ilogb`→logB, `maxnum`→maxNum, `maxnum_mag`→maxNumMag, `minnum`→minNum, `minnum_mag`→minNumMag, `next_down`→nextDown, `next_up`→nextUp, `rem`→remainder, `round_integral_exact`→roundToIntegralExact, `round_integral_nearest_away`→roundToIntegralTiesToAway, `round_integral_nearest_even`→roundToIntegralTiesToEven, `round_integral_negative`→roundToIntegralTowardNegative, `round_integral_positive`→roundToIntegralTowardPositive, `round_integral_zero`→roundToIntegralTowardZero, `scalbn`→scaleB.
- **Quiet-Computational Operations Functions:** `abs`→abs, `copy`→copy, `copysign`→copySign, `negate`→negate.
- **Signaling-Computational Operations Functions:** `quiet_*`→`compareQuiet<Relation>`, `signaling_*`→`compareSignaling<Relation>`.
- **Non-Computational Operations Functions:** predicate/flag/version functions in the section below, plus `defaultMode`→defaultModes, `getBinaryRoundingDirection`, `restoreModes`, `saveModes`, `setBinaryRoundingDirection`, `saveFlags`→saveAllFlags (other names identity-mapped).

### Homogeneous General-Computational Operations Functions

```c
// unary, same input/output format:  <R> __binary<N>_<name>(<T> x);
//   round_integral_{nearest_even|nearest_away|zero|positive|negative|exact}: closest integral value,
//     ties to even / away from zero / toward zero / toward +infinity / toward -infinity / per the
//     applicable rounding-direction
//   next_up : least number in x's format greater than x;  next_down : largest number less than x
// binary: <R> __binary<N>_<name>(<T> x, <T> y);
//   rem : remainder;  minnum/minnum_mag : minimal value / minimal absolute value;
//   maxnum/maxnum_mag : maximal value / maximal absolute value
<R> __binary<N>_scalbn(<T> x, int n);   // computes x X 2n for integer value n
int __binary<N>_ilogb(<T> x);           // exponent part of x as integer
```

### General-Computational Operation Functions

formatOf general-computational operations (result converted to the destination format):

```c
// add / sub / mul / div (→ addition / subtraction / multiplication / division):
//   <R> __binary<N>_{add,sub,mul,div}_binary<A>_binary<B>(<AT> x, <BT> y);
//   documented for every N and every operand-format pair A,B in {32,64} — 8 signatures per operation
// sqrt (→ squareRoot): float __binary32_sqrt_binary32(float x);  float __binary32_sqrt_binary64(double x);
//   source also lists: double __binary32_sqrt_binary32(float x);   // [sic: source garbled]
//                      double __binary32_sqrt_binary64(double x);  // [sic: source garbled]
// fma (→ fusedMultiplyAdd; (x×y) + z): <ret> __binary<N>_fma_<xfmt>_<yfmt>_<zfmt>(...)
//   for every destination N and operand triple from {binary32,binary64} — 16 signatures, e.g.
//   float __binary32_fma_binary32_binary64_binary64(float x, double y, double z);
// from_int32 / from_uint32 / from_int64 / from_uint64 (→ convert) — integral value → floating-point;
//   float __binary32_from_int32(int n);  double __binary64_from_int32(int n); and likewise
//   from_uint32(unsigned int n), from_int64(long long int n), from_uint64(unsigned long long int n)
double __binary32_to_binary64(float x);   // → convertFormat: binary32 → binary64
float  __binary64_to_binary32(double x);  // → convertFormat: rounds binary64 → binary32
float  __binary32_from_string(char * s);  // → convertFromDecimalCharacter: decimal chars → floating-point
double __binary64_from_string(char * s);
void__binary32_to_string(char * s, float x);    // → convertToDecimalCharacter
void__binary64_to_string(char * s, double x);   // [sic: no space after void in source]
float  __binary32_from_hexstring(char * s);     // → convertFromHexCharacter: hex chars → floating-point
double __binary64_from_hexstring(char * s);
void__binary32_to_hexstring(cgar * s, float x);   // → convertToHexCharacter; [sic: source garbled "cgar"]
void__binary64_to_hexstring(char * s, double x);  // [sic: no space after void in source]
```

**Conversion to integer functions** — for every `IT` in {`int32`,`uint32`,`int64`,`uint64`} (`int32`→`int`, `uint32`→`unsigned int`, `int64`→`long long int`, `uint64`→`unsigned long long int`) and every `MODE`, both are documented:

```c
IT __binary32_to_<IT>_<MODE>(float x);
IT __binary64_to_<IT>_<MODE>(double x);
```

| MODE | Semantics (nearest integral value) | IEEE standard equivalent |
| --- | --- | --- |
| rnint | halfway to even, no inexact signal | convertToIntegerTiesToEven |
| int | toward zero, no inexact signal | convertToIntegerTowardZero |
| ceil | toward positive infinity, no inexact signal | convertToIntegerTowardPositive |
| floor | toward negative infinity, no inexact signal | convertToIntegerTowardNegative |
| rninta | halfway away from zero, no inexact signal | convertToIntegerTiesToAway |
| xrnint | halfway to even, signals if inexact | convertToIntegerExactTiesToEven |
| xint | toward zero, signals if inexact | convertToIntegerExactTowardZero |
| xceil | toward positive infinity, signals if inexact | convertToIntegerExactTowardPositive |
| xfloor | toward negative infinity, signals if inexact | convertToIntegerExactTowardNegative |
| xrninta | halfway away from zero, signals if inexact | convertToIntegerExactTiesToAway |

### Quiet-Computational Operations Functions

```c
<R> __binary<N>_copy(<T> x);      // → copy: same format, no sign change
<R> __binary<N>_negate(<T> x);    // → negate: same format, sign reversed
<R> __binary<N>_abs(<T> x);       // → abs: same format, sign set to positive
<R> __binary<N>_copysign(<T> x, <T> y);  // → copySign: copy of x with the same sign as y
```

**NOTE:** For the listed quiet-computational operations functions, when the first input is a signaling NaN, the standard allows two outcomes: signal invalid exception and output a quieted signaling NaN, or output the signaling NaN without signaling any exception.

### Signaling-Computational Operations Functions

All comparisons return `int` (1 = true, 0 = false) and are documented in the four operand-format combinations (`<ret> __binary{32,64}_<name>_binary{32,64}(<float|double> x, <float|double> y)`, e.g. `__binary32_quiet_equal_binary64(float x, double y)`). Source has stray spaces inside `__binary64_quiet_equal_ binary64` and `__binary64_signaling_not_less_binary64 ` [sic: source garbled].

- **Quiet comparisons** — return 1 when the relation named by the suffix holds, else 0; signal invalid operation exception when signaling NaN is in the inputs: `quiet_equal`, `quiet_not_equal`, `quiet_greater`, `quiet_greater_equal`, `quiet_less`, `quiet_less_equal`, `quiet_unordered`, `quiet_not_greater`, `quiet_less_unordered`, `quiet_not_less`, `quiet_greater_unordered`, `quiet_ordered`.
- **Signaling comparisons** — return 1 when the relation named by the suffix holds, else 0; signal invalid operation exception when NaN is in the inputs: `signaling_equal`, `signaling_greater`, `signaling_greater_equal`, `signaling_less`, `signaling_less_equal`, `signaling_not_equal`, `signaling_not_greater`, `signaling_less_unordered`, `signaling_not_less`, `signaling_greater_unordered`.

### Non-Computational Operations Functions

```c
int __binary_is754version1985(void);  // 1 iff environment conforms to IEEE Std. 754-1985, else 0
int __binary_is754version2008(void);  // 1 iff environment conforms to IEEE Std. 754-2008, else 0
// predicate-style: int __binary<N>_<name>(<T> x); returns 1 iff:
//   class : x belongs to the class table below | isSignMinus : negative sign
//   isNormal : normal (not zero, subnormal, infinite, or NaN) | isZero : ±0
//   isSubnormal : subnormal | isInfinite : infinite | isNaN : a NaN | isSignaling : a signaling NaN
//   isCanonical : a finite number, infinity, or NaN that is canonical
//   radix : radix of the format of the input floating-point number
int _binary<N>_totalOrder(<T> x, <T> y);     // 1 iff x and y are total ordered, else 0
int _binary<N>_totalOrderMag(<T> x, <T> y);  // same as totalOrder(abs(x), abs(y))
```

- `is754version1985` **always returns 0** in this library; `is754version2008` **always returns 1**; `isCanonical` **always returns 1** (only canonical floating-point numbers are expected); `radix` **always returns 2** (binary floating-point library).
- `isFinite` returns 1 iff its argument is finite (not infinite or NaN); its calling-interface block is empty in the source [unclear in source]. `totalOrder`/`totalOrderMag` are documented with the single-underscore prefix `_binary32_`/`_binary64_` [sic: as documented].

`class` return values: 0 signalingNaN, 1 quietNaN, 2 negativeInfinity, 3 negativeNormal, 4 negativeSubnormal, 5 negativeZero, 6 positiveZero, 7 positiveSubnormal, 8 positiveNormal, 9 positiveInfinity (the ten classes).

**Exception group flags** for `lowerFlags`, `raiseFlags`, `testFlags`, `testSavedFlags`, `restoreFlags`, `saveFlags`: 1 `__BFP754_INVALID`, 2 `__BFP754_DIVBYZERO`, 4 `__BFP754_OVERFLOW`, 8 `__BFP754_UNDERFLOW`, 16 `__BFP754_INEXACT`.

```c
void __binary_lowerFlags(int x);             // lowers the flags of the exception group specified by the input
void __binary_raiseFlags(int x);             // raises the flags of the exception group specified by the input
int  __binary_testFlags(int x);              // 1 iff any flag of the exception group specified by the input is raised
int  __binary_testSavedFlags(int x, int y);  // 1 iff any flag of the exception group specified by y is raised in x
void __binary_restoreFlags(int x);           // restores the flags to their states represented in x
int  __binary_saveFlags(void);               // returns a representation of the state of all status flags
```

**Rounding direction** — `getBinaryRoundingDirection` returns an integer representing the rounding direction in use: 0 `__BFP754_ROUND_TO_NEAREST_EVEN`, 1 `__BFP754_ROUND_TOWARD_POSITIVE`, 2 `__BFP754_ROUND_TOWARD_NEGATIVE`, 3 `__BFP754_ROUND_TOWARD_ZERO`. Its calling interface, and detailed entries for `defaultMode`, `restoreModes`, `saveModes`, `setBinaryRoundingDirection`, are outside this source range.

### IEEE 754-2008 Binary Floating-Point Conformance Library — gotchas

- `libbfp754` is **not available for SYCL** and cannot be used with SYCL kernels.
- Build flags: `bfp754.h`, `-lbfp754`, `-fp-model strict`; the worked Linux example instead uses `icx -fp-model source -fp-model except binary.c –lbfp754` (en-dash before `lbfp754`) [sic: source garbled].
- formatOf operations must be computed without precision loss before conversion (ordinary hardware double rounding can differ); `roundTiesToAway` is not fully supported; default `roundTiesToEven`; alternate exception handling unsupported; a signaling NaN into a quiet-computational operation may be quieted with invalid signaled or returned unchanged.

## Numeric String Conversion Library

Intel's Numeric String Conversion Library, `libistrconv`, provides a collection of routines for converting between ASCII strings and C data types, which are optimized for performance. The `istrconv.h` header file declares prototypes for the library functions.

- Linux\*: you can link `libistrconv` as a static or shared library.
- Windows\*: you **must** link `libistrconv` as a static library only.

### Use Intel's Numeric String Conversion Library

To use the `libistrconv` library, include the header file `istrconv.h` in your program. The following `conv.c` example illustrates converting between string and floating-point data type:

```c
// conv.c
#include <stdio.h>
#include <istrconv.h>
#define LENGTH 20

int main() {
 const char pi[] = "3.14159265358979323";
 char s[LENGTH];
 int prec;
 float fx;
 double dx;
 printf("PI: %s\n", pi);
 printf("single-precision\n");
  fx = __IML_string_to_float(pi, NULL);
  prec = 6;
  __IML_float_to_string(s, LENGTH, prec, fx);
  printf("prec: %2d, val: %s\n", prec, s);
  printf("double-precision\n");
  dx = __IML_string_to_double(pi, NULL);
  prec = 15;
  __IML_double_to_string(s, LENGTH, prec, dx);
  printf("prec: %2d, val: %s\n", prec, s);
  return 0;
}
```

Compile (see *Invoke the Compiler* for all available compilers and drivers):

```bash
# Linux
icpx conv.c -libistrconv
# Windows
icx conv.c libistrconv.lib
```

Expected output:

```text
PI: 3.14159265358979323

single-precision
prec:  6, val: 3.14159

double-precision
prec: 15, val: 3.14159265358979
```

### Integer Conversion Functions Optimized with SSE4.2 Instructions

Optimized (SSE4.2) versions of these functions move strings between memory and XMM registers directly to maximize performance:

`__IML_int_to_string`, `__IML_uint_to_string`, `__IML_int64_to_string`, `__IML_uint64_to_string`, `__IML_i_to_str`, `__IML_u_to_str`, `__IML_ll_to_str`, `__IML_ull_to_str`, `__IML_string_to_int`, `__IML_string_to_uint`, `__IML_string_to_int64`, `__IML_string_to_uint64`, `__IML_str_to_i`, `__IML_str_to_u`, `__IML_str_to_ll`, `__IML_str_to_ull`

- SSE4.2 versions deployed automatically on post-SSE4.2 processors via Intel runtime processor dispatching, or called directly by defining `"__SSE4_2__"` to the C preprocessor where `<istrconv.h>` is included.
- Generic versions deployed automatically on pre-SSE4.2 processors via dispatch, or called directly by adding the `_generic` suffix to the function names.
- **Caveat:** the SSE4.2 versions do not overwrite memory beyond the boundary, but may cause a memory access violation when the memory immediately trailing the strings is not allocated or accessible. Users concerned about this should use the generic versions.

### Routines to Convert Floating-point Numbers to ASCII Strings

Convert floating-point number `x` to string `s`; `l` = length of the formatted string allowing for full conversion (not including the null terminator). `p` = precision; `n` = buffer size.

| function / signature | description |
|---|---|
| `int __IML_float_to_string(char * s, size_t n, int p, float x);`<br>`int __IML_double_to_string(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*g, p, x)` in stdio.h: `p` = maximum number of significant digits in either fixed-point or exponential notation. If `n` is zero, nothing is written and `s` may be a null pointer. Output beyond the (`n`-1)th character is discarded and a null character is appended at the end. `l` is returned on success; otherwise the result is undefined. |
| `int __IML_float_to_string_f(char * s, size_t n, int p, float x);`<br>`int __IML_double_to_string_f(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*f, p, x)`: `p` = number of digits after the decimal point in fixed-point notation. `n` zero → nothing written, `s` may be null; output beyond (`n`-1)th char discarded, null appended. `l` returned on success; otherwise undefined. |
| `int __IML_float_to_string_e(char * s, size_t n, int p, float x);`<br>`int __IML_double_to_string_e(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*e, p, x)`: `p` = number of digits after the decimal point in exponential notation. `n` zero → nothing written, `s` may be null; output beyond (`n`-1)th char discarded, null appended. `l` returned on success; otherwise undefined. |
| `int __IML_f_to_str(char * s, size_t n, int p, float x);`<br>`int __IML_d_to_str(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*g, p, x)`; `p` = max significant digits in fixed-point or exponential notation. If `l < n`, all output stored in `s` with null terminator at the end. Otherwise output beyond the `n`th character is discarded and **no** null character is appended. If `n` is zero, nothing is written and `s` may be a null pointer. `l` returned on success; otherwise undefined. |
| `int __IML_f_to_str_f(char * s, size_t n, int p, float x);`<br>`int __IML_d_to_str_f(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*f, p, x)`; `p` = digits after the decimal point in fixed-point notation. If `l < n`, all output stored with null terminator; otherwise output beyond the `n`th character discarded and no null appended. `n` zero → nothing written, `s` may be null. `l` returned on success; otherwise undefined. |
| `int __IML_f_to_str_e(char * s, size_t n, int p, float x);`<br>`int __IML_d_to_str_e(char * s, size_t n, int p, double x);` | Similar to `snprintf(s, n, %.*e, p, x)`; `p` = digits after the decimal point in exponential notation. If `l < n`, all output stored with null terminator; otherwise output beyond the `n`th character discarded and no null appended. `n` zero → nothing written, `s` may be null. `l` returned on success; otherwise undefined. |

### Routines to Convert Integers to ASCII Strings

Convert integer `x` to string `s`; `l` = length of the formatted string allowing for full conversion (not including the null terminator).

| function / signature | description |
|---|---|
| `int __IML_int_to_string(char * s, size_t n, int x);`<br>`int __IML_uint_to_string(char * s, size_t n, unsigned int x);`<br>`int __IML_int64_to_string(char * s, size_t n, long long x);`<br>`int __IML_uint64_to_string(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[d\|u\|lld\|llu], x)`. If `n` is zero, nothing is written and `s` may be a null pointer. Output beyond the (`n`-1)th character is discarded and a null character is appended at the end. `l` returned on success; otherwise undefined. |
| `int __IML_int_to_oct_string(char * s, size_t n, int x);`<br>`int __IML_uint_to_oct_string(char * s, size_t n, unsigned int x);`<br>`int __IML_int64_to_oct_string(char * s, size_t n, long long x);`<br>`int __IML_uint64_to_oct_string(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[o\|llo], x)`. `n` zero → nothing written, `s` may be null; output beyond (`n`-1)th char discarded, null appended. `l` returned on success; otherwise undefined. |
| `int __IML_int_to_hex_string(char * s, size_t n, int x);`<br>`int __IML_uint_to_hex_string(char * s, size_t n, unsigned int x);`<br>`int __IML_int64_to_hex_string(char * s, size_t n, long long x);`<br>`int __IML_uint64_to_hex_string(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[x\|llx], x)`. `n` zero → nothing written, `s` may be null; output beyond (`n`-1)th char discarded, null appended. `l` returned on success; otherwise undefined. |
| `int __IML_i_to_str(char * s, size_t n, int x);`<br>`int __IML_u_to_str(char * s, size_t n, unsigned int x);`<br>`int __IML_ll_to_str(char * s, size_t n, long long x);`<br>`int __IML_ull_to_str(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[d\|u\|lld\|llu], x)`. If `l < n`, all output stored in `s` with null terminator at the end; otherwise output beyond the `n`th character discarded and no null appended. `n` zero → nothing written, `s` may be null. `l` returned on success, otherwise undefined. |
| `int __IML_i_to_oct_str(char * s, size_t n, int x);`<br>`int __IML_u_to_oct_str(char * s, size_t n, unsigned int x);`<br>`int __IML_ll_to_oct_str(char * s, size_t n, long long x);`<br>`int __IML_ull_to_oct_str(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[o\|llo], x)`. If `l < n`, all output stored with null terminator; otherwise output beyond the `n`th character discarded and no null appended. `n` zero → nothing written, `s` may be null. `l` returned on success, otherwise undefined. |
| `int __IML_i_to_hex_str(char * s, size_t n, int x);`<br>`int __IML_u_to_hex_str(char * s, size_t n, unsigned int x);`<br>`int __IML_ll_to_hex_str(char * s, size_t n, long long x);`<br>`int __IML_ull_to_hex_str(char * s, size_t n, unsigned long long x);` | Similar to `snprintf(s, n, %[x\|llx], x)`. If `l < n`, all output stored with null terminator; otherwise output beyond the `n`th character discarded and no null appended. `n` zero → nothing written, `s` may be null. `l` returned on success, otherwise undefined. |

### Routines to Convert ASCII Strings to Floating-point Numbers

Convert the initial portion of decimal string `s` to floating-point number `x`. If no conversion could be performed, zero is returned. If the correct value is outside the range of the return type, plus (`+`) or minus (`-`) `HUGE_VALF`, `HUGE_VAL`, or `HUGE_VALL` is returned, and the value of macro `ERANGE` is stored in `errno`.

| function / signature | description |
|---|---|
| `float __IML_string_to_float(const char * nptr, char ** endptr);`<br>`double __IML_string_to_double(const char * nptr, char ** endptr);`<br>`long double __IML_string_to_long_double(const char * nptr, char ** endptr);` | Similar to `strtof(nptr, endptr)`, `strtod(nptr, endptr)`, and `strtold(nptr, endptr)` in stdlib.h, where `endptr` points to the object that stores the final part of `nptr` when `endptr` is not a null pointer. |
| `float __IML_str_to_f(const char * significand, size_t n, int exponent, char ** endptr);`<br>`double __IML_str_to_d(const char * significand, size_t n, int exponent, char ** endptr);`<br>`long double __IML_str_to_ld(const char * significand, size_t n, int exponent, char ** endptr);` | Convert the initial `n` decimal digits of the significand string multiplied by 10 raised to power of `exponent` to floating-point number as return. `endptr` points to the object that stores the final part of `significand`, provided `endptr` is not a null pointer. |

### Routines to Convert ASCII Strings to Integers

Convert the initial portion of string `s` to integer `x`. If no conversion could be performed, zero is returned. If the correct value is outside the range of the return type, `INT_MIN`, `INT_MAX`, `UINT_MAX`, `LLONG_MIN`, `LLONG_MAX`, or `ULLONG_MAX` is returned, and the value of macro `ERANGE` is stored in `errno`. `endptr` points to the object that stores the final part of `nptr` when `endptr` is not a null pointer.

| function / signature | description |
|---|---|
| `int __IML_string_to_int(const char * nptr, char ** endptr);`<br>`unsigned int __IML_string_to_uint(const char * nptr, char ** endptr);`<br>`long long __IML_string_to_int64(const char * nptr, char ** endptr);`<br>`unsigned long long __IML_string_to_uint64(const char * nptr, char ** endptr);` | Similar to `([unsigned] int)strto[u]l(nptr, endptr, 10)` and `strto[u]ll(nptr, endptr, 10)` in stdlib.h. |
| `int __IML_oct_string_to_int(const char * nptr,char ** endptr);`<br>`unsigned int __IML_oct_string_to_uint(const char * nptr,char ** endptr);`<br>`long long __IML_oct_string_to_int64(const char * nptr,char ** endptr);`<br>`unsigned long long __IML_oct_string_to_uint64(const char * nptr,char ** endptr);` | Similar to `([unsigned] int)strto[u]l(nptr, endptr, 8)` and `strto[u]ll(nptr, endptr, 8)` in stdlib.h. |
| `int __IML_hex_string_to_int(const char * nptr,char ** endptr);`<br>`unsigned int __IML_hex_string_to_uint(const char * nptr,char ** endptr);`<br>`long long __IML_hex_string_to_int64(const char * nptr,char ** endptr);`<br>`unsigned long long __IML_hex_string_to_uint64(const char * nptr,char ** endptr);` | Similar to `([unsigned] int)strto[u]l(nptr, endptr, 16)` and `strto[u]ll(nptr, endptr, 16)` in stdlib.h. |

**`__IML_str_to_i`, `__IML_str_to_u`, `__IML_str_to_ll`, `__IML_str_to_ull`** — convert the initial `n` decimal digits (including an optional `+` or `-` sign) pointed to by `nptr` to integral values. When `endptr` is not a null pointer it points to the object that stores the final part of `nptr`. These functions treat any leading whitespace as invalid.

```c
int __IML_str_to_i(const char * nptr, size_t n, char ** endptr);
unsigned int __IML_str_to_u(const char * nptr, size_t n, char ** endptr);
long long __IML_str_to_ll(const char * nptr, size_t n, char ** endptr);
unsigned long long __IML_str_to_ull(const char * nptr, size_t n, char ** endptr);
```

**`__IML_oct_str_to_i`, `__IML_oct_str_to_u`, `__IML_oct_str_to_ll`, `__IML_oct_str_to_ull`** — convert the initial `n` octal digits (including an optional `+` or `-` sign) pointed to by `nptr` to integral values. When `endptr` is not a null pointer it points to the object that stores the final part of `nptr`. These functions treat any leading whitespace as invalid.

```c
int __IML_oct_str_to_i(const char * nptr,size_t n,char ** endptr);
unsigned int __IML_oct_str_to_u(const char * nptr,size_t n,char ** endptr);
long long __IML_oct_str_to_ll(const char * nptr,size_t n,char ** endptr);
unsigned long long __IML_oct_str_to_ull(const char * nptr,size_t n,char ** endptr);
```

**`__IML_hex_str_to_i`, `__IML_hex_str_to_u`, `__IML_hex_str_to_ll`, `__IML_hex_str_to_ull`** — convert the initial `n` hexadecimal digits (including an optional `+` or `-` sign) pointed to by `nptr` to integral values. When `endptr` is not a null pointer it points to the object that stores the final part of `nptr`. These functions treat any leading whitespace as invalid.

```c
int __IML_hex_str_to_i(const char * nptr,size_t n,char ** endptr);
unsigned int __IML_hex_str_to_u(const char * nptr,size_t n,char ** endptr);
long long __IML_hex_str_to_ll(const char * nptr,size_t n,char ** endptr);
unsigned long long __IML_hex_str_to_ull(const char * nptr,size_t n,char ** endptr);
```

## Macros

The Intel® oneAPI DPC++/C++ Compiler supports the ISO Standard predefined macros and additional predefined macros.

**NOTE:** Single capital letter macros are not supported.

### ISO Standard Predefined Macros

The ISO/ANSI standard for the C language requires that certain predefined macros be supplied with conforming compilers. The compiler includes predefined macros in addition to those required by the standard. The default predefined macros differ among Windows\* and Linux\* operating systems. Differences also exist on Linux as a result of the `-std` compiler option.

| Macro | Value |
|---|---|
| `__DATE__` | The date of compilation as an 11-character string literal in the form `mm dd yyyy`. If the day is less than 10 characters, a space is added before the day value. |
| `__FILE__` | A string literal representing the name of the file being compiled. |
| `__LINE__` | The current line number as a decimal constant. |
| `__STDC_HOSTED__` | Defined and value is `1` only when compiling a C translation unit with `/Qstd=c99`. |
| `__STDC_VERSION__` | Defined and value is `199901L` only when compiling a C translation unit with `/Qstd=c99`. [sic: source line-wraps the name as `__STDC_VERSION_` + `_`] |
| `__TIME__` | The time of compilation as a string literal in the form `hh:mm:ss`. |

### Additional Predefined Macros

The compiler includes predefined macros specified by the ISO/ANSI standard and it also supports the predefined macros listed in the following table.

| Macro | OS support | Description |
|---|---|---|
| `__AVX__` | Linux, Windows | Linux: defined as `1` when option `-march=corei7-avx`, `-xAVX`, or higher processor targeting options are specified. Windows: defined as `1` when option `/QxAVX` or higher processor targeting options are specified. |
| `__AVX2__` | Linux, Windows | Linux: defined as `1` when option `-march=core-avx2`, `-xCORE-AVX2`, or higher processor targeting options are specified. Windows: defined as `1` when option `/QxCORE-AVX2` or higher processor targeting options are specified. **NOTE:** when any of the above options are specified, they also define macro `AVX`. |
| `__AVX512BW__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Byte and Word Instructions (BWI). |
| `__AVX512CD__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Conflict Detection Instructions (CDI). |
| `__AVX512DQ__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Doubleword and Quadword Instructions (DQI). |
| `__AVX512ER__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Exponential and Reciprocal Instructions. |
| `__AVX512F__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Foundation instructions. |
| `__AVX512PF__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 PreFetch Instructions (PFI). |
| `__AVX512VL__` | Linux, Windows | Defined as `1` for processors that support Intel® AVX-512 Vector Length Extensions (VLE). |
| `__BASE_FILE__` | Linux, Windows | Name of source file. |
| `__COUNTER__` | Linux, Windows | Defined as zero. |
| `__cplusplus` | Linux | Defined when compiling C++. The setting depends on which `-std=c++nn` option is in effect. The default is `201703L`. |
| `__ELF__` | Linux | Defined as `1` at the start of compilation. |
| `__EXCEPTIONS` | Linux | Defined as `1` when C++ exceptions are enabled (default for C++). Not defined in C or when option `-fno-exceptions` is specified. |
| `__gnu_linux__` | Linux | Defined as `1` at the start of compilation. |
| `__GNUC__` | Linux | Defined as `4`. |
| `__GNUC_MINOR__` | Linux | Defined as `2`. |
| `__GNUC_PATCHLEVEL__` | Linux | Defined as `1`. |
| `__GNUG__` | Linux | Defined as `4` when compiling C++. |
| `_INTEGRAL_MAX_BITS` | Windows | Defined as `64`. |
| `__INTEL_LLVM_COMPILER` | Linux, Windows | The version of the compiler in the form `VVVVMMUU`, where `VVVV` is the major release version, `MM` is the minor release version, and `UU` is the update number. For example, the base release of 2023.1 is represented by the value `20230100`. This symbol is also recognized by CMake. **NOTE:** to identify the Intel® oneAPI DPC++/C++ Compiler you must check for the existence of both `__INTEL_LLVM_COMPILER` and `SYCL_LANGUAGE_VERSION`, where `SYCL_LANGUAGE_VERSION` is part of the SYCL spec. |
| `__INTEL_PREVIEW_BREAKING_CHANGES` | Linux, Windows | Lets a user tell the compiler that they are willing to give up backward compatibility guarantees and lets the compiler enable new backward breaking changes that will appear in the next major release. Set automatically when compiler option `-fpreview-breaking-changes` is specified. The breaking changes specified will be the default in the next major compiler release, so this option lets you prepare for that release. |
| `__LIBSYCL_MAJOR_VERSION` | Linux, Windows | Set to the SYCL runtime library major version. |
| `__LIBSYCL_MINOR_VERSION` | Linux, Windows | Set to the SYCL runtime library minor version. |
| `__LIBSYCL_PATCH_VERSION` | Linux, Windows | Set to the SYCL runtime library patch version. |
| `__linux__`, `__linux`, `linux` | Linux | Defined as `1` at the start of compilation. |
| `__LONG_DOUBLE_SIZE__` | Linux, Windows | Linux: defined as `80`. Windows: defined as `64`; however, if option `Qlong-double` is specified, it is defined as `80`. |
| `__LONG_MAX__` | Linux, Windows | Linux: defined as `9223372036854775807L`. Windows: defined as `2147483647L`. |
| `__LP64__` | Linux | Defined as `1`. |
| `_M_X64` | Windows | Defined as `100`. |
| `MKL_ILP64` | Linux, Windows | Defined as `1` when `-qmkl-ilp64` or `/Qmkl-ilp64` is specified on the command line, or when used with `-fsycl -qmkl`. |
| `__MMX__` | Linux | Defined as `1`. |
| `_MSC_EXTENSIONS` | Windows | Defined when Microsoft extensions are enabled. |
| `_MSC_FULL_VER` | Windows | The Visual C++ version being used. |
| `_MSC_VER` | Windows | The Visual C++ version being used. |
| `_MT` | Windows | Defined as `1` when a multithreaded dynamic-link library (DLL) is used (that is, when option `/MD[d]` or `/MT[d]` is specified). |
| `__NO_MATH_INLINES` | Linux, Windows | Defined as `1`. |
| `_OPENMP` | Linux, Windows | The default is `202011` when you specify option `[q or Q]openmp`. |
| `__OPTIMIZE__` | Linux, Windows | Defined as `1` when optimization is used. Windows: not defined if option `-O0` is specified or in effect. |
| `__pentium4`, `__pentium4__` | Linux | Defined as `1`. |
| `__PIC__`, `__pic__` | Linux, Windows | Linux: defined as `1` when option `-fpic` is specified. Windows: defined as `2`. |
| `__PTRDIFF_TYPE__` | Linux, Windows | Linux: defined as `long int`. Windows: defined as `long long int`. |
| `__REGISTER_PREFIX__` | Linux | Sets the prefix applied to CPU register names in assembly language. |
| `RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR` | Linux, Windows | Due to implementation limitations, writing to raw pointers obtained from `constant_ptr` is not diagnosed by default. You can enable diagnostics by setting this macro, which allows `constant_ptr` to use constant pointers as underlying pointer types. After enabling the macro, conversions from `constant_ptr` to raw pointers return constant pointers, and writing to `const` pointers is diagnosed by the front end. This behavior does not follow the SYCL specification, since `constant_ptr` conversions to the underlying pointer type will return pointers without any additional qualifiers. This macro is **disabled by default**. |
| `__SIZE_TYPE__` | Linux, Windows | Linux: defined as `unsigned long int`. Windows: defined as `unsigned long long int`. |
| `__SSE__` | Linux, Windows | Defined as `1` for processors that support SSE instructions. |
| `__SSE2__` | Linux, Windows | Defined as `1` for processors that support Intel® SSE2 instructions. |
| `__SSE3__` | Linux, Windows | Defined as `1` for processors that support Intel® SSE3 instructions. |
| `__SSE4_1__` | Linux, Windows | Defined as `1` for processors that support Intel® SSE4 instructions. |
| `__SSE4_2__` | Linux, Windows | Defined as `1` for processors that support SSSE4 instructions. [sic: source says "SSSE4"] |
| `__SSSE3__` | Linux, Windows | Defined as `1` for processors that support SSSE3 instructions. |
| `__SYCL_COMPILER_VERSION` | Linux, Windows | The build date of the SYCL library, presented in the format `YYYYMMDD`. **NOTE:** only available after the SYCL library headers are included in the source code. |
| `SYCL2020_CONFORMANT_APIS` (deprecated) | Linux, Windows | Enables compliance with the SYCL 2020 specification. Useful because some current implementations may be widespread and not conform to that specification. When this macro is defined, it currently has no effect on the API. |
| `SYCL2020_DISABLE_DEPRECATION_WARNINGS` | Linux, Windows | Disables warnings coming from usage of SYCL 1.2.1 APIs that are deprecated in SYCL 2020. |
| `SYCL_DISABLE_DEPRECATION_WARNINGS` | Linux, Windows | Disables all deprecation warnings in SYCL runtime headers, including SYCL 1.2.1 deprecations. |
| `SYCL_DISABLE_IMAGE_ASPECT_WARNING` | Linux, Windows | Disables the warning diagnostic issued when calling `device::has(aspect::image)` and `platform::has(aspect::image)`. |
| `SYCL_FALLBACK_ASSERT` | Linux, Windows | Defining as non-zero enables the fallback assert feature even on devices without native support; this process adds overhead associated with submitting kernels that call `assert()`. When this macro is defined as `'0'` or is not defined, the logic for detecting assertion failures in kernels is disabled, so a failed assert does not cause a message to be printed and does not cause the program to abort. Some devices have native support for assertions; the logic for detecting assertion failures is always enabled on these devices regardless of whether this macro is defined, because that logic does not add extra overhead. You can check whether a device has native support for `assert()` via `aspect::ext_oneapi_native_assert`. This macro is **undefined by default**. |
| `SYCL_LANGUAGE_VERSION` | Linux, Windows | An integer reflecting the version number and revision of the SYCL language that is supported by the implementation. Enables compliance with the SYCL 2020 specification. |
| `SYCL_USE_NATIVE_FP_ATOMICS` | Linux, Windows | Enables functions to generate built-in floating-point atomics on the target device. If the target device does not support floating-point atomics, emulated atomics are used instead. **Enabled by default.** |
| `unix`, `__unix`, `__unix__` | Linux | Defined as `1`. |
| `__USER_LABEL_PREFIX__` | Linux | The prefix applied to user labels in assembly language. |
| `__VERSION__` | Linux | The compiler version string. |
| `__WCHAR_T` | Linux | Defined as `1`. |
| `_WCHAR_T_DEFINED` | Windows | Defined when option `/Zc:wchar_t` is specified or `wctype_t` is defined in the header file. |
| `__WCHAR_TYPE__` | Linux, Windows | Linux: defined as `int`. Windows: defined as `unsigned short int`. |
| `_WCTYPE_T_DEFINED` | Windows | Defined when `wctype_t` is defined in the header file. |
| `_WIN64` | Windows | Defined as `1`. |
| `__WINT_TYPE__` | Linux, Windows | Linux: defined as `unsigned int`. Windows: defined as `unsigned short int`. |
| `__x86_64`, `__x86_64__` | Linux | Defined as `1`. |

See also: `arch` compiler option; `march` compiler option; `m` compiler option; `D` compiler option; `U` compiler option; `qopenmp`, `Qopenmp` compiler option; `x`, `Qx` compiler option; ISO Standard Predefined Macros.

### Use Predefined Macros to Specify Intel® Compilers

This topic shows how to use predefined macros to specify an Intel® compiler or version of an Intel compiler.

**Predefined Macros to Specify Compiler and Version**

| Compiler | Predefined Macros to Differentiate from Other Compiler | Notes |
|---|---|---|
| Intel® DPC++ Compiler | `SYCL_LANGUAGE_VERSION`, `__INTEL_LLVM_COMPILER`, `__VERSION` | `SYCL_LANGUAGE_VERSION` is defined in the SYCL specification and should be defined by all SYCL compilers. `__INTEL_LLVM_COMPILER` is used to select the compiler. `__VERSION` is used to select the compiler version. |
| Intel® C++ Compiler | `__INTEL_LLVM_COMPILER`, `__VERSION` | `__INTEL_LLVM_COMPILER` is used to select the compiler. `__VERSION` is used to select the compiler version. |

**Predefined Macros for Intel® DPC++ Compiler.** The following example uses `#if defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)` to define a code block specific to the Intel® DPC++ Compiler:

```c
#if defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)
    // code specific for Intel DPC++ Compiler below
    // ... ...

    // example only
    std::cout << "SYCL_LANGUAGE_VERSION: " << SYCL_LANGUAGE_VERSION << std::endl;
```

## Gotchas & failure modes

- **`align_value` is an unchecked promise.** If the data is not actually aligned to the designated value, the behavior is undefined — the compiler generates code on the assumption of alignment.
- **`allow_cpu_features` changes code generation, it does not check the CPU.** The function "is generated as if the specified features are available", so calling it on a processor lacking those features is a wrong-code/illegal-instruction hazard. Pass `0` for `featp1` when only page-two features apply.
- **`cpu_dispatch`/`cpu_specific` are Intel-only.** On an unsupported non-Intel processor the compile fails with `"invalid option"`.
- **`code_align` on a procedure compounds with a loop attribute:** procedure `code_align(k)` plus loop `code_align(n)` aligns both to `max(n,k)`.
- **`code_align(1)` disables alignment**; omitting `n` means 16 bytes. Value must be a power of 2 in [1, 4096].
- **Intrinsics are host-only** in this compiler; device code cannot call them. They also need `immintrin.h` and a `target` attribute for the intended architecture.
- **`fp-model fast` is the default** and assumes no signed zeros, no infinities, no NaNs, and flushes denormals to zero — numerically significant differences from strict IEEE 754-2008 behavior.
- **Mixing static and dynamic Intel libraries:** by default `libirc`, `libsvml`, `libimf`, `libirng` are linked statically; `-shared` links them dynamically and requires the libraries on the target system. `-shared-intel` links Intel libraries dynamically (smaller binary, runtime dependency); `-static-intel` links them statically. `-fpic` is required when compiling each object file included in a shared library.
- **Order matters when linking:** object files are passed to the linker left to right, then objects/libraries from default configuration files, then default Intel and system libraries. With `icpx lib1.a file.cpp lib2.a` the order is `lib1.a`, `file.o`, `lib2.a`, then defaults.
- **LTO changes the librarian:** if `flto` was used at compile time, create the archive with `llvm-ar`/`llvm-lib`, not `ar`/`lib`.
- **SYCL linked by another compiler fails without the environment:** if the compiler environment is not sourced, linking fails with undefined references in `libsycl.so` and other internal libraries; add the listed `LD_LIBRARY_PATH` directories.
- **`libqkmalloc` is not thread safe** and does not support threaded code such as OpenMP\*; do not use simultaneously from multiple threads, and it is limited to Intel processors (redirects to standard C routines at runtime on non-Intel processors). It requires large throughput workloads for best results.
- **SYCL applications have extra deployment dependencies:** the dependency list from `ldd`/`dumpbin /DEPENDENTS` is not sufficient — redistribute shared libraries for each target being programmed for (device offload).
- **Minor-release compatibility:** a 2021.1-built application works with all 2021.x, but compatibility is not guaranteed with 2022.x or 19.x; major version increments signal API/ABI breaks.
- **`RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR` is disabled by default** and enabling it changes `constant_ptr` conversion semantics away from the SYCL specification.
- **`SYCL2020_CONFORMANT_APIS` is deprecated** and currently has no effect on the API.
- **`SYCL_FALLBACK_ASSERT` is undefined by default**; when it is `0`/undefined, a failed `assert()` in a kernel prints nothing and does not abort, except on devices with native assert support (`aspect::ext_oneapi_native_assert`), where detection is always enabled.
- **Single capital letter macros are not supported.**
- **`__NO_MATH_INLINES` is defined as 1** on both Linux and Windows, and `__OPTIMIZE__` is not defined when `-O0` is specified or in effect (Windows note).

## Source map

- IEEE floating-point special values / IEEE 754-2008 — p. 398
- Attributes (Use Attributes; `align`; `align_value`; `allow_cpu_features`; `code_align`; `const`; `cpu_dispatch`, `cpu_specific`; `target`) — pp. 398–406
- Intrinsics; availability of intrinsics — p. 406
- Libraries (intro; Create Libraries; Static Libraries; Shared Libraries; Use Intel Shared Libraries; Manage Libraries; Compile with SYCL and Link Other Compilers; Other Considerations; Redistribute Libraries; Shared Library Deployment; Resolve Shared Library Dependencies for Private/Public Model; Shared Library Dependencies for Device Offload; minor-release compatibility; Deployment Models; Additional Steps; Redistributable Library Considerations; Intel's Memory Allocator Library `libqkmalloc`) — pp. 406–418
- SIMD Data Layout Templates (SDLT) — pp. 418–479
- Intel® C++ Class Libraries — pp. 480–534
- C++ Asynchronous I/O Extensions — pp. 535–554
- IEEE 754-2008 Binary Floating-Point Conformance Library — pp. 555–578
- Numeric String Conversion Library — pp. 579–586
- Macros (intro; ISO Standard Predefined Macros; Additional Predefined Macros; Use Predefined Macros to Specify Intel® Compilers) — pp. 586–594
