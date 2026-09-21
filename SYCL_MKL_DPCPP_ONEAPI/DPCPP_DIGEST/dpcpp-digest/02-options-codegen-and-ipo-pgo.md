---
chunk: 02-options-codegen-and-ipo-pgo
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 115-227
covers: Code generation options; offload/OpenMP*/SYCL* and parallel processing options; interprocedural optimization (IPO); profile-guided optimization (PGO); optimization report options
---

# Code Generation, Offload/OpenMP*/SYCL*, IPO, PGO, and Optimization Report Options

> **Scope.** dg pp. 115-227: how the compiler selects target instruction sets and generates code, compiles and links OpenMP*/SYCL* offload code, performs interprocedural (IPO/LTO) and profile-guided optimization, and writes optimization reports. Entries preserve exact Linux/Windows spellings, legal values, defaults, platform notes, and deprecations.

## Key facts

- Syntax split: Linux `-x…`, `-m…`, `-f…`; Windows `/Qx…`, `/arch:…`, `/Q…`. Some `-f…` options are documented on Windows too (listed per entry).
- Without `-x`/`-march` (Linux) or `/Qx`/`/arch` (Windows), the default target supports Intel® SSE2; without `-march` the compiler may generate Intel® SSE2 and SSE. Intel® and non-Intel® baselines have identical optimizations and both default to SSE4.2 for x86-based architectures.
- `ICELAKE` (alias of `ICELAKE-CLIENT`) is deprecated in `arch`, `ax, Qax`, `x, Qx`.
- Recurring notes: **host compilation only — no impact on device-specific compilation when offloading is enabled**, and its inverse, **applies only to device-specific compilation when SYCL*/OpenMP* offloading is enabled** (see Gotchas for the split).
- Referenced specs: OpenMP 5.2 (e.g., section 5.8.1), Khronos* Group SYCL* 1.2.1 Specification, SYCL 2020 specification, IEEE 754 / IEEE 754-2008, DWARF2/DWARF3, ELF, LLVM/Clang.
- Numbers: `-fopenmp-target-buffers=4GB` covers >4GB objects (4294959104 bytes); default SYCL divide precision 2.5 ULP and default `sycl::sqrt` 3 ULP without the matching `-foffload-fp32-prec-*` flag; PonteVecchio (PVC) sub-group sizes 16 and 32; PVC register files 128 (small) / 256 (large).
- `-fsycl` compiles a C++ program as a SYCL program instead of plain C++11; `-fsycl-device-obj` defaults to IP-based fat objects (`llvmir`).
- Beginning with release **2025.0**, `-qopt-report`/`/Qopt-report` no longer emits the YAML report; use Clang `-fsave-optimization-record`.
- Deprecated/removal-tracked: `device-math-lib` (no replacement), `fsycl-allow-device-dependencies` (→ `fsycl-allow-device-image-dependencies`), `fsycl-explicit-simd`, `fprofile-ml-use` (no replacement), `m64`/`Qm64` (do nothing), static OpenMP* libraries, `-fp`, `ICELAKE`, `-fsycl-targets` values containing `-sycldevice`.

**Shared value bundles** (`arch`, `ax`, `x`, `march`, `mtune`):

- **Microarchitecture code names** (`arch`/`ax`/`x`; each "may generate instructions for processors that support the specified Intel® processor or microarchitecture code name", and for `x` also optimizes for it): `ALDERLAKE`, `AMBERLAKE`, `ARROWLAKE`, `ARROWLAKE-S`, `BROADWELL`, `CANNONLAKE`, `CASCADELAKE`, `CLEARWATERFOREST`, `COFFEELAKE`, `COOPERLAKE`, `DIAMONDRAPIDS`, `EMERALDRAPIDS`, `GOLDMONT`, `GOLDMONT-PLUS`, `GRANDRIDGE`, `GRANITERAPIDS`, `GRANITERAPIDS-D`, `HASWELL`, `ICELAKE-CLIENT` (or `ICELAKE`), `ICELAKE-SERVER`, `IVYBRIDGE`, `KABYLAKE`, `LUNARLAKE`, `PANTHERLAKE`, `ROCKETLAKE`, `SANDYBRIDGE`, `SAPPHIRERAPIDS`, `SIERRAFOREST`, `SILVERMONT`, `SKYLAKE`, `SKYLAKE-AVX512`, `TIGERLAKE`, `TREMONT`, `WHISKEYLAKE`.
- **ISA bundles:** `CORE-AVX512`/`x86-64-v4` = Intel® AVX-512 Foundation, Conflict Detection Instructions (CDI), Doubleword/Quadword Instructions (DQI), Byte/Word Instructions (BWI), Vector Length Extensions (VLE) + `CORE-AVX2` set. `COMMON-AVX512` = Foundation + CDI + `CORE-AVX2` set. `CORE-AVX2` = Intel® AVX2, Intel® AVX, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3. `CORE-AVX-I` = Float-16 conversion + RDRND, Intel® AVX, Intel® SSE4.2/4.1/3/2, SSE, SSSE3. `AVX2` = Intel® AVX2 + AVX + SSE4.2/4.1/3/2, SSE, SSSE3. `AVX` = Intel® AVX + SSE4.2/4.1/3/2, SSE, SSSE3. `SSE4.2` = SSE4.2/4.1/3/2, SSE, SSSE3. `SSE4.1` = SSE4.1/3/2, SSE, SSSE3. `SSSE3` = SSSE3, Intel® SSE3, SSE2, SSE. `SSE3` = Intel® SSE3, SSE2, SSE.

## Option quick table

| option | syntax (Linux / Windows) | purpose · default · key notes |
|---|---|---|
| `arch` | Windows `/arch:code` | target feature set; default SSE2; host-only; `ICELAKE` deprecated |
| `ax, Qax` | `-axcode` / `/Qaxcode` | auto-dispatch paths; OFF; comma-separated no spaces; host-only |
| `EH` | Windows `/EHtype`, `/EHtype-` | `a`/`s`/`c`; OFF (some EH); `c` needs `a`/`s`; `/EHsc` alt `/GX` |
| `fasynchronous-unwind-tables` | `-f…`/`-fno-…` | unwind precision; default `-f…`; Linux only |
| `fcf-protection, Qcf-protection` | `-fcf-protection[=kw]` / `/Qcf-protection[:kw]` | CET `return`/`branch`/`full`/`none`; default `none`; host-only |
| `fdata-sections, Gw` | `-fdata-sections` / `/Gw` | COMDAT per data item; OFF; pair with linker GC |
| `fexceptions` | `-fexceptions`/`-fno-…` | EH tables; on for C++, off for C; `-fno-` undefines `__EXCEPTIONS` |
| `ffunction-sections, Gy` | `-ffunction-sections` / `/Gy` | COMDAT per function; OFF |
| `fomit-frame-pointer` | `-fomit-…`/`-fno-omit-…` | EBP as GP register; default `-fomit-…`; `-O0`/`-g` flips; `-fp` deprecated |
| `Gd` | Windows `/Gd` | `__cdecl` default; ON |
| `GR` | Windows `/GR`, `/GR-` | RTTI; default `/GR` |
| `guard` | Windows `/guard:kw` | `cf[-]`, `cf,nochecks`, `ehcont[-]`; OFF; compiler+linker; host-only |
| `Gv` | Windows `/Gv` | `__vectorcall` for vector args; OFF; host-only |
| `m, Qm` | `-mcode` / `/Qmcode` | CPUID ISA extensions; default SSE2; host-only |
| `m64, Qm64` | `-m64` / `/Qm64` | deprecated, does nothing |
| `m80387` | `-m80387`/`-mno-…` | x87; default `-m80387`; alt `-m[no-]x87`; host-only |
| `march` | Linux `-march=processor` | baseline CPU features; long lowercase list + `x86-64*`; OFF (SSE2/SSE) |
| `masm` | Linux `-masm=dialect` | `att` (default) / `intel`; host-only |
| `mauto-arch, Qauto-arch` | `-mauto-arch=value` / `/Qauto-arch:value` | x86 auto-dispatch, value = any `[Q]ax` setting; OFF; not with `[Q]x`/`[Q]ax` |
| `mbranches-within-32B-boundaries, Q…` | `-m…`/`-mno-…` ; `/Q…`/`/Q…-` | 32B branch alignment; default `-mno-…`; may hurt debugability |
| `mintrinsic-promote, Qintrinsic-promote` | `-m…` / `/Q…` | promote function ISA for intrinsics; OFF; prefer `__attribute__((target(…)))` |
| `momit-leaf-frame-pointer` | `-momit-…`/`-mno-omit-…` | leaf frame pointer; default varies with `-f[no-]omit-frame-pointer` |
| `mtune, tune` | `-mtune=processor` / `/tune:processor` | tuning without new ISA; default `generic`; backwards compatible |
| `regcall, Qregcall` | `-regcall` / `/Qregcall` | default `__regcall`; OFF; ignored for varargs |
| `x, Qx` | `-xcode` / `/Qxcode` | Intel® features + optimization; OFF; CPU check unless `-O0`/`/Od`; exclusive with `-march`/`/arch` |
| `xHost, QxHost` | `-xHost` / `/QxHost` | highest host ISA; OFF; host-only |
| `device-math-lib` | `-device-math-lib=…`,`-no-…` / `/device-math-lib:…`,`/no-…` | `fp32`,`fp64`; default `fp32, fp64`; deprecated, no replacement |
| `fiopenmp, Qiopenmp` | `-fiopenmp` / `/Qiopenmp` | Intel backend OpenMP*; OFF; alt `-qopenmp`/`/Qopenmp`; not `-fopenmp` |
| `flink-huge-device-code` | `-f…`/`-fno-…` | device code later in binary (>2GB); `-fno-…`; needs `-fsycl` or `-fopenmp-targets` + link action |
| `fno-sycl-libspirv` | both `-fno-sycl-libspirv` | disables libspirv check; check on by default; device-only |
| `foffload-fp32-prec-div` | `-f…`/`-fno-…` | correctly rounded divide; on by default; else 2.5 ULP |
| `foffload-fp32-prec-sqrt` | `-f…`/`-fno-…` | correctly rounded `sycl::sqrt`; on by default; else 3 ULP |
| `fopenmp` | Linux `-fopenmp` | LLVM-community OpenMP*; OFF; no GPU offload; host-only |
| `fopenmp-concurrent-host-device-compile, Q…` | `-f…` / `/Q…` | parallel host+device OpenMP* compile; OFF; experimental |
| `fopenmp-declare-target-scalar-defaultmap, Q…` | `=kw` / `:kw` | `default`,`firstprivate`; default `default`; can change results |
| `fopenmp-device-code-split, Q…` | `=[triple=]per_kernel` / `:[triple=]per_kernel` | SPIR-V* kernel split for AOT; OFF; device-only |
| `fopenmp-device-lib` | `=lib[,…]`,`-fno-…=lib[,…]` | `libm-fp32`,`libm-fp64`,`libc`,`all`; OFF; no spaces; `all` supersedes |
| `fopenmp-device-link, Q…` | `-f…` / `/Q…` | device link at compile step; OFF; SPIR64 only |
| `fopenmp-max-parallel-link-jobs, Q…` | `=num` / `:num` | parallel device-link actions; OFF; device-only |
| `fopenmp-offload-mandatory, Q…` | `-f…` / `/Q…` | device-only target regions; OFF; runtime error if offload fails |
| `fopenmp-target-buffers, Q…` | `=kw` / `:kw` | `default`,`4GB`; default `default`; possible Intel® GPU perf loss |
| `fopenmp-target-default-sub-group-size, Q…` | `=val` / `:val` | SPMD sub-group size; PVC 16/32; OFF; ignored for SIMD kernels |
| `fopenmp-target-loopopt, Q…` | `-f…` / `/Q…` | loop optimizer + autovec for device; OFF; `O2`+; SPIR64 |
| `fopenmp-target-simd, Q…` | `-f…` / `/Q…` | device SIMD loop vectorization; OFF; `O2`+; SPIR64 |
| `fopenmp-target-teams-default-vla-alloc-mode, Q…` | `=arg` / `:arg` | `malloc`,`wilocal` (default) |
| `fopenmp-targets, Qopenmp-targets` | `=triple` / `:triple` | `spir64`,`spir64_x86_64`,`spir64_gen`; OFF; needs OpenMP* enablement |
| `fsycl` | both `-fsycl` | SYCL not C++11; SYCL ON, C++ OFF; Windows sets `/MD`, no `/MT` |
| `fsycl-add-default-spec-consts-image` | `-fsycl-…`/`-fno-sycl-…` | device-image copies with default spec constants; OFF; AOT only |
| `fsycl-allow-device-dependencies` | both forms | device-image dependencies; `-fno-…`; **deprecated** |
| `fsycl-allow-device-image-dependencies` | both forms | device-image dependencies; `-fno-…` |
| `fsycl-dead-args-optimization` | both forms | remove dead kernel args; OFF (may change) |
| `fsycl-device-code-split` | `[=value]` | `per_kernel`,`per_source`,`off`,`auto` (default) |
| `fsycl-device-lib` | `=lib[,…]`,`-fno-…=lib[,…]` | `libm-fp32`,`libm-fp64`,`libc`,`all`; OFF; Windows `-fno-` form garbled in source |
| `fsycl-device-obj` | `=arg` | `llvmir` (default),`spirv`; experimental |
| `fsycl-device-only` | both `-fsycl-device-only` | device-only binary; OFF |
| `fsycl-early-optimizations` | both forms | LLVM opts before SPIR-V*; ON |
| `fsycl-enable-function-pointers` | both `-fsycl-enable-function-pointers` | function pointers/virtual fns; OFF; experimental; CPU-device only |
| `fsycl-esimd-force-stateless-mem` | both forms | stateless ESIMD access; OFF; experimental |
| `fsycl-explicit-simd` | both forms | "Explicit SIMD"; `-fno-…`; **deprecated**, APIs may change |
| `fsycl-force-target` | `=triple` | force extraction target `spir64`/`spir64_x86_64`/`spir64_gen`; OFF |
| `fsycl-fp64-conv-emu` | both `-fsycl-fp64-conv-emu` | fp64 partial emulation; OFF; needs capable Intel GPU |
| `fsycl-help` | `[=arg]` | `x86_64`,`gen`,`all`; OFF; `all` == no arg |
| `fsycl-host-compiler` | `=arg` | host compiler name/path; OFF (Intel® DPC++ Compiler) |
| `fsycl-host-compiler-options` | `="opts"` | options for that compiler; OFF; phase-limiting opts cause UB |
| `fsycl-id-queries-fit-in-int` | both forms | assume ID queries fit `MAX_INT`; ON |
| `fsycl-instrument-device-code` | both forms | ITT device libraries for VTune™; ON |
| `fsycl-link` | both `-fsycl-link` | partial device-binary link; OFF |
| `fsycl-max-parallel-link-jobs` | `=n` | parallel SYCL link processes; `=1`; experimental |
| `fsycl-optimize-non-user-code` | both `-fsycl-optimize-non-user-code` | optimize framework not kernels; OFF; needs `-O0`/`/Od`; host-only |
| `fsycl-pstl-offload` | `[=arg]`,`-fno-…` / `[:arg]`,`/fno-…` | `cpu`,`gpu`; `-fno-…`; oneDPL required |
| `fsycl-rdc` | both forms | relocatable device code; default `-fsycl-rdc`; `-fno-` blocks `SYCL_EXTERNAL` |
| `fsycl-remove-unused-external-funcs` | both forms | remove unused `SYCL_EXTERNAL`; default `-fsycl-…` |
| `fsycl-targets` | `=T1,…,Tn` | `spir64` (default),`spir64_x86_64`,`x86_64`,`spir64_gen`; `-sycldevice` forms deprecated |
| `fsycl-unnamed-lambda` | both forms | unnamed SYCL* lambda kernels; ON |
| `fsycl-use-bitcode` | both `-fsycl-use-bitcode` | LLVM IR bitcode in fat objects; ON |
| `ftarget-compile-fast` | `-f…` / `/f…` | faster, less optimal target code; OFF; experimental; host-only |
| `ftarget-export-symbols` | both forms | expose target library symbols; `-fno-…` |
| `ftarget-register-alloc-mode, Qtarget-register-alloc-mode` | `=device:mode[,…]` / `:…` | `pvc`; `default`,`small`(128 regs),`large`(256),`auto`; Linux `pvc:auto`, Windows `pvc:default` |
| `nolibsycl` | both `-nolibsycl` | no SYCL* runtime link; OFF; host-only |
| `qopenmp, Qopenmp` | `-qopenmp`/`-qno-openmp` ; `/Qopenmp`/`/Qopenmp-` | Intel OpenMP*; off by default; alt `-fiopenmp`/`/Qiopenmp`; host-only |
| `qopenmp-link` | Linux `-qopenmp-link=library` | `static` (deprecated libs),`dynamic` (default); `dynamic`+`-static` error; host-only |
| `qopenmp-simd, Qopenmp-simd` | `-qopenmp-simd`/`-qno-…` ; `/Q…`/`/Q…-` | SIMD-only OpenMP*; off; no OpenMP RTL needed |
| `qopenmp-stubs, Qopenmp-stubs` | `-q…` / `/Q…` | sequential OpenMP*; OFF; stub library linked; host-only |
| `Wno-sycl-strict` | both `-Wno-sycl-strict` | disable strict SYCL* warnings; OFF |
| `Xopenmp-target` | both `-Xopenmp-target-tool=T "options"` | tool `frontend`/`backend`/`linker`; OFF; device-only |
| `Xs` | both `-Xs -option` or `-Xsoption` | backend options; OFF; AOT options are not compiler options |
| `Xsycl-target` | both `-Xsycl-target-tool=T "options"` | tool `frontend`/`backend`/`linker`; OFF; device-only |
| `flto` | `-flto[=arg]`/`-fno-lto` | `full` (default),`thin`; no LTO by default; host-only; `-ipo`/`/Qipo` aliases |
| `ipo, Qipo` | `-ipo`/`-no-ipo` ; `/Qipo`/`/Qipo-` | multifile IPO/WPO; off; one unnamed object file; host-only |
| `fprofile-dwo-dir` | `=dir` / `:dir` | `.dwo` directory; OFF; experimental; host-only |
| `fprofile-ml-use` | `-f…` / `/f…` | ML branch probabilities; OFF; **deprecated**, no replacement |
| `fprofile-sample-generate` | `[=level]` / `[:level]` | `none`,`keep-all-opt` (default),`med-fidelity`,`max-fidelity`; Windows needs LLD |
| `fprofile-sample-use` | `=file`/`-fno-…` | HWPGO data from `llvm-profgen`; off; experimental |
| `qopt-report, Qopt-report` | `[=arg]` | `0`,`1`/`low`,`2`/`medium`,`3`/`high`; OFF; YAML removed in 2025.0 |
| `qopt-report-file, Qopt-report-file` | `=kw` / `:kw` | `filename`,`stderr`,`stdout`; OFF; host-only |
| `qopt-report-names, Qopt-report-names` | `=kw` / `:kw` | `mangled`,`unmangled` (default); must name one |
| `qopt-report-phase, Qopt-report-phase` | `[=list]` / `[:list]` | `cg`,`ipo`,`loop`,`openmp`,`pgo`,`vec`,`all` (default); phase prerequisites |
| `qopt-report-stdout, Qopt-report-stdout` | `-q…` / `/Q…` | report to stdout; OFF; same as file=stdout |

## Code Generation Options

Source: "options that pertain to code generation… listed in alphabetical order." All entries in this section are **host-only** (no device-specific impact under offload).

### `vecabi` — vector function ABI (entry tail, p. 115)

For `#pragma omp declare simd (ompx_processor(core_2_duo_sse4_1)) int foo(int a);` with `-axAVX, CORE-AVX2`: under `gcc`, a vector version is created for Intel® SSE2, AVX, AVX2, AVX512 and these variants are always created independently of target options; under `cmdtarget`, a vector version is created for Intel® SSE2 (default because no `-x` option is used), Intel® SSE4.1 (by vector function specification), and Intel® AVX2 (by the `ax` option value).

> NOTE: To avoid possible link-time and runtime errors, use identical `[Q]vecabi` settings when compiling all files in an application that define or use vector functions, including libraries. If setting `cmdtarget` is specified, options `[Q]x` and/or `[Q]ax` must have identical values.

ABI references: downloadable PDF "Vector Function Application Binary Interface" (Intel®-compatible ABI); GCC vector functions ABI via the Libmvec - vector math library item in the GLIBC wiki at sourceware.org.

### `arch`
Windows only `/arch:code`. Values: shared microarchitecture list (each may generate instructions for the named processor/microarchitecture) plus bundles `CORE-AVX2`, `CORE-AVX-I`, `AVX2`, `AVX`, `SSE4.2`, `SSE4.1`, `SSSE3`, `SSE3`. Default `varies`: Intel® SSE2 if `arch` is unspecified. Generated code should execute on any compatible non-Intel processor supporting the instruction set. See also `x, Qx`; `xHost, QxHost`; `ax, Qax`; `march`; `m`.

### `ax, Qax`
`-axcode` / `/Qaxcode`. Multiple feature-specific auto-dispatch paths for Intel® processors when there is a performance benefit, plus a baseline path; the feature-specific path is usually more optimized than the baseline, whose optimization level is controlled by options such as `O3`. Values: shared microarchitecture list plus bundles `COMMON-AVX512`, `CORE-AVX512`, `CORE-AVX2`, `CORE-AVX-I`, `AVX`, `SSE4.2`, `SSE4.1`, `SSSE3`, `SSE3`; `ATOM_SSE4.2` and `ATOM_SSSE3` may generate MOVBE depending on `-minstruction` (Linux) or `/Qinstruction` (Windows) and optimize for the corresponding Intel Atom® processors. Default `OFF` (feature-specific code is then controlled by `-march`/`-x` or `/arch`/`/Qx`). Multiple values are comma-separated with no spaces. Baseline architecture comes from `-march`/`-x` or `/arch`/`/Qx` and becomes the effective minimum architecture for baseline code; with both `[Q]ax` and `[Q]x`, baseline runs only on Intel® processors compatible with `[Q]x`; with both `-ax` and `-march` (`/Qax` and `/arch`), baseline runs on non-Intel® processors compatible with `-march` (`/arch`). Intel® and non-Intel® baselines have identical optimizations, both defaulting to SSE4.2 for x86-based architectures. The compiler versions a function only if it judges a likely performance gain, then emits a feature-specific and a baseline version; at runtime one is selected by processor, and a non-Intel processor always runs the baseline.

> NOTE: With the `icx` compiler, if you experience any program failure when using `-ax` during this release, remove the option to see if that solves the problem; if it does, report a bug.

```bash
icx -axSKYLAKE file.cpp               ! Linux* systems
icx /QaxSKYLAKE file.cpp              ! Windows* systems
icx -axSKYLAKE,BROADWELL file.cpp     ! Linux* systems
icx /QaxBROADWELL,SKYLAKE file.cpp    ! Windows* systems
```

### `EH`
Windows only `/EHtype`, `/EHtype-`. Values `a` asynchronous C++ EH model; `s` synchronous C++ EH model; `c` assume `extern "C"` functions do not throw (requires `a` or `s`). Default `OFF` (some exception handling by default). The negative form disables handling by `type`, or by the last type if two are given: `/EHsc-` is interpreted as `/EHs`. Alternate `/EHsc`: Linux None, Windows `/GX`.

### `fcf-protection, Qcf-protection`
`-fcf-protection[=keyword]` / `/Qcf-protection[:keyword]`. Intel® CET protection (preliminary support) against vulnerability-exploiting attacks; enforced on CET-capable processors and ignored elsewhere, so safe across a variety of processors. Values `return` shadow stack protection; `branch` endbranch (EB) generation; `full` both (same as no keyword); `none` disabled. Default `none`. Shadow stack protects against return-oriented programming (ROP); EB generation protects against call/jump-oriented programming (COP/JOP), where JOP uses indirect jumps/calls to emulate returns and COP employs indirect calls. Alternate Linux `-qcf-protection`; Windows None.

### `fomit-frame-pointer`
`-fomit-frame-pointer`, `-fno-omit-frame-pointer` (Linux only). Default `-fomit-frame-pointer` (EBP used as a general-purpose register in optimizations); if `-O0` or `-g` is specified the default is `-fno-omit-frame-pointer`. `-fno-omit-frame-pointer` makes the compiler maintain and use EBP as a stack frame pointer for all functions so a debugger can still produce a stack backtrace. `-fno-omit-frame-pointer` is set by `-O0` or `-g`; `-fomit-frame-pointer` is set by `-O1`, `-O2`, or `-O3`. Alternate Linux `-fp` (deprecated). See also `momit-leaf-frame-pointer`.

> NOTE: On Linux, there is currently an issue with GCC 3.2 exception handling; the compiler ignores this option when GCC 3.2 is installed for C++ and exception handling is turned on (the default).

### `guard`
Windows only `/guard:keyword`. Control flow protection. Values `cf[-]` analyze control flow of valid targets for indirect calls and insert runtime target verification (`/guard:cf-` disables); `cf,nochecks` emit only the table of address-taken functions, no checks; `ehcont[-]` generate a sorted list of the relative virtual addresses (RVA) of all valid exception handling continuation targets for a binary, enabling EH Continuation Guard used at runtime for `NtContinue` and `SetThreadContext` instruction pointer validation (`/guard:ehcont-` disables). Default `OFF`. `/guard:cf`, `/guard:cf,nochecks`, and `/guard:ehcont` must be passed to both compiler and linker; code compiled with `/guard:cf` can link to libraries/objects not compiled with it. Added for Microsoft compatibility; `cf` and `ehcont` use the Microsoft implementation.

### `march`
Linux only `-march=processor`. Generates code using the CPU feature set of a specific processor as the baseline. Values `nocona`, `core2`, `penryn`, `bonnell`, `atom`, `silvermont`, `slm`, `goldmont`, `goldmont-plus`, `tremont`, `gracemont`, `nehalem`, `corei7`, `westmere`, `sandybridge`, `corei7-avx`, `ivybridge`, `core-avx-i`, `haswell`, `core-avx2`, `broadwell`, `common-avx512`, `skylake`, `skylake-avx512`, `skx`, `cascadelake`, `cooperlake`, `cannonlake`, `icelake-client`, `rocketlake`, `icelake-server`, `tigerlake`, `sapphirerapids`, `alderlake`, `raptorlake`, `meteorlake`, `sierraforest`, `grandridge`, `graniterapids`, `emeraldrapids`; `x86-64` (generic CPU with 64-bit extensions); `x86-64-v2` (Intel® SSE4.3, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3); `x86-64-v3` (Intel® AVX2, AVX, SSE4.3, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3); `x86-64-v4` (Intel® AVX-512 Foundation, CDI, DQI, BWI, VLE). Default `OFF`: without `-march` the compiler may generate Intel® SSE2 and SSE instructions.

### `mintrinsic-promote, Qintrinsic-promote`
`-mintrinsic-promote` / `/Qintrinsic-promote`. Functions containing calls to intrinsics that require a specific CPU feature have their target architecture automatically promoted to allow the required feature. Default `OFF`: without it, calling such an intrinsic when the specified (or default) target processor lacks the feature reports an error. All code in the promoted function is compiled for that target architecture and will not execute correctly on processors lacking the feature; you are responsible for guarding the execution path at runtime.

> NOTE: We recommend `__attribute__((target(<required target>)))` to mark functions intended for specific target architectures instead of this option; the attribute provides significantly better compile time error checking.

### `momit-leaf-frame-pointer`
`-momit-leaf-frame-pointer`, `-mno-omit-leaf-frame-pointer` (Linux only). Determines whether the frame pointer is omitted or kept in leaf functions. Default `Varies`: if `-fomit-frame-pointer` is specified (or set by default), default `-momit-leaf-frame-pointer`; if `-fno-omit-frame-pointer` is specified, default `-mno-omit-leaf-frame-pointer`.

| Option combination | Result |
|---|---|
| `-fomit-frame-pointer` + `-momit-leaf-frame-pointer` or `-mno-omit-leaf-frame-pointer` | Both are the same as `-fomit-frame-pointer`; frame pointers omitted for all routines. |
| `-fno-omit-frame-pointer -momit-leaf-frame-pointer` | Frame pointer omitted for leaf routines, kept for others — the intended effect. |
| `-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer` | `-mno-omit-leaf-frame-pointer` is ignored since `-fno-omit-frame-pointer` retains frame pointers in all routines. |

Provided for compatibility with gcc. See also `fomit-frame-pointer`.

### `mtune, tune`
`-mtune=processor` / `/tune:processor`. Tuning for specific processors without causing extended instruction sets to be used (unlike `-march`); resulting executables are backwards compatible. Values `generic` (default; compiler's default behavior); `alderlake`, `broadwell`, `cannonlake`, `cascadelake`, `cooperlake`, `goldmont`, `goldmont-plus`, `haswell`, `icelake-server`, `ivybridge`, `rocketlake`, `sandybridge`, `sapphirerapids`, `silvermont`, `skylake`, `skylake-avx512`, `tigerlake`, `tremont` (optimize for the named Intel® processor/microarchitecture); `core-avx2` (Intel® AVX2, AVX, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3); `core-avx-i` (Float-16 conversion + RDRND, Intel® AVX, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3); `corei7-avx` (Intel® AVX, SSE4.2, SSE4.1, SSE3, SSE2, SSE, SSSE3); `corei7` (Intel® SSE4 Efficient Accelerated String and Text Processing; may also generate Intel® SSE4 Vectorizing Compiler and Media Accelerator, SSE3, SSE2, SSE, SSSE3); `atom` (MOVBE; may also generate SSSE3, Intel® SSE3, SSE2, SSE); `core2` (Intel® Core™2 family incl. MMX™, Intel® SSE, SSE2, SSE3, SSSE3). Code built with `-mtune=core2`/`/tune:core2` runs correctly on 4th Generation Intel® Core™ processors but might not run as fast as with `-mtune=haswell`/`/tune:haswell`; code built with `-mtune=haswell` or `-mtune=core-avx2` also runs correctly on Intel® Core™2 but might not run as fast as with `-mtune=core2`. This contrasts with `-march=core-avx2`, which will not run correctly on older processors such as Intel® Core™2. See also `march`.

### `x, Qx`
`-xcode` / `/Qxcode`. Which processor features, instruction sets, and optimizations the compiler may target. Values: shared microarchitecture list (each also optimizes for the named processor/microarchitecture) plus `COMMON-AVX512`, `CORE-AVX512`, `CORE-AVX2`, `CORE-AVX-I`, `AVX`, `SSE4.2`, `SSE4.1`, `ATOM_SSE4.2`, `ATOM_SSSE3`, `SSSE3`, `SSE3`; `ATOM_SSE4.2`/`ATOM_SSSE3` may generate MOVBE depending on `-minstruction` (Linux) or `/Qinstruction` (Windows) and optimize for the corresponding Intel Atom® processors. Default `OFF`: if `-x` or `-march` is not specified (Linux), or `/Qx` or `/arch` is not specified (Windows), the default target architecture supports Intel® SSE2 instructions.

Caveats: executables built with these values run only on Intel® processors supporting the indicated instruction set; do not create binaries for incompatible processors (illegal instruction exception or other unexpected behavior). Compiling `main()` with any code value produces binaries that display a fatal runtime error on unsupported processors, including all non-Intel processors. `-march` (Linux) and `/arch` (Windows) produce binaries that can run on processors not made by Intel that implement the same capabilities; `-x`/`/Qx` enable additional optimizations not enabled by `-march`/`/arch`. Linux: `-x` and `-march` are mutually exclusive — if both are given the compiler uses the last one specified and generates a warning. Windows: `/Qx` and `/arch` are mutually exclusive with the same last-one-wins behavior. All settings perform a CPU check except with `-O0` (Linux) or `/Od` (Windows), where no CPU check is performed.

### Other code generation options (alphabetical)

- **`fasynchronous-unwind-tables`** `-fasynchronous-unwind-tables` / `-fno-asynchronous-unwind-tables` (Linux only). Default `-fasynchronous-unwind-tables`: unwind info precise at an instruction boundary, enabling accurate unwinding at any instruction; table format DWARF2 or DWARF3 depending on system support. `-fno-…` makes it precise at call boundaries only, and the compiler avoids unwind tables for a C++ routine with no objects having destructors and no calls that might throw; a C/C++ or Fortran routine compiled without `-fexceptions` and without traceback information; and a C/C++ or Fortran routine compiled with `-fexceptions` containing no calls that might throw.
- **`fdata-sections, Gw`** `-fdata-sections` / `/Gw`. One COMDAT section per data item. Default `OFF` (compiler does not separate functions into COMDATs). Add `-Wl,--gc-sections` (Linux) or `/link /OPT:REF` (Windows) to remove unused code; per-section placement also lets the linker reorder sections. See also `ffunction-sections, Gy`.
- **`fexceptions`** `-fexceptions`, `-fno-exceptions` (Linux only). EH table generation. Default `-fexceptions` for C++, `-fno-exceptions` for C. `-fno-exceptions` yields smaller code; exception handling constructs (`try`, `throw`) produce an error, exception specifications are parsed but ignored, and `__EXCEPTIONS` is undefined.
- **`ffunction-sections, Gy`** `-ffunction-sections` / `/Gy`. One COMDAT section per function. Default `OFF`. Same GC options/reordering benefit as `fdata-sections`/`Gw`.
- **`Gd`** `/Gd` (Windows only). Default `ON`: `__cdecl` is the default calling convention.
- **`GR`** `/GR`, `/GR-` (Windows only). Default `/GR`: C++ Runtime Type Information (RTTI) enabled; `/GR-` disables.
- **`Gv`** `/Gv` (Windows only). Default `OFF` (`__cdecl`). Uses `__vectorcall` when passing vector type arguments; every function compiles as `__vectorcall` unless declared with a conflicting attribute or named `main`. Microsoft compatibility.
- **`m, Qm`** `-mcode` / `/Qmcode`. Instruction set extensions based on CPUID bits; many Clang `-m` settings supported. Default `varies`: SSE2 if `arch` is unspecified. Code should execute on any compatible non-Intel processor with the instruction set. NOTE: `-m`/`/Qm` enable specific CPUID-based sets; use `-march` (Linux) or `/arch` (Windows) for all instructions of a named microarchitecture.
- **`m64, Qm64`** `-m64` / `/Qm64`. Default: compiler generates Intel® 64 code whether or not the option is given. Legacy, deprecated (will be removed); they do nothing.
- **`m80387`** `-m80387`, `-mno-80387` (Linux only). Default `-m80387`: x87 instructions may be used. `-mno-80387` prevents use and issues an error if the compiler is forced to generate x87. Alternate `-m[no-]x87`.
- **`masm`** `-masm=dialect` (Linux only). Assembler output dialect: `att` (AT&T* syntax, default) or `intel`.
- **`mauto-arch, Qauto-arch`** `-mauto-arch=value` / `/Qauto-arch:value`. Multiple feature-specific auto-dispatch paths for x86 processors when beneficial, plus a baseline path; `value` is any `[Q]ax` setting. Default `OFF` (no additional execution path). Cannot be used with options that may require Intel-specific optimizations (such as `[Q]x` or `[Q]ax`).
- **`mbranches-within-32B-boundaries, Qbranches-within-32B-boundaries`** `-mbranches-within-32B-boundaries`/`-mno-branches-within-32B-boundaries` ; `/Qbranches-within-32B-boundaries`/`/Qbranches-within-32B-boundaries-`. Aligns branches and fused branches on 32-byte boundaries for better performance. Default `-mno-…`/`/Q…-` (not aligned). NOTE: may affect binary utilities usage experience, such as debugability.
- **`regcall, Qregcall`** `-regcall` / `/Qregcall`. `__regcall` becomes the calling convention for functions that do not directly specify one, ensuring as many values as possible are passed/returned in registers; it is the default convention in the compilation unless a declaration specifies another. Default `OFF` (`__regcall` only if a function explicitly specifies it). Ignored for functions with variable arguments; all `__regcall` functions must have prototypes.
- **`xHost, QxHost`** `-xHost` / `/QxHost`. Generates instructions for the highest instruction set available on the compilation host processor; output differs by host. Default `OFF`: if `-x`/`-march` (Linux) or `/Qx`/`/arch` (Windows) is unspecified, the default target supports Intel® SSE2.

## Offload Compilation, OpenMP*, and Parallel Processing Options

Source: "options that pertain to offload compilation, OpenMP*, or parallel processing… listed in alphabetical order."

### `fiopenmp, Qiopenmp`
`-fiopenmp` / `/Qiopenmp`. Enables recognition of OpenMP* features (such as `parallel`, `simd`, and offloading directives) and tells the parallelizer to generate multi-threaded code; alternate name for `-qopenmp` (`/Qopenmp`). Default `OFF`. Enables Intel's implementation of OpenMP* in the compiler back end: the front end produces an IR preserving the parallelism exposed by OpenMP* directives, and the back end uses it for advanced optimizations such as SIMD vectorization. Alternate Linux `-qopenmp`; Windows `/Qopenmp`.

> NOTE: `-fiopenmp` is not the same as `-fopenmp`.
> NOTE: To enable offloading to a specified GPU target you must also specify `-fopenmp-targets` (Linux*) or `/Qopenmp-targets` (Windows).

```bash
icx -fiopenmp foo.c          # OpenMP parallelization (no offloading) of parallel/loop/simd
icpx -qopenmp-simd foo.c     # SIMD vectorization only: no OpenMP parallelization/offloading (-fiopenmp absent)
icpx -fiopenmp -fopenmp-targets=spir64 bar1.cpp   # parallelization + SIMD vectorization + spir64 offload
```

### `fopenmp`
Linux only `-fopenmp`. Enables recognition of OpenMP* features and tells the parallelizer to generate multi-threaded code; it lowers OpenMP constructs in the compiler front end (as implemented by the LLVM community), so it is expected to be less performant than `-fiopenmp` (Intel implementation, backend lowering), and it does not support offloading to GPUs. Default `OFF`. Meant for advanced users who prefer the LLVM-community implementation. Host-only.

> NOTE: `-fopenmp` is not the same as `-fiopenmp`. For full advantage of SIMD vectorization or offloading, use `-qopenmp` or `-fiopenmp`.

### `fopenmp-declare-target-scalar-defaultmap, Qopenmp-declare-target-scalar-defaultmap`
`-fopenmp-declare-target-scalar-defaultmap=keyword` / `/Qopenmp-declare-target-scalar-defaultmap:keyword`. Implicit data-mapping/sharing rules for a scalar referenced in a target pragma when it appears in a `declare target` pragma with a `to` or `link` clause but not clause `device_type (nohost)`. Values `default`: apply implicit rules per the OpenMP* specification — treat the scalar as if it appeared in a `map` clause with map-type `tofrom` when it appears in a `to`/`link` clause of a `declare target` pragma without `device_type (nohost)` and the target construct has no explicit mapping rules for it. `firstprivate`: in the same situation the scalar is not mapped but has an implicit data-sharing attribute of `firstprivate`. Default `...=default` / `...:default`. The option assumes such a scalar has the same value before the target construct (host) and at the beginning of the target region (device), which may enable host-side optimizations; it affects only scalars not appearing in target clauses `map`, `is_device_ptr`, or `has_device_addr`. See OpenMP 5.2, e.g. section 5.8.1. It may behave incorrectly for programs that allow different values of the same `declare target` scalars on entry to target regions.

```c
#pragma omp declare target
int N;
#pragma omp end declare target
...
void program() {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < N; ++i) ...
}
```

Specifying `...=firstprivate` (or explicit `firstprivate(N)`) lets the compiler generate efficient host code issuing the most appropriate number of teams and threads, assuming `N` does not change between the beginning of the target region and the beginning of the `distribute parallel for` region. Without it, `N` must be transferred device-to-host to compute the right team/thread count, which may cost performance.

```c
#include <stdio.h>

#pragma omp declare target
int x = 0; /* host 'x' is 0, target 'x' is 0 */
#pragma omp end declare target

int main() {
  x = -1;                                  /* host 'x' is -1, target 'x' is 0 */
#pragma omp target
  x = 1;                                   /* host 'x' is -1, target 'x' is 1 */
#pragma omp target
  printf("target: %d == 1\n", x);
#pragma omp target update from(x)
    /* host 'x' is 1, target 'x' is 1 */
    printf("host: %d == 1\n", x);
    return 0;
}
```

Correct output: `target: 1 == 1` / `host: 1 == 1`. Output with `...=firstprivate`: `target: -1 == 1` / `host: 0 == 1`.

### `fopenmp-device-code-split, Qopenmp-device-code-split`
`-fopenmp-device-code-split=[triple=]per_kernel` / `/Qopenmp-device-code-split:[triple=]per_kernel`. Parallel compilation of SPIR-V* kernels for OpenMP offload Ahead-Of-Time compilation. `triple` is a device name such as `spir64`, `spir64_gen`, etc.; if specified, splitting applies only to that target. `per_kernel` creates a separate device code module per SYCL* kernel, each containing the kernel and all dependencies (called functions, used variables). Default `OFF`. Use `-fopenmp-max-parallel-link-jobs` (Linux) or `/Qopenmp-max-parallel-link-jobs` (Windows) to cap parallel actions. Device-only under OpenMP offload.

```bash
icpx -fiopenmp -fopenmp-targets=spir64_x86_64 -fopenmp-device-code-split=per_kernel -fopenmp-max-parallel-link-jobs=4 file.cpp
icx /Qiopenmp /Qopenmp-targets:spir64_x86_64 /Qopenmp-device-code-split:per_kernel /Qopenmp-max-parallel-link-jobs:4 file.cpp
```

### `fopenmp-device-link, Qopenmp-device-link`
`-fopenmp-device-link` / `/Qopenmp-device-link`. Performs a device link during the compilation step instead of a link step; with `-c` (Linux) or `/c` (Windows) it produces file device binaries within the generated fat object. Default `OFF` (compiler follows default heuristics). Valid only for SPIR64-based devices; useful for static libraries such as the Intel® oneAPI Math Kernel Library (oneMKL) because it can reduce application compilation time.

> NOTE: It can affect options enabled during the device linking phase because it shifts when compilation steps occur: when generating the fat object, anything that used to impact device linking during the link phase (for example `-Xopenmp-target-backend`) is applied during the compilation phase. Device-only under OpenMP* offload.

### `fopenmp-target-buffers, Qopenmp-target-buffers`
`-fopenmp-target-buffers=keyword` / `/Qopenmp-target-buffers:keyword`. Overcomes incorrect code from some OpenMP* offload SPIR-V* devices when a target object is larger than 4GB. Values `default`: default heuristics, which may produce incorrect code in that case; `4GB`: generate code preventing the issue, required by programs that access target objects larger than 4GB in target code. `4GB` applies to target objects declared in OpenMP* target regions or inside `declare target` functions; target objects in the OpenMP* device data environment; and objects mapped/allocated via OpenMP* APIs (such as `omp_target_alloc`). Default `default`. Threshold is 4GB (4294959104 bytes). `4GB` may decrease performance on Intel® GPUs. Requires `-fopenmp-targets` (Linux*) or `/Qopenmp-targets` (Windows*). May have no effect for some SPIR-V* devices and for offload targets different from SPIR* [sic]. Device-only.

### `fopenmp-target-teams-default-vla-alloc-mode, Qopenmp-target-teams-default-vla-alloc-mode`
`-fopenmp-target-teams-default-vla-alloc-mode=arg` / `/Qopenmp-target-teams-default-vla-alloc-mode:arg`. How local copies are allocated for variable-length/assumed-sized arrays on privatization clauses (`private`, `firstprivate`, etc.) on OpenMP* teams and distribute constructs. Values `malloc`: use `malloc`/`free`, making copies shared across the threads of a team; the buffer size may need adjustment using `LIBOMPTARGET_DYNAMIC_MEMORY_SIZE=<num-mbytes>`. `wilocal` (**default**): stack allocation, copies private to each thread of each team (for example, local to each work-item); more memory but avoids synchronization overhead. When the compiler's analyses determine a VLA need not be shared across threads of a team, it always uses `wilocal` for its private copies. Applies to spir64 devices.

```c
cat tgt_teams_par_priv_vla.c
#include <stdio.h>

void f1(int n) {
  int x[n];
#pragma omp target teams num_teams(1) private(x) thread_limit(4)
  {
#pragma omp parallel shared(x)
#pragma omp critical
    printf("%p\n", &x[0]);
  }
}

int main() { f1(4); }
```

```text
# Each thread in the team has its own x, hence the addresses are different.
icpx -O0 -fiopenmp -fopenmp-targets=spir64 tgt_teams_par_priv_vla.c && ./a.out
0x3f00000000860390
0x3f000000008603a0
0x3f000000008603b0
0x3f000000008603c0

# Every thread in the team shares the same x.
icpx -O0 -fiopenmp -fopenmp-targets=spir64 tgt_teams_par_priv_vla.c -fopenmp-target-teams-default-vla-alloc-mode=malloc && ./a.out
0xff00000026600000
0xff00000026600000
0xff00000026600000
0xff00000026600000
```

The Windows documentation shows identical source, `icx` commands, and output for this example.

### `fopenmp-targets, Qopenmp-targets`
`-fopenmp-targets=triple` / `/Qopenmp-targets:triple`. Enables offloading to a specified GPU target if OpenMP* features are enabled. Values `spir64` (SPIR64-based devices), `spir64_x86_64` (Intel® CPUs), `spir64_gen` (Intel® Processor Graphics). For example, `spir64` makes the compiler generate an x86 + SPIR64 (64-bit Standard Portable Intermediate Representation) fat binary for Intel® GPU devices. Default `OFF` (no fat binaries). Requires enabling OpenMP* features with one of `-qopenmp`, `-fiopenmp`, `-fopenmp` (Linux) or `/Qopenmp`, `/Qiopenmp` (Windows). Example: `icx -fiopenmp -fopenmp-targets=spir64 matmul_offload.cpp -o matmul`. When specified, C++ exception handling is disabled for target compilations; on Linux, host compilations must add `-fno-exceptions` to disable it there too. Device-only under OpenMP* offload.

### `fsycl`
`-fsycl` (Linux and Windows). Compiles a program as a SYCL program rather than plain C++11. Default `SYCL: ON` (compiled as SYCL); `C++: OFF` (compiled as C++11).

> NOTE: On Windows, `-fsycl` sets `/MD`, telling the linker to search for unresolved references in a multithreaded, dynamic-link runtime library; `/MT` cannot be specified. To prevent potential sycl.lib conflicts, add any desired SYCL library in the link command.

### `fsycl-add-default-spec-consts-image`
`-fsycl-add-default-spec-consts-image`, `-fno-sycl-add-default-spec-consts-image` (Linux and Windows). Enables/disables generation of a copy of every device image that uses a specialization constant, replacing all instances of that constant with default values defined in the relevant `specialization_id` variable. Default `OFF` (no copies, constants unchanged). If a device image does not use a specialization constant, no copy is generated; SYCL runtime then chooses between a new generated image and the original depending on whether the constant value changed from the default. Only useful with Ahead of Time (AOT) Compilation. Requires `-fsycl`. Host-only.

```bash
icpx -fsycl -fsycl-add-default-spec-consts-image ./code.cpp
# warning: -fsycl-add-default-spec-consts-image flag has an effect only in Ahead of Time Compilation mode (AOT).
icpx -fsycl -fsycl-add-default-spec-consts-image -fsycl-targets=spir64_gen -Xs "-device skl" ./code.cpp
icpx -fsycl -fno-sycl-add-default-spec-consts-image -fsycl-targets=spir64_gen -Xs "-device skl" ./code.cpp
```

### `fsycl-device-code-split`
`-fsycl-device-code-split[=value]` (Linux and Windows). SYCL* device code module assembly. Values `per_kernel`: separate module per SYCL* kernel with all dependencies (called functions, used variables). `per_source`: separate module per source (translation unit) grouping kernels per-source with all dependencies, including `SYCL_EXTERNAL` macro-marked functions from other translation units. `off`: a single module for all kernels; if `-fsycl-no-rdc` is specified, same as `per_source`. `auto`: heuristic selection; same as specifying the option with no value. Default `auto`, whether unspecified or specified without a value.

> Caution: If `-fno-sycl-rdc` is also specified, `off` is equivalent to `per_source`.

Requires `-fsycl`. Device-only under SYCL offload.

### `fsycl-force-target`
`-fsycl-force-target=triple` (Linux and Windows). Forces the specified target triple when extracting device code from objects on the command line. Values `spir64` (SPIR64-based device), `spir64_x86_64` (Intel® CPU), `spir64_gen` (Intel® Processor Graphics). Default `OFF`: unbundle/extract based on `-fsycl-targets`. You can have both `spir64` and `spir64_gen` in your objects; this option overrides `-fsycl-targets` even if the latter exists. Requires `-fsycl`. Device-only. Example: `icx -fsycl -fsycl-targets=spir64_gen -fsycl-force-target=spir64` — spir64 objects/archives are extracted but spir64_gen targets still compile.

### `fsycl-fp64-conv-emu`
`-fsycl-fp64-conv-emu` (Linux and Windows). fp64 partial emulation for kernels with only fp64 conversion operations and no fp64 computation operations; requires an Intel GPU supporting fp64 partial emulation. Default `OFF` (the compiler will not try fp64 partial emulation for any fp64 conversion operations). Requires `-fsycl`. Device-only. Example (Linux*): `icpx -g -fsycl -fsycl-fp64-conv-emu test.cpp`.

### `fsycl-host-compiler-options`
`-fsycl-host-compiler-options="opts"` (Linux and Windows). Passes options to the compiler specified by `fsycl-host-compiler`; `opts` is a string of compatible options that must appear within quotes, with a space between names if more than one. Default `OFF`.

> NOTE: If `-fsycl-host-compiler=cl` is specified, the host compilation will be performed by the Microsoft* `__cplusplus` preprocessor macro, which depends on `/Qstd` (or MSVC-compatible `/std`). In this case the default is `/Zc:__cplusplus`; override with `/fsycl-host-compiler-options=/Zc:cplusplus-`.
> NOTE: Specifying any kind of phase limiting options (such as `-c`, `-E`, or `-S`) may interfere with the expected output set during host compilation, causing undefined behavior. Requires `-fsycl`.

### `fsycl-id-queries-fit-in-int`
`-fsycl-id-queries-fit-in-int`, `-fno-sycl-id-queries-fit-in-int` (Linux and Windows). Assumes SYCL ID queries fit within `MAX_INT`: `id` class `get()` and `operator[]`; `item` class `get_id()` and `operator[]`; `nd_item` class `get_global_id()`/`get_global_linear_id()` (see the Khronos* Group SYCL* 1.2.1 Specification). Default `ON`. For a larger number of work items use the OFF setting, `-fno-sycl-id-queries-fit-in-int`.

> Caution: Carefully evaluate the OFF setting with a larger number of work items; truncating to data type int is often incorrect in such circumstances. If OFF is used when values fit within `MAX_INT`, it can lead to unexpected performance issues.

Requires `-fsycl`. Device-only.

### `fsycl-max-parallel-link-jobs`
`-fsycl-max-parallel-link-jobs=n` (Linux and Windows). Lets the compiler simultaneously spawn up to `n` processes for actions required to link SYCL applications. Default `-fsycl-max-parallel-link-jobs=1`. Experimental. Limitations: no effect if compiler options such as `c` or `E` are specified; no effect with `-fsycl-device-code-split=off`; spawned processes never exceed the number of device code modules (with `m` kernels, `per_kernel` split, and `n > m`, at most `m`); it is not guaranteed `n` processes are always active (a process is not instantly re-assigned to the next module after finishing); and spawning device link processes cannot be safely combined with build system-level parallelization — a large number risks increased RAM usage, oversubscription, and performance or compilation issues. Requires `-fsycl`. Device-only.

```bash
icx -fsycl -fsycl-max-parallel-link-jobs=4 a.cpp b.cpp c.cpp d.cpp -o a.out
icx -fsycl -fsycl-max-parallel-link-jobs=8 a.o b.o c.o d.so e.a -o b.out
```

### `fsycl-optimize-non-user-code`
`-fsycl-optimize-non-user-code` (Linux and Windows). Optimizes SYCL framework utility functions and leaves kernel code unoptimized for further debugging. Default `OFF`. Requires `-O0` (Linux) or `/Od` (Windows); any other optimization setting, or none, is a compilation error. Requires `-fsycl`. Host-only.

```bash
icpx -fsycl -O0 -fsycl-optimize-non-user-code ./code.cpp   # succeeds
icpx -fsycl -O2 -fsycl-optimize-non-user-code ./code.cpp   # error: only -O0 or /Od can be specified
icpx -fsycl -fsycl-optimize-non-user-code ./code.cpp       # error: -O0 or /Od must be specified
```

### `fsycl-pstl-offload`
`-fsycl-pstl-offload[=arg]`, `-fno-sycl-pstl-offload` / `/fsycl-pstl-offload[:arg]`, `/fno-sycl-pstl-offload`. Automatic offloading of C++ standard parallel algorithms called with `std::execution::par_unseq` policy to a SYCL device, implemented via the oneAPI Data Parallel C++ Library (oneDPL). Values `cpu`, `gpu`; default `-fno-sycl-pstl-offload`; if `arg` is omitted, the default SYCL device is used. oneDPL is required; see its documentation. Requires `-fsycl`.

Restrictions/requirements/limitations: callable objects share SYCL kernel limitations — exceptions not allowed, dynamic memory allocation not allowed, there can be unsupported API from `std` (see SYCL 2020 for the full list). Data placement: only heap memory allocated with C++ standard dedicated facilities can be passed; `std::vector` can be used because it dynamically allocates underneath; host stack memory cannot be used, nor `std::array` or C-style array on the stack — make a deep copy by capture by value or allocate on the heap. Other: only a subset of standard C++ APIs can be used in callable objects (see the oneDPL documentation on Tested Standard C++ APIs); the same argument must be applied to all Translation Units (TU) in an executable or dynamic library. Performance: `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR` may improve memory allocation and `SYCL_CACHE_PERSISTENT` may improve launch time.

> NOTE: Also allows offloading to NVIDIA* and AMD* GPUs besides an Intel GPU. AMD GPU use is Linux only (see the oneAPI for AMD GPUs Guide from Codeplay); NVIDIA GPU use is Linux and Windows (see the oneAPI for NVIDIA® GPUs Guide from Codeplay).

```cpp
#include <algorithm>
#include <vector>
#include <execution>

int main()
{
    std::vector<int> v(1000000);

      // If this code is compiled with -fsycl-pstl-offload=gpu, the
      // for_each algorithm is going to be offloaded to the default
      // SYCL GPU device automatically
      std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](auto& v)
      {
          // do some computation
      });
}
```

### `fsycl-rdc`
`-fsycl-rdc`, `-fno-sycl-rdc` (Linux and Windows). Whether relocatable device code (RDC) is generated during SYCL* offload target compilation. Default `-fsycl-rdc` (generated). `-fno-sycl-rdc` disables RDC, allowing device code to be linked per translation unit; device code then cannot use `SYCL_EXTERNAL` functions, and the option may improve compile time and compiler memory usage. With `-fno-sycl-rdc` plus `-c` (Linux) or `/c` (Windows), the compiler produces final device binaries within the generated fat object. With `-fno-sycl-rdc` plus `-fsycl-max-parallel-link-jobs`, additional device linking parallelism for fat static archives is enabled. With `-fsycl-device-code-split`, `off` becomes equivalent to `per_source`. Requires `-fsycl`. Device-only.

### `fsycl-targets`
`-fsycl-targets=T1,...,Tn` (Linux and Windows; multiple `T` comma-separated). Values `spir64`: default heuristics for SPIR64-based devices (the default; also `spir64-unknown-unknown`). `spir64_x86_64`: Intel® CPUs (also `spir64_x86_64-unknown-unknown`). `x86_64`: code ahead of time for x86_64 CPUs, providing better debuggability (also `x86_64-unknown-unknown`). `spir64_gen`: Intel® Processor Graphics (also `spir64_gen-unknown-unknown`). Default `spir64`. Normally specified when linking, embedding Ahead of Time (AOT) compiled device binaries in the application's fat executable; also usable with `-c` (Linux) or `/c` (Windows) and `-fno-sycl-rdc` when compiling a source file, embedding them in the fat object file.

> NOTE: Long syntax values containing `-sycldevice`, such as `spir64-unknown-unknown-sycldevice`, are still supported but deprecated. Requires `-fsycl`. Device-only.

### `ftarget-compile-fast`
`-ftarget-compile-fast` / `/ftarget-compile-fast`. Less aggressive optimizations to reduce compilation time at the expense of less optimal target code. Default `OFF`. Experimental. Useful during development for fast turnaround, or with `O2`/`O3` for a Just-in-Time (JIT) product where compile-time and execution performance both matter; not recommended with `O2`/`O3` for an Ahead-of-Time (AOT) product where long one-time compilation may be tolerable for best performance, nor if you plan to ship object files as part of a final product. Host-only.

```bash
icx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device skl" test.cpp -ftarget-compile-fast foo.cpp -o:a.out
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device skl" -ftarget-compile-fast foo.cpp -o a.out
icx /Qiopenmp /Qopenmp-targets:spir64_gen -Xopenmp-target-backend=spir64_gen "-device skl" /ftarget-compile-fast foo.cpp /Fo:a.out
```

### `ftarget-register-alloc-mode, Qtarget-register-alloc-mode`
`-ftarget-register-alloc-mode=device-name:reg-mode[, device-name:reg-mode][,...]` / `/Qtarget-register-alloc-mode:device-name:reg-mode[, device-name:reg-mode][,...]`. Register allocation mode for specific hardware, for supported target backends. `device-name`: currently only `pvc` (Ponte Vecchio); more may be added. `reg-mode`: `default` (impose no specification), `small` (small mode; for PVC the 128 register file), `large` (large mode; for PVC the 256 register file), `auto` (internal heuristics based on kernel analysis). Default on Ponte Vecchio: Linux* `-ftarget-register-alloc-mode=pvc:auto`; Windows* `/Qtarget-register-alloc-mode=pvc:default` [sic: source shows `=`]. No effect when targeting hardware other than Ponte Vecchio.

> Caution: For a SYCL* or OpenMP*-offload program compiled for, or run on, Ponte Vecchio, do not set the register allocation mode with Intel® Graphics Compiler (IGC) options such as `-ze-opt-large-register-file` in the `-Xs` high-level option; use `-ftarget-register-alloc-mode` (Linux) or `/Qtarget-register-alloc-mode` (Windows). For other hardware, use the IGC option. Device-only.

```bash
icpx -fiopenmp -fopenmp-targets=spir64 -ftarget-register-alloc-mode=pvc:large a.cpp
icpx -fsycl -ftarget-register-alloc-mode=pvc:large -fsycl-targets=spir64_gen -Xs "-device pvc"
icx /Qiopenmp /Qopenmp-targets:spir64 /Qtarget-register-alloc-mode:pvc:large a.cpp
icx -fsycl /Qtarget-register-alloc-mode:pvc:large -fsycl-targets=spir64_gen -Xs "-device pvc"
```

### `qopenmp, Qopenmp`
`-qopenmp`, `-qno-openmp` / `/Qopenmp`, `/Qopenmp-`. Enables recognition of OpenMP* features (such as `parallel`, `simd`, and offloading directives) and tells the parallelizer to generate multi-threaded code based on OpenMP* directives; the code can run in parallel on uniprocessor and multiprocessor systems. Alternate name for `-fiopenmp` (`/Qiopenmp`). Default `-qno-openmp` or `/Qopenmp-` (no OpenMP* multi-threaded code). Works with any optimization level; no optimization (`-O0` on Linux* or `/Od` on Windows*) helps to debug OpenMP applications. Alternate Linux `-fiopenmp`; Windows `/Qiopenmp`.

> NOTE: To enable offloading to a specified GPU target you must also specify `-fopenmp-targets` (Linux*) or `/Qopenmp-targets` (Windows).
> NOTE: Options that use OpenMP* API are available for both Intel® and non-Intel microprocessors, but may perform additional optimizations on Intel® microprocessors. Major user-visible constructs/features that may perform differently: locks (internal and user visible), the `SINGLE` construct, barriers (explicit and implicit), parallel loop scheduling, reductions, memory allocation, thread affinity, and binding.

```bash
icx -qopenmp foo.c          # OpenMP parallelization (no offloading) of parallel/loop/simd
icpx -qopenmp-simd foo.c    # SIMD vectorization only: no OpenMP parallelization/offloading (-qopenmp absent)
icpx -qopenmp -fopenmp-targets=spir64 bar1.cpp   # parallelization + SIMD vectorization + spir64 offload
```

### `qopenmp-link`
Linux only `-qopenmp-link=library`. Controls whether the compiler links static or dynamic OpenMP* runtime libraries. Values `static` (static OpenMP runtime libraries; note static OpenMP libraries are deprecated), `dynamic` (dynamic OpenMP runtime libraries). Default `-qopenmp-link=dynamic`; however, if Linux* option `-static` is specified the compiler links to static OpenMP runtime libraries. To link the static OpenMP runtime library (RTL) and create a purely static executable you must specify `-qopenmp-link=static`, but the default `dynamic` is strongly recommended. `-qopenmp-link=dynamic` cannot be used with `-static`; specifying both displays an error.

> NOTE: `-static-intel` and `-shared-intel` (Linux*) have no effect on which OpenMP runtime library is linked.
> NOTE: On Linux, the OpenMP runtime library depends on libpthread and libc (libgcc when compiled with gcc); both must be static or both dynamic. If both are static, use the static OpenMP runtime; if both are dynamic, either version may be used. Host-only.

### `qopenmp-simd, Qopenmp-simd`
`-qopenmp-simd`, `-qno-openmp-simd` / `/Qopenmp-simd`, `/Qopenmp-simd-`. Enables/disables OpenMP* SIMD compilation with no impact on other OpenMP features; in that case no OpenMP runtime library is needed to link and no OpenMP runtime initialization code is generated. Default `-qno-openmp-simd` or `/Qopenmp-simd-` (disabled). `[q or Q]openmp` implies `[q or Q]openmp-simd`; specifying this option together with `[q or Q]openmp` can impact other OpenMP features. `-qopenmp-simd` is equivalent to `-fiopenmp-simd`; `/Qopenmp-simd` is equivalent to `/Qiopenmp-simd`. Alternate Linux `-fiopenmp-simd`; Windows `/Qiopenmp-simd`.

> NOTE: Advanced users preferring the LLVM-community OpenMP* can get most of that functionality with `-fopenmp` and `-fopenmp-simd`.

```bash
# Equivalent to only [q or Q]openmp-simd: SIMD support only, no OpenMP library linked,
# only the omp pragmas related to SIMD are processed.
-qno-openmp -qopenmp-simd   # Linux          /Qopenmp- /Qopenmp-simd   # Windows
# SIMD support, OpenMP library linked, and OpenMP runtime initialization code generated.
-qopenmp -qopenmp-simd      # Linux          /Qopenmp /Qopenmp-simd     # Windows
```

### `Xopenmp-target`
`-Xopenmp-target-tool=T "options"` (Linux and Windows). Passes options to the specified tool in the device compilation tool chain for the OpenMP* target. Values `frontend`: the frontend + middle end of the SPIR-V*-based device compiler for target triple `T` (the middle end generates SPIR-V*, which the driver passes to the backend of `T`). `backend`: AOT compilation for target triple `T` and Just in Time (JIT) compilation for `T` at runtime. `linker`: the device code linker for target triple `T`. Some targets may combine frontend and backend; options are then merged. Default `OFF`. Device-only under OpenMP* offload.

### `Xs`
`-Xs -option` or `-Xsoption` (Linux and Windows). Passes options to the backend tool in device compilation; an alternative for `Xsycl-target-backend` or `Xopenmp-target-backend`. To see usable `option` values, specify `-fsycl-help` to display offline-tools help. Default `OFF`.

```bash
-Xsversion      # syntax form -Xsoption
-Xs -version    # syntax form -Xs -option
# both are equivalent to: -Xsycl-target-backend -version   or   -Xopenmp-target-backend -version
```

> NOTE: Under Ahead of Time (AOT) compilation the options passed with `-Xs` are not compiler options. To list options passable with `-Xs` under AOT, specify `-fsycl-help=gen` or `-fsycl-help=x86_64`. Device-only when offloading is enabled.

### `Xsycl-target`
`-Xsycl-target-tool=T "options"` (Linux and Windows). Passes options to the specified tool in the device compilation tool chain for the SYCL* target. Values `frontend` (frontend + middle end of the SPIR-V*-based device compiler for target triple `T`; the middle end generates SPIR-V*, passed by the driver to the backend of `T`), `backend` (AOT compilation for target triple `T` and JIT compilation for `T` at runtime), `linker` (device code linker for target triple `T`). Some targets may combine frontend and backend; options are then merged. Default `OFF`. Device-only.

### Other offload / OpenMP* / parallel options (alphabetical)

- **`device-math-lib`** `-device-math-lib=library`, `-no-device-math-lib=library` / `/device-math-lib:library`, `/no-device-math-lib:library`. Device libraries `fp32` (fp32 device math library), `fp64`; comma-separate to link both (e.g., `fp32, fp64`). Default `fp32, fp64` (both linked). Deprecated; may be removed in a future release; no replacement. See also `fopenmp-device-lib`; `fsycl-device-lib`.
- **`flink-huge-device-code`** `-flink-huge-device-code`, `-fno-link-huge-device-code` (Linux only). Places device code later in the linked binary to prevent 32-bit PC-relative relocations between surrounding Executable and Linkable Format (ELF) sections when device code is larger than 2GB. Default `fno-link-huge-device-code` (no change). Impacts the host link for a full offload compilation; only useful when offloading is performed. Requires `-fsycl` or `-fopenmp-targets`, and a real link action (no effect with `-c` or `-E`). Example: `icx -fsycl -flink-huge-device-code a.cpp b.cpp -o a.out`; `icpx -fiopenmp -fopenmp-targets=spir64 -flink-huge-device-code c.o b.o -o b.out`.
- **`fno-sycl-libspirv`** `-fno-sycl-libspirv` (both OS). Disables the check for libspirv (the SPIR-V* tools library). Default `OFF` (check enabled). Device-only.
- **`foffload-fp32-prec-div`** `-foffload-fp32-prec-div`, `-fno-offload-fp32-prec-div` (both OS). Correctly rounded divide as defined by IEEE 754; otherwise the default precision requirement for divide in SYCL is 2.5 units-in-the-last-place (ULP). Default `-foffload-fp32-prec-div`. Example: `icpx -fsycl -foffload-fp32-prec-div test.cpp`.
- **`foffload-fp32-prec-sqrt`** `-foffload-fp32-prec-sqrt`, `-fno-offload-fp32-prec-sqrt` (both OS). Correctly rounded `sycl::sqrt` per IEEE 754; otherwise the default requirement is 3 ULP. Default `foffload-fp32-prec-sqrt`. Example: `icpx -fsycl -foffload-fp32-prec-sqrt test.cpp`.
- **`fopenmp-concurrent-host-device-compile, Q…`** `-fopenmp-concurrent-host-device-compile` / `/Qopenmp-concurrent-host-device-compile`. Parallel compilation of host and target steps during OpenMP offload compilations (only the steps creating the host and target binaries); may improve compilation times. Default `OFF`. Experimental.
- **`fopenmp-device-lib`** `-fopenmp-device-lib=library[,library,...]`, `-fno-openmp-device-lib=library[,library,...]` (both OS). Values `libm-fp32`, `libm-fp64`, `libc`, `all` (links `libm-fp32`, `libm-fp-64` [sic: source spelling], and `libc`). Comma-separate with no spaces (e.g., `libm-fp32,libc`); `all` supersedes any additional value. Default `OFF` (disables linking to device libraries for this target). `-fno-…` disables linking to the named library. Device-only under OpenMP* offload.
- **`fopenmp-max-parallel-link-jobs, Q…`** `-fopenmp-max-parallel-link-jobs=num` / `/Qopenmp-max-parallel-link-jobs:num`. Maximum number of parallel actions during device linking steps. Default `OFF` (parallelization disabled). Useful with `-fopenmp-device-code-split`/`/Qopenmp-device-code-split`. Device-only.
- **`fopenmp-offload-mandatory, Q…`** `-fopenmp-offload-mandatory` (for Clang compatibility) / `/Qopenmp-offload-mandatory`. Generates only a target device (GPU) version of OpenMP target regions; a runtime error is issued if offloading fails. Default `OFF`: both host (CPU) and device versions are generated, and offload failure executes on the host. Requires `-fopenmp-targets` / `/Qopenmp-targets`. Examples: `icpx -qopenmp -fopenmp-targets=spir64 -fopenmp-offload-mandatory test.cpp`; `icx /Qopenmp /Qopenmp-targets:spir64 /Qopenmp-offload-mandatory test.cpp`.
- **`fopenmp-target-default-sub-group-size, Q…`** `=val` / `:val`. Default sub-group size for SPMD kernels generated for OpenMP* target constructs when offloading to SPIR64-based devices. Supported `val`s depend on the hardware; on PonteVecchio (PVC) the supported values are 16 and 32. Default `OFF` (compiler uses default heuristics for global simd length unless a compiler option specifies otherwise). Ignored for SIMD kernels, i.e. when `-fopenmp-target-simd`/`/Qopenmp-target-simd` is also specified. Requires `-fopenmp-targets=spir64` / `/Qopenmp-targets:spir64`. Device-only.
- **`fopenmp-target-loopopt, Q…`** `-fopenmp-target-loopopt` / `/Qopenmp-target-loopopt`. Loop optimizer and auto-vectorization for OpenMP* offloading device compilation when `O2` or higher is set or specified. Default `OFF` (default heuristics). SPIR64-based devices only. Device-only.
- **`fopenmp-target-simd, Q…`** `-fopenmp-target-simd` / `/Qopenmp-target-simd`. OpenMP* SIMD loop vectorization for offloading device compilation when `O2` or higher is set or specified. Default `OFF`. Ignored unless OpenMP offloading is enabled; SPIR64-based devices only. Device-only.
- **`fsycl-allow-device-dependencies`** `-fsycl-allow-device-dependencies`, `-fno-sycl-allow-device-dependencies` (both OS). Dependencies between device images when splitting device code: the positive form allows them, `-fno-…` constructs complete self-contained images with no dependencies. Default `-fno-…`. Requires `-fsycl`; device-code compilation only. **Deprecated; will be removed. Replacement: `fsycl-allow-device-image-dependencies`.** Examples: `icpx -fsycl test.cpp -fsycl-allow-device-dependencies`; `icpx -fsycl test.cpp -fno-sycl-allow-device-dependencies` (Windows uses `icx`).
- **`fsycl-allow-device-image-dependencies`** Same semantics and default as the deprecated option above; `-fsycl-allow-device-image-dependencies` allows dependencies, `-fno-…` constructs self-contained images. Requires `-fsycl`; device-code compilation only.
- **`fsycl-dead-args-optimization`** `-fsycl-dead-args-optimization`, `-fno-sycl-dead-args-optimization` (both OS). Eliminates SYCL dead kernel arguments; can improve performance. Default `OFF` (not eliminated; may change). Requires `-fsycl`. Device-only.
- **`fsycl-device-lib`** `-fsycl-device-lib=library[,library,...]`, `-fno-sycl-device-lib=library[,library,...]` (Linux; the Windows entry repeats the positive form twice and describes `-fno-` only in text [sic]). Values as for `fopenmp-device-lib`: `libm-fp32`, `libm-fp64`, `libc`, `all` (links `libm-fp32`, `libm-fp-64` [sic], and `libc`). Comma-separate with no spaces; `all` supersedes. Default `OFF`. Requires `-fsycl`. Device-only.
- **`fsycl-device-obj`** `-fsycl-device-obj=arg` (both OS). Format of device code in the resulting object: `llvmir` (Instruction Pointer (IP-based) fat objects), `spirv` (SPIR-V*-based objects). Default `-fsycl-device-obj=llvmir`. Experimental. Specific to the target binary type when bundled with the host object or generated independently with `-fsycl-device-only`.
- **`fsycl-device-only`** `-fsycl-device-only` (both OS). Generates a device-only binary. Default `OFF`. Device-only.
- **`fsycl-early-optimizations`** `-fsycl-early-optimizations`, `-fno-sycl-early-optimizations` (both OS). LLVM-related optimizations before SPIR-V* generation; can improve performance. Default `ON`; `-fno-…` disables. Requires `-fsycl`. Device-only.
- **`fsycl-enable-function-pointers`** `-fsycl-enable-function-pointers` (both OS). Function pointers and virtual-function support for SYCL kernels and device functions. Default `OFF`. Experimental; CPU-device only, cannot currently be used for GPU devices. Requires `-fsycl`. Device-only.
- **`fsycl-esimd-force-stateless-mem`** `-fsycl-esimd-force-stateless-mem`, `-fno-sycl-esimd-force-stateless-mem` (both OS). Enforces stateless memory accesses within ESIMD kernels; uses SYCL* accessors to convert stateful to stateless memory, and SIMD intrinsics that cannot be automatically converted are disabled and reported during compilation. Default `OFF` (stateful not converted). Helps avoid the 4Gb-per-surface limitation where a target does not support stateful accesses. Experimental. Requires `-fsycl`. Device-only.
- **`fsycl-explicit-simd`** `-fsycl-explicit-simd`, `-fno-sycl-explicit-simd` (both OS). Experimental "Explicit SIMD" SYCL* extension for lower-level Intel GPU programming, allowing explicitly vectorized device code. Default `-fno-…` (disabled). APIs may change. Requires `-fsycl`. Device-only. **Deprecated; may be removed.**
- **`fsycl-help`** `-fsycl-help[=arg]` (both OS). Emits help from the device compiler backend. `arg` = `x86_64`, `gen`, or `all`; `=all` outputs help for `x86_64` and `gen`, and `all` is the same as no arg. Default `OFF`. Device-only.
- **`fsycl-host-compiler`** `-fsycl-host-compiler=arg` (both OS). Uses the named compiler or path for host compilation of the overall offloading compilation. Default `OFF` (Intel® DPC++ Compiler). Requires `-fsycl`. Examples: `-fsycl-host-compiler=g++` (looks in PATH); `-fsycl-host-compiler=/usr/bin/g++` (explicit path).
- **`fsycl-instrument-device-code`** `-fsycl-instrument-device-code`, `-fno-sycl-instrument-device-code` (both OS). Links/unlinks Instrumentation and Tracing Technology (ITT) device libraries for VTune™, providing annotations to intercept events inside kernels generated by Just in Time (JIT) compilation. Default `ON`. Requires `-fsycl`. Device-only.
- **`fsycl-link`** `-fsycl-link` (both OS). Partial link of device binaries, wrapped by the offload wrapper so they can be linked by the host compiler or linker; `-fsycl -fsycl-link` makes the driver generate a host linkable device object. Default `OFF`. Requires `-fsycl`. Device-only.
- **`fsycl-remove-unused-external-funcs`** `-fsycl-remove-unused-external-funcs`, `-fno-sycl-remove-unused-external-funcs` (both OS). Removes unused `SYCL_EXTERNAL` functions during compilation of SYCL device code. Default `-fsycl-remove-unused-external-funcs` (they are removed); `-fno-…` disables removal and may improve performance because it prevents `SYCL_EXTERNAL` functions from being treated as entry points. Requires `-fsycl`. Device-only. Examples: `icpx -g -fsycl -fno-sycl-remove-unused-external-funcs test.cpp`; Windows `icx /Zi -fsycl -fno-sycl-remove-unused-external-funcs test.cpp`.
- **`fsycl-unnamed-lambda`** `-fsycl-unnamed-lambda`, `-fno-sycl-unnamed-lambda` (both OS). Unnamed SYCL* kernels defined as lambdas. Default `ON`; `-fno-…` disables. Requires `-fsycl`. Device-only.
- **`fsycl-use-bitcode`** `-fsycl-use-bitcode` (both OS). Produces device code in LLVM Intermediate Representation (IR) bitcode format into fat objects. Default `ON`. Requires `-fsycl`. Device-only.
- **`ftarget-export-symbols`** `-ftarget-export-symbols`, `-fno-target-export-symbols` (both OS). Exposes exported symbols in a generated target library for visibility to other modules; can prevent unresolved symbols at runtime. Default `fno-target-export-symbols`. Device-only when SYCL or OpenMP offloading is enabled. Examples: `icpx -fsycl -fsycl-targets=spir64_gen -ftarget-export-symbols -Xsycl-target-backend "-device *"`; `icpx -fiopenmp -fopenmp-targets=spir64_gen -ftarget-export-symbols -Xopenmp-target-backend "-device *"`; Windows `icx -fsycl …` and `icx -Qopenmp -Qopenmp-targets:spir64_gen -ftarget-export-symbols -Xopenmp-target-backend "-device *"`.
- **`nolibsycl`** `-nolibsycl` (both OS). Disables linking of the SYCL* runtime library. Default `OFF` (library linked). With `icx`/`icpx` it is only effective if `-fsycl` is specified. Host-only.
- **`qopenmp-stubs, Qopenmp-stubs`** `-qopenmp-stubs` / `/Qopenmp-stubs`. Compiles OpenMP* programs in sequential mode: directives are ignored and a stub OpenMP library is linked. Default `OFF` (stub library not linked). Host-only.
- **`Wno-sycl-strict`** `-Wno-sycl-strict` (both OS). Disables warnings that enforce strict SYCL* language compatibility. Default `OFF` (warnings enabled).

## Interprocedural Optimization Options

Source: "options that pertain to interprocedural optimization… listed in alphabetical order." Both are host-only.

### `flto`
`-flto[=arg]`, `-fno-lto` (Linux and Windows). Whole program link time optimization (LTO). Values `full`: merge all input into a single module before LTO; default if `-flto` is given with no argument. `thin`: read information from a summary and then do LTO in parallel; also called ThinLTO, scalable and incremental (see `https://clang.llvm.org/docs/ThinLTO.html`). Default `-fno-lto` (no LTO). `-flto`/`-flto=full` may increase compilation time. Linux: `-ipo` is an alias for `-flto` and equivalent to it; to specify a non-default linker you must also specify `fuse-ld`, otherwise the default linker `ld` is used. Windows: `/Qipo` is an alias for `-flto`; `/Qipo` equals `-flto` during the compile step, and during the link step the compiler automatically adds `-fuse-ld=lld` so lld performs the expected optimizations — this automatic inclusion is only for `/Qipo`, not for `-flto` on Windows.

### `ipo, Qipo`
`-ipo`, `-no-ipo` / `/Qipo`, `/Qipo-`. Interprocedural optimization between files, also called multifile interprocedural optimization (multifile IPO) or Whole Program Optimization (WPO): the compiler performs inline function expansion and other interprocedural optimizations for calls to functions defined in separate files, then creates one object file, and you cannot specify its name. Default `-no-ipo` or `/Qipo-` (not enabled). Linux: `-ipo` automatically sets `-flto`. Windows: `/Qipo` automatically sets `-fuse-ld=lld`.

> NOTE: With option `[q or Q]opt-report`, an optimization report is generated during the compilation step for each compiled file and for the link time compilation. Compilation-step files are named `<file-name>.optrpt`; the link-step file is `ipo_out.optprt` [sic: source spelling].

## Profile Guided Optimization Options

Source: "options that pertain to profile-guided optimization… listed in alphabetical order." All are host-only.

### `fprofile-dwo-dir`
`-fprofile-dwo-dir=dir` / `/fprofile-dwo-dir:dir`. Directory where `.dwo` files are stored when using options `fprofile-sample-generate` and `gsplit-dwarf` (`dir` is the DWARF `.dwo` directory). Default `OFF` (default directory). Experimental.

```bash
# creates a.dwo, b.dwo, c.dwo in profile_dwo; the directory is created if it doesn't exist
icpx -c -fprofile-sample-generate -gsplit-dwarf -fprofile-dwo-dir=profile_dwo a.cpp
icx -c -fprofile-sample-generate -gsplit-dwarf -fprofile-dwo-dir=profile_dwo b.c
icx /c /fprofile-sample-generate -gsplit-dwarf /fprofile-dwo-dir:profile_dwo c.cpp    # Windows
```

### `fprofile-ml-use`
`-fprofile-ml-use` / `/fprofile-ml-use`. Uses a pre-trained machine learning model to predict branch execution probabilities driving profile-guided optimizations. Default `OFF` (default static heuristics). Replaces the default static heuristics and serves as a single-pass proxy to get the performance gains of true 2-pass profiling methods by instrumentation/sampling. **Deprecated; will be removed in a future release. There is no replacement option.**

```bash
icx     -c    -fprofile-ml-use t.c
icpx     -c    -fprofile-ml-use    t.cpp
icx     /c    /fprofile-ml-use    t.cpp     # Windows
```

### `fprofile-sample-generate`
`-fprofile-sample-generate[=level]` / `/fprofile-sample-generate[:level]`. Enables the compiler and linker to generate information and adjust optimization for Hardware Profile-Guided Optimization (HWPGO). Values `none` (same as not specifying the option); `keep-all-opt` (generate HWPGO information without disabling any optimization; default if `level` is not specified); `med-fidelity` (generate HWPGO information and disable some optimizations that inhibit profile fidelity); `max-fidelity` (generate HWPGO information and disable most compiler optimizations, targeting execution count profile fidelity above all else). Default `OFF`. Windows cautions: the LLD linker is required and you must specify `/profile-sample-generate` as a link option if LLD is not invoked by `icx`/`icpx`; do not specify `/Ob0` or `/Ob1` with `/fprofile-sample-generate` or `/fprofile-sample-use` because it will disable inlining. See also `fprofile-sample-use`; `fprofile-dwo-dir`; Hardware Profile-Guided Optimization.

### `fprofile-sample-use`
`-fprofile-sample-use=profile-file`, `-fno-profile-sample-use` / `/fprofile-sample-use:profile-file`, `/fno-profile-sample-use`. Enables the compiler and linker to use Hardware Profile-Guided Optimization (HWPGO) information; `profile-file` is the profile data file generated by `llvm-profgen`. Default `fno-profile-sample-use` (profiling information not used during optimization). Experimental.

> NOTE: On Windows, do not specify `/Ob0` or `/Ob1` with `/fprofile-sample-use` or `/fprofile-sample-generate` because it will disable inlining.

See also `fprofile-sample-generate`; `fprofile-dwo-dir`; Hardware Profile-Guided Optimization.

## Optimization Report Options

Source: "options that pertain to optimization reports… listed in alphabetical order."

### `qopt-report, Qopt-report`
`-qopt-report[=arg]` / `/Qopt-report[=arg]`. Generates a textual file with optimization transformation information for the source file being compiled. Values `0`: disable report generation (default when the option is not specified). `1`/`low`: only positive remarks — report only transformations that actually happened. `2`/`medium`: low details plus negative remarks with a reason why the transformation did not happen; default if `arg` is not specified. `3`/`high`: medium details plus all other details. Default `OFF`. Example: `icx -fiopenmp -qopt-report foo.c` generates `foo.optrpt` containing the optimization report messages. When offloading is enabled, two reports are generated: a host-side report named `foo.optrpt` when there is only CPU compilation, and a device-side report named `foo-xxx-yyy.optrpt`, where `xxx` is `sycl` or `openmp` depending on the kind of offload and `yyy` is the offload target name; for example, with `-fopenmp-targets=spir64` the report is named `foo-openmp-spir64.optrpt`.

> NOTE: In releases prior to 2025.0 this option also produced a YAML-formatted optimization report containing LLVM community optimization remarks. Beginning with release 2025.0 the YAML file is no longer produced by this option; you can still produce the YAML report using the Clang option `-fsave-optimization-record`.

### `qopt-report-file, Qopt-report-file`
`-qopt-report-file=keyword` / `/Qopt-report-file:keyword`. Whether report output goes to a file, stderr, or stdout. Values `filename` (name of the file for the generated report), `stderr`, `stdout` (also specifiable as `-qopt-report-stdout` (Linux) or `/Qopt-report-stdout` (Windows)). Default `OFF` (no report generated). You do not have to specify `[q or Q]opt-report` when using this option. `-qopt-report-file=stdout` (Linux) or `/Qopt-report-file:stdout` (Windows) is the same as `-qopt-report-stdout` (Linux) or `/qopt-report-stdout` (Windows) [sic: source shows lowercase `/q`]. Host-only.

### `qopt-report-names, Qopt-report-names`
`-qopt-report-names=keyword` / `/Qopt-report-names:keyword`. Whether mangled or unmangled names appear in the optimization report. Values `mangled` (report contains mangled names; adds encoding (decoration), appropriate when matching annotations with the assembly listing), `unmangled` (no encoding, appropriate for matching the source listing). Default `unmangled`: if the option is not specified and a report is generated, unmangled names appear. If you use this option you must specify either `mangled` or `unmangled`. You do not have to specify `[q or Q]opt-report`.

### `qopt-report-phase, Qopt-report-phase`
`-qopt-report-phase[=list]` / `/Qopt-report-phase[:list]`. One or more optimizer phases for which optimization reports are generated; more than one phase must be comma-separated. Values `cg` (code generation), `ipo` (Interprocedural Optimization), `loop` (loop nest optimization), `openmp` (OpenMP*), `pgo` (Profile Guided optimization), `vec` (vectorization), `all` (all optimizer phases; default if `list` is not specified). Default `OFF`. Phase prerequisites: `cg` requires option `O1`, `O2` (default), or `O3`; `loop` requires `O2` (default) or `O3`; `openmp` requires option `fiopenmp` (or `/Qiopenmp`) or option `[q or Q]openmp`; `pgo` requires Clang option `-fprofile-use` or `-fprofile-sample-use`; `vec` requires `O2` (default) or `O3`. You do not have to specify `[q or Q]opt-report`; for more detail per phase, specify `[q or Q]opt-report=n` along with it and pick an appropriate `n`. When optimization reporting is enabled, the default is `-qopt-report-phase=all` (Linux*) or `/Qopt-report-phase:all` (Windows*).

### `qopt-report-stdout, Qopt-report-stdout`
`-qopt-report-stdout` / `/Qopt-report-stdout`. The generated report goes to stdout; same as `-qopt-report-file=stdout` (Linux) or `/Qopt-report-file:stdout` (Windows). Default `OFF` (no report generated). You do not have to specify `[q or Q]opt-report`.

## Gotchas & failure modes

- **Host-only vs device-only.** Host-only (no device code effect under offload): the whole code-generation section, `device-math-lib`, `fopenmp`, `flink-huge-device-code`, `fsycl-add-default-spec-consts-image`, `fsycl-optimize-non-user-code`, `ftarget-compile-fast`, `nolibsycl`, `qopenmp-link`, `qopenmp-stubs`, `flto`, `ipo`, all `fprofile-*`. Device-only under SYCL*/OpenMP* offload: `fno-sycl-libspirv`, `fopenmp-device-*`, `fopenmp-target-*`, most `fsycl-*`, `ftarget-export-symbols`, `ftarget-register-alloc-mode`, `Xopenmp-target`, `Xs`, `Xsycl-target`. Wrong-side flags silently do nothing.
- **Exclusive pairs.** `-x` vs `-march` and `/Qx` vs `/arch`: last one wins with a warning. `mauto-arch`/`Qauto-arch` cannot combine with `[Q]x`/`[Q]ax`. `-qopenmp-link=dynamic` with `-static` is an error. On Windows `-fsycl` sets `/MD`, so `/MT` is disallowed.
- **Missing companions.** `-flink-huge-device-code` needs `-fsycl` or `-fopenmp-targets` *and* a real link action. `-fopenmp-targets` needs an OpenMP* enablement option. `-fopenmp-offload-mandatory` and `-fopenmp-target-*` need `-fopenmp-targets`; `fopenmp-target-default-sub-group-size` additionally needs `spir64`. Nearly all `fsycl-*` need `-fsycl`. `fprofile-dwo-dir` matters only with `fprofile-sample-generate` and `gsplit-dwarf`.
- **Offload disables target C++ exceptions.** `-fopenmp-targets`/`/Qopenmp-targets` disables C++ exception handling for target compilations; on Linux host compilations add `-fno-exceptions`.
- **`-fsycl-optimize-non-user-code` requires exactly `-O0` or `/Od`** — any other optimization setting, or none, is a compilation error.
- **`-fno-sycl-rdc` redefines other options.** Device code cannot use `SYCL_EXTERNAL`; with `-fsycl-device-code-split`, `off` ≡ `per_source`; with `-c`/`/c` it emits final device binaries in the fat object.
- **`-fsycl-add-default-spec-consts-image` is AOT-only.** Without AOT it emits `warning: -fsycl-add-default-spec-consts-image flag has an effect only in Ahead of Time Compilation mode (AOT).` The documented working commands enable AOT via `-fsycl-targets=spir64_gen -Xs "-device skl"`.
- **HWPGO on Windows needs LLD.** `/profile-sample-generate` must be a link option when LLD is not invoked by `icx`/`icpx`. Never combine `/Ob0` or `/Ob1` with `/fprofile-sample-generate` or `/fprofile-sample-use` — it disables inlining.
- **Optimization report changes.** As of 2025.0 there is no YAML/LLVM-remarks report from `qopt-report`; use `-fsave-optimization-record`. Offload filenames: host `foo.optrpt`, device `foo-<sycl|openmp>-<target>.optrpt`. With `[Q]ipo`, compilation writes `<file-name>.optrpt` and the link step writes `ipo_out.optprt`.
- **Report phase prerequisites.** `cg` needs `O1`/`O2`/`O3`; `loop` and `vec` need `O2`/`O3`; `openmp` needs OpenMP enabled; `pgo` needs Clang `-fprofile-use` or `-fprofile-sample-use`; `qopt-report-names` requires naming `mangled` or `unmangled`.
- **FP precision flags.** `-foffload-fp32-prec-div` and `-foffload-fp32-prec-sqrt` are enabled by default (correct rounding); relaxed defaults are 2.5 ULP for divide and 3 ULP for `sycl::sqrt`.
- **`fopenmp-declare-target-scalar-defaultmap=firstprivate` can change results.** It assumes a `declare target` scalar has the same value in host code and at the start of the target region; the documented counterexample prints `target: -1 == 1` / `host: 0 == 1` instead of `target: 1 == 1` / `host: 1 == 1`.
- **Large target objects.** `-fopenmp-target-buffers=4GB` guards incorrect code for objects over 4GB (4294959104 bytes) but may reduce performance on Intel® GPUs, may have no effect on some SPIR-V* devices, and covers only the documented object categories. `-fsycl-esimd-force-stateless-mem` is the analogous 4Gb-per-surface workaround; inconvertible SIMD intrinsics are disabled and reported at compile time.
- **Frame-pointer interactions.** `-O0`/`-g` flips the default to `-fno-omit-frame-pointer`; under `-fomit-frame-pointer`, `-mno-omit-leaf-frame-pointer` is ignored; `-mno-omit-leaf-frame-pointer` is meaningless when `-fno-omit-frame-pointer` already retains frame pointers everywhere. On Linux the compiler ignores `-fomit-frame-pointer` when GCC 3.2 is installed for C++ with exception handling on.
- **`/EH` negative form is positional:** `/EHsc-` means `/EHs`; `c` requires `a` or `s`.
- **Deprecations cause future breakage:** `m64`/`Qm64` do nothing and will be removed; `device-math-lib` has no replacement; `fsycl-allow-device-dependencies` → `fsycl-allow-device-image-dependencies`; `fsycl-explicit-simd` and `fprofile-ml-use` are removal-tracked (`fprofile-ml-use` with no replacement); static OpenMP* libraries and `-fp` are deprecated; `ICELAKE` is deprecated; `-fsycl-targets` values containing `-sycldevice` are deprecated.
- **Experimental:** `fopenmp-concurrent-host-device-compile`, `fsycl-device-obj`, `fsycl-enable-function-pointers`, `fsycl-esimd-force-stateless-mem`, `fsycl-max-parallel-link-jobs`, `fprofile-dwo-dir`, `fprofile-sample-use`, `ftarget-compile-fast`. `fsycl-explicit-simd` APIs may change; `fsycl-enable-function-pointers` is CPU-device only.
- **PVC register allocation.** Do not use IGC options such as `-ze-opt-large-register-file` through `-Xs` for Ponte Vecchio SYCL*/OpenMP* programs; use `-ftarget-register-alloc-mode`/`/Qtarget-register-alloc-mode`. Other hardware should use the IGC option. Linux default `pvc:auto` differs from Windows default `pvc:default`.
- **Tool merging.** For `-Xopenmp-target`/`-Xsycl-target`, if a target combines frontend and backend in one component, options passed for those tools are merged.
- **`-Xs` under AOT is not a compiler option.** Enumerate AOT backend options with `-fsycl-help=gen` or `-fsycl-help=x86_64`.
- **`-fsycl-pstl-offload` is all-or-nothing per binary.** The same argument must apply to all TUs in an executable or dynamic library; callables share SYCL kernel restrictions (no exceptions, no dynamic allocation) and only heap-allocated containers (or `std::vector`) can be passed — stack arrays must be copied by value or heap-allocated.
- **`-fsycl-id-queries-fit-in-int` defaults ON.** Using `-fno-…` when values do fit in `MAX_INT` can cause unexpected performance issues; truncating to `int` is often incorrect for large work-item counts.
- **`[Q]x` performs a CPU check** except under `-O0`/`/Od`. Compiling `main()` with any `-x`/`/Qx` code value yields a fatal runtime error on unsupported processors, including all non-Intel processors; use `-march`/`/arch` for non-Intel compatibility.
- **`all` supersedes in `-fopenmp-device-lib`/`-fsycl-device-lib`;** comma lists must contain no spaces.

## Source map

- `vecabi` tail — p. 115
- Code Generation Options (`arch` … `xHost, QxHost`) — pp. 115–147
- Offload Compilation, OpenMP*, and Parallel Processing Options (`device-math-lib` … `Xsycl-target`) — pp. 147–216
- Interprocedural Optimization Options (`flto`, `ipo, Qipo`) — pp. 216–218
- Profile Guided Optimization Options (`fprofile-dwo-dir`, `fprofile-ml-use`, `fprofile-sample-generate`, `fprofile-sample-use`) — pp. 218–223
- Optimization Report Options (`qopt-report` … `qopt-report-stdout`) — pp. 223–227
- No catalogued figures occur in these pages (the figure catalog begins at p. 480).

