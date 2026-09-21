---
title: DPC++ / oneAPI Compiler Digest (LLM ingestion head file)
digest_version: 1.0
created_from: >
  dpcpp-cpp-compiler_developer-guide-reference_2026.0-767253-920771.pdf (976 pp)
  dpcpp-cpp-compiler_get-started-guide_2025.2-767258-855932.pdf (12 pp)
source_publisher: Intel Corporation
compiler_versions: Developer Guide and Reference is valid for compiler 2026.0; Get Started Guide is 2025.2
chunk_count: 19
chunk_dir: output/dpcpp-digest/
figure_count: 14
figures: described in text only; no image files are shipped with this digest
structure: head file output/dpcpp-digest.md + 19 chunk files in output/dpcpp-digest/
---

# DPC++ / oneAPI Compiler Digest — master index

> **How to use this digest.** This is a *compressed knowledge extraction* of the Intel® oneAPI
> DPC++/C++ Compiler documentation, built so a future LLM session can ground itself on DPC++,
> SYCL*, OpenMP* offload, oneAPI Level Zero, the compiler math libraries and the Intel compiler
> options **without hallucinating**. Load this head file first, then load **only the chunk files
> you need** using the index and the "load this when" table below. Everything here traces back to
> the two source PDFs; nothing is invented. When precision matters (exact option syntax, defaults,
> function signatures), read the chunk rather than recalling from memory.

---

## 1. Provenance and validity

| Item | Value |
|---|---|
| Document 1 | Intel® oneAPI DPC++/C++ Compiler **Developer Guide and Reference**, version **2026.0** (976 pages, doc ID 767253-920771) |
| Document 2 | **Get Started with the Intel® oneAPI DPC++/C++ Compiler**, version **2025.2** (12 pages, doc ID 767258-855932) |
| Publisher | Intel Corporation |
| Coverage | Pages 1–976 of document 1 and 1–12 of document 2 — **100% walked, no page omitted, no page assigned to two chunks** |
| Extraction | `pdftotext -layout` text plus 14 content figures extracted from the PDF and visually analyzed |
| Digest built | From the two PDFs above only; no external sources were consulted for technical facts |

**Version-sensitive facts to internalize:**

- The developer guide is valid for compiler **2026.0**. Compiler behavior, defaults and supported
  standards change between releases — treat every default as "as documented for 2026.0".
- The Get Started guide (2025.2) documents drivers and workflows (`icx`, `icpx`, `icx-cl`) that
  remain the same family described in the 2026.0 developer guide, but its version is 2025.2.
- The source documents contain **two inconsistent statements about OpenMP support**:
  the introduction's Key Features says "OpenMP 5.0 Version TR4 and some OpenMP 5.1 features", while
  Standards Support and the OpenMP chapter say "most of OpenMP 5.2 and some OpenMP 6.0 TR12
  features". The OpenMP chapter is the more specific and more recent statement; the discrepancy is
  preserved here rather than silently resolved. **Attribute: `_OPENMP`.**
- **`dpcpp` driver is deprecated** and will be removed in a future release. For SYCL compilation use
  **`-fsycl` with the C++ driver** (`icpx` on Linux, `icx`/`icx-cl` on Windows).
- **macOS* is not supported** by this compiler (use Intel® C++ Compiler Classic for macOS/Xcode*).
- Architecture support is **Intel® 64 only**; no 32-bit OS support.

---

## 2. Topic index — which file answers what

Chunks live in `output/dpcpp-digest/`. All 19 files share the same internal structure
(YAML front matter → *Scope* → *Key facts* → topic sections → option/API quick table →
code examples → figures → *Gotchas & failure modes* → *Source map*), so you can load any chunk
standalone.

### A. Getting started, setup, and calling conventions

| File | Source pp. | What it covers |
|---|---|---|
| [`dpcpp-digest/00-intro-setup.md`](dpcpp-digest/00-intro-setup.md) | 1–51 | Compiler introduction and key features; feature requirements and required companion products; command-line setup (Linux/Windows), environment scripts, component locations, how to invoke the compiler, file extensions, makefiles, CMake, Eclipse*/CDT and Microsoft Visual Studio* integration; converting projects to a selected compiler; **C/C++/SYCL calling conventions** (`__regcall`, `__cdecl`, `__stdcall`, `__fastcall`, `__vectorcall`, `__thiscall`) |
| [`dpcpp-digest/15-getstarted-1-need-oneapi.md`](dpcpp-digest/15-getstarted-1-need-oneapi.md) | GS 1–4 | Get Started (2025.2): what the compiler is, what you need, Linux setup and optional GPU drivers/plugins |
| [`dpcpp-digest/16-getstarted-2-build-and-jupyter.md`](dpcpp-digest/16-getstarted-2-build-and-jupyter.md) | GS 5–8 | Get Started: invoking the compiler from the command line and Eclipse* CDT; hello-world build; Windows/Visual Studio setup; Jupyter notebooks |
| [`dpcpp-digest/17-getstarted-3-fpga-gpu-migration.md`](dpcpp-digest/17-getstarted-3-fpga-gpu-migration.md) | GS 9–12 | Get Started: driver choice, command-line and Visual Studio builds, sample projects, next steps |

### B. Compiler options (the reference bulk)

| File | Source pp. | What it covers |
|---|---|---|
| [`dpcpp-digest/01-options-list-and-optimization.md`](dpcpp-digest/01-options-list-and-optimization.md) | 52–114 | Option conventions and general rules; **the complete alphabetical option list**; Optimization Options (`-O`, `-x`, `-ax`, `-m`, `-qopt-*`, `-fno-alias`, …); Advanced Optimization Options |
| [`dpcpp-digest/02-options-codegen-and-ipo-pgo.md`](dpcpp-digest/02-options-codegen-and-ipo-pgo.md) | 115–227 | Code generation options (`-march`, `-mtune`, `-mcmodel`, ISA selection); **offload / OpenMP* / SYCL* and parallel-processing options**; interprocedural optimization (IPO) options; profile-guided optimization (PGO) options; optimization report options |
| [`dpcpp-digest/03-options-fp-inline-output-preproc.md`](dpcpp-digest/03-options-fp-inline-output-preproc.md) | 228–294 | Floating-point options (`-fp-model`, `-ffp-*`, `-fimf-*`, `-fma`, `-ftz`, `-pc`); inlining options; output/debug/precompiled-header options; preprocessor options (`-D`, `-U`, `-I`, `-include`, `-E`, …) |
| [`dpcpp-digest/04-options-lang-data-diagnostics.md`](dpcpp-digest/04-options-lang-data-diagnostics.md) | 295–344 | Component control (`-Qoption`); language options (`-std`, `-ansi`, `-x`, `-Z*`); data options (`-fcommon`, `-fno-rtti`, `-Qlong-double`, …); compiler diagnostic options (`-w`, `-W*`, `-diag-*`, `-qunknown-option-as-warning`) |
| [`dpcpp-digest/05-options-compat-link-misc.md`](dpcpp-digest/05-options-compat-link-misc.md) | 345–389 | Compatibility options; **linking/linker options** (`-static-intel`, `-shared-intel`, `-L`, `-l`, `-Wl`, `-shared`, `-rdynamic`); miscellaneous options; **deprecated and removed options**; display option information; alternate compiler options; portability and GCC*-compatible warning options |

### C. Language semantics: floating point, attributes, pragmas, macros

| File | Source pp. | What it covers |
|---|---|---|
| [`dpcpp-digest/06-floating-point-operations.md`](dpcpp-digest/06-floating-point-operations.md) | 390–397 | Floating-point programming tradeoffs (accuracy vs reproducibility vs performance); FP optimizations; denormal numbers; FP environment; setting **FTZ and DAZ**; FP tuning; IEEE FP operations |
| [`dpcpp-digest/07a-attributes-intrinsics-macros.md`](dpcpp-digest/07a-attributes-intrinsics-macros.md) | 398–594 | **Attributes** (`align`, `align_value`, `allow_cpu_features`, `code_align`, `const`, `cpu_dispatch`/`cpu_specific`, `target`); **intrinsics** usage rules; **predefined macros** (ISO standard and Intel-specific, including compiler-version detection); and the **Libraries** section body: creating/using/managing/redistributing libraries, the redistributable-library table, Intel's Memory Allocator (`libqkmalloc`), **SIMD Data Layout Templates (SDLT)**, Intel® C++ Class Libraries, C++ Asynchronous I/O, IEEE 754-2008 library, Numeric String Conversion library |
| [`dpcpp-digest/07b-pragmas-and-errors.md`](dpcpp-digest/07b-pragmas-and-errors.md) | 595–621 | **Pragmas**: Intel-specific pragma reference (`block_loop`, `distribute_point`, `inline`/`forceinline`/`noinline`, `ivdep`, `loop_count`, `nofusion`, `novector`, `omp target variant dispatch`, `ompx prefetch data`, `prefetch`/`noprefetch`, `unroll`/`nounroll`, `unroll_and_jam`/`nounroll_and_jam`, `vector`); **supported OpenMP* pragmas** (alphabetical and by category); pragmas compatible with Microsoft* and GCC* compilers; syntactic and semantic errors |

### D. Compilation, environment, and offload

| File | Source pp. | What it covers |
|---|---|---|
| [`dpcpp-digest/08-compilation-and-env-vars.md`](dpcpp-digest/08-compilation-and-env-vars.md) | 622–665 | Compilation phases and defaults; **all documented compile-time and runtime environment variables**; passing options to the linker; alternate tools; configuration files (`*.cfg`) and response files; Linux global symbols and visibility attributes; saving compiler info in the executable; linking debug information; **ahead-of-time (AOT) compilation** for CPU/GPU/non-Intel GPUs; device offload compilation considerations; third-party host compiler for SYCL; Ccache* |
| [`dpcpp-digest/09-openmp.md`](dpcpp-digest/09-openmp.md) | 666–733 | **OpenMP***: enabling and pragma syntax; parallel processing model; worksharing; tasks and scheduling; reductions; controlling thread allocation; **thread affinity (`KMP_AFFINITY`, `OMP_PLACES`, `OMP_PROC_BIND`, `KMP_HW_SUBSET`)** with topology figures; OpenMP library/extension routines and environment variables; memory spaces and allocators; contexts and selectors; offloading SPMD/SIMT and SIMD models; advanced issues; implementation-defined behaviors; worked examples |
| [`dpcpp-digest/10-sycl-offload-sanitizers.md`](dpcpp-digest/10-sycl-offload-sanitizers.md) | 734–743 | **SYCL*** support and the SYCL extension status catalog; redistributing a SYCL* application; **CUDA* → SYCL* migration** (math API mapping); device-side and host-side **compiler sanitizers** (ASan/MSan/TSan) |
| [`dpcpp-digest/11-level-zero.md`](dpcpp-digest/11-level-zero.md) | 744–762 | **Intel® oneAPI Level Zero**: the Level Zero switch, `ONEAPI_DEVICE_SELECTOR`, the Level Zero backend specification (feature-test macro, backend selection, SYCL↔Level Zero interop, handle ownership, buffer/image synchronization, device info), and programming the Level Zero backend |

### E. Optimization, performance, and libraries

| File | Source pp. | What it covers |
|---|---|---|
| [`dpcpp-digest/12-vectorization-and-code-size.md`](dpcpp-digest/12-vectorization-and-code-size.md) | 763–813 | **Automatic vectorization** (guidelines, obstacles, hints, loop constructs); **user-mandated/SIMD vector programming** (`#pragma omp simd`); SIMD-enabled functions and function pointers; **ESIMD** (Explicit SIMD SYCL* extension) and `invoke_simd`; **instrumented PGO (IPGO)** and **hardware PGO (HWPGO)**; high-level optimization (HLO); interprocedural optimization (IPO) and inline expansion; methods to optimize code size |
| [`dpcpp-digest/13-math-library-and-device-lib.md`](dpcpp-digest/13-math-library-and-device-lib.md) | 814–947 | **Optimization reports** (`-qopt-report` family); the **compiler math library** (`mathimf.h`, full function list, C99 macros, `errno`/rounding rules, complex functions); the **SYCL* device library** (basic arithmetic, simple math, integer ops, rounding-mode arithmetic, type casting, `half`/`bfloat16`, and **IMF transcendental math** with accuracy tables and special values) |
| [`dpcpp-digest/14-compat-portability-notices.md`](dpcpp-digest/14-compat-portability-notices.md) | 948–976 | Standards conformance (C/C++, SYCL*, OpenMP*, IEEE 754-2008); **GCC* compatibility and interoperability**; **Microsoft* compatibility**; porting from Microsoft Visual C++* and from GCC* (makefile changes, other considerations); technically relevant notices; the source document's compiler-option and topic indexes |

---

## 3. "Load this when…" — routing by question

| Question shape | Load |
|---|---|
| "How do I invoke/compile with this compiler on Linux or Windows?" | 00, 16 |
| "What does option `-X` do / what is the default?" | 01–05 (pick the range by category, or grep all five) |
| "How do I make the compiler vectorize / optimize this loop?" | 12, then 01/02 for the flags |
| "What does `-qopt-report` output mean?" | 13, then 02 |
| "How do I enable OpenMP / offload to a GPU?" | 09, then 02 (offload options), 11 (Level Zero backend) |
| "Why is my floating-point result different / non-reproducible?" | 06, 03, 13 |
| "How do I write SYCL code with `-fsycl`?" | 10, 11; `dpcpp` is deprecated → use `-fsycl` with the C++ driver |
| "What are the predefined macros / how do I detect the compiler or version?" | 07a, 07b |
| "What attributes/pragmas exist and what is the syntax?" | 07a, 07b |
| "How do I pin OpenMP threads to cores (affinity)?" | 09 (affinity figures and `KMP_AFFINITY`/`OMP_PLACES`/`OMP_PROC_BIND`) |
| "Which environment variables affect compilation or runtime?" | 08, 09 |
| "How do I redistribute Intel libraries with my app?" | 07a, 05, 14 |
| "How do I port from GCC* or MSVC*?" | 14, 05 |
| "How do I profile (PGO) or use IPO?" | 12, 02 |
| "Is `-X` available on Windows too / what is the Windows spelling?" | 01–05 (every entry lists both spellings), 14 |
| "What math functions exist and what is their accuracy?" | 13 |
| "How do I use SIMD Data Layout Templates / Intel C++ classes?" | 07a |
| "What is the compiler's SYCL/OpenMP/C++ standard support?" | 14, 00, 09 |

---

## 4. Cross-cutting quick reference

Short orientation answers. **Each is expanded, with syntax and caveats, in the cited chunk.**

### 4.1 The 60-second mental model

- Two drivers: **`icx`** (C) / **`icpx`** (C++, and SYCL) on Linux; **`icx`** / **`icx-cl`** on
  Windows. `icx-cl` uses MSVC-style options and is **experimental on Linux** (needs the Microsoft
  Visual Studio Package). → `00`, `16`, `17`
- **Option syntax follows the driver**: `-` options for `icx`/`icpx`, `/` options for `icx-cl`.
  The docs write `[Q]x` to mean `-x` (Linux) / `/Qx` (Windows).
- **Default optimization is `-O2`** and **default FP model is `fast`** (`-fp-model=fast`); default
  standard levels are **C17** and **C++17**. → `08`, `01`, `03`
- SYCL compilation is **`-fsycl` with the C++ driver**; `-fsycl` assumes `-fsycl-targets=spir64`
  unless you set it. The old **`dpcpp` driver is deprecated**. → `10`, `16`
- **OpenMP** is enabled by **`-qopenmp`** (Linux) / **`/Qopenmp`** (Windows); target offload adds
  **`-fopenmp-targets=spir64`** / **`/Qopenmp-targets=spir64`**. → `09`
- Intel® oneAPI **Level Zero** is the direct-to-metal accelerator API, also the default offload
  backend when none is specified; **Unified Runtime** adapters sit above it. → `11`

### 4.2 Most-used flags by task

| Task | Flags (Linux / Windows) |
|---|---|
| Optimize | `-O2` (default), `-O3`, `-Ofast`, `-xHOST`, `-xCORE-AVX512` · `/O2`, `/O3`, `/Qx…` |
| Enable OpenMP | `-qopenmp` · `/Qopenmp` |
| Offload to GPU (OpenMP) | `-fopenmp-targets=spir64` · `/Qopenmp-targets=spir64` |
| Compile SYCL | `-fsycl` (+ `-fsycl-targets=…`) |
| Control FP model | `-fp-model=fast|precise|strict` · `/fp:fast|precise|strict` |
| Flush denormals | `-ftz` / `-fno-ftz` (`-ftz` is implied by `-fp-model=fast`) |
| Vectorization report | `-qopt-report[=n] -qopt-report-phase=vec` (n = 0–3) · `/Qopt-report[:n]` |
| Interprocedural optimization | `-ipo` · `/Qipo` |
| PGO | `-fprofile-generate` / `-fprofile-use` (IPGO); `-fprofile-sample-generate` / `-fprofile-sample-use` (HWPGO) |
| Static vs dynamic Intel libs | `-static-intel` (default ON) / `-shared-intel` |
| Disable vectorization | `-no-vec` · `/Qvec-` (or `#pragma novector`) |
| SIMD pragma | `#pragma omp simd`, `#pragma omp declare simd` |

> Verify exact spellings against the relevant chunk — the tables above are orientation only and the
> source lists additional per-OS variants.

### 4.3 Standards/feature posture (as the source states it)

- **C++**: C++17 default; C++20 **Partial**; C++23 **Partial**; front end is **Clang**.
- **C**: C17 default.
- **SYCL**: **SYCL 2020 conformant**.
- **OpenMP**: most of 5.2 and some OpenMP 6.0 TR12 (chapter wording), with the introduction
  additionally mentioning OpenMP 5.0 TR4 / some 5.1 — see the discrepancy note in §1.
- **FP**: close approximation to **IEEE 754-2008**; the Intel® IEEE 754-2008 Binary Floating-point
  Conformance Library covers `binary32`/`binary64`.

→ Details and caveats: `14`, `00`, `09`, `06`.

### 4.4 Gotcha digest (the highest-value traps)

These recur across the source and are the most likely causes of "compiles but wrong" or
"works on my machine" failures. Each is documented with fixes in its chunk.

1. **Host-only vs device options.** A large share of optimization, FP and diagnostic options are
   annotated *"only applies to host compilation… when offloading is enabled"*. Quietly setting them
   does nothing to device code. → `01`–`05`, `12`
2. **`-fp-model=fast` is the default**, so results are *not* reproducible across builds by default
   and lower-accuracy math routines may be used. Use `precise`/`strict` and `-fimf-*` deliberately. → `03`, `06`
3. **`-ffp-accuracy` is incompatible with `-fimf-*`/`-Qimf-*`** — the compiler reports an error. → `03`
4. **Denormals may be flushed** (`FTZ`/`DAZ`) under the fast model, silently changing results. → `06`, `03`
5. **`-g` implies `-O0`** — adding debug info can cost you all optimization. → `03`
6. **`-x` (language) is positional**: it applies only to files listed *after* it; `-x none` restores
   file-extension recognition. → `04`
7. **Options are case sensitive**: `c` prevents linking, `C` keeps comments in preprocessed output. → `01`
8. **Removed vs deprecated options**: deprecated still work (may be removed); removed ones make the
   compiler warn/error — real build breakage on upgrade. `/Zg` has **no replacement**. → `05`, `04`
9. **`dpcpp` is deprecated**; port to `-fsycl` with the C++ driver. → `00`, `10`
10. **Unknown Linux options are errors** unless `-qunknown-option-as-warning` is given. → `04`
11. **IPO is silently skipped** when no input file is a mock object, so multi-file IPO may not happen
    at all; IPO mock objects are compiler-specific and incompatible with other compilers. → `12`, `14`
12. **Never merge frequency and mispredict HWPGO profiles**; HWPGO requires DWARF debug info. → `12`
13. **Optimization reports differ** depending on whether IPO ran at compile time or link time. → `13`
14. **`-static-intel` is ON by default** (except the OpenMP runtime), which affects redistribution and
    binary size. → `05`, `07a`
15. **SIMD-enabled function pointers** are binary-incompatible with regular pointers and **disabled by
    default** (`-Xclang -fsycl-allow-func-ptr` to allow). → `12`
16. **`omp target variant dispatch` is deprecated with support removed** (use `omp dispatch`);
    `omp master` is deprecated (see `omp masked`). → `07b`
17. **Device-side sanitizers are recent and device-limited**: MSan since 2025.1 (local/private memory
    check 2025.2), TSan since 2025.2; OpenMP C/C++ sanitizers run on a GPU device only. → `10`
18. **Experimental SYCL extensions may change or disappear without notice**; check the extension
    status catalog before relying on them. → `10`

---

## 5. Figure index (14 content figures, all analyzed and text-described)

**No image files are shipped with this digest.** Every figure was extracted from the PDF, visually
analyzed, and written up as a **textual description dense enough that the figure is optional** —
those descriptions live inline in the chunk listed below, each marked
`*[caption — source p. N]*` followed by the description. If you have the source PDF, the page number
in the first column locates the original figure.

| Source page | Subject | Chunk |
|---|---|---|
| 480 | SDLT element-wise op layout (A3..A0 / B3..B0 → A3opB3…) | 07a |
| 485 | SIMD type hierarchy for `M64` / `M128` | 07a |
| 507 | Packed FP operand/return layout (`F32vec4` vs `F32vec2`) | 07a |
| 706 | OpenMP topology, default thread-ID mapping 0–7 | 09 |
| 707 | OpenMP compact affinity thread IDs | 09 |
| 707 | OpenMP scatter affinity thread IDs | 09 |
| 708 | OpenMP re-binding with partial parallel regions | 09 |
| 711 | `granularity=fine`/`thread` + `compact` thread-ID sets | 09 |
| 712 | `norespect` modifier thread-ID sets | 09 |
| 717 | Explicit `proclist=` OS proc ID assignment | 09 |
| 718 | `proclist` + `verbose` output variant | 09 |
| 767 | R/G/B three-channel layout (alignment discussion) | 12 |
| 781 | SIMD/auto-vectorization vs OpenMP*/auto-parallelization | 12 |
| 781 | Six-level vectorization control stack | 12 |

The Get Started guide (2025.2) contains no figures.

---

## 6. Digest construction and known limitations

**How it was built.** Both PDFs were converted to text page-by-page (`pdftotext -layout`), the
documents' own tables of contents were parsed to page boundaries, and the 976-page developer guide
was partitioned into **19 non-overlapping page ranges covering every page exactly once**. Each range
was compressed independently against a shared style contract requiring exact preservation of
option/pragma/attribute/macro/function names, defaults, legal values, platform differences,
deprecation notes and all documented examples. All 14 content figures were extracted, visually
analyzed, and written up as textual descriptions inside the relevant chunks (the image files
themselves are not shipped). Coverage and structure were verified programmatically
(page coverage, fence balance, front matter, size ratios).

**Known limitations — read these before trusting an answer blindly:**

1. **Compression is uneven, and the overall ratio is ~41%** (2.32 MB of source text → ~979 KB of
   digest). This is deliberately more generous than a pure summary: the goal is to *supplement*
   thin training data on oneAPI/SYCL/Intel compilers, so technical substance was prioritised over
   raw size reduction. The options chunks (01–03) sit near 43–44% because their source is dominated
   by irreducible per-option tables; `07a` and `13` are denser at ~35–39%; the leanest is `05` at ~28%.
   `12-vectorization-and-code-size` is the least compressed at **~65%** — it is complete and correct
   but retains more verbatim worked examples than the others. If you need a smaller context load,
   summarise that chunk further on the fly rather than trusting a lossy re-read.
2. **This is a digest, not the documentation.** Long-tail detail is compressed. For a
   compliance-critical or safety-critical answer, verify against the original PDF.
3. **The source itself is occasionally ambiguous or garbled** by PDF extraction; chunks flag such
   spots inline as `[sic: source garbled]` or `[unclear in source]` rather than guessing.
4. **The OpenMP version discrepancy** described in §1 is preserved, not resolved.
5. **Option lists reflect 2026.0.** Later compiler releases add, deprecate and remove options.
6. **The source documents' own indexes** (compiler option index, topic index) were captured inside
   chunk `14` and can be used as an additional finding aid.
7. **Get Started chunk 17 covers what its pages actually contain** (driver choice, builds, samples,
   next steps) — the guide does not contain dedicated FPGA/GPU or CUDA-migration chapters, so those
   topics are documented in the developer guide instead (`10`, `11`, `12`).
8. **Chunk `15` has no "Gotchas" section** because its four source pages contain no caveats worth
   recording; this is intentional, not an omission.
9. **All 19 chunk files were checked programmatically** for balanced, language-tagged Markdown code
   fences, presence of front matter, presence of a source map, and page coverage. All 14 figures were
   checked to have a text description present in their chunk.

---

## 7. File map

```
output/
  dpcpp-digest.md          <- this master head file (start here)
  dpcpp-digest/            <- the 19 chunk files
    00-intro-setup.md
    01-options-list-and-optimization.md
    02-options-codegen-and-ipo-pgo.md
    03-options-fp-inline-output-preproc.md
    04-options-lang-data-diagnostics.md
    05-options-compat-link-misc.md
    06-floating-point-operations.md
    07a-attributes-intrinsics-macros.md
    07b-pragmas-and-errors.md
    08-compilation-and-env-vars.md
    09-openmp.md
    10-sycl-offload-sanitizers.md
    11-level-zero.md
    12-vectorization-and-code-size.md
    13-math-library-and-device-lib.md
    14-compat-portability-notices.md
    15-getstarted-1-need-oneapi.md
    16-getstarted-2-build-and-jupyter.md
    17-getstarted-3-fpga-gpu-migration.md
```
