---
chunk: 12-vectorization-and-code-size
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 763-813
covers: Automatic vectorization and its guidelines/hints; user-mandated SIMD vector programming; SIMD-enabled functions and function pointers; Explicit SIMD SYCL* extension (ESIMD); instrumented and hardware profile-guided optimization; high-level optimization (HLO); interprocedural optimization (IPO) and inlining; methods to optimize code size
---

# Vectorization, Explicit SIMD, PGO, IPO, and Code Size

> **Scope.** How the Intel® oneAPI DPC++/C++ Compiler auto-vectorizes loops and which hints/pragmas/options change that; how to write and declare SIMD-enabled (elemental) functions and function pointers; ESIMD; flags for Instrumented and Hardware Profile-Guided Optimization; HLO/IPO/inlining; options that shrink code size.

## Key facts

- **Vectorization** = scalar algorithm (one pair of operands per operation) → vector process where one instruction refers to a vector (adjacent values).
- **Auto-vectorizer**: Intel® SSE/SSE2/SSE3/SSE4/SSSE3, Intel® AVX/AVX2/AVX-512 SIMD; converts sequential SIMD processing up to 16 elements into a parallel operation (by data type); runs when compiler generates packed SIMD instructions to unroll a loop; Intel® 64 only. Intel® Advisor (Intel® oneAPI Base Toolkit) **Vectorization Advisor**.
- Enabled at default optimization for Intel® and non-Intel microprocessors; may call library routines (more gain on Intel); affected by `/arch` (Win), `-m` (Linux), `[Q]x`. Sought at `O2`+; `O1`/`-no-vec` (Linux)/`/Qvec-` (Windows) leave SIMD registers partly unused.
- **User-mandated/SIMD vectorization** supplements auto-vectorization as OpenMP* parallelization supplements auto-parallelization; uses `#pragma omp simd`, and compiler **warns** if it cannot vectorize, whereas hints (even `#pragma vector always`) leave the decision to compiler.
- SIMD-enabled functions were formerly **elemental functions**. SIMD-enabled function pointers are binary-incompatible with regular pointers, **disabled by default**. **ESIMD** = lower-level Intel® GPU programming; always requires **subgroup size one**.
- **IPGO** profiles in software with large instrumentation overhead; **HWPGO** needs no instrumentation, samples the optimized binary on PMU events (Linux perf or SEP). 2024.0 supports **unpredictable branch profiles** (can prefer Conditional Move (CMOV) to branches).
- **IPO:** `[Q]ip` single-file, `[Q]ipo` multi-file; link with `-flto` or `/Qipo`; inlining heuristics differ with `[Q]prof-use`. Code size: `Os` favors size over `O2`; `O1` minimizes size, **implies `Os`**.
- **2024.0 breaking change:** `-mllvm` options are **no longer passed through to linker option processing**; use `-Wl` (e.g. `-Wl,-plugin-opt,-lto-debug-options`).

## Compiler option quick table

| name | purpose / values / default | notes |
|---|---|---|
| `-q[no-]opt-dynamic-align` (Linux), `/Qopt-dynamic-align[-]` (Windows) | dynamic data alignment opts; enabled | helps long trip count loops; disabling may cut perf, can improve bitwise reproducibility |
| `/arch` (Win), `-m` (Linux), `[Q]x` | affect vectorization performed | — |
| `-vec` / `/Qvec`; `-no-vec` / `/Qvec-` | enable / disable vectorization | `O2` already enables; disabling improves compile time |
| `O1` | minimize code size; disables vectorization | implies `Os`; may help very large code with many branches, execution not loop-dominated |
| `O2` | default; some HLO; enables vectorization | — |
| `O3` | best chance for loop transformations optimizing memory access | extra loop optimizations |
| `Os` | favor size over speed | size-neutral opts only; smaller than `O2` |
| `-qopt-report`, `/Qopt-report` | optimization/vectorization report (`-qopt-report=3`, `/Qopt-report=3`) | companion `-qopt-report-phase`, `/Qopt-report-phase` |
| `-xAVX`/`/QxAVX`; `-xCORE-AVX512`, `-Ofast` | instruction selection / example flags | all PGO examples use `-xCORE-AVX512 -Ofast` |
| `[Q]ipo` / `[Q]ip` | multi-file / single-file IPO | inlines cross-file / same-file procedures; compile then link |
| `-flto` or `/Qipo` | link with IPO | invokes compiler a final time over mock objects |
| `[Q]prof-use` | PGO use; changes inlining heuristics | — |
| `[Q]std=c99` | required to use `restrict` | — |
| `-qopenmp`/`-Qopenmp`, `-qopenmp-simd`/`-Qopenmp-simd` | required for `#pragma omp declare simd` (doc: `[q or Q]openmp`, `[q or Q]openmp-simd`) | — |
| `-fuse-ld=lld` | use LLVM linker LLD | Linux default = Berkeley Free Distribution linker; Windows LLD only, default with `/Qipo` |
| `-fprofile-generate=app.profraw` | IPGO step 1 (instrumentation) | name overridable via `LLVM_PROFILE_FILE`; slows execution considerably |
| `-fprofile-use=app.prof` | IPGO step 4 (recompile with profile) | — |
| `-fprofile-sample-generate` | HWPGO step 1 | default: no effect on optimizations or speed; needs DWARF debug info (Linux and Windows) |
| `-fprofile-sample-use=app.freq.prof` | HWPGO step 5 (recompile with PMU profile) | may combine with `-fprofile-sample-generate` |
| `-gsplit-dwarf` + `-fprofile-dwo-dir=<dir>` | split debug info; choose `.dwo` location | default: debug info embedded in object/executable |
| `-mllvm -unpredictable-hints-file=app.misp.prof` | supply branch-mispredict profile | `-mllvm` no longer reaches linker option processing (2024.0+) |
| `-Wl` | pass options to linker (2024.0+) | `-Wl,-plugin-opt,-lto-debug-options` |
| `--print-prog-name=llvm-profgen` | locate product `llvm-profgen` | Windows: `icx /nologo /clang:--print-prog-name=llvm-profgen`; `--include-intel-llvm` on `setvars` adds tools to `PATH` |
| `-fno-builtin` (Linux), `/Qno-builtin-<name>` + `/Oi-` (Windows) | disable inlining/by-name recognition of intrinsics | `fno-builtin` already default with `O1`; `/Qno-builtin-<name>` fine-tunes `/Oi-`; C++ `nolib-inline` disables stdlib/intrinsic inline expansion |
| `-fpic` | position-independent code | default: not generated; needed for function preemption |
| `fno-inline` / `Ob0` | disable/decrease inlining | cuts size, likely hurts perf |
| `Wl, --strip-all` (Linux) | strip symbols | Windows: none; debugging becomes very difficult |
| `shared-intel` (Linux), `MD` (Windows) | dynamically link Intel-provided libraries | default: some Intel support/perf libraries static; `MD` affects all libraries |
| `-fdata-sections -ffunction-sections -Wl,--gc-sections` (Linux), `/Gw /Gy /link /OPT:REF` (Windows) | exclude unused code/data (function-/data-level linking) | linker sees `Wl, --gc-sections` / `link /OPT:REF` |
| `-fno-exceptions` or `-fno-asynchronous-unwind-tables` (Linux) | optimize `.eh_frame` exception data | Windows: none; may shrink up to 15%, platform-dependent |
| `unroll=0` (Linux), `Qunroll:0` (Windows) | disable loop unrolling | already default with `Os`/`O1`; `Qunroll:0` **not available for SYCL** |
| `-c`, `-o`, `-static`; `/c`, `/Fe` | stop after objects; name output; static link | — |
| `-fsycl` | compile SYCL (also ESIMD) | `clang++ -fsycl vadd_usm.cpp` / `icpx -fsycl vadd_usm.cpp` |
| `-fno-sycl-device-code-split-esimd` | keep `invoke_simd()` caller/callee in one module | pair with `-Xclang -fsycl-allow-func-ptr` |
| `-Xclang -fsycl-allow-func-ptr` | allow function pointers in SYCL/ESIMD | not allowed by default |
| `-fsycl-esimd-force-stateless-mem` | enable `sycl::accessor::get_pointer()`/`operator[]` in ESIMD | else accessor memory uses explicit APIs, e.g. `sycl::ext::intel::esimd::block_store(acc, offset)` |

### Pragmas, keywords, attributes, built-ins

| name | purpose / notes |
|---|---|
| `#pragma ivdep` | ignore **assumed** (not proven) vector dependencies; incorrect use → wrong results; `#pragma ivdep!` = ignore assumed dependencies |
| `#pragma novector` | never vectorize this loop; avoids runtime dependency-testing overhead |
| `#pragma vector always` | vectorize regardless of efficiency analysis (safe only if vectorization is safe); `assert` keyword makes heuristic failure an error-level assertion |
| `#pragma vector align` | assert following-loop data aligned (16-byte boundary for Intel® SSE) |
| `#pragma vector {aligned\|unaligned\|always\|temporal\|nontemporal}` | how to vectorize; ignore efficiency heuristics |
| `#pragma loop count (n)` | typical trip count; aids worthwhile/alternative-path decision |
| `#pragma omp simd` | user-mandated SIMD loop; warns if it cannot vectorize; optional clauses |
| `#pragma omp declare simd clauses` | declare SIMD-enabled function/pointer; requires `[q or Q]openmp` or `[q or Q]openmp-simd` |
| `restrict` / `__restrict` | referenced memory not aliased; requires `[Q]std=c99`; reduces portability; runtime alias check still done if used |
| `__declspec(align(n))`, `__declspec(align(n,off))` | align variable to n-byte boundary (offset `off`); address mod n = 0 (or off); `align(base,offset)` needs `0 <= offset < base`, base a power of two |
| `__builtin_assume_aligned (a,n)` | assume array `a` n-byte aligned; for when compiler failed to obtain alignment info |
| `__builtin_assume (cond)` | assume the condition true there; typically conveys alignment properties |
| `__declspec(vector)` / `__attribute__((vector))` | vector declaration for functions/loops; multiple instances in a parallel context not sequenced |
| `__declspec(vector[clauses])` / `__attribute__((vector(clauses)))` | clauses: `linear`, `[no]mask`, `processor(cpuid)`, `uniform`, `vectorlength(n)` |
| `__declspec(vector_variant(clauses))` / `__attribute__((vector_variant(clauses)))` | user-defined vector implementation; `implements` required (`implements (function declarator) [, simd-clauses]`), `simd-clauses` optional |
| `__inline`, `__forceinline`; `#pragma inline`, `#pragma forceinline`; `#pragma noinline` | direct inlining; pragmas inline calls inside the targeted procedure if legal |
| `__attribute__((always_inline))` | GNU inlining attribute; inlines even with no optimization level |
| `[[intel::sycl_explicit_simd]]` | marks ESIMD lambda/kernel; ESIMD APIs usable inside |
| `[[intel::device_indirectly_callable]]`, `SYCL_EXTERNAL`, `SYCL_ESIMD_FUNCTION`, `__regcall`, `[[sycl::reqd_sub_group_size(N)]]` | used in the ESIMD `invoke_simd` example |

## Multi-tile Multi-card Examples (source tail, p. 763)

One `context` shared by multiple root devices: `platform(gpu_selector{})` → `P.get_devices()` → `context(RootDevices)` → per-device `queue(C, D)` with `Q.submit(...)`. Data is sharable across multi-card **but requires explicit copying**. NOTE: allocate/synchronize memory for your programming model/algorithm. Next-step examples: `dgemm`, `gpu2gpu`.

## Vectorization

**Automatic Vectorization.** The auto-vectorizer detects parallelizable operations and converts sequential operations to parallel: e.g. a sequential SIMD instruction processing up to 16 elements becomes one parallel operation (by data type). It occurs when the compiler generates packed SIMD instructions to unroll a loop; packed instructions operate on more than one element, so the loop runs more efficiently. "Auto-vectorization" only emphasizes that the compiler finds and optimizes suitable loops itself, without external input, though keywords/directives may be needed. A variety of auto-vectorizing hints is supported. Intel® 64 only.

**Vectorization Programming Guidelines.** Two factors restrict vectorization. **Hardware:** Intel® SSE vector memory operations are stride-1 only and prefer 16-byte-aligned references, so a loop abstractly recognized as vectorizable may still not vectorize for a distinct target architecture. **Source style:** avoid a pointer unless its association with a variable is established in the same procedure, else the compiler may not prove two memory references distinct. Most inhibitors live in loop structures (keywords, operators, data references, pointer arithmetic, memory operations).

*Vectorize innermost loops — Use:* straight-line code (one basic block); vector data only (arrays/invariant expressions on the RHS; array refs may appear on the LHS); only assignment statements. *Avoid:* function calls (other than math library calls); non-vectorizable operations (loop cannot be vectorized, or an operation is emulated through several instructions); mixing vectorizable types in one loop; data-dependent loop exits. Edit loops only to enable vectorization; do not **unroll loops** (compiler does it automatically) or **split one multi-statement loop into several single-statement loops**.

*Writing vectorizable code:* simple `for` loops, invariant upper iteration limit (innermost of a nest: may be a function of outer indices); straight-line code (no `switch`/`goto`/`return`/most calls; `if` not treatable as masked assignment); avoid inter-iteration dependencies, at least read-after-write; array notation instead of pointers (aliased pointers create unexpected dependencies; the compiler often cannot tell whether pointer code is safe); the loop index directly in subscripts, not a separate counter; efficient memory access (unit-stride inner loops, minimal indirect addressing, 16-byte alignment for Intel® SSE); careful data layout (Intel® SSE data movement much more efficient at 16-byte alignment; success needs a layout that, with restructuring such as loop peeling, gives aligned accesses throughout); aligned data structures via `__declspec(align)`.

> **Caution:** use this hint with care — incorrect usage of aligned data movements results in an exception when using Intel® SSE.

Use **structure of arrays (SoA)** instead of **array of structures (AoS)**. AoS is excellent for encapsulation but hinders vector processing; select appropriate data structures to make vectorization of the resulting code more effective.

*Dynamic Alignment Optimizations* improve vectorized code, especially for **long trip count loops**. Disabling them can decrease performance but may improve bitwise reproducibility of results, factoring data location out of possible discrepancy sources. Enable/disable with `/Qopt-dynamic-align[-]` (Windows) or `-q[no-]opt-dynamic-align` (Linux).

**Use Aligned Data Structures.** Alignment = a memory address' numeric address modulo of powers of two. A datum is **naturally aligned** if its address is aligned to its size, else **misaligned** (8-byte floating-point datum: naturally aligned if its address is aligned to eight). Compiler may align variables to speed memory access; misaligned accesses can incur large performance losses on processors not supporting them in hardware. Example:

```c
struct MyData{ short Data1; short Data2; short Data3; };
```

If `short` is two bytes, each member aligns to a two-byte boundary: `Data1` offset 0, `Data2` offset 2, `Data3` offset 4; size six bytes. Each member type usually has a required alignment (a pre-determined boundary) unless requested otherwise. Where the compiler took sub-optimal alignment decisions, use `__declspec(align(base,offset))` (`0 <= offset < base`, `base` a power of two) to allocate a data structure at `offset` from a certain base. Example loop consuming most execution time:

```c
double a[N], b[N];
for (i = 0; i < N; i++){ a[i+1] = b[i] * 3; }
```

If the first element of both arrays is aligned at a 16-byte boundary, after vectorization either an unaligned load from `b` or an unaligned store into `a` must be used. Peeling off an iteration does not help, but alignment can be enforced:

```c
__declspec(align(16, 8)) double a[N];
__declspec(align(16, 0)) double b[N];   /* or simply "align(16)" */
```

This gives two aligned access patterns after vectorization (assuming 8-byte doubles). With pointers compiler usually cannot determine alignment at compile time. For:

```c
void fill(char *x) { int i; for (i = 0; i < 1024; i++){ x[i] = 1; } }
```

compiler cannot assume alignment of the region; it may vectorize with unaligned data movement instructions or generate the runtime alignment optimization:

```c
peel = x & 0x0f;
if (peel != 0) { peel = 16 - peel; for (i = 0; i < peel; i++) { x[i] = 1; } } /* runtime peeling loop */
for (i = peel; i < 1024; i++) { x[i] = 1; }                                   /* aligned access */
```

Runtime optimization obtains aligned access patterns at the expense of a slight increase in code size and testing. If incoming access patterns are 16-byte aligned, avoid this overhead with `__builtin_assume_aligned`; e.g. where a block of memory with address `n2` is aligned on a 16-byte boundary, use `__builtin_assume (n2%16==0)`.

> **Caution:** incorrect use of aligned data movements results in an exception for Intel® SSE.

**Use Structure of Arrays Versus Array of Structures.** An array is a contiguous collection of data items accessed by an ordinal index; data can be organized as AoS or SoA. AoS works well for encapsulation but poorly for vector processing. AoS: a loop visiting all components of an RGB point before the next point has good locality (all fetched cache-line elements used), but each reference has a non-unit stride, hurting vector performance; a loop visiting only one component has poor locality. SoA: unit-stride references vectorize better and still have good locality within each of the three data streams; SoA may outperform AoS with a vectorizing compiler, though the difference may not be apparent early.

Before vectorizing: make data structures vector-friendly; make inner loop indices correspond to the outermost (last) array index (row-major order); use SoA over AoS — for 3-D coordinates use three separate arrays per component instead of one array of three-component structures, which also avoids dependencies between loops that would prevent vectorization.

With AoS, each iteration produces one result by computing XYZ but uses at best **75%** of the SSE unit (fourth component unused; sometimes only one component, **25%**). With SoA, each iteration produces four results (XXXX, YYYY, ZZZZ) using **100%** of the SSE unit, but the code will likely be three times as long. If the layout is AoS, consider converting to SoA before the critical loop: use the smallest data types giving the needed precision to maximize potential SIMD width (if only 16 bits are needed, `short` vs `int` can mean 8-way vs four-way SIMD parallelism); avoid mixing data types (minimizes conversions); avoid operations not supported in SIMD hardware; use all instruction sets via the appropriate command-line option or the Windows-only IDE option `Project > Properties > C/C++ > Code Generation [Intel C++] > Intel Processor-Specific Optimization` (Intel-processor-only apps) or `Project > Properties > C/C++ > Code Generation > Enable Enhanced Instruction Set` (compatible non-Intel processors).

Vectorizing compilers have built-in efficiency heuristics deciding whether vectorization is likely to improve performance; the Intel® oneAPI DPC++/C++ Compiler disables vectorization of loops with many unaligned or non-unit stride access patterns. If experimentation reveals a win, override with `#pragma vector always` before the loop; the compiler then vectorizes any loop regardless of the efficiency analysis (provided vectorization is safe).

*[Three-channel layout used in the vectorization/alignment discussion — source p. 767]*
Figure (p. 767): three adjacent vertical colour bars labelled **R**, **G**, **B**, each a separate block of memory — the interleaved per-channel layout that makes combined-access alignment/stride unprovable.

**Use Automatic Vectorization / Vectorization Speedup.** For `for (i=0;i<=MAX;i++) c[i]=a[i]+b[i];` (`a`, `b`, `c` integer arrays): without vectorization (`O1`, `-no-vec` Linux, `/Qvec-` Windows) the compiler processes the code with unused space in the SIMD registers, even though each register can hold three additional integers. With vectorization (`O2`+), the compiler may use the additional registers to perform four additions in a single instruction; opportunities are sought at default optimization (`O2`) or higher.

Linux `guided_matmul_opt_report` (source an environment script such as `setvars.sh` in `$ONEAPI_ROOT`; navigate to `oneAPI-samples/DirectProgramming/C++/CompilerInfrastructure/guided_matmul_opt_report`, which multiplies a vector by a matrix with `for (i = 0; i < size1; i++) { b[i] = 0; for (j = 0;j < size2; j++) { b[i] += a[i][j] * x[j]; } }`; default `O2` enables vectorization, so disable it explicitly):

```bash
icx -qopt-report=3 -O2 -xAVX -no-vec Driver.c Multiply.c -o NoVectMult
./NoVectMult
icx -qopt-report=3 -O2 -xAVX -vec        Driver.c Multiply.c -o VectMult
./VectMult
```

Windows (same sample):

```bat
icx-cl /Qopt-report=3 /O2 /QxAVX /Qvec-      Driver.c Multiply.c -o NoVectMult && NoVectMult.exe
icx-cl icx-cl /Qopt-report=3 /O2 /QxAVX /Qvec Driver.c Multiply.c -o VectMult && VectMult.exe
```

The duplicated `icx-cl icx-cl` above is verbatim in the source. The vectorized version may run faster; the non-vectorized time is only slightly faster than compiling with `O1`.

**Obstacles to Vectorization.** These issues do not always prevent vectorization, but frequently cause compiler to decide it would not be worthwhile.

*Non-contiguous memory access:* four consecutive integers/floats, or two consecutive doubles, may be loaded in one SSE instruction; non-adjacent integers must be loaded separately with multiple instructions (much less efficient). Most common: loops with non-unit stride or indirect addressing; the compiler rarely vectorizes these unless computational work exceeds the access overhead:

```c
// arrays accessed with stride 2
for (int i=0; i<SIZE; i+=2) b[i] += a[i] * x[i];
// inner loop accesses a with stride SIZE
for (int j=0; j<SIZE; j++) {
  for (int i=0; i<SIZE; I++)   b[i] += a[i][j] * x[j];
}
// indirect addressing of x using index array
  for (int i=0; i<SIZE; i+=2) b[i] += a[i] * x[index[i]];
```

Typical vectorization report message: `vectorization possible but seems inefficient`; indirect addressing may also report `existence of vector dependence`.

*Data dependencies:* vectorization changes the order of operations within a loop (each SIMD instruction operates on several elements at once), so it is possible only if the change does not change the results. Simplest case: written elements do not appear in any other iteration — all iterations independent, executable in any order, safe for any parallel method. **Read-after-write** (flow) dependency:

```c
A[0]=0;
for (j=1; j<MAX; j++) A[j]=A[j-1]+1;
// equivalent to: A[1]=A[0]+1; A[2]=A[1]+1; A[3]=A[2]+1; A[4]=A[3]+1;
```

The value of `j` propagates to all `A[j]`. This cannot safely be vectorized: if the first two iterations run simultaneously in a SIMD instruction, the second uses `A[1]` before the first computes it. **Write-after-read** (anti-dependency):

```c
for (j=1; j<MAX; j++) A[j-1]=A[j]+1;
// equivalent to: A[0]=A[1]+1; A[1]=A[2]+1; A[2]=A[3]+1; A[3]=A[4]+1;
```

This is not safe for general parallel execution (the write iteration may execute before the read iteration); but, no higher-`j` iteration can complete before a lower-`j` one, so vectorization is safe (it gives the same result as non-vectorized code). The following example may not be safe, since vectorization might overwrite some `A` elements in the first SIMD instruction before the second uses them:

```c
for (j=1; j<MAX; j++) { A[j-1]=A[j]+1; }
```

Read-after-read is not a real dependency and does not prevent vectorization or parallel execution (an unwritten variable can be read any number of times). Write-after-write (output) dependencies, where the same variable is written in more than one iteration, are generally unsafe for parallel execution including vectorization. One important exception containing all the above dependency types:

```c
sum=0;
for (j=1; j<MAX; j++) sum = sum + A[j]*B[j]
```

Although `sum` is read and written every iteration, compiler recognizes reduction idioms and vectorizes them safely. These inter-iteration dependencies are **loop-carried dependencies**. The above are proven dependencies; compiler cannot safely vectorize if even a **potential** dependency exists, e.g. `for (i = 0; i < size; i++) { c[i] = a[i] * b[i]; }` — it must determine whether for some iteration `i`, `c[i]` might refer to the same memory location as `a[i]` or `b[i]` for a different iteration (such locations are **aliased**); e.g. `a[i]` pointing to `c[i-1]` is a read-after-write dependency. If compiler cannot exclude this, it will not vectorize unless given hints.

**Help the Compiler Vectorize.** Ways to provide additional information:

*Pragmas — `#pragma ivdep`:* tells compiler it may safely ignore any **potential** data dependencies (it will not ignore proven dependencies); using it when dependencies exist may lead to incorrect results. There are cases where compiler cannot tell by static dependency analysis that it is safe to vectorize:

```c
void copy(char *cp_a, char *cp_b, int n) {
  for (int i = 0; i < n; i++) { cp_a[i] = cp_b[i]; }
}
```

Without details a vectorizing compiler must conservatively assume the memory regions accessed by `cp_a` and `cp_b` may (partially) overlap, causing potential data dependencies that prohibit straightforward conversion of the loop to SIMD. Compiler may keep the loop serial or generate a runtime test for overlap, where the true-branch loop can be converted to SIMD:

```c
if (cp_a + n < cp_b || cp_b + n < cp_a)   /* vector loop */
  for (int i = 0; i < n; i++) cp_a[i] = cp_b [I];
else                                      /* serial loop */
  for (int i = 0; i < n; i++) cp_a[i] = cp_b[i];
```

Runtime data-dependency testing exploits implicit parallelism in C/C++ code at the expense of a slight increase in code size and testing overhead. If the function is mainly used for small values of `n` or overlapping memory regions, prevent vectorization and the runtime overhead with `#pragma novector` before the loop. If the loop is guaranteed to operate on non-overlapping regions, `#pragma ivdep` tells compiler that conservatively assumed dependencies can be ignored, giving vectorization without runtime data-dependency testing:

```c
#pragma ivdep
void copy(char *cp_a, char *cp_b, int n) { for (int i = 0; i < n; i++) { cp_a[i] = cp_b[i]; } }
```

NOTE: you can also use the `restrict` keyword. Other hints: `#pragma loop count (n)` gives the typical trip count, helping compiler decide whether vectorization is worthwhile or generate alternative code paths; `#pragma vector always` asks compiler to vectorize the loop; `#pragma vector align` asserts the following loop's data is aligned (16-byte boundary for Intel® SSE); `#pragma novector` asks compiler not to vectorize a particular loop.

*Keywords:* `restrict` asserts that the memory referenced by a pointer is not aliased; requires the `[Q]std=c99` option. Declaring `cp_a`/`cp_b` with `restrict` says each pointer provides exclusive access to a memory region: the argument-list qualifier tells compiler there are no other aliases to that memory, and the pointer is the only means of accessing it in that scope. Even if the code vectorizes without `restrict`, compiler checks for aliasing at runtime **if** `restrict` was used:

```c
void copy(char * __restrict cp_a, char * __restrict cp_b, int n) {
  for (int i = 0; i < n; i++) cp_a[i] = cp_b[i];
}
```

Best used when the exclusive-access property holds for pointers used with many loops, because it avoids annotating each vectorizable loop individually. Both `#pragma ivdep` and `restrict` must be used with care: incorrect usage may change the intended semantics. Another loop that may not vectorize from potential aliasing among pointers `a`, `b`, `c`:

```c
void add(float *a, float *b, float *c) {
  for (int i=0; i<SIZE; i++) { c[i] += a[i] + b[i]; }
}
// let the compiler know the pointers are safe with restrict
void add(float * __restrict a, float * __restrict b, float * __restrict c) {
  for (int i=0; i<SIZE; i++) { c[i] += a[i] + b[i]; }
}
```

Down-side of `restrict`: not all compilers support this keyword, so source code may lose portability.

*Options/switches:* **IPO** — `[Q]ipo` across source files supplies additional information (trip counts, alignment, or data dependencies) about a loop and may allow inlining of function calls. **HLO** — `O3` enables additional loop optimizations that make it easier to vectorize the transformed loops.

**Vectorization and Loops.** In rare cases, a successful loop parallelization (automatic or by OpenMP directives) may affect compiler's reported messages for a non-vectorizable loop in a non-intuitive way.

*Loop Constructs* are formed with the usual `for` and `while` constructs and must have a **single entry and a single exit** to be vectorized:

```c
void vec(float a[], float b[], float c[]) {   // vectorizable: if branch inside body
  int i = 0;
  while (i < 100) { a[i] = b[i] * c[i];
    if (a[i] < 0.0) a[i] = 0.0; i++; }
}
void no_vec(float a[], float b[], float c[]) { // non-vectorizable: break is a 2nd exit
  int i = 0;
  while (i < 100) { if (a[i] < 50) break; ++i; }
}
```

*Types of Vectorized Loops.* For integer loops, 128-bit Intel® SSE and Intel® AVX provide SIMD instructions for most arithmetic/logical operators on 32-, 16-, and 8-bit integers, with limited 64-bit support. Vectorization may proceed if the final precision of integer wrap-around arithmetic is preserved (e.g. a 32-bit shift-right is not vectorized in 16-bit mode if the final stored value is a 16-bit integer). Because Intel® SSE and Intel® AVX are not fully orthogonal (shifts on byte operands are unsupported), not all integer operations vectorize. For 32-bit single- and 64-bit double-precision floating point, Intel® SSE provides SIMD instructions for `+`, `-`, `*`, `/`, plus binary MIN and MAX and unary SQRT; other math operators (SIN, COS, TAN) are supported in software by a vector mathematical runtime library provided with compiler.

*To be vectorizable, loops must be:* **countable** — trip count known at entry at runtime, though not necessarily at compile time (the count may be a variable but must remain constant for the loop's duration), so exit must not be data-dependent; **single entry and single exit** (implied by countable); **straight-line code** — SIMD instructions perform the same operation on elements from multiple iterations, so iterations cannot have different control flow or branch; `switch` is not allowed, while `if` is allowed if implementable as masked assignments (usually the case) — the calculation is done for all elements but the result stored only where the mask is true; the **innermost loop of a nest** (only exception: an original outer loop transformed into an inner loop by a prior optimization phase such as unrolling, loop collapsing or interchange, or an original outermost loop transformed to innermost by loop materialization); **without function calls** — even a print statement prevents vectorization (report message typically `non-standard loop is not a vectorization candidate`); the two major exceptions are intrinsic math functions and functions that may be inlined.

Intrinsic math functions are allowed because compiler runtime library contains vectorized versions; most exist in both float and double versions: `acos`, `acosh`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `cbrt`, `ceil`, `cos`, `cosh`, `erf`, `erfc`, `erfinv`, `exp`, `exp2`, `fabs`, `floor`, `fmax`, `fmin`, `log`, `log2`, `log10`, `pow`, `round`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`.

*Statements in the Loop Body.* Vectorizable operations differ for floating-point and integer data. **Integer array operations:** statements may contain `char`, `unsigned char`, `short`, `unsigned short`, `int`, `unsigned int`; calls to `sqrt` and `fabs` are supported; arithmetic is limited to addition, subtraction, bitwise AND/OR/XOR, division (via runtime library call), multiplication, `min`, `max`; data types may be mixed (multiplication, shift, unary operators) but that may lower efficiency. **Other operations:** nothing besides the preceding floating-point and integer operations; `__m64`, `__m128`, `__m256` are **not** vectorizable; the loop body cannot contain function calls; Intel® SSE intrinsics (`_mm_add_ps`) or Intel® AVX intrinsics (`_mm256_add_ps`) are not allowed.

*Loop Exit Conditions.* Loop iterations must be countable and the number of iterations expressed as a constant, a loop invariant expression, or a linear function of outermost loop indices; where a loop's exit depends on computation, the loops are not countable. Countable loop, example one (exit condition "N-1b+1"; `lb` is not affected within the loop):

```c
void cnt1(float a[], float b[], float c[], int n, int lb) {
  int cnt=n, i=0;
  while (cnt >= lb) { a[i] = b[i] * c[i]; cnt--; i++; }
}
```

Countable loop, example two (if branch inside body):

```c
void vec(float a[], float b[], float c[]) {
  int i = 0;
  while (i < 100) { a[i] = b[i] * c[i];
    if (a[i] < 0.0) a[i] = 0.0; i++; }
}
```

Non-countable loop (iterations dependent on `a[i]`):

```c
void no_cnt(float a[], float b[], float c[]) {
  int i=0;
  while (a[i]>0.0) { a[i] = b[i] * c[i]; i++; }
}
```

*Strip-Mining and Cleanup.* Strip-mining (loop sectioning) is a loop transformation technique for enabling SIMD-encoding and improving memory performance. Fragmenting a large loop into strips transforms it in two ways: increasing temporal and spatial locality in the data cache if the data is reusable in different passes; and reducing loop iterations by a factor of the vector length (operations per SIMD operation). With Intel® SSE the vector/strip length is reduced four times: four floating-point data items per single-precision SIMD operation. Compiler automatically strip-mines your loop and generates a **cleanup loop**.

Before: `i=0; while(i<n) { a[i]=b[i]+c[i]; ++i; }`. After vectorization compiler generates two loops:

```c
i=0;
while(i<(n-n%4)) { a[i:i+3]=b[i:i+3]+c[i:i+3]; i=i+4; } // vector strip-mined; [i:i+3] = SIMD
while(i<n) { a[i]=b[i]+c[i]; ++i; }                     // scalar clean-up loop
```

*Loop Blocking* (treat as strip-mining in two or more dimensions) eliminates as many cache misses as possible by transforming the memory domain into smaller chunks rather than sequentially traversing it, each chunk small enough to fit all data for a computation in cache; blocking arrays `A` and `B` into rectangular chunks whose combined size is smaller than the cache improves reuse (you might have to increase cache size to benefit).

Original loop (`#define MAX 7000`, `A`/`B` initialized to `j`, timed via `time`/`printf`): `void add(int a[][MAX], int b[][MAX]) { int i,j; for(i=0;i<MAX;i++) for(j=0;j<MAX;j++) a[i][j] = a[i][j] + b[j][i]; }`. Transformed loop after blocking:

```c
#define BS 8
void add(int a[][MAX], int b[][MAX]) {
  int i, j, ii, jj;
  for(i=0;i<MAX;i+=BS)
   for(j=0; j<MAX;j+=BS)
     for(ii=i; ii<i+BS; ii++)            // outer loop
        for(jj=j;jj<j+BS; jj++)          // Array B experiences one cache miss
           a[ii][jj] = a[ii][jj] + b[jj][ii];  // for every iteration of outer loop
}
```

*Loop Interchange and Subscripts with Matrix Multiply.* Loop interchange transforms a loop nest to improve memory access patterns. In typical matrix multiplication `B(K,J)` is not stride-1 and will not vectorize efficiently; with loops interchanged all references become stride-1.
void matmul_slow(float *a[], float *b[], float *c[]) {   // typical
  int i, j, k, N = 100;
  for (i=0;i<N;i++) for (j=0;j<N;j++) for (k=0;k<N;k++)
        c[i][j]=c[i][j]+a[i][k]*b[k][j];
}
void matmul_fast(float *a[], float *b[], float *c[]) {   // stride -1 after interchange
  int i, j, k, N = 100;
  for (i=0;i<N;i++) for (k=0;k<N;k++) for (j=0;j<N;j++)
        c[i][j]=c[i][j]+a[i][k]*b[k][j];
}
```c

Interchanging loops is not always possible because of dependencies between the loops, which can lead to different results.

## Explicit Vector Programming

**User-Mandated or SIMD Vectorization.** A single-instruction-multiple-data (SIMD) feature supplementing automatic vectorization just as OpenMP* parallelization supplements automatic parallelization. Available for Intel® and non-Intel microprocessors; vectorization may call library routines giving additional gain on Intel microprocessors. Programs resemble those using auto-vectorization hints, and it minimizes the code changes needed for vectorized code. It uses **`#pragma omp simd`** to effect loop vectorization.

*[SIMD and auto-vectorization as supplements to OpenMP* and auto-parallelization — source p. 781]*

Figure (p. 781): left box stacks `SIMD` over `Auto-vectorization` with a vertical double-headed arrow; right box stacks `OpenMP*` over `Auto-parallelization` the same way; horizontal double-headed arrows connect `SIMD`↔`OpenMP*` and `Auto-vectorization`↔`Auto-parallelization`. Meaning: user-mandated/SIMD vectorization supplements automatic vectorization exactly as OpenMP* parallelization supplements automatic parallelization.

*[Six-level vectorization control stack from most programmer control to most ease of use — source p. 781]*

Figure (p. 781): six levels ordered by a vertical double-headed arrow "Programming control ↑ / Ease of use ↓", most control at top: 1. `ASM code (addps)`, 2. `Vector Intrinsics (mm_add_ps())`, 3. `SIMD Intrinsic Class (F32vec4 add)`, 4. `SIMD Vectorization (#pragma simd)`, 5. `Auto-vectorization Hints (#pragma ivdep)`, 6. `Automatic Vectorization`.

For a C++ function `add_floats()` that uses too many unknown pointers for the automatic runtime independence check, you can assert independence with `#pragma ivdep` and let the compiler decide, or enforce vectorization with `#pragma omp simd`. Difference: `#pragma omp simd` makes the compiler generate a **warning** if it cannot vectorize; with auto-vectorization hints actual vectorization is still at the compiler's discretion, even with `#pragma vector always`. `#pragma omp simd` allows optional clauses so the compiler obtains enough information to generate correct vector code.
Figure (p. 781): left box stacks `SIMD` over `Auto-vectorization` with a vertical double-headed arrow; right box stacks `OpenMP*` over `Auto-parallelization` the same way; horizontal double-headed arrows connect `SIMD`↔`OpenMP*` and `Auto-vectorization`↔`Auto-parallelization`. Meaning: user-mandated/SIMD vectorization supplements automatic vectorization exactly as OpenMP* parallelization supplements automatic parallelization.
*Additional Semantics.* A variable may belong to zero or one of `private`, `linear`, `reduction`. Within the vector loop an expression is a **vector value** if it is `private`, `linear`, `reduction`, or has a sub-expression evaluated to a vector value; otherwise it is a **scalar value** (broadcast to all iterations — not necessarily loop invariant, though that is the most frequent usage pattern). A vector value may not be assigned to a scalar L-value (error); a scalar L-value may not be assigned under a vector condition (error); `switch` is not supported. NOTE: describing vector semantics with the SIMD pragma can be difficult for some auto-vectorizable loops, e.g. MIN/MAX reduction in C since the language has no MIN/MAX operators.

*Restrictions on a `#pragma omp declare simd` Declaration.* Not allowed: thread creation/joining via OpenMP `parallel`/`for`/`sections`/`task`/`target`/`teams` and explicit threading API calls; locks, barriers, atomic construct, critical sections; inline ASM, VM, Vector Intrinsics (e.g. SVML); `setjmp`, `longjmp`, SHE and computed GOTO; EH (all vector functions are **noexcept**); `switch` (sometimes converted to `if`, not reliably); `exit()`/`abort()`. Non-vector calls are generally allowed inside vector functions but are **serialized lane-by-lane** and may perform poorly; SIMD-enabled functions may not have side effects except writes by their arguments, which non-vector calls can violate. Formal parameters: (un)signed 8-, 16-, 32-, or 64-bit integer; 32- or 64-bit floating point; 64- or 128-bit complex; a pointer (a C++ reference counts as a pointer data type).

**SIMD-Enabled Functions** (formerly **elemental functions**) express a data-parallel algorithm: written like a regular C/C++ function describing the operation on one element with scalar syntax, callable normally on one element or in a data-parallel context on many. Compiler generates **short vector variants** performing the operation on multiple arguments in one invocation, possibly as fast as the scalar implementation does one, using the CPU vector ISA; a call inside a SIMD loop or another SIMD-enabled function is replaced with the best-fit short-vector variant. Invoked from a pragma `omp` construct, copies may be assigned to different threads (workers) running concurrently (multi-core and vector-ISA parallelism); a short vector function inside a parallel loop achieves vector-level and thread-level parallelism.

*Declare a SIMD-Enabled Function.* Linux: `__attribute__((vector (clauses))) return_typesimd_enabled_function_name(parameters)`; alternatively OpenMP `#pragma omp declare simd clauses`, requiring the `[q or Q]openmp` or `[q or Q]openmp-simd` option. Windows uses clauses in the vector declaration to override defaults. Clauses at the definition declare one or several short vector variants, and multiple vector declarations with different clause sets may attach to one function. The clauses are defined as follows:

| Clause | Definition |
|---|---|
| `processor(cpuid)` | Generate a vector variant using the instructions, caller/callee interface, and default vector-length selection scheme suitable to the specified processor. Highly recommended, especially for wider vector registers (e.g. `core_2nd_gen_avx` and newer). `cpuid` is one of: `core_4th_gen_avx_tsx`, `core_4th_gen_avx`, `core_3rd_gen_avx`, `core_2nd_gen_avx`, `core_aes_pclmulqdq`, `core_i7_sse4_2`, `atom`, `core_2_duo_sse4_1`, `core_2_duo_ssse3`, `pentium_4_sse3`, `pentium_m`, `pentium_4`, `haswell`, `broadwell`, `skylake`, `skylake_avx512`. |
| `vectorlength(n)` / `simdlen(n)` (for `omp declare simd`) | `n` is a vector length, a **power of 2 no greater than 32**. `simdlen` says each routine invocation at the call site should execute the computation equivalent to `n` scalar executions. When omitted compiler selects the length automatically from the return value, parameters, and/or `processor` clause. When multiple variants are called from one vectorization context (e.g. two functions in one vector loop), identical explicit `simdlen` values are advised for good performance. |
| `processor(cpuid)` | Generate a vector variant using the instructions, caller/callee interface, and default vector-length selection scheme suitable to the specified processor. Highly recommended, especially for wider vector registers (e.g. `core_2nd_gen_avx` and newer). `cpuid` is one of: `core_4th_gen_avx_tsx`, `core_4th_gen_avx`, `core_3rd_gen_avx`, `core_2nd_gen_avx`, `core_aes_pclmulqdq`, `core_i7_sse4_2`, `atom`, `core_2_duo_sse4_1`, `core_2_duo_ssse3`, `pentium_4_sse3`, `pentium_m`, `pentium_4`, `haswell`, `broadwell`, `skylake`, `skylake_avx512`. |
| `uniform(param [, param,]…)` | `param` is a formal parameter or `this`. Values of the specified arguments can be broadcast to all iterations as a performance optimization; often useful for more favorable vector memory references. Acknowledging it may allow broadcasts to be hoisted out of the caller loop — evaluate performance implications. Multiple `uniform` clauses merge as a union. |
| `vectorlength(n)` / `simdlen(n)` (for `omp declare simd`) | `n` = vector length, a **power of 2 no greater than 32**. `simdlen`: each routine invocation at the call site executes the computation equivalent to `n` scalar executions. Omitted: the compiler selects the length automatically from the return value, parameters, and/or `processor` clause. When multiple variants are called from one vectorization context (e.g. two functions in one vector loop), identical explicit `simdlen` values are advised for performance. |
| `inbranch` / `notinbranch` | Used with `#pragma omp declare simd`; `inbranch` = `mask`, `notinbranch` = `nomask`. |

Write the code inside your function using existing C/C++ syntax and relevant built-in functions.

*Usage of Vector Function Specifications:* several variants may be defined per routine; on a call the compiler matches variants with actual parameter kinds by priority — if an actual parameter is loop invariant and `uniform` was specified for the corresponding formal parameter, the uniform variant has higher priority. Linear specifications rank high→low: **`linear(uval())`, `linear()`, `linear(val())`, `linear(ref())`**.

```
```c
#pragma omp declare simd                           // universal but slowest; matches all three loops
#pragma omp declare simd linear(in1) linear(ref(in2)) uniform(mul) // matches first loop
#pragma omp declare simd linear(ref(in2))                            // matches second and third loops
#pragma omp declare simd linear(ref(in2)) linear(mul)              // matches second loop
#pragma omp declare simd linear(val(in2:2))                          // matches third loop
extern int func(int* in1, int& in2, int mul);
for (int i = 0; i < nn; i++) c[i] = func(a + i, *(b + i), mul);
    // param1 linear; param2 ref linear; param3 unchanged
for (int i = 0; i < nn; i++) c[i] = func(&a[ndx[i]], b[i], i + 1);
    // param1 unpredictable; param2 ref linear; param3 linear
#pragma omp simd
for (int i = 0; i < nn; i++) { int k = i * 2;  // private vars become arrays: k->k_vec[vector_length]
    c[i] = func(&a[ndx[i]], k, b[i]);
    // param1 unpredictable; param2 ref+val linear; param3 unpredictable
    // (the #pragma simd linear(val(in2:2)) is chosen from the two matching variants)
}
```

*SIMD-Enabled Functions and C++.* **Exception handling:** exceptions are currently not supported in SIMD contexts — they cannot be thrown and/or caught in SIMD loops and SIMD-enabled functions, so all are considered `noexcept` in C++11 terms; this affects short vector variants and the original scalar routine, is enforced at function compilation (checked against `throw` and calls throwing exceptions) and when the SIMD-enabled function call is compiled. **Dynamic polymorphism:** vector specifications are not supported for virtual functions (yet). **C++ type system:** vector attributes are C++11 attributes, not part of a functional type, bound to the function itself. Consequences: template instantiations with SIMD-enabled functions as template parameters won't catch vector attributes, so they cannot be preserved in wrapper templates like `std::bind` that add indirection (the indirection may be optimized away, and the resulting direct call will have all vector attributes); no overloading or template specialization by vector attributes; no functional traits capturing those attributes for template metaprogramming.

```cpp
template <int f(int)>  // Function value template - captures exact function, not a function type
int caller1(int x[100]) { int res = 0;
#pragma omp simd reduction(+:res)
  for (int i = 0; i < 100; i++) { res += f(x[i]); }   // exact function put here upon instantiation
  return res; }
template <typename F>  // Generic functional type template - captures object type for functors or entire
                       // functional type for functions; vector attributes currently are not captured
int caller2(F f, int x[100]) { int res = 0;
#pragma omp simd reduction(+:res)
  for (int i = 0; i < 100; i++) { res += f(x[i]); } // calls matching f indirectly / f.operator() directly
  return res; }
template <typename RET, typename ARG>  // Type-decomposing template; vector attributes would be lost
int caller3(RET (*f)(ARG), int x[100]) { int res = 0;
#pragma omp simd reduction(+:res)
  for (int i = 0; i < 100; i++) { res += f(x[i]); } // calls matching function f indirectly
  return res; }
#pragma omp declare simd
int function(int x);                     // SIMD-enabled function
int nv_function(int x);                  // Regular scalar function
struct functor {                         // Functor class with
#pragma omp declare simd                 // SIMD-enabled operator()
   int operator()(int x); };
int arr[100];
int main() {
   int res;
#pragma noinline
   res  = caller1<function>(arr);     // instantiated for function(); calls short vector variant
#pragma noinline
   res += caller1<nv_function>(arr);  // separately instantiated for nv_function()
#pragma noinline
   res += caller2(function, arr);     // instantiated for int(*)(int); calls scalar function() indirectly
#pragma noinline
   res += caller2(nv_function, arr);  // same instantiation as above, on nv_function
#pragma noinline
   res += caller2(functor(), arr);    // instantiated for functor; calls short vector variant of operator()
#pragma noinline
   res += caller3(function, arr);     // instantiated for <int, int>; calls scalar function() indirectly
#pragma noinline
   res += caller3(nv_function, arr);  // same instantiation as above, on nv_function
   return res; }
```

NOTE: if `caller1`, `caller2`, `caller3` are inlined, the compiler can replace indirect calls with direct calls in all cases — then `caller2(function, arr)` and `caller3(function, arr)` both call short vector variants, as a result of the usual replacement of direct calls by matching short vector variants in the SIMD loop. *Invoke with parallel context:* typically the invocation provides arrays wherever scalar arguments are formal parameters; NOTE that array notation syntax, and calling the SIMD-enabled function from a regular `for` loop, invokes the short vector function each iteration using vector parallelism but in a **serial loop without using multiple cores**. *Limitations:* not allowed within SIMD-enabled functions — `setjmp`/`longjump` calls, exception handling constructs, and any OpenMP construct **except atomic and simd** (see the OpenMP standard).

**SIMD-Enabled Function Pointers.** Without special effort the vector nature of a function is lost through a pointer (it points to the scalar function; short vector variants cannot be called). These pointers support indirect calls to vector variants: a special pointer, **incompatible with a regular function pointer**, referring to an entire set of short vector variants plus the scalar function. The incompatibility risks misuse especially in C++, so support is **disabled by default**. Indirect calls are handled like direct ones, but the available variant set must be associated with the **function pointer variable**, not the target function, because actual targets are unknown at the indirect call; thus all SIMD-enabled functions referenced by such a pointer should have a variant set matching the pointer's. *Declare:* Linux `__attribute__((vector (clauses))) return_type (*function_pointer_name) (parameters)`; alternatively OpenMP `#pragma omp declare simd` (requires `[q or Q]openmp` or `[q or Q]openmp-simd`); Windows `__declspec(vector (clauses)) return_type (*function_pointer_name) (parameters)`. Clauses are those of SIMD-enabled functions. Several vector attributes may attach to one pointer, reflecting all target-function variants; on an indirect call the compiler matches declared variants with actual parameter kinds exactly as for direct calls (the declaration example mirrors the direct-call example with `int (*func)(int* in1, int& in2, int mul);`, the same three loops, and the third loop's comment ending `(the __declspec(vector(linear(val(in2:2)))) will be chosen from the two matching variants)`). Before use, the pointer must be assigned the address of a function or another function pointer; vector function pointers must be compatible at assignment and initialization.

*Vector Function Pointer Compatibility:* (1) assigning a function address — the function must be compatible with the pointer in the usual C/C++ sense, SIMD-enabled, and its variant set a **superset** of those declared for the pointer (includes initializations and passing addresses as parameters); (2) assigning another function pointer — the source must be compatible with the destination in the general C/C++ sense, SIMD-enabled, and its variant set **exactly the same** as the destination's (includes initializations and passing pointers as parameters); (3) assigning a SIMD-enabled function address to a regular function pointer gives the address of a **scalar** function — vector variants cannot be called through it, and it cannot be reinterpreted as or converted into a SIMD-enabled function pointer as in rule 2; (4) assigning a SIMD-enabled function pointer to a regular function pointer matching in the C/C++ sense implicitly dynamically casts the RHS by extracting a scalar function address — vector variants cannot be called and it cannot be reinterpreted/converted as in rule 2. NOTE: SIMD-enabled and regular function pointers are **binary-incompatible** and handled differently; mixing them may give severe unpredictable results. The compiler checks compatibility where C/C++ standards allow, but sometimes cannot (e.g. passing function pointers to undeclared functions or as variable arguments) — refrain there. A SIMD-enabled function pointer may be assigned to a scalar function pointer with a cast as in rule 4, but a SIMD-enabled function pointer cannot refer to a scalar function pointer.

```c
#pragma omp declare simd
int (*ptr1)(int*, int);
#pragma omp declare simd
int (*ptr1a)(int*, int);
#pragma omp declare simd
#pragma omp declare simd linear(a)
typedef int (*fptr_t2)(int* a, int b);
typedef int (*fptr_t3)(int*, int);
fptr_t2 ptr2, ptr2a;  fptr_t3 ptr3;
#pragma omp declare simd
int func1(int* x, int b);
#pragma omp declare simd
#pragma omp declare simd linear(x)
int func2(int* x, int b);
#pragma omp declare simd
#pragma omp declare simd linear(x)
int func3(float* x, int b);
// allowed assignments
ptr1  = func1;   // same prototype and vector spec
ptr2  = func2;   // same prototype and vector spec
ptr1a = ptr1;    // same prototype and vector spec
ptr1a = func2;   // vector spec on function includes all vector spec on pointer
ptr3  = func1;   // scalar pointer with same prototype - use scalar func1
ptr3  = func2;   // scalar pointer with same prototype - use scalar func2
ptr3  = ptr1;    // scalar pointer same prototype - implicit conversion from vector to scalar pointer
ptr3  = ptr2;    // scalar pointer same prototype - implicit conversion from vector to scalar pointer
// disallowed assignments
ptr2 = func1;    // vector spec on function does not have all specs on pointer
ptr2 = func3;    // prototype mismatch although vector spec matched
ptr1 = func3;    // prototype mismatch although vector spec matched
ptr3 = func3;    // prototype mismatch
ptr1 = ptr2;     // pointers should have the same vector spec
ptr2 = ptr3;     // pointers should have the same vector spec
```

*Call Sequence.* An indirect call's target depends on the pointer's dynamic content; in a loop, targets may differ across iterations of a vectorized loop or lanes of a SIMD-enabled function executing the call, so a vectorized indirect call may involve multiple calls to different targets within one SIMD chunk: (1) if the vector function pointer is **uniform** (see the OpenMP specification) or provably uniform, one indirect call to a matched vector variant accessible by the pointer; (2) if not known uniform at compile time, all pointer values in a SIMD chunk may still be the same — checked at runtime, then a single indirect call to a matched vector variant; (3) otherwise lanes sharing the same call target are masked-in and a masked vector variant matching the matched one is invoked for each unique call target — if the masked variant is not provided and the pointer is not proven uniform, the match is rejected and the compiler may **serialize the call** (generate several scalar calls).

```c
#pragma omp declare simd
typedef int (*fptr_t1)(int*, int);
#pragma omp declare simd
int func1(int* x, int b);
fptr_t1 *fptr_array;    // array of vector function pointers
void foo(int N, int *x, int y){
  fptr_t1 ptr1 = func1;
#pragma omp simd
  for (int i = 0; i < N; i++) {
    ptr1(x+i, y);                     // ptr1 is uniform by OpenMP rule.
    fptr_t1 ptr1a = ptr1;
    ptr1a(x+i, y);                    // compiler can prove ptr1a is uniform.
    fptr_t1 ptr1b = fptr_array[i];
    ptr1b(x+i,y);                     // ptr1b may or may not be uniform.
  }
}
```

*SIMD-Enabled Function Pointers and the C++ Type System.* Vector specifications are C++11 attributes, **not part of a pointer type**, though they make the pointer binary-incompatible with another pointer of the same type without the attribute; they are bound to the variable or function argument (an instance of a pointer type) itself, and the pointer's type is unchanged. Consequences: vector attributes on a function argument are not reflected in C++ name mangling, so functions differing only in the vector attributes of a function pointer argument (or lack thereof) have the same name and are treated the same by the C++ linker — this may pass a parameter of incorrect vectorness, sometimes undetectably, so **distinctly name functions having SIMD-enabled function pointers as parameters**. Incorrect interpretation of function pointers is extremely dangerous (may execute unwanted code or non-code), so the compiler issues `Warning #3757: this use of a vector function type is not fully supported` if a vector function pointer is used as a C++ function parameter (ignore only if no ambiguity is possible, e.g. the accepting function has a distinct name and is fully declared before all uses). Templates with SIMD-enabled pointer types as template parameters won't catch vector attributes: the template instantiates with the non-SIMD-enabled pointer type, so all variables, class members, and function arguments bound to that type are regular function pointers; using such templates with a SIMD-enabled function pointer as template function parameter, template class method parameter, or RHS of a template class member assignment causes a dynamic cast to the non-SIMD-enabled pointer and loss of vectorness; no overloading or specialization by the vector attributes of a functional pointer; no functional traits capturing those attributes for template metaprogramming.

```cpp
typedef int (*fptr_t)(int*, int);
#pragma omp declare simd
typedef int (*fptr_t1)(int*, int);
#pragma omp declare simd
#pragma omp declare simd linear(x)
typedef int (*fptr_t2)(int* a, int b);
fptr_t ptr;  fptr_t1 ptr1;  fptr_t2 ptr2;
// function prototypes that only differ in SIMD-enabled function decoration
// All these will have identical mangled names.
void foo(fptr_t);  void foo(fptr_t1);  void foo(fptr_t2);
// template instantiation
template <typename T> void bar(T);
…
  bar(fptr);   // bar<fptr_t>
  bar(fptr1);  // bar<fptr_t>
  bar(fptr2);  // bar<fptr_t>
```

*Indirect Invocation with Parallel Context* (gives instruction-level parallelism via special vector instructions):

```c
#pragma omp declare simd
float (**vf_ptr)(float, float);
a[:] = vf_ptr[:] (b[:],c[:]);                 // operates on the whole extent of arrays a, b, c
a[0:n:s] = vf_ptr[0:n:s] (b[0:n:s],c[0:n:s]); // full array notation: n as extent, s as stride
```

NOTE: array notation syntax, and calling the function from a regular `for` loop, invokes the short vector variant each iteration using vector parallelism, but in a serial loop without utilizing multiple cores.

**Function Annotations and the SIMD Directive for Vectorization.** `__declspec(align(n))` overcomes hardware alignment constraints; auto-vectorization hints address stylistic issues from lexical scope, data dependency, and ambiguity resolution; the SIMD pragma enforces loop vectorization. `__declspec(vector)`/`__attribute__(vector)` and `__declspec(vector[clauses])`/`__attribute__(vector(clauses))` vectorize user-defined functions and loops (for SIMD usage the vector function is called from a loop being vectorized); `__declspec(vector_variant(clauses))`/`__attribute__(vector_variant(clauses))` provide a user-defined vector implementation. The C/C++ extensions for **Array Notations** map operations provide general data-parallel semantics without expressing the implementation strategy, so the same operation is written regardless of problem size; the implementation combines SIMD, loops, and tasking, and a single-dimensional operation can be expressed at two levels using both task constructs and array operations to force a preferred parallel and vector execution. The vector declaration usage model takes a small section of code generated for the function `vectorlength` of the array and exploits SIMD parallelism; task parallelism is implemented at the call site.

| Language Feature | Description |
|---|---|
| `__declspec(align(n))` | Align the variable to an n-byte boundary. Address mod n=0. |
| `__declspec(align(n,off))` | Align to an n-byte boundary with offset `off` within each n-byte boundary. Address mod n=off. |
| `__declspec(vector)` (Windows) / `__attribute__(vector)` (Linux) | Combines with the map operation at the call site to provide data parallel semantics. When multiple instances of the vector declaration are invoked in a parallel context, execution order among them is not sequenced. |
| `__declspec(vector[clauses])` (Windows) / `__attribute__(vector(clauses))` (Linux) | Same, with clauses: `linear` — `linear(param1:step1 [, param2:step2]…)`; `mask` — `[no]mask`; `processor` — `processor(cpuid)`; `uniform` — `uniform(param [, param,]…)`; vector length — `vectorlength(n)`. Multiple instances in a parallel context are not sequenced. |
| `__declspec(vector_variant(clauses))` (Windows) / `__attribute__(vector_variant(clauses))` (Linux) | Vectorize user-defined functions and loops. Clauses: `implements` (required) — `implements (function declarator) [, simd-clauses]`; `simd-clauses` (optional) — one or more clauses allowed for the `vector` attribute. |
| `__builtin_assume_aligned (a,n)` | Assume array `a` is aligned on an n-byte boundary; used where compiler failed to obtain alignment information. |
| `__builtin_assume (cond)` | Assume the represented condition is true where the keyword appears; typically conveys properties such as alignment information. |

**Auto-vectorization hints:** `#pragma ivdep` (ignore assumed vector dependencies); `#pragma vector {aligned|unaligned|always|temporal|nontemporal}` (specify how to vectorize and indicate efficiency heuristics should be ignored — the `assert` keyword with `vector {always}` generates an error-level assertion message if the heuristics indicate the loop cannot be vectorized; use `#pragma ivdep!` to ignore assumed dependencies); `#pragma novector` (never vectorize). NOTE: some pragmas are available for both Intel® and non-Intel microprocessors but may perform additional optimizations on Intel microprocessors. **User-mandated pragma:** `#pragma omp simd` — transform the loop into one executed concurrently using SIMD instructions.

**Explicit SIMD SYCL Extension (ESIMD).** oneAPI provides ESIMD for lower-level Intel® GPU programming: APIs like Intel's GPU Instruction Set Architecture (ISA) for explicitly vectorized device code, giving more control over generated code and less dependence on compiler optimizations. Specification, API demos, and working examples are on GitHub. NOTE: some parts are under active development and APIs in `sycl::ext::intel::experimental::esimd` are subject to change. ESIMD kernels and functions always require **subgroup size one** — no compiler vectorization across work-items in a subgroup.

```cpp
float *A = malloc_shared<float>(Size, q); float *B = malloc_shared<float>(Size, q); float *C = malloc_shared<float>(Size, q);
for (unsigned i = 0; i != Size; i++) { A[i] = B[i] = i; }
q.parallel_for(Size / VL, [=](id<1> i)[[intel::sycl_explicit_simd]] {
   auto offset = i * VL;
   simd<float, VL> va(A + offset);   // pointer arithmetic; offset in elements
   simd<float, VL> vb(B + offset);
   simd<float, VL> vc = va + vb;
   vc.copy_to(C + offset);
}).wait_and_throw();
```

The lambda passed to `parallel_for` is marked `[[intel::sycl_explicit_simd]]`, telling the compiler the kernel is ESIMD-based and that ESIMD APIs can be used inside it; `simd` objects and `copy_to` are available only in the ESIMD extension. *Compile and run* — same commands as standard SYCL:

```bash
clang++ -fsycl vadd_usm.cpp                    # open source oneAPI DPC++ Compiler
icpx -fsycl vadd_usm.cpp                       # Intel(R) oneAPI Toolkit
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out  # run on an Intel-specific GPU via the Level Zero backend
```

The executable runs only on Intel GPU hardware, such as Intel® UHD Graphics 600 or later; the SYCL runtime automatically recognizes ESIMD kernels and dispatches their execution, so no additional setup is needed. Linux and Windows, including OpenCL™ and Level Zero backends, are supported. Regular SYCL and ESIMD kernels can co-exist in the same translation unit and application. *SYCL and ESIMD interoperability:* SYCL kernels can call ESIMD functions using the special `invoke_simd` API (see the `invoke_simd` API specification; examples and test cases are available).

```cpp
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>
constexpr int N = 8;
namespace seoe = sycl::ext::oneapi::experimental::simd;
namespace esimd = sycl::ext::intel::simd;
[[intel::device_indirectly_callable]] SYCL_EXTERNAL seoe::simd<float, N> __regcall
esimd_scale(seoe::simd<float, N> x, float n) SYCL_ESIMD_FUNCTION {   // ESIMD fn
  return esimd::simd<float, N>(x) * n;
}
auto ndr = nd_range<1>{range<1>{global_size}, range<1>{N * num_sub_groups}};
q.parallel_for(ndr, sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(N)]] {
   sycl::sub_group sg = it.get_sub_group();
   float x = ...;  float n = ...;
   // x from each work-item grouped into simd<float,N>; n passed as uniform
   // scalar; result simd<float,N> split into N scalars, assigned to
   // each y of each corresponding N work-items.
   float y = seoe::invoke_simd(sg, esimd_scale, x, seoe::uniform(n));
});
```

```bash
# -fsycl-allow-func-ptr: function pointers are not allowed by default in SYCL/ESIMD programs;
# -fno-sycl-device-code-split-esimd: keep invoke_simd() caller and callee in the same module.
clang++ -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o invoke_simd
IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ./invoke_simd
```

*Restrictions* (NOTE: the compiler does not enforce some extensions, which may lead to undefined program behavior): not supported with ESIMD — C and C++ standard libraries support, device library extensions, a host device; unsupported standard SYCL APIs — 2D and 3D accessors, constant accessors, `sycl::accessor::get_pointer()` and `sycl::accessor::operator[]` (supported only with `-fsycl-esimd-force-stateless-mem`; otherwise accessor memory accesses go via explicit APIs, e.g. `sycl::ext::intel::esimd::block_store(acc, offset)`), accessors with offsets and/or access range specified, `sycl::sampler` and `sycl::stream` classes; other restrictions — only Intel GPU devices, and interoperability between regular SYCL and ESIMD kernels is one-way (regular SYCL kernels can call ESIMD functions but not vice-versa).

## Instrumented Profile-Guided Optimization

Traditional **IPGO**: profile collection is done in software, steps essentially the same on all platforms; instrumentation has significant overhead, which can limit the scenarios in which profile collection can be performed. **HWPGO** may be a better alternative when collection overhead is a concern or PMU-based feedback is needed. See the LLVM Project's Clang Compiler User Manual for more details and other software feedback mechanisms.

1. Compile with optimizations plus `-fprofile-generate=app.profraw` — generates additional code tracking the executable's execution profile; expect the instrumentation to slow execution considerably. No particular linker is required; on Linux, if the linker is invoked directly, add the `libclang_rt.profile.a` library as an input and specify `-u__llvm_profile_runtime` as a command-line flag.
```bash
   icx -xCORE-AVX512 -Ofast -fprofile-generate=app.profraw app.c -o app
```
2. Create a profile by executing the instrumented executable (raw profile data is left on disk per the `-fprofile-generate` option; `app.profraw` can be overridden by the `LLVM_PROFILE_FILE` environment variable).
```bash
   ./app
```
   NOTE: the option supports special specifiers such as `%m` (see *Profiling with Instrumentation* for specifiers and `%m`) helping ensure unique file/directory names when multiple processes use the same file system; there is also an icx-specific expansion **`%e`**, the timestamp of seconds since the Epoch, 1970-01-01 00:00:00 +0000 (UTC).
3. Use the raw instrumentation profile(s) to create an LLVM profile (multiple profraw files may be specified for multiple process invocations):
```bash
   llvm-profdata merge app.profraw --output app.prof
```
   NOTE: this merge also converts the raw profile to a format understood by the compiler, so this step is **required even for a single profraw file**.
4. Recompile specifying the profile information to the compiler:
```bash
   icx -xCORE-AVX512 -Ofast -fprofile-use=app.prof -o app
```

## Hardware Profile-Guided Optimization

HWPGO is an alternative to traditional IPGO. IPGO requires a first compilation phase generating an instrumented binary tracking execution counts from a training run; with HWPGO this instrumentation is not needed — the optimized binary's execution is sampled on PMU events using a tool such as Linux perf or SEP, and a profile is generated from PMU-based data and debug info. Benefits over IPGO: the training binary can be highly optimized and collection can occur in a production environment; the PMU provides new hardware introspection not possible with software instrumentation — e.g. the 2024.0 compiler supports **unpredictable branch profiles**, usable to prefer Conditional Move (CMOV) to conditional branches.

### Execution Frequency Feedback

1. Compile with full optimization plus `-fprofile-sample-generate`. HWPGO needs no instrumentation but **requires DWARF debug information** on Linux and Windows. On Windows take special care to produce usable DWARF debug info: the compilation must include DWARF line number information and the **lld-link** linker must be used with DWARF-enabling flags. On Linux and Windows `-fprofile-sample-generate` also enables additional debug information that may improve profile quality; to simplify, use of `-fprofile-sample-generate` is recommended. By default it does not affect optimizations and should not affect execution speed. By default debug info is embedded in object/executable files; to split it, `-fprofile-sample-generate` with `-gsplit-dwarf -fprofile-dwo-dir=<dir>` specifies where to store split `.dwo` files. On Windows the **lld linker** must be used — the `icx` driver ensures this when `-fprofile-sample-generate` is specified; use `lld-link /profile-sample-generate` when invoking the linker directly.
```bash
   icx -xCORE-AVX512 -Ofast -fprofile-sample-generate app.c -o app
```
2. Create a PMU-based profile using SEP or Perf.
```bash
   # Linux:
   perf record -o app.perf.data -b -c 1000003 -e br_inst_retired.near_taken:uppp -- ./app
```
```bash
   :: Windows:
   sep -start -out app.tb7 -ec BR_INST_RETIRED.NEAR_TAKEN:PRECISE=YES:SA=1000003:pdir:lbr:USR=YES -lbr no_filter:usr -perf-script ip,brstack -app .\app.exe
```
   NOTE: `sep` only includes samples for the executable directly launched by `-app` in `-perf-script` output; invoking `app.exe` via a wrapper script or batch file will not include `app.exe` samples (to be improved). PMU-based profile data is now in `app.perf.data` or `app.tb7`; on Windows a partial textual representation is available as `app.perf.data.script`. The sampling period (`1000003`) may need tuning by application characteristics and execution duration; the period for each event type must be specified to `llvm-profgen` with `--sample-period`.
3. Use the PMU profile to create an LLVM profile describing how frequently source-level code locations were observed executing.
```bash
   # Linux:
   llvm-profgen --perfdata app.perf.data --binary app --output app.freq.prof
```
   Same on Windows except you use the textual `app.perf.data.script` profile:
```bash
   llvm-profgen --perfscript app.perf.data.script --binary app.exe --output app.freq.prof
```
4. If steps 2–3 occurred multiple times, merge profiles with e.g. `llvm-profdata merge --sample run1.freq.prof run2.freq.prof run3.freq.prof --output app.freq.prof` — useful for training against multiple datasets.
5. Recompile specifying the profile information to the compiler (you may add `-fprofile-sample-generate` if additional feedback iterations are desirable):
```bash
   icx -xCORE-AVX512 -Ofast app.c -o app -fprofile-sample-use=app.freq.prof
```
6. Optionally, repeat by jumping back to step 2.

### Execution Frequency and Branch Mispredict Feedback

The compiler can take advantage of both instruction execution and branch mispredict profiles; the two can be collected simultaneously.

1. Generate an instrumented binary:
```bash
   icx -xCORE-AVX512 -Ofast -fprofile-sample-generate app.c -o app
   ```
2. Create a PMU-based profile using SEP or Perf.
   ```bash
   # Linux:
   perf record -o app.perf.data -b -c 1000003 -e br_inst_retired.near_taken:uppp,br_misp_retired.all_branches:upp -- ./app
```
```bash
   :: Windows:
   sep -start -out app.tb7 -ec BR_INST_RETIRED.NEAR_TAKEN:PRECISE=YES:SA=1000003:pdir:lbr:USR=YES,BR_MISP_RETIRED.ALL_BRANCHES:PRECISE=YES:SA=1000003:lbr:USR=YES -lbr no_filter:usr -perf-script event,ip,brstack -app .\app.exe
```
```bash
   PMU-based profile data is now in `app.perf.data` or `app.tb7`; on Windows a partial textual representation is available as `app.perf.data.script`. NOTE: the additional event field requested of `sep` is required so `llvm-profgen` can differentiate between PMU events.
3. Use the single PMU profile to create two types of LLVM profiles: the traditional execution frequency profile, and a profile of mispredicted branches.
```
```bash
   # Linux:
   llvm-profgen --perfdata app.perf.data --binary app --output app.freq.prof --sample-period 1000003 --perf-event br_inst_retired.near_taken:uppp
   llvm-profgen --perfdata app.perf.data --binary app --output app.misp.prof --sample-period 1000003 --perf-event mr_misp_retired.all_branches:upp --leading-ip-only
```
   Same on Windows except you use SEP event names and the textual profile:
```bash
   llvm-profgen --perfscript app.perf.data.script --binary app.exe --output app.freq.prof --sample-period 1000003 --perf-event BR_INST_RETIRED.NEAR_TAKEN:pdir
   llvm-profgen --perfscript app.perf.data.script --binary app.exe --output app.misp.prof --sample-period 1000003 --perf-event BR_MISP_RETIRED.ALL_BRANCHES --leading-ip-only
```
   You should now have two source-level profiles: `app.freq.prof` and `app.misp.prof`.
4. If steps 2–3 occurred multiple times, merge profiles with e.g. `llvm-profdata merge --sample run1.freq.prof run2.freq.prof run3.freq.prof --output app.freq.prof`. NOTE: the frequency and mispredict profiles **should not be merged**.
5. Recompile specifying the profile information to the compiler (you may add `-fprofile-sample-generate` if additional feedback iterations are desirable):
```bash
   icx -xCORE-AVX512 -Ofast app.c -o app -fprofile-sample-use=app.freq.prof -mllvm -unpredictable-hints-file=app.misp.prof
```
6. Optionally, repeat by jumping back to step 2.

### Windows and tool notes

The compiler provides an `llvm-profgen` tool to understand Common Object File Format (COFF) binaries with associated Debugging with Attributed Record Formats (DWARF) debug information; `-fprofile-sample-generate` ensures this debug information is generated. The Linux `perf` tool is unavailable on Windows, but Intel® VTune™ includes a `sep` tool performing the relevant Last Branch Records (LBR) sampling on hardware events on both Windows and Linux. To locate the product's `llvm-profgen`/`llvm-profdata`, use `icx --print-prog-name=llvm-profgen`; on Windows (`icx` is a command-line-style driver) use `icx /nologo /clang:--print-prog-name=llvm-profgen`. Alternatively the `--include-intel-llvm` option to `setvars` scripts places these tools in `PATH`.

## High-Level Optimization

HLO exploits properties of source constructs (loops, arrays) in high-level-language applications. Default `O2` performs some high-level optimizations; `O3` provides the best chance for loop transformations optimizing memory accesses. NOTE: loop optimizations may result in calls to library routines giving additional performance gain on Intel® microprocessors, and additional HLO transformations may be performed for Intel microprocessors. HLO loop transformation techniques: Loop Permutation or Interchange; Loop Distribution; Loop Fusion; Loop Unrolling; Data Prefetching; Scalar Replacement; Unroll and Jam; Loop Blocking or Tiling; Partial-Sum Optimization; Predicate Optimization; Loop Reversal; Profile-Guided Loop Unrolling; Loop Peeling; Data Transformation: Malloc Combining and Memset Combining, Memory Layout Change; Loop Rerolling; Memset and Memcpy Recognition; Statement Sinking for Creating Perfect Loopnests; Multiversioning: Checks include Dependency of Memory References, and Trip Counts; Loop Collapsing.

## Interprocedural Optimization

IPO is an automatic, multi-step process allowing the compiler to analyze code to determine where specific optimizations benefit. It may apply: Alias analysis; Automatic array transposition; C++ class hierarchy analysis; Constant propagation; Dead call deletion; Dead formal argument elimination; Dead function elimination; Forward substitution; Indirect call conversion; Inlining; Mod/ref analysis; Passing arguments in registers to optimize calls and register usage; Points-to analysis; Routine key-attribute propagation; Specialization; Structure splitting and field reordering; Whole program analysis.

**Compile with IPO:** as each source file is compiled, the compiler stores an intermediate representation (IR) of the source in a **mock object file**; mock objects contain IR instead of normal object code, and during the IPO compilation phase only mock object files are visible. **Link with IPO:** linking with `-flto` or `/Qipo` invokes the compiler a final time, performing IPO across all mock object files; mock objects must be linked with the compiler or with LLVM linking tools, and while linking with IPO the compiler and other linking tools compile mock object files and invoke the real/true object file linkers of the platform. NOTE (2024.0+): `-mllvm` options are no longer passed through to linker option processing — use `-Wl`, e.g. `-Wl,-plugin-opt,-lto-debug-options`.

**Whole Program Analysis.** Many IPO optimizations apply, or become much more effective, when the **whole program condition** is satisfied. The compiler reads all IR in the mock file, object files, and library files to determine whether all references are resolved and whether a symbol is defined in a mock object file; symbols in the IR of a mock object file for both data and functions are candidates for manipulation. Two types — **object reader method** (emulates the native linker and attempts to resolve application symbols; if all resolve, the condition is satisfied; more likely to detect it) and **table method** (analyzes mock object files and generates a call-graph; detailed compiler tables cover all functions of important language-specific libraries like `libc`; the compiler builds a call-graph, compares the function table and call-graph, and for each unresolved function attempts to resolve calls by finding an entry in the compiler tables — if it can resolve the call, the condition exists). Most optimizations apply if either type finds the condition, but some require the object reader method's results and some the table method's.

**Use Interprocedural Optimization.** First compile each source file, then link the resulting files.

```bash
# Linux:
icpx -ipo -c a.cpp b.cpp c.cpp      # a.o b.o c.o; -c stops after .o objects (contain IR)
icpx -ipo -o app a.o b.o c.o        # links IR objects; new list of objects to link; produces app
icpx -ipo -o app a.cpp b.cpp c.cpp  # separate compile and link commands combined
```

`icpx` calls GCC `ld` to link the specified object files and produce the application specified by `-o`; the default Linux linker is the standard Berkeley Free Distribution, but LLD can be used via `-fuse-ld=lld`.

```bat
:: Windows:
icx /Qipo /c a.cpp b.cpp c.cpp      :: a.obj b.obj c.obj; /c stops after .obj files (IR)
icx /Qipo /Feapp a.obj b.obj c.obj  :: links IR objects; produces app, specified by /Fe
icx /Qipo /Feapp a.cpp b.cpp c.cpp  :: combined
```

`icx` calls `link.exe`; the executable is named by `/Fe`. Windows' only possible linker is LLD, default with `/Qipo`. NOTES: on Linux, `icpx` uses standard C++ libraries automatically, `icx` does not. Intel linking tools emulate compiling at `-O0` (Linux) and `/Od` (Windows). If multi-file IPO is applied to object files none of which are mock objects, no multi-file IPO is performed — they are simply linked. The 2024.0 `-mllvm`/`-Wl` note applies here too.

**IPO-Related Performance Issues.** Using IPO on very large programs might trigger internal limits of other compiler optimization phases; applications where the compiler lacks sufficient IR coverage for whole program analysis might not perform as well as those with complete IR information. Do not use the link phase of an IPO compilation using mock object files produced for your application by a **different compiler** — Intel® compilers cannot inspect mock object files generated by other compilers for optimization opportunities. Update make files to call the LLVM linker when using IPO from scripts; `-fuse-ld=lld` tells the compiler to use the lld linker.

**Create a Library from IPO Objects.** Libraries are often created with a library manager such as `llvm-ar` (Linux) or `llvm-lib` (Windows); given a list of objects, it inserts them into a named library for subsequent link steps.

```bash
llvm-ar cru user.a a.o b.o        # Linux: create library user.a containing the a.o and b.o objects
```

```bat
:: Windows: create three mock object files (a.obj contains the main subprogram), a library, then link
icx /c /Qipo a.cpp b.cpp c.cpp
llvm-lib -out:main.lib b.obj c.obj
icx -fuse-ld=lld a.obj main.lib -o result.exe
```

**Inline Expansion of Functions.** Inline function expansion does not require the whole program analysis criteria normally required by IPO, so it is one of the most important IPO optimizations. For calls believed frequently executed, compiler often replaces the call instructions with the function code itself; relatively small user functions are inlined more often than relatively large ones. It improves performance by removing the need to set up parameters for a function call, eliminating the function call branch, and propagating constants. Inlining can improve execution time by removing runtime call overhead, but can increase code size, code complexity, and compile times; when instructed to inline, compiler can examine source in a much larger context and find more optimization opportunities. `[Q]ip` (**single-file IPO**) performs inline function expansion for calls to procedures defined within the current source file; `[Q]ipo` (**multi-file IPO**) performs it for calls to procedures defined in other files.

> **Caution:** using the `[Q]ip` and `[Q]ipo` Windows options can, sometimes, substantially increase compile time and code size.

*Select routines for inlining:* compiler selects routines whose inline expansions provide the greatest benefit, using default heuristics; heuristics differ if you use `[Q]prof-use`. With PGO and `[Q]ip` or `[Q]ipo`, the default heuristic focuses on the most frequently executed call sites based on gathered profile information and always inlines very small functions meeting the minimum inline criteria. *Use IPO with PGO:* combining IPO and PGO typically produces better results than IPO alone; PGO's dynamic profiling information usually gives better optimization opportunities than IPO's static profiling information. Compiler estimates frequently executed call sites from source characteristics and applies those estimates to the PGO-based guidelines; that estimate is not always accurate. *Inline expansion of library functions:* by default compiler automatically inlines (expands) several standard and math library functions at the point of call, usually producing faster computation; many routines in `libirc`, `libm`, or `svml` are more highly optimized for Intel microprocessors than for non-Intel microprocessors. `-fno-builtin` (Linux*) or `/Qno-builtin-<name>` and `/Oi-` (Windows*) disable inlining for intrinsic functions and disable by-name recognition support and the resulting optimizations; `/Qno-builtin-<name>` fine-tunes `/Oi-`, which disables almost all intrinsic functions. Use these if you redefine standard library routines with your own same-named version. *Inlining and function preemption on Linux:* you must specify `fpic` to use function preemption; by default compiler does not generate the position-independent code needed for preemption. *Developer-directed inline expansion of user functions:* compiler measures routine size in an abstract value of **intermediate language units**, about equivalent to the number of instructions generated, and classifies routines as relatively small, medium, or large to decide when to inline; if the minimum criteria are met and all else is equal, it favors inlining relatively small functions and not inlining relatively large ones. Functions are typically targeted based on **inlining keywords** (`__inline`, `__forceinline`), **procedure-specific inlining pragmas** (`#pragma inline`, `#pragma forceinline`, which inline calls within the targeted procedure if legal), and **GCC function attributes for inlining** (`__attribute__((always_inline))`, which inline even with no optimization level specified).

## Methods to Optimize Code Size

Two compiler options prioritize code size over performance:

| Option | Result | Notes |
|---|---|---|
| `Os` | Favors size over speed | Enables optimizations that do not increase code size; produces smaller code than `O2`; disables some optimizations that may increase size for a small speed benefit. |
| `O1` | Minimizes code size | Disables even more optimizations generally known to increase size than `Os`. **Specifying `O1` implies `Os`.** As an intermediate step you can replace `O3` with `O2` before `O1`. May improve performance for applications with very large code size, many branches, and execution time not dominated by code within loops. |

Some methods may already be applied by default with `Os`/`O1`; all can be applied at higher optimization levels. Some options will not necessarily reduce code size and may give varying results (good, bad, or neutral) based on target-code characteristics — still, they are the recommended things to try.

- **Disable or decrease the amount of inlining** (call replaced by the function body; context-specific optimization and no call overhead, but size increase can be substantial). *Advantage:* can reduce code size. *Disadvantage:* performance likely sacrificed, especially with many small functions. Linux: `fno-inline`. Windows: `Ob0`.
- **Strip symbols from your binaries.** *Advantage:* noticeably reduces binary size. *Disadvantage:* debugging a stripped application may be very difficult. Linux: `Wl, --strip-all`. Windows: none.
- **Dynamically link Intel-provided libraries** (by default some Intel support and performance libraries are linked statically into every executable, duplicating codes). *Advantage:* performance normally not substantially affected; those library codes no longer contribute to each executable's size, are shared between all executables using them, and are available independent of those executables. *Disadvantage:* dependent libraries must be redistributed with the executable; static linking includes only library content actually used while dynamic libraries contain all content, so the executable itself may be much smaller but the total of executable plus shared libraries/DLLs may be much larger than a static executable — may not be beneficial when building/distributing only a single executable. Linux: `shared-intel`. Windows: `MD` (NOTE: `MD` affects all libraries, not only Intel-provided ones).
- **Exclude unused code and data from the executable** (function-level or data-level linking, without expensive whole-program interprocedural analysis). *Advantage:* only referenced code remains; dead functions and data are stripped; options passed to the linker also let it reorder sections for other possible optimization. *Disadvantage:* object codes may become slightly larger because each function or datum goes into a separate section (that overhead disappears at the linking stage); requires linker support and may increase linking time. Linux: `-fdata-sections -ffunction-sections -Wl,--gc-sections`. Windows: `/Gw /Gy /link /OPT:REF`. Passed to the linker: `Wl, --gc-sections` (Linux), `link /OPT:REF` (Windows).
- **Disable recognition and expansion of intrinsic functions** (intrinsics can be expanded inline or their faster library implementation assumed and linked in; inline expansion of intrinsics is enabled by default). *Advantage:* both object-file size and library-code size brought into an executable can be reduced. *Disadvantage:* can prevent performance optimizations; a slower standard library implementation is used; final executable size can increase when statically pulled library code for an otherwise inlined intrinsic is large. Linux: `fno-builtin`. Windows: `Oi-`. Also: already the default with `O1`; for C++ the Linux option `nolib-inline` disables inline expansion of standard library or intrinsic functions; depending on code characteristics this option can sometimes increase binary size.
- **Optimize exception handling data.** For SYCL, enabling/disabling exception handling is supported for host compilation. If a program needs support, compiler creates a special section of DWARF directives used by the Linux runtime to unwind and catch an exception; this `.eh_frame` information may be shrunk with the options below. *Advantage:* may shrink object/binary size by **up to 15%**, depending on target platform; they control whether unwind information is precise at an instruction boundary or at a call boundary — e.g. `fno-asynchronous-unwind-tables` can be used for programs that may only throw or catch exceptions. *Disadvantage:* both options may change program behavior. Do not use `-fno-exceptions` for programs requiring standard C++ handling for objects of classes with destructors. Do not use `fno-asynchronous-unwind-tables` for functions compiled with `-fexceptions` that call other functions that might throw, or for C++ functions that declare objects with destructors. Linux: `-fno-exceptions` or `-fno-asynchronous-unwind-tables`. Windows: none.
- **Disable loop unrolling** (unrolling increases loop size proportionally to the unroll factor). *Advantage:* code size reduced. *Disadvantage:* performance of otherwise unrolled loops may noticeably degrade because other loop optimizations are limited. Linux: `unroll=0`. Windows: `Qunroll:0` (NOTE: **not available for SYCL**). Already the default with `Os` or `O1`.
- **Disable automatic vectorization** (finds SIMD Intel® SSE/Intel® AVX opportunities; usually transforms loops, increasing code size, sometimes substantially). *Advantage:* compile time also improves substantially. *Disadvantage:* performance of otherwise vectorized loops may suffer substantially; use selectively to suppress vectorization on everything except performance-critical parts. Linux: `no-vec`. Windows: `Qvec-`.

## Gotchas & failure modes

1. **`#pragma ivdep` only overrides *assumed* dependencies**, never proven ones; with real dependencies it yields incorrect results (`#pragma ivdep!` ignores assumed dependencies). `restrict` requires `[Q]std=c99`, is not supported by all compilers (portability loss), and its incorrect use also changes semantics; with `restrict` compiler still performs a runtime aliasing check.
2. **Only `#pragma omp simd` warns** when it cannot vectorize; hints (even `#pragma vector always`) leave the decision to compiler. Under `omp simd` semantics, assigning a vector value to a scalar L-value or assigning a scalar L-value under a vector condition is an error, and `switch` is unsupported.
3. **SIMD-enabled functions are `noexcept`** (C++11 terms), including their scalar routine; exceptions cannot be thrown/caught, `setjmp`/`longjump` and EH constructs are banned, only OpenMP `atomic` and `simd` are allowed inside, and non-vector calls serialize lane-by-lane (can violate the writes-only-via-arguments rule).
4. **`#pragma omp declare simd` requires `[q or Q]openmp` or `[q or Q]openmp-simd`.**
5. **SIMD-enabled function pointers are disabled by default and binary-incompatible** with regular pointers; mixing may give severe unpredictable results. Vector specs are not part of the pointer type, are invisible to name mangling (functions differing only by such attributes get identical mangled names), and are lost through templates/wrappers; compiler emits `Warning #3757: this use of a vector function type is not fully supported` when one is a C++ parameter — give such functions distinct names.
6. `setjmp`/`longjmp`, `SHE`, computed GOTO, inline ASM, VM, vector intrinsics (SVML), locks/barriers/atomics/critical sections, and OpenMP thread creation are disallowed in vector declarations.
7. **`__m64`, `__m128`, `__m256` are not vectorizable** and SSE/AVX intrinsics (`_mm_add_ps`, `_mm256_add_ps`) are disallowed in vectorizable loop bodies. Even a print statement prevents vectorization (`non-standard loop is not a vectorization candidate`); exceptions are intrinsic math functions and functions that can be inlined.
8. **Incorrect aligned data movement raises an exception with Intel® SSE.** Disabling dynamic alignment decreases performance but may improve bitwise reproducibility.
9. **ESIMD:** subgroup size one; Intel GPU devices only; no C/C++ standard libraries, device library extensions, or host device; 2D/3D accessors, constant accessors, accessors with offsets/access range, `sycl::sampler`, `sycl::stream` unsupported; accessor `get_pointer()`/`operator[]` need `-fsycl-esimd-force-stateless-mem`; interoperability one-way (SYCL→ESIMD). `invoke_simd` needs `-fsycl-allow-func-ptr` and `-fno-sycl-device-code-split-esimd`, plus `IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1` at run time; some `sycl::ext::intel::experimental::esimd` APIs are subject to change.
10. **IPGO instrumentation slows execution considerably**; `llvm-profdata merge` is mandatory even for a single profraw file; `%m` and the icx-specific `%e` (seconds since Epoch) create unique profile names; direct Linux linker invocation needs `libclang_rt.profile.a` and `-u__llvm_profile_runtime`.
11. **HWPGO needs DWARF debug info** (use lld/lld-link with DWARF flags on Windows). `sep` samples only the executable launched directly by `-app`, not one started via a wrapper script/batch file. `llvm-profgen` needs `--sample-period`, and multi-event data needs the event name field from `sep`. **Never merge frequency and mispredict profiles.**
12. **IPO mock objects are compiler-specific** — do not link mock object files produced by a different compiler. If no input file is a mock object, multi-file IPO is silently not performed. Update make files to call the LLVM linker (`-fuse-ld=lld`). **2024.0:** `-mllvm` options no longer pass through to the linker; use `-Wl`.
13. **`[Q]ip`/`[Q]ipo` on Windows can substantially increase compile time and code size;** IPO on very large programs might trigger internal limits of other optimizer phases, and incomplete IR coverage weakens whole program analysis.
14. **Code-size caveats:** `Qunroll:0` is unavailable for SYCL; `MD` affects all libraries; dynamic linking requires redistributing libraries and total executable + shared-library size may exceed a static build; stripping symbols makes debugging very difficult; `-fno-exceptions` is unsafe where objects with destructors need standard C++ handling; `fno-asynchronous-unwind-tables` is unsafe for `-fexceptions` functions calling functions that might throw, or C++ functions declaring objects with destructors. Inlining, unrolling, and intrinsic recognition can each increase *or* decrease binary size depending on code characteristics.
15. **`icx` vs `icpx` on Linux:** `icpx` uses standard C++ libraries automatically, `icx` does not; Intel linking tools emulate `-O0` (Linux) / `/Od` (Windows).
16. **Auto-vectorization can grow code size and compile time** (reverse with `no-vec`/`Qvec-`) — apply the size/performance tradeoff selectively.

## Source map

- Multi-tile multi-card tail; Vectorization; Automatic Vectorization — p. 763
- Vectorization Programming Guidelines (innermost-loop guidelines, Restrictions, writing vectorizable code); Dynamic Alignment Optimizations — pp. 764–765
- Use Aligned Data Structures; Use Structure of Arrays Versus Array of Structures — pp. 766–769
- Use Automatic Vectorization; Vectorization Speedup (Linux/Windows guided_matmul_opt_report) — pp. 769–770
- Obstacles to Vectorization — pp. 770–772; Help the Compiler Vectorize — pp. 772–774
- Vectorization and Loops; Types of Vectorized Loops — pp. 774–775
- Loop requirements and statements in the loop body; Loop Constructs; Loop Exit Conditions — pp. 776–777
- Strip-Mining and Cleanup; Loop Blocking; Loop Interchange and Subscripts with Matrix Multiply — pp. 778–780
- Explicit Vector Programming; User-Mandated or SIMD Vectorization (figures) — p. 781
- Additional Semantics; Restrictions on `#pragma omp declare simd` — pp. 782–783
- SIMD-Enabled Functions (clauses, usage, C++) — pp. 783–789
- SIMD-Enabled Function Pointers (compatibility, call sequence, C++ type system, indirect invocation) — pp. 789–795
- Function Annotations and the SIMD Directive for Vectorization — pp. 795–797
- Explicit SIMD SYCL Extension — pp. 797–799
- Instrumented PGO; Hardware PGO; execution-frequency and mispredict workflows; Windows/tool notes — pp. 799–803
- High-Level Optimization — pp. 803–804
- Interprocedural Optimization; Whole Program Analysis; Use Interprocedural Optimization — pp. 804–807
- IPO-Related Performance Issues; Create a Library from IPO Objects — p. 807
- Inline Expansion of Functions — pp. 808–809
- Methods to Optimize Code Size — pp. 809–813
