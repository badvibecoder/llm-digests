---
chunk: 07b-pragmas-and-errors
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 595-621
covers: Pragmas overview/syntax; Intel-specific pragma reference (block_loop, distribute_point, inline/forceinline/noinline, ivdep, loop_count, nofusion, novector, omp target variant dispatch, ompx prefetch data, prefetch/noprefetch, unroll/nounroll, unroll_and_jam/nounroll_and_jam, vector); supported OpenMP* pragmas (alphabetical + categories); Microsoft*/GCC-compatible pragmas; predefined-macro compiler detection
---

# Pragmas — Intel oneAPI DPC++/C++ Compiler

> **Scope.** Lookup for every pragma in this range: syntax, clauses/arguments, defaults, host-vs-device limits, and the OpenMP* pragma inventory by name and category. Also the p. 595 compiler-detection macro examples. Pages 595–621 contain no "Syntactic and Semantic Errors" topic.

## Key facts

- `#pragma` is standard C++, but individual pragmas are machine-/OS-specific and **vary by compiler**. Form: `#pragma <pragma name>`.
- Pragmas can duplicate compiler-option functionality and **override behavior specified by compiler options**.
- Some pragmas work on Intel® and non-Intel microprocessors but may perform **additional optimizations on Intel® microprocessors**.
- **Most Intel-specific pragmas are host-code only unless otherwise noted**; `unroll`/`nounroll` cover both host and device (target device support CPU and GPU).
- OpenMP: compiler **currently supports OpenMP* 5.0 Version TR4, and some OpenMP Version 5.1 pragmas**; Intel-specific clauses noted per pragma.
- Deprecated/removed: `omp target variant dispatch` (**deprecated, support removed**; use `omp dispatch`); `omp master` (**deprecated**; see `omp masked`).
- Hardware scope: `ompx prefetch data` — **Intel® Iris® Xe MAX GPU only**; `prefetch`/`noprefetch` — **Intel® AVX-512 only**.
- Optimization gates: `prefetch`/`noprefetch` and `unroll`/`nounroll` need `O2` or higher; `unroll_and_jam`/`nounroll_and_jam` needs `O3`.

## Code examples — predefined macro compiler detection (p. 595)

```c
#if !defined(SYCL_LANGUAGE_VERSION) && defined (__INTEL_LLVM_COMPILER)
   /* code specific for Intel C++ Compiler below; example only */
   std::cout << "__INTEL_LLVM_COMPILER: " << __INTEL_LLVM_COMPILER << std::endl;
   std::cout << "__VERSION__: " << __VERSION__ << std::endl;
#endif
```
The SYCL* variant also prints `SYCL_LANGUAGE_VERSION`. Output for Intel® oneAPI Toolkit Gold + compiler patch 2021.1.2 — Linux: `SYCL_LANGUAGE_VERSION: 202001`, `__INTEL_LLVM_COMPILER: 202110`, `__VERSION__: Intel(R) Clang Based C++, gcc 4.2.1 mode`; Windows: the same compiler values with `__VERSION__: Intel(R) Clang Based C++, clang 12.0.0`. Non-SYCL builds omit the `SYCL_LANGUAGE_VERSION` line.

## Option / API quick table

| pragma | host/device, gate | key values / default |
|---|---|---|
| `block_loop`/`noblock_loop` | host | `factor(expr)`; `level(...)` = `const1` or `const1:const2`, `m<=8`,`n<=8`,`n>m`; no clause → compiler factor, all levels |
| `distribute_point`, `ivdep`, `nofusion`, `novector` | host | no args |
| `inline`/`forceinline`/`noinline` | host | `[recursive]`; beats function-specific pragmas |
| `loop_count` | host | `(n)`/`=n`, list, `min(n),max(n),avg(n)`; min/max guaranteed |
| `omp target variant dispatch` | — | `device(...)`, `subdevice(...)`, `nowait`, `use_device_pointer (ptr-list)`; **deprecated/removed** |
| `ompx prefetch data` | Xe MAX GPU | modifier 0–7 (default 0) |
| `prefetch`/`noprefetch` | host, `O2`+ | `*:hint[:distance]`; hint 1–4; distance > 0; AVX-512 only |
| `unroll`/`nounroll` | host **and device** (CPU, GPU), `O2`+ | `unroll(n)`; overrides command line |
| `unroll_and_jam`/`nounroll_and_jam` | host, `O3` | `(n)`, `n` integer constant 0–255; not innermost loops |
| `vector` | — | `always[assert]`, `aligned`, `unaligned`, `dynamic_align`, `nodynamic_align`, `temporal`, `nontemporal`, `[no]vecremainder`, `vectorlength(n1[,n2]...)` = 2,4,6,8,16,32,64 |
| Microsoft* set | — | `alloc_text bss_seg code_seg comment component const_seg data_seg fenv_access float_control fp_contract init_seg message optimize pointers_to_members pop_macro push_macro region/endregion section vtordisp warning` |
| GCC set | — | `poison` (also `#pragma POISON`), `options` |

## Intel-Specific Pragma Reference

### block_loop/noblock_loop

`#pragma block_loop [clause[,clause]...]` · `#pragma noblock_loop` — split large iteration-counted loops into smaller iteration groups for better cache use.
- `factor (expr)`: positive scalar constant integer expression, the blocking factor; optional, **max one**; if absent, factor from processor type + memory access patterns applied to the specified levels.
- `level (level_expr[, level_expr]... )`: `const1` or `const1:const2`; `const1` positive integer constant `m <= 8` for level `m` (**immediately following loop = level 1**); `const2` positive integer constant `n <= 8`, `n > m`; `const1:const2` = levels `const1`–`const2`.

Any clause order; no clause → compiler picks the best factor for all levels. **Loop-carried dependence is ignored.** Host only. Documented example forms: `factor(256) level(1)`, `factor(512) level(2)`, `level(2)`+`level(1)` (any order), `level(1:2)` (range), `factor(256)` (all levels), bare `block_loop`, `noblock_loop`. `factor(256) level(1:2)` on a `j`/`i` nest (`f = f + a[i]*b[i]`, then `c[j] = c[j] + f`) adds outer `jj`/`ii` loops of `n/256+1` around `j = (jj-1)*256+1 .. min(jj*256, n)` and `i = (ii-1)*256+1 .. min(ii*256,n)`.

### distribute_point

`#pragma distribute_point` (no arguments) — prefer loop distribution at the indicated location; useful when optimizations like vectorization cannot take place due to excessive register usage.
1. Inside a loop the compiler distributes the loop at that point; **all loop-carried dependencies are ignored**.
2. Inside the loop, pragmas **cannot be placed within an `if` statement**.
3. Outside the loop the compiler distributes by internal heuristic, determines where, and **observes data dependency**.
4. Multiple instances are supported when placed inside the loop.

Host only. Documented examples: `dist1` puts the pragma before `for (int i=1;i<1000;i++)` computing `b[i]=a[i]+1; c[i]=a[i]+b[i]; d[i]=c[i]+1`; `dist2` puts it mid-body between `b[i]` and `c[i]` (distribution starts there, ignoring loop-carried dependency); `loop_distribution_pragma1`/`loop_distribution_pragma2` make the same inside/outside choice for a `NUM`=1024 loop assigning `a[i]=a[i]+i` … `z[i]=z[i]+i`.

### inline, noinline, forceinline

`#pragma inline [recursive]` · `#pragma forceinline [recursive]` · `#pragma noinline` — statement-specific inlining pragmas; placed before a C/C++ statement, each applies to all calls in that statement and in statements nested in it. `recursive`: applies to all calls called by these calls, recursively, down the call chain. `forceinline`: inline whenever the compiler is capable of doing so. `inline`: a hint; the compiler is expected **not** to inline when heuristics find inlining overly aggressive (might slow compilation excessively, create too large an executable, or degrade performance). `noinline`: do not inline. Statement-specific pragmas **take precedence over function-specific pragmas**. Host only. Documented example: `#pragma forceinline recursive` before an inner loop calling `sun(a,b)`, where `sun` calls `fun(a,b)`, applies to both calls.

### ivdep

`#pragma ivdep` (no arguments) — ignore assumed vector dependencies. The compiler treats an assumed dependence as a proven one, which prevents vectorization; this pragma overrides that decision. **Only assumed dependencies are ignored; proven dependencies are not.** Host only. Binds only the `for` loop in the current function, including a `for` loop in a sub-function called by it. The `vector` pragma can also override the vectorizer's efficiency heuristics.

```c
void ignore_vec_dep(int *a, int k, int c, int m) {
#pragma ivdep
for (int i = 0; i < m; i++) a[i] = a[i + k] * c; }  /* no vectorize without ivdep: k unknown, illegal if k<0 */
#pragma ivdep
for (i=1; i<n; i++) { e[ix[2][i]] = e[ix[2][i]]+1.0; e[ix[3][i]] = e[ix[3][i]]+2.0; }
```
See Also: Function Annotations and the SIMD Directive for Vectorization; `novector`; `vector`.

### loop_count

`#pragma loop_count(n)` · `#pragma loop_count=n` · `(n1[, n2]...)` · `=n1[, n2]...` · `min(n),max(n),avg(n)` · `min=n, max=n, avg=n`

Specifies the minimum, maximum, or average iteration count of a `for` loop plus an optional list of commonly occurring values so the compiler can generate multiple versions and perform complete unrolling. `(n)`/`=n`: non-negative integer; the compiler **attempts** to iterate the next loop `n` times, **not guaranteed**. List form: non-negative integers; the compiler attempts those or some other unspecified number, giving unrolling flexibility, **not guaranteed**. `min/max/avg`: non-negative integers, one or more in any order without duplication; the specified maximum, minimum, or average (`n1`) is ensured — **guaranteed for `min` and `max`**. More than one pragma per loop allowed; do not duplicate it. Host only. Example: `#pragma loop_count min(3), max(10), avg(5)` before `for (i=start;i<=end;i++) iret += a;` in `mysum`, called as `mysum(1,10,3)`, `mysum(2,6,2)`, `mysum(5,12,1)` (printing `t1`,`t2`,`t3`).

### nofusion

`#pragma nofusion` (no arguments) — prevent a loop from fusing with adjacent loops; place immediately before the loop that should not be fused. Host only. Example: after a `for(j=0;j<SIZE;j++) A[j]=A[j]+B[j];` loop (`#define SIZE 1024`), the pragma precedes `for (i=0;i<SIZE;i++) k += A[i] + 1;` in `int sub ()`.

### novector

`#pragma novector` (no arguments) — the loop should **never** be vectorized, even if legal; use when vectorization causes a performance regression. Contrast `vector always`. Host only. When the trip count `(ub - lb)` is too low to make vectorization worthwhile, it prevents vectorization even for a loop considered vectorizable. Example: `#pragma novector` before `for(j=lb; j<ub; j++) { a[j]=a[j]+b[j]; }` in `void foo(int lb, int ub)`. See Also: Function Annotations and the SIMD Directive for Vectorization; `vector`.

### omp target variant dispatch

**DEPRECATED — support has been removed from the compiler; `omp dispatch` is the suggested replacement.** Conditionally calls a procedure offload variant if the specified device is available; otherwise executes the procedure on the host.

```c
#pragma omp target variant dispatch {device(integer-expression) | nowait |
subdevice([integer-constant ,] integer-expression [ : integer-expression [ : integer-
expression] ] ) | use_device_pointer (ptr-list)}
```
`device` — only if device `n` is available. `subdevice` — only if the specified tiles or compute slices are available. `nowait` — calls may occur asynchronously (without it, synchronously). `use_device_ptr` — use the device pointer instead of the host pointer when the variant procedure is called `[sic: syntax writes use_device_pointer, argument table writes use_device_ptr]`. With both `device` and `subdevice`, the variant runs only if the tiles/compute slices are available on device `n`, else the base version runs on the host. The pragma emits conditional dispatch code around the procedure call following it; the procedure name must have appeared in an `omp declare variant` pragma in the specification part of the calling scope, and the variant interface must be accessible in the base procedure where the pragma appears.

### ompx prefetch data

```c
#pragma ompx prefetch data( [prefetch-hint-modifier:] arrsect [, arrsect] ) [if (condition)]
```
Issues a prefetch to pre-load the specified array sections. `arrsect` — a **contiguous** array section (stride not specified, or constant 1, as defined in OpenMP 5.1). `prefetch-hint-modifier` — optional implementation-defined positive constant literal integer 0–7 inclusive, **assumed 0 when not specified**: 0 no operation; 1 L1 uncached + L3 uncached; 2 L1 uncached + L3 cached; 3 L1 cached + L3 uncached; 4 L1 cached + L3 cached; 5 L1 streaming load + L3 uncached; 6 L1 streaming load + L3 cached; 7 L1 and L3 cached memory load, and invalidate L1 cache. `if` — optional condition, the same as the existing `if` clause for the parallel construct in OpenMP 5.1; prefetch only if `condition` is true. **Intel® Iris® Xe MAX GPU only.** Example (in a loop `m < 1024`): `#pragma ompx prefetch data(4: y[m+16], z[m+16]) if(m%16==0 && (m+16) < 1024)` before `x[m] = y[m] + z[m];` — `4` means prefetch to L1 and L3 cache.

### prefetch/noprefetch

```c
#pragma prefetch
#pragma prefetch *:hint[:distance]
#pragma prefetch [var1 [: hint1 [: distance1]] [, var2 [: hint2 [: distance2]]]...]
#pragma noprefetch [var1 [, var2]...]
```
Invite the compiler to issue or disable data-prefetch requests; hints affect compiler heuristics. **Applies only to Intel® AVX-512.** `var` — optional memory reference. `hint` — optional type: **1** integer data that will be reused; **2** integer and floating-point data reused from L2 cache; **3** data reused from L3 cache; **4** data that will not be reused; requires `var`. `distance` — optional integer **greater than 0**, iterations ahead at which a prefetch is issued before the corresponding load or store; requires `var` and `hint`. No arguments → all arrays accessed in the immediately following loop are prefetched. If the loop includes `A(j)`, `#pragma prefetch A` before it inserts prefetches for `A(j + d)`, `d` determined by the compiler. `#pragma prefetch *` with hint and distance prefetches all array accesses in the loop. **Option `O2` or higher must be in effect.** `noprefetch` hints not to generate prefetches for some memory references. Host only. Documented examples: `#pragma prefetch htab_p:1:30` and `#pragma prefetch htab_p:0:6` before the loop initializing `htab_p[i*m1 + j]` (issues vprefetch1 at distance 30 and vprefetch0 at distance 6 vectorized iterations ahead; without the pragmas the compiler chooses both distances); `#pragma noprefetch b` with `#pragma prefetch a` on `for(i=0; i<m; i++) { a[i]=b[i]+1; }`; and inside a sparse-row loop `for (i=i0; i!=i1; i+=is)` the combination `#pragma noprefetch col`, `#pragma prefetch value:1:80`, `#pragma prefetch x:1:40` before `for(; ip<srow[i+1]; c=col[++ip]) sum -= value[ip] * x[c];`.

### unroll/nounroll

`#pragma unroll` · `#pragma unroll(n)` · `#pragma nounroll` — unroll or do not unroll a loop; `n` = unrolling factor (number of times). Must precede the `for` statement for each `for` loop it affects. If `n` is specified the optimizer unrolls `n` times; if omitted the optimizer assigns the number. Supported when option `O2` or higher is in effect; **overrides any command-line loop-unrolling setting**. Applies to an innermost loop or an outer loop. `nounroll` stops unrolling a specified loop (use when unrolling increases register pressure and code size). Supported in **both host and device code**; target device support **CPU and GPU**. Examples: `#pragma unroll(4)` before an inner `for (int i = 1; i < 100; i++)` doing `b[i]=a[i]+1; d[i]=c[i]+1;`; and `#pragma unroll (4)` before an outer `for (int i = 0; i < 4; i++)` over `int dir[4]= {1,2,3,4}; int data[10];` whose inner loop is `for (int j = dir[i]; data[j]==N ; j+=dir[i]) m++;`. Placed before the **first** `for` loop it unrolls the outer loop **completely**; placed before the inner as well as the outer `for` loop, the compiler honors both.

### unroll_and_jam/nounroll_and_jam

`#pragma unroll_and_jam` · `#pragma unroll_and_jam (n)` · `#pragma nounroll_and_jam` — enable/disable loop unrolling and jamming; only for **iterative `for` loops**. `n` = unrolling factor, an **integer constant from 0 through 255**. Partially unrolls one or more loops higher in the nest than the innermost and fuses/jams the resulting loops back together, allowing more reuses. **Not effective on innermost loops** — ensure the immediately following loop is not innermost after compiler-initiated interchanges complete. Specifying it is a hint that the sequence is legal and profitable; the compiler enables the transformation whenever possible. Must precede the `for` statement for each loop it affects; if `n` is specified the optimizer unrolls `n` times, and **if `n` is omitted or outside the allowed range the optimizer assigns the number**; correct code is generated by comparing `n` and the loop count. **Supported only when compiler option `O3` is set**; overrides any command-line loop-unrolling setting. `nounroll_and_jam` hints not to unroll (use when unrolling increases register pressure and code size in a nested or imperfect nested loop). Host only. Example: `#pragma unroll_and_jam (6)` before `for (i = 1; i < n; i++)` and again before the `j` loop, whose body runs `for (k = 1; k < n; k++){ a[i][j] += b[i][k]*c[k][j]; }`.

### vector

```c
#pragma vector {always[assert]|aligned|unaligned|dynamic_align|nodynamic_align|
temporal|nontemporal|[no]vecremainder|vectorlength(n1[, n2]...)}
```
`always [assert]` — override any efficiency heuristic when deciding to vectorize or not, and vectorize non-unit strides or very unaligned memory accesses; controls the subsequent loop; optional `assert` generates a diagnostic message if the loop cannot be vectorized. `aligned`/`unaligned` — use aligned / unaligned data movement instructions for all array references when vectorizing. `dynamic_align`/`nodynamic_align` — perform / disable dynamic alignment optimization for the loop. `temporal` — temporal (non-streaming) stores on systems based on all supported architectures, unless otherwise specified. `nontemporal` — non-temporal (streaming) stores likewise; you must also insert any fences needed for correct memory ordering within/across threads (typically a `_mm_sfence` intrinsic call just after the loops where the compiler may insert streaming store instructions). `vecremainder`/`novecremainder` — vectorize / do not vectorize the remainder loop when the original loop is vectorized. `vectorlength (n1[, n2]...)` — which vector length/factor to use for the main vector loop.

- The compiler **does not apply the pragma to nested loops**; each nested loop needs a preceding pragma statement, placed before the loop control statement.
- Specify **only one** of `aligned`/`unaligned`. **Caution:** with `aligned` the loop must really be vectorizable that way, otherwise **the compiler generates incorrect code**; all-aligned array references cause a **run-time exception** if some access patterns are actually unaligned.
- Dynamic alignment peels iterations from the vector loop into a scalar loop (possibly itself vectorized) before the vector loop so it aligns with a particular memory reference. `dynamic_align` enables it but efficiency heuristics still decide whether it applies; `nodynamic_align` disables it. **By default the compiler does not perform the optimization.**
- `temporal`/`nontemporal` control how stores of register contents to storage are performed (streaming vs non-streaming) on systems based on Intel® 64 architectures. **By default the compiler automatically determines whether a streaming store is used for each variable.** Streaming stores may significantly improve performance for large numbers on certain processors, but **misuse can significantly degrade performance**.
- `vectorlength`: `n` is an integer power of 2; **must be 2, 4, 6, 8, 16, 32, or 64** `[sic: source lists 6, not a power of 2]`; with more than one value the vectorizer chooses by cost model decision.
- **NOTE:** use `pragma vector` with care; override efficiency heuristics only if absolutely sure vectorization improves performance.

```c
void vec_aligned(float *a, int m, int c) {
  // Alignment unknown but compiler will still generate aligned load/store instructions
  #pragma vector aligned
  for (i = 0; i < m; i++) a[i] = a[i] * c; }

void vec_always(int *a, int *b, int m) {
  #pragma vector always
  for(int i = 0; i <= m; i++) a[32*i] = b[99*i]; }

float a[1000];
void foo(int N){ int i;
  #pragma vector nontemporal
  for (i = 0; i < N; i++) { a[i] = 1; } }
```
Generated assembly for `vector nontemporal` (for large `N`, significant gains result on systems with processors that have Streaming SIMD Extensions (SSE) support over non-streaming implementations):
```text
  .B1.2:
movntps XMMWORD PTR _a[eax], xmm0
movntps XMMWORD PTR _a[eax+16], xmm0
add eax, 32
cmp eax, ebx
jl .B1.2
```
See Also: Function Annotations and the SIMD Directive for Vectorization.

## Supported OpenMP* Pragmas

Compiler supports **OpenMP* 5.0 Version TR4, and some OpenMP Version 5.1 pragmas**; details are in the OpenMP Application Program Interface Version 5.1 specification. Intel-specific clauses are noted per pragma.

## Alphabetical List of Supported OpenMP* Pragmas

| Pragma | Description |
|---|---|
| `omp allocate` | memory allocators for allocation/deallocation |
| `omp atomic` | atomic computation |
| `omp barrier` | wait until all team threads arrive |
| `omp cancel` | cancel the innermost enclosing region of the specified type; encountering task goes to the end of the cancelled construct |
| `omp cancellation point` | tasks check for cancellation of the innermost enclosing region of the specified type; no thread/task synchronization |
| `omp critical` | block accessible by one thread at a time |
| `omp declare reduction` | user-defined reduction (UDR) functions (reduction identifiers) for a `reduction` clause |
| `omp declare simd` | SIMD function version processing multiple arguments per invocation from a SIMD loop |
| `omp declare target` | functions/variables created or mapped to a device |
| `omp declare variant` | variant of a base procedure and its context |
| `omp dispatch` | whether a procedure variant is called for a given procedure |
| `omp distribute` | iterations distributed among the initial threads of all thread teams in a league |
| `omp distribute parallel for` | parallel loop across threads of multiple teams |
| `omp distribute parallel for simd` | same, concurrently using SIMD |
| `omp distribute simd` | distributed across the teams region's primary threads, using SIMD |
| `omp flush` | a thread's temporary view of memory becomes consistent with memory |
| `omp for` | work-sharing loop; iterations executed in parallel by the team's threads |
| `omp for simd` | iterations split across team threads, each thread's iterations also concurrent using SIMD |
| `omp interop` | foreign runtime context and its runtime characteristics |
| `omp loop` | associated loop iterations can execute in any order or concurrently |
| `omp masked` | structured block for a subset of the current team's threads |
| `omp master` **(deprecated; see `omp masked`)** | block executed only once by the primary thread of the team |
| `omp ordered` | team executes in the natural order of loop iterations; stand-alone, cross-iteration dependences in a doacross loop-nest. Intel-specific clauses: `ompx_monotonic` — new list item value on each iteration of the associated SIMD loop(s) = original list item before entering the loop + (number of iterations where the conditional update happens before the current one) × `linear-step`; the sequentially last iteration's value is assigned to the original list item; use with `simd`. `ompx_overlap` — block that executes scalar for overlapping `inx` values and parallel for different `inx` values within a SIMD loop; use with `simd`. |
| `omp parallel` | structured block run in parallel by a team of threads |
| `omp parallel for` | abbreviated parallel region containing only a FOR construct |
| `omp parallel for simd` | parallel construct containing one `for simd` construct and no other statement |
| `omp parallel sections` | parallel construct containing only a `sections` construct |
| `omp requires` | features an implementation must support for correct compilation and execution |
| `omp scan` | scan computation updating each list item in each iteration of an enclosing SIMD loop nest |
| `omp scope` | block executed by all team threads, with additional OpenMP* operations specifiable |
| `omp sections` | structured blocks distributed among the team's threads |
| `omp simd` | loop executed concurrently using SIMD instructions. Intel-specific clause `ompx_assert` — compiler generates an error message if the loop is not vectorized for any reason |
| `omp single` | block executed by only one thread in the team |
| `omp target` | creates a device data environment and executes the construct on that device |
| `omp target data` | maps variables to a device data environment for the region's extent |
| `omp target enter data` | variables are mapped to a device data environment |
| `omp target exit data` | variables are unmapped from a device data environment |
| `omp target parallel` | creates a device data environment and executes the parallel region on that device |
| `omp target parallel for` | abbreviated target construct containing an `omp target parallel for` construct and no other statement |
| `omp target parallel for simd` | target construct containing an `omp target parallel for simd` construct and no other statement |
| `omp target parallel loop` | abbreviated target region containing only a parallel loop construct |
| `omp target simd` | target construct containing an `omp simd` construct and no other statement |
| `omp target teams` | device data environment executed on the same device; league of thread teams with each team's primary thread executing the block |
| `omp target teams distribute` | device data environment; iterations distributed among the primary threads of all thread teams in a league from a `teams` construct |
| `omp target teams distribute parallel for` | as above, plus a loop executable in parallel by threads of multiple teams from a `teams` construct |
| `omp target teams distribute parallel for simd` | as above; the loop is distributed across the teams, executed concurrently using SIMD |
| `omp target teams distribute simd` | device data environment; iterations distributed among the primary threads of all thread teams in a league from a `teams` construct, concurrently using SIMD |
| `omp target teams loop` | abbreviated target region containing only a `teams loop` construct |
| `omp target update` | makes device data environment list items consistent with their original list items |
| `omp task` | code block whose execution may be deferred |
| `omp taskgroup` | wait until all enclosed and descendant tasks complete |
| `omp taskloop` | iterations of associated `for` loops executed using OpenMP tasks |
| `omp taskloop simd` | loop executed concurrently using SIMD whose iterations also execute in parallel using OpenMP* tasks |
| `omp taskwait` | wait for child tasks generated since the beginning of the current task |
| `omp taskyield` | current task can be suspended here in favor of a different task |
| `omp teams` | league of thread teams inside a target region, executing a block in each team's initial thread |
| `omp teams distribute` | league of thread teams; iterations distributed among the primary threads of all thread teams in the league |
| `omp teams distribute parallel for` | league of thread teams; the associated loop can execute in parallel by threads of multiple teams |
| `omp teams distribute parallel for simd` | as above, concurrently using SIMD in parallel by threads of multiple teams |
| `omp teams distribute simd` | league of thread teams; the loop is distributed across the primary threads of the teams and executed concurrently using SIMD |
| `omp teams loop` | abbreviated `teams` construct containing only a `loop` construct |
| `omp threadprivate` | globally-visible variables allocated private to each thread |

## Categories of Supported OpenMP* Pragmas

- **Parallelism:** `omp parallel`. **Tasking:** `omp task`, `omp taskloop`.
- **Worksharing:** `omp for`, `omp loop`, `omp scope`, `omp sections`, `omp single`.
- **Synchronization:** `omp atomic`, `omp barrier`, `omp critical`, `omp flush`, `omp masked`, `omp master` (deprecated, see `omp masked`), `omp ordered`, `omp taskgroup`, `omp taskwait`, `omp taskyield`.
- **Data Environment:** `omp threadprivate`.
- **Offload Target Control:** `omp declare target`, `omp declare variant`, `omp dispatch`, `omp distribute`, `omp interop`, `omp requires`, `omp target`, `omp target data`, `omp target enter data`, `omp target exit data`, `omp target update`, `omp teams`.
- **Vectorization:** `omp scan`, `omp simd`, `omp declare simd`. **Cancellation:** `omp cancel`, `omp cancellation point`. **User-Defined Reduction:** `omp declare reduction`. **Memory Space Allocation:** `omp allocate`.
- **Combined and Composites:** `omp distribute parallel for`¹, `omp distribute parallel for simd`¹, `omp distribute simd`¹, `omp for simd`¹, `omp parallel for`, `omp parallel for simd`, `omp parallel sections`, `omp target parallel`, `omp target parallel for`, `omp target parallel for simd`, `omp target parallel loop`, `omp target simd`, `omp target teams`, `omp target teams distribute`, `omp target teams distribute parallel for`, `omp target teams distribute parallel for simd`, `omp target teams distribute simd`, `omp target teams loop`, `omp taskloop simd`¹, `omp teams distribute`, `omp teams distribute parallel for`, `omp teams distribute parallel for simd`, `omp teams distribute simd`, `omp teams loop`. **Combined constructs** are shortcuts for one construct immediately nested inside another (identical semantics to explicitly nesting); **composite constructs** (¹) add semantics or use non-conforming nesting.

## Pragmas Compatible with Other Compilers

## Pragmas Compatible with the Microsoft* Compiler

| Pragma | Description |
|---|---|
| `alloc_text` | code section where specified function definitions reside |
| `bss_seg` | segment for uninitialized variables in the `.obj` file |
| `code_seg` | code section where functions are allocated |
| `comment` | places a comment record into an object/executable file |
| `component` | controls collecting of browse/dependency information from source files |
| `const_seg` | segment where functions are stored in the `.obj` file |
| `data_seg` | default section for initialized data |
| `fenv_access` | a program may test status flags or run under a non-default control mode |
| `float_control` | floating-point behavior for a function |
| `fp_contract` | allows or disallows the implementation to contract expressions |
| `init_seg` | section containing C++ initialization code for the translation unit |
| `message` | displays the specified string literal to stdout |
| `optimize` | optimizations for functions below the pragma or until the next `optimize` pragma; **partly** supports the Microsoft implementation |
| `pointers_to_members` | whether a pointer to a class member can be declared before its class definition; controls pointer size and interpretation code |
| `pop_macro` | sets the specified macro to the value on top of the stack |
| `push_macro` | saves the specified macro's value on top of the stack |
| `region`/`endregion` | Microsoft Visual Studio* Code Editor code segment that expands/contracts via outlining |
| `section` | creates a section in an `.obj` file; stays valid for the rest of the compilation |
| `vtordisp` | `on` enables hidden `vtordisp` members, `off` disables them; `push` pushes the current setting onto the internal compiler stack, `pop` removes the top record and restores the removed value |
| `warning` | selective modification of compiler warning message behavior |

## Pragmas Supported for GCC-Compatible Compilers

| Pragma | Description |
|---|---|
| `poison` | labels identifiers to remove; compiling a "poisoned" identifier errors; `#pragma POISON` also supported; GCC-compatible |
| `options` | sets the alignment of fields in structures |

## Gotchas & failure modes

- **`omp target variant dispatch` removed** — migrate to `omp dispatch`; the procedure must appear in an `omp declare variant` in the calling scope's specification part, its interface accessible in the base procedure. **`omp master` deprecated** — migrate to `omp masked`.
- **`#pragma vector aligned` can silently generate incorrect code** for unaligned accesses; at run time an exception, not a compile error. Only one of `aligned`/`unaligned`.
- **`vector nontemporal` needs manual fences** (`_mm_sfence` after the loops is typical); misuse of streaming stores can degrade performance. **Dynamic alignment off by default** — `dynamic_align` only permits it; `nodynamic_align` disables it.
- **`ivdep` ignores only assumed dependencies**; unsafe use yields wrong code. **`block_loop` ignores loop-carried dependences** — blocking can change results where a real dependence exists.
- **`distribute_point`** cannot be inside an `if` when used in a loop; inside a loop it ignores all loop-carried dependencies, outside it observes data dependency.
- **Optimization gates:** `prefetch`/`noprefetch` and `unroll`/`nounroll` need `O2`+; `unroll_and_jam`/`nounroll_and_jam` needs `O3` and is ineffective on innermost loops. `unroll` and `unroll_and_jam` override the command-line loop-unrolling setting; `unroll_and_jam (n)` range is 0–255 (omitted/out-of-range → optimizer chooses).
- **Host-only Intel pragmas:** `block_loop`, `distribute_point`, `inline`/`forceinline`/`noinline`, `ivdep`, `loop_count`, `nofusion`, `novector`, `prefetch`/`noprefetch`, `unroll_and_jam`/`nounroll_and_jam`; only `unroll`/`nounroll` cover host and device.
- **Hardware-scoped:** `ompx prefetch data` — Xe MAX GPU only; `prefetch`/`noprefetch` — Intel® AVX-512 only.
- **Pragmas beat options** where a pragma duplicates a compiler option. **`vector` does not propagate to nested loops** — each nested loop needs its own preceding pragma.
- **No "Syntactic and Semantic Errors" material exists in pp. 595–621**; the range ends with the GCC-compatible pragma tables.

## Source map

- Predefined-macro compiler-detection examples — p. 595; Pragmas intro, Use Pragmas, Intel-Specific Pragma Reference table — pp. 595–597
- `block_loop`/`noblock_loop` — pp. 597–598; `distribute_point` — pp. 598–600; `inline`/`noinline`/`forceinline` — pp. 600–602
- `ivdep` — p. 602; `loop_count` — pp. 603–604; `nofusion` — p. 604; `novector` — pp. 604–605
- `omp target variant dispatch` — pp. 605–606; `ompx prefetch data` — pp. 606–607; `prefetch`/`noprefetch` — pp. 607–608
- `unroll`/`nounroll` — pp. 608–609; `unroll_and_jam`/`nounroll_and_jam` — pp. 609–610; `vector` — pp. 611–613
- Supported OpenMP* Pragmas alphabetical list — pp. 614–617; categories — pp. 618–620
- Pragmas Compatible with Other Compilers (Microsoft*, GCC-compatible) — pp. 620–621
