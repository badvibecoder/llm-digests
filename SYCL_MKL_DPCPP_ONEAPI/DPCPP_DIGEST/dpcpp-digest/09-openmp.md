---
chunk: 09-openmp
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0
source_pages: 666-733
covers: OpenMP* support in the Intel® oneAPI DPC++/C++ Compiler — pragma syntax, compile options, parallel model, worksharing, tasks, scheduling, reductions, thread allocation, runtime library and Intel extension routines, performance/stubs libraries, affinity interface and topology, memory spaces/allocators, contexts/selectors, SPMD/SIMD offload models, advanced issues, implementation-defined behaviors, examples
---

# OpenMP* Support

> **Scope.** This chunk lets a reader add OpenMP* to a DPC++/C++ build (`-qopenmp`/`/Qopenmp`, `spir64` targets), reason about the parallel/worksharing/tasking model, control thread count and affinity (`KMP_HW_SUBSET`, `KMP_AFFINITY`, `GOMP_CPU_AFFINITY`, `OMP_PROC_BIND`), choose performance vs stubs libraries, and understand OpenMP contexts/selectors, memory spaces/allocators, and the SPMD/SIMT vs SIMD offload models. Every runtime library and Intel extension routine and every documented default is listed.

## Key facts

- Compliance: **OpenMP C++ API specification 5.0**, **most of OpenMP 5.1 and 5.2**, **some of OpenMP 6.0 Version TR12**.
- Enable with **`-qopenmp` (Linux*)** / **`/Qopenmp` (Windows*)**; works with `-O0` (Linux) / `/Od` (Windows) and any of `O1`, `O2`, `O3`. Without it pragmas are comments → single-threaded build, no source change.
- Target offload (OpenMP 4.0+): **`-fopenmp-targets=spir64`** (Linux), **`/Qopenmp-targets=spir64`** (Windows). "Basics of Compilation" prints the Windows form `/Qopenmp-targets:spir64` [sic: source inconsistent].
- The compiler defines **`_OPENMP`**.
- Intel extensions = runtime routines + environment variables. **A runtime routine call overrides the corresponding environment variable.**
- Available on Intel® and non-Intel microprocessors, but may add Intel® optimizations. Constructs/features that may differ: **locks (internal and user visible), the SINGLE construct, barriers (explicit and implicit), parallel loop scheduling, reductions, memory allocation, and thread affinity and binding.**
- OpenMP implementations are not guaranteed interoperable across compilers.
- The Intel OpenMP runtime creates as many threads as available logical processors unless `omp_set_num_threads()` is used.
- `#include <omp.h>`; headers are in `../include` (Linux*) or `..\include` (Windows*) of the compiler installation.

## Option / API quick table

| name | purpose | key values/default | notes |
|---|---|---|---|
| `-qopenmp` / `/Qopenmp` | Enable OpenMP code generation | — | Omitting makes pragmas comments |
| `-fopenmp-targets=spir64` / `/Qopenmp-targets=spir64` | Compile `target` regions for offload | `spir64` | Required for GPU offload |
| `-fiopenmp` | Enable OpenMP with offload model options | — | With `-fopenmp-targets=spir64` |
| `-fopenmp-target-simd` / `/Qopenmp-target-simd` | Select offloading SIMD model | — | SIMD1 kernel + explicit SIMD inside kernel |
| `-qopenmp-stubs` / `/Qopenmp-stubs` | Serial stubs for OpenMP + Intel routines | — | For OpenMP-API code built serially |
| `-qopenmp-link` | Linux static/dynamic OpenMP lib | `dynamic` (default); `static` | Static "not recommended" |
| `KMP_HW_SUBSET` | Hardware resources used | e.g. `2s,12c,2t` | Warns+ignores if more than system has |
| `KMP_AFFINITY` | Bind threads to hardware | defaults `noverbose`, `respect`, `granularity=core`; type `none` | Set before first parallel region/before some API calls |
| `GOMP_CPU_AFFINITY` | Explicit OS proc ID list (Linux) | implies `granularity=fine` | Alias for `KMP_AFFINITY="granularity=fine,proclist=[<proc_list>],explicit"` |
| `KMP_TOPOLOGY_METHOD` | Force topology modeling | `cpuid_leaf31`, `cpuid_leaf11`, `cpuid_leaf4`, `cpuinfo`, `group`, `flat`, `hwloc` | — |
| `KMP_CPUINFO_FILE` | Corrected cpuinfo for topology | `<temp_file>` | Overrides `/proc/cpuinfo` and APIC decoding; works on Windows |
| `KMP_LIBRARY` | Runtime execution mode | `throughput` (default), `turnaround`, `serial` | — |
| `KMP_STACKSIZE` / `KMP_BLOCKTIME` | Thread private stack; sleep wait | — | Also `kmp_set_stacksize_s()`, `kmp_set_blocktime()` |
| `OMP_NUM_THREADS` | Threads for parallel regions | — | With `KMP_HW_SUBSET` can over/under subscribe |
| `OMP_SCHEDULE` | Schedule for `schedule(runtime)` | string as on the parallel construct | Invalid → `run-sched-var` ICV `static` |
| `OMP_PROC_BIND` | Thread affinity policy | `true`, `false`, list of `master` (deprecated), `primary`, `close`, `spread` | Invalid → `bind-var` ICV `false` |
| `OMP_THREAD_LIMIT`, `OMP_MAX_ACTIVE_LEVELS` | Program thread limit; nested active region limit | — | See Implementation-Defined Behaviors |
| `OMP_STACKSIZE` | Per-thread stack | default 1MB–4MB by architecture; up to 256MB on Linux* | — |
| `OMP_DYNAMIC`, `OMP_NESTED`, `OMP_CANCELLATION` | Dynamic threads; nested parallelism; cancellation | disabled by default | `omp_get_cancellation()` affected by `OMP_CANCELLATION` |

## OpenMP* Support

OpenMP provides symmetric multiprocessing (SMP): it removes low-level iteration-space partitioning, data sharing, thread creation, scheduling, and synchronization, and benefits shared-memory multiprocessor/multi-core systems including Intel® Hyper-Threading Technology (Intel® HT Technology). The compiler transforms code per pragma placement; industry-standard pragmas are supported plus Intel-specific extensions (runtime routines, environment variables). **Parallel Processing with OpenMP**: the compiler produces a multithreaded version whose parallelism is implemented by threads executing parallel regions/constructs. **Using Other Compilers**: the OpenMP specification does not define interoperability of implementations, and different compilers might not provide OpenMP source compatibility, so the same sources may not yield the expected parallel results.

### Add OpenMP* Support

1. Add the appropriate OpenMP pragmas.
2. Compile with `-qopenmp` (Linux*) or `/Qopenmp` (Windows*) to enable recognition of OpenMP parallel and loop transformation pragmas.
3. For large local/temporary arrays, raise runtime stack space and the per-thread stack with `OMP_STACKSIZE` or the corresponding library routines. Other environment variables can control multi-threaded execution.

## OpenMP Pragma Syntax

Header: `#include <omp.h>`. Syntax:

```text
<prefix> <pragma> [<clause>, ...] <newline>
```

- `<prefix>` — required; must be `#pragma omp`.
- `<pragma>` — valid OpenMP pragma immediately after the prefix.
- `[<clause>]` — optional; any order, repeatable unless otherwise restricted.
- `<newline>` — required; precedes the enclosed structured block.

Pragmas are comments if `/Qopenmp` (Windows) or `-qopenmp` (Linux) is omitted.

```c
#include <omp.h>
void simple_omp(int *a){
  int i;
  #pragma omp parallel for
  for (i=0; i<1024; i++)
    a[i] = i*2;
}
```

**Compile the Application.** `-qopenmp`/`/Qopenmp` enable the parallelizer; code runs on single-processor, multi-processor, or multi-core systems.

```bash
icpx -qopenmp source_file
icpx -qopenmp -c parallel.cpp
```
```bash
icpx -qopenmp -fopenmp-targets=spir64 offload.cpp    # Linux target offload
```
```text
icx /Qopenmp source_file
icx /Qopenmp /c parallel.c
icx /Qopenmp /Qopenmp-targets=spir64 offload.c       # Windows target offload
```

**Configure the OpenMP Environment.** Set `OMP_NUM_THREADS` before running. See also `c` compiler option, `O` compiler option, OpenMP* Examples, `qopenmp, Qopenmp` compiler option, Supported Environment Variables.

## Parallel Processing Model

- Execution begins as one **initial thread of execution**, sequential until the first parallel construct.
- `omp parallel` defines the parallel construct. The initial thread creates a **team of threads** and becomes its **primary thread**; all enclosed statements run in parallel by every thread, including called routines.
- **Static extent** = statements enclosed lexically in a construct. **Dynamic extent** = all statements encountered during execution of a construct, including called routines.
- At the end of the structured block a thread waits for all team threads; the team is dissolved and only the primary thread continues. Other threads wait until needed for another team. Teams can be created/dissolved many times.

```c
main() {                         // Begin serial execution.
  ...                            // Only the initial thread executes
  #pragma omp parallel           // Begin a parallel construct and form a team.
  {
    #pragma omp sections         // Begin a worksharing construct.
    {
        #pragma omp section      // One unit of work.
       {...}
       #pragma omp section       // Another unit of work.
        {...}
    }                            // Wait until both units of work complete.
    ...                          // This code is executed by each team member.
    #pragma omp for nowait       // Begin a worksharing Construct
           for(...) {            // Each iteration chunk is unit of work.
             ...                  // Work is distributed among the team members.
           }                      // End of worksharing construct.
                                  // nowait was specified so threads proceed.
           #pragma omp critical   // Begin a critical section.
           {...}                  // Only one thread executes at a time.
           ...                    // This code is executed by each team member.
           #pragma omp barrier    // Wait for all team members to arrive.
           ...                    // This code is executed by each team member.
       }                          // End of Parallel Construct
                                  // Disband team and continue serial execution.
       ...                        // Possibly more parallel constructs.
   }                              // End serial execution.
```

**Use Orphaned Pragmas.** Pragmas in the dynamic but not static extent of the parallel construct are **orphaned pragmas**; they let you parallelize constructs at the top of the call tree and control called routines with minimal changes to the sequential program.

```c
int main(void) {
  #pragma omp parallel { phase1(); }
}
void phase1(void) {
  #pragma omp for // This is an orphaned pragma.
  for(i=0; i < n; i++) { some_work(i); }
}
```

The `omp for` is orphaned because the parallel region is not lexically present in `phase1`.

**Data Environment.** Control it with the construct's clauses; privatize named global-lifetime objects with `threadprivate`. Commonly used clauses (full list in the OpenMP specification): `default`, `shared`, `private`, `firstprivate`, `lastprivate`, `reduction`, `linear`, `map`. With no data scope attribute clause, behavior follows the OpenMP specification's default scoping rules.

## Determine How Many Threads to Use

- Delay the thread-count decision to runtime when workload depends on input (matrix size, database size, image/video size and resolution, depth/breadth/bushiness of tree structures, size of list-based structures) or when processor count varies widely.
- For unpredictable work, use a calibration step; expensive results can be stored persistently (e.g., file system).
- Avoid more threads than processing units: the OS multiplexes threads and performance is typically sub-optimal.
- Libraries should let users select the thread count; outer-level parallelism may make library parallelism unnecessary or disruptive.
- Use `num_threads` on parallel regions for thread count and `if` to decide whether to use multiple threads at all. `omp_set_num_threads()` also affects parallel regions created by the calling thread; `num_threads` is local.
- Explicit counts: (1) on many-processor systems you use only some processors; (2) on few-processor systems you may force oversubscription and poor performance.
- The runtime creates as many threads as available logical processors unless `omp_set_num_threads()` is used. `omp_get_thread_limit()` and `omp_get_max_active_levels()` report limits. `OMP_THREAD_LIMIT` limits the whole program; `OMP_MAX_ACTIVE_LEVELS` limits active nested parallel regions.

**Binding Sets and Binding Regions.**
- **Binding task set**: tasks affected by, or providing context for, execution of the region — all tasks, current team tasks, all current-team tasks generated in the region, the binding implicit task, or the generating task.
- **Binding thread set**: threads affected by, or providing context for, the region — all threads on a device, all threads in a contention group, all primary threads executing an enclosing teams region, the current team, or the encountering thread.
- **Binding region**: the enclosing region determining execution context and effect scope. `omp ordered` → innermost enclosing `omp for` loop region; `omp taskwait` → innermost enclosing `omp task` region; other constructs whose binding thread set is the current team or binding task set is the current team tasks → innermost enclosing region; constructs whose binding task set is the generating task → that task's region.
- An `omp parallel` construct need not be active to be a binding region; a construct need not be explicit; a region never binds outside the innermost enclosing parallel region.

## Worksharing Using OpenMP*

Worksharing distributes work across threads. Most loops with no loop-carried dependencies can be threaded with one statement immediately before the loop; maximum performance comes from threading **hotspots** (most time-consuming loops). Options using OpenMP exist for Intel® and non-Intel microprocessors but may add Intel® optimizations (construct list as in Key facts).

```c
#pragma omp parallel for
for (i=0; i < numPixels; i++) {
  pGrayScaleBitmap[i] = (unsigned BYTE)
    (pRGBBitmap[i].red * 0.299 +
     pRGBBitmap[i].green * 0.587 +
     pRGBBitmap[i].blue * 0.114);
}
```

With `for`, each iteration executes exactly once (on a different thread if available). `for` alone distributes only among **existing** threads; `parallel for` first creates a team. With no `num_threads` clause, OpenMP decides thread creation/synchronization/destruction.

**Five restrictions on which loops can be threaded:**
1. Loop variable signed/unsigned integer, random access iterator, or pointer.
2. Comparison `loop_variable <, <=, >, >=, or != loop_invariant_expression` of compatible type.
3. Increment must be addition or subtraction by a loop invariant value.
4. With `<` or `<=` the variable must increment every iteration; with `>` or `>=` it must decrement every iteration.
5. Body must be single-entry-single-exit: no jumps from inside to outside except the exit statement terminating the whole application; `goto`/`break` must stay within the loop; exceptions must be caught within the loop.

Non-conforming loops can frequently be rewritten to satisfy these.

**Basics of Compilation.** OpenMP pragmas need an OpenMP-compatible compiler and thread-safe libraries. `-qopenmp`/`/Qopenmp` generates multi-threaded code; omitting it ignores pragmas. For GPU offload of `target` constructs, `-fopenmp-targets=spir64` (Linux) and `/Qopenmp-targets:spir64` (Windows) are needed [sic: Windows spelling as printed in this topic].

```c
#ifdef _OPENMP
   fn();
#endif
```

**A Few Simple Examples.** Array clip to `0 <= x <= 255`, serial then threaded:

```c
for (i=0; i < numElements; i++) {
  if (array[i] < 0) array[i] = 0;
  else if (array[i] > 255) array[i] = 255;
}
#pragma omp parallel for
for (i=0; i < numElements; i++) {
  if (array[i] < 0) array[i] = 0;
  else if (array[i] > 255) array[i] = 255;
}
```

Square-root table 0..100 — change the loop variable to a signed or unsigned integer and add the pragma:

```c
double value; double roots[100];                       // serial
for (value = 0.0; value < 100.0; value ++) { roots[(int)value] = sqrt(value); }
int value; double roots[100];                          // threaded
#pragma omp parallel for
for (value = 0; value < 100; value ++) { roots[value] = sqrt((double)value); }
```

**Avoid Data Dependencies and Race Conditions.** Data dependencies exist when different iterations (specifically on different threads) read or write the same shared-memory location. The compiler threads this factorial loop but threading fails — a **race condition**:

```c
// Each loop iteration writes a value that a different iteration reads.
#pragma omp parallel for
for (i=2; i < 10; i++) { factorial[i] = i * factorial[i-1]; }
```

Races require shared resources (memory) and parallel execution; fix by rewriting the loop or choosing an algorithm without the race. They are hard to detect because threads may win in an order that happens to work — working once does not mean working always. Test on varied machines (Intel® HT Technology, multiple physical processors). Traditional debuggers are useless (they stop one thread while others continue, changing behavior); thread checking tools can help.

**Manage Shared and Private Data.** Shared = all threads access the same location. Private = a separate copy per thread, destroyed when the loop ends. **By default all variables are shared except the loop variable, which is private.** Make memory private by (1) declaring the variable inside the loop — really inside the parallel pragma — without `static`, or (2) specifying the `private` clause.

```c
// WRONG: temp is shared; one thread may write while another reads.
#pragma omp parallel for
for (i=0; i < 100; i++) { temp = array[i]; array[i] = do_something(temp); }

// RIGHT: variables declared within a parallel construct are by definition private
#pragma omp parallel for
for (i=0; i < 100; i++) { int temp; temp = array[i]; array[i] = do_something(temp); }

// RIGHT: explicit clause
#pragma omp parallel for private(temp)
for (i=0; i < 100; i++) { temp = array[i]; array[i] = do_something(temp); }
```

Examine all memory references, including called functions. Variables declared within a parallel construct are private **except** with `static`, because static variables are not allocated on the stack.

**Reductions.** `sum` must be shared for correctness but private to avoid races; `reduction` efficiently combines the reduction of one or more variables.

```c
sum = 0;
#pragma omp parallel for reduction(+:sum)
for (i=0; i < 100; i++) { sum += array[i]; }
```

Private copies of `sum` are made per thread; on exit their values are combined into the one global copy.

| Operation | private Variable Initialization Value |
|---|---|
| `+` (addition) | 0 |
| `-` (subtraction) | 0 |
| `*` (multiplication) | 1 |
| `&` (bitwise and) | `~0` |
| `\|` (bitwise or) | 0 |
| `^` (bitwise exclusive or) | 0 |
| `&&` (conditional and) | 1 |
| `\|\|` (conditional or) | 0 |

Multiple reductions per loop use comma-separated variables and operations. Reduction variables: can be listed in just one reduction; cannot be declared constant; cannot be declared private in the parallel construct.

**Load Balancing and Loop Scheduling.** Load balancing keeps processors busy; poor balancing in loops often comes from variation in iteration compute time. Find sets of iterations with similar cost (all even vs all odd; first half vs second half). If iterations are roughly uniform, use `schedule` to distribute them in roughly equal amounts. Large chunks reduce memory conflicts/false sharing (loops touch memory sequentially) but can hurt load balancing; the reverse is also true — measure both.

```c
#pragma omp parallel for schedule(kind [, chunk size])
```

Source says "Four different loop scheduling types (kinds)" but the table lists five [sic]. The optional `chunk`, when specified, must be a positive integer.

| Kind | Description |
|---|---|
| `static` | Equal-sized chunks or as equal as possible when iterations are not evenly divisible by threads × chunk size. Default chunk size `loop_count/number_of_threads`. Set chunk to 1 to interleave iterations. |
| `dynamic` | Internal work queue gives a chunk-sized block per thread; a finished thread retrieves the next block from the top. Default chunk size 1. Careful: extra overhead. |
| `guided` | Like `dynamic` but chunk size starts large and decreases to handle imbalance; the optional chunk is the minimum chunk size. Default approximately `loop_count/number_of_threads`. |
| `auto` | Scheduling delegated to the compiler; it may choose any mapping of iterations to threads in the team. |
| `runtime` | Uses `OMP_SCHEDULE` (a string formatted exactly as it would appear on the parallel construct) to select one of the three loop-scheduling types. |

```c
for (i=0; i < NumElements; i++) { array[i] = StartVal; StartVal++; }   // dependency: cannot parallelize as-is
#pragma omp parallel for
for (i=0; i < NumElements; i++) { array[i] = StartVal + i; }           // no dependency; benefits from SIMD
```

The rewrite is not 100% identical because `StartVal` is not incremented; after the parallel loop it differs from the serial version. If the value is needed afterwards:

```c
// This works and is identical to the serial version.
#pragma omp parallel for
for (i=0; i < NumElements; i++) { array[i] = StartVal + i; }
StartVal += NumElements;
```

### OpenMP Tasking Model

A **task** is an instance of executable code and its data environment that can be scheduled for execution by threads.

**The task Construct.**

```c
void test1(LIST *head) {
  #pragma omp parallel shared(head)
  {
    #pragma omp single
     {
        LIST *p = head;
        while (p != NULL) {
          #pragma omp task firstprivate(p)
          { do_work1(p); }
          p = p->next;
        }
    }
  }
}
```

The binding thread set of the task region is the current parallel team; a task region binds to the innermost enclosing parallel region. On encountering a task construct a task is generated from the structured block; the encountering thread may immediately execute or defer it. A task may nest inside an outer task, but the inner task region is not part of the outer task region.

**Use Clauses with the task Construct.** The task construct takes optional clauses; the task data environment follows the data-sharing attribute clauses and any defaults. Generate N tasks with one thread and execute with the team:

```c
double data[N];
int i;
#pragma omp parallel shared(data)
{
  #pragma omp single private(i)
  {
    for (i=0, i<N; i++)
    {
       #pragma omp task firstprivate(i) shared(data))
       { do_work(data, i); }
    }
  }
}
```

**Task Scheduling.** At a **task scheduling point** a thread may switch tasks, suspending the current task and beginning or resuming a different task bound to the current team. The OpenMP 5.1 specifications list all locations; examples: where a task is explicitly generated; immediately following generation of an explicit task; after the last instruction of a task region; in a `taskwait` region; in a `taskyield` region; in implicit and explicit barrier regions.

> **NOTE.** Task scheduling points divide task regions into parts; each part runs start to finish without interruption. Parts of the same task region execute in the order encountered. Without task synchronization constructs the order across different schedulable tasks is unspecified. A correct program must behave correctly under all conceivable scheduling sequences.

**The taskwait Construct.** `taskwait` waits on child tasks generated since the beginning of the current task. It binds to the current task region; its binding thread set is the current team. It includes an implicit task scheduling point; the current task region is suspended until all child tasks generated before the `taskwait` region complete.

```c
#pragma omp task // TASK1
{
  ...
  #pragma omp task // TASK 2 (child of TASK1)
  { do_work1(); }
  #pragma omp task // TASK3 (child of TASK 1)
  {
    ...
    #pragma omp task // TASK4 (child of TASK3, not TASK1)
    { do_work2(); }
    ...
  }
  #pragma omp taskwait // suspend TASK1; wait for TASK2 and TASK3 to complete
  ...
}
```

**The taskyield Construct.** `taskyield` allows the current task to be suspended at that point and the thread to switch to a different task; use it to provide an explicit task scheduling point at a particular point in the task. See also `OMP_SCHEDULE`, `qopenmp, Qopenmp`, Supported Environment Variables.

## Control Thread Allocation

`KMP_HW_SUBSET` controls allocation of hardware resources; `KMP_AFFINITY` controls how threads are bound to them.

**Control Thread Distribution.** `KMP_HW_SUBSET` often specifies three topology layers: sockets, cores per socket, threads per core.

- `KMP_HW_SUBSET=2s,12c,2t` = two sockets, 12 cores/socket, two threads/core = **48** hardware threads.
- More layers (NUMA domain, tile, etc.) may be specified: `KMP_HW_SUBSET=2s,2n,8c,2t` = two sockets, two NUMA domains/socket, eight cores/NUMA domain, two threads/core = **64** hardware threads.
- Historically an unspecified layer means **all** resources in that layer. Use `KMP_AFFINITY=verbose` to see detected layers. E.g. `KMP_HW_SUBSET=2s,2t` = two sockets, all cores per socket (possibly all resources of other detected layers), two threads per layer.
- Core attributes are appended to the core layer with a colon (`:`): (1) core type `intel_core` or `intel_atom`; (2) core efficiency `effnum`, `num` a non-negative integer from zero to the number of detected core efficiencies minus one, larger = more performant. E.g. `KMP_HW_SUBSET=4c:eff0,5c:eff1` selects all sockets, four efficiency-0 cores, five efficiency-1 cores, all threads per those cores.
- To request **all** resources at a layer, use an optional asterisk (`*`) instead of a positive integer: `KMP_HW_SUBSET=*c:eff0` or `KMP_HW_SUBSET=c:eff0` requests all cores of efficiency 0.

Example: 24 cores with four hardware threads each. Two threads/core often beats one; three or four may or may not improve performance.

| To Assign This Number of Threads ... | ... Use This Setting |
|---|---|
| 24 | `KMP_HW_SUBSET=24c,1t` |
| 48 | `KMP_HW_SUBSET=24c,2t` |
| 72 | `KMP_HW_SUBSET=24c,3t` |
| 96 | `KMP_HW_SUBSET=24c,4t` |

> **Caution.** Using `OMP_NUM_THREADS` with this variable can result in over or under subscription.

> **NOTE.** Specifying more resources than the system has makes the runtime warn and ignore the setting (e.g. `KMP_HW_SUBSET=24c,5t` where each core has four hardware threads).

**Control Thread Bindings.** `KMP_AFFINITY` binds threads to resources allocated by `KMP_HW_SUBSET`. Recommended types: `compact` — distribute threads sequentially among the cores; `scatter` — round robin among the cores, one thread per core initially then repeat.

`KMP_HW_SUBSET=2c,3t` (three threads/core on two cores):

| Affinity | OpenMP Threads on Core 0 | OpenMP Threads on Core 1 |
|---|---|---|
| `KMP_AFFINITY=compact` | 0, 1, 2 | 3, 4, 5 |
| `KMP_AFFINITY=scatter` | 0, 2, 4 | 1, 3, 5 |

**Determine the Best Setting.** (1) Ensure OpenMP code works first. (2) Establish a baseline. (3) Measure one/two/three/four threads per core with `KMP_HW_SUBSET`. (4) Measure binding threads to cores with `KMP_AFFINITY`. See also Thread Affinity Interface, Supported Environment Variables.

## OpenMP* Library Support

### OpenMP* Runtime Library Routines

Runtime routines manage parallel mode; many have corresponding environment variables settable as defaults. **A runtime routine call overrides any corresponding environment variable.** Running them may initialize the OpenMP runtime so later programmatic environment-variable settings have no effect — use the Intel extension `kmp_set_defaults()` instead. The compiler supports all OpenMP runtime library routines; include declarations with `#include <omp.h>` (headers in `../include` Linux* / `..\include` Windows*). OpenMP specification has details.

**Thread Team Routines** (thread teams in the current contention group): `void omp_set_num_threads(int nthreads)` threads for subsequent regions created by the calling thread; `int omp_get_num_threads(void)` threads in the current parallel region (not necessarily the `omp_set_num_threads()` value inherited); `int omp_get_max_threads(void)` threads available to subsequent regions created by the calling thread; `int omp_get_thread_num(void)` calling thread's number in the current region; `int omp_in_parallel(void)` TRUE in the dynamic extent of a parallel region executing in parallel, else FALSE; `void omp_set_dynamic(int dynamic_threads)` enable (TRUE)/disable (FALSE) dynamic thread adjustment (**disabled by default**); `int omp_get_dynamic(void)` TRUE if dynamic adjustment enabled; `int omp_get_cancellation(void)` TRUE if cancellation enabled (affected by `OMP_CANCELLATION`); `void omp_set_nested(int nested)` **(Deprecated)** enable/disable nested parallelism (**disabled by default**); `int omp_get_nested(void)` **(Deprecated)** TRUE if nested parallelism enabled; `void omp_set_schedule(omp_sched_t kind,int chunk_size)` schedule applied when `'runtime'` is the kind; `void omp_get_schedule(omp_sched_kind *kind,int *chunk_size)` returns that schedule; `int omp_get_thread_limit(void)` max simultaneously executing threads in the program; `int omp_get_supported_active_levels(void)` active parallelism levels supported; `void omp_set_max_active_levels(int max_active_levels)` limits nested active parallel regions (argument must be non-negative); `int omp_get_max_active_levels(void)` that maximum; `int omp_get_level(void)` nested regions (active or inactive) enclosing the task, excluding the implicit parallel region; `int omp_get_ancestor_thread_num(int level)` ancestor thread number at a nest level; `int omp_get_team_size(int level)` team size for an ancestor/current thread at a nested level; `int omp_get_active_level(void)` nested, active regions enclosing the task.

**Thread Affinity Routines** (affinity policies in effect): `omp_proc_bind_t omp_get_proc_bind(void)` active policy (initializable by `OMP_PROC_BIND`), used for subsequent nested regions; `int omp_get_num_places(void)` places in the initial task's place list (usually threads, cores, or sockets); `int omp_get_place_num_procs(int place_num)` processors in place `place_num`, zero if negative or `>= omp_get_num_places()`; `void omp_get_place_proc_ids(int place_num, int *ids)` non-negative identifiers of processors in that place, meaning and array order implementation defined, `ids` must hold `omp_get_place_num_procs(place_num)` elements, no effect out of range; `int omp_get_place_num(void)` place the encountering thread is bound to, 0..`omp_get_num_places()-1`, or `-1` if unbound; `int omp_get_partition_num_places(void)` places in the innermost implicit task's place partition; `void omp_get_partition_place_nums(int *place_nums)` place numbers of the place-partition-var ICV of the innermost implicit task; `void omp_set_affinity_format(const char *format)` sets affinity-format-var ICV on the device; `size_t omp_get_affinity_format(char *buffer, size_t size)` returns that ICV; `void omp_display_affinity(const char *format)` prints thread affinity info using the format; `size_t omp_capture_affinity(char *buffer, size_t size, const char *format)` writes it into a buffer.

**Teams Region Routines:** `int omp_get_num_teams(void)` initial teams in the current teams region; `int omp_get_team_num(void)` initial team number of the calling thread; `void omp_set_num_teams(int num_teams)` teams for subsequent teams regions without a `num_teams` clause; `int omp_get_max_teams(void)` upper bound on teams creatable by a later teams construct without `num_teams`; `void omp_set_teams_thread_limit(int thread_limit)` max threads per contention group created by a teams construct; `int omp_get_teams_thread_limit(void)` that maximum.

**Tasking Routines:** `int omp_get_max_task_priority(void)` max value for the `priority` clause; `int omp_in_explicit_task(void)` TRUE in an explicit task region; `int omp_in_final(void)` TRUE in a final task region.

**Resource Relinquishing Routines** (**host device only**): `int omp_pause_resource(omp_pause_resource_t kind, int device_num)` relinquish OpenMP resources on a device (zero on success, non-zero otherwise); `int omp_pause_resource_all(omp_pause_resource_t kind)` same for all devices.

**Device Information Routines:** `int omp_get_num_procs(void)` processors available; `void omp_set_default_device(int device_number)` sets default device number; `int omp_get_default_device(void)` returns it; `int omp_get_num_devices(void)` number of target devices; `int omp_get_device_num(void)` device the calling thread runs on; `int omp_is_initial_device(void)` TRUE if on the host device; `int omp_get_initial_device(void)` host device number (implementation defined; valid in all device constructs/routines only if within 0..`omp_get_num_devices()-1`, otherwise only in device memory routines and not the `device` clause).

**Device Memory Routines:** `void *omp_target_alloc(size_t size, int device_num)` allocate device memory, return device pointer; `void omp_target_free(void *device_ptr, int device_num)` free memory from `omp_target_alloc`; `int omp_target_is_present(const void *ptr, int device_num)` TRUE if `device_num` is host or `ptr` has corresponding storage in that device data environment; `int omp_target_is_accessible(const void *ptr, size_t size, int device_num)` TRUE if `size` bytes at `ptr` are accessible from that device; `int omp_target_memcpy(void *dst, const void *src, size_t length, size_t dst_offset, size_t src_offset, int dst_device_num, int src_device_num)` copy `length` bytes from `src`+`src_offset` (device `src_device_num`) to `dst`+`dst_offset` (device `dst_device_num`), zero on success, non-zero on failure, `omp_get_initial_device` references the host, **contains a task scheduling point, effect unspecified inside a target region**; `int omp_target_memcpy_rect(void *dst, const void *src, size_t element_size, int num_dims, const size_t *volume, const size_t *dst_offsets, const size_t *src_offsets, const size_t *dst_dimensions, const size_t *src_dimensions, int dst_device_num, int src_device_num)` copy a rectangular subvolume (`volume` = elements per dimension, offsets = elements from origin, dimensions = length per dimension), **max dimensions supported is three or more**, zero on success, non-zero otherwise, if both `dst` and `src` are NULL returns the number of dimensions supported for the device numbers, **contains a task scheduling point, effect unspecified inside a target region**; `int omp_target_associate_ptr(const void *host_ptr, const void *device_ptr, size_t size, size_t device_offset, int device_num)` map a device pointer (possibly from `omp_target_alloc`) to a host pointer; `int omp_target_disassociate_ptr(const void *ptr, int device_num)` remove that association; `void *omp_get_mapped_ptr(const void *ptr, int device_num)` device pointer associated with a host pointer.

**Lock Routines:** `void omp_init_lock(omp_lock_t *lock)` unlocked state; `void omp_init_nest_lock(omp_nest_lock_t *lock)` unlocked, nesting count zero; `void omp_init_lock_with_hint(omp_lock_t *lock, omp_sync_hint_t hint)` unlocked, optionally choose implementation from `hint` (hints per OpenMP specification); `void omp_init_nest_lock_with_hint(omp_nest_lock_t *lock, omp_sync_hint_t hint)` same, nesting count zero; `void omp_destroy_lock(omp_lock_t *lock)` uninitialized; `void omp_destroy_nest_lock(omp_nest_lock_t *lock)` uninitialized; `void omp_set_lock(omp_lock_t *lock)` wait until available then own; `void omp_set_nest_lock(omp_nest_lock_t *lock)` wait, increment nesting count if already owned; `void omp_unset_lock(omp_lock_t *lock)` release (**undefined if the thread does not own the lock**); `void omp_unset_nest_lock(omp_nest_lock_t *lock)` decrement nesting count, release at zero (**undefined if not owner**); `int omp_test_lock(omp_lock_t *lock)` TRUE on success else FALSE; `int omp_test_nest_lock(omp_nest_lock_t *lock)` nesting count on success else zero.

**Timing Routines:** `double omp_get_wtime(void)` elapsed wall clock seconds relative to an arbitrary reference time that does not change during execution; `double omp_get_wtick(void)` seconds between successive clock ticks.

**Event Routines:** `void omp_fulfill_event(omp_event_handle_t event)` fulfill and destroy the event.

**Interoperability Routines:** `int omp_get_num_interop_properties(const omp_interop_t interop)` number of implementation-defined properties (total = value minus `omp_ipr_first`); `omp_intptr_t omp_get_interop_int(const omp_interop_t interop, omp_interop_property_t property_id, int *ret_code)` requested integer property, zero on error/unavailable; `void *omp_get_interop_ptr(const omp_interop_t interop, omp_interop_property_t property_id, int *ret_code)` requested pointer property, NULL on error/unavailable; `const char *omp_get_interop_str(const omp_interop_t interop, omp_interop_property_t property_id, int *ret_code)` requested string property, NULL on error/unavailable; `const char *omp_get_interop_name(const omp_interop_t interop, omp_interop_property_t property_id)` property name; `const char *omp_get_interop_type_desc(const omp_interop_t interop, omp_interop_property_t property_id)` human-readable property type; `const char *omp_get_interop_rc_desc(const omp_interop_t interop, omp_interop_rc_t ret_code)` human-readable return code.

**Memory Management Routines:** `omp_allocator_handle_t omp_init_allocator(omp_memspace_handle_t memspace, int ntraits, const omp_alloctrait_t traits[])` create an allocator for `memspace`; `void omp_destroy_allocator(omp_allocator_handle_t allocator)` release it; `void omp_set_default_allocator(omp_allocator_handle_t allocator)` default for allocation calls, `allocate` directives and `allocate` clauses without an allocator; `omp_allocator_handle_t omp_get_default_allocator(void)` that handle; `void *omp_alloc(size_t size, omp_allocator_handle_t allocator)` allocate `size` bytes; `void *omp_aligned_alloc(size_t alignment, size_t size, omp_allocator_handle_t allocator)` allocate, byte-aligned to at least the max of `malloc`'s alignment, the allocator's alignment trait, and the alignment argument; `void omp_free(void *ptr, omp_allocator_handle_t allocator)` deallocate memory from an OpenMP allocation routine; `void *omp_calloc(size_t nmemb, size_t size, omp_allocator_handle_t allocator)` array of `nmemb` × `size` bytes; `void *omp_aligned_calloc(size_t alignment, size_t nmemb, size_t size, omp_allocator_handle_t allocator)` same with the aligned rule; `void *omp_realloc(void *ptr, size_t size, omp_allocator_handle_t allocator, omp_allocator_handle_t free_allocator)` deallocate at `ptr` and allocate `size` bytes from `allocator`, contents preserved up to the minimum of the old allocated size and `size`.

**Tool Control Routines:** `int omp_control_tool(int command, int modifier, void *arg)` pass commands to an active tool.

**Environment Display Routines:** `void omp_display_env(int verbose)` display the OpenMP version number and initial values of ICVs associated with environment variables.

**Device Runtime Routines Available on GPU** (CPU and GPU): `omp_get_device_num`, `omp_get_max_threads`, `omp_get_num_devices`, `omp_get_num_procs`, `omp_get_num_teams`, `omp_get_num_threads`, `omp_get_team_num`, `omp_get_team_size`, `omp_get_thread_limit`, `omp_get_thread_num`, `omp_in_parallel`, `omp_is_initial_device`.

### Intel® Compiler Extension Routines to OpenMP*

Groups: get/set the execution environment; get/set the stack size for parallel threads; memory allocation; get/set the thread sleep time for the throughput execution mode; target memory allocation. They are for low-level tuning and are generally not recognized by other OpenMP-compliant compilers, which may fail at the link stage. To execute them use `/Qopenmp-stubs` (Windows*) or `-qopenmp-stubs` (Linux*). Environment variables can usually replace them (e.g. `OMP_STACKSIZE` instead of `kmp_set_stacksize_s()`). **A runtime call to an Intel extension routine takes precedence over the corresponding environment variable setting.**

**Execution Environment:** `void kmp_set_defaults(char const *)` set OpenMP environment variables given as a `"|"`-separated list; `void kmp_set_library_throughput(void)` throughput mode (**the default**; lets the application determine the runtime environment; multi-user environments); `void kmp_set_library_turnaround(void)` turnaround mode (dedicated parallel/single-user environments); `void kmp_set_library_serial(void)` serial mode; `void kmp_set_library(int)` mode by value — **1** Serial, **2** Turnaround, **3** Throughput — call before the first parallel region is executed; `int kmp_get_library(void)` current mode with the same 1/2/3 values.

**Stack Size:** `size_t kmp_get_stacksize_s(void)` bytes allocated per parallel thread as its private stack (settable with `kmp_set_stacksize_s()` before the first parallel region or via `KMP_STACKSIZE`); `int kmp_get_stacksize(void)` **backwards compatibility only**, use `kmp_get_stacksize_s()`; `void kmp_set_stacksize_s(size_t size)` set that size, also via `KMP_STACKSIZE`, must be called before the first (dynamically executed) parallel region to take effect; `void kmp_set_stacksize(int size)` **backward compatibility only**, use `kmp_set_stacksize_s()`.

**Memory Allocation:** `kmp_malloc()`, `kmp_calloc()`, and `kmp_realloc()` allocate from a heap local to each thread; memory **must also be freed by `kmp_free()`**. Cross-thread allocate-then-free works but costs a slight performance penalty. `void* kmp_malloc(size_t size)`; `void* kmp_calloc(size_t nelem, size_t elsize)`; `void* kmp_realloc(void* ptr, size_t size)`; `void* kmp_free(void* ptr)` (memory must come from one of the three above).

**Thread Sleep Time.** In the throughput libraries, threads wait at the ends of parallel regions and then sleep after a period set by `KMP_BLOCKTIME` or `kmp_set_blocktime()`. `int kmp_get_blocktime(void)` milliseconds waited after completing a parallel region before sleeping; `void kmp_set_blocktime(int msec)` sets it, affecting the calling thread and any OpenMP team threads it forms but **not** other threads.

**Target Memory Allocation:** `void *omp_target_alloc_host(size_t size, int device_num)` `size` bytes in host memory, same pointer usable on the host and all supported devices, null on failure; `void *omp_target_alloc_device(size_t size, int device_num)` `size` bytes owned by `device_num` in device memory if present, generally device-only but copyable, null on failure; `void *omp_target_alloc_shared(size_t size, int device_num)` `size` bytes shared by host and the specified device and intended to migrate, null on failure; `void *ompx_target_realloc(void *ptr, size_t size, int device_num)` reallocate device memory, device-only, contents preserved up to min(old size, `size`); `void *ompx_target_realloc_host(void *ptr, size_t size, int device_num)` same, accessible by host and all supported devices; `void *ompx_target_realloc_device(void *ptr, size_t size, int device_num)` same, device-only; `void *ompx_target_realloc_shared(void *ptr, size_t size, int device_num)` same, accessible by host and the specified device; `void *ompx_target_aligned_alloc(size_t alignment, size_t size, int device_num)` aligned device memory, device-only; `void *ompx_target_aligned_alloc_host(size_t alignment, size_t size, int device_num)` aligned, host + all supported devices; `void *ompx_target_aligned_alloc_device(size_t alignment, size_t size, int device_num)` aligned, device-only; `void *ompx_target_aligned_alloc_shared(size_t alignment, size_t size, int device_num)` aligned, host + specified device; `void *ompx_target_aligned_alloc_shared_with_hint(size_t align, size_t size, int access_hint, int device_num)` aligned, host + specified device, `access_hint` ∈ {`ompx_mem_hint_read_mostly`, `ompx_mem_hint_prefer_device`, `ompx_mem_hint_non_atomic_mostly`, `ompx_mem_hint_cached`, `ompx_mem_hint_uncached`}.

**Target Offload:** `int ompx_get_device_info(int devce_num, int info_id, size_t info_size, void *info_value, size_t *info_size_ret)` device info requested by `info_id`; with `info_value` NULL and `info_size` 0 returns the correct size in `info_size_ret`; zero if successful, non-zero otherwise and non-zero if the info is unavailable in the backend [sic: source spells the parameter `devce_num`]; `int ompx_get_num_subdevices(int device_num, int level)` subdevices supported at `level`; `int ompx_target_register_host_pointer(void *ptr, size_t size, int device_num)` register `ptr` for efficient copy to a device pointer allocated for `device_num`, **non-zero if successful, zero otherwise**, **Linux only**; `void ompx_target_unregister_host_pointer(void *ptr, int device_num)` unregister `ptr`, **Linux only**; `int ompx_target_prefetch_shared_mem(size_t num_ptrs, void **ptrs, size_t *sizes, int device_num)` prefetch shared memory in (`num_ptrs`, `ptrs`) on `device_num`, zero if successful; `int ompx_get_device_from_ptr(const void *ptr)` OpenMP device number on which device pointer `ptr` is allocated, valid device number or a negative number on failure.

`ompx_get_device_info` `info_id` constants (type → meaning): `ompx_devinfo_ccs_id` (`int32_t`) compute command streamer (CCS) ID if supported; `ompx_devinfo_eu_simd_width` (`uint32_t`) physical EU SIMD width; `ompx_devinfo_eus_per_subslice` (`uint32_t`) EUs per sub-slice; `ompx_devinfo_global_mem_cache_size` (`uint64_t`) cache size in bytes; `ompx_devinfo_global_mem_size` (`uint64_t`) total memory size in bytes available to the device; `ompx_devinfo_local_mem_size` (`uint32_t`) max shared local memory per group in bytes; `ompx_devinfo_max_clock_frequency` (`uint32_t`) max clock frequency in MHz; `ompx_devinfo_max_mem_alloc_size` (`size_t`) max memory allocation size in bytes; `ompx_devinfo_name` (`char *`) device name; `ompx_devinfo_num_eus` (`uint32_t`) total EUs; `ompx_devinfo_num_slices` (`uint32_t`) number of slices; `ompx_devinfo_num_threads_per_eu` (`uint32_t`) threads per EU; `ompx_devinfo_pci_id` (`uint32_t`) device ID from PCI configuration; `ompx_devinfo_plugin_name` (`char *`) offload backend name; `ompx_devinfo_subslices_per_slice` (`uint32_t`) sub-slices per slice; `ompx_devinfo_tile_id` (`int32_t`) tile ID if supported.

See also `openmp-stubs, Qopenmp-stubs` compiler option, OpenMP* Runtime Library Routines, OpenMP* Support Libraries.

### OpenMP* Support Libraries

**Performance** libraries support parallel OpenMP execution; **Stubs** libraries support serial execution (stubs for OpenMP routines and extended Intel-specific routines). Each kind is available for dynamic and static linking on Linux*; **only dynamic linking is supported on Windows***.

| Library kind | Option | Linux dynamic / static | Windows dynamic / static |
|---|---|---|---|
| Performance | `-qopenmp` / `/Qopenmp` | `libiomp5.so` / `libiomp5.a` | `libiomp5md.lib`, `libiomp5md.dll` / None |
| Stubs | `-qopenmp-stubs` / `/Qopenmp-stubs` | `libiompstubs5.so` / `libiompstubs5.a` | `libiompstubs5md.lib`, `libiompstubs5md.dll` / None |

Many routines are more optimized for Intel® microprocessors than non-Intel microprocessors (construct list as in Key facts).

**Execution Modes.** Selected at runtime with `KMP_LIBRARY`:
- **throughput (default)** — allows yielding to other programs and adjusts resource usage for efficient execution in a dynamic environment. In a multi-user environment with non-constant load or unpredictable job streams, throughput may be better, minimizing total time to run multiple jobs. Worker threads yield while waiting. After a parallel region, threads wait for work, then sleep so resources can be used by non-OpenMP threaded code between regions or by other applications; the wait is set by `KMP_BLOCKTIME` or `kmp_set_blocktime()`. Small blocktime suits non-OpenMP threaded code between parallel regions; large blocktime suits threads reserved solely for OpenMP but may penalize other concurrently running OpenMP/threaded applications.
- **turnaround** — keeps all processors active, minimizing a single job's time; worker threads actively wait without yielding (still subject to `KMP_BLOCKTIME`). Best in a dedicated (batch/single user) environment with exclusively allocated processors. **NOTE:** avoid over-allocating system resources (too many threads or too few processors at runtime); over-allocation causes poor performance — if it occurs, use throughput mode.
- **serial** — forces parallel applications to run as a single thread.

### Use the OpenMP* Libraries

Set up the environment so the OpenMP library is available at link time: on Linux source the appropriate `setvars` script file; on Windows run the appropriate `.bat` file or use the command-line window in the compiler program folder. Use the `omp.h` provided by the compiler you compile with (e.g. GCC's `omp.h` with GCC). With GCC or the Microsoft Compiler you may inadvertently use inappropriate header/module files; copy the file(s) to a separate directory and add it with `-I` to avoid this. If structures/classes contain members typed from `omp.h`, all sources using them should use the same `omp.h`.

```bash
icpx -qopenmp hello.cpp                 # compile+link; dynamic by default
# static link (not recommended): add -qopenmp-link=static
# -qopenmp-link controls static vs dynamic OpenMP libs on Linux; default -qopenmp-link=dynamic
```

Linux object-level interoperability with GCC (`<install_dir>` = location of the installed Intel OpenMP library):

```bash
gcc -fopenmp -c foo.c bar.c             # -c prevents linking at this step
gcc foo.o bar.o -liomp5 -lpthread -L<install_dir>/lib
# Alternate: let icx link so -liomp5, -L and -lpthread are not needed
gcc -fopenmp -c foo.c
icx -qopenmp -c bar.c
icx -qopenmp foo.o bar.o
```

Mixed C/C++/Fortran objects link with GNU, GCC, or Intel oneAPI DPC++/C++ Compiler compilers (`ibar.c`, `gbar.c`, `foo.f`; main in `ibar.c`):

```bash
icx -qopenmp -c ibar.c
gcc -fopenmp -c gbar.c
ifx -qopenmp -c foo.f
icx -qopenmp foo.o ibar.o gbar.o
```

If the main program were in Fortran file `foo.f`, linking must be done by `ifx`. **Do not mix objects created with the GNU Fortran Compiler (`gfortran`)**; recompile all Fortran sources with `ifx`, or all with `gfortran` (Linux-only). Linking with `gfortran` requires the Intel® OpenMP compatibility library and Intel® `irc` libraries via `-l`, the Linux pthread library via `-l`, and the Intel® library path via `-L`; `-fopenmp` is not needed on the link line:

```bash
gfortran -fopenmp -c foo.f
gfortran foo.o ibar.o gbar.o -lirc -liomp5 -lpthread -lc -L<install_dir>/lib                 # component layout
gfortran foo.o ibar.o gbar.o -lirc -liomp5 -lpthread -lc -L<install_dir>/<toolkit_version>/lib  # unified layout
# Alternate: link with icx but pass the gfortran libraries
icx -qopenmp -c ibar.c
gfortran -fopenmp -c foo.f
icx -qopenmp foo.o ibar.o -lgfortran -L<install_dir_of_gfortran_libraries>
```

```text
icx /MD /Qopenmp hello.cpp                                        # Windows; dynamic by default
cl /MD /openmp hello.cpp /link /nodefaultlib:vcomp libiomp5md.lib  # MSVC: Intel compat lib, suppress vcomp
cl /MD /openmp /c f1.c f2.c                                       # /c prevents linking
icx /MD /Qopenmp /c f3.c f4.c
icx /MD /Qopenmp f1.obj f2.obj f3.obj f4.obj /Feapp /link /nodefaultlib:vcomp   # /Fe names the exe
link f1.obj f2.obj f3.obj f4.obj /out:app.exe /nodefaultlib:vcomp libiomp5md.lib  # Visual C++ linker
```

**Visual Studio.** (1) `Project > Properties` → `Configuration Properties > Linker > General > Additional Library Directories` → e.g. `<Intel_compiler_installation_path>\<version>\lib`. (2) `Configuration Properties > Debugging > Environment` → e.g. `C:\Program Files (x86)\Common Files\intel\Shared Libraries`. (3) `Configuration Properties > Linker > Command Line > Additional Options` → `/nodefaultlib:vcomp libiomp5md.lib`.

See also `qopenmp, Qopenmp`, Using IPO, OpenMP Support Libraries, `qopenmp-link, Qopenmp-link`.

### Thread Affinity Interface

The Intel® runtime library can bind OpenMP* threads to physical processing units, controlled by `KMP_AFFINITY`. Thread affinity restricts execution of certain threads (virtual execution units) to a subset of physical processing units and can dramatically affect speed. Supported on Windows* and on Linux* systems whose kernel supports thread affinity. Three interfaces make up the Intel OpenMP Thread Affinity Interface:
- **High-level** — environment variable determines machine topology and assigns threads by physical location; controlled entirely by `KMP_AFFINITY`.
- **Mid-level** — environment variable explicitly specifies which processors (integer IDs) bind to threads; compatible with GCC* `GOMP_CPU_AFFINITY`, also invocable via `KMP_AFFINITY`. `GOMP_CPU_AFFINITY` is Linux-only; Windows/Linux users can use the similar `KMP_AFFINITY` functionality.
- **Low-level** — APIs let threads call into the runtime to specify processors; similar to `sched_setaffinity` on Linux or `SetThreadAffinityMask` on Windows. Some `KMP_AFFINITY` options affect it: type `disabled` disables the low-level interface; `KMP_AFFINITY`/`GOMP_CPU_AFFINITY` can set the initial affinity mask which the low-level API can retrieve.

Terms: total processing elements = number of **OS thread contexts**; each is an **Operating System processor (OS proc)** with a unique **OS proc ID**; a **package** is a single or multi-core processor chip; the **OpenMP Global Thread ID (GTID)** uniquely identifies threads known to the Intel OpenMP runtime library. The first library-initializing thread gets GTID 0. Normally `nthreads-var - 1` new threads get GTIDs 1..`nthreads-var - 1`, each equal to the thread number from `omp_get_thread_num()`. High/mid-level interfaces rely on GTIDs, so they are of limited use with nested parallelism; the low-level interface does not use GTIDs and works with arbitrarily many levels.

**The KMP_AFFINITY Environment Variable.** Set it before the first parallel region, or certain API calls including `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls (see Low Level Affinity API).

```text
KMP_AFFINITY=[<modifier>,...]<type>[,<permute>][,<offset>]
```

E.g. `KMP_AFFINITY=verbose,none` lists a machine topology map.

| Argument | Default | Description |
|---|---|---|
| `modifier` | `noverbose`, `respect`, `granularity=core` | Optional keyword+specifier string. Values: `granularity=<specifier>` (`fine`, `thread`, `core`, `tile`, `die`, `module`, `l1_cache`, `l2_cache`, `l3_cache`, `node` (also `numa_domain`), `group`, `socket`), `norespect`, `noverbose`, `nowarnings`, `noreset`, `proclist={<proc-list>}`, `respect`, `verbose`, `warnings`, `reset`. `<proc-list>` syntax under mid-level affinity interface. **NOTE:** on Windows with multiple processor groups, `norespect` is assumed when the process affinity mask equals a single processor group (default on Windows); otherwise `respect` is used. |
| `type` | `none` | Required. Values: `balanced`, `compact`, `disabled`, `explicit`, `none`, `scatter`, `logical` (deprecated; instead use `compact` but omit any permute value), `physical` (deprecated; instead use `scatter`, possibly with an offset value). `logical`/`physical` are deprecated but supported for backward compatibility. |
| `permute` | `0` | Optional positive integer. Not valid with type `explicit`, `none`, or `disabled`. |
| `offset` | `0` | Optional positive integer. Not valid with type `explicit`, `none`, or `disabled`. |

**Affinity Types.** Type is the only required argument.
- **`none` (default)** — no binding to particular thread contexts; if the OS supports affinity the compiler still uses the interface to determine machine topology. `KMP_AFFINITY=verbose,none` lists a topology map.
- **`balanced`** — places threads on separate cores until all cores have at least one thread, like `scatter`; when multiple hardware thread contexts per core must be used it keeps OpenMP thread numbers close together, which `scatter` does not. **CPU only, single socket systems.** NOTE: `OMP_PROC_BIND=spread` is similar and available on all platforms including multi-socket CPU systems.
- **`compact`** — assigns OpenMP thread `<n>`+1 to a free thread context as close as possible to where thread `<n>` was placed; the nearer a node is to the root, the more significance it has when sorting threads.
- **`disabled`** — completely disables the thread affinity interfaces, as if affinity were unsupported by the OS. Includes `kmp_set_affinity` and `kmp_get_affinity`, which have no effect and return a non-zero error code.
- **`explicit`** — assigns threads to OS proc IDs given by the `proclist=` modifier, **required** for this type.
- **`scatter`** — distributes threads as evenly as possible across the system; the opposite of `compact`, so the leaves of the node are most significant when sorting the topology map.

**Deprecated types `logical` and `physical`.** A single trailing integer is an **offset** specifier (for `compact`/`scatter` it is a **permute** specifier). `logical` assigns threads to consecutive logical processors (hardware thread contexts); equivalent to `compact` except permute is not allowed — `KMP_AFFINITY=logical,n` ≡ `KMP_AFFINITY=compact,0,n` (regardless of `granularity=fine`). `physical` assigns threads to consecutive physical processors (cores): with one thread context per core it equals `logical`; with multiple thread contexts per core it equals `compact` with permute 1, i.e. `KMP_AFFINITY=physical,n` ≡ `KMP_AFFINITY=compact,1,n` (regardless of `granularity=fine`). The compiler should permute the innermost topology level to the outermost (presumably the thread context level); `physical` does not support the permute specifier.

**Examples of Types compact and scatter.** Topology: two processors, each two cores, each core with Intel® Hyper-Threading Technology (Intel® HT Technology) enabled. Default mapping of OpenMP threads to hardware thread contexts:

*[OpenMP thread affinity topology (default mapping) — source p. 706]*

Text description: Machine/Node → Package 0 and Package 3 → each Package has Core 0 and Core 1 → each Core has two thread contexts (0 and 1). OpenMP global thread IDs left-to-right: 0,1 (pkg0 core0), 2,3 (pkg0 core1), 4,5 (pkg3 core0), 6,7 (pkg3 core1). This is the default mapping.

`KMP_AFFINITY=granularity=fine,compact`:

*[compact affinity thread ID renumbering — source p. 707]*

Text description: **compact** thread IDs core0/p0 → 0,4; core1/p0 → 2,6; core0/p3 → 1,5; core1/p3 → 3,7 — consecutive IDs land on the same core's two thread contexts.

`KMP_AFFINITY=granularity=fine,scatter`:

*[scatter affinity thread ID renumbering — source p. 707]*

Text description: **scatter** thread IDs core0/p0 → 3,4; core1/p0 → 5,6; core0/p3 → 7,0; core1/p3 → 1,2 — IDs spread across packages/cores rather than packed.

**permute and offset Combinations.** For `compact` and `scatter` both are allowed; with only one integer specified the compiler treats it as `permute`. **Both default to 0.** `permute` controls which levels are most significant when sorting the topology map: a value forces the specified number of most significant levels to become least significant and inverts the order of significance. The root node is not a separate level for sorting. `offset` is the starting position for thread assignment. (Source p. 707 figure: result of `KMP_AFFINITY=granularity=fine,compact,0,5` — the offset shifts the starting thread assignment position.)

With the same hardware running an OpenMP application with data sharing between consecutive loop iterations, consecutive threads should bind close together (`KMP_AFFINITY=compact`) to minimize communication overhead, cache line invalidation overhead, and page thrashing. If the application also has parallel regions that do not use all threads, avoid binding multiple threads to the same core while leaving other cores unused (a thread executes faster where it does not compete with another active thread on the same core). `KMP_AFFINITY=granularity=fine,compact,1,0`:

*[compact with permute 1 offset 0 — source p. 708]*

Text description: thread IDs 0,4 / 1,5 / 2,6 / 3,7 (core0/p0, core1/p0, core0/p3, core1/p3). Thread n+1 binds as close as possible to thread n but on a different core; once each core has one thread, subsequent threads fill the available cores in the same order on different thread contexts.

**Modifier Values for Affinity Types.** Modifiers precede `type` and are optional; **without a modifier, `noverbose`, `respect`, and `granularity=core` are used automatically.** Interpreted left to right; a conflicting modifier is ignored (e.g. `KMP_AFFINITY=verbose,noverbose,scatter` ≡ `KMP_AFFINITY=verbose,scatter`).
- **`noverbose` (default)** — no verbose messages.
- **`verbose`** — prints packages, cores per package, thread contexts per core, and OpenMP thread bindings to physical thread contexts. Binding is shown indirectly as hardware thread context → OS processor (proc) ID mappings; each thread's affinity mask is printed as a set of OS processor IDs. Example `KMP_AFFINITY=verbose,scatter` on a dual core, two-processor system with Intel® HT Technology disabled:

```text
KMP_AFFINITY: Initial OS proc set respected: 0,1,2,3
KMP_AFFINITY: affinity capable, using hwloc.
KMP_AFFINITY: 4 available OS procs
KMP_AFFINITY: Uniform topology
KMP_AFFINITY: 2 sockets x 2 cores/socket x 1 threads/core (4 total cores)
KMP_AFFINITY: OS proc to physical thread map:
KMP_AFFINITY: OS proc 0 maps to socket 0 core 0 thread 0
KMP_AFFINITY: OS proc 2 maps to socket 0 core 1 thread 0
KMP_AFFINITY: OS proc 1 maps to socket 3 core 0 thread 0
KMP_AFFINITY: OS proc 3 maps to socket 3 core 1 thread 0
KMP_AFFINITY: pid 79739 tid 79739 thread 0 bound to OS proc set 0
KMP_AFFINITY: pid 79739 tid 79740 thread 2 bound to OS proc set 2
KMP_AFFINITY: pid 79739 tid 79741 thread 3 bound to OS proc set 3
KMP_AFFINITY: pid 79739 tid 79742 thread 1 bound to OS proc set 1
```

  Message meanings: "affinity capable" — compiler, OS, and hardware all support affinity, binding possible; "decoding x2APIC ids" — topology found by binding a thread to each OS processor and decoding `cpuid`; "using hwloc" — Portable Hardware Locality* (hwloc) used; "using /proc/cpuinfo" — Linux only, cpuinfo used; "using flat" — OS processor ID assumed equal to physical package ID, used if nothing else works and possibly inaccurate; "uniform topology" — full tree with no missing leaves at any level. The OS proc → thread context ID mapping prints next, then OpenMP thread context ID binding unless the type is `none`.
- **`granularity`** — binding threads to packages and cores often gains performance with Intel® HT Technology enabled, but binding each thread to a particular thread context on a core usually is not beneficial. Granularity is the lowest level threads may float within a topology map: `core` (default; threads bound to a core float between its thread contexts); `fine` or `thread` (finest; each thread binds to a single thread context; functionally equivalent); `tile`, `die`, `module`, `node` (also `numa_domain`), `group`, `l1_cache`, `l2_cache`, `l3_cache`, `socket` (threads float between the hardware thread contexts representing that resource; e.g. `granularity=socket` allows moving between the socket's hardware threads). [Source formatting unclear] Only available when Intel® Hybrid Technology is detected in the machine topology: `core_type` or `core_efficiency`.

  Example `KMP_AFFINITY=verbose,granularity=core,compact` on the same dual core, two-processor system **with** Intel® HT Technology enabled (source prints "sockects"):

```text
KMP_AFFINITY: Initial OS proc set respected: 0-7
KMP_AFFINITY: decoding x2APIC ids.
KMP_AFFINITY: 8 available OS procs
KMP_AFFINITY: Uniform topology
KMP_AFFINITY: 2 sockects x 2 cores/socket x 2 threads/core (4 total cores)
KMP_AFFINITY: OS proc 0 maps to socket 0 core 0 thread 0
KMP_AFFINITY: OS proc 4 maps to socket 0 core 0 thread 1
KMP_AFFINITY: OS proc 2 maps to socket 0 core 1 thread 0
KMP_AFFINITY: OS proc 6 maps to socket 0 core 1 thread 1
KMP_AFFINITY: OS proc 1 maps to socket 3 core 0 thread 0
KMP_AFFINITY: OS proc 5 maps to socket 3 core 0 thread 1
KMP_AFFINITY: OS proc 3 maps to socket 3 core 1 thread 0
KMP_AFFINITY: OS proc 7 maps to socket 3 core 1 thread 1
KMP_AFFINITY: pid 40880 tid 40880 thread 0 bound to OS proc set 0,4
KMP_AFFINITY: pid 40880 tid 40882 thread 2 bound to OS proc set 2,6
KMP_AFFINITY: pid 40880 tid 40884 thread 4 bound to OS proc set 1,5
KMP_AFFINITY: pid 40880 tid 40886 thread 6 bound to OS proc set 3,7
```

  With `granularity=fine` (or `thread`) + `compact`, thread contexts carry OpenMP global thread ID sets core0/p0 {0,1}, core1/p0 {2,3}, core0/p3 {4,5}, core1/p3 {6,7} — each thread binds to a single hardware thread context and consecutive IDs fill one core before moving on:

*[granularity=fine (thread) with compact — source p. 711]*

  `KMP_AFFINITY=verbose,granularity=fine,compact` or `KMP_AFFINITY=verbose,granularity=thread,compact` (same topology header) instead gives:

```text
KMP_AFFINITY: pid 40895 tid 40895 thread 0 bound to OS proc set 0
KMP_AFFINITY: pid 40895 tid 40896 thread 1 bound to OS proc set 4
KMP_AFFINITY: pid 40895 tid 40897 thread 2 bound to OS proc set 2
KMP_AFFINITY: pid 40895 tid 40898 thread 3 bound to OS proc set 6
KMP_AFFINITY: pid 40895 tid 40899 thread 4 bound to OS proc set 1
KMP_AFFINITY: pid 40895 tid 40900 thread 5 bound to OS proc set 5
KMP_AFFINITY: pid 40895 tid 40901 thread 6 bound to OS proc set 3
KMP_AFFINITY: pid 40895 tid 40902 thread 7 bound to OS proc set 7
```

  `granularity=fine` always binds each thread to a single OS processor; equivalent to `granularity=thread`, currently the finest level.
- **`respect` (default)** — respect the process' original affinity mask, specifically the mask in place for the thread that initializes the OpenMP runtime library. **Linux**: respect the initializing thread's mask. **Windows**: respect the original process affinity mask. NOTE: on Windows with multiple processor groups, `norespect` is the default when the process affinity mask equals a single processor group (default on Windows); otherwise `respect` is the default.

  Example `KMP_AFFINITY=verbose,compact` on the same system with Intel® HT Technology enabled and initial affinity mask `{4,5,6,7}` (thread context 1 on every core) models the machine as dual core, two-processor with HT disabled:

```text
KMP_AFFINITY: Initial OS proc set respected: 4-7
KMP_AFFINITY: 4 available OS procs
KMP_AFFINITY: Uniform topology
KMP_AFFINITY: 2 sockets x 2 cores/socket x 1 threads/core (4 total cores)
KMP_AFFINITY: OS proc 4 maps to socket 0 core 0 thread 1
KMP_AFFINITY: OS proc 6 maps to socket 0 core 1 thread 1
KMP_AFFINITY: OS proc 5 maps to socket 3 core 0 thread 1
KMP_AFFINITY: OS proc 7 maps to socket 3 core 1 thread 1
KMP_AFFINITY: pid 41032 tid 41032 thread 0 bound to OS proc set 4
KMP_AFFINITY: pid 41032 tid 41033 thread 1 bound to OS proc set 6
KMP_AFFINITY: pid 41032 tid 41034 thread 2 bound to OS proc set 5
KMP_AFFINITY: pid 41032 tid 41035 thread 3 bound to OS proc set 7
```

  Because four thread contexts are accessible, four threads were created by default. Figure: corresponding topology and thread placement when eight threads are requested via `OMP_NUM_THREADS=8`; the `norespect` modifier gives sets core0/p0 {0,4}, core1/p0 {1,5}, core0/p3 {2,6}, core1/p3 {3,7} (on Windows with multiple processor groups, `norespect` is default when the process affinity mask equals a single processor group).

*[norespect modifier thread ID sets — source p. 712]*

  With local `cpuid` information it is not always possible to distinguish a machine without Intel® HT Technology from one that supports but disabled it. The compiler therefore omits a level whose elements had no siblings, except the package level, which is always modeled even with a single package.
- **`norespect`** — do not respect the process' original affinity mask; bind threads to all OS processors. In early runtime versions supporting only `physical`/`logical`, `norespect` was the default and not recognized as a modifier. The default changed to `respect` when `compact`/`scatter` were added, so bindings may have changed with newer compilers for a partial initial thread affinity mask.
- **`nowarnings`** — no affinity warning messages.
- **`warnings` (Default)** — print them.
- **`noreset` (Default)** — do not reset the primary thread's affinity after each outermost parallel region; preserves it between parallel regions. E.g. with `KMP_AFFINITY=compact,granularity=core` the primary thread's affinity is set to the first core for the first parallel region and kept for the thread's lifetime, even during serial regions.
- **`reset`** — reset the primary thread's affinity after each outermost parallel region to the initial affinity before OpenMP was initialized.

**Determine Machine Topology.** With an APIC (Advanced Programmable Interrupt Controller), the compiler uses `cpuid` to get package id, core id, and thread context id. Normally each thread context gets a unique APIC ID at boot; other `cpuid` data plus the number of OS thread contexts determine how to break the APIC ID into package ID, core ID, and thread context ID. The APIC ID may use the legacy leaf 4 method or the more modern leaf 11 and leaf 31. **Only 256 unique APIC IDs are available in leaf 4; leaf 11 and leaf 31 have no such limitation.** Core ids on a package and thread context ids on a core are normally contiguous; package id numbering gaps are common.

If topology cannot be determined by any other method but the OS supports affinity, a warning prints and topology is assumed **flat** (OS process N maps to package N; one thread context per core; one core per package). If topology cannot be accurately determined, copy `/proc/cpuinfo` to a temporary file, correct errors, and pass it via `KMP_CPUINFO_FILE=<temp_filename>`. If every core has one thread context the thread context level does not appear; if every package has one core the core level does not appear. The map need not be a full tree. **The package level always appears, even with a single package.**

**KMP_CPUINFO_FILE and /proc/cpuinfo.** The runtime can detect Linux topology by parsing `/proc/cpuinfo`. If that file (or a device mapped into the Linux file system) is insufficient or erroneous, copy it to a writable `<temp_file>`, correct/extend it, and set `KMP_CPUINFO_FILE=<temp_file>`; the runtime then reads that file instead of `/proc/cpuinfo` or decoding APIC IDs — **the file overrides those methods**. Usable on Windows too, where `/proc/cpuinfo` does not exist. Entries are `name : value` lines, blank-line separated per processor element; only these fields are used:

| Field | Description |
|---|---|
| `processor :` | OS ID for the processing element; must be unique. **`processor` and `physical id` are the only required fields.** |
| `physical id :` | Package ID (physical chip ID). A package may contain multiple cores; the package level always exists in the runtime model. |
| `core id :` | Core ID; defaults to 0 if absent. If every package has one core, the core level does not exist in the map (even with non-zero core ID fields). |
| `apicid :` | Thread ID; defaults to 0 if absent. If every core has one thread, the thread level does not exist in the map (even with non-zero thread ID fields). |
| `node_n id :` | Extension for NUMA interconnect nodes at different levels; arbitrarily many levels `n`. `node_0` is closest to the package level; multiple level-0 nodes comprise a level-1 node, and so on. |

Fields must be spelled exactly as shown, lowercase, then optional whitespace, a colon (`:`), optional whitespace, then the integer ID; other fields are ignored. On many Linux variants the thread id field is missing and a field labeled `siblings` gives threads per node or nodes per package; the runtime **ignores** `siblings` so it can distinguish it from thread id, and the warning `Physical node/pkg/core/thread ids not unique` appears (unless `nowarnings`).

**Windows Processor Groups.** On 64-bit Windows multiple processor groups accommodate more than 64 processors; each group is limited to at most **sixty-four (64)** processors. If multiple groups are detected, the default is a 2-level tree: level 0 = processors in a group, level 1 = the groups. Threads are assigned to a group until as many threads are bound as there are processors in the group; subsequent threads go to the next group, and so on. By default threads float among all processors in a group, i.e. **granularity equals the group (`granularity=group`)**. This can be overridden with another type like `compact` or `scatter`, but the granularity must then be fine enough to prevent binding a thread to multiple processors in different groups.

**Use a Specific Machine Topology Modeling Method (KMP_TOPOLOGY_METHOD):** `cpuid_leaf31` decode APIC IDs per leaf 31; `cpuid_leaf11` per leaf 11; `cpuid_leaf4` per leaf 4; `cpuinfo` parse `/proc/cpuinfo` if `KMP_CPUINFO_FILE` is not specified (Linux only), otherwise use `KMP_CPUINFO_FILE` (Windows or Linux); `group` 2-level map of processors in a group (level 0) and the groups (level 1) (Windows 64-bit only); `flat` flat (linear) processor list; `hwloc` model as the Portable Hardware Locality* (hwloc) library does — most detailed, including but not limited to NUMA nodes, packages, cores, hardware threads, caches, and Windows processor groups.

**Explicitly Specify OS Processor IDs (GOMP_CPU_AFFINITY, KMP_AFFINITY).** Set `GOMP_CPU_AFFINITY` or `KMP_AFFINITY` **before the first parallel region** and **before `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls** (see Low Level Affinity API). On Linux, `GOMP_CPU_AFFINITY` syntax is identical to `libgomp`'s (assume `<proc_list>` produces the entire string): `<proc_list> := <entry> | <elem> , <list> | <elem> <whitespace> <list>`; `<elem> := <proc_spec> | <range>`; `<proc_spec> := <proc_id>`; `<range> := <proc_id> - <proc_id> | <proc_id> - <proc_id> : <int>`; `<proc_id> := <positive_int>`. OS processors are assigned in order of OpenMP Global Thread IDs; if more threads than list elements are created, assignment occurs **modulo the list size** (GTID `n` binds to element `n mod <list_size>`).

Example: the dual core, dual-package machine without Intel® HT Technology. With six OpenMP threads instead of the default 4 (oversubscribing), `GOMP_CPU_AFFINITY=3,0-2` binds threads as shown — same as compiling with `gcc` and linking with `libgomp`:

*[GOMP_CPU_AFFINITY explicit OS proc assignment — source p. 717]*

Text description: each Core box has a child box with its OS proc ID: core0/p0 → 0, core1/p0 → 2, core0/p3 → 1, core1/p3 → 3. OpenMP global thread ID sets: core0/p0 {1,5}, core1/p0 {3}, core0/p3 {2}, core1/p3 {0,4}.

The same syntax gives the OS proc ID list in `proclist=[<proc_list>]`. Difference for strictly `libgomp` semantics: **`GOMP_CPU_AFFINITY` implies `granularity=fine`**; specifying the list in `KMP_AFFINITY` without `granularity=` does not change the default, so threads may float between thread contexts on a core:

```text
GOMP_CPU_AFFINITY=<proc_list>  is an alias for
KMP_AFFINITY="granularity=fine,proclist=[<proc_list>],explicit"
```

In `KMP_AFFINITY` the syntax is extended to OS processor ID **sets** — processors among which a thread may execute ("float"), in brackets: `<proc_list> := <proc_id> | { <float_list> }`; `<float_list> := <proc_id> | <proc_id> , <float_list>`. This is like `granularity=` but more flexible: a thread's OS processors may exclude nearby ones and include distant ones. Extending the example, threads 2 and 3 may float between OS processors 1 and 2 via `KMP_AFFINITY="granularity=fine,proclist=[3,0,{1,2},{1,2}],explicit"`:

*[proclist float list assignment — source p. 718]*

Text description: `verbose` execution output variant; thread ID sets core0/p0 {1,5}, core1/p0 {2,3}, core0/p3 {2,3}, core1/p3 {0,4}.

With `verbose` also specified, output includes `Initial OS proc set respected: 0,1,2,3`, `4 available OS procs`, `Uniform topology`, `2 sockets x 2 cores/socket x 1 threads/core (4 total cores)`, the OS proc→socket/core/thread map, and bindings: thread 0 → set 3, thread 1 → set 0, thread 2 → set 1,2, thread 3 → set 1,2, thread 4 → set 3, thread 5 → set 0.

**Low Level Affinity API.** Each OpenMP thread can determine its OS procs and bind with `kmp_set_affinity`, instead of an environment variable set before execution (or the `kmp_settings` interface before the first parallel region). `kmp_affinity_mask_t` is defined in `omp.h`.
- `int kmp_set_affinity (kmp_affinity_mask_t *mask)` — set the current thread's mask to `*mask` (OS proc IDs built with the calls below); the thread executes only on OS procs in the set; zero (0) on success, non-zero error code otherwise.
- `int kmp_get_affinity (kmp_affinity_mask_t *mask)` — store the current thread's mask in `*mask`, which must have been initialized with `kmp_create_affinity_mask()`; zero (0) on success, non-zero otherwise.
- `int kmp_get_affinity_max_proc (void)` — maximum OS proc ID on the machine plus 1; all OS proc IDs are in 0 (inclusive) .. value (exclusive).
- `void kmp_create_affinity_mask (kmp_affinity_mask_t *mask)` — allocate a new mask and initialize `*mask` to the empty set. The implementation may represent it as the set, a pointer to it, or an index into a table; make no assumptions.
- `void kmp_destroy_affinity_mask (kmp_affinity_mask_t *mask)` — deallocate; each create needs a corresponding destroy.
- `int kmp_set_affinity_mask_proc (int proc, kmp_affinity_mask_t *mask)` — add `proc` if not present; zero (0) on success, non-zero otherwise.
- `int kmp_unset_affinity_mask_proc (int proc, kmp_affinity_mask_t *mask)` — remove `proc` if present; zero (0) on success, non-zero otherwise.
- `int kmp_get_affinity_mask_proc (int proc, kmp_affinity_mask_t *mask)` — 1 if `proc` is in `*mask`, else 0.

After a successful `kmp_set_affinity()` the thread remains bound until at least the end of the parallel region unless reset by another call. Between parallel regions the mask and bindings are thread private data objects with the persistence described in the OpenMP Application Program Interface. Persistence between two consecutive active parallel regions requires **all three**: neither region is nested inside another explicit parallel region; the number of threads executing both is the same; the `dyn-var` ICV in the enclosing task region is false at entry to both. Thus a parallel region at program start whose sole purpose is setting each thread's mask can mimic `KMP_AFFINITY`, if those rules hold.

```c
// Force the executing thread to execute on logical CPU i
// Returns 1 on success, 0 on failure.
int forceAffinity(int i)
{
kmp_affinity_mask_t mask;
kmp_create_affinity_mask(&mask);
kmp_set_affinity_mask_proc(i, &mask);
return (kmp_set_affinity(&mask) == 0);
}
```

This assumes knowledge of the OS proc ID → physical element mapping of the target machine; elsewhere it still runs but bindings may differ and you might explicitly force a bad distribution.

> **Caution.** This interface gives complete control of the hardware resources on which threads run, requiring detailed knowledge of how logical CPUs (the OS enumeration of hardware threads) map to physical hardware — a mapping that likely differs across machines, so you risk binding machine-specific information into code and forcing bad affinities elsewhere. It may also let you ignore resource limitations set by the program startup mechanism, such as Message Passing Interface (MPI), intended to prevent multiple OpenMP processes on a node from sharing hardware threads; the runtime neither prevents this nor warns you. Prefer higher level affinity settings: they are more portable and need no low-level knowledge.

### OpenMP* Memory Spaces and Allocators

OpenMP* provides **memory spaces** for storage/retrieval of variables; the appropriate space depends on how a variable is used. Each space has a unique allocator that allocates/deallocates memory in it; allocators allocate contiguous space not overlapping any other allocation in the space. Multiple spaces with different traits may map to one memory resource.

| Allocator Trait | Values That Can Be Specified | Default Value |
|---|---|---|
| `access` | `all`, `cgroup`, `pteam`, `thread` | All |
| `alignment` | positive integer, a power of 2, in bytes | 1 byte |
| `fallback` | `abort_fb`, `allocator_fb`, `default_mem_fb`, `null_fb` | `default_mem_fb` |
| `fb_data` | an allocator handle | None |
| `partition` | `blocked`, `environment`, `interleaved`, `nearest` | environment |
| `pinned` | `true`, `false` | false |
| `pool_size` | a positive integer value | Implementation defined |
| `sync_hint` | `contended`, `uncontended`, `private`, `serialized` | contended |

- `access`: `all` — accessible by all threads in the allocating device (**default**); `cgroup` — accessible by all threads of the same contention group (access from outside is undefined behavior); `pteam` — accessible by all threads binding to the same parallel region (access otherwise undefined); `thread` — accessible only by the allocating thread (allocation by another thread undefined).
- `alignment`: byte-aligned to at least the trait value; default **1 byte**. Directives and runtime allocator routines specifying alignment can also affect it.
- `fallback`: `abort_fb` terminates the program; `allocator_fb` retries with the allocator in `fb_data`; `default_mem_fb` retries in `omp_default_mem_space` (**default**; that allocator's traits should be default except `fallback=null_fb`); `null_fb` returns zero.
- `fb_data`: the fallback allocator; the failing allocator's `fallback` must be `allocator_fb` for it to be used.
- `partition`: `blocked` — approximately equal blocks, one per storage resource; `environment` — placement by the runtime execution environment (**default**); `interleaved` — round-robin across storage resources; `nearest` — nearest storage resource to the requesting thread.
- `pinned`: `true` keeps each allocation in the same storage resource until deallocated; default **false**.
- `pool_size`: total bytes available when there have been no allocations. Scope follows `access`: `all` = all allocations for all threads with access; `cgroup` = allocations from threads in the same contention group; `pteam` = allocations within the same parallel team; `thread` = allocations from each thread using the allocator. A request larger than `pool_size` is not fulfilled.
- `sync_hint`: `contended`/`uncontended` — many/few threads anticipated to request simultaneously (**default `contended`**); `private` — all requests from one thread (**undefined if two or more threads use it**); `serialized` — one thread requests at a time (**undefined if two request simultaneously**).

**Five predefined memory spaces:** `omp_default_mem_space` (system default), `omp_large_cap_mem_space` (large capacity), `omp_high_bw_mem_space` (high bandwidth), `omp_low_lat_mem_space` (low latency), `omp_const_mem_space` (optimal storage of constant values; initialized with compile-time constant expressions or a `firstprivate` clause; **writing to variables there is undefined behavior**). **Three more are extensions to the OpenMP standard:** `omp_target_host_mem_space` (host memory accessible by the device), `omp_target_shared_mem_space` (memory that can migrate between host and device), `omp_target_device_mem_space` (memory accessible to the device).

| Allocator Name | Associated Memory Space | Non-Default Trait Values |
|---|---|---|
| `omp_default_mem_alloc` | `omp_default_mem_space` | `fallback=null_fb` |
| `omp_large_cap_mem_alloc` | `omp_large_cap_mem_space` | none |
| `omp_low_lat_mem_alloc` | `omp_low_lat_mem_space` | none |
| `omp_high_bw_mem_alloc` | `omp_high_bw_mem_space` | none |
| `omp_const_mem_alloc` | `omp_const_mem_space` | none |
| `omp_cgroup_mem_alloc` | implementation/system defined | `access=cgroup` |
| `omp_pteam_mem_alloc` | implementation/system defined | `access=pteam` |
| `omp_thread_mem_alloc` | implementation/system defined | `access=thread` |
| `omp_target_host_mem_alloc` | `omp_target_host_mem_space` | none |
| `omp_target_shared_mem_alloc` | `omp_target_shared_mem_space` | none |
| `omp_target_device_mem_alloc` | `omp_target_device_mem_space` | none |

> **NOTE.** `ifx` does not recognize the allocator names listed as implementation/system defined. `omp_large_cap_mem_space`, `omp_low_lat_mem_space`, `omp_high_bw_mem_space`, and `omp_const_mem_space` have the same effect as specifying `omp_default_mem_space`.

### OpenMP* Contexts

At each point of an OpenMP* program an OpenMP context describes: the devices where parts of the program execute; implementation supported functionality such as target instruction sets; the active OpenMP constructs; and available dynamic values. Trait sets: **construct, dynamic, device, implementation, and target_device**; the trait category determines the context selector syntax that matches it.

**construct Trait Set** — pragma names of all enclosing constructs up to an `omp target` construct; each enclosing directive name is a trait. Composite and combined constructs are added as distinct constructs in the nesting order specified by the construct. It is implementation defined whether an implementation adds an `omp dispatch` construct; if added it is only for the target call of the code. Constructs are ordered `c1, … cN` (`c1` outermost, `cN` innermost). At a point not enclosed in an `omp target` construct, in order:
1. Procedures with `omp declare simd` get the `omp simd` trait added as construct trait `c1` for generated `omp simd` versions (+1 trait).
2. Function variants generated by `omp declare variant` get constructs `c1` to `cM` added at the beginning as `c1, … cM` (+M traits).
3. The `omp target` trait is added at the beginning of a device routine as `c1` for versions generated for target regions (+1 trait).

The clause list trait `omp simd` matches clauses accepted in an `omp declare simd` pragma with the same names and semantics; it minimally defines the `simdlen` property and either the `inbranch` or `notinbranch` property. Other construct traits are non-property traits.

**device Trait Set** — characteristics of the device targeted at that point. A target-device set exists per supported target device. Required traits for device and target_device sets: `kind` (`kind-name-list`) — `any` (as if no kind selector), `host` (the host device), `nohost` (not the host device), plus values in the OpenMP Additional Definitions document; `arch` (`architecture-name-list`) — implementation defined architectures supported; `isa` (`isa-name-list`) — implementation-defined instruction set architectures supported; `vendor` (`vendor-name-list`) — a supported vendor-name value from the OpenMP Additional Definitions document. The `target_device` set must also include `device_num` (the device number). `arch`, `isa`, `kind`, and `vendor` are name-list traits.

**implementation Trait Set** — supported functionality at that point: `extension` (`extension-name-list`) lists implementation-specific extensions (names implementation defined); `vendor` (`vendor-name-list`); a `requires` (`requires-clause-list`) trait — a clause-list trait whose properties are the clauses specified in the `requires` pragma before that point, including implementation-defined implicit requirements. `vendor` and `extension` are name-list traits. Implementations may define additional device, target_device, and implementation traits (extension traits). The **dynamic trait set** specifies dynamic properties at any execution point; the `data state` trait refers to the complete data state accessible at runtime.

### OpenMP* Context Selectors

```text
context-selector          Is trait-set-selector [, trait-set-selector [, . . . ]]
trait-set-selector        Is trait-set-selector-name= {trait-selector [, trait-selector [, . . .]]}
                          Note that the curly braces are part of the required syntax.
trait-set-selector-name   Is construct, device, implementation, target_device, or user.
trait-selector            Is trait-set-selector-name [ ( [trait-score : ] trait-property [, trait-property [, . . .]] ) ]
trait-score               Is score (score-expression)
score-expression          Is a scalar-integer-constant-expression with a non-negative value.
trait-property            Is trait-property-name
                          or trait-property-clause
                          or trait-property-expression
                          or trait-property-extension
trait-property-name       Is kind, isa, arch, or vendor
                          or a default-character-constant
trait-property-clause     Is a clause, as defined in the OpenMP 5.2 Specification.
trait-property-expression Is a scalar-expression
                          or a scalar-integer-expression
trait-property-extension  Is trait-property-name
                          or identifier ( trait-property-extension [, trait-property-extension [, . . .]] )
                          or a constant-integer-expression
```

- Name-list trait-selectors (`kind`, `isa`, `arch` in device and target_device sets) take `trait-property-name`; **at least one trait-property must be specified**.
- Clause-list trait-selectors (a `simd` trait in the construct set, a `requires` clause in the implementation set) take a `trait-property-clause` matching the OpenMP clause; **at least one trait-property is required for a `requires` selector**.
- The `isa` construct context selector set specifies construct traits that should be active; its trait selectors are OpenMP directive names of context-matching constructs.
- The `trait-property-clause` syntax for a `simd` trait-selector in a construct set is that of a valid `omp declare simd` clause with the same restriction.
- Device and implementation selector sets define traits that should be active in the OpenMP context's trait sets.
- `target_device` specifies traits active in the target device trait set for the device identified by `device_num`; if `device_num` is specified for `target_device`, **only one trait-property-expression may be specified**.
- The `kind` selector of device/target_device may specify `host`, `nohost`, or `any`; if `any` is specified, neither `host` nor `nohost` may appear in the same selector.
- `atomic_default_mem_order` can select the implementation trait set; only one trait-property may appear, an identifier that is a valid argument to the `atomic_default_mem_order` clause in an `omp requires` pragma.
- The `requires` selector can also select the implementation set; syntax and restrictions match a valid `omp requires` clause.
- The `user` selector defines a condition selector for additional user-defined conditions; it must contain one trait-property-expression that is a logical expression and must evaluate to true for the selector to be true. If not a constant expression the selector is dynamic, otherwise static.

The **dynamic** part of a context selector is its user selector set (if not static) plus its target_device selector set; all other parts are **static**. In the `match` clause of `omp declare variant`: a reference to a formal parameter of the base function refers to the actual parameter associated with it; otherwise a reference to a variable or function refers to the one accessible in the scope of the pragma containing the context selector. Except in a construct selector set, each trait-property may be specified only once; each trait-set-selector-name may appear once; a given trait-selector-name may appear only once. A trait-score cannot be specified for construct, device, or target_device trait selector sets. The `device_num` expression must evaluate to a non-negative integer ≤ the value returned by `omp_get_num_devices()`.

### Score and Match Context Selectors

A context is compatible with a context selector if: all `user` trait set conditions evaluate to true; all traits/properties defined by `implementation`, `device`, and `construct` sets are active in the corresponding context trait set; all traits/properties defined by `target_device` are active in the target-device trait set for the `device_num` device; selectors in the construct set specify the same construct ordering as the context's construct trait set; for each selector the properties specified are a subset of the corresponding context trait's properties; and no implementation-defined selector specified is ignored by the implementation.

Additional `simd` matching: selector `aligned (list :n)` matches context `aligned (list :m)` if `n` is a multiple of `m`; selector `simdlen (n)` matches context `simdlen (m)` if `m` is a multiple of `n`.

Scoring compatible context selectors: trait selectors with a trait-score take the trait-score expression value; each specified construct trait selector matching the context construct trait gets `2p-1` where `p` is the position of the corresponding trait `cp` in the context trait set specified by the context selector (if those traits appear multiple times, the highest valued subset containing all selectors in the same order is used); if specified, `kind`, `arch`, and `isa` get `2n`, `2n+1`, and `2n+2` where `n` is the number of traits in the construct set; other selectors get zero; values for implementation-defined selectors are defined by the implementation; a context selector that is a strict subset of another gets score zero, and for other selectors the final value is the sum of the specified selector's values plus 1.

## OpenMP* Offloading SPMD/SIMT and SIMD Models

For Intel GPUs, OpenMP kernel generation supports **SPMD (Single Program Multiple Data)** and **SIMD (Single Instruction Multiple Data)**. Differences: **Data Parallelism** — SPMD exploits multiple threads simultaneously operating on different data elements, SIMD executes the same operation on multiple data elements; **Granularity** — SPMD is coarser (a thread may handle a significant portion of the computation), SIMD is finer (individual instructions and vectorized data elements); **Syntax** — both use OpenMP directives, whose specific forms (e.g. `parallel for` for SPMD, `simd` for SIMD) reflect the models.

**SPMD (default for OpenMP offloading).** The OpenMP SPMD (also known as SIMT, Single Instruction Multiple Threads) model is a common GPU programming model. Below, at the kernel level Loop-A and its entire body are vectorized with SIMD8, SIMD16, or SIMD32 for Intel® ARC GPU with native SIMD8 hardware support (the compiler generates SIMD8, SIMD16, or SIMD32 kernels). For Intel® GPU Max Series with native SIMD16 support, Loop-A and its body are vectorized with SIMD16 or SIMD32 — SIMD16 or SIMD32 kernels for SIMT16 or SIMT32 thread execution.

```c
#pragma omp target teams distribute parallel for // Loop-A is vectorized with SIMD8, SIMD16, or
SIMD32 based on the HW width of the GPU SIMD hardware unit.
for (int a = 0; a < M; a++) {
    code 1;
    for (int b = 0; b < N; b++)
        code 2;
    for (int c = 0; c < K; c++)
        code 3;
    code 4;
}
```

The SPMD model is the **default** for OpenMP offloading, enabled with `-fiopenmp -fopenmp-targets=spir64`; the compiler generates SIMD8, SIMD16, or SIMD32 kernels like 8-way, 16-way, or 32-way SIMT parallelism for the outer Loop-A.

**SIMD model.** OpenMP SIMD is a common CPU programming model. The Intel GPU has a SIMD engine in its execution unit, so the compiler can generate explicit SIMD code from the seamless transition from well-tuned CPU code with the outer-parallel-inner-simd scheme. Below, Loop-A and its entire body are **not** vectorized at the kernel level (the compiler generates the SIMD1 kernel for thread execution):

```c
#pragma omp target teams distribute parallel for    // SIMD1 kernel is generated for Loop-A
for (int a = 0; a < M; a++) {
    code 1;
    #pragma omp simd simdlen(32)                 // Loop-B is vectorized with SIMD32 in the kernel
    for (int b = 0; b < N; b++)
        code 2;
    #pragma omp simd simdlen(8)                  // Loop-C is vectorized with SIMD8 in the
kernel
    for (int c = 0; c < K; c++)
        code 3;
    code 4;
}
```

The SIMD model is controlled by `-fiopenmp -fopenmp-targets=spir64 -fopenmp-target-simd`; it generates a SIMD1 kernel with explicit SIMD code inside the kernel for OpenMP SIMD loops, giving more flexibility in register allocation and SIMD width control for OpenMP SIMD loops inside target regions (or kernels). See also `fiopenmp, Qiopenmp`, `fopenmp-target-simd, Qopenmp-target-simd`, `fopenmp-targets, Qopenmp-targets`.

## OpenMP* Advanced Issues

Include the `omp.h` header (installed in the INCLUDE directory) and compile with `/Qopenmp` (Windows*) or `-qopenmp` (Linux*). The alphabet example illustrates: (1) using functions instead of pragmas requires rewriting code — extra debugging/testing/maintenance; (2) it becomes difficult to compile without OpenMP support; (3) simple bugs are easy — the loop fails to print all letters when the number of threads is not a multiple of 26; (4) you lose adjustable loop scheduling without your own work-queue algorithm, being limited to your own (most likely static) scheduling.

```c
#include <stdio.h>
#include <omp.h>

int main(void) {
    int i;
    omp_set_num_threads(4);

    #pragma omp parallel private(i)
    {
        // OMP_NUM_THREADS is not a multiple of 26,
        // which can be considered a bug in this code.
        int LettersPerThread = 26 / omp_get_num_threads();
        int ThisThreadNum = omp_get_thread_num();
        int StartLetter = 'a'+ThisThreadNum*LettersPerThread;
        int EndLetter = 'a'+ThisThreadNum*LettersPerThread+LettersPerThread;

        for (i=StartLetter; i<EndLetter; i++) { printf("%c", i); }
    }
    printf("\n");
    return 0;
}
```

Debugging threaded applications is complex: debuggers change runtime performance, masking race conditions, and even print statements can mask issues via synchronization and OS functions. OpenMP distinguishes private/shared variables and inserts code. A debugger supporting OpenMP helps examine variables and step through threaded code; Intel® Inspector detects many hard-to-find threading errors analytically, and sometimes process of elimination suffices. **NOTE:** Intel® Inspector has been deprecated; see the Intel® Inspector End of Life Announcement.

**Most mistakes are race conditions**, mostly shared variables that should be private. Inspect variables inside parallel regions, then called functions. By default stack variables are private, but the C/C++ keyword `static` puts the variable on the global heap and thus shares it for OpenMP loops. `default(none)` finds hard-to-spot variables: every variable must then have a data-sharing attribute clause:

```c
#pragma omp parallel for default(none) private(x,y) shared(a,b)
```

Uninitialized variables are another common mistake: **private variables have no initial values on entering a parallel construct**. Use `firstprivate`/`lastprivate` only when necessary (extra overhead). If a bug remains, reduce scope: binary-hunt; force sections serial with `if(0)` on the parallel construct or by commenting out the pragma; force large chunks into critical sections and find code that works inside but fails outside. If still broken, run everything serially with `KMP_LIBRARY=serial`. If not using OpenMP API calls, compile without `/Qopenmp`/`-qopenmp` to confirm the serial version; if using API calls, use `/Qopenmp-stubs`/`-qopenmp-stubs`.

**Performance** depends largely on: underlying single-threaded code performance; CPU utilization, idle threads, and load balancing; the percentage executed in parallel; synchronization and communication among threads; thread create/manage/destroy/synchronize overhead, worsened by single↔parallel transitions (**fork-join transitions**); shared-resource limits (memory, bus bandwidth, CPU execution units); and memory conflicts from shared or falsely shared memory. Start from a properly constructed parallel algorithm — parallelizing a bubble-sort is not a good start, even in hand-optimized assembly. Keep scalability in mind: running well on 2 CPUs is less efficient than on n CPUs; OpenMP chooses the thread count, so programs working well regardless of thread count are desirable. Producer/consumer architectures are rarely efficient (made for two threads). Verify efficiency on the targeted Intel® architecture; a single-threaded version helps — turn off `/Qopenmp`/`-qopenmp` or build with `/Qopenmp-stubs`/`-qopenmp-stubs`, optimize that, then generate the multi-threaded version. Try the different scheduling clauses; if parallel-region overhead is large versus compute time, use an `if` clause to execute the section serially. See also OpenMP* Runtime Library Routines, Worksharing Using OpenMP*, `qopenmp, Qopenmp`, `qopenmp-stubs, Qopenmp-stubs`.

## OpenMP* Implementation-Defined Behaviors

See the OpenMP API specification for the full list; Internal Control Variables (ICVs) are discussed there.

| Name | Description |
|---|---|
| `single` construct | First thread to encounter it executes the structured block. |
| `teams` construct | Number of teams created is 1 if the `num_teams` clause is not specified. |
| `dist_schedule` clause, `distribute` construct | Without `dist_schedule`, the schedule for `distribute` is `static`. |
| `omp_set_num_threads` routine | If the argument is not a positive integer, Intel's OpenMP implementation sets the first element of the `nthreads-var` ICV of the current task to 1. |
| `omp_set_max_active_levels` routine | A negative argument is ignored; the last valid setting is used. |
| `omp_get_max_active_levels` routine | From within any explicit parallel region the binding thread set (and binding region if required) is the current task region. |
| `OMP_SCHEDULE` environment variable | If the value does not conform to the specified format, the `run-sched-var` ICV is set to `static`. |
| `OMP_NUM_THREADS` environment variable | If any list value is negative the whole list is ignored; if any value is zero it is set to 1. |
| `OMP_PROC_BIND` environment variable | If the value is not `true`, `false`, or a comma separated list of `master` (deprecated), `primary`, `close`, or `spread`, Intel's OpenMP implementation sets the `bind-var` ICV to `false`. |
| `OMP_DYNAMIC` environment variable | If neither `true` nor `false`, the `dyn-var` ICV is set to `false`. |
| `OMP_NESTED` environment variable | If neither `true` nor `false`, the `nest-var` ICV is set to `false`. |
| `OMP_STACKSIZE` environment variable | If the value does not conform to the format or the implementation cannot provide that stack size, the `stacksize-var` ICV is set to the default: **1MB to 4MB depending on the architecture**. On Linux* the implementation can set it up to **256MB**, respecting the OS stack size limit. |
| `OMP_MAX_ACTIVE_LEVELS` environment variable | If the value is a negative integer or exceeds the supported number of parallel levels, the `max-active-levels-var` ICV is set to 1. |
| `OMP_THREAD_LIMIT` environment variable | If the requested value exceeds the supported thread count, or is a negative integer, the `thread-limit-var` ICV is set to the maximum number of threads supported on the platform; if the value is zero it is set to 1. |
| Runtime library definitions | Intel's OpenMP implementation provides both `omp.h` and `omp-tools.h`. |

## OpenMP* Examples

**A Simple Difference Operator.** A simple parallel loop where the work per iteration differs; dynamic scheduling improves load balancing. `nowait` is used because there is an implicit barrier at the end of the parallel region, so a barrier at the end of the `for` region is unnecessary.

```c
void for1(float a[], float b[], int n) {
   int i, j;
   #pragma omp parallel shared(a,b,n)
   {
     #pragma omp for schedule(dynamic,1) private (i,j) nowait
      for (i = 1; i < n; i++)
         for (j = 0; j < i; j++)
           b[j + n*i] = (a[j + n*i] + a[j + n*(i-1)]) / 2.0;
   }
}
```

**Two Difference Operators: for Loop Version.** Two parallel loops fused to reduce fork/join overhead. The first `for` pragma has `nowait` because all data in the second loop differs from all data in the first.

```c
void for2(float a[], float b[], float c[], float d[], int n, int m) {
   int i, j;
   #pragma omp parallel shared(a,b,c,d,n,m) private(i,j)
   {
     #pragma omp for schedule(dynamic,1) nowait
     for (i = 1; i < n; i++)
       for (j = 0; j < i; j++)
         b[j + n*i] = ( a[j + n*i] + a[j + n*(i-1)] )/2.0;
     #pragma omp for schedule(dynamic,1) nowait
     for (i = 1; i < m; i++)
         for (j = 0; j < i; j++)
           d[j + m*i] = ( c[j + m*i] + c[j + m*(i-1)] )/2.0;
     }
}
```

**Two Difference Operators: sections Version.** Identical logic to the preceding `for` example but using `sections`; speedup is limited to two because there are only two units of work, whereas the `for` example has `(n-1) + (m-1)` units.

```c
void sections1(float a[], float b[], float c[], float d[], int n, int m) {
   int i, j;
   #pragma omp parallel shared(a,b,c,d,n,m) private(i,j)
   {
      #pragma omp sections nowait
      {
         #pragma omp section
           for (i = 1; i < n; i++)
             for (j = 0; j < i; j++)
               b[j + n*i] = ( a[j + n*i] + a[j + n*(i-1)] )/2.0;
          #pragma omp section
           for (i = 1; i < m; i++)
             for (j = 0; j < i; j++)
               d[j + m*i] = ( c[j + m*i] + c[j + m*(i-1)] )/2.0;
        }
     }
}
```

**Update a Shared Scalar.** A `single` construct updates an element of the shared array `a`. The optional `nowait` after the first loop is omitted because it is necessary to wait at the end of the loop before the `single` construct, to avoid a race condition.

```c
void sp_1a(float a[], float b[], int n) {
   int i;
   #pragma omp parallel shared(a,b,n) private(i)
   {
      #pragma omp for
        for (i = 0; i < n; i++)
          a[i] = 1.0 / a[i];
        #pragma omp single
          a[0] = MIN( a[0], 1.0 );
        #pragma omp for nowait
        for (i = 0; i < n; i++)
        b[i] = b[i] / a[i];
     }
}
```

**More Samples.** Additional OpenMP code samples are in the oneAPI Samples GitHub* repository; see also the OpenMP API Examples document on the OpenMP website.

## Gotchas & failure modes

- **Omitting `-qopenmp`/`/Qopenmp`** silently makes all pragmas comments → single-threaded build, no error; use `-qopenmp-stubs`/`/Qopenmp-stubs` if code calls OpenMP API routines.
- **GPU offload needs both switches**: `target` regions need `-fopenmp-targets=spir64` (Linux) / `/Qopenmp-targets=spir64` (Windows) *plus* `-qopenmp`; the SIMD model additionally needs `-fopenmp-target-simd` and `-fiopenmp`.
- **Races from shared variables** are the most common mistake. Everything is shared by default except the loop variable; `static` variables are shared even inside a parallel construct. Fix with `private(...)`, declaration inside the construct (no `static`), or `default(none)`.
- **Private variables are uninitialized** on entry to a parallel construct; `firstprivate`/`lastprivate` cost overhead.
- **Reduction variables** cannot be `const`, cannot be private in the parallel construct, and can appear in only one reduction.
- **`OMP_NUM_THREADS` + `KMP_HW_SUBSET` can over- or under-subscribe** (explicit Caution).
- **`KMP_HW_SUBSET` beyond system capacity is ignored after a warning** (e.g. `KMP_HW_SUBSET=24c,5t` on a 4-threads/core machine); an unspecified layer means *all* resources in it.
- **Setting `KMP_AFFINITY`/`GOMP_CPU_AFFINITY` too late has no effect**: before the first parallel region and before `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls.
- **`KMP_AFFINITY=disabled`** also disables the low-level API — `kmp_set_affinity`/`kmp_get_affinity` have no effect and return a non-zero error code.
- **`OMP_PROC_BIND=spread` vs `KMP_AFFINITY=balanced`**: `balanced` is CPU-only for **single socket** systems; `spread` works on all platforms including multi-socket.
- **Deprecated `logical`/`physical`** may become unsupported; they treat a single trailing integer as an **offset**, while `compact`/`scatter` treat it as a **permute**. The default changed from `norespect` to `respect` when `compact`/`scatter` were added, so bindings may differ for a partial initial affinity mask.
- **`respect` differs by OS**: Linux respects the initializing thread's mask; Windows respects the process mask. On Windows with multiple processor groups, `norespect` is default when the process affinity mask equals a single processor group.
- **`granularity=fine` equals `granularity=thread`** and always binds a thread to one OS processor; default granularity is `core`, and the Windows multi-group default is `group`.
- **APIC/CPUINFO limits**: only 256 unique APIC IDs in `cpuid` leaf 4 (leaf 11/31 unlimited); `siblings` is ignored, possibly producing `Physical node/pkg/core/thread ids not unique`.
- **`omp_const_mem_space` writes are undefined behavior.**
- **`omp_pause_resource*` works only on the host device**; **`ompx_target_register_host_pointer`/`ompx_target_unregister_host_pointer` are Linux-only.**
- **Mixing compilers**: implementations are not guaranteed interoperable. Linking with GCC/`gfortran` or MSVC needs the Intel compatibility library and suppression of the other runtime (`/nodefaultlib:vcomp libiomp5md.lib`; `-liomp5 -lpthread -L<install_dir>/lib`). Do not mix `gfortran` objects; use `ifx` for all Fortran sources or `gfortran` for all.
- **Use the matching `omp.h`**: if structures/classes contain members typed from `omp.h`, all sources using them need the same `omp.h`.
- **Static linking is not recommended** and is unavailable on Windows (dynamic only for performance and stubs libraries).
- **Runtime calls beat environment variables**, but calling runtime routines may initialize the runtime so later environment settings have no effect — use `kmp_set_defaults()`. Intel extension routines are not recognized by other OpenMP-compliant compilers and can break their link stage; `kmp_set_library()` must precede the first parallel region and `kmp_set_stacksize_s()` the first dynamically executed parallel region.
- **`kmp_free()` is mandatory** for `kmp_malloc()`/`kmp_calloc()`/`kmp_realloc()` memory; cross-thread free works but costs performance.
- **Low-level affinity is an expert interface**: it can bind machine-specific information into code, ignore MPI-set resource limits, and force bad affinities without warning.
- **Turnaround mode with over-allocated resources performs poorly** — use throughput. `omp_set_dynamic` is disabled by default; `omp_set_nested`/`omp_get_nested` are deprecated.
- **`omp_target_memcpy`/`omp_target_memcpy_rect` are unspecified inside a target region** and both contain a task scheduling point.

## Source map

- OpenMP* Support; Parallel Processing; Using Other Compilers; Add OpenMP* Support — pp. 666–667
- OpenMP Pragma Syntax; Compile the Application; Configure the OpenMP Environment — pp. 667–668
- Parallel Processing Model; Use Orphaned Pragmas; Data Environment — pp. 668–669
- Determine How Many Threads to Use; Binding Sets and Binding Regions — pp. 670–671
- Worksharing Using OpenMP*; Basics of Compilation; A Few Simple Examples — pp. 671–672
- Avoid Data Dependencies and Race Conditions; Manage Shared and Private Data — pp. 672–674
- Reductions — pp. 674–675; Load Balancing and Loop Scheduling — pp. 675–676
- OpenMP Tasking Model (task, clauses, scheduling, taskwait, taskyield) — pp. 676–678
- Control Thread Allocation (KMP_HW_SUBSET, KMP_AFFINITY compact/scatter, best setting) — pp. 678–680
- OpenMP* Library Support; Runtime Library Routines (all groups) — pp. 680–690
- Intel® Compiler Extension Routines (execution environment, stack size, memory allocation, thread sleep time, target memory allocation, target offload) — pp. 690–696
- OpenMP* Support Libraries; performance/stubs; execution modes; use the libraries (Linux/Windows, Visual Studio) — pp. 696–702
- Thread Affinity Interface; KMP_AFFINITY; affinity types; compact/scatter examples; permute/offset; modifiers — pp. 702–713
- Determine Machine Topology; KMP_CPUINFO_FILE and /proc/cpuinfo; Windows Processor Groups; KMP_TOPOLOGY_METHOD — pp. 713–716
- Explicitly Specify OS Processor IDs (GOMP_CPU_AFFINITY, KMP_AFFINITY); Low Level Affinity API — pp. 716–720
- OpenMP* Memory Spaces and Allocators — pp. 720–724
- OpenMP* Contexts; Context Selectors; Score and Match Context Selectors — pp. 724–727
- OpenMP* Offloading SPMD/SIMT and SIMD Models — pp. 727–728
- OpenMP* Advanced Issues; Performance — pp. 728–730
- OpenMP* Implementation-Defined Behaviors — pp. 730–732
- OpenMP* Examples (for1, for2, sections1, sp_1a, more samples) — pp. 732–733

