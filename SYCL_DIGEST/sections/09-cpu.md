## 09 · Programming for CPUs (SYCL/DPC++ portability notes)

### CPU model: mapping and vocabulary

**Fact.** Typical CPU = cc-NUMA (cache-coherent non-uniform memory access): each socket owns a memory subset; cache-coherent interconnect gives one system memory view. Aggregate bandwidth scales with socket count; **latency is not fixed** — remote-socket access costs extra hops.

Reference topology used by the source: 2 sockets, 2 cores/socket, 4 hardware threads/core → hardware threads numbered 0–15. Hardware threads are the execution vehicles (instruction streams). Within a socket, memory latency is uniform and predictable.

**Fact.** Work-group → thread mapping is **not specified** by SYCL; implementations may expose tuning control via compiler options or environment variables. Work-groups may be formed **implicitly** by compiler/runtime from a plain `parallel_for(range)`, not only from `nd_range`.

Round-robin illustration (1 socket, 4 cores, 2 HW threads/core = 8 threads; 1024 elements; work-groups of 32 work-items → 32 work-groups; work-group scheduling round-robin, i.e. `thread-id = work-group-id mod 8`; 8 work-groups run concurrently per round; each thread executes 4 work-groups).

**Rule.** Eliminate dependences between work-items within a work-group when targeting CPUs. Fine-grained synchronization is costly on CPUs; thread context-switch overhead is high; more resident software threads than cores is beneficial (core can switch to a ready thread instead of idling).

**Rule.** Restrict unavoidable dependences to work-items **within one sub-group**, where the sub-group barrier (`group_barrier(sg)`) can be treated by the compiler as a **no-op** under a SIMD execution model — no runtime synchronization cost.

### SIMD / sub-group on CPU

| Fact | Detail |
|---|---|
| Fixed SIMD register widths (Intel Xeon) | 128-bit `XMM`, 256-bit `YMM`, 512-bit `ZMM` — not variable-length |
| Vector length / SIMD width | data elements processed simultaneously by one instruction |
| 512-bit regs | eight 64-bit (double) calculations per machine instruction |
| Sub-group size report | implementation doing loop vectorization instead of sub-group execution **likely reports sub-group size of one** |
| Vectorizer choice under ND-range | fastest-growing (unit-stride) dimension |
| Vectorization legality | no cross-work-item dependences in the same sub-group, or compiler must preserve cross-work-item **forward** dependences |
| `REQD_SUB_GROUP_SIZE` | `[[sycl::reqd_sub_group_size(N)]]`, sub-group execution model |

**Fact.** `vec` load/store/swizzle operate directly on vector variables, telling the compiler elements are contiguous from one uniform location → optimized contiguous loads/stores. Expect `vec` to eventually be **deprecated** in favor of a more explicit vector type (e.g. `std::simd`) once available.

**Fact.** Under `single_task` there are no work-items to vectorize over; compiler/runtime choose explicit SIMD or scalar. `vec`-typed variables inside `single_task` may map to SIMD instructions; swizzles may benefit if SIMD swizzle instructions exist.

```cpp
// vector types + swizzle under single_task
sycl::vec<int, 4> old_v = sycl::vec<int, 4>(0, 100, 200, 300);
sycl::vec<int, 4> new_v = sycl::vec<int, 4>();
new_v.rgba() = old_v.abgr();
```

**Gotcha.** Fixed vector sizes may not match SIMD register width across CPU generations/vendors → scalability limitation of `vec`-based interfaces.

### Masking

| Codegen | Cost |
|---|---|
| No masking | 2 `vmulps` per cycle |
| Merge masking | 2 `vmulps` every 4 cycles — **dependence on destination register** |
| Zero masking | no destination dependence → 2 `vmulps` per cycle |

**Rule.** Mask bits come from conditionals (`if`, `a = b > a ? a : b`, variable-trip loops, `switch`). Masking cost = extra mask blend on each load + destination dependence. Use only when necessary; balance masking vs. code branches.

**Rule.** ND-range work-group size should be evenly divisible by the processor's SIMD width to minimize masking; a non-divisible work-group leaves part of it executing masked.

**Fact.** Cache-aligned data > non-aligned. Compilers may peel/version loops (masked head to first aligned address, unmasked body, masked remainder) at the cost of code size; do the equivalent by hand or ensure allocations are appropriately aligned.

### Data layout & lanes

**Gotcha.** AOS (`struct {float x; float y; float z; float w;} a[4]`) → gathers/scatters: stride of `a[0].x, a[1].x, ...` is 4, not unit-stride 1. Hardware gather/scatter does not remove the need to transform — it needs significantly higher bandwidth and latency than contiguous loads.

**Rule.** Prefer SOA (`struct {float x[4]; float y[4]; float z[4]; float w[4];} a;`) → unit-stride (contiguous) vector loads/stores, e.g. `x[wi] = a.x[wi];` loads `x[0:4]`.

**Rule.** Do AOS→SOA (or AOSOA) at program level across all users of the structure; doing it at loop level adds costly format conversions before/after the loop. Compiler may still do vector-load-and-shuffle for AOS at some cost. Compilers may perform horizontal or vertical expansion for vector-typed members based on hardware.

**Gotcha.** `get_global_id(0)` returns `size_t` (64-bit) — do not narrow to `int`: `a[get_global_id(0)]` may become a unit-stride vector load, but `a[(int)get_global_id(0)]` may become a non-unit-stride gather, because narrowing conversion wraparound is unspecified/undefined behavior. Same detail applies to most `get_*_id()` family functions; bounded returns (e.g. max id in a work-group) usually fit `MAX_INT`.

**Fact.** DPC++ assumes no overflow of ID values and vectorizes accordingly. `-fno-sycl-id-queries-fit-in-int` tells the compiler overflow is possible and vectorized ID-derived accesses may be unsafe — large performance impact; use whenever assuming no overflow is unsafe.

**Rule.** Ensure global ID values fit in 32-bit `int`; otherwise use `-fno-sycl-id-queries-fit-in-int` for correctness, accepting lower performance.

### DPC++ CPU tuning knobs (not defined by SYCL)

| Env var | Values | Default | Effect |
|---|---|---|---|
| `DPCPP_CPU_CU_AFFINITY` | `spread`, `close` | — | bind software threads to hardware threads |
| `DPCPP_CPU_NUM_CUS` | `[n]` | number of hardware threads | threads used for kernel execution |
| `DPCPP_CPU_PLACES` | `sockets`, `numa_domains`, `cores`, `threads` | `cores` | places for affinity (like `OMP_PLACES` in OpenMP 5.1) |
| `DPCPP_CPU_SCHEDULE` | `dynamic`, `affinity`, `static` | `dynamic` | TBB partitioner for work-group scheduling |

Affinity formulas: `spread: boundHT = (tid mod numHT) + (tid mod numSocket) × numHT)`; `close: boundHT = tid mod (numSocket × numHT)`. `tid` = software thread id, `boundHT` = hardware thread bound, `numHT` = hardware threads per socket, `numSocket` = sockets.

| `DPCPP_CPU_SCHEDULE` | TBB partitioner | Behavior |
|---|---|---|
| `dynamic` | `auto_partitioner` | sufficient splitting to balance load among worker threads |
| `affinity` | `affinity_partitioner` | improves cache affinity, proportional splitting when mapping subranges to worker threads |
| `static` | `static_partitioner` | distributes iterations among worker threads as uniformly as possible |

Grain size controls TBB work splitting; default grain size 1 = all work-groups can execute independently. Work-group scheduling on Intel OpenCL CPU runtime is handled by Threading Building Blocks (TBB).

```bash
export DPCPP_CPU_PLACES=numa_domains
export DPCPP_CPU_CU_AFFINITY=close
```

Perf evidence: ~30% gain on Intel Xeon 2 sockets × 28 cores × 2 HW threads @ 2.5 GHz. `DPCPP_CPU_PLACES` recommended together with `DPCPP_CPU_CU_AFFINITY`. Lack of affinity tuning does not necessarily mean lower performance — parallel thread count often matters more than binding; benchmark.

### First touch / allocation

**Rule.** Memory is stored where first touched. Serial host initialization associates all memory with the host thread's socket; later access from other sockets crosses the interconnect. Parallelize initialization kernels to control first-touch placement across sockets (~2× gain on Intel Xeon in the source's example).

**Fact.** `malloc_shared<T>(n, q)` USM allocations and `buffer`/`accessor` both appear in the CPU code paths here; first-touch placement applies to the memory the kernel initializes.

```cpp
// sub-group vectorization of a forward (read-before-write) dependence
const int n = 16, w = 16;
range<2> G = {n, w};
range<2> L = {1, w};
q.parallel_for(nd_range<2>{G, L},
  [=](nd_item<2> it) [[sycl::reqd_sub_group_size(w)]] {
    const int i = it.get_global_id(0);
    sub_group sg = it.get_sub_group();
    for (int j = sg.get_local_id()[0]; j < n; j += w) {
      auto va = a[i * n + j + 1]; // load before update
      group_barrier(sg);
      a[i * n + j] = va + i + 2;
    }
    group_barrier(sg);
  }).wait();
```

Work-group formed as `(1, 8)`; loop iterations distributed over sub-group work-items, eight-way SIMD. Gains are largest when this loop dominates kernel time.

### Portability deltas vs GPU

| Dimension | CPU | GPU-path consequence |
|---|---|---|
| Work-item mapping | software threads on cores/HW threads; mapping unspecified | do not assume hardware-thread == work-item |
| Work-group size | choose to divide SIMD width and avoid masking | GPU uses hardware work-group sizing rules |
| Explicit grouping | needed only to force sub-group/SIMD structure | GPU generally requires/benefits from explicit local ranges |
| Local memory | no explicit fast local memory described | local memory is a GPU throughput lever |
| Barriers | expensive; sub-group barrier can become no-op | work-group barriers are the normal GPU sync |
| Sync strategy | remove intra-work-group dependences; confine to sub-group | GPU tolerates more fine-grained sync |
| Consistency | hardware cache-coherent across sockets; relaxed atomic order can avoid hardware consistency cost | GPU memory model needs explicit fences/atomics |
| Vectorization | mostly implicit; `vec`/swizzle explicit, fixed widths | GPU sub-group width is the SIMD unit |

### Key gotchas

- Never narrow `get_global_id()`/`get_*_id()` results to `int` — risks non-unit-stride gathers and UB wraparound.
- Use `-fno-sycl-id-queries-fit-in-int` when global IDs may exceed 32-bit `int`, or accept incorrect vectorization.
- Size ND-range work-groups as multiples of the SIMD width, or accept merge-masking costs.
- Set thread affinity (`DPCPP_CPU_CU_AFFINITY=close`, `DPCPP_CPU_PLACES=numa_domains`); no binding risks cache ping-pong.
- Parallelize initialization kernels — serial first touch pins all memory to one socket.
- Convert AOS to SOA at program level, not per-loop; loop-level conversion adds transform costs.
- Do not assume a sub-group size > 1 on CPU; implementations may report sub-group size one.
- Replace cross-work-item dependences with sub-group-scoped dependences plus `group_barrier(sg)`.
- Do not rely on explicit `vec` widths surviving CPU generation/vendor SIMD-width changes.
- Keep `DPCPP_CPU_NUM_CUS` at the hardware-thread count by default; both over- and under-subscription lose throughput.
- Masking costs a blend per load and a destination dependence — prefer zero masking or branches.
