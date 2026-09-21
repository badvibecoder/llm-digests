## 05 · Work-groups, sub-groups and intra-kernel communication

### Hierarchy and model

- **work-item** = one kernel instance, leaf of the hierarchy; runs in any order; communicates only via atomics to local/global memory or group collectives (`select_from_group`, `group_barrier`).
- **work-group** = grouping of work-items in an ND-range kernel; same-work-group work-items get extra scheduling guarantees ⇒ may cooperate and communicate safely.
- **sub_group** = implementation-defined subset of a work-group executing together on the same hardware resources or with extra scheduling guarantees; **no sub-group local memory**.
- Group concepts (barriers, local memory, collectives) exist **only** in ND-range kernels; `item` (basic data-parallel) cannot query a group, `nd_item` can.
- **Rule (communication).** No co-scheduling guarantee across work-groups ⇒ communicate only within your own work-group, else deadlock.
- **Rule (mapping).** Sub-group size is fixed (1-D) per device + kernel + ND-range; the work-item→sub-group mapping is implementation-defined (not queryable, no request mechanism). Range `{4,4}` with max sub-group size 8 ⇒ two sub-groups of 8 **or** four of 4; some mappings set the size to the extent of the highest-numbered (contiguous) dimension.
- **Rule (progress).** Sub-group work-items have no forward-progress guarantee; they may run sequentially, switching only at collectives. All-sub-group progress is implementation-defined, non-portable.

**Queries.** `item.get_group()` (work-group from `nd_item`), `item.get_sub_group()`, `item.get_local_id()`/`get_local_range()`/`get_global_id()` (each also `(dim)`), `g.get_group_id()`/`item.get_group(0)`, `g.leader()` (`if (g.leader())`); sub-group: `sg.get_local_id()[0]`, `sg.get_local_range()[0]` (size).

**Fact.** Not in source: `get_sub_group_size`, `get_sub_group_id`, `get_num_sub_groups`, `get_max_sub_group_size`, `get_sub_group_local_id`, `group_local_memory`, bare `broadcast`, `ext_oneapi_get_composite_device`.

### Work-group local memory

- **Rule.** Local memory = communication space for one work-group's work-items (USM/buffers would need a dedicated allocation partitioned across work-groups); **uninitialized** at work-group start, does **not** persist after it ends ⇒ temporary storage only. A work-group sees all global memory plus **only its own** local memory.
- **Fact.** Local memory is a software abstraction over global memory on many CPUs; dedicated hardware (faster) on many GPUs. `info::device::local_mem_type`: `info::local_mem_type::local` = dedicated storage (e.g. SRAM); `info::local_mem_type::global` = abstraction over global memory, possibly no perf gain.

```cpp
accessor dataAcc{dataBuf, h};                              // global
auto localIntAcc   = local_accessor<int, 1>(16, h);        // 1-D, 16 ints
auto localFloatAcc = local_accessor<float, 2>({4, 4}, h);  // 2-D, 4x4 floats
```

**Rule.** `local_accessor`: built in a command group handler from a **type + range**, never from a `buffer`; 1-D/2-D/3-D; always `read_write` (local memory starts uninitialized, results must be viewable); optionally atomic.

**Gotcha.** Stride equal to the bank count ⇒ every access conflicts, all serialize (worst); stride sharing no common factor with the bank count ⇒ full performance. Pad local structures to choose a good stride.

### Barriers

**Fact.** `group_barrier(group)` (1) synchronizes execution of the group's work-items and (2) synchronizes each work-item's view of memory (memory consistency / fencing). `group_barrier(item.get_group())` = work-group barrier (CUDA `__syncthreads`); `group_barrier(sg)` with `sg = item.get_sub_group()` = sub-group barrier (CUDA `__syncwarp`).

- **Rule.** All group work-items must execute the barrier or none may; branching around it leaves the rest waiting forever. Unlike CUDA `__syncthreads`, SYCL `group_barrier` synchronizes **all** work-items in the work-group, so early-exited work-items are not excused. Fix: move an early `return`/range check after the barrier, or drop it.
- **Rule (scope).** `group_barrier` takes one optional `fence_scope` argument for its memory operations; omitted ⇒ default scope derived from the group (usually correct, explicit scope rarely required). Scope affects **only** memory consistency — the synchronized work-item set is set solely by the group object.
- **Fact.** Every group barrier is by default an **acquire-release fence** for all address spaces reachable by the calling work-item, so preceding writes are visible to at least all work-items in the same group (the group's `fence_scope` member variable).
- **Fact.** `memory_scope` enumerators: `memory_scope::work_item` (image ops only), `memory_scope::sub_group`, `memory_scope::work_group`, `memory_scope::device`, `memory_scope::system`. `memory_scope::device` is legal on `group_barrier` but usually unnecessary and may cost more.
- **Fact.** Migrated `nd_item` barrier code: `item_ct1.barrier(sycl::access::fence_space::local_space);`.

### Collective functions and group algorithms

**Rule.** Collectives must be encountered by **all** group work-items in converged control flow (all encounter the call or none), with arguments agreeing on the operation; `joint_reduce` requires identical arguments from all work-items. First argument is a `group` **or** `sub_group` in place of an execution policy.

**Fact.** Only primitive data types and built-in operators `plus`, `multiplies`, `bit_and`, `bit_or`, `bit_xor`, `logical_and`, `logical_or`, `minimum`, `maximum`. "joint" algorithms read/write memory visible to all group work-items; "group" algorithms operate over the group itself with inputs/outputs in work-item private memory.

**Group-algorithm names** (C++ correspondences in §06). All group types: `sycl::joint_any_of`/`sycl::any_of_group`, `sycl::joint_all_of`/`sycl::all_of_group`, `sycl::joint_none_of`/`sycl::none_of_group`. `sub_group` only: `sycl::shift_group_left`, `sycl::shift_group_right`, `sycl::permute_group_by_xor`.

**Broadcast.** `group_broadcast(group, value, local_id)` copies `value` from work-item `local_id` to all group work-items; `local_id` must be group-uniform, hence so is the result. Work-groups and sub-groups; the sub-group form is a specialized `select_from_group` (group-uniform shuffle index) that may compile better.

**Votes.** `any_of_group(g, cond)` / `all_of_group(g, cond)` / `none_of_group(g, cond)`: `cond` true for ≥1 / all / none of the group; work-groups and sub-groups, useful for convergence tests. `joint_any_of`, `joint_all_of`, `joint_none_of` let group work-items cooperate over a data range.

**Shuffles (sub-group only).**
- `select_from_group(sg, x, id)` — most general; arbitrary pair communication via precomputed permutation indices; may cost more, prefer specialized forms.
- `shift_group_left(sg, x, n)` / `shift_group_right(sg, x, n)` — shift by a fixed number of elements; values returned to shifted-out work-items are **undefined**.
- `permute_group_by_xor(sg, x, mask)` — swaps values with the work-item whose sub-group local id XOR `mask` equals one's own; expresses neighbor swap and reversal.

**Reductions / scans.** `reduce_over_group`, `joint_reduce`, `exclusive_scan_over_group`, work-group `inclusive_scan` — see §06 (no device-wide barrier ⇒ a global scan needs multiple kernels).

### Sub-group sizes and SIMD on GPUs

**Fact.** On many GPUs a sub-group = work-items processed by one **instruction stream** (SIMD); they exchange data and synchronize inexpensively without local memory. SIMD widths of 16, 32 or more are common. SIMD efficiency = performance vs. scalar streams; divergent control flow runs both paths with channels masked/predicated, worst case cutting efficiency by the SIMD width.

**Rule.** Sub-group size is chosen by the implementation by default; it may differ per kernel and must not be assumed. A size can be requested at compile time but must be compatible with the device.

**Queries.** `info::device::sub_group_sizes` / `sub_group_sizes` = device-supported sizes; `compile_sub_group_size` / `compile_num_sub_groups` = size/count specified by a kernel, else `0`; `max_sub_group_size` = maximum for a kernel launched with the specified work-group size; `max_num_sub_groups` = maximum number of sub-groups for a kernel; `preferred_work_group_size_multiple` = make work-group size a multiple of this for best performance (must not exceed `work_group_size`).

**Fact.** `reqd_sub_group_size(dim)` requires the kernel to launch with the specified sub-group size (kernel queries need the kernel-bundle mechanisms of Ch. 10). A kernel violating a required condition (e.g. `sub_group_sizes`) produces a **runtime error**.

**Rule.** Size the work-group to maximize SIMD efficiency: a size not evenly divisible by the processor's SIMD width may leave channels disabled for the whole kernel; use `preferred_work_group_size_multiple`. A single-work-item work-group (`nd_range<1>{M, 1}`) is likely very slow — most GPUs mask off all but one SIMD channel.

```cpp
// reqd_sub_group_size attribute + sub-group barrier
const int n = 16, w = 16;
range<2> G = {n, w};  range<2> L = {1, w};
q.parallel_for(nd_range<2>{G, L},
  [=](nd_item<2> it) [[sycl::reqd_sub_group_size(w)]] {
    sub_group sg = it.get_sub_group();
    for (int j = sg.get_local_id()[0]; j < n; j += w) {
      auto va = a[it.get_global_id(0) * n + j + 1];
      group_barrier(sg);
      // ...
    }
    group_barrier(sg);
  }).wait();
```

### Canonical ND-range kernels

```cpp
// Tiled matrix multiply: local memory as explicit cache + work-group barriers
constexpr int tile_size = 16;
auto tileA = local_accessor<T, 1>(tile_size, h);
h.parallel_for(nd_range<2>{{M, N}, {1, tile_size}}, [=](nd_item<2> item) {
    int m = item.get_global_id()[0], n = item.get_global_id()[1], i = item.get_local_id()[1];
    T sum = 0;
    for (int kk = 0; kk < K; kk += tile_size) {
      tileA[i] = matrixA[m][kk + i];
      group_barrier(item.get_group());   // consistent view of the tile
      for (int k = 0; k < tile_size; k++) { /* ...  sum += tileA[k] * matrixB[kk + k][n]; */ }
    }
    group_barrier(item.get_group());     // all local reads complete
    matrixC[m][n] = sum;
});
```

**Rule.** Work-group-local use almost always needs barriers. Tile size need not equal the work-group size, but a multiple is common and simplifies transfers. A barrier is also required between one tile's compute and the next tile's load, else the current tile may be overwritten before another work-item finishes with it; end-of-loop and start-of-iteration placement are equally correct.

```cpp
// Sub-group exchange (shuffle): no local memory, no explicit sync
q.parallel_for(nd_range<2>{{2, 16}, {2, 16}}, [=](auto item) {
  auto index = item.get_global_linear_id();
  auto sg = item.get_sub_group();
  auto neighbor = permute_group_by_xor(sg, item.get_local_id(1), 1);
  buffer[index] = neighbor;
}).wait();
```

### Key gotchas
- Never branch around `group_barrier`; all group work-items must reach it or none — else deadlock.
- Communicate only within your own work-group; cross-work-group waits can deadlock (no co-scheduling guarantee).
- Local memory is uninitialized per work-group and dies with it; never rely on zeroed or persistent contents.
- A `local_accessor` must be `read_write`; built from type + range, not a buffer.
- Barrier scope does not change which work-items synchronize; only the group object does.
- Do not assume sub-group size, work-item mapping, or 32; query it and keep kernels size-agnostic.
- Shuffles are sub-group-only; `shift_group_left/right` returns **undefined** values to shifted-out work-items.
- Keep work-group size a multiple of the SIMD width / `preferred_work_group_size_multiple`; single-work-item work-groups waste all but one SIMD channel.
