## 06 · Common parallel patterns and SYCL idioms

### Pattern vocabulary

| Pattern | Definition | Constraints / affinity |
|---|---|---|
| Map | one output per input element, independent | any device; scales with hardware parallelism; no inter-element reuse |
| Stencil | function of an element **and** its stencil neighbors → one output | out-of-place → independent; in-place cuts footprint; small → GPU scratchpad, large → CPU caches, small inputs on FPGA → systolic arrays; efficient blocking needs compile-time block/neighborhood/function knowledge |
| Reduction | combine partials with an operator typically associative and commutative (e.g. addition) | tree = `log2(N)` combinations; combination order not assumed; balance partial-compute vs. combine; reduce on the producing device |
| Scan | generalized prefix sum over a binary associative operator | inclusive: element `i` = sum over `[0, i]`; exclusive: `[0, i)`; small problems fit CPUs, FPGAs pipeline; no ND-range-wide barrier → multiple kernels via global memory |
| Pack | discard by Boolean condition, pack kept elements contiguously | mask precomputed or online; output index = exclusive scan over the condition; preserves order → `std::copy_if`, `std::stable_partition` (`std::partition` needs no order) |
| Unpack | contiguous input elements → noncontiguous output, others untouched | inverse of pack; fills "gaps"; also built on scan |

### SYCL reduction library

```cpp
h.parallel_for(
    range<1>{N}, reduction(sum, plus<>()),
    [=](id<1> i, auto& sum) { sum += data[i]; });
```

- **Rule.** Body references no reduction; the variable + functor suffice for an optimized reduction sequence.
- **Rule.** Result not written back until kernel completion; buffer → device/host accessor, USM → may need synchronization/memory movement.
- **Rule.** Variable opaque during execution: no intermediate values, no update except through the specified combination function.
- **Rule.** `reducer<T, BinaryOperation>::combine(const T&)` merges one work-item's partial into the reduction variable (behavior implementation-defined); shorthand operators (`+=` for `plus`) are equivalent to `combine()`.
- **Rule.** Array reductions add `operator[]`, returning another `reducer` (not an element reference) with the same `combine()`/shorthand operators.
- Overload families (buffer / USM `T*` / `span<T, Extent>`, each with and without identity), return type, `property_list`, and built-in operators: see §03.

Array reduction histogram (`span` sizes it; `operator[]` touches only the updated bin):

```cpp
h.parallel_for(
    range{N},
    reduction(span<int, 16>(histogram, 16), plus<>()),
    [=](id<1> i, auto& histogram) { histogram[i % B]++; });
```

#### User-defined reductions

**Why.** Tree reductions accumulate per-work-item partials in private variables: initializing from the first contribution costs uninitialized-value tracking; initializing to a known operator identity is cheaper.
- **Rule.** Identity auto-determined only for simple arithmetic types with standard function objects (e.g. `plus`).
- **Rule.** User-defined reductions: trivially copyable types, combination functions with no side effects; passing the identity explicitly can improve performance.

```cpp
template <typename T, typename I>
using minloc = minimum<std::pair<T, I>>;
*res = identity;
auto red = sycl::reduction(res, identity, minloc<float, int>());
// in kernel:
std::pair<float, int> partial = {data[i], i};
res.combine(partial);
```

### Group algorithms

- **Rule.** Called from device code already executing in parallel; first argument is a `group`/`sub_group` in place of an execution policy.
- **Rule.** Acts like a group barrier: all group work-items must reach the same algorithm in converged control flow and agree on the operation (`sycl::joint_reduce` needs identical arguments across work-items).
- **Rule.** Primitive types and built-in operators only: `plus`, `multiplies`, `bit_and`, `bit_or`, `bit_xor`, `logical_and`, `logical_or`, `minimum`, `maximum`.
- **Rule.** `joint_` prefix → STL-like, collaborative, on memory visible to all group work-items. `_group` suffix → operates on an implicit range reflecting the group, on work-item private memory.

| C++ Algorithm | SYCL "Joint" | SYCL "Group" | Group Types |
|---|---|---|---|
| `std::any_of` | `sycl::joint_any_of` | `sycl::any_of_group` | All |
| `std::all_of` | `sycl::joint_all_of` | `sycl::all_of_group` | All |
| `std::none_of` | `sycl::joint_none_of` | `sycl::none_of_group` | All |
| `std::shift_left` | N/A | `sycl::shift_group_left` | `sub_group` |
| `std::shift_right` | N/A | `sycl::shift_group_right` | `sub_group` |
| N/A | N/A | `sycl::permute_group_by_xor` | `sub_group` |

`sub_group`-only rows are the shuffle operations. Other collectives used below: `sycl::joint_reduce`, `sycl::reduce_over_group`, `sycl::exclusive_scan`, `sycl::exclusive_scan_over_group`, work-group `inclusive_scan`.

```cpp
// work-group reduces a memory range; elements auto-distributed over the group
int sum = joint_reduce(
    g, input + g.get_group_id() * elements_per_reduction,
    input + (g.get_group_id() + 1) * elements_per_reduction, plus<>());
// work-group reduces private values; each work-item contributes one value
int sum = reduce_over_group(g, x, plus<>());
```

### Direct programming

**Map** — basic parallel kernel: `q.parallel_for(N, [=](id<1> i) { output[i] = sqrt(input[i]); }).wait();`

#### Stencil — spatial blocking in work-group local memory

Naïve form indexes the halo and re-reads neighbors from global memory.

Recipe: local range `B×B`; tile range = `local_range + range<2>(2, 2)` (halo included); `local_accessor<float, 2>(tile_size, h)`; strided tile load incl. halo; `group_barrier(it.get_group())` between load and use; compute from tile with local indices shifted `+1`.

```cpp
q.submit([&](handler& h) {
  accessor input{input_buf, h};
  accessor output{output_buf, h};
  constexpr size_t B = 4;
  range<2> local_range(B, B);
  range<2> tile_size = local_range + range<2>(2, 2);
  auto tile = local_accessor<float, 2>(tile_size, h);
  h.parallel_for(
      nd_range<2>(stencil_range, local_range),
      [=](nd_item<2> it) {
        id<2> lid = it.get_local_id();
        range<2> lrange = it.get_local_range();
        for (int ti = lid[0]; ti < B + 2; ti += lrange[0]) {
          int gi = ti + B * it.get_group(0);
          for (int tj = lid[1]; tj < B + 2; tj += lrange[1]) {
            int gj = tj + B * it.get_group(1);
            tile[ti][tj] = input[gi][gj];
          }
        }
        group_barrier(it.get_group());
        int gi = it.get_global_id(0) + 1;
        int gj = it.get_global_id(1) + 1;
        int ti = it.get_local_id(0) + 1;
        int tj = it.get_local_id(1) + 1;
        float self = tile[ti][tj];
        float north = tile[ti - 1][tj];
        float east = tile[ti][tj + 1];
        float south = tile[ti + 1][tj];
        float west = tile[ti][tj - 1];
        output[gi][gj] = (self + north + east + south + west) / 5.0f;
      });
});
```

#### Reduction, both ways

```cpp
// naive: one atomic operation per work-item
q.parallel_for(N, [=](id<1> i) {
  atomic_ref<int, memory_order::relaxed,
            memory_scope::system,
            access::address_space::global_space>(
      *sum) += data[i];
}).wait();

// work-group reduction, one atomic per group
q.parallel_for(nd_range<1>{N, B}, [=](nd_item<1> it) {
  int i = it.get_global_id(0);
  auto grp = it.get_group();
  int group_sum = reduce_over_group(grp, data[i], plus<>());
  if (grp.leader()) {
    atomic_ref<int, memory_order::relaxed,
              memory_scope::system,
              access::address_space::global_space>(
        *sum) += group_sum;
  }
}).wait();
```

**Fact.** Choice depends on atomic support, work-group local memory size, global memory size, fast device-wide barriers, dedicated reduction instructions; some architectures prefer (or require) a tree reduction via `log2(N)` kernel calls.
**Rule.** Hand-write reductions only for cases the SYCL reduction library cannot express or for device-specific tuning, and only after confirming built-in reductions underperform.

#### Scan (inclusive, device-wide) — three kernel enqueues

```cpp
// Phase 1: per-block inclusive scan; block total -> tmp[group_id]
q.submit([&](handler& h) {
  auto local = local_accessor<int32_t, 1>(L, h);
  h.parallel_for(nd_range<1>(N, L), [=](nd_item<1> it) {
    int i = it.get_global_id(0), li = it.get_local_id(0);
    local[li] = input[i];
    group_barrier(it.get_group());
    for (int32_t d = 0; d <= log2((float)L) - 1; ++d) {   // manual local tree
      uint32_t stride = (1 << d);
      int32_t update = (li >= stride) ? local[li - stride] : 0;
      group_barrier(it.get_group());
      local[li] += update;
      group_barrier(it.get_group());
    }
    output[i] = local[li];
    if (li == it.get_local_range()[0] - 1) tmp[it.get_group(0)] = local[li];
  });
}).wait();
// Phase 2: scan the G block totals in one work-group (size G, in/out = tmp)
// Phase 3: add the preceding block total to each element
q.parallel_for(nd_range<1>(N, L), [=](nd_item<1> it) {
  int g = it.get_group(0);
  if (g > 0) output[it.get_global_id(0)] += tmp[g - 1];
}).wait();
```

Phases 1–2 differ only in range size and I/O handling; work-group `inclusive_scan` replaces the manual tree (one barrier per sweep).

#### Pack / Unpack

```cpp
// group pack on top of an exclusive scan
uint32_t index = exclusive_scan(g, (uint32_t)predicate, plus<>());
if (predicate) dst[index] = value;

// unpack on top of an exclusive scan
uint32_t index = exclusive_scan(sg, (uint32_t)predicate, plus<>());
return (predicate) ? new_value[index] : original_value;
```

Device-wide pack (all ND-range elements) likewise needs global memory and multiple enqueues (it depends on an exclusive scan); group/sub-group scope does not.

```cpp
// sub-group pack building a neighbor list
sub_group sg = it.get_sub_group();
uint32_t k = 0;
for (int j = sg.get_local_id()[0]; j < N; j += sg.get_local_range()[0]) {
  float r = distance(position[i], position[j]);
  uint32_t pack = (i != j) and (r <= CUTOFF);
  uint32_t offset = exclusive_scan_over_group(sg, pack, plus<>());
  if (pack) neighbors[i * MAX_K + k + offset] = j;
  k += reduce_over_group(sg, pack, plus<>());   // packed so far
}
num_neighbors[i] = reduce_over_group(sg, k, maximum<>());
```

```cpp
// unpack: load balancing under divergent control flow
while (any_of_group(sg, i < Nx)) {
  uint32_t converged = next_iteration(params, i, j, count, cr, ci, zr, zi,
                                      mandelbrot);
  if (any_of_group(sg, converged)) {
    uint32_t index = exclusive_scan_over_group(sg, converged, plus<>());
    i = (converged) ? iq + index : i;  // converged pixels replaced, others kept
    iq += reduce_over_group(sg, converged, plus<>());
  }
}
if (converged) reset(params, i, j, count, cr, ci, zr, zi);
```

**Gotcha.** Unpack load balancing pays completion-check + unpack overhead; gate it on a heuristic (e.g. active work-items below a threshold).

### Key gotchas

- Never read a `reduction` variable mid-kernel; combine it only with the specified operator.
- Results valid only after kernel completion; buffer → accessor, USM → synchronization.
- Never assume a reduction/scan combination order; use associative (commutative where required) operators.
- Stage stencil tiles in `local_accessor`; direct global neighbor reads repeat memory traffic.
- `group_barrier` between tile load and tile use, or halo cells are garbage; tile range must include the halo.
- Device-wide scan needs multiple kernels; SYCL has no ND-range-wide barrier.
- `joint_*` / `*_group` need converged control flow and identical arguments across the group.
- Group algorithms take only primitive types and built-in operators, not user-defined ones.
- `shift_group_left`, `shift_group_right`, `permute_group_by_xor` are `sub_group`-only.
- Array-reduction `operator[]` returns a `reducer`, not an element reference.
- Prefer vendor libraries and the SYCL reduction library; measure before replacing with kernels.
