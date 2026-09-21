## 03 · Expressing parallelism: kernels, ranges, nd_range

### Kernel forms

| Form | Construct | Classes | Exposes |
|---|---|---|---|
| Basic data-parallel | `parallel_for(range, kernel)` | `range`, `id`, `item` | nothing locality-aware |
| ND-range data-parallel | `parallel_for(nd_range, kernel)` | `range`, `id`, `nd_range`, `nd_item`, `group`, `sub_group` | work-groups, sub-groups, group barriers, group-local memory, group functions, group algorithms (scan, reduce) |
| Hierarchical data-parallel | `parallel_for_work_group` / `parallel_for_work_item` | `h_item` | experimental nested-loop syntax |

- **Basic** = descriptive (implementation schedules; range = execution range, instance = `item`); **ND-range** = prescriptive (mapping obeyed; groups run in any order).
- **Hierarchical** experimental: immature compiler support, many performance features incompatible/inaccessible; spec advises new code against it.
- Kernel = one operation instantiated many times, not a loop; instances SPMD — assume parallel execution even when sequential.
- Dimensions `0`..`N-1`, `N-1` contiguous; multidim list/constructor arguments read left to right from dimension 0; >3 dimensions must be linearized manually.

**Handler submit overloads:**

```cpp
template <typename KernelName, typename KernelType>
void single_task(KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
void parallel_for(range<Dims> num_work_items, KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
void parallel_for(nd_range<Dims> execution_range, KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
void parallel_for_work_group(range<Dims> num_groups, KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
void parallel_for_work_group(range<Dims> num_groups,
                             range<Dims> group_size, KernelType kernel);
```

### Class accessors

All: `template <int Dimensions = 1>`; dims compile-time fixed, extents runtime; constructors take 1/2/3 `size_t`. `range`/`id` add `get(int)`, `operator[]`, arithmetic; `range` adds `size()` (product).

| Class | Extra template param | Accessors |
|---|---|---|
| `range<D>` | — | `get(int)`, `operator[](int)` (`size_t&` and const), `size()` |
| `id<D>` | — | `get(int)`, `operator[](int)` (`size_t&` and const) |
| `item<D>` | `bool WithOffset = true` | `get_id()`, `get_id(int)`, `operator[](int)`, `get_range()`, `get_range(int)`, `get_offset()`, `get_linear_id()` |
| `nd_range<D>` | — | `get_global_range()`, `get_local_range()`, `get_group_range()` |
| `nd_item<D>` | — | `get_global_id()`, `get_global_id(int)`, `get_global_linear_id()`, `get_global_range()`, `get_global_range(int)`, `get_local_id()`, `get_local_id(int)`, `get_local_linear_id()`, `get_local_range()`, `get_local_range(int)`, `get_group()`, `get_sub_group()` |
| `group<D>` | — | `get_id()`, `get_id(int)`, `get_linear_id()`, `get_group_range()`, `get_group_range(int)`, `get_local_range()`, `get_local_range(int)` |
| `sub_group` | not templated (`id<1>`/`range<1>`) | `get_group_id()`, `get_group_range()`, `get_local_id()`, `get_local_range()`, `get_max_local_range()` |

- `item` = execution range + index; `id` = index only. Both exist only as kernel-function arguments (no free queries).
- `nd_range` never mentions sub-groups: sub-group range not settable at construction nor queryable (devices often support one valid size).
- `group` accessors duplicated on `nd_item` (`group.get_group_id()` ≡ `item.get_group_id()`, `group.get_local_range()` ≡ `item.get_local_range()`); `sub_group` is the only sub-group route. `group::leader()`/`group::get_group_id()` let one work-item write a group result.
- Sub-group `get_local_range()` = sub-group size; `get_max_local_range()` = max over sub-groups in the parent work-group; may differ for tiny work-groups or dimensions not divisible by sub-group size.
- `group` reads better and enables generics via `sycl::is_group_v` (no official C++20 Group concept).

### Work-groups, sub-groups, size rules

- **Rule.** Work-group size must divide the ND-range exactly in each dimension.
- **Rule.** Work-groups run in any order; different work-groups cannot communicate/synchronize except via atomic memory operations to global memory.
- **Rule.** Within a work-group: local memory, barriers, fences, group functions, group algorithms (reductions, scans); no forward-progress guarantee — sequential execution between barriers/collectives is valid; hand-coded sync may deadlock; only provided barriers/collectives are guaranteed-safe (§05).
- **Rule.** Sub-group size is fixed and 1D per device + kernel + ND-range; dimension 2 of the ND-range/work-group becomes dimension 0 of the sub-group. No sub-group local memory; exchange via "shuffle" group algorithms; no required forward-progress guarantee.
- **Gotcha.** Work-item-to-sub-group mapping is neither queryable nor requestable: a `{4,4}` range with max sub-group size 8 may map to two sub-groups of 8 or four of 4 — use 1D work-groups, or a multidim work-group whose highest-numbered (contiguous) dimension divides by the required sub-group size.
- **Fact.** Work-group count need not relate to compute-unit count; groups may greatly exceed the simultaneously-executable count — do not use `max_compute_units` in program logic.
- **Fact.** Work-group size is per-kernel and runtime-configured. `reqd_work_group_size(dim0[, dim1[, dim2]])` requires a size, `work_group_size_hint(dim0[, dim1[, dim2]])` hints one, `reqd_sub_group_size(dim)` requires a sub-group size, `device_has(aspect, ...)` requires aspects (throws at submit if absent). Kernel/device sizing queries: §07, §11.

### Work mapping and kernel naming

- **One-to-one** — range size exactly matches work (simplest; implementation maps work-items to hardware). **Many-to-one** — range = number of workers, not work; needs a kernel parameter for total work plus a loop; round-robin start = global index, stride = total work-items (linear work-items touch contiguous memory); work may also be split across groups or within groups.
- Anonymous lambda kernels may need a kernel name template parameter to identify them uniquely: `h.parallel_for<class Add>(size, [=](id<1> i) { ... });`; optional for most SYCL 2020 compilers, unnamed lambdas preferred when no name is required (naming mechanics: §11).

### Reductions

- **Rule.** Combines partial results with an operator typically associative and commutative (e.g. addition); never assume a combination order.
- **Rule.** Result not written back until kernel completion; afterwards behaves like any other variable (accessor for a buffer; possibly explicit synchronization/memory movement for USM).
- **Rule.** The variable cannot be inspected during execution and cannot be updated except by the specified combination function.
- **Fact.** Buffer-/USM-initialized reduction is scalar (first object in an array); `span`-initialized is an array reduction with independent components — equivalent to N scalar reductions of the same type/operator.

Three families (buffer, USM pointer `T*`, `span<T, Extent>`), two overloads each (with/without identity), all taking a trailing `const property_list& properties = {}`:

```cpp
// buffer
template <typename BufferT, typename BinaryOperation>
unspecified reduction(BufferT variable, handler& h, BinaryOperation combiner,
                      const property_list& properties = {});
template <typename BufferT, typename BinaryOperation>
unspecified reduction(BufferT variable, handler& h, const BufferT::value_type& identity,
                      BinaryOperation combiner, const property_list& properties = {});
// USM pointer
template <typename T, typename BinaryOperation>
unspecified reduction(T* variable, BinaryOperation combiner,
                      const property_list& properties = {});
template <typename T, typename BinaryOperation>
unspecified reduction(T* variable, const T& identity, BinaryOperation combiner,
                      const property_list& properties = {});
// span
template <typename T, typename Extent, typename BinaryOperation>
unspecified reduction(span<T, Extent> variables, BinaryOperation combiner,
                      const property_list& properties = {});
template <typename T, typename Extent, typename BinaryOperation>
unspecified reduction(span<T, Extent> variables, const T& identity,
                      BinaryOperation combiner, const property_list& properties = {});
```

- **Fact.** Return type of `reduction` is unspecified; the `reduction` class is implementation-defined (different classes/algorithms allowed). Future versions may allow requesting an algorithm, likely via `property_list`.
- **Rule.** `combine()` merges one work-item's partial into the variable's value (implementation-defined); shorthand operators (`+=` for plus) are identical to `combine()`; array reductions add `operator[]` returning another `reducer` (not a reference) with the same `combine()`/shorthand operators.
- **Fact.** Built-in operators recognized by SYCL: `plus`, `multiplies`, `bit_and`, `bit_or`, `bit_xor`, `logical_and`, `logical_or`, `minimum`, `maximum`.
- **Rule.** Identity is auto-determined only for simple arithmetic types with a standard function object (e.g. `plus`); other reductions should supply `identity` to avoid detecting uninitialized private partials.
- **Rule.** User-defined reductions are limited to trivially copyable types and side-effect-free combination functions.

```cpp
template <typename T, typename BinaryOperation, /* implementation-defined */>
class reducer {
  void combine(const T& partial);   // combine partial result with reducer's value
};
// Other operators are available for standard binary operations
template <typename T>
auto& operator+=(reducer<T, plus::<T>>&, const T&);
```

```cpp
template <typename T, typename I>
using minloc = minimum<std::pair<T, I>>;
std::pair<float, int> identity = {
    std::numeric_limits<float>::max(), std::numeric_limits<int>::min()};
*res = identity;
auto red = sycl::reduction(res, identity, minloc<float, int>());
// kernel: h.parallel_for(range<1>{N}, red, [=](id<1> i, auto& res) {
//   res.combine({data[i], i}); });
```

### Group algorithms

Device-code collectives taking a `group`/`sub_group` object as first argument (in place of an execution policy); each acts as a barrier — all group work-items must reach it in converged control flow with identical arguments. Names: `sycl::joint_reduce`/`sycl::reduce_over_group`; `sycl::exclusive_scan`, work-group `inclusive_scan`, `sycl::exclusive_scan_over_group`; `sycl::joint_any_of`/`joint_all_of`/`joint_none_of` / `sycl::any_of_group`/`all_of_group`/`none_of_group`; `sub_group`-only `sycl::shift_group_left`, `sycl::shift_group_right`, `sycl::permute_group_by_xor`. `joint_*` vs `*_group` semantics and the full algorithm list: §06.

### Code

```cpp
// basic range kernel
h.parallel_for(range{N}, [=](id<1> idx) { c[idx] = a[idx] + b[idx]; });
// 2D id: subscript operators or indexing by id
h.parallel_for(range{N, N}, [=](id<2> idx) {
  int j = idx[0], i = idx[1];
  for (int k = 0; k < N; ++k) c[j][i] += a[j][k] * b[k][i]; // or c[idx] += a[id(j,k)] * b[id(k,i)]
});
// nd_range kernel
range global{N, N}, local{B, B};
h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
  int j = it.get_global_id(0), i = it.get_global_id(1); // ...
});
// many-to-one; N = work, W = workers; stride = total work-items
size_t N = ..., W = ...;
h.parallel_for(range{W}, [=](item<1> it) {
  for (int i = it.get_id()[0]; i < N; i += it.get_range()[0]) output[i] = function(input[i]);
});
// reduction as a parallel_for argument, not a body call
h.parallel_for(range<1>{N}, reduction(sum, plus<>()),
               [=](id<1> i, auto& sum) { sum += data[i]; });
// array reduction (histogram); span sizes the reduction
h.parallel_for(range{N}, reduction(span<int, 16>(histogram, 16), plus<>()),
               [=](id<1> i, auto& histogram) { histogram[i % B]++; });
```

**Fact.** A reduction argument makes the kernel function take an extra reducer (captured as `auto&`); ND-range forms accept them too: `h.parallel_for(nd_range<1>{...}, reduction(...), [=](nd_item<1> it, auto& ...) { ... })`.

### Key gotchas

- Work-group size must divide the ND-range exactly in each dimension — else illegal.
- Never sync across work-groups via scheduling assumptions; not guaranteed.
- Hand-coded intra-work-group sync may deadlock; use provided barriers/collectives.
- Do not assume sub-group size, mapping, or forward progress; query and size for it.
- Highest-numbered (contiguous) work-group dimension must divide by required sub-group size.
- Do not inspect or non-combiner-update a reduction variable in the kernel; result valid only after completion.
- Identity is auto-detected only for standard arithmetic types + standard function objects; supply it otherwise.
- User-defined reductions: trivially copyable types, side-effect-free combiners only.
- Anonymous kernel lambdas: add a kernel name template parameter when required, else use the unnamed form.
- Basic kernels get no local memory, barriers, sub-groups, or group algorithms — use nd_range.
- One-to-one kernels must not launch more range items than work exists; use many-to-one.
- Prefer runtime device/kernel queries over hard-coded work-group sizes or `max_compute_units`.
