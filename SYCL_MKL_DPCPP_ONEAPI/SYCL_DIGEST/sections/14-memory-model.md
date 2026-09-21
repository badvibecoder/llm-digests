## 14 · Memory model, atomics and synchronization primitives

### Model & vocabulary

- Work-item ≡ C++ "thread of execution" (C++17), **weakly parallel forward progress guarantees**; a memory order constrains **all other memory operations (to any address)**. Scheduling: §04.
- **Data race.** `data[i] += x` = load → add → store; concurrent non-atomic access can lose an update. Mixing atomic and non-atomic accesses to the same location = **undefined behavior**.
- Atomics on the same memory do not overlap, but only if *all* accesses there are atomic. Atomicity removes overlap, not nondeterminism: no ordering guarantee vs non-atomic ops.

### `memory_order` enumerators

C++ `consume` (similar to `acquire`; discouraged since C++17) is not part of SYCL.

| Enumerator | Exact semantics |
|---|---|
| `memory_order::relaxed` | Reads/writes can be reordered before or after the operation with no restrictions. No ordering guarantees. |
| `memory_order::acquire` | Read/write operations appearing **after** the operation in the program must occur after it (cannot be reordered before it). |
| `memory_order::release` | Read/write operations appearing **before** the operation must occur before it; preceding writes are guaranteed visible to other work-items synchronized by a corresponding acquire operation (an atomic op on the same variable using `memory_order::acquire`, or a barrier function). |
| `memory_order::acq_rel` | Acts as both acquire and release. Reads/writes cannot be reordered around the operation; preceding writes made visible as for `memory_order::release`. |
| `memory_order::seq_cst` | Acts as acquire, release, or both depending on whether the operation is a read, a write, or a read–modify–write. All operations with this order are observed in one sequentially consistent order. |

Loads: `relaxed`/`acquire`/`seq_cst` (loads do not write → no release). Stores: `relaxed`/`release`/`seq_cst` (stores do not read → no acquire). RMW (`fetch_*`, `exchange`, `compare_exchange_*`) and fences: all five.

- Acquire-release communicates values between thread pairs through memory; relaxed suits shared counters/histograms and cannot implement a lock.
- `seq_cst` adds one global order over all atomics, ordering communication between *groups*, not just pairs, of threads.

### `memory_scope` enumerators

Scope = **minimum set of work-items** the constraint applies to. C++ assumes one device/address space; SYCL has host + accelerators and private/local/global address spaces, possibly disjoint per device.

| Enumerator | Exact semantics |
|---|---|
| `memory_scope::work_item` | Constraint applies only to the calling work-item. Only useful for image operations — all other operations within a work-item already execute in program order. |
| `memory_scope::sub_group` | Constraint applies only to work-items in the same sub-group as the calling work-item. |
| `memory_scope::work_group` | Constraint applies only to work-items in the same work-group as the calling work-item. |
| `memory_scope::device` | Constraint applies only to work-items executing on the same device as the calling work-item. |
| `memory_scope::system` | Constraint applies to all work-items in the system. |

**Rule.** Barring device capability limits, all scopes are valid arguments to all atomic and fence operations. Scope may be **automatically demoted** to a narrower scope in exactly three situations:

1. Atomic operation updates work-group local memory → any scope broader than `memory_scope::work_group` is narrowed (local memory is visible only within the work-group).
2. Device does not support USM → `memory_scope::system` is always equivalent to `memory_scope::device` (buffers cannot be accessed concurrently by multiple devices).
3. Atomic operation uses `memory_order::relaxed` → no ordering guarantees exist and the scope argument is effectively ignored.

### Fences and barriers

| API | Form | Behavior |
|---|---|---|
| `group_barrier` | `group_barrier(group)`, `group_barrier(group, fence_scope)` | Synchronization point **and** acquire-release fence to all address spaces accessible by the calling work-item; preceding writes become visible to at least all other work-items in the same group (the group's `fence_scope` member variable). Explicit `memory_scope` overrides the default. Synchronizes within one group only, never across groups. |
| `atomic_fence` | `atomic_fence(order, scope)` | Ordering only, no waiting; caller chooses `memory_order` and `memory_scope`. |

- A wider `group_barrier` scope (e.g. `memory_scope::device`) is usually unnecessary.
- `access::fence_space` is the migrated form (`item_ct1.barrier(sycl::access::fence_space::local_space);`); `group_barrier` takes a `memory_scope` `fence_scope` instead. See §05.

### Device capability queries

| Query | Supported by | Guaranteed minimum on all devices |
|---|---|---|
| `atomic_memory_order_capabilities` | atomic operations on a device | `memory_order::relaxed` |
| `atomic_fence_order_capabilities` | fence operations | `memory_order::relaxed`, `memory_order::acquire`, `memory_order::release`, `memory_order::acq_rel` |
| `atomic_memory_scope_capabilities` | atomic operations | `memory_scope::work_group` |
| `atomic_fence_scope_capabilities` | fence operations | `memory_scope::work_group` |

**Why.** Fences are essential for reasoning about memory order with barriers, so the fence order minimum exceeds the atomic minimum; the queries cover hardware that cannot support the full C++ memory model.

**Rule.** Two portable strategies: (1) `seq_cst` + system fences, relaxing only when tuning — matches C++, simplest, slowest; (2) relaxed + work-group fences, strengthening only where correctness requires — matches OpenCL/pre-SYCL-2020 defaults, portable, more complex.

### `atomic` (deprecated) vs `atomic_ref` vs `std::atomic`

| Class | Ownership / copy / default | Kernel usability |
|---|---|---|
| `std::atomic` (C++11) | owns data; not movable/copyable | **compiler error in device kernels** — host objects cannot be transferred; fine in host code |
| `cl::sycl::atomic` (SYCL 1.2.1) | does **not** own data; defaults to relaxed | **deprecated in SYCL 2020**; use `atomic_ref` |
| `std::atomic_ref` (C++20) / `sycl::atomic_ref` | does not own data; built from an existing non-atomic variable | every access must be atomic for the reference's lifetime — host creates/transfers non-atomic data, device treats it as atomic |

### The `atomic_ref` class

```cpp
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace> class atomic_ref {
  using value_type = T;
  static constexpr size_t required_alignment = /* implementation-defined */;
  static constexpr bool is_always_lock_free = /* implementation-defined */;
  static constexpr memory_order default_read_order = memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order = memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;
  explicit atomic_ref(T& obj);
  atomic_ref(const atomic_ref& ref) noexcept;
};
```

**Fact.** Three template args beyond `std::atomic_ref`: `DefaultOrder`, `DefaultScope`, `AddressSpace` (optional; omitted = any address space — set explicitly to avoid overheads). `required_alignment` and `is_always_lock_free` are implementation-defined constants.
**Rule.** `DefaultOrder`/`DefaultScope` are part of the type, applying only where defaults cannot be overridden (e.g. `+=`); other orders/scopes are runtime arguments. `seq_cst`/`system` defaults restrict code to the most capable devices; `relaxed`/`work_group` defaults surprise migrating C++ code.
**Rule (lifetime).** While any `atomic_ref` to an object exists, **all** accesses to it must be atomic.

All types:

```cpp
void store(T operand, memory_order order = default_write_order, memory_scope scope = default_scope) const noexcept;
T operator=(T desired) const noexcept;   // equivalent to store
T load(memory_order order = default_read_order, memory_scope scope = default_scope) const noexcept;
operator T() const noexcept;   // equivalent to load
T exchange(T operand, memory_order order = default_read_modify_write_order, memory_scope scope = default_scope) const noexcept;
bool compare_exchange_weak(T &expected, T desired, memory_order success, memory_order failure, memory_scope scope = default_scope) const noexcept;
bool compare_exchange_weak(T &expected, T desired, memory_order order = default_read_modify_write_order, memory_scope scope = default_scope) const noexcept;
bool compare_exchange_strong(T &expected, T desired, memory_order success, memory_order failure, memory_scope scope = default_scope) const noexcept;
bool compare_exchange_strong(T &expected, T desired, memory_order order = default_read_modify_write_order, memory_scope scope = default_scope) const noexcept;
```

Integral only — `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_min`, `fetch_max`, each `Integral fetch_X(Integral operand, memory_order order = default_read_modify_write_order, memory_scope scope = default_scope) const noexcept`; plus:

```cpp
Integral operator++(int) const noexcept;   Integral operator--(int) const noexcept;
Integral operator++() const noexcept;      Integral operator--() const noexcept;
Integral operator+=(Integral) const noexcept;   Integral operator-=(Integral) const noexcept;
Integral operator&=(Integral) const noexcept;   Integral operator|=(Integral) const noexcept;
```

Floating-point only — `fetch_add`, `fetch_sub`, `fetch_min`, `fetch_max`, same shape with `Floating`; plus:

```cpp
Floating operator+=(Floating) const noexcept;   Floating operator-=(Floating) const noexcept;
```

**Fact.** Covers all fundamental integer, floating-point, and pointer types. Only 32-bit atomic types are guaranteed on all devices; 64-bit support is optional. Atomic FP types must be supported even without native hardware support; many devices emulate FP addition via compare-exchange.

### Atomics with buffers vs USM

```cpp
// buffers: non-atomic storage; accessor needs read/write permission
atomic_ref<int, memory_order::relaxed, memory_scope::system,
          access::address_space::global_space> a(acc[j]);
a += 1;                              // += shorthand for fetch_add
// USM: same, with atomic_ref(data[j]) — no buffer or accessor
```

**Fact.** Atomic data cannot be allocated and moved host↔device; allocate non-atomic buffer/USM data and access it through an atomic reference. Use atomics only for concurrently accessed locations, or only between two work-group barriers when work-items touch local memory concurrently.

### Real usage: histogram, relaxed atomics + barriers

```cpp
auto local = local_accessor<uint32_t, 1>{B, h};
h.parallel_for(nd_range<1>{num_groups * num_items, num_items}, [=](nd_item<1> it) {
  auto grp = it.get_group();
  for (int32_t b = it.get_local_id(0); b < B; b += it.get_local_range(0)) local[b] = 0;  // P1: no atomics
  group_barrier(grp);
  const auto [group_start, group_end] = distribute_range(grp, N);
  for (int i = group_start + it.get_local_id(0); i < group_end; i += it.get_local_range(0)) {
    atomic_ref<uint32_t, memory_order::relaxed, memory_scope::work_group,     // P2: local writes
              access::address_space::local_space>(local[input[i] % B])++;
  }
  group_barrier(grp);
  for (int32_t b = it.get_local_id(0); b < B; b += it.get_local_range(0))     // P3: global writes
    atomic_ref<uint32_t, memory_order::relaxed, memory_scope::system,
              access::address_space::global_space>(histogram[b]) += local[b];
});
```

**Fact.** Barrier = sync point + acquire-release fence, so phase writes are visible to the work-group later. P1: independent locations → no atomicity. P2: same local locations → `relaxed` + `work_group`, sync deferred to phase end. P3: independent local reads, colliding global writes → `relaxed` + `system`.

### Real usage: device-wide latch (non-portable, expert-only)

**Rule.** Device-wide synchronization is **not portable**; ordering guarantees are orthogonal to forward progress guarantees and work-group scheduling is implementation-defined (§04), so the spin may deadlock. Correctness requires all three: (1) orders at least as strict as `memory_order::acq_rel`; (2) each group's elected leader progresses independently, else a spinning work-item starves work-items yet to increment the counter; (3) all ND-range work-groups execute simultaneously with strong forward progress guarantees. The cross-group handoff uses `acq_rel` + `memory_scope::device` on the shared counter, spinning on a `load()` that synchronizes with prior releases:

```cpp
group_barrier(grp);              // elect one work-item per work-group
if (grp.leader()) {
  atomic_ref<size_t, memory_order::acq_rel, memory_scope::device,
            access::address_space::global_space> atomic_counter(counter);
  atomic_counter++;              // signal arrival, visible device-wide
  while (atomic_counter.load() != expected) { }  // sync with prior releases
}
group_barrier(grp);
```

### Key gotchas

- Mixing atomic and non-atomic accesses to the same location = undefined behavior.
- Atomicity gives non-overlap, not order.
- `std::atomic` cannot appear in a device kernel; `cl::sycl::atomic` is deprecated in SYCL 2020 — use `atomic_ref`.
- While any `atomic_ref` to an object lives, every access to it must go through `atomic_ref`.
- Loads cannot use `memory_order::release`; stores cannot use `memory_order::acquire`; `memory_order::consume` is not in SYCL.
- `memory_order::relaxed` makes the scope argument effectively ignored.
- Local-memory atomics demote scopes broader than `memory_scope::work_group`; non-USM devices degrade `system` to `device`.
- Give `atomic_ref` an explicit `access::address_space` — omitting it costs performance.
- Only 32-bit atomic types are guaranteed; 64-bit support is optional.
- Device-wide spin-latch synchronization is non-portable and can deadlock.
- A group barrier synchronizes only its own group.
