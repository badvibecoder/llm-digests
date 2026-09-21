## 02 · Memory: USM, buffers, accessors, data movement

### Model

Three abstractions: **USM** (pointer-based), **`buffer`** (data, not address-bound), **`image`** (special buffer: image formats, `sampler` reads; no further API in source). Buffers/images are accessed only via `accessor` objects.

| Strategy | Mover | Trade-off |
|---|---|---|
| Explicit | programmer (`memcpy`/`copy`) | full control of how much/when; overlaps compute with transfer; tedious, error-prone |
| Implicit | runtime/driver | less effort, fewer errors; less control, may not overlap optimally |

**Fact.** USM device allocations ⇒ explicit movement; host and shared allocations ⇒ implicit.

### Decision table: USM vs buffers

| Condition | Choose |
|---|---|
| Full control of all movement from the start | explicit USM device allocations |
| Porting pointer-based C/C++ (`malloc`/`new`, linked lists, trees) | USM |
| Prefer dependences between *kernels* over *data* | buffers |
| Prefer ordering by computation (in-order queue, events, `wait`) | USM |
| Must work on every device | buffers (some devices support no/limited USM modes) |

Styles may be mixed for different data.

### USM allocation types

| Type | Lives in | Host-accessible | Migrates host↔device | Notes |
|---|---|---|---|---|
| device | device-attached ((G)DDR, HBM) | no | n/a | kernels of that device; explicit `memcpy` only |
| host | host memory | yes | no | same pointer valid host+device; device access remote over PCI-E |
| shared | host or device-attached | yes | yes, automatically | device hits device-local memory after migration |

**Why host.** Rarely accessed data; dataset too large for device memory; device lacks shared-allocation support.

**Gotcha.** Touching a device allocation on the host ⇒ incorrect data or crash. A device may support none, some, or all USM types.

### USM allocation API

```cpp
// C-style: size in bytes, returns void*, type-agnostic. Named forms also exist
// as (size_t size, const queue&); aligned_alloc_device|_host|_shared prepend size_t alignment.
void *malloc_device(size_t size, const device &dev, const context &ctxt);
void *malloc_host  (size_t size, const context &ctxt);
void *malloc_shared(size_t size, const device &dev, const context &ctxt);
// single-function form: allocation type is an argument
void *malloc(size_t size, const device &dev, const context &ctxt, usm::alloc kind);
void *malloc(size_t size, const queue &q, usm::alloc kind);
// C++-style: template <typename T>, Count in elements, returns T*
template <typename T> T *malloc_device(size_t Count, const queue &Q);
template <typename T> T *malloc_host(size_t Count, const context &Ctxt);
template <typename T> T *malloc_shared(size_t Count, const device &Dev, const context &Ctxt);
template <typename T> T *malloc(size_t Count, const queue &Q, usm::alloc Kind);
```

**Fact.** Legal alignments are powers of two; `malloc*` memory is default-aligned, though devices commonly align maximally. malloc-style interfaces do **not** invoke constructors.

**Rule.** Allocation needs (1) the type (name or `usm::alloc kind`), (2) a `context` — or a `queue`, which supplies it, (3) for device and some shared allocations, the `device`. All USM allocations, queues and kernels must share the same context.

```cpp
template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {              // allocate(size_t), deallocate(T*, size_t)
  using value_type = T;
  // propagate_on_container_{copy_assignment,move_assignment,swap} = std::true_type
  template <typename U> struct rebind { typedef usm_allocator<U, AllocKind, Alignment> other; };
  usm_allocator() = delete;        // ctor binds context+device, or a queue; optional property_list
};  // equal only if same USM kind, alignment, context, device
usm_allocator<float, usm::alloc::shared> alloc(q);  float *f3 = alloc.allocate(N);  free(f3, q);
```

### USM queries

| Query | Returns |
|---|---|
| `get_pointer_type(ptr, ctxt)` | `usm::alloc`: `host`, `device`, `shared`, or `unknown` |
| `get_pointer_device(ptr, ctxt)` | `device` the pointer was allocated against; for host allocations, the first device in the context (avoids throwing in templated code) |

Device aspects, tested with `device::has(...)`: `aspect::usm_device_allocations`, `aspect::usm_host_allocations`, `aspect::usm_atomic_host_allocations`, `aspect::shared_allocations`, `aspect::atomic_shared_allocations` (shared allocations concurrently accessible and atomically modifiable by host and device), `aspect::usm_system_allocations`. `dev.has(aspect::usm_shared_allocations)` selects `malloc_shared` vs `malloc_device` vs a `buffer` fallback.

### `buffer`

```cpp
template <typename T, int Dimensions, AllocatorT allocator>
class buffer;
```

| Template param | Meaning |
|---|---|
| `T` | element type; must be **device copyable** (extends trivially copyable recursively: `std::array`, `std::pair`, `std::tuple`, `std::span` specializations over device-copyable types) |
| `Dimensions` | 1, 2, or 3 |
| `AllocatorT` | C++ allocator for host allocations; optional, default `buffer_allocator<T>` |

```cpp
buffer<int, 2, buffer_allocator<int>> b1{range<2>{2, 5}};
buffer<int, 2> b2{range{2, 5}};                   // CTAD is all-or-none
buffer b5{myDoubles, range{4}};                   // host pointer, copied at construction
buffer b6{myConstDbls, range{5}};                 // T deduced double, not const double
buffer b7{sharedPtr, range{1}};                   // shared_ptr, ref-counted
buffer b9{myVec};                                 // contiguous container
buffer<int, 2> b10{range{2, 5}};
buffer b11{b10, id{0, 0}, range{1, 5}};           // sub-buffer of b10; b12 uses id{1, 0}
// InputIterator form: buffer b8{myVec.begin(), myVec.end()};
// allocator-taking form: buffer<float, 1, std::allocator<float>> b4{range(20), myFloatAlloc};
```

**Rule.** A sub-buffer requires a parent buffer reference, a base `id`, and a range; it cannot be created from a sub-buffer. Multiple sub-buffers may overlap; non-overlapping ones may run concurrently.

**Fact.** Buffer queries: range, total elements, bytes required, allocator in use, whether sub-buffer. A buffer can be reinterpreted as one with different characteristics.

| Property | Semantics |
|---|---|
| `property::buffer::use_host_ptr{}` | no host allocation, allocator ignored; must use the passed host pointer's memory (device may still cache it). **Only** valid with a host pointer constructor argument |
| `property::buffer::use_mutex{myMutex}` | requires a host pointer; host may read updated values while the buffer is alive; query `get_property<property::buffer::use_mutex>().get_mutex_ptr()`. `host_accessor` preferred |
| `property::buffer::context_bound{q.get_context()}` | locks the buffer to one context; use on another context ⇒ runtime error; debugging aid |

**Write-back rules.**
- Buffer initialized from a host pointer to **non-const** data ⇒ that pointer is updated with the latest data when the buffer is destroyed (as if `set_final_data` was called with it). Passing a host pointer promises the runtime you will not access that host memory during the buffer's lifetime.
- `set_final_data(...)` accepts a raw pointer, a C++ `OutputIterator`, or a `std::weak_ptr`; data written there on destruction (with `std::weak_ptr`, nothing if it expired/deleted). `set_write_back(...)` controls whether writeback occurs.

### `accessor`

Five template parameters: element type, dimensionality, access mode, access target, placeholder.

| Template param | Values | Default |
|---|---|---|
| access mode | `access::mode::read`, `access::mode::write`, `access::mode::read_write` | `read_write` for non-const `T`, `read` for const `T` |
| access target | `access::target::device`, `access::target::host_task` | `device` |
| placeholder | `access::placeholder::false_t` | deduced from the constructor used |

**Fact.** Only `device` and `host_task` targets exist in SYCL 2020 C++; the old local-memory and host targets are gone — the host target is replaced by `host_accessor`, local memory handled differently (see §05).

Deduction tags — constructor parameter; CTAD deduces mode+target:

| Tag | `access::mode` | `access::target` |
|---|---|---|
| `read_only` | `read` | `device` |
| `read_write` | `read_write` | `device` |
| `write_only` | `write` | `device` |
| `read_only_host_task` | `read` | `host_task` |

`no_init` — optional **accessor property**: previous buffer contents may be discarded; removes pre-kernel data movement for a full-overwrite kernel.

```cpp
accessor pc{c_buf};                       // placeholder: declared outside a command group
q.submit([&](handler &h) {
  accessor a{buf_a, h, write_only, no_init};
  accessor b{buf_b, h, read_only};        // tags need no explicit template args (CTAD)
  h.parallel_for(N, [=](id<1> i) { c[i] += a[i] + b[i]; });
});
q.submit([&](handler &h) { h.require(pc);  // bind placeholder to command group
  h.parallel_for(N, [=](id<1> i) { pc[i]++; }); });
host_accessor result{buf_c, read_only};   // no handler argument
auto acc = a_buf.get_access<access::mode::read>(h);   // alternative extraction
```

**Rule.** `host_accessor` construction blocks until data is available on the host (kernels **and** copy complete); it takes no handler. A buffer is locked to the host while a `host_accessor` is valid — keep its scope tight; a `read_only` host accessor means later device kernels need no write-back.

**Accessor operations.** `operator[](id)` or `operator[](size_t)`; multi-dimensional accessors return an object indexed again (`a[i][j]`); `get_pointer()`/`data()`, forward/backward iterators, container-like interface.

### `handler` data-movement API

```cpp
class handler {
  void depends_on(event);  void depends_on(const std::vector<event> &);
  void memcpy(void* Dest, const void* Src, size_t Count);               // bytes
  template <typename T> void copy(const T* Src, T* Dest, size_t Count); // elements
  template <typename T> void fill(void* Ptr, const T& Pattern, size_t Count);
  // kernel submission members (single_task, parallel_for over range/nd_range): see §03
};
```

Accessor `copy` overloads are templated `<T_Src, T_Dst, Dims, access::mode AccessMode, access::target AccessTarget, access::placeholder IsPlaceholder = access::placeholder::false_t>` and pair `accessor` ↔ `shared_ptr_class`, `accessor` ↔ pointer (`T_Dst *` / `const T_Src *`), and `accessor` ↔ `accessor`.

```cpp
template <typename T, int Dims, access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
void update_host(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder> Acc);
```

**Rule.** `update_host` guarantees the memory object accessed by the accessor is updated on the host after this action executes. `memcpy`, `fill`, `prefetch`, `mem_advise` and `memset` exist on **both** `handler` and `queue`; `queue::parallel_for(range, event, kernel)` takes a dependence `event`.

```cpp
// explicit USM movement: Dest first for memcpy
int *device_array = malloc_device<int>(N, q);
q.submit([&](handler &h) { h.memcpy(device_array, &host_array[0], N * sizeof(int)); }).wait();
q.submit([&](handler &h) { h.parallel_for(N, [=](id<1> i) { device_array[i]++; }); }).wait();
q.submit([&](handler &h) { h.memcpy(&host_array[0], device_array, N * sizeof(int)); }).wait();
free(device_array, q);
```

Implicit USM pattern: allocate with `malloc_host` / `malloc_shared`, access the pointers directly in kernel and host code — no copies. Include `#include <sycl/sycl.hpp>`.

### Semantics and rules

- Data dependences: RAW (true/flow), WAR (anti-dependence), WAW (output); RAR adds no restriction. Buffers express most dependences via accessors; USM needs explicit events/`depends_on` or an in-order queue (see §04).
- A kernel that only writes a buffer may still need the buffer's original contents on the device — not all elements are guaranteed written, unless the access discards previous contents (`no_init`).
- `access::mode::read` ⇒ no buffer copy-back after the kernel; `access::mode::write` ⇒ results may need copying back.

**Coverage note.** Source does not define `discard_write`/`discard_read_write`, `get_host_access`, `atomic_accessor`, or `image`/`sampler`/`image_accessor` APIs; work-group local memory is `local_accessor<T, Dims>` (§05).

### Key gotchas
- Never dereference a device allocation on the host: wrong data or crash; use `memcpy`/`copy`.
- Give every USM allocation, queue and kernel the same `context`; USM pointers do not cross contexts.
- Do not touch host memory passed to a `buffer` (`use_host_ptr`/`use_mutex`) while the buffer is alive.
- `handler::memcpy(Dest, Src, bytes)` but `handler::copy(Src, Dest, elements)` — argument order differs.
- `host_accessor` blocks and locks the buffer to the host — construct late, scope tightly, destroy before device reuse.
- `buffer` write-back happens at destruction, not kernel completion; control it with `set_final_data`/`set_write_back`.
- A `const` host pointer still yields a writable `buffer<T>`; const-ness is not deduced — write results via `set_final_data`.
- Default accessor mode is `read_write`; specify `read_only`/`write_only` to avoid false dependences and superfluous movement.
- Use `no_init` only for full-overwrite kernels, never for read–modify–write or partial writes.
