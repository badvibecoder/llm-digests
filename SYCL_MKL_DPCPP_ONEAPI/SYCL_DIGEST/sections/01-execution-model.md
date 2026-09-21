## 01 · Execution model, queues, devices, contexts

### Model

**Fact.** SYCL is **single-source**: one translation unit holds device code (kernels) and the host code orchestrating execution.

| Region | Runs on | Content |
|---|---|---|
| Host code | CPU running the OS process | standard C++ + SYCL constructs; defines data, dependences, submits work; may compute and use libraries |
| Device code | accelerator (may be the host CPU exposed as a device) | restricted subset; logically independent of the host processor |

**Rule.** Host and devices are logically independent; host runs native C++, devices run device code.

**Fact.** Implementations build on lower-level APIs (OpenCL, Level Zero, CUDA, others); the target API is a **SYCL backend**. Each object is associated with a backend, queried via `get_backend()` (platform/device/context/queue). Interop: see §20.

**Fact.** Device code executes **asynchronously** (host submits and continues; work starts only when dependences are satisfied); is **restricted** (no dynamic memory allocation, no RTTI); some functions/queries exist **only in device code** (e.g. work-item identifier queries).

**Fact.** Work submitted to queues = **actions**: device-code execution and memory movement commands.

**Rule.** Device code is *permitted* to start once node dependences are met, never *guaranteed*. Only host blocking operations (host accessors, queue wait) force start; otherwise the runtime may optimize for power or congestion.

### Type map

| Concept | SYCL type / API | CUDA analogue (source-given) |
|---|---|---|
| kernel | `Kernel` | Kernel |
| N-dimensional parallel index space | `range` (generally), `nd_range` (with grouping) | Grid (always has grouping) |
| kernel instance at a point in the index space | Work-Item | Thread |
| application-defined group that can communicate/synchronize | Work-Group | Block |
| implementation-defined group with communication/sync | Sub-Group | Warp |
| memory exchanged among instances in a group | Local Memory | Shared Memory |
| group synchronization | `group_barrier()` | `__syncthreads()`, `__syncwarp()`, `coop_group.sync()` |
| work submission target bound to one device | Queue | Stream |
| execution target | `device` | — |
| grouping of devices/backend | `platform` | — |
| shared object namespace for devices | `context` | — |

### Device selection

**Fact.** A `DeviceSelector` is a callable taking a `const` reference to a `device` and returning an integer score used to rank devices.

**Rule.** Runtime scores the selector on all available root devices and picks the highest. Score must be nonnegative for selection; negative ⇒ device guaranteed not selected; ties ⇒ runtime picks one of the tied devices.

**Fact.** Selectors and selection constructs:

| Selector | Selects | Form |
|---|---|---|
| `default_selector_v` | any device of the implementation's choosing (implicit for `queue q;`) | `queue q;` / `queue q{default_selector_v};` |
| `cpu_selector_v` | device identifying as CPU | `queue q{cpu_selector_v};` |
| `gpu_selector_v` | device identifying as GPU | `queue q{gpu_selector_v};` |
| `accelerator_selector_v` | device identifying as "accelerator" (includes FPGAs) | `queue q{accelerator_selector_v};` |
| `ext::intel::fpga_selector_v` | FPGA (`#include <sycl/ext/intel/fpga_extensions.hpp>`); **DPC++ only, not available in SYCL** | `queue q{ext::intel::fpga_selector_v};` |
| custom callable | fixed positive score per class; name/vendor string match on `info::device::name` / `info::device::vendor`; any device or platform query | `queue q{my_selector};`; score `-1` vetoes |
| `aspect_selector` | aspects exhibited / forbidden (forms below) | `queue q{aspect_selector(...)};` |

**Fact.** `aspect_selector` forms:

- `aspect_selector(aspect::fp16, aspect::gpu)` (comma-delimited) — all listed aspects must be exhibited.
- `aspect_selector(std::vector{aspect::fp64, aspect::fp16}, std::vector{aspect::gpu, aspect::accelerator})` — first vector must be present; second must **not** be present.

```cpp
int my_selector(const device &dev) {   // -1 vetoes; all scores -1 ⇒ exception
  if (dev.get_info<info::device::name>().find("pac_a10") != std::string::npos &&
      dev.get_info<info::device::vendor>().find("Intel") != std::string::npos)
    return 1;
  return -1;
}
```

### Queue

```cpp
class queue {
public:
  queue(const property_list & = {});
  queue(const async_handler &, const property_list & = {});
  template <typename DeviceSelector>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {});
  queue(const device &, const property_list & = {});
  queue(const context &, const device &, const property_list & = {});

  template <typename T> event submit(T);
  void wait();
  void wait_and_throw();
};
```

**Rule.** A queue is bound to exactly **one** device, fixed at construction; work executes on that device. It cannot map to a collection of devices, spread work across devices, or bind to more than one device. Multiple queues may exist (different devices, different host threads, or several on one device, where work is combined); no cross-device load balancing exists.

**Fact.** Default construction (`queue q;`) uses `default_selector_v`; at least one device is always available, so one is always selected; it may be the host CPU (not guaranteed).

**Fact.** `property::queue::in_order()` (source form) executes tasks in submission order. Queues are **out-of-order by default**; set the property at queue creation. In-order is deterministic but serializes tasks with no dependences between them. Command-group/event dependences: see §4.

**Fact.** `property::queue::enable_profiling`: requires `device::has(aspect::queue_profiling)`. Profiling descriptors: see §12 (also §4).

**Fact.** Queue-level shortcuts for the equivalent `handler` calls, returning an `event`:

```cpp
event memset(void* ptr, int value, size_t count);
event memcpy(void* dest, const void* src, size_t count);
template <typename KernelName, typename KernelType>
event single_task(KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
event parallel_for(range<Dims> num_work_items, KernelType kernel);
template <typename KernelName, typename KernelType, int Dims>
event parallel_for(nd_range<Dims> execution_range, KernelType kernel);
// event-list overloads: same three kernel forms plus
// const std::vector<event>& events, waiting before executing the kernel
```

### Command groups and the task graph

**Fact.** Task graph = graph of nodes; each node holds one action for a device, most commonly a data-parallel kernel. Edges = dependences defining when a node's work may begin, usually from data dependences; explicit custom dependences also available.

**Fact.** `q.submit(cg)` takes one command group (lambda or function object) per call. Inside: (1) host code setting up dependences — runs immediately on the host before `submit` returns (e.g. accessor creation); (2) at most one action — `single_task`, `parallel_for` (device code) or `copy`, `fill`, `update_host`, `memset`, `memcpy` (explicit memory operation).

**Rule.** At most one action per command group — more is an error; all other code runs in the host program immediately. Depth: see §4.

**Fact.** `handler::host_task` submits arbitrary C++ as a task-graph action run on the host once dependences are satisfied; no device-code restrictions (may use `std::cout`, `printf`, OpenCL interop). Runs asynchronously, integrated with dependence tracking. Order with `h.depends_on(e)` or accessors parameterized by `target::host_task`.

### Device, platform, context queries

| Entry point | Purpose |
|---|---|
| `dev.get_info<info::device::name>()`, `info::device::vendor` | device name / vendor |
| `dev.get_info<info::device::max_work_group_size>()`, `global_mem_size`, `local_mem_size` | numeric device limits |
| `dev.has(aspect::fp16)` | aspect test |
| `dev.is_cpu()`, `dev.is_gpu()` | equivalent to `has(aspect::cpu)`, `has(aspect::gpu)` |
| `device::get_devices(info::device_type t = info::device_type::all)` | devices of a type |
| `platform::get_platforms()` | platforms |
| `p.get_info<info::platform::name>()` | platform name |
| `p.get_devices()` | devices in a platform |
| `p.get_backend()`, `d.get_backend()`, `c.get_backend()`, `q.get_backend()` | associated backend |
| `q.get_device()`, `q.get_context()` | queue's device/context |

```cpp
for (auto& p : platform::get_platforms()) for (auto& d : p.get_devices()) { /* p.get_backend() */ }
```

**Fact.** `context` may be created implicitly or explicitly, from a native backend context via `make_context<backend::...>`, or with an async handler (below). Most applications need no explicit context management (§20 for `make_context` / `get_native`).

### Async errors and queues

**Fact.** Synchronous SYCL errors are `sycl::exception`, derived from `std::exception`; caught with standard try/catch, message via `e.what()`.

**Fact.** The asynchronous handler is a `std::function` registered with a **queue or context** at construction time:

```cpp
queue my_queue{ gpu_selector_v, handle_async_error };
context my_context{ handle_async_error };
```

**Rule.** It must accept a `sycl::exception_list` and is passed the available unprocessed asynchronous exceptions. Define it on queues unless contexts are already explicitly managed.

**Fact.** Invocation times (exact set): `queue::throw_asynchronous()`; `queue::wait_and_throw()`; `event::wait_and_throw()`; destruction of a queue; destruction of a context.

**Rule.** With no handler for the queue or its parent context, the **default asynchronous handler** prints the exceptions and calls `std::terminate()` (abnormal termination).

**Gotcha.** A handler that prints and returns lets execution continue with no error-recovery strategy; terminate inside it unless comprehensive error management exists.

**Fact.** Error taxonomy and `errc` codes: see §12.

### Code: queue creation variants and device selection

```cpp
queue q0;                                    // default -> default_selector_v
queue q1{gpu_selector_v};                    // device-class selector
queue q2{property::queue::in_order{}};       // property-list form
queue q3{my_selector};                       // custom callable selector
```

**Gotcha.** If no device of the required class is available, the selector throws (source: `runtime_error`; caller code catches `sycl::exception`). Catch it to fall back to a less desirable device class.

### Key gotchas

- Queue→device binding is fixed at construction; a queue cannot be rebound, shared across devices, or load-balanced.
- `queue q;` chooses arbitrarily and ignores the kernel; the host CPU is not guaranteed.
- A selector score must be nonnegative to be selected; return `-1` to veto; all devices vetoed ⇒ exception.
- Tied top scores are resolved arbitrarily by the runtime — do not rely on which device wins.
- Out-of-order is the default queue type; only `property::queue::in_order()` gives submission-order execution.
- In-order queues serialize independent tasks; use explicit dependences on out-of-order queues to overlap.
- `property::queue::enable_profiling` needs `device::has(aspect::queue_profiling)`, else a synchronous `errc::feature_not_supported` exception.
- Device code is only *allowed* to start when dependences are met; never guaranteed without a blocking host operation.
- Never rely on async errors appearing promptly: they surface only at `wait_and_throw`, `throw_asynchronous`, or queue/context destruction.
- A device may enumerate once per backend that supports it; selecting a device does not select a backend.
- Exceeding one action per command group is an error, not a supported pattern.
