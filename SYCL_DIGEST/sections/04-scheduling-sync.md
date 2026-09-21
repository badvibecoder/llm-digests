## 04 · Scheduling, dependences and host synchronization

### Model

Task graph = nodes + edges. Node = **command group**; edge = **dependence**. Sequencing applies to kernel executions and to data movement.

| Data dependence | Meaning |
|---|---|
| Read-after-Write (RAW) | task reads data produced by another task (data flow) |
| Write-after-Read (WAR) | task updates data after another task read it |
| Write-after-Write (WAW) | two tasks write the same data |

**Rule.** SYCL executes correctly *based on the expressed dependences*; the programmer must ensure the graph expresses all dependences. Graphs may be linear chains or hundreds of nodes.

### Command group anatomy

Three possible contents; **action** is the only required one, most also express dependences, plus arbitrary host C++ code.

| Action type | USM forms | buffer forms |
|---|---|---|
| kernel | `parallel_for`, `single_task` | `parallel_for`, `single_task` |
| explicit data movement | `memcpy`, `memset`, `fill` | `copy`, `fill`, `update_host` |

**Rule.** A command group may perform only a **single** action.

**Rule.** The host portion of a command group runs **immediately on the host, before `submit` returns**, exactly once. The kernel / explicit data operation is enqueued for device execution.

Command groups are expressed as a lambda passed to `queue::submit`, or via shortcut methods on `queue` taking a kernel plus event-based dependences.

### Declaring dependences — 3 mechanisms

| Mechanism | Form |
|---|---|
| in-order queue semantics | implicit dependence between successively enqueued command groups; one task cannot execute until the previously submitted task completes |
| event-based, command group style | `handler::depends_on(event)` or `depends_on(std::vector<event>)` inside a `submit` lambda |
| event-based, shortcut style | `queue::parallel_for` / `queue::single_task` with an extra `event` or `std::vector<event>` parameter |
| accessors (data) | `accessor` creation on the same `buffer` lets the runtime derive dependences |

**Rule.** Out-of-order queue ⇒ no implicit ordering; dependences must be explicit (events or accessors).

**Gotcha.** USM allocations have **no implicit dependences** — the runtime does not track reads/writes of raw pointers. Ordering must come from `depends_on`/event parameters or an in-order queue. Buffers + accessors are the only implicit-dependence path.

Accessors declare intended read/write use; examples given: one kernel reading data another produces, two kernels writing the same data, one kernel modifying data after another read it.

Accessor deduction tag from source: `read_only` tells the runtime the data is only read, not produced.

### Implicit data movement

- USM: host and shared allocations. Host allocations access data remotely (no real movement); shared allocations may migrate host↔device automatically. Nothing needed in command groups.
- `prefetch`: like `memcpy` but starts migrating a shared allocation early; treated as a **hint**, does not invalidate pointer values, not required for correctness. Command groups often do **not** depend on prefetches.
- Buffers: accessor creation effectively creates an **extra, hidden node** in the graph. If data is not on the device, the runtime must move it there before the kernel may execute; the runtime tracks where the current version of a buffer resides. Only after the movement may the submitted kernel execute.
- Host accessor creation schedules movement of buffer data back to the host before the accessor is usable.

### Synchronizing with the host

| Mechanism | Semantics |
|---|---|
| `queue::wait` | blocks host until every command group submitted to the queue completes; coarse-grained |
| `queue::wait_and_throw` | same, and processes asynchronous errors |
| `event::wait` | fine-grained; synchronize on a specific action/command group |
| `event::wait` (static) | accepts a `std::vector` of events |
| `event::wait_and_throw` | blocking + asynchronous error processing for one event |
| `host_accessor` | makes data available on host **and** defines a new dependence between the accessing graph and the host; **blocking** |
| buffer/image destruction | when a buffer is destroyed it waits for all command groups using it to finish; then data is copied back if initialized with a host pointer or if a host pointer was passed to `set_final_data` |
| `property::buffer::use_mutex` | memory shared with host app, governed by the mutex used to initialize the buffer; host takes the mutex lock when safe to access |

**Rule.** `submit` returns an `event` (as does submitting the action/command group for `memcpy`-style operations) that can order that work with other command groups.

**Rule.** While a device uses a buffer, a `host_accessor` keeps its data on the host; a buffer cannot be used on a device while a host accessor exists. Common pattern: create host accessors inside nested C++ scopes so the data is freed when no longer needed.

**Gotcha.** A host accessor gives an up-to-date host view but does **not** guarantee the original host memory (specified at buffer creation) is updated; that memory cannot be safely accessed until the buffer is destroyed, unless the mutex mechanism is used.

### `handler::host_task`

**Fact (DPC++/SYCL 2020).** `handler::host_task` submits arbitrary C++ as an action in the task graph, executed on the **host** once task graph dependences are satisfied. Body need not follow device-code restrictions (e.g. `std::cout`, `printf`, OpenCL interop allowed). Executes **asynchronously** relative to the host program. Combines with `handler::depends_on(e)` or with accessors parameterized by `target::host_task`.

```cpp
// explicit dependence chain: out-of-order queue + events + host_task
queue q;                                     // out-of-order
int *data = malloc_shared<int>(N, q);
auto e = q.parallel_for(N, [=](id<1> i) { data[i] = 1; });
q.submit([&](handler &h) {
  h.depends_on(e);
  h.host_task([=]() { /* arbitrary host C++ */ });
});
q.wait();
```

```cpp
// Y pattern, out-of-order: shortcut form takes a vector of events
auto e1 = q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });
auto e2 = q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });
auto e3 = q.parallel_for(
    range{N}, {e1, e2},
    [=](id<1> i) { data1[i] += data2[i]; });   // {e1,e2} converts to std::vector<event>
q.single_task(e3, [=]() { /* ... */ });
```

```cpp
// implicit dependences: accessors on the same buffer, out-of-order queue
buffer<int> data{range{N}};
q.submit([&](handler &h) {
  accessor a{data, h};
  h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
});
q.submit([&](handler &h) {
  accessor a{data, h};
  h.single_task([=]() { /* reads a */ });
});
host_accessor h_a{data};   // blocks until data is on the host; also a graph/host sync point
```

```cpp
// in-order queue: order implied by submission order
queue q{property::queue::in_order()};
q.parallel_for(N, [=](id<1> i) { data[i] = 1; });
q.single_task([=]() { /* ... */ });
q.wait();
```

### Asynchronous errors

**Fact.** Asynchronous handler = user function registered with contexts and/or queues, passed to the context or queue constructor as a `std::function` (function, lambda, or function object). Must accept a `sycl::exception_list`. Invoked with the list of unprocessed asynchronous exceptions available at invocation time.

```cpp
auto handle_async_error = [](exception_list elist) {
  for (auto &e : elist) {
    try { std::rethrow_exception(e); }
    catch (sycl::exception &e) { std::cout << e.what() << "\n"; }
  }
};
queue my_queue{ gpu_selector_v, handle_async_error };
context my_context{ handle_async_error };
```

**Rule.** Handler invocation points (only these):
1. host calls `queue::throw_asynchronous()` on a specific queue
2. host calls `queue::wait_and_throw()` on a specific queue
3. host calls `event::wait_and_throw()` on a specific event
4. a queue is destroyed
5. a context is destroyed

**Rule.** If no handler is defined for the queue or its parent context, the **default** asynchronous handler runs: reports list contents, then `std::terminate` (abnormal termination).

**Rule.** Prefer registering handlers on queues, not explicit contexts (contexts are normally created implicitly). After reporting, terminate inside the handler unless comprehensive error recovery is in place; silent continuation risks wrong results.

**Gotcha.** Without explicit `wait_and_throw()` / `throw_asynchronous()`, asynchronous errors surface only during teardown; a watchdog/convergence loop may never see them — call them at regular, controlled points.

**Gotcha.** C++ exception handling (`throw`) is disallowed inside **device** code; signal device-side errors by non-exception means (error log buffer, invalid result value). Throwing inside a `host_task` is a valid way to generate an asynchronous error.

### Queue profiling

**Fact.** `device::has(aspect::queue_profiling)` reports support; profiling is activated by `property::queue::enable_profiling` in the queue constructor's optional final property list. Enabling makes the runtime capture profiling info for command groups submitted to that queue, available via `event::get_profiling_info`.

Source names the parameter *family* and demonstrates two members; the full enumerator set is
defined by the SYCL spec's "Profiling information descriptors for the SYCL event class" table,
plus any backend restrictions.

| Info descriptor | Captures |
|---|---|
| `info::event_profiling::command_start` | actual start of execution on the device |
| `info::event_profiling::command_end` | completion on the device |

**Rule.** Each descriptor returns a timestamp = nanoseconds elapsed since an implementation-defined time base. Events sharing a backend share the time base, so timestamp differences are nanoseconds between them.

**Gotcha.** On a device without `aspect::queue_profiling`, passing `property::queue::enable_profiling` throws a **synchronous** exception with `errc::feature_not_supported`. Enabling profiling adds overhead — enable during development/tuning only, disable for production.

```cpp
queue q(property::queue::enable_profiling{});
event e          = q.submit([&](handler &h) { /* kernel */ });
q.wait();
double timeA =
    (e.template get_profiling_info<info::event_profiling::command_end>() -
     e.template get_profiling_info<info::event_profiling::command_start>());
```

Device-side timing excludes host↔device transfer time, so it is more precise than host timing (`std::chrono`) for device work.

### Key gotchas
- USM pointers carry no implicit dependences: order USM kernels explicitly via events, `depends_on`, or an in-order queue.
- A command group performs exactly one action; never submit a kernel plus a memory op in one group.
- Command-group host code runs at `submit` time, not when dependences are met — no device data there.
- Out-of-order queues provide no ordering; only in-order queues imply dependence between successive submissions.
- `host_accessor` blocks and pins data to the host; scope it tightly and destroy it before device reuse.
- Host accessor does not update original host memory provided at buffer creation; `set_final_data` or destroy the buffer.
- Buffer destructor waits for all command groups using it; `use_mutex` buffers keep shared memory governed by the mutex.
- Do not make later command groups depend on `prefetch`; it is a hint and not required for correctness.
- Call `queue::wait_and_throw()` / `throw_asynchronous()` at controlled points, else async errors surface only at teardown.
- Never let an async handler return silently after an unrecoverable error; terminate after reporting.
- `throw` is illegal in device code; use a log buffer or sentinel result instead.
- `enable_profiling` requires `aspect::queue_profiling`, else throws `errc::feature_not_supported`; it adds overhead, so disable in production.
