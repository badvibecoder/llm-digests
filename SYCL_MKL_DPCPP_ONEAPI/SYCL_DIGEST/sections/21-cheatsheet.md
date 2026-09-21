## 21 · One-page cheat sheet

### Decision table

| If you need… | Use |
|---|---|
| Port pointer C/C++ | USM: `malloc_device`, `malloc_host`, `malloc_shared` |
| Runtime-derived dependences, any device | `buffer` + `accessor` (creation is the dependence) |
| One operation, no index space | `single_task` |
| Data-parallel, no groups/SLM/barriers | `parallel_for(range)`; kernel takes `id` or `item` |
| Work-groups, sub-groups, SLM, barriers | `parallel_for(nd_range)`; kernel takes `nd_item` |
| Deterministic order | `property::queue::in_order()` |
| Overlap independent submissions | out-of-order queue (default) + `handler::depends_on(event)`, event parameter, or accessors |
| Block host; async errors | `queue::wait()` / `queue::wait_and_throw()` |
| Block host on one command group | `event::wait()` |
| Read buffer on host | `host_accessor` |
| Work-group scratch | `local_accessor<T, D>(range, h)`; never from a `buffer`; always `read_write` |
| Intra-group communication | `local_accessor` + `group_barrier`; sub-group: `select_from_group`, `shift_group_left`/`shift_group_right`, `permute_group_by_xor`, `group_broadcast` |
| Group reduction / scan | `reduce_over_group`, `joint_reduce`, `exclusive_scan_over_group`, or `reduction(...)` as a `parallel_for` argument |
| One work-item writes group result | `group::leader()` |
| Relaxed counter/histogram | `atomic_ref<T, memory_order::relaxed, memory_scope::work_group, access::address_space::local_space>` |
| Work-item communication; lock | `memory_order::acq_rel` (or `acquire`/`release`) on `atomic_ref` |
| Any device atomic | `atomic_ref`; never `std::atomic`; `cl::sycl::atomic` deprecated in SYCL 2020 |
| Device by aspect / specific | `gpu_selector_v` / `aspect_selector(aspect::fp16, aspect::gpu)` / custom `DeviceSelector` scoring >= 0; `-1` vetoes |
| Host CPU for debug/fallback | `cpu_selector_v` |

### API skeleton

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr size_t N = 1024, B = 64;                      // B divides N
queue q{gpu_selector_v, property::queue::in_order()};   // + async_handler 2nd arg
int* u = malloc_shared<int>(N, q);                      // USM

buffer<int, 1> a{range<1>{N}}, c{range<1>{N}};          // nd_range path
q.submit([&](handler& h) {
  accessor A{a, h, read_write};
  accessor C{c, h, write_only, no_init};
  auto tile = local_accessor<int, 1>(B, h);
  h.parallel_for<class Tiled>(nd_range<1>{N, B}, [=](nd_item<1> it) {
    tile[it.get_local_id(0)] = A[it.get_global_id(0)];
    group_barrier(it.get_group());                      // all work-items or none
    C[it.get_global_id(0)] = tile[it.get_local_id(0)] * 2;
  });
});
host_accessor H{c, read_only};                          // blocks

// USM: no implicit dependences - wait REQUIRED before host reads
q.parallel_for(range<1>{N}, [=](id<1> i) { u[i] *= 2; }).wait();

// reduction: USM-pointer form; reducer is an extra kernel argument
int* total = malloc_shared<int>(1, q); *total = 0;
q.parallel_for(range<1>{N}, reduction(total, plus<>()),
               [=](id<1> i, auto& red) { red += u[i]; }).wait();
assert(*total >= 0);
free(u, q); free(total, q);
```

### Build commands

```bash
source /opt/intel/oneapi/setvars.sh   # or <install>/<toolkit-version>/oneapi-vars.sh
sycl-ls                               # platforms/devices

icpx -fsycl -O2 prog.cpp -o prog                                          # JIT (SPIR-V, default)
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc" prog.cpp -o prog # Intel GPU AOT
icpx -fsycl -fsycl-targets=spir64_x86_64 -Xs "-device avx2" prog.cpp -o prog  # CPU AOT
ocloc compile --help                                                      # -device GPU names
dpcpp -g -O0 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \       # AOT debug (kbl)
  -Xs "-device kbl -internal_options -cl-kernel-debug-enable -options -cl-opt-disable" prog.cpp
```

### Intel GPU rules of thumb

- Mapping: work-item -> SIMD lane; sub-group -> SIMD width/EU thread; work-group -> compute unit; ND-range -> GPU.
- Sub-group size fixed per device+kernel+ND-range; 16/32+ common. Query `info::device::sub_group_sizes`, `max_sub_group_size`, `compile_sub_group_size`; never assume.
- Work-group size: multiple of `preferred_work_group_size_multiple`, else SIMD channels disabled.
- Never submit `nd_range<1>{M, 1}`: one-work-item group masks all but one SIMD lane.
- Portable mapping: 1-D work-groups, or highest dim divisible by the sub-group size.
- `reqd_sub_group_size(dim)` only for supported sizes, else the build fails.
- Converged control flow: divergence masks both paths; parallelize the more converged dimension.
- Register pressure: prefer 32-bit arithmetic/indexing; 64-bit ops (incl. `size_t`) cost more and spill.
- Occupancy = resident streams vs. theoretical total; 2-D ranges raise it.
- Never branch on `max_compute_units`; express parallelism for the runtime to map.
- SLM beats global memory even on a cache hit; uninitialized on entry, dead on exit.
- Bank conflicts: stride = bank count serializes; pad local structures.
- Count cache lines: adjacent elements peak; every-other or misaligned ~half; one line per work-item worst.
- `prefetch`/`mem_advise` are hints; a prefetch `event` in `parallel_for` overlaps transfer and compute.
- `CL_OUT_OF_RESOURCES` (-5) ~ SLM over the limit; check `info::device::local_mem_size`.
- AOT = device binary, fixed target; JIT = SPIR-V (`-fsycl-targets=spir64`), portable; FPGA is AOT only.

### Top 15 mistakes

- Wait (`.wait()`/`host_accessor`) before host reads.
- Never dereference `malloc_device` on the host; use `memcpy`/`copy`.
- Order USM work (`handler::depends_on`, event parameter, `property::queue::in_order()`); no implicit dependences.
- All USM allocations, queues, kernels in one `context`; pointers do not cross.
- One action per command group; host code runs at `submit` time.
- Scope `host_accessor` tightly: it blocks and pins the buffer.
- Argument order: `handler::memcpy(Dest, Src, bytes)` vs `handler::copy(Src, Dest, elements)`.
- Each work-group dimension divides the ND-range exactly, else the launch is illegal.
- Never branch around `group_barrier`: all work-items reach it.
- Synchronize only within your own work-group; cross-work-group waits can deadlock.
- `local_accessor` from type + range, never a `buffer`; dead on exit.
- Do not assume sub-group size or work-item->sub-group mapping; query descriptors.
- `atomic_ref` only; never mix atomic and non-atomic accesses; specify `access::address_space`, scope.
- `memory_order`: no `release` on loads, no `acquire` on stores, `relaxed` ignores scope, `consume` absent.
- Capture lambdas by value; never the `handler`; no `throw`, `mutable`, non-`void` return.

### Environment variables worth knowing

| Variable | Values | Purpose |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `backend:device_type:device_num` (e.g. `level_zero:gpu`, `opencl:gpu:0`, `cpu`) | select devices/backends |
| `SYCL_UR_TRACE`, `SYCL_PI_TRACE` | `1` \| `2` \| `-1` | runtime trace: `1` basic, `2` all, `-1` +debug |
| `ZE_DEBUG` | any value | log Level Zero API calls/events |
| `SYCL_PRINT_EXECUTION_GRAPH` | `0` \| `1` | dump graph as DOT files (Linux) |
