## 15 · Backend interoperability and SYCL extensions

**Model.** A *SYCL backend* is the lower-level API a SYCL implementation builds on to reach hardware: OpenCL, Level Zero, CUDA, or others. Implementations may drive multiple backends simultaneously. Every SYCL object carries an associated backend; query with `get_backend()` on a platform, device, context, or queue. A device supported by several backends may enumerate as a separate SYCL device for each.

```cpp
// Backend associated with each platform
for (auto& p : platform::get_platforms())
  std::cout << p.get_info<info::platform::name>() << " " << p.get_backend();
```

| `backend` enumerator | Platform name in the source's example output |
|---|---|
| `backend::opencl` | Portable Computing Language; Intel(R) OpenCL HD Graphics; Intel(R) FPGA Emulation Platform for OpenCL(TM) |
| `backend::ext_oneapi_level_zero` | Intel(R) Level-Zero |
| `backend::ext_oneapi_cuda` | NVIDIA CUDA BACKEND |
| `backend::ext_oneapi_hip` | AMD HIP BACKEND |

**Rule.** SYCL 2020 defines interoperability for OpenCL backends only; other backends get it through extensions (`sycl_ext_oneapi_backend_level_zero`, see below).
**Why.** Interop lets you incrementally add SYCL to an existing low-level-API codebase, or extract native objects to call existing low-level libraries. Cost: extra code paths per backend, reduced portability (execution restricted to one backend's devices).

### API surface: directions
| Direction | API |
|---|---|
| native object → SYCL object | `make_context`, `make_queue`, `make_event`, `make_kernel`, `make_device`, `make_platform`, `make_buffer`, `make_subbuffer`, `make_kernel_bundle` |
| SYCL object → native object (free function) | `get_native<backend>(syclObj)` |
| SYCL object → native object inside runtime-scheduled host code | `interop_handle` member functions, passed to `host_task` |

Generic shape: `make_<syclObj><backend::B>( nativeObj [, context] [, extraArgs] )`. Parameter sets differ per backend; `make_device`/`make_context`/`make_buffer` are the same function names across OpenCL and Level Zero.

```cpp
// SYCL objects built from native OpenCL objects
context c = make_context<backend::opencl>(openclContext);
device  d = make_device<backend::opencl>(openclDevice);
buffer  data_buf = make_buffer<backend::opencl, int>(openclBuffer, c);
queue q{c, d};
q.submit([&](handler& h) {
  accessor data_acc{data_buf, h};
  h.parallel_for(size, [=](id<1> i) { data_acc[i] = data_acc[i] + 1; });
}).wait();
```

```cpp
// Level Zero: make_* take an ownership field; pointer + ownership for buffers
device d = make_device<backend::ext_oneapi_level_zero>(level0Device);
context c = make_context<backend::ext_oneapi_level_zero>(
    {level0Context, {d}, ext::oneapi::level_zero::ownership::keep});
buffer data_buf = make_buffer<backend::ext_oneapi_level_zero, int>(
    {level0Ptr, ext::oneapi::level_zero::ownership::keep}, c);
```

**Rule (ownership).** OpenCL: the SYCL implementation uses OpenCL reference counting to manage native object lifetimes. Level Zero: ownership is explicit — `ext::oneapi::level_zero::ownership::keep` means the application retains ownership and must free the native object; if the SYCL implementation takes ownership, the native object is destroyed with the SYCL object.

### Extracting native objects
| Call | Native type |
|---|---|
| `get_native<backend::opencl>(d)` | `cl_device_id` |
| `get_native<backend::opencl>(c)` | `cl_context` |
| `get_native<backend::ext_oneapi_level_zero>(d)` | `ze_device_handle_t` |
| `get_native<backend::ext_oneapi_level_zero>(c)` | `ze_context_handle_t` |

**Rule.** Extracted objects can do anything the native API allows (queries, allocation, command queues, compiling programs, executing kernels), then are released with that API: `clReleaseDevice`, `clReleaseContext`, `clReleaseMemObject`; `zeMemFree(level0Context, level0Ptr)`.
**Fact.** Examples compile with `#include <sycl/sycl.hpp>`; `cl_*` and `ze_*` types are used directly.

### `interop_handle` in `host_task`
Use when only one specific operation in the SYCL task graph needs a backend API. The `interop_handle` represents the SYCL runtime state at host-task invocation and exposes native objects for the queue, device, context, and the buffers captured by the host task.

| Member | Returns |
|---|---|
| `get_native_device<backend>()` | native device handle |
| `get_native_context<backend>()` | native context handle |
| `get_native_queue<backend>()` | native queue handle |
| `get_native_mem<backend>(accessor)` | native memory object(s) |

```cpp
q.submit([&](handler& h) {
  accessor a{b, h};
  h.host_task([=](interop_handle ih) {
    auto openclDevice = ih.get_native_device<backend::opencl>();
    size_t sz = 0;
    clGetDeviceInfo(openclDevice, CL_DEVICE_NAME, 0, nullptr, &sz);
    std::string openclDeviceName(sz, ' ');
    clGetDeviceInfo(openclDevice, CL_DEVICE_NAME, sz, &openclDeviceName[0], nullptr);
    auto openclMem = ih.get_native_mem<backend::opencl>(a)[0]; // vector, take [0]
    clGetMemObjectInfo(openclMem, CL_MEM_SIZE, sizeof(sz), &sz, nullptr);
  });
});
```

```cpp
// Level Zero inside a host task
auto level0Context = ih.get_native_context<backend::ext_oneapi_level_zero>();
auto ptr = ih.get_native_mem<backend::ext_oneapi_level_zero>(a);
size_t sz = 0;
zeMemGetAddressRange(level0Context, ptr, nullptr, &sz);
```

**Rule.** Operations performed from a host task are scheduled with the other operations in the SYCL queue.
**Rule.** `interop_handle::get_native_mem` returns a *vector* of memory objects (SYCL 2020); for one buffer, use the first element. Return types of `interop_handle` members must match the corresponding `get_native` free functions.

### Kernel interoperability
Two mechanisms, both reworked in SYCL 2020 to route through backend interop.

**(a) API-defined kernel objects.** Create the kernel with the native API, import with `make_kernel<backend>(nativeKernel, context)`.

```cpp
auto openclContext = get_native<backend::opencl>(c);
cl_program p = clCreateProgramWithSource(openclContext, 1, &kernelSource, nullptr, nullptr);
clBuildProgram(p, 0, nullptr, nullptr, nullptr, nullptr);
cl_kernel k = clCreateKernel(p, "add", nullptr);
auto sk = make_kernel<backend::opencl>(k, c);          // SYCL kernel from OpenCL kernel
q.submit([&](handler& h) {
  accessor data_acc{data_buf, h};
  h.set_args(data_acc);                                // explicit args required
  h.parallel_for(size, sk);
});
```

**(b) Non-SYCL source languages / IR.** Kernel contents come from source or an IR not defined by SYCL (reuse kernel libraries, DSLs emitting IR).

```cpp
// Online compiler (experimental): OpenCL C -> SPIR-V; Level Zero module -> SYCL kernel
online_compiler<source_language::opencl_c> compiler(d);
std::vector<byte> spirv = compiler.compile(kernelSource);
ze_module_handle_t level0Module = nullptr;
auto skb = make_kernel_bundle<backend::ext_oneapi_level_zero,
                              bundle_state::executable>({level0Module}, c);
auto sk  = make_kernel<backend::ext_oneapi_level_zero>({skb, level0Kernel}, c);
// then: h.set_args(data_acc); h.parallel_for(size, sk);
```

**Rule.** When a kernel is created via the low-level API (or non-SYCL source), the SYCL compiler has no visibility into it: every kernel argument must be passed explicitly with `set_arg()` / `set_args()`.
**Rule.** SYCL and the low-level API must agree on a convention for passing objects as kernel arguments; the convention belongs to the backend interop specification (example: accessor `data_acc` passed as global pointer arg `data`). SYCL 2020 leaves the precise semantics of `set_arg()` / `set_args()` to each backend specification — more backend-specific code.
**Gotcha.** The old `build_with_source` function was removed in SYCL 2020. If a backend cannot consume a source language directly, compile it with the online compiler extension to a format the backend supports (OpenCL C → SPIR-V for Level Zero); a kernel is usable by any backend the online compiler can target.

### Choosing a device for a backend
**Rule.** Score devices by their associated backend inside custom device-selection logic, or by a lambda equivalent to `default_selector_v`: iterate `device::get_devices(t)` for `info::device_type t = info::device_type::all`, return the first `d` with `d.get_backend() == b`, else `throw sycl::exception(errc::runtime, ...)`. This uses only standard SYCL queries, so it is portable. DPC++ runtime alternative for prototyping only: `ONEAPI_DEVICE_SELECTOR` limits enumerated devices by device type or backend — external configuration, not a production solution.

### Contexts and scope
**Rule.** Kernels and USM allocations are generally inaccessible from a SYCL context other than the one they were created in. A native-backend context created directly with the backend API generally will not have access to objects created in a different SYCL context (and vice versa), even when both are associated with the same backend.
**Rule.** To share objects safely, either create the SYCL context from the native context with `make_context`, or obtain the native context from the SYCL context with `get_native`.
**Fact.** Several SYCL implementations with CUDA and HIP backends already have some interoperability support; check the implementation's documentation for which backends and whether interop is supported.

### Interop-relevant extensions
| Extension | Status | Purpose |
|---|---|---|
| `sycl_ext_oneapi_backend_level_zero` | DPC++ extension | Level Zero backend interop: `backend::ext_oneapi_level_zero` with `get_native` free functions, `make_*` creation, and `ext::oneapi::level_zero::ownership` |
| `sycl_ext_intel_online_compiler` | experimental (subject to change or removal) | Online compiler from a source language a backend cannot consume directly to a backend-supported format, e.g. OpenCL C → SPIR-V |

**Fact.** oneAPI guide: extensions are the rapid-experimentation vehicle in open standards bodies (Khronos Group); the DPC++-supported list lives on the "SYCL Extensions" page. Full index: see §20.
**Fact (Intel-specific SYCL knobs named in the oneAPI guide).**

| Feature | Kind | Purpose |
|---|---|---|
| `[[cl::reqd_work_group_size(X, Y, Z)]]` | attribute | Set work-group size (X, Y, Z are ND-range dimensions) so the compiler optimizes more aggressively; prefer specifying a work-group size |
| `[[intel::kernel_args_restrict]]` | kernel attribute | Compiler ignores dependencies between accessor arguments, enabling more aggressive optimization |
| `-Xsfpc` | compiler option | Remove intermediary floating-point rounding operations/conversions where possible; carry extra bits to maintain precision |
| `-Xsno-accessor-aliasing` | compiler option | Ignore dependencies between accessor arguments in a SYCL kernel |

### Key gotchas
- Build the SYCL context with `make_context` from the native context, or `get_native` it — unrelated contexts silently do not share objects.
- Pass every argument explicitly via `set_arg()` / `set_args()` for API-defined or non-SYCL kernels; the compiler cannot see them.
- `interop_handle::get_native_mem` returns a vector — take `[0]` for one buffer; match free-function return types.
- Level Zero: pass `ext::oneapi::level_zero::ownership::keep` or the runtime destroys your native object.
- Free native objects per their own API (`clRelease*`, `zeMemFree`); SYCL will not free `keep`-owned handles.
- Interop is unportable by construction: expect one code path per backend and restriction to its devices.
- `build_with_source` was removed in SYCL 2020; use source/IR import, `make_kernel_bundle`, or the online compiler.
- `sycl_ext_intel_online_compiler` is experimental — subject to change or removal.
- `ONEAPI_DEVICE_SELECTOR` is external configuration; do not rely on it in production code.
- One device may enumerate per backend; backend selection can return duplicates or throw `errc::runtime` when none match.
