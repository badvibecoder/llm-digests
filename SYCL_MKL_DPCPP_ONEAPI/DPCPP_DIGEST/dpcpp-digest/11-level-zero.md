---
chunk: 11-level-zero
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 744-762
covers: Host-side/Clang sanitizers (p.744); Intel® oneAPI Level Zero API and switch; Level Zero backend specification (feature-test macro, backend selection, SYCL↔Level Zero interop, handle ownership, buffer/image synchronization, device info); multi-card/multi-tile programming (USM, buffers, queues)
---

# Intel® oneAPI Level Zero

> **Scope.** Level Zero switch (packages, loader, driver, backend selection); Level Zero SYCL backend specification (`SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO`, `ONEAPI_DEVICE_SELECTOR`, `get_native`/`make_*` interop, handle ownership, buffer/image synchronization, device info); multi-card/multi-tile programming (discovery, USM, buffers, queues). Page 744 also carries host-side sanitizer variables and Clang sanitizer OS support.

## Key facts

- Level Zero = **direct-to-metal interfaces to offload accelerator devices**; adaptable to function pointers, virtual functions, unified memory, I/O. Influenced by OpenCL™ API, Vulkan* and GPU architecture, but designed to evolve independently and be supportable across different compute device architectures.
- Most applications should **not** need its extra control; it targets explicit controls needed by higher-level runtime APIs/libraries.
- Packages: `intel-level-zero-gpu` and `level-zero`.
- Loader **discovers all Level Zero drivers** and is also the SDK (headers + libraries for building Level Zero programs).
- GPU driver: open-source, regularly released, **not shipped with DPC++**, installed independently; Level Zero driver + OpenCL™ driver ship in the **same package**.
- Unified Runtime (UR) adapters translate the UR API to a low-level runtime; the Level Zero UR adapter serves Level Zero devices.
- Default: if Level Zero reports support for the installed GPU, SYCL uses it; devices Level Zero does not support (**CPU**) stay on OpenCL™.
- `SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO` values `1`–`5`; extension follows **SYCL 2020** backend spec; older Level Zero interop APIs **deprecated, removed next release**.
- Level Zero backend targets **all Intel GPUs starting with Gen9**.
- Level Zero API is **not thread-safe**; default ownership `transfer` (SYCL runtime owns) unless `ownership::keep`.

## Host-Side Compiler Sanitizers (p.744)

Host-side sanitizers do not support `RTLD_DEEPBIND` when a sanitized application loads other shared libraries; these clear the flag:

| Environment Variable | Description |
|---|---|
| `export ASAN_OPTIONS=clear_deepbind_flag=1` | AddressSanitizer (Asan) |
| `export MSAN_OPTIONS=clear_deepbind_flag=1` | MemorySanitizer (MSan) |
| `export TSAN_OPTIONS=clear_deepbind_flag=1` | ThreadSanitizer (Tsan) |

- NOTE: if the application strongly depends on `RTLD_DEEPBIND`, issues may arise when enabling this flag for host-side sanitizers.
- Host-side MSan can give **false negatives**: the instrumented app relies on instrumented versions of all libraries (GPU drivers such as Intel® Graphics Compiler libraries, SYCL* runtime libraries, others) to propagate correct memory information; many are C++, so MSan-instrumented libc++ would be needed too — **not provided**.

### Clang Sanitizers

Provided with the compiler, **host code only**: AddressSanitizer (Linux, Windows); LeakSanitizer (Linux); MemorySanitizer (Linux); ThreadSanitizer (Linux); UndefinedBehaviorSanitizer (Linux, Windows).

## Intel® oneAPI Level Zero Switch

DPC++ is one oneAPI component; Level Zero provides low-level direct-to-metal interfaces tailored to oneAPI devices. More info: the oneAPI Specification.

### DPC++ Backends

- **SYCL Device Selection:** UR discovers all devices through all UR adapters; one physical device can appear as several SYCL devices if several adapters support it (e.g. OpenCL™ Gen90 and Level Zero Gen90). Runtime selects via built-in (e.g. `gpu_selector`) or user-defined device selectors.
- **Discovery of Multiple UR Adapters:** the same GPU card can be seen as multiple GPU devices under different UR adapters. **NOTE:** OpenCL™ and/or Level Zero runtimes must be installed correctly and independently for UR to see their devices; the SYCL specification does not define which device is used when several match criteria (e.g. `is_gpu()`).
- **Default Preference is Given to a Level Zero GPU:** by default, if Level Zero reports support for the installed GPU, SYCL uses it — for built-in and custom device selectors unless something changes the default. CPU devices keep running on OpenCL™.
- **How to See Where the Code is Running:** `SYCL_PI_TRACE=1` reports the built-in selectors' choice; `SYCL_PI_TRACE=-1` enables verbose UR tracing of all devices found by UR discovery.
- **How to Find all DPC++ Backends and Supported Devices Discovered in the System:** `sycl-ls` queries all platforms/devices of the backends and prints device info + ID numbers; each line prints the `ONEAPI_DEVICE_SELECTOR` pieces **backend name**, **device_type**, **device_id**. Verbose: `$ sycl-ls --verbose` (same choices as built-in and custom device selectors).

## ONEAPI_DEVICE_SELECTOR

With nothing else set, all present platforms/devices are available; default choice usually a Level Zero GPU if available. This variable limits the choice and exposes GPU sub-devices or sub-sub-devices as individual devices.

**Syntax (BNF):**

```text
ONEAPI_DEVICE_SELECTOR = <selector-string>
<selector-string> ::= { <accept-filters> | <discard-filters> | <accept-filters>;<discard-filters> }
<accept-filters> ::= <accept-filter>[;<accept-filter>...]
<discard-filters> ::= <discard-filter>[;<discard-filter>...]
<accept-filter> ::= <term>
<discard-filter> ::= !<term>
<term> ::= <backend>:<devices>
<backend> ::= { * | level_zero | opencl | cuda | hip | esimd_emulator } // case insensitive
<devices> ::= <device>[,<device>...]
<device> ::= { * | cpu | gpu | <num> | <num>.<num> | <num>.* | *.* | <num>.<num>.<num> | <num>.<num>.* | <num>.*.* | *.*.* } // case insensitive
```

**Semantics**

1. A term selects devices from one backend; `cpu`/`gpu` select all devices of that type there, or use **zero-based** index, or `*` (all backend devices).
2. Dot syntax (e.g. `<num>.<num>`) exposes GPU sub-devices as SYCL root devices: `1.0` = first sub-device of the second device; `<num>.*` = all sub-devices of that device; `*.*` = all sub-devices of all GPU devices.
3. `*` matches all backends/devices/sub-devices of the pattern, but a **warning is generated if it matches nothing** (e.g. `*:gpu` with no GPU anywhere; `level_zero:*.*` with no partitionable Level Zero GPU).
4. Indices are zero-based, **unique only within a backend** (`level_zero:0` ≠ `cuda:0`); check with `sycl-ls`. Backends can expose the same hardware (both `level_zero` and `opencl` expose Intel GPUs).
5. Given a sub-device, a further layer (sub-sub-device) is selectable with `.` plus `*` or a numeric index. `ONEAPI_DEVICE_SELECTOR=level_zero:0.*.*` partitions device 0 into sub-devices then each into sub-sub-devices; only the grandchild sub-sub-devices become available (device 0 and its child partitions do not).
6. A filter = term + action: accept `<term>`, discard `!<term>`. Discarding filters, if any, **must all appear at the end**.
7. If filters accept and discard the same device, **discard wins** (e.g. `*:gpu;!cuda:*`).
8. With only discarding filters, an accepting filter matching all devices (but not sub-devices/sub-sub-devices) is implicitly included: `!*:cpu` accepts all but CPU; `opencl:*;!*:cpu` accepts all OpenCL™ devices except OpenCL™ CPU devices.
9. Rejecting an already-omitted device is legal, has no effect; the device stays omitted.

| Example | Result |
|---|---|
| `ONEAPI_DEVICE_SELECTOR=opencl:*` | Only OpenCL™ devices. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | Only GPU devices on the Level Zero platform. |
| `ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"` | GPU devices from both Level Zero and OpenCL™. Escaping (like quotation marks) will likely be needed when using semi-colon separated entries. |
| `ONEAPI_DEVICE_SELECTOR=opencl:gpu,cpu` | Only CPU and GPU devices on the OpenCL™ platform. |
| `ONEAPI_DEVICE_SELECTOR=opencl:0` | Only the device with index 0 on the OpenCL™ backend. |
| `ONEAPI_DEVICE_SELECTOR=hip:0,2` | Only devices with indices of 0 and 2 from the HIP backend. |
| `ONEAPI_DEVICE_SELECTOR=opencl:0.*` | All the sub-devices from the OpenCL™ device with index 0 are exposed as SYCL root devices. No other devices are available. |
| `ONEAPI_DEVICE_SELECTOR=opencl:0.2` | The third sub-device (2 in zero-based counting) of the OpenCL™ device with index 0 will be the sole device available. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:*,*.*` | Each Level Zero device (card) is exposed as a SYCL root device and each sub-device is also exposed as a SYCL root device. |
| `ONEAPI_DEVICE_SELECTOR="opencl:*;!opencl:0"` | All OpenCL™ devices except for the device with index 0. |
| `ONEAPI_DEVICE_SELECTOR="!*:cpu"` | All devices except for CPU devices. |

**Notes**

- Backend always required (error if absent) and MUST be followed by `:` plus **at least one device specifier**, else error.
- Sub-device syntax partitions the root device per `info::partition_property::partition_by_affinity_domain` and `info::partition_affinity_domain::next_partitionable`; the root device is determined by the underlying backend.
- Level Zero backend: `ZE_FLAT_DEVICE_HIERARCHY` also affects root-device exposure; for Intel GPUs the syntax can expose tiles or CCSs as SYCL root devices, with exact mapping set by that variable.
- `;` and `!` are shell-special — quote the selector string if it contains them.

## Intel® oneAPI Level Zero Backend Specification

Adds a Level Zero backend for SYCL on the Level Zero runtime enabled with the oneAPI Level Zero Specification; aims at best possible SYCL performance on supported targets. Implementations must predefine `SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO` to one of:

| Value | Description |
|---|---|
| 1 | Initial extension version. |
| 2 | Added support for the `make_buffer()` API. |
| 3 | Added `device` member to `backend_input_t<backend::ext_oneapi_level_zero, queue>`. |
| 4 | Change the definition of `backend_input_t` and `backend_return_t` for the `queue` object, which changes the API for `make_queue` and `get_native` (when applied to `queue`). |
| 5 | Added support for `make_image()` API. |

**NOTE:** follows the SYCL 2020 backend specification; prior Level Zero interop APIs are deprecated and **will be removed in the next release**.

**Prerequisites:** Level Zero loader and drivers must be installed for the SYCL runtime to recognize and enable the backend (see Intel® oneAPI DPC++/C++ Compiler System Requirements).

### User-visible Level Zero Backend Selection and Default Backend

```cpp
enum class backend {
   // ...
   ext_oneapi_level_zero,
   // ...
};
```

- **Environment variable:** `ONEAPI_DEVICE_SELECTOR` with `level_zero` as backend.
- **Programming API:** the Filter Selector extension (SYCL Proposals: Filter Selector) — like `ONEAPI_DEVICE_SELECTOR`, this selector filters.
- With neither, the implementation chooses Level Zero for GPU devices supported by the installed Level Zero runtime. The serving backend for a SYCL platform is queried with the `get_backend()` member function of `sycl::platform`.

### Interoperability with the Level Zero API

Headers, **in this order**:

```cpp
#include "level_zero/ze_api.h"
#include "sycl/ext/oneapi/backend/level_zero.hpp"
```

Notation below: `B` = `backend::ext_oneapi_level_zero`; `Own{X}` = `ext::oneapi::level_zero::ownership Ownership{ext::oneapi::level_zero::ownership::X}` (every `Ownership` member defaults to `transfer`).

#### Mapping of SYCL Objects to Level Zero Handles

| SYCL Type | `backend_return_t<B, SyclType>` | `backend_input_t<B, SyclType>` |
|---|---|---|
| `platform` | `ze_driver_handle_t` | `ze_driver_handle_t` |
| `device` | `ze_device_handle_t` | `ze_device_handle_t` |
| `context` | `ze_context_handle_t` | `struct { ze_context_handle_t NativeHandle; std::vector<device> DeviceList; Own{transfer}; }` |
| `queue` | `ze_command_queue_handle_t` | `struct { ze_command_queue_handle_t NativeHandle; Own{transfer}; }` — **deprecated in Version 3**.<br>`struct { ze_command_queue_handle_t NativeHandle; device Device; Own{transfer}; }` — **supported since Version 3**. |
| `event` | `ze_event_handle_t` | `struct { ze_event_handle_t NativeHandle; Own{transfer}; }` |
| `kernel_bundle` | `std::vector<ze_module_handle_t>` | `struct { ze_module_handle_t NativeHandle; Own{transfer}; }` |
| `kernel` | `ze_kernel_handle_t` | `struct { kernel_bundle<bundle_state::executable> KernelBundle; ze_kernel_handle_t NativeHandle; Own{transfer}; }` |
| `buffer` | `void *` | `struct { void *NativeHandle; Own{transfer}; }` |

#### Obtaining Native Level Zero Handles from SYCL Objects

```cpp
template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>
```

- Supported for `platform`, `device`, `context`, `queue`, `event`, `kernel_bundle`, `kernel`.
- `get_native(queue)` returns `ze_command_queue_handle_t` or `ze_command_list_handle_t` per how the queue was created: SYCL-constructor queues use a default documented for `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`; `make_queue()` queues follow their input argument and are **not affected** by that default/variable.
- **Not supported for `buffer`.** Use `interop_handle` (SYCL spec "Class interop_handle"): `get_native_mem<B>` returns the value from `zeMemAllocShared()`, `zeMemAllocDevice()`, or `zeMemAllocHost()`, not directly host-accessible — data may need copying to the host. Allocation type = `type` member of `ze_memory_allocation_properties_t` from `zeMemGetAllocProperties`.

```cpp
Queue.submit([&](handler &CGH) {
    auto BufferAcc = Buffer.get_access<access::mode::write>(CGH);
    CGH.host_task([=](const interop_handle &IH) {
        void *DevicePtr =
            IH.get_native_mem<backend::ext_oneapi_level_zero>(BufferAcc);
        ze_memory_allocation_properties_t MemAllocProperties{};
        ze_result_t Res = zeMemGetAllocProperties(
            ZeContext, DevicePtr, &MemAllocProperties, nullptr);
        ze_memory_type_t ZeMemType = MemAllocProperties.type;
    });
 }).wait();
```

#### Construct a SYCL Object from a Level Zero Handle

`sycl`-namespace free functions specialized for the Level Zero backend; all take an `Ownership` member defaulting to transfer.

| Function | Description |
|---|---|
| `make_platform<B>(const backend_input_t<B, platform> &)` | Platform from `ze_driver_handle_t`; environment has a fixed set (`sycl::platform::get_platforms()`) — copy of one entry, no new platform. |
| `make_device<B>(const backend_input_t<B, device> &)` | Device from `ze_device_handle_t`; devices counted by `sycl::device::get_devices()`, sub-devices by `sycl::device::create_sub_devices(...)`; no new device. |
| `make_context<B>(const backend_input_t<B, context> &)` | Context from `ze_context_handle_t`, created against `DeviceList`; **≥1 device**, all from the same SYCL platform and same Level Zero driver. |
| `make_queue<B>(const backend_input_t<B, queue> &, const context &Context)` | Queue from `ze_command_queue_handle_t`; `Context` valid/Level Zero; `Device` member must be in `Context`. **Deprecated** variant: queue attached to the **first device in Context**. Since **version 4**: accepts `ze_command_queue_handle_t` **or** `ze_command_list_handle_t` (immediate command list generally faster); accepts a `Properties` member with any SYCL queue-constructor property **except** `compute_index` (built into the command queue/list). |
| `make_event<B>(const backend_input_t<B, event> &, const context &Context)` | Event from `ze_event_handle_t`; `Context` valid/Level Zero; event should come from an event pool created in the same context. |
| `make_kernel_bundle<B, bundle_state::executable>(const backend_input_t<B, kernel_bundle<bundle_state::executable>> &, const context &Context)` | Executable `kernel_bundle` from `ze_module_handle_t`; module on the same valid context, **fully linked** (no further `zeModuleDynamicLink`). With transfer the runtime destroys the input module — **no outstanding `ze_kernel_handle_t`** to the underlying `ze_module_handle_t` when the interop `kernel_bundle` destructor runs. |
| `make_kernel<B>(const backend_input_t<B, kernel> &, const context &Context)` | Kernel from `ze_kernel_handle_t`; `KernelBundle` names the module's `kernel_bundle` — **exactly one** Level Zero module; context valid, module created on it. With transfer the runtime destroys the input kernel. |
| `template <backend Backend, typename T, int Dimensions = 1, typename AllocatorT = buffer_allocator<std::remove_const_t<T>>> buffer<T, Dimensions, AllocatorT> make_buffer(const backend_input_t<Backend, buffer<T, Dimensions, AllocatorT>> &, const context &Context)` | **Revision 2.** Buffer from a pointer returned by `zeMemAllocShared()`, `zeMemAllocDevice()`, or `zeMemAllocHost()`; `Context` valid/Level Zero, associated with a single device matching the allocation device, memory allocated on it; buffer **accessible in other contexts** too. With transfer the runtime frees the allocation. See Buffer Synchronization Rules. |
| `buffer<T, Dimensions, AllocatorT> make_buffer(const backend_input_t<Backend, buffer<T, Dimensions, AllocatorT>> &, const context &Context, event AvailableEvent)` (same template) | **Revision 2.** Same semantics/restrictions; `AvailableEvent` must be a valid SYCL event and the buffer must wait for it to signal the native memory handle is ready. |
| `template<backend Backend, int Dimensions = 1, typename AllocrT = sycl::image_allocator> image<Dimensions, AllocrT> make_image(const backend_input_t<Backend, image<Dimensions, AllocrT>> &backendObject, const context &targetContext);` | **Revision 5.** Image from `ze_image_handle_t`; Level Zero cannot supply image info, so it must be provided (struct below). |
| `image<Dimensions, AllocrT> make_image(const backend_input_t<Backend, image<Dimensions, AllocrT>> &backendObject, const context &targetContext, event availableEvent);` (same template) | **Revision 5.** Same semantics/restrictions; `availableEvent` must be a valid SYCL event and the image must wait for it to signal the native handle is ready. |

`make_image` `backend_input_t` struct:

```cpp
struct type {
    ze_image_handle_t ZeImageHandle;
    sycl::image_channel_order ChanOrder;
    sycl::image_channel_type ChanType;
    sycl::range<Dimensions> Range;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
};
```

- `Range` ordering `(width)`, `(width, height)`, `(width, height, depth)` for 1D/2D/3D, matching the `ze_image_desc` used to create the handle. Width-first/depth-last holds for the SYCL 1.2.1 images supported here; future `sampled_image`/`unsampled_image` might differ.
- Image usable only on the **single device where created** (may be relaxed); `Context` valid/Level Zero, image created on it, accessible only from kernels submitted to a queue using that context.
- Transfer (default): runtime frees the allocation when `~image` fires, destructor may not need to block. `keep`: destructor does not free, **blocks until all queue work on the image completes**; caller frees.

```cpp
ze_image_handle_t ZeHImage;
// ... user provided LevelZero ZeHImage image
// handle gotten somehow (possibly zeImageCreate)

// the informational data that matches ZeHImage
sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
sycl::image_channel_type ChanType = sycl::image_channel_type::unsigned_int8;
size_t width = 4;
size_t height = 2;
sycl::range<2> ImgRange_2D(width, height);

constexpr sycl::backend BE = sycl::backend::ext_oneapi_level_zero;
sycl::backend_input_t<BE, sycl::image<2>> ImageInteropInput{
    ZeHImage,
    ChanOrder,
    ChanType,
    ImgRange_2D,
    sycl::ext::oneapi::level_zero::ownership::transfer };

sycl::image<2> Image_2D = sycl::make_image<BE, 2>(ImageInteropInput, Context);
```

#### Level Zero Handle Ownership and Thread-safety

Level Zero does **not reference-count** its objects. Default is transfer to the SYCL runtime; some interop APIs allow keeping ownership.

```cpp
namespace sycl { namespace ext { namespace oneapi { namespace level_zero {
enum class ownership { transfer, keep };
} } } }
```

1. **SYCL Runtime Takes Ownership (default):** any `make_*` without explicit `ownership::keep` gives the handle to the runtime. Do **not use** the handle after the last host copy of the SYCL object is destroyed; do **not destroy** it (SYCL Common Reference Semantics).
2. **Application Keeps Ownership (explicit):** with `ownership::keep` the runtime neither owns nor destroys the handle; the app destroys it when done, but **not before** the last host copy of the SYCL object is destroyed.
3. **Obtaining a Native Handle Does Not Change Ownership:** `get_native<B>` leaves ownership unchanged; the handle may not be used after the last host copy of the SYCL object is destroyed unless created with `ownership::keep`.
4. **Multi-threaded Environment:** the Level Zero API is **not thread-safe** — no simultaneous use of a handle from different threads; the runtime owns handles and should not be used to access them directly afterwards.

#### Interoperability Buffer Synchronization Rules

- The buffer uses the Level Zero allocation for its **full lifetime**; that allocation's contents are **unspecified** during the lifetime.
- Modifying the Level Zero allocation while the SYCL buffer lives is **undefined behavior**; initial buffer contents = allocation contents at buffer construction.
- Destructor behavior depends on `Ownership`, triggering only when the last reference count drops (SYCL spec "Buffer Synchronization Rules"):
  - `keep`: destructor **blocks until all queue work on the buffer completes**; contents **not copied back** to the allocation.
  - `transfer`: destructor need not block despite outstanding work; runtime frees the allocation **asynchronously** when no longer used in queues.

### Level Zero Additional Functionality

#### Device Information Descriptors

```cpp
sycl::queue Queue;
auto Device = Queue.get_device();

size_t freeMemory =
  Device.get_backend_info<sycl::ext::oneapi::level_zero::info::device::free_memory>();
```

`sycl::ext::oneapi::level_zero::info::device::free_memory` returns the number of bytes of free memory for the device.

```cpp
namespace sycl { namespace ext { namespace oneapi { namespace level_zero {
namespace info { namespace device {
struct free_memory {
    using return_type = size_t;
};
} } } } } }
```

## Programming with the Intel® oneAPI Level Zero Backend

Supported scenarios for multi-card and multi-tile programming.

### Device Discovery

**Scaling:** `ZE_FLAT_DEVICE_HEIRARCHY` [sic: source garbled] affects how the driver/l0 exposes GPU devices. Allowed values for `ZE_FLAT_DEVICE_HIERARCHY`: `FLAT`, `COMPOSITE`, `COMBINED`. `ONEAPI_DEVICE_SELECTOR` queries the exposed devices and chooses how to order and filter them.

**Root-devices:** Intel GPUs are SYCL GPU devices (root-devices), found with `sycl-ls`:

```bash
sycl-ls
```

```text
[opencl:gpu:0] Intel(R) OpenCL HD Graphics, Intel(R) UHD Graphics 630 [0x3e92] 3.0 [21.49.21786]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz 2.1
[2020.11.11.0.03_160000]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) UHD Graphics 630 [0x3e92] 1.2
[1.2.21786]
[host:host:0] SYCL host platform, SYCL host device 1.2 [1.2]
```

`sycl-ls` shows all devices/platforms of all SYCL backends seen by the runtime; here the CPU (OpenCL™ backend) and two GPUs corresponding to the single physical GPU (OpenCL™ or Level Zero). Filter observable root-devices with:

- **Option One:** `ONEAPI_DEVICE_SELECTOR`:
  ```bash
  ONEAPI_DEVICE_SELECTOR=ext_oneapi_level_zero sycl-ls
  ```
  output = the `[ext_oneapi_level_zero:gpu:0] ... [1.2.21786]` line above, alone.
- **Option Two:** a similar API (Filter Selector), e.g. `filter_selector("ext_oneapi_level_zero")` only sees Level Zero operated devices.
- Multiple GPUs = multiple root-devices: on **Linux**, multiple SYCL root-devices of the **same** SYCL platform; on **Windows**, root-devices of **multiple different** SYCL platforms.
- Emulate multiple GPU cards with `CreateMultipleRootDevices=N NEOReadDebugKeys=1`:
  ```bash
  CreateMultipleRootDevices=2 NEOReadDebugKeys=1 ONEAPI_DEVICE_SELECTOR=ext_oneapi_level_zero sycl-ls
  ```
  output = the same `[ext_oneapi_level_zero:gpu:0] ... [1.2.21786]` line plus an identical `[ext_oneapi_level_zero:gpu:1] ... [1.2.21786]` line.
  **NOTE:** `CreateMultipleRootDevices` is experimental, not validated, debug/experimental purposes only.

**Sub-devices:** multi-tile Intel GPU hardware partitions into sub-devices corresponding to physical tiles:

```cpp
try {
  vector<device> SubDevices = RootDevice.create_sub_devices<
  sycl::info::partition_property::partition_by_affinity_domain>(
  sycl::info::partition_affinity_domain::next_partitionable);
}
```

- Each call returns the **same sub-devices in their persistent order**.
- `ZE_AFFINITY_MASK` controls exposed sub-devices.
- `partition_by_affinity_domain` is the **only** partitioning type for Intel GPUs; `next_partitionable` and `numa` the **only** supported properties.
- `CreateMultipleSubDevices=N NEOReadDebugKeys=1` emulates multiple GPU tiles. **NOTE:** experimental, not validated, debug/experimental purposes only.

### Contexts

Contexts give resource isolation and sharing; a context may hold one or multiple devices, and root-devices and sub-devices can mix in one context but must come from the **same SYCL platform**. A `kernel_bundle` for a multi-device context is built to **each root-device** in the context; for multiple sub-devices of the same root-device only a **single build** (to that root-device) is needed.

### Memory

#### Unified Shared Memory (USM)

| Allocation | Accessible by | Data location | Copy/sync |
|---|---|---|---|
| `malloc_device` | only the specified device (not other devices in the context, not the host) | always on the device; fastest for kernel execution | explicit copy to host or other devices in the context |
| `malloc_host` | host and any other device in the context | always on the host, accessed via Peripheral Component Interconnect (PCI) from devices | none for host/device sync |
| `malloc_shared` | only the host and the specified device | migrates (operated by the Level Zero driver) between host and device for faster access | none between host and device; needed for other devices in the context |

- Root-device memory is accessible by **all its sub-devices (tiles)**.
- In a context of multiple sub-devices of one root-device, use `malloc_device` on that root-device rather than slower `malloc_host`; with `malloc_device` an explicit copy out to the host is needed to see the data there.

#### Buffers

SYCL buffers created against a context map under the hood to Level Zero USM:

- Integrated device: allocated on the **host**, accessible by host and device **without copying**.
- Context with sub-devices of the same root-device (possibly including the root-device): allocated on that **root-device**, accessible by all devices in the context; host synchronization done by the SYCL runtime with map/unmap performing implicit copies when necessary.
- Context with devices from **different root-devices**: allocated on the **host** (accessible to all devices).

### Queues

A SYCL queue always attaches to a **single device** in a potential multi-device context. Scenarios, most to least performant:

**Scenario One** — single sub-device context, queue on that sub-device (tile): execution/visibility limited to that sub-device; best performance per tile.

```cpp
try {
  vector<device> SubDevices = ...;
  for (auto &D : SubDevices) {
    // Each queue is in its own context, no data sharing across them.
    auto Q = queue(D);
    Q.submit([&](handler& cgh) {...});
  }
}
```

**Scenario Two** — context with multiple sub-devices of the same root-device (multi-tile): queues attach to sub-devices for explicit scaling; the root-device should **not** be passed to this context (better performance).

```cpp
try {
  vector<device> SubDevices = ...;
  auto C = context(SubDevices);
  for (auto &D : SubDevices) {
    // All queues share the same context, data can be shared across queues.
    auto Q = queue(C, D);
    Q.submit([&](handler& cgh) {...});
  }
}
```

**Scenario Three** — context with a single root-device, queue on that root-device: work automatically distributed across all sub-devices/tiles by **implicit scaling by the driver**; simplest multi-tile enablement but cannot target specific tiles.

```cpp
try {
  // The queue is attached to the root-device, driver distributes to sub-devices, if any.
  auto D = device(gpu_selector{});
  auto Q = queue(D);
  Q.submit([&](handler& cgh) {...});
}
```

**Scenario Four** — contexts with multiple root-devices (multi-card): most unrestrictive, queues on different root-devices; most sharing possibilities at the cost of slow host-memory access or explicit copies.

## Option / API quick table

| name | purpose | key values/default | notes |
|---|---|---|---|
| `intel-level-zero-gpu`, `level-zero` | required packages | — | driver + OpenCL™ driver same package; not shipped with DPC++ |
| `SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO` | backend feature-test macro | `1`,`2`,`3`,`4`,`5` | 1 initial; 2 `make_buffer()`; 3 `device` member; 4 new `queue` interop; 5 `make_image()` |
| `ONEAPI_DEVICE_SELECTOR` | limit/expose devices; sub-devices as root devices | `<backend>:<devices>`; accept `<term>`, discard `!<term>`; backends `*`,`level_zero`,`opencl`,`cuda`,`hip`,`esimd_emulator` | backend + `:` + ≥1 device else error; discard filters last and win; warning on empty wildcard match |
| `SYCL_PI_TRACE` | show chosen/running device | `1`; `-1` verbose UR discovery | — |
| `sycl-ls`, `$ sycl-ls --verbose` | list backends/platforms/devices + IDs | — | prints backend name, device_type, device_id |
| `filter_selector("ext_oneapi_level_zero")` | API backend selection | — | Filter Selector extension |
| `ZE_FLAT_DEVICE_HIERARCHY` | how driver/l0 exposes GPU devices | `FLAT`, `COMPOSITE`, `COMBINED` | sets root/sub-device mapping; p.759 spells `ZE_FLAT_DEVICE_HEIRARCHY` [sic] |
| `ZE_AFFINITY_MASK` | exposed sub-devices | — | — |
| `CreateMultipleRootDevices=N NEOReadDebugKeys=1`, `CreateMultipleSubDevices=N NEOReadDebugKeys=1` | emulate multiple GPU cards / tiles | `N` | experimental, not validated |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | queue vs command-list default for SYCL-ctor queues | — | does not affect `make_queue()` queues |
| `sycl::get_native<B>` | native handle from a SYCL object | `backend_return_t<...>` | all listed types except `buffer` |
| `interop_handle::get_native_mem<B>` | native memory pointer for a buffer | `zeMemAlloc*` value | not directly host-accessible |
| `ext::oneapi::level_zero::ownership` | handle ownership | `transfer` (default), `keep` | no reference-counting in Level Zero |
| `...::level_zero::info::device::free_memory` | device info descriptor | `return_type = size_t` | with `device::get_backend_info<>()` |
| `ASAN_OPTIONS`/`MSAN_OPTIONS`/`TSAN_OPTIONS` `=clear_deepbind_flag=1` | clear `RTLD_DEEPBIND` for host sanitizers | `1` | see p.744 |

## Gotchas & failure modes

- **Thread-safety:** one Level Zero handle must never be used from different threads simultaneously; transferred handles must not be used/destroyed after the last host copy of the SYCL object dies.
- **Ownership mix-ups:** `keep` → app destroys the handle after the last host copy of the SYCL object is destroyed; `transfer` → app must not destroy it. `make_kernel_bundle` + transfer → no outstanding `ze_kernel_handle_t` when the interop destructor runs.
- **Buffer interop UB:** touching the Level Zero allocation while the SYCL buffer lives is undefined; `keep` blocks in the destructor and does not copy back; `transfer` frees asynchronously.
- **Image `keep`:** `~image` blocks until all queue work on the image completes and the caller frees; images are confined to their creating device and same-context queues; `make_image` needs explicit channel order/type and range matching the original `ze_image_desc` (width first, depth last).
- **Queue interop:** `compute_index` is not allowed in `make_queue()` `Properties` (v4+); `get_native(queue)` may return `ze_command_queue_handle_t` or `ze_command_list_handle_t`; `make_queue()` queues ignore `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`; immediate command lists generally faster.
- **`get_native` on `buffer` unsupported** → `interop_handle::get_native_mem<B>`, pointer not directly host-accessible.
- **Duplicate device visibility:** `opencl:gpu:0` and `ext_oneapi_level_zero:gpu:0` are the same GPU; indices are per-backend and the SYCL spec does not define which matching device is chosen — pin the backend with `ONEAPI_DEVICE_SELECTOR`.
- **Selector errors:** missing backend, or backend without `:` + device specifier, throws; discard filters must be last; quote strings with `;` or `!`.
- **Platform difference:** multiple GPUs = multiple root-devices of one SYCL platform on Linux, of different SYCL platforms on Windows.
- **Performance/limits:** root-device in a sub-device context, or `malloc_host` instead of `malloc_device` on a root-device, is slower; Scenario One best per tile, Scenario Four most flexible/slowest; only `partition_by_affinity_domain` with `next_partitionable`/`numa` for Intel GPUs.
- **Deprecations:** pre-v3 `queue` `backend_input_t` (Version 3); all pre-SYCL-2020 Level Zero interop APIs (removal next release).
- **Sanitizer limits:** no `RTLD_DEEPBIND` without `clear_deepbind_flag=1`; MSan false negatives for GPU-driver/SYCL* runtime libraries; `CreateMultipleRootDevices`/`CreateMultipleSubDevices` experimental, not validated.

## Source map

- Host-Side Compiler Sanitizers; Clang Sanitizers; Level Zero API objective — p. 744
- Level Zero Switch (packages, loader, driver), DPC++ Backends — pp. 745–746
- ONEAPI_DEVICE_SELECTOR (grammar, semantics, examples, notes) — pp. 746–749
- Backend Specification, macro values, Prerequisites, backend selection — p. 749
- Interop headers, handle mapping — pp. 750–751
- `get_native`, `interop_handle::get_native_mem` — p. 752
- `make_platform` … `make_image`, image struct + example — pp. 752–757
- Ownership and Thread-safety; Buffer Synchronization Rules — pp. 757–758
- Device Information Descriptors — pp. 758–759
- Discovery, Scaling, Root-devices, Sub-devices — pp. 759–761
- Contexts; Memory (USM, Buffers) — p. 761
- Queues, Scenarios One–Four — pp. 761–762
