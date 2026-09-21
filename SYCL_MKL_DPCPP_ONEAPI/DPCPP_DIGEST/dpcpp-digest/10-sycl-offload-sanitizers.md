---
chunk: 10-sycl-offload-sanitizers
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0
source_pages: 734-743
covers: SYCL* extension status catalog, redistributing a SYCL* application, CUDA* to SYCL* math API mapping, device-side compiler sanitizers ASan/MSan/TSan
---

# SYCL* Support, Redistribution, CUDA* Migration, and Compiler Sanitizers

> **Scope.** Which SYCL* extensions are Supported vs Experimental, which libraries must ship with a SYCL* application, which CUDA* (SDK11.4) math APIs map to `sycl::ext::intel::math::`, and how to build/run device-side AddressSanitizer, MemorySanitizer, and ThreadSanitizer for SYCL* and OpenMP*.

## Key facts

- The compiler compiles C++ and SYCL* source files with code for both CPU and a wide range of compute accelerators.
- **Experimental** extension APIs are unstable, may be changed or even removed without prior notice, and are not recommended for production; **Supported** APIs are generally stable and backward compatible.
- Compiler sanitizers detect buffer overflows, dangling-pointer access, use of uninitialized memory, and other undefined behavior; they work with OpenMP* and SYCL*, on **CPU devices and Ponte Vecchio GPUs, Linux***.
- Device-side MSan since **2025.1**, its local/private memory check since **2025.2**; device-side TSan since **2025.2**.
- OpenMP (C/C++) sanitizers run on a GPU device only. Level Zero is the default backend when none is specified.

## SYCL Support

The Intel® oneAPI DPC++/C++ Compiler compiles C++ and SYCL* source files with code for both CPU and a wide range of compute accelerators.

## SYCL Extensions

Beyond core SYCL* features the compiler implements the extensions below, each with its source status.

**Supported:** C and C++ Standard Libraries Support, sycl_ext_intel_buffer_location, sycl_ext_intel_cslice, sycl_ext_intel_dataflow_pipes, sycl_ext_intel_device_info, sycl_ext_intel_esimd, sycl_ext_intel_esimd_functions, sycl_ext_intel_fpga_device_selector, sycl_ext_intel_fpga_lsu, sycl_ext_intel_fpga_reg, sycl_ext_intel_kernel_args_restrict, sycl_ext_intel_legacy_image, sycl_ext_intel_mem_channel_property, sycl_ext_intel_queue_immediate_command_list, sycl_ext_intel_queue_index, sycl_ext_intel_usm_address_spaces, sycl_ext_oneapi_accessor_properties, sycl_ext_oneapi_assert, sycl_ext_oneapi_backend_level_zero, sycl_ext_oneapi_bfloat16, sycl_ext_oneapi_default_context, sycl_ext_oneapi_discard_queue_events, sycl_ext_oneapi_dot_accumulate, sycl_ext_oneapi_enqueue_barrier, sycl_ext_oneapi_filter_selector, sycl_ext_oneapi_free_function_queries, sycl_ext_oneapi_local_memory, sycl_ext_oneapi_memcpy2d, sycl_ext_oneapi_peer_access, sycl_ext_oneapi_prod, sycl_ext_oneapi_queue_empty, sycl_ext_oneapi_queue_priority, sycl_ext_oneapi_srgb, sycl_ext_oneapi_sub_group_mask, sycl_ext_oneapi_use_pinned_host_memory_property, sycl_ext_oneapi_usm_device_read_only, sycl_ext_oneapi_weak_object, sycl_khr_default_context

**Experimental:** sycl_ext_codeplay_enqueue_native_command, sycl_ext_codeplay_max_registers_per_work_group_query, sycl_ext_intel_cache_config, sycl_ext_intel_data_flow_pipes_properties, sycl_ext_intel_fp_control, sycl_ext_intel_fpga_task_sequence, sycl_ext_intel_grf_size, sycl_ext_intel_matrix, sycl_ext_oneapi_annotated_arg, sycl_ext_oneapi_annotated_ptr, sycl_ext_oneapi_auto_local_range, sycl_ext_oneapi_bfloat16_math_functions, sycl_ext_oneapi_bindless_images, sycl_ext_oneapi_complex, sycl_ext_oneapi_composite_device, sycl_ext_oneapi_copy_optimize, sycl_ext_oneapi_cuda_async_barrier, sycl_ext_oneapi_cuda_tex_cache_read, sycl_ext_oneapi_device_architecture, sycl_ext_oneapi_device_global, sycl_ext_oneapi_enqueue_functions, sycl_ext_oneapi_graph, sycl_ext_oneapi_group_load_store, sycl_ext_oneapi_group_sort, sycl_ext_oneapi_in_order_queue_events, sycl_ext_oneapi_invoke_simd, sycl_ext_oneapi_kernel_arg_properties, sycl_ext_oneapi_kernel_compiler, sycl_ext_oneapi_kernel_compiler_opencl, sycl_ext_oneapi_kernel_compiler_spirv, sycl_ext_oneapi_kernel_properties, sycl_ext_oneapi_matrix, sycl_ext_oneapi_max_work_group_query, sycl_ext_oneapi_native_math, sycl_ext_oneapi_non_uniform_groups, sycl_ext_oneapi_prefetch, sycl_ext_oneapi_private_alloca, sycl_ext_oneapi_profiling_tag, sycl_ext_oneapi_properties, sycl_ext_oneapi_raw_kernel_arg, sycl_ext_oneapi_root_group, sycl_ext_oneapi_uniform, sycl_ext_oneapi_user_defined_reductions, sycl_ext_oneapi_virtual_mem

## Redistribute Your SYCL* Application

### Files to Include when Redistributing Your SYCL Applications

- **Always (minimum):** `libsycl.so.8`, `libur_loader.so.0`
- **By backend:** OpenCL™ → `libur_adapter_opencl.so.0`; Level Zero (default backend when none is specified) → `libur_adatper_level_zero.so.0` [sic: source spelling]. `libze_trace_collector.so` is required for runtime-level debugging of Level Zero.
- **By optional feature:** XPTI → `libxtpi.a` [sic: source spelling] and `libxptifw.so`

### Linking

- **Static:** libraries are already in the application; no further redistribution required.
- **Dynamic**, either *redistribute with your application* (files in your installer, installed with the app in the same location as the executable) or *require users to install as a prerequisite* (Intel redistributable packages: Linux via the typical system package managers, Windows as a standalone download; see *Single Component Downloads and Runtime Versions*).

## CUDA* to SYCL* Migration

Not intended to be all-inclusive; APIs will be added in subsequent releases. Every CUDA* API below (SDK11.4) has an equivalent under `sycl::ext::intel::math::`; Feature for all rows is **Mathematical Functions**.

| Subfeature | CUDA* APIs (SDK11.4) |
|---|---|
| Double Precision Mathematical Functions | cyl_bessel_i0, cyl_bessel_i1, j1, jn, y0, y1, yn |
| Single Precision Mathematical Functions | cyl_bessel_i0f, cyl_bessel_i1f, j0f, j1f, jnf, y0f, y1f, ynf |
| Half2 Math Functions | h2rcp |
| Half Precision Conversion and Data Movement | `__ldcs`, `__ldcg`, `__ldca`, `__double2half` |
| Bfloat162 Comparison Functions | `__bfloat16_as_ushort` |

## Compiler Sanitizers

Detect bugs/errors including buffer overflows, dangling pointers, use of uninitialized memory, and other undefined behavior; they work with OpenMP* and SYCL*.

### System Requirements

- **Platform Support:** CPU devices and Ponte Vecchio GPUs, on Linux*.
- **GPU Configuration:** supports a Ponte Vecchio GPU card. For the Level Zero runtime, use `ZE_AFFINITY_MASK=0` to set this configuration.

### Limitations

1. Kernel execution is sequential — concurrency is forced into sequential execution when device-side ASan is enabled.
2. Device-side ASan may increase private memory usage, reducing the maximum workgroup size a kernel supports; you may then see `UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE`. Fix by reducing the SYCL local workgroup size or the OpenMP overlay management protocol (OMP) teams.
3. Many workgroups on a GPU may make device-side ASan skip an out-of-bound check for private/local memory, and device-side MSan skip uninitialized-usage checks for private/local memory.
4. OpenMP (C/C++) only supports execution on a GPU device.

### Device-Side Compiler Sanitizers

Activate with `-Xarch_device -fsanitize=<kind>`. All documented compile commands:

```bash
# OpenMP* offload (spir64 target)
icpx -fiopenmp -fopenmp-targets=spir64 -Xarch_device -fsanitize=address -g -O2 -o demo demo.cpp
icpx -fiopenmp -fopenmp-targets=spir64 -Xarch_device -fsanitize=memory -g -o demo demo.cpp
icpx -fiopenmp -fopenmp-targets=spir64 -Xarch_device -fsanitize=thread -g -o demo demo.cpp
# SYCL*
icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o demo demo.cpp
icpx -fsycl -Xarch_device -fsanitize=memory -g -o demo demo.cpp
icpx -fsycl -Xarch_device -fsanitize=thread -g -o demo demo.cpp
```

All documented run invocations (the sanitizers differ only in the `UR_ENABLE_LAYERS` layer):

```bash
# OpenMP*: GPU only. Substitute UR_LAYER_ASAN / UR_LAYER_MSAN / UR_LAYER_TSAN.
export LIBOMPTARGET_PLUGIN=unified_runtime
export UR_ENABLE_LAYERS=UR_LAYER_ASAN
export ZE_AFFINITY_MASK=0
./demo
# SYCL* on GPU
export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
./demo
# SYCL* on CPU
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
./demo
```

> NOTE: Setting the `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` environment variable for OpenMP offload on GPU devices will cause errors.

#### Device-Side AddressSanitizer

Device-side AddressSanitizer (ASan) supports these checks for OpenMP and SYCL* device code:

- **Both OpenMP C/C++ and SYCL*:** invalid argument, memory leak, memory overhead statistics, misaligned access, multiple error reports, nullpointer access, out-of-bounds on device global, out-of-bounds on global unified shared memory (USM), out-of-bounds on local, out-of-bounds on private, use-after-free.
- **SYCL only (n/a for OpenMP C/C++):** bad-context, bad-free, double-free, out-of-bounds on memory buffer.
- **Split:** kernel filter — Supported for SYCL, **Not supported** for OpenMP C/C++.

**Compiler Flags**

| Flag | Default | Meaning |
|---|---|---|
| `-mllvm -asan-spir-locals=0 \| 1` | 1 | ASan detection for local memory; 0 disables this check. Runtime flag: `detect_locals`. |
| `-mllvm -asan-spir-privates=0 \| 1` | 1 | ASan detection for private memory; 0 disables this check. Runtime flag: `detect_privates`. |

**Runtime Flags** — passed to device-side ASan with the `UR_LAYER_ASAN_OPTIONS` environment variable:

```bash
UR_LAYER_ASAN_OPTIONS=redzone:32 ./demo
UR_LAYER_ASAN_OPTIONS="quarantine_size_mb:0; print_stats:1" ./demo
```

| Flag | Default | Meaning |
|---|---|---|
| detect_kernel_arguments | true | Detect invalid kernel arguments, such as using pointers from a different context or device. |
| detect_leaks | true | Memory leak detection. |
| detect_locals | true | Detect out-of-bounds errors on local memory, such as shared local memory (SLM). |
| detect_privates | true | Detect out-of-bounds errors on private memory. |
| halt_on_error | true | Crash after the first error report; effective only if compiled with `-fsanitize-recover=address`. |
| print_stats | false | Print memory overhead statistics after printing an error message. |
| quarantine_size_mb | 0 | Quarantine size (MB) for detecting use-after-free; lower values reduce memory usage but increase false negatives. 0 = no quarantine (use-after-free still detected). |
| redzone | 16 | Minimal size (bytes) of redzones around USM heap objects; requirement `redzone` >= 16, a power of two. |

> NOTE: A larger redzone size may help capture out-of-bounds errors with a large offset, but may exhaust your device memory.

#### Device-Side MemorySanitizer

Device-side MemorySanitizer (MSan), an LLVM tool that detects use of uninitialized memory (UUM) in C/C++ code, supports OpenMP and SYCL device code starting with the 2025.1 release; activate with `-Xarch_device -fsanitize=memory`. This SYCL accelerator extension provides a device-side MSan focusing on device USM to address common use cases. The 2025.2 release extended support to a local and private memory check, enabled via the compiler options below; that check increases runtime overhead.

**Compiler Flags**

| Flag | Default | Meaning |
|---|---|---|
| `-mllvm -msan-spir-locals=0 \| 1` | 1 | MSan detection for local memory; 0 disables this check. |
| `-mllvm -msan-spir-privates=0 \| 1` | 1 | MSan detection for private memory; 0 disables this check. |

**Runtime Flags** — passed to device-side MSan with the `UR_LAYER_MSAN_OPTIONS` environment variable:

```bash
UR_LAYER_MSAN_OPTIONS=recover:true ./demo
```

| Flag | Default | Meaning |
|---|---|---|
| recover | false | Terminate kernel execution after the first error; if true, the kernel continues executing even after an error is detected. |

#### Device-Side ThreadSanitizer

Device-side ThreadSanitizer (TSan), an LLVM tool that detects data races in C/C++ code, supports OpenMP and SYCL device code starting with the Intel® oneAPI DPC++/C++ Compiler 2025.2 release; activate with `-Xarch_device -fsanitize= thread` [sic: source shows a space after `=`], focusing on device USM to address common use cases.

## Gotchas & failure modes

- **`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` breaks OpenMP offload** — it causes errors there; OpenMP needs `LIBOMPTARGET_PLUGIN=unified_runtime` + `UR_ENABLE_LAYERS=UR_LAYER_ASAN`/`UR_LAYER_MSAN`/`UR_LAYER_TSAN` + `ZE_AFFINITY_MASK=0`. Only the SYCL path uses `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`; SYCL CPU runs use `ONEAPI_DEVICE_SELECTOR=opencl:cpu`.
- **ASan serializes kernels** and inflates private memory, which can raise `UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE`; shrink the SYCL local workgroup size or OpenMP OMP teams.
- **Checks can be silently skipped:** with many GPU workgroups, ASan may skip out-of-bound checks for private/local memory and MSan may skip uninitialized-usage checks there.
- **`halt_on_error` needs `-fsanitize-recover=address`** at compile time to have any effect.
- **`quarantine_size_mb=0`** still detects use-after-free but without quarantine, so false negatives are likelier; raising it costs memory. **`redzone` must be >= 16 and a power of two** and can exhaust device memory at large sizes.
- **Redistribution:** minimum is `libsycl.so.8` + `libur_loader.so.0`; backend/XPTI files are conditional. Level Zero is the default backend, so `libur_adatper_level_zero.so.0` [sic] is needed even when unselected; the source spells the XPTI files `libxtpi.a` [sic] and `libxptifw.so`.
- **Experimental extensions may be changed or removed without notice** and are not recommended for production. **CUDA* migration coverage is partial** — the table is not all-inclusive.

## Source map

- SYCL Support / SYCL Extensions — pp. 734–736
- Redistribute Your SYCL* Application — p. 737
- CUDA* to SYCL* Migration — pp. 737–738
- Compiler Sanitizers intro, System Requirements, Limitations — pp. 738–739
- Device-Side AddressSanitizer — pp. 739–741
- Device-Side MemorySanitizer — p. 742
- Device-Side ThreadSanitizer, See Also — p. 743

## See Also (source)

`fiopenmp`, `Qiopenmp` compiler option; `fopenmp` compiler option; `fsycl` compiler option.
