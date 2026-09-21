## 20 · SYCL extension index and terminology

### Extension index

**Fact.** The oneAPI guide's "SYCL* Extensions" section names **no individual extension**; it defers to the external authoritative list: "For a list of SYCL extensions supported by the Intel® oneAPI DPC++ Compiler, refer to SYCL Extensions." The digest is deliberately partial here — consult that page for the complete set. The DPC++ book chapters do name the constructs below; `ch15` and `ch18` name none explicitly (they reference extensions only generically).

**Fact.** SYCL 2020 defines backend interoperability only for OpenCL (`backend::opencl`); other backends are provided via extensions.

| Extension / construct | Namespace or header | Purpose |
|---|---|---|
| `sycl_ext_oneapi_backend_level_zero` | DPC++ ext; Level Zero interop | create SYCL objects from native Level Zero objects: `make_device`, `make_context`, `make_buffer`, `make_kernel_bundle`, `make_kernel`, `get_native` |
| `backend::ext_oneapi_level_zero` | `sycl::backend` enumerator (`ext_oneapi_*`) | Level Zero backend (platform "Intel(R) Level-Zero") |
| `backend::ext_oneapi_cuda` | `sycl::backend` enumerator (`ext_oneapi_*`) | NVIDIA CUDA backend |
| `backend::ext_oneapi_hip` | `sycl::backend` enumerator (`ext_oneapi_*`) | AMD HIP backend |
| `ext::oneapi::level_zero::ownership::keep` | `sycl::ext::oneapi::level_zero` | ownership policy when constructing a context/buffer from Level Zero objects |
| `sycl_ext_intel_online_compiler` | `sycl_ext_intel_*` naming; experimental | online compiler from a source language the backend cannot consume (OpenCL C → SPIR-V): `online_compiler<source_language::opencl_c> compiler(d); compiler.compile(src)`; replaces removed `build_with_source` |
| `ext::intel::pipe` | `sycl::ext::intel`; `<sycl/ext/intel/fpga_extensions.hpp>` | type-identified FIFO between kernels/host/peripherals; `pipe<class some_pipe, int>` with `::read()`, `::write()` |
| `ext::intel::fpga_selector_v` | `sycl::ext::intel`; FPGA header | select a device that identifies itself as an FPGA |
| `ext::intel::fpga_emulator_selector_v` | `sycl::ext::intel`; FPGA header | select the FPGA emulator for rapid emulation (supports FPGA pipes and `fpga_reg`) |
| `fpga_reg` | FPGA kernel variable (no namespace stated) | kernel variable available on the FPGA emulator along with FPGA pipes |
| `sycl_ext_oneapi_properties` | DPC++ ext; experimental | compile-time properties for classes such as buffers and accessors |
| `sycl_ext_oneapi_annotated_ptr` | DPC++ ext | pointers annotated with information beyond address spaces; could inform `sycl::multi_ptr` |
| `sycl_ext_oneapi_kernel_properties` | DPC++ ext | kernel configuration controls; could replace C++ attributes |
| `sycl_ext_oneapi_device_global` | DPC++ ext | desired memory behavior and access controls |
| `sycl_ext_oneapi_prefetch` | DPC++ ext | prefetch / access controls |
| `sycl_ext_oneapi_invoke_simd` | DPC++ ext | `invoke_simd` (modelled on `std::invoke`) calls explicitly vectorized (SIMD) code from an SPMD kernel |
| `[[intel::kernel_args_restrict]]` | vendor attribute (no ext namespace stated) | compiler ignores dependencies between accessor arguments; more aggressive optimization |
| `[[cl::reqd_work_group_size(X, Y, Z)]]` | vendor attribute; X, Y, Z = integer ND-range dimensions | specifies work-group size |
| `-Xsfpc` | compiler option | removes intermediary floating-point rounding operations and conversions where possible; carries extra bits to maintain precision |
| `-Xsno-accessor-aliasing` | compiler option | ignores dependencies between accessor arguments in a SYCL* kernel |

**Fact.** Core SYCL 2020 properties (not extensions) named in the sources: `property::queue::in_order`, `property::queue::enable_profiling`, `property::buffer::use_host_ptr`, `property::buffer::use_mutex`, `property::buffer::context_bound`. Pipe identity shape: `template <typename name, typename dataT, size_t min_capacity = 0> class pipe;`.

**Fact.** Not found in the searched sources: any `sycl::ext::oneapi::experimental::` name, or a `properties/ext_oneapi_properties` header path.

### Glossary

| Term | Definition |
|---|---|
| Accelerator | Specialized component whose compute resources quickly execute a subset of operations. Examples: CPU, FPGA, GPU |
| Accessor | Declares desired location (host, device) and mode (read, write) of access |
| Application Scope | Code that executes on the host |
| Buffers | Memory object conveying type and item count of data to the device for computation |
| Command Group Scope | Code acting as interface between host and device |
| Command Queue | Issues command groups concurrently |
| Compute Unit | Grouping of processing elements into a "core" with shared elements; faster access than memory on other compute units of the device |
| Device | Accelerator component with compute resources executing a subset of operations quickly; a CPU employed as a device acts as an accelerator. Examples: CPU, FPGA, GPU |
| Device Code | Code executing on the device, not the host; via lambda, functor, or kernel class |
| DPC++ | Open source project adding SYCL* support to the LLVM C++ compiler |
| Fat Binary | Application binary with device code for multiple devices; generic code (SPIR-V) + target-specific executable code |
| Fat Library | Archive/library of object code for multiple devices; generic (SPIR-V) + target-specific object code |
| Fat Object | File of object code for multiple devices; generic (SPIR-V) + target-specific object code |
| Host | CPU-based system executing the primary program portion: application scope and command group scope |
| Host Code | Code compiled by the host compiler, executing on the host, not the device |
| Images | Formatted opaque memory object accessed via built-in function; typically pixels in a format like RGB |
| Kernel Scope | Code that executes on the device |
| ND-Range | N-Dimensional Range: group of work-items across one, two, or three dimensions |
| Processing Element | Individual computation engine making up a compute unit |
| Single Source | Code in the same file executing on host and accelerator(s) |
| SPIR-V | Binary intermediate language representing graphical-shader stages and compute kernels |
| SYCL | Standard for a cross-platform abstraction layer; heterogeneous code in standard ISO C++ with host and kernel code in one source file |
| Work-Groups | Collection of work-items executing on a compute unit |
| Work-Item | Basic unit of computation; associated with a kernel executing on a processing element |

Optimization guidance: see §21 (Intel GPU rules of thumb, top mistakes) and §19 (profiling, tuning cycle).

### oneAPI library compatibility

| Policy | Statement |
|---|---|
| Semantic versioning | Toolkits and components use semantic versioning; apps may load dynamic libraries at runtime, so cross-release compatibility matters for APIs and ABIs shipped with oneAPI Toolkits |
| Target support | oneAPI applications are supported on 64-bit target devices |
| Non-breaking | New device drivers, dynamic libraries and compilers do not break previously deployed oneAPI-built applications; current APIs are not removed or modified without notice and a major-version iteration |
| Version match | Header files and libraries must be the same release version (never 2021.2 oneMKL headers with 2021.1) |
| Backward compat | New dynamic libraries from the Intel compilers work with applications built by older compiler versions |
| No forward compat | Older dynamic libraries do not work with newer oneAPI compilers |
| Newer-only routines | Newer oneAPI dynamic libraries may contain routines absent from earlier versions |
| Testing | Test thoroughly before deploying against a oneAPI library |

### Key gotchas

- The oneAPI guide names zero extensions; never fabricate names or headers beyond the verified table above.
- Table namespaces are as written in source (`ext::oneapi`, `ext::intel`, `ext_oneapi_*`); it is not a complete ext classification.
- `[[cl::reqd_work_group_size(X, Y, Z)]]` takes integer ND-range dimensions, not an object.
- `-Xsno-accessor-aliasing` is safe only when accessor arguments truly do not alias; source gives no safety check.
- `[[intel::kernel_args_restrict]]` asserts no dependency between accessor arguments; a false assertion permits miscompilation.
- Do not mix toolkit header and library versions across releases.
- Older oneAPI dynamic libraries never work with newer compilers; only the reverse is supported.
- Unrolling loops can change concurrent memory access behavior — recheck coalescing.
- Cross-kernel handoff of a global write is unsupported; use a pipe instead.
- Numerical results may legitimately differ after optimization; validate tolerance before accepting.
