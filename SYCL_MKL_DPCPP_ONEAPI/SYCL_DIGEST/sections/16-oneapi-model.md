## 16 · oneAPI programming model and SYCL execution/memory hierarchy

### oneAPI and its toolkits

**Fact.** oneAPI is a unified programming model and tool portfolio expressing parallelism with SYCL in modern C++ across scalar, vector, matrix and spatial (SVMS) CPU/GPU/AI/FPGA accelerators (CPU → FPGA). Performance libraries are custom-coded per target architecture, so one call is optimized across architectures.

| Product / tool | Role |
|---|---|
| Intel oneAPI Base Toolkit | DPC++/C++ Compiler, DPC++ Compatibility Tool (CUDA→SYCL migration), select libraries, analysis tools |
| Intel HPC Toolkit | complementary tools by developer workload need |
| Intel oneAPI DPC++/C++ Compiler | direct programming of accelerator-targeted code; online + offline compilation for CPU and GPU targets, offline for FPGA targets |

Two portable heterogeneous-computing methods: Data Parallel C++ with SYCL, and OpenMP for C, C++ and Fortran. Libraries: §18; tools: §19.

### SYCL and DPC++

**Fact.** SYCL is a cross-platform abstraction layer for heterogeneous processors in standard ISO C++ (host and kernel code in one source file, no new keywords or pragmas; parallelism via C++ classes (`buffer`, `queue`, `accessor`, `parallel_for`)). DPC++ adds SYCL to the LLVM C++ compiler and ships as the Intel oneAPI DPC++/C++ Compiler in the Base Toolkit. API: §01; `sycl::device_has`/`kernel_bundle`: §11; build: §17.

### The two programming models

| Model | Languages | Compiler | Parallelism expressed by |
|---|---|---|---|
| Data Parallel C++ with SYCL | ISO C++ + SYCL | Intel oneAPI DPC++/C++ Compiler | C++ classes: `queue`, `parallel_for`, `nd_range`, accessors |
| OpenMP offload | C, C++, Fortran | DPC++/C++ Compiler (C/C++); Intel Fortran Compiler Classic / Intel Fortran Compiler (Fortran) | `#pragma omp target` (C/C++), `!$omp target` (Fortran) |

**Fact.** Intel implements OpenMP version 5. Offload support: DPC++/C++ Compiler (Base and HPC Toolkits); Fortran compilers (HPC Toolkit). OpenMP is not supported for FPGA devices.

**Rule (`target` construct).** Transfers control from host to target device; variables are mapped host↔device; the host thread waits until offloaded computations complete. `nowait` → encountering thread does not wait for the target region. `target data` maps variables to the accelerator and keeps them in the target data region for its extent (useful across multiple target regions).

| Clause | Form |
|---|---|
| `DEVICE` | `DEVICE (integer-expression)` |
| `IF` | `IF ([TARGET DATA:] scalar-logical-expression)` |
| `MAP` | `MAP ([[map-type-modifier[,]] map-type: ] list)` |
| `SUBDEVICE` | `SUBDEVICE ([integer-constant ,] integer-expression [ : integer-expression [ : integer-expression]])` |
| `USE_DEVICE_ADDR` | `(list)` — available only in `ifx` |
| `USE_DEVICE_PTR` | `(ptr-list)` |

Map types: `alloc`, `to`, `from`, `tofrom`, `delete`, `release`; `target update` or the `always` map-type-modifier synchronizes an original host variable with its device variable.

```cpp
#pragma omp target map(tofrom:fa), map(to:fb,a)
#pragma omp parallel for firstprivate(a)
for(k=0; k<FLOPS_ARRAY_SIZE; k++)
  fa[k] = a * fa[k] + fb[k]
```

```fortran
!$omp target map(to: a, b ) map(tofrom: c )
!$omp parallel do private(j,i,k)
...
!$omp end parallel do
!$omp end target
```

**Fact.** Scalars (`FLOPS_ARRAY_SIZE`, `n`) are implicitly mapped; loop indices are implicitly private per the OpenMP specification (`private` optional). Map `tofrom` for input+output, `to` for input-only.

| Routine | Purpose |
|---|---|
| `int omp_get_num_procs (void)` | number of processors available to the device |
| `void omp_set_default_device(int device_num)` | controls default target device for offloading code or data |
| `int omp_get_default_device(void)` | returns the default target device |
| `int omp_get_num_devices(void)` | number of non-host devices available for offloading code or data |
| `int omp_get_device_num(void)` | device number on which the calling thread is executing |
| `int omp_is_initial_device(int device_num)` | true if current task executes on the host device, else false |
| `int omp_get_initial_device(void)` | device number representing the host device |

**Fact.** A per-region `device` clause selects the target device for that offload region. `LIBOMPTARGET_DEVICETYPE=[ CPU | GPU ]` selects device type; if unavailable it errors only with `OMP_TARGET_OFFLOAD=mandatory`, else falls back to the initial device. Compile: `-fiopenmp -fopenmp-targets=spir64` (`icx`/`ifx`); see §17.

### Device selection

**Rule.** `ONEAPI_DEVICE_SELECTOR` limits the runtimes, device types and device IDs usable by the DPC++ runtime to a subset of all combinations; token form `backend:type:id` (e.g. `opencl:gpu:1`, `level_zero:gpu:0`, `host:host:0`). IDs match those returned by the SYCL API, `clinfo` or `sycl-ls` (numbering starts at 0) and are unrelated to device type or runtime. Requesting a filtered-out device via a programmatic special selector (like `gpu_selector`) raises an exception. Full syntax: `sycl/doc/EnvironmentVariables.md` on GitHub. Selectors: §01.

**Gotcha.** That exception can be thrown when an ahead-of-time (AOT) compiled binary runs on a platform lacking the specified device type.

**Fact.** `sycl-ls` enumerates the devices available in the system; run it before any SYCL/DPC++ program to confirm configuration. It prints the `ONEAPI_DEVICE_SELECTOR` string as a prefix of each device listing. Output format: `[ONEAPI_DEVICE_SELECTOR] Platform_name, Device_name, Device_version [driver_version]`.

```bash
$ sycl-ls
[opencl:gpu:1] Intel OpenCL HD Graphics, Intel UHD Graphics 630 [0x3e92] 3.0 [21.37.20939]
[level_zero:gpu:0] Intel oneAPI Level Zero, Intel UHD Graphics 630 [0x3e92] 1.1 [1.2.20939]
[host:host:0] SYCL host platform, SYCL host device 1.2 [1.2]
```

### SYCL execution and memory hierarchy

Work-items → work-groups → sub-groups; execution model, queues, contexts and kernels: §01, §03; USM, buffers/accessors and memory scopes: §02.

### Key gotchas

- Handle `sycl::exception` from `gpu_selector`/`cpu_selector`/`accelerator_selector`: the requested device type may be absent, including in AOT binaries.
- `ONEAPI_DEVICE_SELECTOR` filtering out a device makes a programmatic selector for it throw; keep filter and code in agreement.
- `sycl-ls` IDs are neither type-bound nor runtime-bound — never assume `:1` is a GPU.
- `SUBDEVICE` is ignored under `ZE_FLAT_DEVICE_HIERARCHY=FLAT|COMBINED`, `LIBOMPTARGET_DEVICES=SUBDEVICE/SUBSUBDEVICE`, or `ONEAPI_DEVICE_SELECTOR`.
- OpenMP device type unavailable errors only with `OMP_TARGET_OFFLOAD=mandatory`; otherwise it silently falls back to the host.
- OpenMP is not supported for FPGA devices; `target` offload is synchronous unless `nowait` is present.
