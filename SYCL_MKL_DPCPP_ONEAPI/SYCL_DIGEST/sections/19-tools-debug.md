## 19 · Debugging, profiling and performance tuning

Failure surfaces: host code, device code, host↔device communication stack.

### Debug tool matrix

| Tool | Debugs |
|---|---|
| Environment variables | OpenMP/SYCL runtime diagnostics, no rebuild |
| `onetrace` (PTI for GPU) | Level Zero **and** OpenCL; backend errors; host/device profiling |
| `ze_tracer` / `cl_tracer` | Level Zero only / OpenCL only; silent on the other backend |
| Intercept Layer for OpenCL Applications | OpenCL backend; buffer overwrites, leaks, mismatched pointers, runtime error detail (wider than `onetrace`) |
| Intel Distribution for GDB | source-level host+device debugging; CPU/GPU/FPGA-emulation; kernel breakpoints |
| Intel Inspector (HPC Toolkit) | memory + threading errors incl. offload failures; **CPU target only** |
| Intel Advisor | Offload Modeling; GPU Roofline |
| Intel VTune Profiler | host + connected-GPU performance, local or remote |
| SYCL Exception Handler | DPC++ runtime errors surfaced as exceptions |

`clinfo` / `sycl-ls` device IDs are the IDs used by `ONEAPI_DEVICE_SELECTOR` (§17). In-application: compare kernel output vs expected, return intermediates, print from kernels (note thread/SIMD lane).

### Environment variables

| Variable | Effect | Values / example |
|---|---|---|
| `LIBOMPTARGET_DEBUG=<Num>` | OpenMP offload runtime output: runtimes used, device, libraries, alloc/dealloc size+address, copies/mappings, launches (args, SIMD width, group info) | `0` off (default); `1` basic plugin actions; `2` + GPU runtime API args |
| `LIBOMPTARGET_INFO=<Num>` | libomptarget offload info | bitmask `0,1,2,4,8,32` (default `0`): `1` data args on kernel entry; `2` mapped address exists; `4` device pointer map dump on offload failure; `8` mapping-table change; `32` data copied to/from device |
| `LIBOMPTARGET_PLUGIN_PROFILE=<Enable>[,<Unit>]` | offload perf: transfers (read/write), allocation, module build (JIT), per-kernel times | `F` off (default); `T` ms; `T,usec` µs. `<Enable> := 1\|T`; `<Unit> := usec\|unit_usec` (µs default). `export LIBOMPTARGET_PLUGIN_PROFILE=T,usec` |
| `LIBOMPTARGET_PLUGIN=<Name>` | force backend; no other RTL loaded | `LEVEL0`/`LEVEL_ZERO`/`level0`, `OPENCL`/`opencl`, `CUDA`/`cuda`, `X86_64`/`x86_64`, `NIOS2`/`nios2`. Default GPU=`LEVEL0`, CPU/FPGA=`OPENCL`. Level Zero GPU-only |
| `LIBOMPTARGET_PROFILE=<FileName>` | time profile output like Clang `-ftime-trace` | |
| `LIBOMPTARGET_DEVICES=<DeviceKind>` | subdevice exposure | `DEVICE`/`device` (default; `subdevice` clause supported); `SUBDEVICE` (1st level, clause ignored); `SUBSUBDEVICE` (2nd level; single compute slice also needs `CFESingleSliceDispatchCCSMode=1`); `ALL` (deprecated, unsupported on Intel GPU) |
| `LIBOMPTARGET_LEVEL0_MEMORY_POOL=<Option>` | reusable memory pool | `0` or `<MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]`; `<MemType> := all\|device\|host\|shared`; MB. Default `device,1,4,256,host,1,4,256,shared,8,4,256` |
| `LIBOMPTARGET_LEVEL0_STAGING_BUFFER_SIZE=<Num>` | staging buffer KB for two-step host↔device copies; discrete devices | default `16` |
| `LIBOMPTARGET_LEVEL_ZERO_COMMAND_BATCH=<Value>` | command batching per target region | `<Type>[,<Count>]`; `none\|NONE` off (default), `copy\|COPY`, `compute\|COMPUTE` (also disables copy engine); no `<Count>` = all eligible |
| `LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=<Value>` | immediate command list | `1\|T\|t`, `0\|F\|f`, `compute`, `copy`, `all`; default `"all"`; true ≡ `compute` |
| `OMP_TARGET_OFFLOAD` | OpenMP standard offload control | `MANDATORY`; `CPU` |
| `LIBOMPTARGET_DEVICETYPE=cpu` | run OpenMP kernels on CPU device | |
| `LIBOMPTARGET_OPENCL_COMPILATION_OPTIONS` / `LIBOMPTARGET_LEVEL0_COMPILATION_OPTIONS` | per-backend kernel compile options | `"-g -cl-opt-disable"` |
| `ONEAPI_DEVICE_SELECTOR` | limit runtimes/types/device IDs (IDs = `sycl-ls`/`clinfo`/SYCL API order, from 0); a filtered programmatic request (e.g. `gpu_selector`) throws | `opencl:cpu`; `opencl:gpu`; `opencl:gpu:2`; `level_zero:gpu:1`; `opencl:cpu,level_zero`; `host`; default all |
| `SYCL_UR_TRACE` | SYCL runtime debug output | `1` plugins and devices discovered/used; `2` SYCL API calls with arguments and result values; `-1` all tracing; default disabled |
| `ZE_DEBUG` | Level Zero output: APIs called, event information | any value = enabled; default disabled |
| `IGC_ShaderDumpEnable=1` | LLVM/assembly/ISA code from Intel Graphics Compiler → `/tmp/IntelIGC/<application_name>` | default `0` |
| `IGC_DumpToCurrentDir=1` | write those files to current directory instead (use a temp dir) | default `0` |
| `CL_CONFIG_USE_VTUNE=True` / `CL_CONFIG_USE_VECTORIZER=false` | analysis/tracing in JIT compilers/runtimes | |

### Trace the offload process

- **Kernel setup time** (JIT compile + copy) skews benchmarks. OpenMP: `LIBOMPTARGET_PLUGIN_PROFILE=T[,usec]` reports `ModuleBuild`. SYCL: `onetrace`/`ze_tracer` Device Timing + Device Timeline.
- **Buffers**: `LIBOMPTARGET_DEBUG` reports addresses and sizes; `SYCL_UR_TRACE` reports only API calls. Low level: `onetrace`/`ze_tracer` Call Logging (`zeMemAllocDevice` gives buffer size); Device Timeline = append/submit/start/end per activity.
- **Transfer / per-kernel time**: OpenMP reports `DataAlloc`, `DataRead`, `DataWrite` (aggregate only) and `Kernel#…`. SYCL: Device Timing / Device Timeline. OpenCL intercept flags: `CallLoggingElapsedTime`, `DevicePerformanceTiming`, `DevicePerformanceTimeKernelInfoTracking`, `DevicePerformanceTimeLWSTracking`, `DevicePerformanceTimeGWSTracking`, `ChromePerformanceTiming`, `ChromePerformanceTimingInStages`, `BuildLogging`, `KernelInfoLogging`, `CallLogging`, `HostPerformanceTiming`, `HostPerformanceTimeLogging`, `ChromeCallLogging`.
- **When kernels run / threads created**: breakpoint inside a device kernel (Intel Distribution for GDB); inspect kernel arguments, thread creation/destruction, `info thread`.

### Debug offload: CPU vs GPU

- **Sanity check**: rerun on the other runtime (OpenCL vs Level Zero) or device via `LIBOMPTARGET_PLUGIN` + `OMP_TARGET_OFFLOAD` (OpenMP), `ONEAPI_DEVICE_SELECTOR` (SYCL). Reproduces across runtimes → not the runtime; across devices → not the hardware.
- **CPU**: "host" implementation is native, debuggable like non-offloaded code; CPU OpenCL runs the OpenCL runtime to TBB-parallel code — pointers meaningful, data inspectable. OpenMP host: drop `target`/`device` constructs; `LIBOMPTARGET_PLUGIN=OPENCL` without GPU offload → OpenMP runtime + TBB. SYCL: `ONEAPI_DEVICE_SELECTOR=host` = single-threaded (races/deadlocks); `opencl:cpu` = CPU OpenCL + TBB.
- **GPU**: Intel Distribution for GDB + compatible GPU. Breakpoint at a source line inside the kernel; `step` into functions, `next` over calls; `scheduler-locking step` first.

GDB GPU thread format `<inferior_number>.<thread_number>:<SIMD Lane/s>`; `2.3:[1 3 5 7]` = SIMD lanes 1,3,5,7 of thread 3, inferior 2 (host = inferior 1, GPU = inferior 2, auto-created). `info threads` lists ids, stop locations, active SIMD lanes (`-gid` = global thread IDs); `thread 3:4` = thread 3 lane 4, `thread :6` = lane 6 of current thread, `thread 7` = thread 7; the default lane is the previously selected one if still active, else the first active. Apply broadly: `thread apply 2.5:3-5 print element`, `thread apply :3 :5 :6 print element`. `help <CMD>` for command help. SIMD lanes cannot be resumed individually. All-stop (default) resumes all threads; `set scheduler-locking` prevents resuming others; non-stop resumes only the current thread unless `continue -a`.

```bash
# SYCL: -g -O0 for host and JIT kernel; kernel flags taken at link time
icpx -fsycl -g -O0 -c myprogram.cpp
icpx -fsycl -g -O0 myprogram.o

# AOT debug build (dpcpp), target KBL
dpcpp -g -O0 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
  -Xs "-device kbl -internal_options -cl-kernel-debug-enable -options -cl-opt-disable" myprogram.cpp
ocloc compile --help     # available GPU device options

# OpenMP (kernel flags taken at link time)
icpx -fiopenmp -O0 -fopenmp-targets=spir64 -c -g myprogram.cpp
icpx -fiopenmp -O0 -fopenmp-targets=spir64 -g myprogram.o
export LIBOMPTARGET_OPENCL_COMPILATION_OPTIONS="-g -cl-opt-disable"
export LIBOMPTARGET_LEVEL0_COMPILATION_OPTIONS="-g -cl-opt-disable"
```

### Failure and correctness workflows

**Generic OpenCL error** — `~/clintercept.conf`: `SimpleDumpProgramSource=1`, `CallLogging=1`, `LogToFile=1`, `BuildLogging=1`, `ErrorLogging=1`, `USMChecking=1` (+ `KernelInfoLogging=1`, `DevicePerformanceTiming=1`, `DevicePerformanceTimeLWSTracking=1`, `DevicePerformanceTimeGWSTracking=1` for profiling). Run `<OCL_Intercept_Install_Dir>/bin/cliloader/cliloader -d ./<app_name> <app_args>`; outputs `~CLIntercept_Dump/<app_name>/clintercept_report.txt`, `clintercept_log.txt`. Build with `cmake -DENABLE_CLIPROF=TRUE -DENABLE_CLILOADER=TRUE`.

**Level Zero root cause** — `./onetrace -c ./<app_name> <app_args> [2> log.txt]` (`ze_tracer` identical; `cl_tracer` for OpenCL). Log pairs `>>>> [ts] zeKernelSetArgumentValue: hKernel = … argIndex = 1 pArgValue = …` / `<<<< [ts] -> ZE_RESULT_ERROR_INVALID_ARGUMENT (2013265924)`; localize via call name, result code, `argIndex`, `pArgValue`, `hKernel` (→ kernel name).

**JIT compilation failures** (bad offload language use) exit with an error: try AOT; else OpenCL backend + `SimpleDumpProgramSource` + `BuildLogging`.

**Escalation**: Level Zero error → try OpenCL backend; if it works, report against Level Zero. If it reproduces, try OpenCL CPU: OpenMP `OMP_TARGET_OFFLOAD=CPU`, SYCL `ONEAPI_DEVICE_SELECTOR=opencl:cpu`.

**Lambda capture trap** (common in migrated CUDA kernels): in a `[=]` `parallel_for` lambda, `factor` means `this->factor` and `data` means `this->data`, so `this` is captured; the non-trivial structure copy produces an illegal-arguments error at `kernel(data, b, factor, LEN, item_ct1)`. Fix: `auto data2 = data; auto factor2 = factor;` then `kernel(data2, b, factor2, LEN, item_ct1)` — works if the class is in USM (if only `data` is USM it still crashes). Re-declaring the variables in local scope with the same names also works. **SYCL exception handler**: see §12.

**Correctness**: run the host and offload implementations, capture every kernel's inputs/outputs, compare within epsilon (device order/precision can differ in the last digit or two), then debug the offending kernel with Intel Distribution for GDB. `printf` works inside kernels (SYCL, C++ OpenMP offload; `print *, ...` for Fortran) — print thread and SIMD lane and synchronize so printed state is consistent. SYCL: use the `stream` class, or:

```cpp
#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
  static const CL_CONSTANT char _format[] = format; \
  sycl::ONEAPI::experimental::printf(_format, ## __VA_ARGS__); }
// usage: PRINTF("My integer variable:%d\n", (int) x);
```

### Intel Inspector

Cannot analyze offload code talking to GPU/FPGA; force kernels onto a CPU target.

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:cpu    # SYCL (or cl::sycl::queue Queue(cl::sycl::cpu_selector{});)
export OMP_TARGET_OFFLOAD=MANDATORY         # OpenMP
export LIBOMPTARGET_DEVICETYPE=cpu
export CL_CONFIG_USE_VTUNE=True             # analysis/tracing in JIT compilers/runtimes
export CL_CONFIG_USE_VECTORIZER=false
inspxe-cl -c mi3 -- <app> [app_args]        # memory
inspxe-cl -c ti3 -- <app> [app_args]        # threading
inspxe-cl -report=problems -report-all      # report
```

Flags bad pointers passed to the OpenCL backend, or a wrong pointer from the wrong thread.

### Profiling

- **Intel VTune Profiler**: GPU Offload view — GPU hotspots incl. transfer time per kernel; GPU Compute/Media Hotspots view — `Dynamic Instruction Count` micro analysis; transfer vs compute over time; whether a kernel has enough work.
- **Intel Advisor**: Offload Modeling — recommends offloadable host OpenMP parts, models target GPUs, estimates speedup/bottlenecks/transfer cost; GPU Roofline — per-kernel memory-subsystem and compute-unit utilization. Already-offloading apps: point the environment at the OpenCL device on the CPU.
- **Timelines**: VTune, or `onetrace`/`ze_tracer`/`cl_tracer`/Intercept Layer (no graphical timeline — script it).

```bash
# VTune collect, Level Zero backend
vtune -collect-with runss -knob enable-gpu-level-zero=true finalization-mode=none -app-working-dir <app_working_dir> <app>
# VTune collect, OpenCL backend
vtune -collect-with runss -knob collect-programming-api=true finalization-mode=none -r <result_dir_name> -app-working-dir <app_working_dir> <app>
# VTune report
vtune --report hotspots --group-by=source-computing-task --sort-desc="Total Time" -r <result_dir_name>
```

### Performance tuning cycle

The baseline/verify cycle and the general, loop, memory and SYCL-specific optimization tasks: see §21 `### Intel GPU rules of thumb` and `### Top 15 mistakes`. Use a profiling tool such as Intel VTune Profiler to find bottlenecks.

### Library compatibility and mixed toolchains

Version compatibility (headers/libraries same release; deployed apps not broken by new drivers/libs/compilers; backward but not forward compatibility): see §20 `### oneAPI library compatibility`.

- DPC++ SYCL uses the TBB runtime for device code on the CPU → OpenMP + SYCL on a CPU can oversubscribe threads; profile to confirm.
- OpenMP directives cannot appear inside SYCL device kernels; SYCL cannot appear inside OpenMP target regions (SYCL in host-CPU OpenMP code is fine). OpenMP and SYCL device parts cannot cross-depend — linked into separate binaries inside the fat binary. Direct OpenMP↔SYCL runtime interaction unsupported (OpenMP device memory object used in SYCL = unspecified behavior).
- OpenMP not supported for FPGA devices.
- Mixed build `icpx -fsycl -fiopenmp -fopenmp-targets=spir64 offloadOmp_dpcpp.cpp`; host-only OpenMP `icpx -fsycl -fiopenmp omp_dpcpp.cpp`.

### Key gotchas

- `ze_tracer` / `cl_tracer` / Intercept Layer print nothing when paired with the other backend.
- Intel Inspector cannot analyze GPU/FPGA offload; force CPU with `ONEAPI_DEVICE_SELECTOR=opencl:cpu`.
- `ONEAPI_DEVICE_SELECTOR` IDs are enumeration order, not type-based; a filtered `gpu_selector` request throws.
- Kernel debug flags are taken at link time — pass `-g -O0` when linking too.
- Unhandled asynchronous SYCL errors terminate; a print-only handler can let incorrect results continue.
- Level Zero is GPU-only; `LIBOMPTARGET_PLUGIN=LEVEL0` is invalid for CPU/FPGA offload.
- `LIBOMPTARGET_DEVICES=ALL` is unsupported on Intel GPU and deprecated.
