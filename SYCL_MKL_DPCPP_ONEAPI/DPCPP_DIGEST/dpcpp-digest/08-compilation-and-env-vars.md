---
chunk: 08-compilation-and-env-vars
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 622-665
covers: Compilation overview/defaults; all compile-time and runtime environment variables; linker options; alternate tools; config/response files; Linux global symbols and visibility; saving compiler info; linking debug info; AOT compilation (CPU/GPU/non-Intel GPUs); device offload compilation; third-party SYCL host compilers; Ccache*
---

# Compilation, Environment Variables, Linking, and AOT Compilation

> **Scope.** How the Intel® oneAPI DPC++/C++ Compiler drives compilation and linking: phases,
> defaults, and customization via configuration files, response files, and environment variables.
> Complete compile-time and runtime environment-variable tables (OpenMP*, SYCL*, libomptarget,
> OpenCL™, caches). Linker options, alternate tools, Linux* symbol visibility, compiler-info and
> debug-info handling, Ahead of Time (AOT) compilation, offload compilation caveats, third-party
> SYCL host compilers, Ccache*.

## Key facts

- Phases: **Preprocessing, Semantic parsing, Optimization, Code generation, Linking**. The compiler does the first four, then invokes the linker — `ld` (Linux), `link` (Windows).
- Defaults: `-O2`; floating point model `fast` (`-fp-model=fast`); C17; C++17; libstdc++ with system headers/libs (Linux) or MSVC headers/libs (Windows); SVML and specific interfaces enabled to call into the Intel libirc library.
- `-c` (Linux) / `/c` (Windows) prevents default linking; pass the object on the command line to build the final binary.
- Calling the linker directly on Linux requires explicitly passing the needed system and Intel libraries.
- Path lists use a platform `PATH_SEPARATOR`: colon (`:`) Linux, semicolon (`;`) Windows.
- Some variables work on Intel® and non-Intel microprocessors but may optimize more on Intel microprocessors.
- `Any(*)` = effective when set to any non-null value.
- **Since version 2024.0** options given with the Clang `-mllvm` flag are no longer passed through to linker option processing; use `-Wl`.
- AOT binaries are device-specific and will not run elsewhere; detect the target at runtime and error if absent (async exception handler recommended).
- Syntactic/semantic errors go to `stderr` with the erroneous source line; they suppress object code for that module and prevent linking, but parsing continues to find other errors. Examples: `expected ';' at end of declaration`, `unexpected type name 'b': expected expression`. The p622 `weak` pragma is GCC-compatible.

## Option / API quick table

| name | purpose |
|---|---|
| `-c` / `/c`; `-Ldirectory`; `-shared`, `-static`, `-static-libgcc`, `-static-intel`, `-shared-libgcc`, `-shared-intel` | Link control (Linux) |
| `-Qoption,tool,list` / `/Qoption,tool,optlist`; `-Wl,optlist`; `-Xlinker val`; `/link` | Pass options to another tool/linker (`Qoption` not for SYCL) |
| `-g`, `-gsplit-dwarf` (Linux); `/Z7`, `/debug`, `/Zi` (Windows) | Debug information |
| `-fopenmp-targets=`, `-Xopenmp-target-backend`; `-fsycl-targets=`, `-Xsycl-target-backend`, `-Xs` | AOT device targets / backend options |
| `-fopenmp-device-code-split=`; `-fsycl-device-code-split=` | Code split (`per_kernel`; SYCL also `per_source`, `off`, `auto`=default) |
| `-fsycl-host-compiler`, `-fsycl-host-compiler-options`; `-fsycl-help=gen`, `-fsycl-help=x86_64` | SYCL host compiler; list backend AOT options |

## Compilation

### Compilation Overview

Customize compilation with **Configuration Files**, **Environment variables**, or **Response Files**; additional include directories can also be added (Specify Compiler Files). Sources are processed in the phase order above. *Preprocessing:* set system/user header locations, macros, stop after preprocessing, send preprocessed output to `stdout`, or use your own preprocessor to produce a preprocessed file to pass in. *Compiling:* options are optional but control code generation, optimization, output file (type/name/location), linking properties, executable size and speed. *Linking:* invoke the linker through the compiler (default) or call it directly.

## Supported Environment Variables

### Compiler Compile-Time Environment Variables

`(W)` = Windows*, `(L)` = Linux.

| Variable | Description |
|---|---|
| `CL` (W) | Files/options used most often. Cannot contain an equal sign; use `#`: `SET CL=/Dtest#100`. |
| `_CL_` (W) | Listed with `CL` (same equal-sign note). |
| `ICXCFG` | Configuration file for `icx` invocations, instead of the default configuration file. |
| `ICPXCFG` (L) | Configuration file for `icpx` invocations, instead of the default. |
| `__INTEL_PRE_CFLAGS` | Compiler options added at the **prefix** position; extension of the `icx.cfg` facility. |
| `__INTEL_POST_CFLAGS` | Compiler options added at the **suffix** position. Order: `icx <PRE flags> <flags from configuration file> <flags from the compiler invocation> <POST flags>`. |
| `PATH` | Directories searched for binary executables; on Windows also affects DLL search. |
| `TMP`, `TMPDIR`, `TEMP` | Temporary-file location; search order `TMP`, `TMPDIR`, `TEMP`. If none specified/writeable/found: `/tmp` (Linux) or current directory (Windows). On Windows cannot be set from Visual Studio*. |
| `LD_LIBRARY_PATH` (L) | Paths added to dirs searched for shared objects (`.so`). |
| `INCLUDE` (W) | Paths added to the include path list for C and C++ compilations. |
| `LIB` (W) | Paths added to dirs for all libraries used by compiler and linker. |
| `CPATH` | Paths added to the **user** include path for C and C++. |
| `C_INCLUDE_PATH` | Paths added to the **system** include path for C compilations. |
| `CPLUS_INCLUDE_PATH` | Paths added to the **system** include path for C++ compilations. |
| `DEPENDENCIES_OUTPUT` (L) | Make dependency output based on non-system header files; system headers ignored. |
| `GCC_EXEC_PREFIX` (L) | Alternative names for the linker (`ld`) and assembler (`as`). |
| `LIBRARY_PATH` (L) | Paths added to dirs for all libraries used by compiler and linker. |
| `SUNPRO_DEPENDENCIES` (L) | As `DEPENDENCIES_OUTPUT`, but system header files are **not** ignored. |

Path-list variables above use `PATH_SEPARATOR` delimiters. **Default configuration files:** `icx.cfg` (Linux* or Windows) or `icpx.cfg` (Linux), in the compiler executable's directory; use `ICXCFG` (Linux, Windows) or `ICPXCFG` (Linux) to point elsewhere. The driver warns about an option overridden by an environment variable only with `-w3` (Linux) or `/W5` (Windows).

### Compiler Runtime Environment Variables

`Any(*)` = effective when set to any non-null value.

| Variable | Default | Description / values |
|---|---|---|
| `CL_CONFIG_CPU_EXPENSIVE_MEM_OPT` | `0` | Bitmap of expensive memory optimizations (more JIT time, perf gain). Only LSB available: `0` = OpenCL™ address space alias analysis. |
| `CL_CONFIG_CPU_FORCE_LOCAL_MEM_SIZE` | `32KB` | Forces CPU `CL_DEVICE_LOCAL_MEM_SIZE`; include units (`8MB`, `8192KB`, `8388608B`). Host app needs sufficient stack; recommendation: stack size = 2× local memory size (covers app + OpenCL™ Runtime overheads). |
| `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE` | `32KB` | Forces CPU `CL_DEVICE_PRIVATE_MEM_SIZE`; unit required (`8MB`, `8192KB`, `8388608B`). Host app needs sufficient stack. |
| `CL_CONFIG_CPU_STREAMING_ALWAYS` | `FALSE` | Use of non-temporal instructions. |
| `DPCPP_CPU_CU_AFFINITY` | Not set | CPU thread affinity. `Close`: pin threads successively through available cores. `Spread`: spread threads over available cores. `Master`: threads share the master's cores; if `DPCPP_CPU_CU_AFFINITY` is set the master thread is pinned too, else not pinned. Similar to OpenMP* `OMP_PROC_BIND`. |
| `DPCPP_CPU_NUM_CUS` | Not set; determined by Intel® oneAPI Threading Building Blocks (oneTBB) | Threads for kernel execution. Max should be the hardware-thread count to avoid oversubscription; `1` runs all workgroups sequentially in one thread (debugging). Similar to `OMP_NUM_THREADS`. |
| `DPCPP_CPU_PLACES` | `cores` | Places affinities are set to: `{ sockets \| numa_domains \| cores \| threads }`. Similar to `OMP_PLACES`. With `numa_domains` the oneTBB NUMA API is used (analogous to `OMP_PLACES=numa_domains` in the OpenMP 5.1 Specification): the oneTBB task arena binds to the NUMA node and SYCL `nd_range` is uniformly distributed to task arenas. Use with `DPCPP_CPU_CU_AFFINITY`. |
| `DPCPP_CPU_SCHEDULE` | `dynamic` | Work-group scheduling (oneTBB with the OpenCL™ CPU driver); selects the oneTBB petitioner. `dynamic`: `auto_partitioner`, enough splitting to balance load. `affinity`: `affinity_partitioner`, improves `auto_partitioner` cache affinity by mapping subranges to worker threads. `static`: `static_partitioner`, distributes range iterations as uniformly as possible; relies on grain-size for chunking. Grain-size default `1` = every work-group independently executable. |
| `GOMP_CPU_AFFINITY` (L) | Affinity disabled | GNU extension recognized by the Intel OpenMP compatibility library: list of OS processor IDs. Set before the first parallel region or certain API calls including `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls. |
| `GOMP_STACKSIZE` (L) | See `OMP_STACKSIZE` | GNU extension recognized by the Intel OpenMP compatibility library; same as `OMP_STACKSIZE`. `KMP_STACKSIZE` > `GOMP_STACKSIZE` > `OMP_STACKSIZE`. |
| `KMP_AFFINITY` | `noverbose,warnings,noreset,respect,granularity=core,none` (Linux); `noverbose,warnings,noreset,norespect,granularity=group,compact,0,0` (Windows with multiple processor groups) | Binds threads to physical processing units. Set before the first parallel region or certain API calls including `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls. On Windows with multiple processor groups, `norespect` is assumed when the process affinity mask equals a single processor group (the Windows default); otherwise `respect`. |
| `KMP_ALL_THREADS` | No enforced limit | Limits simultaneously-executing threads. If reached and another native OS thread hits OpenMP calls/constructs the program may abort with an error; if reached at parallel-region start, a one-time warning may report a reduced team but execution continues. Only for `[q or Q]openmp`-compiled programs. |
| `KMP_BLOCKTIME` | `0` ms with Intel® Hybrid Technology detected; `200` ms otherwise | Busy-wait time after a parallel region before sleeping. Suffixes `us`/`ms`; no suffix = milliseconds. `infinite` = unlimited. Related: `KMP_LIBRARY`. |
| `KMP_CPUINFO_FILE` | None | Alternate machine-topology file; must use the `/proc/cpuinfo` format. |
| `KMP_DETERMINISTIC_REDUCTION` | `FALSE` | `TRUE`/`FALSE`: fixed ordering of reduction operations for the reduction clause, giving consistent floating-point reduction results run to run (identical round-off errors). Requires `-fp-model precise` (Linux) / `-fp:precise` (Windows). |
| `KMP_DYNAMIC_MODE` | `tcm` when the Thread Composability Manager library is available; else `thread_limit` | Thread-count method for a parallel region when `OMP_DYNAMIC=TRUE`: `tcm` (Thread Composability Manager); `load_balance` (avoid more threads than available execution units); `thread_limit` (avoid more than total execution units). |
| `KMP_HIDDEN_HELPER_AFFINITY` (L) | `noverbose,warnings,granularity=core,none` | Binds hidden helper threads to physical processing units. Set before the first hidden helper task, parallel region, or certain API calls including `omp_get_max_threads()`, `omp_get_num_procs()`, and any affinity API calls. Syntax as `KMP_AFFINITY`, except `reset`/`noreset` and `respect`/`norespect` are unavailable. |
| `KMP_HOT_TEAMS_MAX_LEVEL` | `1` | Max nested level of hot teams (a hot team stays ready for reuse by later parallel regions; a cold team is freed after each parallel region, its threads going to a common pool). For `2` and above, nested parallelism should be enabled. |
| `KMP_HOT_TEAMS_MODE` | `0` | Behavior when a hot team's thread count is reduced. `0`: extra threads freed to the common pool. `1`: kept in the team in reserve for faster reuse. |
| `KMP_HW_SUBSET` | If omitted, all available hardware resources | Hardware-resource subset — see sub-section below. |
| `KMP_INHERIT_FP_CONTROL` | `TRUE` | `TRUE`/`FALSE`: copy the primary thread's floating-point control settings to OpenMP worker threads at the start of each parallel region. |
| `KMP_LIBRARY` | `throughput` | OpenMP runtime execution mode: `serial`, `turnaround`, or `throughput`. |
| `KMP_PLACE_THREADS` | n/a | **Deprecated**; use `KMP_HW_SUBSET`. |
| `KMP_SETTINGS` | `FALSE` | `TRUE`/`FALSE`: print OpenMP runtime environment variables during execution (user-defined settings plus effective runtime values). |
| `KMP_STACKSIZE` | `4m` | Bytes per OpenMP thread private stack; recommended `16m`. Suffixes `B`, `G`, `K`, `M`, `T`; no suffix assumes `K`. Overrides `GOMP_STACKSIZE`, which overrides `OMP_STACKSIZE`. |
| `KMP_TOPOLOGY_METHOD` | `all` | Topology modeling method: `all` (most appropriate); `cpuid_leaf31`/`cpuid_leaf11`/`cpuid_leaf4` (decode APIC IDs per that `cpuid` leaf); `cpuinfo` (parse `/proc/cpuinfo`; Linux only — uses `KMP_CPUINFO_FILE` if set); `group` (Windows only; 2-level map, level 0 = processors in a group, level 1 = groups; **deprecated, will be removed, use `all`**); `flat` (flat/linear processor list); `hwloc` (as the hwloc library; most detailed — NUMA nodes, packages, cores, hardware threads, caches, Windows processor groups). |
| `KMP_USER_LEVEL_MWAIT` | `FALSE` | `TRUE`/`FALSE`: use user-level `mwait` instead of sleeping waiting threads if available (ring3 or WAITPKG). |
| `KMP_VERSION` | `FALSE` | `TRUE`/`FALSE`: print OpenMP runtime version info during execution. |
| `KMP_WARNINGS` | `TRUE` | `TRUE`/`FALSE`: show OpenMP runtime warnings during execution. |
| `LIBOMPTARGET_DEBUG` | `0` | Offload-runtime debug output. `0`: off. `1`: basic plugin debug (device detection, kernel compilation, memory copies, kernel invocations, other plugin actions). `2`: also which GPU runtime API functions are called with which arguments. |
| `LIBOMPTARGET_DEVICES` | `DEVICE` | Subdevice exposure. `DEVICE`/`device`: only top-level devices; `subdevice` clause supported. `SUBDEVICE`/`subdevice`: only first-level subdevices; aborts on the `subdevice` clause. `SUBSUBDEVICE`/`subsubdevice`: only second-level; aborts on the `subdevice` clause. |
| `LIBOMPTARGET_DEVICETYPE` | `GPU` | Offload device type: `GPU`/`gpu`, `CPU`/`cpu`. Plugins: Level Zero only GPU; OpenCL™ both; X86_64 ignores this variable. |
| `LIBOMPTARGET_DYNAMIC_MEMORY_SIZE` | `1` | MB preallocated for in-kernel device `malloc`. Values: non-negative integer. |
| `LIBOMPTARGET_INFO` | `0` | Basic offload info. `0`: off. `1`: print all data arguments on entering an OpenMP device kernel. `2`: mapped address already exists in the device mapping table. `4`: dump the device pointer map if offloading fails. `8`: an entry changed in the device mapping table. `32`: data copied to/from the device. |
| `LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE` | `async` | How each command in a target region executes when immediate command lists are fully enabled via `LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=all`. No effect on integrated devices. `sync`: host waits for the submitted command. `async`: host does not wait, synchronization later when required. `async_ordered`: as `async`, execution ordered. |
| `LIBOMPTARGET_LEVEL_ZERO_COMPILATION_OPTIONS` | n/a | Extra build options for native target program binaries. Values: valid Level Zero module build options. |
| `LIBOMPTARGET_LEVEL_ZERO_DEFAULT_TARGET_MEM` | `DEVICE` | Memory type returned by `omp_target_alloc`. `DEVICE`/`device`: device-owned, explicit data movement. `SHARED`/`shared`: shared ownership, implicit movement. `HOST`/`host`: host-owned, implicit movement. |
| `LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL` | Equivalent to `device,1,4,256,host,1,4,256,shared,8,4,256` | Memory pool configuration — grammar/examples below. |
| `LIBOMPTARGET_LEVEL_ZERO_STAGING_BUFFER_SIZE` | `16` | Staging buffer KB; temporary storage for two-step host↔device copies; discrete devices only. Values: non-negative integers; `0` disables. |
| `LIBOMPTARGET_LEVEL_ZERO_USE_COPY_ENGINE` | `all` | Copy engines for transfers when supported. `0`/`F`/`f`: disable. `main`: main engines only. `link`: link engines only. `all`: all engines. |
| `LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST` | `all` for Xe HPC devices; `0` otherwise | Immediate command list for computation and/or copy. `0`/`F`/`f`: disable. `compute`: computation only. `copy`: copy only. `all`: both. |
| `LIBOMPTARGET_OPENCL_COMPILATION_OPTIONS` | n/a | Extra compilation options for target programs built from SPIR-V* target images. Values: valid OpenCL™ compilation options. |
| `LIBOMPTARGET_OPENCL_LINKING_OPTIONS` | n/a | Extra linking options when linking target programs. Values: valid OpenCL™ linking options. |
| `LIBOMPTARGET_PLUGIN` | `LEVEL_ZERO` | Offload plugin. `LEVEL_ZERO`/`LEVEL0`/`level_zero`/`level0`: Intel® oneAPI Level Zero (Level Zero). `OPENCL`/`opencl`: OpenCL™. `X86_64`/`x86_64`: X86_64 plugin. |
| `LIBOMPTARGET_PLUGIN_PROFILE` | Disabled | Basic plugin profiling, printed when the program finishes. Syntax `<Value>[,usec]`, `<Value>=1 \| T \| t`; microseconds if `,usec` appended, else milliseconds. |
| `OCL_ICD_ENABLE_TRACE` | `FALSE` | `TRUE`/`FALSE`: trace mechanism in the OpenCL™ Installable Client Driver (ICD) loader. Values: `OCL_ICD_ENABLE_TRACE=T`, `=1`, `=True`. |
| `OMP_AFFINITY_FORMAT` | `'OMP: pid %P tid %i thread %n bound to OS proc set {%A}'` | Format for displayed affinity info. Fields: `%t`/`%{team_num}`=`omp_get_team_num()`; `%T`/`%{num_teams}`=`omp_get_num_teams()`; `%L`/`%{nesting_level}`=`omp_get_level()`; `%n`/`%{thread_num}`=`omp_get_thread_num()`; `%a`/`%{ancestor_tnum}`=`omp_get_ancestor_thread_num(omp_get_level() – 1)`; `%H`/`%{host}`=host device name; `%P`/`%{process_id}`=process ID; `%i`/`%{native_thread_id}`=native thread ID; `%A`/`%{thread_affinity}`=processor IDs the thread may run on. |
| `OMP_ALLOCATOR` | `omp_default_mem_alloc` | Default allocator for allocation calls/directives/clauses without one. Syntax `<PredefinedMemAllocator> \| <PredefinedMemSpace> \| <PredefinedMemSpace>:<Traits>`. Supported: `omp_default_mem_alloc`, `omp_default_mem_space`; with libmemkind + system support: `omp_high_bw_mem_alloc`, `omp_high_bw_mem_space`, `omp_large_cap_mem_alloc`, `omp_large_cap_mem_space`. |
| `OMP_CANCELLATION` | `FALSE` | Cancellation of the innermost enclosing region of the given type. `TRUE` enables the `cancel` construct and cancellation points; `FALSE` disables cancellation (construct and points ignored), e.g. `OMP_CANCELLATION=TRUE`. With cancellation enabled, internal barrier code should repeatedly check the global flag; a thread observing cancellation leaves the barrier prematurely with return value `1` (may wake others), otherwise `0`. |
| `OMP_DEBUG` | `DISABLED` | Whether the runtime collects information an OMPD library may need for tool support: `ENABLED`/`DISABLED`. Host OpenMP runtime only. |
| `OMP_DEFAULT_DEVICE` | n/a | Device used in a target region; overridden by `omp_set_default_device` or a `device` clause. If no device with that number exists the code runs on the host; if unset, device `0` is used. |
| `OMP_DISPLAY_AFFINITY` | `FALSE` | Display formatted affinity info for all OpenMP threads on entering the first parallel region and whenever information covered by `OMP_AFFINITY_FORMAT` specifiers changes. `TRUE`/`FALSE`. |
| `OMP_DISPLAY_ENV` | `FALSE` | `TRUE`/`FALSE`: print to `stderr` the OpenMP version number and environment-variable values. Values `TRUE`, `FALSE`, or `VERBOSE`, e.g. `OMP_DISPLAY_ENV=TRUE`. |
| `OMP_DYNAMIC` | `TRUE` when `TCM_ENABLE=1` and the Thread Composability Manager library is available; else `FALSE` | `TRUE`/`FALSE`: dynamic adjustment of thread count, e.g. `OMP_DYNAMIC=TRUE`. |
| `OMP_MAX_ACTIVE_LEVELS` | `1` | Max levels of parallel nesting. Values: non-negative integer. |
| `OMP_MAX_TASK_PRIORITY` | `0` | Initial value controlling task priorities. Values: non-negative integer. |
| `OMP_NESTED` | n/a | **Deprecated**; use `OMP_MAX_ACTIVE_LEVELS`. |
| `OMP_NUM_TEAMS` | `1` | Max teams created by a `teams` construct (sets `nteams-var` ICV). Values: positive integer. |
| `OMP_NUM_THREADS` | Number of processors visible to the OS running the program | Max threads for parallel regions absent another value. Value: single integer or numeric abstract name (all levels), or comma-separated positive integers and/or numeric abstract names (one per nesting level). First position = outer-most level; a value may be omitted but the comma must remain; omitting the first value uses the normal outer-level default, omitting another level inherits the previous level. See the `OMP_PLACES` note on numeric abstract names. Applies to `[q or Q]openmp`. Syntax `OMP_NUM_THREADS=value[,value]*`. |
| `OMP_PLACES` | n/a | Explicit ordered place list — see sub-section below. |
| `OMP_PROC_BIND` | `FALSE` | Affinity policy for parallel regions at the corresponding nesting level; `TRUE`/`FALSE` thread binding. Enabled = `KMP_AFFINITY=scatter`; disabled = `KMP_AFFINITY=none`. Values: `TRUE`, `FALSE`, or comma-separated `PRIMARY`, `MASTER` (deprecated), `CLOSE`, `SPREAD`. If `FALSE`, threads may move between places, affinity is disabled, and `proc_bind` clauses are ignored. Otherwise threads should not move, affinity is enabled, and the initial thread binds to the first place. `PRIMARY`: all threads share the primary thread's place. `CLOSE`: successive places near the primary thread's binding. `SPREAD`: primary thread's partition subdivided; threads bound to single place successive sub-partitions. Note: `KMP_AFFINITY` > (`GOMP_CPU_AFFINITY`, `OMP_PROC_BIND`); `GOMP_CPU_AFFINITY` > `OMP_PROC_BIND`. |
| `OMP_SCHEDULE` | `static`, no chunk size specified | Runtime schedule type and optional chunk size. Syntax `OMP_SCHEDULE="[modifier:]kind[,chunk_size]"`; `modifier` = `monotonic` or `nonmonotonic`; `kind` = `static`, `dynamic`, `guided`, or `auto`; `chunk_size` = positive integer. |
| `OMP_STACKSIZE` | `4M` | Bytes per OpenMP thread private stack; recommended `16M`. Suffixes `B`, `G`, `K`, `M`, `T`; no suffix assumes `K`. Does not affect native OS threads created by the user program or the thread running the sequential part. `kmp_{set,get}_stacksize_s()` set/retrieve it; `kmp_set_stacksize_s()` must be called from the sequential part before the first parallel region or it has no effect. `KMP_STACKSIZE` overrides `OMP_STACKSIZE`. Syntax `OMP_STACKSIZE=value`. |
| `OMP_TARGET_OFFLOAD` | `DEFAULT` | `MANDATORY`: terminate if a device construct or device memory routine is encountered and the device is unavailable/unsupported. `DISABLED`: no target offloading, execute on host. `DEFAULT`: offload if the device is available and supported. |
| `OMP_TEAMS_THREAD_LIMIT` | `<NumberOfProcessors> / <nteams-var ICV>` | Max OpenMP threads per team created by a `teams` construct. Values: positive integer or numeric abstract name (see `OMP_PLACES` note). |
| `OMP_THREAD_LIMIT` | No enforced limit | Limits simultaneously-executing threads in an OpenMP contention group. Value: positive integer or numeric abstract name (see `OMP_PLACES` note). If reached and another native OS thread hits OpenMP calls/constructs the program can abort with an error; at parallel-region start a one-time warning may report a reduced team but execution continues. Only for `[q or Q]openmp`. `omp_get_thread_limit()` returns it. `KMP_ALL_THREADS` overrides `OMP_THREAD_LIMIT`. Syntax `OMP_THREAD_LIMIT=value`. |
| `OMP_TOOL` | `ENABLED` | Whether the runtime tries to register a first-party tool using the OpenMP Tool (OMPT) interface: `ENABLED`/`DISABLED`. Host OpenMP runtime only. |
| `OMP_TOOL_LIBRARIES` | Empty | First-party OMPT tool locations: dynamically-loadable library names with the OS-specific path separator. Host OpenMP runtime only. |
| `OMP_TOOL_VERBOSE_INIT` | `DISABLED` | Verbose OMPT tool-registration logging. `DISABLED`: none. `STDOUT`: stdout. `STDERR`: stderr. `File_Name`: that location. Host OpenMP runtime only. |
| `OMP_WAIT_POLICY` | `PASSIVE` | Waiting threads spin (`ACTIVE`) or yield (`PASSIVE`). `ACTIVE` alias for `KMP_LIBRARY=turnaround`; `PASSIVE` alias for `KMP_LIBRARY=throughput`. Syntax `OMP_WAIT_POLICY=value`. |
| `ONEAPI_DEVICE_SELECTOR` | See `ONEAPI_DEVICE_SELECTOR` | Limits devices for a running SYCL-using application; limits by type (GPUs, accelerators) or backend (Level Zero, OpenCL™). Replaces `SYCL_DEVICE_FILTER`. Syntax shared with OpenMP; also chooses sub-devices. |
| `SYCL_CACHE_DIR` | Path | Persistent cache root. Defaults `$XDG_CACHE_HOME/libsycl_cache` (Linux), `%AppData%\libsycl_cache` (Windows); if `XDG_CACHE_HOME` is unset, `$HOME/.cache/libsycl_cache`. If none are set, the persistent cache is disabled. |
| `SYCL_CACHE_DISABLE_PERSISTENT` (**deprecated**) | Any(*) | Has no effect. |
| `SYCL_CACHE_EVICTION_DISABLE` | Any(*) | Setting it turns cache eviction off. |
| `SYCL_CACHE_IN_MEM` | `1` | Enable (`1`)/disable (`0`) in-memory caching of device compiled code; caches and reuses JIT-compiled binaries. |
| `SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE` | `1GB` | Max cached device image size in bytes; too-big kernels may overload disk fast. |
| `SYCL_CACHE_MAX_SIZE` | Positive integer | Evict when total cached-image size exceeds this many megabytes (default `8 192` = 8 GB). `0` disables size-based eviction. |
| `SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE` | `0` | Min device-code image size in bytes worth caching (disk access may cost more than JIT). `0` caches all images. |
| `SYCL_CACHE_PERSISTENT` | `Off` | Persistent device compiled-code cache: `1` on, `0` off; when on, JIT binaries are cached and reused. Default off. |
| `SYCL_CACHE_THRESHOLD` | Positive integer | Eviction threshold in days (default `7` = 1 week). `0` disables time-based eviction. |
| `SYCL_DEVICE_ALLOWLIST` | See `SYCL_DEVICE_ALLOWLIST` | Filters out non-matching devices. `BackendName`: `host`, `opencl`, `level_zero`, `cuda`. `DeviceType`: `host`, `cpu`, `gpu`, `acc`. `DeviceVendorId`: `uint32_t` hex (`0xXYZW`). `DriverVersion`, `PlatformVersion`, `DeviceName`, `PlatformName`: regular expression. Escape special characters such as parenthesis. Only devices satisfying all values and the RegEx are selected; specify multiple devices with the piping symbol (`\|`). |
| `SYCL_DEVICE_FILTER` (**deprecated**) | `backend:device_type:device_num` | Use `ONEAPI_DEVICE_SELECTOR` instead. |
| `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING` | Any(*) | Disables automatic rounding-up of `parallel_for` invocation ranges. |
| `SYCL_EAGER_INIT` | `0` | Non-zero enables: initialize as much as possible at object construction rather than lazily. May do redundant warmup work but gives fastest execution on later hot/reportable paths; also instructs PI plugins to do the same. |
| `SYCL_ENABLE_DEFAULT_CONTEXTS` | `1` or `0` | Enable (`1`)/disable (`0`) creation of default platform contexts; each platform's default context contains all its devices (Platform Default Contexts extension). Enabled by default on Linux, disabled on Windows. |
| `SYCL_ENABLE_FUSION_CACHING` | `1` | Enable (`1`)/disable (`0`) caching of JIT compilations for kernel fusion; avoids re-running the JIT pipeline when the same kernel sequence is fused repeatedly. |
| `SYCL_ENABLE_PCI` | `1` | `1` obtains the GPU PCI address with the Level Zero backend. Kept for compatibility and **immediately deprecated**. |
| `SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR` | Any(*) | Disable the USM allocator in the Level Zero plugin (requests go directly to the Level Zero runtime). |
| `SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY` | Any(*) | Enable kernels with indirect access and the corresponding deferred release of memory allocations in the Level Zero plugin. |
| `SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE` | See `SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE` | Preferred work-group size of reduction. |
| `SYCL_RT_WARNING_LEVEL` | `0` | Higher = more runtime warnings/performance hints. `0`: none. `1`: performance warnings from device runtime/codegen. `>1` reserved for future use. |
| `SYCL_UR_USE_LEVEL_ZERO_V2` | Integer | Enable (`1`)/disable (`0`) a preview Level Zero adapter with a redesigned architecture optimizing different queue modes (immediate/batched, in-order/out-of-order); expected to reduce runtime overhead; currently only immediate, in-order mode. |
| `SYCL_USM_HOSTPTR_IMPORT` | Integer | Non-zero enables host-data promotion to USM for buffers created with a host pointer, improving transfer performance. Also set `SYCL_HOST_UNIFIED_MEMORY=1`. |

#### `KMP_HW_SUBSET` details

Subset of available hardware resources at each topology layer, given as units per upper layer unit from the top down (sockets, cores per socket, threads per core). Alternative to complicated explicit affinity settings or a limiting process affinity mask. An offset selects resources; attributes (when available) select subsets. Extended syntax exists with `KMP_TOPOLOGY_METHOD=hwloc`, allowing extra resources such as NUMA nodes and groups of hardware resources sharing cache levels.

```text
[:][num_units]ID[@offset][:attribute] [,[num_units]ID[@offset][:attribute]...]
```

- Leading colon (`:`) = **explicit** subset; default **implicit**.
- `num_units`: positive integer (exact count) or `*` (all resources at that layer); omitted ⇒ `*` semantics.
- `ID` (case-insensitive): `C` cores per die (if any) or per socket; `D` dies per socket; `S` sockets; `T` hardware threads per core.
- `offset`: units to skip.
- `attribute` (core layer only, Intel® Hybrid Technology machines): core type `intel_atom`/`intel_core`; core efficiency `effnum`, `num` from 0 to (detected core efficiencies − 1), e.g. `eff0` (higher = more performant). More core efficiencies than core types is possible; view with `KMP_AFFINITY=verbose`.
- Cache can be a unit, e.g. `L2`, or `LL` for last level cache.

`hwloc` extended IDs: `N` = NUMA nodes per upper layer unit (e.g. per socket); `TI` = tiles per upper layer unit (e.g. per NUMA node). Specifying any `numa`/`tile` unit auto-sets `KMP_TOPOLOGY_METHOD=hwloc`.

Semantics: **explicit** — detected layers omitted from the subset are ignored; only listed layers are used. **implicit** — socket, core, thread types are implied included; other layers are not and are ignored if unspecified; omitting socket/core/thread means all their resources are used. Implicit is the default. The runtime warns and ignores the setting if: a resource's detection is unsupported by the chosen detection method; a resource is specified twice (exception: attributes differentiate it); or attributes are unavailable, undetected, or conflicting. Does not work if OpenMP affinity is disabled.

Implicit examples: `2s,4c,2t` (first 2 sockets s0–s1, first 4 cores per socket c0–c3, first 2 threads/core); `2s@2,4c@8,2t` (skip s0–s1, use s2–s3; skip c0–c7, use next 4 per socket c8–c11; first 2 threads/core); `5C@1,3T` (all sockets, skip first core, next 5 cores, first 3 threads/core); `1T` (all cores on all sockets, 1 thread/core); `1s, 1d, 1n, 1c, 1t` (1 socket, 1 die/socket, 1 NUMA node/die, 1 core/NUMA node, 1 thread/core → a single hardware thread); `4c:intel_atom,5c:intel_core` (all sockets, first 4 Intel Atom® processor cores and first 5 Intel® Core™ processor cores per socket); `2c:eff0,3c:eff1` (all sockets, first 2 efficiency-0 cores and first 3 efficiency-1 cores per socket).

Explicit examples: `:2s,6t` (exactly first two sockets, 6 threads/socket); `:1t@7` (skip t0–t6, use exactly t7); `:5c,1t` (exactly first 5 cores c0–c4, first thread each).

Use the `verbose` modifier in `KMP_AFFINITY` to see the result; the runtime prints topology before and after the setting to `stderr`:

```text
KMP_HW_SUBSET=1N,1L2,1L1,1T →
Info #191: KMP_AFFINITY: 1 socket x 4 NUMA domains/socket x 8 tiles/NUMA domain x 2 cores/tile x 4 threads/core. (64 total cores)
Info #191: KMP_HW_SUBSET 1 socket x 1 NUMA domain/socket x 1 tile/NUMA domain x 1 core/tile x 1 thread/core (1 total cores)
```

#### `OMP_PLACES` details

Explicit ordered place list: an abstract name for a set of places, or nonnegative numbers. `!` excludes the number/place immediately after it. Each explicit nonnegative number is one unique OS-defined logical processor number (on Intel® Architecture Processors, one unique hardware thread); the set resembles an OS affinity mask.

Interval notation `<lower-bound>:<length>:<stride>` denotes `"<lower-bound>, <lower-bound> + <stride>, ..., <lower-bound> +(<length>-1)*<stride>."`; omitted `<stride>` = unit stride. Intervals can specify numbers within a place and sequences of places.

```bash
# EXPLICIT LIST EXAMPLE
setenv OMP_PLACES "{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}"
setenv OMP_PLACES "{0:4},{4:4},{8:4},{12:4}"
setenv OMP_PLACES "{0:4}:4:4"
```

If all explicit numerical values are invalid, `OMP_PLACES` is ignored and set to one place representing all hardware resources available to the initial thread; if only some are invalid, the invalid ones are ignored and valid ones used.

Abstract names: `threads` (single hardware thread); `cores` (single core with one or more hardware threads); `ll_caches` (cores sharing the last level cache); `numa_domains` (cores whose closest device memory is the same memory at similar distance); `sockets` (single socket of one or more cores). Depending on runtime/topology these may also exist: `dice`, `modules`, `tiles`, `l1_caches`, `l2_caches`, `l3_caches` (each one or more cores). With Intel® Hybrid Technology, `cores:<attribute>` where `<attribute>` is core type (`intel_atom`/`intel_core`) or core efficiency (`effnum` from 0 to detected core efficiencies − 1):

```bash
OMP_PLACES=cores:intel_core
OMP_PLACES=cores:eff1
```

Resources are ordered so consecutive resources are close together. Fewer requested places `N` than available ⇒ first `N` of the ordered list (e.g. 4 cores requested when more exist → first 4); more requested than available ⇒ all places. Append a positive parenthesized number for list length, `abstract_name(num-places)`, optionally with stride `abstract_name(num-places:stride)`:

```bash
# ABSTRACT NAMES EXAMPLE

# Set to list of all available threads.
setenv OMP_PLACES threads

# Set to list of first four threads.
setenv OMP_PLACES threads(4)

# Set to list of four threads, beginning with the first thread
# and then skipping every other thread.
setenv OMP_PLACES threads(4:2)
```

Numeric abstract names: prepend `n_` to an abstract name above, e.g. `n_cores`, `n_sockets`; the name is replaced by that resource's count (8 detected cores ⇒ `n_cores` = `8`). Only for OpenMP variables that explicitly mention support.

#### `LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL` grammar

```text
<PoolInfoList>=<PoolInfo>[,<PoolInfoList>]
<PoolInfo>=<MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]
```

`<MemType>` = `all`, `device`, `host`, or `shared`; `<AllocMax>`, `<Capacity>`, `<PoolSize>` = positive integer or empty. A pool is a list of blocks serving at least `<Capacity>` allocations of up to `<AllocMax>` size from one block, total ≤ `<PoolSize>`. If `<PoolInfoList>` lists only a subset of `{device, host, shared}`, defaults apply to the rest, and a type's pool is disabled by giving `0` as its `<AllocMax>`. `0` disables the memory pool.

- `all,2,8,1024`: pool for all memory types; up to eight 2MB blocks from one Level Zero-allocated block, 1GB total pool size.
- `device,1,4,512`: device-memory pool; up to four 1MB blocks from one Level Zero-allocated block, 512MB total pool size; defaults control other memory types.

## Pass Options to the Linker

Note: since version 2024.0 options given with the Clang `-mllvm` flag are no longer passed through to linker option processing; use `-Wl`, e.g. to pass `-lto-debug-options`:

```bash
-Wl,-plugin-opt,-lto-debug-options
```

**Linux** (compile-time options that take effect at link time for the `ld` linker; see the `ld` man page):

| Option | Description |
|---|---|
| `-Ldirectory` | Linker searches `directory` for libraries. |
| `-Qoption,tool,list` | Pass an argument list to another program in the compilation sequence (assembler, linker). |
| `-shared` | Build a Dynamic Shared Object (DSO), not an executable. |
| `-shared-libgcc` | Opposite of `-static-libgcc`: GNU standard libraries linked dynamically, letting the user override static linking when `-static` is used. By default all C++ standard and support libraries are linked dynamically. |
| `-shared-intel` | All Intel-provided libraries linked dynamically. |
| `-static` | Link all libraries statically. Without: `/lib/ld-linux.so.2` linked in, all other libs dynamic. With: `/lib/ld-linux.so.2` not linked in, all other libs static. |
| `-static-libgcc` | GNU standard libraries linked statically. |
| `-static-intel` | Intel-provided libraries linked statically; opposite of `-shared-intel`. |
| `-Wl,optlist` | Pass a comma-separated `optlist` of options to the linker. |
| `-Xlinker val` | Pass `val` (linker option, object, or library) directly to the linker. |

**Windows.** Linker options do not work for SYCL kernels, but do work for host code (including linker-option pragmas). Use `link` to pass options to the linker at compile time:

```bat
icx a.cpp libfoo.lib /link -delayload:comct132.dll
```

The compiler recognizes `libfoo.lib` as a library to link with `a.cpp`, so it need not follow `link`; it does not recognize `-delayload:comct132.dll`, so `link` routes it to the linking phase:

```cpp
#pragma comment(linker, "/defaultlib:mylib.lib")
```

On C++, `Qoption` passes options to various tools including the linker, and `#pragma comment` also passes options; both pragmas below link `mylib.lib` at link time:

```cpp
#pragma comment(lib, "mylib.lib")
```

## Specify Alternate Tools

Does not apply to SYCL. `Qoption` passes the options in `optlist` (comma-separated) to a tool:

```bash
-Qoption,tool,optlist      # Linux
```

```bat
/Qoption,tool,optlist      rem Windows
```

`tool` = compilation tool receiving the options; `optlist` = one or more valid argument strings for it. Include the hyphen for a command-line option; enclose arguments containing a space or tab in `""`; separate multiple arguments with commas.

## Use Configuration Files

Configuration files automate command-line entries and are processed automatically on every run of the Intel® oneAPI DPC++/C++ Compiler. Any valid command-line option may be inserted. Options are processed in file order, followed by the command-line options at invocation. Note: they execute on **every** run; for varying per-project needs use Response Files.

Default files: `icx.cfg` and `icpx.cfg` (Linux) or `icx.cfg` (Windows), in the compiler executable's directory. For a different file use `ICCCFG`/`ICPCCFG` (Linux) or `ICLCFG` (Windows). Note: using a different configuration file makes the default file(s) ignored.

```text
## Sample icpx.cfg file        (Linux)
  -I/my_headers

## Sample icx.cfg file         (Windows)
  /Ic:\my_headers
```

The compiler reads the file and invokes the `I` option on every run, along with command-line options.

## Use Response Files

Response files hold options for particular compilations/projects in individual files and are invoked as options on the command line. They are **not** processed automatically: an unnamed response file is never invoked.

```text
Linux:   vi response1.txt → -fp-model=precise ; vi response2.txt → -O0
Windows: notepad response1.txt → /fp:precise ; notepad response2.txt → /O0
```

Options or file names may appear on a line in a response file; several response files can be referenced in one command line:

```bash
icpx @response1.txt prog1.cpp @response2.txt prog2.cpp     # Linux
```

```bat
icx @response1.txt prog1.cpp @response2.txt prog2.cpp      rem Windows
```

An at sign (`@`) must precede the response file name on the command line.

## Global Symbols and Visibility Attributes for Linux*

A **global symbol** is visible outside its compilation unit (source file plus includes) — in C/C++, anything at file level without `static`:

```cpp
int x = 5;         // global data definition
extern int y;      // global data reference
int five()         // global function definition
  { return 5; }
extern int four(); // global function reference
```

A program is a main program file plus possibly shareable object (`.so`) files defining data/functions it references; shareable objects may reference each other. If several simultaneously executing processes map a shareable object, only one copy of its read-only portion is resident in physical memory. Main program plus referenced shareable objects are the program's **components**.

Every global symbol definition or reference has a **visibility attribute** controlling how (or if) it may be referenced outside its component. Five values:

- `external`: treat the symbol as defined in another component — for a definition, assume it will be overridden (preempted) by a same-named definition elsewhere (see Symbol Preemption). For a function symbol, the compiler knows it must be called indirectly and can inline the indirect call stub.
- `default`: other components can reference it; the definition may be overridden (preempted) by a same-named definition elsewhere.
- `protected`: other components can reference it, but it cannot be preempted by a same-named definition elsewhere.
- `hidden`: other components cannot directly reference it, but its address might be passed indirectly (e.g. as a call argument, or stored in a data item referenced by a function in another component).
- `internal`: cannot be referenced outside its defining component, directly or indirectly.

Static local symbols (`static` at file scope or elsewhere) usually have hidden visibility — not directly referenceable by other components (or other compilation units in the same component), but possibly referenced indirectly. Visibility applies to references as well as definitions: a reference's visibility attribute asserts the corresponding definition will have that visibility.

Explicit visibility via the attribute on a data/function declaration:

```cpp
int i __attribute__ ((visibility("default")));
void __attribute__ ((visibility("hidden"))) x () {...}
extern void y() __attribute__ ((visibility("protected")));
```

The visibility declaration attribute value overrides the default set by `-fpic` or `-fno-common`.

## Save Compiler Information in Your Executable

```bash
# Linux: view information stored in the object file
objdump -sj comment a.out
strings -a a.out | grep comment:
```

```bat
rem Windows: view linker directives stored in string format in the object file
link /dump /directives filename.obj
rem the ?-comment linker directive shows compiler version information
findstr "Compiler" filename.exe
```

## Link Debug Information

- **Linux**: `g` at compile time generates symbolic debugging information in the object file. `gsplit-dwarf` creates a separate object file with DWARF debug information; since the DWARF object is not used by the linker, this reduces linker debug-info processing and yields a smaller executable.
- **Windows**: `Z7` at compile time or `debug` at link time generates symbolic debugging information in the object file. `Zi` at link time generates executables with debug information in the `.pdb` file.

## Ahead of Time Compilation

AOT helps when the target device is known in advance. Benefits: no compilation time at application run; no JIT bugs for the target (found during AOT and resolved); final target-device code can be tested as-is before delivery. A program built with AOT for specific target device(s) will not run on different device(s) — detect the proper target at runtime and error if the targeted device is absent; exception handling with an asynchronous exception handler is recommended.

AOT targets: SYCL → **Intel® CPUs** and **Intel® Processor Graphics**; OpenMP → **Intel® Processor Graphics**.

**Prerequisites.** Targeting a GPU requires the OpenCL™ Offline Compiler (OCLOC), which generates binaries using OpenCL™ (SYCL only) or the Intel® oneAPI Level Zero (Level Zero) backend. OCLOC is not packaged with the compiler and must be installed separately: install the GPU drivers, whether or not an Intel GPU is present.

**Requirements for Accelerators — GPUs:** Intel® UDH Graphics for 11th generation Intel processors or newer; Intel® Iris® Xe graphics; Intel® Arc™ graphics; Intel® Data Center GPU Flex Series; Intel® Data Center GPU Max Series.

### AOT Compilation Supported Options for OpenMP

`-fopenmp-target` — device target; `-Xopenmp-target-backend` — options to the backend tool. `-Xopenmp-target-backend` is a general device target option; with multiple targets (e.g. `-fopenmp-targets=spir64,spir64_gen`) its options apply to **all** targets. Add specificity with e.g. `Xopenmp-target-backend=spir64_gen <option>`. Under AOT these are OCLOC options, **not** compiler options. List them with `-fsycl-help=gen`.

### AOT Compilation Supported Options for SYCL

`-fsycl-target` — device target; `-Xsycl-target-backend` — options to the backend tool. `-Xsycl-target-backend` is a general device target option; with multiple targets (e.g. `-fopenmp-targets=spir64_gen,spir64_x86_64`) its options apply to **all** targets. Add specificity with e.g. `Xsycl-target-backend=spir64_gen <option>`. Under AOT these are not compiler options. List them with `-fsycl-help=gen` or `-fsycl-help=x86_64`.

### Use AOT for the Target Device (Intel® CPUs)

SYCL compilation is available only with the C/C++ compiler; SYCL-generated objects can be linked with the Fortran compiler — `-fsycl` with `ifx` allows this, restricted to `spir64`, `spir64_gen`, and `spir64_x86_64`.

AOT target arguments: `-fsycl-targets=spir64_x86_64`; `-Xsycl-target-backend "-march=<arch>"` where `<arch>` is `avx` (Intel® AVX), `avx2` (Intel® AVX2), `avx512` (Intel® AVX-512), or `sse4.2` (Intel® SSE4.2).

AVX2 code generation:

```bash
icpx -fsycl -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=avx2" main.cpp    # Linux*
```

```bat
icx -fsycl /EHsc -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=avx2" main.cpp   rem Windows*
```

Multiple source files for CPU targeting (SYCL only): compile normal files (no SYCL kernels) into host objects, then compile the kernel file and link.

```bash
# Linux
icpx -c main.cpp      # creates the host object used below
icpx -c -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=mavx2" mandel.cpp
icpx -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=mavx2" mandel.o main.o
```

```bat
rem Windows
icx /EHsc -c main.cpp
icx /EHsc -c -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=mavx2" mandel.cpp
icx -fsycl-targets=spir64_x86_64 -Xsycl-target-backend "-march=mavx2" mandel.obj main.obj
```

### Use AOT for Integrated Graphics (Intel® GPU)

**OpenMP** — `-Xopenmp-target-backend` is general-purpose; its arguments apply to all offline compilation invocations. Relevant: `-Xopenmp-target-backend "-device <arch>"` (`<arch>` = target device); `-fopenmp-targets=spir64_gen`; `-fopenmp-device-code-split=<value>` for an OpenMP device code split, `<value>` = `per_kernel` (one device code module per OpenMP kernel).

**SYCL** — `-Xsycl-target-backend` is general-purpose; arguments apply to all offline compilation invocations. Relevant: `-Xsycl-target-backend "-device <arch>"`; `-fsycl-targets=spir64_gen`; `-fsycl-device-code-split=<value>`, `<value>` = `per_kernel` (module per SYCL kernel), `per_source` (module per source/translation unit), `off` (no split), `auto` (heuristic — **default**, same as specifying `-fsycl-device-code-split` with no `<value>`).

Run `ocloc compile --help` for your OCLOC's supported target device types; look for `-device <device_type>`. Listing multiple target devices compiles for each and creates a **fat-binary** containing all device binaries.

```bash
# OpenMP for Linux
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device skl" vector-add.cpp
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device skl,icllp" vector-add.cpp
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device *" vector-add.cpp
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend=spir64_gen "-device *" vector-add.cpp

# SYCL for Linux
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device skl" vector-add.cpp
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device skl,icllp" vector-add.cpp
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" vector-add.cpp

# Multiple OCLOC options via -Xs:
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device tgllp --format zebin -options <-user-option1> -options <-user-option2>" vector-add.cpp
# ...or via -Xsycl-target-backend:
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device tgllp --format zebin -options <-user-option1> -options <-user-option2>" vector-add.cpp
```

```bat
rem SYCL for Windows
icx -fsycl /EHsc -fsycl-targets=spir64_gen -Xsycl-target-backend "-device skl" vector-add.cpp
icx -fsycl /EHsc -fsycl-targets=spir64_gen -Xsycl-target-backend "-device skl,icllp" vector-add.cpp
icx -fsycl /EHsc -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" vector-add.cpp
icx -fsycl /EHsc -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" vector-add.cpp
```

Multiple source files for GPU targeting:

```bash
# Linux
icpx -c main.cpp
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" mandel.o main.o
```

```bat
rem Windows
icx /c main.cpp
icx -fsycl /EHsc -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" mandel.cpp main.obj
```

### Use AOT in Microsoft Visual Studio

SYCL only. CPU — compile `Configuration Properties > DPC++ > General > Specify SYCL offloading targets for AOT compilation`; link `Configuration Properties > Linker > General > Specify CPU Target Device for AOT compilation`. GPU — compile the same DPC++ property; link `Configuration Properties > Linker > General > Specify GPU Target Device for AOT compilation`.

### Available GPU Platforms

Model names/processor lists abbreviated from the source table. Format: GPU model | segment | product code | **AOT compilation device name** | compatible targets.

| GPU Model | Segment | Product Code | AOT Device Name | Compatible Targets |
|---|---|---|---|---|
| Arc™ 140T GPU (Core™ Ultra 9 285H, 7 265H, 5 235H) | Mobile | Arrow Lake-H | `arl-h` | — |
| Graphics (Core™ Ultra 9 285T/285K/285HX/285, 7 265T/265K/265HX/265, 5 245T/245K/245HX/245, 235T/235) | Desktop/Mobile | Arrow Lake-S | `mtl-u` (or `arl-u`, `arl-s`) | `mtl` |
| Graphics (Core™ Ultra 7 265U, 5 235U) | Mobile | Arrow Lake-U | `mtl-u` (or `arl-u`, `arl-s`) | `mtl` |
| Arc™ B580, B570 Graphics | Desktop | Battlemage | `bmg-g21` | `bmg` |
| Arc™ graphics 140V (Core™ Ultra 9 288V, 7 268V/266V/258V/256V) | Mobile | Lunar Lake | `lnl-m` | `bmg` |
| Arc™ graphics 130V (Core™ Ultra 5 238V/236V/228V/226V) | Mobile | Lunar Lake | `lnl-m` | `bmg` |
| Arc™ graphics (Core™ Ultra 9 185H, 7 165H/155H, 5 135H/125H) | Mobile | Meteor Lake-H | `mtl-h` | `mtl` |
| Arc™ graphics (Core™ Ultra 7 165HL/155HL, 5 135HL/125HL) | Embedded | Meteor Lake-H | `mtl-h` | `mtl` |
| Graphics (Core™ Ultra 7 165U/164U/155U, 5 135U/134U/125U) | Mobile | Meteor Lake-U | `mtl-u` (or `arl-u`, `arl-s`) | `mtl` |
| Graphics (Core™ Ultra 7 165UL/155UL, 5 135UL/125UL, 3 105UL) | Embedded | Meteor Lake-U | `mtl-u` (or `arl-u`, `arl-s`) | `mtl` |
| MAX® 1550, MAX® 1100 | Data Center | Ponte Vecchio | `pvc` | — |
| Flex 170 | Data Center | Arctic Sound | `ats-m150` | `dg2` |
| Flex 140 | Data Center | Arctic Sound | `ats-m75` | `dg2` |
| Arc™ A770, A750, A580 | Desktop | Alchemist | `acm-g10` (or `dg2-g10`, `ats-m150`) | `dg2` |
| Arc™ A770M, A730M, A550M | Mobile | Alchemist | `acm-g10` (or `dg2-g10`, `ats-m150`) | `dg2` |
| Arc™ A380, A310, Arc™ Pro A40/A50 | Desktop | Alchemist | `acm-g11` (or `dg2-g11`, `ats-m75`) | `dg2` |
| Arc™ A370M, A350M, Arc™ Pro A30M | Mobile | Alchemist | `acm-g11` (or `dg2-g11`, `ats-m75`) | `dg2` |
| Arc™ A380E, A370E, A350E, A310E | Embedded | Alchemist | `acm-g11` (or `dg2-g11`, `ats-m75`) | `dg2` |
| UHD Graphics | Mobile | Alder Lake-N | `adl-n` | — |
| UHD Graphics, Iris® Xe graphics | Mobile | Alder Lake-P | `adl-p` | — |
| UHD Graphics 770/730/710 | Mobile | Alder Lake-S | `adl-s` | — |
| UHD Graphics 617/615 | Mobile | Amber Lake | `aml` | — |
| HD Graphics, HD Graphics 505/500 | Mobile | Apollo Lake, Broxton | `apl` (or `bxt`) | — |
| Iris® Plus graphics 655/645, UHD Graphics 630/610/P630 | Mobile | Coffee Lake | `cfl` | — |
| UHD Graphics | Mobile | Comet Lake | `cml` | — |
| Iris® Xe MAX graphics, Iris® Xe graphics, Iris® Xe MAX 100, Server GPU SG-18M | Mobile/Server | DG1 | `dg1` | — |
| UHD Graphics | Mobile | Elkhart Lake, Jasper Lake | `ehl jsl` | — |
| UHD Graphics 605/600 | Mobile | Gemini Lake | `glk` | — |
| HD Graphics, UHD Graphics, Iris® Plus Graphics | Mobile | Ice Lake | `icllp` | — |
| HD Graphics 635, Iris® Plus Graphics 650/640, HD Graphics 630/620/P630/615/610, UHD Graphics 617/615 | Mobile | Kaby Lake | `kbl` | — |
| UHD Graphics 750/730/P750 | Mobile | Rocket Lake | `rkl` | — |
| Iris® Xe Graphics, UHD Graphics | Mobile | Raptor Lake-P | `rpl-p` | — |
| UHD Graphics 770/730/710 | Mobile | Raptor Lake-S | `rpl-s` | — |
| HD Graphics 535/530/520/515/510/P530, Iris® Pro Graphics 580/P580, Iris® Graphics 555/550/540/P555 | Mobile | Intel® microarchitecture code name Skylake | `skl` | — |
| UHD Graphics, Iris® Xe Graphics | Mobile | Tiger Lake | `tgllp` | — |
| UHD Graphics, UHD Graphics 620 | Mobile | Whiskey Lake | `whl` | — |

### Use AOT with Non-Intel GPUs

**SYCL.** Besides Intel GPUs, SYCL applications can be compiled once and run on AMD* and NVIDIA* GPUs. One binary can contain device code for AMD GPUs, NVIDIA GPUs, or any SPIR-V*-supporting device including Intel GPUs.

NVIDIA GPUs for Linux (NVIDIA `sm_80` architecture):

```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 -o sycl-app sycl-app.cpp
```

`-Xsycl-target-backend=nvptx64-nvidia-cuda` sends the following flag only to the backend for that target; `--offload-arch=sm_80` specifies the NVIDIA GPU architecture (compute capability) for AOT. Note: `--offload-arch=<arch>` differs from the `-device <intel-arch>` syntax the Intel-target toolchain requires.

AMD GPUs for Linux (AMD GPU `gfx90a` architecture):

```bash
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -o sycl-app sycl-app.cpp
```

`-Xsycl-target-backend=amdgcn-amd-amdhsa` sends the following flag only to the backend for that target; `--offload-arch=gfx90a` specifies the AMD GPU architecture for AOT.

**Using Alias Targets.** The driver offers alias targets per target/architecture pair for shorter, more readable command lines; `-Xsycl-target-backend` flags may be omitted. This outputs one binary with device code that runs directly (no JIT) on AMD GPUs, NVIDIA GPUs, and Ponte Vecchio Intel GPUs, or with JIT on any SPIR-V-supporting device (e.g. Intel integrated GPUs):

```bash
icpx -fsycl -fsycl-targets=intel_gpu_pvc,amd_gpu_gfx90a,nvidia_gpu_sm_80 \
      -o sycl-app sycl-app.cpp
```

Equivalent to:

```bash
icpx -fsycl -fsycl-targets=spir64_gen,amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
-Xsycl-target-backend=spir64_gen '-device pvc' \
-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a \
-Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 \
-o sycl-app sycl-app.cpp
```

## Device Offload Compilation Considerations

SYCL compilation generates host and target binaries from one source file, creating file dependencies from device to host compilation. Those dependent files are **integration files** included in the host-side compilation. An **integration footer** is appended to the original source before compilation; a new temporary source file is generated and treated as the host source file. It is a new source dependency and can break build environments that track generated files — configure them for this additional intermediate file. It is generated in the common temporary location set by the `TMP` then `TEMP` environment variables.

## Use a Third-Party Compiler as a Host Compiler for SYCL Code

Three rules: (1) host code can be compiled with any compiler; (2) files containing device code must be compiled with the Intel® oneAPI DPC++/C++ Compiler; (3) the final program must be linked with the Intel® oneAPI DPC++/C++ Compiler.

```bash
icpx -fsycl -c michigan.cpp    # 1. may contain host and device code
g++ -c erie.cpp                # 2. host code only
ifx -c ontario.f90             # 3. host code only
icx -c huron.cpp               # 4. host code only
icpx -fsycl -c superior.cpp    # 5. may contain host and device code
# 6. final linkage with the Intel compiler
icpx -fsycl -o greatlakes.out michigan.o superior.o huron.o erio.o ontario.o
```

Mixing another SYCL* compiler with the Intel® oneAPI DPC++/C++ Compiler is not currently supported.

**External Compiler Options.** `fsycl-host-compiler` uses the specified compiler for host compilation of the offloading compilation; `fsycl-host-compiler-options` passes options to it. Example: `a.cpp` and `b.cpp` contain SYCL code, `main.cpp` contains C++ code; `g++` compiles host code and `icpx -fsycl` compiles SYCL:

```bash
# 1. Build environment (GCC version 5.1 or above must be installed and accessible):
source /opt/intel/oneapi/setvars.sh                         # Component directory layout
source /opt/intel/oneapi/<toolkit_version>/oneapi-vars.sh   # Unified directory layout

# 2. SYCL headers location:
export INCLUDEDIR=<Location of SYCL headers>

# 3. -fsycl-host-compiler makes a third-party compiler do host compilation (device
#    compilation stays with the Intel compiler); creates fat objects (device + host code):
icpx -fsycl -fsycl-host-compiler=g++ a.cpp -o a.o
icpx -fsycl -fsycl-host-compiler=g++ b.cpp -o b.o

# 4. Compile other C++ (or non-SYCL) code using G++:
g++ -std=c++17 main.cpp -c -fPIC -I$INCLUDEDIR

# 5. Final link:
icpx -fsycl main.o a.o b.o -o finalexe.exe
```

## Ccache*

Ccache* is a compiler cache tool that speeds recompilation by caching previous compilations and detecting repeats (common in CI/CD systems). It can be used with Intel compilers on Linux*, including for SYCL code with `icpx`.

```bash
ccache icx -c test.c                # direct compilation
ccache icpx -fsycl -c sycl_test.cpp
cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..   # CMake
```

## Gotchas & failure modes

Synthesis of traps stated in the rows above (details there):

- **Linking:** `-c`/`/c` stops linking — pass objects later or no binary results. A direct Linux linker call omits the Intel/system libraries the compiler normally injects. `-mllvm` no longer reaches linker option processing since 2024.0 (use `-Wl`). Windows linker options and `Qoption` do not apply to SYCL kernels; `Qoption` does not apply to SYCL at all.
- **Environment files:** configuration files run on **every** invocation (use response files per project); `ICCCFG`/`ICPCCFG`/`ICLCFG` disables the default config file(s); a response file needs a leading `@`. Windows `CL` cannot contain `=` (use `#`); `TMP`/`TMPDIR`/`TEMP` cannot be set from Visual Studio*.
- **OpenMP runtime:** affinity variables (`GOMP_CPU_AFFINITY`, `KMP_AFFINITY`, `KMP_HIDDEN_HELPER_AFFINITY`) must be set before the first parallel region / hidden helper task or before `omp_get_max_threads()`, `omp_get_num_procs()`, and affinity API calls. `KMP_DETERMINISTIC_REDUCTION=TRUE` needs `-fp-model precise` (Linux) / `-fp:precise` (Windows). `KMP_HW_SUBSET` is ignored if OpenMP affinity is disabled and auto-forces `hwloc` for `numa`/`tile` units. Precedence: `KMP_STACKSIZE` > `GOMP_STACKSIZE` > `OMP_STACKSIZE`; `KMP_ALL_THREADS` > `OMP_THREAD_LIMIT`; `KMP_AFFINITY` > (`GOMP_CPU_AFFINITY`, `OMP_PROC_BIND`); `GOMP_CPU_AFFINITY` > `OMP_PROC_BIND`.
- **Deprecated/ineffective:** `KMP_PLACE_THREADS`, `OMP_NESTED`, `KMP_TOPOLOGY_METHOD=group`, `OMP_PROC_BIND` value `MASTER`, `SYCL_DEVICE_FILTER`, `SYCL_CACHE_DISABLE_PERSISTENT` (no effect), `SYCL_ENABLE_PCI` (immediately deprecated).
- **AOT/offload:** binaries are device-specific — detect the target at runtime and error if absent (async exception handler). GPU AOT needs separately installed OCLOC; `-X*‑target-backend` args are OCLOC options, not compiler options, and `--offload-arch=<arch>` ≠ `-device <intel-arch>`. A SYCL build adds an intermediate host source file (integration footer) under `TMP` then `TEMP`.
- **SYCL limits:** mixing another SYCL* compiler is unsupported (host code may come from another compiler; device code and final linking must use the Intel compiler); `-fsycl` with `ifx` is restricted to `spir64`, `spir64_gen`, `spir64_x86_64`; third-party-host-compiler builds need GCC 5.1+; `SYCL_USM_HOSTPTR_IMPORT` also needs `SYCL_HOST_UNIFIED_MEMORY=1`; a persistent SYCL cache is disabled if `SYCL_CACHE_DIR`, `XDG_CACHE_HOME`, and `HOME`/`AppData%` are all unavailable.
- **Code split:** OpenMP supports only `per_kernel`; SYCL supports `per_kernel`, `per_source`, `off`, `auto` (default).

## Source map

- Syntactic and Semantic Errors; Compilation Overview/defaults/customization — pp. 622–624
- Compile-time env vars — pp. 624–626; runtime env vars incl. `KMP_HW_SUBSET`, `OMP_PLACES`, memory-pool grammar — pp. 626–647
- Linker options — pp. 647–649; alternate tools — p. 649; configuration files — pp. 649–650; response files — pp. 650–651
- Global symbols/visibility — pp. 651–652; saving compiler info and debug info — p. 652
- AOT compilation and GPU platform table — pp. 653–663; device offload considerations — p. 664; third-party host compilers — pp. 664–665; Ccache* — p. 665
