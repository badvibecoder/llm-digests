## 12 · Practical concerns, debugging and pitfalls

### Adding SYCL to an existing C++ program
**Pipeline.** (1) Make shared mutable data race-free; (2) add concurrency/parallelism; (3) tune — do (1) first. Mixing OpenMP/MPI/TBB is legal but an extra concern. Start from one isolated, maximally parallel point, then extend; refactoring may create more.

**Link rule.** The *same* toolchain that compiled the SYCL device code must link. A different or non-SYCL-aware compiler yields a non-functional program — it cannot integrate host and device code.

```bash
icpx -fsycl vector_add.cpp                  # compile + link
icpx -fsycl -c myprogram.cpp -o myprogram.o # compile
icpx -fsycl myprogram.o -o myprogram        # link (must be SYCL-aware)
icpx -fsycl axpy.o -lsycl -lOpenCL -lpthread -lm -ldl -o axpy.out
```

### Fat binaries and the compilation model
Fat (multiarchitecture) binary: one binary holding all compiled + intermediate device code; behaves like `a.out`/`a.exe`, automating device-code selection.

| Mode | When | Tradeoff |
|---|---|---|
| AOT | device target known at compile time | saves runtime; more compile time, fatter binary, less portable; catches errors at build time |
| JIT | default for most devices incl. GPUs | portable, incl. devices absent at build time; failures surface at runtime |

**Rule.** Use JIT unless a known need (FPGA: synthesis place-and-route is very slow); carrying both maximizes portability. Device code may be compiled separately and combined into the fat binary later (bundler/unbundler); required for FPGA.

### Multiple compilers, ABI and libraries
- SYCL-capable compilers link object code from other C++ compilers; same concerns as any mixed C++ build: name mangling, standard libraries, calling conventions.
- **Rule.** Use the SYCL runtime shipped with the compiler that built the program. Never mix SYCL compilers and SYCL runtimes — implementations and data layouts for important SYCL objects may differ.
- Interop with non-SYCL source languages (OpenCL, C, CUDA, precompiled IR) is a separate feature (§15).

### Multiple translation units
**Rule.** Functions defined in another translation unit but called inside a kernel need `SYCL_EXTERNAL`; without it they are compiled for non-device use only, making the device call illegal.

**Restrictions on `SYCL_EXTERNAL`** (none apply within one TU): functions only; no raw pointer parameter or return types (explicit pointer classes instead); cannot call a `parallel_for_work_item` method; cannot be called from a `parallel_for_work_group` scope.

```
error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
```
Missing attribute on the definition → `terminate called after throwing an instance of '...compile_program_error'...`, `error: undefined reference to ...`.

`SYCL_EXTERNAL` is optional (SYCL does not require compiler support); DPC++ supports it.

**Performance.** Scattering device code across translation units can trigger more JIT invocations than colocated code (implementation-dependent). Mitigations: group device code in one translation unit; use AOT.

### Initializing data and accessing kernel outputs
**Rule.** A `buffer` built from a host allocation (array, `vector`) owns that allocation for its whole lifetime; do not touch the host allocation until the buffer is destroyed. Buffer destruction *and* host accessor creation synchronize the task graph.

**Gotcha.** Reading a host allocation while its buffer is alive gives initial/stale values — the kernel may not have started. Use scoped host accessors.

```cpp
queue q;
buffer<int> in_buf{N}, out_buf{N};
{ // CRITICAL: scope for host_accessor lifetime
  host_accessor in_acc{in_buf}, out_acc{out_buf};
  // fill input, zero output ...
} // CRITICAL: close scope so host accessors are destroyed
q.submit([&](handler& h) {
  accessor in{in_buf, h}; accessor out{out_buf, h};
  h.parallel_for(range{N}, [=](id<1> idx) { out[idx] = in[idx]; });
});
host_accessor A{out_buf};   // buffer still alive; safe
```
**Rule.** Once a `host_accessor` constructor returns, all earlier writes to that buffer have executed and are visible.

**Gotcha — deadlock.** A live `host_accessor` forbids device use of that buffer; the runtime does not analyze the host program, so only its destructor signals completion. Waiting on the host (`queue::wait()`, another host accessor) while earlier host accessors remain in scope hangs the program. Destroy host accessors as soon as unneeded.

### Debugging
**Tools.** See §19 `### Debug tool matrix`. `onetrace` works on either backend; `ze_tracer`/`cl_tracer` give nothing on the other.

**Rule.** Intel Inspector cannot catch errors in offload code running on GPU/FPGA; run kernels on a CPU target first (commands in §19 `### Intel Inspector`):
```bash
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
inspxe-cl -c mi3 -- <app>   # memory analysis
inspxe-cl -c ti3 -- <app>   # threading analysis
```

**Tips.** CPU execution is a powerful debugging tool: races, deadlocks and transfer errors often vanish; still run on the target device often. On deadlock, check host accessors are destroyed and work-items obey the spec's synchronization rules. For dependency bugs, switch out-of-order queues to in-order and sprinkle `queue.wait()`; if failures change or disappear, that logic is the suspect. Intermittent "fails until I run it in the debugger" means dependencies are unsynchronized. Kernel-code debugging: start on CPU. Breakpoints go before or inside `parallel_for`, never on the `parallel_for` itself.

```cpp
q.submit([&](handler &h) {
  stream out(1024, 256, h);
  h.parallel_for(range{8}, [=](id<1> idx) { out << "ID:" << idx << "\n"; });
});
```
DPC++ also has an experimental C-style `printf` in kernels, with restrictions (see §19).

**Flags.** Start with `-g`; add `-ferror-limit=1` and `-Werror -Wall -Wpedantic`. Debug build for host + JIT kernel; kernel flags are taken at link time: `icpx -fsycl -g -O0 -c myprogram.cpp` then `icpx -fsycl -g -O0 myprogram.o`. AOT GPU debug build (kernel debug mode required): `dpcpp -g -O0 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device kbl -internal_options -cl-kernel-debug-enable -options -cl-opt-disable" myprogram.cpp`.

### GDB on device code
- GPU threads print as `<inferior_number>.<thread_number>:<SIMD Lane/s>`; `2.3:[1 3 5 7]` = lanes 1, 3, 5, 7 of thread 3 on inferior 2. Inferior 1 = host, inferior 2 = GPU (created automatically).
- Switching: `thread 3:4` (thread 3, lane 4), `thread :6` (lane 6 of current thread), `thread 7`; default lane = previously selected if active, else first active. `thread apply 2.5:3-5 print element`; `thread apply :3 :5 :6 print element`.
- All-stop stops all threads at a breakpoint; non-stop shows the stop event while others run. `set scheduler-locking` prevents resuming other threads; SIMD lanes resume only with their thread. In non-stop mode pass `-a` to `continue` to resume all. Before stepping, `set scheduler-locking step`; `step` enters functions, `next` steps over.

### Debugging runtime failures
JIT runtime failure means: an explicitly used feature the hardware cannot support (e.g. `fp16`, `simd8`), a compiler/runtime bug, or nonsense that tripped the runtime. Rebuild with AOT where available — compile-time messages carry far more information. Runtime failures need not abort; catch, avoid, or both.

```
terminate called after throwing an instance of 'sycl::_V1::runtime_error'
  what():  Native API failed. Native API returns: ...
```
```
terminate called after throwing an instance of 'sycl::_V1::compile_program_error'
  what():  The program was built for 1 devices
error: Kernel compiled with required subgroup size 8, which is unsupported on this platform
error: backend compiler failed build.
-11 (PI_ERROR_BUILD_PROGRAM_FAILURE)
```
```
terminate called after throwing an instance of 'cl::sycl::invalid_object_error'
what(): SYCL buffer size is zero. To create a device accessor, SYCL buffer size must be
greater than zero. -30 (CL_INVALID_VALUE)
terminate called after throwing an instance of 'cl::sycl::runtime_error'
what(): NULL pointer argument in memory copy operation. -30 (CL_INVALID_VALUE)
```

**Gotcha — generic backend errors from logic bugs:** buffer from the wrong context passed to a kernel; `this` instead of a class element; a host buffer instead of a device buffer; an uninitialized pointer passed even if unused.

**Gotcha — `[=]` captures `this`.** A bare `data`/`factor` member is really `this->data`/`this->factor`, so `this` is copied into the kernel and OpenCL/Level Zero fails with an illegal-arguments error. Fix: hoist to locals (`auto data2 = data; auto factor2 = factor;`) and call `kernel(data2, b, factor2, LEN, item_ct1)`, or redeclare locals with the same names. Unchanged code works only if `MyClass` itself (not just `data`) is in USM; common when migrating CUDA kernels.

**Triage.** Errors reproducing across runtimes mostly eliminate the runtime; errors reproducing on all devices mostly eliminate bad hardware. Try `ONEAPI_DEVICE_SELECTOR=opencl:cpu` or `=host` (single-threaded; isolates races/deadlocks) before deeper debugging.

### Error handling
| Class | Detected | Mechanism |
|---|---|---|
| Synchronous | a host operation (API call, object construction), thrown immediately | C++ exceptions; catch `sycl::exception` |
| Asynchronous | during task-graph execution, decoupled in time from host | `async_handler` on a queue/context, invoked at controlled points |

`sycl::exception` derives from `std::exception`. Implementations report as many errors synchronously as possible. With no handling code the default is abnormal termination (`terminate called after throwing an instance of '...'`; an unhandled asynchronous error gives `terminate called without an active exception`) — errors are never silently lost.

```cpp
try {
  buffer<int> b{range{16}};
  buffer<int> b2(b, id{8}, range{16});  // throws: sub-buffer larger than parent
} catch (sycl::exception &e) { std::cout << e.what() << "\n"; return 1; }
```

**Asynchronous handler.** A `std::function` (function, lambda or function object) passed to a queue or context constructor, accepting `sycl::exception_list`. Invocation points (only these): `queue::throw_asynchronous()`, `queue::wait_and_throw()`, `event::wait_and_throw()`, queue destruction, context destruction — without them errors surface only at teardown, so call them at regular controlled points. Default handler and the handler's exception filter: see §04 `### Asynchronous errors`. SYCL disallows `throw` in device code; signal failure via a logging buffer or a defined invalid result.

```cpp
throw sycl::exception(errc::runtime, "Could not find a device with the requested backend!");
```
```cpp
catch (const cl::sycl::exception& e) {
  auto clError = e.get_cl_code();
  bool hascontext = e.has_context();
  std::cout << e.what() << "CL ERROR CODE : " << clError << std::endl;
}
```

**Error codes named in the sources:**

| Code | Numeric | Raised by |
|---|---|---|
| `errc::feature_not_supported` | — | `queue` constructor with `property::queue::enable_profiling` when the device lacks `aspect::queue_profiling` (synchronous) |
| `errc::runtime` | — | user-constructed `sycl::exception` (e.g. no device with requested backend) |
| `PI_ERROR_INVALID_VALUE` | `-30` | illegal sub-buffer size; zero-size SYCL buffer; null pointer in memory copy |
| `PI_ERROR_BUILD_PROGRAM_FAILURE` | `-11` | backend compiler failed build (e.g. unsupported required subgroup size) |
| `PI_ERROR_DEVICE_NOT_FOUND` | — | no device of requested type, e.g. `info::device_type::gpu` |
| `CL_INVALID_VALUE` | `-30` | OpenCL-backend spelling of the invalid-value cases above |
| `CL_INVALID_ARG_VALUE` | `-50` | e.g. `clSetKernelArgMemPointerINTEL` on an UNKNOWN pointer with no shared system pointer support |
| `ZE_RESULT_ERROR_INVALID_ARGUMENT` | `2013265924` | e.g. `zeKernelSetArgumentValue` with a bad argument |

**Call-log triage:** failing call, reason code, `argIndex`, `pArgValue`, `hKernel` — see §19 `### Failure and correctness workflows`.

### Queue profiling and timing
See §04 `### Queue profiling`.

### Environment variables
DPC++ debug options (Windows and Linux unless noted):

| Variable | Value | Effect |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | selector string | limit devices/backends available when the SYCL application runs |
| `SYCL_PI_TRACE` | `1` basic, `2` advanced, `-1` all | Plugin Interface (PI): `1` plugin/device discovery, `2` all PI calls, `-1` all levels |
| `SYCL_PRINT_EXECUTION_GRAPH` | `always`, or selective: `before_addCG`, `after_addCG`, `before_addCopyBack`, `after_addCopyBack`, `before_addHostAcc`, `after_addHostAcc` | DOT-extension files tracing the execution graph |
| `SYCL_UR_TRACE` | `1`, `2`, `-1`; default disabled | `1` plugins/devices discovered, `2` SYCL API calls with args/results, `-1` all tracing |
| `ZE_DEBUG` | any value; default disabled | Level Zero APIs called, event information |

**`ONEAPI_DEVICE_SELECTOR`.** Limits runtimes, device types and device IDs; IDs match those from the SYCL API, `clinfo` or `sycl-ls` (numbering starts at 0) and are unrelated to type or runtime. Requesting via a programmatic special selector (like `gpu_selector`) a device filtered out by this variable causes an exception. Default: all available runtimes and devices. Examples: `opencl:cpu`, `opencl:gpu`, `opencl:gpu:2` (third device, must be a GPU), `level_zero:gpu:1` (second device, must be a GPU), `opencl:cpu,level_zero`, `host` (host device, single-threaded).

**IGC dump variables** (`IGC_ShaderDumpEnable`, `IGC_ShaderDumpEnableAll`, `IGC_DumpToCurrentDir`) and **offload-runtime variables** (`LIBOMPTARGET_*`, `OMP_TARGET_OFFLOAD`): see §19 `### Environment variables`.

### Key gotchas
- Never touch a host allocation while its `buffer` is alive: values are stale or the kernel has not run.
- Destroy `host_accessor`s promptly or the buffer is locked to the host and the program deadlocks.
- A breakpoint on `parallel_for` fires many times; break before or inside it instead.
- Call `queue::wait_and_throw()`/`throw_asynchronous()` explicitly; teardown-only reporting hides errors.
- Never let an async handler return normally after an error unless recovery is genuinely safe.
- Kernel-called functions from another translation unit need `SYCL_EXTERNAL`, else link/runtime failure.
- Never mix SYCL compilers and SYCL runtimes, and never link with a non-SYCL-aware toolchain.
- `[=]` captures `this` for member accesses — copy members into locals before the kernel.
- `enable_profiling` on a device without `aspect::queue_profiling` throws `errc::feature_not_supported`.
- Prefer JIT for portability; use AOT to turn runtime JIT failures into build-time diagnostics.
- Use `ONEAPI_DEVICE_SELECTOR=opencl:cpu` or `=host` to isolate races, deadlocks and logic errors.
- No `throw` in device code; report device-side failures via buffers or sentinel values.
