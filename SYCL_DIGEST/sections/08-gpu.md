## 08 · Programming for Intel GPUs

### Hardware mapping

| SYCL unit | Intel GPU mapping |
|---|---|
| work-item | one SIMD lane |
| sub-group | SIMD width of work-items executing in parallel; mapped to one GPU EU thread |
| work-group | compute unit = Xe core, aka sub-slice (the SM analog) |
| global ND-range | the entire GPU |

**Device.** A compute device = many GPU Compute Engines (CE) = Execution Units (EU) = Xe Vector Engines (XVE), plus caches, SLM and HBM. GPUs trade single-instruction-stream features for more processors: give a large element range. Say *instruction stream*, not "thread".

**Runtime.** Offload defaults to Level Zero; OpenCL selectable. `sycl-ls` lists one entry per runtime/device, e.g. `GPU : OpenCL 3.0 NEO [...]`, `GPU : 1.1[...]`; select via `gpu_selector` or `ONEAPI_DEVICE_SELECTOR`.

### SIMD, divergence, occupancy

| Topic | Fact / rule |
|---|---|
| SIMD width | width = elements per instruction; 16, 32 or more common (Xeon CPU: 128-bit XMM / 256-bit YMM / 512-bit ZMM) |
| SPMD / divergence | implementation picks the element group per SIMD instruction stream; width stays an implementation detail unless sub-groups are explicit. Divergent paths run with channels masked/predicated, worst case dropping SIMD efficiency by the SIMD width; prefer the more converged dimension (mechanics: §05) |
| Occupancy | resident streams hide latency: a long-latency op (e.g. memory read) switches to a ready stream. Occupancy = executing / theoretical total. Low occupancy ⇏ low performance, high ⇏ high; more streams hide more latency; 2-D beats 1-D |
| Gotcha | avoid `max_compute_units` in program logic; its definition is not crisp enough for tuning — express parallelism and let the runtime map it |

### Work-group sizing and limits

| Rule | Detail |
|---|---|
| Work-group size | multiple of `preferred_work_group_size_multiple` (kernel/device query); a size not divisible by the SIMD width may run with channels disabled for the whole kernel |
| Single-work-item group | `nd_range<1>{M, 1}` is likely very slow: many GPUs mask all SIMD channels but one |
| Work-item→sub-group mapping | implementation-defined, not queryable/requestable. Use 1-D work-groups, or multidimensional ones whose highest dimension is divisible by the required sub-group size: range `{4, 4}` with max sub-group 8 → two sub-groups of 8 or four of 4 |
| Sub-group size | fixed per device + kernel + ND-range, implementation-chosen by default; requestable via `reqd_sub_group_size(dim)` at compile time if device-compatible (§05) |

| Device query | Meaning |
|---|---|
| `max_work_group_size` | max work-items per work-group on one compute unit (min 1) |
| `max_work_item_sizes` | max work-items per dimension (min `(1, 1, 1)`) |
| `local_mem_size` | local memory bytes (min 32 K) |
| `local_mem_type` | `info::local_mem_type::local` = dedicated SRAM; `::global` = abstraction over global memory |
| `global_mem_cache_line_size` | global cache line bytes |
| `global_mem_cache_size` | global cache bytes |
| `sub_group_sizes` | supported sub-group sizes |

**Fact.** Kernel info (supported/preferred work-group sizes, per-work-item private memory) comes from kernel `get_info` with a device argument; violating a required condition (e.g. `sub_group_sizes`) is a runtime error.

### Global memory access

- **Coalescing.** Performance = GPU cache lines accessed. All accesses in one line → peak; two lines (every other element, or a misaligned start) → about half; a unique line per work-item (very strided or random) → lowest.
- **Layout.** Row-parallel matmul (id `m`): neighbors access `matrixA` at indices differing by `K` → highly strided. Column-parallel (id `n`): `matrixA` indices identical, `matrixB` consecutive → efficient; column wins.
- **Rule.** Minimize cache lines touched; cut redundant global traffic by staging reusable tiles in work-group local memory or via sub-groups. For unclear dimension choice, build and profile variants per dimension.

### Shared local memory (SLM)

**Fact.** SLM usually beats global memory in bandwidth and latency, even on a cache hit; it is divided into banks.

| SLM access | Result |
|---|---|
| each work-item hits a different bank | full performance |
| consecutive local addresses | usually different banks → always full performance regardless of start |
| several work-items hit one bank | bank conflict → access serialized |
| stride = number of banks | every access conflicts (worst) |
| stride sharing no common factor with the bank count | full performance |
| stride 2 | conflicts only if the bank count is even |
| fix | pad local structures to choose a conflict-free stride |

**Fact.** `local_accessor` declares SLM from a type and range (never a buffer), always `read_write`, uninitialized at work-group entry and dead after it, may be atomic; synchronize with `group_barrier` (§05, §09).

- **Gotcha.** Exceeding the SLM limit fails at runtime: with `PrintDebugMessages=1` + `NEOReadDebugKeys=1` a `Size of SLM (656384) larger than available (131072)` failure appears as `-5 (CL_OUT_OF_RESOURCES)` from Level Zero's `zeKernelSuggestGroupSize` (`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, `0x1879048195`); `onetrace -c ./myapp` shows the same. Check available SLM before sizing tiles.

### Hiding latency and overlapping transfer

- **Fact.** `prefetch` (base pointer + byte count) and `mem_advise` (device-specific advice, e.g. read-only so the runtime may copy instead of migrate) overlap transfer with compute; issue prefetches early and chain the returned `event` into the dependent `parallel_for`. Both are hints only: correctness holds if a prefetch has not completed, pointer values stay valid, and command groups need not depend on it (signatures: §02, §04).
- **Fact.** Frequent submission cuts latency but its overhead cuts throughput; batching amortizes overhead but raises latency. Dedicated device memory wins for frequent data when transfer overlaps GPU execution; infrequent/unpredictable access may favor remote/system memory. Running part of an algorithm inefficiently on one device may beat transferring it.
- **Rule.** Each submission/completion step (GPU → kernel-mode driver → user-mode driver → runtime) adds latency; if driver heuristics seem inefficient, use API-/implementation-specific overrides (backend interop, Ch. 20).

```cpp
int HW_SPECIFIC_ADVICE_RO = 0; // real values documented by your backend
q.mem_advise(read_only_data, BLOCK_SIZE, HW_SPECIFIC_ADVICE_RO);
event e = q.prefetch(data, BLOCK_SIZE * sizeof(int));   // issue early
q.parallel_for(range{BLOCK_SIZE}, e, [=](id<1> i) { data[i] += read_only_data[i]; });
// ... e = q.prefetch(next_block, BLOCK_SIZE * sizeof(int)); overlaps next with current
```

### AOT vs JIT for Intel GPUs

| | JIT (default) | AOT (offline) |
|---|---|---|
| device chosen | runtime | compile time (`-device`) |
| startup | slower (compiles at run) | faster |
| portability | devices unknown at build | target known; less portable |
| use | default, unknown target | known target; good for debugging |

```bash
# AOT for a specific Intel GPU
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device <device name>" a.cpp b.cpp -o app.out
# OpenMP offload equivalent
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device <device name>" a.cpp b.cpp -o app.out
# list valid GPU device names/options on this machine
ocloc compile --help
# AOT debug build; example targets kbl, disables optimization, enables kernel debug
dpcpp -g -O0 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
  -Xs "-device kbl -internal_options -cl-kernel-debug-enable -options -cl-opt-disable" myprogram.cpp
```

- **Facts.** JIT and AOT images may both sit in one fat binary (host plus embedded device code). AOT moves compilation into the build, giving compile-time diagnostics, so a runtime failure is easier to diagnose after an AOT rebuild. JIT is unsupported on FPGA; only AOT is (FPGA device binary: ELF on Linux, PE on Windows).
- **Gotcha.** `error: Kernel compiled with required subgroup size 8, unsupported on this platform` + `-11 (PI_ERROR_BUILD_PROGRAM_FAILURE)`: the requested `reqd_sub_group_size` (SIMD8) was invalid (device supported SIMD16).

| GPU debug env var | Value | Purpose |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `backend:device_type:device_num` | limit devices/backends (e.g. Level Zero, OpenCL) |
| `SYCL_UR_TRACE` | `1\|2\|-1` | `1` basic runtime trace, `2` all API traces, `-1` all + debug |
| `ZE_DEBUG` | any value | Level Zero API calls and events |
| `SYCL_PI_TRACE` | `1\|2\|-1` | runtime plugin tracing |
| `SYCL_PRINT_EXECUTION_GRAPH` | `0\|1` | dump execution graph to DOT files (Linux) |
| `IGC_ShaderDumpEnable` / `IGC_ShaderDumpEnableAll` | `0\|1` | Intel Graphics Compiler (JIT) dump (Linux) |

### Compute-bound optimization

| Area | Rule |
|---|---|
| FP precision | 32-bit float is highly optimized; many GPUs support 16-bit (precision for speed); `double` is supported but costs more — 32-bit ops usually win |
| Integer width | 32-bit beats 64-bit, 16-bit may beat both. Addressing uses 64-bit `size_t`: keep most calculation 32-bit; 16-bit indexing may suffice for small local-memory buffers |
| Optional support | test `aspect::fp16` (`sycl::half`) and `aspect::fp64` (`double`) — §07 |
| Math built-ins | most GPUs do not implement the `sycl` built-ins natively and expand them to long instruction sequences; fast/native variants with reduced/implementation-defined accuracy can be an order of magnitude faster (list: §13) |
| Multiply-add | nearly all GPUs have `mad`/`fma`; compilers contract multiplies/adds automatically and `mad`/`fma` can be called explicitly — never disable floating-point contraction |
| Specialized | dot-product-and-accumulate etc. reachable only via compiler optimization, SYCL extensions or backend interop |
| Libraries | prefer vendor libraries (oneMKL, dispatching to Intel MKL, cuBLAS, hipBLAS) over reimplementing tuned routines |

### Sub-group kernel: no local memory, no barrier

```cpp
// Tiled matmul using sub-group broadcast; requires sub-group size >= tile_size
constexpr int tile_size = 4;
h.parallel_for(
  nd_range<2>{{M, N}, {1, tile_size}},
  [=](nd_item<2> item) {
    auto sg = item.get_sub_group();
    int m = item.get_global_id()[0], n = item.get_global_id()[1], i = item.get_local_id()[1];
    T sum = 0;
    for (int kk = 0; kk < K; kk += tile_size) {
      T tileA = matrixA[m][kk + i];
      for (int k = 0; k < tile_size; k++)  // no local memory, no barrier
        sum += group_broadcast(sg, tileA, k) * matrixB[kk + k][n];
    }
    matrixC[m][n] = sum;
  });
```

`reqd_sub_group_size(w)` usage and the sub-group/work-group sizing query table are in §05.

### Key gotchas
- Make the work-group size a multiple of `preferred_work_group_size_multiple`; otherwise channels stay disabled for the whole kernel.
- Never submit an `nd_range` work-group of one work-item; all but one SIMD channel are masked.
- Request `reqd_sub_group_size` only for device-supported sizes; otherwise the build fails at runtime/AOT.
- Do not assume work-item→sub-group mapping; use 1-D work-groups or a divisible highest dimension.
- Keep control flow converged; divergent branches execute both paths with masked lanes.
- Give the GPU enough work for high occupancy and memory-latency hiding.
- Count cache lines, not bytes; misaligned or strided global access can halve throughput; a unique line per work-item is worst.
- Avoid stride equal to the SLM bank count; pad local structures to remove common factors.
- Local memory is per-work-group scratch: uninitialized on entry, dead on exit; `group_barrier` before/after exchange.
- Prefetch is a hint only; never depend on it for correctness.
- `CL_OUT_OF_RESOURCES` (-5) may mean SLM over the hardware limit; check available SLM before sizing tiles.
- 64-bit fp/int and native math built-ins are not fast; use smaller types or fast/native variants when accuracy allows.
