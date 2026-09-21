## 10 · FPGA programming (condensed)

`(DPC++ ext)` FPGA selectors and pipes are the only DPC++ extensions in this topic.

### Model: spatial pipeline vs ISA/SIMD
| | ISA-based (CPU/GPU) | FPGA (spatial, data flow) |
|---|---|---|
| Hardware | fixed regions reused over time | each operation gets its own device region |
| Parallelism | SIMD lanes across work-items | pipeline parallelism; scalar data flow |
| Data movement | register file / memory | output of one instruction wired to next |
| Divergence | expensive | both sides of a branch share hardware |
| Loop-carried deps | break or synchronize | handled natively by backward pipeline communication |
| "Occupancy" | resident warps/threads | average pipeline-stage utilization over time (different meaning) |

**Rule.** Each kernel generates its own spatial pipeline consuming device area; kernels execute concurrently with independent forward progress, so a stalled kernel does not block others.
**Rule.** Pipelining is automatic — register insertion and balancing are the compiler's job, never manual.
**Rule.** No/low work-item communication → `parallel_for` over a range or `nd_range`; one work-item per cycle enters stage 1. Serial state / loop-carried dependence → `single_task` with a loop; iterations, not work-items, fill stages. Work-groups and work-group local memory are implemented efficiently from on-chip resources.
**Rule.** Loops are fine and are the main occupancy mechanism; iterations overlap. Backward data flow is expressed only by loops or intra-kernel pipes with ND-range kernels.

### Initiation interval (II)
**Fact.** II = clock cycles between starting successive loop iterations. II=1 ideal; II=N idles each stage (N−1)/N of cycles (II=2 → 50% occupancy).
**Fact.** II>1 arises when a data dependence takes multiple cycles — often an off-chip memory lookup on the critical path.
**Rule.** Static reports give each loop's II and why it exceeds 1; restructure loop compute accordingly. Nested loops can interleave outer iterations with an II>1 inner loop to refill stages.

### Compile flow (ahead-of-time)
| Stage | Form | Purpose |
|---|---|---|
| Emulation | `icpx -fsycl -fintelfpga my_source_code.cpp` + `ext::intel::fpga_emulator_selector_v` | correctness at host compile speed; emulator runs on host |
| Static reports | same toolchain, generated quickly | compiler-inferred structures + bottlenecks; primary optimization guide |
| Hardware | `icpx -fsycl -fintelfpga my_source_code.cpp -Xshardware` | device binary; place-and-route can take hours |

**Rule.** Compilation is ahead-of-time: the device binary is embedded in the DPC++ executable. Running it auto-programs the attached FPGA on first kernel submission (one-time delay); resubmissions do not pay it.
**Rule.** `-Xshardware` and the selector are a matched pair. Emulation = omit `-Xshardware` and use the emulator selector (source also spells these `INTEL::fpga_selector_v` / `INTEL::fpga_emulator_selector_v`).
**Rule.** Do not exceed 90% of any FPGA resource, and never 90% of multiple resources — routing exhaustion lowers frequency. FPGA clock frequency is lower than fixed silicon; compare throughput (ops/sec), not frequency.

```cpp
#include <sycl/ext/intel/fpga_extensions.hpp> // fpga_selector_v, fpga_emulator_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;
queue q{ext::intel::fpga_selector_v};          // hardware
queue q{ext::intel::fpga_emulator_selector_v}; // emulation
```

### Pipes API
On-chip-memory FIFOs carrying data plus implicit empty/full control signals → no off-chip-memory cost, lightweight synchronization. Four connectivity types: inter-kernel, intra-kernel, host, I/O peripherals. Identity is type-based — same pipe type means same FIFO.

```cpp
template <typename name, typename dataT, size_t min_capacity = 0>
class pipe;

using my_pipe = ext::intel::pipe<class some_pipe, int>; // alias = recommended idiom

// Blocking
T read();
void write(const T &data);
// Non-blocking
T read(bool &success_code);
void write(const T &data, bool &success_code);
```

| Item | Fact |
|---|---|
| `name` | tag type; establishes pipe identity |
| `dataT` | unit type of each read/write |
| `min_capacity` | default `0` = automatic selection; if set, guarantees that many words writable before any read |
| Blocking | waits/pauses execution until the operation succeeds |
| Non-blocking | returns immediately, sets `success_code` |
| Read / write success | pipe was not empty / not full |
| No accessor or event dependence between two kernels | runtime runs both concurrently; they communicate through the pipe instead of buffers or USM |

**Rule.** Set `min_capacity` when the producer must write all outputs before the consumer runs, or to decouple burst-producing kernels at the cost of FPGA memory.

```cpp
// inter-kernel pipe: ND-range writer + single_task loop reader
q.submit([&](handler& h) {
  auto a = accessor(b_in, h);
  h.parallel_for(count, [=](auto idx) { my_pipe::write(a[idx]); });
});
q.submit([&](handler& h) {
  auto a = accessor(b_out, h);
  h.single_task([=]() {
    for (int i = 0; i < count; i++) { a[i] = my_pipe::read(); }
  });
});
```

### GPU idioms that hurt on FPGA
| GPU idiom | FPGA consequence | Instead |
|---|---|---|
| SIMD lanes / sub-groups for sharing | Pipeline is scalar, not vectorized across work-items | Plain scalar code; fine-grained dependences are cheap in-pipeline |
| Batching work to avoid inter-thread communication | Unnecessary: communication across work-items, even across work-groups, is handled in the pipeline | Loops with carried state, or intra-kernel pipes |
| Avoiding loops / manual unrolling | Loops create occupancy; iterations overlap across stages | Keep loops; reduce II |
| Mass thread occupancy to hide memory latency | Compiler balances the pipeline to assumed latencies; irregular access stalls earlier stages | Simplify addressing, maximize static coalescing |
| Assuming "occupancy" = resident threads | FPGA occupancy = average pipeline-stage utilization | Think queue of pending work feeding stage 1 |
| Tuning to a fixed memory system (banks, ports) | Memory systems are compiler-customized per kernel | Tune via static reports |

### Key gotchas
- Never omit `-Xshardware` for hardware, and never pair `fpga_selector_v` with an emulation build.
- Do not hand-insert pipeline registers; the compiler does register insertion and balancing.
- Add accessor or event dependences between pipe-connected kernels and they serialize.
- Alias pipe types; a mismatched type parameter silently creates a different FIFO.
- Non-blocking access needs the `bool &success_code` overloads; blocking forms report no failure.
- `min_capacity = 0` is automatic selection, not zero capacity.
- Check each loop's II in static reports; II>1 halves occupancy silently.
