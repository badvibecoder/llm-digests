## 00 · Quick start: minimal SYCL for Intel GPUs

### Smallest complete program (buffer + accessor)

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr size_t N = 1024;
  std::vector<int> in(N, 1), out(N, 0);

  buffer<int, 1> buf_in{in.data(), range<1>{N}};
  buffer<int, 1> buf_out{out.data(), range<1>{N}};

  queue q{gpu_selector_v};                       // Intel GPU
  q.submit([&](handler& h) {
    accessor a{buf_in, h, read_only};
    accessor b{buf_out, h, write_only};
    h.parallel_for(range<1>{N}, [=](id<1> i) {
      b[i] = a[i] * 2;
    });
  });                                            // buffer dtor + host access sync back
  host_accessor res{buf_out, read_only};
  assert(res[0] == 2);
}
```

### Smallest complete program (USM)

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr size_t N = 1024;
  queue q{gpu_selector_v};
  int* data = malloc_shared<int>(N, q);          // host+device visible
  q.parallel_for(range<1>{N}, [=](id<1> i) {
    data[i] *= 2;
  }).wait();                                     // REQUIRED before host reads
  assert(data[0] == 2);
  free(data, q);
}
```

**Rule.** All SYCL constructs live in `namespace sycl`; one include, `<sycl/sycl.hpp>`, defines
them all. Every other line runs on the host.

### Build

```bash
# one-time environment setup
source /opt/intel/oneapi/setvars.sh

# JIT everything (device code compiled at runtime, per target)
icpx -fsycl -O2 prog.cpp -o prog

# AOT for a specific Intel GPU (no JIT at run time)
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc" prog.cpp -o prog

./prog
sycl-ls                          # list platforms/devices the runtime can see
```

See §17 for the full flag set, target triples, CMake integration and CPU/FPGA flows.

### Execution model vocabulary

| Term | Meaning | CUDA analogue |
|---|---|---|
| platform | a backend's set of devices (OpenCL, Level Zero, ...) | — |
| device | the accelerator that executes kernels | device |
| context | SYCL objects + memory shared between queues; owns USM allocations | context |
| queue | the only connection used to direct work to a device | stream |
| kernel | one operation, instantiated many times (SPMD, not a loop) | kernel |
| command group | one `submit` closure: dependences + one action | — |
| work-item | one kernel instance | thread |
| work-group | work-items sharing local memory and barriers | block |
| sub-group | hardware SIMD execution unit within a work-group | warp |
| nd_range | global range + local (work-group) range | grid + block dims |

**Rule.** `range`/`nd_range` dimensions are numbered `0..N-1`; dimension `N-1` is contiguous.
Max 3 dimensions; beyond that, linearize manually.

### Memory hierarchy

| Space | Where | Declared as |
|---|---|---|
| global | device memory, any work-item | USM pointer, `accessor`, `buffer` |
| local | per-work-group scratchpad | `local_accessor` |
| private | per-work-item registers/stack | ordinary local variable |
| constant | read-only, optimized broadcast | `constant` accessor / `accessor` on constant buffer |

**Rule.** Host and device memory are distinct unless USM `malloc_shared`/`malloc_host` is used.
**Rule.** USM creates **no implicit dependences**; `buffer`/`accessor` do.

### The three axes you must choose first

| Question | Options | Section |
|---|---|---|
| Memory model | USM (pointer-based, explicit sync) vs buffers/accessors (graph-inferred sync) | §02 |
| Parallelism expression | `parallel_for(range)` vs `parallel_for(nd_range)` vs `single_task` | §03 |
| Ordering | in-order queue vs out-of-order + explicit dependences | §04 |

### Key gotchas
- Omitting `.wait()` (USM) or the host accessor (buffers) reads pre-kernel results silently.
- `queue q;` with no selector picks `default_selector_v`, which may not be the Intel GPU.
- A kernel is not a loop: every work-item runs the body; guard work with the id/range.
- Capturing the `handler` or a `host_accessor` in a kernel body is a compile error.
- `q.parallel_for(N, ...)` and `q.parallel_for(range{N}, ...)` are different overloads; be consistent.
- Without `-fsycl`, the kernel is ordinary unoptimized host code (or fails to link).
- A device that lacks a required aspect fails at runtime, not compile time — query before relying on it.
- Kernel names are needed for caching/queries; unnamed lambdas are accepted by current DPC++ but not portable.
