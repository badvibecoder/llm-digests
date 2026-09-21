## 13 · Libraries: C++ standard library and oneDPL

### SYCL built-in functions

**Fact.** Built-ins live in namespace `sycl`, usable on host and device. Spec: SYCL 2020 §4.17.5–4.17.9 (`registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html`).

| Category | Examples |
|---|---|
| Floating-point math | `asin`, `acos`, `log`, `sqrt`, `floor` |
| Integer | `abs`, `max`, `min` |
| Common | `clamp`, `smoothstep` |
| Geometric | `cross`, `dot`, `distance` |
| Relational | `isequal`, `isless`, `isfinite` |

**Fact.** DPC++ precision-control flags: `-mfma`, `-ffast-math`, `-ffp-contract=fast`.

**Rule.** A SYCL math function is *not* required to produce the same numeric result as its C/C++ `std` counterpart for a given hardware target; variation is permitted for hardware characteristics/limits. `sycl::log` == `std::log` may hold in practice (DPC++), never by contract.

**Rule — always `sycl::`.** Invoke built-ins with an explicit `sycl::` prefix. Bare `sqrt()` is not guaranteed to resolve to the SYCL built-in on all implementations even after `using namespace sycl;`. Avoid `using namespace sycl;` entirely; qualify `std::` and `sycl::` explicitly to avoid unresolvable conflicts and differing precision guarantees.

```cpp
accA[idx] = std::log(accA[idx]);
accB[idx] = sycl::log(accB[idx]);
if (!sycl::isequal(accA[idx], accB[idx])) { accP[0] = false; }
```

### C++ standard library in device code

**Rule.** The SYCL spec does not guarantee `std::` functions in device code. DPC++ (and other compilers) support a tested set as an extension: include the corresponding C++ header and use the `std` namespace, exactly as on the host.

**Fact.** Tested with libstdc++ (GNU) gcc 7.5.0+, libc++ (LLVM) clang 11.0+, MSVC Standard C++ Library VS 2019+; the same run on host CPU.

**Fact.** Linux DPC++ default is GNU libstdc++ → no compile/link option needed. libc++ requires `-stdlib=libc++ -nostdinc++` **and** a runtime rebuilt with libc++; not recommended without a specific reason.

**Fact.** The tested-coverage table marks std APIs `Y` per CPU/GPU/FPGA target; blank = incomplete coverage.

**Gotcha.** A `std::` function not marked `Y` can cause functional incorrectness or build failure on an untested target device.

```cpp
h.single_task([=]() { std::swap(a[0], a[1]); });   // std:: in device code
```

### oneDPL — headers and namespaces

**Fact.** oneDPL (oneAPI DPC++ Library) targets the DPC++/C++ compiler; not part of SYCL 2020, but implemented on top of SYCL → compatible with any SYCL 2020 compiler. Install via Intel oneAPI Base Toolkit. All headers under `oneapi/dpl`; most classes/functions in namespace `oneapi::dpl`.

**Fact.** Components: Parallel STL (usage instructions, macros); Extension API (Parallel Algorithms, Iterators, Function Object Classes, Range-Based API); Tested Standard C++ APIs; Random Number Generator.

| Include | Provides |
|---|---|
| `#include <oneapi/dpl/algorithm>` | algorithms (per-algorithm choice) |
| `#include <oneapi/dpl/numeric>` | numeric algorithms |
| `#include <oneapi/dpl/memory>` | memory algorithms |
| `#include <oneapi/dpl/execution>` | SYCL execution policies (required for policies) |
| `#include <oneapi/dpl/iterator>` | `oneapi::dpl::begin` / `oneapi::dpl::end` (buffer iterators) |
| `#include <sycl/sycl.hpp>` | SYCL |

**Fact.** Extension API iterators (`oneapi::dpl::`, oneDPL ext): `counting_iterator`, `zip_iterator`, `discard_iterator`, `permutation_iterator`, `transform_iterator`, `host_iterator`.

### SYCL execution policies

**Fact.** C++17 parallel algorithms take an execution policy as first argument, denoting how the algorithm may execute (threads, SIMD, or both). Standard policies: `seq`, `unseq`, `par`, `par_unseq`.

**Rule.** Only the parallel *unsequenced* policy (`par_unseq`) can be safely offloaded to SYCL devices. `par` and the others conflict with SYCL work-item forward-progress guarantees.

**Fact.** A SYCL-aware policy inherits a standard C++ execution policy, encapsulates a SYCL device or queue, and allows an optional kernel name. Usable with all standard C++ algorithms that support execution policies per C++17.

Three steps:
1. `#include <oneapi/dpl/execution>`.
2. Create a policy object: standard policy type + optional `class` kernel-name template argument + one constructor argument — a SYCL queue, a SYCL device, a SYCL device selector, or an existing policy object with a different kernel name.
3. Pass the policy object to an algorithm.

**Fact.** `oneapi::dpl::execution::dpcpp_default` is a predefined `device_policy` with default kernel name and default queue; use directly or as the base for custom policies.

```cpp
using namespace oneapi::dpl::execution;
auto policy_b = device_policy<parallel_unsequenced_policy, class PolicyB>{
    sycl::device{sycl::gpu_selector{}}};
std::for_each(policy_b, ...);
auto policy_c = device_policy<parallel_unsequenced_policy, class PolicyC>{
    sycl::default_selector{}};
auto policy_d = make_device_policy<class PolicyD>(default_policy);
auto policy_e = make_device_policy<class PolicyE>(sycl::queue{});
```

**Fact.** A policy exposes `queue()`: `policy.queue().get_device().get_info<info::device::name>()`.

**Fact.** Algorithms: all C++17 `std` algorithms accepting an execution policy — `std::sort`, `std::reduce`, `std::transform`, `std::inclusive_scan`, `std::for_each`, `std::fill`; oneDPL extension `oneapi::dpl::binary_search`. Other headers are selected per algorithm (`<oneapi/dpl/algorithm>`, `<oneapi/dpl/numeric>`, `<oneapi/dpl/memory>`).

### oneDPL with buffers

**Fact.** `oneapi::dpl::begin(buf)` and `oneapi::dpl::end(buf)` accept a SYCL buffer and return objects of unspecified type satisfying:
- CopyConstructible, CopyAssignable, comparable with `==` and `!=`.
- Valid expressions `a + n`, `a - n`, `a - b` (a, b iterators; n integer).
- No-argument `get_buffer()` returning the SYCL buffer passed to `begin`/`end`.

**Rule.** Requires `#include <oneapi/dpl/iterator>`; not included by default, and not needed with USM.

**Fact.** The algorithm stays in `std::`; only the execution policy is in a nonstandard namespace.

```cpp
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>
sycl::queue q;
sycl::buffer<int> buf{1000};
auto buf_begin = oneapi::dpl::begin(buf);
auto buf_end   = oneapi::dpl::end(buf);
auto policy = oneapi::dpl::execution::make_device_policy<class fill>(q);
std::fill(policy, buf_begin, buf_end, 42);
```

**Fact.** Host-side iterators + default policy: `std::fill(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(), 42);` creates a temporary SYCL buffer, copies data in, processes on device, copies back. Prefer existing SYCL buffers — fewer host/device transfers, no buffer create/destroy overhead.

**Fact.** `oneapi::dpl::binary_search(policy, k_beg, k_end, v_beg, v_end, r_beg)` writes, for the *i*-th search element, a Boolean found-result to the *i*-th result element; returns an iterator one past the last assigned result. Input sequence is assumed sorted per the supplied comparator; with no comparator, a function object using `operator<` is used.

Buffer recipe: (1) create SYCL iterators from buffers; (2) create a named policy from an existing policy; (3) invoke the parallel algorithm.

### oneDPL with USM

**Fact.** With USM, pass the pointer to the start and one-past-the-end of the allocation directly as iterators. Policy and allocation must be created for the same queue or context — otherwise undefined behavior at runtime (not oneDPL-specific; always true for USM).

**Rule.** Same USM allocation processed by several algorithms → use an in-order queue or explicitly wait for each algorithm before the next; always wait for completion before host access.

```cpp
sycl::queue q;
const int n = 10;
int* h_head = sycl::malloc_host<int>(n, q);
int* d_head = sycl::malloc_device<int>(n, q);
std::fill(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n, 78);
q.wait();
q.memcpy(h_head, d_head, n * sizeof(int));
q.wait();
// ...
sycl::free(h_head, q);
sycl::free(d_head, q);
```

**Fact.** USM allocator with containers: `sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(q); std::vector<int, decltype(alloc)> vec(n, alloc);` — the vector manages memory normally but allocates through an internal `sycl::malloc_shared`; `begin()`/`end()` return iterators stepping through a USM allocation. Convenient when migrating existing C++ container/algorithm code.

### Error handling with SYCL execution policies

**Rule.** For algorithms run with SYCL-aware execution policies, *all* error handling (synchronous or asynchronous) is the caller's responsibility:
- No exceptions are thrown explicitly by the algorithms.
- Exceptions thrown by the runtime on the host CPU, including SYCL synchronous exceptions, are passed through to the caller.
- SYCL asynchronous errors are not handled by oneDPL; handle them (if desired) with the usual SYCL asynchronous exception mechanisms.

### Device reduce/sort: USM and buffers

```cpp
// USM pointers as iterators
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
sycl::queue q;
int* d = sycl::malloc_device<int>(n, q);
q.memcpy(d, h, n * sizeof(int));
q.wait();
auto policy = oneapi::dpl::execution::make_device_policy<class dev_sort>(q);
std::sort(policy, d, d + n);
int sum = std::reduce(policy, d, d + n);
q.wait();
```

```cpp
// SYCL buffer via oneapi::dpl::begin/end
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
sycl::queue q;
sycl::buffer<int> buf{host_data};
auto b = oneapi::dpl::begin(buf);
auto e = oneapi::dpl::end(buf);
auto policy = oneapi::dpl::execution::make_device_policy<class buf_sort>(q);
std::sort(policy, b, e);
int sum = std::reduce(policy, b, e);
```

### Key gotchas
- Prefix every built-in with `sycl::`; bare `sqrt()`/`log()` may bind elsewhere or fail to use the SYCL built-in.
- Never rely on `using namespace sycl;`; qualify `std::` and `sycl::` explicitly.
- Never assert bit-equality between SYCL math built-ins and `std::` counterparts.
- `std::` in device code is a compiler extension, not SYCL-guaranteed; verify per target.
- Offload only `par_unseq` policies; `par` and others are unsafe on SYCL devices.
- Buffer algorithms need `<oneapi/dpl/iterator>` plus `oneapi::dpl::begin`/`end`; USM does not.
- Keep policy and USM allocation on the same queue/context; mismatch is undefined behavior.
- `q.wait()` before any host access to USM results.
- oneDPL does not service async SYCL errors; attach a queue async handler.
- Reuse existing SYCL buffers; host-side iterators force temporary-buffer copies.
- Give each policy a unique `class` kernel name to avoid kernel collisions.
