# Monte Carlo Pi Estimation

## Domain & Purpose
oneMKL **RNG** (random number generation) domain only. Three programs estimate &pi; by random
sampling, each using a different RNG calling convention: host API writing into a `sycl::buffer`
(`mc_pi.cpp`), host API writing into USM memory (`mc_pi_usm.cpp`), and the RNG *device* API
called from inside a kernel (`mc_pi_device_api.cpp`).

## Problem & Math
Sample `n_points` 2D points uniformly in the unit square; the source comment gives the default
distribution bounds as `a = 0.0f, b = 1.0f`. Count points accepted by the code's test
`sycl::length(r) <= 1.0f` (the neighboring source comment writes `x ^ 2 + y ^ 2 < 1.0f`). The
ratio of areas is `pi/4`, so:

```
estimated_pi = n_under_curve / ((double)n_points) * 4.0;
abs_error    = std::fabs(pi - estimated_pi);   // pi = 3.1415926535897932384626433832795
```

Test criterion: `if(abs_error > 1.0e-4) -> "TEST FAILED"`, `return 1`; else `"TEST PASSED"`, `return 0`.

## oneMKL Routines Used
Headers: host API `#include "oneapi/mkl.hpp"`; device API `#include "oneapi/mkl/rng/device.hpp"`.
Both files do `using namespace oneapi;` and then call everything through `mkl::`.

Host-side engine construction (identical in `mc_pi.cpp` and `mc_pi_usm.cpp`):

```cpp
static const auto seed = 7777;
// Create an object of basic random number generator (engine)
mkl::rng::philox4x32x10 engine(q, seed);
// Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
mkl::rng::uniform distr;
```

Host-side generation, buffer overload (`mc_pi.cpp`; return value discarded => declared return
type not established, see gaps):

```cpp
sycl::buffer<float, 1> rng_buf(n_points * 2);
mkl::rng::generate(distr, engine, n_points * 2, rng_buf);
```

Host-side generation, USM overload (`mc_pi_usm.cpp`; returns something usable as a kernel
dependency, named `event`):

```cpp
float* rng_ptr = sycl::malloc_shared<float>(n_points * 2, q);
auto event = mkl::rng::generate(distr, engine, n_points * 2, rng_ptr);
```

Device-side engine + distribution + generation (`mc_pi_device_api.cpp`), all inside the kernel
lambda, `vec_size = 2`:

```cpp
mkl::rng::device::philox4x32x10<vec_size> engine(seed, id_global * count_per_thread * vec_size);
mkl::rng::device::uniform distr;
...
r = mkl::rng::device::generate(distr, engine);   // r is sycl::vec<float, vec_size>
```

Call shape as used: host `mkl::rng::generate(distr, engine, n_points * 2, destination)` —
distribution first, engine second, requested count third, destination (buffer or pointer)
fourth. The device call `mkl::rng::device::generate(distr, engine)` takes no count argument and
returns one vector of `vec_size` floats per call. Engine construction order:
`philox4x32x10(q, seed)` (queue, seed), and `device::philox4x32x10<vec_size>(seed, <offset
expression>)` whose second argument in the source is `id_global * count_per_thread * vec_size`.

Templating shown: only `device::philox4x32x10<vec_size>` with `vec_size = 2`; the non-device
`philox4x32x10` and `uniform` are used **without** explicit template arguments.

## Key Code Patterns
Queue with async exception handler (all three files, verbatim apart from the indentation of the
enclosing `main`):

```cpp
auto exception_handler = [&](sycl::exception_list exceptions) {
    for(std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
            std::terminate();
        }
    }
};
sycl::queue q(sycl::default_selector_v, exception_handler);
```

Ordering — the queue is constructed with only the selector and the exception handler, so
dependencies are expressed explicitly or via the buffer accessor model:
* buffer version: the reduction kernel reads `rng_buf` through
  `rng_buf.template get_access<sycl::access::mode::read>(h)`, which orders it after `generate`.
* USM version: the event returned by `generate` is passed as the second argument of
  `q.parallel_for(sycl::range<1>(...), event, reductor, lambda)` and the whole thing ends in
  `.wait_and_throw()`.
* device version: everything happens inside one kernel; results are accumulated with
  `sycl::atomic_ref`.

Buffer-API reduction with `sycl::reduction` + `sycl::vec<float,2>::load` (`mc_pi.cpp`):

```cpp
auto rng_acc = rng_buf.template get_access<sycl::access::mode::read>(h);
auto reductor = sycl::reduction(count_buf, h, size_t(0), std::plus<size_t>());
h.parallel_for(sycl::range<1>(n_points / count_per_thread), reductor,
    [=](sycl::item<1> item, auto& sum) {
        sycl::vec<float, 2> r;
        ...
        r.load(i + item.get_id(0) * count_per_thread, rng_acc.template get_multi_ptr<sycl::access::decorated::yes>());
        if(sycl::length(r) <= 1.0f) { count++; }
```

USM-API reduction over raw pointers (`mc_pi_usm.cpp`) — `sycl::malloc_host` for the counter,
plain pointer reduction:

```cpp
size_t *n_under_curve = sycl::malloc_host<size_t>(1, q);
*n_under_curve = 0;
auto reductor = sycl::reduction(n_under_curve, size_t(0), std::plus<size_t>{});
q.parallel_for(sycl::range<1>(n_points / count_per_thread), event, reductor,
               [=](sycl::item<1> item, auto& sum) { ... }).wait_and_throw();
...
sycl::free(rng_ptr, q);
sycl::free(n_under_curve, q);
```

Device-API atomic accumulation into a buffer accessor (`mc_pi_device_api.cpp`):

```cpp
auto count_acc = count_buf.template get_access<sycl::access::mode::write>(h);
sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space> atomic_counter { count_acc[0] };
...
atomic_counter.fetch_add(count);
```

`n_under_curve` is a plain `size_t` local in the buffer and device versions, captured by a
`count_buf` placed inside an inner `{...}` block so the buffer destructor completes before the
result is read. The two declarations differ lexically: `mc_pi.cpp` uses
`sycl::buffer<size_t> count_buf{ &n_under_curve, 1 };` (braced init, default dimension), while
`mc_pi_device_api.cpp` uses `sycl::buffer<size_t, 1> count_buf(&n_under_curve, 1);`. The USM
version reads `*n_under_curve` directly.

Command-line argument handling (identical in all three): `n_points` defaults to
`n_samples = 120'000'000`; `if(argc >= 2) { n_points = atol(argv[1]); if(n_points == 0) n_points = n_samples; }`.

## Build & Run
Linux (`GNUmakefile`), `make` builds and immediately runs all three programs; `make clean` removes
them; targets: `default`, `all`, `run`, `clean`, `.PHONY: clean run all`.

```
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations $(MKL_LIBS)

icpx mc_pi.cpp            -fsycl -o mc_pi            $(DPCPP_OPTS)
icpx mc_pi_usm.cpp        -fsycl -o mc_pi_usm        $(DPCPP_OPTS)
icpx mc_pi_device_api.cpp -fsycl -o mc_pi_device_api $(DPCPP_OPTS)
```

Windows (`makefile`, for NMAKE), `nmake` to build/run, `nmake clean`:

```
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=rng /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib
icx-cl -fsycl mc_pi.cpp /Femc_pi.exe $(DPCPP_OPTS)
```
plus `mc_pi_usm.exe` and `mc_pi_device_api.exe` rules; extra target `pseudo: clean run all`.

Environment: source the oneAPI `setvars` script first (README lists `. /opt/intel/oneapi/setvars.sh`
for system-wide installs and `. ~/intel/oneapi/setvars.sh` for private installs).
Device selection: the programs run on the **default SYCL device**; set `ONEAPI_DEVICE_SELECTOR`
to `"*:cpu"` or `"*:gpu"` to choose (README). Optional CLI arg = number of points.

Expected output (README, verbatim shape; `120000000` points):

```
Monte Carlo pi Calculation Simulation
Buffer API            <- README label; mc_pi.cpp actually prints "Buffer Api"
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14106
Exact value of Pi = 3.14159
Absolute error = 0.000530387

TEST PASSED
```
`mc_pi_usm` prints the same numbers with the label `Unified Shared Memory Api`;
`mc_pi_device_api` prints `Device Api` and, in the README example,
`Estimated value of Pi = 3.14159`, `Absolute error = 5.95359e-06`.

## Gotchas & Invariants
* **Point count is truncated, not rounded.** Work-item counts use integer division:
  `n_points / count_per_thread` (`count_per_thread = 32`) in the buffer and USM versions, and
  `n_points / (count_per_thread * vec_size / 2)` (also 32) in the device version. A `n_points`
  that is not a multiple of the divisor silently drops the remainder, while the estimator's
  denominator is still the original `n_points`.
* The buffer/USM kernels consume random values in `sycl::vec<float,2>` chunks, so
  `generate(..., n_points * 2, ...)` is called with twice the point count — one float pair per point.
* The sample fixes `seed = 7777`; the README example output shows the buffer and USM examples
  reporting identical estimates (`3.14106`), while the device-API example reports `3.14159`.
* Device API creates one engine per work item whose second constructor argument is
  `id_global * count_per_thread * vec_size`, i.e. it varies with the global work-item id.
* Device API accumulates from potentially many work items into one location; the source uses
  `sycl::atomic_ref` with `sycl::memory_order::relaxed` and `sycl::memory_scope::device`, and
  increments via `fetch_add`.
* `count_buf`/`count_acc` in the device version is opened `sycl::access::mode::write`; the host
  value `n_under_curve` is only valid after the enclosing buffer scope ends (the inner `{ }`).
* USM allocations must be released: `sycl::free(rng_ptr, q); sycl::free(n_under_curve, q);`.
  `sycl::malloc_shared` is used for the random values and `sycl::malloc_host` for the counter.
* Failure path is `std::terminate()` for both caught async SYCL exceptions and any other
  exception (`catch (...) { std::cout << "Failure" << std::endl; std::terminate(); }`), so a broken
  run aborts rather than returning a diagnostic.
* Both makefiles pass `-DMKL_ILP64` together with `-qmkl-ilp64` and `-qmkl-sycl-impl=rng`; the
  source states no further effect of `-qmkl-sycl-impl=rng` on which oneMKL domains are linked.

## Explicit gaps
* Declared signatures of `oneapi::mkl::rng::generate` are never shown: parameter types, whether
  the buffer overload returns `void` or an event, and the exact USM overload set are not
  established. Only the call forms above (with `auto event = ...` working on the pointer overload)
  are grounded.
* `oneapi::mkl::rng::philox4x32x10`'s constructor parameter types (queue by reference vs value,
  seed type) are not shown — only the call `philox4x32x10 engine(q, seed);`.
* `oneapi::mkl::rng::uniform`'s template parameter list / default arguments are not shown; the
  only evidence for the default type and bounds is the source comment `by default float, a = 0.0f,
  b = 1.0f`. Whether the sampled interval is open or closed at either end is not stated.
* `oneapi::mkl::rng::device::generate`'s full overload set and return type are not shown beyond
  `r = mkl::rng::device::generate(distr, engine)` returning something assignable to
  `sycl::vec<float, 2>`.
* `oneapi::mkl::rng::device::philox4x32x10`'s constructor parameter types (seed type, offset
  type) are not shown; only the two-argument call form. The meaning/units of the second argument
  `id_global * count_per_thread * vec_size`, and the consequence of reusing it across work items,
  are likewise not stated in the source.
* `MKL_LIBS` is referenced in `DPCPP_OPTS` in `GNUmakefile` but never defined there, and
  `GNUmakefile` passes no include path; no CMake build exists in this directory.
* Why `-fsycl-device-code-split=per_kernel` and `-fno-sycl-early-optimizations` are used (and
  whether they are required) is not explained.
* No source statement establishes run-to-run or cross-device reproducibility of `seed = 7777`;
  the README's differing device-API estimate (`3.14159`) versus buffer/USM (`3.14106`) is not
  explained.

## Source Map
* `README.md` — purpose, algorithm sketch, VS Code notes, build/run instructions, `ONEAPI_DEVICE_SELECTOR` usage, example output for all three programs.
* `GNUmakefile` — Linux `icpx` build with `MKL_COPTS`/`DPCPP_OPTS`, targets `mc_pi`, `mc_pi_usm`, `mc_pi_device_api`, `run`, `clean`.
* `makefile` — Windows NMAKE build (`icx-cl`, `.exe` targets, `OpenCL.lib`, `/Qmkl-sycl-impl=rng`), target `pseudo`.
* `mc_pi.cpp` — buffer-API estimate: `mkl::rng::philox4x32x10` + `mkl::rng::uniform` + `mkl::rng::generate` into `sycl::buffer`, `sycl::reduction` count kernel.
* `mc_pi_usm.cpp` — USM-API estimate: same host RNG calls into `sycl::malloc_shared`, event-ordered `parallel_for`, pointer-based `sycl::reduction`.
* `mc_pi_device_api.cpp` — device-API estimate: in-kernel `mkl::rng::device::philox4x32x10<vec_size>` / `device::uniform` / `device::generate`, `sycl::atomic_ref` accumulation.
