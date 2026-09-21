# oneMKL Device API (In-Kernel) Patterns

## Domain & Purpose

oneMKL domain: **random number generation (RNG)**, device API — engines and distributions
constructed and consumed *inside* a SYCL kernel rather than on the host. Two samples
establish the pattern: Monte Carlo Pi estimation and Multiple Simple Random Sampling
without replacement ("lottery"). Device API header: `"oneapi/mkl/rng/device.hpp"` (the
host/USM variants instead include `"oneapi/mkl.hpp"` — verified in `mc_pi_usm.cpp:21` and
`lottery_usm.cpp:19`).

## Problem & Math

**Monte Carlo Pi.** Draw `n_points` random 2D points in the unit square; the fraction
falling inside the quarter-circle equals `pi/4`.

```
estimated_pi = n_under_curve / ((double)n_points) * 4.0;
// "under curve" iff sycl::length(r) <= 1.0f  (code uses <=, while the source comment
// above the containment test writes the inequality as x ^ 2 + y ^ 2 < 1.0f)
```

**Lottery (partial Fisher–Yates shuffle).** For each experiment, start `local_buf`
holding the naturals `1..N`, then for `i = 0..M-1` draw a uniform `res` in `[0,1)`
and swap:

```cpp
auto j = i + (size_t)(res * (float)(n - i));
std::swap(local_buf[i], local_buf[j]);
```

Each of `num_exp` experiments yields `M` unique values from `1..N`. Defaults
`M = 6`, `N = 49`, `num_exp = 11969664` (`1 <= M <= N`, per README).

## oneMKL Routines Used

All names below appear verbatim in the sources. Both files do `using namespace oneapi;`,
so call sites are also written `mkl::rng::device::...`.

**Device engine (vector case, `mc_pi_device_api.cpp`, 2-wide):**

```cpp
mkl::rng::device::philox4x32x10<vec_size> engine(seed, id_global * count_per_thread * vec_size);
mkl::rng::device::uniform distr;
r = mkl::rng::device::generate(distr, engine);
```

- `vec_size` is `constexpr size_t vec_size = 2;` — the engine's explicit template argument.
- Constructor arguments in order: `(seed, offset)`. `seed = 7777`; offset is a per-work-item stream position.
- `uniform distr;` takes no arguments here; source comment: "by default float, a = 0.0f, b = 1.0f".
- The result of `generate(distr, engine)` is assigned to `sycl::vec<float, vec_size> r;`.
  The source never names the return type; it only shows the result is assignable to that
  vector type.

**Device engine (scalar case, `lottery_device_api.cpp`):**

```cpp
oneapi::mkl::rng::device::philox4x32x10 engine(seed, id * m);
oneapi::mkl::rng::device::uniform distr;
auto res = oneapi::mkl::rng::device::generate(distr, engine);
```

- `philox4x32x10` carries **no** explicit template argument; the result of `generate` is
  held by `auto` (used as a scalar `float` in `res * (float)(n - i)`). Constructor args:
  `(seed, offset)`, `seed = 777`, offset `id * m`.

Summary of identifiers actually observed:

| Identifier | Observed form |
|---|---|
| `oneapi::mkl::rng::device::philox4x32x10` | `philox4x32x10<vec_size>` and bare `philox4x32x10` |
| `oneapi::mkl::rng::device::uniform` | `uniform distr;` (no template args) |
| `oneapi::mkl::rng::device::generate` | `generate(distr, engine)` — 2 arguments |
| Header | `#include "oneapi/mkl/rng/device.hpp"` |

Engine/distribution objects are constructed **inside the kernel lambda** in both samples:
in `mc_pi_device_api.cpp` inside the `parallel_for` lambda passed to `sycl::handler`; in
`lottery_device_api.cpp` inside the `nd_item` lambda. The lottery launch is
`sycl::nd_range<1>(num_exp, 1)`, i.e. a local range of 1, so each work-group holds a
single work-item and the engine is effectively one per experiment (group), keyed on
`item.get_group(0)`.

## Key Code Patterns

**Queue construction with an async exception handler.** Both samples:

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

No property list is passed — the samples do not request an in-order queue.

**Buffer + `get_access` + host copy-back via scope.** The pi sample accumulates into a
1-element host `size_t` through a buffer, and reads the host variable only after the
buffer's scope has closed:

```cpp
size_t n_under_curve = 0;
{
    sycl::buffer<size_t, 1> count_buf(&n_under_curve, 1);
    q.submit([&] (sycl::handler& h) {
        auto count_acc = count_buf.template get_access<sycl::access::mode::write>(h);
        /* ... kernel ... */
    });
}   // buffer destroyed here -> data copied back to n_under_curve
estimated_pi = n_under_curve / ((double)n_points) * 4.0;
```

**Device-wide atomic accumulation.** Each work-item adds its private count:

```cpp
sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_counter { count_acc[0] };
// ...
atomic_counter.fetch_add(count);
```

**Per-work-item stream offset (deterministic, non-overlapping streams).** `mc_pi` gives
each work-item a disjoint block of the engine's stream:

```cpp
constexpr size_t count_per_thread = 32;
constexpr size_t vec_size = 2;
h.parallel_for(sycl::range<1>(n_points / (count_per_thread * vec_size / 2)),
[=](sycl::item<1> item) {
    size_t id_global = item.get_id(0);
    mkl::rng::device::philox4x32x10<vec_size> engine(seed, id_global * count_per_thread * vec_size);
```

Each work-item runs `count_per_thread` iterations and each `generate` call fills the
whole `vec_size`-wide vector, so the offset stride `count_per_thread * vec_size` equals
the per-work-item consumption. The range divisor is
`count_per_thread * vec_size / 2` = 32, i.e. `count_per_thread` points per work-item
(each 2D point consumes 2 = `vec_size` components; `vec_size` is 2 here).

**Per-work-group stream offset + local memory (lottery).** One work-group per experiment;
`local_buf` is `n` elements of `size_t`:

```cpp
sycl::local_accessor<size_t> local_buf(sycl::range<1>{n}, h);
h.parallel_for(sycl::nd_range<1>(num_exp, 1),
    [=](sycl::nd_item<1> item) {
    size_t id = item.get_group(0);
    // Let buf contain natural numbers 1, 2, ..., N
    for (size_t i = 0; i < n; ++i) {
        local_buf[i] = i + 1;
    }
    // Create an object of basic random number generator (engine)
    oneapi::mkl::rng::device::philox4x32x10 engine(seed, id * m);
    // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
    oneapi::mkl::rng::device::uniform distr;
```

Each group consumes exactly `m` engine values, matching the offset stride `id * m`.
Results land in the host buffer via `res_acc[id * m + i] = local_buf[i];`.

**Launch guarded by `try`/`catch(...)`.** `sycl::queue q(sycl::default_selector_v, exception_handler);`
and the kernel launch sit inside a `try` whose `catch (...)` prints `"Failure"` and calls
`std::terminate()`. No explicit queue wait and no event `wait_and_throw()` call appears in
either device-API file; completion is enforced by buffer scope exit. (The sibling
`mc_pi_usm.cpp`, `lottery.cpp` and `lottery_usm.cpp` do call `.wait_and_throw()`.)

## Build & Run

No `CMakeLists.txt` exists in either sample directory (the only CMake files in the sample
tree are under `american_options/`), so the makefiles are the build contract.

**Environment:** source the oneAPI `setvars` script first
(Linux: `. /opt/intel/oneapi/setvars.sh` or `. ~/intel/oneapi/setvars.sh`).

**Linux (GNU make).** Both `GNUmakefile` files define:

```
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng
```

`random_sampling_without_replacement/GNUmakefile` then defines:

```
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations
```

while `monte_carlo_pi/GNUmakefile` appends `$(MKL_LIBS)`:

```
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations $(MKL_LIBS)
```

The per-binary recipe is the same in both files (target `mc_pi_device_api` shown):

```
mc_pi_device_api: mc_pi_device_api.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
```

Targets: `default` and `all` both alias `run`; `run` depends on the three binaries
(`mc_pi mc_pi_usm mc_pi_device_api`, or `lottery lottery_usm lottery_device_api`) and
executes them; `clean` removes the binaries. `MKL_LIBS` is referenced by
`monte_carlo_pi/GNUmakefile` but defined nowhere in these files.

**Windows (nmake).** `icx-cl -fsycl` with:

```
/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=rng /DMKL_ILP64 /EHsc
-fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib
```

Recipe form: `icx-cl -fsycl mc_pi_device_api.cpp /Femc_pi_device_api.exe $(DPCPP_OPTS)`.

**Device selection.** Computation runs on the default SYCL device
(`sycl::default_selector_v`). README: set `ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or
`"*:gpu"` to pick the device.

**Run + arguments.**
- `./mc_pi_device_api [n_points]` — optional `argv[1]` parsed with `atol`; `0` falls back
  to `n_samples = 120'000'000`.
- `./lottery_device_api [m] [n] [num_exp]` — read only when `argc >= 4`; any of
  `m == 0 || n == 0 || num_exp == 0 || m > n` resets all three to the defaults.

**Expected output (verbatim from README).**

```
./mc_pi_device_api

Monte Carlo pi Calculation Simulation
Device Api
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14159
Exact value of Pi = 3.14159
Absolute error = 5.95359e-06

TEST PASSED
```

```
./lottery_device_api

Multiple Simple Random Sampling without replacement
Device Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

TEST PASSED
```

The pi program prints `TEST FAILED` and returns `1` when
`abs_error > 1.0e-4`; the lottery program returns `1` on a failed duplicate check.

## Gotchas & Invariants

- **Buffer scope is the synchronization.** `n_under_curve` / `result_vec` are read on the
  host only after the enclosing `{ ... }` block destroys the `sycl::buffer`. Moving the
  host read inside that block yields stale/racy data.
- **Grid divisibility (pi).** The range is
  `n_points / (count_per_thread * vec_size / 2)` = `n_points / 32` (integer division),
  but the estimate divides by the original `n_points`. `n_points` must therefore be a
  multiple of `count_per_thread * vec_size / 2`; the default `120000000` is. A
  non-multiple silently counts fewer points than the denominator assumes.
- **Offset stride must match consumption.** The pi engine offset is
  `count_per_thread * vec_size` and the per-item consumption is exactly that. Changing
  `count_per_thread` or `vec_size` without changing the multiplier makes concurrent
  work-items draw overlapping streams. Same invariant in lottery: offset `id * m`,
  consumption `m` values per group.
- **`local_accessor` size = `n` (49 by default), one work-group per experiment**
  (`sycl::nd_range<1>(num_exp, 1)`). `n` is bounded by device local memory.
- **`<=` not `<`** in the pi containment test: `if(sycl::length(r) <= 1.0f)`; the source
  comment above it says `< 1.0f`.
- **`res` is assumed in `[0,1)`.** `j = i + (size_t)(res * (float)(n - i))` has no bounds
  check; `(float)` truncation keeps `j` in `[i, n-1]` only under that assumption.
- **`-qmkl-sycl-impl=rng`** is present in both samples' compile flags alongside
  `-DMKL_ILP64` / `-qmkl-ilp64`; the source does not state whether it is mandatory.
- **No explicit queue wait** and no in-order queue property are used in the device-API
  files; correctness relies purely on buffer scoping plus the async exception handler.
- **Source quirk in the lottery verifier:** it reports failure when duplicates exist
  (`adjacent_find(...) != first_iter + m`) **and** `count_if(..., val > n && val >= 0) == 0`.
  The comment says "if all elements are in the [0, n] range", but for `size_t` the
  predicate `val > n && val >= 0` is not a range check for `[0, n]`, and the sample's
  shuffle yields values in `1..n`. Treat this check as sample-specific, not a validated
  invariant (also note it sorts ranges that begin at `result_vec.begin() + m * i`).

## Explicit gaps

- The **full signature/overload set of `oneapi::mkl::rng::device::generate` is not
  shown.** Only the two-argument call form `generate(distr, engine)` appears; whether
  overloads accept output pointers, counts, or explicit result types is not established
  here.
- The **template parameter list and defaults of `philox4x32x10`** are not shown — only an
  explicit `<vec_size>` instantiation and a bare instantiation appear. Nothing establishes
  what the template parameter means or what its default is.
- The **scalar type of `uniform`** is asserted only by a source comment ("by default
  float, a = 0.0f, b = 1.0f"); no explicit template argument or member is written, and no
  way to set `a` / `b` appears in these files.
- **The stream-advance semantics of a `generate` call are not stated.** The statements
  that one call consumes `vec_size` values and that the offset stride must equal
  consumption are inferred from the offset arithmetic and the `sycl::vec<float, vec_size>`
  result, not from any documented behavior in the source.
- **No other engine constructors, engines, or distributions** are demonstrated; the README
  only states generically that "oneMKL provides many other generators and distributions."
- **No device-side error/status handling** appears: nothing shows a `sycl::exception` or
  status code originating inside a kernel, and no `try`/`catch` exists within kernel code.
- **No host-side device API "finalize"/destroy/reset call and no scratchpad/workspace
  query** appears in either file.
- **Queue ordering is not established** — no in-order queue property and no explicit
  queue-wait or event-wait call in the device-API files, so their ordering assumptions
  beyond buffer scoping are unstated.
- **That buffer-scope exit synchronizes the device work and copies data back** is standard
  SYCL semantics; the source files never state it, yet the host reads depend on it.
- **`-fsycl-device-code-split=per_kernel` and `-fno-sycl-early-optimizations`** appear
  only as sample build flags; the source does not say whether they are required.
- **`MKL_LIBS`** is referenced by `monte_carlo_pi/GNUmakefile` but defined nowhere in
  these files, so the effective link line is not fully determined by the source.
- **No CMake build** is provided for either sample, so no CMake invocation can be quoted.

## Source Map

- `monte_carlo_pi/mc_pi_device_api.cpp` — in-kernel RNG pi estimate: `philox4x32x10<2>` + `uniform` + `generate`, `sycl::buffer` accumulation with `sycl::atomic_ref`, `default_selector_v` queue with async exception handler.
- `monte_carlo_pi/GNUmakefile` — Linux build: `icpx $< -fsycl -o $@ $(DPCPP_OPTS)`, `-DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl=rng`, `run`/`clean` targets, runs all three binaries; `DPCPP_OPTS` appends `$(MKL_LIBS)`.
- `monte_carlo_pi/makefile` — Windows `nmake` build using `icx-cl -fsycl` and `/Qmkl-sycl-impl=rng`.
- `monte_carlo_pi/README.md` — algorithm narrative, build/run steps, `ONEAPI_DEVICE_SELECTOR` guidance, expected output for `mc_pi`, `mc_pi_usm`, `mc_pi_device_api`.
- `random_sampling_without_replacement/lottery_device_api.cpp` — in-kernel partial Fisher–Yates lottery: scalar `philox4x32x10` seeded with `(seed, id * m)`, `sycl::local_accessor`, one work-group per experiment, duplicate/range verifier.
- `random_sampling_without_replacement/GNUmakefile` — Linux build flags (no `$(MKL_LIBS)`) and `run`/`clean` targets for `lottery`, `lottery_usm`, `lottery_device_api`.
- `random_sampling_without_replacement/makefile` — Windows `nmake` build (plus commented-out Linux lines).
- `random_sampling_without_replacement/README.md` — partial Fisher–Yates description, parameters `M=6`, `N=49`, expected output for all three programs.
- `random_sampling_without_replacement/lottery_usm.cpp` — contrast only: host (USM) API using `#include "oneapi/mkl.hpp"`, `oneapi::mkl::rng::philox4x32x10 engine(q, seed)`, `oneapi::mkl::rng::generate(distr, engine, m * num_exp, rng_buf)`, `event.wait_and_throw()`.
