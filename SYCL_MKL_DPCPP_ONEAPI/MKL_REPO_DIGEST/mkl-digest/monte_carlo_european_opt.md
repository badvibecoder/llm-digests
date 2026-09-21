# Monte Carlo European Option Pricing

## Domain & Purpose

oneMKL domain: **Random Number Generation (RNG)** — both the host API (`oneapi/mkl.hpp`,
`oneapi::mkl::rng`) and the device-side API (`oneapi/mkl/rng/device.hpp`,
`oneapi::mkl::rng::device`). BLAS/LAPACK/FFT/VM are not used.

Prices European call options by Monte Carlo simulation of a stochastic stock-price model, then
validates the prices against a host-side Black-Scholes reference.

## Problem & Math

Path model (log2 domain, matching the `M_LOG2E`-scaled constants in `montecarlo.hpp`):

```
RLog2E   = -risk_free * M_LOG2E
MuLog2E  = M_LOG2E * (risk_free - 0.5 * volatility * volatility)
VLog2E   = M_LOG2E * volatility

per path step (vectorized, VEC_SIZE lanes):
  VBySqrtT = VLog2E * sycl::sqrt(option_years)
  MuByT    = MuLog2E * option_years
  gaussian distr(MuByT, VBySqrtT)
  rng_val  = stock_price * sycl::exp2(gaussian_sample) - option_strike
  payoff   = sycl::max(rng_val, 0)

per option, summing the work-group reduction over local_size work-items
(block_n * VEC_SIZE samples each, i.e. local_size * block_n * VEC_SIZE samples in total,
which equals path_length under the default constants):
v0 = sum(payoff),  v1 = sum(payoff^2)
call_result     = exp2(RLog2E * option_years) * v0 / path_length
std_dev         = sqrt((path_length * v1 - v0 * v0) / (path_length * (path_length - 1)))
call_confidence = exp2(RLog2E * option_years) * std_dev * (1.96 / sqrt(path_length))
```

Constants (verbatim from `src/montecarlo.hpp`): `risk_free = 0.06f`, `volatility = 0.10f`, `num_options = 384000`, `path_length = 262144`, `num_iterations = 5`.

Reference check uses the Black-Scholes formula with
`N(d) = 1/2 + 1/2 * std::erf(d / std::sqrt(2.))` and returns `S * N_d1 - L * std::exp(-r * t) * N_d2`.

## oneMKL Routines Used

Headers: `#include <oneapi/mkl.hpp>` and `#include <oneapi/mkl/rng/device.hpp>`.
Alias: `namespace mkl_rng = oneapi::mkl::rng;`

**Host generator engine types** (selected by preprocessor macro):

```cpp
#if USE_PHILOX
using EngineTypeHost   = mkl_rng::philox4x32x10;
using EngineTypeDevice = mkl_rng::device::philox4x32x10<VEC_SIZE>;
#elif USE_MRG
using EngineTypeHost   = mkl_rng::mrg32k3a;
using EngineTypeDevice = mkl_rng::device::mrg32k3a<VEC_SIZE>;
#else
using EngineTypeHost   = mkl_rng::mcg59;
using EngineTypeDevice = mkl_rng::device::mcg59<VEC_SIZE>;
#endif
```

Host engine construction — queue first, seed second:

```cpp
EngineTypeHost engine(
#if !INIT_ON_HOST
    my_queue,
#else
    sycl::queue{sycl::cpu_selector_v},
#endif
    rand_seed); // random number generator object
```

**Host generate** — `oneapi::mkl::rng::generate(distribution, engine, count, usm_ptr)`; assigned
with `auto` in the source, so the return type is not literally named (each yields an event used as
a kernel dependency):

```cpp
auto rng_event_1 = mkl_rng::generate(mkl_rng::uniform<DataType>(5.0, 50.0), engine, num_options, h_stock_price_ptr);
auto rng_event_2 = mkl_rng::generate(mkl_rng::uniform<DataType>(10.0, 25.0), engine, num_options, h_option_strike_ptr);
auto rng_event_3 = mkl_rng::generate(mkl_rng::uniform<DataType>(1.0, 5.0), engine, num_options, h_option_years_ptr);
```

Distribution `oneapi::mkl::rng::uniform<DataType>` is constructed with two numeric bounds at the
call sites; ranges used: stock price `(5.0, 50.0)`, strike `(10.0, 25.0)`, option years `(1.0, 5.0)`.

**Device generate** — `oneapi::mkl::rng::device::generate(distribution, engine_state)`, called
inside a SYCL kernel, returning a vector indexed by lane (the source indexes it over
`lane < VEC_SIZE`). Distribution `oneapi::mkl::rng::device::gaussian<DataType>` is constructed at
the call site as `(MuByT, VBySqrtT)`:

```cpp
mkl_rng::device::gaussian<DataType> distr(MuByT, VBySqrtT);
...
auto rng_val_vec = mkl_rng::device::generate(distr, local_state);
```

**Device engine state construction** (in the init kernel), engine differs per generator:

```cpp
#if USE_MRG
    constexpr std::uint32_t seed = 12345u;
    rng_states[id] = EngineTypeDevice({ seed, seed, seed, seed, seed, seed }, { 0, (4096 * id) });
#else
    rng_states[id] = EngineTypeDevice(rand_seed, id * ITEMS_PER_WORK_ITEM * VEC_SIZE * block_n);
#endif
```

`rand_seed` is `constexpr int rand_seed = 777;`.

## Key Code Patterns

**Queue and USM shared allocation via `sycl::usm_allocator`** — host vectors that the device
writes directly, no explicit copy step:

```cpp
sycl::queue my_queue;
sycl::usm_allocator<DataType, sycl::usm::alloc::shared> alloc(my_queue);
std::vector<DataType, decltype(alloc)> h_call_result(num_options, alloc);
std::vector<DataType, decltype(alloc)> h_call_confidence(num_options, alloc);
std::vector<DataType, decltype(alloc)> h_stock_price(num_options, alloc);
DataType* h_call_result_ptr = h_call_result.data();
```

`h_option_strike` and `h_option_years` are declared identically, and each vector gets its own
`.data()` pointer. `decltype(alloc)` is required as the vector's allocator argument, passed as the
second constructor argument.

**Device-only USM allocation with an explicit deleter** (state array; not on the host):

```cpp
auto deleter = [my_queue](auto* ptr) {sycl::free(ptr, my_queue);};
auto rng_states_uptr = std::unique_ptr<EngineTypeDevice, decltype(deleter)>(sycl::malloc_device<EngineTypeDevice>(n_states, my_queue), deleter);
auto* rng_states = rng_states_uptr.get();
```

**Grid sizing** — `local_size = 256`, `VEC_SIZE = 8`, `ITEMS_PER_WORK_ITEM = 4`:

```cpp
const std::size_t global_size = (num_options * local_size) / ITEMS_PER_WORK_ITEM; // It requires num_options be divisible by ITEMS_PER_WORK_ITEM
const int block_n = path_length / (local_size * VEC_SIZE);
```

**State-init kernel with explicit event dependencies** — the `std::vector<sycl::event>` form of
`parallel_for`, terminated by `wait_and_throw()` so the RNG host results are ready:

```cpp
my_queue.parallel_for<k_initialize_state<DataType>>(
    sycl::range<1>(n_states),
    std::vector<sycl::event>{rng_event_1, rng_event_2, rng_event_3},
    [=](sycl::item<1> idx) {
        auto id = idx[0];
        ...
})
.wait_and_throw();
```

Both lambdas are given explicit kernel names via the template argument
(`k_initialize_state<DataType>`, `k_MonteCarlo<DataType, ITEMS_PER_WORK_ITEM>`), forward-declared
at file scope with `// can be useful for profiling`. The **main kernel** takes an `nd_range` and
a `sycl::nd_item<1>`:

```cpp
my_queue.parallel_for<k_MonteCarlo<DataType, ITEMS_PER_WORK_ITEM>>(
    sycl::nd_range<1>({global_size}, {local_size}),
    [=](sycl::nd_item<1> item)
    {
        auto local_state = rng_states[item.get_global_id()];
        ...
})
.wait_and_throw();
```

`local_state` is a by-value copy of `rng_states[item.get_global_id()]`; anything `device::generate`
does to it is never written back to `rng_states`.

**Work-item / work-group mapping and reduction:**

```cpp
const std::size_t i_options = item.get_group_linear_id() * ITEMS_PER_WORK_ITEM + i;
v0 = sycl::reduce_over_group(item.get_group(), v0, std::plus<>());
v1 = sycl::reduce_over_group(item.get_group(), v1, std::plus<>());
if(item.get_local_id() == 0)
{
    h_call_result_ptr[i_options] = call_result;
    h_call_confidence_ptr[i_options] = call_confidence;
}
```

Reduction is plain SYCL (`sycl::reduce_over_group`) over per-work-item partials, not an oneMKL
routine; only local id 0 stores, so the group-reduced value is written once.

**Precision dispatch** reads `sycl::aspect::fp64` from the default device, and one
`catch (sycl::exception e)` wraps the whole `run()` body:

```cpp
sycl::queue test_queue;
is_fp64 = test_queue.get_device().has(sycl::aspect::fp64);
```

```cpp
if (is_fp64) {
    run<double>();
} else {
    std::cout<<"Warning: could not find a device with double precision support. Single precision is used."<<std::endl;
    run<float>();
}
```

```cpp
catch (sycl::exception e) {
    std::cout << e.what();
    exit(1);
}
```

## Build & Run

Linux (`GNUmakefile`, target `montecarlo`, single translation unit):

```
icpx src/montecarlo_main.cpp -o montecarlo -DMKL_ILP64 $(GENERATOR) -qmkl-ilp64 -qmkl-sycl-impl=rng -Wall -Wformat-security -Werror=format-security -fsycl -DINIT_ON_HOST=$(init_on_host)
```

with `GENERATOR` empty by default, `-DUSE_MRG` for `generator=mrg`, `-DUSE_PHILOX` for
`generator=philox`; `init_on_host ?= 0` and `generator ?= mcg59`. Allowed values are validated:
`mrg philox mcg59` (error: `"You use unknown generator. Please, use mrg philox or mcg59 (default)"`).

Windows (`makefile`, nmake syntax, target `montecarlo` producing `montecarlo.exe`, compiler `icx`):

```
DPCPP_OPTS=/I"$(MKLROOT)\include" /DMKL_ILP64 $(GENERATOR) /EHsc -fsycl $(INIT_ON_HOST) /Qmkl-ilp64 /Qmkl-sycl-impl=rng OpenCL.lib
icx src/montecarlo_main.cpp /omontecarlo.exe $(DPCPP_OPTS)
```

with `GENERATOR=/DUSE_MRG` or `/DUSE_PHILOX` and `INIT_ON_HOST=/DINIT_ON_HOST=1`.

Targets: `make` (build `montecarlo`), `make clean` (`-rm -f montecarlo`), `make generator=mrg`,
`make generator=philox`, `make init_on_host=1`. Windows adds `pseudo` (`clean` then `all`) and uses
the same targets under `nmake`.

Environment: source `setvars` first (`. /opt/intel/oneapi/setvars.sh` for system-wide installs).
Device selection uses `ONEAPI_DEVICE_SELECTOR` set to `"*:cpu"` or `"*:gpu"`; default is the
default SYCL device. No CMakeLists.txt is present in this sample.

Expected output shape (README example, MCG59 default):

```
MonteCarlo European Option Pricing in Double precision
Pricing 384000 Options with Path Length = 262144, sycl::vec size = 8, Options Per Work Item = 4 and Iterations = 5
Completed in 67.6374 seconds. Options per second = 22709.3
Running quality test...
L1_Norm          = 0.000480579
Average RESERVE  = 12.9099
Max Error        = 0.123343
TEST PASSED!
```

The printed generator string is `PHILOX4x32x10`, `MRG32k3a`, or `MCG59`, chosen by the same
`USE_PHILOX` / `USE_MRG` macros as the engine type.

## Gotchas & Invariants

- **Divisibility:** `global_size = (num_options * local_size) / ITEMS_PER_WORK_ITEM` — the comment
  states it requires `num_options` divisible by `ITEMS_PER_WORK_ITEM`. `block_n = path_length /
  (local_size * VEC_SIZE)` is integer division, so a `path_length` that is not an exact multiple of
  `local_size * VEC_SIZE` (256 * 8 = 2048) truncates `block_n`; the accumulated sample count
  (`local_size * block_n * VEC_SIZE`) then falls short of `path_length` while the normalization
  still divides by `path_length`. The `montecarlo.hpp` comments read `//Should be > 1`
  (`num_options`) and `//Should be > 16` (`path_length`).
- **`VEC_SIZE` and `ITEMS_PER_WORK_ITEM`** come from `#define` defaults 8 and 4, overridable only
  by predefinition (`#ifndef` guards). They must stay consistent between the device engine's
  vector width (`device::mcg59<VEC_SIZE>`) and the `lane < VEC_SIZE` payoff loop.
- **Generator macro is mandatory for the intended engine:** building without any of
  `USE_PHILOX`/`USE_MRG` silently selects `mcg59` via the `#else` branch.
- **State offsets must not collide:** each non-MRG work-item state is offset by
  `id * ITEMS_PER_WORK_ITEM * VEC_SIZE * block_n`; `mrg32k3a` uses `{ 0, (4096 * id) }` with a
  fixed `seed = 12345u`. `n_states = global_size`, i.e. one state per global work-item.
- **Free with the same context:** device USM is released by `sycl::free(ptr, my_queue)` through a
  `unique_ptr` deleter; the deleter lambda captures `my_queue` by value (in the source it is written
  `[my_queue](auto* ptr)`).
- **Timing excludes iteration 0:** `if(i != 0) total_time += tt.duration();` and throughput divides
  by `num_iterations - 1` — the first iteration is warm-up and is not counted.
- **Pass criterion is the reserve ratio, not the error:** `check()` prints `TEST PASSED` and only
  fails via `exit(1)` when `sum_reserve / num_options <= 1.0f`. `sum_reserve` accumulates
  `h_call_confidence[opt] / delta` only when `delta > 1e-6`; the README example output shows
  `TEST PASSED!` with a trailing `!` while `montecarlo.hpp` prints `"TEST PASSED"` without one.
- **Out-of-order queue:** removing the init kernel's event dependency vector, its
  `wait_and_throw()`, or the main kernel's `wait_and_throw()` races the
  `h_stock_price_ptr` / `h_option_strike_ptr` / `h_option_years_ptr` reads.

## Explicit gaps

- The exact declared signatures/return types of `oneapi::mkl::rng::generate` and
  `oneapi::mkl::rng::device::generate` are **not** in these files — both are consumed with `auto`.
  Do not assume an argument list beyond the usage verbatim above.
- `oneapi::mkl::rng::uniform<DataType>`'s constructor parameter list and default template
  arguments are only evidenced by the `(5.0, 50.0)`-style call sites; the header is not in the
  workspace. Likewise the declared return type of `device::generate` is inferred only from the
  `rng_val[lane]` element access (`lane < VEC_SIZE`).
- The meaning of the `mrg32k3a` device-state arguments `{ seed x6 }` and `{ 0, (4096 * id) }` is not
  documented in these files, nor is whether `device::generate` advancing the by-value `local_state`
  is intended to persist across the outer `num_iterations` loop.
- No CMake build path exists in this sample; only `GNUmakefile` (Linux) and `makefile` (Windows).
  `-qmkl-sycl-impl=rng` / `/Qmkl-sycl-impl=rng` and `-qmkl-ilp64` / `/Qmkl-ilp64` are shown
  verbatim in those files but their semantics are not explained there.
- README's claim that prices are "averaged using reduction functions" does not name an oneMKL
  routine; the source uses the SYCL `sycl::reduce_over_group` only.
- The `sycl::vec` type named in the README's output text is never spelled out in the source; only
  element access `rng_val[lane]` and `VEC_SIZE` appear. Only `sycl::exception` is caught, so
  behavior for other oneMKL error types is not shown.
- The README Purpose paragraph describes estimating both call and put option values, but the source
  computes and stores call results only (`h_call_result`, `h_call_confidence`); no put-option
  computation appears in these files.
- The README's example first output line (`MonteCarlo European Option Pricing in Double precision`)
  omits the `" using " << <generator> << " generator."` suffix that the current source prints, so
  the README sample output does not match the source format byte-for-byte.

## Source Map

- `README.md` — purpose, generator/`init_on_host` build options, `ONEAPI_DEVICE_SELECTOR`, example output.
- `GNUmakefile` — Linux `icpx` build, `generator`/`init_on_host` variables, `all`/`clean` targets.
- `makefile` — Windows `nmake` `icx` build, same variables in `/D` form, `pseudo` target.
- `src/montecarlo_main.cpp` — queue/USM setup, host `mkl_rng::generate` calls, device state init kernel, main Monte Carlo kernel, precision dispatch, exception handling.
- `src/montecarlo.hpp` — simulation constants, precision/`VEC_SIZE`/`ITEMS_PER_WORK_ITEM` macros, log2 constants, Black-Scholes reference and `check()` quality test.
- `src/timer.hpp` — `timer` class over `std::chrono::steady_clock` with `start`/`stop`/`duration`.
