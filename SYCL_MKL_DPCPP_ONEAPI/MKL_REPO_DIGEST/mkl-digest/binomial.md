# Binomial Option Pricing (SYCL)

## Domain & Purpose

oneMKL domain: **RNG** (Random Number Generation). A SYCL implementation of a
European-option binomial pricing model that fills a randomly generated portfolio
of stock prices, strikes and maturities with the oneMKL RNG API, then prices every
option on-device.

## Problem & Math

Given per-option inputs `S` (stock price), `X` (strike), `T` (years), constants
`volatility = 0.10f` and `risk_free = 0.06f`, and `num_steps = 2048`:

- CRR lattice: `dt = T/num_steps`, `v_dt = volatility*sqrt(dt)`, `r_dt = risk_free*dt`,
  `i_f = exp(r_dt)`, `df = exp(-r_dt)`, `u = exp(v_dt)`, `d = exp(-v_dt)`,
  `pu = (i_f - d)/(u - d)`, `pd = 1 - pu`, `pu_df = pu*df`, `pd_df = pd*df`.
- Terminal payoff at node `j` (`j = 0..num_steps`), computed as `S*exp(v_dt*(2*j - num_steps)) - X` clamped at 0, i.e. `S * u^j * d^(num_steps-j)`.
- Backward induction `V[j] = pu_df*V[j+1] + pd_df*V[j]`, applied for `j = 0..block_size-1` inside the outer time loop that walks `i` from `num_steps` down to 1; the triangular truncation is the guard `if (block_start <= i)`.
- Correctness check against Black-Scholes (`BlackScholesRefImpl` in `binomial_main.cpp`),
  reported as an L1 norm and a hard-coded pass threshold `errorVal < 5e-4`.

## oneMKL Routines Used

Header and alias used by the sample:

```cpp
#include <oneapi/mkl.hpp>
namespace mkl_rng = oneapi::mkl::rng;
```

The routine names below are the canonical `oneapi::mkl::rng::` qualified forms; in
the source they are spelled through the `mkl_rng` alias (e.g. `mkl_rng::generate`).

**1. `oneapi::mkl::rng::philox4x32x10` — engine object.** Constructed with a queue
and a seed; the README calls it "a Philox 4x32x10 generator ... a lightweight
counter-based RNG well-suited for parallel computing."

```cpp
constexpr int rand_seed = 777;
mkl_rng::philox4x32x10 engine(
#if !INIT_ON_HOST
      *binomial_queue,
#else
      sycl::queue{sycl::cpu_selector_v},
#endif
      rand_seed);
```

Argument order as used: `(queue, seed)`. `binomial_queue` is a `sycl::queue*`.

**2. `oneapi::mkl::rng::uniform<DATA_TYPE>` — distribution object.** Constructed with
two scalar bounds; the template argument is the sample's `DATA_TYPE` (`double` or `float`).

```cpp
sycl::event event_1 = mkl_rng::generate(
    mkl_rng::uniform<DATA_TYPE>(5.0, 50.0), engine, opt_n, h_stock_price);
sycl::event event_2 = mkl_rng::generate(
    mkl_rng::uniform<DATA_TYPE>(10.0, 25.0), engine, opt_n, h_option_strike);
sycl::event event_3 = mkl_rng::generate(
    mkl_rng::uniform<DATA_TYPE>(1.0, 5.0), engine, opt_n, h_option_years);
sycl::event::wait({event_1, event_2, event_3});
```

**3. `oneapi::mkl::rng::generate` — the only generation call.** Argument order as
used: `(distribution, engine, count, output_pointer)`, returning `sycl::event`.
`count` is `constexpr int opt_n`; the output pointer is a USM shared `DATA_TYPE*`; all
three calls reuse the same `engine` and the same `count`. Bounds passed: stock price
`5.0, 50.0`; strike `10.0, 25.0`; years `1.0, 5.0` (two-argument `uniform` constructor;
see Explicit gaps on interval semantics).

No BLAS, LAPACK, sparse, DFT, or VM routine is used; the pricing kernel is hand-written SYCL (`k_binomial`), not an oneMKL routine.

## Key Code Patterns

**Queue**: a single heap `sycl::queue` built with no property list; the sample never
requests in-order execution from SYCL, so completion is enforced by explicit waits.

```cpp
sycl::queue* binomial_queue;
...
binomial_queue = new sycl::queue;
```

Default device selection; `is_fp64()` probes for double support with a throwaway queue:

```cpp
bool is_fp64() {
    sycl::queue test_queue;
    return test_queue.get_device().has(sycl::aspect::fp64);
}
```

**USM allocation** — four shared allocations of `opt_n` elements, then an
`opt_n`-element fill (note the typed zero value and the third `count` argument):

```cpp
  h_call_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_stock_price = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_option_strike = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_option_years = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);

  binomial_queue->fill(h_call_result, DATA_TYPE(0), opt_n);
```

Teardown pairs each `malloc_shared` with `sycl::free` on the same queue
(`sycl::free(h_call_result, *binomial_queue);` and likewise for the other three),
then `delete binomial_queue;`.

**ND-range and work partition** — each option gets a whole work-group:

```cpp
constexpr int block_size = num_steps / wg_size;
static_assert(block_size * wg_size == num_steps);
...
h.parallel_for<k_binomial<DATA_TYPE>>(
    sycl::nd_range(sycl::range<1>(opt_n * wg_size),
                   sycl::range<1>(wg_size)),
```

Global id → option index: `const size_t opt = item.get_global_id(0) / wg_size;`
`wg_size = 128`, `sg_size = 32` are file-scope `constexpr int` in `binomial_sycl.cpp`.

**Kernel-name template**: `template<typename Type> class k_binomial;` is declared up
front and passed as `h.parallel_for<k_binomial<DATA_TYPE>>` ("can be useful for profiling").

**SLM scratchpad** — a size-`wg_size + 1` local accessor for the cross-work-item halo:

```cpp
sycl::local_accessor<DATA_TYPE> slm_call{wg_size + 1, h};
```

Per-item private array `DATA_TYPE local_call[block_size + 1];` is allocated inside
the kernel; the `+1` slot holds the neighbour's first element.

**Kernel attributes**, written after the `nd_item` lambda:

```cpp
[=](sycl::nd_item<1> item)
    [[intel::kernel_args_restrict]] [[sycl::reqd_sub_group_size(sg_size)]]
```

**Barrier discipline** — barriers are issued only when `wg_size > sg_size`
(guarded at runtime, with both sides static constants):

```cpp
slm_call[local_id] = local_call[0];
if (wg_size > sg_size) {
  item.barrier(sycl::access::fence_space::local_space);
}
local_call[block_size] = slm_call[local_id + 1];
if (wg_size > sg_size) {
  item.barrier(sycl::access::fence_space::local_space);
}
```

**Class capture constraint** — `this` is not captured; members are copied to locals
before `submit`:

```cpp
// "this" can not be captured to the kernel. So, we need to copy internals of
// the class to local variables
DATA_TYPE* h_stock_price_local = this->h_stock_price;
```

**Device info queries** for the banner: `get_device().get_info<sycl::info::device::name>()`
and `get_info<sycl::info::device::driver_version>()`.

**Host RNG mode** (`INIT_ON_HOST=1`): the engine is built on
`sycl::queue{sycl::cpu_selector_v}` while the data buffers stay allocated on
`binomial_queue`.

**No exception handling anywhere** — no `try`/`catch` or other exception handling appears in the sample.

## Build & Run

Environment: source the `setvars` script of the oneAPI installation
(`. /opt/intel/oneapi/setvars.sh` for system-wide installs) before building.

Linux (`GNUmakefile`), target `all` → `binomial_sycl`:

```make
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng

binomial_sycl: src/binomial_sycl.cpp src/binomial_main.cpp src/binomial.hpp
	icpx -fsycl -O3 -DSMALL_OPT_N=0 -DVERBOSE=1 -DREPORT_COLD=1 -DREPORT_WARM=1 -DINIT_ON_HOST=$(init_on_host)  $(MKL_COPTS) -o $@ src/binomial_main.cpp src/binomial_sycl.cpp
```

- `make` builds; `make clean` runs `rm -f binomial_sycl`.
- `init_on_host ?= 0`; host-side RNG is selected with `make init_on_host=1`.
- Integration flags: `-qmkl-ilp64` plus `-DMKL_ILP64`, and `-qmkl-sycl-impl=rng`.

Windows (`makefile`, `nmake`), target `binomial_sycl.exe`:

```make
!if "$(init_on_host)" == "1"
	INIT_ON_HOST=/DINIT_ON_HOST=1
!endif

DPCPP_OPTS=-O3 /I"$(MKLROOT)\include" /DMKL_ILP64 /DVERBOSE=1 /DSMALL_OPT_N=0 /DREPORT_COLD=1 /DREPORT_WARM=1 $(INIT_ON_HOST) -fsycl /EHsc /Qmkl-ilp64 /Qmkl-sycl-impl=rng OpenCL.lib

binomial_sycl.exe: src\binomial_sycl.cpp src\binomial_main.cpp src\binomial.hpp
	icx $(DPCPP_OPTS) src\binomial_sycl.cpp src\binomial_main.cpp /obinomial_sycl.exe
```

`nmake init_on_host=1` enables host RNG; `nmake clean` deletes `binomial_sycl.exe`.

Compile-time switches referenced by the makefiles and sources: `SMALL_OPT_N`
(`0` → `opt_n = 8*1024*1024`; the `#if SMALL_OPT_N` branch instead sets `480`),
`VERBOSE`, `REPORT_COLD`, `REPORT_WARM`, `INIT_ON_HOST`. The GNUmakefile always
passes all five, with `init_on_host ?= 0` supplying `-DINIT_ON_HOST=0` by default;
the Windows `makefile` defines `INIT_ON_HOST` only inside its
`!if "$(init_on_host)" == "1"` block. Only `VERBOSE` has a fallback in
`binomial.hpp` (`#ifndef VERBOSE` / `#define VERBOSE 0`); the others are not
defined there, so if they are left undefined the preprocessor treats them as `0`
(`REPORT_COLD`/`REPORT_WARM` off, `INIT_ON_HOST` off, `opt_n = 8*1024*1024`).

Device selection: the sample runs on the default SYCL device; the README documents
setting `ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"`. Run the produced binary
(`./binomial_sycl`). The only other environment references in the read files are the
`setvars` script in the README and `$(MKLROOT)\include` in the Windows `makefile`
(normally set by `setvars`).

Expected output lines (README example, `./binomial_sycl`):

```
Double Precision Binomial Option Pricing version 1.8 running on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz using DPC++, workgroup size 128, sub-group size 32.
Compiler Version: Intel(R) oneAPI DPC++/C++ Compiler 2023.1.0 (2023.1.0.20230320), LLVM 16.0 based.
Driver Version  : 2023.15.3.0.20_160000
Build Time      : May  3 2023 03:51:58
Input Dataset   : 8388608
Pricing 8388608 Options with time step of 2048.
Cold iteration.
Completed in  379.85246 seconds. Options per second: 22083.85865
Warm iteration.
Completed in  378.85559 seconds. Options per second: 22141.96683
Time Elapsed =   378.85559 seconds
Creating the reference result...
L1 norm: 2.645901E-06
TEST PASSED
```

`run()` always prints the banner and the `Cold iteration.` body; the
`Completed in ...` lines are printed only under `REPORT_COLD` / `REPORT_WARM`.
`check()` prints `Creating the reference result...`, the L1 norm, and
`TEST PASSED` / `TEST FAILED` only when `VERBOSE` is nonzero, and calls
`exit(1)` on failure. The warm iteration re-runs `body()` with unchanged input data.

## Gotchas & Invariants

- `num_steps` must be exactly divisible by `wg_size`: `block_size = num_steps / wg_size`
  is guarded by `static_assert(block_size * wg_size == num_steps)`. With the shipped
  constants `2048 / 128 = 16`.
- SLM sizing is load-bearing: `slm_call` is sized `wg_size + 1` and the kernel reads
  `slm_call[local_id + 1]` for `local_id == wg_size - 1`. The extra element also
  receives the last terminal node, written only by `local_id == wg_size - 1`.
- Global range is `opt_n * wg_size`; every work-group must be fully populated. The
  option index is derived by integer division by `wg_size`, so a differently-shaped
  range silently misindexes the input arrays.
- The `if (wg_size > sg_size)` tests around both `item.barrier(sycl::access::fence_space::local_space)`
  calls involve two `constexpr` values; with the shipped `wg_size = 128` and
  `sg_size = 32` the comparison is a constant true and the barriers are always
  executed. The sample does not exercise or document a `wg_size <= sg_size`
  configuration.
- `[[sycl::reqd_sub_group_size(sg_size)]]` pins sub-group size to 32; both `wg_size`
  and `sg_size` are also printed in the banner, so they must agree with the values
  shown in the README output.
- All four arrays are `sycl::malloc_shared` on USM, and `h_call_result` is zero-filled
  by `binomial_queue->fill(...)` before the kernel; the matching `sycl::free` calls use
  the same queue as the `malloc_shared` that produced each pointer.
- The three `generate` calls return events that must all complete before the kernel
  reads the inputs; the sample joins them with `sycl::event::wait({event_1, event_2, event_3})`
  and separately calls `binomial_queue->wait()` after `submit` in `body()`.
- The queue has no in-order property; nothing may assume sequential execution.
- Cross-item handoff: the kernel's `slm_call[local_id] = local_call[0]` write and
  `local_call[block_size] = slm_call[local_id + 1]` read are separated by the two
  barriers above, and that ordering is what makes the handoff correct.
- `float` results are accepted only when no `fp64` device is found; `main` prints
  `Warning: could not find a device with double precision support. Single precision is used.`
  and instantiates `Binomial<float>`. The pass threshold `5e-4` is shared by both types.
- `check()` compares against a host reference computed with `BlackScholesRefImpl`
  using the same `risk_free` and `volatility`; it uses `%E` formatting and prints
  `Avg. diff:` instead of `L1 norm:` when `sum_ref <= 1E-5`.
- The banner prints `Cold iteration.` before the first `body()` and `Warm iteration.`
  before the second; `timer t{}` is restarted before the warm run and the final
  `Time Elapsed` line is printed after it, so that line reports the warm `body()`
  duration rather than the cold one. Neither the code nor the README states why the
  two runs differ.

## Explicit gaps

- Full declarations of `oneapi::mkl::rng::generate`, `oneapi::mkl::rng::uniform` and
  `oneapi::mkl::rng::philox4x32x10` (all template parameters, defaulted parameters
  such as dependency events, and available overloads) are **not** in these files;
  only the call sites quoted above are established. Copy the signatures from the
  oneMKL RNG headers, not from this sample.
- The interval semantics of `uniform<DATA_TYPE>(a, b)` (open/closed bounds) are not
  stated by the sample or README.
- `DLL_EXPORT` appears in the explicit instantiation definitions
  (`template DLL_EXPORT Binomial<double>::Binomial();`) but is defined nowhere in the
  read files and never passed by either makefile (`-D`/`/D`); how it is meant to be
  defined is not established.
- No workspace/scratchpad query is shown for the RNG calls; whether a scratchpad
  overload exists or is required is not established here.
- No `CMakeLists.txt`, CMake target, or `icpx`/`icx` version requirement appears in the
  read files; only the two makefiles are established build paths.
- Neither the code nor the README explains the cold/warm distinction beyond the two
  printed labels; no cause for the timing difference is given in the read files.
- No guidance is given on how many options fit in device memory for the default
  `opt_n = 8*1024*1024`, and no timing/variance guarantee accompanies the two printed
  iterations — only the `L1 norm` (or `Avg. diff`) line reflects accuracy.

## Source Map

- `README.md` — purpose, build/run instructions, `init_on_host=1`, `ONEAPI_DEVICE_SELECTOR` (`*:cpu` / `*:gpu`), example output.
- `GNUmakefile` — Linux `icpx` build (`-fsycl -O3`, `-DMKL_ILP64`, `-qmkl-ilp64`, `-qmkl-sycl-impl=rng`), `all` / `clean`, `init_on_host`.
- `makefile` — Windows `nmake`/`icx` build, `DPCPP_OPTS`, `binomial_sycl.exe`, `clean`.
- `src/binomial.hpp` — `Binomial<DATA_TYPE>` class declaration, constants (`volatility`, `risk_free`, `num_steps`, `opt_n`), `VERBOSE` default, `timer`, `is_fp64()`.
- `src/binomial_main.cpp` — `BlackScholesRefImpl` host reference, `Binomial<DATA_TYPE>::check()` L1-norm validation, `main` fp64/float dispatch.
- `src/binomial_sycl.cpp` — queue and USM setup, `philox4x32x10` + `uniform` + `generate` calls, `k_binomial` nd-range kernel with SLM halo, `run()` banner/timing, `is_fp64()`, explicit instantiations.
