# Black-Scholes Option Pricing (SYCL)

## Domain & Purpose

oneMKL domain: **Random Number Generation (RNG)** only. The sample generates a
random portfolio (stock prices, option strikes, option years) with the oneMKL RNG
API, then prices European call and put options with the Black-Scholes formula
inside a hand-written SYCL kernel. No BLAS/LAPACK/DFT routine is called.

## Problem & Math

Reference implementation in `black_scholes.hpp` (verbatim body):

```cpp
double N_d1 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r + 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
double N_d2 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r - 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
call_result = (S * N_d1 - L * std::exp(-r * t) * N_d2);
```

Kernel-side variant (`USE_CNDF_C` false) computes the same with `sycl::erf`,
`sycl::log`, `sycl::sqrt`, `sycl::exp`, and put via put-call parity:

```cpp
const DATA_TYPE call_val = s * n_d1 - XexpRT * n_d2;
const DATA_TYPE put_val = call_val + XexpRT - s;
```

Constants (`black_scholes.hpp`): `constexpr float volatility = 0.30f;`
`constexpr float risk_free = 0.02f;`. Problem sizes: `opt_n` is
`8 * 1024 * 1024` when `SMALL_OPT_N` is falsy, else `480`; `ITER_N` defaults to `512`.
Portfolio ranges: stock price uniform `(5.0, 50.0)`, strike uniform `(10.0, 25.0)`,
years uniform `(1.0, 5.0)`. Seed `constexpr int rand_seed = 777;`.

## oneMKL Routines Used

The only oneMKL header included is `#include <oneapi/mkl.hpp>`; no finer-grained
MKL header is included. In the constructor the source declares the alias
`namespace mkl_rng = oneapi::mkl::rng;` and then uses only `mkl_rng::`-qualified
names. The fully-qualified expansions below name the same oneMKL entities; the
verbatim spellings in this source are the `mkl_rng::` forms.

Engine construction — `mkl_rng::philox4x32x10`, i.e.
`oneapi::mkl::rng::philox4x32x10`, constructed from a `sycl::queue` and an integer
seed. Verbatim, with the queue argument selected by the `INIT_ON_HOST` macro:

```cpp
mkl_rng::philox4x32x10 engine(
#if !INIT_ON_HOST
    *black_scholes_queue,
#else
    sycl::queue{sycl::cpu_selector_v},
#endif // !INIT_ON_HOST
    rand_seed);
```

Generation — `mkl_rng::generate`, i.e. `oneapi::mkl::rng::generate`, four
arguments in this order: distribution object, engine, count, output USM pointer.
The distribution is `mkl_rng::uniform<DATA_TYPE>`, i.e.
`oneapi::mkl::rng::uniform<DATA_TYPE>`, constructed with `(min, max)`. Verbatim:

```cpp
sycl::event event_1 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(5.0, 50.0), engine, opt_n, h_stock_price);
sycl::event event_2 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(10.0, 25.0), engine, opt_n, h_option_strike);
sycl::event event_3 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(1.0, 5.0), engine, opt_n, h_option_years);
sycl::event::wait({event_1, event_2, event_3});
```

Each `generate` call returns a `sycl::event`; the three events are waited on
together. `opt_n` is a `size_t`; the output pointers are `DATA_TYPE*` returned by
`sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue)`.

No other oneMKL symbol appears anywhere in the two source files.

## Key Code Patterns

**Queue** — default-constructed with no properties, held as a global pointer
(`black_scholes_queue = new sycl::queue;`), and released in the destructor with
`delete black_scholes_queue;`. The source never requests
`sycl::property::queue::in_order`, so the queue has SYCL's default out-of-order
semantics. Device choice is left to the SYCL default selector (overridable by
`ONEAPI_DEVICE_SELECTOR`).

**USM allocation** — all five buffers are `sycl::malloc_shared`, so host code reads
the results directly after `queue->wait()`:

```cpp
h_call_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
h_put_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
h_stock_price = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
h_option_strike = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
h_option_years = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
```

Freed with `sycl::free(ptr, *black_scholes_queue)` for each pointer, then the queue
is deleted. No `memset`/initialization of the buffers is performed.

**Kernel name and launch** — named kernel class template forward-declared as
`template<typename Type, int> class k_BlackScholes;` and launched with the
block size as a non-type template argument:

```cpp
black_scholes_queue->parallel_for<k_BlackScholes<DATA_TYPE, block_size>>(sycl::nd_range(sycl::range<1>(opt_n / block_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] [[sycl::reqd_sub_group_size(sg_size)]] {
```

Sizes: `sg_size = 32`; `wg_size = 256` and `block_size = 4` by default, or
`wg_size = 128` / `block_size = 1` when `NON_DEFAULT_SIZE` is truthy. Each
work-item processes `block_size` elements with an unrolled strided loop:

```cpp
for (size_t opt = group_id * block_size * wg_size + local_id, i = 0; i < block_size; opt += wg_size, i++) {
```

Class members cannot be captured into the kernel, so `body()` first copies them to
local variables (`h_stock_price_local`, `h_option_years_local`,
`h_option_strike_local`, `h_call_result_local`, `h_put_result_local`) and captures
those by `[=]`.

**Ordering / synchronization** — no submission uses `depends_on`; ordering is
enforced explicitly with `sycl::event::wait({...})` after the three RNG calls and
with `black_scholes_queue->wait()` after each launch of `body()`. `run()` calls
`body()` once before the timed loop, so that first call is not counted by
`timer t{}`.

**Precision dispatch** — `main` probes the default device and instantiates
`BlackScholes<double>` or `BlackScholes<float>`:

```cpp
sycl::queue test_queue;
is_fp64 = test_queue.get_device().has(sycl::aspect::fp64);
```

**No exception handling** — no `try`/`catch`, no `sycl::exception` handling, and no
check of RNG error/status in either file.

## Build & Run

Environment: source the oneAPI `setvars` script first
(`. /opt/intel/oneapi/setvars.sh` for system-wide installs,
`. ~/intel/oneapi/setvars.sh` for private installs). `MKLROOT` is used by the
Windows makefile.

Linux (`GNUmakefile`), targets `all` (default, builds `black_scholes_sycl`) and
`clean`:

```make
MKL_COPTS = -DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl=rng

black_scholes_sycl: src/black_scholes_sycl.cpp
	icpx -O3 -g -fsycl $(MKL_COPTS) -DVERBOSE=1 -DSMALL_OPT_N=0 -DINIT_ON_HOST=$(init_on_host) -o $@  src/black_scholes_sycl.cpp
```

`init_on_host ?= 0` by default; build the RNG-on-host variant with `make init_on_host=1`.
Run with `./black_scholes_sycl`; `make clean` runs `rm -f black_scholes_sycl`.

Windows (`makefile`, nmake), targets `all` and `clean`:

```make
DPCPP_OPTS=-O3 /I"$(MKLROOT)\include" /DMKL_ILP64 /DVERBOSE=1 /DSMALL_OPT_N=0 $(INIT_ON_HOST) -fsycl /EHsc /Qmkl-ilp64 /Qmkl-sycl-impl=rng OpenCL.lib

black_scholes_sycl.exe: src\black_scholes_sycl.cpp
	icx $(DPCPP_OPTS) src\black_scholes_sycl.cpp /oblack_scholes_sycl.exe
```

`!if "$(init_on_host)" == "1"` adds `INIT_ON_HOST=/DINIT_ON_HOST=1`. Build/run with
`nmake`, cleanup with `nmake clean`.

Device selection: default SYCL device; set
`ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"` to force a device. If the device
lacks `sycl::aspect::fp64` the program prints
`Warning: could not find a device with double precision support. Single precision is used.`
and runs the `float` path.

Output begins with a banner built from `sizeof(DATA_TYPE)`, `MAJOR`=1, `MINOR`=6,
`sycl::info::device::name`, `wg_size`, `sg_size`, then
`sycl::info::device::driver_version`, `__DATE__`/`__TIME__`,
`Input Dataset   : <opt_n>`, a pricing/iteration line, timing lines, and finally
`L1 norm: <value>` plus `TEST PASSED` (or `TEST FAILED` and `exit(1)`).
README example lines include:

```
Pricing 16777216 Options in 512 iterations, 8589934592 Options in total.
Completed in    1.41111 seconds. GOptions per second:    6.08735
Time Elapsed =     1.41111 seconds
L1 norm: 1.385136E-16
TEST PASSED
```

## Gotchas & Invariants

- `opt_n` must be divisible by `block_size * wg_size`: the global range is
  `opt_n / block_size`, the local range is `wg_size`, so the group count is
  `(opt_n / block_size) / wg_size` and the covered element count is
  `group_count * block_size * wg_size`. With `opt_n = 8 * 1024 * 1024`,
  `block_size = 4`, `wg_size = 256` (8192 groups) coverage is exact. Otherwise the
  launch does not satisfy SYCL's requirement that the global range be a multiple of
  the local range (and any truncation of the group count would leave the tail of the
  portfolio unprocessed); the kernel contains no bounds check. The shipped
  `SMALL_OPT_N=1` value is 480, and 480 / 4 = 120, which is neither a multiple of
  `wg_size` (256) nor even >= `wg_size` at the default sizes.
- `wg_size` (256 or 128) must remain a multiple of `sg_size` (32), since
  `[[sycl::reqd_sub_group_size(sg_size)]]` is applied unconditionally.
- Only the uniform `(a,b)` bounds and the count are passed; the RNG writes
  directly into shared USM that the kernel later reads. With an out-of-order queue,
  the `sycl::event::wait({event_1, event_2, event_3})` is the only thing preventing
  a race — removing it is a silent correctness bug.
- Results are read on the host from `sycl::malloc_shared` memory, so
  `black_scholes_queue->wait()` after each `body()` loop is required; `check()`
  itself performs no synchronization.
- `USE_CNDF_C`, `NON_DEFAULT_SIZE`, `SMALL_OPT_N`, `INIT_ON_HOST` and `ITER_N` are
  macro switches. Neither makefile defines `USE_CNDF_C` or `NON_DEFAULT_SIZE`, so
  those `#if` branches are inert as shipped unless the user adds `-D` flags.
- `INIT_ON_HOST=1` runs the RNG on `sycl::queue{sycl::cpu_selector_v}` while the
  pricing kernel still runs on `black_scholes_queue`; correctness relies on shared
  USM being host-accessible.
- Verification is a relative L1 error (`sum_delta / sum_ref`) when `sum_ref > 1E-5`,
  otherwise an average difference; the pass threshold is `errorVal < 5e-4`. The
  comparison is against a CPU recomputation, not an external oracle.
- `BlackScholesRefImpl` in `black_scholes.hpp` is a non-inline free function defined
  in a header; it is only safe because exactly one translation unit includes it.

## Explicit gaps

- The exact declaration/overload set of `oneapi::mkl::rng::generate` is not shown;
  only the 4-argument form `(distribution, engine, count, pointer)` returning
  `sycl::event` is established. Whether other overloads (e.g. with `std::int64_t`
  count or explicit dependencies) exist is not established by these files.
- The full constructor signature of `oneapi::mkl::rng::philox4x32x10` is not shown;
  only that it accepts a `sycl::queue` plus `rand_seed` (an `int`).
- The full template/typedef name of the `uniform` distribution type is not
  established; only `mkl_rng::uniform<DATA_TYPE>(min, max)` as used.
- Queue ordering is not stated in the sources; the out-of-order behavior described
  above follows the SYCL default when `sycl::property::queue::in_order` is not
  requested, not an explicit statement in the sample.
- No RNG status/exception handling or error-reporting API is demonstrated.
- No explicit byte alignment, leading-dimension, or memory-padding requirements are
  stated anywhere in these files.
- No CMake build file is present in this sample directory, so CMake target names and
  options are not established here.
- The `#if !SYCL_LANGUAGE_VERSION` guard line in `black_scholes_sycl.cpp` contains a
  stray extra double quote in the `#error` string as shipped.
- Values of README-reported timings are hardware-dependent and not reproducible
  invariants.

## Source Map

- `README.md` — purpose, RNG/Philox 4x32x10 summary, Linux/Windows build and run
  instructions, `ONEAPI_DEVICE_SELECTOR` device selection, example output, `init_on_host=1` note.
- `GNUmakefile` — Linux `icpx` build flags (`-DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl=rng`),
  `init_on_host` variable, `all`/`clean` targets.
- `makefile` — Windows `nmake`/`icx` build flags (`/Qmkl-ilp64 /Qmkl-sycl-impl=rng`),
  conditional `INIT_ON_HOST`, `all`/`clean` targets.
- `src/black_scholes.hpp` — `BlackScholes<DATA_TYPE>` class declaration, `volatility`/`risk_free`
  constants, `opt_n`/`ITER_N`/`VERBOSE` macros, `BlackScholesRefImpl`, `check()` L1 verification,
  `timer` helper.
- `src/black_scholes_sycl.cpp` — queue and shared-USM setup, `philox4x32x10` engine and three
  `generate` calls, SYCL pricing kernel `k_BlackScholes`, timing loop, fp64-aspect dispatch in `main`.
