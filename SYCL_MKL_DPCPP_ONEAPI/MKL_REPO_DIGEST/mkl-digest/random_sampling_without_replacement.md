# Random Sampling Without Replacement (Lottery)

## Domain & Purpose

oneMKL domain: **RNG** (`oneapi::mkl::rng`), in two variants — the host-side
queue-based API (`oneapi/mkl.hpp`) and the in-kernel **device** API
(`oneapi/mkl/rng/device.hpp`).

Generates `K>>1` simple random length-`M` samples *without replacement* from a
population of size `N` (`1 <= M <= N`) using a Partial Fisher–Yates shuffle.
Three programs illustrate the same algorithm through different RNG APIs:
`lottery.cpp` and `lottery_usm.cpp` use the host queue-based API and are
byte-identical apart from their printed banner (`lottery_usm.cpp` also prints an
extra `Results with Host API:` line); `lottery_device_api.cpp` calls the RNG from
inside the kernel.

## Problem & Math

Each experiment is a partial length-`M` random shuffle of the `N`-element
population `{1, 2, ..., N}` ("lottery `M` of `N`"). One experiment per work-group:

1. Initialize local buffer to `local_buf[i] = i + 1` for `i` in `[0, N)`.
2. For `i` in `[0, M)`:
   - draw uniform `u` in `[0, 1)`,
   - `j = i + (size_t)(u * (float)(n - i))`, so `j` in `{i, ..., N-1}`,
   - swap `local_buf[i]` and `local_buf[j]`.
3. Emit `local_buf[0..M)` as the sample.

`m * num_exp` uniform floats are consumed; results land at
`result_ptr[id * m + i]` (host programs) or `res_acc[id * m + i]` (device program).

## oneMKL Routines Used

`using namespace oneapi;` is in effect in all three files, but every call below is
written with its full qualification in the source.

**Host / queue API** — include `"oneapi/mkl.hpp"`:

```cpp
// Create an object of basic random number generator (engine)
oneapi::mkl::rng::philox4x32x10 engine(q, seed);
// Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
oneapi::mkl::rng::uniform distr;

float* rng_buf = sycl::malloc_device<float>(m * num_exp, q);

// Random number generation
auto event = oneapi::mkl::rng::generate(distr, engine, m * num_exp, rng_buf);
// Make sure, that generation is finished
event.wait_and_throw();
```

- `philox4x32x10` — the engine name used; the README states a "Philox 4x32x10
  generator is used, and a uniform distribution is a basis for the algorithm".
- `uniform distr;` — constructed with **no** arguments in all three programs; the
  in-source comment documents the default as `float`, `a = 0.0f`, `b = 1.0f`.
- `generate(distr, engine, m * num_exp, rng_buf)` — argument order as used:
  distribution, engine, count, output pointer. Return value is an event that is
  explicitly waited on. Only this 4-argument overload form is shown by the source.
- `seed` is `static const auto seed = 777;` in all three files.

**Device (in-kernel) API** — include `"oneapi/mkl/rng/device.hpp"`:

```cpp
// inside h.parallel_for lambda, once per work-group:
// Create an object of basic random number generator (engine)
oneapi::mkl::rng::device::philox4x32x10 engine(seed, id * m);
// Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
oneapi::mkl::rng::device::uniform distr;

for (size_t i = 0; i < m; ++i) {
    auto res = oneapi::mkl::rng::device::generate(distr, engine);
    auto j = i + (size_t)(res * (float)(n - i));
    std::swap(local_buf[i], local_buf[j]);
}
```

- The device engine constructor is called with two arguments, written here as
  `(seed, id * m)`; the source gives no parameter names, units, or documented
  meaning for the second argument (see Explicit gaps).
- `oneapi::mkl::rng::device::generate(distr, engine)` is assigned to a scalar
  (`auto res`) and used as a single number — no explicit count/pointer arguments
  appear in this call.

## Key Code Patterns

**Queue construction with async exception handler** (identical in all three
programs; uses `sycl::default_selector_v`):

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

**USM allocation + generate/wait/consume** (`lottery.cpp`, `lottery_usm.cpp`):

```cpp
float* rng_buf = sycl::malloc_device<float>(m * num_exp, q);
auto event = oneapi::mkl::rng::generate(distr, engine, m * num_exp, rng_buf);
event.wait_and_throw();          // "Make sure, that generation is finished"
...
event = q.submit([&] (sycl::handler& h) { ... });
event.wait_and_throw();
```

The host-API `generate` event is waited **before** the shuffle kernel is
submitted, and the kernel event is waited before the host reads the results. The
queue is constructed with a selector and an exception handler only; these files
do not state its ordering property, so the explicit waits are what the source
shows as the ordering mechanism.

**Result storage differs per program:**

- `lottery.cpp`: `sycl::malloc_shared<size_t>(m * num_exp, q)` → `size_t* result_ptr`.
- `lottery_usm.cpp`: identical `sycl::malloc_shared<size_t>(m * num_exp, q)`.
- `lottery_device_api.cpp`: host `std::vector<size_t> result_vec(m * num_exp)` wrapped
  in `sycl::buffer<size_t, 1> result_buf(result_vec.data(), result_vec.size())`,
  accessed inside the scope with
  `auto res_acc = result_buf.template get_access<sycl::access::mode::write>(h);`
  The buffer is created inside a nested `{ ... }` block that closes before the
  results are printed and validated, and no explicit `q.wait()` appears in the
  file, so completion is left to the buffer's scope exit.

**Work decomposition** (both host and device shuffle kernels): one work-group per
experiment, exactly one work-item per group, running group id indexes the
experiment:

```cpp
sycl::local_accessor<size_t> local_buf(sycl::range<1>{n}, h);
h.parallel_for(sycl::nd_range<1>(num_exp, 1),
    [=](sycl::nd_item<1> item) {
    size_t id = item.get_group(0);
    // Let buf contain natural numbers 1, 2, ..., N
    for (size_t i = 0; i < n; ++i) {
        local_buf[i] = i + 1;
    }
```

Because each group has a single work-item, the read-modify-write swaps into
`local_buf` need no barrier; `item.get_group(0)` (not `get_global_id`) is the
experiment index.

## Build & Run

Environment first (README): `. /opt/intel/oneapi/setvars.sh` (or
`. ~/intel/oneapi/setvars.sh` for private installs).

**Linux (GNUmakefile, GNU Make):**

```make
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=rng
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

lottery: lottery.cpp
		icpx $< -fsycl -o $@ $(DPCPP_OPTS)
lottery_usm: lottery_usm.cpp
		icpx $< -fsycl -o $@ $(DPCPP_OPTS)
lottery_device_api: lottery_device_api.cpp
		icpx $< -fsycl -o $@ $(DPCPP_OPTS)
```

- `make` (targets `default` / `all` / `run`) builds all three and then runs
  `./lottery`, `./lottery_usm`, `./lottery_device_api`.
- `make clean` runs `-rm -f lottery lottery_usm lottery_device_api`.
- `.PHONY: clean run all`.

**Windows (makefile, NMAKE):** `nmake` builds/runs, `nmake clean` deletes the exes.

```make
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=rng /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

lottery.exe: lottery.cpp
	icx-cl -fsycl lottery.cpp /Felottery.exe $(DPCPP_OPTS)
```

The Windows makefile additionally defines a `pseudo: clean run all` target, whose
`clean` recipe is `del /q lottery.exe lottery_usm.exe lottery_device_api.exe`, and
keeps commented-out `icpx` rules. No CMakeLists is referenced by any listed file.

**Device selection** (README): computations run on the default SYCL device; set
`ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"`.

**Runtime arguments** (all three programs, same logic): with `argc >= 4`,
`m = atol(argv[1]); n = atol(argv[2]); num_exp = atol(argv[3]);`. If
`m == 0 || n == 0 || num_exp == 0 || m > n`, all three revert to the defaults
`m_def = 6`, `n_def = 49`, `num_exp_def = 11969664`.

**Expected output** (README example; README shows a separate block per binary).
The `./lottery` block is quoted verbatim:

```
./lottery

Multiple Simple Random Sampling without replacement
Buffer Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

TEST PASSED
```

The README's `./lottery_usm` block is the same shape with the banner
`Unified Shared Memory API` and an extra `Results with Host API:` line before the
three `Sample` lines; the `./lottery_device_api` block is the same shape with the
banner `Device API`. Note that the C++ sources print `Buffer Api`,
`Unified Shared Memory Api` and `Device Api` respectively — the latter two differ
in case from the README banners.

`lottery_usm.cpp` prints an extra `Results with Host API:` line before the
samples. Only the last 3 experiments are printed
(`for (size_t i = num_exp - 3; i < num_exp; ++i)`). The failure path prints
`Failure` and calls `std::terminate()`; a duplicate-detection failure prints
`TEST FAILED` and returns 1.

## Gotchas & Invariants

- **`N` must fit in local memory.** `sycl::local_accessor<size_t> local_buf(sycl::range<1>{n}, h)`
  allocates `N` `size_t` per work-group; the host-side argument validation only
  checks `m == 0 || n == 0 || num_exp == 0 || m > n`, so the source contains no
  check against device local-memory limits and such a failure would surface at
  runtime, not in the argument parsing.
- **One work-item per group is load-bearing.** Swaps into `local_buf` are
  unsynchronized; increasing the group size without adding barriers is a data race.
- **Validation covers only zero/order checks.** The argument check is exactly
  `m == 0 || n == 0 || num_exp == 0 || m > n`; there is no check against overflow
  of `m * num_exp`. `generate` is asked for exactly `m * num_exp` floats and the
  shuffle indexes `rng_buf[id * m + i]`, so buffer size and the two loops must
  agree in the source as written.
- **Ordering is explicit, not implicit.** The host `generate` event is waited with
  `event.wait_and_throw()` before the shuffle kernel is submitted; the shuffle
  event is waited before the host reads `result_ptr`. In the device-API program
  no wait appears at all — the `std::vector` is read only after the inner scope
  holding `sycl::buffer` has exited.
- **Device-API offset must be per-experiment.** `philox4x32x10 engine(seed, id * m)`
  passes a different second argument per group; the source gives no semantics for
  that argument, and reusing one value across groups would make the groups draw
  the same numbers (inference from the usage, not a documented guarantee).
- **Float-precision index mapping.** `j = i + (size_t)(res * (float)(n - i))`: the
  product is `float` and truncated, so the sample's randomness is only as good as
  the uniform `float` draw.
- **The correctness check is weak.** The `if` condition combines the duplicate
  result with `std::count_if(...) == 0` via `&&`, where the predicate is
  `val > n && val >= 0` (`size_t` is always `>= 0`), so a range violation alone
  does not fail the test. `TEST PASSED` here is not strong evidence of correctness.
- **`lottery.cpp` prints "Buffer Api"** although it uses `sycl::malloc_shared`
  USM and its file comment says "USM-based interface"; only
  `lottery_device_api.cpp` uses a real `sycl::buffer`. `lottery.cpp` and
  `lottery_usm.cpp` are otherwise byte-identical.
- The device-API file includes only `"oneapi/mkl/rng/device.hpp"`, not `"oneapi/mkl.hpp"`;
  the other two include `"oneapi/mkl.hpp"` and no device header.
- All three programs mutate the results in place with `std::sort` during
  validation, so `result_ptr` / `result_vec` are sorted after the check loop.

## Explicit gaps

- The exact **full signature/overload set** of `oneapi::mkl::rng::generate` is not
  established by these files; only the call
  `oneapi::mkl::rng::generate(distr, engine, m * num_exp, rng_buf)` (host) and
  `oneapi::mkl::rng::device::generate(distr, engine)` (device) are shown. Whether
  additional overloads (e.g. explicit result-type or offsets) exist is not stated.
- The template parameters of `oneapi::mkl::rng::uniform` and of the engines are
  never written explicitly; the in-source comment only says the distribution
  default is `float` with `a = 0.0f`, `b = 1.0f`.
- No `oneapi::mkl::rng::device` engine reset/destructor API, scratchpad query,
  `skip_ahead`, or `enqueued`/`wait` helper is referenced anywhere in these files.
- The **sequence/offset semantics** of the device engine's second constructor
  argument are not given by the source at all: no parameter names, no units, no
  counter-vs-offset meaning, and no relationship to the host `philox4x32x10`
  stream for the same seed. The only evidence is the call site `id * m`.
- The **queue ordering property** (in-order vs out-of-order) is not stated in any
  of the three files; only the explicit waits and the `sycl::buffer` scope exit are
  visible.
- The README's expected sample values are illustrative; determinism across the
  host and device APIs (identical last-3 samples per the README output) is not
  demonstrated by any test code in these files.
- No CMake build path or `CMakeLists.txt` is present among the listed files, so a
  CMake invocation cannot be quoted.
- The relationship between `-qmkl-sycl-impl=rng` and required device link
  libraries is not explained; only the literal flags are given.

## Source Map

- `README.md` — purpose, Partial Fisher–Yates description, M=6/N=49/11969664 defaults, all three binaries' expected output, `setvars` and `ONEAPI_DEVICE_SELECTOR` guidance.
- `makefile` — NMAKE/Windows build: `icx-cl -fsycl`, `/Qmkl-*` flags, `OpenCL.lib`, targets `run`/`clean`/`pseudo`.
- `GNUmakefile` — Linux/GNU Make build: `icpx -fsycl`, `-qmkl-*` flags, targets `run`/`clean`.
- `lottery.cpp` — host RNG API with `sycl::malloc_shared` results; prints "Buffer Api".
- `lottery_usm.cpp` — byte-identical to `lottery.cpp` except the banner "Unified Shared Memory Api" and the `Results with Host API:` line.
- `lottery_device_api.cpp` — `oneapi/mkl/rng/device.hpp` in-kernel engine/distribution/generate inside a `sycl::buffer`-backed kernel; prints "Device Api".
