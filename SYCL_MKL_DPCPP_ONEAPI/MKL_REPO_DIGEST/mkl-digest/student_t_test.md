# Student's t-Test

## Domain & Purpose
- oneMKL domain touched: **Vector Statistics** — `oneapi::mkl::stats` (statistics) and `oneapi::mkl::rng` (RNG engines, distributions, generation).
- Two programs implement the same test twice: `t_test.cpp` with the SYCL **buffer** API, `t_test_usm.cpp` with **USM** (Unified Shared Memory). Both generate ~1,000,000 Gaussian samples, compute mean/central moment, and decide whether the null hypothesis is accepted or rejected.

## Problem & Math
- One-sample t-test: `|mean - expected_mean| * sqrt(n) / sqrt(variance) < threshold`.
- Two-sample t-test, chosen by an equality-of-variance guard:
```cpp
bool almost_equal = (variance1_acc[0] < 2 * variance2_acc[0]) ||
                    (variance2_acc[0] < 2 * variance1_acc[0]);
```
- `almost_equal == true` branch (verbatim, `t_test.cpp` lines 102-107):
```cpp
if ((std::abs(mean1_acc[0] - mean2_acc[0]) /
     std::sqrt((static_cast<RealType>(1.0) / static_cast<RealType>(n1) +
                static_cast<RealType>(1.0) / static_cast<RealType>(n2)) *
              ((n1 - 1) * (n1 - 1) * variance1_acc[0] +
               (n2 - 1) * (n2 - 1) * variance2_acc[0]) /
               (n1 + n2 - 2))) < static_cast<RealType>(threshold)) {
```
- `almost_equal == false` branch: `|mean1 - mean2| / std::sqrt(variance1 + variance2) < threshold`.
- `threshold` is a hard-coded constant, not computed by any oneMKL routine:
```cpp
// T-test threshold which corresponds to 5% significance level and infinite
// degrees of freedom
static const float threshold = 1.95996f;
```
- Return convention of both `t_test` overloads: `-1` something went wrong (initial value, never overwritten on failure), `1` null hypothesis accepted, `0` null hypothesis rejected.

## oneMKL Routines Used
Call forms are quoted exactly as they appear in the sources. The sources contain only call sites, not routine declarations, so no complete prototype is reproduced here.

- Dataset constructor — **buffer** version (`t_test.cpp`):
```cpp
auto dataset =
    oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
        1, n, r);
```
  - `r` is `sycl::buffer<RealType, 1>&`; `n` is `std::int64_t`.
  - Template argument is the layout enum value `oneapi::mkl::stats::layout::row_major`; no other layout appears in the sources.

- Dataset constructor — **USM** version (`t_test_usm.cpp`): identical spelling but `r` is `RealType*`:
```cpp
auto dataset =
    oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
        1, n, r);
```

- `oneapi::mkl::stats::mean` — output is a `sycl::buffer<RealType, 1>&` in the buffer version, a `RealType*` in the USM version:
```cpp
oneapi::mkl::stats::mean(q, dataset, mean_buf);   // t_test.cpp
oneapi::mkl::stats::mean(q, dataset, mean);       // t_test_usm.cpp, mean is RealType*
```
  - Argument order in both: `queue`, input dataset, output mean.

- `oneapi::mkl::stats::central_moment` — note the argument order is `(queue, mean, dataset, variance)`, i.e. the mean (computed by the preceding call) is passed **between** the queue and the dataset:
```cpp
oneapi::mkl::stats::central_moment(q, mean_buf, dataset, variance_buf); // buffer
oneapi::mkl::stats::central_moment(q, mean, dataset, variance);         // USM (RealType*)
```
  - The call passes no moment-order argument, and no file names a moment order; the result is stored in a local named `variance`. Which central moment the 4-argument call computes is not stated in these files (see Explicit gaps).

- RNG engine (`oneapi::mkl::rng::default_engine`), constructed with the queue and an `int` seed:
```cpp
static const int seed = 7777;
oneapi::mkl::rng::default_engine engine(q, seed);
```

- Distribution (`oneapi::mkl::rng::gaussian`) templated on the real type, constructed as `(mean, std_dev)`:
```cpp
oneapi::mkl::rng::gaussian<fp_type> distribution(mean, std_dev);
```
  - `fp_type` is `float` in both programs.

- Generation (`oneapi::mkl::rng::generate`) — argument order `(distribution, engine, count, output)`:
```cpp
oneapi::mkl::rng::generate(distribution, engine, n_points, rng_buf0); // buffer
oneapi::mkl::rng::generate(distribution, engine, n_points, rng_arr0); // USM (fp_type*)
```
  - `n_points` is `size_t`.

- No BLAS/LAPACK/DFT/Sparse/VM routines are used. `std::sqrt` and `std::abs` do the arithmetic on the host.

## Key Code Patterns
- Queue constructed with an async exception handler via `sycl::default_selector_v`; the handler walks `sycl::exception_list` and prints `e.what()`:
```cpp
auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception during generation:\n"
                << e.what() << std::endl;
    }
  }
};
sycl::queue q(sycl::default_selector_v, exception_handler);
```
- Synchronous error path in `main`: the whole oneMKL workload is inside `try { ... } catch (...) { std::cout << "Failure" << std::endl; std::terminate(); }`.
- Buffer-API output for a scalar statistic is a 1-element SYCL buffer, read back with a `host_accessor`:
```cpp
sycl::buffer<RealType, 1> mean_buf(sycl::range{1});
sycl::buffer<RealType, 1> variance_buf(sycl::range{1});
...
sycl::host_accessor mean_acc(mean_buf);
sycl::host_accessor variance_acc(variance_buf);
```
- `q.wait_and_throw()` is called after the mean/central_moment/generate calls before host reads (`host_accessor` construction, or reading USM shared memory):
```cpp
oneapi::mkl::stats::mean(q, dataset, mean_buf);
q.wait_and_throw();
oneapi::mkl::stats::central_moment(q, mean_buf, dataset, variance_buf);
q.wait_and_throw();
```
  - Both two-array overloads issue four stats calls (`mean`/`central_moment` for each dataset) with only three `q.wait_and_throw()` calls: no wait separates `central_moment(q, mean1_buf, dataset1, variance1_buf)` from the following `mean(q, dataset2, mean2_buf)` in `t_test.cpp`, and no wait separates `central_moment(q, mean1, dataset1, variance1)` from `mean(q, dataset2, mean2)` in `t_test_usm.cpp`. The one-array overloads in both files wait after every stats call. The files do not comment on the omission.
- USM scalars/results use **shared** USM so the host can read them directly after the wait, and are freed with `sycl::free`:
```cpp
RealType* mean = sycl::malloc_shared<RealType>(1, q);
RealType* variance = sycl::malloc_shared<RealType>(1, q);
...
sycl::free(mean, q);
sycl::free(variance, q);
```
- Sample arrays: `sycl::buffer<fp_type, 1> rng_buf0(n_points);` (buffer API) vs `fp_type* rng_arr0 = sycl::malloc_shared<fp_type>(n_points, q);` (USM API).
- The **same** `engine` object is used for both `generate` calls, so the second call continues the engine's stream rather than restarting it.
- Queue is not requested in-order; the code relies on `q.wait_and_throw()` and buffer/accessor dependencies for ordering.
- `make_dataset` + the stats routines are called from host code with the queue as the first argument; no `submit` / command-group is written by the sample. The sources ignore these calls' return values, and the files never show whether they are asynchronous (see Explicit gaps).

## Build & Run
Sources of truth: `GNUmakefile` (Linux, GNU Make) and `makefile` (Windows, NMAKE).

Linux:
```make
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl="stats,rng"
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

t_test: t_test.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

t_test_usm: t_test_usm.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)
```
- Targets: `default`/`all` -> `run`; `run: t_test t_test_usm` then executes `./t_test` and `./t_test_usm`; `clean` runs `-rm -f t_test t_test_usm`.
- So `make` builds **and runs** both programs; `make clean` removes the two binaries.

Windows (NMAKE):
```make
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl="stats,rng" /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

t_test.exe: t_test.cpp
	icx-cl -fsycl t_test.cpp /Fet_test.exe $(DPCPP_OPTS)
```
- Targets: `default`/`all` -> `run` (executes `.\t_test.exe` then `.\t_test_usm.exe`); `clean` runs `del /q t_test.exe t_test_usm.exe`; `pseudo: clean run all`.
- `MKLROOT` is used on Windows for the include path (`/I"$(MKLROOT)\include"`).
- Device selection (README): the sample computes on the default SYCL device; set `ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"` to choose.
- Runtime arguments (from `main`): `argv[1]` = number of samples (`std::atol`; `0` falls back to `n_samples == 1000000`), `argv[2]` = mean (`std::atof`; NaN/Inf falls back to `expected_mean`), `argv[3]` = std dev (`std::atof`; `<= 0` falls back to `expected_std_dev`).

Expected output (README example for `./t_test`):
```
./t_test

Student's T-test Simulation
Buffer Api
-------------------------------------
Number of random samples = 1000000 with mean = 0, std_dev = 1
T-test result with expected mean: 1
T-test result with two input arrays: 1

TEST PASSED
```
The README repeats the block for `./t_test_usm` with `Unified Shared Memory Api` in place of `Buffer Api`. `main` prints `TEST FAILED` and returns `1` if either result is not `1`.

## Gotchas & Invariants
- At every call site `central_moment`'s 2nd argument is the **mean** (computed by the preceding `mean` call) and its 3rd is the dataset; the files do not document the routine's parameter list, so this order is established only by these call sites.
- Results are only guaranteed valid on the host after `q.wait_and_throw()` (USM shared writes) or after the relevant `host_accessor` is constructed (buffers). In the USM program every host read is preceded by a wait.
- Neither two-array overload waits between the first `central_moment` and the second `mean` (see Key Code Patterns); the one-array overloads wait after every stats call. Do not read the omission as "waits are optional" for USM: the USM two-array overload still waits before its host reads.
- Scalar statistics buffers/allocations are sized exactly `1` (`sycl::range{1}`, `sycl::malloc_shared<RealType>(1, q)`).
- `n` / `n1` / `n2` parameters are `std::int64_t`, while `n_points` in `main` is `size_t` and `n_samples` is `static const int`; the sample mixes these types freely.
- The two `t_test` overloads are distinguished purely by argument shape (buffer vs `RealType*`, and 4 vs 5 arguments) and share the same return-code contract; a `-1` result means no branch assigned a decision.
- USM memory is allocated with `sycl::malloc_shared` and must be released with `sycl::free(ptr, q)`; the USM program frees both statistics temporaries inside `t_test` and both sample arrays inside the `main` `try` block.
- `oneapi::mkl::rng::generate` is issued twice against one `engine`; the sample never re-seeds between the two calls.
- `std_dev <= 0` is replaced by the default `1.0f`, and a non-finite mean by `0.0f`; a user-supplied `0` sample count is replaced by `1000000`. A `variance` of `0` is not guarded, so the one-sample expression divides by `std::sqrt(variance)`.
- Threshold decisions are strict `<` comparisons against `static_cast<RealType>(threshold)`; `res` defaults to `-1` and any exception path terminates the process rather than returning `-1`.
- Both programs are `float` (`using fp_type = float;`); the `t_test` template `RealType` is instantiated only with `float` in this sample.

## Explicit gaps
- The full declared signatures (parameter lists, default arguments, free-function vs. overload set) of `oneapi::mkl::stats::make_dataset`, `oneapi::mkl::stats::mean`, `oneapi::mkl::stats::central_moment`, `oneapi::mkl::rng::default_engine`, `oneapi::mkl::rng::gaussian`, and `oneapi::mkl::rng::generate` are **not present** in these files; only the call forms quoted above are established.
- The meaning of the leading literal `1` in `make_dataset<...>(1, n, r)` is not stated by any of the files.
- The available values of `oneapi::mkl::stats::layout` beyond `row_major` are not shown; `row_major` is the only one used.
- Whether `make_dataset` / `mean` / `central_moment` accept non-buffer, non-pointer output types, multiple dimensions, or an explicit dimensions-vector argument is not shown.
- Whether oneMKL provides a dedicated `variance` routine: the sample stores a `central_moment` result in a local named `variance`, and the files never name a variance routine.
- `central_moment` is called with no moment-order argument; the files never state which central moment is computed or whether it is normalized by `n` or `n - 1`. The t-test formulas are transcribed verbatim without asserting a normalization convention.
- The pooled-variance two-sample expression uses `(n1 - 1) * (n1 - 1) * variance1 + (n2 - 1) * (n2 - 1) * variance2` over `(n1 + n2 - 2)` (the buffer version reads `variance1_acc[0]`/`variance2_acc[0]`, the USM version `variance1[0]`/`variance2[0]`); the files neither explain nor justify that expression, so this digest does not claim it matches any standard pooled formula.
- No CMake or CMakeLists-based build path is present among these files; only the GNU Make and NMAKE makefiles.
- `oneapi::mkl::rng::default_engine`'s concrete underlying engine, seeding semantics, reproducibility guarantees, and any required alignment for its `RealType*` outputs are not specified.
- The return values of the stats and RNG calls are ignored in the sources; the files never show what those calls return, so whether any of them is asynchronous or blocking is not established.
- Whether the statistics routines require a scratchpad/workspace query or particular queue properties is never shown; no descriptor or workspace object is constructed, and the queue is built from a selector plus an exception handler only.
- No device-side (kernel) or explicit `q.submit` usage appears, so nothing in these files establishes oneMKL behavior inside a command group.

## Source Map
- `README.md` — purpose, build instructions for Linux (`make`) and Windows (`nmake`), `ONEAPI_DEVICE_SELECTOR` device selection, expected sample output.
- `GNUmakefile` — Linux build: `icpx ... -fsycl` compile lines, `MKL_COPTS`/`DPCPP_OPTS`, `run`/`all`/`clean` targets.
- `makefile` — Windows NMAKE build: `icx-cl -fsycl` compile lines, `MKLROOT` include path, `/Qmkl-*` flags, `clean`/`pseudo` targets.
- `t_test.cpp` — buffer-API implementation: both `t_test` overloads using `sycl::buffer` + `host_accessor`, plus `main` with `default_engine`, `gaussian<float>`, two `generate` calls.
- `t_test_usm.cpp` — USM-API implementation: same algorithm with `sycl::malloc_shared`/`sycl::free` and raw-pointer outputs for `make_dataset`, `mean`, `central_moment`, `generate`.
