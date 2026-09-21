# American Options Pricing (Longstaff-Schwartz, RNG Host vs Device API)

Sample path: `input/oneMKL-samples/american_options/`. Built via **CMake** (the only sample in the repository with a CMake build; every other sample ships GNUmakefile/NMAKE only). Two executables from one source: `american_monte_carlo_host` and `american_monte_carlo_device`.

## Domain & Purpose

oneMKL **RNG** (`oneapi::mkl::rng`), demonstrating **both** models of the RNG API in one sample:

- **Host API** — `oneapi::mkl::rng::mrg32k3a` engine + `oneapi::mkl::rng::generate` called from the host onto a USM buffer, with the random numbers stored and then consumed by a separate kernel.
- **Device API** — `oneapi::mkl::rng::device::mrg32k3a` + `oneapi::mkl::rng::device::generate` called **inside** the kernel, so random numbers are generated and consumed immediately and the storage for them disappears.

Scope: American option pricing by Monte Carlo (Longstaff-Schwartz with an SVD-based regression). The sample is a migration of an NVIDIA CUDA sample to SYCL (via the Intel DPC++ Compatibility Tool); a pre-migration CUDA source `longstaff_schwartz_svd_2.cu` sits beside the SYCL one.

## Problem & Math

Monte Carlo path generation for American option valuation: `num_paths` price paths over `num_timesteps`, driven by standard-normal increments, then a backward induction that at each timestep exercises early if the immediate payoff beats the discounted continuation value estimated by regression (SVD least squares on basis functions). Relevant numeric constants are `gaussian<double>(0.0, 1.0)` draws and `SVD_NUM_ITERATIONS`.

This chapter documents the oneMKL RNG surface, which is the sample's stated purpose; the regression/basis-function math is hand-written in the sample and not a oneMKL routine.

## oneMKL Routines Used

Headers (both models; note the device header is separate from the umbrella `oneapi/mkl.hpp`):

```cpp
#include <oneapi/mkl/rng/device.hpp>
#include <oneapi/mkl.hpp>
```

**Device API — engine, distribution and generate, inside the path-generation kernel** (`longstaff_schwartz_svd_2.dp.cpp:152-159`, guarded by `#if USE_DEVICE_API`):

```cpp
#if USE_DEVICE_API
  oneapi::mkl::rng::device::mrg32k3a engine(0, (path * num_timesteps));
  oneapi::mkl::rng::device::gaussian<double> distr(0.0, 1.0);
  ...
    auto res = oneapi::mkl::rng::device::generate(distr, engine);
```

Call-site facts established by the source:

| routine | call form | notes |
|---|---|---|
| `oneapi::mkl::rng::device::mrg32k3a` | `(0, (path * num_timesteps))` | two arguments: a seed-like first value (`0`) and a per-work-item offset/stream position derived from the path index times step count |
| `oneapi::mkl::rng::device::gaussian<double>` | `(0.0, 1.0)` | explicit `double` template argument; two scalar arguments — standard-normal parameters |
| `oneapi::mkl::rng::device::generate` | `(distr, engine)` | two arguments, distribution first then engine by name; result assigned to `auto res` |

The device engine's per-path stream offset `path * num_timesteps` is how each path gets an independent, non-overlapping random stream; the sample does not document the parameter names or the units (counter vs offset).

**Host API — engine and generate enqueued from the host** (`:1024-1026`, guarded by `#if !USE_DEVICE_API`):

```cpp
#if !USE_DEVICE_API
    auto engine = oneapi::mkl::rng::mrg32k3a(*stream, 12354);
    oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<double>(0.0, 1.0), engine,
        num_timesteps * num_paths, d_samples);
#endif
```

Call-site facts:

| routine | call form | notes |
|---|---|---|
| `oneapi::mkl::rng::mrg32k3a` | `(*stream, 12354)` | engine constructed from a SYCL queue (dereferenced `stream`, a pointer) plus a seed |
| `oneapi::mkl::rng::gaussian<double>` | `(0.0, 1.0)` | same distribution shape as the device model, but in the host namespace |
| `oneapi::mkl::rng::generate` | `(distribution, engine, count, output pointer)` | count is `num_timesteps * num_paths`; output is `d_samples` (a device buffer) |

The host-model `generate` call is the same 4-argument shape used by the other RNG samples in this repository: `(distribution, engine, count, output USM pointer)`. Its result is discarded here.

**Explicit gaps:** the full declared signatures/overload sets of `mrg32k3a` (both models), `gaussian`, and `generate` are not present in the sample — only the call sites above. The exact seed parameter type, whether `mrg32k3a` is a class template, the meaning of the device engine's second argument, and the declared parameter types of `generate` are not established by these files. No oneMKL routine other than RNG appears in the sample.

## Key Code Patterns

**The pedagogical core: the device API deletes the storage buffer.** The README states the migration requires (1) a `USE_DEVICE_API` macro to distinguish implementations, (2) *removing the memory allocation that used to store random numbers*, (3) introducing device API calls in `generate_paths_kernel` so numbers are generated inside the kernel and consumed immediately, and (4) *not* calling the RNG host API as a separate call. That is the one-file, one-macro comparison this sample exists to show.

**Compile-time selection of the API model, not a runtime choice.** The two models are separate builds selected by the CMake option, which passes different macros *and* different link libraries (see Build & Run). `WITH_FUSED_BETA=1` is defined in both branches.

**Host model links the oneMKL SYCL RNG library explicitly; the device model does not.** This is the practical link-line consequence of the same design choice and is the most reusable detail here:

```cmake
# USE_DEVICE_API=1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -DWITH_FUSED_BETA=1 -DUSE_DEVICE_API=1 ")
add_executable (american_monte_carlo_device longstaff_schwartz_svd_2.dp.cpp)
target_link_libraries(american_monte_carlo_device sycl)

# default (host API)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -DWITH_FUSED_BETA=1 ")
add_executable (american_monte_carlo_host longstaff_schwartz_svd_2.dp.cpp)
target_link_libraries(american_monte_carlo_host sycl mkl_sycl_rng mkl_intel_ilp64 mkl_sequential mkl_core)
```

Note the contrast with the repository's other samples, which select domains with the compiler driver flag `-qmkl-sycl-impl=...` instead of spelling library names. Here the CMake build names the libraries directly (`mkl_sycl_rng`, `mkl_intel_ilp64`, `mkl_sequential`, `mkl_core`).

**FP64 capability is asserted, not branched.** Unlike the other samples, which test for FP64 and fall back to `float`, this sample uses the oneAPI sample-common helper, which throws when the aspect is missing:

```cpp
internal::has_capability_or_fail(stream->get_device(), sycl::aspect::fp64);
```

(`utils.hpp` implements `has_capability_or_fail`; the helper lives in the sample, not oneMKL.)

**`generate_paths_kernel` is the named kernel that hosts the device RNG calls.**

## Build & Run

Environment first (README): source `setvars` — Linux system-wide `. /opt/intel/oneapi/setvars.sh`, private install `. ~/intel/oneapi/setvars.sh`.

**CMake build (the sample's only build system; there is no makefile).** `CMAKE_CXX_COMPILER` is forced to `icpx`, the project name is `american_options`, the default `CMAKE_BUILD_TYPE` is `RelWithDebInfo` (set forcibly when unset), and executables land in `${CMAKE_BINARY_DIR}/bin` via `CMAKE_RUNTIME_OUTPUT_DIRECTORY`.

```
$ mkdir build
$ cd build
$ cmake ..              # host-API build  (or: cmake -D USE_DEVICE_API=1 ..  for the device API)
$ make
```

**Run targets** (custom targets that `cd` into `${CMAKE_SOURCE_DIR}/src/` before running the binary, because the sample reads data files relative to `src/`):

```
make run_host_api      # host RNG API build
make run_device_api    # device RNG API build (requires USE_DEVICE_API=1 configured)
```

**Device selection.** GPU by default; for CPU the README uses the OpenCL-style selector form:

```
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
make run_host_api
unset ONEAPI_DEVICE_SELECTOR
```

**Explicit gaps:** no expected-output block is given in the README, so no output format is documented here. No Linux/Windows NMAKE alternative exists for this sample. The README's `cmake_minimum_required`/`project` details are in `CMakeLists.txt`, not the README.

## Gotchas & Invariants

- **`USE_DEVICE_API` changes the link line, not just a macro.** Building with `-D USE_DEVICE_API=1` links only `sycl`; the default host-API build additionally requires `mkl_sycl_rng mkl_intel_ilp64 mkl_sequential mkl_core`. Getting this wrong is a link error, not a runtime one.
- **`make run_device_api` only works if the build was configured with `-D USE_DEVICE_API=1`.** The run target is created by whichever branch CMake took; the two branches are mutually exclusive.
- **The run targets `cd` into `src/` first** — the program resolves its input data relative to `src/`, so launching `bin/american_monte_carlo_host` directly from another directory can fail to find data.
- **`-fsycl` is supplied by the CMake build**, and `WITH_FUSED_BETA=1` is defined in both branches (it gates a kernel that assembles beta coefficients from partial sums).
- **FP64 is required, checked with `internal::has_capability_or_fail`** — the sample does not degrade to single precision, so a device without `fp64` fails rather than running reduced precision.
- **Two RNG models means two different headers**: `oneapi/mkl.hpp` for the host API and `oneapi/mkl/rng/device.hpp` for the device API.
- **The device engine's stream offset is per-path** (`path * num_timesteps`); dropping it would give every path the same random sequence and silently correlate paths.
- **A CUDA source is present** (`longstaff_schwartz_svd_2.cu`, ~54 KB) alongside the SYCL `.dp.cpp` (~49 KB). It is the pre-migration reference and is not part of either CMake target.
- **Explicit gap:** the README gives no expected output, no performance numbers, and no argument list, so invocation arguments and output format are not established by these files.

## Source Map

- `README.md` — purpose, host-vs-device RNG migration steps, CMake configure/build/run commands, `ONEAPI_DEVICE_SELECTOR=opencl:cpu` CPU path.
- `CMakeLists.txt` — top-level: forces `icpx`, `project(american_options)`, default `RelWithDebInfo`, sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_BINARY_DIR}/bin`, `add_subdirectory(src)`.
- `src/CMakeLists.txt` — the `USE_DEVICE_API` branch: flags, executable names (`american_monte_carlo_device` / `american_monte_carlo_host`), link libraries, and the `run_device_api` / `run_host_api` custom targets.
- `src/longstaff_schwartz_svd_2.dp.cpp` — the SYCL implementation: device RNG calls at `:152-159`, host RNG calls at `:1024-1026`, `generate_paths_kernel`, `WITH_FUSED_BETA` fused-beta kernel, Longstaff-Schwartz backward induction, `internal::has_capability_or_fail` FP64 assertions.
- `src/longstaff_schwartz_svd_2.cu` — the original CUDA implementation of the same algorithm (migration reference).
- `src/utils.hpp` — sample-local `internal` helpers, **not** oneMKL: `has_capability_or_fail(dev, aspect)` (throws `std::runtime_error` naming the device when `fp64` is absent) and `exclusive_scan(item, input, init, binary_op, group_aggregate)` wrapping `sycl::exclusive_scan_over_group` + `sycl::group_broadcast`.
