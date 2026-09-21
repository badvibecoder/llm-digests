# Matrix Multiplication with oneMKL

## Domain & Purpose

oneMKL **BLAS** domain only (dense linear algebra). The sample is a GEMM benchmark: it multiplies two large matrices on a SYCL device and reports achieved FLOPS, plus a correctness check against the all-ones input.

## Problem & Math

Dense matrix-matrix multiply with scaling and update:

```
C := alpha * op(A) * op(B) + beta * C
```

As used here `op(A) = A`, `op(B) = B` (both `transpose::N`), `alpha = 1`, `beta = 0`, so it reduces to `C = A * B` with `A` being `M x K`, `B` being `K x N`, `C` being `M x N`. The sample prints the problem as `(M x K) x (K x N)`.

Performance accounting, verbatim from `matrix_mul_mkl.cpp`:

```cpp
/* Calculate and display performance */
auto op_count = double(M) * double(N) * double(K) * 2;
auto flops = op_count / avg;

flops *= 1e-9;
char unit = 'G';
if (flops >= 1000.) {
    flops *= 1e-3;
    unit = 'T';
}
if (flops >= 1000.) {
    flops *= 1e-3;
    unit = 'P';
}
```

`op_count` multiplies by 2 for one multiply plus one add per element, and `flops = op_count / avg` is therefore a *rate* (FLOP/s) despite the variable name; `avg` is seconds per GEMM call.

## oneMKL Routines Used

Only one oneMKL routine is called in the entire sample.

**`oneapi::mkl::blas::gemm`** — written unqualified as `blas::gemm` after `using namespace oneapi::mkl;` (the using-directive sits inside the `time_gemms` lambda).

Exact call site (`matrix_mul_mkl.cpp:47-51`), verbatim:

```cpp
using namespace oneapi::mkl;
using namespace std::chrono;
auto start = steady_clock::now();
for (int i = 0; i < runs; i++)
    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
Q.wait_and_throw();
```

Argument order and role *as passed at this call site*. The argument names below are the conventional BLAS positional names, used only for orientation; the declared oneMKL prototype is not visible in these files (see **Explicit gaps**), so only the source values in the third column are established by the sample:

| # | Argument | Value in source | Meaning |
|---|----------|-----------------|---------|
| 1 | queue | `Q` (`sycl::queue&`, parameter named `Q`) | queue the GEMM is submitted to |
| 2 | transa | `transpose::N` | `op(A) = A` |
| 3 | transb | `transpose::N` | `op(B) = B` |
| 4 | m | `M` (`int`) | rows of `op(A)` / rows of `C` |
| 5 | n | `N` (`int`) | cols of `op(B)` / cols of `C` |
| 6 | k | `K` (`int`) | cols of `op(A)` / rows of `op(B)` |
| 7 | alpha | `1` (int literal) | scalar multiplier on `op(A)*op(B)` |
| 8 | a | `A` (device `T*`) | matrix A buffer |
| 9 | lda | `lda` (`int`) | leading dimension of A |
| 10 | b | `B` (device `T*`) | matrix B buffer |
| 11 | ldb | `ldb` (`int`) | leading dimension of B |
| 12 | beta | `0` (int literal) | scalar multiplier on incoming `C` |
| 13 | c | `C` (device `T*`) | matrix C buffer, updated in place |
| 14 | ldc | `ldc` (`int`) | leading dimension of C |

Notes established by the source:

- No layout namespace qualifier is written: the call is exactly `blas::gemm`, with no row-major or column-major namespace in front of it. The README states this is "the default column-major layout, the traditional choice for BLAS".
- No events/dependency-list argument is passed; the return value of `gemm` is discarded (the statement's value is never used).
- `alpha`/`beta` are written as bare `1` and `0` int literals; no explicit cast to `T` appears.
- The routine is templated on the element type implicitly through `A`/`B`/`C`, which are `T*` from `malloc_device<T>`; instantiations exercised are `test<float>`, `test<double>`, `test<half>` (`half` = `sycl::half` via `using namespace sycl;`).

Header: `#include <oneapi/mkl.hpp>`.

## Key Code Patterns

**Device / context / queue setup** (`main`), no properties anywhere — the queue is plain and therefore not in-order:

```cpp
device D(default_selector_v);
device_info(D);
context C(D);
queue Q(C, D);
```

**USM device allocation** with padded leading dimensions:

```cpp
int lda = nice_ld<T>(M);
int ldb = nice_ld<T>(K);
int ldc = nice_ld<T>(M);

auto A = malloc_device<T>(lda * K, Q);
auto B = malloc_device<T>(ldb * N, Q);
auto C = malloc_device<T>(ldc * N, Q);
```

**Leading-dimension padding** (`utilities.hpp`) — byte-based, 512-byte aligned plus a 256-byte offset:

```cpp
template <typename T>
int nice_ld(int x)
{
    x = std::max(x, 1);
    x *= sizeof(T);
    x = (x + 511) & ~511;
    x += 256;
    x /= sizeof(T);
    return x;
}
```

**Filling device buffers** by replicating a host staging buffer (device copy is asynchronous; `replicate_data` waits at the end):

```cpp
template <typename T>
void replicate_data(sycl::queue &Q, T *dst, size_t dst_elems, const T *src, size_t src_elems)
{
    while (dst_elems > 0) {
        auto copy_elems = std::min(dst_elems, src_elems);
        Q.copy(src,  dst, copy_elems);
        dst += copy_elems;
        dst_elems -= copy_elems;
    }
    Q.wait();
}
```

**Sync point after the timed loop** — because the queue is out-of-order, timing is only valid after `Q.wait_and_throw()`; the `steady_clock` interval brackets the submit loop *and* the wait, so host submission overhead is included.

**Read-back and verification** of the all-ones product (`C` entries must equal `T(K)`), capped at `rd_size` elements:

```cpp
size_t elems = std::min(ldc * N, rd_size);
Q.copy(C, host_data, elems).wait();
```

**Warm-up before timing** (README: "to allow oneMKL to initialize and prepare GEMM kernels for execution"):

```cpp
std::cout << " -> Warmup...\n";
(void) time_gemms(1);

/* Time one GEMM call, and estimate how many calls will be required to keep the
 * GPU busy for 1s. */
auto tare = time_gemms(1);
int ncalls = std::max(4, std::min(1000, int(1. / tare)));
auto time = time_gemms(ncalls + 1) - tare;
auto avg = time / ncalls;
```

**Exception handling** — a single `try` around all device work, catching only `sycl::exception`:

```cpp
} catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    ...
    return 139;
}
```

**FP64 gating** via the device's `double_fp_config`:

```cpp
static
bool device_has_fp64(sycl::device const& D) {
    return (D.get_info<sycl::info::device::double_fp_config>().size() != 0);
}
```

**Teardown:** `free(C, Q); free(B, Q); free(A, Q);`

**Command-line parsing:** if `argv[1]` starts with a letter (`std::isalpha(argv[1][0])`) it is consumed as the type; then `M = N = K = atoi(argv[1])`, and if more than 3 args remain, `N = atoi(argv[2])` and `K = atoi(argv[3])`. Bounds: `M <= 0 || N <= 0 || K <= 0 || M > 1000000 || N > 1000000 || K > 1000000` triggers `usage()` (`std::exit(1)`). Types accepted: `double`, `single`, `float`, `half`, `all`.

## Build & Run

Environment first (README): source `setvars` — Linux `. /opt/intel/oneapi/setvars.sh` (sudo) or `. ~/intel/oneapi/setvars.sh` (user); Windows `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`.

**Linux (`GNUmakefile`)** — `make` picks `GNUmakefile`; default target is `run`.

```make
INCLUDE_COMMON=../../../common
MKL_COPTS = -DMKL_ILP64  -qmkl-ilp64 -qmkl-sycl-impl=blas
DPCPP_OPTS = -O2 $(MKL_COPTS)

matrix_mul_mkl: matrix_mul_mkl.cpp
	icpx -fsycl -I$(INCLUDE_COMMON) $< -o $@ $(DPCPP_OPTS)
```

Expanded command line:

```
icpx -fsycl -I../../../common matrix_mul_mkl.cpp -o matrix_mul_mkl -O2 -DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl=blas
```

Targets: `all` -> `matrix_mul_mkl`; `run` -> `./matrix_mul_mkl single` then `./matrix_mul_mkl double`; `clean` -> `-rm -f matrix_mul_mkl`; `.PHONY: clean run all`.

**Windows (`makefile`, NMAKE)** — default target `run`.

```make
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=blas /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

matrix_mul_mkl.exe: matrix_mul_mkl.cpp
	icx-cl -fsycl matrix_mul_mkl.cpp /Fematrix_mul_mkl.exe $(DPCPP_OPTS)
```

Targets: `all` -> `matrix_mul_mkl.exe`; `run` -> `.\matrix_mul_mkl.exe single` then `.\matrix_mul_mkl.exe double`; `clean` -> `del /q matrix_mul_mkl.exe matrix_mul_mkl.exp matrix_mul_mkl.lib`; `pseudo` -> `clean run all`. The `/I` include path expands `$(MKLROOT)`, so `MKLROOT` must be defined for that build.

**Device selection:** the sample uses the default SYCL device. README: set `ONEAPI_DEVICE_SELECTOR` to `"*:cpu"` or `"*:gpu"` to pick the device; the usage text also mentions `ONEAPI_DEVICE_SELECTOR`.

**Expected output** (first lines, assembled from the format strings in `device_info` and `test`; not a captured run):

```
oneMKL DPC++ GEMM benchmark
---------------------------
Platform:                ...
Device:                  ...
Driver_version:          ...
Core/EU count:           ...
Maximum clock frequency: ... MHz
FP64 capability:         yes|no

Benchmarking (4096 x 4096) x (4096 x 4096) matrix multiplication, single precision
 -> Initializing data...
 -> Verification...gemm  passes. for type: single precision
 -> Warmup...
 -> Timing...

Average performance: ...GF
```

`Average performance` is printed as value + unit char + `F`, unit in `{G, T, P}`.

## Gotchas & Invariants

- **The `transpose::N` / `transpose::N` + plain `blas::gemm` form is column-major.** Because A is `lda x K` and B is `ldb x N` with `lda >= M`, `ldb >= K`, and C is `ldc x N` with `ldc >= M`, the buffers must be sized by the *padded leading dimension* (`lda * K`, `ldb * N`, `ldc * N`), not by `M*K` etc.
- **Leading dimensions must not be the raw dimensions in this sample.** `nice_ld<T>` deliberately pads each of `lda`, `ldb`, `ldc` (byte-rounded to 512, plus 256 bytes, then converted back to elements), so `lda != M` in general. Passing `M`/`K` as leading dimensions while allocating `nice_ld`-sized buffers would break the layout.
- **`C`'s incoming contents are never read because `beta = 0`.** The sample achieves this by never initializing `C` on the device and calling `gemm` with `beta = 0`; only the `alpha * op(A) * op(B)` term contributes. If `beta` were nonzero, uninitialized device memory would be the addend.
- **Correctness check is partial.** It only inspects `min(ldc * N, rd_size)` elements (`rd_size = 1048576`) and compares against exact `T(K)`; it is not a full-matrix check.
- **Warm-up matters.** The first `gemm` is untimed (JIT/oneMKL initialization); timing uses `time_gemms(ncalls + 1) - tare`, i.e. the first of `ncalls+1` calls is subtracted as host overhead.
- **Out-of-order queue.** `Q.wait_and_throw()` (not `Q.wait()`) is used before stopping the clock; `Q.copy(...).wait()` is used for the read-back event.
- **`half` instantiation exists** (`test<half>`) and its verification compares `host_data[...] != T(K)`, so the ones-sum must be exactly representable for the chosen `K`.
- **`main` returns 139 from the `sycl::exception` handler**, `1` if `g_success` is false, `0` otherwise.
- **`test<double>`'s return value is not folded into `g_success`** — in the `"double"` branch the call is `test<double>(Q, M, N, K);` with no `g_success = ...` assignment, so a failed double-precision verification does not change the exit code. In the `"all"` branch it is assigned.
- **The `"none"` default-type path is dead.** `if ("none" == type) std::string type = device_has_fp64(D) ? "double" : "float";` declares a *shadowing* local `type`; the outer `type` stays `"none"`, falls through every `else if`, and reaches `else { type = "none"; usage(pname); }`. Invoking the binary with a numeric first argument (no type word) therefore prints usage and exits 1. Always pass an explicit type, as the `run` target does (`single`, `double`).
- **`M`, `N`, `K` come from the positional arguments that remain after the optional type word is shifted off** (`argc--; argv++`): a single remaining numeric argument sets all three (`M = N = K = std::atoi(argv[1])`), and when `argc > 3` after the shift, `N = std::atoi(argv[2])` and `K = std::atoi(argv[3])`. In a raw `pname type M N K` command line those are the original `argv[2..4]`; without a type word they are the original `argv[1..3]`.
- **Linux include path is `-I../../../common`**, not the sample directory; `utilities.hpp` is still resolved because `#include "utilities.hpp"` is a quoted include relative to the including file.
- **ILP64 is selected on both platforms** (`-DMKL_ILP64 -qmkl-ilp64` on Linux, `/Qmkl-ilp64` on Windows). What that implies for the `int` values passed to `gemm` is not explained by these files (see Explicit gaps).

## Explicit gaps

- The **declared signature/overload set of `oneapi::mkl::blas::gemm`** is not shown in any of these files. Only the 14-argument call site is visible; the parameter types (`std::int64_t` vs `int` for `m,n,k,lda,ldb,ldc`; `T` vs `const T&` for `alpha`/`beta`; whether an event-returning and/or dependency-list overload exists) cannot be established from the source.
- No **layout-namespace-qualified form of `gemm`** appears; the sample relies on the default layout only, so the exact spelling of any row-major/column-major namespace is not established here.
- The **`oneapi::mkl::transpose` enum's full set of values** is not established; only `transpose::N` is used, and no other enumerator appears in the sample.
- No **scratchpad / workspace query** or scratchpad USM allocation appears; whatever workspace this `gemm` call requires is not documented by these files.
- No **GEMM variant other than the plain `gemm` call** appears (no batched or bias-fused form), so those APIs are not established here.
- **`half`-precision GEMM support requirements** (device/extension gating beyond the FP64 check) are not established; the source only instantiates `test<half>`.
- No **CMakeLists.txt** exists in this sample directory, so no CMake build line is available.
- No **ahead-of-time / device-specific target flags** appear; the Linux build line is JIT-oriented, and the Windows line's `-fsycl-device-code-split=per_kernel` has no Linux counterpart.
- **`MKL_ILP64` semantics vs the compile flag** (whether the define alone suffices, and its interaction with `-qmkl-ilp64`) is not explained by the source.
- In the **`all` branch**, the `std::string type` declared in `main` is reassigned to `"half"`, `"float"`, `"double"` before each `test<T>` call; this affects only the `type` value printed by the `sycl::exception` handler.
- Whether `nice_ld<T>` guarantees `ld >= max(x,1)` is not asserted anywhere in the source; it follows from the arithmetic (`ceil(x*sizeof(T)/512)*512 + 256 >= x*sizeof(T)`), but the files state no such invariant.

## Source Map

- `README.md` — purpose, "default column-major layout" statement, warm-up and alignment best practices, `setvars` setup, `ONEAPI_DEVICE_SELECTOR` guidance, example output.
- `makefile` — Windows NMAKE build: `icx-cl` command line, `/Qmkl-ilp64` `/Qmkl-sycl-impl=blas`, `OpenCL.lib`, targets `all` / `run` / `clean` / `pseudo`.
- `GNUmakefile` — Linux GNU Make build: `icpx -fsycl`, `INCLUDE_COMMON=../../../common`, `-DMKL_ILP64 -qmkl-ilp64 -qmkl-sycl-impl=blas -O2`, targets `all` / `run` / `clean`.
- `matrix_mul_mkl.cpp` — the benchmark: `test<T>` template, the single `blas::gemm` call site, argument parsing, `device_info`, `device_has_fp64`, FP64 gating, `sycl::exception` handling, FLOPS math.
- `utilities.hpp` — `type_string<T>` specializations, `nice_ld<T>` leading-dimension padding, `generate_ones`, `generate_random_data`, `replicate_data`.
