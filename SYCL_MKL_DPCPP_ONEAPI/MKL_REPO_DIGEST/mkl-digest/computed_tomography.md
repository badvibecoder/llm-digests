# Computed Tomography (Fourier Reconstruction)

## Domain & Purpose

oneMKL domain: `oneapi::mkl::dft` (Discrete Fourier Transform). The sample inverts the Radon transform of a 24-bit uncompressed BMP image by Fourier reconstruction: a batched 1D real forward DFT samples the image spectrum on a polar grid, a sample-provided SYCL kernel interpolates it onto a Cartesian grid, and one 2D real backward DFT produces the reconstructed image. Everything below is taken from the four files read for this chapter: `README.md`, `makefile`, `GNUmakefile`, `computed_tomography.cpp`.

## Problem & Math

Radon transform of `f(y,x)`: `radon[f](theta, v) = integral of f along the line x*cos(theta) + y*sin(theta) = v`.

Sampling grid used by the sample (`p` = number of projection angles, `q` = samples per angle, `S` = scanning width):

```
theta(i) = -0.5*M_PI + i*M_PI/p                  // 0 <= i < p
v(j)     = (-1.0 + (2.0*j + 1.0)/q)*0.5*S        // 0 <= j < q
```

Fourier slice theorem: `radon_hat(theta, ksi) = f_hat(ksi*sin(theta), ksi*cos(theta))`. Midpoint-rule approximation of the row DFT (eq. 1a in the source, whose row-slice notation is quoted as written):

```
radon_hat(theta(i), r/S)
  ~ exp(1i*M_PI*r*(1.0 - 1.0/q)) * (S/q) * DFT(R[i][0:q[; r)      // (eq. 1a)
```

Backward step (eq. 2), with `q x q` complex `G_HAT` interpolated from the polar samples, zeroed for `hypot(mm,nn) > 0.5*q`, and with a doubled real part on the `hypot(mm,nn) == 0.5*q` circle when exactly one of `mm`, `nn` is non-zero:

```
G[i,j] = (1/S^2) * iDFT(G_HAT; i, j)                              // (eq. 2)
```

`S = S_to_D * D`, where `D = std::hypot(hh, ww)` is the image diagonal, `hh = original.h*in_pix_len`, `ww = original.w*in_pix_len`, `in_pix_len = 1.0` (a `constexpr double`).

## oneMKL Routines Used

Namespace alias and descriptor type used verbatim throughout the source:

```cpp
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

namespace dft_ns = oneapi::mkl::dft;
using real_descriptor_t =
    dft_ns::descriptor<dft_ns::precision::DOUBLE, dft_ns::domain::REAL>;
using complex_t = std::complex<double>;
```

1D real descriptor: length from a single integer, then config parameters, then `commit`:

```cpp
real_descriptor_t radon_dft(q);
radon_dft.set_value(dft_ns::config_param::NUMBER_OF_TRANSFORMS, p);
radon_dft.set_value(dft_ns::config_param::FWD_DISTANCE, R.real_padded_width());
radon_dft.set_value(dft_ns::config_param::BWD_DISTANCE, R.complex_padded_width());
radon_dft.set_value(dft_ns::config_param::FORWARD_SCALE, S / q);
radon_dft.commit(R.queue);
auto compute_radon_hat = dft_ns::compute_forward(radon_dft, R.data, {dep});
```

`R.data` is `double*`; the dependency list is a braced initializer list holding one `sycl::event` (`dep` is a `sycl::event&` parameter of `reconstruction_from_radon`); the return value is consumed as a `sycl::event` (passed to `cgh.depends_on`).

2D real descriptor from an initializer list of sizes, backward scale, commit, backward compute:

```cpp
real_descriptor_t q_by_q_real_dft({q, q});
// Default strides are set by default for in-place DFTs (consistently with
// the implementation of padded_matrix)
q_by_q_real_dft.set_value(dft_ns::config_param::BACKWARD_SCALE, 1.0 / (S*S));
q_by_q_real_dft.commit(image.queue);
auto compute_g_values =
    dft_ns::compute_backward(q_by_q_real_dft, image.data, {interp_ev});
compute_g_values.wait();
```

Exact oneMKL identifiers confirmed present, fully qualified (the source reaches them through `dft_ns` = `oneapi::mkl::dft`): `oneapi::mkl::dft::descriptor`, `oneapi::mkl::dft::precision::DOUBLE`, `oneapi::mkl::dft::domain::REAL`, `oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS`, `oneapi::mkl::dft::config_param::FWD_DISTANCE`, `oneapi::mkl::dft::config_param::BWD_DISTANCE`, `oneapi::mkl::dft::config_param::FORWARD_SCALE`, `oneapi::mkl::dft::config_param::BACKWARD_SCALE`, `oneapi::mkl::dft::compute_forward`, `oneapi::mkl::dft::compute_backward`; plus the descriptor member functions `set_value` and `commit`, which the source calls as `radon_dft.set_value(...)`, `radon_dft.commit(R.queue)`, `q_by_q_real_dft.set_value(...)`, `q_by_q_real_dft.commit(image.queue)`. In the source all of these are spelled through the alias: `dft_ns::descriptor`, `dft_ns::precision::DOUBLE`, `dft_ns::domain::REAL`, `dft_ns::config_param::<name>`, `dft_ns::compute_forward`, `dft_ns::compute_backward`. The only overloads shown for `compute_forward` / `compute_backward` are the three-argument form `(descriptor, data_ptr, {event})`.

## Key Code Patterns

Queue creation requires fp64, and the failure path is a `sycl::exception` catch in `main`:

```cpp
sycl::queue main_queue;
try {
    // This sample requires double-precision floating-point arithmetic:
    main_queue = sycl::queue(sycl::aspect_selector({sycl::aspect::fp64}));
} catch (sycl::exception &e) {
    std::cerr << "Could not find any device with double precision support."
              << "Exiting." << std::endl;
    return 0;
}
```

USM is shared, allocated and freed through the queue; the padded layout is the whole point of the `padded_matrix` helper:

```cpp
inline int complex_padded_width() const {
    return (w / 2 + 1);
}
inline int real_padded_width() const {
    return 2 * complex_padded_width();
}
void allocate(int _h, int _w) {
    deallocate();
    h   = _h;
    w   = _w;
    data = sycl::malloc_shared<double>(h * real_padded_width(), queue);
}
void deallocate() {
    if (data)
        sycl::free(data, queue);
    data = nullptr;
    h = w = 0;
}
```

Element access for every real-domain matrix is `data[i * real_padded_width() + j]`; the DFT forward-domain distance is `real_padded_width()` and the backward-domain distance is `complex_padded_width()`. The in-place DFT's complex output is read by casting the same buffer:

```cpp
const complex_t *R_data_c = reinterpret_cast<complex_t*>(R.data);
complex_t *G_HAT = reinterpret_cast<complex_t*>(image.data);
const int R_data_c_ldw  = R.complex_padded_width();
const int G_HAT_ldw     = image.complex_padded_width();
```

Explicit cross-kernel ordering with SYCL dependencies and one host-side wait:

```cpp
auto interp_ev = image.queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(compute_radon_hat);
    ...
    cgh.parallel_for<class interpolateKernelClass>(
        sycl::range<2>(q, q/2 + 1), [=](sycl::item<2> item) { ... });
});
```

`radon_ev.wait(); // make sure it completes before exporting data` precedes the host-side BMP write of the radon image. `main` constructs `sycl::queue main_queue;` with no properties, and no in-order queue property is requested anywhere in the source; ordering is instead made explicit. `sycl::reduction(L1_error, sycl::plus<>())` is passed to a `cgh.parallel_for<class compute_errors_class>(sycl::range<2>(q, q), global_integral, ...)` in `compute_errors`, followed by `.wait()` then `sycl::free(L1_error, errors.queue)`.

Only defaults are relied on for placement: the source notes "oneMKL DFT descriptor operate in-place by default" and "Default strides are set by default for in-place DFTs".

## Build & Run

Linux (`GNUmakefile`), targets `default`/`all` -> `run`, plus `clean` and a `.PHONY: clean run all` line:

```
MKL_COPTS = -qmkl-ilp64 -qmkl-sycl-impl=dft
DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel

computed_tomography: computed_tomography.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

run: computed_tomography
	./computed_tomography
```

Windows (`makefile`, for NMAKE), targets `default`/`all` -> `run`, `clean`, `pseudo`:

```
DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl-ilp64 /Qmkl-sycl-impl=dft /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

computed_tomography.exe: computed_tomography.cpp
	icx-cl -fsycl $? /Fe$@ $(DPCPP_OPTS)

run: computed_tomography.exe
	.\computed_tomography.exe
```

The sample directory contains no `CMakeLists.txt` (its files are `computed_tomography.cpp`, `GNUmakefile`, `makefile`, `README.md`, `License.txt`, `input.bmp`), so no CMake build path exists for this sample. Environment: source `setvars` (`. /opt/intel/oneapi/setvars.sh` system-wide, `. ~/intel/oneapi/setvars.sh` private). Device selection: `ONEAPI_DEVICE_SELECTOR` set to `"*:cpu"` or `"*:gpu"` (otherwise the default SYCL device is used). `clean` removes, on Linux, `computed_tomography radon.bmp restored.bmp errors.bmp`; on Windows, `computed_tomography.exe computed_tomography.exp computed_tomography.lib radon.bmp restored.bmp errors.bmp`.

Command-line arguments: `p q in radon_out restored_out S_to_D err_out crop`, defaults `p=400`, `q=400`, `in="input.bmp"`, `radon_out="radon.bmp"`, `restored_out="restored.bmp"`, `S_to_D=1.0`, `err_out="errors.bmp"`, `crop=1`. Validation: invalid if `argc > 9 || p <= 0 || q <= 0 || S_to_D < 1.0 || crop < 0 || crop > 1`. `-h`/`-H` as `argv[1]` prints usage and exits (`EXIT_SUCCESS`).

Expected output (README, 400x400):

```
./computed_tomography
Reading original image from input.bmp
Generating Radon transform data from input.bmp
Saving Radon transform data in radon.bmp
Reconstructing image from the Radon projection data
        Step 1 - Batch of 400 real 1D in-place forward DFTs of length 400
        Step 2 - Interpolating spectrum from polar to cartesian grid
        Step 3 - In-place backward real 2D DFT of size 400x400
Saving restored image in restored.bmp
```

Exit codes: `EXIT_SUCCESS` normally; `EXIT_FAILURE` (and `errors.bmp` written) when `mean_error / max_input_value > 0.1` (`arbitrary_error_threshold`, a `constexpr double`). The fp64-queue failure path instead prints a message and `return 0` from `main` (exit status 0). `die()` prints `Fatal error: ...` to stderr and calls `std::exit(EXIT_FAILURE)`.

## Gotchas & Invariants

- Real in-place batched DFTs require forward-domain padding: for unit-stride real in-place transforms the source states the distance in the forward domain (elements real) must be twice the distance in the backward domain (elements complex), so `FWD_DISTANCE` is set to `real_padded_width()` (= `2*(w/2+1)`) and `BWD_DISTANCE` to `complex_padded_width()` (= `w/2+1`). The source's comment says padding is required in the forward domain "to store all the backward domain's elements"; how a mis-sized buffer would fail is not described.
- `complex_padded_width()` is `w/2 + 1`, the size the sample reserves for the backward (complex) domain of a real 1D transform of length `w`; `real_padded_width()` is twice that.
- Data must be `double` (fp64): the queue is created with `sycl::aspect_selector({sycl::aspect::fp64})`; if that construction throws `sycl::exception`, `main` reports it and returns 0 rather than failing.
- Scaling is applied through the descriptor, not by post-processing: `FORWARD_SCALE = S/q` for step 1 and `BACKWARD_SCALE = 1.0/(S*S)` for step 3.
- Ordering across the DFT and the interpolation kernel is explicit (`{dep}`, `cgh.depends_on`, `.wait()`); nothing in the source relies on in-order queue execution.
- The interpolation kernel runs over `sycl::range<2>(q, q/2 + 1)` and leaves the conjugate half unset; the source justifies this by `G_HAT[q - m][q - n] = conj(G_HAT[m][n])`, which it says is consistent with the requirements for a well-defined backward real 2D DFT.
- The sample computes `theta` over `[-0.5*M_PI, +0.5*M_PI]` and maps out-of-range `i` to the complex conjugate of `R_data_c[(i%p) * R_data_c_ldw + r]`; `i % (2*p) >= p` triggers `std::conj`.
- The `q/2` divisions in the sample are integer divisions (source note), so odd `q` is not validated (only `q > 0` is) and its behavior is not established by the source.
- The 2D descriptor is built with `{q, q}` and no strides/distances are set; the source states only that defaults apply for in-place DFTs.
- Accuracy guidance is emitted only in the error path: it recommends that `p*S_to_D` and `q` be commensurate, that `q` be increased to `ceil(S/(2.0*in_pix_len))` or more to alleviate blurring, and otherwise that `S_to_D` and `q` be increased proportionally.
- `padded_matrix` is non-copyable (`padded_matrix(const padded_matrix&) = delete;` also deletes `operator=`), owns its USM, and frees in `deallocate()` and its destructor.

## Explicit gaps

- The full overload set of `oneapi::mkl::dft::compute_forward` / `compute_backward` is not shown; only `(descriptor, double*, {sycl::event})` appears. Variants taking explicit strides, a `sycl::queue`, or an in-place/out-of-place flag are not established by these files.
- Whether `.set_value` calls must precede `.commit` is not stated; the source only shows that ordering.
- No descriptor `destroy`/reset call appears anywhere; the source relies on descriptor lifetime alone, so required cleanup semantics are not established.
- The exact `sycl::aspect_selector` overload used (`{sycl::aspect::fp64}`) is quoted from the source, but its contract is not documented there.
- Whether `FWD_DISTANCE`/`BWD_DISTANCE` are counted in elements of the respective domain type is implied by the code but never stated in the source.
- What ordering a default-constructed `sycl::queue` has is not stated in the source; it only shows that no in-order property is requested.
- No `CMakeLists.txt` exists in the sample directory, so no CMake build path or CMake flags for this sample are established.
- The meaning of the compiler flags `-qmkl-ilp64` and `-qmkl-sycl-impl=dft` is not explained in any file read.
- Constraints on `q` (odd vs even) and on `p` are only soft recommendations in the error path; hard divisibility requirements are not established.

## Source Map

- `computed_tomography.cpp` — full sample: `padded_matrix` USM helper, BMP read/write, `acquire_radon` Radon-transform SYCL kernel, `reconstruction_from_radon` (both oneMKL DFT calls), `compute_errors`, `main` with CLI parsing and validation.
- `README.md` — purpose, Fourier-reconstruction description, build/run instructions, `ONEAPI_DEVICE_SELECTOR`, expected output.
- `GNUmakefile` — Linux `icpx` build with `-qmkl-ilp64 -qmkl-sycl-impl=dft -fsycl-device-code-split=per_kernel`, `run`/`clean` targets.
- `makefile` — Windows NMAKE build with `icx-cl`, `DPCPP_OPTS`, `run`/`clean`/`pseudo` targets.
