# Batched Linear Solver (Fortran, OpenMP Offload)

## Domain & Purpose

oneMKL **LAPACK** domain, *batched* LU factorization and solve, driven from Fortran via **OpenMP target offload**
(`!$omp target data` + `!$omp dispatch`) rather than SYCL queues/USM. The sample solves `batch_size` independent
`n x n` linear systems with `nrhs` right-hand sides each, then validates the result with a relative-residual check.

## Problem & Math

For each batch member `i`: factor `A_i = P_i L_i U_i`, then solve `A_i X_i = B_i` using the factored form.
`getrf_batch_strided` computes the factorization and pivot indices in place in `a`; `getrs_batch_strided` consumes
the factored `a` plus `ipiv` and overwrites the RHS array `b` with the solution `X`.

Verification (host side, plain BLAS):

```fortran
call dgemv('N', n, n, 1.0d0, a_orig(:,i), lda, b(:,(i-1)*nrhs+j), 1, 0.0d0, x, 1)
residual = norm2(b_orig(:,(i-1)*nrhs+j) - x(:)) / norm2(b_orig(:,(i-1)*nrhs+j))
```

Thresholds: `threshold = 1.0d-9` for `real(kind=8)`, `threshold = 1.0e-5` for `real(kind=4)` (`-DSP`).

## oneMKL Routines Used

Routines are invoked as **plain Fortran external calls** (no `use ... , only:` of individual procedure names is shown
for them). The module used is selected by preprocessor macro:

```fortran
#if defined(MKL_ILP64)
!$  use onemkl_lapack_omp_offload_ilp64   ! 64-bit
#else
!$  use onemkl_lapack_omp_offload_lp64    ! 32-bit
#endif
```

The offload machinery is pulled in by:

```fortran
!$ include "mkl_omp_offload.f90"
```

### Factorization — double precision (as called)

```fortran
!$omp target data map(tofrom:a) map(from:ipiv) map(from:info)
    !$omp dispatch
    call dgetrf_batch_strided(n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, info)
!$omp end target data
```

### Factorization — single precision (`-DSP`)

```fortran
call sgetrf_batch_strided(n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, info)
```

### Solve — double precision (as called)

```fortran
!$omp target data map(to:a) map(to:ipiv) map(tofrom: b) map(from:info)
    !$omp dispatch
    call dgetrs_batch_strided('N', n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, &
                                            b, ldb, stride_b, batch_size, info)
!$omp end target data
```

### Solve — single precision (`-DSP`)

```fortran
call sgetrs_batch_strided('N', n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, &
                                    b, ldb, stride_b, batch_size, info)
```

### Argument order, by position, with the values this sample passes

`getrf_batch_strided` (both precisions) — 9 arguments:

| # | passed value | note |
|---|---|---|
| 1 | `n` | first dimension argument; sample passes `n` (square case) |
| 2 | `n` | second dimension argument; sample passes `n` |
| 3 | `a` | input matrices on entry, factored matrices on exit (in place) |
| 4 | `lda` | `= n` |
| 5 | `stride_a` | `= n * lda` (elements between batch members) |
| 6 | `ipiv` | pivot output array |
| 7 | `stride_ipiv` | `= n` |
| 8 | `batch_size` | number of matrices in the batch |
| 9 | `info` | integer array of length `batch_size`; named `info` in `lu_solve_omp_offload.F90`, `info_rf` in the optimized variant |

`getrs_batch_strided` (both precisions) — 13 arguments:

| # | passed value | note |
|---|---|---|
| 1 | `'N'` | character trans argument |
| 2 | `n` | order of the factored matrices |
| 3 | `nrhs` | number of right-hand sides |
| 4 | `a` | factored matrices from `getrf_batch_strided` |
| 5 | `lda` | `= n` |
| 6 | `stride_a` | `= n * lda` |
| 7 | `ipiv` | pivots from `getrf_batch_strided` |
| 8 | `stride_ipiv` | `= n` |
| 9 | `b` | RHS on entry, solutions on exit (in place) |
| 10 | `ldb` | `= n` |
| 11 | `stride_b` | `= n * nrhs` |
| 12 | `batch_size` | number of systems |
| 13 | `info` | integer array of length `batch_size`; named `info` in `lu_solve_omp_offload.F90`, `info_rs` in the optimized variant |

The sample never uses an argument other than `'N'` for the `getrs_batch_strided` character argument.

### Host-side BLAS used only for verification

```fortran
call dgemv('N', n, n, 1.0d0, a_orig(:,i), lda, b(:,(i-1)*nrhs+j), 1, 0.0d0, x, 1)
```

Single-precision counterpart: `call sgemv('N', n, n, 1.0, a_orig(:,i), lda, b(:,(i-1)*nrhs+j), 1, 0.0, x, 1)`.

## Key Code Patterns

**No SYCL queue, no USM.** This sample contains no SYCL queue object, no USM allocation, and no scratchpad/workspace
query for the batch routines. Device work is expressed purely as OpenMP target regions; `!$omp dispatch` routes the
oneMKL call to the device while already inside the `target data` environment. That environment is created by the
included `mkl_omp_offload.f90`.

**Array layout / batching convention (Fortran column-major).** The batch index is the *second* array dimension, and
each matrix occupies a contiguous block of `stride_a` elements:

```fortran
lda         = n
stride_a    = n * lda
stride_ipiv = n
ldb         = n
stride_b    = n * nrhs
allocate (a(stride_a, batch_size), b(n, batch_size*nrhs),  &
          ipiv(stride_ipiv, batch_size), info(batch_size), &
          stat = allocstat, errmsg = allocmsg)
if (allocstat > 0) stop trim(allocmsg)
```

**Two regions (naive) vs one fused region (optimized).** `lu_solve_omp_offload.F90` uses two separate
`!$omp target data` regions, so `a` and `ipiv` are copied host↔device around each of the two calls. The optimized
variant fuses both dispatches into one region and changes the map clauses:

```fortran
!$omp target data map(to:a) map(tofrom: b) map(from:info_rf, info_rs)    &
!$                          map(alloc:ipiv(1:stride_ipiv, 1:batch_size))
    !$omp dispatch
    call dgetrf_batch_strided(n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, info_rf)

    !$omp dispatch
    call dgetrs_batch_strided('N', n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, &
                               b, ldb, stride_b, batch_size, info_rs)
!$omp end target data
```

The optimized mapping matches the README's stated invariant that factored matrices and pivots can be discarded:
`a` is `map(to:)` (never copied back — the host copy is left untouched), `ipiv` is `map(alloc:)` (device-only, never
transferred), and only the solution array `b` is `map(tofrom:)`. Because both calls share the region, the device-side
factored `a` is available to the second call without return to the host.

**Separate `info` arrays in the optimized version.** `info_rf` and `info_rs` are distinct arrays of length
`batch_size` so factorization and solve status can be attributed after the region closes:

```fortran
if (any(info_rf .ne. 0)) then
    print *, 'Error: getrf_batch_strided returned with errors.'
    stop
elseif (any(info_rs .ne. 0)) then
    print *, 'Error: getrs_batch_strided returned with errors.'
    stop
else
```

The non-optimized version uses one `info` array and checks `any(info .ne. 0)` immediately after each region.

**Timing** is host-side only, around the target regions: `call system_clock(start_time)` …
`call system_clock(end_time)`, then `cycle_time = dble(end_time - start_time) / dble(clock_precision)`;
`call system_clock(count_rate = clock_precision)` runs once before the cycle loop.

**Cycle loop.** `cycles` iterations, each re-randomizing `a` and `b`, saving `a_orig = a` / `b_orig = b` before the
regions, and accumulating `total_time`. A fresh factorization is therefore performed every cycle.

## Build & Run

Targets: `default`, `all` and `run_all` all resolve to the same thing; `make clean` removes artifacts.

```make
IFX_OPTS = -i8 -free -qmkl-ilp64
IFX_OPTS_OFFLOAD = -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -qmkl-sycl-impl=lapack

lu_solve: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS)

lu_solve_omp_offload: lu_solve_omp_offload.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)

lu_solve_omp_offload_optimized: lu_solve_omp_offload_optimized.F90
	ifx $< -o $@ $(IFX_OPTS) $(IFX_OPTS_OFFLOAD)
```

Expanded commands, as printed by the sample's own example session:

```
ifx lu_solve_omp_offload.F90 -o lu_solve -i8 -free -qmkl-ilp64
ifx lu_solve_omp_offload.F90 -o lu_solve_omp_offload -i8 -free -qmkl-ilp64 -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -qmkl-sycl-impl=lapack
ifx lu_solve_omp_offload_optimized.F90 -o lu_solve_omp_offload_optimized -i8 -free -qmkl-ilp64 -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -qmkl-sycl-impl=lapack
```

Note the CPU-only `lu_solve` binary is built from the **same** `lu_solve_omp_offload.F90` source but without the
offload/OpenMP flags (the README labels it "CPU-only, OpenMP disabled"), so three binaries are produced from two
source files.

`make` / `make run_all` also runs each binary:

```
./lu_solve -n 64 -b 8 -r 1 -c 2
./lu_solve_omp_offload -n 64 -b 8 -r 1 -c 2
./lu_solve_omp_offload_optimized -n 64 -b 8 -r 1 -c 2
```

Command-line options parsed by the program (pairs, no error checks; unknown option prints
`Unrecognized command-line option:` and `stop`s): `-n` matrix dimension, `-b` batch size, `-r` number of RHS,
`-c` cycles. Defaults in source: `n = 64`, `batch_size = 4096`, `nrhs = 1`, `cycles = 5`.

Environment: source the oneAPI `setvars` script before building. Device selection uses the default SYCL device;
`ONEAPI_DEVICE_SELECTOR` may be set to `"*:cpu"` or `"*:gpu"`. Single-precision-only devices (e.g. gen11, gen12)
require compiling with `-DSP`. The README's own suggested benchmark run is `-n 16000 -b 8 -r 1 -c 5` (the makefile's
default test sizes are too small to justify offload).

Expected output shape, quoted from the README's example session for `./lu_solve -n 64 -b 8 -r 1 -c 2`:

```
 Matrix dimensions:                    64
 Batch size:                     8
 Number of RHS:                     1
 Number of test cycles:                     2
 Computation completed successfully  2.849000000000000E-002 seconds
 Computation completed successfully  4.600000000000000E-005 seconds
 Total time:  2.853600000000000E-002 seconds
```

In that example (and in the two offloaded runs) the first reported cycle is much slower than the second; the source
does not state why.

## Gotchas & Invariants

- **Every batch call is preceded by `!$omp dispatch`.** The sample never calls `dgetrf_batch_strided` /
  `dgetrs_batch_strided` bare inside a `target data` region; each dispatch directive immediately precedes its call.
  Both sources also begin with `!$ include "mkl_omp_offload.f90"`, whose contents are outside this sample directory.
- **`a` is consumed in place.** On exit from `getrf_batch_strided` the array holds factored matrices, and
  `getrs_batch_strided` must be given that same factored array. The sample preserves a separate `a_orig` for the
  residual check.
- **`b` is consumed in place**: RHS on entry, solutions on exit. Keep `b_orig` if the RHS is needed afterwards.
- **Map clauses differ between the two variants.** In the optimized version `a` is `map(to:)` and `ipiv` is
  `map(alloc:)`, so the factorization results are not returned to the host; only `b` is `map(tofrom:)`. The README
  states the fusing exists to minimize host-device data transfer, but the source does not state whether splitting the
  region again would be a correctness or only a performance change.
- **`info` is per-batch, not scalar.** Its declared length is `batch_size`, and the check is `any(info .ne. 0)`.
  The optimized version keeps separate arrays (`info_rf`, `info_rs`) per call. Error handling is `print` + `stop`
  (whole-program abort); there is no recovery path.
- **Stride/leading-dimension relationships used here:** `lda = n`, `stride_a = n * lda`, `stride_ipiv = n`,
  `ldb = n`, `stride_b = n * nrhs`. `a` is allocated as `a(stride_a, batch_size)` and `ipiv` as
  `ipiv(stride_ipiv, batch_size)`, i.e. batch index in the second dimension.
- **`ipiv` is dimensioned to match the `map` section** in the optimized variant:
  `map(alloc:ipiv(1:stride_ipiv, 1:batch_size))` addresses the full allocated array.
- **Integer width is coupled to the build.** `-i8` and `-DMKL_ILP64` (with `-qmkl-ilp64`) are used together; the
  `MKL_ILP64` macro is what selects `use onemkl_lapack_omp_offload_ilp64` versus `..._lp64`. All `integer`
  declarations in the source are default kind and rely on `-i8`.
- **Conditioning is engineered, not required:** the diagonal/off-diagonal band boost (`+ 50.0`, `+ 20.0`) exists
  only to make random matrices well conditioned so the residual test passes. It is not an API requirement.
- **Residual failures do not abort:** they print `Warning: relative residual of ...` and continue; only nonzero
  `info` triggers `stop`.
- **No destroy/reset calls.** There is no handle to free, no descriptor to destroy, and no explicit warm-up call in
  the source. The much slower first cycle in the README's example output is not explained by anything in the source.
- Both binaries deallocate all arrays at the end; no device-side cleanup API is called between cycles.

## Explicit gaps

- **Argument names/semantics are not given by this source.** Only positional order and the concrete values passed
  are visible. Whether positions 1–2 of `getrf_batch_strided` are formally `m` and `n`, and the exact declared
  argument names of either routine, is **not established here**; the README only links to the oneMKL developer
  reference pages for `getrf_batch_strided` and `getrs_batch_strided`.
- Only the `'N'` case of the `getrs_batch_strided` character argument appears; any transpose/conjugate variant, and
  its permitted values, is not shown.
- No single-precision `real(kind=4)` run is demonstrated; the `-DSP` flag's effect is only stated in prose.
- Whether `stride_ipiv = n` is a hard requirement or merely a sufficient choice is not established.
- The mechanism by which `!$omp dispatch` selects and configures the device (queue creation, context, any internal
  scratchpad sizing) lives in `mkl_omp_offload.f90` and the `onemkl_lapack_omp_offload_*` modules, which are **not**
  part of this sample directory and were not read; whether `!$ include "mkl_omp_offload.f90"` is strictly required to
  compile is therefore not established.
- **Build-flag discrepancy:** the README's `-DSP` note shows a longer offload link line
  (`-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl`),
  whereas the makefile uses `-qmkl-sycl-impl=lapack` instead. Which is current is not established.
- **Filename/flag discrepancy:** the source file header comments reference different source names
  (`lu_solve_omp_offload_ex1_timer.F90`, `lu_solve_omp_offload_ex3_timer.F90`) and add `-qmkl-ilp64` to the GPU compile line; the makefile/README build lines are what is documented above.
- No workspace/scratchpad query routine for the batched factorization appears in the sample.
- No enumeration of possible `info` values (e.g. which `i` indicates a zero pivot) is provided beyond `!= 0`.
- `n`, `batch_size`, `nrhs`, `cycles` are declared `integer` with no argument validation; divisibility/range
  constraints imposed by the library are not stated.
- The cause of the much slower first timing cycle in the README's example output is not stated in the source.

## Source Map

- `README.md` — purpose, the two batch routine names, build/run narrative, `ONEAPI_DEVICE_SELECTOR`, `-DSP` note, example output.
- `makefile` — `IFX_OPTS` / `IFX_OPTS_OFFLOAD`, three build targets, `run_all` invocation lines, `clean`.
- `lu_solve_omp_offload.F90` — two separate `!$omp target data` regions; single `info` array; CPU and offload builds share this source.
- `lu_solve_omp_offload_optimized.F90` — one fused `!$omp target data` region with `map(to:a)`, `map(alloc:ipiv(...))`, separate `info_rf` / `info_rs`.
