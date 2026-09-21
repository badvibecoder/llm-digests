# LAPACK Routines, Part 1: Batch, Linear Equations, SVD

This chapter covers the first block of oneMKL `oneapi::mkl::lapack` routines (source pages 393-562): general-matrix bidiagonal/QR/RQ/LU solvers, the `*_batch` strided and group families, Hermitian eigen-solvers, and orthogonal-matrix generators. Each computational routine exists as a SYCL-buffer overload (returns `void`) and, except where noted, a USM overload (returns `sycl::event`). Every computational routine needs a scratchpad sized by its paired `*_scratchpad_size` function.

## Overview

**Namespace.** All signatures are inside `namespace oneapi::mkl::lapack`. The source pages for these LAPACK routines **do not state Include Files**; the only header printed in this page range belongs to the sparse routines (`oneapi/mkl/spblas.hpp`), which are outside this chapter.

**Linear-system matrix classes supported by the LAPACK package (intro, p393).** General; Banded; Symmetric or Hermitian positive-definite (full, packed, rectangular full packed (RFP)); Symmetric or Hermitian positive-definite banded; Symmetric or Hermitian indefinite (full and packed); Symmetric or Hermitian indefinite banded; Triangular (full, packed, RFP); Triangular banded; Tridiagonal; Diagonally dominant tridiagonal.

**Package-wide restrictions (p393).** `NOTE Different arrays used as parameters to oneMKL LAPACK routines must not overlap.` and `Warning LAPACK routines assume that input matrices do not contain IEEE 754 special values such as INF or NaN values. Using these special values may cause LAPACK to return unexpected results or become unstable.`

**Two API shapes.** *Buffer*: `sycl::buffer<T> &`, `sycl::buffer<RT> &`, `sycl::buffer<int64_t> &`; returns `void`; throws `mkl::lapack::exception` (batch: `mkl::lapack::batch_exception`). *USM*: `T *`, `RT *`, `int64_t *`; returns `sycl::event`; extra trailing `const std::vector<sycl::event> &events = {}` described as "List of events to wait for before starting computation. Defaults to empty list."; the return value is "Output event to wait on to ensure computation is complete."

**Scratchpad.** `scratchpad` is intermediate-result memory; `scratchpad_size` is "Size of scratchpad memory as a number of floating point elements of type T" and must be at least the value returned by the matching `*_scratchpad_size`. A too-small scratchpad surfaces as `info` equal to the passed `scratchpad_size` with non-zero `detail()`/`get_detail()`.

**Real type.** `RealT` (USM `gebrd`), `realT` (buffer `gebrd`), `RT` (`heevd`/`heevx`/`hegvd`/`hegvx`), or `T::value_type` (the `heevx`/`hegvx` scratchpad sizers). For real precisions `RT == T`. Supported precisions: `float`, `double`, `std::complex<float>`, `std::complex<double>`; the Hermitian routines accept only the complex types.

**Device-support annotations (verbatim meaning).** `CPU, GPU*` with `*Interface support only; all computations are performed on the CPU.`; `CPU and GPU*` with `*Hybrid support; some computations are performed on the CPU.` (also printed `CPU and GPU^` / `^Hybrid support; some computations are performed on the CPU.`); `CPU and GPU` or `CPU, GPU` (full support); bare `CPU` or `GPU` where only that device is supported.

**Batch model.** *Strided*: one flat array holds `batch_size` matrices separated by `stride_a` (also `stride_b`, `stride_tau`, `stride_s`, `stride_u`, `stride_vt`, `stride_ipiv`, `stride_ainv`). *Group*: `group_count` groups, per-group scalars indexed `_g` (`m_g`, `n_g`, `lda_g`, `ldb_g`, `nrhs_g`, `trans_g`, `k_g`), and `group_sizes[g]` problems in group `g`; total `batch_size` "is a sum of all parameter group sizes"; group pointer arrays (`T **a`, `int64_t **ipiv`, ...) "must be device-accessible."

**Error model.** `info = -i` (single) or `info = -n` (batch): i-th/n-th parameter illegal. `info` equal to the passed `scratchpad_size` with non-zero `detail()`/`get_detail()`: scratchpad too small. Batch routines also expose `ids()` (failing matrices in the batch) and `infos()` (first zero diagonal elements).

**Enums (spellings as printed).** `mkl::transpose` (`nontrans`, `trans`, `conjtrans`; also `mkl::transpose::notrans` in some batch text), `mkl::job` (`novec`, `vec`), `mkl::jobsvd` (printed `jobsvd::vectors`, `job::somevec`, `jobsvd::vectorsina`, `job::novec`), `mkl::uplo` (`upper`, `lower`), `mkl::rangev` (`all`, `values`, `indices`), `mkl::generate` (`q`, `p`).

## Routines

### gebrd

Reduces a general m-by-n matrix A to bidiagonal form B by an orthogonal (unitary) transformation; Q and P are not formed explicitly, only as products of elementary reflectors.
```cpp
namespace oneapi::mkl::lapack {
void gebrd(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<realT> &d, sycl::buffer<realT> &e, sycl::buffer<T> &tauq, sycl::buffer<T> &taup, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event gebrd(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, RealT *d, RealT *e, T *tauq, T *taup, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
Precision/devices: all four types — `CPU, GPU*` (interface only). `m` rows (0 ≤ m), `n` columns (0 ≤ n), `a` size ≥ `lda*n`, `lda` leading dimension. Outputs: if `m >= n`, diagonal and first super-diagonal of `a` hold the upper bidiagonal B, elements below the diagonal with `tauq` represent Q, elements above the first superdiagonal with `taup` represent P; if `m < n`, diagonal and first sub-diagonal hold the lower bidiagonal B, elements below the first subdiagonal with `tauq` represent Q, elements above the diagonal with `taup` represent P. `d` size ≥ `max(1, min(m,n))` (diagonal of B); `e` size ≥ `max(1, min(m,n) - 1)` (off-diagonal of B); `tauq` size ≥ `max(1, min(m, n))`; `taup` size ≥ `max(1, min(m, n))`.

### gebrd_scratchpad_size

Element count for the `gebrd` (buffer or USM) scratchpad.
```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t gebrd_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda)
}
```
`m` (0 ≤ m), `n` (0 ≤ n), `lda`. `mkl::lapack::exception` on a bad argument (position via `info()`).

### geinv_batch

Batch inverses `Ai^-1` of general matrices `Ai`, `i ∈ {1...batch_size}`; `batch_size` is the sum of `group_sizes`. Group Version only.
```cpp
namespace oneapi::mkl::lapack {
sycl::event geinv_batch(sycl::queue &queue, int64_t *n, T **a, int64_t *lda, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`float`/`double`/`std::complex<float>`/`std::complex<double>` — `CPU and GPU`. `n` = `group_count` orders `n_g` (0 ≤ `n_g`); `a` = `batch_size` pointers to `Ai`; `lda` = `group_count` values `lda_g` (`n_g` ≤ `lda_g`); `group_count` ≥ 0. Output `a` overwritten by the `n_g`-by-`n_g` matrices `Ai^-1`. Errors: `mkl::lapack::batch_exception`, `info = -n`, scratchpad condition.

### geinv_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t geinv_batch_scratchpad_size(sycl::queue &queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes)
}
```
`float`/`double`/`std::complex<float>`/`std::complex<double>` — `CPU and GPU`. Same input meanings as `geinv_batch`.

### gels

Least squares solution of an overdetermined system or minimum norm solution of an underdetermined system, by QR factorization of full-rank matrices; USM Version only; `m < n` is **not supported**.
```cpp
namespace oneapi::mkl::lapack {
sycl::event gels(sycl::queue &queue, mkl::transpose trans, int64_t m, int64_t n, int64_t nrhs, T *a, int64_t lda, T *b, int64_t ldb, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU`. If `m >= n` and `trans = transpose::nontrans`: `min ||A*X - B||`. If `m >= n` and `trans = (transpose::trans or tranpose::conjtrans)`: `min ||X|| s.t. A^H * X = B`. On exit B is overwritten with X. `trans` — real: `mkl::tranpose::nontrans` or `mkl::transpose::trans`; complex: `mkl::tranpose::nontrans` or `mkl::transpose::conjtrans` (verbatim spellings). `m >= n >= 0`; `nrhs >= 0`; `lda >= max(1,m)`; `b` must have room for X, i.e. `max(m,n)`-by-`nrhs` (m-by-nrhs if `nontrans`, else n-by-nrhs); `ldb >= max(1, max(m,n))`. Outputs: `a` holds triangular R; `b` holds the solutions.

### gels_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t gels_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb)
}
```
`trans=trans` → minimum norm solution of `A^T*X = B`; `conjtrans` → `A^H*X = B`; `nontrans` → `minimize || B - A*X ||`. `m ≥ 0`, `n ≥ 0`, `nrhs ≥ 0`, `lda >= max(1,m)`, `ldb >= max(1, max(m,n))`.

### gels_batch

Batch least squares solutions by QR factorization, same two cases as `gels`; B overwritten by X. Strided: "Currently only the `m >= n` and `trans = transpose::nontrans` case is supported"; Group: "Currently only `m >= n` case is supported".
```cpp
namespace oneapi::mkl::lapack {
void gels_batch(sycl::queue &queue, mkl::transpose trans, int64_t m, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<T> &b, int64_t ldb, int64_t stride_b, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event gels_batch(sycl::queue &queue, mkl::transpose trans, int64_t m, int64_t n, int64_t nrhs, T *a, int64_t lda, int64_t stride_a, T *b, int64_t ldb, int64_t stride_b, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event gels_batch(sycl::queue &queue, mkl::transpose *trans, int64_t *m, int64_t *n, int64_t *nrhs, T **a, int64_t *lda, T **b, int64_t *ldb, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
Strided variants: all four types — `GPU`. Group variant: all four types — `CPU and GPU`. `m >= n >= 0`; `nrhs >= 0`; `a` holds `batch_size` m-by-n matrices; `lda >= max(1,m)`; `stride_a >= max(1, lda * n)`; `b` each `Bi` sized to hold X (`max(m,n)`-by-`nrhs`, m-by-nrhs if `nontrans` else n-by-nrhs); `ldb >= max(1,max(m,n))`; `stride_b >= max(1, ldb * nrhs)`; `batch_size >= 0`. Group: `trans_g`, `m_g`, `n_g`, `nrhs_g`, device-accessible `a`/`b` pointers, `lda_g`, `ldb_g`, `group_count` ≥ 0, `group_sizes`. Outputs: `a` holds R — **the tau vectors are not recorded**; `b` holds the solutions. Errors: `mkl::lapack::batch_exception`; `info = -i`; scratchpad under-size; `info = 0` means `Ai` does not have full rank — batch indexes via `ids()`, first zero diagonal element indexes via `infos()`.

### gels_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t gels_batch_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size)
int64_t gels_batch_scratchpad_size(sycl::queue &queue, mkl::transpose *trans, int64_t *m, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes)
}
```
Strided: "Only `trans = mkl::transpose::nontrans` case is currently supported." Group: "Only the `trans = mkl::transpose::nontrans` case is currently supported." Constraints as in `gels_batch`.

### geqrf

QR factorization of a general m-by-n matrix A; no pivoting; Q is a product of `min(m, n)` elementary reflectors, not formed explicitly.
```cpp
namespace oneapi::mkl::lapack {
void geqrf(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event geqrf(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU*` (`*Hybrid support; some computations are performed on the CPU.`). `m` (0 ≤ m); `n` (0 ≤ n); `a` size ≥ `lda*n`; `lda` ≥ `max(1, m)`. Output: `a` — on/above the diagonal the `min(m,n)`-by-n upper trapezoidal R (upper triangular if `m >= n`), below the diagonal with `tau` the matrix Q. `tau` size ≥ `max(1, min(m, n))` (buffer) / ≥ `min(m,n)` (USM).

### geqrf_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t geqrf_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda)
}
```
`m` (0 ≤ m), `n` (0 ≤ n), `lda`.

### geqrf_batch

Batch `Qi*Ri` factorizations of general m-by-n matrices `Ai`; no pivoting; `Qi` not formed explicitly.
```cpp
namespace oneapi::mkl::lapack {
void geqrf_batch(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<T> &tau, int64_t stride_tau, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event geqrf_batch(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, int64_t stride_a, T *tau, int64_t stride_tau, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event geqrf_batch(sycl::queue &queue, int64_t *m, int64_t *n, T **a, int64_t *lda, T **tau, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All three variants, all four types — `CPU and GPU*` (hybrid). `m` (0 ≤ m), `n` (0 ≤ n), `lda >= max(1, m)`, `stride_a >= max(1, lda * n)`, `stride_tau >= max(1, min(m,n))`, `batch_size >= 0`; group uses `m_g`, `n_g`, `lda_g >= max(1, m_g)`, `a` pointers each of size `lda_g*n_g`, `tau` pointers, `group_count`, `group_sizes`. Outputs: `a` — on/above the diagonal the `min(m,n)`-by-n upper trapezoidal `Ri`; below it with `tau_i` the matrix `Qi`. `tau` — batch of `tau_i` each of size `min(m,n)`; group: `Ri` is `min(m_g,n_g)`-by-`n_g`.

### geqrf_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t geqrf_batch_scratchpad_size(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes)
geqrf_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size)
}
```
The Strided signature is printed **without a return type**. Group inputs `m_g`, `n_g`, `lda_g >= max(1, m_g)`, `group_count` ≥ 0, `group_sizes`. Strided inputs as in `geqrf_batch`.

### gerqf

RQ factorization of a general m-by-n matrix A; no pivoting; Q is a product of `min(m, n)` elementary reflectors. `NOTE This routine supports the Progress Routine feature.`
```cpp
namespace oneapi::mkl::lapack {
void gerqf(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event gerqf(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU` only. `m` (0 ≤ m), `n` (0 ≤ n), `a` size ≥ `lda*n`, `lda >= max(1, m)`. Output: `a` — if `m <= n`, the upper triangle of `a(1:m, n-m+1:n)` holds the m-by-m upper triangular R; if `m >= n`, elements on and above the (m-n)-th subdiagonal hold the m-by-n upper trapezoidal R; remaining elements with `tau` represent Q. `tau` size ≥ `min(m,n)`. Errors (USM page uses `get_info()`/`get_detail()`): `info = -i`; scratchpad under-size.

### gerqf_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template <typename T> int64_t gerqf_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda)
}
```
`m` (0 ≤ m), `n` (0 ≤ n), `lda` ≥ `max(1,m)`. Bad-argument position via `get_info()` (as printed).

### gesv

Solves `A*X = B` for a square coefficient matrix with multiple right-hand sides, via `A = P*L*U` (P permutation, L unit lower triangular, U upper triangular). USM Version only.
```cpp
namespace oneapi::mkl::lapack {
sycl::event gesv(sycl::queue &queue, int64_t n, int64_t nrhs, T *a, int64_t lda, int64_t *ipiv, T *b, int64_t ldb, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU, GPU`. `n` order of A (0 ≤ n); `nrhs >= 0`; `a` n-by-n A; `b` n-by-nrhs B; `ipiv` size ≥ `max(1, n)` "as returned by `getrf` (USM Version)". Outputs: `a` overwritten by L and U (`A = P*L*U`; unit diagonal of L not stored); `b` overwritten by X. Errors: `info = -i`; `info = i` the i-th diagonal element of U is zero and the solve could not be completed; scratchpad under-size.

### gesv_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t gesv_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb)
}
```
The printed signature contains `mkl::transpose trans`, but the source's own parameter list describes only `queue`, `n` (order of A and rows of B, 0 ≤ n), `nrhs` (0 ≤ nrhs), `lda`, `ldb` — no `trans`.

### gesvd

Singular value decomposition of a general rectangular m-by-n matrix A, optionally computing left and/or right singular vectors; `A = U*Sigma*V^T` (real) / `A = U*Sigma*V^H` (complex), `Sigma` m-by-n diagonal, U m-by-m, V n-by-n orthogonal/unitary; singular values real, non-negative, descending.
```cpp
namespace oneapi::mkl::lapack {
void gesvd(sycl::queue &queue, mkl::jobsvd jobu, mkl::jobsvd jobvt, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<RealT> &s, sycl::buffer<T> &u, int64_t ldu, sycl::buffer<T> &vt, int64_t ldvt, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event gesvd(sycl::queue &queue, mkl::jobsvd jobu, mkl::jobsvd jobvt, int64_t m, int64_t n, T *a, int64_t lda, RealT *s, T *u, int64_t ldu, T *vt, int64_t ldvt, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU` only. `jobu`/`jobvt` must be `jobsvd::vectors`, `job::somevec`, `jobsvd::vectorsina`, or `job::novec` (verbatim); `jobvt` and `jobu` cannot both be `jobsvd::vectorsina`. `m` (0 ≤ m), `n` (0 ≤ n), `a` size ≥ `lda*n`. Outputs: `a` is overwritten with the first `min(m,n)` columns of U if `jobu = jobsvd::vectorsina`, with the first `min(m,n)` rows of `VT`/`VH` if `jobvt = jobsvd::vectorsina`, otherwise destroyed; `s` size ≥ `max(1, min(m,n))` with `s(i) >= s(i+1)`; `u` size ≥ `ldu*m` (`jobu = jobsvd::vectors`) or ≥ `ldu*min(m, n)` (`jobu = job::somevec`), not referenced for `job::novec`/`jobsvd::vectorsina`; `vt` size ≥ `ldvt*n`, not referenced for `job::novec`/`jobsvd::vectorsina`. Errors: `info = -i`; `info = i` gives how many superdiagonals of the intermediate bidiagonal form B did not converge to zero and `scratchpad(2:min(m,n))` holds the unconverged superdiagonal elements of an upper bidiagonal B whose diagonal is in `s` (not necessarily sorted); B satisfies `A = U*B*VT` so it has the same singular values as A and singular vectors related by U and VT; scratchpad under-size.

### gesvd_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t gesvd_scratchpad_size(sycl::queue &queue, mkl::jobsvd jobu, mkl::jobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt)
}
```
`jobu`/`jobvt` must be `jobsvd::vectors`, `jobsvd::somevec`, `jobsvd::vectorsina`, or `jobsvd::novec`; the page also states "`jobvt` and `jobu` cannot both be `job::overwritevec`" (spelled differently from the computational routine).

### gesvda_batch

Truncated SVD factorizations for a batch of general matrices: `Ai = Ui * Si * Vi^T`, or the low-rank product `Ai = Pi * Qi` with `Pi = Ui * Si, Qi = Vi^T` when rows ≥ columns else `Pi = Ui, Qi = Si * Vi^T`. Three modes: by `irank`, by `tolerance` (values below tolerance treated as zero and set to zero), or by effective rank (values below tolerance times the largest singular value treated as zero). Can also compute singular values only.
```cpp
namespace oneapi::mkl::lapack {
void gesvda_batch(sycl::queue &queue, sycl::buffer<int64_t> &iparm, sycl::buffer<int64_t> &irank, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<RealT> &s, int64_t stride_s, sycl::buffer<T> &u, int64_t ldu, int64_t stride_u, sycl::buffer<T> &vt, int64_t ldvt, int64_t stride_vt, RealT tolerance, sycl::buffer<T> &residual, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
void gesvda_batch(sycl::queue &queue, int64_t *iparm, int64_t *irank, int64_t m, int64_t n, T *a, int64_t lda, int64_t stride_a, RealT *s, int64_t stride_s, T * u, int64_t ldu, int64_t stride_u, T * vt, int64_t ldvt, int64_t stride_vt, RealT tolerance, RealT *residual, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU, GPU*` (interface only). The USM signature is printed with return type `void` (see Gotchas).
`iparm` (dimension 16; `iparm[4]`-`iparm[15]` reserved; `*` = default): `iparm[0]`: `-1` use defaults (`iparm[0-2]=0, iparm[3]=1`), `0*` use `irank`, `1` use `tolerance`, `2` use effective rank. `iparm[1]`: `0*` both singular values and vectors, `1` singular values only. `iparm[2]`: `0*` three-matrix form `Ai = Ui * Si * Vi^T`, `1` low-rank product `Ai = Pi * Qi`. `iparm[3]`: source prints "If `iparm[2]=0`, the residual vector is not computed; If `iparm[2]=1*`, the residual vector is computed" (index printed as `iparm[2]` in both branches).
Inputs: `irank[i]` = number of singular values/vectors for `Ai` when `iparm[0]=0` or `-1`; `lda >= max(1, m)`; `stride_a >= max(1, lda * n)`; `stride_s >= min(m,n)`; `ldu >= max(1, m)`; `stride_u >= max(1, ldu * m)`; `ldvt >= max(1, n)`; `stride_vt >= max(1, ldvt * n)`; `tolerance` only used for `iparm[0]=1` and `2`; `batch_size`. Outputs: `irank` (computed counts when `iparm[0]=-1` or `0`); `a` unchanged unless the residual is required, else holds `Ai - Ui*Si*Vi^T` (`iparm[2]=0`) or `Ai - Pi*Qi` (`iparm[2]=1`); `s` size `min(m,n)*batch_size`; `u` size ≥ `stride_u*batch_size`; `vt` size ≥ `stride_vt*batch_size`; `residual` dimension `batch_size`, and if `iparm[3]=1` holds the Frobenius norm `|| Ai - Ui * Si * Vi^T ||` (`iparm[2]=0`) or `|| Ai - Pi * Qi ||` (`iparm[2]=1`). Errors: `mkl::lapack::batch_exception`; `info = -n` illegal parameter; `1` internal memory allocation failed; `2` invalid input parameter; `3` algorithm error computing singular values; `4` empty structure or matrix array.

### gesvda_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t gesvda_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t ldu, int64_t stride_u, int64_t ldvt, int64_t stride_vt, int64_t batch_size)
}
```
`lda >= max(1, m)`, `stride_a >= max(1, lda * n)`, `ldu >= max(1, m)`, `stride_u >= max(1, ldu * m)`, `ldvt >= max(1, n)`, `stride_vt >= max(1, ldvt * n)`. The prose also lists `stride_s`, which is absent from the printed signature.

### getrf

LU factorization `A = P*L*U` of a general m-by-n matrix with partial pivoting and row interchanges (L unit lower triangular / lower trapezoidal if `m > n`; U upper triangular / upper trapezoidal if `m < n`).
```cpp
namespace oneapi::mkl::lapack {
void getrf(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &ipiv, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrf(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, int64_t *ipiv, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU*` (hybrid). `m` (0 ≤ m), `n` (0 ≤ n), `a` size ≥ `lda * max(1, n)`, `lda >= max(1, m)`. Outputs: `a` overwritten by L and U (unit diagonal of L not stored); `ipiv` size ≥ `max(1,min(m,n))`, and for `1 <= i <= min(m, n)` row `i` was interchanged with row `ipiv(i)`. Errors: `info = -i`; `info = i` means `u_ii` is 0 — factorization completed but U is exactly singular (division by 0 if U is used to solve); scratchpad under-size.

### getrf_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t getrf_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda)
}
```
`m` (0 ≤ m), `n` (0 ≤ n), `lda >= max(1, m)`.

### getrf_batch

Batch LU factorizations `Ai = Pi * Li * Ui` with partial pivoting.
```cpp
namespace oneapi::mkl::lapack {
void getrf_batch(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<int64_t> &ipiv, int64_t stride_ipiv, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrf_batch(sycl::queue &queue, int64_t *m, int64_t *n, T **a, int64_t *lda, int64_t **ipiv, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event getrf_batch(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All three variants, all four types — `CPU and GPU`. `m` (m ≥ 0), `n` (n ≥ 0), `lda >= max(1, m)`, `stride_a >= max(1, lda * n)`, `stride_ipiv >= max(1, min(m,n))`, `batch_size >= 0`. Group: `m_g`, `n_g`, `lda_g >= max(1, m_g)`, `a` and `ipiv` as `batch_size` device-accessible pointers, each `ipiv_i` size ≥ `max(1,min(m_g, n_g))`, `group_count`, `group_sizes`. Outputs: `a` overwritten by `Li` and `Ui`; `ipiv` batch where for `1 <= k <= min(m,n)` row `k` of `Ai` was interchanged with row `ipiv_i(k)`. Errors: `mkl::lapack::batch_exception`; `info = -n`; scratchpad under-size; `info = 0` means some `Ui` diagonal is zero — indexes via `ids()`/`infos()`.

### getrf_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t getrf_batch_scratchpad_size(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes)
int64_t getrf_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size)
}
```
Group variant: all four types — `CPU and GPU`. Constraints as in `getrf_batch`.

### getrfnp

LU factorization `A = L*U` of a general m-by-n matrix **without pivoting** (L unit lower triangular / lower trapezoidal if `m > n`; U upper triangular / upper trapezoidal if `m < n`).
```cpp
namespace oneapi::mkl::lapack {
void getrfnp(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrfnp(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU*` (hybrid). `m` (m ≥ 0), `n` (n ≥ 0), `a` size ≥ `lda * max(1, n)`, `lda >= max(1, m)`. Output: `a` overwritten by L and U (unit diagonal of L not stored). Errors: `info = -i`; `info = i` means `u_ii` is 0 (U exactly singular); scratchpad under-size.

### getrfnp_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t getrfnp_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda)
}
```
`m` (m ≥ 0), `n` (n ≥ 0), `lda >= max(1, m)`.

### getrfnp_batch

Batch LU factorizations `Ai = Li * Ui` **without pivoting**.
```cpp
namespace oneapi::mkl::lapack {
void getrfnp_batch(sycl::queue &queue, int64_t m, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrfnp_batch(sycl::queue &queue, int64_t *m, int64_t *n, T **a, int64_t *lda, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event getrfnp_batch(sycl::queue &queue, int64_t m, int64_t n, T *a, int64_t lda, int64_t stride_a, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All three variants, all four types — `CPU and GPU`. `m` (m ≥ 0), `n` (n ≥ 0), `lda >= max(1, m)`, `stride_a >= max(1, lda * n)`, `batch_size >= 0`; group uses `m_g`, `n_g`, `lda_g`, device-accessible `a` pointers, `group_count`, `group_sizes`. Output: `a` overwritten by `Li` and `Ui`. Errors: `mkl::lapack::batch_exception`; `info = 0` with a zero diagonal in some `Ui` (indexes via `ids()`/`infos()`).

### getrfnp_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t getrfnp_batch_scratchpad_size(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes)
int64_t getrfnp_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size)
}
```
Constraints as in `getrfnp_batch`.

### getri

Computes `inv(A)` of a general matrix A; call `getrf` first to factorize A.
```cpp
namespace oneapi::mkl::lapack {
void getri(sycl::queue &queue, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &ipiv, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getri(sycl::queue &queue, int64_t n, T *a, int64_t lda, const int64_t *ipiv, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU`. `n` (0 ≤ n); `a` the buffer/pointer returned by `getrf`, size ≥ `lda*max(1,n)`; `lda` (`n <= lda`); `ipiv` as returned by `getrf`, dimension ≥ `max(1, n)`. Output: `a` overwritten by the n-by-n matrix `inv(A)` (the buffer page's input prose says "the n-by-n matrix A").

### getri_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t getri_scratchpad_size(sycl::queue &queue, int64_t n, int64_t lda)
}
```
`n` (0 ≤ n), `lda` (`lda >= n`).

### getri_batch

Batch inverses `Ai^-1` of general matrices `Ai` LU-factored by a matching `getrf_batch` variant. In-place (Buffer Strided, Group, USM Strided) and Out-of-place (Buffer Strided, USM Strided) forms exist.
```cpp
namespace oneapi::mkl::lapack {
void getri_batch(sycl::queue &queue, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<int64_t> &ipiv, int64_t stride_ipiv, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getri_batch(sycl::queue &queue, int64_t *n, T **a, int64_t *lda, const int64_t * const *ipiv, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event getri_batch(sycl::queue &queue, int64_t n, T *a, int64_t lda, int64_t stride_a, const int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
void getri_batch(sycl::queue &queue, int64_t n, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<int64_t> &ipiv, int64_t stride_ipiv, sycl::buffer<T> &ainv, int64_t ldainv, int64_t stride_ainv, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getri_batch(sycl::queue &queue, int64_t n, T *a, int64_t lda, int64_t stride_a, const int64_t *ipiv, int64_t stride_ipiv, T* ainv, int64_t ldainv, int64_t stride_ainv, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All variants, all four types — `CPU and GPU`. `n` (n ≥ 0); `a` result of `getrf_batch`; `lda >= max(1, n)`; `stride_a >= max(1, lda * n)`; `ipiv` as returned by `getrf_batch` (`const int64_t * const *` group/USM forms); `stride_ipiv >= max(1, n)`; `batch_size >= 0`; out-of-place adds `ldainv >= max(1, n)` and `stride_ainv >= max(1, ldainv * n)`. Group: `n_g`, `lda_g >= max(1, n_g)`, device-accessible `a`/`ipiv` pointers, `group_count`, `group_sizes`. Outputs: `a` (in-place) or `ainv` (out-of-place) overwritten by the n-by-n matrices `Ai^-1`. Errors: `mkl::lapack::batch_exception`; `info = -n`; scratchpad under-size.

### getri_batch_scratchpad_size

Three distinct printed signatures share this name: in-place Strided, Group, Out-of-place Strided.
```cpp
namespace oneapi::mkl::lapack {
int64_t getri_batch_scratchpad_size(sycl::queue &queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size)
int64_t getri_batch_scratchpad_size(sycl::queue &queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes)
int64_t getri_batch_scratchpad_size(sycl::queue &queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldainv, int64_t stride_ainv, int64_t batch_size)
}
```
Group inputs `n_g` (n_g ≥ 0), `lda_g >= max(1, n_g)`, `group_count`, `group_sizes`.

### getrs

Solves `A*X = B` (`trans=mkl::transpose::nontrans`), `A^T*X = B` (`trans`), or `A^H*X = B` (`conjtrans`) for an LU-factored square coefficient matrix; call `getrf` first.
```cpp
namespace oneapi::mkl::lapack {
void getrs(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &ipiv, sycl::buffer<T> &b, int64_t ldb, sycl::buffer<T> &scratchpad, int64_t   scratchpad_size)
sycl::event getrs(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, const T *a, int64_t lda, const int64_t *ipiv, T *b, int64_t ldb, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU` (buffer) / `CPU, GPU` (USM). `n` order of A and rows of B (0 ≤ n); `nrhs >= 0`; `a` factorization from `getrf`, size ≥ `lda*n`; `ipiv` size ≥ `max(1, n)`; `b` size ≥ `ldb*nrhs`; `ldb`. Output: `b` overwritten by X. Errors: `info = -i`; `info = i` the i-th diagonal element of U is zero and the solve could not be completed.

### getrs_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t getrs_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb)
}
```
`n` (0 ≤ n), `nrhs` (0 ≤ nrhs), `lda`, `ldb`.

### getrs_batch

Batch solves `Ai * Xi = Bi`, `Ai^T * Xi = Bi`, `Ai^H * Xi = Bi` from LU-factored matrices; precede with the matching `getrf_batch` variant.
```cpp
namespace oneapi::mkl::lapack {
void getrs_batch(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<int64_t> &ipiv, int64_t stride_ipiv, sycl::buffer<T> &b, int64_t ldb, int64_t stride_b, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrs_batch(sycl::queue &queue, mkl::transpose *trans, int64_t *n, int64_t *nrhs, const T * const *a, int64_t *lda, const int64_t * const *ipiv, T **b, int64_t *ldb, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event getrs_batch(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, const T *a, int64_t lda, int64_t stride_a, const int64_t *ipiv, int64_t stride_ipiv, T *b, int64_t ldb, int64_t stride_b, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All three variants, all four types — `CPU and GPU`. `n` (n ≥ 0); `nrhs` (nrhs ≥ 0); `a` factorizations from `getrf_batch`; `lda >= max(1, n)`; `stride_a >= max(1, lda * n)`; `ipiv`; `stride_ipiv >= max(1, n)`; `ldb >= max(1, n)`; `stride_b >= max(1, ldb * nrhs)`; `batch_size >= 0`. Group: `trans_g`, `n_g`, `nrhs_g`, device-accessible `a`/`ipiv`/`b` pointers, `lda_g >= max(1, n_g)`, `ldb_g >= max(1, n_g)`, `group_count`, `group_sizes`. Output: `b` overwritten by `Xi`. Errors: `mkl::lapack::batch_exception`; `info = -n`; scratchpad under-size; `info = 0` when some `Ui` has a zero diagonal (indexes via `ids()`/`infos()`).

### getrs_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t getrs_batch_scratchpad_size(sycl::queue &queue, mkl::transpose *trans, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes)
int64_t getrs_batch_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size)
}
```
Group variant: all four types — `CPU and GPU`. Group uses `trans_g`, `n_g`, `nrhs_g`, `lda_g >= max(1, n_g)`, `ldb_g >= max(1, n_g)`, `group_count`, `group_sizes`.

### getrsnp_batch

Batch solves `Ai * Xi = Bi`, `Ai^T * Xi = Bi`, `Ai^H * Xi = Bi` from **no-pivot** LU factorizations; precede with `getrfnp_batch` (without pivoting).
```cpp
namespace oneapi::mkl::lapack {
void getrsnp_batch(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<T> &b, int64_t ldb, int64_t stride_b, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event getrsnp_batch(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, const T *a, int64_t lda, int64_t stride_a, T *b, int64_t ldb, int64_t stride_b, int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
All four types — `CPU and GPU`. `n` (n ≥ 0); `nrhs` (nrhs ≥ 0); `a` factorizations from `getrfnp_batch`; `lda >= max(1, n)`; `stride_a >= max(1, lda * n)`; `ldb >= max(1, n)`; `stride_b >= max(1, ldb * nrhs)`; `batch_size >= 0`. Output: `b` overwritten by `Xi`. Errors: `mkl::lapack::batch_exception` including the `info = 0` zero-diagonal case with `ids()`/`infos()`.

### getrsnp_batch_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
int64_t getrsnp_batch_scratchpad_size(sycl::queue &queue, mkl::transpose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size)
}
```
Constraints as in `getrsnp_batch`.

### heevd

Computes all eigenvalues and, optionally, all eigenvectors of a complex Hermitian matrix A using a divide and conquer algorithm: `A = Z*Lambda*Z^H`, Lambda real diagonal with eigenvalues `lambda_i`, Z the (complex) unitary matrix whose columns are the eigenvectors `z_i`, so `A*z_i = lambda_i*z_i` for `i = 1, 2, ..., n`. With only eigenvalues requested it uses the Pal-Walker-Kahan variant of the QL or QR algorithm.
```cpp
namespace oneapi::mkl::lapack {
void heevd(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<RT> &w, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event heevd(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo, int64_t n, T *a, int64_t lda, RT *w, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`T = std::complex<float>`/`RT = float` and `T = std::complex<double>`/`RT = double`; buffer `CPU and GPU*` (interface only); USM `CPU and GPU*` for float, `CPU and GPU^` for double (hybrid). `jobz` must be `job::novec` or `job::vec`; `uplo` must be `uplo::upper` or `uplo::lower` (which triangle of A is stored); `n` (0 ≤ n); `a` size ≥ `lda*n`; `lda >= max(1,n)`. Outputs: `a` — if `jobz = job::vec`, overwritten by the unitary matrix Z of eigenvectors; `w` size ≥ n, eigenvalues ascending if `info = 0`. Errors: `info = -i`; `info = i` with `jobz = job::novec` — failed to converge, i is the number of off-diagonal elements of an intermediate tridiagonal form that did not converge to zero; with `jobz = job::vec` — failed to compute an eigenvalue while working on the submatrix in rows/columns `info/(n+1)` through `mod(info,n+1)`; scratchpad under-size.

### heevd_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> heevd_scratchpad_size(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo, int64_t n, int64_t lda)
}
```
Printed **without a return type**; the Return Values section says it returns "The number of elements of type T ...". `jobz` (`job::novec`/`job::vec`), `uplo` (`uplo::upper`/`uplo::lower`), `n` (0 ≤ n), `lda >= max(1,n)`.

### heevx

Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix A — eigenpairs `(lambda, z)` with `A*z = z*lambda` — selected by a range of values or a range of indices. Eigenvalues use the bisection algorithm; eigenvectors use a combination of a modified twisted factorization algorithm based on Inderjit Dhillon and Beresford Parlett's work and the inverse iteration algorithm followed by Gram-Schmidt orthogonalization.
```cpp
namespace oneapi::mkl::lapack {
void heevx(sycl::queue &queue, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda, RT vl, RT vu, int64_t il, int64_t iu, RT abstol, sycl::buffer<int64_t> &m, sycl::buffer<RT> &w, sycl::buffer<T> &z, int64_t ldz, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event heevx(sycl::queue &queue, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, T *a, int64_t lda, RT vl, RT vu, int64_t il, int64_t iu, RT abstol, int64_t *m, RT *w, T *z, int64_t ldz, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`std::complex<float>`/`float` and `std::complex<double>`/`double`; buffer `CPU and GPU*` (interface only); USM `CPU and GPU*` for float and `CPU and GPU^` for double (hybrid). `range` must be `rangev::all` (all eigenvalues/eigenvectors), `rangev::values` (eigenvalues in `(vl, vu]`), or `rangev::indices` (the `il`-th through `iu`-th). `a` holds the stored triangle of A, size ≥ `lda*n`; `lda >= max(1,n)`; `vl < vu` (not referenced for `all`/`indices`); `1 <= il <= iu <= n` if `n > 0`, `il = 1` and `iu = 0` if `n = 0` (not referenced for `all`/`values`); `abstol` — an approximate eigenvalue is accepted as converged when it lies in an interval `[a,b]` of width ≤ `abstol + eps * max( |a|,|b| )`, `eps` being machine precision, and if `abstol <= 0` then `eps*|T|` is used instead where `|T|` is the 1-norm of the tridiagonal matrix from reducing A to tridiagonal form; `ldz >= max(1,n)`. Outputs: `a` — the stored triangle including the diagonal is destroyed; `m` total eigenvalues found (`0 <= m <= n`); `w` size ≥ n, first `m` elements are the selected eigenvalues ascending; `z` — if `jobz = job::vec`, first `m` columns are the orthonormal eigenvectors, column i for `w(i)`; if `jobz = job::novec`, Z is not referenced. Supply at least `max(1, m)` columns in Z — for `rangev::values` the exact `m` is unknown in advance so an upper bound must be used. Errors: `info = -i`; `info = i` with `jobz = job::novec` (failed to converge) or `job::vec` (failed to compute an eigenvalue); scratchpad under-size.

### heevx_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t heevx_scratchpad_size(sycl::queue &queue, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, int64_t lda, T::value_type vl, T::value_type vu, int64_t il, int64_t iu, T::value_type abstol, int64_t ldz)
}
```
Input semantics identical to `heevx` (including the `abstol` convergence definition and the `il`/`iu` constraints).

### hegvd

Computes all eigenvalues, and optionally eigenvectors, of a complex generalized Hermitian positive-definite eigenproblem by a divide and conquer method: `A*x = lambda*B*x`, `A*B*x = lambda*x`, or `B*A*x = lambda*x`; A and B Hermitian with B positive definite.
```cpp
namespace oneapi::mkl::lapack {
void hegvd(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &b, int64_t ldb, sycl::buffer<RT> &w, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event hegvd(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *b, int64_t ldb, RT *w, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`std::complex<float>`/`float` (`CPU and GPU*`) and `std::complex<double>`/`double` (`CPU and GPU^`). `itype` must be 1 (`A*x = lambda*B*x`), 2 (`A*B*x = lambda*x`), or 3 (`B*A*x = lambda*x`); `jobz` (`job::novec`/`job::vec`); `uplo` (`uplo::upper`/`uplo::lower`, applied to both A and B); `n` (0 ≤ n); `a` size ≥ `lda*n`; `lda >= max(1,n)`; `b` size ≥ `ldb*n`; `ldb >= max(1,n)`. Outputs: `a` — if `jobz = job::vec` and `info = 0`, the matrix Z of eigenvectors normalized so `Z^H*B*Z = I` for `itype = 1` or 2 and `Z^H*inv(B)*Z = I` for `itype = 3`; if `jobz = job::novec`, the stored triangle of A including the diagonal is destroyed. `b` — if `info <= n`, overwritten by U or L from the Cholesky factorization `B = U^H*U` or `B = L*L^H`. `w` size ≥ n, eigenvalues ascending if `info = 0`. Errors: `info = -i`; for `info <= n`, `info = i` with `jobz = job::novec` (failed to converge; i is the number of non-converged off-diagonal elements of an intermediate tridiagonal form) or with `jobz = job::vec` (failed to compute an eigenvalue while working on the submatrix in rows/columns `info/(n+1)` through `mod(info,n+1)`); for `info > n`, `info = n + i` for `1 <= i <= n` means the leading minor of order i of B is not positive-definite — factorization of B could not be completed and no eigenvalues or eigenvectors were computed; scratchpad under-size.

### hegvd_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t hegvd_scratchpad_size(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb)
}
```
`itype` (1, 2, or 3), `jobz`, `uplo`, `n` (0 ≤ n), `lda >= max(1,n)`, `ldb >= max(1,n)`.

### hegvx

Computes selected eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian positive-definite eigenproblem `A*x = lambda*B*x`, `A*B*x = lambda*x`, or `B*A*x = lambda*x`, selected by range of values or range of indices. Same algorithms as `heevx` (bisection; twisted factorization + inverse iteration + Gram-Schmidt).
```cpp
namespace oneapi::mkl::lapack {
void hegvx(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &b, int64_t ldb, RT vl, RT vu, int64_t il, int64_t iu, RT abstol, sycl::buffer<int64_t> &m, sycl::buffer<RT> &w, sycl::buffer<T> &z, int64_t ldz, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event hegvx(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *b, int64_t ldb, RT vl, RT vu, int64_t il, int64_t iu, RT abstol, int64_t *m, RT *w, T *z, int64_t ldz, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`std::complex<float>`/`float` (`CPU and GPU*`) and `std::complex<double>`/`double` (`CPU and GPU^`). `itype` (1, 2, or 3); `jobz`; `range` (`rangev::all`/`values`/`indices` as in `heevx`); `uplo` (applied to A and B); `n` (0 ≤ n); `a` (`lda*n`); `lda >= max(1,n)`; `b` (`ldb*n`); `ldb >= max(1,n)`; `vl`/`vu`; `il`/`iu` (one-based, `1 <= il <= iu <= n` if `n > 0`, `il = 1`, `iu = 0` if `n = 0`); `abstol` (same convergence criterion as `heevx`); `ldz >= max(1,n)`. Outputs: `a` stored triangle including the diagonal destroyed; `b` overwritten by U or L from the Cholesky factorization `B = U^T*U` or `B = L*L^T`; `m` total eigenvalues found (`0 <= m <= n`); `w` first `m` eigenvalues ascending; `z` first `m` columns are the orthonormal eigenvectors when `jobz = job::vec`, else not referenced (supply at least `max(1, m)` columns). Errors: `info = -i`; `info = i` as in `heevx`; scratchpad under-size.

### hegvx_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t hegvx_scratchpad_size(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb, T::value_type vl, T::value_type vu, int64_t il, int64_t iu, T::value_type abstol, int64_t ldz)
}
```
Input semantics identical to `hegvx`.

### hetrd

Reduces a complex Hermitian matrix A to tridiagonal form T by a unitary similarity transformation: `A = Q*T*Q^H`; Q is not formed explicitly but represented as a product of `n-1` elementary reflectors.
```cpp
namespace oneapi::mkl::lapack {
sycl::event hetrd(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t   lda, sycl::buffer<T> &d, sycl::buffer<T> &e, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event hetrd(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *d, T *e, T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
The buffer overload is printed with return type `sycl::event` (unlike other buffer routines here). `std::complex<float>` and `std::complex<double>` — `CPU and GPU*` (interface only). `uplo` picks which triangle of A is stored; `n` (0 ≤ n); `a` size `(lda,*)`; `lda >= max(1, n)`. Outputs: `a` — for `uplo::upper`, diagonal and first superdiagonal overwritten by T and elements above the first superdiagonal with `tau` represent Q; for `uplo::lower`, diagonal and first subdiagonal overwritten by T and elements below the first subdiagonal with `tau` represent Q. `d` diagonal of T, size ≥ `max(1, n)`; `e` off-diagonal of T, size ≥ `max(1, n-1)`; `tau` size ≥ `max(1, n)` stores `(n-1)` reflectors and `tau(n)` is used as workspace.

### hetrd_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t hetrd_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)
}
```
`uplo`, `n` (0 ≤ n), `lda >= max(1, n)`.

### hetrf

Computes the Bunch-Kaufman factorization of a complex Hermitian matrix A: `A = U*D*U^H` if `uplo = uplo::upper`, `A = L*D*L^H` if `uplo = uplo::lower`. U and L are products of permutation and triangular matrices with unit diagonal (upper triangular for U, lower for L); D is Hermitian block-diagonal with 1-by-1 and 2-by-2 diagonal blocks, and U/L have 2-by-2 unit diagonal blocks corresponding to the 2-by-2 blocks of D. `NOTE This routine supports the Progress Routine feature.`
```cpp
namespace oneapi::mkl::lapack {
void hetrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &ipiv, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event hetrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, int64_t *ipiv, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`std::complex<float>` and `std::complex<double>` — `CPU` only. `uplo` — `uplo::upper`: `a` stores the upper triangle and A is factored as `U*D*U^H`; `uplo::lower`: `a` stores the lower triangle and A is factored as `L*D*L^H`; `n` (0 ≤ n); `a` size ≥ `lda*n`. Outputs: `a` stored triangle overwritten by details of D and the multipliers for U (or L); `ipiv` size ≥ `max(1, n)` — if `ipiv(i) = k > 0`, `d_ii` is a 1-by-1 block and row/column i was interchanged with row/column k; if `uplo = mkl::uplo::upper` and `ipiv(i) = ipiv(i-1) = -m < 0`, D has a 2-by-2 block in rows/columns i and i-1 and row/column i-1 was interchanged with row/column m; if `uplo = mkl::uplo::lower` and `ipiv(i) = ipiv(i+1) = -m < 0`, D has a 2-by-2 block in rows/columns i and i+1 and row/column i+1 was interchanged with row/column m. Errors: `info = -i`; `info = i` means `d_ii` is 0 — factorization completed but D is exactly singular (division by 0 if D is used to solve); scratchpad under-size.

### hetrf_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t hetrf_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)
}
```
`uplo` (with the factored form spelled out as above), `n` (0 ≤ n), `lda`.

### orgbr

Generates the whole or part of the real orthogonal matrices Q and P^T formed by `gebrd`; use after `sgebrd`/`dgebrd`. Common usage (verbatim): whole m-by-m Q `orgbr(queue, generate::q, m, m, n, a, ...)` (`a` must have at least m columns); n leading columns of Q if `m > n` `orgbr(queue, generate::q, m, n, n, a, ...)`; whole n-by-n P^T `orgbr(queue, generate::p, n, n, m, a, ...)` (`a` must have at least n rows); m leading rows of P^T if `m < n` `orgbr(queue, generate::p, m, n, m, a, ...)`.
```cpp
namespace oneapi::mkl::lapack {
void orgbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, T *a, int64_t lda, const T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`float` and `double` — `CPU` only. `gen` must be `generate::q` (generate Q) or `generate::p` (generate P^T); `m` rows of Q or P^T (0 ≤ m), with `m >= n >= min(m, k)` for `generate::q` and `n >= m >= min(n, k)` for `generate::p`; `n` (0 ≤ n); `k` = number of columns of the original m-by-k matrix reduced by `gebrd` (`generate::q`), or rows of the original k-by-n matrix (`generate::p`); `a` memory returned by `gebrd`; `tau` size `min(m,k)` for `generate::q` / `min(n,k)` for `generate::p`, the scalars returned by `gebrd` in `tauq` or `taup`. Output: `a` overwritten by n leading columns of the m-by-m orthogonal matrix Q or P^T (or the leading rows/columns thereof) as specified by `gen`, `m`, `n`.

### orgbr_scratchpad_size

```cpp
namespace oneapi::mkl::lapack {
template<typename T> int64_t orgbr_scratchpad_size(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, int64_t lda)
}
```
Input semantics identical to `orgbr`.

### orgqr

Generates the whole or part of the m-by-m orthogonal matrix Q of the QR factorization formed by `geqrf`; usually Q comes from an m-by-p matrix A with `m >= p`. Common usage (verbatim): whole Q `mkl::orgqr(queue, m, m, p, a, lda, tau, ...)`; leading p columns `mkl::orgqr(queue, m, p, p, a, lda, tau, ...)`; `Qk` of the leading k columns `mkl::orgqr(queue, m, m, k, a, lda, tau, ...)`; leading k columns of `Qk` `mkl::orgqr(queue, m, k, k, a, lda, tau, ...)`.
```cpp
namespace oneapi::mkl::lapack {
void orgqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, T *a, int64_t lda, const T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`float` and `double` — `CPU and GPU*` (interface only). `m` (0 ≤ m); `n` (0 ≤ n); `k` number of elementary reflectors (`0 <= k <= n`); `a` result of `geqrf`; `lda >= m`; `tau` result of `geqrf`. Output: `a` overwritten by n leading columns of the m-by-m orthogonal matrix Q. Errors: `info = -i`; the USM page lists `info = i`, `d_ii` is 0 (evidently copied text, inconsistent with `orgqr`); scratchpad under-size.

### orgqr_batch

Generates the whole or part of the orthogonal matrices `Qi` of the batch QR factorizations formed by `geqrf_batch`. Common usage (verbatim): whole `orgqr_batch(queue, m, m, p, a, ...)`; leading p columns `orgqr_batch(queue, m, p, p, a, ...)`; `Qik` `orgqr_batch(queue, m, m, k, a, ...)`; leading k columns of `Qik` `orgqr_batch(queue, m, k, k, a, ...)`.
```cpp
namespace oneapi::mkl::lapack {
void orgqr_batch(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda, int64_t stride_a, sycl::buffer<T> &tau, int64_t stride_tau, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgqr_batch(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *k, T **a, int64_t *lda, const T * const *tau, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
}
```
`float` and `double` — both variants `CPU and GPU*` (interface only). Strided: `m` (m ≥ 0), `n` (n ≥ 0), `k` (`0 <= k <= n`), `a` result of `geqrf_batch` (Buffer Strided), `lda >= max(1,m)`, `stride_a >= max(1, lda * n)`, `tau` result of `geqrf_batch`, `stride_tau >= max(1, min(m,n))`, `batch_size >= 0`. Group: `m`/`n`/`k` as previously supplied to `geqrf_batch` (Group Version) with `0 <= k_g <= n_g`, device-accessible `a`/`tau` pointers, `lda_g >= max(1, m_g)`, `group_count`, `group_sizes`. Output: `a` overwritten by n leading columns (strided) or by `n_g` leading columns of the `m_g`-by-`m_g` orthogonal matrices `Qi` (group).

## Formulas

Verbatim from the formula images (red tags given as `pNNNN#i`):

Reduction to bidiagonal form:
- `gebrd (p393, p0393#0, m >= n): A = Q*B*P^H = Q * ( B_1 ; 0 ) * P^H = Q_1 * B_1 * P^H` (`B_1` n-by-n upper diagonal; `Q_1` = first n columns of Q).
- `gebrd (p394, p0394#0, m < n): A = Q*B*P^H = Q * ( B_1  0 ) * P^H = Q * B_1 * P_1^H` (`B_1` m-by-m lower diagonal; `P_1` = first m columns of P).
- `gebrd USM (p396, p0396#0, m >= n): A = Q*B*P^H = Q * ( B_1 ; 0 )^T * P^H = Q_1 * B_1 * P^H` (the `^T` is printed outside the parenthesized column).
- `gebrd USM (p396, p0396#1, m < n): A = Q*B*P^H = Q * ( B_1  0 ) * P^H = Q * B_1 * P_1^H`

LU factorization with partial pivoting:
- `getrf USM (p453, p0453#0): A = P * L * U,`
- `getrf_batch Buffer Strided (p454, p0454#0): A_i = P_i x L_i x U_i`
- `getrf_batch Group (p456, p0456#0): A_i = P_i x L_i x U_i`
- `getrf_batch USM Strided (p458, p0458#0): A_i = P_i x L_i x U_i`

LU factorization without pivoting:
- `getrfnp USM (p465, p0465#0): A = L * U,`
- `getrfnp_batch Buffer Strided (p468, p0468#0): A_i = L_i x U_i`
- `getrfnp_batch Group (p470, p0470#0): A_i = L_i x U_i`
- `getrfnp_batch USM Strided (p472, p0472#0): A_i = L_i x U_i`

Total: 12 formula images transcribed (11 PNGs; `p0396` carries two formulas).

Formulas stated in prose (not images) in this range:
- `gels / gels_batch (p402, p405, p407, p411): min ||A*X - B||` for `m >= n`, `trans = transpose::nontrans`; `min ||X|| s.t. A^H * X = B` for `m >= n`, `trans = (transpose::trans or tranpose::conjtrans)`.
- `gesvd (p434, p437): A = U*Sigma*V^T` (real); `A = U*Sigma*V^H` (complex); `Sigma` m-by-n diagonal, U m-by-m, V n-by-n orthogonal/unitary; singular values real, non-negative, descending.
- `gesvda_batch (p442, p446): A_i = U_i * S_i * V_i^T; A_i = P_i * Q_i` with `P_i = U_i * S_i, Q_i = V_i^T` if rows >= columns, else `P_i = U_i, Q_i = S_i * V_i^T`; residual norms `|| Ai - Ui * Si * Vi^T ||` and `|| Ai - Pi * Qi ||`.
- `gesv (p431): A*X = B` via `A = P*L*U`.
- `getri / getri_batch (p476, p479, p486, p481): inv(A)`, `A_i^-1`.
- `getrs / getrs_batch / getrsnp_batch (p492, p496, p500, p507): A*X = B`, `A^T*X = B`, `A^H*X = B` per `trans`.
- `heevd (p512): A = Z*Lambda*Z^H`, `A*z_i = lambda_i*z_i` for `i = 1, 2, ..., n`.
- `heevx (p517): A*z = z*lambda`.
- `hegvd / hegvx (p525, p531): A*x = lambda*B*x`, `A*B*x = lambda*x`, `B*A*x = lambda*x`; normalization `Z^H*B*Z = I` (`itype = 1 or 2`), `Z^H*inv(B)*Z = I` (`itype = 3`).
- `hetrd (p540): A = Q*T*Q^H`.
- `hetrf (p544, p546): A = U*D*U^H` (`uplo::upper`), `A = L*D*L^H` (`uplo::lower`).

## Conventions & Gotchas

- **Leading dimensions are row strides**, not array sizes. Common minima: `lda >= max(1, m)` (general), `lda >= max(1, n)` (square/Hermitian), `lda >= m` (`orgqr`), `n <= lda` (`getri`/`getri_scratchpad_size`). Array-size requirements are separate (`lda*n`, `lda * max(1, n)`, `ldb*n`, `ldvt*n`).
- **No array overlap** is permitted among LAPACK routine parameters; **no NaN or INF inputs** (see package-wide warnings in the Overview).
- **Buffer vs USM.** Buffer calls return `void` (except `hetrd`, printed as `sycl::event`) and take no dependency vector; USM calls return `sycl::event` and take `const std::vector<sycl::event> &events = {}`. USM inputs are often `const` (`const T *a`, `const int64_t *ipiv` in `getrs`, `orgbr`, `orgqr`, `getri_batch`, `getrs_batch`, `getrsnp_batch`).
- **Scratchpad sizing** is in elements of type T, not bytes, and must come from the matching `*_scratchpad_size`; too-small scratchpads report `info == scratchpad_size` with non-zero `detail()`/`get_detail()`.
- **Strided strides.** `stride_a >= max(1, lda * n)`; `stride_b >= max(1, ldb * nrhs)`; `stride_ipiv >= max(1, n)` for the `getri`/`getrs` families but `>= max(1, min(m,n))` for the `getrf` family; `stride_tau >= max(1, min(m,n))`; `stride_s >= min(m,n)`; `stride_u >= max(1, ldu * m)`; `stride_vt >= max(1, ldvt * n)`; `stride_ainv >= max(1, ldainv * n)`.
- **Group batches.** `batch_size` = sum of `group_sizes`; per-group scalars indexed `_g`; pointer arrays must be device-accessible.
- **Required call ordering.** `getrs`/`getrs_batch` need `getrf`/`getrf_batch`; `getrsnp_batch` needs `getrfnp_batch`; `getri`/`getri_batch` need `getrf`/`getrf_batch`; `orgqr` needs `geqrf`; `orgqr_batch` needs `geqrf_batch`; `orgbr` needs `gebrd`.
- **Availability.** Strided `gels_batch` is `GPU` only; `gesvd`, `gerqf`, `orgbr`, `hetrf` are `CPU` only; `gebrd`, `geqrf`, `geqrf_batch`, `getrf`, `getrfnp`, `heevd`, `heevx`, `hegvd`, `hegvx`, `hetrd`, `orgqr`, `orgqr_batch`, `gesvda_batch` carry interface-only or hybrid footnotes; `geinv_batch`, `gels` (USM), `gesv` (USM), `gels_batch` (Group), `getrf_batch`, `getrfnp_batch`, `getri`, `getri_batch`, `getrs`, `getrs_batch`, `getrsnp_batch` are full `CPU and GPU`/`CPU, GPU`.
- **`hetrd` buffer overload returns `sycl::event`**, unlike every other buffer routine in this range.
- **`gesvda_batch` USM signature is printed `void`** yet still takes `events` and has an "Output event to wait on" Return Values section.
- **Missing return types.** `heevd_scratchpad_size` and `geqrf_batch_scratchpad_size` (Strided Version) are printed with no return type, though both are documented as returning an element count.
- **`gesv_scratchpad_size` prints an extra `mkl::transpose trans`** that its parameter table never describes; `gesv` itself has no `trans`.
- **Enum spelling is inconsistent in the source.** `gesvd` writes `jobsvd::vectors`, `job::somevec`, `jobsvd::vectorsina`, `job::novec`, while `gesvd_scratchpad_size` writes all four with the `jobsvd::` prefix plus a stray `job::overwritevec`. `gels`/`gels_batch` print `tranpose::nontrans` (typo for `transpose`) beside `transpose::trans`/`conjtrans`. `getrs_batch` text mixes `mkl::transpose::notrans` and `nontrans`.
- **Exception accessors differ by page.** Most use `info()` / `detail()`; `gerqf` (USM), `gerqf_scratchpad_size`, `gesvd` (USM) and `gesvd_scratchpad_size` use `get_info()` / `get_detail()`. `hegvd` prints `job::vec` as `job:vec` (on both the buffer and USM pages).
- **Pivot array element type.** Descriptions require integer pivot indices (`int64_t *ipiv` in USM), but the buffer signatures of `getrf`, `getri`, `getrs`, and `hetrf` are extracted as `sycl::buffer<T> &ipiv`; prefer `sycl::buffer<int64_t>` when writing code.
- **`getri_batch_scratchpad_size` is overloaded by form** (in-place Strided, Group, Out-of-place Strided) — pick the signature matching the `getri_batch` variant being called.
- **`heevx`/`hegvx` output sizing.** Only `m` eigenvalues are returned; when `range = rangev::values` the exact `m` is unknown in advance, so `z` must be sized with an upper bound of at least `max(1, m)` columns.

## Explicit gaps

- **Include Files are not stated** for any LAPACK routine in these pages; no header is given for the buffer or USM forms.
- **`orgqr_scratchpad_size` and `orgqr_batch_scratchpad_size`** are referenced by `orgqr`/`orgqr_batch` but their sections are not present in pages 546-562.
- **Only one overload is documented** for `gesv` (USM), `gels` (USM), and `geinv_batch` (Group); no buffer form of `gesv`/`gels` and no strided form of `geinv_batch` appears here.
- **`ungbr`** (complex analogue of `orgbr`) is named by `gebrd` but not documented in this page range.
- **`stride_s`** appears in the `gesvda_batch_scratchpad_size` prose but not in its printed parameter list.
- The eigenvalue-diagonal symbol in the `heevd`/`heevx`/`hegvd`/`hegvx` prose was lost in text extraction; the ASCII form `Lambda` is inferred from the surrounding sentence ("is a real diagonal matrix whose diagonal elements are the eigenvalues").
- The `gebrd USM` m >= n image (p0396#0) prints a superscript `^T` outside the parenthesized `B_1` column where the non-USM page does not; the discrepancy is in the source rendering and is reproduced rather than resolved.
- `tau` sizing for USM `geqrf` is printed "at least min(m,n)" while the buffer page says "at least max(1, min(m, n))"; both spellings are recorded.
