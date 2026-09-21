# LAPACK Routines, Part 2: Eigenvalues, Orthogonal, Triangular

This chapter covers the `oneapi::mkl::lapack` routines documented on PDF pages 546–703: the Bunch-Kaufman factorizations `hetrf`/`sytrf`, the orthogonal/unitary generators and appliers (`org*`, `orm*`, `ung*`, `unm*`), the Cholesky family (`potrf`, `potri`, `potrs`, with the `potrf_batch`/`potrs_batch` batch forms), the symmetric/generalized-symmetric eigensolvers (`syevd`, `syevx`, `sygvd`, `sygvx`), the tridiagonal reduction `sytrd`, and the triangular routines (`trtri`, `trtrs`). Most routines appear in a buffer form (`sycl::buffer<T>&`, returns `void`) and/or a USM form (raw pointers, returns `sycl::event`), with batch forms as strided or group variants. These pages end where the separate Vector Mathematical Functions (VM) domain begins; that domain is out of scope here.

## Overview

**Namespace.** Every routine here belongs to `namespace oneapi::mkl::lapack`.

**Include files.** Not stated on any of these pages (the source lacks an Include Files section for every routine in this chapter). Only the namespace is given. Do not infer a header from this chapter alone.

**Two API families.**
- *Buffer*: `sycl::buffer<T>&` operands, returns `void`, reports failures by throwing; has **no** `events` parameter.
- *USM*: raw `T *` operands, returns an output `sycl::event` ("Output event to wait on to ensure computation is complete"), and takes a trailing `const std::vector<sycl::event> &events = {}` — "List of events to wait for before starting computation. Defaults to empty list."

**Scratchpad model.** Every computational routine takes `scratchpad` plus `scratchpad_size`: "Size of scratchpad memory as a number of floating point elements of type T. Size should not be less than the value returned by the `<routine>_scratchpad_size` function." Sizing is in elements of `T`, not bytes. Every `*_scratchpad_size` entry point that prints a return type returns `int64_t` (the lone exception is `orgtr_scratchpad_size`, which prints none — see Explicit gaps); most are declared `template<typename T>`, while the batch group/strided sizes are non-template in the extracted text.

**Precision/device notation** used below:
- `CPU and GPU` — as listed, unqualified.
- `CPU and GPU*` + `*Interface support only; all computations are performed on the CPU.`
- `CPU and GPU*` / `CPU, GPU*` + `*Hybrid support; some computations are performed on the CPU.`
- `CPU and GPU^` + `^Hybrid support; some computations are performed on the CPU.`
- plain `CPU` — no device form offered.

**Integer types.** All dimensions, leading dimensions, strides, group counts and `batch_size` are `int64_t`; USM pivot arrays are `int64_t *ipiv`.

**Enums (verbatim values).** `mkl::uplo` = `uplo::upper` / `uplo::lower`; `mkl::job` = `job::novec` / `job::vec` (parameter is `mkl::job jobz`); `mkl::rangev` = `rangev::all` / `rangev::values` / `rangev::indices`; `mkl::side` = `side::left` / `side::right`; `mkl::transpose` = `transpose::nontrans` / `transpose::trans` / `transpose::conjtrans`; `mkl::diag` = `diag::nonunit` / `diag::unit`; `mkl::generate` = `generate::q` / `generate::p`.

**Exception model (applies to every routine; not repeated below).** Computational routines throw `mkl::lapack::exception`; batch routines throw `mkl::lapack::batch_exception`. Recurring contract: "You can obtain the info code of the problem using the `info()` method of the exception object: If `info = -i`, the i-th parameter had an illegal value. If `info` is equal to the value passed as scratchpad size, and `detail()` returns non-zero, then the passed scratchpad has an insufficient size, and the required size should not be less than the value returned by the `detail()` method of the exception object." A `*_scratchpad_size` function throws `mkl::lapack::exception` "when an incorrect argument value is supplied. You can determine the position of the incorrect argument by the `info()` method of the exception object." Batch routines add: "If `info = -n`, the n-th parameter had an illegal value." Where a per-problem factorization/solve can fail: "If info is zero, then the diagonal element of some of Ui is zero, and the solve could not be completed. The indexes of such matrices in the batch can be obtained with the ids() method of the exception object. You can obtain the indexes of the first zero diagonal elements in these Ui matrices using the infos() method of the exception object."

**Batch data layouts.** *Strided*: scalar `n`, `lda`, `stride_a`, `batch_size`. *Group*: `int64_t *m, int64_t *n, ... int64_t group_count, int64_t *group_sizes`, with `a` as `T **`; "Array element with index g specifies the number of problems to solve for each of the groups of parameters g. So the total number of problems to solve, batch_size, is a sum of all parameter group sizes."

**Transpose convention gotcha.** For `ormqr`, `ormrq`, `unmqr`, `unmrq` the source states: "If `trans=mkl::transpose::trans`, the routine multiplies C by Q. If `trans=mkl::transpose::nontrans`, the routine multiplies C by QT." For `ormtr`/`unmtr` it states the opposite: "If trans = transpose::nontrans, the routine multiplies C by Q. If trans = transpose::trans, the routine multiplies C by QT." Use each routine's own wording (repeated in its entry). Signatures below join parameters onto fewer lines than the PDF; token names, order and default arguments are verbatim.

## Routines

### hetrf

USM only on these pages (no buffer signature given). Bunch-Kaufman factorization of a complex Hermitian matrix: if `uplo=uplo::upper`, `A = U*D*UH`; if `uplo=uplo::lower`, `A = L*D*LH`. U/L are products of permutation and unit-diagonal triangular matrices, D is Hermitian block-diagonal with 1-by-1 and 2-by-2 blocks; U and L have 2-by-2 unit diagonal blocks matching D's 2-by-2 blocks. Precisions/devices: `std::complex<float>` CPU, `std::complex<double>` CPU.
```cpp
// hetrf (USM Version), p546
sycl::event hetrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, int64_t *ipiv,
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
In: `uplo` (which triangle of A is stored and how A is factored); `n` order of A (`0 ≤ n`); `a` coefficients of A, upper or lower triangle per `uplo`, size ≥ `lda*n`; `lda`; `scratchpad`; `scratchpad_size` (from `hetrf_scratchpad_size`). Out: `a` "overwritten by details of the block-diagonal matrix D and the multipliers used to obtain the factor U (or L)"; `ipiv` size ≥ `max(1, n)` — `ipiv(i) = k > 0` ⇒ `dii` is a 1-by-1 block and row/column i was interchanged with row/column k. If `uplo = mkl::uplo::upper` and `ipiv(i)=ipiv(i-1)=-m<0`, D has a 2-by-2 block in rows/columns i and i-1 and row/column (i-1) was interchanged with row/column m. If `uplo = mkl::uplo::lower` and `ipiv(i)=ipiv(i+1)=-m<0`, D has a 2-by-2 block in rows/columns i and i+1 and row/column (i+1) was interchanged with row/column m.
Error: `If info = i, dii is 0. The factorization has been completed, but D is exactly singular. Division by 0 will occur if you use D for solving a system of linear equations.`

### hetrf_scratchpad_size

Element count for the scratchpad of the hetrf (buffer or USM version) function.
```cpp
template<typename T> int64_t hetrf_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p549
```
`queue`, `uplo`, `n` (`0 ≤ n`), `lda`. Returns "The number of elements of type T the scratchpad memory ... must be able to hold."

### orgbr

Generates the whole or part of the real orthogonal matrices Q and P^T formed by `gebrd`; use after `sgebrd`/`dgebrd`. Precisions/devices: `float` CPU, `double` CPU.
```cpp
void orgbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a,   // p549
  int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, T *a, int64_t lda,  // USM, p551
  const T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`gen` must be `generate::q` (matrix Q) or `generate::p` (matrix PT). `m` rows of Q/PT (`0 ≤ m`): if `gen= generate::q`, `m ≥ n ≥ min(m, k)`; if `gen= generate::p`, `n ≥ m ≥ min(n, k)`. `n` rows of Q/PT (`0 ≤ n`, see `m`). `k`: for `generate::q` the columns of the original m-by-k matrix reduced by gebrd; for `generate::p` the rows of the original k-by-n matrix. `a` = memory returned by gebrd; `tau` = size `min(m,k)` (`generate::q`) or `min(n,k)` (`generate::p`), scalar factors as returned by gebrd in `tauq` or `taup`. Out: `a` "Overwritten by n leading columns of the m-by-m orthogonal matrix Q or PT (or the leading rows or columns thereof) as specified by gen, m, and n."
Documented usage: whole m-by-m Q → `orgbr(queue, generate::q, m, m, n, a, ...)` (array `a` must have ≥ m columns); n leading columns of Q if `m > n` → `orgbr(queue, generate::q, m, n, n, a, ...)`; whole n-by-n PT → `orgbr(queue, generate::p, n, n, m, a, ...)` (array `a` must have ≥ n rows); m leading rows of PT if `m < n` → `orgbr(queue, generate::p, m, n, m, a, ...)`.

### orgbr_scratchpad_size

```cpp
template<typename T> int64_t orgbr_scratchpad_size(sycl::queue &queue, mkl::generate gen,
  int64_t m, int64_t n, int64_t k, int64_t lda)  // p554
```
Same meanings/constraints as `orgbr`. Returns the element count for the orgbr (buffer or USM version) scratchpad.

### orgqr

Generates the whole or part of the m-by-m real orthogonal matrix Q of the QR factorization formed by `geqrf`. Devices: `float`/`double`, each `CPU and GPU*` (*interface support only).
```cpp
void orgqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda,   // p555
  sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, T *a, int64_t lda, const T *tau,  // USM, p556
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`m` rows of A (`0 ≤ m`); `n` columns of A (`0 ≤ n`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` result of geqrf; `lda ≥ m`; `tau` result of geqrf. Out: `a` "Overwritten by n leading columns of the m-by-m orthogonal matrix Q."
Usage: whole Q → `mkl::orgqr(queue, m, m, p, a, lda, tau, ...)`; leading p columns → `mkl::orgqr(queue, m, p, p, a, lda, tau, ...)`; Qk of leading k columns → `mkl::orgqr(queue, m, m, k, a, lda, tau, ...)`; leading k columns of Qk → `mkl::orgqr(queue, m, k, k, a, lda, tau, ...)`.
NOTE: the USM exception table also lists `If info = i, dii is 0 ... D is exactly singular` (a factorization condition, inconsistent with QR generation) — transcribed as printed.

### orgqr_batch

Batch real orthogonal matrices Qi of the QR factorizations formed by `geqrf_batch`. Devices: `float`/`double`, each `CPU and GPU*` (*interface support only).
```cpp
void orgqr_batch(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda,  // Buffer Strided, p558
  int64_t stride_a, sycl::buffer<T> &tau, int64_t stride_tau, int64_t batch_size,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgqr_batch(sycl::queue &queue, int64_t m, int64_t n, int64_t k, T *a, int64_t lda,  // USM Strided, p562
  int64_t stride_a, const T *tau, int64_t stride_tau, int64_t batch_size, T *scratchpad,
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event orgqr_batch(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *k, T **a, int64_t *lda,  // Group, p560
  const T * const *tau, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
Strided: `m` rows of Ai (`m ≥ 0`), `n` columns (`n ≥ 0`), `k` elementary reflectors defining Qi (`0 ≤ k ≤ n`), `a`/`tau` from `geqrf_batch`, `lda ≥ max(1,m)`, `stride_a ≥ max(1, lda * n)`, `stride_tau ≥ max(1, min(m,n))`, `batch_size ≥ 0`, `scratchpad_size` from `orgqr_batch_scratchpad_size` (Strided Version). Group: `m`/`n`/`k`/`lda` are arrays of `group_count` per-group `mg`/`ng`/`kg`/`ldag` as previously supplied to `geqrf_batch` (Group Version), with `mg ≥ 0`, `ng ≥ 0`, `0 ≤ kg ≤ ng`, `ldag ≥ max(1, mg)`; `group_count` ≥ 0.
Out: strided `a` "is overwritten by a batch of n leading columns of the m-by-m orthogonal matrices Qi"; group `a` "Matrices pointed to by array a are overwritten by ng leading columns of the mg-by-mg orthogonal matrices Qi, where g is an index of group of parameters corresponding to Qi."
Usage: `orgqr_batch(queue, m, m, p, a, ...)`, `orgqr_batch(queue, m, p, p, a, ...)`, `orgqr_batch(queue, m, m, k, a, ...)`, `orgqr_batch(queue, m, k, k, a, ...)`.

### orgqr_batch_scratchpad_size

```cpp
int64_t orgqr_batch_scratchpad_size(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda,  // Group, p564
  int64_t group_count, int64_t *group_sizes)
int64_t orgqr_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t k, int64_t lda,  // Strided, p566
  int64_t stride_a, int64_t stride_tau, int64_t batch_size)
```
Devices for the Group size function: `float`/`double`, `CPU and GPU` (no asterisk). Returns the element count for the matching `orgqr_batch` scratchpad.

### orgqr_scratchpad_size

```cpp
template<typename T> int64_t orgqr_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n,
  int64_t k, int64_t lda)  // p567
```
`m` rows (`0 ≤ m`), `n` columns (`0 ≤ n`), `k` (`0 ≤ k ≤ n`), `lda ≥ m`. Returns the element count for the orgqr (buffer or USM version) scratchpad.

### orgtr

Explicitly generates the n-by-n real orthogonal matrix Q formed by `sytrd` when reducing a real symmetric matrix A to tridiagonal form: `A = Q*T*QT`. Use after `sytrd`. Devices: `float` CPU, `double` CPU.
```cpp
void orgtr(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,   // p568
  sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event orgtr(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, const T *tau,  // USM, p569
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo` must be `uplo::upper` or `uplo::lower`, "Uses the same uplo as supplied to the sytrd function"; `n` order of Q (`0 ≤ n`); `a` = buffer/pointer returned by sytrd, size ≥ `lda*n`; `lda ≥ n`; `tau` = tau returned by sytrd (the USM input list instead says "The pointer to tau returned by the geqrf (USM Version) function." — as printed). Out: `a` "Overwritten by the orthogonal matrix Q."

### orgtr_scratchpad_size

```cpp
template<typename T> orgtr_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p571
```
NOTE: the extracted Syntax shows **no return type** (printed exactly as above). `uplo`, `n` (`0 ≤ n`), `lda ≥ n`. "Return Values: The number of elements of type T the scratchpad memory to be passed to the orgtr (buffer or USM version) function must be able to hold."

### ormqr

Multiplies a real matrix by the orthogonal Q of the QR factorization formed by `geqrf`; forms one of `Q*C`, `QT*C`, `C*Q`, `C*QT` (result overwrites C). Buffer devices `float`/`double` `CPU and GPU*`; USM devices `float`/`double` `CPU, GPU*`; both footnotes say `*Hybrid support; some computations are performed on the CPU.`
```cpp
void ormqr(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // p572
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ormqr(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // USM, p573
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`side` = `mkl::side::left` ⇒ Q or QT applied from the left; `mkl::side::right` ⇒ from the right. `trans`: "If `trans=mkl::transpose::trans`, the routine multiplies C by Q. If `trans=mkl::transpose::nontrans`, the routine multiplies C by QT." `m` rows in A (`0 ≤ m`); `n` columns in A (`0 ≤ n ≤ m`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` = result of geqrf, size ≥ `lda*k`; `tau` = tau returned by geqrf; `c` = matrix C, size ≥ `ldc*n`; `ldc`. Out: `c` "Overwritten by the product Q*C, QT*C, C*Q, or C*QT (as specified by left_right and trans)."

### ormqr_scratchpad_size

```cpp
template<typename T> int64_t ormqr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc)  // p575
```
Inputs as for `ormqr`. Returns the element count for the ormqr (buffer or USM version) scratchpad.

### ormrq

Multiplies a real m-by-n matrix C by Q or QT, where Q is the real orthogonal matrix defined as a product of k elementary reflectors Hi: `Q = H1H2 ... Hk` as returned by the RQ factorization routine `gerqf`; forms `Q*C`, `QT*C`, `C*Q`, or `C*QT` (overwriting C). Devices: `float` CPU, `double` CPU.
```cpp
void ormrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // p577
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ormrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // USM, p578
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`side`, `trans` (same printed convention as `ormqr`); `m` rows in A (`0 ≤ m`); `n` columns in A (`0 ≤ n ≤ m`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` = result of `gerqf`, size ≥ `lda*k`; `tau` = tau returned by `gerqf`; `c` = matrix C, size ≥ `ldc*n`; `ldc`. Out: `c`.
Application Notes: "The complex counterpart of this routine is unmrq."

### ormrq_scratchpad_size

```cpp
template<typename T> int64_t ormrq_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc)  // p580
```
Inputs as for `ormrq`. Returns the element count for the ormrq (buffer or USM version) scratchpad.

### ormtr

Multiplies a real matrix C by Q or QT, Q being the real orthogonal matrix formed by `sytrd` when reducing a real symmetric A to tridiagonal form `A = Q*T*QT`; forms `Q*C`, `QT*C`, `C*Q`, or `C*QT` (overwriting C). Devices: `float` CPU, `double` CPU.
```cpp
void ormtr(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans, int64_t m, int64_t n,  // p581
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ormtr(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans, int64_t m, int64_t n,  // USM, p583
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
"In the descriptions below, r denotes the order of Q: `r = m` if `left_right = side::left` (USM: `side = side::left`); `r = n` if `left_right = side::right` (USM: `side = side::right`)."
`side` must be `side::left` or `side::right`; `uplo` must be `uplo::upper` or `uplo::lower`, using the same uplo supplied to sytrd; `trans` must be `transpose::nontrans` or `transpose::trans`, and here "If trans = transpose::nontrans, the routine multiplies C by Q. If trans = transpose::trans, the routine multiplies C by QT." `m` rows in C (`m ≥ 0`); `n` columns in C (`n ≥ 0`); `a` = array returned by sytrd; `lda ≥ max(1, r)`; `tau` = tau returned by sytrd, dimension ≥ `max(1, r-1)`; `c` = matrix C, size ≥ `ldc*n`; `ldc ≥ max(1, m)`. Out: `c` "Overwritten by the product Q*C, QT*C, C*Q, or C*QT (as specified by side and trans)."

### ormtr_scratchpad_size

```cpp
template<typename T> int64_t ormtr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::uplo uplo,
  mkl::transpose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc)  // p585
```
Same `r` definition and input meanings as `ormtr`. Returns the element count for the ormtr (buffer or USM version) scratchpad.

### potrf

Cholesky factorization of a symmetric (Hermitian) positive-definite matrix: `A = UT*U` (real) / `A = UH*U` (complex) if `uplo=mkl::uplo::upper`; `A = L*LT` (real) / `A = L*LH` (complex) if `uplo=mkl::uplo::lower`. Devices (both forms): `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
void potrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,   // p587
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event potrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *scratchpad,  // USM, p588
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo`: if `mkl::uplo::upper`, `a` stores the upper triangular part of A and the strictly lower triangular part is not referenced; if `mkl::uplo::lower`, the lower triangular part is stored and the strictly upper part is not referenced. `n` (`0 ≤ n`); `a` input matrix, size ≥ `lda*n`; `lda`. Out: `a` "is overwritten by the Cholesky factor U or L, as specified by uplo."
Error: "If `info = i`, and `detail()` returns 0, the leading minor of order i (and therefore the matrix A itself) is not positive-definite, and the factorization could not be completed. This may indicate an error in forming the matrix A."

### potrf_batch

Batch Cholesky factorizations of Ai, iϵ{1...batch_size}: `Ai = UiT * Ui` (real) / `Ai = UiH * Ui` (complex) if `uplo = mkl::uplo::upper`; `Ai = LiT * Li` (real) / `Ai = LiH * Li` (complex) if `uplo = mkl::uplo::lower` (lower case printed with the same factor order as the upper case — see Explicit gaps). Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
void potrf_batch(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,  // Buffer Strided, p590
  int64_t stride_a, int64_t batch_size, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event potrf_batch(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, int64_t stride_a,  // USM Strided, p594
  int64_t batch_size, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event potrf_batch(sycl::queue &queue, mkl::uplo *uplo, int64_t *n, T **a, int64_t *lda,  // Group, p592
  int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
Strided: `uplo` selects which triangular parts of Ai are stored; `n` order of Ai (`n ≥ 0`); `a` batch of input matrices Ai, "each of Ai being of size lda*n"; `lda ≥ max(1, n)`; `stride_a ≥ max(1, lda * n)`; `batch_size ≥ 0`. Group: `uplo` array of `group_count` `uplog`; `n` array of `group_count` `ng` (`ng ≥ 0`); `a` array of `batch_size` pointers to Ai, each of size `ldag*ng`; `lda` array of `group_count` `ldag` (`ldag ≥ max(1, ng)`).
Out: strided "The batch array a is overwritten by the Cholesky factor Ui or Li, as specified by uplo."; group "The matrices pointed to by array a are overwritten by the Cholesky factors Ui or Li, as specified by uplog from the corresponding group of parameters."
Errors: standard batch contract plus the `info is zero` / `ids()` / `infos()` case (see Overview).

### potrf_batch_scratchpad_size

```cpp
int64_t potrf_batch_scratchpad_size(sycl::queue &queue, mkl::uplo *uplo, int64_t *n, int64_t *lda,  // Group, p596
  int64_t group_count, int64_t *group_sizes)
int64_t potrf_batch_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda,  // Strided, p598
  int64_t stride_a, int64_t batch_size)
```
Devices for the Group size function: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`. Returns the element count for the matching `potrf_batch` scratchpad.

### potrf_scratchpad_size

```cpp
template<typename T> int64_t potrf_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p599
```
`uplo` (same meaning as `potrf`), `n` (`0 ≤ n`), `lda` ("The leading dimension of A."). Returns the element count for the potrf (buffer or USM version) scratchpad.

### potri

Inverse `inv(A)` of a symmetric positive-definite (Hermitian positive-definite for complex) matrix A. **Before calling, call `potrf` to factorize A.** Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU` (USM list prints "CPU, GPU").
```cpp
void potri(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,   // p600
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event potri(sycl::queue &queue, mkl::uplo uplo, int64_t n, T* a, int64_t lda, T *scratchpad,  // USM, p601
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo`: "Indicates how the input matrix A has been factored: If `uplo = mkl::uplo::upper`, the upper triangle of A is stored. If `uplo = mkl::uplo::lower`, the lower triangle of A is stored." `n` (`0 ≤ n`); `a` = "the factorization of the matrix A, as returned by potrf (USM Version)", size ≥ `lda*n`; `lda`. Out: `a` "Overwritten by the upper or lower triangle of the inverse of A. Specified by uplo."
Error: "If `info = i`, the i-th diagonal element of the Cholesky factor (and therefore the factor itself) is zero, and the inversion could not be completed."

### potri_scratchpad_size

```cpp
template<typename T> int64_t potri_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p603
```
`uplo`, `n` (`0 ≤ n`), `lda`. Returns the element count for the potri (buffer or USM version) scratchpad.

### potrs

Solves `A*X = B` for X, with a symmetric positive-definite (Hermitian positive-definite for complex) A given its Cholesky factorization, and multiple right-hand sides in the columns of B. **Before calling, you must call `potrf` to compute the Cholesky factorization of A.** Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
void potrs(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda,   // p604
  sycl::buffer<T> &b, int64_t ldb, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event potrs(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t nrhs, const T *a, int64_t lda,  // USM, p606
  T *b, int64_t ldb, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo`: "If `uplo=mkl::uplo::upper`, the upper triangle U of A is stored, where `A = UT*U` for real data, `A = UH*U` for complex data. If `uplo=mkl::uplo::lower`, the upper triangle L of A is stored, where `A = L*LT` for real data, `A = L*LH` for complex data." (the word "upper" for the lower case is as printed). `n` order of A (`0 ≤ n`); `nrhs` (`0 ≤ nrhs`); `a` = factorization as returned by potrf, size ≥ `lda*n`; `b` = matrix B whose columns are the right-hand sides, size ≥ `ldb*nrhs`; `ldb`. Out: `b` "is overwritten by the solution matrix X."
Error: "If `info = i`, the i-th diagonal element of the Cholesky factor is zero, and the solve could not be completed."

### potrs_batch

Solves `Ai*Xi = Bi` for `i ϵ {1...batch_size}` given the Cholesky factorization of Ai: `Ai = UiT*Ui` (real) / `Ai = UiH*Ui` (complex) if `uplo=mkl::uplo::upper`; `Ai = Li*LiT` (real) / `Ai = Li*LiH` (complex) if `uplo=mkl::uplo::lower`. **Before calling, matrices Ai should be factorized by a call to `potrf_batch`** (Buffer Strided / USM Strided / Group Version respectively). Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
void potrs_batch(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t nrhs, sycl::buffer<T> &a, int64_t lda,  // Buffer Strided, p608
  int64_t stride_a, sycl::buffer<T> &b, int64_t ldb, int64_t stride_b, int64_t batch_size,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event potrs_batch(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t nrhs, const T *a, int64_t lda,  // USM Strided, p612
  int64_t stride_a, T *b, int64_t ldb, int64_t stride_b, int64_t batch_size, T *scratchpad,
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event potrs_batch(sycl::queue &queue, mkl::uplo *uplo, int64_t *n, int64_t *nrhs, const T * const *a,  // Group, p610
  int64_t *lda, T **b, int64_t *ldb, int64_t group_count, int64_t *group_sizes, T *scratchpad,
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
Strided: `uplo` selects the stored triangle of the factored Ai (`Ai = UiT*Ui`/`UiH*Ui` for upper, `Ai = Li*LiT`/`Li*LiH` for lower — the lower case here uses the mathematically expected order); `n` order of Ai (`n ≥ 0`); `nrhs` (`nrhs ≥ 0`); `a` = batch of factorizations from `potrf_batch`; `lda ≥ max(1, n)`; `stride_a ≥ max(1, lda * n)`; `b` = batch of Bi; `ldb ≥ max(1, n)`; `stride_b ≥ max(1, ldb * nrhs)`; `batch_size ≥ 0`; `scratchpad_size` from the stride version of `potrs_batch_scratchpad_size`. Group: `uplo` array of `group_count` `uplog`; `n` array of `group_count` `ng` (`ng ≥ 0`); `nrhs` array of `group_count` `nrhsg` (`nrhsg ≥ 0`); `a` = `batch_size` pointers to factored Ai; `lda` array of `ldag` (`ldag ≥ max(1, ng)`); `b` = `batch_size` pointers to Bi, each of size `ldbg*nrhsg`; `ldb` array of `ldbg` (`ldbg ≥ max(1, ng)`).
Out: strided "The batch array b is overwritten by the solution matrix Xi."; group "The matrices pointed to by array b are overwritten by the solution matrices Xi."
Errors: standard batch contract plus the `info is zero` / `ids()` / `infos()` case.

### potrs_batch_scratchpad_size

```cpp
int64_t potrs_batch_scratchpad_size(sycl::queue &queue, mkl::uplo *uplo, int64_t *n, int64_t *nrhs,  // Group, p615
  int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes)
int64_t potrs_batch_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t nrhs,  // Strided, p617
  int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size)
```
Devices for the Group size function: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`. Returns the element count for the matching `potrs_batch` scratchpad.

### potrs_scratchpad_size

```cpp
template<typename T> int64_t potrs_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n,
  int64_t nrhs, int64_t lda, int64_t ldb)  // p618
```
`uplo`, `n` (`0 ≤ n`), `nrhs` (`0 ≤ nrhs`), `lda` ("The leading dimension of A."), `ldb` ("The leading dimension of B."). Returns the element count for the potrs (buffer or USM version) scratchpad.

### syevd

All eigenvalues and, optionally, all eigenvectors of a real symmetric matrix A by divide and conquer: spectral factorization `A = Z*λ*ZT`, with Λ diagonal holding eigenvalues λi and Z orthogonal with eigenvector columns zi, so `A*zi = λi*zi` for `i = 1, 2, ..., n`. With eigenvectors requested it uses divide and conquer; with only eigenvalues it uses "the Pal-Walker-Kahan variant of the QL or QR algorithm." Buffer devices `float`/`double` `CPU and GPU*` (*interface only); USM devices `float` `CPU and GPU*` (*interface only), `double` `CPU and GPU^` (^hybrid).
```cpp
void syevd(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,  // p619
  sycl::buffer<T> &w, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event syevd(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *w,  // USM, p621
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`jobz` must be `job::novec` (only eigenvalues) or `job::vec` (eigenvalues and eigenvectors); `uplo` must be `uplo::upper` (a stores the upper triangular part of A) or `uplo::lower`; `n` order of A (`0 ≤ n`); `a` the symmetric triangle, size ≥ `lda*n`; `lda` ≥ `max(1,n)`; `scratchpad_size` from `syevd_scratchpad_size`. Out: `a` — "If jobz = job::vec, then on exit this buffer is overwritten by the orthogonal matrix Z which contains the eigenvectors of A."; `w` — array size ≥ n; "If info = 0, contains the eigenvalues of the matrix A in ascending order."
Errors: illegal parameter; if `jobz = job::novec`, `info = i` ⇒ "the algorithm failed to converge; i indicates the number of off-diagonal elements of an intermediate tridiagonal form which did not converge to zero"; if `jobz = job::vec`, `info = i` ⇒ "the algorithm failed to compute an eigenvalue while working on the submatrix lying in rows and columns info/(n+1) through mod(info,n+1)".

### syevd_scratchpad_size

```cpp
template<typename T> int64_t syevd_scratchpad_size(sycl::queue &queue, mkl::job jobz, mkl::uplo uplo,
  int64_t n, int64_t lda)  // p623
```
`jobz`, `uplo`, `n` (`0 ≤ n`), `lda` ≥ `max(1,n)`. Returns the element count for the syevd (buffer or USM version) scratchpad.

### syevx

Selected eigenvalues and, optionally, eigenvectors of a real symmetric A: selected eigenpairs (λ, z) with `A*z = z*λ`; selection by value range or index range. Selected eigenvalues use the bisection algorithm; when eigenvectors are requested, "a combination of modified twisted factorization algorithm based on Inderjit Dhillon and Beresford Parlett's work and the inverse iteration algorithm followed by Gram-Schmidt orthogonalization." Buffer devices `float`/`double` `CPU and GPU*` (*interface only); USM devices `float` `CPU and GPU*` (*interface only), `double` `CPU and GPU^` (^hybrid).
```cpp
void syevx(sycl::queue &queue, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n,  // p624
  sycl::buffer<T> &a, int64_t lda, T vl, T vu, int64_t il, int64_t iu, T abstol,
  sycl::buffer<int64_t> &m, sycl::buffer<T> &w, sycl::buffer<T> &z, int64_t ldz,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event syevx(sycl::queue &queue, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n,  // USM, p627
  T *a, int64_t lda, T vl, T vu, int64_t il, int64_t iu, T abstol, int64_t *m, T *w, T *z, int64_t ldz,
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`jobz` = `job::novec` or `job::vec`. `range` must be `rangev::all` (all eigenvalues/eigenvectors), `rangev::values` (eigenvalues in the interval `(vl, vu]` plus corresponding eigenvectors), or `rangev::indices` (the il-th through iu-th). `uplo` = `uplo::upper`/`uplo::lower` (which triangle of A is stored); `n` (`0 ≤ n`); `a` size ≥ `lda*n`; `lda` ≥ `max(1,n)`. `vl`/`vu`: lower/upper bounds when `range = rangev::values`, "vl must be less than vu", not referenced for `rangev::all`/`rangev::indices`. `il`/`iu`: **one-based** indices when `range = rangev::indices`, "Must be `1 ≤ il ≤ iu ≤ n` if `n > 0` and `il = 1` and `iu = 0` if `n = 0`", not referenced otherwise. `abstol`: "The absolute error tolerance for the eigenvalues. An approximate eigenvalue is accepted as converged when it is determined to lie in an interval [a,b] of width less than or equal to `abstol + eps * max( |a|,|b| )`, where eps is the machine precision. If abstol is less than or equal to zero, then `eps*|T|` will be used in its place, where `|T|` is the 1-norm of the tridiagonal matrix obtained by reducing A to tridiagonal form." `ldz` ≥ `max(1,n)`.
Out: `a` — "On exit the upper triangle (if `uplo = uplo::upper`) or the lower triangle (if `uplo = uplo::lower`) of A, including the diagonal, is destroyed."; `m` — total number of eigenvalues found, `0 ≤ m ≤ n`; `w` — array size ≥ n, first m elements are the selected eigenvalues in ascending order; `z` — if `jobz = job::vec`, the first m columns are the orthonormal eigenvectors, column i corresponding to `w(i)`; if `jobz = job::novec`, Z is not referenced. NOTE: "the user must ensure that at least `max(1, m)` columns are supplied in the array Z; if `range = rangev::values`, the exact value of m is not known in advance and an upper bound must be used."
Errors: illegal parameter; `jobz = job::novec` ⇒ `info = i` means the algorithm failed to converge; `jobz = job::vec` ⇒ `info = i` means the algorithm failed to compute an eigenvalue.

### syevx_scratchpad_size

```cpp
template<typename T> int64_t syevx_scratchpad_size(sycl::queue &queue, mkl::job jobz, mkl::rangev range,
  mkl::uplo uplo, int64_t n, int64_t lda, T vl, T vu, int64_t il, int64_t iu, T abstol, int64_t ldz)  // p630
```
Same input meanings as `syevx`. Returns the element count for the syevx (buffer or USM version) scratchpad.

### sygvd

All eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem of the form `A*x = λ*B*x`, `A*B*x = λ*x`, or `B*A*x = λ*x`, where A and B are symmetric and B is also positive definite; divide and conquer algorithm. Devices: `float` `CPU, GPU*` (*interface only); `double` `CPU, GPU^` (^hybrid).
```cpp
void sygvd(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a,  // p632
  int64_t lda, sycl::buffer<T> &b, int64_t ldb, sycl::buffer<T> &w, sycl::buffer<T> &scratchpad,
  int64_t scratchpad_size)
sycl::event sygvd(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::uplo uplo, int64_t n, T *a,  // USM, p634
  int64_t lda, T *b, int64_t ldb, T *w, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
(The USM device table is headed `hegvd (USM version) supports the following precision and devices.` — a wrong routine name in the source.)
`itype` must be 1, 2, or 3: "if itype= 1, the problem type is A*x = lambda*B*x; if itype= 2, the problem type is A*B*x = lambda*x; if itype= 3, the problem type is B*A*x = lambda*x." `jobz` = `job::novec`/`job::vec`; `uplo` = `uplo::upper`/`uplo::lower` (if upper, a and b store the upper triangular parts of A and B; if lower, the lower parts); `n` order of A and B (`0 ≤ n`); `a` triangle of A, size ≥ `lda*n`; `lda` ≥ `max(1,n)`; `b` triangle of B, size ≥ `ldb*n`; `ldb` ≥ `max(1,n)`.
Out: `a` — "On exit, if `jobz = job::vec`, then if `info = 0`, a contains the matrix Z of eigenvectors. The eigenvectors are normalized as follows: `if itype= 1 or 2, ZT*B*Z = I; if itype= 3, ZT*inv(B)*Z = I;` If `jobz = job::novec`, then on exit the upper triangle (if `uplo = uplo::upper`) or the lower triangle (if `uplo = uplo::lower`) of A, including the diagonal, is destroyed."; `b` — "On exit, if `info ≤ n`, the part of b containing the matrix is overwritten by the triangular factor U or L from the Cholesky factorization `B = UT*U` or `B = L*LT`."; `w` — size ≥ n, if `info = 0` holds the eigenvalues of A in ascending order.
Errors: illegal parameter. `For info ≤ n`: `jobz = job::novec` and `info = i` ⇒ "the algorithm failed to converge; i indicates the number of off-diagonal elements of an intermediate tridiagonal form which did not converge to zero"; `jobz = job:vec` and `info = i` ⇒ "the algorithm failed to compute an eigenvalue while working on the submatrix lying in rows and columns info/(n+1) through mod(info,n+1)" (the `job:vec` spelling is as printed). `For info > n`: "If info = n + i, for 1 ≤ i ≤ n, then the leading minor of order i of B is not positive-definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed."

### sygvd_scratchpad_size

```cpp
template<typename T> int64_t sygvd_scratchpad_size(sycl::queue &queue, int64_t itype, mkl::job jobz,
  mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb)  // p637
```
`itype` (1/2/3), `jobz`, `uplo`, `n` (`0 ≤ n`), `lda` ≥ `max(1,n)`, `ldb` ≥ `max(1,n)`. Returns the element count for the sygvd (buffer or USM version) scratchpad.

### sygvx

Selected eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem `A*x = λ*B*x`, `A*B*x = λ*x`, or `B*A*x = λ*x`, A and B symmetric with B positive definite; selection by value range or index range, using the bisection algorithm (and, when eigenvectors are requested, modified twisted factorization plus inverse iteration and Gram-Schmidt orthogonalization). Devices: `float` `CPU and GPU*` (*interface only); `double` `CPU and GPU^` (^hybrid).
```cpp
void sygvx(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::rangev range, mkl::uplo uplo, int64_t n,  // p638
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &b, int64_t ldb, T vl, T vu, int64_t il, int64_t iu,
  T abstol, sycl::buffer<int64_t> &m, sycl::buffer<T> &w, sycl::buffer<T> &z, int64_t ldz,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event sygvx(sycl::queue &queue, int64_t itype, mkl::job jobz, mkl::rangev range, mkl::uplo uplo,  // USM, p641
  int64_t n, T *a, int64_t lda, T *b, int64_t ldb, T vl, T vu, int64_t il, int64_t iu, T abstol,
  int64_t *m, T *w, T *z, int64_t ldz, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`itype` (1/2/3 as in `sygvd`); `jobz`; `range` (`rangev::all`/`rangev::values`/`rangev::indices`, interval `(vl, vu]`); `uplo`; `n` order of A and B (`0 ≤ n`); `a` triangle of A (size ≥ `lda*n`), `lda` ≥ `max(1,n)`; `b` triangle of B (size ≥ `ldb*n`), `ldb` ≥ `max(1,n)`; `vl`/`vu`/`il`/`iu` under the same rules as `syevx` (one-based indices); `abstol` with the same tolerance definition as `syevx`; `ldz` ≥ `max(1,n)`.
NOTE: in the USM input list only, `uplo` is described as "If `uplo = uplo::upper`, a stores the upper triangular part of A. If `uplo = uplo::lower`, a stores the lower triangular part of A." (the buffer version says a and b).
Out: `a` — stored triangle of A including the diagonal is destroyed; `b` — "On normal exit, the part of b containing the matrix is overwritten by the triangular factor U or L from the Cholesky factorization `B = UT*U` or `B = L*LT`."; `m` — total number of eigenvalues found, `0 ≤ m ≤ n`; `w` — first m elements are the selected eigenvalues in ascending order; `z` — if `jobz = job::vec`, the first m columns are the orthonormal eigenvectors, and the user must supply at least `max(1, m)` columns; if `jobz = job::novec`, Z is not referenced.
Errors: illegal parameter; `jobz = job::novec` ⇒ `info = i` means the algorithm failed to converge; `jobz = job::vec` ⇒ `info = i` means the algorithm failed to compute an eigenvalue.

### sygvx_scratchpad_size

```cpp
template<typename T> int64_t sygvx_scratchpad_size(sycl::queue &queue, int64_t itype, mkl::job jobz,
  mkl::rangev range, mkl::uplo uplo, int64_t n, int64_t lda, int64_t ldb, T vl, T vu, int64_t il,
  int64_t iu, T abstol, int64_t ldz)  // p645
```
Same input meanings as `sygvx`. Returns the element count for the sygvx (buffer or USM version) scratchpad.

### sytrd

Reduces a real symmetric matrix A to symmetric tridiagonal form T by an orthogonal similarity transformation `A = Q*T*QT`; Q is not formed explicitly but represented as a product of `n-1` elementary reflectors. Devices: `float`/`double`, each `CPU and GPU*` (*interface support only).
```cpp
void sytrd(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,   // p646
  sycl::buffer<T> &d, sycl::buffer<T> &e, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad,
  int64_t scratchpad_size)
sycl::event sytrd(queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, T *d, T *e, T *tau,  // USM, p648
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
NOTE: the USM first parameter is printed as `queue &queue` (no `sycl::`).
`uplo` = `uplo::upper`/`uplo::lower` (which triangle of A is stored); `n` order of A (`0 ≤ n`); `a` matrix A, size `(lda,*)`, upper or lower triangle per `uplo`; `lda` ≥ `max(1,n)`.
Out: `a` — if `uplo = uplo::upper`, "the diagonal and first superdiagonal of A are overwritten by the corresponding elements of the tridiagonal matrix T, and the elements above the first superdiagonal, with the buffer tau, represent the orthogonal matrix Q as a product of elementary reflectors"; if `uplo = uplo::lower`, the analogous statement holds for the diagonal/first subdiagonal and the elements below the first subdiagonal. `d` — diagonal of T, dimension ≥ `max(1, n)`; `e` — off-diagonal of T, dimension ≥ `max(1, n-1)`; `tau` — size ≥ `max(1, n)`, "Stores (n-1) scalars that define elementary reflectors in decomposition of the unitary matrix Q in a product of n-1 elementary reflectors. `tau (n)` is used as workspace."

### sytrd_scratchpad_size

```cpp
template<typename T> int64_t sytrd_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p650
```
`uplo`, `n` (`0 ≤ n`), `lda` ≥ `max(1,n)`. Returns the element count for the sytrd (buffer or USM version) scratchpad.

### sytrf

Bunch-Kaufman factorization of a real/complex symmetric matrix: `A = U*D*UT` if `uplo=uplo::upper`; `A = L*D*LT` if `uplo=uplo::lower`. U/L are products of permutation and unit-diagonal triangular matrices and D is a symmetric block-diagonal matrix with 1-by-1 and 2-by-2 blocks; U and L have 2-by-2 unit diagonal blocks matching D. Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU` only.
```cpp
void sytrf(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,  // p651
  sycl::buffer<T> &ipiv, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event sytrf(queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, int64_t *ipiv,  // USM, p653
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
NOTE: the buffer signature declares `sycl::buffer<T> &ipiv` (element type T), while `ipiv` is described as holding integer interchanges and the USM form uses `int64_t *ipiv`; and the USM first parameter is printed as `queue &queue`. Both as printed.
`uplo` (which triangle is stored and how A is factored); `n` order of A (`0 ≤ n`); `a` coefficients of A, size ≥ `lda*n`; `lda`.
Out: `a` "The upper or lower triangular part of a is overwritten by details of the block-diagonal matrix D and the multipliers used to obtain the factor U (or L)."; `ipiv` size ≥ `max(1, n)`: `ipiv(i)=k>0` ⇒ `dii` is 1-by-1 and row/column i was interchanged with row/column k; if `uplo=mkl::uplo::upper` and `ipiv(i)=ipiv(i-1)=-m<0` ⇒ 2-by-2 block in rows/columns i and i-1, with row/column (i-1) interchanged with row/column m; if `uplo=mkl::uplo::lower` and `ipiv(i)=ipiv(i+1)=-m<0` ⇒ 2-by-2 block in rows/columns i and i+1, with row/column (i+1) interchanged with row/column m.
Error: `If info = i, dii is 0. The factorization has been completed, but D is exactly singular. Division by 0 will occur if you use D for solving a system of linear equations.`

### sytrf_scratchpad_size

```cpp
template<typename T> int64_t sytrf_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p656
```
`uplo` (as for `sytrf`), `n` (`0 ≤ n`), `lda`. Returns the element count for the sytrf (buffer or USM version) scratchpad.

### trtri

Computes `inv(A)` of an upper or lower triangular matrix A. USM only on these pages (no buffer signature given). Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
// trtri (USM Version), p656
sycl::event trtri(sycl::queue &queue, mkl::uplo uplo, mkl::diag diag, int64_t n, T *a, int64_t lda,
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo`: if `uplo::upper`, `a` stores the upper triangular part of A and the strictly lower part is not referenced; if `uplo::lower`, `a` stores the lower triangular part and the strictly upper part is not referenced. `diag`: if `diag::nonunit`, A is not unit triangular; if `diag::unit`, A is unit triangular and "the diagonal elements are assumed to be 1 and are not referenced." `n` order of A (`n ≥ 0`); `a` size ≥ `lda * max(1, n)`; `lda ≥ max(1, n)`.
Out: `a` — the stored triangle is overwritten by its inverse.
Error: "If `info = i`, and `detail()` returns 0, A(i,i) is exactly zero. The triangular matrix is singular and its inverse can not be computed."

### trtri_scratchpad_size

```cpp
template<typename T> int64_t trtri_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, mkl::diag diag,
  int64_t n, int64_t lda)  // p658
```
`uplo`, `diag` (same meanings as `trtri`), `n` (`n ≥ 0`), `lda ≥ max(1, n)`. Returns the element count for the trtri (buffer or USM version) scratchpad.

### trtrs

Solves for X, with multiple right-hand sides stored in B: `A*X = B` if `transa =transpose::nontrans`; `AT*X = B` if `transa =transpose::trans`; `AH*X = B` if `transa =transpose::conjtrans` (complex matrices only). Devices: `float`, `double`, `std::complex<float>`, `std::complex<double>`, each `CPU and GPU`.
```cpp
void trtrs(sycl::queue &queue, mkl::uplo uplo, mkl::transpose trans, mkl::diag diag, int64_t n,  // p659
  int64_t nrhs, sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &b, int64_t ldb,
  sycl::buffer<T> &scratchpad, int64_t   scratchpad_size)
sycl::event trtrs(sycl::queue &queue, mkl::uplo uplo, mkl::transpose trans, mkl::diag diag, int64_t n,  // USM, p661
  int64_t nrhs, const T *a, int64_t lda, T *b, int64_t ldb, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`uplo` (A upper or lower); `trans` — `transpose::nontrans` solves `A*X = B`, `transpose::trans` solves `AT*X = B`, `transpose::conjtrans` solves `AH*X = B`; `diag` (`diag::nonunit` or `diag::unit`, whose diagonal elements "are assumed to be 1 and not referenced in the array a"); `n` = "The order of A; the number of rows in B; n ≥ 0."; `nrhs` = "The number of right-hand sides; nrhs ≥ 0."; `a` size ≥ `lda*n`; `lda ≥ max(1, n)`; `b` size ≥ `ldb*nrhs`; `ldb ≥ max(1, n)`.
Out: `b` "Overwritten by the solution matrix X."
The buffer "Return Values" section says: `info` — "Buffer containing error information. If info = 0, the execution is successful. If info = -i, the i-th parameter had an illegal value." (no `info` parameter exists in the signature — as printed).

### trtrs_scratchpad_size

```cpp
template<typename T> int64_t trtrs_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, mkl::transpose trans,
  mkl::diag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb)  // p664
```
Same input meanings as `trtrs`. Returns the element count for the trtrs (buffer or USM version) scratchpad.

### ungbr

Generates the whole or part of the complex unitary matrices Q and P^H formed by `gebrd`; use after `cgebrd`/`zgebrd`. Devices: `std::complex<float>` CPU, `std::complex<double>` CPU.
```cpp
void ungbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a,  // p664
  int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ungbr(sycl::queue &queue, mkl::generate gen, int64_t m, int64_t n, int64_t k, T *a, int64_t lda,  // USM, p667
  const T *tau, T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`gen` must be `generate::q` or `generate::p`; `m` rows of Q/PT to be returned (`0 ≤ m`), with `m ≥ n ≥ min(m, k)` for `generate::q` and `n ≥ m ≥ min(n, k)` for `generate::p`; `n` (`0 ≤ n`); `k` = columns of the original m-by-k matrix reduced by gebrd (`generate::q`) or rows of the original k-by-n matrix (`generate::p`); `a` = memory returned by gebrd; `tau` — "For `gen= generate::q`, the array tauq is returned by the gebrd function. For `gen= generate::p`, the array taup is returned by the gebrd function. The dimension of tau must be at least `min(m,k)` for `gen = generate::q`, or `min(n,k)` for `gen = generate::p`."
Out: `a` — "Overwritten by n leading columns of the m-by-m unitary matrix Q or PT, (or the leading rows or columns thereof) as specified by gen, m, and n." (the USM output text instead says "orthogonal matrix Q or PT").
Usage: whole m-by-m Q → `ungbr(queue, generate::q, m, m, n, a, ...)`; n leading columns of Q if `m > n` → `ungbr(queue, generate::q, m, n, n, a, ...)`; whole n-by-n P^T → `ungbr(queue, generate::p, n, n, m, a, ...)`; m leading rows of P^T if `m < n` → `ungbr(queue, generate::p, m, n, m, a, ...)`.

### ungbr_scratchpad_size

```cpp
template<typename T> int64_t ungbr_scratchpad_size(sycl::queue &queue, mkl::generate gen,
  int64_t m, int64_t n, int64_t k, int64_t lda)  // p669
```
Same meanings/constraints as `ungbr`. Returns the element count for the ungbr (buffer or USM version) scratchpad.

### ungqr

Generates the whole or part of the m-by-m complex unitary matrix Q of the QR factorization formed by `geqrf`. Devices: `std::complex<float>`/`std::complex<double>`, each `CPU and GPU*` (*interface support only).
```cpp
void ungqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda,   // p671
  sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ungqr(sycl::queue &queue, int64_t m, int64_t n, int64_t k, T *a, int64_t lda, const T *tau,  // USM, p672
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`m` rows of A (`0 ≤ m`); `n` columns of A (`0 ≤ n`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` result of geqrf; `lda ≥ m`; `tau` result of geqrf. Out: `a` "Overwritten by n leading columns of the m-by-m unitary matrix Q."
Usage: whole Q → `mkl::ungqr(queue, m, m, p, a, lda, tau, ...)`; leading p columns → `mkl::ungqr(queue, m, p, p, a, lda, tau, ...)`; Qk of the leading k columns → `mkl::ungqr(queue, m, m, k, a, lda, tau, ...)`; leading k columns of Qk → `mkl::ungqr(queue, m, k, k, a, lda, tau, ...)`.

### ungqr_batch

Batch complex unitary matrices Qi of the QR factorizations formed by `geqrf_batch`. Devices: `std::complex<float>`/`std::complex<double>`, each `CPU and GPU*` (*interface support only).
```cpp
void ungqr_batch(sycl::queue &queue, int64_t m, int64_t n, int64_t k, sycl::buffer<T> &a, int64_t lda,  // Buffer Strided, p674
  int64_t stride_a, sycl::buffer<T> &tau, int64_t stride_tau, int64_t batch_size,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ungqr_batch(sycl::queue &queue, int64_t m, int64_t n, int64_t k, T *a, int64_t lda,  // USM Strided, p678
  int64_t stride_a, const T *tau, int64_t  stride_tau, int64_t batch_size, T *scratchpad,
  int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
sycl::event ungqr_batch(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *k, T **a, int64_t *lda,  // Group, p676
  const T * const *tau, int64_t group_count, int64_t *group_sizes, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
Strided: `m` rows of Ai (`m ≥ 0`), `n` columns (`n ≥ 0`), `k` elementary reflectors defining Qi (`0 ≤ k ≤ n`), `a`/`tau` from `geqrf_batch`, `lda ≥ max(1,m)`, `stride_a ≥ max(1, lda * n)`, `stride_tau ≥ max(1, min(m,n))`, `batch_size ≥ 0`. Group: arrays `mg`, `ng`, `kg` (`0 ≤ kg ≤ ng`), `ldag` (`ldag ≥ max(1, mg)`) as previously supplied to `geqrf_batch` (Group Version), plus `group_count` and `group_sizes`.
Out: strided `a` "is overwritten by a batch of n leading columns of the m-by-m unitary matrices Qi"; group `a` "Matrices pointed to by array a are overwritten by ng leading columns of the mg-by-mg unitary matrices Qi, where g is an index of group of parameters corresponding to Qi."
Usage: `ungqr_batch(queue, m, m, p, a, ...)`, `ungqr_batch(queue, m, p, p, a, ...)`, `ungqr_batch(queue, m, m, k, a, ...)`, `ungqr_batch(queue, m, k, k, a, ...)`.

### ungqr_batch_scratchpad_size

```cpp
int64_t ungqr_batch_scratchpad_size(sycl::queue &queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda,  // Group, p680
  int64_t group_count, int64_t *group_sizes)
int64_t ungqr_batch_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n, int64_t k, int64_t lda,  // Strided, p681
  int64_t stride_a, int64_t stride_tau, int64_t batch_size)
```
Devices: `std::complex<float>`/`std::complex<double>`, `CPU and GPU`. Returns the element count for the matching `ungqr_batch` scratchpad.

### ungqr_scratchpad_size

```cpp
template<typename T> int64_t ungqr_scratchpad_size(sycl::queue &queue, int64_t m, int64_t n,
  int64_t k, int64_t lda)  // p682
```
`m` (`0 ≤ m`), `n` (`0 ≤ n`), `k` (`0 ≤ k ≤ n`), `lda ≥ m`. Returns the element count for the ungqr (buffer or USM version) scratchpad.

### ungtr

Explicitly generates the n-by-n complex unitary matrix Q formed by `hetrd` when reducing a complex Hermitian matrix A to tridiagonal form; use after `hetrd`. Devices: `std::complex<float>` CPU, `std::complex<double>` CPU. (The USM precision table prints `float` CPU / `double` CPU — as printed; the routine is complex-only per its Description and the buffer table.)
```cpp
void ungtr(sycl::queue &queue, mkl::uplo uplo, int64_t n, sycl::buffer<T> &a, int64_t lda,   // p683
  sycl::buffer<T> &tau, sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event ungtr(sycl::queue &queue, mkl::uplo uplo, int64_t n, T *a, int64_t lda, const T *tau,  // USM, p685
  T *scratchpad, int64_t scratchpad_size, const std::vector<sycl::event> &events = {})
```
`uplo` must be `uplo::upper` or `uplo::lower`, "Uses the same uplo as supplied to the hetrd function"; `n` order of Q (`0 ≤ n`); `a` = array returned by hetrd, size ≥ `lda*n`; `lda ≥ n`; `tau` = tau returned by hetrd, "The dimension of tau must be at least `max(1,n-1)`." Out: `a` "Overwritten by the unitary matrix Q."

### ungtr_scratchpad_size

```cpp
template<typename T> int64_t ungtr_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)  // p686
```
`uplo`, `n` (`0 ≤ n`), `lda ≥ n`. Returns the element count for the ungtr (buffer or USM version) scratchpad.

### unmqr

Multiplies a rectangular complex matrix C by Q or Q^H from `geqrf`: forms `Q*C`, `QH*C`, `C*Q`, or `C*QH` (overwriting C). Devices: `std::complex<float>`/`std::complex<double>`, each `CPU and GPU*` (*hybrid support; some computations are performed on the CPU).
```cpp
void unmqr(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // p687
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event unmqr(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // USM, p689
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`side` (`mkl::side::left` ⇒ applied from the left; `mkl::side::right` ⇒ from the right); `trans` (as printed: "If `trans=mkl::transpose::trans`, the routine multiplies C by Q. If `trans=mkl::transpose::nontrans`, the routine multiplies C by QT."); `m` rows in A (`0 ≤ m`); `n` columns in A (`0 ≤ n ≤ m`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` = result of geqrf, size ≥ `lda*k`; `tau` = tau returned by geqrf; `c` = matrix C, size ≥ `ldc*n`; `ldc`. Out: `c` "Overwritten by the product Q*C, QT*C, C*Q, or C*QT (as specified by side and trans)."

### unmqr_scratchpad_size

```cpp
template<typename T> int64_t unmqr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc)  // p691
```
Inputs as for `unmqr`. Returns the element count for the unmqr (buffer or USM version) scratchpad.

### unmrq

Multiplies a complex m-by-n matrix C by Q or Q^H, where Q is the complex unitary matrix defined as a product of k elementary reflectors H(i) of order n, as returned by the RQ factorization routine `gerqf`; forms `Q*C`, `QH*C`, `C*Q`, or `C*QH`. Devices: `std::complex<float>` CPU, `std::complex<double>` CPU.
```cpp
void unmrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // p692
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event unmrq(sycl::queue &queue, mkl::side side, mkl::transpose trans, int64_t m, int64_t n, int64_t k,  // USM, p694
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
`side`, `trans` (same printed convention as `unmqr`); `m` rows in A (`0 ≤ m`); `n` columns in A (`0 ≤ n ≤ m`); `k` elementary reflectors defining Q (`0 ≤ k ≤ n`); `a` = result of `gerqf`, size ≥ `lda*k`; `tau` = tau returned by `gerqf`; `c` = matrix C, size ≥ `ldc*n`; `ldc`. Out: `c`.

### unmrq_scratchpad_size

```cpp
template<typename T> int64_t unmrq_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::transpose trans,
  int64_t m, int64_t n, int64_t kint64_t lda, int64_t ldc)  // p696
```
NOTE: the `k`/`lda` line is printed exactly as `int64_t kint64_t lda,` in the extracted text (a source/extraction defect); the intended parameters are `int64_t k, int64_t lda`. Inputs otherwise as for `unmrq`. Returns the element count for the unmrq (buffer or USM version) scratchpad.

### unmtr

Multiplies a complex matrix C by Q or Q^H, where Q is the unitary matrix formed by `hetrd` when reducing a complex Hermitian A to tridiagonal form; use after `hetrd`; forms `Q*C`, `QH*C`, `C*Q`, or `C*QH`. Devices: `std::complex<float>` CPU, `std::complex<double>` CPU.
```cpp
void unmtr(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans, int64_t m, int64_t n,  // p697
  sycl::buffer<T> &a, int64_t lda, sycl::buffer<T> &tau, sycl::buffer<T> &c, int64_t ldc,
  sycl::buffer<T> &scratchpad, int64_t scratchpad_size)
sycl::event unmtr(sycl::queue &queue, mkl::side side, mkl::uplo uplo, mkl::transpose trans, int64_t m, int64_t n,  // USM, p699
  const T *a, int64_t lda, const T *tau, T *c, int64_t ldc, T *scratchpad, int64_t scratchpad_size,
  const std::vector<sycl::event> &events = {})
```
"In the descriptions below, r denotes the order of Q: `r = m` if `side = side::left`; `r = n` if `side = side::right`."
`side` (`side::left`/`side::right`); `uplo` must be `uplo::upper` or `uplo::lower`, using the same uplo supplied to `hetrd (USM Version)`; `trans` — "If trans = transpose::nontrans, the routine multiplies C by Q. If trans = transpose::trans, the routine multiplies C by QT."; `m` rows in C (`m ≥ 0`); `n` columns in C (`n ≥ 0`); `a` = array returned by `hetrd (USM Version)`; `lda ≥ max(1, r)`; `tau` = tau returned by `hetrd (USM Version)`, dimension ≥ `max(1, r-1)`; `c` = matrix C, size ≥ `ldc*n`; `ldc ≥ max(1, m)`. Out: `c`.

### unmtr_scratchpad_size

```cpp
template<typename T> int64_t unmtr_scratchpad_size(sycl::queue &queue, mkl::side side, mkl::uplo uplo,
  mkl::transpose trans, int64_t n, int64_t lda, int64_t ldc)  // p701
```
NOTE: the extracted Syntax omits `m` (it declares only `n`, `lda`, `ldc`) while the Input Parameters prose describes both `m` ("The number of rows in the matrix C (m ≥ 0)") and `n`; the parameter list is transcribed as printed and should be treated as suspect. Returns the element count for the unmtr (buffer or USM version) scratchpad.

### Domain boundary after unmtr_scratchpad_size

Pages 702–703 introduce the separate **Vector Mathematical Functions (VM)** domain, not `lapack` routines: "oneMKL Vector Mathematics functions (VM) compute a mathematical function of each of the vector elements", split into VM Mathematical Functions and VM Service Functions ("set/get the accuracy modes and the error codes, and create error handlers for mathematical functions"), plus a **Special Value Notations** topic. Facts stated there (kept only as a boundary marker): VM interfaces are given in `oneapi/mkl/vm.hpp`; examples are in `${MKL}/share/doc/mkl/examples/sycl/vml/source`; "All the VM mathematical functions can perform in-place operations, where the input and output arrays are at the same memory locations. For VM mathematical functions with positive increment indexing, in-place operations are supported only when the input and output increments have the same value."; `CONJ(x+i·y)=x-i·y` and `CIS(y)=cos(y)+i·sin(y)`.

## Formulas

No formula *images* exist for pages 546–703 (the formula manifest lists none), so every definition below was transcribed from inline text on the cited page. The source prints conjugates as `H` suffixes (`UH` = U^H) and transposes as `T` suffixes (`UT` = U^T).

- hetrf (p546): `A = U*D*UH` if `uplo=uplo::upper`; `A = L*D*LH` if `uplo=uplo::lower`; U/L are products of permutation and unit-diagonal triangular matrices (upper for U, lower for L), D is Hermitian block-diagonal with 1-by-1 and 2-by-2 diagonal blocks.
- orgbr (p549–552): generates the whole or part of the orthogonal matrices Q and P^T from `gebrd`; whole Q = `orgbr(queue, generate::q, m, m, n, a, ...)`; n leading columns of Q (m > n) = `orgbr(queue, generate::q, m, n, n, a, ...)`; whole n-by-n P^T = `orgbr(queue, generate::p, n, n, m, a, ...)`; m leading rows of P^T (m < n) = `orgbr(queue, generate::p, m, n, m, a, ...)`.
- orgqr (p555–557): whole Q = `mkl::orgqr(queue, m, m, p, a, lda, tau, ...)`; leading p columns = `mkl::orgqr(queue, m, p, p, a, lda, tau, ...)`; Qk of leading k columns = `mkl::orgqr(queue, m, m, k, a, lda, tau, ...)`; leading k columns of Qk = `mkl::orgqr(queue, m, k, k, a, lda, tau, ...)`; output `a` holds n leading columns of the m-by-m orthogonal Q.
- orgqr_batch (p558–563): same four patterns for Qi/Qik: `orgqr_batch(queue, m, m, p, a, ...)`, `orgqr_batch(queue, m, p, p, a, ...)`, `orgqr_batch(queue, m, m, k, a, ...)`, `orgqr_batch(queue, m, k, k, a, ...)`; `a` is overwritten by a batch of n leading columns of the m-by-m orthogonal Qi.
- orgtr (p568–570): explicitly generates the n-by-n orthogonal Q from `sytrd` for `A = Q*T*QT`.
- ormqr (p572–574): forms one of `Q*C`, `QT*C`, `C*Q`, `C*QT` (overwriting C), selected by `side` and `trans`.
- ormrq (p576–579): `Q = H1H2 ... Hk` as returned by `gerqf`; forms one of `Q*C`, `QT*C`, `C*Q`, `C*QT`.
- ormtr (p581–584): with `A = Q*T*QT` from `sytrd`, forms one of `Q*C`, `QT*C`, `C*Q`, `C*QT`; `r` (order of Q) = `m` if `side::left`, `= n` if `side::right`.
- potrf (p587–589): `A = UT*U` real / `A = UH*U` complex if `uplo=mkl::uplo::upper`; `A = L*LT` real / `A = L*LH` complex if `uplo=mkl::uplo::lower`; L lower, U upper triangular.
- potrf_batch (p590–595), for `i ϵ {1...batch_size}`: `Ai = UiT * Ui` real / `Ai = UiH * Ui` complex if `uplo = mkl::uplo::upper`; `Ai = LiT * Li` real / `Ai = LiH * Li` complex if `uplo = mkl::uplo::lower` (the lower-case line is printed with the upper-case factor order — see Explicit gaps).
- potri (p600–602): computes `inv(A)` of a symmetric (Hermitian) positive-definite A given the `potrf` Cholesky factorization.
- potrs (p604–607): solves `A*X = B` with `A = UT*U` real / `UH*U` complex if `uplo=mkl::uplo::upper`, or `A = L*LT` real / `L*LH` complex if `uplo=mkl::uplo::lower`.
- potrs_batch (p608–615), for `i ϵ {1...batch_size}`: solves `Ai*Xi = Bi` with `Ai = UiT*Ui` real / `Ai = UiH*Ui` complex if `uplo=mkl::uplo::upper`, and `Ai = Li*LiT` real / `Ai = Li*LiH` complex if `uplo=mkl::uplo::lower`.
- syevd (p619–622): `A = Z*λ*ZT` with Λ diagonal holding eigenvalues λi and Z orthogonal with eigenvector columns zi; `A*zi = λi*zi` for `i = 1, 2, ..., n`.
- syevx (p624–629): selected eigenpairs (λ, z) with `A*z = z*λ`; convergence when an approximate eigenvalue lies in an interval `[a,b]` of width `<= abstol + eps * max( |a|,|b| )`, or `eps*|T|` when `abstol <= 0` (`|T|` = 1-norm of the tridiagonal form of A).
- sygvd (p632–636): `A*x = λ*B*x`, `A*B*x = λ*x`, or `B*A*x = λ*x` for `itype` = 1, 2, 3; normalization `if itype= 1 or 2, ZT*B*Z = I; if itype= 3, ZT*inv(B)*Z = I`; b overwritten by the factor of `B = UT*U` or `B = L*LT` when `info ≤ n`.
- sygvx (p638–644): same generalized forms and `itype` selection as sygvd, with selected eigenpairs; same `abstol` criterion as syevx; b overwritten by the Cholesky factor `B = UT*U` or `B = L*LT`.
- sytrd (p646–650): `A = Q*T*QT`; Q represented by `n-1` elementary reflectors, with `tau` holding `(n-1)` scalars (`tau(n)` is workspace).
- sytrf (p651–655): `A = U*D*UT` if `uplo=uplo::upper`; `A = L*D*LT` if `uplo=uplo::lower`, with the same block structure as hetrf (symmetric case).
- trtri (p656–658): computes `inv(A)` of an upper or lower triangular A (unit or non-unit diagonal per `diag`).
- trtrs (p659–663): `A*X = B` if `transa =transpose::nontrans`; `AT*X = B` if `transa =transpose::trans`; `AH*X = B` if `transa =transpose::conjtrans` (complex only).
- ungbr (p664–668): generates the whole or part of the unitary Q and P^H from `gebrd`, with the same four usage patterns as orgbr written as `ungbr(...)`.
- ungqr (p670–673): whole Q = `mkl::ungqr(queue, m, m, p, a, lda, tau, ...)`; leading p columns = `mkl::ungqr(queue, m, p, p, a, lda, tau, ...)`; Qk of leading k columns = `mkl::ungqr(queue, m, m, k, a, lda, tau, ...)`; leading k columns of Qk = `mkl::ungqr(queue, m, k, k, a, lda, tau, ...)`; `a` holds n leading columns of the m-by-m unitary Q.
- ungqr_batch (p674–679): same four patterns for Qi/Qik; `a` is overwritten by a batch of n leading columns of the m-by-m unitary Qi.
- ungtr (p683–685): explicitly generates the n-by-n unitary Q from `hetrd`; p683 states `A = Q*T*QH`, p685 states `A = Q*T*QT` (as printed).
- unmqr (p687–690): forms one of `Q*C`, `QH*C`, `C*Q`, `C*QH` (overwriting C), Q from `geqrf`.
- unmrq (p692–695): Q is "a product of k elementary reflectors H(i) of order n: `Q = H(1)H* H(2)H*…*H(k)H` as returned by the RQ factorization routine gerqf" (the superscripts are garbled in extraction); forms one of `Q*C`, `QH*C`, `C*Q`, `C*QH`.
- unmtr (p697–700): with `A = Q*T*QH` from `hetrd` (p697; p699 states `A = Q*T*QT` as printed), forms one of `Q*C`, `QH*C`, `C*Q`, `C*QH`; `r` (order of Q) = `m` if `side::left`, `= n` if `side::right`.

## Conventions & Gotchas

- **Buffer vs USM.** Buffer routines return `void` and throw; USM routines return an output `sycl::event` and take `const std::vector<sycl::event> &events = {}`. Never pass `events` to a buffer routine: the parameter does not exist there.
- **Scratchpad sizing is mandatory.** Call the matching `*_scratchpad_size` first; the value is a count of elements of type `T`. On a too-small scratchpad, `info` equals the `scratchpad_size` you passed and `detail()` returns the required size.
- **In-place results.** `potrf`/`potri` overwrite `a`; `potrs`/`trtrs` overwrite `b`; `orm*`/`unm*` overwrite `c`; `org*`/`ung*` overwrite `a`; eigensolvers overwrite `a` (and `b` for the generalized ones). No routine returns a separate output array for the overwritten operand.
- **Leading dimensions / strides.** `potrf`: size of `a` ≥ `lda*n`; `orgqr`/`ungqr`: `lda ≥ m`; `trtri`/`trtrs`: `lda ≥ max(1, n)`, `ldb ≥ max(1, n)`; `syevd`/`syevx`/`sygvd`/`sygvx`/`sytrd`: `lda` (and `ldb`, `ldz`) ≥ `max(1,n)`; batch strided: `lda ≥ max(1, n)` and `stride_a ≥ max(1, lda * n)`; `potrs_batch` adds `ldb ≥ max(1, n)`, `stride_b ≥ max(1, ldb * nrhs)`; QR batch generators use `stride_tau ≥ max(1, min(m,n))`.
- **Batch shapes.** Group variants use `T **a` (and `const T * const *tau` for generators, `const T * const *a` for `potrs_batch` group); per-group parameters are arrays indexed by `g`; total `batch_size` is the sum of `group_sizes`.
- **Indexing.** `syevx`/`sygvx` `il`/`iu` are explicitly **one-based** ("Must be `1 ≤ il ≤ iu ≤ n` if `n > 0` and `il = 1` and `iu = 0` if `n = 0`"). Every other number here is a size/count, not an element index.
- **Unknown output size.** With `range = rangev::values`, the number of found eigenvalues `m` is not known in advance: supply at least `max(1, m)` columns in `Z` using an upper bound. `m` is an output (buffer `sycl::buffer<int64_t>&`, USM `int64_t *`).
- **Selection interval.** `rangev::values` selects eigenvalues in the half-open interval `(vl, vu]` with `vl` < `vu`; `vl`/`vu` are not referenced for `rangev::all`/`rangev::indices`.
- **Destructive solvers.** `syevx`/`sygvx` destroy the stored triangle of A including the diagonal; generalized solvers replace the stored part of `b` with the Cholesky factor.
- **Generalized problem types.** `itype` ∈ {1, 2, 3} ⇒ `A*x = lambda*B*x`, `A*B*x = lambda*x`, `B*A*x = lambda*x`. Normalization depends on `itype` (`ZT*B*Z = I` for 1 and 2, `ZT*inv(B)*Z = I` for 3). B is assumed positive definite; if it cannot be factored, `info = n + i`.
- **Algorithm selection.** `syevd`/`sygvd` use divide and conquer when eigenvectors are requested; `syevd` falls back to the Pal-Walker-Kahan variant of QL/QR for eigenvalues only; `syevx`/`sygvx` use bisection, plus modified twisted factorization / inverse iteration with Gram-Schmidt orthogonalization for eigenvectors.
- **Result reuse.** `orgqr`/`ungqr` and `ormqr`/`unmqr` consume `geqrf` output; `orgbr`/`ungbr` consume `gebrd`; `orgtr`/`ormtr` consume `sytrd`; `ungtr`/`unmtr` consume `hetrd`; `ormrq`/`unmrq` consume `gerqf`; `potri`/`potrs` require prior `potrf`; `potrs_batch` requires prior `potrf_batch`.
- **Device restrictions.** Each routine is CPU-only or CPU+GPU with the footnote shown in its entry; the routine-level precision/device table is authoritative, and `*_scratchpad_size` tables sometimes differ (e.g. `orgqr_batch_scratchpad_size` lists `CPU and GPU` without the interface-only asterisk).
- **Integer type.** All signature parameters are `int64_t`; USM `hetrf`/`sytrf` pivots are `int64_t *ipiv` (but see the sytrf buffer quirk below).

## Explicit gaps

- **No formula images exist for pages 546–703.** `formula-manifest.json` lists none for this range and every chunk header 038–047 says `(none)`; all equations in `## Formulas` came from inline text. No image-based cross-check of any formula was possible, and PDF formulas whose layout did not survive text extraction cannot be verified.
- **Include files are not stated** for any routine in this chapter; only the namespace is given. Don't infer a header from this chapter alone.
- **No buffer (`sycl::buffer`) signature is given here for `hetrf` or `trtri`** — both appear as USM-only on these pages.
- **`orgtr_scratchpad_size` has no return type in the extracted Syntax** (printed as `orgtr_scratchpad_size(sycl::queue &queue, mkl::uplo uplo, int64_t n, int64_t lda)`).
- **`unmrq_scratchpad_size` Syntax prints `int64_t kint64_t lda,`** — a missing comma/line break between `k` and `lda`.
- **`unmtr_scratchpad_size` Syntax omits `m`** (declares `n`, `lda`, `ldc` only) although its prose describes `m`; the correct parameter list is not stated unambiguously.
- **`sytrf` buffer signature declares `sycl::buffer<T> &ipiv`** (element type `T`) although `ipiv` holds integer interchanges and the USM form uses `int64_t *ipiv` — likely a documentation defect.
- **`sytrd (USM Version)` first parameter prints as `queue &queue`** without `sycl::`; `sytrf (USM Version)` likewise prints `sycl::event sytrf(queue &queue, ...)`.
- **`trtrs` buffer "Return Values" describes an `info` buffer** ("Buffer containing error information. If info = 0, the execution is successful...") that does not exist in the signature.
- **`ungtr (USM Version)` precision table lists `float`/`double`**, contradicting its Description and the buffer version's complex-only table.
- **`sygvd (USM Version)` device table is headed `hegvd (USM version)`** — a wrong routine name in the source heading.
- **Inconsistent lower-triangular factorization formulas.** `potrf_batch` prints `Ai = LiT * Li` / `Ai = LiH * Li` for the lower case, while `potrs_batch` prints `Ai = Li*LiT` / `Ai = Li*LiH`; both are verbatim above, and the mathematically consistent lower Cholesky form is `Ai = Li*LiH`.
- **Inconsistent tridiagonal-reduction statements.** `ungtr` says `A = Q*T*QH` on p683 but `A = Q*T*QT` on p685; `unmtr` says `A = Q*T*QH` on p697 but `A = Q*T*QT` on p699.
- **`unmrq` reflector product is garbled**: "Q = H(1)H* H(2)H*…*H(k)H" — the conjugate-transpose superscript placement did not survive extraction, so the exact product form is not reliably legible.
- **`orgqr (USM Version)` exception table includes a `dii is 0` / `D is exactly singular` condition** belonging to a factorization routine, not to QR generation; transcribed as printed.
- **The complex `unm*` routines describe the result as `Q*C, QT*C, C*Q, C*QT`** in their parameter/output prose while their Descriptions use `QH`; and `unmqr`'s `side` text says "Q or QT" even though the routine is documented as Q/Q^H. Both wordings appear as printed.
- **`ungbr (USM Version)` output text says "orthogonal matrix Q or PT"** for a complex routine; the source uses orthogonal/unitary interchangeably in these output sentences.
- **No `syevr`, `sygvr`, or `heevd`/`heevx`/`hegvd` sections appear in these pages** (only the real `sy*` eigensolvers and the complex `hetrf`), and no `potri_batch`/`trtri` buffer variants are given.
