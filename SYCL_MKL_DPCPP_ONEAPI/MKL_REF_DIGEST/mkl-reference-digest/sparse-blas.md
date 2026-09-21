# Sparse BLAS: Handle Contract, Storage Formats, and Routines

oneMKL Sparse BLAS is the C++ with SYCL interface to sparse matrix/vector operations. Its central object is the opaque `oneapi::mkl::sparse::matrix_handle_t`, which holds a *view* of user-owned sparse arrays plus library-owned optimization state that persists across calls. This chapter covers the handle contract, the four supported storage formats (CSR, COO, CSC, BSR), the supported data/integer types, and every documented routine grouped as state management, analysis (`optimize_*`), execution, and helpers. All identifiers and signatures are copied verbatim from the oneMKL - Data Parallel C++ Developer Reference (2026.0); line wrapping inside signatures is the only formatting change.

## Overview

### Objects and structures

`sparse::matrix_handle_t` (pointer to an opaque sparse matrix object); `sparse::property` (user-provided matrix data guarantees, such as sorted input data); `sparse::omatadd_alg` and `sparse::omatadd_descr_t` (algorithm enum and opaque descriptor for `sparse::omatadd`); `sparse::matrix_view_descr`, `sparse::matmat_request`, `sparse::matmat_descr_t` (for `sparse::set_matmat_data` / `sparse::matmat`); `sparse::omatconvert_alg` and `sparse::omatconvert_descr_t` (for `sparse::omatconvert`).

### Routine groups

1. **State management** — init/destroy and setting data, formats, properties (`init_matrix_handle`, `release_matrix_handle`, `set_<format>_data`, `set_matrix_property`, descriptor init/set/get/release).
2. **Analysis** (inspector/optimize stage) — `optimize_gemv`, `optimize_trmv`, `optimize_trsv`, `optimize_gemm`, `optimize_trsm`.
3. **Execution** — `gemv`, `gemvdot`, `symv`, `trmv`, `trsv`, `gemm`, `trsm`, `omatadd`, `matmat`, `matmatd`.
4. **Helper** — `omatcopy`, `omatconvert`, `sort_matrix`, `update_diagonal_values`.

The user data is **not** changed by any analysis/optimization routine. Optimizations may be reused by multiple execution routines: an analysis routine is typically called once per operation while the corresponding execution routines may be called many times. Execution routines perform the actual matrix-matrix/matrix-vector operations using the data, optimizations, and properties stored in the handle.

### Supported data and integer types

`<DATA_TYPE>`: `float`, `double`, `std::complex<float>`, `std::complex<double>`. `<INT_TYPE>`: `std::int32_t`, `std::int64_t`. Unless a routine's own "Formats Supported" list says otherwise, execution routines accept all four `<DATA_TYPE>` values.

### Include files

All Sparse BLAS declarations are in `oneapi/mkl/spblas.hpp`, which is also where the `oneapi::mkl::sparse::property` enum class is defined. The reference lists this single header for every routine below.

### Buffer vs USM duality

Most routines have a `sycl::buffer` form (returning `void`) and a USM-pointer form (returning `sycl::event`, usually with a trailing `const std::vector<sycl::event> &dependencies = {}`). `set_<format>_data` buffer forms return `void` and have **no** `dependencies` parameter. The reference notes that most `sycl::buffer` oneMKL APIs do not have the ability to pass in dependencies; passing `{}` or a dependency vector to a `sycl::buffer` call is "not recommended ... but supported." For `sycl::buffer` analysis routines, both `r(...)` and `static_cast<void>(r(...))` are listed as recommended uses, while assigning the event is "allowed ... but most sycl::buffer oneMKL APIs do not have ability to pass in dependencies."

### Matrix handle contract between User and Library

The handle is created by `sparse::init_matrix_handle` and filled by `sparse::set_<format>_data(q, handle, /*user data*/)`. Unlike most other oneMKL domains, the handle **along with the user data persists outside of individual calls**; when the user subsequently calls a Sparse BLAS API with that handle, the library uses the data stored in it. Both parties can access the data at the same time, hence this implicit contract.

**Description.** Providing device USM pointers binds the handle to the `sycl::context` and `sycl::device` associated with the device USM pointer; it can be used on any device within the context that is compatible with that `sycl::device` (can read from that device peer to peer). Using shared and host USM pointers or `sycl::buffer`s binds the handle to the associated `sycl::context` and it could be used on any device in the context. It is the **User's responsibility** to make sure they use the handle with appropriate queues and devices within the given context.

The handle is best described as a "view of User's matrix data arrays with an opaque state attached to it": lightweight to begin with, but through `sparse::optimize_*` APIs and in some cases execution APIs (like `sparse::matmat()`) the hidden state can grow with internal optimizations/structures that persist through the lifetime of the handle.

**User agreements.** (1) The User owns any data provided to the handle and is responsible to create and dispose of it correctly; data can be disposed of only after all uses of the handle and the release of the handle have finished. (2) The User agrees **not to modify the data arrays directly while they are in a** `sparse::matrix_handle_t`; any changes must happen before attaching to a handle or, when available, indirectly through a Sparse BLAS library API that states it will modify the handle data in specified ways.

**Library agreements.** (1) The Library may create and store data in the handle during a Sparse BLAS API call; such data is owned by the Library and disposed of at the respective matrix handle release, and the User cannot access it. (2) The Library agrees to not modify the user-provided data arrays in the handle unless through a library API that specifically states it may change the data (such as `sparse::sort_matrix()` or `sparse::omatcopy()`).

**Advanced use recommendations.** (1) The `sparse::matrix_handle_t` object is **not currently considered thread-safe**, so it must be used serially on the host. (2) Most Sparse BLAS operations consume the user-provided data read-only; exceptions are APIs that fill a handle such as `sparse::matmat()` for the output `C` matrix (`A` and `B` are still read-only), `sparse::omatcopy()`, and APIs that explicitly state they modify user data such as `sparse::sort_matrix()`. (2a) It is recommended not to use the same data arrays in multiple handles on the same context and device; with care it may be possible to do so safely, but doing so may affect performance. (2b) It is also recommended not to use the raw data arrays of a handle read-only in other kernels or library calls independent of the handle on the same context and device; with care it may be possible, but it may affect performance.

**Canonical workflow:** (A) allocate user memory for the sparse matrix (SYCL USM device alloc or `sycl::buffer`); (B) do anything with the matrix arrays, such as memcpy data from host; (C) create a sparse matrix handle and pass in the matrix arrays; (D) make library calls; (E) destroy the handle via `sparse::release_matrix_handle()`; (F) do anything with user memory — modify, copy, etc.; (G) create another handle using the same arrays; (H) make other library calls; (I) destroy that handle via `release_matrix_handle()`; (J) do whatever you want with user memory.

### Storage formats

Supported formats: **Compressed Sparse Row (CSR)**, **Coordinate (COO)**, **Compressed Sparse Column (CSC)**, **Block Compressed Sparse Row (BSR)**.

- **Repeated indices/non-zeros:** it is possible to create matrices with 'repeated' indices where the value of the non-zero is considered as a sum of its values where the row and column indices are repeated. Using such matrices with oneMKL is currently **undefined behavior with no guarantees of correctness**; it is user-responsibility to ensure non-zero values are not repeated and are appropriately 'compressed' into a single non-zero value. Unless explicitly specified, oneMKL documentation assumes arrays without repeated indices.
- **Explicit zeros:** all references to 'non-zeros' mean **structural** non-zeros stored in the matrix, **including explicitly stored zero values**.

**CSR** (sometimes called 3-array CSR or CSR3): represented by scalar sizes `(nrows, ncols, nnz)`, three data arrays `row_ptr`, `col_ind`, `values`, and the `index_base` parameter.

| Element | Meaning |
|---|---|
| `nrows`, `ncols` | Number of rows / columns in the sparse matrix. |
| `nnz` | Number of stored elements (sometimes called number of non-zeros). |
| `index` | Whether the matrix has zero or one-based indexing. |
| `values` | Array of the `nnz` stored element values, stored row by row. |
| `col_ind` | Integer array of `nnz` column indices such that `col_ind[i]` is the column number of the element stored in `values[i]`. |
| `row_ptr` | Integer array of size `nrows + 1`; element `j` gives the position in `values` of the first non-zero element in row `j`, equal to `row_ptr[j] - index`. `row_ptr[nrows]` stores the sum of `nnz` and `index`, i.e. `nnz = row_ptr[nrows] - index`. |

The 3-array CSR format oneMKL supports has **sorted rows by definition**, but column indices within each row may or may not be sorted. oneMKL **assumes CSR matrix handles to be unsorted by default**. Sortedness states: Unsorted (column indices within each row may not be ordered) → property `None (default)`; Sorted (by rows and by columns within each row) → `sparse::property::sorted`.

**COO**: represented by scalar sizes `(nrows, ncols, nnz)`, three data arrays `row_ind`, `col_ind`, `values`, and the `index` parameter. Each tuple `(row_ind[i], col_ind[i], values[i])` represents a non-zero value.

| Element | Meaning |
|---|---|
| `nrows`, `ncols` | Number of rows / columns in the sparse matrix. |
| `nnz` | Number of non-zeros in the sparse matrix. |
| `index` | Whether the matrix has zero or one-based indexing. |
| `row_ind` | Integer array of row indices, such that `row_ind[i]` is the row number of the non-zero value in `values[i]`. |
| `col_ind` | Integer array of column indices, such that `col_ind[i]` is the column number of the non-zero value in `values[i]`. |
| `values` | Array that contains the non-zero elements, preferably but not necessarily stored in a sorted order. |

NOTE: the transpose of a COO matrix represented by `(row_ind, col_ind, values)` is `(col_ind, row_ind, values)`; interchanging the row and column index arrays gives the transposed form.

COO sortedness states (five broad states, two unique): Arbitrary, non-unique unsorted order → `None (default)`; Unsorted CSR-style/partially sorted (sorted row indices, column indices within rows unsorted) → `None (support may be added in the future)`; Unsorted CSC-style/partially sorted (sorted column indices, row indices within columns unsorted) → `None (no plans to add support)`; Sorted CSR-style (rows then columns within rows) → `sparse::property::sorted`; Sorted CSC-style (columns then rows within columns) → `None (no plans to add support)`. oneMKL assumes COO arrays fully unsorted by default and currently has **no plans to add support** for either sorted or unsorted CSC-style states. `sparse::property::sorted` must **not** be specified on a COO handle created with CSC-style-sorted arrays; because the transpose of a COO matrix is immediately available by interchanging row and column index arrays, the transpose of a CSC-style-sorted COO matrix is automatically CSR-style-sorted, so users can create a handle for the transposed matrix with the arrays interchanged and set `sorted` on that handle. For example `sparse::gemv(q, mkl::transpose::nontrans, cooAMatrix, ...)` cannot use `sorted` on sorted-CSC-style arrays in `cooAMatrix`, but `sparse::gemv(q, mkl::transpose::trans, transCooAMatrix, ...)` can use it on `transCooAMatirx` (spelling as in the source), which has sorted-CSR-style arrays.

**CSC** (3-array CSC or CSC3): represented by `(nrows, ncols, nnz)`, `col_ptr`, `row_ind`, `values`, and the `index_base` parameter. `values` contains the `nnz` stored element values compressed column by compressed column; `row_ind` is an integer array of `nnz` row indices such that `row_ind[i]` is the row number of the element stored in `values[i]`; `col_ptr` is an integer array of size `ncols + 1` where element `j` gives the position in `values` of the first non-zero element in column `j` of A, equal to `col_ptr[j] - index`, and `col_ptr[ncols]` stores the sum of `nnz` and `index`, i.e. `nnz = col_ptr[ncols] - index`.

The 3-array CSC format has **sorted columns by definition**, but row indices within each column may be unsorted. Default is unsorted (property `None (default)`); `sparse::property::sorted` means sorted by columns and sorted by rows within each column.

**BSR** (3-array BSR / BSR3): represented by scalar sizes `blk_nrows`, `blk_ncols`, `row_blk_size`, `col_blk_size`, `blk_nnz`, plus the `index_base` parameter and three arrays `bsr_row_ptr`, `bsr_col_ind`, `bsr_values`. Block sizes (rather than full `nrows`/`ncols`) are used so the matrix is fully blocked without partial blocks on the edges.

| Element | Meaning |
|---|---|
| `blk_nrows`, `blk_ncols` | Number of block rows / block columns in the sparse matrix. |
| `row_blk_size`, `col_blk_size` | Number of rows / columns for the dense blocks in the matrix. |
| `blk_nnz` | Number of stored block elements (number of non-zero blocks). |
| `blk_layout` | Whether the dense blocks are stored in row-major or column-major layout. |
| `index` | Whether the matrix has zero or one-based indexing. |
| `bsr_values` | Array of the `blk_nnz * row_blk_size * col_blk_size` stored element values, stored block row by block row using `blk_layout`. |
| `bsr_col_ind` | Integer array of `blk_nnz` column block indices, such that `bsr_col_ind[i]` is the column number of the block element stored in `bsr_values[i * row_blk_size * col_blk_size]`. |
| `bsr_row_ptr` | Integer array of size `blk_nrows + 1`; element `j` gives the position of the block element in `bsr_col_ind` that is the first non-zero block element in block row `j` of A, and when scaled by size of blocks also gives the position in `bsr_values`. `bsr_col_ind[bsr_row_ptr[j]-index] * col_blk_size` is the column index of the first stored element in the j'th block row, and `bsr_values[(bsr_row_ptr[j]-index) * row_blk_size * col_blk_size]` is the first stored value in that block row. `bsr_row_ptr[blk_nrows]` stores the sum of `blk_nnz` and `index`, i.e. `blk_nnz = bsr_row_ptr[blk_nrows] - index`. |

There is **no leading dimension for the blocks** in the format definition, meaning the leading dimension should be the same as `row_blk_size` or `col_blk_size` depending on `blk_layout`. The format has sorted block rows by definition; block column indices within each block row may be unsorted. Default unsorted; `sparse::property::sorted` means sorted by block rows and sorted by block columns within each block row. Square **or rectangular** blocks are supported; square blocks are important for a triangular-solve-like operation, as that is all that will be supported, but for matrix multiplications rectangular sizes are supported.

### Device/format support summary

- **CSR** — CPU/GPU for every routine that lists CSR, except `update_diagonal_values`, which is **GPU only**.
- **COO** — CPU/GPU for `gemv`, `optimize_gemv`, `optimize_gemm`, `gemm`, `omatcopy`, `omatconvert`; **CPU only** for `optimize_trmv`, `optimize_trsv`, `gemvdot`, `trmv`, `trsv`, `optimize_trsm`, `trsm`.
- **CSC** — CPU/GPU for `gemv`; **GPU only** for `optimize_gemv`.
- **BSR** — CPU/GPU for `gemv`; **GPU only** for `optimize_gemv`.

### Error handling

Most routines note "Refer to Error Handling for a detailed description of the possible exceptions thrown." Concrete facts stated in this text: `sparse::trsv` and `sparse::trsm` throw an exception of type `onemkl::invalid_value()` when `diag::nonunit` is selected and a diagonal value is missing from the sparsity profile or is zero. `set_matrix_property` accepts properties **as truth from the user without verification**. `init_matrix_handle` and `init_matmat_descr` state "otherwise it throws an exception"; the `set_<format>_data` routines only cross-reference Error Handling and state no explicit failure behavior of their own.

## Routines

### init_matrix_handle

Allocates memory on the heap for a `sparse::matrix_handle_t` object, initializes its internals to default values, and returns the address to that object; otherwise it throws an exception.

```cpp
namespace oneapi::mkl::sparse {
    void init_matrix_handle (oneapi::mkl::sparse::matrix_handle_t *p_spMat)
}
```

Include: `oneapi/mkl/spblas.hpp`. `p_spMat` (input): the address of the handle to be initialized; it is recommended that the handle object be set to `nullptr` before entry to avoid confusion about whether it is previously initialized, and this initialization routine must **only** be called on an uninitialized matrix_handle object. `p_spMat` (output): on return the address is updated to point to a newly allocated and initialized `matrix_handle_t` object that can be filled and used to perform sparse BLAS operations.

```cpp
using namespace oneapi::mkl;
sparse::matrix_handle_t spMat = nullptr;
sparse::init_matrix_handle(&spMat);
sycl::event ev_set = sparse::set_csr_data(queue, spMat, /*matrix sizes and arrays */);
sycl::event ev_release = sparse::release_matrix_handle(queue, &spMat, dependencies);
ev_release.wait();  // make it blocking
```

### release_matrix_handle

Releases internal data in the provided `matrix_handle_t` object, waits for the dependencies to be finished when provided, then frees the handle itself.

```cpp
namespace oneapi::mkl::sparse {
    sycl::event release_matrix_handle (sycl::queue & queue,
        oneapi::mkl::sparse::matrix_handle_t *p_spMat,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `queue`: SYCL command queue used for SYCL kernels execution. `p_spMat`: the address of the handle to be released, initialized with `init_matrix_handle` and filled with user data using one of the `set_<sparse_matrix_type>_data` routines; the `optimize_xyz` routines may also have created additional internally allocated data which is released in this call. `dependencies`: any events the handle depends on before executing the release. Output `p_spMat`: updated to point to a null object, with the passed-in handle scheduled for deallocation and cleanup. Return `sycl::event`: can be waited upon or added as a dependency for the completion of the deallocation cleanup routines.

Formats supported for `p_spMat`: CSR on CPU/GPU, COO on CPU/GPU, CSC on CPU/GPU, BSR on CPU/GPU.

NOTE (oneMKL 2023.2 and beyond): although the address of the `matrix_handle_t` object is passed in (`matrix_handle_t *p_spMat`), it is dereferenced (`matrix_handle_t handle = *p_spMat;`) and updated to null **synchronously**, then the handle is passed into the asynchronously scheduled destruction routines, ensuring asynchronous destruction is safe. In previous releases the dereferencing occurred asynchronously, which can cause segmentation faults if `p_spMat` is a stack variable (such as `&handle`) and goes out of scope before the asynchronous execution (including the dereferencing) begins.

### set_csr_data

Takes a matrix handle and user-provided CSR matrix arrays and fills the internal CSR data structure of the matrix handle for a matrix of dimensions `nrows`-by-`ncols`.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void set_csr_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> &row_ptr,
        sycl::buffer<INT_TYPE, 1> &col_ind, sycl::buffer<DATA_TYPE, 1> &values);
 }
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event set_csr_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, INT_TYPE *row_ptr, INT_TYPE *col_ind,
        DATA_TYPE *values, const std::vector<sycl::event> &dependencies = {} );
}
```

Include: `oneapi/mkl/spblas.hpp`. `spMat`: handle to object containing sparse matrix and other internal data for subsequent Sparse BLAS operations. `nrows`/`ncols`: number of rows / columns of the input matrix. `nnz`: number of stored elements (sometimes called number of non-zeros). `index`: `oneapi::mkl::index_base::zero` — zero-based (C-style) indexing, indices start at 0 — or `oneapi::mkl::index_base::one` — one-based (Fortran-style) indexing, indices start at 1. `row_ptr`: SYCL memory object of length `nrows+1` containing row-wise starting and ending locations of non-zero column indices and values in index-based numbering. `col_ind`: SYCL memory object of column indices in index-based numbering, length ≥ `nnz`. `values`: SYCL memory object with the non-zero elements, length ≥ `nnz`. `dependencies`: USM API only. Output `spMat`; output `sycl::event` (USM API only): tracks the completion of asynchronous events enqueued during the call that continue the chain of events from the input dependencies.

Input array summary (for `sycl::buffer` inputs the arrays are `sycl::buffer<T, 1> &`; for USM inputs they must be device-accessible and of type `T *`):

| array | length (elements) | T | USM Memory Type |
|---|---|---|---|
| `row_ptr` | `nrows + 1` | `INT_TYPE` | device accessible (USM device or USM shared or USM host) |
| `col_ind` | `nnz` | `INT_TYPE` | device accessible (USM device or USM shared or USM host) |
| `values` | `nnz` | `DATA_TYPE` | device accessible (USM device or USM shared or USM host) |

In general, using USM device memory will provide better performance than USM shared, which in turn will give better performance than USM host, but all are supported as they are all device-accessible.

### set_csc_data

Takes a matrix handle and user-provided CSC matrix arrays and fills the internal CSC data structure of the matrix handle.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void set_csc_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> &col_ptr,
        sycl::buffer<INT_TYPE, 1> &row_ind, sycl::buffer<DATA_TYPE, 1> &values);
 }
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event set_csc_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, INT_TYPE *col_ptr, INT_TYPE *row_ind,
        DATA_TYPE *values, const std::vector<sycl::event> &dependencies = {} );
}
```

Include: `oneapi/mkl/spblas.hpp`. `col_ptr`: SYCL memory object of length `ncols+1` containing col-wise starting and ending locations of non-zero row indices and values in index-based numbering. `row_ind`: row indices in index-based numbering; must be of length ≥ `nnz`. `values`: non-zero elements, length ≥ `nnz`. Other parameters as for `set_csr_data`. Array summary: `col_ptr` — `ncols + 1`, `INT_TYPE`, device accessible; `row_ind` — `nnz`, `INT_TYPE`, device accessible; `values` — `nnz`, `DATA_TYPE`, device accessible.

### set_coo_data

Takes a matrix handle and user-provided COO matrix arrays and fills the internal COO data structure of the matrix handle.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void set_coo_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> &row_ind,
        sycl::buffer<INT_TYPE, 1> &col_ind, sycl::buffer<DATA_TYPE, 1> &values);
 }
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event set_coo_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t nrows, const std::int64_t ncols, const std::int64_t nnz,
        oneapi::mkl::index_base index, INT_TYPE *row_ind, INT_TYPE *col_ind,
        DATA_TYPE *values, const std::vector<sycl::event> &dependencies = {} );
}
```

Include: `oneapi/mkl/spblas.hpp`. `row_ind`, `col_ind`, `values`: SYCL memory objects of length `nnz` each, holding row indices, column indices, and the stored elements respectively, in index-based numbering. Output `spMat` is described in the source as the handle "for subsequent DPC++ Sparse BLAS operations"; output `sycl::event` (USM API only). Array summary: `row_ind` — `nnz`, `INT_TYPE`, device accessible; `col_ind` — `nnz`, `INT_TYPE`, device accessible; `values` — `nnz`, `DATA_TYPE`, device accessible.

### set_bsr_data

Takes a matrix handle and user-provided BSR matrix arrays and fills the internal BSR data structure of the matrix handle; dense blocks may be square or rectangular with row-major or column-major layout, stored in a compressed sparse row format of blocks. The matrix is of dimensions `blk_nrows * row_blk_size`-by-`blk_ncols * col_blk_size`.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void set_bsr_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t blk_nrows, const std::int64_t blk_ncols, const std::int64_t blk_nnz,
        const std::int64_t row_blk_size, const std::int64_t col_blk_size,
        oneapi::mkl::layout blk_layout, oneapi::mkl::index_base index,
        sycl::buffer<INT_TYPE, 1> &bsr_row_ptr, sycl::buffer<INT_TYPE, 1> &bsr_col_ind,
        sycl::buffer<DATA_TYPE, 1> &bsr_values );
 }
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event set_bsr_data (sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::int64_t blk_nrows, const std::int64_t blk_ncols, const std::int64_t blk_nnz,
        const std::int64_t row_blk_size, const std::int64_t col_blk_size,
        oneapi::mkl::layout blk_layout, oneapi::mkl::index_base index, INT_TYPE *bsr_row_ptr,
        INT_TYPE *bsr_col_ind, DATA_TYPE *bsr_values,
        const std::vector<sycl::event> &dependencies = {} );
}
```

Include: `oneapi/mkl/spblas.hpp`. `blk_nrows`/`blk_ncols`: number of block rows / block columns. `blk_nnz`: number of blocks (sometimes called number of nonzero blocks). `row_blk_size`/`col_blk_size`: number of rows / cols in each dense block. `blk_layout`: `oneapi::mkl::layout::row_major` (rows are concatenated one after another) or `oneapi::mkl::layout::col_major` (columns are concatenated one after another). `bsr_row_ptr`: array of length `blk_nrows+1` with row-wise starting and ending locations of non-zero column block indices and block of values in index-based numbering. `bsr_col_ind`: block column indices in index-based numbering; array must be at least `blk_nnz` in length. `bsr_values`: blocks of stored matrix elements concatenated one after another in `blk_layout` format; array must be at least `row_blk_size * col_blk_size * blk_nnz` in length. Array summary: `bsr_row_ptr` — `blk_nrows + 1`, `INT_TYPE`, device accessible; `bsr_col_ind` — `blk_nnz`, `INT_TYPE`, device accessible; `bsr_values` — `row_blk_size * col_blk_size * blk_nnz`, `DATA_TYPE`, device accessible.

### set_matrix_property

Sets matrix properties present in the user-provided data in the `sparse::matrix_handle_t` that can serve as optimization hints for library algorithms. Properties are **not verified by the library** but accepted as truth from the user who specified them.

```cpp
namespace oneapi::mkl::sparse {
    void set_matrix_property(oneapi::mkl::sparse::matrix_handle_t spMat,
                             oneapi::mkl::sparse::property property_value);
}
namespace oneapi::mkl::sparse {
    enum class property : char {
        symmetric,
        sorted
    };
}
```

Include: `oneapi/mkl/spblas.hpp`. `symmetric` refers to the matrix being symmetric and the full pattern is present in the data arrays; `sorted` indicates the matrix data is sorted in whatever manner is natural to the particular matrix format (see `sparse::sort_matrix`). `spMat`: handle where the property will be set, created using one of the `set_<sparse_matrix_type>_data` routines. `property_value`: `sparse::property::symmetric` — data in the handle represents a symmetric matrix and has the full pattern and values provided; `sparse::property::sorted` — data in the handle is sorted according to natural state for the given format. Setting properties may affect performance as certain internal optimizations may not need to be done if they are present. Properties may also be set internally by the library when applicable (for instance after a call to `sparse::sort_matrix`).

Formats supported in `set_matrix_property()`: CSR on CPU/GPU, COO on CPU/GPU. NOTE: for matrices in COO format, property is currently **ignored** but may enable enhanced performance for that format in the future.

```cpp
sparse::set_matrix_property(spMat, sparse::property::symmetric);
sparse::set_matrix_property(spMat, sparse::property::sorted);
sparse::optimize_trsv(main_queue, uplo::lower, transpose::nontrans,
                      diag::nonunit, spMat);
sparse::optimize_gemv(main_queue, transpose::nontrans, spMat);
```

An example is installed under `share/doc/mkl/examples/sycl/sparse_blas/source/csr_conjugate_gradient.cpp`.

### init_omatadd_descr / release_omatadd_descr

Initialize / release a `sparse::omatadd_descr_t` object; for use with `sparse::omatadd`.

```cpp
namespace oneapi::mkl::sparse {
    struct omatadd_descr;        /* Forward declaration of opaque omatadd operation descriptor */
    typedef omatadd_descr *omatadd_descr_t;        /* User-facing type for use in omatadd APIs */
    /* Host-side/non-blocking */
    void init_omatadd_descr(sycl::queue &queue, omatadd_descr_t *p_descr);
    /* Asynchronous/non-blocking */
    sycl::event release_omatadd_descr(sycl::queue &queue, omatadd_descr_t descr,
        const std::vector<sycl::event> &dependencies = {});
}
```

`p_descr` is the pointer to the descriptor object, used for allocating it. The descriptor stores input data, operation-specific information, and the user-provided temporary workspace.

### optimize_gemv

Performs internal optimizations for `oneapi::mkl::sparse::gemv` by analyzing the matrix structure; optimized data is then stored in the matrix handle.

```cpp
// Using USM and SYCL buffers:
namespace oneapi::mkl::sparse {
    sycl::event optimize_gemv (sycl::queue &queue, oneapi::mkl::transpose opA,
        oneapi::mkl::sparse::matrix_handle_t A,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `opA`: `oneapi::mkl::transpose::nontrans` (non-transpose, op(A) = A), `oneapi::mkl::transpose::trans` (transpose, op(A) = A^T), `oneapi::mkl::transpose::conjtrans` (conjugate transpose, op(A) = A^H). `A`: handle created using one of the `set_<sparse_matrix_type>_data` routines. `dependencies`: vector of type `std::vector<sycl::event> &`. Return `sycl::event`.

Formats supported in `sparse::optimize_gemv()` for `A`: CSR on CPU/GPU, COO on CPU/GPU, CSC on **GPU**, BSR on **GPU**. The source lists `sparse::optimize_gemv(queue, opA, A)`, `(..., A, {})`, `(..., A, dependencies)` as allowed in the USM case; for `sycl::buffer` the recommended uses are `sparse::optimize_gemv(queue, opA, A)` and `static_cast<void>(sparse::optimize_gemv(queue, opA, A))`.

### optimize_trmv

Performs internal optimizations for `oneapi::mkl::sparse::trmv` by analyzing the matrix structure; optimized data is stored in the matrix handle.

```cpp
// Using USM and SYCL buffers:
namespace oneapi::mkl::sparse {
    sycl::event optimize_trmv (sycl::queue &queue, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::transpose opA, oneapi::mkl::diag diag_val,
        oneapi::mkl::sparse::matrix_handle_t A,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `uplo_val`: `oneapi::mkl::uplo::lower` (lower triangular part processed) or `oneapi::mkl::uplo::upper` (upper triangular part processed). `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `diag_val`: `oneapi::mkl::diag::nonunit` (diagonal elements might not be equal to one) or `oneapi::mkl::diag::unit` (diagonal elements are equal to one). **NOTE: currently the only supported case for `diag_val` is `oneapi::mkl::diag::nonunit`.**

Formats supported in `sparse::optimize_trmv()` for `A`: CSR on CPU/GPU, COO on CPU. Return `sycl::event`.

### optimize_trsv

Performs internal optimizations for `oneapi::mkl::sparse::trsv` by analyzing the provided matrix structure and operation parameters; optimized data is stored in the matrix handle.

```cpp
// Using USM and SYCL buffers:
 namespace oneapi::mkl::sparse {
     sycl::event optimize_trsv (sycl::queue &queue, oneapi::mkl::uplo uplo_val,
         oneapi::mkl::transpose  opA, oneapi::mkl::diag diag_val,
         oneapi::mkl::sparse::matrix_handle_t A,
         const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `diag_val`: `oneapi::mkl::diag::nonunit` (diagonal elements might not be equal to one) or `oneapi::mkl::diag::unit` (diagonal elements are equal to one); no nonunit-only restriction is stated for `optimize_trsv`. Formats supported in `sparse::optimize_trsv()` for `A`: CSR on CPU/GPU, COO on CPU. Return Values (USM Only): `sycl::event`.

### optimize_gemm

Performs internal optimizations for `oneapi::mkl::sparse::gemm` by analyzing the provided matrix structure and operation parameters; optimized data is stored in the matrix handle. There are **two versions**: one analyzes the sparse matrix pattern only, the other takes information about the layout and dimensions of the dense matrices and may allow further optimizations.

```cpp
// Using USM and SYCL buffers:
 namespace oneapi::mkl::sparse {
     sycl::event optimize_gemm (sycl::queue &queue, oneapi::mkl::transpose  opA,
         oneapi::mkl::sparse::matrix_handle_t A,
         const std::vector<sycl::event> &dependencies = {});
     sycl::event optimize_gemm (sycl::queue &queue, oneapi::mkl::layout layout_val,
         oneapi::mkl::transpose  opA, oneapi::mkl::transpose  opX,
         oneapi::mkl::sparse::matrix_handle_t A, const std::int64_t columns,
         const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `layout_val`: storage scheme in memory for the dense matrices; this layout applies to **both** `X` and `Y` dense matrices. `opA`: op() on the input sparse matrix (nontrans/trans/conjtrans). `opX`: op() on the input dense matrix (nontrans/trans/conjtrans); **NOTE (placed after the `opX` description in the source): currently the only supported case for this operation is `oneapi::mkl::transpose::nontrans`.** `columns`: number of columns of matrix `op(X)` and `Y` of the following `sparse::gemm()` routine.

Formats supported in `sparse::optimize_gemm()` for `A`: CSR on CPU/GPU, COO on CPU/GPU. **NOTE: `sparse::optimize_gemm()` would be mainly beneficial to the COO sparse matrix on GPU device.** Return `sycl::event`.

### optimize_trsm

Performs internal optimizations for `oneapi::mkl::sparse::trsm` by analyzing the provided matrix structure and operation parameters; optimized data is stored in the matrix handle. There are **two versions** (pattern only, and pattern plus dense layout/dimensions).

```cpp
// Using USM and SYCL buffers:
 namespace oneapi::mkl::sparse {
     sycl::event optimize_trsm (sycl::queue &queue, oneapi::mkl::uplo uplo_val,
         oneapi::mkl::transpose  opA, oneapi::mkl::diag diag_val,
         oneapi::mkl::sparse::matrix_handle_t A,
         const std::vector<sycl::event> &dependencies = {});
     sycl::event optimize_trsm (sycl::queue &queue, oneapi::mkl::layout layout_val,
         oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose  opA, oneapi::mkl::diag diag_val,
         oneapi::mkl::sparse::matrix_handle_t A, const std::int64_t columns,
         const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `layout_val`: storage scheme in memory for the dense matrices; applies to both `X` and `Y`. `uplo_val`: `uplo::lower` (lower triangular part processed) or `uplo::upper` (upper triangular part processed). `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `diag_val`: `diag::nonunit` (diagonal elements might not be equal to one) or `diag::unit` (diagonal elements are equal to one). `columns`: number of columns of matrix `op(X)` and `Y` of the following `sparse::trsm()` routine.

Formats supported in `sparse::optimize_trsm()`: CSR on CPU/GPU, COO on CPU. Return Values (USM Only): `sycl::event`.

### gemv

Computes a sparse matrix-dense vector product, `y <- alpha * op(A) * x + beta * y`, where `alpha` and `beta` are scalars, `A` is a general sparse matrix of dimensions `num_rows` rows and `num_cols` columns, and `op()` is the matrix modifier below. The dense vectors `x` and `y` are appropriately sized based on the matrix product dimensions of `op(A)`.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void gemv (sycl::queue &queue, oneapi::mkl::transpose opA, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &x,
        const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &y)
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event gemv (sycl::queue &queue, oneapi::mkl::transpose opA, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *x, DATA_TYPE beta,
        DATA_TYPE *y, const std::vector<sycl::event> &dependencies= {})
}
```

Include: `oneapi/mkl/spblas.hpp`. `alpha`, `beta`: scalars. `x`: SYCL buffer or device-accessible USM pointer of size at least equal to the number of columns of the input matrix if `opA = oneapi::mkl::transpose::nontrans` and at least the number of rows of the input matrix otherwise. `y` (output): SYCL buffer or device-accessible USM pointer of size at least equal to the number of rows if `opA = nontrans` and at least the number of columns otherwise; overwritten by the updated vector. Note the USM form takes non-`const` `beta` while the buffer form takes `const DATA_TYPE beta` (verbatim from the source).

Formats supported in `sparse::gemv()`: CSR on CPU/GPU, COO on CPU/GPU, CSC on CPU/GPU, BSR on CPU/GPU. Return Values (USM Only): `sycl::event`. Examples: `share/doc/mkl/examples/sycl/sparse_blas/source/coo_gemv.cpp`, `.../csr_gemv_usm.cpp`.

### gemvdot

Computes a sparse matrix-dense vector product with dot product, defined as `y <- alpha * op(A) * x + beta * y` and `d <- x . y`, where `A` is a general sparse matrix, `alpha` and `beta` are scalars, `x`, `y`, `d` are dense vectors, and `op()` is the matrix modifier.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void gemvdot (sycl::queue &queue, oneapi::mkl::transpose opA, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &x,
        const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &y, sycl::buffer<DATA_TYPE, 1> &d)
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event gemvdot (sycl::queue &queue, oneapi::mkl::transpose opA, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *x, const DATA_TYPE beta,
        DATA_TYPE *y, DATA_TYPE *d, const std::vector<sycl::event> &dependencies={})
}
```

Include: `oneapi/mkl/spblas.hpp`. `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `x`: size at least the number of columns of the input matrix if `opA == nontrans` and at least the number of rows otherwise. `y` (output): size at least the number of rows if `opA == nontrans` and at least the number of columns otherwise; overwritten by the updated vector. `d` (output): overwritten by the dot product of `x` and `y`.

Formats supported in `sparse::gemvdot()`: CSR on CPU/GPU, COO on **CPU only**. Return Values: `sycl::event`. Example: `share/doc/mkl/examples/sycl/sparse_blas/source/csr_gemvdot.cpp`.

### symv

Computes a sparse matrix-dense vector product for a symmetric matrix built from the lower or upper triangular part of the input matrix: `y <- alpha * sym(A) * x + beta * y`, where `alpha` and `beta` are scalars, `A` is a **real-valued square** sparse matrix of dimension `n` rows and columns, `sym(A)` is a matrix modifier which symmetrizes according to the `oneapi::mkl::uplo` value, and `x` and `y` are dense vectors.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void symv (sycl::queue &queue, oneapi::mkl::uplo uplo_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &x,
        const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &y)
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event symv (sycl::queue &queue, oneapi::mkl::uplo uplo_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *x, const DATA_TYPE beta,
        DATA_TYPE *y, const std::vector<sycl::event> &dependencies = {})
}
```

Include: `oneapi/mkl/spblas.hpp`. `uplo_val`: `oneapi::mkl::uplo::lower` (lower part used for symmetric product) or `oneapi::mkl::uplo::upper` (upper part used). For a given matrix decomposition into lower, diagonal and upper parts `A = L + D + U`, `symv` with `uplo::lower` performs the matrix product with the symmetrized lower part and with `uplo::upper` using the symmetrized upper part. `x`: size at least the number of columns `n` of the input matrix. `y` (output): size at least the number of rows `n` of the input matrix; overwritten by the updated vector.

Formats supported in `sparse::symv()` for `A`: **CSR on CPU/GPU** (only). Return Values (USM Only): `sycl::event`. Example: `share/doc/mkl/examples/sycl/sparse_blas/source/csr_symv.cpp`.

### trmv

Computes a sparse matrix-dense vector product over upper or lower triangular parts of the matrix, `y <- alpha * op(A) * x + beta * y`, where `alpha` and `beta` are scalars, `A` is a sparse triangular matrix of size `m` rows by `m` columns, and `x` and `y` are dense vectors of size `m`.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void trmv (sycl::queue &queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose opA,
        oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &x,
        const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &y)
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event trmv(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::transpose opA, oneapi::mkl::diag
        diag_val, const DATA_TYPE alpha, oneapi::mkl::sparse::matrix_handle_t A,
        const DATA_TYPE *x, const DATA_TYPE beta, DATA_TYPE *y,
        const std::vector<sycl::event> &dependencies = {})
}
```

Include: `oneapi/mkl/spblas.hpp`. `uplo_val`: `uplo::lower` or `uplo::upper` (the corresponding triangular matrix part is processed). `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `diag_val`: `diag::nonunit` (diagonal elements might not be equal to one) or `diag::unit` (diagonal elements are equal to one). For the decomposition `A = L + D + U`, the product with `uplo::lower` or `uplo::upper` uses the appropriate part for `diag::nonunit`, or for `diag::unit` performs the appropriate product for `L + I` or `U + I` where `I` is the identity matrix. `x`: size at least the number of columns if `opA = nontrans`, and at least the number of rows otherwise. `y` (output): size at least the number of rows if `opA = nontrans`, and at least the number of columns otherwise; overwritten by the updated vector.

Formats supported in `sparse::trmv()` for `A`: CSR on CPU/GPU, COO on CPU. Return Values (USM Only): `sycl::event`. Example: `share/doc/mkl/examples/sycl/sparse_blas/source/csr_trmv.cpp`.

### trsv

Solves a system of linear equations for a triangular sparse matrix, `op(A) * y = alpha * x`, where `A` is a sparse triangular matrix of size `m` rows by `m` columns. The dense vectors `x` and `y` must be at least of length `m`; `x` is input right hand side data and `y` is the resulting output vector.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void trsv (sycl::queue &queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose opA,
        oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &x,
        sycl::buffer<DATA_TYPE, 1> &y);
    [[deprecated("Use oneapi::mkl::sparse::trsv(queue, uplo_val, opA, diag_val, /* alpha */ 
1.0, ...) instead.")]]
    void trsv (sycl::queue &queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose opA,
        oneapi::mkl::diag diag_val, oneapi::mkl::sparse::matrix_handle_t A,
        sycl::buffer<DATA_TYPE, 1> &x, sycl::buffer<DATA_TYPE, 1> &y);
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::transpose opA, oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *x, DATA_TYPE *y,
        const std::vector<sycl::event> &dependencies = {});
    [[deprecated("Use oneapi::mkl::sparse::trsv(queue, uplo_val, opA, diag_val, /* alpha */ 
1.0, ...) instead.")]]
    sycl::event trsv(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::transpose opA, oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *x, DATA_TYPE *y,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `uplo_val`: `uplo::lower` / `uplo::upper` (forward or backward substitution using respectively `L + D` or `U + D` for `diag::nonunit`, or `L + I` / `U + I` for `diag::unit`). `opA`: nontrans/trans/conjtrans. **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `diag_val`: `diag::nonunit` — diagonal elements are used as provided in the sparse matrix; `diag::unit` — the value of one is substituted for the diagonal elements in the triangular solve algorithm. **NOTE: if `diag::nonunit` is selected, all diagonal values must be present in the sparse matrix sparsity profile and must not be zero valued. This is not necessary for the `diag::unit` case. An exception of type `onemkl::invalid_value()` will be thrown in case this is violated.** `alpha`: scalar. `A`: handle to object containing sparse matrix `A` and other internal data. `x`: size at least the number of columns of the input matrix if `opA = nontrans` and at least the number of rows otherwise; it is the input vector `x`. `y` (output): size at least the number of rows if `opA = nontrans` and at least the number of columns otherwise; the solution of the triangular solve is filled into this array.

Formats supported in `sparse::trsv()` for `A`: CSR on CPU/GPU, COO on CPU. **NOTE: while it can make sense to pass in the same array for both RHS (`x`) and solution (`y`) in order to update the RHS in-place, in-place `sparse::trsv()` is currently not supported.** Return Values (USM Only): `sycl::event`. Examples: `csr_trsv.cpp`, `csr_trsv_usm.cpp`.

Source inconsistency (reproduced verbatim above): the deprecated `sycl::buffer` overload omits `alpha`, while the deprecated USM overload still declares `const DATA_TYPE alpha`, and both carry the same "use /* alpha */ 1.0" deprecation message.

### gemm

Computes a sparse matrix-dense matrix product, `Y <- alpha * op(A) * op(X) + beta * Y`, where `alpha` and `beta` are scalars, `A` is a sparse matrix of size `num_rows` rows by `num_cols` columns, and `op()` is the matrix modifier for `A` and `X`. The dense matrices `X` and `Y` are stored with row-major or column-major layout and have an appropriate number of rows for the matrix product and `columns` number of columns.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void gemm(sycl::queue &queue, oneapi::mkl::layout layout_val, oneapi::mkl::transpose opA,
      oneapi::mkl::transpose opX, const DATA_TYPE alpha, matrix_handle_t A,
      sycl::buffer<DATA_TYPE, 1> &X, const std::int64_t columns, const std::int64_t ldx,
      const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &Y, const std::int64_t ldy);
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event gemm(sycl::queue &queue, oneapi::mkl::layout layout_val,
        oneapi::mkl::transpose opA, oneapi::mkl::transpose opX, const DATA_TYPE alpha,
        matrix_handle_t A, const DATA_TYPE *X, const std::int64_t columns,
        const std::int64_t ldx, const DATA_TYPE beta, DATA_TYPE *Y, const std::int64_t ldy,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `layout_val`: storage scheme in memory for the dense matrices; applies to **both** `X` and `Y`. `opA`: op() on input matrix `A`; `opX`: op() on input matrix `X` (nontrans/trans/conjtrans). **NOTE (placed after the `opX` description in the source): currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** `alpha`, `beta`: scalars. `X`: SYCL buffer or device-accessible USM pointer of size at least `rows*cols` (assuming `opX == nontrans`); `Y` (output): size at least `rows*cols`, overwritten by the updated matrix `Y`.

Shape rules with `opX == nontrans`:

| | `layout=col-major` | `layout=row-major` |
|---|---|---|
| rows (number of rows in X) | `ldx` | if `op(A) = A` (`opA == nontrans`), number of columns in A; if `op(A) = A^T` (`opA == trans`), number of rows in A |
| cols (number of columns in X) | `columns` | `ldx` |
| rows (number of rows in Y) | `ldy` | if `op(A) = A` (`opA == nontrans`), number of rows in A; if `op(A) = A^T` (`opA == trans`), number of columns in A |
| cols (number of columns in Y) | `columns` | `ldy` |

`columns`: number of columns of matrix `Y`. `ldx`: leading dimension of `X`; must be positive, and at least `columns` if `layout_val=row-major` or at least number of columns in `A` if `layout_val=col-major`. `ldy`: leading dimension of `Y`; must be positive, and at least `columns` if `layout_val=row-major` or at least number of rows in `A` if `layout_val=col-major`.

Formats supported in `sparse::gemm()`: CSR on CPU/GPU, COO on CPU/GPU. Return Values (USM Only): `sycl::event`. Examples: `coo_gemm_row_major.cpp`, `csr_gemm_row_major_usm.cpp`, `csr_gemm_col_major.cpp`, `csr_gemm_col_major_usm.cpp`.

### trsm

Solves a system of linear equations with multiple right-hand sides for a triangular sparse matrix, `op(A) * Y = alpha * op(X)`, where `A` is a sparse triangular matrix of size `m` rows by `m` columns. `X` and `Y` are stored with row-major or column-major layout and are of dimensions `op(X)`/`op(Y)` rows by `columns` columns; `X` is input right hand side data and `Y` is the resulting output matrix.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void trsm (sycl::queue &queue, oneapi::mkl::layout layout_val,
        oneapi::mkl::transpose opA, oneapi::mkl::transpose opX, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, sycl::buffer<DATA_TYPE, 1> &X,
        const std::int64_t columns, const std::int64_t ldx, sycl::buffer<DATA_TYPE, 1> &Y,
        const std::int64_t ldy);
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event trsm(sycl::queue &queue, oneapi::mkl::layout layout_val,
        oneapi::mkl::transpose opA, oneapi::mkl::transpose opX, oneapi::mkl::uplo uplo_val,
        oneapi::mkl::diag diag_val, const DATA_TYPE alpha,
        oneapi::mkl::sparse::matrix_handle_t A, const DATA_TYPE *X,
        const std::int64_t columns, const std::int64_t ldx, DATA_TYPE *Y,
        const std::int64_t ldy, const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `layout_val`: storage scheme for the dense matrices; applies to both `X` and `Y`. `opA`: op() on the sparse input matrix; `opX`: op() on the dense input matrix (nontrans/trans/conjtrans). **NOTEs: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`** (stated separately after both the `opA` and the `opX` description). `uplo_val`: `uplo::lower` / `uplo::upper`. `diag_val`: `diag::nonunit` — diagonal elements are used as provided in the sparse matrix; `diag::unit` — the value of one is substituted. **NOTE: if `diag::nonunit` is selected, all diagonal values must be present in the sparse matrix sparsity profile and must not be zero valued. This is not necessary for the `diag::unit` case. An exception of type `onemkl::invalid_value()` will be thrown in case this is violated.** The decomposition `A = L + D + U` gives forward/backward substitution using `L + D` or `U + D` (`diag::nonunit`) or `L + I` / `U + I` (`diag::unit`).

`X`: right hand side dense matrix of dimension `op(X)` by `columns` with leading dimension stride `ldx` based on the provided layout of `X`. For `opX == nontrans`:

| | `layout=col-major` | `layout=row-major` |
|---|---|---|
| nrows of `op(X)` | `ldx` | `m` |
| ncols of `op(X)` | `columns` | `ldx` |

`columns`: number of columns of `op(X)` and `Y`. `ldx`: leading dimension of `X`; must be positive, and larger than or equal to the actual usable row (`layout::row_major`) or column (`layout::col_major`) dimension of `X`. `ldy`: leading dimension of `Y`; must be positive, and at least `columns` if `layout_val=row-major` or at least number of rows in `op(A)` if `layout_val=col-major`. `Y` (output): dense matrix solution of the triangular solve, of dimension `op(Y)` by `columns` with leading dimension stride `ldy`: nrows of `op(Y)` is `ldy` (col-major) or `m` (row-major); ncols of `op(Y)` is `columns` (col-major) or `ldy` (row-major).

Formats supported in `sparse::trsm()` for `A`: CSR on CPU/GPU, COO on CPU. **NOTE: in-place `sparse::trsm()` is currently not supported.** Return Values (USM Only): `sycl::event`. Examples: `csr_trsm.cpp`, `csr_trsm_usm.cpp`.

### omatadd (multi-stage)

Computes general sparse matrix-sparse matrix addition with sparse matrix output, `C = alpha * op(A) + beta * op(B)`, where `A`, `B`, `C` are sparse matrices with mathematically consistent sizes. The output `C` is **not guaranteed to be sorted** on exit; `sparse::sort_matrix()` is provided if that is necessary for subsequent usage. The inputs need not be sorted, but if both inputs are sorted — via `sort_matrix()` or `sparse::property::sorted` — that may significantly improve performance. The routine is broken into four APIs (two lightweight, two computationally expensive).

| Stage | Description |
|---|---|
| `omatadd_buffer_size` | Return size of temporary workspace. |
| `omatadd_analyze` | Count the number of non-zero values (nnzC) in the output sparse matrix. |
| `omatadd_get_nnz` | Return the calculated nnzC count of the output sparse matrix. |
| `omatadd` | Perform union of sparsity pattern and floating point accumulations into user-provided arrays of the output sparse matrix. |

```cpp
namespace oneapi::mkl::sparse {
    enum class omatadd_alg : std::int32_t {
        default_alg = 0
    };
    struct omatadd_descr;        /* Forward declaration of opaque omatadd operation descriptor */
    typedef omatadd_descr *omatadd_descr_t;        /* User-facing type for use in omatadd APIs */
    /* Host-side/non-blocking */
    void init_omatadd_descr(sycl::queue &queue, omatadd_descr_t *p_descr);
    /* Asynchronous/non-blocking */
    sycl::event release_omatadd_descr(sycl::queue &queue, omatadd_descr_t descr,
        const std::vector<sycl::event> &dependencies = {});
    /* Combined USM/sycl::buffer API, host-side/non-blocking */
    void omatadd_buffer_size(sycl::queue &queue, transpose opA, transpose opB,
        matrix_handle_t A,           /* oneMKL Input sparse matrix handle */
        matrix_handle_t B,           /* oneMKL Input sparse matrix handle */
        matrix_handle_t C,          /* oneMKL Output sparse matrix handle */
        omatadd_alg alg, omatadd_descr_t descr,            /* omatadd operation descriptor */
        std::int64_t &sizeTempWorkspace);  /* Size of temporary workspace */
    /* sycl::buffer API, asynchronous/non-blocking */
    void omatadd_analyze(sycl::queue &queue, transpose opA, transpose opB,
        matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, omatadd_alg alg,
        omatadd_descr_t descr,
        sycl::buffer<std::uint8_t, 1> *tempWorkspace); /* Temporary workspace */
    /* USM API, asynchronous/non-blocking */
    sycl::event omatadd_analyze(sycl::queue &queue, transpose opA, transpose opB,
        matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, omatadd_alg alg,
        omatadd_descr_t descr, void *tempWorkspace,                    /* Temporary workspace */
        const std::vector<sycl::event> &dependencies = {});
    /* Combined USM/sycl::buffer API, synchronous/blocking */
    void omatadd_get_nnz(sycl::queue &queue, transpose opA, transpose opB,
        matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, omatadd_alg alg,
        omatadd_descr_t descr,
        std::int64_t &nnzC,            /* Returned non-zero count of C matrix */
        const std::vector<sycl::event> &dependencies = {});
    /* Combined USM/sycl::buffer API, asynchronous/non-blocking */
    sycl::event omatadd(sycl::queue &queue, transpose opA, transpose opB,
        const DATA_TYPE alpha,                             /* A-scaling factor */
        matrix_handle_t A, const DATA_TYPE beta,                              /* B-scaling factor */
        matrix_handle_t B, matrix_handle_t C,                               /* User arrays filled */
        omatadd_alg alg, omatadd_descr_t descr,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`.

Workflow: (1) **Before** — use `set_<xyz>_data` with dummy arguments for row, column, and data arrays to set the sparse matrix format and the output 0-/1-based indexing (rows/columns may be set to zero or be mathematically consistent at this stage); create the `omatadd_descr_t` using `init_omatadd_descr` and decide on an algorithm through `omatadd_alg`; **do not change this enum** between calls with a given set of input arguments and descriptor. (2) `omatadd_buffer_size` — non-blocking host-side API that does not access the input matrix arrays; get the workspace size, then allocate the workspace. (3) `omatadd_analyze` — non-blocking asynchronous API that accesses and analyzes the sparsity patterns; the workspace array is internally stored in the descriptor, so do not modify or free it for the duration of its use or for the lifetime of the descriptor. (4) `omatadd_get_nnz` — blocking API to get the number of non-zeros in `C`; allocate the row, column, and data arrays of `C`; call `set_<xyz>_data` again with the valid newly allocated arrays (the output 0-/1-based indexing must not be changed, and the number of rows and columns must be mathematically consistent with the input sizes). (5) `omatadd` — non-blocking asynchronous call performing the union of the sparsity pattern and floating point accumulations. (6) **After** — release the descriptor using `release_omatadd_descr`; **reusing the descriptor for another addition operation is currently undefined behavior**, but may be enabled in a future oneMKL release. Release the temporary workspace, or reuse it once the descriptor has been released. If sorted output is needed, call `sparse::sort_matrix()`.

- `opA`, `opB`: op() on input matrices as `oneapi::mkl::transpose` enums; **all combinations of opA and opB are supported**.
- `alpha`, `beta`: scalars to scale `op(A)` and `op(B)` respectively.
- `A`, `B`: input handles; need not be sorted, but performance may significantly benefit from sorting. **The order of `A` and `B` must not be changed across API calls.** Formats supported for `A` and `B`: CSR on CPU/GPU.
- `alg`: choice of algorithm; for a given set of inputs and descriptor it must not be changed across API calls. Currently only `omatadd_alg::default_alg = 0` is available.
- `descr`/`p_descr`: the descriptor object and the pointer used for allocating it, created/destroyed with `init_omatadd_descr`/`release_omatadd_descr`.
- `tempWorkspace`: SYCL-aware container (`sycl::buffer` or device-accessible USM pointer) of `sizeTempWorkspace` bytes; must remain valid through the full multi-stage calls and must not be modified between API calls or for the lifetime of `descr`. For `sycl::buffer` inputs it is `sycl::buffer<std::uint8_t> *`; for USM it is a device-accessible `void *` (USM device recommended for best performance, shared/host supported).
- `dependencies`: events the routine being called depends on to complete first, if any.
- `C` (input/output): the output handle; the 0- or 1-based indexing parameter set in the `C` handle is an input. Arrays are user-allocated/user-owned, stored in the handle with `set_<xyz>_data`, and filled by the library; output arrays are not guaranteed to be sorted. Formats supported for `C`: CSR on CPU/GPU. **NOTE: aliasing the `C` matrix handle or arrays with either of the input `A` and `B` handles or their arrays (therefore attempting an "in-place" addition operation) is undefined behavior.**
- Output `sizeTempWorkspace`: `std::int64_t`, size in bytes of `tempWorkspace`, obtained from `omatadd_buffer_size`. Output `nnzC`: `std::int64_t`, format specific number of non-zeros in `C`, obtained from `omatadd_get_nnz`.
- Return `sycl::event` where applicable: in USM APIs it must be carried over and added as a dependency for the completion of subsequent stages.
- Note: unlike `sparse::matmat()`, the `omatadd` routines currently do **not** support addition involving only the sparsity patterns without any floating point values.
- Examples: `csr_omatadd.cpp`, `csr_omatadd_usm.cpp`.

### init_matmat_descr

Allocates and initializes a `oneapi::mkl::sparse::matmat_descr_t` object to default values, otherwise it throws an exception.

```cpp
namespace oneapi::mkl::sparse {
    void init_matmat_descr ( oneapi::mkl::sparse::matmat_descr_t *p_descr );
}
```

Include: `oneapi/mkl/spblas.hpp`. `p_descr` (input): address of an uninitialized (`nullptr`) matmat descriptor object to be initialized in this routine. `p_descr` (output): on return the address is updated to point to a newly allocated and initialized `sparse::matmat_descr_t` object ready for use in matmat routines.

### set_matmat_data

Sets the appropriate `oneapi::mkl::sparse::matrix_view_descr` and `oneapi::mkl::transpose` values in the `oneapi::mkl::sparse::matmat_descr_t` object reflecting the `sparse::matmat` operation to be performed (`C = op(A) * op(B)`).

```cpp
namespace oneapi::mkl::sparse {
    enum class matrix_view_descr : std::int32_t {
        general
    };
    void set_matmat_descr (sparse::matmat_descr_t    descr,
        sparse::matrix_view_descr viewA, transpose                 opA,
        sparse::matrix_view_descr viewB, transpose                 opB,
        sparse::matrix_view_descr viewC);
}
```

Include: `oneapi/mkl/spblas.hpp`. `viewA`, `viewB`, `viewC`: how the `A`, `B`, `C` matrix representations are to be viewed; **currently only the general type is supported**, where `general` view assumes all data is populated in the handle for both lower, diagonal and upper portions of the matrix. `opA`, `opB`: op() on the input matrices (nontrans/trans/conjtrans). **NOTE: currently the only supported case for operation is `oneapi::mkl::transpose::nontrans`.** Output `descr`: the descriptor defining the operation to be performed by `sparse::matmat`.

Source inconsistency: the routine is titled **`set_matmat_data`** but its Syntax block names the function **`set_matmat_descr`** (reproduced verbatim above); the usage examples call `sparse::set_matmat_data(descr, viewA, opA, viewB, opB, viewC)`.

### get_matmat_data

Queries the `matrix_view_descr` and `transpose` values in the `matmat_descr_t` object reflecting the `sparse::matmat` operation to be performed.

```cpp
namespace oneapi::mkl::sparse {
    void get_matmat_descr (sparse::matmat_descr_t    descr,
        sparse::matrix_view_descr &viewA, transpose                 &opA,
        sparse::matrix_view_descr &viewB, transpose                 &opB,
        sparse::matrix_view_descr &viewC);
}
```

Include: `oneapi/mkl/spblas.hpp`. Input `descr`: the descriptor object. Output `viewA`, `viewB`, `viewC`: how the `A`, `B`, `C` matrix representations are viewed (**currently only `general` is supported**); output `opA`, `opB`: op() on the input matrices (**currently only `oneapi::mkl::transpose::nontrans` is supported**).

Source inconsistencies: the routine is titled **`get_matmat_data`** but its Syntax block names the function **`get_matmat_descr`**; the usage example calls `sparse::get_matmat_data(descr, viewA, opA, viewV, opB, viewC);`, where `viewV` appears nowhere in the signature, and shows `init_matmat_descr(descr)`/`release_matmat_descr(descr)` without the address-of operator (both verbatim from the source).

### release_matmat_descr

Releases a `oneapi::mkl::sparse::matmat_descr_t` object and sets it to NULL.

```cpp
namespace oneapi::mkl::sparse {
    void release_matmat_descr ( oneapi::mkl::sparse::matmat_descr_t *p_descr );
}
```

Include: `oneapi/mkl/spblas.hpp`. `p_descr` (input): address of an initialized matmat descriptor object (previously allocated in `sparse::init_matmat_descr()`) to be deallocated in this routine. `p_descr` (output): on return the address is updated to point to a null `sparse::matmat_descr_t`; the matmat descriptor object that was previously pointed to will have been deallocated.

### matmat

Computes a sparse matrix-sparse matrix product, `C = op(A) * op(B)`, where `A`, `B`, `C` are appropriately-sized sparse matrices stored in `matrix_handle_t` objects (currently only CSR is supported). The output `C` is not guaranteed to be sorted on exit; `sparse::sort_matrix()` is provided if that is necessary. `B` must be in a sorted state prior to the call; `A` does not require a sorted state, but performance can benefit from it.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
   void matmat(sycl::queue &queue, sparse::matrix_handle_t A, sparse::matrix_handle_t B,
                sparse::matrix_handle_t C, sparse::matmat_request req,
                sparse::matmat_descr_t descr, sycl::buffer<std::int64_t, 1> *sizeTempBuffer,
                sycl::buffer<std::uint8_t, 1> *tempBuffer);
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
   sycl::event matmat(sycl::queue &queue, sparse::matrix_handle_t A,
                       sparse::matrix_handle_t B, sparse::matrix_handle_t C,
                       sparse::matmat_request req, sparse::matmat_descr_t descr,
                       std::int64_t *sizeTempBuffer, void *tempBuffer,
                       const std::vector<sycl::event> &dependencies);
}
namespace oneapi::mkl::sparse {
    enum class matmat_request : std::int32_t {
        get_work_estimation_buf_size, work_estimation, get_compute_structure_buf_size,
        compute_structure, finalize_structure, get_compute_buf_size, compute, get_nnz,
        finalize
    };
}
```

Include: `oneapi/mkl/spblas.hpp`.

- `A`: matrix handle for the first matrix; does not need to be in a sorted state (source spells this "`sparse::matmt()`" in one place), but performance may benefit from it. Formats supported for `A`: CSR on CPU/GPU.
- `B`: matrix handle for the second matrix; **currently must be in a sorted state** as input to `sparse::matmat()` — use `sparse::sort_matrix()` to ensure the sorted property. NOTE: the plan is to remove this sorted restriction in a future release. Formats supported for `B`: CSR on CPU/GPU.
- `C` (output): the output matrix handle; its sparse matrix format arrays are allocated by the user and put into the handle using a `set_<xyz>_data` routine, and filled by the library as part of the matmat operation. The output may not be sorted, so `sparse::sort_matrix()` is provided for convenience. Formats supported for `C`: CSR on CPU/GPU.
- `req`: the `matmat_request` stage in the multi-stage algorithm.
- `descr`: the `matmat_descr_t` describing the operation to be executed; manipulated using `sparse::init_matmat_descr`, `sparse::set_matmat_data` and `sparse::release_matmat_descr`.
- `sizeTempBuffer`: a SYCL aware container (`sycl::buffer` or **host-accessible** USM pointer) of the length of one `std::int64_t` representing the size in bytes of `tempBuffer`. For the `matmat_request` stages with the `get_xyz` naming convention the value is set by the library to inform the user how much memory to allocate; in the other `work_estimation` and `compute`/`compute_structure` stages it is passed in along with `tempBuffer`, informing the library how much space was provided in bytes. For `sycl::buffer` inputs it is `sycl::buffer<std::int64_t>`; for USM inputs it must be host-accessible and of `std::int64_t *` type (USM host generally gives better performance than USM shared; both are supported).

| `sizeTempBuffer` filled in stage | size (in bytes) of which array(s) | USM Memory Type |
|---|---|---|
| `get_work_estimation_buf_size` | `temp_buffer1` in `work_estimation` | host accessible (USM host or USM shared) |
| `get_compute_buf_size` or `get_compute_structure_buf_size` | `temp_buffer2` in `compute` or `compute_structure` | host accessible (USM host or USM shared) |
| `get_nnz` | `C` colind/values arrays for `finalize`\|`finalize_structure` | host accessible (USM host or USM shared) |

- `tempBuffer`: a SYCL-aware container (`sycl::buffer` or device-accessible USM pointer) of `sizeTempBuffer` bytes used as a temporary workspace. There are **two stages where separate workspaces must be passed in** (`work_estimation` and `compute`/`compute_structure`); they must remain valid through the full multi-stage algorithm as both may be used until the last `finalize`/`finalize_structure` request is completed. For `sycl::buffer` inputs it is `sycl::buffer<std::uint8_t>`; for USM it must be device-accessible and is passed as `void *`.

| `tempBuffer` array provided in stage | size of array set in stage | USM Memory Type |
|---|---|---|
| `temp_buffer1` — `work_estimation` | `get_work_estimation_buf_size` | device accessible (USM device or USM shared or USM host) |
| `temp_buffer2` — `compute` or `compute_structure` | `get_compute_buf_size` or `get_compute_structure_buf_size` | device accessible (USM device or USM shared or USM host) |

- `dependencies` (USM APIs only): events the current stage depends on. Return Values (USM Only): `sycl::event`.

Computational stages: `work_estimation` — initial estimation of work and load balancing (upper bound estimate on size of the C matrix data); `compute`/`compute_structure` — internal products for computing the C matrix including the calculation of the size of C matrix data and filling the row pointer array for C; `finalize`/`finalize_structure` — any remaining internal products and accumulation and transfer into final C matrix arrays. `compute_structure` and `finalize_structure` and their helpers can be used if the final result desired is purely the sparsity pattern of `C`.

Common workflow: (1) before the stages, allocate the `C` matrix row pointer array and input it into the `C` matrix handle with dummy arguments for column and data arrays; (2) `get_work_estimation_buf_size` → allocate the work estimation temporary workspace → `work_estimation`; (3) `get_compute_buf_size` → allocate the compute temporary workspace → `compute`; (4) `get_nnz` → allocate the `C` column and data arrays and input into the C matrix handle → `finalize`; (5) after the stages, release or reuse the matmat descriptor, release any temporary workspace arrays, and release or use the `C` matrix handle.

**Simplified workflow:** there is a simplifying option to skip the `get_xxx_buf_size` queries for the `work_estimation` and `compute`/`compute_structure` stages and pass in null pointers for the `sizeTempBuffer` and `tempBuffer` arguments in the API for those stages. In this case the library handles the allocation and memory management of the temporary arrays themselves, and the internally allocated temporary arrays live until the `C` matrix handle is destroyed. However, you are **always** expected to query the size of `C` matrix data and allocate the `C` matrix arrays yourself. Examples: `csr_matmat.cpp`, `csr_matmat_simplified.cpp`, `csr_matmat_structure_only.cpp`, `csr_matmat_usm.cpp`, `csr_matmat_simplified_usm.cpp`, `csr_matmat_structure_only_usm.cpp`.

### matmatd

Computes a sparse matrix-sparse matrix product with a dense matrix result, `C <- alpha * op(A) * op(B) + beta * C`, where `alpha` and `beta` are scalars, `A` and `B` are sparse matrices, `C` is a dense matrix of size `c_nrows` rows by `c_ncols` columns, and `op()` is the matrix modifier for `A` and `B`. The dense matrix object `C` is stored with row-major or column-major layout.

```cpp
// Using SYCL buffers:
namespace oneapi::mkl::sparse {
    void matmatd(sycl::queue &queue, oneapi::mkl::layout c_layout, oneapi::mkl::transpose opA,
      oneapi::mkl::transpose opB, const DATA_TYPE alpha, matrix_handle_t A, matrix_handle_t B,
      const DATA_TYPE beta, sycl::buffer<DATA_TYPE, 1> &C
      const std::int64_t c_nrows, const std::int64_t c_ncols, const std::int64_t ldc);
}
// Using USM pointers:
namespace oneapi::mkl::sparse {
    sycl::event matmatd(sycl::queue &queue, oneapi::mkl::layout c_layout,
        oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, const DATA_TYPE alpha,
        matrix_handle_t A, matrix_handle_t B, const DATA_TYPE beta, DATA_TYPE *C,
        const std::int64_t c_nrows, const std::int64_t c_ncols, const std::int64_t ldc,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `c_layout`: storage scheme in memory for the dense matrix `C`. `opA`: op() on input matrix `A`; `opB`: op() on input matrix `B` (nontrans/trans/conjtrans). No nontrans-only restriction is stated for `matmatd`. `alpha`, `beta`: scalars. `A`, `B`: handles created using one of the `set_<sparse_matrix_type>_data` routines; formats supported for `A` and `B`: CSR on CPU/GPU.

`C` (output): SYCL buffer or device-accessible USM pointer of size at least `rows*cols`, where (for `layout=col_major`) rows = `ldc` and cols = `c_nrows`, and (for `layout=row_major`) rows = `c_ncols` and cols = `ldc`; overwritten by the updated matrix `C`. `c_nrows`/`c_ncols`: number of rows / columns of matrix `C`. `ldc`: leading dimension of matrix `C`; must be positive, and at least `c_ncols` if `c_layout=row_major` or at least `c_nrows` if `c_layout=col_major`. Return Values (USM Only): `sycl::event`.

Source artifact: the buffer Syntax block as extracted is missing a comma after `sycl::buffer<DATA_TYPE, 1> &C` (reproduced verbatim above). Examples: `csr_matmatd.cpp`, `csr_matmatd_usm.cpp`.

### omatcopy

Performs an out-of-place copy/transpose of user data in a matrix handle into user data in a new handle, **without changing the sparse matrix format**. A transpose operation and/or a change in array indexing may be performed in the copy; only out-of-place copy/transpose is supported.

```cpp
// Using USM and SYCL buffers:
namespace oneapi::mkl::sparse {
    sycl::event omatcopy (sycl::queue &queue, oneapi::mkl::transpose transpose_val,
        oneapi::mkl::sparse::matrix_handle_t spMat_in,
        oneapi::mkl::sparse::matrix_handle_t spMat_out,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `transpose_val`: `oneapi::mkl::transpose::nontrans` (op(A) = A), `oneapi::mkl::transpose::trans` (op(A) = A^T), `oneapi::mkl::transpose::conjtrans` (op(A) = A^H). `spMat_in`: handle to the input sparse matrix, created using one of the `set_<sparse_matrix_type>_data` routines. `spMat_out`: handle to the output sparse matrix, created the same way; its dimensions/array lengths must be the same as the input handle for `nontrans`, while for `trans` and `conjtrans` the number of rows and columns of the output must be the number of columns and rows of the input, respectively; for a transpose operation the number of non-zeros remains unchanged. `dependencies`: events the routine depends on.

Formats supported in `sparse::omatcopy()` for `spMat_in` and for `spMat_out`: CSR on CPU/GPU, COO on CPU/GPU. **NOTE: `spMat_out` must be created with the same `<sparse_matrix_type>` as `spMat_in`** (stated in both directions). **NOTE: this routine modifies arrays provided to oneMKL through the `set_<sparse_matrix_type>_data` routines; arrays of only the output matrix handle are modified, and those of the input handle are not.** Return Values: `sycl::event`.

### omatconvert (multi-stage)

Performs an out-of-place conversion of user data in a matrix handle into user data in a new handle **in a different sparse matrix format**. The only formats supported are CSR and COO, and the input and output formats **must be different** (use `sparse::omatcopy` to copy a matrix, including potentially changing the `index_base` or copying the transpose). Users are responsible for most memory allocation, including temporary workspace and the output matrix data.

| Stage | Description |
|---|---|
| `omatconvert_buffer_size` | Return the size of the temporary workspace. |
| `omatconvert_analyze` | Count the number of non-zero values (nnzOut) in the output sparse matrix. |
| `omatconvert_get_nnz` | Return the calculated nnzOut count of the output sparse matrix. |
| `omatconvert` | Perform the conversion from the input to the output matrix. |

```cpp
namespace oneapi::mkl::sparse {
   enum class omataconvert_alg : std::int32_t {
      default_alg = 0                                 /* More may be added in the future */
   };
   struct omatconvert_descr;     /* Forward declaration of opaque omatconvert operation descriptor */
   typedef omatconvert_descr *omatconvert_descr_t; /* User-facing type for use in omatconvert APIs */
   /* Host-side/non-blocking */
   void init_omatconvert_descr(sycl::queue &queue, omatconvert_descr_t *p_descr);
   /* Asynchronous/non-blocking */
   sycl::event release_omatconvert_descr(sycl::queue &queue, omatconvert_descr_t descr,
       const std::vector<sycl::event> &dependencies = {});
   /* Combined USM/sycl::buffer API, host-side/non-blocking  */
   void omatconvert_buffer_size(sycl::queue &queue, matrix_handle_t spMat_in,
      matrix_handle_t spMat_out, omatconvert_alg alg, omatconvert_descr_t descr,
      std::int64_t &sizeTempWorkspace);
   /* sycl::buffer API, asynchronous/non-blocking */
   sycl::event omatconvert_analyze(sycl::queue &queue, matrix_handle_t spMat_in,
      matrix_handle_t spMat_out, omatconvert_alg alg, omatconvert_descr_t descr,
      void *tempWorkspace, const std::vector<sycl::event> &dependencies = {});
   /* USM API, asynchronous/non-blocking */
   void omatconvert_analyze(sycl::queue &queue, matrix_handle_t spMat_in,
      matrix_handle_t spMat_out, omatconvert_alg alg, omatconvert_descr_t descr,
      sycl::buffer<std::uint8_t, 1> *tempWorkspace);
   /* Combined USM/sycl::buffer API, synchronous/blocking */
   void omatconvert_get_nnz(sycl::queue &queue, matrix_handle_t spMat_in,
      matrix_handle_t spMat_out, omatconvert_alg alg, omatconvert_descr_t descr,
      std::int64_t &nnzOut, const std::vector<sycl::event> &dependencies = {});
   /* Combined USM/sycl::buffer API, asynchronous/non-blocking */
   sycl::event omatconvert(sycl::queue &queue, matrix_handle_t spMat_in,
      matrix_handle_t spMat_out, omatconvert_alg alg, omatconvert_descr_t descr,
      const std::vector<sycl::event> &dependencies = {});
 }
```

Include: `oneapi/mkl/spblas.hpp`.

Workflow: (1) **Before the stages** — create a handle for the output matrix using `oneapi::mkl::sparse::set_<sparse_matrix_type>_data`, where `<sparse_matrix_type>` is the desired output format; the indexing for the output matrix can differ from that of the input; the number of non-zero elements can be set to 0 and dummy arguments used for the matrix representation array (e.g. for CSR: rowptr, colind and values); allocate and initialize an `omatconvert_descr_t` using `init_omatconvert_descr` and decide on an algorithm through `omatconvert_alg`. **The algorithm must not be changed within the series of omatconvert API calls.** (2) `omatconvert_buffer_size` — non-blocking API that may access the input matrix arrays; get the workspace size and allocate the workspace. (3) `omatconvert_analyze` — non-blocking asynchronous API that accesses/analyzes the sparsity pattern of the input matrix and the already known parameters of the output matrix; the workspace is internally stored in the descriptor, so do not modify or free it for the duration of its use, which is governed by the lifetime of the descriptor. (4) `omatconvert_get_nnz` — blocking API returning the non-zero count to the host; allocate the appropriately sized output arrays; call `set_<sparse_matrix_type>_data` again with the valid newly allocated row, column, and data arrays (the output 0-/1-based indexing must not be changed, and dimensions must be mathematically consistent with the input matrix sizes). (5) `omatconvert` — non-blocking asynchronous API performing the conversion. (6) **After** — release or reuse the descriptor and the temporary workspace arrays.

Conversion algorithms and supported configurations:

| Conversion type | Supported algorithms | Algorithm description |
|---|---|---|
| CSR -> COO | `omatconvert_alg::default` | Default algorithm for the conversion from CSR to COO |
| COO -> CSR | `omatconvert_alg::default` | Default algorithm for the conversion from COO to CSR. Duplicate elements are condensed. |

- `spMat_in`, `spMat_out`: handles created using one of the `set_<sparse_matrix_type>_data` routines; the dimensions of the sparse matrices in the input and output handles must be the same, and it is required to have allocated the data for the arrays of the output handle. Formats supported for both: CSR on CPU/GPU, COO on CPU/GPU. **NOTE: the sparse format of `spMat_out` must be different than that of `spMat_in`. If they are the same, then use `sparse::omatcopy` instead.**
- `alg`: the `omatconvert_alg` enum specifying the algorithm to use for the conversion.
- `descr`: the descriptor storing input data, operation-specific information, and the user-provided temporary workspace; created/destroyed with `sparse::init_omatconvert_descr` and `sparse::release_omatconvert_descr`.
- `tempWorkspace`: a SYCL-aware container (`sycl::buffer` or device-accessible USM pointer) of `sizeTempWorkspace` bytes; must remain valid through the full multi-stage calls and should not be modified in between calls. For `sycl::buffer` inputs it is `sycl::buffer<std::uint8_t> *`; for USM it is a device-accessible `void *` (USM device recommended, shared/host supported).
- Output `sizeTempWorkspace`: `std::int64_t`, size in bytes to allocate for `omatconvert_analyze` and `omatconvert`, obtained from `omatconvert_buffer_size`. Output `nnzOut`: `std::int64_t`, format specific number of non-zeros in the output matrix, obtained from `omatconvert_get_nnz`. Output `spMat_out`: on return the data for the sparse matrix is filled according to the requested `<sparse_matrix_type>`.
- Return Values: `sycl::event` (can be waited upon or added as a dependency for the completion of the routine).
- Source inconsistency (reproduced verbatim): the enum is declared as `enum class omataconvert_alg` (an extra "a"), while the text heading, the API parameters, and the algorithm tables refer to `omatconvert_alg`, `omatconvert_alg::default`. The declared enumerator is `default_alg`, so the tables' `omatconvert_alg::default` matches no declared enumerator (use `omatconvert_alg::default_alg`, spelling of the enum aside).
- Examples: `csr2coo_omatconvert.cpp`, `coo2csr_omatconvert_usm.cpp`.

### sort_matrix

Performs in-place sorting of user-provided sparse matrix arrays stored in a given sparse matrix handle.

```cpp
// Using USM and SYCL buffers:
namespace oneapi::mkl::sparse {
    sycl::event sort_matrix (sycl::queue &queue,
        oneapi::mkl::sparse::matrix_handle_t spMat,
        const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `spMat`: handle created using one of the `set_<sparse_matrix_type>_data` routines. `dependencies`: events the routine depends on. Formats supported in `sparse::sort_matrix()` for `spMat`: CSR on CPU/GPU. **NOTE: this routine modifies user-arrays provided to oneMKL through the `set_<sparse_matrix_type>_data` routines.** Return Values: `sycl::event`.

### update_diagonal_values

Changes the values of the entries on the main diagonal of a sparse matrix; it also updates any data structures related to optimizations that have already been done (such as with `oneapi::mkl::sparse::optimize_trsv` or `oneapi::mkl::sparse::optimize_gemv`). The existing sparse structure must already have **all diagonal entries included** in the sparsity pattern.

```cpp
// Using SYCL buffers:
 namespace oneapi::mkl::sparse {
     void update_diagonal_values(sycl::queue &queue,
         oneapi::mkl::sparse::matrix_handle_t spMat,
         sycl::buffer<DATA_TYPE, 1> &new_diag_values);
}
// Using USM pointers:
 namespace oneapi::mkl::sparse {
     sycl::event update_diagonal_values(sycl::queue &queue,
         oneapi::mkl::sparse::matrix_handle_t spMat, std::int64_t length,
         const DATA_TYPE* new_diag_values,
         const std::vector<sycl::event> &dependencies = {});
}
```

Include: `oneapi/mkl/spblas.hpp`. `spMat`: handle created using one of the `set_<sparse_matrix_type>_data` routines. `new_diag_values`: SYCL buffer or device-accessible USM pointer of size at least equal to the smaller of the number of rows and number of columns of the input matrix. `length` (USM only): length of the input vector `new_diag_values`; must be at least equal to the smaller of the number of rows and number of columns of the input matrix. `dependencies`: events the routine depends on.

Formats supported in `sparse::update_diagonal_values()` for `spMat`: **CSR on GPU** (only). Return Values (USM Only): `sycl::event`.

## Formulas

`op(A)` is defined identically in several routines: `op(A) = A` for `oneapi::mkl::transpose::nontrans`; `op(A) = A^T` for `oneapi::mkl::transpose::trans`; `op(A) = A^H` for `oneapi::mkl::transpose::conjtrans`.

### Execution- and helper-routine definitions

- `gemv (p342)`: `y <- alpha * op(A) * x + beta * y`
- `gemv (p342)`: `op(A) = { A, oneapi::mkl::transpose::nontrans ; A^T, oneapi::mkl::transpose::trans ; A^H, oneapi::mkl::transpose::conjtrans }`
- `gemvdot (p344)`: `y <- alpha * op(A) * x + beta * y`
- `gemvdot (p344)`: `d <- x . y` (dot product of `x` and `y`)
- `gemvdot (p344)`: `op(A) = { A, oneapi::mkl::transpose::nontrans ; A^T, oneapi::mkl::transpose::trans ; A^H, oneapi::mkl::transpose::conjtrans }`
- `symv (p346)`: `y <- alpha * sym(A) * x + beta * y`
- `trmv (p349)`: `y <- alpha * op(A) * x + beta * y`
- `trmv (p349)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `trsv (p351)`: `op(A) * y = alpha * x`
- `trsv (p351)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `gemm (p355)`: `Y <- alpha * op(A) * op(X) + beta * Y`
- `gemm (p355)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `trsm (p358)`: `op(A) * Y = alpha * op(X)`
- `trsm (p358)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `omatadd (p362)`: `C = alpha * op(A) + beta * op(B)`
- `omatadd (p363)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `matmat (p373)`: `C = op(A) * op(B)`
- `matmat (p374)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`
- `matmatd (p379)`: `C <- alpha * op(A) * op(B) + beta * C`
- `matmatd (p380)`: `op(A) = { A, nontrans ; A^T, trans ; A^H, conjtrans }`

### Storage-format illustrative matrices (from the format examples)

- `CSR Case 1 (p302)`, sorted square, zero-based indexing, `nrows=3, ncols=3, nnz=5, index=0, row_ptr=[0,2,4,5], col_ind=[0,2,1,2,0], values=[1.0,2.0,-1.0,4.0,3.0]`: `A = [[1.0,0.0,2.0],[0.0,-1.0,4.0],[3.0,0.0,0.0]]`.
- `CSR Case 2 (p303)`, sorted rectangular, one-based indexing, empty row, `nrows=4, ncols=5, nnz=7, index=1, row_ptr=[1,3,6,6,8], col_ind=[1,3,2,3,5,1,4], values=[1.0,2.0,-1.0,4.0,1.0,3.0,1.0]`: `A = [[1.0,0.0,2.0,0.0,0.0],[0.0,-1.0,4.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0],[3.0,0.0,0.0,1.0,0.0]]`.
- `CSR Case 3 (p304)`, unsorted rectangular, zero-based indexing, `nrows=4, ncols=5, nnz=10, index=0, row_ptr=[0,2,5,9,10], col_ind=[0,2,4,1,2,1,2,0,3,0], values=[1.0,2.0,1.0,-1.0,4.0,2.0,3.0,1.0,4.0,3.0]`: `A = [[1.0,0.0,2.0,0.0,0.0],[0.0,-1.0,4.0,0.0,1.0],[1.0,2.0,3.0,4.0,0.0],[3.0,0.0,0.0,0.0,0.0]]`.
- `COO Case 1 (p306)`, unsorted rectangular, zero-based indexing, `nrows=4, ncols=5, nnz=10, index=0, row_ind=[2,1,0,1,2,3,0,1,2,2], col_ind=[1,4,2,1,0,0,0,2,3,2], values=[2.0,1.0,2.0,-1.0,1.0,3.0,1.0,4.0,4.0,3.0]`: the dense matrix equals the CSR Case 3 matrix above.
- `COO Case 2 (p307)`, sorted rectangular, one-based indexing, same dense matrix as CSR Case 2, in two fully sorted forms. CSR-style sorted: `row_ind=[1,1,2,2,2,4,4], col_ind=[1,3,2,3,5,1,4], values=[1.0,2.0,-1.0,4.0,1.0,3.0,1.0]`. CSC-style sorted: `row_ind=[1,4,2,1,2,4,2], col_ind=[1,1,2,3,3,4,5], values=[1.0,3.0,-1.0,2.0,4.0,1.0,1.0]`.
- `COO Case 3 (p308)`, partially sorted rectangular, one-based indexing, same dense matrix as CSR Case 2. CSR-style unsorted/partially sorted: `row_ind=[1,1,2,2,2,4,4], col_ind=[1,3,5,2,3,4,1], values=[1.0,2.0,1.0,-1.0,4.0,1.0,3.0]`. CSC-style unsorted/partially sorted: `row_ind=[4,1,2,1,2,4,2], col_ind=[1,1,2,3,3,4,5], values=[3.0,1.0,-1.0,2.0,4.0,1.0,1.0]`.
- `CSC Case 1 (p310)`, sorted square, zero-based indexing, `nrows=3, ncols=3, nnz=5, index=0, col_ptr=[0,2,3,5], row_ind=[0,2,1,0,1], values=[1.0,3.0,-1.0,2.0,4.0]`: `A = [[1.0,0.0,2.0],[0.0,-1.0,4.0],[3.0,0.0,0.0]]`.
- `CSC Case 2 (p310)`, sorted rectangular, one-based indexing, empty row, `nrows=4, ncols=5, nnz=7, index=1, col_ptr=[1,3,4,6,7,8], row_ind=[1,4,2,1,2,4,2], values=[1.0,2.0,-1.0,2.0,4.0,1.0,1.0]`: `A = [[1.0,0.0,2.0,0.0,0.0],[0.0,-1.0,4.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0],[3.0,0.0,0.0,1.0,0.0]]`.
- `CSC Case 3 (p311)`, unsorted rectangular, zero-based indexing, `nrows=4, ncols=5, nnz=10, index=0, col_ptr=[0,3,5,8,9,10], row_ind=[0,2,3,1,2,0,1,2,2,1], values=[1.0,1.0,3.0,-1.0,2.0,2.0,4.0,3.0,4.0,1.0]`: `A = [[1.0,0.0,2.0,0.0,0.0],[0.0,-1.0,4.0,0.0,1.0],[1.0,2.0,3.0,4.0,0.0],[3.0,0.0,0.0,0.0,0.0]]`.
- `BSR Case 1 (p313)`, sorted square, zero-based indexing, 2x2 blocks in row-major layout, `blk_nrows=4, blk_ncols=4, blk_nnz=7, row_blk_size=2, col_blk_size=2, blk_layout=RM, index=0, bsr_row_ptr=[0,2,4,6,7], bsr_col_ind=[0,2,0,3,1,2,1], bsr_values=[1.2,-3.4,0.7,4.0,1.5,-3.8,2.6,-1.1,-0.9,2.2,3.7,-1.3,4.0,-2.7,1.8,-3.2,-1.4,2.9,3.1,-0.5,-3.6,0.8,2.3,-2.0,1.9,-2.4,-3.0,0.6]`: `A = [[1.2,-3.4,0,0,1.5,-3.8,0,0],[0.7,4.0,0,0,2.6,-1.1,0,0],[-0.9,2.2,0,0,0,0,4.0,-2.7],[3.7,-1.3,0,0,0,0,1.8,-3.2],[0,0,-1.4,2.9,-3.6,0.8,0,0],[0,0,3.1,-0.5,2.3,-2.0,0,0],[0,0,1.9,-2.4,0,0,0,0],[0,0,-3.0,0.6,0,0,0,0]]`.
- `BSR Case 2 (p314)`, sorted rectangular, one-based indexing, empty block row, 2x3 blocks in row-major layout, `blk_nrows=3, blk_ncols=2, blk_nnz=2, row_blk_size=2, col_blk_size=3, blk_layout=RM, index=1, bsr_row_ptr=[1,2,2,3], bsr_col_ind=[1,2], bsr_values=[1.0,0.0,2.0,0.0,-1.0,4.0,0.0,2.0,0.0,-1.0,1.0,3.0]`: `A = [[1.0,0,2.0,0,0,0],[0,-1.0,4.0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,2.0,0],[0,0,0,-1.0,1.0,3.0]]`.
- `BSR Case 3 (p315)`, unsorted rectangular, zero-based indexing, 2x2 blocks in **column-major** layout, `blk_nrows=2, blk_ncols=3, blk_nnz=4, row_blk_size=2, col_blk_size=2, blk_layout=CM, index=0, bsr_row_ptr=[0,2,4], bsr_col_ind=[2,0,0,1], bsr_values=[0.0,1.0,-1.0,0.5,1.0,0.0,0.0,-1.0,1.0,3.0,2.0,0.0,3.0,0.0,4.0,0.0]`: `A = [[1.0,0,0,0,0,-1.0],[0,-1.0,0,0,1.0,0.5],[1.0,2.0,3.0,4.0,0,0],[3.0,0,0,0,0,0]]`.

### Out-of-domain formula images encountered in these chunks

These images were listed in the chunk headers but belong to other domains; transcribed here so nothing from the images is lost.

- `Numerical Reproducibility (p294)`: a floating-point summation `a_0 + a_1 + ... + a_n`, used to illustrate that the results of a summation depend on the order of operations because floating-point addition is not associative. This belongs to the BLAS reproducibility discussion, not Sparse BLAS.
- `gebrd (p393, LAPACK domain)`: `A = Q * B * P^H = Q * ( B_1 ; 0 ) * P^H = Q_1 * B_1 * P^H` (for m >= n).
- `gebrd (p394, LAPACK domain)`: `A = Q * B * P^H = Q * ( B_1  0 ) * P^H = Q * B_1 * P_1^H` (for m < n).
- `gebrd (p396#0, LAPACK domain, USM version)`: `A = Q * B * P^H = Q * ( B_1 ; 0 )^T * P^H = Q_1 * B_1 * P^H`.
- `gebrd (p396#1, LAPACK domain, USM version)`: `A = Q * B * P^H = Q * ( B_1  0 ) * P^H = Q * B_1 * P_1^H`.

## Conventions & Gotchas

- **Indexing:** `oneapi::mkl::index_base::zero` (C-style, indices start at 0) or `oneapi::mkl::index_base::one` (Fortran-style, indices start at 1); `index` selects how *all* arrays of that handle are indexed. It is part of the handle state and must not be changed between the stages of a multi-stage operation.
- **Leading dimensions:** `ldx`, `ldy`, `ldc` must be positive and at least the usable row (row-major) or column (col-major) dimension. BSR blocks have **no** leading dimension: it must equal `row_blk_size` (row-major blocks) or `col_blk_size` (column-major blocks).
- **In-place:** `sparse::trsv()` and `sparse::trsm()` do **not** support in-place operation even though passing the same array for RHS and solution can make sense. `sparse::omatadd()` aliasing the output `C` handle/arrays with either input handle/arrays is **undefined behavior**. `omatcopy` and `omatconvert` are out-of-place only.
- **Sortedness default:** CSR, COO, CSC, BSR handles are assumed **unsorted by default**; `sparse::property::sorted` is a user promise never verified by the library. `matmat` currently requires `B` sorted. `omatadd` and `matmat` outputs are not guaranteed sorted — call `sort_matrix()`.
- **Repeated indices:** undefined behavior; compress duplicates first. Explicitly stored zeros count as structural non-zeros. COO `sorted` must not be set on CSC-style-sorted arrays (interchange `row_ind`/`col_ind` to get a CSR-style-sorted transpose instead).
- **Scratchpad/workspace sizing:** `omatadd_buffer_size` → `sizeTempWorkspace` (bytes) for `omatadd_analyze`; `omatadd_get_nnz` → `nnzC` for the user-allocated `C` arrays. `omatconvert_buffer_size` → `sizeTempWorkspace`; `omatconvert_get_nnz` → `nnzOut`. `matmat` uses `matmat_request::get_work_estimation_buf_size` and `get_compute_buf_size`/`get_compute_structure_buf_size` → `sizeTempBuffer` (one `std::int64_t`, bytes), and `get_nnz` → size of the `C` colind/values arrays; the two workspaces (`temp_buffer1`, `temp_buffer2`) must stay valid until the last `finalize`/`finalize_structure`.
- **Dependency events:** USM forms return a `sycl::event`; for multi-stage USM APIs (omatadd, omatconvert, matmat) the event must be carried over as a dependency for the next stage. Most `sycl::buffer` forms return `void` and do not accept dependencies (passing them is "not recommended ... but supported"). In the extracted text, only the USM `matmat` form declares the dependency vector **without** a default; `trsv`, `trsm`, `gemv`, `gemvdot`, `trmv`, `symv`, `gemm`, `matmatd`, the `optimize_*` routines, `omatcopy`, `sort_matrix`, `update_diagonal_values`, and the USM `set_<format>_data` forms all use `dependencies = {}`.
- **Workspace ownership:** if `nullptr` is passed for `sizeTempBuffer`/`tempBuffer` in `work_estimation` and `compute`/`compute_structure`, the library allocates and manages them until the `C` matrix handle is destroyed; otherwise the user owns and must keep them alive. `matmat`'s `sizeTempBuffer` must be **host**-accessible even though `tempBuffer` must be device-accessible.
- **Descriptor reuse:** reusing an `omatadd_descr_t` for another addition is currently undefined behavior. `omatadd_alg`, the omatconvert algorithm, and the matmat operation must not change across the stages of a single operation; `opA`/`opB` order must not change across `omatadd` calls.
- **Handle thread-safety:** `sparse::matrix_handle_t` is not thread-safe and must be used serially on the host. It is bound to the context/device of the data it was created with, and the user is responsible for using it with compatible queues/devices. The user must not modify the arrays directly while they are in a handle.
- **`diag::nonunit` precondition:** all diagonal values must be present in the sparsity profile and must not be zero, otherwise `onemkl::invalid_value()` is thrown (stated for `trsv` and `trsm`). `update_diagonal_values` likewise requires all diagonal entries to already be in the sparsity pattern.
- **Integer type requirements:** `<INT_TYPE>` must be `std::int32_t` or `std::int64_t`; index and pointer arrays (CSR `row_ptr`/`col_ind`, CSC `col_ptr`/`row_ind`, COO `row_ind`/`col_ind`, BSR `bsr_row_ptr`/`bsr_col_ind`) use `<INT_TYPE>`, while all size parameters (`nrows`, `ncols`, `nnz`, `blk_nrows`, `blk_ncols`, `blk_nnz`, `row_blk_size`, `col_blk_size`, `columns`, `c_nrows`, `c_ncols`, `length`, `ldx`/`ldy`/`ldc`) are `std::int64_t`.
- **USM memory guidance:** device memory generally outperforms shared, which outperforms host, for the sparse data arrays and `tempBuffer`; but `matmat`'s `sizeTempBuffer` and the `omatadd`/`omatconvert` `tempWorkspace` place their own host/device requirements as noted above.

## Explicit gaps

- This chapter is Sparse BLAS only. The opening pages of these chunks also carry BLAS-level material — **Compute Modes** (`oneapi::mkl::blas::compute_mode`, per-call/per-source-file/runtime selection, `MKL_BLAS_COMPUTE_MODE`, `MKL_VERBOSE`) and **Numerical Reproducibility** (CNR modes: on CPU for all BLAS DPC++ APIs, on GPU for level-3 routines and level-3 extensions) — which belong to the BLAS domain and are not documented here. The p294 formula image is transcribed above.
- The chunks end inside the **LAPACK** section (`gebrd`, `gebrd` USM version, `gebrd_scratchpad_size`, `geinv_batch` group version, `geinv_batch_scratchpad_size` group version). Those routines and their formula images (p393, p394, p396) are outside this chapter's scope; the formula images are transcribed above for completeness, but their signatures are not documented here.
- Source text inconsistencies reproduced rather than silently fixed: the `set_matmat_data` heading vs. `set_matmat_descr` syntax block; the `get_matmat_data` heading vs. `get_matmat_descr` syntax block and the `viewV` typo in its example; the `omataconvert_alg` spelling in the enum declaration vs. `omatconvert_alg` elsewhere; the asymmetric deprecated `trsv` overloads (buffer form without `alpha`, USM form with `alpha`); the missing comma after `sycl::buffer<DATA_TYPE, 1> &C` in `matmatd`'s buffer syntax; `transCooAMatirx` and `sparse::matmt()` misspellings in the source prose.
- Inline formulas inside several parameter descriptions did not survive extraction at all (for example "Specifies the scalar, ." and "Non-transpose, ."), and the extracted text renders `AT`/`AH` where the formula images use `A^T`/`A^H`; the formulas above follow the images, and nothing has been reconstructed for the missing inline text.
- No default value or device restriction is stated for several parameters. In particular, no default is given for `alg`/`descr` in `omatadd`/`omatconvert`/`matmat`, `matmat`'s USM `dependencies` has no default in the extracted signature, and `matmatd` states no restriction on `opA`/`opB`. Where the source is silent, this chapter says so rather than assuming.
- The `gemm` and `trsm` dense-matrix shape tables render their inline conditions as math images that did not survive text extraction (for example the `gemm` row-major cells were reduced to "if ,"); the table cells above (including `matmatd`'s `C` shape table) were re-read from the page images, so the non-conditional entries (`ldx`, `ldy`, `ldc`, `columns`, `m`, `c_nrows`, `c_ncols`, "number of rows/columns in A") are exact. The inline conditions omitted by extraction are given here as `op(A) = A` (`opA == nontrans`) and `op(A) = A^T` (`opA == trans`), which is the reading the surviving `A`/`A^T` fragments support.
