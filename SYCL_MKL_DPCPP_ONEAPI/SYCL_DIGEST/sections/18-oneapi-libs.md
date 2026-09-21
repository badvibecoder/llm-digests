## 18 · oneAPI libraries (oneMKL, oneTBB, oneDNN, oneDAL, oneCCL)

**Source scope.** The source's oneAPI library coverage is overview-level: namespace/object vocabulary, includes, and one full oneMKL buffer sample. It states **no** signature, engine, distribution, primitive, collective, or algorithm names beyond those below. Do not infer missing APIs from this section.

| Library | Usage (source) |
|---|---|
| Intel® oneAPI DPC++ Library (oneDPL) | high performance parallel applications — see §13 |
| Intel® oneAPI Math Kernel Library (oneMKL) | highly optimized, extensively parallelized math routines |
| Intel® oneAPI Threading Building Blocks (oneTBB) | combine TBB-based multicore CPU parallelism with SYCL device-accelerated parallelism |
| Intel® oneAPI Data Analytics Library (oneDAL) | big data analysis and distributed computation |
| Intel® oneAPI Collective Communications Library (oneCCL) | Deep Learning / Machine Learning workloads |
| Intel® oneAPI Deep Neural Network Library (oneDNN) | neural-network building blocks for Intel Architecture Processors and Intel Processor Graphics |

Sample catalog: `https://oneapi-src.github.io/oneAPI-samples/` (GitHub*).

### oneDPL

See **§13** (oneDPL). Not duplicated here.

### oneMKL

Two-level namespaces (SYCL interfaces):

| Namespace | Contains |
|---|---|
| `oneapi::mkl` | common elements between various domains |
| `oneapi::mkl::blas` | dense vector-vector, matrix-vector, matrix-matrix low-level operations |
| `oneapi::mkl::lapack` | higher-level dense matrix operations: matrix factorizations, eigensolvers |
| `oneapi::mkl::rng` | random number generators for various probability density functions |
| `oneapi::mkl::stats` | basic statistical estimates for single/double precision multi-dimensional datasets |
| `oneapi::mkl::vm` | vector math routines |
| `oneapi::mkl::dft` | fast Fourier transform operations |
| `oneapi::mkl::sparse` | sparse matrix-vector multiplication, sparse triangular solver |

Computation areas with optimized SYCL interfaces: BLAS/LAPACK dense linear algebra, Sparse BLAS, RNG, VM, FFT. Also usable via OpenMP* offload for C and Fortran interfaces.

**Includes** (exactly as the sample shows):

```cpp
// Standard SYCL header
#include <CL/sycl.hpp>
// STL classes
#include <exception>
#include <iostream>
// Declarations for Intel oneAPI Math Kernel Library SYCL/DPC++ APIs
#include "oneapi/mkl.hpp"
```

**Rule.** oneMKL depends on the Intel® oneAPI DPC++/C++ Compiler and Intel oneDPL. Build with the DPC++ compiler, make SYCL headers available, and link with oneMKL using the **DPC++ linker**.

**Rule.** SYCL interfaces take device-accessible **USM pointers** for input data (vectors, matrices). **Many** interfaces also accept `sycl::buffer` objects in place of those USM pointers. No link flags or library names are given by the source.

**Rule.** Interfaces are overloaded on floating-point type: single precision `float`, double precision `double`, half precision `half`, and complex via `std::complex<float>` / `std::complex<double>` (several general matrix multiply APIs).

**Rule (exceptions).** Host-side errors are caught with standard C++ exception handling. Device-side errors are **asynchronous**: stored in an exception list and processed later by a user-provided queue exception handler taking `sycl::exception_list`. Call `my_queue.wait_and_throw()` to hand caught asynchronous exceptions to the handler.

Smallest viable shape (buffer variant; `C = alpha * A * B + beta * C`):

```cpp
// create execution queue on my gpu device with exception handler attached
sycl::queue my_queue(my_device, my_exception_handler);
// create sycl buffers of matrix data for offloading between device and host
sycl::buffer<double, 1> A_buffer(A.data(), A.size());
sycl::buffer<double, 1> B_buffer(B.data(), B.size());
sycl::buffer<double, 1> C_buffer(C.data(), C.size());
try {
  using oneapi::mkl::blas::gemm;
  using oneapi::mkl::transpose;
  gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha,
       A_buffer, ldA, B_buffer, ldB, beta, C_buffer, ldC);
} catch (sycl::exception const& e) { /* ... */ }
// ensure any asynchronous exceptions caught are handled before proceeding
my_queue.wait_and_throw();
```

Call shape implied for `oneapi::mkl::blas::gemm`: `(queue, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`; leading dimensions `ldA`, `ldB`, `ldC`; A is m-by-k, B is k-by-n, C is m-by-n.

**Buffer-vs-USM.** The sample is buffer-based (`sycl::buffer<double, 1>` built from `A.data(), A.size()`). Runtime transfers buffer data host↔device automatically; by the time an accessor is created for `C_buffer`, data has been silently transferred back to host if necessary:

```cpp
auto C_accessor = C_buffer.template get_access<sycl::access::mode::read>();
```

**Gotcha.** Result data stays in the buffer; unless explicitly copied elsewhere it is available **only through accessors**, and only while `C_buffer` is in scope.

### oneTBB

**Fact.** C++ library for task-based shared-memory parallel programming **on the host**. Provides for CPUs, beyond SYCL*/ISO C++: generic parallel algorithms, concurrent containers, scalable memory allocator, work-stealing task scheduler, low-level synchronization primitives. Compiler-independent; portable across processors/OSes. Used by other oneAPI libraries (oneMKL, oneDNN) to express multithreading parallelism for CPUs.

**Fact.** Used with the Intel® oneAPI DPC++/C++ Compiler like any other C++ compiler. oneTBB **does not directly use accelerators**; combine with SYCL*, OpenMP* offload, and other oneAPI libraries to use all hardware.

**Fact.** The SYCL runtime itself uses oneTBB for CPU work-group scheduling; partitioner selection is via `DPCPP_CPU_SCHEDULE` (`dynamic` = auto_partitioner, `affinity` = affinity_partitioner, `static` = static_partitioner) — see §09.

Samples (`https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneTBB`), all prepared for CPU and GPU:

| Sample | Pattern |
|---|---|
| `tbb-async-sycl` | oneTBB Flow Graph asynchronous node (SYCL, GPU) + functional node (CPU) |
| `tbb-task-sycl` | two oneTBB tasks; one runs SYCL code, one runs oneTBB code |
| `tbb-resumable-tasks-sycl` | oneTBB resumable task (SYCL, GPU) + `parallel_for` (CPU) |

Source names no `tbb::`-qualified types, no `task_group`, and no flow-graph API signatures.

### oneDAL

**Fact.** Optimized algorithmic building blocks for all stages of data analytics: preprocessing, transformation, analysis, modeling, validation, decision making — in batch, online, and distributed modes. C++ and Java* APIs; connectors to Spark* and Hadoop*; Python* wrappers are part of Intel® Distribution for Python*.

**Fact.** Provides DPC++ SYCL API extensions to the traditional C++ interface and enables GPU usage for some algorithms. DPC++ applications select CPU or GPU by **picking the proper device selector**.

**Rule.** New capabilities: extract SYCL* buffers from numeric tables and pass them to a custom kernel; create numeric tables from SYCL buffers. Algorithms reuse SYCL buffers to keep GPU data resident and avoid repeated host↔device copies.

Sample starting point: `https://github.com/oneapi-src/oneDAL/tree/master/examples/oneapi/dpc/source/svm`.

Source names no `oneapi::dal` symbols, `get_table`, `kmeans`, `linear_regression`, `decision_forest`, or `train`/`infer`/`compute` flow.

### oneCCL

**Fact.** Scalable, high-performance communication library for DL/ML workloads. Built on lower-level communication middleware — **MPI and libfabrics**. DL-specific optimizations: prioritization, persistent operations, out of order execution. DPC++-aware API across hardware targets (CPUs, GPUs). Interconnects: Intel® Omni-Path Architecture (Intel® OPA), InfiniBand*, Ethernet.

**Rule (backend selection).** `SYCL*`-aware API is optional. Choose CPU vs SYCL backend when creating the oneCCL **stream object**:

| Backend | First argument to stream creation |
|---|---|
| CPU backend | `ccl_stream_host` |
| SYCL backend | `ccl_stream_cpu` or `ccl_stream_gpu`, depending on device type |

**Rule (collective buffers on a SYCL stream).** C API: communication buffers must be `sycl::buffer` objects **cast to `void*`**. C++ API: buffers are passed **by reference**.

Docs: `https://oneapi-src.github.io/oneCCL/`. Samples: `https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneCCL` (includes a Getting Started sample).

Source names no `allreduce`, `allgather`, `broadcast`, `alltoall`, or point-to-point routine names.

### oneDNN

**Fact.** Open-source performance library for deep learning; basic neural-network building blocks optimized for Intel Architecture Processors and Intel Processor Graphics. Supports C and C++ interfaces, OpenMP*, Intel® oneAPI Threading Building Blocks, OpenCL™ runtimes; adds SYCL*/DPC++ API and runtime support. Distributed in the Intel® oneAPI DL Framework Developer Toolkit, Intel® oneAPI Base Toolkit, and via apt and yum.

**Fact.** Detects ISA at runtime and uses online generation to deploy code optimized for the latest supported ISA.

Packages (dependencies not bundled; resolve at build time):

| Configuration | Dependency |
|---|---|
| `cpu_dpcpp_gpu_dpcpp` | DPC++ runtime |
| `cpu_iomp` | Intel OpenMP* runtime |
| `cpu_gomp` | GNU* OpenMP runtime |
| `cpu_vcomp` | Microsoft* Visual C++ OpenMP runtime |
| `cpu_tbb` | Intel oneAPI Threading Building Blocks |

**Rule.** Under SYCL*, oneDNN uses the DPC++ SYCL runtime for CPU/GPU hardware; interop API covers (a) constructing oneDNN objects from existing SYCL objects, (b) accessing SYCL objects for existing oneDNN objects. **You must include `dnnl_sycl.hpp` to enable the SYCL-interop API.** OpenMP needs no interop API because it does not pass runtime objects.

Object mapping:

| oneDNN object | SYCL object |
|---|---|
| Engine | `cl::sycl::device` and `cl::sycl::context` |
| Stream | `cl::sycl::queue` |
| Memory | `cl::sycl::buffer<uint8_t, 1>` or USM pointer |

| oneDNN object | Construct from SYCL object |
|---|---|
| Engine | `dnnl::sycl_interop::make_engine(sycl_dev, sycl_ctx)` |
| Stream | `dnnl::sycl_interop::make_stream(engine, sycl_queue)` |
| Memory (USM based) | `dnnl::memory(memory_desc, engine, usm_ptr)` |
| Memory (buffer based) | `dnnl::sycl_interop::make_memory(memory_desc, engine, sycl_buf)` |

| oneDNN object | Extract SYCL object |
|---|---|
| Engine | `dnnl::sycl_interop::get_device(engine)`, `dnnl::sycl_interop::get_context(engine)` |
| Stream | `dnnl::sycl_interop::get_queue(stream)` |
| Memory (USM pointer) | `dnnl::memory::get_data_handle()` |
| Memory (buffer) | `dnnl::sycl_interop::get_buffer(memory)` |

**Rule.** Internally, library memory objects use **1D `uint8_t`** SYCL buffers; buffers of a different type may be used to initialize and access memory, and are reinterpreted to `cl::sycl::buffer<uint8_t, 1>`.

Sample (Getting Started): `https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDNN`.

Source names no `dnnl::engine` / `dnnl::stream` / `dnnl::memory::desc` constructors, no convolution/matmul/reorder primitives, no `execute`, and no `get_dnnl_engine`.

### Key gotchas

- Treat this section as namespace/object vocabulary only; get real signatures from library docs, not from here.
- For oneMKL, prefer USM pointers; use buffer overloads only where the source says "many" interfaces support them.
- Wrap oneMKL calls in `try`/`catch` **and** attach a queue exception handler for asynchronous errors.
- Always call `wait_and_throw()` before reading results, or device exceptions are lost.
- Copy buffer results into host containers before the `sycl::buffer` scope ends; accessors die with it.
- Include quoted `"oneapi/mkl.hpp"` plus `<CL/sycl.hpp>`; link through the DPC++ linker.
- Match the oneDNN package to your runtime (`cpu_iomp`, `cpu_gomp`, `cpu_vcomp`, `cpu_tbb`, `cpu_dpcpp_gpu_dpcpp`) or ABI/link failures follow.
- Include `dnnl_sycl.hpp` before using any `sycl_interop` call; build oneDNN engines/streams via `make_engine`/`make_stream`, not direct construction.
- Pass typed buffers to oneDNN memory; they are reinterpreted to `cl::sycl::buffer<uint8_t, 1>`.
- Select the oneCCL stream backend up front: `ccl_stream_host` for CPU, `ccl_stream_cpu`/`ccl_stream_gpu` for SYCL.
- Pass oneCCL SYCL buffers as `void*` in C, by reference in C++; mismatching the API breaks silently.
- oneTBB never runs on the accelerator; keep GPU work in SYCL and CPU work in oneTBB.
