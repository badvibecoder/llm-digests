---
chunk: 15-getstarted-1-need-oneapi
source: dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 1-4
covers: Get Started Ch.1 intro (product scope, Find More) and Get Started on Linux (CLI env setup, optional GPU drivers/plugins)
---

# Get Started with the Intel® oneAPI DPC++/C++ Compiler

> **Scope.** 2025.2 *Get Started* Ch. 1 opening: compiler scope/targets, "Find More" references, Linux CLI setup. Extract ends mid-bullet in the AMD\* GPU item; *Get Started on Windows* and *Compile and Execute Sample Code* appear only in the Contents.

## Key facts

- Optimizes for **Intel® 64 on Linux\* and Windows\***; supports the latest **C, C++, and SYCL\*** standards.
- Faster code via **SIMD (Single Instruction Multiple Data) vectorization**, **Intel® Performance Libraries** integration, and **OpenMP\* 5.0/5.1**, exploiting **core count** and **vector register width** in **Intel® Xeon® processors and compatible processors**.
- Compiles **C++-based SYCL source files for a wide range of compute accelerators**; part of the **Intel® oneAPI Toolkits**.
- Contents: *Get Started on Linux* (p. 4), *Get Started on Windows* (p. 7), *Compile and Execute Sample Code* (p. 11).
- Notices: © Codeplay Software Limited; Intel optimizations may not optimize to the same degree for non-Intel products.

## Find More

Names and scope of the referenced material: Release Notes (known issues, current info); **Intel® oneAPI Programming Guide** (SYCL/OpenMP offload, target accelerators, oneAPI library introductions); **Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference** (options, attributes); **Intel® oneAPI Data Parallel C++ Forum** (DPC++ and Intel® C++ Compiler forums Q&A); **Intel® oneAPI DPC++/C++ Compiler Documentation** (tutorials, training); **SYCL Specification Version 1.2.1 PDF** (SYCL integrates OpenCL™ devices with modern C++); **SYCL: C++ Programming for Heterogeneous Parallel Computing** (overview); **The GNU\* C++ Library - Using Dual ABI** (dual ABI). **NOTE:** samples in the **oneAPI Samples Catalog (GitHub\*)** for **CPUs, GPUs, and FPGAs**.

## Get Started on Linux

**Before You Begin.** **Unified Directory Layout** (implemented **2024.0**): with multiple toolkit versions installed, ensures correct component versions for each. Pre-2024.0 **Component Directory Layout** still supported on new and existing installations. Details: *Use the setvars and oneapi-vars Scripts with Linux*.

**Set Environment Variables for CLI Development.** CLI use requires configuring compiler environment variables: run `setvars` (Component Directory Layout) or `oneapi-vars` (Unified Directory Layout). Changes sourced by `setvars.sh` or `oneapi-vars.sh` apply by default **only to the terminal session where you sourced the script**; every new terminal must source it again. Details: *Use the setvars and oneapi-vars Scripts with Linux*. Optionally use one-time setup for `setvars.sh` per *Use Environment Modulefiles with Linux*.

**GPU Drivers or Plugins (Optional).** oneAPI C++/SYCL\* apps can run on **Intel, AMD\*, or NVIDIA\* GPUs**; install the corresponding drivers or plugins first. **Intel GPU:** latest Intel GPU drivers. **AMD GPU:** *[unclear in source: extract ends mid-bullet]*

## Source map

GS p. 2 title/Contents · p. 3 Ch. 1 intro, Find More, sample-catalog note · p. 4 notices (legal only), *Before You Begin*, *Set Environment Variables for CLI Development*, *GPU Drivers or Plugins (Optional)*
