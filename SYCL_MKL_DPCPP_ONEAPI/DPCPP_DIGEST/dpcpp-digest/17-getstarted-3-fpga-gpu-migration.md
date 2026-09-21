---
chunk: 17-getstarted-3-fpga-gpu-migration
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0 / dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 9-12
covers: Get Started: driver choice and invocation, Visual Studio setup, first build, samples, next steps
---

# Get Started: driver, Visual Studio, first build

> **Scope.** Driver choice, command-line and SYCL `-fsycl` invocation, Visual Studio setup, a first build, and the four samples. No FPGA/GPU-migration instructions occur in this range.

## Key facts

- `icx` is the recommended default driver for Windows; `icx-cl` is its Microsoft-compatible variant (experimental on Linux, requires the Microsoft Visual Studio Package).
- C/C++ drivers: Linux `icx-cl`; Windows `icx`, `icx-cl`; MSVC-style options.
- .NET CLR C++ project types (CLR Class Library, CLR Console App, CLR Empty Project; varies by Visual Studio version) are unsupported.

## Invoking the compiler on the command line

```text
{compiler driver} [option] file1 [file2...]
```

```bash
icx hello-world.cpp          # from a Visual Studio command prompt
icx -fsycl hello-world.cpp   # SYCL: -fsycl with the C++ driver
```

**NOTE** With `-fsycl`, `-fsycl-targets=spir64` is assumed unless `-fsycl-targets` is explicitly set.

## Option 2: Use Visual Studio

- New DPC++ projects are automatically configured for the compiler; new Microsoft Visual C++ (MSVC) projects must be configured manually.
- **Platform Toolset**: **Intel® oneAPI DPC++ Compiler** for C++ with SYCL; **Intel C++ Compiler \<major version\>** (example 2025) invokes `icx`; **Intel C++ Compiler \<major.minor\>** (example 19.2) invokes `icl`. **Project > Intel Compiler > Use Intel oneAPI DPC++/C++ Compiler** applies a version to all supported platforms and configurations of the selected project(s).
- **Select Compiler Version**: **Tools > Options > Intel Compilers and Libraries > \<compiler selection\> > Compilers**, where `<compiler selection>` is `C++` or `DPC++`; choose under **Selected Compiler**.
- **Switch back to Visual C++**: **Intel Compiler > Use Visual C++** updates the solution file; configurations of affected projects are cleaned unless **Do not clean project(s)** is selected, else rebuild so all sources compile with the new compiler.

## Build a Program From the Command Line

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello, world!\n";

         return 0;
}
```

- `icx hello-world.cpp` produces `hello-world.exe`; running `hello-world.exe` outputs `Hello, world!`.
- Two-step build: `icx hello-world.cpp /c /Fohello-world.obj` (`/c` prevents linking, `/Fo` names the object file), then `icx hello-world.obj /Fehello-world.exe` (`/Fe` names the generated executable). See *Compiler Options* for available options.

## Compile and Execute Sample Code

| Sample Project | Description |
|---|---|
| OpenMP Offload Sample | The OpenMP* Offload sample demonstrates some of the new OpenMP Offload features supported by the compiler. |
| Base: Vector Add Sample | "Hello, World!" equivalent for data parallel programs; verifies environment setup and demonstrates core DPC++ features. |
| Matrix Multiply Sample | Multiplies two large matrices and verifies results; implemented with DPC++ and with OpenMP. |
| Adaptive Noise Reduction Sample | DPC++ reference design for a highly optimized image sensor adaptive noise reduction (ANR) algorithm on an FPGA. |

**NOTE** The oneAPI Samples Catalog (GitHub*) lists them all; they help you develop, offload, and optimize multiarchitecture applications targeting CPUs, GPUs, and FPGAs.

## Next Steps

- Use the latest oneAPI Code Samples and follow along with the Intel® oneAPI Training Resources.
- Explore the Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference.

## Gotchas & failure modes

- `-fsycl` silently assumes `-fsycl-targets=spir64` unless `-fsycl-targets` is set.
- `icx-cl` on Linux is experimental and requires the Microsoft Visual Studio Package.
- New MSVC projects are **not** auto-configured (DPC++ projects are); set the Platform Toolset manually.
- Switching to Visual C++ cleans affected configurations by default; opting out requires a manual rebuild or stale objects remain.
- `/c`, `/Fo`, `/Fe` and the `.exe` output are Windows-style.

## Source map

- Driver, invocation syntax and examples, Visual Studio — pp. 9–10
- Build a Program From the Command Line — pp. 10–11
- Compile and Execute Sample Code — p. 11
- Next Steps — p. 12
