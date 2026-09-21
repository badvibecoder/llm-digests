---
chunk: 16-getstarted-2-build-and-jupyter
source: dpcpp-cpp-compiler_get-started-guide_2025.2
source_pages: 5-8
covers: Get Started — compiler drivers/CLI, Eclipse* CDT, hello-world build, Get Started on Windows (Visual Studio*, env vars, GPU plugins)
---

# Get Started: Invoke the Compiler, Build a Program, Get Started on Windows

> **Scope.** Driver selection and CLI invocation (Linux/Windows), Eclipse* CDT integration, the `hello-world.cpp` build, and *Get Started on Windows*. The extract ends mid-table on GS p. 8; **no Jupyter notebook material is present**.

## Key facts

- **Option Style** follows the driver: Clang-style (`icx`, `icx-cc`, `icpx`) vs MSVC-style (`icx-cl`).
- SYCL compilation = C++ driver + `-fsycl`; `-fsycl` assumes **`-fsycl-targets=spir64`** unless `-fsycl-targets` is set explicitly.
- Linux `icx-cl` is **experimental** and requires the Microsoft Visual Studio Package.
- Visual Studio* integration: **Visual Studio 2022**, **Visual Studio 2019**.

## Invoke the Compiler from the Command Line

| Language | Driver (Linux) | Driver (Windows) | Option Style |
|---|---|---|---|
| C | `icx`, `icx-cc` | `icx-cc` | Clang-style |
| C++ | `icpx` | `icpx` | Clang-style |
| C/C++ | `icx-cl` (see NOTE) | `icx`, `icx-cl` | MSVC-style |

- `icx` — recommended default C driver for Linux and recommended default driver for Windows. A C++ source file passed to `icx` is compiled as a C++ file; use `icx` to link C object files.
- `icx-cc` — Microsoft-compatible variant of `icx`.
- `icpx` — recommended default C++ driver for Linux. A C source file passed to `icpx` is compiled as a C++ file; use `icpx` to link C++ object files.
- `icx-cl` — Microsoft-compatible variant of `icx`. **NOTE:** on Linux, `icx-cl` is experimental and requires the Microsoft Visual Studio Package.

```text
{compiler driver} [option] file1 [file2...]
```

```bash
icpx hello-world.cpp                 # example
icpx -fsycl hello-world.cpp          # SYCL compilation, C++ driver
```

**NOTE:** With `-fsycl`, `-fsycl-targets=spir64` is assumed unless the `-fsycl-targets` is explicitly set in the command.

**GPU drivers/plugins, GS p. 5 tail (section opens GS p. 4).** AMD GPU: install the oneAPI for AMD GPUs plugin from Codeplay; Download oneAPI for AMD GPUs. NVIDIA GPU: install the oneAPI for NVIDIA® GPUs plugin from Codeplay; Download oneAPI for NVIDIA® GPUs. AMD/NVIDIA compilation details: *oneAPI for AMD GPUs Get Started Guide*, *oneAPI for NVIDIA® GPUs Get Started Guide*.

## Invoke the Compiler From the Eclipse* CDT

Install the Intel® Compiler Eclipse CDT Plugin: *Help > Install New Software > Add > Archive*; pick the `.zip` file that starts with `com.intel.dpcpp.compiler` under `<install_dir>/compiler/<version>/linux/ide_support`; select the options beginning with Intel and follow the installation instructions (restart Eclipse when asked). Then set the toolchain to **Intel DPC++/C++ Compiler** via *Project > Properties > C/C++ Build > Toolchain Editor*, and create/manage build configurations via *Project > Properties > C/C++ Build > Settings*.

## Example: Build a Program From the Command Line

1. Create `hello-world.cpp`:

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello, world!\n";

    return 0;
}
```

2. Open a terminal.
3. If not using one-time setup for `setvars.sh`, source setvars. Component Directory Layout: `source <install-dir>/setvars.sh`. Unified Directory Layout: `source <install-dir>/<toolkit-version>/oneapi-vars.sh`. For `<install-dir>` on system-wide or private installations see *Use the setvars and oneapi-vars Scripts with Linux*.
4. `icpx hello-world.cpp -o hello-world` — `-o` specifies the file name for the generated output.
5. Run the executable, which outputs:

```text
Hello, world!
```

Two-step alternative: `icpx hello-world.cpp -c` (`-c` prevents linking at this step), then `icpx hello-world.o -o hello-world` (`-o` specifies the generated executable file name). See *Compiler Options*.

## Get Started on Windows

### Before You Begin — Set Environment Variables

- Integrates into Visual Studio 2022 and Visual Studio 2019.
- **Visual Studio Community\* or higher** is required for full functionality within Visual Studio, including debugging and development; **Visual Studio Express\*** allows only command-line builds.
- For all versions, **Microsoft C++\* support must be selected** as part of the install; Visual Studio 2017 and later require a **custom install** to select it.
- Windows* typically needs no manual environment variables — the compiler command-line window sets them automatically. If needed, run the environment script per *Use the setvars and oneapi-vars Scripts with Windows*.

### GPU Drivers or Plugins (Optional)

oneAPI applications using C++ and SYCL* can run on Intel, AMD\*, or NVIDIA\* GPUs; install the matching drivers or plugins first. **Intel GPU:** latest Intel GPU drivers. **NVIDIA GPU:** install the oneAPI for NVIDIA® GPUs plugin from Codeplay; Download oneAPI for NVIDIA® GPUs.

### Option 1: Use the Command Line in Visual Studio

Same multiple-driver model. The extract's copy of the table has only the **C** and **C++** rows (it ends mid-table), identical to the table above.

## Gotchas & failure modes

- **Driver, not extension, sets language:** `icx` compiles a C++ source file as C++; `icpx` compiles a C source file as C++. Link C objects with `icx`, C++ objects with `icpx`.
- **SYCL target is assumed:** `icpx -fsycl` implies `-fsycl-targets=spir64`.
- **Linux `icx-cl`** is experimental and needs the Microsoft Visual Studio Package.
- **Windows gates:** Express\* = command-line only; Community\*+ for in-IDE debugging/development; Microsoft C++* support must be selected (custom install from Visual Studio 2017).
- **Layout picks the script:** Component (`source <install-dir>/setvars.sh`) vs Unified (`source <install-dir>/<toolkit-version>/oneapi-vars.sh`).
- **AMD/NVIDIA support needs Codeplay plugins.**

## Source map

- GPU plugin tail; driver table — GS p. 5
- `icx-cl` Linux NOTE; CLI syntax/examples; `-fsycl` NOTE; Codeplay pointers; Eclipse* CDT — GS p. 6
- Build example; *Get Started on Windows* → *Set Environment Variables* — GS p. 7
- Visual Studio* requirements; *GPU Drivers or Plugins (Optional)*; *Option 1: Use the Command Line in Visual Studio* (truncated) — GS p. 8
