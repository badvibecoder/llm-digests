---
chunk: 00-intro-setup
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0
source_pages: 1-51
covers: Compiler introduction, key features, help/support and related reading; feature requirements; command-line setup (Linux and Windows), environment scripts, drivers, file extensions, makefiles, CMake, Eclipse and Microsoft Visual Studio* integration; converting projects to a selected compiler; C/C++/SYCL calling conventions including __regcall.
---

# Intel® oneAPI DPC++/C++ Compiler — Introduction, Setup, and Calling Conventions

> **Scope.** Select the right compiler driver for a language/platform, configure the compiler
> environment, invoke the compiler and know which options and file extensions it accepts, drive it
> from makefiles, CMake, Eclipse, or Microsoft Visual Studio*, and apply the exact C/C++/SYCL
> calling conventions and `__regcall` register/classification rules. Valid for compiler version
> **2026.0**.

## Key facts

- Part of the Intel® oneAPI Toolkit or a **standalone compiler**. Valid for compiler version **2026.0**.
- Architecture: **Intel® 64 architecture** only. OS: Linux and Windows on Intel® 64 systems. 32-bit OSes are **not** supported; **macOS* is not supported** (for macOS/Xcode* use **Intel® C++ Compiler Classic**).
- IDEs: **Eclipse*/CDT (Linux only)**, **Microsoft Visual Studio* (Windows only)**.
- Standards: **C++ 20, SYCL, most of OpenMP 5.2, and some OpenMP 6.0 TR12 features**. (The Key Features list instead says OpenMP 5.0 Version TR4 and some OpenMP 5.1 features — both appear in the source.)
- Key features: Intel® oneAPI Level Zero (direct-to-metal interfaces to offload accelerator devices), OpenMP* support, pragmas, offload support (SYCL*, OpenMP, parallel processing options), latest standards.
- **Clang options** are supported but not documented; check `-help` for whether a particular option is supported.
- **`dpcpp` driver is deprecated** and will be removed in a future release. For SYCL compilation use **`-fsycl` with the C++ driver**.
- Features labeled **experimental** may be implemented or removed in a future release.
- The environment must be configured per terminal session; otherwise the compiler can behave unpredictably.

## Intel® oneAPI DPC++/C++ Compiler Introduction

Unless specified otherwise the guide applies to all supported architectures and OSes. Guide sections: Introduction (feature requirements, support, related information); Compiler Setup (invoking the compiler from the command line or an IDE); Compiler Reference (options, compiler limits, libraries); Compilation (environment variables, configuration files); Optimization and Programming; Compatibility and Portability.

### Feature Requirements

Compilation may fail if an option is specified but its required product is not installed; remove the option from the command line and recompile.

| Feature | Requirement |
|---|---|
| `-qtbb`, `-tbb`, `/Qtbb` | Intel® oneAPI Threading Building Blocks (oneTBB) install |
| `-mkl`, `-qmkl`, `-qmkl-ilp64`, `/Qmkl`, `/Qmkl-ilp64` | Intel® oneAPI Math Kernel Library (oneMKL) install |
| `-daal`, `-qdaal`, `/Qdaal` | Intel® oneAPI Data Analytics Library (oneDAL) install |
| `-ipp`, `-qipp`, `/Qipp` | Intel® Integrated Performance Primitives (Intel® IPP) install |
| Use crypto (links to Intel® Cryptography Primitives Library) | Intel® Cryptography Primitives Library install |
| Thread Checking | Intel® Inspector install — **deprecated** |
| Trace Analyzing and Collecting | Intel® Trace Analyzer and Collector install — **deprecated**; related options may require a set-up script |

See the Release Notes for supported architectures, operating systems, and IDEs for this release.

## Get Help and Support

Documentation, Context Sensitive/F1 Help, and previous PDF/FARHTML versions of this guide: Explore Our Documentation page, the compiler main page, and the Download Documentation: Intel® Compiler (Current and Previous) page. For downloaded HTML use Google Chrome* (with Mozilla Firefox* the Search tab may not work; use the Contents and Index tabs). Support and warranty requests go through the Online Service Center; support requires product registration at the Intel Registration Center. System requirements and supported architectures/OSes/IDEs are in the Release Notes. Forums: Software Development Tools (general), Intel® C++ Compilers, Intel® oneAPI Data Parallel C++.

**Related information:** Development Tools page; *Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL*; Get Started with the Intel® oneAPI DPC++/C++ Compiler; the compiler main page; Intel® Guides and Tutorials; Intel® Intrinsics Guide; Intel® oneAPI Programming Guide; Intel® Technical Articles and How-Tos. Additional reading (threading background, not endorsed over others): Reinders, *Intel Threading Building Blocks* (2007) and *Pro TBB* (2019); Chapman et al., *Using OpenMP* (2007); Jansen, *Basic Parallel Programming with OpenMP* (2017); Mattson et al., *The OpenMP Common Core* (2019); Woodring & Cohen, *WIN32 Multithreaded Programming* (1997); Akhter & Roberts, *Multi-Core Programming* (2006).

## Compiler Setup

The compiler is usable from the command line, Eclipse, or Microsoft Visual Studio.

### Use the Command Line

#### Specify Component Locations

Set environment variables defining compiler-related component locations **for each terminal session**; otherwise the compiler can behave unpredictably. Scripts: Linux **`setvars.sh`**, Windows **`setvars.bat`**.

> **Unified Directory Layout** (implemented in **2024.0**) ensures that with multiple toolkit versions installed, the development environment contains the correct component versions for each installed version. The pre-2024.0 **Component Directory Layout** is still supported.

Use **`setvars`** (Component Directory Layout) or **`oneapi-vars`** (Unified Directory Layout). Changes apply only to the sourced terminal session; source again in each new session. One-time setup for `setvars.sh` is available via environment modulefiles.

**Linux:**

```bash
source /<install-dir>/setvars.sh <arg1> <arg2> … <argn>
source /opt/intel/oneapi/setvars.sh intel64
. /<install-dir>/setvars.sh <arg1> <arg2> … <argn>
source /<install-dir>/setvars.sh --help     # more setvars usage information
```

Arguments: `intel64` (generate code and use libraries for Intel® 64 architecture-based targets); `--include-intel-llvm` (adds the Intel Compiler's clang binaries folder `bin-llvm` to the `PATH`). To run it in all sessions, add the `source setvars.sh` command to the startup file (e.g. `.bash_profile`). Unset variables can break execution of compiled programs:

```text
./a.out: error while loading shared libraries:
libimf.so: cannot open shared object file: No such file or directory
```

**Windows** — normally `setvars.bat` is unnecessary: the Start menu shortcuts *Intel oneAPI command prompt for \<target architecture\> for Visual Studio \<year\>* set the variables automatically. Run it if the command line was opened without one of those items, or from your own script. It inserts DLL directories used by the compiler and libraries at the **beginning** of `Path`, so they are searched before the original Windows `Path` — important if the original `Path` has same-named files.

```text
<install-dir>\setvars.bat [<arg1>] [<arg2>]
```

`<arg1>` optional: `intel64` (host and target) or `--include-intel-llvm`. `<arg2>` optional: `vs2022` (Microsoft Visual Studio 2022) or `vs2019` (Microsoft Visual Studio 2019). With multiple editions installed, automatic search precedence within a year is **Professional**, **Enterprise**, then **Community**; the preferred edition can be set with `VS20??INSTALLDIR` (`VS2022INSTALLDIR`, `VS2019INSTALLDIR`, etc.). Defaults: `<arg1>` = `intel64`; `<arg2>` = highest installed Microsoft Visual Studio version detected during installation.

#### Invoke the Compiler

| Language | Driver for Linux | Driver for Windows | Option style | Notes |
|---|---|---|---|---|
| C | `icx`, `icx-cc` | `icx-cc` | Clang-style | `icx` is the recommended default C driver for Linux |
| C++ | `icpx` | `icpx` | Clang-style | `icpx` is the recommended default C++ driver for Linux |
| C/C++ | `icx-cl` (see notes) | `icx`, `icx-cl` | MSVC-style | `icx` is the recommended default driver for Windows; `icx-cl` is the Microsoft-compatible variant of `icx` |

- `icx` with a C++ source file compiles it as C++; use `icx` to link C object files. `icpx` with a C source file compiles it as C++; use `icpx` to link C++ object files.
- NOTE: on Linux `icx-cl` is **experimental** and requires the Microsoft Visual Studio Package.

```text
{compiler driver} [option] file1 [file2...]
```

`option`: on Linux, letters preceded by `-`; on Windows, `-` or `/`, including linker options. Options are not required; the default behavior implies some options are ON by default.

```bash
icpx hello-world.cpp
icpx -fsycl hello-world.cpp     # SYCL: use -fsycl with the C++ driver
```

- **Linux:** compiles and links the input source file(s); produces one executable **`a.out`** in the current directory.
- **Windows:** compiles and links the input file(s), producing object file(s) named after the respective source file(s) with an `.obj` extension; produces one executable named after the **first input file on the command line** with an `.exe` extension; all files go in the current directory.

Other methods: **makefiles** (many files, various paths, saved for multiple compilations); **batch files** (a `.bat` file running the compiler with a desired option set).

#### Use the Command Line on Windows

A Start menu shortcut opens the command line with the environment already set: Start menu > **Intel oneAPI 2025** folder > your component. `exit` ends the session. Any Windows command prompt command plus some additional commands is available.

### File Extensions

**Input:**

| OS-agnostic | Linux | Windows | Interpretation | Action |
|---|---|---|---|---|
| `file.c` | — | — | C source file | Passed to the compiler |
| `file.C`, `file.CC`, `file.cc`, `file.cpp`, `file.cxx` | — | — | C++ source file | Passed to the compiler |
| — | `file.a`, `file.so` | `file.lib` | Library file | Passed to the linker |
| `file.i` | — | — | Preprocessed file | Passed to the compiler |
| — | `file.o` | `file.obj` | Object file | Passed to the linker |
| — | `file.s`, `file.S` | `file.asm` | Assembly file | Passed to the assembler |

**Output:**

| OS-agnostic | Linux | Windows | Description |
|---|---|---|---|
| `file.i` | — | — | Preprocessed file: produced with `-E` |
| — | `file.o` | `file.obj` | Object files: produced with `-c`; rename with `-o` (Linux) or `/Fo` (Windows) |
| — | `file.s` | `file.asm` | Assembly language file: produced with `-S`; rename with `-s` `[sic: source says -s; the "Specify Assembly Files" topic uses -S -o]` (Linux) or `/Fa` (Windows) |
| — | `a.out` | `file.exe` | Executable file: produced by default compilation; rename with `-o` (Linux) or `/Fe` (Windows) |

### Use Makefiles for Compilation

**Linux.** Ensure `/usr/bin` and `/usr/local/bin` are in `PATH`. With the C shell, add to `.cshrc`:

```bash
setenv PATH /usr/bin:/usr/local/bin:$PATH
```

The makefile must include `CC=icx`, `CC=icpx`, or `CC=icpx -fsycl`; use the same setting on the command line. For a GCC makefile, change command line options not recognized by the compiler. Run:

```bash
make -f yourmakefile
```

where `-f` specifies the makefile name.

**Windows.** Use `nmake`:

```text
nmake /f [makefile_name.mak] CPP=[compiler_name] [LINK32=[linker_name]
```

`[sic: source shows an unbalanced bracket]`

```bat
nmake /f your_project.mak CPP=icx LINK32=link
```

`/f` — nmake option specifying a makefile; `your_project.mak` — makefile generating object and executable files; `CPP` — preprocessor/compiler generating object and executable files (macro name may differ); `LINK32` — linker used. `nmake` creates object files (`.obj`) and executable files from `your_project.mak`.

> NOTE: if you have link/xilink specific options not accepted by `icx-cl -fsycl`, place any linker specific options after the `/link` option.

### Use CMake with the Compiler

**Linux** — enabled using the `icx` (variant) binary; you may need to set `CC`/`CXX` or `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` to `icx`/`icpx`:

```bash
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx …
```

**Windows** — enabled using the `icx` (variant) binary; set `CC`/`CXX` or `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` to `icx`. The supported generator is **Ninja**:

```bat
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -GNinja …
```

> NOTE: if your Microsoft Visual Studio 2022 default CMake is older than **3.23.0**, install CMake **3.25** (or above) and update Visual Studio with the new CMake executable (edit `CMakeSettings.json`). Linux works with CMake **3.23.5** and later. Support for GNU-like drivers (`icx-cc`, `icpx`) on Windows needs a newer CMake.

**Enable the Compiler** — **`IntelSYCLConfig`** (recommended: compatible with de-facto industry standards; `IntelDPCPPConfig` may be deprecated in the future) or **`IntelDPCPPConfig`**:

```cmake
# IntelSYCLConfig
if (CMAKE_HOST_WIN32)
# need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
cmake_minimum_required(VERSION 3.25)
else()
# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
cmake_minimum_required(VERSION 3.23.5)
endif()

find_package(IntelSYCL REQUIRED)

add_executable(target_proj A.cpp B.cpp offload1.cpp offload2.cpp)
add_sycl_to_target(TARGET target_proj SOURCES offload1.cpp offload2.cpp)
```

```cmake
# IntelDPCPPConfig
cmake_minimum_required(VERSION 3.23.0)
find_package(IntelDPCPP REQUIRED)
```

`find_package(IntelSYCL REQUIRED)` imports `IntelSYCLConfig.cmake`; `find_package(IntelDPCPP REQUIRED)` imports `IntelDPCPPConfig.cmake`. Both ship with the compiler; the package directory is the parent directory of the `icx` bin directory. Specifying no sources to `add_sycl_to_target()` adds SYCL compilation to **all** sources and may greatly increase compilation time. Then select the C/C++ compilers and run CMake.

**Build and Run.**

```cmake
if (CMAKE_HOST_WIN32)
    # need at least CMake 3.25 for IntelLLVM support of IntelSYCL package on Windows
    cmake_minimum_required(VERSION 3.25)
else()
    # CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.23.5)
endif()

project(simple-sycl LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

add_executable(simple simple.cpp)
add_sycl_to_target(TARGET simple SOURCES simple.cpp )
```

`project` names the project and states it uses C++ (C, Fortran, or others go in `LANGUAGES`); IntelSYCL is in CMake's search path after `setvars.sh` (Linux) or `setvars.bat` (Windows) and sets the compiler/linker flags required for SYCL. Example `simple.cpp`:

```cpp
#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

int main(int argc, char* argv[])
{
    sycl::queue queue;

    std::cout << "Using "
        << queue.get_device().get_info<sycl::info::device::name>()
        << std::endl;

    // Compute the first n_items values in a well known sequence
    constexpr int n_items = 16;
    int *items = sycl::malloc_shared<int>(n_items, queue);

    queue.parallel_for(sycl::range<1>(n_items), [items] (sycl::id<1> i) {
        double x1 = pow((1.0 + sqrt(5.0))/2, i);
        double x2 = pow((1.0 - sqrt(5.0))/2, i);
        items[i] = round((x1 - x2)/sqrt(5));
    }).wait();

    for(int i = 0 ; i < n_items ; ++i) {
        std::cout << items[i] << std::endl;
    }
    free(items, queue);

    return 0;
}
```

```bash
# Linux
mkdir build && cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./simple
```

```bat
:: Windows
mkdir build && cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icx ..
cmake –build .
.\simple.exe
```

> The Linux Makefile generator is known to work with Intel oneAPI compilers and CMake; other generators may work but have not been thoroughly tested.

### Use Compiler Options

A compiler option is a **case-sensitive** command line expression that changes the compiler's default operation, controlling code generation, optimization, output file (type, name, location), linking properties, executable size, and executable speed. Options are not required to compile.

**Linux syntax** — invocation is `icx`, `icpx`, or `icpx -fsycl`:

```text
[invocation] [option] [@response_file] file1 [file2...]
```

Files: C or C++ source (`.C`, `.c`, `.cc`, `.cpp`, `.cxx`, `.c++`, `.i`, `.ii`), assembly (`.s`, `.S`), object (`.o`), static library (`.a`). Compile C sources with `icx`; C++ or mixed with `icpx`; SYCL sources with `icpx -fsycl`.

**Windows syntax** — invocation is `icx`:

```text
[invocation] [option] [@response_file] file1 [file2 ...] [/link linker_option]
```

Files: C or C++ source (`.c`, `.cc`, `.cpp`, `.cxx`, `.i`), assembly (`.asm`), object (`.obj`), static library (`.lib`). The optional `response_file` lists compiler options to include during compilation.

**Default Operation.** Many options are on by default — these examples include **O2** (and other defaults):

```bash
icpx main.c        # Linux
icx main.c         # Windows
```

**Configuration file** options override competing defaults (an `O3` in the configuration file replaces the default `O2`). NOTE: default `.cfg` files are **not** valid for the compiler — use `-config<name>`; `<name>` can be a file in the `bin` directory or a full path.

Precedence, lowest to highest: defaults → configuration file → options in the command line **environment variable** → options on the command line; so `O1` beats competing defaults, configuration files, and the environment variable. Certain `#pragma` statements override command-line options: a function preceded by `#pragma optimize("", off)` has optimization turned off even when `O2` is the default, `O3` is in the configuration file, and `O1` is on the command line for the rest of the program.

**Competing options** are read **left to right**; the one **furthest to the right** is used:

```bash
icpx –xSSSE3 main.c file1.c –xSSE4.2 file2.c      # Linux
icx /QxSSSE3 main.c file1.c /QxSSE4.2 file2.c     # Windows
```

`[Q]xSSSE3` (or `O1`) and `[Q]xSSE4.2` (or `O2`) are two forms of the same option, only one of which can be used; the last wins. All command-line options compile **each** file. Rare exception: the `-x` type option on Linux:

```bash
icpx -x c file1 -x c++ file2 -x assembler file3
```

**Options with arguments.** Options can be a single letter (such as `E`). `O` takes a single-value argument determining the optimization degree; others require at least one argument and can accept multiple. For most such options the compiler warns if the option and argument are not recognized — `O9` warns, is ignored, and compilation proceeds. `O` does not require an argument, but `I` requires one identifying the directory to add to the include file search path; used without an argument, the compiler will not finish its compilation.

**Other forms.** Some options toggle via the negation convention; e.g. `[Q]ipo` includes **`-no-ipo`** (Linux) and **`/Qipo-`** (Windows).

**Option categories:** Advanced Optimization; Code Generation; Compatibility; Compiler Diagnostics; Component Control (**Not available for device compilation.**); Data; Floating Point; Inlining; Interprocedural Optimizations (IPO); Language; Linking/Linker; Miscellaneous; Offload Compilation, OpenMP, and Parallel Processing; OpenMP and Parallel Processing; Optimization; Optimization Report; Output; Preprocessor.

### Specify Compiler Files

**Include files** are searched in this order: (1) directories specified by the `I` option, (2) directories specified in the environment variables, (3) default include directories. `-nostdinc` (Linux) or `X` (Windows) removes the default directories from the search path:

```bash
icpx -nostdinc -I/alt/include prog1.cpp     # Linux
icx /X /I\alt\include prog1.cpp             # Windows
```

**Assembly files** — `-S` with `-o` (Linux) or `/Fa` (Windows) names the assembly file; these generate `myasm.s` (Linux) or `myasm.asm` (Windows):

```bash
icpx -S -o myasm.s x.cpp     # Linux
icx /Famyasm x.cpp           # Windows
```

**Object files** — `-c` with `-o` (Linux) or `/Fo` (Windows) names the object file; these generate `myobj.o` (Linux) or `myobj.obj` (Windows):

```bash
icpx -c -o myobj.o x.cpp     # Linux
icx /Fomyobj x.cpp           # Windows
```

### Convert Projects to Use a Selected Compiler

`ICProjConvert<version>.exe` transforms Intel® C++ projects into Microsoft Visual C++ projects, or vice versa:

```text
ICProjConvert<version>.exe <sln_file | prj_files> </VC[:"VCtoolset name"] | /IC[:"ICtoolset name"]> [/q] [/nologo] [/msvc] [/s] [/f]
```

`version` — ICProjConvert version number, values **191 or 192**; `sln_file` — path to the solution file to be modified; `prj_files` — space separated project files (or wildcard) to be modified; `/VC` — convert to the Microsoft Visual C++ project system (with `VCtoolset name` **v142** for Visual Studio 2019 or **v143** for Visual Studio 2022); `/IC` — convert to the Intel® C++ project system (with `ICtoolset name` such as **Intel C++ Compiler 2021.1**; supported values may differ by integration version); `/q` — quiet mode (all information messages except errors hidden); `/nologo` — suppress the startup banner; `/msvc` — set the compiler to Microsoft Visual C++; `/s` — search project files through all subdirectories; `/f` — force an update even for an unsupported type or unsupported properties; `/?` or `/h` — help.

```bat
ICProjConvert<version>.exe *.icproj /s /VC
```

> NOTE: if you uninstall the compiler, `ICProjConvert<version>.exe` remains in `Program Files (x86)\Common Files\Intel\shared files\ia32\Bin` and can transform Intel® C++ projects back into Microsoft Visual C++.

### Use Eclipse

On Linux the compiler integrates with Eclipse and C/C++ Development Tooling (CDT) to develop, build, and debug projects.

> NOTES: Eclipse and CDT are **not bundled**. Installed via `sudo sh ./<installer>.sh` → open with `sudo ./eclipse` (root); via `sh ./<installer>.sh` → `./eclipse` (current user); opening as current user after installing as root means the integration will not be available.

- **Add the Compiler to Eclipse** (manual plug-in install only) via **Help > Install New Software**: install the `.zip` under `<install_dir>/compiler/<version>/share/ide_support/eclipse/compiler` named `com.intel.compiler` (C/C++) or `com.intel.dpcpp.compiler` (SYCL), selecting **Intel® Software Development Tools > Intel® C++ Compiler Integration** (C/C++) or **Intel® oneAPI DPC++ Compiler Integration > Intel® oneAPI DPC++ Compiler Integration** (SYCL), then restart Eclipse.
- **Multiversion support:** latest used by default; select at project Properties > **C/C++ Build > Settings** > **Intel(R) oneAPI DPC++ Compiler** tab (DPC++) or **Intel® C++ Compiler** tab (C++) > version row > **Use Selected** / **Use Latest** > **Apply**. An Eclipse-specified compiler environment overrides all other environment specifications.
- **New project:** File > New > Project... > **C/C++ Project** > type > name (`hello_world`) > **Project Type > Executable > Hello World C++ Project** (C++) or **Hello World DPC++ Project** (DPC++) > **Toolchains > Intel C++ Compiler** (C++) or **Intel(R) oneAPI DPC++ Compiler** (DPC++); **Debug** and **Release** are created by default. **Help > Cheat Sheets** lists cheat sheets (Intel cheat sheets under **Intel(R) C++ Compiler**).
- **Set options** at **Properties > C/C++ Build > Settings > Tool Settings** under **Intel C Compiler** / **Intel C++ Compiler** / **Intel C++ Linker** (C++) or **Intel® oneAPI DPC++ Compiler** / **Intel® oneAPI DPC++ Linker** (DPC++); unlisted options go in **C/C++ Build Settings > Settings > \<Compiler\> > Command Line > Additional Options**. **Exclude sources:** **Resource Configurations > Exclude from build**. **Build/run:** **Project > Build Project**; **Run As > Local C/C++ Application**. **Error Parser** (default): **C/C++ Build > Settings > Error Parsers** — **Intel C++ Error Parser** checked, **CDT Visual C Error Parser** / **Microsoft Visual C Error Parser** unchecked.
- **Project types:** **Executable**, **Shared Library**, **Static Library**, **Makefile** (choose Makefile if one exists). **Export makefiles:** **File > Export > General > File system > Next** > check `hello_world` and `Release` and all sources > **Browse** to a directory > **Finish** (**Create directory structure for files** required; files such as `hello_world.o` may be deselected).

```text
**** Build of configuration Debug for project hello_world ****
make all
Building file: ../src/hello_world.cpp
Invoking: Intel C++ Compiler
icpx -g -O0 -MMD -MP -MF"src/hello_world.d" -MT"src/hello_world.d" -c -o "src/hello_world.o"
"../src/hello_world.cpp"
Finished building: ../src/hello_world.cpp

Building target: hello_world
Invoking: Intel C++ Linker
icpx -O0 -o "hello_world" ./src/hello_world.o
Finished building target: hello_world

Build Finished. 0 errors, 0 warnings.
```

```text
**** Build of configuration Debug for project DPCPPhelloworld ****
make all
Building file: ../main.cpp
Invoking: Intel(R) oneAPI DPC++ Compiler
icpx -fsycl -g -Wall -O0 -I/home/sys_idebuilder/eclipse-workspace/DPCPPhelloworld -MMD -MP -c -o
"main.o" "../main.cpp"
Finished building: ../main.cpp

Building target: DPCPPhelloworld
Invoking: Linker
icpx -fsycl -o "DPCPPhelloworld" ./main.o        -lsycl -lOpenCL
Finished building target: DPCPPhelloworld

Build Finished. 0 errors, 0 warnings.
```

```text
rm -rf    ./new_source_file.o     ./new_source_file.d    hello_world

Building file: ../new_source_file.c
Invoking: Intel C++ Compiler
icx -O2 -MMD -MP -MF"new_source_file.d" -MT"new_source_file.d" -c -o "new_source_file.o" "../
new_source_file.c"
Finished building: ../new_source_file.c

Building target: hello_world
Invoking: Intel C++ compiler
icx -o "hello_world" ./new_source_file.o
Finished building target: hello_world
```

- **Use Intel Libraries with Eclipse** via **Project > Properties > C/C++ Build > Settings > Intel C/C++ Compiler > Performance Library Build Components** (C++) or **Intel® oneAPI DPC++ Compiler > Performance Library Build Components** (DPC++):

| Property | Values |
|---|---|
| Use Intel® oneAPI Data Analytics Library (oneDAL) | `None` (disable); `Use threaded Intel® oneDAL`; `Use non-threaded Intel® oneDAL` |
| Use Intel® Integrated Performance Primitives Libraries | `None` (disable); `Use main libraries set`; `Use non-pic version of libraries` (no position-independent code); `Use main libraries and cryptography library` |
| Use Intel® oneAPI Math Kernel Library | `None` (disables); `Use threaded Intel® oneMKL library`; `Use non-threaded Intel® oneMKL library`; `Use Intel® oneMKL Cluster and sequential Intel® oneMKL libraries` — **only available for Intel C Compiler or Intel C++ Compiler** |
| Use Intel® oneAPI Threading Building Blocks | Enables the library and brings in the associated headers |

### Use Microsoft Visual Studio

Develops C++ or DPC++ applications (static library `.LIB`, dynamic link library `.DLL`, main executable `.EXE`). **Unsupported project types:** Class Library; CLR Console Application; CLR Empty Project; Windows Forms Application; Windows Forms Control Library.

- **Select the compiler:** set **Platform Toolset** (`Configuration Properties > General`) to `<compiler selection>`, or use **Project > Intel Compiler > Use Intel oneAPI DPC++/C++ Compiler**. NOTE: **Intel(R) oneAPI DPC++ Compiler** invokes `icx-cl -fsycl`; **Intel C++ Compiler \<major version\>** (example 2021) invokes `icx`; **Intel C++ Compiler \<major.minor\>** (example 19.2) invokes `icl`. Options at **Project > Properties > C/C++ > Command Line > Additional Options**; rebuild with **Build > Project only > Rebuild** / **Build > Rebuild Solution**; revert via right-click > **Intel Compiler > Use Visual C++**.
- **MSVC runtime environment:** **Configuration Properties > Debugging > Enable Intel® oneAPI DPC++ Compiler Runtime Environment** = `Yes` (or **Project > Enable Intel® oneAPI DPC++ Compiler Runtime Environment** / **Enable DPC++ Runtime Environment** for all configurations). Verify with **C/C++ > General > Suppress Startup Banner** = `No`:

```text
Intel(R) oneAPI DPC++/C++ Compiler for applications running on XXXX, Version XX.X.X
```

- **Versions:** **Tools > Options > Intel Compilers and Libraries > \<compiler\> > Compilers** (`<compiler>` = `C++` or `DPC++`) > **Selected Compiler** (default `<Latest>`); **Environment** is only available for `icx`. Use `rel_intelc` / `debug_intelc` configurations for Intel-specific options.
- **Base Platform Toolset:** defaults to the toolset environment of the Visual Studio version in use; values **v143** (Microsoft Visual Studio 2022), **v142** (Microsoft Visual Studio 2019); lists installed non-Intel toolsets only. Set at **Configuration Properties > General > Intel Specific > Base Platform Toolset**, or:

```bat
Msbuild.exe myproject.vcxproj /p:BasePlatformToolset=v142
Msbuild.exe myproject.vcxproj /p:PlatformToolset="Intel C++ Compiler 2021" /p:BasePlatformToolset=v142
```

The first builds with the Visual Studio 2019 environment when **Platform Toolset** is already the compiler; the second also sets **Platform Toolset** to the compiler.

- **Intel® Libraries** (oneDAL, Intel® IPP, oneTBB, oneMKL; Intel® C++, Intel® oneAPI DPC++, Microsoft Visual C++* project types) at **Project > Properties > Configuration Properties > Intel Libraries for oneAPI**:
  - **Use oneDAL:** `No`; `Default Linking Method` (parallel dynamic oneDAL libraries); `Multi-threaded Static Library` (parallel static); `Single-threaded Static Library` (sequential static); `Multi-threaded DLL` (parallel dynamic); `Single-threaded DLL` (sequential dynamic).
  - **Use Intel® IPP:** `No`; `Default Linking Method` (dynamic Intel® IPP libraries); `Static Library`; `Dynamic Library`.
  - **Use oneTBB:** `No`; `Use oneTBB` = `Yes`; `Instrument for use with Analysis Tools` = `Yes` to analyze release mode (not required for debug).
  - **Use oneMKL:** `No`; `Parallel`; `Sequential`; `Cluster`. The target platform of an Intel® oneAPI DCP++ project `[sic]` is **x64**, so a final **Use interface** selection appears: if selected, the **ilp** oneMKL libraries and **`MKL_ILP64`** preprocessor definition are added to the command lines; otherwise the **lp** libraries are used.
  - Standalone versions at **Tools > Options > Intel Compilers and Libraries > Intel Libraries for oneAPI** (`oneDAL`, `Intel IPP`, `oneTBB`, `oneMKL`; `Reset All` = latest libraries, the default). **MPI:** **Use oneMKL** = `Cluster`, **Use MPI Library** = **Intel® MPI Library** or **MS-MPI**, then build.
- **PGO** at **Tools > Intel Compiler > Insturmented Profile Guided Optimization...** `[sic]`. Phase 1 - Instrument (option shown in Compiler Options): **Enable Function Ordering in the optimized application**; **Enable Static Data Layout in the optimized application**; **Instrument with guards for threaded application** (produces a static profile information file `.spi`, increasing parallel-build time); deselect to skip. Phase 2 - Run Instrumented Application(s); Phase 3 - Optimize with Profile Data.
- **HWPGO** at **Tools > Intel Compiler > Hardware Profile-Guided Optimization**. Phase 1 - Generate Application uses **`/fprofile-sample-generate`**: `/fprofile-sample-generate`, `=keep-all-opt`, `=med-fidelity`, `=max-fidelity` (the default has no optimization impact and equals `=keep-all-opt`). Phase 2 - Hardware Profiling creates a PMU-based profile via the **SEP** tool of the Intel® VTune™ Profiler and LLVM profile data files via **`llvm-profgen`**; profile types `Execution Frequency`, `Execution Frequency and Branch Mispredict`. Phase 3 - Optimize with Profile Data (each phase can be deselected to skip); **Profile Directory** (Edit/Browse), **Show this dialog next time**, **Save Settings**, **Run**, **Cancel**.
- **Code Coverage** at **Tools > Intel Compiler > Code Coverage...**: Phase 1 - Instrument (**Instrument with guards for threaded applications**), Phase 2 - Run Instrumented Application(s) (**Application Invocations...**), Phase 3 - Generate Report (**Settings...**); HTML report plus text/XML export. Settings (**Tools > Options > Intel Compilers and Libraries > Code Coverage**, and Code Coverage Settings) — **Codecov Options:** Additional Options (passed to `Codecov.exe`), Ignore Object Unwind Handlers, Show Execution Counts, Treat Partially-covered Code As Fully-covered; **Profmerge Options:** Suppress Startup Banner, Verbose, Additional Options (passed to `Profmerge.exe`), Dump Profile Information, Exclude Functions (comma `,` separated; `.` is a wildcard in function names).
- **Optimization Reports:** enable at **Configuration Properties > C/C++ > Diagnostics [Intel C ++]** with a non-default **Optimization Diagnostics Level** / **Optimization Diagnostics Phase** / **Optimization Diagnostics Routine**, then build → **Compiler Optimization Report** (phases **PGO, LNO, PAR, VEC, Offload (Linux* only), OpenMP*, CG**; grouped by loops or flat), **Compiler Inline Report** (IPO phase; **Just My Code**; right-click > **Intel Compiler > Show Inline report for \<function name\>** / **Show where \<function name\> in inlined** `[sic]`), and editor annotations. Viewing settings at **Tools > Options > Intel Compilers and Libraries > Optimization Reports**: Always Show Compiler Inline Report; Always Show Compiler Optimization Report (higher priority; if both True this window has focus by default); Show Optimization Notes in Text Editor Margin; Collapse by Default; Show Optimization Notes; **Site** — Caller Site, Callee Site, Caller and Callee Sites.

## Compiler Reference

This section contains compiler reference information, for example compiler options, compiler limits, and libraries.

### C/C++/SYCL Calling Conventions

Calling conventions set the rules on how arguments are passed to a function and how values are returned from it.

**Linux:**

| Calling Convention | Compiler Option | Description |
|---|---|---|
| `__attribute((cdecl))` | None | Default for C/C++/SYCL programs; can be specified on a function with variable arguments |
| `__attribute((stdcall))` | None | Arguments passed on the stack; cannot be specified on a function with variable arguments |
| `__attribute__((regcall))` | `-regcall` makes `__regcall` the default calling convention for functions in the compilation, unless another calling convention is specified on a declaration | As many arguments as possible passed in registers; registers used whenever possible to return values. Ignored on a function with variable arguments |
| `__attribute__((vectorcall))` | None | A function passing vector type arguments should use vector registers |

**Windows:**

| Calling Convention | Compiler Option | Description |
|---|---|---|
| `__cdecl` | `/Gd` | Default for C/C++/SYCL programs; can be specified on a function with variable arguments |
| `__stdcall` | `/Gz` | Standard calling convention used for Win32 API functions |
| `__fastcall` | `/Gr` | Arguments are passed in registers rather than on the stack |
| `__regcall` | `/Qregcall` makes `__regcall` the default calling convention for functions in the compilation, unless another calling convention is specified on a declaration | As many arguments as possible passed in registers; registers used whenever possible to return values. Ignored on a function with variable arguments. For the Intel-compatible vector functions ABI, download the Vector Function Application Binary Interface PDF |
| `__thiscall` | None | Default calling convention used by C++ member functions that do not use variable arguments |
| `__vectorcall` | `/Gv` | A function passing vector type arguments should use vector registers |

**The `__regcall` calling convention** is unique to the Intel oneAPI DPC++/C++ Compiler. Place the keyword before a function declaration:

```c
__attribute__((regcall)) foo (int I, int j);   // Linux
__regcall int foo (int i, int j);              // Windows
```

**Available `__regcall` registers** (all can pass/return values except those reserved by the compiler; used in the order shown):

| Register Class/Architecture | Intel® 64 for Linux | Intel® 64 for Windows |
|---|---|---|
| GPR | RAX, RCX, RDX, RDI, RSI, R8, R9, R10, R11, R12, R14, R15 | RAX, RCX, RDX, RDI, RSI, R8, R9, R11, R12, R14, R15 |
| FP | ST0 | ST0 |
| MMX | None | None |
| XMM | XMM0 - XMM15 | XMM0 - XMM15 |
| YMM | YMM0 - YMM15 | YMM0 - YMM15 |
| ZMM | ZMM0 - YMM15 `[sic]` | ZMM0 - YMM15 `[sic]` |

**`__regcall` data type classification** (parameters and return values pass in the registers of these classes):

| Type (Signed and Unsigned) | Intel® 64 |
|---|---|
| `bool`, `char`, `int`, `enum`, `_Decimal32`, `long`, pointer | GPR |
| `short`, `__mmask{8,16,64}` | GPR |
| `long long`, `__int64` | GPR |
| `_Decimal64` | GPR |
| `long double` | FP |
| `float`, `double`, `float128`, `_Decimal128` | XMM |
| `__m128`, `__m128i`, `__m128d` | XMM |
| `__m256`, `__m256i`, `__m256d` | YMM |
| `__m512`, `__m512i`, `__m512d` | ZMM |
| complex type, struct, union | See Structured Data Type Classification Rules |

> NOTES: all types assigned to XMM, YMM, or ZMM in a non-SSE target are passed in the **stack**; for structured types, GPR class classification is used. Types smaller than registers of their class are passed in the lower part of those registers (`float` in the lower four bytes of an XMM register).

**`__regcall` structured data type classification rules.** Structures/unions and complex types are classified similarly to the x86_64 ABI, with these exceptions: no limitation on the overall size of a structure; the register classes for basic types are given in Data Type Classifications.

## Option / API quick table

| Name | Purpose | Key values / default | Notes |
|---|---|---|---|
| `icx` | C driver (Linux); recommended Windows driver | Clang-style | Compiles a C++ source file as C++; link C object files |
| `icx-cc` | Microsoft-compatible variant of `icx` | Clang-style | Listed as C driver for Windows and Linux |
| `icpx` | C++ driver (Linux) | Clang-style | Recommended default C++ driver; compiles a C source file as C++ |
| `icx-cl` | MSVC-style driver | MSVC-style | Microsoft-compatible variant for C/C++; **experimental on Linux**, requires the Microsoft Visual Studio Package |
| `icpx -fsycl` / `icx-cl -fsycl` | Invoke the SYCL compiler | — | `dpcpp` is deprecated; use `-fsycl` with the C++ driver |
| `icl` | Invoked by **Intel C++ Compiler \<major.minor\>** (e.g. 19.2) | — | Intel® C++ Compiler Classic |
| `-help` | Check whether a Clang option is supported | — | Clang options supported but undocumented |
| `-E` / `-c` / `-S` | Produce preprocessed (`file.i`) / object / assembly file | — | `-c` renames with `-o` or `/Fo`; `-S` renames with `-s` `[sic]` or `/Fa` |
| `-o` | Rename output object/assembly/executable (Linux) | — | Windows: `/Fo`, `/Fa`, `/Fe` |
| `-I` | Add directory to include search path | — | **Requires an argument**; without one the compiler will not finish compilation |
| `-nostdinc` / `X` | Remove default include directories | — | Search order: `I` dirs, environment-variable dirs, defaults |
| `O` (`O1`, `O2`, `O3`, `O9`) | Optimization level | `O2` among the defaults | `O9` warns, is ignored, compilation proceeds |
| `-x` | Select language for subsequent files (Linux) | `c`, `c++`, `assembler` | Per-file language; rare exception to the all-files rule |
| `[Q]x...` | Target ISA options | `-xSSSE3`, `-xSSE4.2` / `/QxSSSE3`, `/QxSSE4.2` | Furthest-right competing option wins |
| `-no-ipo` / `/Qipo-` | Negation form of `[Q]ipo` | — | Negation convention toggles the option off |
| `-regcall` / `/Qregcall` | Make `__regcall` the default calling convention | — | Overridden by an explicit convention on a declaration |
| `/Gd`, `/Gz`, `/Gr`, `/Gv` | Windows calling convention defaults | `__cdecl`, `__stdcall`, `__fastcall`, `__vectorcall` | `/Gd` is the default |
| `-config<name>` | Use a configuration file | `<name>` in `bin` or full path | Default `.cfg` files are **not** valid |
| `@response_file` | Text file listing compiler options | — | Accepted on Linux and Windows |
| `/link` | Introduce linker options (Windows) | — | Linker-specific options for `icx-cl -fsycl` must follow `/link` |
| `/fprofile-sample-generate` | HWPGO phase 1 | `=keep-all-opt`, `=med-fidelity`, `=max-fidelity` | Default equals `=keep-all-opt`; no optimization impact |
| `MKL_ILP64` | Preprocessor definition added when **Use interface** oneMKL is selected in Visual Studio | — | Otherwise the `lp` oneMKL libraries are used |
| `-qtbb`, `-tbb`, `/Qtbb` | TBB linkage | — | Requires oneTBB install |
| `-mkl`, `-qmkl`, `-qmkl-ilp64`, `/Qmkl`, `/Qmkl-ilp64` | MKL linkage | — | Requires oneMKL install |
| `-daal`, `-qdaal`, `/Qdaal` | oneDAL linkage | — | Requires oneDAL install |
| `-ipp`, `-qipp`, `/Qipp` | Intel® IPP linkage | — | Requires Intel® IPP install |

## Code examples

All examples appear inline in the topic sections; canonical command forms:

```text
{compiler driver} [option] file1 [file2...]
icpx hello-world.cpp ; icpx -fsycl hello-world.cpp                # Linux C++ / SYCL
icx main.c                                                        # Windows default-operation example
make -f yourmakefile ; nmake /f your_project.mak CPP=icx LINK32=link
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx …          # Linux
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -GNinja …   # Windows
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx .. && cmake --build . && ./simple
ICProjConvert<version>.exe *.icproj /s /VC
Msbuild.exe myproject.vcxproj /p:BasePlatformToolset=v142
icpx -nostdinc -I/alt/include prog1.cpp ; icpx -S -o myasm.s x.cpp ; icpx -c -o myobj.o x.cpp
```

## Gotchas & failure modes

- **`dpcpp` is deprecated** — use `-fsycl` with the C++ driver.
- **Unset environment variables** cause unpredictable behavior and load-time failures (`libimf.so: cannot open shared object file`); re-source `setvars.sh`/`setvars.bat` every session.
- **`setvars.bat` prepends DLL directories to `Path`**, preceding the original Windows `Path`; same-named files can collide.
- **Default `.cfg` files are not valid** — use `-config<name>`.
- **Precedence:** defaults < configuration file < command-line environment variable < command line; `#pragma optimize("", off)` overrides even a winning command-line option for that function.
- **Competing options:** rightmost wins; all command-line options apply to every file except with the Linux `-x` type option.
- **Missing products fail compilation:** `-qtbb`/`-tbb`/`/Qtbb`, `-mkl`/`-qmkl`/`-qmkl-ilp64`/`/Qmkl`/`/Qmkl-ilp64`, `-daal`/`-qdaal`/`/Qdaal`, `-ipp`/`-qipp`/`/Qipp` need oneTBB/oneMKL/oneDAL/Intel® IPP installed.
- **`I` without an argument** means the compiler will not finish its compilation; **`O9`** warns, is ignored, and compilation proceeds (silent-success trap).
- **Windows accepts `-` or `/`; Linux only `-`.** Linux-only `-x` type options and `-nostdinc` have `/Q...`/`/X` counterparts on Windows.
- **CMake minimums:** Visual Studio 2022 default CMake < 3.23.0 needs CMake 3.25+ and a `CMakeSettings.json` update; Linux needs 3.23.5+; Ninja is the supported Windows generator; GNU-like drivers (`icx-cc`, `icpx`) on Windows need a newer CMake.
- **`add_sycl_to_target()` with no sources** adds SYCL compilation to all sources and can greatly increase compile time.
- **Eclipse:** root/current-user mismatch disables the integration; Eclipse and CDT are not bundled; right-pane files deselected during makefile export are silently omitted.
- **`icx-cl` on Linux is experimental** (requires the Microsoft Visual Studio Package); unsupported MSVC project types: Class Library, CLR Console Application, CLR Empty Project, Windows Forms Application, Windows Forms Control Library.
- **MSVC selection trap:** **Intel C++ Compiler \<major\>** (e.g. 2021) invokes `icx`; **Intel C++ Compiler \<major.minor\>** (e.g. 19.2) invokes `icl`. Combo boxes default to `<Latest>`.
- **Linker options with `icx-cl -fsycl`** must follow `/link`.
- **MPI support** requires **Use oneMKL = Cluster** plus **Use MPI Library** = Intel® MPI Library or MS-MPI.
- **oneMKL interface selection in Visual Studio** adds `MKL_ILP64` and the `ilp` libraries only when **Use interface** is selected; otherwise `lp` libraries are used.
- **`__regcall` is ignored on functions with variable arguments**; on a non-SSE target all XMM/YMM/ZMM-classified types are passed on the stack.
- **Deprecated:** Intel® Inspector (Thread Checking), Intel® Trace Analyzer and Collector (Trace Analyzing and Collecting).

## Source map

- Doc validity, key features, key sections; Clang option support; notices (dpcpp deprecation, experimental features, no macOS*); additional resources — pp. 1, 6–7
- Introduction (architecture/OS/standards support) — p. 7; Feature Requirements — p. 8; Get Help and Support — pp. 8–9; Related Information — pp. 9–10
- Specify Component Locations — pp. 10–12; Invoke the Compiler — pp. 12–14; Command Line on Windows — pp. 14–15
- File Extensions — pp. 15–16; Makefiles — pp. 16–17; CMake — pp. 17–20; Use Compiler Options — pp. 20–23; Specify Compiler Files — pp. 23–24; Convert Projects — pp. 24–25
- Use Eclipse — pp. 25–31; Use Microsoft Visual Studio — pp. 31–48
- Compiler Reference intro — p. 49; C/C++/SYCL Calling Conventions, `__regcall`, registers, data type classification — pp. 49–51
