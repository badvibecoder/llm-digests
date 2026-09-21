## 17 · Build system, environment setup and compilation

### Installation layouts

| Layout | Root |
|---|---|
| Component (pre-2024.0) | `/opt/intel/oneapi/`; Windows `C:\Program Files (x86)\Intel\oneAPI\`; each component owns `env/vars.sh` / `env\vars.bat`; `latest` = newest version |
| Unified (2024.0+) | `/opt/intel/oneapi/<toolkit-version>/` or `C:\Program Files (x86)\Intel\oneAPI\<toolkit-version>\`; components merged into `bin`, `lib`, `include`, `share`; `env/vars.sh` unused |

`setvars` defines only `ONEAPI_ROOT`; `oneapi-vars` defines the common vars (`LD_LIBRARY_PATH` → `$ONEAPI_ROOT/lib`, `CPATH` → `$ONEAPI_ROOT/include`). Both set/overwrite `ONEAPI_ROOT`, which since 2024.0 the installer does not add (define in shell init or `/etc/environment`).

### setvars.sh / oneapi-vars.sh (Linux)

```bash
source <install-dir>/setvars.sh                       # Component layout
source <install-dir>/<toolkit-version>/oneapi-vars.sh # Unified layout
bash -c 'source <install-dir>/setvars.sh ; exec csh'  # non-POSIX shell (csh)
```

- Verify `SETVARS_COMPLETED=1` (Windows: `set | find "SETVARS_COMPLETED"`); changes are session-local.
- Windows: `setvars.bat`; unified `<install-dir>\<toolkit-version>\oneapi-vars.bat`; PowerShell `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`.
- **Rule.** Top-level script refuses a second invocation in one session (PATH/CPATH overflow); `--force` overrides.
- **Gotcha.** `--force` pollutes args on bash 3.x/4.x (`echo ${@}` → `advisor=latest ccl=latest ...`). Workaround: `SETVARS_ARGS="--force" source <install-dir>/setvars.sh`. Fine on bash 5.x, zsh, ksh, dash.
- `--config=file` limits components/selects versions; **only `setvars` supports it, not `oneapi-vars`**. Unrecognized args forward to every component `vars` script: common `ia32`, `intel64` (target arch); Windows also `vs2017`, `vs2019`, `vs2022`.

```bash
source <install-dir>/setvars.sh --config="full/path/to/your/config.txt"
```

### Config file format

Newline-delimited `key=value`; `key` = component folder under `$ONEAPI_ROOT`; `value` = version dir, `latest`, or `exclude` (e.g. `mkl=1.1`, `dldt=exclude`). Last duplicate key wins. `default=exclude` excludes ALL `env/vars.sh` except keys listed after it.

**Gotcha.** Config files require `setvars.sh` in the Component layout; `oneapi-vars.sh` (Unified) does not support them.

### Modulefiles (Linux only)

Needs Tmod ≥ 4.2, Tcl ≥ 8.4, Lmod ≥ 8.7.44; do not mix with `setvars`.

```bash
module use /opt/intel/oneapi/2024.1/etc/modulefiles/
module load tbb ; module load compiler
module avail ; module list ; module unload tbb
```

- Auto-load `prereq` modulefiles: `module config auto_handling 1`, `export MODULES_AUTO_HANDLING=1` (1 enables, 0 disables), `--auto`. **Gotcha.** `HINT: the following module must be loaded first` = auto-load off.
- 2024.0+ Unified: `/opt/intel/oneapi/<toolkit-version>/etc/modulefiles` (add to `MODULEPATH`) ; `modulefiles-setup.sh` gathers versioned symlinks: `--output-dir=<path>`, `--ignore-latest`. Compiler modulefiles: `/opt/intel/compiler/<component-version>/etc/modulefiles/` (2024.0+), `.../modulefiles/` (≤2023).

### CMake integration

- Config discovery: Linux/macOS system `/usr/local/lib/cmake`, user `~/lib/cmake`; Windows registry `HKEY_LOCAL_MACHINE\Software\Kitware\CMake\Packages\`.
- Components with CMake support: DPC++ Compiler; IPP; MPI; oneCCL; oneDAL; oneDNN; oneDPL; oneMKL; oneTBB; oneVPL. Use as system libraries: `find_package(tbb)` makes the project use the oneTBB package; the source shows only this component-level `find_package(<component>)` pattern.
- Debug builds: `CMAKE_BUILD_TYPE=Debug` plus `-O0` appended to `CMAKE_CXX_FLAGS_DEBUG`:

```cmake
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
```

### Compiler drivers

| Driver | Host | Options |
|---|---|---|
| `icpx -fsycl` | Linux | GCC-style `-` |
| `icx-cl -fsycl` | Windows | MSVC `/`, Visual Studio |
| `dpcpp` | both | legacy alias used in examples |

- **Rule.** Single-source: one `icpx -fsycl` command generates both host and device code.
- OpenMP offload without SYCL: `icpx -fiopenmp -fopenmp-targets=<arch>`; with SYCL add `-fsycl` (Windows: `icx`/`icx-cl` with `/Qiopenmp /Qopenmp-targets:<arch>`). OpenMP target C: `icx -fiopenmp -fopenmp-targets=spir64 code.c`; Fortran: `ifx -fiopenmp -fopenmp-targets=spir64 code.f90`.

### Compilation flow

| Flow | Device code path | Runtime behavior |
|---|---|---|
| Traditional (host-only) | none | front end → back end object → linker → executable |
| JIT (default for device) | device → SPIR-V embedded in fat binary | runtime translates SPIR-V per available device |
| AOT | device → SPIR-V → device object at compile time for a specified device | device executable embedded, loaded directly |

- **Rule.** AOT requires the target device at compile time; faster start-up. JIT compiles at run time (slower start; large device code hurts). **Fact.** JIT is not supported for FPGA.
- Fat binary = host binary with embedded device code; host ELF (Linux)/PE (Windows), device SPIR-V (JIT) or executable (AOT). AOT executable formats: CPU ELF (Linux)/PE (Windows); GPU ELF (Windows, Linux); FPGA ELF (Linux)/PE (Windows). Fat Library/Object = archive/object with generic SPIR-V plus target-specific object code for multiple devices.

### Compiler option table

| Option | Purpose |
|---|---|
| `-fsycl` | enable SYCL single-source offload compilation |
| `-fsycl-targets=<target-list>` | select device code targets (AOT selection); comma-separated |
| `-Xs "<args>"` | pass arguments to the device compiler, e.g. `-device <name>` |
| `-fsycl-device-code-split=per_kernel` | split device code per kernel |
| `-foffload-static-lib=libstlib.a` | link a static library containing device code |
| `-fintelfpga` | enable FPGA compilation |
| `-Xshardware` | FPGA hardware (AOT) image; omit for emulation |
| `-fiopenmp`, `-fopenmp-targets=<arch>` | OpenMP offload |
| `-Xopenmp-target-backend "<args>"` | backend args for OpenMP offload |
| `-g`, `-O0` | debug info; disable optimizations |
| `-I${MKLROOT}/include`, `-c`, `-o` | include path, compile-only, output name |

**Fact.** Attested `-fsycl-targets` values: `spir64` (SPIR-V/JIT), `spir64_x86_64` (CPU AOT), `spir64_gen` (GPU AOT), `spir64_gen-unknown-unknown-sycldevice` (GPU AOT debug), `nvptx64-nvidia-cuda` (NVIDIA plugin).

### Canonical command lines

```bash
# JIT: compile and link in one step
icpx -fsycl vector_add.cpp
# CPU offload, then host-only build
icpx -fsycl simple-iota-dp.cpp -o simple-iota
icpx -g -o matrix_mul_omp src/matrix_mul_omp.cpp

# CPU AOT
icpx -fsycl -fsycl-targets=spir64_x86_64 \
  -Xs "-device <CPU optimization flags>" a.cpp b.cpp -o app.out
# -march=<instruction_set_arch>: sse42, avx2, avx512

# GPU AOT
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device <device name>" a.cpp b.cpp -o app.out
# GPU AOT via OpenMP offload
icpx -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device <device name>" a.cpp b.cpp -o app.out
ocloc compile --help      # list valid GPU device names/options

# GPU AOT debug (targets kbl)
dpcpp -g -O0 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
  -Xs "-device kbl -internal_options -cl-kernel-debug-enable -options -cl-opt-disable" myprogram.cpp

# Multiarchitecture
icpx -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda migrated.cpp -o migrated

# FPGA hardware (AOT); emulation = drop -Xshardware
icpx -fsycl -fintelfpga my_source_code.cpp -Xshardware
icpx -fsycl -fintelfpga my_source_code.cpp

# Static library with device code (dynamic-library device code NOT supported)
icpx -fsycl -c static_lib.cpp
ar cr libstlib.a static_lib.o
icpx -fsycl -c a.cpp
icpx -fsycl -foffload-static-lib=libstlib.a a.o -o a.exe

# oneMKL axpy: compile then link
icpx -fsycl -I${MKLROOT}/include -c axpy.cpp -o axpy.o
icpx -fsycl axpy.o -fsycl-device-code-split=per_kernel \
  "${MKLROOT}/lib/intel64"/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group \
  "${MKLROOT}/lib/intel64"/libmkl_intel_ilp64.a \
  "${MKLROOT}/lib/intel64"/libmkl_sequential.a \
  "${MKLROOT}/lib/intel64"/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL \
  -lpthread -lm -ldl -o axpy.out
```

**Rule.** `MKLROOT` must be set (`echo ${MKLROOT}`); source `setvars.sh`/`oneapi-vars.sh`, or set it to the folder containing `lib` and `include`.

### CPU flow

- Offload uses the OpenCL runtime plus oneTBB. Work-groups map to logical cores; work-items to CPU SIMD lanes. Verify runtime with `sycl-ls`; select with `cpu_selector` or `export ONEAPI_DEVICE_SELECTOR=cpu`. Use AOT/offline compilation for a specific CPU architecture.

| Env var | Values | Meaning |
|---|---|---|
| `DPCPP_CPU_CU_AFFINITY` | `close` \| `spread` \| `master` | thread affinity; `master` pins master thread only if set; default: not set; analogous to `OMP_PROC_BIND` |
| `DPCPP_CPU_SCHEDULE` | `dynamic` \| `affinity` \| `static` | oneTBB partitioner (`auto_`/`affinity_`/`static_partitioner`); default: `Dynamic` |
| `DPCPP_CPU_NUM_CUS` | integer | threads for kernel execution; `1` runs work-groups sequentially (debug); default: not set (oneTBB-determined) |
| `DPCPP_CPU_PLACES` | `sockets` \| `numa_domains` \| `cores` \| `threads` | affinity places; default: `cores`; analogous to `OMP_PLACES` |

### GPU flow

- Defaults to Level Zero runtime; OpenCL is selectable. Mapping: work-item → SIMD lane; subgroup → SIMD width / GPU EU thread; work-group → compute unit (Xe core / sub-slice); global NDRange → whole GPU.
- Setup: install Intel GPU drivers, add user to `video` group, verify with `sycl-ls` (two GPU entries with OpenCL + Level Zero drivers). AMD GPUs need the Codeplay oneAPI for AMD GPUs plugin (Linux only); NVIDIA GPUs need the oneAPI for NVIDIA GPUs plugin (Linux and Windows).
- Select with `gpu_selector`, `default_selector`, or `ONEAPI_DEVICE_SELECTOR=backend:device_type:device_num`. **Rule.** Prefer AOT over JIT for GPU start-up and specific-architecture targeting.
- Debug: `SYCL_UR_TRACE` (`1` basic trace, `2` all API traces, `-1` all of 2 plus more); `ZE_DEBUG` (any value; Level Zero APIs called/event info). OpenMP: `LIBOMPTARGET_DEVICETYPE` (`cpu`\|`gpu`), `LIBOMPTARGET_DEBUG=1`, `LIBOMPTARGET_INFO`; `OMP_TARGET_OFFLOAD=mandatory` forces offload instead of host fallback. `CL_OUT_OF_RESOURCES`: `PrintDebugMessages=1` and `NEOReadDebugKeys=1` print shared-local-memory size vs. hardware limit.

### Key gotchas

- `oneapi-vars.sh`/`.bat` does not support `--config`; config files require Component-layout `setvars`.
- A second `source setvars.sh` in a session is skipped; use `--force`, preferably via `SETVARS_ARGS`.
- Source `setvars` (or set `MKLROOT`) before compiling oneMKL code.
- FPGA supports AOT only (`-fintelfpga -Xshardware`); JIT is unavailable.
- AOT needs the target device at compile time; JIT binaries cannot add AOT targets later.
- Linking a dynamic library containing device code is unsupported; use `-foffload-static-lib`.
- Get valid `-Xs "-device ..."` GPU names from `ocloc compile --help`.
- Set `CMAKE_BUILD_TYPE=Debug` and add `-O0` to `CMAKE_CXX_FLAGS_DEBUG` for debuggable kernels.
- `icpx -fsycl` takes GCC-style `-` options; `icx-cl -fsycl` takes MSVC `/` options.
- `icc` modulefile removed in 2024.0; use `icx`/`icpx`.
