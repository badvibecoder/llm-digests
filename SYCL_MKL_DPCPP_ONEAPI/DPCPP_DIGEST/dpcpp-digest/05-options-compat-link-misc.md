---
chunk: 05-options-compat-link-misc
source: dpcpp-cpp-compiler_developer-guide-reference_2026.0
source_pages: 345-389
covers: Compatibility options, linking/linker options, miscellaneous options, deprecated and removed compiler options, display option information, alternate compiler options, portability and GCC-compatible warning options
---

# Compatibility, Linking, Miscellaneous, Deprecated, and Portability Compiler Options

> **Scope.** Intel® oneAPI DPC++/C++ Compiler options for linking/linker control, driver and
> miscellaneous behavior (help, version, intermediate files, sysroot, source type), and compatibility
> surfaces (deprecated/removed, alternate, GCC-/Microsoft-compatible option lists). The quick table is
> the per-option reference.

## Key facts

- **Host-only note.** Most options carry *"This option only applies to host compilation. When
  offloading is enabled, it does not impact device-specific compilation."* Such rows are marked `H`.
  Default linker: `ld` (Linux), `link` (Windows).
- `-static-intel` defaults to **ON**, except the Intel OpenMP* runtime (dynamic; override
  `-qopenmp-link=static`) and Intel® libraries under `shared` (override with `-static-intel`).
  `-fsycl` sets `/MD`, so `/MT` cannot be used.
- **Deprecated** options still work now but may not later; **removed** options make the compiler warn,
  ignore them, and continue compiling. Alternate Compiler Options are **not valid for SYCL**;
  portability and GCC-compatible warning options **do not apply for SYCL**. No cataloged figure is in
  pages 345–389.

## Option quick table

`H` = source adds the host-compilation-only note; `(T)` = truncated/garbled in source (see Gotchas).

| Option | Purpose / values | Default | Notes |
|---|---|---|---|
| *(name lost to page break)* | base-toolchain location | `OFF`: heuristics | H; `(T)` |
| `/vmv` | pointers to members, any inheritance type | `OFF` | Win; needs `/vmg` |
| `/Fn` (`F`) | stack reserve, bytes (dec or C-style hex, e.g. `/F0x1000`) | `OFF`: OS chooses | Win; H |
| `/fixed` | load at preferred base address only | `OFF` | Win; H |
| `-fortlib` | link Fortran libraries | `OFF` | Lin; link time only |
| `-fuse-ld=keyword` | non-default linker: `bfd`, `gold`, `lld`, `llvm-lib` | `ld` Lin / `link` Win | H; below |
| `-lstring` | find library `libstring` | `OFF`: standard dirs | Lin; H; order matters |
| `-Ldir` | search `dir` before standard dirs | `OFF` | Lin; H |
| `/LD`, `/LDd` | link as DLL, not `.exe` | `OFF` | Win; H; `d` = debug |
| `/link` | pass following options to linker | `OFF` | Win; H; Lin `-Wl`/`-Xlinker` |
| `/MD`, `/MDd` | multithreaded DLL runtime refs | `OFF`: static runtime | Win; H; `-fsycl` sets `/MD` |
| `/MT`, `/MTd` | multithreaded static runtime refs | `/MT` | Win; H; `-fsycl` bars `/MT` |
| `-nodefaultlibs` | no standard libraries | `OFF` | Lin; GNU compat; H |
| `-no-intel-lib[=library]` / `/Qno-intel-lib[:library]` | skip named/all Intel® libraries | `OFF`: heuristics | H; values below |
| `-nostartfiles` | no standard startup files | `OFF` | Lin; H |
| `-nostdlib` | no standard libraries or startup files | `OFF` | Lin; GNU compat; H |
| `-pie` / `-no-pie` | PIC in an executable | `-no-pie` | Lin |
| `-pthread` (also `-pthreads`) | use the pthread library | `OFF` | Lin; H; below |
| `-shared` | DSO instead of an executable | `OFF`: executable | Lin; H; needs `fpic` |
| `-shared-intel` | link Intel® libraries dynamically | `OFF`; opposite of `-static-intel` | Lin; H |
| `-shared-libgcc` | link libgcc dynamically | `-shared-libgcc` | Lin; opposite `-static-libgcc` |
| `-static` | no linking with shared libraries | `OFF` | Lin; H |
| `-static-intel` | link Intel® libraries statically | `ON` (two exceptions) | Lin; H; below |
| `-static-libgcc` | link libgcc statically | `OFF`: dynamic | Lin; traceback needs it |
| `-static-libstdc++` | link libstdc++ statically | `OFF`: dynamic | Lin; H |
| `-Tfilename` | linker reads link commands from a file | `OFF` | Lin; H |
| `-u symbol` | the specified symbol is undefined | `OFF`: standard rules | Lin |
| `-v [filename]` | display **and execute** tool commands | `OFF`; no file = version | Lin; space before `filename` |
| `-Wa,option1[,option2,...]` | pass options to assembler | `OFF` | Lin; H; ignored if not invoked |
| `-Wl,option1[,option2,...]` | pass options to linker | `OFF` | Lin; H; `= -Qoption,link,options` |
| `-Wp,option1[,option2,...]` | pass options to preprocessor | `OFF` | Lin; H; `= -Qoption,cpp,options` |
| `-Xlinker option` | one linker option, passed directly | `OFF` | Lin; H; below |
| `/Zl` | omit library names from object file | `OFF`: names included | Win; H |
| `-dryrun` | show tool commands, do not execute | `OFF`: executed silently | Lin |
| `-dumpmachine` | target machine and OS | `OFF` | Lin; no compile; H |
| `-dumpversion` | compiler version | `OFF` | Lin; no compile; H |
| `-fpreview-breaking-changes` | next-major breaking changes | `OFF` | Lin+Win; sets `__INTEL_PREVIEW_BREAKING_CHANGES` |
| `-help` / `/help` | all options, alphabetical | `OFF` | H; categories below |
| `/MP[processMax]` | parallel compile processes | `OFF`: single; no max = per CPU | Win; not linking/LTCG |
| `/nologo` | no compiler version information | `OFF` | Win; H |
| `-save-temps`, `-no-save-temps` / `/Qsave-temps`, `/Qsave-temps-` | save intermediate files | Lin `-no-save-temps`; Win `.obj` only | `/Qsave-temps` C++ only; SYCL `None` |
| `/showIncludes` | list include files, nested too | `OFF` | Win |
| `-sox[=keyword[,keyword]]` / `-no-sox` | embed build options in object/exe | `-no-sox`; `inline`, `profile`, `secure`, `secure-defines` | Lin; H |
| `--sysroot=dir` | root for headers and libraries | `Off`: defaults | Lin; gcc compat; below |
| `/Tcfilename` | file processed as C | `OFF` | Win; H |
| `/TC` | all/unrecognized as C | `OFF` | Win; H |
| `/Tpfilename` | file processed as C++ | `OFF` | Win |
| `TP` | *(deprecated)* | replacement `None` | Win; no detailed entry |
| `--version` | GCC-style version info | `OFF` | Lin; H |

## Compatibility options (p. 345; source heading not in extract)

- Unnamed option (`[unclear in source]`: name severed by the page break) sets the base-toolchain
  location; default `OFF` (heuristics); H. The only `gcc-toolchain` here is the Removed table's
  replacement for `gcc-name`/`gxx-name`.
- `vmv` (`/vmv`, Windows): pointers to members of any inheritance type; **`/vmg` is also required**.

## Linking or Linker Options

- `-fuse-ld`: `bfd`/`gold` Linux-only, `llvm-lib` Windows-only (Linux-only = gcc compatibility);
  Windows `/Qipo` auto-sets `-fuse-ld=lld`; see `flto`, `ipo`/`Qipo`.
- `-l`: search order standard dirs → `-L` dirs → the `-l` library; `-l` must follow the last object
  file it applies to. `/link` sends every following option straight to the linker.
- `-no-intel-lib` values: `libirc` Intel® C/C++ library; `libimf` Intel® oneAPI DPC++/C++ Compiler Math
  library; `libm` Windows-only = `libimf`; `libsvml` Intel® Short Vector Math library; `libirng`
  Random Number Generator library. Comma-separate; omitting `library` links none.
- `-pthread` also `-pthreads`; also set by `-fiopenmp`, `-qmkl`, `-debug=parallel`, `-fortlib`.
- `-shared`: needs `fpic` per object; links all libraries dynamically; Intel® libraries go dynamic too
  unless `-static-intel`.
- `-static-intel`/`-shared-intel` add library names at the linking driver command. `-static-intel` `ON`
  exceptions: (1) Intel OpenMP* runtime dynamic — prevent with `-qopenmp-link=static`; (2) Intel®
  libraries dynamic under `shared` — prevent with `static-intel` too. `-static-intel` +
  `-mcmodel=medium`/`-mcmodel=large` = **error**; a library with no static version = **diagnostic**;
  those `-mcmodel` values set `-shared-intel`.
- `-shared-libgcc` (default) overrides `-static`'s all-static behavior; `-static-libgcc` is needed for
  traceback (static libgcc prints backtraces); `-static` cannot statically link libraries without a
  static version.
- `-Wa`/`-Wl`/`-Wp`: not driver-processed, **ignored if that tool is not invoked**; `-Wl` ≡
  `-Qoption,link,options`; `-Wp` ≡ `-Qoption,cpp, options`; Windows `-Wl` = `/link`.
- `-Xlinker` passes arguments straight through; `-Xlinker -shared` passes only `-shared` with no
  shared-object linkage work; compound options repeat the flag (`-Xlinker -L -Xlinker $HOME/lib`).

## Miscellaneous Options

- `-fpreview-breaking-changes`: enabled changes become the next major release's defaults; with `-fsycl`
  the driver links `libsycl-preview`; the source NOTE inconsistently names
  `-fpreview-breaking-release` `[sic]`. `/MP`: no `processMax` = one process per effective OS
  processor; compilation only, not linking/LTCG.
- `-save-temps`/`/Qsave-temps`: names derive from the source file, in the working directory; the save
  form writes Linux `.o` / Windows `.obj` (source: "the `.obj` file object `.o` file is saved"
  `[sic: source garbled]`). `-no-save-temps` (Linux): `.o` goes to `/tmp`, deleted after `ld`,
  preprocessed file not kept; `/Qsave-temps-` (Windows): `.obj` not kept after linking, preprocessed
  file not kept. Only intermediates normally created during compilation are saved.
- `-sox` embeds data in each object file/assembly output; no keyword = options **and version** of the
  executable's objects; the executable grows slightly and strings land in its header. `inline` =
  functions inlined per object; `profile` = data when profile was used with PGO-enabling Clang options
  like `-fprofile-use`/`-fprofile-instr-use`; `secure` = strips directory names and their options;
  `secure-defines` = strips command-line `-D`.
- `--sysroot=dir`: `dir` holds copies of target libraries in corresponding subdirectories; with normal
  `/usr/include` and `/usr/lib`, `--sysroot=/mydir` searches `/mydir/usr/include` and
  `/mydir/usr/lib`. gcc compatibility; not Windows-to-Windows native, but supported for
  Windows-host → Linux-target.

## Deprecated and Removed Compiler Options

Deprecated options remain supported now but may not be later. Removed options: the compiler warns,
**ignores the option**, and proceeds with compilation. Both source lists are not exhaustive.

**Deprecated for SYCL** (Linux and Windows): `fsycl-explicit-simd` → `None`.

**Other Deprecated** — Linux: `daal` → `qdaal`, `device-math-lib` → `None`, `tbb` → `qtbb`.
Windows: `device-math-lib` → `None`, `TP` → `None`, `Zg` → `None`.

**Removed** — Linux: `c99` → `std=c99`; `check-uninit` → `check=uninit`; `foffload-static-lib` →
`None`; `fsycl-add-targets` → `None`; `fsycl-link-huge-device-code` → `flink-huge-device-code`;
`fsycl-link-targets` → `None`; `gcc-name` and `gxx-name` → "No exact replacement; use
`gcc-toolchain`"; `std=c9x` → `std=c99`; `syntax` → `fsyntax-only`. Windows: `Qc99` → `Qstd=c99`.

## Display Option Information

`help` alone lists **all** options; a functional category gives groupings (e.g. `-help diagnostics`
Linux, `/help diagnostics` Windows); other categories — see `help`.

## Alternate Compiler Options

**Not valid for SYCL applications.** Some alternates are deprecated and may be removed.

- Linux — Code Generation: `-fp` → `-fomit-frame-pointer`; Advanced Optimizations: `-funroll-loops` →
  `-unroll`; Linking or Linker: `-i-dynamic` → `-shared-intel`, `-i-static` → `-static-intel`.
- Linux — "OpenMP* and Parallel Processing Options": no entries listed.
- Windows — OpenMP* and Parallel Processing Options: `/openmp` → `/Qopenmp`.

## Portability and GCC-Compatible Warning Options

Does not apply for SYCL. The Intel® compiler supports many options valid on other compilers; options
unique to either compiler are not listed.

### Linux: options supported by both the Intel® compiler and the GCC Compiler

```text
-ansi,-B,-C,-c,-D,-dD,-dM,-E,-fargument-noalias,-fargument-noalias-global,
-fcf-protection,-fdata-sections,-ffunction-sections,-f[no-]builtin,-f[no-]common,
-f[no-]freestanding,-f[no-]gnu-keywords,-f[no-]inline,-f[no-]inline-functions,
-f[no-]math-errno,-f[no-]operator-names,-f[no-]stack-protector,-f[no-]unsigned-bitfields,
-fpack-struct,-fpermissive,-fPIC,-fpic,-fshort-enums,-fsyntax-only,-funroll-loops,
-funsigned-char,-fverbose-asm,-H,-help,-I,-idirafter,-imacros,-iprefix,-iwithprefix,
-iwithprefixbefore,-l,-L,-M,-march,-mcpu,-MD,-MF,-MG,-MM,-MMD,-m[no-]ieee-fp,-MP,
-MQ,-msse,-msse2,-msse3,-MT,-nodefaultlibs,-nostartfiles,-nostdinc,-nostdinc++,-nostdlib,
-o,-O,-O0,-O1,-O2,-O3,-Os,-p,-P,-S,-shared,-static,-std,-trigraphs,-U,-u,-v,-V,
-Wall,-Werror,-W[no-]cast-qual,-W[no-]comment,-W[no-]comments,-W[no-]deprecated,
-W[no-]fatal-errors,-W[no-]format-security,-W[no-]main,-W[no-]missing-declarations,
-W[no-]missing-prototypes,-W[no-]overflow,-W[no-]overloaded-virtual,-W[no-]pointer-arith,
-W[no-]return-type,-W[no-]strict-prototypes,-W[no-]trigraphs,-W[no-]uninitialized,
-W[no-]unknown-pragmas,-W[no-]unused-function,-W[no-]unused-variable,-X,-x assembler-with-cpp,
-x c,-x c++,-Xlinker
```

Many GCC-compatible warning options are recognized but **not documented**; accepted but undocumented
ones behave as described in the GCC documentation. Docs: `man gcc`, the GCC website, or search
"gcc warning options".

### Windows: options supported by both the Intel® compiler and the Microsoft C++ Compiler

```text
/C,/c,/D<name>{=|#}<text>,/E,/EH{a|s|c|r},/EP,/F<n>,/Fa[file],/FA[{c|s|cs}],/FC,
/Fe<file>,/FI<file>,/Fo<file>,/fp:<model>,/Fp<file>,/FR[<file>],/GA,/Gd,/GF,/GR[-],
/GS[-],/Gs[<n>],/Gy[-],/GZ,/H<n>,/help,/I<dir>,/J,/LD,/LDd,/link,/MD,/MDd,/MT,
/MTd,/nologo,/O1,/O2,/Od,/Oi[-],/Os,/Ot,/Ox,/P,/QIfist[-]
```

For `<n>` values and other details, see the Microsoft Visual Studio C++ documentation.

## Code examples

`-fortlib` mixed Fortran / C / C++ build (C/C++ links to the Fortran libraries at link time):

```bash
icx mymain.c -c
…
ifx sub1.f90 -c
icx -fortlib mymain.o sub1.o
```

`-fpreview-breaking-changes`:

```bash
> icpx -fpreview-breaking-changes -fsycl a.cpp -o a.out
> icx -fpreview-breaking-changes a.cpp -o a.out
> icpx -fpreview-breaking-changes -fiopenmp -fopenmp-targets=spir64 test.cpp
```

`-sox` (comments = documented equivalence/masking semantics):

```bash
icpx -sox file.cpp
icpx -sox=inline,profile file.cpp
icpx -sox=inline -sox=profile file.cpp           // same as -sox=inline,profile
icpx -sox=inline -no-sox -sox=profile file.cpp   // same as -sox=profile
icpx -sox=secure-defines simple.cpp -D_HELLO_    // -D_HELLO_ not in .comment
icpx -sox=secure,secure-defines simple.cpp -I/tmp/dir -D_HELLO_  // -I/tmp/dir, -D_HELLO_ not in .comment
```

Compound linker option; displaying option information:

```bash
-Xlinker -L -Xlinker $HOME/lib
-help diagnostics     # Linux
/help diagnostics     # Windows
```

## Gotchas & failure modes

1. `-static-intel` (default ON) errors with `-mcmodel=medium`/`large`, which set `-shared-intel`.
2. `-shared` makes Intel® libraries dynamic despite that default; the OpenMP* runtime stays dynamic
   unless `-qopenmp-link=static`; `-static` leaves no-static-version libraries dynamic.
3. Traceback needs `-static-libgcc`; `-Xlinker -shared` does no shared-object linkage work and
   compound `-Xlinker` args need the flag repeated.
4. `-l` must follow its object files; `-L` dirs precede standard dirs.
5. `-Wa`/`-Wl`/`-Wp` are ignored when their tool is not invoked.
6. Removed options are ignored with a warning (`-c99`; `-fsycl-add-targets`/`-fsycl-link-targets` have
   no replacement); deprecated ones still warn (`daal`, `tbb`, `fsycl-explicit-simd`,
   `device-math-lib`, `TP`, `Zg`).
7. `-fpreview-breaking-changes` links `libsycl-preview` with `-fsycl`; `--sysroot` is
   Windows-host→Linux-target only.
8. `-save-temps` leaves `.o` in the working directory; `/Qsave-temps` is C++ only, `None` for SYCL.
9. Alternate names and portability/GCC lists are invalid for SYCL; `/Qipo` (Windows) sets
   `-fuse-ld=lld`; `/MP` compiles only (no linking/LTCG).

## Source map

- Compatibility options — p. 345 · Linking or Linker Options (`F` … `Zl`) — pp. 345–370
- Miscellaneous Options (`dryrun` … `version`) — pp. 370–382
- Deprecated and Removed — pp. 382–383 · Display Option Information — pp. 383–384
- Alternate Compiler Options — p. 384 · Portability and GCC-Compatible Warning Options — pp. 384–389
