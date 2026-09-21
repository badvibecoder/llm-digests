# Session 1 notes — SYCL / oneAPI on Intel Arc Pro B70

**Purpose.** Accumulating lessons file. Each session gets its own `sessionN-notes.md`; keep these
focused on things that cost time or would **silently produce wrong results**. Add measured facts,
not impressions — and mark clearly whether a number was *measured* or *inferred*.

**Session 1 scope:** validate that the agent + harness can drive an Intel Arc GPU through the
docker container, then benchmark vector engines (XVE) and tensor engines (XMX) per dtype.

---

## 1. Environment (measured)

| Item | Value |
|---|---|
| Host | Intel(R) Core(TM) Ultra 5 250K Plus (`opencl:cpu`) |
| GPU | **Intel(R) Graphics [0xe223]** = Arc Pro B70 (Intel SKU 245797) |
| Driver | 1.14.37020+3, Level Zero V2 20.2.0 |
| Backends | `level_zero:0` (default) and `opencl:gpu:1` |
| Compute units | 256 |
| Global memory | 30.30 GiB |
| L2 / SLM | 24 MiB / 128 KiB |
| Sub-group sizes | 16, 32 |
| Max work-group | 1024 |
| Aspects | fp16 ✅ fp64 ✅ atomic64 ✅ usm_device_allocations ✅ queue_profiling ✅ |
| Extensions | `cl_khr_fp16`, `cl_khr_fp64`, `cl_khr_int64_{base,extended}_atomics`, `cl_intel_bfloat16_conversions`, `cl_khr_subgroups`, `cl_intel_required_subgroup_size` |

## 2. Workflow that works

```bash
./start-oneapi.sh                                  # long-lived container, $PWD -> /workspace
./cmd-oneapi.sh icpx -fsycl -O2 prog.cpp -o prog   # compile inside container
./cmd-oneapi.sh ./prog                             # run inside container
./cmd-oneapi.sh bash -lc '<cmd>'                   # arbitrary shell (needed for env vars)
./stop-oneapi.sh                                   # teardown; /workspace survives
```
- Container name is `sycl-test` (all three scripts must agree).
- `$PWD` at `start-oneapi.sh` time becomes the container's `/workspace` — run it from the project dir.
- Plain `icpx -fsycl -O2` is enough, including for **ESIMD/DPAS** — no `-fsycl-esimd` flag.

## 3. Bugs, traps and caveats (the important section)

| # | Symptom | Cause | Fix / rule |
|---|---|---|---|
| 1 | **DPAS returns silently wrong data** (fp16 72.2461 vs 64; s8 240 vs 128) | Execution size **N=8**. Compiles, passes every `static_assert`, runs with **no SYCL exception** | **Use ExecN=16 only.** N=8 is not implemented on this device. Always assert an analytically-known result |
| 2 | int2 far slower than int4 (183 vs 651 Top/s) | Header clamps `OpsPerChannel` to 8 → int2 gets **K=64, same as int4**, not 128 | Do not assume 2-bit gives 2× int4. **int4 is the sweet spot**; int2 is a 3.5× regression |
| 3 | Process dies with exit **139** (SIGSEGV) | Dereferencing a `malloc_device` pointer on the host | Never deref device USM on host — `q.memcpy(&host_var, dev_ptr, n)` then `.wait()` |
| 4 | Throughput numbers ~2× too low | Divided per-phase FLOPs by **total** elapsed including other phases | Compute each phase's metric against **that phase's** time (or its own device time) |
| 5 | Wrong claim: "int8/int16 promote to 32-bit" | Inferred, not measured | Measured: uint8 42.5 / uint16 41.3 vs uint32 6.73 Top/s, emitted as `:w` word ops. **Not promoted** |
| 6 | `joint_matrix` enum lists fp32/fp64 — looks supported | `matrix_type` is a *type* list, not an *XMX* list | fp32/fp64/int16/int32/int64 are **not tensor ops**; they'd run on the vector engine |
| 7 | No fp8 type anywhere | Not in oneAPI 2026.0 SYCL tree at all | fp8 is untestable here; don't promise it |
| 8 | `-Wdeprecated-declarations` on `sycl::backend::host` | Removed backend in 2026.0 | Drop the case; use `default:` |
| 9 | `info::device::extensions` deprecation warning | Deprecated in SYCL 2020 in favour of aspects | Still works; prefer `aspects` for new code |
| 10 | `MKLROOT` likely unset | `cmd-oneapi.sh` is plain `docker exec` (non-login) — **never sources `setvars.sh`** | `icpx` works via image PATH, but for oneMKL check `./cmd-oneapi.sh bash -lc 'echo $MKLROOT'` first |
| 11 | `permission denied ... /var/run/docker.sock` | Session credentials lacked the `docker` gid (host-side, not the scripts) | **Not** a script bug. `chmod +x` does **not** fix it; don't chase that |
| 12 | `docker run` failed when `/dev/dri` absent | No GPU passthrough nodes on that host state | Passthrough must exist before `start-oneapi.sh` can work |

## 4. Verification methodology (why the numbers are trustworthy)

1. **Warmup before timing** — lazy JIT means first submission is not representative.
2. **Checksum / assert results** — otherwise `-O2` may delete the work or hide a wrong answer.
   For DPAS with all-ones operands, one call adds exactly `K`; `chains × iters` → `chains·iters·K`.
3. **Batch submissions** to ~150 ms per `q.wait()` so per-kernel launch overhead doesn't dominate.
4. **Two clocks** — wall (`steady_clock`) and device (`property::queue::enable_profiling` +
   `event::get_profiling_info<info::event_profiling::command_start/command_end>`); gate on
   `aspect::queue_profiling`.
5. **ISA attribution** — never label a result "XMX" or "fp16" without checking the ISA:
   ```bash
   ./cmd-oneapi.sh bash -lc 'rm -rf /tmp/IntelIGC; IGC_ShaderDumpEnable=1 ./bin 1 >/dev/null 2>&1;
     find /tmp/IntelIGC -name "*.asm" | head'
   ```
   Then grep: `dpas` (tensor), `:hf` (fp16), `:f` (fp32), `:df` (fp64), `:w`/`:d` (16/32-bit int).
6. **Geometry sweep** — peak depends on launch shape; best-of-N, not one config.

## 5. Measured throughput (session 1)

Device-timed, 8 s/dtype, checksum-verified, ISA-attributed. Top/s counts 2 ops per MAC.

**XVE (vector engines), register MAD chains:**
| dtype | Top/s | ISA evidence |
|---|---|---|
| fp16 | **44.0** | `:hf`, 264 `mad` |
| fp32 | **22.4** | `:f`, 264 `mad` |
| fp64 | **1.40** | `:df` native (527 `mad`), gated 1/16 |
| uint8 | 42.5 | `:w` word ops |
| uint16 | 41.3 | `:w` word ops |
| uint32 | 6.73 | `:d` |

**XMX (tensor engines), ESIMD DPAS, M=1, ExecN=16:**
| dtype | Top/s | K |
|---|---|---|
| fp16 | 164.6 | 16 |
| bf16 | 162.6 | 16 |
| tf32 | 69.2 | 8 |
| int8 (s/u) | 326.3 / 320.3 | 32 |
| **int4 (s/u)** | **645.8 / 651.3** | 64 |
| int2 (s/u) | 179.4 / 182.6 | 64 (clamped) |

**Ratios:** XMX/XVE ≈ **3.7×** fp16, **7.7×** int8. fp64 = **1/16** of fp32. Peak observed = int4 ≈ 651 Top/s.
**Caveat:** M=1 only — an M-sweep (M=2/4/8) should push XMX higher, so the XMX column is a **floor**.

## 6. DPAS invocation reference (validated on this device)

```cpp
// template order: <SystolicDepth=8, RepeatCount M, T(result), CT(acc), BT, AT, BPrecision, APrecision>
C = xmx::dpas<8, 1, float, float, sycl::half, sycl::half,
              dpas_argument_type::fp16, dpas_argument_type::fp16>(C, B, A);
```
- `SystolicDepth` **must** be 8; `RepeatCount` 1..8; **ExecN must be 16**.
- Extents: `AN = M·K·bits/(8·sizeof(AT))`, `BN = K·N·bits/(8·sizeof(BT))`, `C = M·N`.
  Sub-byte AT/BT are 1-byte `char`, so extents are in **bytes**.
- `K = 8 × clamp(32/bits, ≤8)`: fp16/bf16 16, tf32 8, int8 32, int4 64, int2 **64**.

| dtype | BT/AT | precision args | K | A | B | C |
|---|---|---|---|---|---|---|
| fp16 | `half` | `fp16,fp16` | 16 | 16 | 256 | 16 |
| bf16 | `bfloat16` | `bf16,bf16` | 16 | 16 | 256 | 16 |
| tf32 | `tfloat32` | `tf32,tf32` | 8 | 8 | 128 | 16 |
| int8 s/u | `signed/unsigned char` | `s8/u8` | 32 | 32 | 512 | 16 |
| int4 s/u | `signed/unsigned char` | `s4/u4` **explicit** | 64 | 32 | 512 | 16 |
| int2 s/u | `signed/unsigned char` | `s2/u2` **explicit** | 64 | 16 | 256 | 16 |

- **int4/int2 precision must be passed explicitly** — deduction yields s8/u8 and the extent
  `static_assert` fires ("The first matrix multiplier has wrong size").
- All-ones byte patterns: u8/s8 `0x01`, u4/s4 `0x11`, u2/s2 `0x55`.
- Launch: `q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) SYCL_ESIMD_KERNEL { ... })`.
  Use **many work-items** (`single_task` = one thread → one XMX unit → useless for throughput).
- `RepeatCount=8` verified working for fp16 (`A=simd<half,128>`, `B=simd<half,256>`, `C=simd<float,128>`).
- Device bitcode/ISA shows the real `__esimd_dpas2` / `dpas` instruction — not scalar emulation.

## 7. Corrections log (claims I made that turned out wrong)

- ❌ "Sub-32-bit integers are promoted to 32-bit on the XVE." → **Measured false**; they run word-width at ~6× uint32.
- ❌ "`gpu_bench` computes 10.9 TFLOP/s fp32." → **Diluted** by phase interleaving; true value ≈ **22.4 TFLOP/s**.
- ⚠️ "`gpu_bench` measured 430 GB/s." → Also diluted; a clean bandwidth-only measurement was **not** done. Treat as unverified.

## 8. Untested — carry into session 2

- **buffers + accessors** (everything so far used USM only)
- **SLM / `local_accessor` / `group_barrier`**, reductions, group algorithms, sub-group shuffles
- **atomics** (`atomic_ref`) and the memory model
- **AOT build** (`-fsycl-targets=spir64_gen -Xs "-device ..."`), CMake, multi-TU `SYCL_EXTERNAL`
- **oneMKL / oneDPL / oneDNN** — linking, `MKLROOT`, buffer-vs-USM overloads (digest §18 is entry-point level only)
- **`joint_matrix`** unified API (only ESIMD DPAS was exercised)
- XMX **M-sweep** for true peak; clean **read-only bandwidth** measurement
- `dtype_probe.cpp` — scalar fp16/bf16/fp32/fp64/int8→64 all PASS on device

## 9. Session 1 artifacts

`start-oneapi.sh`, `cmd-oneapi.sh`, `stop-oneapi.sh` (name `sycl-test`) · `hello.cpp` ·
`gpu_bench.cpp` (mixed FMA + stream soak) · `dtype_probe.cpp` (scalar type coverage) ·
`xve_bench.cpp` (vector sweep) · `xmx_bench.cpp` (DPAS sweep) · `esimd_recon/` (DPAS shapes,
traps, `RESULTS.md`) · `SYCL_DIGEST/` (reference digest)
