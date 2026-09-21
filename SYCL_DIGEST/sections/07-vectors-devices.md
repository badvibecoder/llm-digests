## 07 · Vectors, math arrays, device information and specialization

### Two interpretations of vector types

| Interpretation | Meaning | SYCL type |
|---|---|---|
| Convenience type | groups data operated on as a group (RGB pixel); adds `+`, math fns | `marray` (SYCL 2020); `vec` usually used this way |
| SIMD type | maps to a hardware SIMD instruction (e.g. `float8` → 8 lanes) | none in SYCL 2020; explicit-SIMD extensions exist |

- `marray` (new in 2020) is a convenience type, unrelated to SIMD hardware instructions; local to one work-item; `float4 y4` storage equals `float y4[4]`.
- **Gotcha.** An escaped vector address (`dowork(&y4)` expecting a `vec_y[8][4]` layout) with non-inlined callees forces the compiler to honor the array layout and emit gather/scatter; an unobservable address may be transposed to consecutive accesses.
- Use `marray` where it makes logical sense; investigate lowering only at hotspots.

### `marray<T, N>`

- Templated on `DataT` and `NumElements`; `DataT` must be a C++ numeric type.
- **Rule.** `NumElements` is a positive integer; `NumElements == 1` makes a `marray` implicitly convertible to the equivalent scalar. Aliases require `N` ∈ {2, 3, 4, 8, 16}.
- `std::array`-like container plus math operators (`+`, `+=`) and SYCL math functions (`sin`, `cos`), applied elementwise.

| Type alias | marray equivalent |
|---|---|
| `mcharN` | `marray<int8_t, N>` |
| `mucharN` | `marray<uint8_t, N>` |
| `mshortN` | `marray<int16_t, N>` |
| `mushortN` | `marray<uint16_t, N>` |
| `mintN` | `marray<int32_t, N>` |
| `muintN` | `marray<uint32_t, N>` |
| `mlongN` | `marray<int64_t, N>` |
| `mulongN` | `marray<uint64_t, N>` |
| `mhalfN` | `marray<half, N>` |
| `mfloatN` | `marray<float, N>` |
| `mdoubleN` | `marray<double, N>` |
| `mboolN` | `marray<bool, N>` |

```cpp
marray<float, 4> input{1.0004f, 1e-4f, 1.4f, 14.0f};
buffer in_buf(&input, range{1});
q.submit([&](handler& cgh) {
  accessor in_acc{in_buf, cgh};
  cgh.parallel_for(range<1>(M), [=](id<1> idx) { auto r = cos(in_acc[0]); }); // elementwise; M gives parallelism
});
```

### `vec<T, N>`

- SYCL 1.2.1 and 2020; recommended only for **loads/stores**, **backend-native vector interop**, **swizzles**.
- **Rule.** `NumElements` must be 1, 2, 3, 4, 8, or 16; any other value is a compilation failure. `DataT` may be any basic scalar type supported in device code.
- Aliases have **no** `m` prefix, for 2, 3, 4, 8, 16: `uint4` = `vec<uint32_t, 4>`, `float16` = `vec<float, 16>`.

#### load / store

```cpp
template <access::address_space AddressSpace, access::decorated IsDecorated>
void load(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr);
template <access::address_space AddressSpace, access::decorated IsDecorated>
void store(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const;
```

- `load` reads `NumElements` values into the channels from the `multi_ptr` at `NumElements * offset` **elements of `DataT`**; `store` is the reverse. **Rule.** Parameter is a `multi_ptr` (not an accessor or raw pointer) whose `DataT` must match the `vec` component type.

```cpp
float16 inpf16;
inpf16.load(idx, acc.get_multi_ptr<access::decorated::no>());
float16 result = inpf16 * 2.0f;
result.store(idx, acc.get_multi_ptr<access::decorated::no>());
```

**Why.** Vectors hint each work-item accesses a contiguous block (may improve bandwidth); do not expect SIMD-instruction mapping.

#### Backend-native interop

- `vec`'s member type `vector_t` is the backend-native vector type if one exists (device code only); `vec` converts to/from it, enabling backend-native kernel calls.

#### Swizzles

```cpp
template <int... swizzleindexes>
__swizzled_vec__ swizzle() const;
__swizzled_vec__ XYZW_ACCESS() const;
__swizzled_vec__ RGBA_ACCESS() const;
__swizzled_vec__ INDEX_ACCESS() const;
#ifdef SYCL_SIMPLE_SWIZZLES
// Available only when numElements <= 4
// XYZW_SWIZZLE is all permutations with repetition of:
// x, y, z, w, subject to numElements
__swizzled_vec__ XYZW_SWIZZLE() const;
// Available only when numElements == 4
// RGBA_SWIZZLE is all permutations with repetition of: r, g, b, a.
__swizzled_vec__ RGBA_SWIZZLE() const;
#endif
```

- `swizzle<...>()`: variadic pack of integer template args, each index in [0, `NumElements`-1] (e.g. `vec.swizzle<2, 1, 0, 3>()`); returns `__swizzled_vec__`, an implementation-defined temporary. **Rule.** the swizzle runs when the returned instance is used in an expression, not at call.
- Simple swizzles `{x, y, z, w}` and `{r, g, b, a}`: only for vectors ≤ 4 elements, and only if `SYCL_SIMPLE_SWIZZLES` is defined **before any SYCL header files**.
- `a.wxyz()` (from `{1,2,3,4}` → `{4,1,2,3}`) ≡ `a.argb()`; result size need not match the original.

```cpp
auto e = a[idx];
float w = e.w();
float4 sw = e.xyzw();
sw = e.xyzw() * sw.wzyx();
sw = sw + w;
a[idx] = sw.xyzw();
```

#### `vec` size vs sub-group width

**Rule.** `vec`/`marray` size is unrelated to sub-group width; parallelism comes from the range (`h.parallel_for(range<1>(8), ...)`, sub-group size 8) and a work-item's vector is private (see §01).

### Device enumeration and queries

Member fns: `is_cpu()`, `is_gpu()`, `is_accelerator()`, `get_info`, `has` (`is_cpu()` == `has(aspect::cpu)`, `is_gpu()` == `has(aspect::gpu)`). `device::get_info` tags: `info::device::...`; `kernel::get_info` takes a `device` arg (queries may be device-specific). **Gotcha.** Queries need installed user-level drivers; failing enumeration usually means misinstalled drivers.

```cpp
// clang++ -fsycl fig_12_5_curious.cpp -o curious   (also: sycl-ls lists devices)
for (auto const& p : platform::get_platforms())
  for (auto const& d : p.get_devices())
    d.get_info<info::device::name>();  // p.get_info<info::platform::name>() likewise
```

#### Correctness queries

| `info::device::` tag | Fact |
|---|---|
| `device_type` | `cpu`, `gpu`, `accelerator`, `custom`, `automatic`, `all`; usually tested via `is_cpu()`, `is_gpu()`, ... |
| `max_work_item_sizes` | max work-items per dimension of the `nd_range` work-group; min (1, 1, 1) |
| `max_work_group_size` | max work-items per work-group on one compute unit; min 1 |
| `global_mem_size` | global memory size in bytes |
| `local_mem_size` | local memory size in bytes; min 32 K |
| `max_compute_units` | indicative of available parallelism; implementation-defined — avoid in program logic |
| `sub_group_sizes` | sub-group sizes supported by the device |

**Gotcha.** Submitting a kernel that violates a required condition (e.g. `sub_group_sizes`) generates a runtime error.
**Gotcha.** Do not branch on `max_compute_units`; express parallelism and let the runtime map it.

#### Tuning queries

| `info::device::` tag | Fact |
|---|---|
| `global_mem_cache_line_size` | cache line size, bytes |
| `global_mem_cache_size` | cache size, bytes |
| `local_mem_type` | `info::local_mem_type::local` (dedicated SRAM) or `info::local_mem_type::global` (abstraction over global memory, potentially no gain) |
| `preferred_work_group_size` | preferred work-group size for a kernel on a specific device |
| `preferred_work_group_size_multiple` | work-group size should be a multiple of this; must not exceed `work_group_size` |

#### Kernel queries (require kernel bundles)

| Tag | Fact |
|---|---|
| `work_group_size` | max work-group size usable for a kernel on a specific device |
| `compile_work_group_size` | work-group size specified by the kernel if applicable; else (0, 0, 0) |
| `compile_sub_group_size` | sub-group size specified by the kernel if applicable; else 0 |
| `compile_num_sub_groups` | number of sub-groups specified by the kernel if applicable; else 0 |
| `max_sub_group_size` | max sub-group size for a kernel launched with the specified work-group size |
| `max_num_sub_groups` | max number of sub-groups for a kernel |

#### Aspects

All Boolean; implementations may add more. `cpu`/`gpu`/`accelerator`/`custom` are mutually exclusive; `fp16`, `fp64`, `atomic64` optional.

| Aspect | The device... |
|---|---|
| `aspect::cpu` | executes code on a CPU |
| `aspect::gpu` | executes code on a GPU |
| `aspect::accelerator` | executes code on an accelerator |
| `aspect::custom` | executes fixed functions only; no programmable kernels |
| `aspect::emulated` | runs in an emulator; not for performance (debug, profiling) |
| `aspect::host_debuggable` | can fully support standard debugging |
| `aspect::fp16` | supports the `sycl::half` data type |
| `aspect::fp64` | supports the `double` data type |
| `aspect::atomic64` | supports 64-bit atomic operations |
| `aspect::image` | supports images |
| `aspect::online_compiler` | supports online compilation/linking of device code (`build()`, `compile()`, `link()`) |
| `aspect::online_linker` | (as above: online compilation/linking) |
| `aspect::queue_profiling` | supports queue profiling |
| `aspect::usm_device_allocations` | corresponding USM capability |
| `aspect::usm_host_allocations` | corresponding USM capability |
| `aspect::usm_atomic_host_allocations` | corresponding USM capability |
| `aspect::usm_shared_allocations` | corresponding USM capability |
| `aspect::usm_atomic_shared_allocations` | corresponding USM capability |
| `aspect::usm_system_allocations` | shares data allocated by system allocators |

- `aspect::queue_profiling` gates `property::queue::enable_profiling`; that property on a device without the aspect throws synchronously with `errc::feature_not_supported`.
- `any_device_has_v<aspect>` / `all_devices_have_v<aspect>` avoid instantiating templated kernels for unsupported features.
- Memory-model queries documented elsewhere: `atomic_memory_order_capabilities`, `atomic_fence_order_capabilities`, `atomic_memory_scope_capabilities`, `atomic_fence_scope_capabilities` (see memory-model section).

### Kernel specialization

Attributes (standard, not deprecated):

| Standard attribute | Specifies |
|---|---|
| `device_has(aspect, ...)` | only attribute usable on a non-kernel function as well as a kernel; kernel must launch only on devices meeting the listed aspect(s) |
| `reqd_work_group_size(dim0)` / `(dim0, dim1)` / `(dim0, dim1, dim2)` | requires launch with the specified work-group size |
| `work_group_size_hint(dim0)` / `(dim0, dim1)` / `(dim0, dim1, dim2)` | hints the kernel will most likely launch with the specified size |
| `reqd_sub_group_size(dim)` | requires the specified sub-group size |

- **Rule.** Submitting a kernel to a device lacking a listed aspect throws an exception.
- **Rule.** The compiler diagnoses a kernel (or any function it calls) that uses an optional feature (e.g. `fp16`) whose aspect is not listed.
- **Gotcha.** `device_has` does **not** affect device selection — host code must check aspects before submitting.
- **Gotcha.** `using namespace sycl` does not apply to attributes; write `[[sycl::device_has(...)]]`.

```cpp
queue q; constexpr int size = 16; std::array<double, size> data;
if (q.get_device().has(aspect::fp64)) {
  buffer B{data};
  q.submit([&](handler& h) {
    accessor A{B, h};
    h.parallel_for(size, [=](auto& idx) [[sycl::device_has(aspect::fp64)]] { A[idx] = idx * 2.0; });
  });
} // else: portable float path when fp64 is absent
```

**Guidance.** Parameterize one kernel by queried features instead of duplicating kernels; query real capabilities, never model/marketing numbers. Use a separate kernel when the algorithm itself differs.

### Not documented in the assigned source

`specialization_constant`, `kernel_handler`, `get_specialization_constant`, `lo`/`hi`/`even`/`odd` swizzle accessors, `as` conversions, `device_image` aspects, `-fsycl-device-code-split`: the source only says the standard defines "specialization constants" (not discussed) and points to an experimental compile-time property extension; no signatures given.

### Key gotchas

- `vec` size must be 1, 2, 3, 4, 8, or 16; anything else fails compilation.
- `marray` aliases exist only for N ∈ {2, 3, 4, 8, 16}; `NumElements` may be any positive integer.
- The `m` prefix distinguishes them: `float4` is `vec`, `mfloat4` is `marray`; mixing them is a silent logic bug.
- Define `SYCL_SIMPLE_SWIZZLES` before any SYCL header, or `xyzw()`-style calls are unavailable.
- Simple swizzles only for ≤ 4 elements; RGBA swizzles only for exactly 4.
- `__swizzled_vec__` evaluates lazily; the swizzle runs when the result is used.
- `load`/`store` need a `multi_ptr` whose element type matches the `vec` component type.
- `device_has` specializes but does not select a device; guard submission with `has(aspect::...)`.
- `[[sycl::device_has(...)]]` needs the `sycl::` qualifier even under `using namespace sycl`.
- `property::queue::enable_profiling` throws `errc::feature_not_supported` without `aspect::queue_profiling`.
- Never branch on `max_compute_units`; let the runtime map parallelism.
- Escaping a vector's address blocks layout transposition and can force gather/scatter.
