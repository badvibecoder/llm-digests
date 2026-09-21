## 11 · Defining kernels: lambdas, functors, kernel bundles

**Kernel.** Unit of computation instantiated many times in parallel; each instance = one index in the parallel execution space.

### Three ways to represent a kernel

| Form | Pros | Cons |
|---|---|---|
| Lambda expression | concise at point of use; familiar modern C++; capture rules pass data to kernel automatically | cannot be templated; do not assemble as a library (like regular functions) without extra work; syntax may be unfamiliar |
| Named function object (functor) | can be templated, reused, shipped as part of a library like any C++ class; more control over data passed into kernel | more code than a lambda; kernel args must be passed explicitly, not captured automatically |
| Interoperability with other languages/APIs | reuse of previously written kernels/libraries; incremental SYCL adoption in large codebases | see §20 |

### Kernel lambda expression

Full syntax: `[ capture-list ] ( params ) -> ret { body }`

| Element | Kernel rules |
|---|---|
| `capture-list` | capture **by copy only**. Use `[=]` (implicitly captures all variables by value) or name each captured variable in a comma-separated list. A variable used in the kernel but not captured by value → compile-time error. Global variables are not captured (C++ standard). |
| `params` | depends on how the kernel was invoked; identifies the work-item index in the parallel execution space (see §4). |
| `body` | operations performed at each index. |
| specifiers (`mutable` etc.) | **none defined by SYCL 2020** — not supported. |
| exception specification | supported but **must be `noexcept`** if provided; exceptions are not supported for kernels. |
| lambda attributes | supported; control kernel compilation, e.g. `reqd_work_group_size`, `device_has` (see kernel attributes; §12). |
| `-> ret` | may be specified but **must be `void`** if provided; non-void return types unsupported. |

```cpp
h.parallel_for(
  size,
  [=](id<1> i) { data_acc[i] = data_acc[i] + 1; }
);
```

Outer command-group lambda captures `[&]` (handler by reference); the kernel lambda uses `[=]`.

All optional elements together:

```cpp
q.submit([&](handler& h) {
  accessor data_acc{data_buf, h};
  h.parallel_for(
    nd_range{{size}, {8}},
    [=](id<1> i) noexcept [[sycl::reqd_work_group_size(8)]]
    ->void { data_acc[i] = data_acc[i] + 1; });
});
```

Global variables not captured; non-global **static** variables usable in a kernel only if `const`.

### Naming kernel lambdas

Lambdas are anonymous; some cases require an explicit kernel name template parameter. Name = a `class` used only as a tag:

```cpp
// "class Add" names the kernel lambda expression
h.parallel_for<class Add>(size, [=](id<1> i) {
  data_acc[i] = data_acc[i] + 1;
});
```

- Kernel name template parameter is **optional for most SYCL 2020 compilers**; when not required, prefer the unnamed form (less verbose).
- Naming a kernel lambda lets a host-code compiler identify which kernel to invoke when the kernel was compiled by a separate device-code compiler; enables runtime introspection of a compiled kernel and building a kernel by name.
- Naming is **mandatory** to selectively compile or to query a kernel: without a name there is no way to identify the kernel to compile or query.
- Tools mixing SYCL implementations from multiple vendors may require names for defined interaction within a single compilation; also improves kernel-name display in debug tools and layers.
- `(SYCL 2020)` Naming lambdas is optional per the specification.

### Named function objects (functors)

- Member variables = state the kernel operates on; overloaded `operator()` invoked per work-item.
- Args passed via constructor; arg to `operator()` depends on launch, same as lambdas.
- No kernel name template parameter needed: all function objects are named, so the host compiler uses the function object type to identify device code **even if the functor is templated**.
- Type must satisfy SYCL 2020 **device copyable** rules (informally: safely copyable byte-by-byte, so members reach device code). Any trivially copyable C++ type is implicitly device copyable.
- Easier to analyze/optimize, debug, reuse; ship as separate header or library.
- `(SYCL 2020)` based on C++17: C++20 templated lambda expressions are **not supported** for kernels.

```cpp
class Add {
public:
  Add(accessor<int> acc) : data_acc(acc) {}
  void operator()(id<1> i) const {
    data_acc[i] = data_acc[i] + 1;
  }
private:
  accessor<int> data_acc;
};
// ...
h.parallel_for(size, Add(data_acc));
```

**Two valid attribute positions on a functor** (a lambda has only one):

```cpp
class AddWithAttribute {
public:
  AddWithAttribute(accessor<int> acc) : data_acc(acc) {}
  [[sycl::reqd_work_group_size(8)]] void operator()(
      id<1> i) const { data_acc[i] = data_acc[i] + 1; }
private:
  accessor<int> data_acc;
};
class MulWithAttribute {
public:
  MulWithAttribute(accessor<int> acc) : data_acc(acc) {}
  void operator()
  [[sycl::reqd_work_group_size(8)]] (id<1> i) const {
    data_acc[i] = data_acc[i] * 2; }
private:
  accessor<int> data_acc;
};
```

### Kernel attributes

Standard (non-deprecated) attributes:

| Attribute | Effect |
|---|---|
| `device_has(aspect, ...)` | kernel may only launch on devices meeting the listed aspects. **Only attribute usable on a non-kernel function**, in addition to decorating a kernel. |
| `reqd_work_group_size(dim0)` / `(dim0, dim1)` / `(dim0, dim1, dim2)` | kernel must launch with the specified work-group size. |
| `work_group_size_hint(dim0)` / `(dim0, dim1)` / `(dim0, dim1, dim2)` | hints kernel will most likely launch with the specified size. |
| `reqd_sub_group_size(dim)` | requires the specified sub-group size. |

Resolved via namespace qualifier even under `using namespace sycl;` (C++ attribute lookup quirk):

```cpp
h.parallel_for(
  size, [=](auto& idx)
  [[sycl::device_has(aspect::fp64)]] { A[idx] = idx * 2.0; });
```

**Rules.** Attribute benefits: (1) kernel **throws an exception** if submitted to a device missing one of the listed aspects; (2) compiler **diagnostic** if the kernel (or any function it calls) uses an optional feature (e.g. fp16) whose aspect is not listed. Submitting a kernel that violates a required condition (e.g. `sub_group_sizes`) → runtime error.

**Gotcha.** `sycl::device_has()` does **not** affect device selection — host code must check `q.get_device().has(aspect::...)` before submitting.

### Kernel bundles

`kernel_bundle` = container for the SYCL kernels/functions used by an application; the number of bundles depends on the compiler (one bundle may hold many kernels; several bundles may hold few).

| `bundle_state` | Meaning |
|---|---|
| input | typically intermediate representation; must be JIT-compiled before executing on a device |
| object | usually compiled but not linked, like object files |
| executable | fully compiled to device code, ready to execute. AOT-compiled bundles start here |

- Most compilers compile kernels to IR first for portability → bundles usually start in input state; runtimes then compile input → executable **lazily**, on demand. Fast startup; first use of a kernel is slower (compile + submit).
- `get_kernel_bundle<bundle_state::executable>(context)` compiles **all kernels for all devices in the context**.
- `get_kernel_bundle<bundle_state::executable>(context, {device}, {kid})` compiles **only the named kernel for the given device**. `handler::use_kernel_bundle(kb)` uses the precompiled bundle.
- Devices with `aspect::online_compiler` may support the `build()`, `compile()`, and `link()` functions (advanced; not covered here).

```cpp
auto kid = get_kernel_id<class Add>();
auto kb = get_kernel_bundle<bundle_state::executable>(
    q.get_context(), {q.get_device()}, {kid});
q.submit([&](handler& h) {
  h.use_kernel_bundle(kb);
  accessor data_acc{data_buf, h};
  h.parallel_for<class Add>(range{size}, [=](id<1> i) {
    data_acc[i] = data_acc[i] + 1; });
});
```

Query a compiled kernel via `kernel_bundle::get_kernel(kid)` and `kernel::get_info<...>(device)`:

```cpp
auto kernel = kb.get_kernel(kid);
kernel.get_info<info::kernel_device_specific::work_group_size>(
    q.get_device());
kernel.get_info<
    info::kernel_device_specific::preferred_work_group_size_multiple>(
    q.get_device());
```

| Kernel query | Returns |
|---|---|
| `work_group_size` | max work-group size usable to execute the kernel on a specific device |
| `compile_work_group_size` | work-group size specified by the kernel if applicable; else `(0, 0, 0)` |
| `compile_sub_group_size` | sub-group size specified by the kernel if applicable; else `0` |
| `compile_num_sub_groups` | number of sub-groups specified by the kernel if applicable; else `0` |
| `max_sub_group_size` | max sub-group size for a kernel launched with the specified work-group size |
| `max_num_sub_groups` | max number of sub-groups for a kernel |
| `preferred_work_group_size` | preferred work-group size for the kernel on a specific device |
| `preferred_work_group_size_multiple` | best performance when work-group size is a multiple of this value; must not be greater than `work_group_size` |

All under `info::kernel_device_specific::`; queries are device-specific, so `get_info` takes a `device`.

### Multiple translation units & device code

- Functions called inside a kernel but defined in a different translation unit need `SYCL_EXTERNAL`; otherwise the compiler only compiles them for host use.
- Restrictions on `SYCL_EXTERNAL` functions: only functions; **no raw pointers** as parameter or return types (use explicit pointer classes); cannot call a `parallel_for_work_item` method; cannot be called from within a `parallel_for_work_group` scope.
- `(DPC++ ext)` `SYCL_EXTERNAL` is an optional feature in general; DPC++ supports it.
- Anonymous kernels spanning translation units, multi-vendor tooling, or any kernel query still need names (see naming above).

```
error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
```
```
terminate called after throwing an instance of '...compile_program_error'...
error: undefined reference to ...
```

**Why.** Scattered device code across translation units may trigger more JIT compilations than colocated device code (implementation-dependent). Mitigate by (1) grouping device code in the same translation unit, (2) ahead-of-time compilation.

### Key gotchas
- Capture only by value in kernel lambdas; `[&]` on a kernel lambda is illegal.
- Never capture the `handler`; create accessors inside the command group and capture those.
- Anonymous lambda with no kernel name cannot be selectively compiled or queried — name it.
- Functor attribute goes on `operator()`; two valid positions, not the single lambda position.
- Functor must be device copyable; trivially copyable is implicitly so.
- `noexcept` or nothing; non-`void` returns and `mutable` are not supported in kernels.
- Static variables are usable only if `const`; globals are never captured.
- `device_has` does not filter device selection — check `has()` yourself.
- Prefer unnamed lambdas unless importing kernel name template parameters for compile options.
- `get_kernel_bundle<executable>` compiles everything for the context; restrict with explicit devices and kernel ids.
