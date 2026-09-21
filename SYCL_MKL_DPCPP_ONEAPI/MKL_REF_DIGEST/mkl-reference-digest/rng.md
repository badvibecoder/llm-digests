# Random Number Generators: Engines and Distributions

This chapter covers the oneMKL RNG (Vector Statistics) domain: pseudorandom, quasi-random and non-deterministic **engines** (basic random number generators, BRNG), **distribution** classes, the `generate` entry points, engine-state **service routines**, and the **device** API callable from SYCL kernels with its host-side helpers. Host (manual-offload) routines are in `oneapi::mkl::rng` (include `oneapi/mkl/rng.hpp`); device routines are in `oneapi::mkl::rng::device` (include `oneapi/mkl/rng/device.hpp`). Engines hold generator state and a `sycl::queue`; distributions hold statistical parameters and a generation method; `generate` combines the two.

## Overview

**PRNG definition.** A pseudo-random number generator is defined by a structure (S, μ, f, U, g): S a finite set of states (state space); μ a probability distribution on S for the initial state (or seed) s0; f:S→S the transition function; U a finite set of output symbols; g:S→U an output function. Generation: (1) generate the initial state (the seed) s0 according to μ and compute u0=g(s0); (2) iterate for ... and ... (iteration symbols lost in extraction). Output values are the random numbers. Random variates are generated in two steps: i.i.d. uniform (0, 1) variates, then transformations to imitate arbitrary distributions.

**Classification.** Engines (BRNG) hold state and are the source of i.i.d. random variables; Transformation classes hold a distribution's parameters (including generation method); the Generate function obtains random numbers from an engine with statistics defined by a distribution; Service routines modify engine state (`skip_ahead`, `leapfrog`).

**Usage model (host).** (1) Create/initialize the BRNG object (use `skip_ahead` or `leapfrog` if required — used in parallel with random number generation for CPU devices); (2) create/initialize the distribution object; (3) call `generate`. Buffer API returns `void`, USM API returns `sycl::event`.
```cpp
oneapi::mkl::rng::philox4x32x10 engine(queue, SEED);
oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf> distr(5.0, 2.0);
{ sycl::buffer<double, 1> r_buf(r.data(), r.size()); oneapi::mkl::rng::generate(distr, engine, n, r_buf); }
auto event = oneapi::mkl::rng::generate(distr, engine, n, r.data()); event.wait(); // USM
```
Manual-offload structure (p989 diagram): **Engines** (`oneapi::mkl::rng::mt19937`, `mrg32k3a`, `mcg59`, `philox4x32x10`, ... — source of randomness, hold state) and **Distributions** (`uniform`, `gaussian`, `lognormal`, `poisson`, `bernoulli`, ... — transform engine output, hold parameters) feed **Service Routines** (`template <typename Engine> oneapi::mkl::rng::skip_ahead(Engine& engine, ...)`, `oneapi::mkl::rng::leapfrog(Engine& engine, ...)`) and **Generation Routines** (`template <typename Distr, typename Engine> oneapi::mkl::rng::generate(const Distr&, Engine& engine, ...)`).

**Device support.** Host device (current CPU), CPU device (CPU using OpenCL™), GPU device. All DPC++ routines of oneMKL RNG support at least CPU devices. GPU devices are supported for engines `oneapi::mkl::rng::mcg31m1`, `mcg59`, `mrg32k3a`, `mt19937`, `mt2203`, `philox4x32x10`, `sobol`. GPU devices are supported for all distributions. NOTE: some distributions use double precision inside their implementations, so not all GPUs are supported for them. Device routines are callable from kernels and from the host (`oneapi::mkl::rng::device::routine(...)`). Examples: host `${MKL}/share/doc/mkl/examples/sycl/rng/source`, device `${MKL}/examples/dpcpp_device/rng/source`.

**Generation modes.** Accurate and fast. Accurate mode produces random numbers lying completely within the definitional domain for all values of the distribution parameters; fast mode is higher-performance and guarantees definitional-domain membership except for some specific parameter values. The mode is set by the `method` parameter. Accurate-mode support (source table, line-wrap artifacts removed; source prints `unform_method` once): uniform (Discrete) via `oneapi::mkl::rng::uniform_method::accurate`; exponential via `exponential_method::icdf_accurate`; weibull via `weibull_method::icdf_accurate`; rayleigh via `rayleigh_method::icdf_accurate`; lognormal via `lognormal_method::icdf_accurate` and `lognormal_method::box_muller2_accurate`; gamma via `gamma_method::marsaglia_accurate`; beta via `beta_method::cja_accurate`.

**Host method types** (each family also has `::by_default`): `uniform_method::{standard, accurate}`; `gaussian_method::{box_muller, box_muller2, icdf}`; `geometric_method::icdf`; `exponential_method::{icdf, icdf_accurate}`; `weibull_method::{icdf, icdf_accurate}`; `cauchy_method::icdf`; `rayleigh_method::{icdf, icdf_accurate}`; `lognormal_method::{icdf, icdf_accurate, box_muller2, box_muller2_accurate}`; `gumbel_method::icdf`; `bernoulli_method::icdf`; `gamma_method::{marsaglia, marsaglia_accurate}`; `beta_method::{cja, cja_accurate}`; `chi_square_method::gamma_based`; `gaussian_mv_method::{box_muller, box_muller2, icdf}`; `binomial_method::btpe`; `poisson_method::{ptpe, gaussian_icdf_based}`; `poisson_v_method::gaussian_icdf_based`; `hypergeometric_method::h2pe`; `negative_binomial_method::nbar`; `multinomial_method::poisson_icdf_based`. `binomial_method::btpe`, `poisson_method::ptpe`, `hypergeometric_method::h2pe` and `negative_binomial_method::nbar` are acceptance/rejection methods (four/four/three/five regions; conditions lost); `multinomial_method::poisson_icdf_based` is the Poisson Approximation method.

**Host data types / BRNG data types.** `uniform` (Continuous), `gaussian`, `gaussian_mv`, `exponential`, `laplace`, `weibull`, `cauchy`, `rayleigh`, `lognormal`, `gumbel`, `gamma`, `beta`, `chi_square`: float, double / float, double. `uniform` (Discrete): integer; GPU float for `uniform_method::standard`, double for `accurate` (source prints `oneapi::mkl::rng::method::accurate`); CPU double for both. `uniform_bits`, `bits`: integer/integer (32-bit and 64-bit chunk rows). `bernoulli`, `geometric`: integer/float. `binomial`, `hypergeometric`: integer/double. `poisson`: integer; float for `poisson_method::gaussian_icdf_based`, float/double for `poisson_method::ptpe` (conditions lost). `poisson_v`, `negative_binomial`: integer/double. `multinomial`: integer; CPU - float GPU - double. NOTE: in case of integer check desired distribution for supported data types. Source quirk: a table row labelled `gamma` describes the Beta distribution.

**Device data types / methods.** Continuous (`uniform`, `gaussian`, `exponential`, `lognormal`, `beta`, `gamma`): float, double / float, double. Discrete: `uniform` (Discrete) integer, BRNG float for `standard`, double for `accurate`, raw output for 64-bit integers; `bits` integer/integer; `uniform_bits` integer/integer; `bernoulli` integer/float; `poisson` integer/double; `geometric` integer, BRNG float for 32-bit and double for 64-bit integers. Device methods: `device::uniform_method::{standard, accurate}`, `device::gaussian_method::{box_muller2, icdf}`, `device::lognormal_method::box_muller2`, `device::exponential_method::{icdf, icdf_accurate}`, `device::poisson_method::devroye`, `device::bernoulli_method::icdf`, `device::geometric_method::icdf`, `device::beta_method::{cja, cja_accurate}`, `device::gamma_method::{marsaglia, marsaglia_accurate}`, each plus `::by_default`. To get `sycl::half` results, generate floats and convert (`static_cast<sycl::half>` for `VecSize == 1`, else `res.template convert<sycl::half>()`).

**Shared class boilerplate.** Every host engine class also declares `Engine(const Engine& other); Engine(Engine&& other); Engine& operator=(const Engine& other); Engine& operator=(Engine&& other); ~Engine();` — stated once here (verbatim per class) and omitted from the per-engine listings. Host distribution classes declare no copy/move members in their syntax blocks.

## Routines
### generate (host)
Obtain random numbers from a given engine with proper statistics of a given distribution; submits a kernel into the queue held by the engine and fills the output with n random numbers.
```cpp
namespace oneapi::mkl::rng { template<typename Distr, typename Engine>
  void generate (const Distr& distr, Engine& engine, std::int64_t n,
                 sycl::buffer<typename Distr::result_type, 1>& r)                    // Buffer API
  template<typename Distr, typename Engine>
  sycl::event generate (const Distr& distr, Engine& engine, std::int64_t n,
                        typename Distr::result_type* r,
                        const std::vector<sycl::event> & dependencies)                // USM API
}
```
Include `oneapi/mkl/rng.hpp`. `distr` `const Distr&`; `engine` `Engine&`; `n` `std::int64_t` number of random values; `dependencies` (USM only) events to wait for, if any; `r` output vector. USM returns the event after submitting the task from the engine. The syntax block shows no default for `dependencies`.
### mrg32k3a (host engine)
The combined multiple recursive pseudorandom number generator MRG32k3a [L'Ecuyer99a]. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class mrg32k3a { public:
    static constexpr std::uint32_t default_seed = 1; mrg32k3a(sycl::queue queue, std::uint32_t seed = default_seed);
    mrg32k3a(sycl::queue queue, std::uint32_t seed, mrg32k3a_mode::optimal mode); mrg32k3a(sycl::queue queue, std::uint32_t seed, mrg32k3a_mode::custom mode);
    mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed); mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed, mrg32k3a_mode::optimal mode);
    mrg32k3a(sycl::queue queue, std::initializer_list<std::uint32_t> seed, mrg32k3a_mode::custom mode); };
}
```
`queue` valid `sycl::queue` (generate submits kernels in it); `seed` initial conditions of the generator/engine state; `mode` `mrg32k3a_mode::optimal` or `mrg32k3a_mode::custom`. NOTE: sub-sequence parallelization [L'Ecuyer02] is supported via `mode`; `optimal` uses the optimal number of sub-streams, `custom` passes a user-defined number (e.g. `mrg32k3a_mode::custom my_mode{123}`); helper structure template `mrg32k3a_mode::optimal_v` exists. Currently supported only for uniform and gaussian distributions on GPU device.
### philox4x32x10 (host engine)
Philox4x32-10 counter-based pseudorandom number generator with period ... PHILOX4X32X10 [Salmon11]. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class philox4x32x10 { public:
      static constexpr std::uint64_t default_seed = 0; philox4x32x10(sycl::queue queue, std::uint64_t seed = default_seed);
      philox4x32x10(sycl::queue queue, std::initializer_list<std::uint64_t> seed); };
}
```
`seed` `std::uint64_t` / `std::initializer_list<std::uint64_t>`. NOTE: host `default_seed` is 0 while the device engine's `default_seed` is 1.
### mcg31m1 (host engine)
The 31-bit multiplicative congruential pseudorandom number generator MCG(...) [L'Ecuyer99a]. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class mcg31m1 { public:
    static constexpr std::uint32_t default_seed = 1; mcg31m1(sycl::queue queue, std::uint32_t seed = default_seed); };
}
```
`seed` `std::uint32_t` initial conditions of the engine.
### mcg59 (host engine)
The 59-bit multiplicative congruential pseudorandom number generator MCG(...) from NAG Numerical Libraries [NAG]. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class mcg59 { public:
    static constexpr std::uint64_t default_seed = 1; mcg59(sycl::queue queue, std::uint64_t seed = default_seed); };
}
```
`seed` `std::uint64_t`. NOTE: device `mcg59` takes a `std::uint32_t` seed.
### r250 (host engine)
The 32-bit generalized feedback shift register pseudorandom number generator GFSR(250,103) [Kirkpatrick81]. **CPU** only.
```cpp
namespace oneapi::mkl::rng { class r250 { public:
    static constexpr std::uint32_t default_seed = 1; r250(sycl::queue queue, std::uint32_t seed = default_seed);
    r250(sycl::queue queue, std::vector<std::uint32_t> seed); };
}
```
Parameter-table `seed` types: `std::uint32_t` / `std::initializer_list<std::uint32_t>` (second constructor in the syntax block takes `std::vector<std::uint32_t>` — source inconsistent).
### wichmann_hill (host engine)
Wichmann-Hill pseudorandom number generator (a set of 273 basic generators) from NAG Numerical Libraries [NAG]. **CPU** only.
```cpp
namespace oneapi::mkl::rng { class wichmann_hill { public:
    static constexpr std::uint32_t default_seed = 1; wichmann_hill(sycl::queue queue, std::uint32_t seed = default_seed);
    wichmann_hill(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx); wichmann_hill(sycl::queue queue, std::initializer_list<std::uint32_t> seed);
    wichmann_hill(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx); };
}
```
`engine_idx` `std::uint32_t`: index of the engine from the set (set contains 273 basic generators).
### mt19937 (host engine)
Mersenne Twister pseudorandom number generator MT19937 [Matsumoto98] with period length ... of the produced sequence. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class mt19937 { public:
    static constexpr std::uint32_t default_seed = 1; mt19937(sycl::queue queue, std::uint32_t seed = default_seed);
    mt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed); };
}
```
### sfmt19937 (host engine)
SIMD-oriented Fast Mersenne Twister SFMT19937 [Saito08] with period length ... of the produced sequence. **CPU** only.
```cpp
namespace oneapi::mkl::rng { class sfmt19937 { public:
    static constexpr std::uint32_t default_seed = 1; sfmt19937(sycl::queue queue, std::uint32_t seed = default_seed);
    sfmt19937(sycl::queue queue, std::initializer_list<std::uint32_t> seed); };
}
```
### mt2203 (host engine)
Set of 6024 Mersenne Twister generators MT2203 [Matsumoto98], [Matsumoto00]; each generates a sequence of period ... ; parameters provide mutual independence of the sequences. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class mt2203 { public:
    static constexpr std::uint32_t default_seed = 1; mt2203(sycl::queue queue, std::uint32_t seed = default_seed);
    mt2203(sycl::queue queue, std::uint32_t seed, std::uint32_t engine_idx); mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed);
    mt2203(sycl::queue queue, std::initializer_list<std::uint32_t> seed, std::uint32_t engine_idx); };
}
```
`engine_idx` `std::uint32_t`: index of the engine from the set (set contains 6024 basic generators); enables up to 6024 independent sequences. Source prints `~mt2203()` without a semicolon.
### ars5 (host engine)
ARS-5 counter-based pseudorandom number generator with period ..., using instructions from the AES-NI set ARS5 [Salmon11]. **CPU** only.
```cpp
namespace oneapi::mkl::rng { class ars5 { public:
      static constexpr std::uint64_t default_seed = 0; ars5(sycl::queue queue, std::uint64_t seed = default_seed);
      ars5(sycl::queue queue, std::initializer_list<std::uint64_t> seed); };
}
```
### sobol (host engine)
Sobol quasi-random number generator [Sobol76], [Bratley88], working in arbitrary dimension. CPU and GPU.
```cpp
namespace oneapi::mkl::rng { class sobol { public:
    static constexpr std::uint32_t default_dimensions_number = 1; sobol(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number);
    sobol(sycl::queue queue, std::vector<std::uint32_t>& direction_numbers); };
}
```
`dimensions` `std::uint32_t` number of dimensions; `direction_numbers` `std::vector<std::uint32_t>` user-defined direction numbers (non-const reference). NOTE: quasi-random, so statistics can be worse than pseudo-random generators if used incorrectly; for distributions requiring more than 1 random number from BRNG (e.g. gaussian with the `box_muller` and `box-muller2` methods) set `dimensions` to 2 or more.
### niederreiter (host engine)
Niederreiter quasi-random number generator [Bratley92], working in arbitrary dimension. **CPU** only.
```cpp
namespace oneapi::mkl::rng { class niederreiter { public:
      static constexpr std::uint32_t default_dimensions_number = 1; niederreiter(sycl::queue queue, std::uint32_t dimensions = default_dimensions_number);
      niederreiter(sycl::queue queue, std::vector<std::uint32_t>& irred_polynomials); };
}
```
`irred_polynomials` described in the parameter table as "User-defined direction numbers" (as printed).
### nondeterministic (host engine)
Non-deterministic random number generator (RDRAND-based) [AVX], [IntelSWMan]. **CPU** only; no seed parameter.
```cpp
namespace oneapi::mkl::rng { class nondeterministic { public:
      nondeterministic(sycl::queue queue); };
}
```
### leapfrog (host service routine)
Proceed state of engine using the leapfrog method: generates random numbers with non-unit stride, splitting the sequence into non-overlapping subsequences across `stride` nodes; initializes the stream for node `idx`, 0 ≤ idx < stride.
```cpp
namespace oneapi::mkl::rng { template<typename Engine>
  void leapfrog (Engine& engine, std::uint64_t idx, std::uint64_t stride)
}
```
`engine` engine object which supports leapfrog; `idx` `std::uint64_t` index of the computational node; `stride` `std::uint64_t` largest number of computational nodes. NOTE: supported only for generators that allow splitting by the leapfrog method. Example: `mkl::rng::leapfrog(engine_1, 0, 3)`, `(engine_2, 1, 3)`, `(engine_3, 2, 3)`.
### skip_ahead (host service routine)
Advances the state of the given engine as if it had generated 32 random bits a specified number of times; splits the sequence into non-overlapping blocks of `num_to_skip` (block-splitting method, unlimited nodes).
```cpp
namespace oneapi::mkl::rng { template<typename Engine> void skip_ahead (Engine& engine, std::uint64_t num_to_skip)   // Common Interface
  template<typename Engine> void skip_ahead (Engine& engine,                             // Partitioned number
                                             std::initializer_list<std::uint64_t> num_to_skip)
}
```
`engine` engine object which supports the block-splitting method; `num_to_skip` `std::uint64_t` number of skipped elements or `std::initializer_list<std::uint64_t>` partitioned number of skipped elements. NOTES: only for generators that allow skipping; for quasi-random generators the method works with components of quasi-random vectors, so to skip NS vectors set `num_to_skip = num_to_skip * dim`. Above 2^64 the partitioned interface is used, encoding `num_to_skip[0] + num_to_skip[1]*2^64 + num_to_skip[2]*2^128 + ... + num_to_skip[n-1]*2^(64*(n-1))`; below 2^64 both interfaces work.
### save_state
Save the engine state in binary format to the file or memory buffer.
```cpp
namespace oneapi::mkl::rng { template<typename Engine> void save_state (Engine& engine, std::uint8_t* mem);                    // Save to Memory
  template<typename Engine> void save_state (Engine& engine, const std::string& filename);          // Save to File
}                                                                      // File form deprecated since 2024.1 release
```
Note: The Save to File function is deprecated. Use the Save to Memory function instead. Include `oneapi/mkl/rng.hpp`. `engine` `Engine&` whose state would be saved; `mem` `std::uint8_t*` memory you allocate (use `get_state_size` for the byte count); `filename` `const std::string&`.
### load_state
Loads the state of the random number engine from the provided memory buffer or file, then creates new engine object (a different `sycl::queue` may be used for the new engine).
```cpp
namespace oneapi::mkl::rng { template<typename Engine> Engine load_state (const sycl::queue& queue, const std::uint8_t* mem);  // Load from Memory
  template<typename Engine> Engine load_state (const sycl::queue& queue, const std::string& filename);
}                                                                      // File form deprecated since 2024.1 release
```
Note: The Load from File function is deprecated. Use the Load from Memory function instead. Include `oneapi/mkl/rng.hpp`. `queue` `const sycl::queue&` used for the newly-created engine; `mem` `const std::uint8_t*`; `filename` `const std::string&`. Returns `Engine`: new engine object from the given queue and loaded state. The example uses `oneapi::mkl::rng::default_engine` and `load_state<oneapi::mkl::rng::default_engine>(cpu_queue, mem_buf)` (that class is not otherwise documented here).
### get_state_size
Returns the size in bytes which is needed to store the state of a given random number engine.
```cpp
namespace oneapi::mkl::rng { template<typename Engine> std::int64_t get_state_size (Engine& engine);
}
```
`engine` `Engine&` engine object whose state size would be calculated. Returns `std::int64_t`: size in bytes.
### uniform (Continuous) (host distribution)
Uniform random numbers over [a, b) (constraint formula lost). CPU and GPU. `Type` = float | double; `Method` = `oneapi::mkl::rng::uniform_method::standard` | `accurate`; inputs `a` (left bound a), `b` (right bound b).
```cpp
namespace oneapi::mkl::rng { template<typename Type = float, typename Method = uniform_method::by_default>
  class uniform { public:
    using method_type = Method; using result_type = Type; uniform(): uniform(static_cast<Type>(0.0), static_cast<Type>(1.0)){} explicit uniform(Type a, Type b); explicit uniform(const param_type& pt); Type a() const; Type b() const; param_type param() const; void param(const param_type& pt); };
}
```
### gaussian (host distribution)
Normal (Gaussian) random numbers with mean (a) and standard deviation (stddev, σ) (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `gaussian_method::box_muller` | `box_muller2` | `icdf`; inputs `mean`, `stddev`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = gaussian_method::by_default>
  class gaussian { public:
    using method_type = Method; using result_type = RealType; gaussian(): gaussian(static_cast<RealType>(0.0), static_cast<RealType>(1.0)){} explicit gaussian(RealType mean, RealType stddev); explicit gaussian(const param_type& pt); RealType mean() const; RealType stddev() const; param_type param() const; void param(const param_type& pt); };
}
```
### exponential (host distribution)
Exponentially distributed random numbers with displacement a and scalefactor ... (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `exponential_method::icdf` | `icdf_accurate`; inputs `a` (displacement a), `beta` (scalefactor). Source quirk: constructor parameters are named `mean, stddev`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = exponential_method::by_default>
  class exponential { public:
    using method_type = Method; using result_type = RealType; exponential(): exponential((RealType)0.0, (RealType)1.0){} explicit exponential(RealType mean, RealType stddev); explicit exponential(const param_type& pt); RealType a() const; RealType beta() const; param_type param() const; void param(const param_type& pt); };
}
```
### laplace (host distribution)
Laplace distributed random numbers with mean value (or average) a and scalefactor b, where σ = b√2. CPU and GPU. `RealType` = float | double; `Method` = `laplace_method::icdf`; inputs `a` (mean value a), `b` (scalefactor b).
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = laplace_method::by_default>
  class laplace { public:
    using method_type = Method; using result_type = RealType; laplace(): laplace((RealType)0.0, (RealType)1.0){} explicit laplace(RealType a, RealType b); explicit laplace(const param_type& pt); RealType a() const; RealType b() const; param_type param() const; void param(const param_type& pt); };
}
```
### weibull (host distribution)
Weibull distributed random numbers with displacement a, scalefactor β and shape ... (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `weibull_method::icdf` | `icdf_accurate`; inputs `alpha` (shape α), `a` (displacement a), `beta` (scalefactor β). Source quirk: default constructor writes `(Real_Type)1.0`.
```cpp
namespace oneapi::mkl::rng {
template<typename RealType = float, typename Method = weibull_method::by_default>
class weibull { public:
  using method_type = Method; using result_type = RealType;
  weibull(): weibull((Real_Type)1.0, (RealType)0.0, (RealType)1.0){} explicit weibull(RealType alpha, RealType a, RealType beta); explicit weibull(const param_type& pt);
  RealType alpha() const; RealType a() const; RealType beta() const; param_type param() const; void param(const param_type& pt); };
}
```
### cauchy (host distribution)
Cauchy distributed random values with displacement (a) and scalefactor (b, β) (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `cauchy_method::icdf`; inputs `a`, `b`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = cauchy_method::by_default>
  class cauchy { public:
    using method_type = Method; using result_type = RealType; cauchy(): cauchy((RealType)0.0, (RealType)1.0){} explicit cauchy(RealType a, RealType b); explicit cauchy(const param_type& pt); RealType a() const; RealType b() const; param_type param() const; void param(const param_type& pt); };
}
```
### rayleigh (host distribution)
Rayleigh distributed random values with displacement (a) and scalefactor (b, β) (constraint lost); special case of the Weibull distribution with shape parameter ... . CPU and GPU. `RealType` = float | double; `Method` = `rayleigh_method::icdf` | `icdf_accurate`; inputs `a`, `b`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = rayleigh_method::by_default>
  class rayleigh { public:
    using method_type = Method; using result_type = RealType; rayleigh(): rayleigh((RealType)0.0, (RealType)1.0){} explicit rayleigh(RealType a, RealType b); explicit rayleigh(const param_type& pt); RealType a() const; RealType b() const; param_type param() const; void param(const param_type& pt); };
}
```
### lognormal (host distribution)
Lognormal random numbers with average of distribution (m, a) and standard deviation (s, σ) of subject normal distribution, displacement (displ, b) and scalefactor (scale, β) (constraints lost). CPU and GPU. `RealType` = float | double; `Method` = `lognormal_method::box_muller2` | `box_muller2_accurate` | `icdf` | `icdf_accurate`; inputs `m`, `s`, `displ`, `scale` (defaults `(RealType)0.0`, `(RealType)1.0`).
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = lognormal_method::by_default>
  class lognormal { public:
    using method_type = Method; using result_type = RealType; lognormal(): lognormal((RealType)0.0, (RealType)1.0, (RealType) 0.0, (RealType)1.0){} explicit lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0); explicit lognormal(const param_type& pt);
    RealType m() const; RealType s() const; RealType displ() const; RealType scale() const; param_type param() const; void param(const param_type& pt); };
}
```
### gumbel (host distribution)
Gumbel distributed random values with displacement a and scalefactor (b, β) (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `gumbel_method::icdf`; inputs `a`, `b`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = gumbel_method::by_default>
  class gumbel { public:
    using method_type = Method; using result_type = RealType; gumbel(): gumbel((RealType)0.0, (RealType)1.0){} explicit gumbel(RealType a, RealType b); explicit gumbel(const param_type& pt); RealType a() const; RealType b() const; param_type param() const; void param(const param_type& pt); };
}
```
### gamma (host distribution)
Gamma distributed random values with shape parameter ..., displacement a and scale parameter ... (constraints lost). CPU and GPU. `RealType` = float | double; `Method` = `gamma_method::marsaglia` | `marsaglia_accurate`; inputs `alpha` (shape α), `a` (displacement a), `beta` (scalefactor β).
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = gamma_method::by_default>
  class gamma { public:
    using method_type = Method; using result_type = RealType; gamma(): gamma((RealType)1.0, (RealType)0.0, (RealType)1.0){} explicit gamma(RealType alpha, RealType a, RealType beta); explicit gamma(const param_type& pt); RealType alpha() const; RealType a() const; RealType beta() const; param_type param() const; void param(const param_type& pt); };
}
```
### beta (host distribution)
Beta distributed random values with shape parameters p and q, displacement a and scale parameter (b, β) (constraints lost). CPU and GPU. `RealType` = float | double; `Method` = `beta_method::cja` | `cja_accurate`; inputs `p`, `q`, `a`, `b`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = beta_method::by_default>
  class beta { public:
    using method_type = Method; using result_type = RealType; beta(): beta((RealType)1.0, (RealType)1.0, (RealType)(0.0), (RealType)(1.0)){} explicit beta(RealType p, RealType q, RealType a, RealType b); explicit beta(const param_type& pt); RealType p() const; RealType q() const; RealType a() const; RealType b() const; param_type param() const; void param(const param_type& pt); };
}
```
### chi_square (host distribution)
Chi-square distributed random values with n degrees of freedom (constraint lost). CPU and GPU. `RealType` = float | double; `Method` = `chi_square_method::gamma_based`; input `n` `std::int32_t` degrees of freedom.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, typename Method = chi_square_method::by_default>
  class chi_square { public:
    using method_type = Method; using result_type = RealType; chi_square(): chi_square(5){} explicit chi_square(std::int32_t n); explicit chi_square(const param_type& pt); std::int32_t n() const; param_type param() const; void param(const param_type& pt); };
}
```
### gaussian_mv (host distribution)
Random numbers from d-variate normal (Gaussian) distribution with mean (a) and variance-covariance matrix C, a ∈ R^d, C a d x d symmetric positive-definite matrix; C = T T^T with T the lower triangular Cholesky factor of C. CPU and GPU. `RealType` = float | double; `layout Layout = layout::packed` with values `layout::packed`, `layout::full`, `layout::diagonal`; `Method` = `gaussian_mv_method::box_muller` | `box_muller2` | `icdf`; inputs `dimen` `std::uint32_t`, `mean` `sycl::span<RealType>`, `matrix` `sycl::span<RealType>`.
```cpp
namespace oneapi::mkl::rng { template<typename RealType = float, layout Layout = layout::packed, typename Method = gaussian_mv_method::by_default>
  class gaussian_mv { public:
    using method_type = Method; using result_type = RealType; static constexpr layout layout_type = Layout; explicit gaussian_mv(std::uint32_t dimen, sycl::span<RealType> mean, sycl::span<RealType> matrix); explicit gaussian_mv(const param_type& pt); std::uint32_t dimen() const; sycl::span<RealType> mean() const; sycl::span<RealType> matrix() const; param_type param() const; void param(const param_type& pt); };
}
```
NOTE: when passing a `sycl::span` constructed over a user's memory, users must manage the memory under `sycl::span` themselves and must not destroy it while data are processed.
### uniform (Discrete) (host distribution)
Uniform random numbers over [a, b). CPU and GPU. Partial specialization of the continuous `uniform` template. `Method` = `uniform_method::standard` | `accurate`; inputs `a` (`std::int32_t` / `std::uint32_t`) left bound a; `b` (`std::int32_t` / `std::uint32_t`) right bound b.
```cpp
namespace oneapi::mkl::rng { template<typename Method>
  class uniform<std::(u)int32_t, Method> { public:
    using method_type = Method; using result_type =  std::(u)int32_t; uniform(): uniform((std::(u)int32_t )(0), std::is_same<Method, uniform_method::standard>::value ? (1 << 23) : std::numeric_limits<Type>::max()){} explicit uniform(std::(u)int32_t a,  std::(u)int32_t b); explicit uniform(const param_type& pt);
    std::(u)int32_t a() const; std::(u)int32_t b() const; param_type param() const; void param(const param_type& pt); };
}
```
NOTE: `oneapi::mkl::rng::uniform_method::standard` uses the `s` BRNG type on GPU devices, which might cause incorrect statistics (rounding error) when (condition formula lost); use `uniform_method::accurate` instead.
### uniform_bits (host distribution)
Uniformly distributed bits in 32/64-bit chunks, each bit in the chunk uniformly distributed; not supported for all engines. CPU and GPU. `UIntType` = `std::uint32_t` | `std::uint64_t` (the chunk size).
```cpp
namespace oneapi::mkl::rng {
template<typename UIntType = std::uint32_t>
  class uniform_bits { using result_type = UIntType }
}
```
### bits (host distribution)
Bits of underlying engine (BRNG) integer reccurence (spelling as printed). CPU and GPU. `UIntType` = `std::uint32_t`.
```cpp
namespace oneapi::mkl::rng { template<typename UIntType = std::uint32_t>
  class bits { using result_type = UIntType };
}
```
NOTE: lower bits of linear congruential generators are less random than higher bits (see [Knuth81]); typically only the 24 higher bits of a 32-bit LCG can be considered random.
### bernoulli (host distribution)
Bernoulli distributed random values: 1 with probability of success p, 0 with probability 1 - p (constraint lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t` (table prints `typename_IntType = std::int32_t`); `Method` = `bernoulli_method::icdf`; input `p` `float`.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = bernoulli_method::by_default>
  class bernoulli { public:
    using method_type = Method; using result_type = IntType; bernoulli(): bernoulli(0.5f){} explicit bernoulli(float p); explicit bernoulli(const param_type& pt); float p() const; param_type param() const; void param(const param_type& pt); };
}
```
### geometric (host distribution)
Geometrically distributed random values: number of independent Bernoulli trials preceding the first success with single-trial success probability p (constraint lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `geometric_method::icdf`; input `p` `float`.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = geometric_method::by_default>
  class geometric { public:
    using method_type = Method; using result_type = IntType; geometric(): geometric(0.5){} explicit geometric(float p); explicit geometric(const param_type& pt); float p() const; param_type param() const; void param(const param_type& pt); };
}
```
### binomial (host distribution)
Binomially distributed random numbers: number of successes in m independent Bernoulli trials with single-trial success probability p (constraints lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `binomial_method::btpe`; inputs `ntrials` `std::int32_t` (constructor parameter named `ntrial`), `p` `double`.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = binomial_method::by_default>
  class binomial { public:
    using method_type = Method; using result_type = IntType; binomial(): binomial(5, 0.5){} explicit binomial(std::int32_t ntrial, double p); explicit binomial(const param_type& pt); std::int32_t ntrial() const; double p() const; param_type param() const; void param(const param_type& pt); };
}
```
### hypergeometric (host distribution)
Hypergeometrically distributed random values with lot size l, size of sampling s and number of marked elements m (constraints lost); sampling without replacement of exactly s elements from a lot of l with m "marked" elements. CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `hypergeometric_method::h2pe`; inputs `l`, `s`, `m`. Source quirk: constructor writes `std::int32_T` for `s` and `m`.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = hypergeometric_method::by_default>
  class hypergeometric { public:
    using method_type = Method; using result_type = IntType; hypergeometric(): hypergeometric(1, 1, 1){} explicit hypergeometric(std::int32_t l, std::int32_T s, std::int32_T m); explicit hypergeometric(const param_type& pt); std::int32_t s() const; std::int32_t m() const; std::int32_t l() const; param_type param() const; void param(const param_type& pt); };
}
```
### poisson (host distribution)
Poisson distributed random values with distribution parameter λ (constraint lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `poisson_method::ptpe` | `gaussian_icdf_based`; input `lambda` `double`.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = poisson_method::by_default>
  class poisson { public:
    using method_type = Method; using result_type = IntType; poisson(): poisson(0.5){} explicit poisson(double lambda); explicit poisson(const param_type& pt); double lambda() const; param_type param() const; void param(const param_type& pt); };
}
```
### poisson_v (host distribution)
Poisson distributed random values with varying mean: n Poisson random numbers with distribution parameter ... (conditions lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `poisson_v_method::gaussian_icdf_based`; input `lambda` `sycl::span<double>` array of n distribution parameters λ (user must keep the memory alive).
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = poisson_v_method::by_default>
  class poisson_v { public:
    using method_type = Method; using result_type = IntType; explicit poisson_v(sycl::span<double> lambda); explicit poisson_v(const param_type& pt); sycl::span<double> lambda() const; param_type param() const; void param(const param_type& pt); };
}
```
### negative_binomial (host distribution)
Negative binomial (or Pascal) distributed random numbers with distribution parameters a and p (constraints lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `negative_binomial_method::nbar`; inputs `a` `double` first distribution parameter, `p` `double` second distribution parameter.
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = negative_binomial_method::by_default>
  class negative_binomial { public:
    using method_type = Method; using result_type = IntType; negative_binomial(): negative_binomial(0.1, 0.5){} explicit negative_binomial(double a, double p); explicit negative_binomial(const param_type& pt); double a() const; double p() const; param_type param() const; void param(const param_type& pt); };
}
```
### multinomial (host distribution)
Multinomially distributed random numbers with `ntrial` independent trials and k mutually exclusive outcomes with probabilities ... (constraints lost). CPU and GPU. `IntType` = `std::int32_t` | `std::uint32_t`; `Method` = `multinomial_method::poisson_icdf_based`; inputs `ntrial` `std::int32_t` number of independent trials m (constructor takes `double ntrial`), `p` `sycl::span<double>` probability vector of outcomes (length k, keep alive). NOTE: on GPU only input parameters following the condition ... are supported (inequality lost).
```cpp
namespace oneapi::mkl::rng { template<typename IntType = std::int32_t, typename Method = multinomial_method::by_default>
  class multinomial { public:
    using method_type = Method; using result_type = IntType; explicit multinomial(double ntrial, sycl::span<double> p); explicit multinomial(const param_type& pt); std::int32_t ntrial() const; sycl::span<double> p() const; param_type param() const; void param(const param_type& pt); };
}
```
### oneapi::mkl::rng::device::generate
Entry point to obtain random numbers from a given engine with proper statistics of a given distribution (device API). `distr` `Distr&`, `engine` `Engine&`. Returns `sycl::vec<typename Distr::result_type, Engine::vec_size>` or `typename Distr::result_type` (scalar when `vec_size = 1`) filled with random numbers. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename Distr, typename Engine>
  auto generate(Distr& distr, Engine& engine) ->
    typename std::conditional<Engine::vec_size == 1, typename Distr::result_type,
                              sycl::vec<typename Distr::result_type, Engine::vec_size>>::type
}
```
### oneapi::mkl::rng::device::generate_single
Entry point to obtain a single random number from a given vector engine with proper statistics of a given distribution. Returns `typename Distr::result_type`; may be used for engines with `vec_size > 1`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename Distr, typename Engine>
  typename Distr::result_type generate_single(Distr& distr, Engine& engine)
}
```
### mrg32k3a (device engine)
The combined multiple recursive pseudorandom number generator MRG32k3a [L'Ecuyer99a]. `VecSize` `std::int32_t`: size of the vector produced by `generate`; values 1, 2, 3, 4, 8, or 16 as the `sycl:vec` class size; default 1 returns a single random number. `seed` initial conditions of the engine state; `offset` number of skipped elements, initializer_list offset = `num_to_skip[0] + num_to_skip[1]*2^64 + num_to_skip[2]*2^128 + ... + num_to_skip[n-1]*2^(64*(n-1))`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<std::int32_t VecSize = 1>
  class mrg32k3a { public:
    static constexpr std::uint32_t default_seed = 1; static constexpr std::int32_t vec_size = VecSize; mrg32k3a() : mrg32k3a(default_seed) {}
    mrg32k3a(std::uint32_t seed, std::uint64_t offset = 0); mrg32k3a(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0); mrg32k3a(std::uint32_t seed, std::initializer_list<std::uint64_t> offset); mrg32k3a(std::initializer_list<std::uint32_t> seed, std::initializer_list<std::uint64_t> offset); };
}
```
### philox4x32x10 (device engine)
A Philox4x32-10 counter-based pseudorandom number generator [Salmon11]. `VecSize` as for device `mrg32k3a`; `seed` `std::uint64_t` / `std::initializer_list<std::uint64_t>`; `offset` number of skipped elements (initializer_list formula as above). Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<std::int32_t VecSize = 1>
  class philox4x32x10 { public:
    static constexpr std::uint64_t default_seed = 1; static constexpr std::int32_t vec_size = VecSize; philox4x32x10() : philox4x32x10(default_seed) {}
    philox4x32x10(std::uint64_t seed, std::uint64_t offset = 0); philox4x32x10(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0); philox4x32x10(std::uint64_t seed, std::initializer_list<std::uint64_t> offset); philox4x32x10(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset); };
}
```
### mcg31m1 (device engine)
The 31-bit multiplicative congruential pseudorandom number generator MCG(1132489760, 232-1) [L'Ecuyer99a]. `seed` `std::uint32_t` / `std::initializer_list<std::uint32_t>`; `offset` `std::uint64_t` number of skipped elements. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<std::int32_t VecSize = 1>
  class mcg31m1 { public:
    static constexpr std::uint32_t default_seed = 1; static constexpr std::int32_t vec_size = VecSize; mcg31m1() : mcg31m1(default_seed) {}
    mcg31m1(std::uint32_t seed, std::uint64_t offset = 0); mcg31m1(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0); };
}
```
### mcg59 (device engine)
The 59-bit multiplicative congruential pseudorandom number generator MCG(1313, 259) from NAG Numerical Libraries [NAG]. `seed` `std::uint32_t` / `std::initializer_list<std::uint32_t>`; `offset` `std::uint64_t`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<std::int32_t VecSize = 1>
  class mcg59 { public:
    static constexpr std::uint32_t default_seed = 1; static constexpr std::int32_t vec_size = VecSize; mcg59() : mcg59(default_seed) {}
    mcg59(std::uint32_t seed, std::uint64_t offset = 0); mcg59(std::initializer_list<std::uint32_t> seed, std::uint64_t offset = 0); };
}
```
### pcg64_dxsm (device engine)
A permuted congruential pseudorandom number generator PCG64 DXSM with period ... [pcg2014]. `seed` initial conditions of the 128 bit engine state; `offset` number of skipped elements, initializer_list offset = `num_to_skip[0] + num_to_skip[1]*2^64`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<std::int32_t VecSize = 1>
  class pcg64_dxsm { public:
    static constexpr std::uint64_t default_seed = 0; static constexpr std::int32_t vec_size = VecSize; pcg64_dxsm() : pcg64_dxsm(default_seed) {}
    pcg64_dxsm(std::uint64_t seed, std::uint64_t offset = 0); pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0); pcg64_dxsm(std::uint64_t seed, std::initializer_list<std::uint64_t> offset); pcg64_dxsm(std::initializer_list<std::uint64_t> seed, std::initializer_list<std::uint64_t> offset); };
}
```
### engine_descriptor (host-side helper)
Provides an abstraction over `sycl::buffer` to initialize and store engines' states between SYCL kernels. `Engine` = engine whose state is held; `queue` `sycl::queue&` task submitted to initialize engine states on the device; `range` `sycl::range<1>` number of engines; `seed` `std::uint64_t`; `offset` `std::uint64_t` number of skipped elements (initializer_list formula as for `mrg32k3a`); `func` `InitEngineFunc` functor taking `sycl::item<1>` and returning `Engine`. NOTE: the constructor submits a `parallel_for` task: each engine is initialized as `Engine{seed, id * offset}` for the scalar offset case, where `id` is a value from 0 to `range.size()`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<Engine>
  class engine_descriptor { public:
    engine_descriptor(sycl::queue& queue, sycl::range<1> range, std::uint64_t seed, std::uint64_t offset); template<typename InitEngineFunc> engine_descriptor(sycl::queue& queue, sycl::range<1> range, InitEngineFunc func);
    engine_accessor<Engine> get_access(sycl::handler& cgh); };
}
```
### engine_accessor (host-side helper)
Provides an abstraction over `sycl::accessor` to access engines held by `engine_descriptor` in kernels, by the `load()` and `store()` functions. `Engine` = engine whose state is held. Include `oneapi/mkl/rng/device.hpp`. (Source prints this syntax block under the class name `engine_descriptor`; reproduced verbatim.)
```cpp
namespace oneapi::mkl::rng::device { template<Engine>
  class engine_descriptor { public:
    Engine load(size_t id) const; void store(Engine engine, size_t id) const; };
}
```
### skip_ahead (device service routine)
Advances the state of the given engine as if it had generated 32 random bits a specified number of times. `engine` engine object which supports the block-splitting method; `num_to_skip` `std::uint64_t` number of skipped elements or `std::initializer_list<std::uint64_t>` partitioned number. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename Engine> void skip_ahead (Engine& engine, std::uint64_t num_to_skip)                    // Common Interface
  template<typename Engine> void skip_ahead (Engine& engine,                                              // Partitioned number
                                             std::initializer_list<std::uint64_t> num_to_skip)
}
```
### uniform (Continuous) (device distribution)
Uniform random numbers over [a, b), a, b ∈ R, a < b. `Type` = float | double; `Method` = `oneapi::mkl::rng::device::uniform_method::by_default` | `standard` | `accurate`; inputs `a`, `b` (no default template arguments are shown). Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename Type, typename Method>
  class uniform { public:
    using method_type = Method; using result_type = Type; uniform(): uniform((Type)0.0, (Type)1.0){}; explicit uniform(Type a, Type b); explicit uniform(const param_type& pt); Type a() const; Type b() const; param_type param() const; void param(const param_type& pt); };
}
```
### gaussian (device distribution)
Normally distributed random numbers with mean (a) and standard deviation (stddev, σ). `RealType` = float | double; `Method` = `device::gaussian_method::by_default` | `box_muller2` | `icdf`; inputs `mean`, `stddev`. Include `oneapi/mkl/rng/device.hpp`. NOTES: to enable `device::gaussian_method::box_muller2` based on Vector Math functions define the `MKL_RNG_USE_BINARY_CODE` macro and link static oneMKL libraries per the Link Line Advisor (may improve performance); `device::gaussian_method::icdf` is available only if `MKL_RNG_USE_BINARY_CODE` is defined, and with `float` requires device double-precision support.
```cpp
namespace oneapi::mkl::rng::device { template<typename RealType, typename Method>
  class gaussian { public:
     using method_type = Method; using result_type = RealType; gaussian(): gaussian((RealType)0.0, (RealType)1.0){} explicit gaussian(RealType mean, RealType stddev); explicit gaussian(const param_type& pt);
     RealType mean() const; RealType stddev() const; param_type param() const; void param(const param_type& pt); };
}
```
### lognormal (device distribution)
Lognormally distributed random numbers with average of distribution (m, a) and standard deviation (s, σ) of subject normal distribution, displacement (displ, b) and scalefactor (scale, β). `RealType` = float | double; `Method` = `device::lognormal_method::by_default` | `box_muller2`; inputs `m`, `s`, `displ`, `scale`. Include `oneapi/mkl/rng/device.hpp`. NOTE: to enable `device::lognormal_method::box_muller2` define `MKL_RNG_USE_BINARY_CODE` and link static oneMKL libraries per the Link Line Advisor.
```cpp
template<typename RealType, typename Method>
class lognormal { public:
   using method_type = Method; using result_type = RealType;
   lognormal() : lognormal((RealType)0.0, (RealType)1.0, (RealType) 0.0, (RealType)1.0){}
   explicit lognormal(RealType m, RealType s, RealType displ = (RealType)0.0, RealType scale = (RealType)1.0); explicit lognormal(const param_type& pt);
   RealType m() const; RealType s() const; RealType displ() const; RealType scale() const; param_type param() const; void param(const param_type& pt); };
```
### exponential (device distribution)
Exponentially distributed random numbers with displacement ... and scalefactor ... . `RealType` = float | double; `Method` = `device::exponential_method::by_default` | `icdf` | `icdf_accurate`; inputs `a` (displacement a), `beta` (scalefactor). Include `oneapi/mkl/rng/device.hpp`. NOTE: to enable extra Vector Math optimizations define `MKL_RNG_USE_BINARY_CODE` and link static oneMKL libraries per the Link Line Advisor.
```cpp
namespace oneapi::mkl::rng::device { template<typename RealType, typename Method>
  class exponential { public:
    using method_type = Method; using result_type = RealType; exponential(): exponential((RealType)0.0, (RealType)1.0){} explicit exponential(RealType a, RealType beta); explicit exponential(const param_type& pt); RealType a() const; RealType beta() const; param_type param() const; void param(const param_type& pt); };
}
```
### uniform (Discrete) (device distribution)
Uniform random numbers over [a, b). `Method` = `device::uniform_method::standard` | `accurate`; inputs `a`, `b` of type `std::int8_t`, `std::uint8_t`, `std::int16_t`, `std::uint16_t`, `std::int32_t`, `std::uint32_t`, `std::int64_t` or `std::uint64_t`. Include `oneapi/mkl/rng/device.hpp`. NOTES: `device::uniform_method::standard` uses `float` for underlying BRNG calls, which might cause incorrect statistics (rounding error) when (condition lost) — use `accurate` instead; for `(u)int8`/`(u)int16` numbers are always generated as single-precision floats; the sample rejection method is used for the 64-bit return type, enabling non-overlapping parallel sequences — provide an ample offset between engines. Source quirks: `sizeof(Type) == 32` (almost certainly the 32-bit test) and bare `uniform_method::standard` without the `device::` qualifier.
```cpp
namespace oneapi::mkl::rng::device { template<typename Type, typename Method>
  class uniform { public:
    using method_type = Method; using result_type = Type; uniform(): uniform((Type)0, std::is_integral<Type>::value
                 ? ((std::is_same_v<Method, uniform_method::standard> && sizeof(Type) == 32) ? (1 << 23) : std::numeric_limits<Type>::max())
                 : (Type)1) {}; explicit uniform(Type a, Type b); explicit uniform(const param_type& pt);
    Type a() const; Type b() const; param_type param() const; void param(const param_type& pt); };
}
```
### bits (device distribution)
Generates bits of underlying engine (BRNG) integer sequence. `UIntType` = `std::uint32_t` for `philox4x32x10`, `mrg32k3a` and `mcg31m1` engines; `std::uint64_t` for `mcg59` engine. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename UIntType = std::uint32_t>
  class bits { using result_type = UIntType; };
}
```
### uniform_bits (device distribution)
Uniformly distributed bits in 32/64-bit chunks, each bit in the chunk uniformly distributed; supported for `philox4x32x10` and `mcg59` engines; for 64-bit chunks twice as much engine offset needs to be provided. `UIntType` = `std::uint32_t` | `std::uint64_t`. Include `oneapi/mkl/rng/device.hpp`. Examples: convert to small types via `static_cast<DesiredType>` / `sycl::vec<DesiredType, VecSize>::convert`, or extract four `std::uint8_t` from one `std::uint32_t` by masking `0xff` and shifting right by 8.
```cpp
namespace oneapi::mkl::rng::device { template<typename UIntType = std::uint32_t>
  class uniform_bits { using result_type = UIntType; };
}
```
### poisson (device distribution)
Poisson distributed random values with distribution parameter λ. `IntType` = `std::int32_t` | `std::int64_t`; `Method` = `device::poisson_method::by_default` | `devroye`; input `lambda` `double`. Include `oneapi/mkl/rng/device.hpp`. NOTE: `device::poisson_method::devroye` uses table-lookup for small lambdas (... 60), Devroye's method [Devroye] for medium lambdas (60 ... 1000), Gaussian approximation for huge lambdas (... 1000); the comparison operators are formulas that did not survive extraction.
```cpp
namespace oneapi::mkl::rng::device { template<typename IntType, typename Method>
  class poisson { public:
    using method_type = Method; using result_type = IntType; poisson(): poisson(0.5){} explicit poisson(double lambda); explicit poisson(const param_type& pt); double lambda() const; param_type param() const; void param(const param_type& pt); };
}
```
### bernoulli (device distribution)
Bernoulli distributed random values with probability p of a single trial success. `IntType` = `std::int8_t` | `std::uint8_t` | `std::int16_t` | `std::uint16_t` | `std::int32_t` | `std::uint32_t`; `Method` = `device::bernoulli_method::by_default` | `icdf`; input `p` `float`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename IntType, typename Method>
  class bernoulli { public:
    using method_type = Method; using result_type = IntType; bernoulli(): bernoulli(0.5f){} explicit bernoulli(float p); explicit bernoulli(const param_type& pt); float p() const; param_type param() const; void param(const param_type& pt); };
}
```
### geometric (device distribution)
Geometrically distributed random values with probability p of a single trial success. `IntType` = `std::int32_t` | `std::uint32_t` | `std::int64_t` | `std::uint64_t`; `Method` = `device::geometric_method::by_default` | `icdf`; input `p` `float`. Include `oneapi/mkl/rng/device.hpp`.
```cpp
namespace oneapi::mkl::rng::device { template<typename IntType, typename Method>
  class geometric { public:
    using method_type = Method; using result_type = IntType; geometric(): geometric(0.5f){} explicit geometric(float p); explicit geometric(const param_type& pt); float p() const; param_type param() const; void param(const param_type& pt); };
}
```
### beta (device distribution)
Beta distributed random numbers with shape parameters ..., displacement ... and scale parameter ... . `RealType` = float | double; `Method` = `device::beta_method::by_default` | `cja` | `cja_accurate`; inputs `p` (shape p), `q` (shape q), `a` (displacement), `b` (scalefactor). Special method `std::size_t count_rejected_numbers() const`: amount of random numbers rejected during the last `generate` call (0 if no calls). Include `oneapi/mkl/rng/device.hpp`. NOTE: to enable extra Vector Math optimizations define `MKL_RNG_USE_BINARY_CODE` and link static oneMKL libraries per the Link Line Advisor.
```cpp
namespace oneapi::mkl::rng::device { template<typename RealType, typename Method>
  class beta { public:
    using method_type = Method; using result_type = RealType; beta() : beta((RealType)1.0, (RealType)1.0, (RealType)0.0, (RealType)1.0){} explicit beta(RealType p, RealType q, RealType a, RealType b); explicit beta(const param_type& pt); RealType p() const; RealType q() const; RealType a() const; RealType b() const; param_type param() const; std::size_t count_rejected_numbers() const; void param(const param_type& pt); };
}
```
### gamma (device distribution)
Gamma distributed random numbers with shape ..., displacement ... and scale parameter ... . `RealType` = float | double; `Method` = `device::gamma_method::by_default` | `marsaglia` | `marsaglia_accurate`; inputs `alpha` (shape), `a` (displacement a), `beta` (scalefactor). Special method `std::size_t count_rejected_numbers() const` as for device `beta`. Include `oneapi/mkl/rng/device.hpp`. NOTE: to enable extra Vector Math optimizations define `MKL_RNG_USE_BINARY_CODE` and link static oneMKL libraries per the Link Line Advisor.
```cpp
namespace oneapi::mkl::rng::device { template<typename RealType, typename Method>
  class gamma { public:
    using method_type = Method; using result_type = RealType; gamma() : gamma((RealType)1.0, (RealType)0.0, (RealType)1.0){} explicit gamma(RealType alpha, RealType a, RealType beta); explicit gamma(const param_type& pt); RealType alpha() const; RealType a() const; RealType beta() const; param_type param() const; std::size_t count_rejected_numbers() const; void param(const param_type& pt); };
}
```
### count_engine_adaptor (device engine adaptor)
Engine adaptor that counts how many times random 32 bits were taken from the engine during the `generate` call; especially useful for acceptance-rejection distributions such as beta and gamma. `Engine` = type of the provided random engine; `engine` (`const Engine&` or `Engine&&`) constructs the adaptor over an engine; `params` (`Params...`) constructs the adaptor generating an engine inside. Getters: `std::size_t get_count() const` (as printed in the prose) returns the count of random numbers generated by `generate`; `const Engine& base() const` returns the underlying engine. Include `oneapi/mkl/rng/device.hpp`. NOTE: the syntax block declares `std::int64_t get_count() const` while the description says `std::size_t get_count() const` — source inconsistent.
```cpp
namespace oneapi::mkl::rng::device { template <typename Engine>
 class count_engine_adaptor { public:
     static constexpr std::int32_t vec_size = Engine::vec_size; explicit count_engine_adaptor(const Engine& engine); explicit count_engine_adaptor(Engine&& engine);
     template <typename... Params> count_engine_adaptor(Params... params); std::int64_t get_count() const; const Engine& base() const; };
}
```
## Formulas

Host distributions. Notation: `alpha`, `beta`, `sigma` written out; `C_n^k` binomial coefficient; `floor(x)` printed as `|_x_|` in the source.

- `uniform` (Continuous) pdf (p1020#1; same on p1076#0 for the device class): `f_{a,b}(x) = { 1/(b-a), x in [a,b) ; 1, x not in [a,b) }, -inf < x < +inf`. The source image really shows `1` on the `x not in [a,b)` branch; mathematically the density is 0 there (source error, reproduced verbatim).
- `uniform` (Continuous) cdf (p1021#0; same on p1076#1): `F_{a,b}(x) = { 0, x < a ; (x-a)/(b-a), a <= x < b ; 1, x >= b }, -inf < x < +inf`.
- `gaussian` pdf (p1022#0; same on p1078#0): `f_{a,sigma}(x) = 1/(sigma*sqrt(2*pi)) * exp( -((y-a)^2)/(2*sigma^2) ) dy, -inf < x < +inf` (source uses `y` and a trailing `dy` in the density).
- `gaussian` cdf (p1022#1; same on p1078#1): `F_{a,sigma}(x) = integral_{-inf}^{x} 1/(sigma*sqrt(2*pi)) * exp( -((y-a)^2)/(2*sigma^2) ) dy, -inf < x < +inf`.
- `gaussian` cdf via standard normal (p1078#2, device only): `F_{a,sigma}(x) = phi((x-a)/sigma)`.
- `exponential` pdf (p1023#0; same on p1081#0): `f_{a,beta}(x) = { (1/beta)*exp(-(x-a)/beta), x >= a ; 0, x < a }, -inf < x < +inf`.
- `exponential` cdf (p1024#0; same on p1081#1): `F_{a,beta}(x) = { 1 - exp(-(x-a)/beta), x >= a ; 0, x < a }, -inf < x < +inf`.
- `laplace` standard deviation (p1025#0): `sigma = b*sqrt(2)`.
- `laplace` pdf (p1025#1): `f_{a,b}(x) = (1/(2b))*exp( -|x-a|/b ), -inf < x < +inf`.
- `laplace` cdf (p1025#2): `F_{a,b}(x) = { (1/2)*exp((x-a)/b), x < a ; 1 - (1/2)*exp(-(x-a)/b), x >= a }, -inf < x < +inf`.
- `weibull` pdf (p1027#0): `f_{a,alpha,beta}(x) = { (alpha/beta^alpha)*(x-a)^(alpha-1)*exp( -((x-a)/beta)^alpha ), x >= a ; 0, x < a }` (subscripts printed `a,alpha,beta`).
- `weibull` cdf (p1027#1): `F_{a,alpha,beta}(x) = { 1 - exp( -((x-a)/beta)^alpha ), x >= a ; 0, x < a }, -inf < x < +inf`.
- `cauchy` pdf (p1028#0, source labels it `F`): `F_{a,beta}(x) = 1 / ( pi*beta*( 1 + ((x-a)/beta)^2 ) ), -inf < x < +inf`.
- `cauchy` cdf (p1028#1): `F_{a,beta}(x) = 1/2 + (1/pi)*arctan((x-a)/beta), -inf < x < +inf`.
- `rayleigh` pdf (p1030#0): `f_{a,beta}(x) = { (2(x-a)/beta^2)*exp( -((x-a)^2)/beta^2 ), x >= a ; 0, x < a }, -inf < x < +inf`.
- `rayleigh` cdf (p1030#1; source conditions use `alpha`): `F_{alpha,beta}(x) = { 1 - exp( -((x-a)^2)/beta^2 ), x >= alpha) ; 0, x < alpha }, -inf < x < +inf`.
- `lognormal` pdf (p1031#0; same on p1080#0): `f_{a,sigma,b,beta}(x) = { 1/( sigma*(x-b)*sqrt(2*pi) ) * exp( -( ln((x-b)/beta) - a )^2 / (2*sigma^2) ), x > b ; 0, x <= b }`.
- `lognormal` cdf (p1031#1; same on p1080#1): `F_{a,sigma,b,beta}(x) = { Phi( ( ln((x-b)/beta) - a ) / sigma ), x > b ; 0, x <= b }`.
- `gumbel` pdf (p1033#0): `f_{a,beta}(x) = (1/beta)*exp((x-a)/beta)*exp( -exp((x-a)/beta) ), -inf < x < +inf`.
- `gumbel` cdf (p1033#1; source uses `alpha`): `F_{alpha,beta}(x) = 1 - exp( -exp((x-alpha)/beta) ), -inf < x < +inf`.
- `gamma` pdf (p1034#0): `f_{alpha,a,beta}(x) = { ( 1/( Gamma(alpha)*beta^alpha ) )*(x-a)^(alpha-1)*e^{-(x-a)/beta}, x >= a ; 0, x < a }, -inf < x < +inf`.
- `gamma` cdf (p1035#0): `F_{alpha,a,beta}(x) = { integral_a^x ( 1/( Gamma(alpha)*beta^alpha ) )*(y-a)^(alpha-1)*e^{-(y-a)/beta} dy, x >= a ; 0, x < a }, -inf < x < +inf`.
- `beta` pdf (p1036#0): `f_{p,q,a,beta}(x) = { ( 1/( B(p,q)*beta^{p+q-1} ) )*(x-a)^{p-1}*(beta+a-x)^{q-1}, a <= x < a+beta ; 0, x < a, x >= a+beta }, -inf < x < inf`.
- `beta` cdf (p1036#1, source labels it `f`): `f_{p,q,a,beta}(x) = { 0, x < a ; integral_a^x ( 1/( B(p,q)*beta^{p+q-1} ) )*(y-a)^{p-1}*(beta+a-y)^{q-1} dy, a <= x < a+beta ; 1, x >= a+beta }, -inf < x < inf`.
- `chi_square` pdf (p1038#0, source labels it `F_n` and includes `dy`): `F_n(x) = { ( x^{(n-2)/2} * e^{-x/2} ) / ( 2^{n/2} * Gamma(n/2) ) dy, x >= 0 ; 0, x < 0 }`.
- `chi_square` cdf (p1038#1): `F_n(x) = { integral_0^x ( y^{(n-2)/2} * e^{-y/2} ) / ( 2^{n/2} * Gamma(n/2) ) dy, x >= 0 ; 0, x < 0 }`.
- `gaussian_mv` pdf (p1039#0): `f_{a,C}(x) = 1 / sqrt( det(2*pi*C) ) * exp( -(1/2)*(x-a)^T * C^{-1} * (x-a) )`.
- `uniform` (Discrete) probability distribution (p1041#0; same on p1083#0): `P(X = k) = 1/(b-a), k in { a, a+1, ..., b-1 }`.
- `uniform` (Discrete) cdf (p1041#1; same on p1083#1): `F_{a,b}(x) = { 0, x < a ; (x-a+1)/(b-a), a <= x < b, x in R ; 1, x >= b }`.
- `bernoulli` cdf (p1045#0; same on p1089#0): `F_p(x) = { 0, x < 0 ; 1-p, 0 <= x < 1, x in R ; 1, x >= 1 }`.
- `geometric` cdf (p1046#0; same on p1090#1): `F_p(x) = { 0, x < 0 ; 1 - (1-p)^{floor(x+1)}, x >= 0 }, x in R`.
- `binomial` probability distribution (p1047#0): `P(X = k) = C_m^k * p^k * (1-p)^{m-k}, k in { 0, 1, ..., m }`.
- `binomial` cdf (p1048#0): `F_{m,p}(x) = { 0, x < 0 ; sum_{k=0}^{floor(x)} C_m^k * p^k * (1-p)^{m-k}, 0 <= x < m), x in R ; 1, x > m }`.
- `hypergeometric` probability distribution (p1049#0): `P(X = k) = ( C_m^k * C_{l-m}^{s-k} ) / C_l^s`.
- `hypergeometric` cdf (p1049#1): `F_{l,s,m}(x) = { 0, x < max(0, s+m-l) ; sum_{k=max(0,s+m-l)}^{floor(x)} ( C_m^k * C_{l-m}^{s-k} ) / C_l^s, max(0, s+m-l) <= x <= min(s,m) ; 1, x > min(s,m) }`.
- `poisson` probability distribution (p1051#0; same on p1087#0): `P(X = k) = ( lambda^k * e^{-lambda} ) / k!`.
- `poisson` cdf (p1051#1; same on p1087#1): `F_lambda(x) = { sum_{k=0}^{floor(x)} ( lambda^k * e^{-lambda} ) / k!, x >= 0 ; 0, x < 0 }, x in R`.
- `poisson_v` probability distribution (p1052#0): `P(X_i = k) = ( lambda_i^k * exp(-lambda_i) ) / k!, k in { 0, 1, 2, ... }`.
- `poisson_v` cdf (p1052#1): `F_{lambda_i}(x) = { sum_{k=0}^{floor(x)} ( lambda_i^k * e^{-lambda_i} ) / k!, x >= 0 ; 0, x < 0 }, x in R`.
- `negative_binomial` probability distribution (p1054#0): `P(X = k) = C_{a+k-1}^k * p^a * (1-p)^k, k in { 0, 1, 2, ... }`.
- `negative_binomial` cdf (p1054#1): `F_{a,p}(x) = { sum_{k=0}^{floor(x)} C_{a+k-1}^k * p^a * (1-p)^k, x >= 0 ; 0, x < 0 }, x in R`.
- `multinomial` probability distribution (p1055#0): `P(X_1 = x_1, ..., X_k = x_k) = ( m! / prod_{i=1}^{k} x_i! ) * prod_{i=1}^{k} p_i^{x_i}, 0 <= x_i <= m, sum_{i=1}^{k} x_i = m`.
- `negative_binomial_method::nbar` acceptance/rejection condition in the host method table (p1020#0): `(a-1)*(1-p)/p >= 100`.

Device distributions (p1076-p1093); identical to the host formula unless stated:

- device `uniform` (Continuous) pdf (p1076#0) — as host (including the `1` branch; source labels it `F_{a,b}`); cdf (p1076#1) — as host.
- device `gaussian` pdf (p1078#0) — as host (`y`, trailing `dy`); cdf (p1078#1) — as host; cdf via standard normal (p1078#2): `F_{a,sigma}(x) = phi((x-a)/sigma)`.
- device `lognormal` pdf (p1080#0), cdf (p1080#1) — as host.
- device `exponential` pdf (p1081#0), cdf (p1081#1) — as host.
- device `uniform` (Discrete) probability distribution (p1083#0), cdf (p1083#1) — as host.
- device `poisson` probability distribution (p1087#0), cdf (p1087#1) — as host.
- device `bernoulli` cdf (p1089#0) — as host.
- device `geometric` probability distribution (p1090#0): `P(X = k) = p * (1-p)^k, k = { 0, 1, 2, ... }.`; cdf (p1090#1) — as host.
- device `beta` pdf (p1092#0): `f_{p,q,alpha,beta}(x) = { ( 1/( B(p,q)*beta^{p+q-1} ) )*(x-a)^{p-1} * (beta+alpha-x)^{q-1}, alpha <= x < alpha+beta ; 0, x < alpha, x >= alpha+beta }` (first factor prints `a`, conditions `alpha`).
- device `beta` cdf (p1092#1, source labels it `F_{a,b}`): `F_{a,b}(x) = { integral_alpha^x ( 1/( B(p,q)*beta^{p+q-1} ) )*(y-alpha)^{p-1} * (beta+alpha-y)^{q-1} dy, alpha <= x < alpha+beta, x in R ; 1, x >= alpha+beta ; 0, x < alpha }`.
- device `gamma` pdf (p1093#0): `f_{a,alpha,beta}(x) = { ( 1/( Gamma(alpha)*beta^alpha ) )*(x-a)^{alpha-1}*e^{-(x-a)/beta}, x >= a ; 0, x < a }`.
- device `gamma` cdf (p1093#1): `F_{a,alpha,beta}(x) = { integral_a^x ( 1/( Gamma(alpha)*beta^alpha ) )*(y-a)^{alpha-1}*e^{-(y-a)/beta} dy, x >= a ; 0, x < a }`.
Diagram pages p989 and p1058 are RNG structure/usage schematics and carry no formulas. The two splitting figures do carry substantive content and are described here:

- **`leapfrog` interleaved splitting (p1007#0):** numbered stream `1..21` with three nodes taking every third element — node 1 generates `1, 4, 7, 10, 13, 16, 19`; node 2 generates `2, 5, 8, 11, 14, 17, 20`; node 3 generates `3, 6, 9, 12, 15, 18, 21`. This is `leapfrog(engine, idx, stride)` with `stride = 3`: each node's elements form the arithmetic progression starting at `idx+1` with step `stride`.
- **Block (contiguous) splitting (p1008#0):** numbered stream `1..21` split into three contiguous blocks — node 1 generates `1, 2, 3, 4, 5, 6, 7`; node 2 generates `8, 9, 10, 11, 12, 13, 14`; node 3 generates `15, 16, 17, 18, 19, 20, 21`. This is the `skip_ahead` model: each node skips a contiguous prefix.

The practical contrast: `leapfrog` interleaves streams (stride-separated), `skip_ahead` partitions them into contiguous blocks.

62 labelled formula images/expressions were transcribed (42 host-side page tags, 20 device-side page tags), plus the two splitting diagrams described above.

## Conventions & Gotchas

- **Engine owns the queue.** Every host engine constructor takes a `sycl::queue`; `generate` submits kernels into that queue. The device API takes no queue: pass explicit `seed` and `offset`.
- **Return types.** Host buffer `generate` returns `void`; host USM `generate` returns `sycl::event`; device `generate` returns `Distr::result_type` when `Engine::vec_size == 1`, else `sycl::vec<Distr::result_type, Engine::vec_size>`; device `generate_single` always returns a scalar.
- **Dependencies.** Only USM interfaces carry `const std::vector<sycl::event>& dependencies`; the host `generate` syntax shows no default for it.
- **vec_size / element counts.** Device engines use `template<std::int32_t VecSize = 1>` with `VecSize` ∈ {1, 2, 3, 4, 8, 16} (as `sycl:vec` class size); work-items cover `n / VecSize`.
- **Offsets / splitting.** Device `offset` = skipped elements; the `initializer_list<std::uint64_t>` form encodes `num_to_skip[0] + num_to_skip[1]*2^64 + num_to_skip[2]*2^128 + ...` (needed above 2^64). Host `skip_ahead` uses the same encoding. `leapfrog(engine, idx, stride)` = interleaved streams; `skip_ahead` = contiguous blocks. For quasi-random generators skip components, not vectors (`num_to_skip = num_to_skip * dim`).
- **Accurate vs fast.** Pick `*_accurate` when values must stay in the definitional domain for all parameter values; `uniform_method::standard` uses the `s` BRNG type on GPU (host) / `float` (device) and can give incorrect statistics due to rounding over wide integer ranges.
- **Integer distributions.** `bernoulli`, `geometric`, `binomial`, `hypergeometric`, `poisson`, `poisson_v`, `negative_binomial`, `multinomial` are integer-valued (host `std::int32_t`/`std::uint32_t`; device `poisson` also `std::int64_t`, device `geometric` also 64-bit, device `bernoulli` also 8-/16-bit). `bits`/`uniform_bits` return raw BRNG bits.
- **`sycl::span` lifetime.** `gaussian_mv` (mean, matrix), `poisson_v` (lambda) and `multinomial` (p) hold spans; the user must keep the underlying memory alive.
- **State serialization.** `get_state_size(engine)` → allocate `std::uint8_t[mem_size]` → `save_state(engine, mem)`; `load_state<Engine>(queue, mem)` creates a new engine and may use a different queue (the example saves a GPU engine state and loads it for a CPU queue). File-based forms are deprecated since 2024.1.
- **Host/device differences.** Host `philox4x32x10` `default_seed = 0` vs device `default_seed = 1`; host `mcg59` seed `std::uint64_t` vs device `mcg59` seed `std::uint32_t`; device `pcg64_dxsm` `default_seed = 0`; quasi-random `default_dimensions_number = 1` (use ≥ 2 when a distribution needs more than one BRNG value).
- **Engine state across kernels.** Store engines in `sycl::buffer`/USM, or use `engine_descriptor`/`engine_accessor` with `load(id)`/`store(engine, id)`; the `engine_descriptor` constructor itself submits a `parallel_for` initializing `Engine{seed, id * offset}`.
- **Rejection bookkeeping.** Device `beta`/`gamma` expose `count_rejected_numbers()`; `count_engine_adaptor` exposes `get_count()` and `base()`.
- **Build switch.** `MKL_RNG_USE_BINARY_CODE` plus static oneMKL libraries (per the Link Line Advisor) enables device `gaussian_method::box_muller2`, device `gaussian_method::icdf` (needs device double precision for `float`), device `lognormal_method::box_muller2`, and extra Vector-Math optimizations for `exponential`, `beta`, `gamma`.
- **Template defaults.** Host distributions default value types to `float` (or `std::int32_t` for integer distributions) and methods to `<family>_method::by_default`; `gaussian_mv` defaults to `layout::packed`. Device distribution syntax blocks show no default template arguments.
- **Host `uniform` (Discrete)** is the partial specialization `class uniform<std::(u)int32_t, Method>` with `result_type = std::(u)int32_t`.

## Explicit gaps

- Chunk-073 continues past RNG into **Summary Statistics** (`oneapi::mkl::stats::dataset`, `make_dataset`, `raw_sum`, `central_sum`, layouts `row_major`/`col_major`, methods `fast`/`one_pass`, include `oneapi/mkl/stats.hpp`) — a separate domain, not documented here. Chunk-066 opens with the tail of the Vector Math domain (`fdim`, `fmax`, `fmin`, `maxmag`, `minmag`), also out of scope.
- Parameter-constraint formulas were not among the supplied formula images (the text reads "where ." or "( )"). Missing: constraints for uniform/laplace/weibull/cauchy/rayleigh/lognormal/gumbel/gamma/beta/bernoulli/geometric/binomial/hypergeometric/poisson/poisson_v/negative_binomial/multinomial; the device `uniform_method::standard` rounding condition; the GPU `multinomial` condition on k; the `gamma_method::marsaglia`, `beta_method::cja` and `chi_square_method::gamma_based` case conditions; the `poisson`/`poisson_v` `gaussian_icdf_based` λ ranges.
- Poisson `devroye` λ inequalities: endpoints 60 and 1000 are visible but the comparison operators are not.
- The PRNG iteration/termination step (`for ... and ...`) and the output symbols `u_i` did not survive extraction.
- Periods printed as empty formulas: PHILOX4X32X10, MT19937/SFMT19937, MT2203, ARS5, PCG64 DXSM, and the MCG constant arguments in the descriptive tables.
- `gaussian_method::box_muller`/`box_muller2` and `lognormal_method::box_muller2` closed forms are referenced by the method table ("according to the formula(s): .") but their images were not in the supplied set.
- The `VS Notes` document repeatedly referenced for engine support, rejection thresholds and state sizes is not in these chunks, so per-engine `skip_ahead`/`leapfrog` support lists are not reproducible.
- `oneapi::mkl::rng::default_engine` appears only in the `save_state`/`load_state` example; no class documentation is present.
- Source typos reproduced and flagged inline: `unform_method::accurate`, `oneapi::mkl::rng::method::accurate`, `Real_Type`, `std::int32_T`, `typename_IntType`, `reccurence`, `sizeof(Type) == 32`, `~mt2203()` without semicolon, exponential constructor parameters `mean, stddev`, and the `engine_accessor` syntax printed under `engine_descriptor`.
