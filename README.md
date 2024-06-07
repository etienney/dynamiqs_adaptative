<h1 align="center">
    <img src="./docs/media/dynamiqs-logo.png" width="520" alt="dynamiqs library logo">
</h1>

[P. Guilmin](https://github.com/pierreguilmin), [R. Gautier](https://github.com/gautierronan), [A. Bocquet](https://github.com/abocquet), [E. Genois](https://github.com/eliegenois)

[![ci](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml?query=branch%3Amain)  ![python version](https://img.shields.io/badge/python-3.9%2B-blue) [![chat](https://badgen.net/badge/icon/on%20slack?icon=slack&label=chat&color=orange)](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA) [![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow)](https://github.com/dynamiqs/dynamiqs/blob/main/LICENSE) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

High-performance quantum systems simulation with JAX.

**dynamiqs** is a Python library for **GPU-accelerated** and **differentiable** quantum simulations. Solvers are available for the Schrödinger equation, the Lindblad master equation, and the stochastic master equation. The library is built with [JAX](https://jax.readthedocs.io/en/latest/index.html) and the main solvers are based on [Diffrax](https://github.com/patrick-kidger/diffrax).

Documentation is available on our website, <https://www.dynamiqs.org>; see the [Python API](https://www.dynamiqs.org/python_api/index.html) for a list of all implemented functions.

The main features of **dynamiqs** are:

- Running simulations on **CPUs** and **GPUs** with high-performance.
- Executing many simulations **concurrently** by batching over Hamiltonians, initial states or jump operators.
- Computing **gradients** of arbitrary functions with respect to arbitrary parameters of the system.
- Full **compatibility** with the [JAX](https://jax.readthedocs.io/en/latest/index.html) ecosystem with a [QuTiP](https://qutip.org/)-like API.

We hope that this library will prove useful to the community for e.g. simulation of large quantum systems, gradient-based parameter estimation or quantum optimal control. The library is designed for large-scale problems, but also runs efficiently on CPUs for smaller problems.

> [!WARNING]
> This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

## Installation

We will soon make a first release of the library on PyPi. In the meantime, you can install directly from source:

```shell
pip install git+https://github.com/dynamiqs/dynamiqs.git
```

## Examples

### Simulate a lossy quantum harmonic oscillator

This first example shows simulation of a lossy harmonic oscillator with Hamiltonian $H=\omega a^\dagger a$ and a single jump operator $L=\sqrt{\kappa} a$.

```python
import dynamiqs as dq
import jax.numpy as jnp

# parameters
n = 128      # Hilbert space dimension
omega = 1.0  # frequency
kappa = 0.1  # decay rate
alpha = 1.0  # initial coherent state amplitude

# initialize operators, initial state and saving times
a = dq.destroy(n)
H = omega * dq.dag(a) @ a
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = dq.coherent(n, alpha)
tsave = jnp.linspace(0, 1.0, 101)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
print(result)
```

```text
==== MEResult ====
Solver  : Tsit5
States  : Array complex64 (101, 128, 128) | 12.62 Mb
Infos   : 7 steps (7 accepted, 0 rejected)
```

### Compute gradients with respect to some parameters

Suppose that in the above example, we want to compute the gradient of the number of photons in the final state, $\bar{n} = \mathrm{Tr}[a^\dagger a \rho(t_f)]$, with respect to the decay rate $\kappa$ and the initial coherent state amplitude $\alpha$.

```python
import dynamiqs as dq
import jax.numpy as jnp
import jax

# parameters
n = 128      # Hilbert space dimension
omega = 1.0  # frequency
kappa = 0.1  # decay rate
alpha = 1.0  # initial coherent state amplitude

def population(omega, kappa, alpha):
    """Return the oscillator population after time evolution."""
    # initialize operators, initial state and saving times
    a = dq.destroy(n)
    H = omega * dq.dag(a) @ a
    jump_ops = [jnp.sqrt(kappa) * a]
    psi0 = dq.coherent(n, alpha)
    tsave = jnp.linspace(0, 1.0, 101)

    # run simulation
    result = dq.mesolve(H, jump_ops, psi0, tsave)

    return dq.expect(dq.number(n), result.states[-1]).real

# compute gradient with respect to omega, kappa and alpha
grad_population = jax.grad(population, argnums=(0, 1, 2))
grads = grad_population(omega, kappa, alpha)
print(f'Gradient w.r.t. omega={grads[0]:.2f}')
print(f'Gradient w.r.t. kappa={grads[1]:.2f}')
print(f'Gradient w.r.t. alpha={grads[2]:.2f}')
```

```text
Gradient w.r.t. omega=0.00
Gradient w.r.t. kappa=-0.90
Gradient w.r.t. alpha=1.81
```

## More features!

Below are some cool features of **dynamiqs** that are either already available or planned for the near future.

**Solvers**

- Choose between a variety of solvers, from **modern** ODE solvers (e.g. Tsit5 and PID controllers for adaptive step-sizing) to **quantum-tailored** solvers that preserve the physicality of the evolution (the state trace and positivity are preserved).
- Simulate **time-varying problems** (both Hamiltonian and jump operators) with support for various formats (piecewise constant operator, constant operator modulated by a time-dependent factor, etc.).
- Define a **custom save function** during the evolution (e.g. to register only the state purity, to track a subsystem by taking the partial trace of the full system, or to compute the population in the last Fock states to regularise your QOC problem).
- Easily implement **your own solvers** by subclassing our base solver class and focusing directly on the solver logic.
- Simulate SME trajectories **orders of magnitude faster** by batching the simulation over the stochastic trajectories.
- Use **adaptive step-size solvers** to solve the SME (based on Brownian bridges to generate the correct statistics).
- **Parallelise** large simulations across multiple CPUs/GPUs.

**Gradients**

- Choose between **various methods** to compute the gradient, to tradeoff speed and memory (e.g. use the optimal online checkpointing scheme of [Diffrax](https://github.com/patrick-kidger/diffrax) to compute gradients for large systems).
- Compute gradients with **machine-precision accuracy**.
- Evaluate **derivatives with respect to evolution time** (e.g. for time-optimal quantum control).
- Compute **higher order derivatives** (e.g. the Hessian).

**Utilities**

- Balance **accuracy and speed** by choosing between single precision (`float32` and `complex64`) or double precision (`float64` and `complex128`).
- Plot beautiful figures by using our **handcrafted plotting function**.
- Apply any functions to **batched arrays** (e.g. `dq.wigner(states)` to compute the wigners of many states at once).
- Use **QuTiP objects as arguments** to any functions (e.g. if you have existing code to define your Hamiltonian in QuTiP, or if you want to use our nice plotting functions on a list of QuTiP states).

**Library development**

- Enjoy **modern software development practices and tools**.
- Build confidence from the **analytical tests** that verify state correctness and gradient accuracy for every solver, at each commit.

**Coming soon**

- Discover a custom **sparse format**, with substantial speedups for large systems.
- Use **implicit** ODE solvers.
- Simulate using propagators solvers based on **Krylov subspace methods**.
- **Benchmark code** to compare solvers and performance for different systems.

## The dynamiqs project

**Philosophy**

There is a noticeable gap in the availability of an open-source library that simplifies gradient-based parameter estimation and quantum optimal control. In addition, faster simulations of large systems are essential to accelerate the development of quantum technologies. The **dynamiqs** library addresses both of these needs. It aims to be a fast and reliable building block for **GPU-accelerated** and **differentiable** solvers. We also work to make the library compatible with the existing Python ecosystem (i.e. JAX and QuTiP) to allow easy interfacing with other libraries.

**Team and sponsoring**

The library is being developed by a **team of physicists and developers**. We are working with theorists, experimentalists, machine learning practitioners, optimisation and numerical methods experts to make the library as useful and as powerful as possible. The library is sponsored by the startup [Alice & Bob](https://alice-bob.com/), where it is being used to simulate, calibrate and control chips made of superconducting-based dissipative cat qubits.

**History**

Development started in early 2023, the library was originally based on PyTorch with homemade solvers and gradient methods. It was completely rewritten in JAX in early 2024 for performance.

## Let's talk!

If you're curious, have questions or suggestions, wish to contribute or simply want to say hello, please don't hesitate to engage with us, we're always happy to chat! You can join the community on Slack via [this invite link](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA), open an issue on GitHub, or contact the lead developer via email at <pierreguilmin@gmail.com>.

## Contributing

We warmly welcome all contributions. If you're a junior developer or physicist, you can start with a small utility function, and move on to bigger problems as you discover the library's internals. If you're more experienced and want to implement more advanced features, don't hesitate to get in touch to discuss what would suit you. Please refer to [CONTRIBUTING.md](https://github.com/dynamiqs/dynamiqs/blob/main/CONTRIBUTING.md) for detailed instructions.

## Citing dynamiqs

If you have found this library useful in your academic research, you can cite:

```bibtex
@unpublished{guilmin2024dynamiqs,
  title  = {dynamiqs: an open-source Python library for GPU-accelerated and differentiable simulation of quantum systems},
  author = {Pierre Guilmin and Ronan Gautier and Adrien Bocquet and {\'{E}}lie Genois},
  year   = {2024},
  url    = {https://github.com/dynamiqs/dynamiqs}
}
```

> P. Guilmin, R. Gautier, A. Bocquet, E. Genois. dynamiqs: an open-source Python library for GPU-accelerated and differentiable simulation of quantum systems (2024), in preparation.
