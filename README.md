# Dynamiqs_adaptative
A version of dynamiqs implementing a solver adaptative in the size of the modes for the Lindblad master equation.

## Truncation estimator 

This version of dynamiqs implements an estimator of the error made by the truncation from an infinite Fock Hilbert space $$\mathcal{H}$$ to a finite one $$\mathcal{H}_N \subset \mathcal{H}$$.

## Adaptative solver

This option allows to dynamically adjust the truncation of the Hilbert space, enabling fully adaptive simulations of the density matrix.

## Non-trivial truncations

For multimode simulations, the trivial finite subspace to simulate on, that we call $$\mathcal{H}_N$$ is:

$$ \mathcal{H}_N = \text{Span}\left( \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq i_1 \leq N_1, \cdots, 0 \leq i_m \leq N_m \right) \quad  N_1,\ ...,\ N_m \in \mathcal{N}^m $$

But it could be any finite subspace for instance:

$$ \mathcal{H}_N = \text{Span}\left(  \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq \sum_j i_j \leq N \right) \quad  n \in \mathcal{N} $$

Depending on ones needs some subspaces may be more interesting than others. We allow one to choose such non-trivial truncations. 
<!-- $$ \mathcal{H}_N = \text{Span}\{ \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq \sum_{j=0}^{m} i_j \leq N \} $$ -->

# Installation 

```shell
pip install git+https://github.com/etienney/dynamiqs_adaptative.git
```


# Examples

## Truncation estimator 

This is a basic example of a 1-mode dynamic defined in H, jump_ops, and an initial rho, with the adaptative solver using all its options.
```python
import dynamiqs_adaptative as dq
import jax.numpy as jnp
dq.set_precision('double') 

def H(n):
    return jnp.zeros((n,n))
def jump_ops(n, alpha=1.0):
    a = dq.destroy(n)
    kappa=1.0
    return [jnp.sqrt(kappa)*(a@a-jnp.identity(n)*alpha**2)]
def rho(n):
    return dq.fock_dm(n,0) #|0><0|

ntemps, steps = 1.0, 100
t_span = jnp.linspace(0, ntemps, steps)
n=70 
tensorisation=(n,) # We have to specify to the computer that the density matrix is defined as 1-mode of size n
res = dq.mesolve(
    H(n), jump_ops(n), rho(n), t_span,
    solver=dq.solver.Tsit5(atol= 1e-14, rtol= 1e-14, max_steps = 10000),
    options = dq.Options(estimator=True, tensorisation=tensorisation)
)
print(res)
```

The output specifies the smaller space the simulation has been done on and the output of the estimator:
```Markdown
|███████████████████████████████████████████████████████████████████████████████████████████████| 100.0% ◆ elapsed 366.38ms ◆ remaining 0.00ms
==== MEResult ====
Solver           : Tsit5
States           : Array complex128 (292, 40, 40) | 7.13 Mb
Estimator        : [5.8022265e-14+0.j]
Simulation size  : (36,)
Original size    : (40,)
Infos            : 341 steps (291 accepted, 50 rejected)
```
We know that for the above dynamics run on a Hilbert subspace $$\mathcal{H}_N \subset \mathcal{H}$$ with $$\mathcal{H}_N = \text{Span}\left( \ket{n} \mid 0 \leq n \leq 36 \right)$$, the error made by truncating the dynamics from $$\mathcal{H}$$ to $$\mathcal{H}_N$$ is less than 5.8022265e-14. Note that it may be limited by the solver precision set at 1e-14. (It is indeed the case here.)

## Adaptative solver

This is a basic example of a simulation with dynamical reshaping for a 1-mode dynamic defined in H, jump_ops, and an initial rho, with the adaptative solver using all its options.
```python
import dynamiqs_adaptative as dq

n=70 # the maximum size we can access numerically
tensorisation=(n,)
solver_atol, solver_rtol = 1e-14, 1e-14
estimator_rtol = 500 # Such that the minimal precision we aim at for the estimator is limit = 500*(solver_atol + solver_rtol)*t/(total_time)
downsizing_rtol = 5 # Such that if we have an estimator that is under limit/5 we reduce the size of our objects to win computation time
initial_inequality, extend_by, reduce_by = 50, 4, 4 # such that we select some initial states respecting the inequalities set in the first parameters among the variable "inequalities" (next line) for the parameter initial_inequality,
# and we will reshape respecting some inequalities for a parameter + "extend_by" while extending or the parameter - "reduce_by" while downsizing
inequalities = [[lambda a: a, initial_inequality, extend_by, reduce_by]] # a list of some inequalities set as [the inequality as a lambda function, the initial parameter for those inequalites, by how much we extend the parameter that control via the inequality the states we look at, by how much we reduce it]

res = dq.mesolve(
    H(n), jump_ops(n), rho(n), t_span, 
    solver=dq.solver.Tsit5(atol = solver_atol, rtol = solver_rtol, max_steps=10000), 
    options = dq.Options(estimator=True, reshaping=True, tensorisation=tensorisation, inequalities=inequalities,
    estimator_rtol=estimator_rtol, downsizing_rtol=downsizing_rtol)
)
```

This is the same for a 2-mode simulation using some non-trivial truncation.
```python
import dynamiqs_adaptative as dq
n_a, n_b = 60, 30
tensorisation=(n_a, n_b)
solver_atol, solver_rtol = 1e-14, 1e-14
estimator_rtol = 50000000
downsizing_rtol = 5
def ineq(a, b):
    return 0.5*a+b
inequalities = [[ineq, 6, 7, 5]]
res = dq.mesolve(H(n_a, n_b), jump_ops(n_a, n_b), rho(n_a, n_b), t_span, 
    solver=dq.solver.Tsit5(atol = solver_atol, rtol = solver_rtol, max_steps=3000), 
    options = dq.Options(estimator=True, reshaping=True, tensorisation=tensorisation, inequalities=inequalities,
    estimator_rtol=estimator_rtol, downsizing_rtol=downsizing_rtol)
)
```

## Non-trivial truncations

The branch "dynamiqs_adaptative - Inequalities ab initio" allows to run faster a simulation that would only need to use some non-trivial truncations.
