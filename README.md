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



## Adaptative solver

This in an example of a reshaping for some dynamic defined in H, jump_ops, and an initial rho, with the adaptative solver using all its options.
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
    H(n), jump_ops(n), rho(n), t_span, solver=dq.solver.Tsit5(atol = solver_atol, rtol = solver_rtol, max_steps=10000), 
    options = dq.Options(estimator=True, reshaping=True, tensorisation=tensorisation, inequalities=inequalities,estimator_rtol=estimator_rtol, downsizing_rtol=downsizing_rtol)
)
```

## Non-trivial truncations

The branch "dynamiqs_adaptative - Inequalities ab initio" allows to run faster a simulation that would only need to use some non-trivial truncations.
