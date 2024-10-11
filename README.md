<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
$$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$
# Dynamiqs_adaptative
A version of dynamiqs implementing a solver adaptative in the size of the modes for the Lindblad master equation.

## Truncation estimator 

This version of dynamiqs implements an estimator of the error made by the truncation from an infinite Fock space to a finite one.

## Adaptative solver

It allows to dynamically adjust the truncation of the Hilbert space, enabling fully adaptive simulations of the density matrix.

## Non-trivial truncations

For multimode simulations, the usual finite subspace to simulate on, that we call $$\mathcal{H}_N \subset \mathcal{H}$$ is:

$$ \mathcal{H}_N = \text{Span}\left( \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq i_1 \leq N_1,\, \cdots,\, 0 \leq i_m \leq N_m \right) $$

But it could be any finite subspace for instance:

$$ \mathcal{H}_N = \text{Span}\left(  \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq \sum_j i_j \leq N \right)$$

Depending on ones needs some subspaces may be more interesting than others.
<!-- $$ \mathcal{H}_N = \text{Span}\{ \ket{i_1} \otimes \cdots \otimes \ket{i_m} \mid 0 \leq \sum_{j=0}^{m} i_j \leq N \} $$ -->

# Installation 

```shell
pip install git+https://github.com/etienney/dynamiqs_adaptative.git
```
