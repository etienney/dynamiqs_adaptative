<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

When \( a \ne 0 \), there are two solutions to \( ax^2 + bx + c = 0 \) given by

$$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

# Dynamiqs_adaptative
A version of dynamiqs implementing a solver adaptative in the size of the modes for the Lindblad master equation.

## Truncation estimator 

This version of dynamiqs implements an estimator of the error made by the truncation from an infinite Fock space to a finite one.

## Adaptative solver

It allows to dynamically adjust the truncation of the Hilbert space, enabling fully adaptive simulations of the density matrix.

## Non-trivial truncations



# Installation 

```shell
pip install git+https://github.com/etienney/dynamiqs_adaptative.git
```
