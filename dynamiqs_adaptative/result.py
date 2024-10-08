from __future__ import annotations

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from .gradient import Gradient
from .options import Options
from .solver import Solver

__all__ = ['SEResult', 'MEResult']


def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'
    else:
        return f'{mem / 1024**3:.2f} Gb'


def array_str(x: Array | None) -> str | None:
    return None if x is None else f'Array {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


# the Saved object holds quantities saved during the equation integration
class Saved(eqx.Module):
    ysave: Array
    Esave: Array | None
    extra: PyTree | None

class Saved_estimator(Saved):
    # save additional data needed to compute the estimator
    destimator: Array | None
    estimator: Array | None
    time: Array | None
    inequalities: Array | None
    
    def __init__(self, ysave, Esave, extra, destimator, estimator, time, inequalities):
        super().__init__(ysave, Esave, extra)
        self.destimator = destimator
        self.estimator = estimator
        self.time = time
        self.inequalities = inequalities


class Result(eqx.Module):
    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    _saved: Saved
    infos: PyTree | None

    @property
    def states(self) -> Array:
        return self._saved.ysave

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra
    
    @property
    def estimator(self) -> Array:
        if self.options.estimator:
            return self._saved.estimator
        else:
            raise ValueError('Calling estimator without using it does not make sense. '
                             'Try putting \'options = dq.Options(estimator=True)\'')
    @property
    def destimator(self) -> Array:
        return self._saved.destimator

    
    @property
    def time(self) -> PyTree | None:
        return self._saved.time
    
    @property
    def inequalities(self)-> Array | None:
        return self._saved.inequalities
    
    def _str_parts(self) -> dict[str, str]:
        if self.options.estimator:
            if self.options.tensorisation is None:
                simu_size = ((self.states).shape)[0] - self.options.trunc_size
                given_size = ((self.states).shape)[0]
            else:
                simu_size = [
                    self.options.tensorisation[i] - self.options.trunc_size[i]
                    for i in range(len(self.options.tensorisation))
                ]
                given_size = self.options.tensorisation
                # print it in a nicer way, without the "Array"
                simu_size = tuple(arr.item() for arr in simu_size)
                given_size = tuple(arr.item() for arr in given_size)
            estimator = self.estimator[-1]

        return {
            'Solver  ': type(self.solver).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'States  ': array_str(self.states) ,
            'Estimator ': (
                estimator if self.options.estimator else None
            ),
            'Simulation size ': (
                simu_size if self.options.estimator and not self.options.reshaping 
                else None
            ),
            'Original size ': (
                given_size if self.options.estimator and not self.options.reshaping 
                else None
            ),
            'Expects ': array_str(self.expects),
            'Extra   ': (
                eqx.tree_pformat(self.extra) if self.extra is not None else None
            ),
            'Infos   ': (
                self.infos if self.infos is not None and not self.options.reshaping 
                else None
            ),
        }

    def __str__(self) -> str:
        parts = self._str_parts()

        # remove None values
        parts = {k: v for k, v in parts.items() if v is not None}

        # pad to align colons
        padding = max(len(k) for k in parts) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return f'==== {self.__class__.__name__} ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError


class SEResult(Result):
    r"""Result of the Schrödinger equation integration.

    Attributes:
        states _(array of shape (..., ntsave, n, 1))_: Saved states.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting states and expectation values are batched according to the
        leading dimensions of the Hamiltonian `H` and initial state `psi0`. The
        behaviour depends on the value of the `cartesian_batching` option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...psi0
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `psi0` has shape _(4, n, 1)_,

            then `states` has shape _(2, 3, 4, ntsave, n, 1)_.

        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...psi0  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `psi0` has shape _(3, n, 1)_,

            then `states` has shape _(2, 3, ntsave, n, 1)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """


class MEResult(Result):
    """Result of the Lindblad master equation integration.

    Attributes:
        states _(array of shape (..., ntsave, n, n))_: Saved states.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting states and expectation values are batched according to the
        leading dimensions of the Hamiltonian `H`, jump operators `jump_ops` and initial
        state `rho0`. The behaviour depends on the value of the `cartesian_batching`
        option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...L0, ...L1, (...), ...rho0
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
            - `rho0` has shape _(7, n, n)_,

            then `states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.
        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...L0 = ...L1 = (...) = ...rho0  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
            - `rho0` has shape _(3, n, n)_,

            then `states` has shape _(2, 3, ntsave, n, n)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """
