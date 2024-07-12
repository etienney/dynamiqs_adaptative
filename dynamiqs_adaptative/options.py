from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jax import Array
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter

__all__ = ['Options']


class Options(eqx.Module):
    """Generic options for the quantum solvers.

    Args:
        save_states: If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        verbose: If `True`, print information about the integration, otherwise
            nothing is printed.
        cartesian_batching: If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        progress_meter: Progress meter indicating how far the solve has progressed.
            Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress meter. Pass
            `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        t0: Initial time. If `None`, defaults to the first time in `tsave`.
        save_extra _(function, optional)_: A function with signature
            `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
            This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers (see
            [`SEResult`][dynamiqs.SEResult] or [`MEResult`][dynamiqs.MEResult]).
        estimator: if 'True' activates an estimator to verify that the truncature in
            space is large enough.
        tensorisation: (expects estimator to be 'True')
            if not 'None' explain to the estimator that we are dealing with
            a n dimensional object. An input could be (2,3) for an object tensorised 
            according to ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)) for instance.
        trunc_size: The larger space needed to compute the error, 
            (See [an explaination on the technique not made yet](https://itdoesnotexistyet))
            By default a function [link] try to guess the degree of the jump operators
            and the Hamiltionian you put in a and dag(a). Then k is 2 times the max of
            those degree (See [demo]). It supposes that H and the jump operators are
            polynomial in creation and destruction operators.
            If otherwise, you have to figure it out and give it in options to the 
            estimator.
            With n cavities trunc_size is thus a list.
            Works poorly if H and the jumps operators are time dependant (unable to
            work if the polynomial degree varies with time, which can happen for an 
            operator in a + cos(t)a@a for instance)
        estimator_rtol: Defines the relative tolerance of the estimator towards the
            tolerance of the solver. The estimator on the error cannot be beneath
            the solver's precision, so when branching the estimator to your computation,
            it checks that " estimated_error < estimator_rtol * solver_tolerance ",
            and output an error if it's not the case.
            The default value (100) has been set empirically.
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    save_extra: callable[[Array], PyTree] | None = None
    estimator: bool = False
    tensorisation: tuple | None = None
    trunc_size: Array | None = None
    estimator_rtol : float | None = 100

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        save_extra: callable[[Array], PyTree] | None = None,
        estimator: bool = False,
        tensorisation: tuple | None = None,
        trunc_size: Array | None = None,
        estimator_rtol : float | None = 100,
    ):
        if progress_meter is None:
            progress_meter = NoProgressMeter()

        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.estimator = estimator
        self.tensorisation = tensorisation
        self.trunc_size = trunc_size
        self.estimator_rtol = estimator_rtol
        
        # make `save_extra` a valid Pytree with `Partial`
        if save_extra is not None:
            save_extra = jtu.Partial(save_extra)
        self.save_extra = save_extra

    def __str__(self) -> str:
        return tree_str_inline(self)
