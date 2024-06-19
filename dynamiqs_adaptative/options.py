from __future__ import annotations

import equinox as eqx
import jax
from jax import Array
from jaxtyping import PyTree, ScalarLike

from .time_array import TimeArray

from ._utils import tree_str_inline

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
        t0: Initial time. If `None`, defaults to the first time in `tsave`.
        save_extra _(function, optional)_: A function with signature
            `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
            This can be used to save additional arbitrary data during the
            integration. the results are saved in extra of [dynamiqs/result.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/result.py)
        estimator: if 'True' activates an estimator to verify that the truncature in
            space is large enough. The simulation will be run on smaller cavities'
            size than the given operators' size. 
        tensorisation: (expects estimator to be 'True')
            if not 'None' explain to the estimator that we are dealing with
            a n dimensional object. An input could be (2,3) for an object tensorised 
            according to ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)) for instance.
        inequalities: (expects estimator and reshaping to be 'True', tensorisation to be 
            given)
            For a n-dimensional object, you can give your own truncature to the
            estimator. It has to be formated like a list of 2 objects list [param, f].
            param: a float.
            f: a function that has len("number of dimensions") inputs and outputs a 
            float.
            Exemple: [param = 2, f = def f(i, j): return i+j] for a 2D tensorisation
            gives [lambda i, j: i+j < 2]    
            Default inequalities are rectangular ones (rectangular in the dimensions),
            such as [lambda i0, .., in: i0 < param0, etc...]
        reshaping: (expects estimator to be 'True')
            if 'True' activates a dynamic reshaping of the simulation's size.
            It expects that you give operators much larger than needed for the
            simulation, since their shape will be dynamically reshaped to the faster fit
            that allows the error to be below an estimator of it.
            If the operators given end up to not be large enough for your simulation,
            you will be noticed.
        estimator_rtol: Defines the relative tolerance of the estimator towards the
            tolerance of the solver. The estimator on the error cannot be beneath
            the solver's precision, so when branching the estimator to your computation,
            it checks that " estimated_error < estimator_rtol * solver_tolerance ",
            and output an error if it's not the case.
            The default value (100) has been set empirically.
        trunc_size: The 'k' needed to compute the error, 
            (See [an explaination on the technique not made yet](https://itdoesnotexistyet))
            By default a function [link] try to guess the degree of the jump operators
            and the Hamiltionian you put in a and dag(a). Then k is 2 times the max of
            those degree (See [demo]). It supposes that H and the jump operators are
            polynomial in creation and destruction operators.
            If otherwise, you have to figure it out and give it in options to the 
            estimator.
            With n cavities k is the maximum regarding all directions.
            Works poorly if H and the jumps operators are time dependant (unable to
            work if the trunc_size varies with time, which can happen for an operator in
            a + cos(t)a@a for instance)
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: ScalarLike | None = None
    save_extra: callable[[Array], PyTree] | None = None
    estimator: bool = False
    reshaping: bool | list = False
    tensorisation: tuple | None = None
    inequalities: list | None = None
    estimator_rtol : int | None = 100
    trunc_size: Array | None = None
    # parameters the user is not supposed to touch
    projH: TimeArray | None = None
    projL: TimeArray | None = None
    mask: Array | None = None

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        t0: ScalarLike | None = None,
        save_extra: callable[[Array], PyTree] | None = None,
        estimator: bool = False,
        reshaping: bool | list = False,
        tensorisation: tuple | None = None,
        inequalities: list | None = None,
        estimator_rtol : int | None = 100,
        trunc_size: Array | None = None,
        projH: TimeArray | None = None,
        projL: TimeArray | None = None,
        mask: Array | None = None,
    ):
        
        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.t0 = t0
        self.estimator = estimator
        self.tensorisation = tensorisation
        self.inequalities = inequalities
        self.reshaping = reshaping
        self.estimator_rtol = estimator_rtol
        self.trunc_size = trunc_size
        self.projH = projH
        self.projL = projL
        self.mask = mask

        # make `save_extra` a valid Pytree with `jax.tree_util.Partial`
        if save_extra is not None:
            save_extra = jax.tree_util.Partial(save_extra)
        self.save_extra = save_extra

    def __str__(self) -> str:
        return tree_str_inline(self)
