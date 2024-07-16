from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from .._checks import check_shape, check_times
from .._utils import cdtype
from ..core._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_solver_class,
)
from ..gradient import Gradient
from ..options import Options
from ..result import MEResult
from ..solver import (
    Dopri5,
    Dopri8,
    Euler,
    Kvaerno3,
    Kvaerno5,
    Propagator,
    Rouchon1,
    Solver,
    Tsit5,
)
from ..time_array import Shape, TimeArray
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEDopri8, MEEuler, MEKvaerno3, MEKvaerno5, METsit5
from .mepropagator import MEPropagator
from .merouchon import MERouchon1

# from ..a_posteriori.utils.mesolve_fcts import (
    # mesolve_estimator_init,
    # mesolve_iteration_prepare
# )
from ..estimator.saves import collect_saved_estimator
from ..estimator.mesolve_fcts import mesolve_estimator_init
from ..estimator.utils.warnings import warning_estimator_tol_reached
# from ..a_posteriori.utils.utils import find_approx_index, put_together_results
# from ..a_posteriori.n_D.inequalities import *
# from ..a_posteriori.n_D.reshapings import (
#     reduction_nD, extension_nD, projection_nD, mask, dict_nD
# )
# from ..a_posteriori.n_D.reshaping_y import reshaping_init, reshaping_extend
# from ..a_posteriori.n_D.estimator_derivate_nD import estimator_derivate_opti_nD
from ..utils.utils import dag
from ..result import Result, Saved
import time

__all__ = ['mesolve']


def mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MEResult:
    r"""Solve the Lindblad master equation.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the Lindblad master
    equation ($\hbar=1$)
    $$
        \frac{\dd\rho(t)}{\dt} = -i[H(t), \rho(t)]
        + \sum_{k=1}^N \left(
            L_k(t) \rho(t) L_k^\dag(t)
            - \frac{1}{2} L_k^\dag(t) L_k(t) \rho(t)
            - \frac{1}{2} \rho(t) L_k^\dag(t) L_k(t)
        \right),
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators at time $t$.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`][dynamiqs.constant],
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See
        the [Time-dependent operators](../../tutorials/time-dependent-operators.md)
        tutorial for more details.

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the initial density
        matrix `rho0` can be batched to solve multiple master equations concurrently.
        All other arguments are common to every batch. See the
        [Batching simulations](../../tutorials/batching-simulations.md) tutorial for
        more details.

    Args:
        H _(array-like or time-array of shape (nH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        rho0 _(array-like of shape (nrho0?, n, 1) or (nrho0?, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] (supported:
            [`Tsit5`][dynamiqs.solver.Tsit5], [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Euler`][dynamiqs.solver.Euler],
            [`Rouchon1`][dynamiqs.solver.Rouchon1],
            [`Rouchon2`][dynamiqs.solver.Rouchon2],
            [`Propagator`][dynamiqs.solver.Propagator]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.MEResult`][dynamiqs.MEResult] object holding the result of the Lindblad
            master  equation integration. Use the attributes `states` and `expects`
            to access saved quantities, more details in
            [`dq.MEResult`][dynamiqs.MEResult].
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    rho0 = jnp.asarray(rho0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None
    estimator = jnp.zeros(1, dtype = cdtype())
    dt0 = None

    # === estimator part
    options, Hred, Lsred, _mask, tensorisation = (
        mesolve_estimator_init(options, H, jump_ops, tsave)
    )

    # === check arguments
    _check_mesolve_args(H, jump_ops, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # === convert rho0 to density matrix
    rho0 = todm(rho0)

    if options.estimator and options.tensorisation is not None and options.reshaping:
        # a first reshaping to reduce 
        ti0 = time.time()
        (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod, ineq_set
        ) = reshaping_init(
            options, H, jump_ops, Hred, Lsred, _mask, rho0, tensorisation, tsave, 
            solver.atol
        )
        print(time.time() - ti0)
        new_tsave = tsave
        L_reshapings = [0]
        old_steps = len(tsave) 
        estimator_all = []
        rho_all = []
        while True: # do while syntax in Python
            mesolve_iteration = _vectorized_mesolve(
            H_mod, jump_ops_mod, rho_mod, new_tsave, exp_ops, solver, gradient, options
            , Hred_mod, Lsred_mod, _mask_mod, estimator, dt0
            )
            (rho_all, estimator_all, L_reshapings, estimator, new_tsave, true_time,
            dt0, options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
            tensorisation_mod) = mesolve_iteration_prepare(mesolve_iteration, old_steps, 
            tsave, L_reshapings, rho_all, estimator_all, H, jump_ops, options, 
            H_mod, jump_ops_mod, Hred_mod, Lsred_mod, _mask_mod, tensorisation_mod, 
            solver, ineq_set)
            
            if true_time[-1]==tsave[-1] and L_reshapings[-1]!=1: # do while syntax
                break
        # put the results in the usual dynamiqs format
        mesolve_result = mesolve_iteration[3].result(
        Saved(put_together_results(rho_all, 2), None, None, 
        put_together_results(estimator_all, 2, True)), None)
    else:
        # we implement the jitted vmap in another function to pre-convert QuTiP objects
        # (which are not JIT-compatible) to JAX arrays
        mesolve_result = _vectorized_mesolve(
            H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
            , Hred, Lsred, _mask, estimator, dt0
        )
    if options.estimator and not options.reshaping:
        # warn the user if the estimator's tolerance has been reached
        mesolve_result = collect_saved_estimator(mesolve_result)
        warning_estimator_tol_reached(mesolve_result, options, solver)
    return mesolve_result
    
@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_mesolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    rho0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
    Hred: TimeArray | None,
    Lsred: list[TimeArray] | None,
    _mask: Array | None,
    estimator: Array | None,
    dt0: float | None
) -> MEResult:
    # === vectorize function
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.

    # the result is vectorized over `_saved` and `infos`
    out_axes = MEResult(False, False, False, False, 0, 0)
    
    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(
            H.shape[:-2], rho0.shape[:-2], *[jump_op.shape[:-2] for jump_op in jump_ops]
        )

        def broadcast(x: TimeArray) -> TimeArray:
            return x.broadcast_to(*(broadcast_shape + x.shape[-2:]))

        H = broadcast(H)
        jump_ops = list(map(broadcast, jump_ops))
        rho0 = jnp.broadcast_to(rho0, broadcast_shape + rho0.shape[-2:])

    n_batch = (
        H.in_axes,
        [jump_op.in_axes for jump_op in jump_ops],
        Shape(rho0.shape[:-2]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape()
    )
    
    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_mesolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_mesolve, n_batch, out_axes)

    # === apply vectorized function
    return f(
            H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
            , Hred, Lsred, _mask, estimator, dt0
        )


def _mesolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    rho0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
    Hred: TimeArray | None,
    Lsred: list[TimeArray] | None,
    _mask: Array | None,
    estimator: Array | None,
    dt0: float,
) -> MEResult:
    # === select solver class
    solvers = {
        Euler: MEEuler,
        Rouchon1: MERouchon1,
        Dopri5: MEDopri5,
        Dopri8: MEDopri8,
        Tsit5: METsit5,
        Kvaerno3: MEKvaerno3,
        Kvaerno5: MEKvaerno5,
        Propagator: MEPropagator,
    }
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(
            tsave, rho0, H, exp_ops, solver, gradient, options, jump_ops
            , Hred, Lsred, _mask, estimator, dt0
        )
    
    # === run solver
    result = solver.run()

    # === return result
    return result  # noqa: RET504

def _check_mesolve_args(
    H: TimeArray, jump_ops: list[TimeArray], rho0: Array, exp_ops: Array | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)')

    # === check jump_ops
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)')

    if len(jump_ops) == 0:
        logging.warn(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check rho0 shape
    check_shape(rho0, 'rho0', '(?, n, 1)', '(?, n, n)', subs={'?': 'nrho0?'})

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'nE'})
