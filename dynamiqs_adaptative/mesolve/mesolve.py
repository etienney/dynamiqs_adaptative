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
    compute_vmap,
    get_solver_class,
    is_timearray_batched,
)
from ..gradient import Gradient
from ..options import Options
from ..result import MEResult
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import TimeArray, ConstantTimeArray
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEDopri8, MEEuler, METsit5
from .mepropagator import MEPropagator

from ..a_posteriori.one_D.degree_guesser_1D import degree_guesser_list
from ..a_posteriori.n_D.degree_guesser_nD import degree_guesser_nD_list
from ..a_posteriori.n_D.projection_nD import projection_nD, dict_nD, mask
from ..a_posteriori.n_D.tensorisation_maker import tensorisation_maker
from ..a_posteriori.utils.utils import find_approx_index
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

    # === estimator part
    t0 = tsave[0]
    if options.estimator:
        if options.trunc_size is None:
            if (
                type(H)!=ConstantTimeArray or 
                any([type(jump_ops[i])!=ConstantTimeArray 
                for i in range(len(jump_ops))])
            ):
                jax.debug.print(
                    'WARNING : If your array is not time dependant, beware that '
                    'the truncature required to compute the estimator won\'t be '
                    'trustworthy. See [link to the article] for more details. '
                )
            if options.tensorisation is None:
                # Find the truncature needed to compute the estimator
                trunc_size = degree_guesser_list(
                    H(t0), jnp.stack([L(t0) for L in jump_ops])
                )
                # for the 2 see [the article]
                trunc_size = 2 * trunc_size
                tmp_dic=options.__dict__
                tmp_dic['trunc_size']=int(trunc_size)
                options=Options(**tmp_dic) 
            else:
                t0 = time.time()
                H0 = H(t0)
                L0 = jnp.stack([L(t0) for L in jump_ops])
                lazy_tensorisation = options.tensorisation
                # Find the truncature needed to compute the estimator
                trunc_size = degree_guesser_nD_list(H0, L0, lazy_tensorisation)
                # for the 2 see [the article]
                trunc_size = [2 * x for x in trunc_size]
                # tansform the trunctature into inegalities
                inequalities = [
                lambda *args, idx=idx, lt=lazy_tensorisation: 
                args[idx] <= lt[idx] - (trunc_size[idx]+1)
                for idx in range(len(lazy_tensorisation))
                ]
                tensorisation = tensorisation_maker(lazy_tensorisation)
                _mask = mask(H0, dict_nD(tensorisation, inequalities))
                Hred, *Lsred = projection_nD(
                   [H0] + list(L0), tensorisation, inequalities, _mask
                )
                # We setup the results in options
                tmp_dic=options.__dict__
                tmp_dic['trunc_size'] = [x.item() for x in jnp.array(trunc_size)]
                options=Options(**tmp_dic) 
                # reconvert to Timearray args
                Hred = _astimearray(Hred)
                Lsred = [_astimearray(L) for L in Lsred]
                t1 = time.time()
                # print(t1-t0)

    # === check arguments
    _check_mesolve_args(H, jump_ops, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # === convert rho0 to density matrix
    rho0 = todm(rho0)

    # we implement the jitted vmap in another function to pre-convert QuTiP objects
    # (which are not JIT-compatible) to JAX arrays
    if options.estimator and options.tensorisation and not options.reshaping:
        a = _vmap_mesolve(
                H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
                , Hred, Lsred, _mask, estimator 
            )
        return a
    elif options.estimator and options.tensorisation and options.reshaping:
        a = _vmap_mesolve(
                H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
                , Hred, Lsred, _mask, estimator
            )
        while a[1][0]!=tsave[-1]:
            if options.save_states:
                estimator = a[0].estimator[-1]
            else:
                estimator = a[0].estimator
            steps = len(tsave) - find_approx_index(tsave, a[1]) + 1 # +1 for the case under
            new_tsave = jnp.linspace(a[1][0], tsave[-1], steps) # problem: it's not true time so the algo "clips" to the nearest value
            # print(tsave, new_tsave)
            a = _vmap_mesolve(
                H, jump_ops, rho0, new_tsave
                , exp_ops, solver, gradient, options
                , Hred, Lsred, _mask, estimator
            )
        return a[0]
    else:
        return _vmap_mesolve(
                H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
                , None, None, None, None
            )
    

@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vmap_mesolve(
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
    estimator: Array | None
) -> MEResult:
    # === vectorize function
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
    if options.estimator and options.tensorisation:
        is_batched = (
            is_timearray_batched(H),
            [is_timearray_batched(jump_op) for jump_op in jump_ops],
            rho0.ndim > 2,
            False,
            False,
            False,
            False,
            False,
            is_timearray_batched(Hred),
            [is_timearray_batched(L) for L in Lsred],
            False,
            False, # estimateur = False ?
        )
    else:
        is_batched = (
            is_timearray_batched(H),
            [is_timearray_batched(jump_op) for jump_op in jump_ops],
            rho0.ndim > 2,
            False,
            False,
            False,
            False,
            False,
        )
    # the result is vectorized over `_saved` and `infos`
    out_axes = MEResult(None, None, None, None, 0, 0)

    # compute vectorized function with given batching strategy
    f = compute_vmap(_mesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    if options.estimator and options.tensorisation:
        return f(
            H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
            , Hred, Lsred, _mask, estimator
        )
    else:
        return f(
            H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
            , None, None, None, None
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
) -> MEResult:
    # === select solver class
    solvers = {
        Euler: MEEuler,
        Dopri5: MEDopri5,
        Dopri8: MEDopri8,
        Tsit5: METsit5,
        Propagator: MEPropagator,
    }
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    if options.estimator and options.tensorisation:
        solver = solver_class(
            tsave, rho0, H, exp_ops, solver, gradient, options, jump_ops
            , Hred, Lsred, _mask, estimator
        )
    else:
        solver = solver_class(
            tsave, rho0, H, exp_ops, solver, gradient, options, jump_ops
            , None, None, None, None
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
