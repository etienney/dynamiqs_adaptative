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
from ..time_array import TimeArray
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEDopri8, MEEuler, METsit5
from .mepropagator import MEPropagator

from ..a_posteriori.utils.mesolve_fcts import (
    mesolve_estimator_init,
    latest_non_inf_index
)
from ..a_posteriori.utils.utils import find_approx_index
from ..a_posteriori.n_D.inequalities import *
from ..a_posteriori.n_D.projection_nD import (
    reduction_nD, extension_nD, projection_nD, mask, dict_nD
)
from ..a_posteriori.n_D.reshapings import reshaping_init, reshaping_extend
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
    options, Hred, Lsred, _mask, inequalities, tensorisation = (
        mesolve_estimator_init(options, H, jump_ops, tsave)
    )
    L_reshapings = [] # to store the reshapings if options.reshaping

    # === check arguments
    _check_mesolve_args(H, jump_ops, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # === convert rho0 to density matrix
    rho0 = todm(rho0)

    if options.estimator and options.tensorisation is not None and options.reshaping:
        # a first reshaping to reduce 
        ti0 = time.time()
        options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho0_mod, _mask_mod, tensorisation_mod = reshaping_init(
            options, H, jump_ops, Hred, Lsred, _mask, rho0, tensorisation, tsave, solver.atol
        )
        print(time.time() - ti0)
        a = _vmap_mesolve(
            H_mod, jump_ops_mod, rho0_mod, tsave, exp_ops, solver, gradient, options
            , Hred_mod, Lsred_mod, _mask_mod, estimator, L_reshapings
        )
        old_steps = len(tsave) 
        true_time = a[1][jnp.isfinite(a[1])]
        while true_time[-1]!=tsave[-1]:
            true_time = a[1][jnp.isfinite(a[1])]
            true_steps = len(true_time)
            rho_mod =  a[2].rho[true_steps - 2] # -1 because we want rho and estimator values at true_time[-1] and len adds one
            true_estimator = a[2].err[true_steps - 2]
            print("estimator:", true_estimator,"time: ", true_time)
            print("rho: ", rho_mod)
            L_reshapings = a[-1]
            approx_index = find_approx_index(tsave, true_time[-2]) # enlever le abs ?
            new_steps = old_steps - find_approx_index(tsave, true_time[-2]) + 1 # +1 for the case under
            new_tsave = jnp.linspace(true_time[-2], tsave[-1], new_steps) # problem: it's not true (diffrax) time so the algo "clips" to the nearest value
            # because it is stored differently if save_states is on...
            # latest_index = latest_non_inf_index(a[0].estimator) # rem : c'est pas le mçeme que find_approx_index ?
            # t = tsave[approx_index] # -1 since we are redoing the problem
            # rho = a[0].states[latest_index]
            # estimator = a[0].estimator[latest_index-2]
            print(a[0].estimator)
            if L_reshapings[-1]==1 and not jnp.isfinite(a[0].estimator[-1]): # isfinite to check if we aren't on the last reshaping
                te0 = time.time()
                options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, tensorisation_mod = (
                reshaping_extend(options, H, jump_ops, rho_mod,
                    tensorisation_mod, true_time)
                )
                print("temps du reshaping: ", time.time() - te0)
            a = _vmap_mesolve(
                H_mod, jump_ops_mod, rho_mod, new_tsave
                , exp_ops, solver, gradient, options
                , Hred_mod, Lsred_mod, _mask_mod, true_estimator, L_reshapings
            )
        return a[0]
    else:
        # we implement the jitted vmap in another function to pre-convert QuTiP objects
        # (which are not JIT-compatible) to JAX arrays
        return _vmap_mesolve(
                H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
                , Hred, Lsred, _mask, estimator, L_reshapings
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
    estimator: Array | None,
    L_reshapings: list | None,
) -> MEResult:
    # === vectorize function
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
    is_batched = (
        is_timearray_batched(H),
        [is_timearray_batched(jump_op) for jump_op in jump_ops],
        rho0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
        is_timearray_batched(Hred) if Hred is not None else False,
        [is_timearray_batched(L) for L in Lsred] if Lsred is not None else False,
        False,
        False, # estimateur = False ?
        False,
    )

    # the result is vectorized over `_saved` and `infos`
    out_axes = MEResult(None, None, None, None, 0, 0)

    # compute vectorized function with given batching strategy
    f = compute_vmap(_mesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(
            H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
            , Hred, Lsred, _mask, estimator, L_reshapings
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
    L_reshapings: list | None,
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
    solver = solver_class(
            tsave, rho0, H, exp_ops, solver, gradient, options, jump_ops
            , Hred, Lsred, _mask, estimator, L_reshapings
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
