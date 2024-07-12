# ruff: noqa: ARG001

from __future__ import annotations

from jaxtyping import ArrayLike

from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Solver
from ..time_array import TimeArray

__all__ = ['smesolve']


def smesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    etas: ArrayLike,
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    tmeas: ArrayLike | None = None,
    ntrajs: int = 10,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> Result:
    r"""Solve the diffusive stochastic master equation (SME).

    Warning:
        This function has not been ported to JAX yet. The following documentation is
        a draft API, copied from the old PyTorch version of the library.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho(t=0)$, according to the diffusive SME in Itô
    form ($\hbar=1$)
    $$
        \begin{split}
            \dd\rho(t) =&~ -i[H(t), \rho(t)] \dt \\\\
            &+ \sum_{k=1}^N \left(
                L_k(t) \rho(t) L_k(t)^\dag
                - \frac{1}{2} L_k(t)^\dag L_k(t) \rho(t)
                - \frac{1}{2} \rho(t) L_k(t)^\dag L_k(t)
            \right)\dt \\\\
            &+ \sum_{k=1}^N \sqrt{\eta_k} \left(
                L_k(t) \rho(t)
                + \rho(t) L_k(t)^\dag
                - \tr{(L_k(t)+L_k(t)^\dag)\rho(t)}\rho(t)\ \dd W_k(t)
            \right),
        \end{split}
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$, $\{L_k(t)\}$ is a collection
    of jump operators at time $t$, each continuously measured with efficiency
    $0\leq\eta_k\leq1$ ($\eta_k=0$ for purely dissipative loss channels) and
    $\dd W_k(t)$ are independent Wiener processes.

    Note-: Diffusive vs. jump SME
        In quantum optics the _diffusive_ SME corresponds to homodyne or heterodyne
        detection schemes, as opposed to the _jump_ SME which corresponds to photon
        counting schemes. No solver for the jump SME is provided yet, if this is needed
        don't hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    The measured signals $I_k(t)=\dd y_k(t)/\dt$ verifies:
    $$
        \dd y_k(t) =\sqrt{\eta_k} \tr{(L_k(t) + L_k(t)^\dag) \rho(t)} \dt + \dd W_k(t).
    $$

    Note-: Signal normalisation
        Sometimes the signals are defined with a different but equivalent normalisation
        $\dd y_k'(t) = \dd y_k(t)/(2\sqrt{\eta_k})$.

    The signals $I_k(t)$ are singular quantities, the solver returns the averaged signals
    $J_k(t)$ defined for a time interval $[t_0, t_1)$ by:
    $$
        J_k([t_0, t_1)) = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} I_k(t) \dt
        = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} \dd y_k(t).
    $$
    The time intervals for integration are defined by the argument `tmeas`, which
    defines `len(tmeas) - 1` intervals. By default, `tmeas = tsave`, so the signals
    are averaged between the times at which the states are saved.

    Note-: Defining a time-dependent Hamiltonian or jump operator
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`][dynamiqs.constant],
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the initial density
        matrix `rho0` can be batched to solve multiple SMEs concurrently. All other
        arguments are common to every batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for
        more details.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        etas _(array-like of shape (nL,))_: Measurement efficiencies, must be of the
            same length as `jump_ops` with values between 0 and 1. For a purely
            dissipative loss channel, set the corresponding efficiency to 0. No
            measurement signal will be returned for such channels.
        rho0 _(array-like of shape (brho?, n, 1) or (brho?, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        tmeas _(array-like of shape (ntmeas,), optional)_: Times between which
            measurement signals are averaged and saved. Defaults to `tsave`.
        ntrajs: Number of stochastic trajectories to solve concurrently.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].
    """  # noqa: E501
    return NotImplementedError
