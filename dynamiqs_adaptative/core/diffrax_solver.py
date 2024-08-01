from __future__ import annotations

import warnings
import jax
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
from jax import Array
from jaxtyping import PyTree
import jax.numpy as jnp


from ..gradient import Autograd, CheckpointAutograd
from .abstract_solver import BaseSolver
from ..options import Options

from .abstract_solver import State
from ..estimator.saves import save_estimator
from .._utils import cdtype
from ..estimator.condition_reshapings import condition_diffrax


class DiffraxSolver(BaseSolver):
    # Subclasses should implement:
    # - the attributes: stepsize_controller, dt0, max_steps, diffrax_solver, terms
    # - the methods: result, infos

    stepsize_controller: dx.AbstractVar[dx.AbstractStepSizeController]
    dt0: dx.AbstractVar[float | None]
    max_steps: dx.AbstractVar[int]
    diffrax_solver: dx.AbstractVar[dx.AbstractSolver]
    terms: dx.AbstractVar[dx.AbstractTerm]
    options: Options

    def __init__(self, *args):
        # pass all init arguments to `BaseSolver`
        super().__init__(*args)

    def run(self) -> PyTree:
        # TODO: remove once complex support is stabilized in diffrax
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            # === prepare diffrax arguments
            if self.options.estimator:
                subsaveat_a = dx.SubSaveAt(t0 =True, steps=True, fn=save_estimator)
                saveat = dx.SaveAt(subs=[subsaveat_a], controller_state = True)
            else:
                fn = lambda t, y, args: self.save(y)  # noqa: ARG005
                subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
                subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
                saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            # stop the diffrax integration if condition is reached (we will then restart
            # a diffrax integration with a reshaping of H, L, rho)
            event = dx.DiscreteTerminatingEvent(cond_fn=condition_diffrax(self))
            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.ts[-1],
                dt0=self.dt0,
                y0=State(
                    self.y0, # the solution at current time
                    jnp.zeros(1, dtype = cdtype()),
                    self.estimator, # the estimator at current time
                    self.t0.astype(cdtype()), # the time
                ) if self.options.estimator else self.y0,
                discrete_terminating_event=event if self.options.reshaping else None,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                progress_meter= (
                    self.options.progress_meter.to_diffrax() 
                    if not self.options.reshaping else dx.NoProgressMeter()
                ), 
                args = [
                    self.H, self.Ls, self.Hred, self.Lsred, self._mask, self.ineq_set
                ] 
            )

        # === collect and return results
        if self.options.estimator and not self.options.reshaping:
            # jax.debug.print("ee{e}", e = self.diffrax_solver.interpolation_cls)
            saved = solution.ys[0]
            return self.result(saved, infos=self.infos(solution.stats))
        elif not self.options.reshaping:
            save_a, save_b = solution.ys
            saved = self.collect_saved(save_a, save_b[0])
            return self.result(saved, infos=self.infos(solution.stats))
        else:
            saved = solution.ys[0]
            # give additional infos needed for the reshaping
            # jax.debug.print("fin saved: {res}", res = solution.stats)
            return self.result(saved, infos=self.infos(solution.stats))
            # return [self.result(saved, infos=self.infos(solution.stats)), 
            #     solution.ts[-1], save_c, self, solution.result,
            # ]

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedSolver(DiffraxSolver):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    stepsize_controller: dx.AbstractStepSizeController = dx.ConstantStepSize()
    max_steps: int = 150  # TODO: fix hard-coded max_steps

    @property
    def dt0(self) -> float:
        return self.solver.dt

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])


class EulerSolver(FixedSolver):
    diffrax_solver: dx.AbstractSolver = dx.Euler()


class AdaptiveSolver(DiffraxSolver):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean()} steps ({self.naccepted.mean()}'
                    f' accepted, {self.nrejected.mean()} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )


    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
            jump_ts=self.discontinuity_ts,
        )

    @property
    def max_steps(self) -> int:
        return self.solver.max_steps

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )


class Dopri5Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri5()


class Dopri8Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri8()


class Tsit5Solver(AdaptiveSolver):
    diffrax_solver = dx.Tsit5()


class Kvaerno3Solver(AdaptiveSolver):
    diffrax_solver = dx.Kvaerno3()


class Kvaerno5Solver(AdaptiveSolver):
    diffrax_solver = dx.Kvaerno5()