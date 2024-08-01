from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array

from ..core.abstract_solver import State
from ..estimator.reshapings import projection_nD
import time


import diffrax as dx
import jax.numpy as jnp

from ..core.abstract_solver import MESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Kvaerno3Solver,
    Kvaerno5Solver,
    Tsit5Solver,
)
from ..utils.utils import dag



class MEDiffraxSolver(DiffraxSolver, MESolver):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.


        def vector_field_estimator_nD(t, y: State, _):
            # run the simulation for a smaller size than the defined tensorisation.
            # instead of really reducing the operators to a smaller sized space, we will
            # project all of them on a smaller sized space (the result is the same)

            t0 = time.time()    
            y_true = jnp.array(y.rho)
            rho = projection_nD(y_true, self._mask)
            t1 = time.time()
            # jax.debug.print("{a}", a= t121-t120)
            derr = 0
            err = 0
            t = 0
            Hred = self.Hred(t)
            Lsred = jnp.stack([L(t) for L in self.Lsred])
            t2 = time.time()
            Lsd = dag(Lsred)
            LdL = (Lsd @ Lsred).sum(0)
            tmp = (-1j * Hred - 0.5 * LdL) @ rho + 0.5 * (Lsred @ rho @ Lsd).sum(0)
            drho = tmp + dag(tmp)
            t3 = time.time()
            
            # jax.debug.print("{a} et {b}", a= t1-t0, b = t3-t1)
            return State(drho, derr, err, t)

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)
        
        if self.options.estimator:
            return dx.ODETerm(vector_field_estimator_nD) 
        else:
            return dx.ODETerm(vector_field)

class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass


class MEDopri8(MEDiffraxSolver, Dopri8Solver):
    pass


class METsit5(MEDiffraxSolver, Tsit5Solver):
    pass


class MEKvaerno3(MEDiffraxSolver, Kvaerno3Solver):
    pass


class MEKvaerno5(MEDiffraxSolver, Kvaerno5Solver):
    pass
