import diffrax as dx
import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array

from ..a_posteriori.one_D.extension_1D import extension_1D
from ..a_posteriori.one_D.reduction_1D import reduction_1D
from ..a_posteriori.one_D.estimator_derivate_1D import (
    estimator_derivate_simple, estimator_derivate_opti
)
from ..a_posteriori.n_D.estimator_derivate_nD import (
    estimator_derivate_simple_nD, estimator_derivate_opti_nD
)
from ..a_posteriori.one_D.degree_guesser_1D import degree_guesser_list
from ..a_posteriori.n_D.projection_nD import projection_nD
from ..a_posteriori.n_D.tensorisation_maker import tensorisation_maker
from ..time_array import TimeArray
from ..core.abstract_solver import State
import time

from ..core.abstract_solver import MESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
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
            # run the simulation for a smaller size than the defined tensorisation..
            # instead of really reducing the operators to a smaller sized space, we will
            # project all of them on a smaller sized space (the result is the same)
            t1 = time.time()    
            y_true = y.rho
            GLs = jnp.stack([L(t) for L in self.Ls])
            rho = projection_nD(
                [y_true], None, None, self._mask
            )[0]
            t121 = time.time()
            # jax.debug.print("{a}", a= t121-t120)
            H = self.Hred(t)
            Ls = jnp.stack([L(t) for L in self.Lsred])
            t13 = time.time()
            Ls = jnp.stack(Ls)
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            t2 = time.time()
            tmp = (-1j * H - 0.5 * LdL) @ rho + 0.5 * (Ls @ rho @ Lsd).sum(0)
            drho = tmp + dag(tmp)
            t3 = time.time()
            derr = estimator_derivate_opti_nD(
                drho, self.H(t), GLs, rho
            )
            t4 = time.time()
            # jax.debug.print("{z}, {e}", z =t3-t2, e=t4-t3)
            return State(drho, derr)

        def vector_field_estimator_1D(t, y: State, _):  # noqa: ANN001, ANN202
            # run the simulation for the size n-k (k defined in the function from where
            # it is called), and add an estimator of the error made by truncating
            y_true = y.rho
            N,temp = y_true.shape
            # guessing the degree of the polynomial. if H and L are time dependant,
            # it should be executed at each t... not really efficient though     
            k = self.options.trunc_size
            Ls = jnp.stack([reduction_1D(L(t),N-k) for L in self.Ls])
            H = reduction_1D(self.H(t),N-k)
            rho = reduction_1D(y_true,N-k)
            rho_N = extension_1D(rho,N)

            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * H - 0.5 * LdL) @ rho + 0.5 * (Ls @ rho @ Lsd).sum(0)

            drho = extension_1D(tmp + dag(tmp),N)
            derr = estimator_derivate_opti(rho_N, self.Ls, self.H(t), t, N, k)
            # jax.debug.print('erreur instantanée : {res}', res=derr)
            # jax.debug.print('erreur totale : {res}', res=y.err)

            return State(drho, derr)

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)
        
        if self.options.estimator:
            if self.options.tensorisation is not None:
                return dx.ODETerm(vector_field_estimator_nD) 
            else:
                return dx.ODETerm(vector_field_estimator_1D)
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
