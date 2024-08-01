import jax
import jax.numpy as jnp
from .reshaping_y import error_reducing
from .utils.warnings import check_max_reshaping_reached


def erreur_tol_fct(estimator_rtol, solver_atol, solver_rtol, rho):
    erreur_tol = (
        estimator_rtol * (solver_atol + 
        jnp.linalg.norm(rho, ord='nuc') * solver_rtol)
    ) 
    return erreur_tol


def condition_extend(erreur, erreur_tol, max_not_reached):
    extend = jax.lax.cond(((erreur >= erreur_tol) & 
        max_not_reached), lambda: True, lambda: False
    )
    return extend


def condition_reducing(
    t_event, t1, erreur, erreur_tol, error_red, downsizing_rtol, taille_rho, steps, 
    not_extending
):  
    reduce = jax.lax.cond(
        ((erreur + error_red) <= ((t_event + 1e-16)/t1) * (erreur_tol/downsizing_rtol))
        & (taille_rho > 10) & (steps > 5) & (not_extending), 
        lambda: True, lambda: False # taille_min=100 bcs overhead too big to find useful to downsize such little matrices. num_steps_min=5 bcs the first iterations may look okay after an extension but it will rapidly goes up again
    )
    # jax.debug.print("{erreur}, {erreur_tol}, {error_red}, {downsizing_rtol}, {taille_rho}, {steps}, {reduce}", erreur=erreur, erreur_tol=erreur_tol, error_red=error_red, downsizing_rtol=downsizing_rtol, taille_rho=taille_rho, steps=steps, reduce=reduce)
    return reduce


def condition_diffrax(self):

    def condition(state, **kwargs):
        dt = state.tnext - state.tprev
        index = state.save_state[0].save_index
        dest = state.save_state[0].ys.destimator[index-1]
        est = state.save_state[0].ys.estimator[index-1][0]

        erreur_tol = erreur_tol_fct(
            self.options.estimator_rtol, self.solver.atol,
            self.solver.rtol, state.y.rho
        )
        not_max = not check_max_reshaping_reached(self.options, self.Hred)
        extend = condition_extend(dest, erreur_tol, not_max)
        error_red = error_reducing(state.y.rho, self.options, kwargs['args'][5])
        not_extending = jnp.logical_not(extend)
        reduce = condition_reducing(
            state.tprev, kwargs['t1'], 
            est, erreur_tol, error_red, self.options.downsizing_rtol, 
            len(state.y.rho[0]), state.num_steps, not_extending
        )
        # jax.debug.print("error verif. err {a} tol {b}, only dest {o}", a=dest*dt, b=erreur_tol, o= dest)
        # jax.debug.print("t0: {a} and tprev {b} and dt0 {c} ", a=kwargs['t0'], b=state.tprev, c=kwargs['dt0'])
        jax.debug.print("e:{a} r:{b}, time: {tprev}"
        , a=extend, b=reduce, tprev = state.tprev)
        return jax.lax.cond(extend | reduce, lambda: True, lambda: False)
    return condition

