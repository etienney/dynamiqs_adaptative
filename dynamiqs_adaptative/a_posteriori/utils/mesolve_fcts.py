from ...a_posteriori.n_D.reshapings import (
    reshaping_extend, check_max_reshaping_reached, error_reducing, reshapings_reduce
)
from ..one_D.degree_guesser_1D import degree_guesser_list
from ..n_D.degree_guesser_nD import degree_guesser_nD_list
from ..n_D.projection_nD import projection_nD, dict_nD, mask
from ..n_D.tensorisation_maker import tensorisation_maker
from ..n_D.inequalities import generate_rec_ineqs
from ...core._utils import _astimearray
# from ...mesolve.mesolve import _vmap_mesolve
from .utils import find_approx_index, prod
from ...a_posteriori.n_D.estimator_derivate_nD import estimator_derivate_opti_nD

from ...utils.utils import dag
from ...time_array import ConstantTimeArray
import diffrax as dx
import time as time
from ...options import Options

import jax
import jax.numpy as jnp
import math

def mesolve_estimator_init(options, H, jump_ops, tsave):
    # to init the arguments necessary for the estimator and the reshaping

    if options.estimator:
        t0 = tsave[0]
        H0 = H(t0)
        L0 = jnp.stack([L(t0) for L in jump_ops])
        lazy_tensorisation = options.tensorisation
        tmp_dic=options.__dict__
        if lazy_tensorisation is None:
            lazy_tensorisation = [len(H0[0])]
            tmp_dic['tensorisation']=lazy_tensorisation
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
                t0 = tsave[0]
                # Find the truncature needed to compute the estimator
                trunc_size = degree_guesser_list(H0, L0)
                # for the 2 see [the article]
                trunc_size = 2 * trunc_size
                tmp_dic['trunc_size']=int(trunc_size)
            else:
                # Find the truncature needed to compute the estimator
                trunc_size = degree_guesser_nD_list(H0, L0, lazy_tensorisation)
                # for the 2 see [the article]
                # We setup the results in options
                tmp_dic['trunc_size'] = [2 * x.item() for x in jnp.array(trunc_size)]
        inequalities = generate_rec_ineqs(
            [(a - 1) - b for a, b in zip(lazy_tensorisation, options.trunc_size)]
        )# -1 since list indexing strats at 0
        tmp_dic['inequalities'] = [
        [None, (a - 1) - b] for a, b in zip(lazy_tensorisation, options.trunc_size)
        ]
        tensorisation = tensorisation_maker(lazy_tensorisation)
        _mask = mask(H0, dict_nD(tensorisation, inequalities, options))
        Hred, *Lsred = projection_nD(
            [H0] + list(L0), tensorisation, inequalities, options, _mask
        )
        options = Options(**tmp_dic)
        # reconvert to Timearray args
        Hred = _astimearray(Hred)
        Lsred = [_astimearray(L) for L in Lsred]
    else:
        # setup empty values
        options, Hred, Lsred, _mask, inequalities, tensorisation = options, None, None, 0, 0, 0
    return options, Hred, Lsred, _mask, inequalities, tensorisation

def mesolve_warning(solution, options, solver):
    estimator_final = solution.estimator[-1][0]
    rho_final = solution.states[-1]
    if (estimator_final > options.estimator_rtol * (solver.atol + 
        jnp.linalg.norm(rho_final, ord='nuc') * solver.rtol)
    ):
        jax.debug.print(
            'WARNING : At this truncature of your simulation\'s size, '
            'it\'s not possible to warranty anymore the accuracy of '
            'your results. Try to enlarge the truncature'
        )
        jax.debug.print(
            "estimated error = {err} > {estimator_rtol} * tolerance = {tol}", 
            err = ((estimator_final).real.astype(float)), 
            estimator_rtol = options.estimator_rtol,
            tol = options.estimator_rtol * 
            (solver.atol + jnp.linalg.norm(rho_final, ord='nuc') * solver.rtol)     
        )
    return None

def mesolve_iteration_prepare(mesolve_iteration, old_steps, tsave, L_reshapings, rho_all
    , estimator_all, H, jump_ops, options, H_mod, jump_ops_mod, Hred_mod, 
    Lsred_mod, _mask_mod, tensorisation_mod, solver):
    true_time = mesolve_iteration[1][jnp.isfinite(mesolve_iteration[1])]
    true_steps = len(true_time)
    last_state_index = max(0,true_steps - 2)
    dt0 = true_time[-1] - true_time[-2]
    print("dt0:", dt0)
    new_steps = old_steps - find_approx_index(tsave, true_time[last_state_index]) + 1
    new_tsave = jnp.linspace(true_time[last_state_index], tsave[-1], new_steps) 
    print(options.tensorisation)
    print(tensorisation_mod[-1])
    rho_mod =  mesolve_iteration[2].rho[last_state_index]
    estimator = mesolve_iteration[2].err[last_state_index]
    # useful to recompute the error to see if it was an extend or a reduce
    rho_erreur = mesolve_iteration[2].rho[last_state_index + 1]
    estimator_erreur = mesolve_iteration[2].err[last_state_index + 1]
    rho_all.append(mesolve_iteration[2].rho[:true_steps])
    estimator_all.append(mesolve_iteration[2].err[:true_steps])
    if check_max_reshaping_reached(options, H_mod):
        L_reshapings.append(2)
        print("""WARNING: your space wasn't large enough to capture the dynamic up to
              the tolerance. Give a larger max space[link to what it means] or try to 
              see if your dynamic isn't exploding""")
    elif dx.RESULTS.discrete_terminating_event_occurred==mesolve_iteration[-1]:
        erreur_tol = (true_time[-1] * 
            options.estimator_rtol * (solver.atol + 
            jnp.linalg.norm(rho_erreur, ord='nuc') * solver.rtol)
        )
        if (estimator_erreur).real and not check_max_reshaping_reached(options, H_mod) >= erreur_tol:
            L_reshapings.append(1)
        if ((estimator_erreur).real + error_reducing(rho_erreur, options) <= 
            erreur_tol/options.downsizing_rtol
            and len(rho_erreur) > 100): # 100 bcs overhead too big to find useful to downsize such little matrices :
            L_reshapings.append(-1)
        print("error seuil en dehors", erreur_tol, estimator_erreur)
    else:
        L_reshapings.append(0)
    # print("estimator all qui charge:", estimator_all)
    if (L_reshapings[-1]==1
    ):# and not jnp.isfinite(a[0].estimator[-1]): # isfinite to check if we aren't on the last reshaping
        te0 = time.time()
        (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod) = (reshaping_extend(options, H, jump_ops, rho_mod,
        tensorisation_mod, true_time)
        )
        print("temps du reshaping: ", time.time() - te0)
        Lsred_mod_eval = jnp.stack([L(0) for L in Lsred_mod])
        Lsd = dag(Lsred_mod_eval)
        LdL = (Lsd @ Lsred_mod_eval).sum(0)
        tmp = (-1j * Hred_mod(0) - 0.5 * LdL) @ rho_mod + 0.5 * (Lsred_mod_eval @ rho_mod @ Lsd).sum(0)
        drho = tmp + dag(tmp)
        print("estimator calculé:", estimator_derivate_opti_nD(drho, H_mod(0), 
        jnp.stack([L(0) for L in jump_ops_mod]), rho_mod))
    # elif (L_reshapings[-1]==-1
    # ):
    #     (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
    #     tensorisation_mod) = (
    #         reshapings_reduce(options, H, jump_ops, rho_mod, tensorisation_mod, tsave)
    #     )
    print("estimator:", estimator,"time: ", true_time)
    print("L_reshapings:", L_reshapings)
    # print("Ls", [jump_ops_mod[0](0)[i][i].item() for i in range(len(jump_ops_mod[0](0)[0]))], "\n", [Lsred_mod[0](0)[i][i].item() for i in range(len(Lsred_mod[0](0)[0]))], "\nrho", [rho_mod[i][i].item() for i in range(len(rho_mod[0]))], "\n mask", _mask_mod[0], "\ntensor", tensorisation_mod, "\nest", true_estimator)
    return (rho_all, estimator_all, L_reshapings, estimator, new_tsave, true_time,
            dt0, options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
            tensorisation_mod)

def latest_non_inf_index(lst):
    # find the latest elements in a list that is not inf
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != math.inf:
            return i
    return None  # If all elements are `inf`, return None

# problem of cycling imports... not understood
# def mesolve_vmap_reshaping(
#     H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options, Hred, Lsred, _mask
#     , estimator
#     ):
#     a = _vmap_mesolve(
#         H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
#         , Hred, Lsred, _mask, estimator
#     )
#     while a[1][0]!=tsave[-1]:
#         if options.save_states:
#             estimator = a[0].estimator[-1]
#         else:
#             estimator = a[0].estimator
#         steps = len(tsave) - find_approx_index(tsave, a[1]) + 1 # +1 for the case under
#         new_tsave = jnp.linspace(a[1][0], tsave[-1], steps) # problem: it's not true time so the algo "clips" to the nearest value
#         # print(tsave, new_tsave)
#         a = _vmap_mesolve(
#             H, jump_ops, rho0, new_tsave
#             , exp_ops, solver, gradient, options
#             , Hred, Lsred, _mask, estimator
#         )
#     return a[0]
