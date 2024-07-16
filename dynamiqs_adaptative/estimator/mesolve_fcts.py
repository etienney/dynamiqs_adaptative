import jax
import jax.numpy as jnp

from ..core._utils import _astimearray
from ..options import Options

from .utils.warnings import warning_bad_TimeArray, warning_size_too_small
from .degree_guesser import degree_guesser_nD_list
from .reshapings import mask, projection_nD, dict_nD
from .utils.utils import tensorisation_maker
from .inequalities import generate_rec_ineqs

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
        tensorisation = tensorisation_maker(lazy_tensorisation)
        if options.trunc_size is None:
            warning_bad_TimeArray(H, jump_ops)
            # Find the truncature needed to compute the estimator
            trunc_size = degree_guesser_nD_list(H0, L0, lazy_tensorisation)
            trunc_size = [2 * x.item() for x in jnp.array(trunc_size)]
            # for the "2 *" see [the article]
            tmp_dic['trunc_size'] = trunc_size
            warning_size_too_small(tensorisation, trunc_size)
        ineq_params = [(a - 1) - b for a, b in 
            zip(lazy_tensorisation, options.trunc_size)
        ] # -1 since list indexing starts at 0
        inequalities = generate_rec_ineqs(ineq_params)
        _mask = mask(H0, dict_nD(tensorisation, inequalities))
        Hred, *Lsred = [projection_nD(x, _mask) for x in [H0] + list(L0)]
        # reconvert to Timearray args
        Hred = _astimearray(Hred)
        Lsred = [_astimearray(L) for L in Lsred]
        # print(Hred, Lsred, jump_ops, type(Hred), type(Lsred), type(jump_ops))
        options = Options(**tmp_dic)
    else:
        # setup empty values
        options, Hred, Lsred, _mask, tensorisation = (
            options, None, None, None, None
        )
    return options, Hred, Lsred, _mask, tensorisation

def mesolve_iteration_prepare(mesolve_iteration, old_steps, tsave, L_reshapings, rho_all
    , estimator_all, H, jump_ops, options, H_mod, jump_ops_mod, Hred_mod, 
    Lsred_mod, _mask_mod, tensorisation_mod, solver, ineq_set):
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
            print("eeeeeee")
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
        tensorisation_mod, true_time[-2], ineq_set)
        )
        print("temps du reshaping: ", time.time() - te0)
        Lsred_mod_eval = jnp.stack([L(0) for L in Lsred_mod])
        Lsd = dag(Lsred_mod_eval)
        LdL = (Lsd @ Lsred_mod_eval).sum(0)
        tmp = (-1j * Hred_mod(0) - 0.5 * LdL) @ rho_mod + 0.5 * (Lsred_mod_eval @ rho_mod @ Lsd).sum(0)
        drho = tmp + dag(tmp)
        print("estimator calculé:", estimator_derivate_opti_nD(drho, H_mod(0), 
        jnp.stack([L(0) for L in jump_ops_mod]), rho_mod))
    elif (L_reshapings[-1]==-1
    ):
        (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod) = (
            reshapings_reduce(options, H, jump_ops, rho_mod, tensorisation_mod, 
            true_time[-2], ineq_set)
        )
    print("estimator:", estimator,"time: ", true_time)
    print("L_reshapings:", L_reshapings)
    # print("Ls", [jump_ops_mod[0](0)[i][i].item() for i in range(len(jump_ops_mod[0](0)[0]))], "\n", [Lsred_mod[0](0)[i][i].item() for i in range(len(Lsred_mod[0](0)[0]))], "\nrho", [rho_mod[i][i].item() for i in range(len(rho_mod[0]))], "\n mask", _mask_mod[0], "\ntensor", tensorisation_mod, "\nest", true_estimator)
    return (rho_all, estimator_all, L_reshapings, estimator, new_tsave, true_time,
            dt0, options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
            tensorisation_mod)

