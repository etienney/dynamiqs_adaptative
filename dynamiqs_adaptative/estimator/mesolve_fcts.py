import jax
import jax.numpy as jnp
import diffrax as dx
import time

from ..core._utils import _astimearray
from ..options import Options
from .._utils import cdtype


from .utils.warnings import (
    warning_bad_TimeArray,
    warning_size_too_small,
    check_in_max_truncature,
    check_max_reshaping_reached,
    check_not_under_truncature
)
from .degree_guesser import degree_guesser_nD_list
from .reshapings import mask, projection_nD, dict_nD, dict_nD_reshapings
from .utils.utils import tensorisation_maker, find_approx_index
from .inequalities import generate_rec_ineqs
from .reshaping_y import (
    error_reducing,
    reshaping_extend,
    reshapings_reduce,
    reshaping_init
)
from .estimator import compute_estimator
from ..utils.utils import dag


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

def mesolve_iteration_prepare(
    rho_all, estimator_all, time_all, inequalities_all,
    L_reshapings, tsave, old_steps, options,
    mesolve_iteration, solver, ineq_set,
    H, jump_ops, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, _mask_mod, tensorisation_mod
):
    rextend_args = None
    print("infos sur la run", mesolve_iteration.infos)
    true_time = mesolve_iteration._saved.time
    true_steps = len(true_time) - 1
    dt0 = true_time[-1] - true_time[-2]
    print("dt0:", dt0)
    new_steps = old_steps - find_approx_index(tsave, true_time[-2]) + 1
    new_tsave = jnp.linspace(true_time[-2], tsave[-1], new_steps) 
    print("tensorisation max", options.tensorisation)
    print("largest tens reached", tensorisation_mod[-1])
    rho_mod =  mesolve_iteration.states[-2]
    estimator = jnp.zeros(1, cdtype())
    estimator = estimator.at[0].set(mesolve_iteration.estimator[-2])
    print("we restart with this base value", estimator)
    # useful to recompute the error to see if it was an extend or a reduce
    rho_erreur = mesolve_iteration.states[-1]
    estimator_erreur = mesolve_iteration.estimator[-1]
    rho_all.append(mesolve_iteration.states[:-1])
    estimator_all.append(mesolve_iteration.estimator[:-1])
    time_all.append(mesolve_iteration.time[:-1])
    inequalities_all.append(mesolve_iteration.inequalities[:-1])
    if check_max_reshaping_reached(options, H_mod):
        L_reshapings.append(2)
        print("""WARNING: your space wasn't large enough to capture the dynamic up to
              the tolerance. Give a larger max space[link to what it means] or try to 
              see if your dynamic isn't exploding""")
    erreur_tol = (true_time[-1] * 
        options.estimator_rtol * (solver.atol + 
        jnp.linalg.norm(rho_erreur, ord='nuc') * solver.rtol)
    )
    if (
        (estimator_erreur).real and not 
        check_max_reshaping_reached(options, H_mod) >= erreur_tol
    ):
        L_reshapings.append(1)
    elif (
        (estimator_erreur).real + error_reducing(rho_erreur, options, ineq_set) <= 
        erreur_tol/options.downsizing_rtol
        and len(rho_erreur) > 100 and len(true_time) > 4 
    ): # 100 bcs overhead too big to find useful to downsize such little matrices. 4 bcs the first iterations may look okay after an extension but it will rapidly goes up again.
        print("reducing set")
        L_reshapings.append(-1)
    te0 = time.time()
    if (L_reshapings[-1]==1
    ):
        (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod, rextend_args) = (
            reshaping_extend(options, H, jump_ops, rho_mod,
            tensorisation_mod, true_time[-2], ineq_set)
        )
        print("temps du reshaping: ", time.time() - te0)
        print("estimator calculé:", compute_estimator(
            H_mod, jump_ops_mod,
            Hred_mod, Lsred_mod,
            rho_mod, true_time[-2])
        )
    elif (L_reshapings[-1]==-1):
        (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod, rextend_args) = (
            reshapings_reduce(options, H, jump_ops, rho_mod, tensorisation_mod, 
            true_time[-2], ineq_set)
        )
        print("temps du reshaping: ", time.time() - te0)
    print("estimator:", estimator,"time: ", true_time)
    print("L_reshapings:", L_reshapings)
    # print("Ls", [jump_ops_mod[0](0)[i][i].item() for i in range(len(jump_ops_mod[0](0)[0]))], "\n", [Lsred_mod[0](0)[i][i].item() for i in range(len(Lsred_mod[0](0)[0]))], "\nrho", [rho_mod[i][i].item() for i in range(len(rho_mod[0]))], "\n mask", _mask_mod[0], "\ntensor", tensorisation_mod, "\nest", true_estimator)
    return (
        rho_all, estimator_all, time_all, inequalities_all,
        L_reshapings, new_tsave, estimator, true_time, dt0, options, 
        H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation_mod, rextend_args
    )

