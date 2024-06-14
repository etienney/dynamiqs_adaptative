from ..one_D.degree_guesser_1D import degree_guesser_list
from ..n_D.degree_guesser_nD import degree_guesser_nD_list
from ..n_D.projection_nD import projection_nD, dict_nD, mask
from ..n_D.tensorisation_maker import tensorisation_maker
from ...core._utils import _astimearray
from ...mesolve.mesolve import _vmap_mesolve
from .utils import find_approx_index
from ...time_array import ConstantTimeArray
from ...options import Options

import jax
import jax.numpy as jnp

def mesolve_estimator_init(options, H, jump_ops, tsave):
    # to init the arguments necessary for the estimator and the reshaping

    # setup empty values
    Hred, Lsred, _mask = None, None, None
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
                t0 = tsave[0]
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
                t0 = tsave[0]
                H0 = H(t0)
                L0 = jnp.stack([L(t0) for L in jump_ops])
                lazy_tensorisation = options.tensorisation
                # Find the truncature needed to compute the estimator
                trunc_size = degree_guesser_nD_list(H0, L0, lazy_tensorisation)
                # for the 2 see [the article]
                trunc_size = [2 * x for x in trunc_size]
                # tansform the trunctature into inegalities (+1 to account for the fact  
                # that matrix index start at 0)
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
    return Hred, Lsred, _mask, options

def mesolve_warning(L):
    save_b,  estimator_rtol, atol , rtol = L
    jax.debug.print(
        'WARNING : At this truncature of your simulation\'s size, '
        'it\'s not possible to warranty anymore the accuracy of '
        'your results. Try to enlarge the truncature'
    )
    jax.debug.print(
        "estimated error = {err} > {estimator_rtol} * tolerance = {tol}", 
        err = ((save_b.err[0][0]).real.astype(float)), tol = 
        estimator_rtol * (atol + jnp.linalg.norm(save_b.rho[0], ord='nuc') * rtol)
        , estimator_rtol = estimator_rtol 
    )
    return None

def mesolve_vmap_reshaping(
    H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options, Hred, Lsred, _mask
    , estimator
    ):
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
