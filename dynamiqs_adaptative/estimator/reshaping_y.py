from .reshapings import (
    projection_nD, dict_nD, mask, reduction_nD, extension_nD, red_ext_full, 
    red_ext_zeros, dict_nD_reshapings
)
from .inequalities import generate_rec_ineqs, ineq_from_params
from ..core._utils import _astimearray
from ..options import Options

import jax.numpy as jnp
from .inequalities import generate_rec_ineqs, generate_rec_func, update_ineq
from .utils.utils import (
    prod, 
    ineq_to_tensorisation, 
    tensorisation_maker,
    find_reextension_params
)
import itertools


def reshaping_init(
         options, H, jump_ops, Hred, Lsred, _mask, rho0, tensorisation, tsave, atol
    ):
    # On commence en diminuant beaucoup la taille par rapport à la saisie utiliateur
    # qui correspond à la taille maximal atteignable

    # We first check if it does not create too much of an approximation to modify the
    # matrix size this way
    """WARNING we do it by checking the diagonal values, which is not theoreticaly
    justified. it's just a guess.
    Also we just do an absolute tolerance, relative tolerance would need to compute
    a norm which is expensive."""
    if options.inequalities is None:
        ineq_params =  [(a-1)//2 for a, b in 
            zip(options.tensorisation, options.trunc_size)
        ]
        up = options.trunc_size
        down = options.trunc_size
        ineq_set = [generate_rec_func(j) for j in range(len(ineq_params))]
        inequalities = ineq_from_params(ineq_set, ineq_params)# /!\ +1 since lazy_tensorisation starts counting at 0
    else:
        len_ineq = len(options.inequalities)
        ineq_params = [options.inequalities[i][1] for i in  
            range(len_ineq)
        ]
        ineq_set = [options.inequalities[i][0] for i in  
            range(len_ineq)
        ]
        up = [options.inequalities[i][2] for i in  
            range(len_ineq)
        ]
        down = [options.inequalities[i][3] for i in  
            range(len_ineq)
        ]
        inequalities = ineq_from_params(ineq_set, ineq_params)
    if (
        trace_ineq_states(tensorisation, inequalities, 
        rho0) > 1/len(tsave) * (atol)
    ):# verification that we don't lose too much info on rho0 by doing this
        raise ValueError("""Your initial state is already populated in the high Fock
            states dimensions. This technique won't work.
            """)
        return H, jump_ops, H, jump_ops, rho0, None, tensorisation, options
    else:
        tmp_dic=options.__dict__
        options_ineq_0 = [None] * (len(options.tensorisation)) # JAX cannot stand lambda functions
            # when called with options so it is necessary
        tmp_dic['inequalities'] = [[a, b, c , d] for a, b, c, d in 
            zip(options_ineq_0, ineq_params, up, down)
        ]
        options=Options(**tmp_dic) 

        temp = red_ext_full(
            [H(tsave[0])] + [rho0] + [L(tsave[0]) for L in jump_ops],
            tensorisation, inequalities, options
        )
        H_mod, rho0_mod, *jump_ops_mod = temp[0]
        tensorisation_mod = temp[1]
        _mask_mod = mask(
            H_mod, 
            jnp.array(dict_nD_reshapings(tensorisation_mod, inequalities, options, 'proj'))
        )
        Hred_mod, rho0_mod, *Lsred_mod = [
            projection_nD(x, _mask_mod) for x in [H_mod] + [rho0_mod] + 
            [L for L in jump_ops_mod]
        ]
        rextend_args = find_reextension_params(tensorisation_mod, options.tensorisation)

        H_mod = _astimearray(H_mod)
        jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
        Hred_mod = _astimearray(Hred_mod)
        Lsred_mod = [_astimearray(L) for L in Lsred_mod]

        return (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho0_mod, _mask_mod, 
            tensorisation_mod, ineq_set, rextend_args
        )
    
def reshaping_extend(
        options, H, Ls, rho, tensorisation, t, ineq_set
    ):
    H = H(t)
    Ls = jnp.stack([L(t) for L in Ls])
    old_inequalities = ineq_from_params(ineq_set, [options.inequalities[i][1] for i in 
        range(len(options.inequalities))]
    )
    options = update_ineq(options, direction='up')
    inequalities = ineq_from_params(ineq_set, [options.inequalities[i][1] for i in 
        range(len(options.inequalities))]
    )
    temp = extension_nD(
        [rho], options, inequalities, tensorisation
    )
    rho_mod = jnp.array(temp[0])[0]
    tensorisation = temp[1]
    _mask_mod = mask(
        rho_mod, 
        jnp.array(dict_nD_reshapings(tensorisation, inequalities, options, 'proj'))
    )
    temp = red_ext_full([H] + [L for L in Ls], 
        tensorisation_maker(options.tensorisation), inequalities, options
    )
    H_mod, *Ls_mod = temp[0]
    tensorisation = temp[1]
    Hred_mod, *Lsred_mod =  [
        projection_nD(x, _mask_mod) for x in [H_mod] + [L for L in Ls_mod]
    ]
    rextend_args = find_reextension_params(tensorisation, options.tensorisation)
    
    H_mod = _astimearray(H_mod)
    Ls_mod = [_astimearray(L) for L in Ls_mod]
    Hred_mod = _astimearray(Hred_mod)
    Lsred_mod = [_astimearray(L) for L in Lsred_mod]
    return (
        options, H_mod, Ls_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation, rextend_args
    )

def reshapings_reduce(options, H, jump_ops, rho_mod, tensorisation, t, ineq_set
    ):
    options = update_ineq(options, direction='down')
    inequalities = ineq_from_params(ineq_set, [options.inequalities[i][1] for i in 
        range(len(options.inequalities))]
    )
    temp = red_ext_full(
        [H(t)] + [L(t) for L in jump_ops],
        tensorisation_maker(options.tensorisation), inequalities, options
    )
    H_mod, *jump_ops_mod = temp[0]
    rho_mod = red_ext_zeros(
        [rho_mod], tensorisation, inequalities, options
    )[0][0]
    tensorisation = temp[1]
    _mask_mod = mask(
        H_mod, 
        jnp.array(dict_nD_reshapings(tensorisation, inequalities, options, 'proj'))
    )
    Hred_mod, rho_mod, *Lsred_mod = [
        projection_nD(x, _mask_mod) for x in [H_mod] + [rho_mod] + 
        [L for L in jump_ops_mod]
    ]
    rextend_args = find_reextension_params(tensorisation, options.tensorisation)

    H_mod = _astimearray(H_mod)
    jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
    Hred_mod = _astimearray(Hred_mod)
    Lsred_mod = [_astimearray(L) for L in Lsred_mod]
    return (
        options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation, rextend_args
    )

def error_reducing(rho, options):
    # compute the error made by reducing rho
    rec_ineq = [a[1] - b for a, b in 
        zip(options.inequalities, options.trunc_size)]
    inequalities = generate_rec_ineqs(rec_ineq)
    rec_ineq_prec = [a[1] for a in options.inequalities]
    inequalities_previous = generate_rec_ineqs(rec_ineq_prec)
    tensorisation = ineq_to_tensorisation(inequalities_previous, options.tensorisation)
    _mask = mask(
        rho, 
        dict_nD_reshapings(tensorisation, inequalities, options, usage = 'reduce')
    )
    proj_reducing = projection_nD(rho, _mask)
    # jax.debug.print("rho{a}\nand reduced{b}", a=rho, b=proj_reducing[0])
    return jnp.linalg.norm(proj_reducing[0]-rho, ord='nuc')
# 
def trace_ineq_states(tensorisation, inequalities, rho):
    # compute a partial trace only on the states concerned by the inequalities (ie that
    # would be suppresed if one applied a reduction_nD on those)
    dictio = dict_nD(tensorisation, inequalities)
    return sum([rho[i][i] for i in dictio])


