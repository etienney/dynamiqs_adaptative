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
    find_reextension_params,
    to_hashable
)
from ..time_array import (
    ModulatedTimeArray, 
    PWCTimeArray, 
    ConstantTimeArray, 
    SummedTimeArray,
    pwc
)
import copy


def reshaping_init(
         options, H, jump_ops, rho0, tensorisation, tsave, atol
    ):
    # On commence en diminuant beaucoup la taille par rapport à la saisie utiliateur
    # qui correspond à la taille maximal atteignable

    # We first check if it does not create too much of an approximation to modify the
    # matrix size this way
    """WARNING we do it by checking the diagonal values, which is not theoreticaly
    justified. it's just a guess.
    Also we just do an absolute tolerance, relative tolerance would need to compute
    a norm which is expensive."""
    rextend_args = []
    if options.inequalities is None:
        ineq_params =  [(a-1)//2 for a, b in 
            zip(options.tensorisation, options.trunc_size)
        ]
        up = options.trunc_size
        down = options.trunc_size
        ineq_set = [generate_rec_func(j) for j in range(len(ineq_params))]
        inequalities = ineq_from_params(ineq_set, ineq_params)
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
        options_ineq_0 = [0] * (len(options.tensorisation)) # JAX cannot stand lambda functions
            # when called with options so it is necessary to replace ineq_set by 0 (not None otherwise # (not None otherwise it is not arrayable)
        tmp_dic['inequalities'] = [[a, b, c , d] for a, b, c, d in 
            zip(options_ineq_0, ineq_params, up, down)
        ]
        options=Options(**tmp_dic) 

        temp = get_array(H)
        len_H_array = temp[0]
        H_array = temp[1:][0]
        Ls_array = jnp.stack([L.array for L in jump_ops])
        temp = red_ext_full(
            H_array + [rho0] + [L for L in Ls_array],
            tensorisation, inequalities, options
        )
        tensorisation_mod = temp[1]
        H_mod = temp[0][0:len_H_array]
        rho0_mod = temp[0][len_H_array]
        Ls_mod = temp[0][len_H_array+1:]
        _mask_mod = mask(
            H_mod[0], 
            jnp.array(dict_nD_reshapings(tensorisation_mod, inequalities, options, 'proj'))
        )
        temp = [
            projection_nD(x, _mask_mod) for x in H_mod + [rho0_mod] + 
            [L for L in Ls_mod]
        ]
        Hred_mod = temp[0:len_H_array]
        rho0_mod = temp[len_H_array]
        Lsred_mod = temp[len_H_array+1:]
        rextend_args.append(find_reextension_params(tensorisation_mod, options.tensorisation))
        ineq_set = to_hashable(ineq_set)
        # reconvert to Timearray args
        H_mod = re_timearray(H_mod, H)
        Ls_mod = [re_timearray(L, or_L) for L,or_L in zip(Ls_mod, jump_ops)]
        Hred_mod = re_timearray(Hred_mod, H)
        Lsred_mod = [re_timearray(L, or_L) for L,or_L in zip(Lsred_mod, jump_ops)]
        return (options, H_mod, Ls_mod, Hred_mod, Lsred_mod, rho0_mod, _mask_mod, 
            tensorisation_mod, ineq_set, rextend_args
        )
    

def reshaping_extend(
        options, H, Ls, rho, tensorisation, t, ineq_set, rextend_args
    ):
    temp = get_array(H)
    len_H_array = temp[0]
    H_array = temp[1:][0]
    Ls_array = jnp.stack([L.array for L in Ls])
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
    temp = red_ext_full(H_array + [L for L in Ls_array], 
        tensorisation_maker(options.tensorisation), inequalities, options
    )
    tensorisation = temp[1]
    H_mod = temp[0][0:len_H_array]
    Ls_mod = temp[0][len_H_array:]
    temp = [
            projection_nD(x, _mask_mod) for x in H_mod + [L for L in Ls_mod]
        ]
    Hred_mod = temp[0:len_H_array]
    Lsred_mod = temp[len_H_array:]
    rextend_args.append(find_reextension_params(tensorisation, options.tensorisation))
    # reconvert to Timearray args
    H_mod = re_timearray(H_mod, H)
    Ls_mod = [re_timearray(L, or_L) for L, or_L in zip(Ls_mod, Ls)]
    Hred_mod = re_timearray(Hred_mod, H)
    Lsred_mod = [re_timearray(L, or_L) for L, or_L in zip(Lsred_mod, Ls)]
    return (
        options, H_mod, Ls_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation, rextend_args
    )


def reshapings_reduce(
        options, H, Ls, rho_mod, tensorisation, t, ineq_set, rextend_args
    ):
    temp = get_array(H)
    len_H_array = temp[0]
    H_array = temp[1:][0]
    Ls_array = jnp.stack([L.array for L in Ls])
    options = update_ineq(options, direction='down')
    inequalities = ineq_from_params(ineq_set, [options.inequalities[i][1] for i in 
        range(len(options.inequalities))]
    )
    temp = red_ext_full(H_array + [L for L in Ls_array], 
        tensorisation_maker(options.tensorisation), inequalities, options
    )
    H_mod = temp[0][0:len_H_array]
    Ls_mod = temp[0][len_H_array:]
    rho_mod = red_ext_zeros(
        [rho_mod], tensorisation, inequalities, options
    )[0][0]
    tensorisation = temp[1]
    _mask_mod = mask(
        H_mod[0], 
        jnp.array(dict_nD_reshapings(tensorisation, inequalities, options, 'proj'))
    )
    temp = [
        projection_nD(x, _mask_mod) for x in H_mod + [rho_mod] + 
        [L for L in Ls_mod]
    ]
    Hred_mod = temp[0:len_H_array]
    rho_mod = temp[len_H_array]
    Lsred_mod = temp[len_H_array+1:]
    rextend_args.append(find_reextension_params(tensorisation, options.tensorisation))
    # reconvert to Timearray args
    H_mod = re_timearray(H_mod, H)
    Ls_mod = [re_timearray(L, or_L) for L,or_L in zip(Ls_mod, Ls)]
    Hred_mod = re_timearray(Hred_mod, H)
    Lsred_mod = [re_timearray(L, or_L) for L,or_L in zip(Lsred_mod, Ls)]
    return (
        options, H_mod, Ls_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, 
        tensorisation, rextend_args
    )


def error_reducing(rho, options, ineq_set):
    # compute the error made by reducing rho
    options_copy = copy.deepcopy(options)
    old_inequalities = ineq_from_params(ineq_set, [options_copy.inequalities[i][1] for i in 
        range(len(options_copy.inequalities))]
    )
    options_copy = update_ineq(options_copy, direction='down')
    inequalities = ineq_from_params(ineq_set, [options_copy.inequalities[i][1] for i in 
        range(len(options_copy.inequalities))]
    )
    tensorisation = ineq_to_tensorisation(old_inequalities, options_copy.tensorisation)
    _mask = mask(
        rho, 
        dict_nD_reshapings(tensorisation, inequalities, options_copy, usage = 'reduce')
    )
    proj_reducing = projection_nD(rho, _mask)
    # print(proj_reducing[0], rho[0])
    return jnp.linalg.norm(proj_reducing-rho, ord='nuc')


def trace_ineq_states(tensorisation, inequalities, rho):
    # compute a partial trace only on the states concerned by the inequalities (ie that
    # would be suppresed if one applied a reduction_nD on those)
    dictio = dict_nD(tensorisation, inequalities)
    return sum([rho[i][i] for i in dictio])


def re_timearray(operator, old_operator):
    # put the operator back to their original time_array form
    # only work for ModulatedTimeArray and PWCTimeArray
    time_array_class = type(old_operator)
    if time_array_class==ModulatedTimeArray:
        operator = ModulatedTimeArray(old_operator.f, operator, old_operator._disc_ts)
    elif time_array_class==PWCTimeArray:
        operator = pwc(old_operator.times, old_operator.values, operator)
    elif time_array_class==SummedTimeArray:
        operator = sum(re_timearray(x, y) for x, y in zip(operator, old_operator.timearrays))
    elif time_array_class==ConstantTimeArray:
        if len(operator)==1: # to accept constant time array in H
            operator = operator[0]
    else:
        print(
            'WARNING : If your operators are time dependant, using another class than '
            'ModulatedTimeArray, PWCTimeArray, or ConstantTimeArray (constant) '
            'the program\'s results won\'t be trustworthy'
        )
    operator = _astimearray(operator)
    return operator


def get_array(operator):
    # designed to get the array in case of a SummedTimeArray
    if type(operator)!=SummedTimeArray:
        return 1, [operator.array]
    else:
        L=[]
        for x in operator.timearrays:
            L.append(x.array)
        return len(operator.timearrays), L