from ..n_D.projection_nD import projection_nD, dict_nD, mask, reduction_nD, extension_nD
from ..n_D.tensorisation_maker import tensorisation_maker
from ..n_D.inequalities import generate_rec_ineqs
from ...core._utils import _astimearray
from ...options import Options

import jax.numpy as jnp

def reshaping_init(
        tsave, H, jump_ops, Hred, Lsred, rho0, tensorisation, options, _mask, solver
    ):
    # On commence en diminuant beaucoup la taille par rapport à la saisie utiliateur
    # qui correspond à la taille maximal atteignable

    # We first check if it does not create too much of an approximation to modify the
    # matrix size this way
    """WARNING we do it by checking the diagonal values, which is not theoreticaly
    justified.
    Also we just do an absolute tolerance, relative tolerance would need to compute
    a norm which is expensive."""
    inequalities = generate_rec_ineqs(
            [a//2 for a, b in 
            zip(options.tensorisation, options.trunc_size)]
    )
    if (
        trace_ineq_states(tensorisation, inequalities, rho0) > 1/len(tsave) * 
        (solver.atol)
    ):
        print("""Your initial state is already populated in the high Fock states
              dimensions. Computation may be slower.
              """)
        return H, jump_ops, Hred, Lsred, rho0_mod, _mask
    else:
        H_mod, rho0_mod, *jump_ops_mod = reduction_nD(
            [H(tsave[-1])] + [rho0] + [L(tsave[-1]) for L in jump_ops],
            tensorisation, inequalities
        )
        inequalities = generate_rec_ineqs(
            [a//2 - b for a, b in 
            zip(options.tensorisation, options.trunc_size)]
        )
        # we set our rectangular params
        tmp_dic=options.__dict__
        tmp_dic['inequalities'] = [[None, a//2] for a in options.tensorisation]
        options=Options(**tmp_dic) 
        _mask = mask(H_mod, dict_nD(tensorisation, inequalities))
        Hred_mod, rho0_mod, *Lsred_mod = projection_nD(
            [H_mod] + [rho0_mod] + [L for L in jump_ops_mod], 
            None, None, _mask
        )
        H_mod = _astimearray(H_mod)
        jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
        Hred_mod = _astimearray(Hred_mod)
        Lsred_mod = [_astimearray(L) for L in Lsred_mod]
        return H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho0_mod, _mask
    
# def reshaping_extend(
#         tsave, H, jump_ops, Hred, Lsred, rho0, tensorisation, options, _mask, solver):
#     if 
#     return H, jump_ops, Hred, Lsred, rho0_mod, _mask

def trace_ineq_states(tensorisation, inequalities, rho):
    # compute a partial trace only on the states concerned by the inequalities (ie that
    # would be suppresed if one applied a reduction_nD on those)
    dictio = dict_nD(tensorisation, inequalities)
    return sum([rho[i][i] for i in dictio])