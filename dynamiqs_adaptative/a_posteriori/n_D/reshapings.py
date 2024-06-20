from ..n_D.projection_nD import projection_nD, dict_nD, mask, reduction_nD, extension_nD
from ..n_D.tensorisation_maker import tensorisation_maker
from ..n_D.inequalities import generate_rec_ineqs
from ...core._utils import _astimearray
from ...options import Options

import jax.numpy as jnp
import itertools

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
    ):# verification that we don't lose too much info on rho0 by doing this
        print("""Your initial state is already populated in the high Fock states
              dimensions. Computation may be slower.
              """)
        return H, jump_ops, Hred, Lsred, rho0_mod, _mask
    else:
        temp = reduction_nD(
            [H(tsave[-1])] + [rho0] + [L(tsave[-1]) for L in jump_ops],
            tensorisation, inequalities
        )
        H_mod, rho0_mod, *jump_ops_mod = temp[0]
        tensorisation = temp[1]
        inequalities = generate_rec_ineqs(
            [a//2 - b for a, b in 
            zip(options.tensorisation, options.trunc_size)]
        )
        print(inequalities)
        # we set our rectangular params in options
        tmp_dic=options.__dict__
        tmp_dic['inequalities'] = [
            [None, a//2 - b] for a, b in zip(options.tensorisation, options.trunc_size)
        ]
        options=Options(**tmp_dic) 
        _mask_mod = mask(H_mod, dict_nD(tensorisation, inequalities))
        Hred_mod, rho0_mod, *Lsred_mod = projection_nD(
            [H_mod] + [rho0_mod] + [L for L in jump_ops_mod],
            None, None, _mask_mod
        )
        H_mod = _astimearray(H_mod)
        jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
        Hred_mod = _astimearray(Hred_mod)
        Lsred_mod = [_astimearray(L) for L in Lsred_mod]
        return H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho0_mod, _mask_mod, tensorisation
    
def reshaping_extend(
        t0, H, Ls, rho, tensorisation, options, solver
    ):
    H = H(t0)
    Ls = jnp.stack([L(t0) for L in Ls])
    # extend by trunc_size in all directions
    inequalities = generate_rec_ineqs(
        [a[1] + b for a, b in 
        zip(options.inequalities, options.trunc_size)]
    )
    extended_inequalities = inequalities = generate_rec_ineqs(
        [a[1] + 2 * b for a, b in 
        zip(options.inequalities, options.trunc_size)]
    )
    tmp_dic=options.__dict__
    tmp_dic['inequalities'] = [
        [None, a[1] + b] for a, b in zip(options.inequalities, options.trunc_size)
    ]
    options = Options(**tmp_dic)
    temp = extension_nD(
        [rho], tensorisation, options.tensorisation, extended_inequalities, options
    )
    rho_mod = jnp.array(temp[0])[0]
    print(rho_mod.shape)
    tensorisation = temp[1]
    _mask = mask(rho_mod, dict_nD(tensorisation, inequalities))
    max_tensorisation = list(
        itertools.product(*[range(max_dim) for max_dim in options.tensorisation])
    )
    temp = reduction_nD([H] + [L for L in Ls], max_tensorisation, extended_inequalities)
    H, Ls = temp[0]
    Hred, *Lsred = projection_nD([H] + [L for L in Ls], None, None, _mask)
    H_mod = _astimearray(H)
    Ls_mod = [_astimearray(L) for L in Ls]
    Hred_mod = _astimearray(Hred)
    Lsred_mod = [_astimearray(L) for L in Lsred]
    return H_mod, Ls_mod, Hred_mod, Lsred_mod, rho_mod, _mask, tensorisation
    return 0

def trace_ineq_states(tensorisation, inequalities, rho):
    # compute a partial trace only on the states concerned by the inequalities (ie that
    # would be suppresed if one applied a reduction_nD on those)
    dictio = dict_nD(tensorisation, inequalities)
    return sum([rho[i][i] for i in dictio])