from ..n_D.projection_nD import (
    projection_nD, dict_nD, mask, reduction_nD, extension_nD, red_ext_full, 
    red_ext_zeros
)
from ..n_D.tensorisation_maker import tensorisation_maker
from ..n_D.inequalities import generate_rec_ineqs, ineq_from_params
from ...core._utils import _astimearray
from ...options import Options

import jax.numpy as jnp
from ..n_D.inequalities import generate_rec_ineqs, generate_rec_func, update_ineq
from ..utils.utils import prod, ineq_to_tensorisation, to_hashable
from ..one_D.degree_guesser_1D import degree_guesser_list
from ..n_D.degree_guesser_nD import degree_guesser_nD_list
from ...time_array import ConstantTimeArray, TimeArray
from ...utils.states import fock_dm
from ...utils.operators import destroy
from ...utils.utils.general import tensor, dag
from ...utils.random import rand_dm
from .estimator_derivate_nD import estimator_derivate_opti_nD
import jax

import jax.numpy as jnp
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
        tensorisation = temp[1]
        _mask_mod = mask(H_mod, dict_nD(tensorisation, inequalities, options, 'proj'))
        Hred_mod, rho0_mod, *Lsred_mod = projection_nD(
            [H_mod] + [rho0_mod] + [L for L in jump_ops_mod],
            None, None, None, _mask_mod
        )

        H_mod = _astimearray(H_mod)
        jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
        Hred_mod = _astimearray(Hred_mod)
        Lsred_mod = [_astimearray(L) for L in Lsred_mod]
        return (options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho0_mod, _mask_mod, 
            tensorisation, ineq_set
        )
    
def reshaping_extend(
        options, H, Ls, rho, tensorisation, t, ineq_set
    ):
    H = H(t)
    Ls = jnp.stack([L(t) for L in Ls])
    options = update_ineq(options, direction='up')
    inequalities = ineq_from_params(ineq_set, [options.inequalities[i][1] for i in 
        range(len(options.inequalities))]
    )
    temp = extension_nD(
        [rho], options, inequalities
    )
    rho_mod = jnp.array(temp[0])[0]
    tensorisation = temp[1]
    _mask = mask(rho_mod, dict_nD(tensorisation, inequalities, options, 'proj'))
    temp = red_ext_full([H] + [L for L in Ls], 
        tensorisation_maker(options.tensorisation), inequalities, options
    )
    H_mod, *Ls_mod = temp[0]
    tensorisation = temp[1]
    Hred_mod, *Lsred_mod = projection_nD(
        [H_mod] + [L for L in Ls_mod], None, None, None, _mask)
    
    H_mod = _astimearray(H_mod)
    Ls_mod = [_astimearray(L) for L in Ls_mod]
    Hred_mod = _astimearray(Hred_mod)
    Lsred_mod = [_astimearray(L) for L in Lsred_mod]
    return options, H_mod, Ls_mod, Hred_mod, Lsred_mod, rho_mod, _mask, tensorisation

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
    _mask_mod = mask(H_mod, dict_nD(tensorisation, inequalities, options, 'proj'))
    Hred_mod, rho_mod, *Lsred_mod = projection_nD(
        [H_mod] + [rho_mod] + [L for L in jump_ops_mod],
        None, None, None, _mask_mod
    )

    H_mod = _astimearray(H_mod)
    jump_ops_mod = [_astimearray(L) for L in jump_ops_mod]
    Hred_mod = _astimearray(Hred_mod)
    Lsred_mod = [_astimearray(L) for L in Lsred_mod]
    return options, H_mod, jump_ops_mod, Hred_mod, Lsred_mod, rho_mod, _mask_mod, tensorisation

def error_reducing(rho, options):
    # compute the error made by reducing rho
    rec_ineq = [a[1] - b for a, b in 
        zip(options.inequalities, options.trunc_size)]
    inequalities = generate_rec_ineqs(rec_ineq)
    rec_ineq_prec = [a[1] for a in options.inequalities]
    inequalities_previous = generate_rec_ineqs(rec_ineq_prec)
    tensorisation = ineq_to_tensorisation(inequalities_previous, options.tensorisation)
    proj_reducing = projection_nD([rho], tensorisation, inequalities, options)
    # jax.debug.print("rho{a}\nand reduced{b}", a=rho, b=proj_reducing[0])
    return jnp.linalg.norm(proj_reducing[0]-rho, ord='nuc')
# 
def trace_ineq_states(tensorisation, inequalities, rho):
    # compute a partial trace only on the states concerned by the inequalities (ie that
    # would be suppresed if one applied a reduction_nD on those)
    dictio = dict_nD(tensorisation, inequalities)
    return sum([rho[i][i] for i in dictio])

def check_max_reshaping_reached(options, obj: TimeArray):
    return prod(options.tensorisation)==len(obj(0)[0])

def check_not_under_truncature(tensorisation, trunc_size):
    """
    Check if the a certain tensorsation is not under 2 times the truncature to avoid
    problems with the estimator.
    For instance if we have trunc_size = [8,4] and we check the tensorisation [7,1], it 
    returns true, while [8,1] returns False (tensorisation starting at 0)
    """
    return any(tensorisation[j] > max(trunc_size[j] - 1, 0)
        for j in range(len(trunc_size))
    )

def check_in_max_truncature(tensorisation, options):
    """
    Add this tensorisation to the mask for projecting if it's near the max attainable 
    tensorisation. (needed to compute the estimator)
    """
    return any(tensorisation[j] > options.tensorisation[j] - 1 - 
        options.trunc_size[j] for j in range(len(options.tensorisation))
    )

def unit_test_mesolve_estimator_init():
    from ..utils.mesolve_fcts import mesolve_estimator_init
    def run_mesolve_estimator_init(lazy_tensorisation):
        tsave = jnp.linspace(0, 1 , 100)
        product_rho = prod([a for a in lazy_tensorisation])
        H = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        Ls = [jnp.arange(1, product_rho**(2) + 1).reshape(product_rho, product_rho),jnp.arange(2, product_rho**(2) + 2).reshape(product_rho, product_rho)]
        options = Options(estimator=True, tensorisation=lazy_tensorisation, reshaping=False, trunc_size=[1 + i for i in range(len(lazy_tensorisation))]) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        print("H:", H(0))
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        print("Hred:", res_init[1](0), "\nmask:", res_init[3], "\ntensorisation:", res_init[5])
        return res_init
    lazy_tensorisation_1D = [5]
    lazy_tensorisation_2D = [3,4]
    res_init_1D = run_mesolve_estimator_init(lazy_tensorisation_1D)
    res_init_2D = run_mesolve_estimator_init(lazy_tensorisation_2D)
    expected_Hred_1D = matrix = jnp.array([
        [ 0.,  1.,  2.,  3.,  0.],
        [ 5.,  6.,  7.,  8.,  0.],
        [10., 11., 12., 13.,  0.],
        [15., 16., 17., 18.,  0.],
        [ 0.,  0.,  0.,  0.,  0.]])
    expected_mask_1D = jnp.array([
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True, False],
        [False, False, False, False, False]])
    expected_tensorisation_1D = [[0], [1], [2], [3], [4]]
    expected_mask_2D = jnp.array([
        [ True,  True, False, False,  True,  True, False, False, False, False, False, False],
        [ True,  True, False, False,  True,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [ True,  True, False, False,  True,  True, False, False, False, False, False, False],
        [ True,  True, False, False,  True,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False]])
    expected_tensorisation_2D = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
    return (jnp.array_equal(expected_Hred_1D, res_init_1D[1](0)) and 
            jnp.array_equal(expected_mask_1D, res_init_1D[3]) and 
            jnp.array_equal(expected_tensorisation_1D, res_init_1D[5]) and
            jnp.array_equal(expected_mask_2D, res_init_2D[3]) and 
            jnp.array_equal(expected_tensorisation_2D, res_init_2D[5])
    )

def unit_test_reshaping_init():
    from ..utils.mesolve_fcts import mesolve_estimator_init
    def run_reshaping_init(lazy_tensorisation):
        tsave = jnp.linspace(0, 1 , 100)
        product_rho = prod([a for a in lazy_tensorisation])
        H = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        Ls = [jnp.arange(1, product_rho**(2) + 1).reshape(product_rho, product_rho), 
            jnp.arange(2, product_rho**(2) + 2).reshape(product_rho, product_rho)
        ]
        rho0 = fock_dm(lazy_tensorisation, [0 for x in lazy_tensorisation]) 
        options = Options(estimator=True, tensorisation=lazy_tensorisation, 
            reshaping=True, trunc_size=[1 + i for i in range(len(lazy_tensorisation))]
        ) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        print("H:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[5], tsave, atol
        )
        print("ineq", resh_init[0].inequalities, "\ntrunc size:", 
            resh_init[0].trunc_size, "\nH_mod:", resh_init[1](0)[0], "\nHred_mod:", 
            resh_init[3](0)[0], "\nmask:", resh_init[6][0], "\ntensorisation:", 
            resh_init[7]
        )
        return resh_init
    
    lazy_tensorisation_1D = [10]
    lazy_tensorisation_2D = [9,7]
    resh_init_1D = run_reshaping_init(lazy_tensorisation_1D)
    resh_init_2D = run_reshaping_init(lazy_tensorisation_2D)
    expected_H_mod_1D = jnp.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected_Hred_mod_1D = jnp.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
    expected_mask_1D = jnp.array([ True,  True,  True,  True, True, False])
    expected_tensorisation_1D = [(0,), (1,), (2,), (3,), (4,), (5,)]
    expected_ineq_1D = [[None, 4, 1, 1]]

    expected_H_mod_2D = jnp.array(
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 
        25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40])
    expected_Hred_mod_2D = jnp.array(
        [0, 1, 2, 3, 0, 0, 7, 8, 9, 10, 0, 0, 14, 15, 16, 17, 0, 0, 21, 22, 23, 24, 0, 
        0, 28, 29, 30, 31, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_mask_2D = jnp.array(
        [True,True,True,True,False,False,True,True,True,True,False,False,True,True,True,
        True,False,False,True,True,True,True,False,False,True,True,True,True,False,
        False,False,False,False,False,False,False])
    expected_tensorisation_2D = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), 
        (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1), (4, 2), 
        (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
    expected_ineq_2D = [[None, 4, 1, 1], [None, 3, 2, 2]] 
    working = [
        jnp.array_equal(expected_H_mod_1D, resh_init_1D[1](0)[0]),
        jnp.array_equal(expected_Hred_mod_1D, resh_init_1D[3](0)[0]),
        jnp.array_equal(expected_mask_1D, resh_init_1D[6][0]),
        jnp.array_equal(expected_tensorisation_1D, resh_init_1D[7]),
        (expected_ineq_1D == resh_init_1D[0].inequalities),
        jnp.array_equal(expected_H_mod_2D, resh_init_2D[1](0)[0]),
        jnp.array_equal(expected_Hred_mod_2D, resh_init_2D[3](0)[0]),
        jnp.array_equal(expected_mask_2D, resh_init_2D[6][0]),
        jnp.array_equal(expected_tensorisation_2D, resh_init_2D[7]),
        (expected_ineq_2D == resh_init_2D[0].inequalities),
    ]
    print(working)
    return (all(working))

def unit_test_reshaping_extend():
    from ..utils.mesolve_fcts import mesolve_estimator_init
    def run_extension(lazy_tensorisation):
        tsave = jnp.linspace(0, 1 , 100)
        product_rho = prod([a for a in lazy_tensorisation])
        H = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        Ls = [jnp.arange(1, product_rho**(2) + 1).reshape(product_rho, product_rho), 
            jnp.arange(2, product_rho**(2) + 2).reshape(product_rho, product_rho)
        ]
        rho0 = fock_dm(lazy_tensorisation, [0 for x in lazy_tensorisation]) 
        options = Options(estimator=True, tensorisation=lazy_tensorisation, 
            reshaping=True, trunc_size=[1 + i for i in range(len(lazy_tensorisation))]
        ) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        print("\nH:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[5], tsave, atol)
        n, _ = jnp.array(resh_init[1](0)).shape
        rho = jnp.arange(0,n**2).reshape(n,n) # on refait un rho pour voir l'effet du reshaping
        print("rho", rho[0])
        print("objects sizes after initial reshaping (+trunc_size(", options.trunc_size,
             ") to add)", resh_init[0].inequalities
        )
        resh_ext = reshaping_extend(resh_init[0], H, Ls, rho, resh_init[7], tsave[0], 
            resh_init[8])
        print("H_mod:", resh_ext[1](0)[0], "\nHred_mod:", resh_ext[3](0)[0], "\nmask:", 
            resh_ext[6][0], "\nrho", resh_ext[5][0], "\ntensorisation:", resh_ext[7], 
            "\nineq", resh_ext[0].inequalities
        )
        return resh_ext
    
    def run_estimator_on_extension_2D():
        # to check that after a standard extension we get 0 for the estimator
        def jump_ops2(n_a,n_b):
            kappa=1.0
            a_a = destroy(n_a)
            identity_a = jnp.identity(n_a)
            a_b = destroy(n_b)
            identity_b = jnp.identity(n_b)
            tensorial_a_a = tensor(a_a,identity_b)
            tensorial_a_b = tensor(identity_a,a_b)
            # jump_ops = [jnp.sqrt(kappa)*(tensorial_a_a@tensorial_a_a-jnp.identity(n_a*n_b)*alpha**2),jnp.identity(n_a*n_b)]
            # jump_ops = [jnp.sqrt(kappa)*(tensorial_a_a@tensorial_a_a-jnp.identity(n_a*n_b)*alpha**2)]
            jump_ops = [jnp.sqrt(kappa)*tensorial_a_b]
            return jump_ops
        def H2(n_a,n_b):
            a_a = destroy(n_a)
            identity_a=jnp.identity(n_a)
            a_b = destroy(n_b)
            identity_b=jnp.identity(n_b)
            tensorial_a_a=tensor(a_a,identity_b) #tensorial operations, verified to be in the logical format a_a (x) identity_b
            tensorial_a_b=tensor(identity_a,a_b)
            H=(
                (tensorial_a_a@tensorial_a_a)@dag(tensorial_a_b) + 
            dag(tensorial_a_a@tensorial_a_a)@tensorial_a_b - 
            tensorial_a_b -
            dag(tensorial_a_b)
            )
            return H
        def rho2(n_a,n_b):
            key =  jax.random.PRNGKey(42)
            return rand_dm(key, (n_a*n_b,n_a*n_b)) #|00><00|
        lazy_tensorisation = [15,19]
        tsave = jnp.linspace(0, 1 , 100)
        H = H2(*lazy_tensorisation)
        Ls = jump_ops2(*lazy_tensorisation)
        rho0 = rho2(*lazy_tensorisation)
        options = Options(estimator=True, tensorisation=lazy_tensorisation, reshaping=True, trunc_size=[2*x for x in degree_guesser_nD_list(H, Ls, lazy_tensorisation)]) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 10000 # to force reshaping
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], res_init[3], rho0, res_init[5], tsave, atol)
        n, _ = jnp.array(resh_init[1](0)).shape
        print("\nobjects sizes after initial reshaping (+trunc_size to add)", resh_init[0].inequalities, "\ntrunc_size:", options.trunc_size)
        resh_ext = reshaping_extend(resh_init[0], H, Ls, resh_init[5], resh_init[7], tsave[0], resh_init[8])
        print("\nH_mod:", resh_ext[1](0)[0], "\nHred_mod:", resh_ext[3](0)[0], "\nmask:", resh_ext[6][0], "\nrho", resh_ext[5][0], "\ntensorisation:", resh_ext[7], "\nineq", resh_ext[0].inequalities)
        Lsred_mod = jnp.stack([L(0) for L in resh_ext[4]])
        Lsd = dag(Lsred_mod)
        LdL = (Lsd @ Lsred_mod).sum(0)
        tmp = (-1j * resh_ext[3](0) - 0.5 * LdL) @ resh_ext[5] + 0.5 * (Lsred_mod @ resh_ext[5] @ Lsd).sum(0)
        drho = tmp + dag(tmp)
        return estimator_derivate_opti_nD(drho, resh_ext[1](0), jnp.stack([L(0) for L in resh_ext[2]]), resh_ext[5])
    
    # test 1D
    lazy_tensorisation_1D = [10]
    resh_ext_1D = run_extension(lazy_tensorisation_1D)
    # test 2D
    lazy_tensorisation_2D = [7,11]
    resh_ext_2D = run_extension(lazy_tensorisation_2D)

    expected_H_mod_1D = jnp.array([0, 1, 2, 3, 4, 5, 6])
    expected_Hred_mod_1D = jnp.array([0, 1, 2, 3, 4, 5, 0])
    expected_mask_1D = jnp.array([True, True, True, True, True, True, False])
    expected_rho_1D = jnp.array([0, 1, 2, 3, 4, 5, 0])
    expected_tensorisation_1D = [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
    expected_ineq_1D = [[None, 5, 1, 1]]
    
    expected_tensorisation_2D_last = (5, 9)
    expected_shape_objects = prod([x+1 for x in expected_tensorisation_2D_last])
    expected_mask_2D = jnp.array([True, True, True, True, True, True, True, True, False,
        False, True, True, True, True, True, True, True, True, False, False, True, True,
        True, True, True, True, True, True, False, False, True, True, True, True, True, 
        True, True, True, False, False, True, True, True, True, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, 
        False])
    expected_H_mod_2D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 
        19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 
        41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 
        63, 64]
    expected_ineq_2D = [[None, 4, 1, 1], [None, 7, 2, 2]]

    estimator = run_estimator_on_extension_2D()

    working = [
        jnp.array_equal(expected_H_mod_1D, resh_ext_1D[1](0)[0]) ,
        jnp.array_equal(expected_Hred_mod_1D, resh_ext_1D[3](0)[0]) ,
        jnp.array_equal(expected_mask_1D, resh_ext_1D[6][0]) ,
        jnp.array_equal(expected_rho_1D, resh_ext_1D[5][0]) ,
        jnp.array_equal(expected_tensorisation_1D, resh_ext_1D[7]) ,
        (expected_ineq_1D == resh_ext_1D[0].inequalities) ,
        jnp.array_equal(expected_H_mod_2D, resh_ext_2D[1](0)[0]) ,
        jnp.array_equal(expected_mask_2D, resh_ext_2D[6][0]) ,
        (expected_ineq_2D == resh_ext_2D[0].inequalities) ,
        (expected_tensorisation_2D_last == resh_ext_2D[7][-1]) , 
        (expected_shape_objects==jnp.array(resh_ext_2D[5]).shape[0]) ,
        estimator==0
    ]
    print(working)
    return (all(working))

def unit_test_error_reducing():
    def run_test(lazy_tensorisation):
        product_rho = prod([a for a in lazy_tensorisation])
        rho = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        tensorisation_max = [4 * a for a in lazy_tensorisation]
        trunc_size = jnp.array([1 + i for i in range(len(lazy_tensorisation))])
        rec_ineq_prec = jnp.array([a - 1 for a in lazy_tensorisation])
        inequalities_previous = generate_rec_ineqs(rec_ineq_prec)
        options = Options(trunc_size=trunc_size, tensorisation=tensorisation_max,
            inequalities=[[a, b] for a, b in zip(inequalities_previous, rec_ineq_prec)]
        )
        return error_reducing(rho, options)
    lazy_tensorisation_2D = [3,4]
    res = run_test(lazy_tensorisation_2D)
    expected_res = 1123.5209
    return (expected_res == res)

def unit_test_reshaping_reduce():
    from ..utils.mesolve_fcts import mesolve_estimator_init
    def run_reshaping_reduce(lazy_tensorisation):
        tsave = jnp.linspace(0, 1 , 100)
        product_rho = prod([a for a in lazy_tensorisation])
        H = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        Ls = [jnp.arange(1, product_rho**(2) + 1).reshape(product_rho, product_rho), 
            jnp.arange(2, product_rho**(2) + 2).reshape(product_rho, product_rho)
        ]
        rho0 = fock_dm(lazy_tensorisation, [0 for x in lazy_tensorisation]) 
        options = Options(estimator=True, tensorisation=lazy_tensorisation, 
            reshaping=True, trunc_size=[1 + i for i in range(len(lazy_tensorisation))]
        ) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        print("\nH:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[5], tsave, atol)
        n, _ = jnp.array(resh_init[1](0)).shape
        rho = jnp.arange(0,n**2).reshape(n,n) # on refait un rho pour voir l'effet du reshaping
        print("rho", rho[0])
        print("objects sizes after initial reshaping (+trunc_size(", options.trunc_size,
             ") to add)", resh_init[0].inequalities
        )
        resh_red = reshapings_reduce(resh_init[0], H, Ls, rho, resh_init[7], tsave[0], 
            resh_init[8])
        print("H_mod:", resh_red[1](0)[0], "\nHred_mod:", resh_red[3](0)[0], "\nmask:", 
            resh_red[6][0], "\nrho", resh_red[5][0], "\ntensorisation:", resh_red[7], 
            "\nineq", resh_red[0].inequalities
        )
        return resh_red
    
    # standard test 1D
    lazy_tensorisation_1D = [10]
    resh_init_1D = run_reshaping_reduce(lazy_tensorisation_1D)
    # standard test 2D
    lazy_tensorisation_2D = [6,10]
    resh_init_2D = run_reshaping_reduce(lazy_tensorisation_2D)

    expected_H_mod_1D = jnp.array([0, 1, 2, 3, 4])
    expected_Hred_mod_1D = jnp.array([0, 1, 2, 3, 0])
    expected_mask_1D = jnp.array([True, True, True, True, False])
    expected_tensorisation_1D = [(0,), (1,), (2,), (3,), (4,)]
    expected_ineq_1D = [[None, 3, 1, 1]]

    expected_H_mod_2D = jnp.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23
        , 24]) # just the first line
    expected_Hred_mod_2D = jnp.array([0, 1, 2, 0, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0])
    expected_mask_2D = jnp.array([True, True, True, False, False, True, True, True, 
        False, False, False, False, False, False, False])
    expected_tensorisation_2D = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1),
        (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    expected_ineq_2D = [[None, 1, 1, 1], [None, 2, 2, 2]]

    working = [jnp.array_equal(expected_H_mod_1D, resh_init_1D[1](0)[0]) ,
        jnp.array_equal(expected_Hred_mod_1D, resh_init_1D[3](0)[0]) ,
        jnp.array_equal(expected_mask_1D, resh_init_1D[6][0]) ,
        jnp.array_equal(expected_tensorisation_1D, resh_init_1D[7]) ,
        (expected_ineq_1D == resh_init_1D[0].inequalities) ,
        jnp.array_equal(expected_Hred_mod_2D, resh_init_2D[3](0)[0]) ,
        jnp.array_equal(expected_H_mod_2D, resh_init_2D[1](0)[0]) ,
        jnp.array_equal(expected_mask_2D, resh_init_2D[6][0]) ,
        jnp.array_equal(expected_tensorisation_2D, resh_init_2D[7]) ,
        (expected_ineq_2D == resh_init_2D[0].inequalities)
    ]
    print(working)
    return (all(working))



