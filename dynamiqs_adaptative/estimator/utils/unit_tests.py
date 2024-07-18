from ...core._utils import _astimearray
from ...options import Options
from ..._utils import cdtype
import jax.numpy as jnp
from .utils import (
    prod,
    tensorisation_maker
)
from ..estimator import compute_estimator
from ..inequalities import generate_rec_ineqs
from ..degree_guesser import degree_guesser_nD_list
from ...utils.states import fock_dm
from ...utils.operators import destroy
from ...utils.utils.general import tensor, dag
from ...utils.random import rand_dm
import jax

from ..mesolve_fcts import (
    mesolve_estimator_init,
)

from  ..reshaping_y import (
    reshaping_extend,
    reshaping_init,
    reshapings_reduce,
    error_reducing,
)

from ..reshapings import (
    projection_nD,
    extension_nD,
    red_ext_full,
    red_ext_zeros,
    mask,
    dict_nD,
    dict_nD_reshapings,
    extended_tensorisation
)

from ..degree_guesser import degree_guesser_nD



# === reshapings unit tests
def unit_test_dicts():
    # the simple dict
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    res_simple = dict_nD(original_tensorisation, inequalities)
    expected_simple = [2,5]

    # the reshaping version - general case
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    options = Options(tensorisation=[100,100], trunc_size=[0,0])
    res_gen = dict_nD_reshapings(original_tensorisation, inequalities)
    expected_gen = [2,5]

    # the reshaping version - under truncature case extend
    original_tensorisation = ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    options = Options(tensorisation=[100,100], trunc_size=[0,4])
    res_under_trunc = dict_nD_reshapings(original_tensorisation, inequalities, options)
    expected_under_trunc = [6, 7]

    # the reshaping version - under truncature case proj and red
    original_tensorisation = ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    options = Options(tensorisation=[100,100], trunc_size=[0,4])
    res_under_trunc_proj = dict_nD_reshapings(
        original_tensorisation, inequalities, options, usage='proj'
    )
    expected_under_trunc_proj = [6, 7]

    # the reshaping version - max truncature touched for proj and red
    original_tensorisation = tensorisation_maker((6,6))
    inequalities = [lambda i, j: i + j < 8]
    options = Options(tensorisation=[7,7], trunc_size=[0,3])
    res_max_trunc_proj = dict_nD_reshapings(
        original_tensorisation, inequalities, options, usage='proj'
    )
    # print([original_tensorisation[i] for i in res_max_trunc_proj])
    expected_max_trunc_proj =  [4, 5, 10, 11, 16, 17, 22, 23, 28, 29, 33, 34, 35]

    # the reshaping version - max truncature touched for proj and red and under
    # truncature touched
    original_tensorisation = tensorisation_maker((8,8))
    inequalities = [lambda i, j: 4 * i + j <= 5]
    options = Options(tensorisation=[9,9], trunc_size=[2,3])
    res_max_under_trunc_proj = dict_nD_reshapings(
        original_tensorisation, inequalities, options, usage='proj'
    )
    # print([original_tensorisation[i] for i in res_max_under_trunc_proj])
    # print(res_max_under_trunc_proj)

    expected_max_under_trunc_proj =  [6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
        62, 63]

    results = [
        jnp.array_equal(res_simple, expected_simple),
        jnp.array_equal(res_gen, expected_gen),
        jnp.array_equal(res_under_trunc, expected_under_trunc),
        jnp.array_equal(res_under_trunc_proj, expected_under_trunc_proj),
        jnp.array_equal(res_max_trunc_proj, expected_max_trunc_proj),
        jnp.array_equal(res_max_under_trunc_proj, expected_max_under_trunc_proj),
    ]
    print(results)
    return all(results)


def unit_test_mask():
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    obj = jnp.arange(1, 37).reshape(6, 6)
    res = mask(obj, dict_nD(original_tensorisation, inequalities))
    expected_mask = [True, True, False, True, True, False]

    results = [
        jnp.array_equal(res[0], expected_mask) 
    ]
    print(results)
    return all(results)


def unit_test_projection_nD():
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    obj = jnp.arange(1, 37).reshape(6, 6)
    _mask = mask(obj, dict_nD(original_tensorisation, inequalities))
    res = projection_nD(obj, _mask)
    expected_result =   jnp.array([
        [1, 2, 0, 4, 5, 0],
        [7, 8, 0, 10, 11, 0],
        [0, 0, 0, 0, 0, 0], 
        [19, 20, 0, 22, 23, 0],
        [25, 26, 0, 28, 29, 0],
        [0, 0, 0, 0, 0, 0]])

    results = [
        jnp.array_equal(res, expected_result) ,
    ]
    print(results)
    return all(results)


def unit_test_extended_tensorisation():
    # simple case
    inequalities = [lambda i, j: i + j <= 4]
    options = Options(tensorisation=[100,100], trunc_size=[1,3])
    res_simple = extended_tensorisation(inequalities, options)
    expected_simple = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), 
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), 
        (3, 5), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3)
    ]

    # reaching max_tens case
    inequalities = [lambda i, j: i + j <= 8]
    options = Options(tensorisation=[7,8], trunc_size=[1,3])
    res_max_tens = extended_tensorisation(inequalities, options)
    expected_max_tens = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), 
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), 
        (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), 
        (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), 
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), 
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)
    ]

    results = [
        jnp.array_equal(res_simple, expected_simple) ,
        jnp.array_equal(res_max_tens, expected_max_tens) ,
    ]
    print(results)
    return all(results)


def unit_test_reduction_nD():
    def run_reduction_zeros(actual, cut, trunc_size, max_tensorisation):
        tensorisation = tensorisation_maker(actual)
        ineq_params =  [x for x in cut]
        inequalities = generate_rec_ineqs(ineq_params)
        # print("square ineq: ", ineq_params)
        product = prod([a for a in actual])
        objs = [jnp.arange(0, product**(2)).reshape(product, product)]
        # print("objs :", objs[0][0])
        # print("trunc_size:", trunc_size)
        up = trunc_size
        down = trunc_size
        options = Options(trunc_size=trunc_size, 
            inequalities = [[None, ineq_params[i], up[i], down[i]] 
            for i in range(len(max_tensorisation))],
            tensorisation = max_tensorisation
        )
        temp = red_ext_zeros(objs, tensorisation, inequalities, options)
        return temp
    
    def run_reduction_full(actual, cut, trunc_size, max_tensorisation):
        tensorisation = tensorisation_maker(actual)
        ineq_params =  [x for x in cut]
        inequalities = generate_rec_ineqs(ineq_params)
        # print("square ineq: ", ineq_params)
        product = prod([a for a in max_tensorisation])
        objs = [jnp.arange(0, product**(2)).reshape(product, product)]
        # print("objs :", objs[0][0])
        # print("trunc_size:", trunc_size)
        up = trunc_size
        down = trunc_size
        options = Options(trunc_size=trunc_size, 
            inequalities = [[None, ineq_params[i], up[i], down[i]] 
            for i in range(len(max_tensorisation))],
            tensorisation = max_tensorisation
        )
        temp = red_ext_full(objs, tensorisation, inequalities, options)
        return temp
    
    # standard test
    actual = [7,8]
    cut = [5,4]
    trunc_size = [1,1]
    max_tensorisation = [15, 15]
    test_2D = run_reduction_zeros(actual, cut, trunc_size, max_tensorisation)
    # print("reduction: ", test_2D[0][0][0], "\ntensorisation:", test_2D[1])  
    # standard test but with a reduction that doesn't put zero (but the obj of max_size)
    actual = [7,8]
    cut = [5,4]
    trunc_size = [1,1]
    max_tensorisation = [15, 15]
    test_2D_full = run_reduction_full(actual, cut, trunc_size, max_tensorisation)
    # print("reduction: ", test_2D_full[0][0][0], "\ntensorisation:", test_2D_full[1]) 
    # test to see if we don't cut under trunc_size
    actual = [9,9]
    cut = [6,2]
    trunc_size = [4,4] # the 2nd index is cut
    max_tensorisation = [30, 30]
    test_2D_trunc = run_reduction_zeros(actual, cut, trunc_size, max_tensorisation)
    # print("reduction: ", test_2D_trunc[0][0][0], "\ntensorisation:", test_2D_trunc[1])   
    # print("same len:", len(test_2D_trunc[0][0][0]), len(test_2D_trunc[1]))

      
    expected_result_2D =  [0, 1, 2, 3, 4, 0, 8, 9, 10, 11, 12, 0, 16, 17, 18, 19, 20, 0, 
        24, 25, 26, 27, 28, 0, 32, 33, 34, 35, 36, 0, 40, 41, 42, 43, 44, 0, 0, 0, 0, 0,
        0, 0]
    expected_tensorsiation_2D = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), 
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), 
        (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1), (4, 2), 
        (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 0), 
        (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]

    expected_result_2D_full = [0,1,2,3,4,5,15,16,17,18,19,20,30,31,32,33,34,35,45,46,47,
        48,49,50,60,61,62,63,64,65,75,76,77,78,79,80,90,91,92,93,94,95]
    expected_tensorsiation_2D_full = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 3), 
        (2, 4), (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1), 
        (4, 2), (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), 
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
    
    expected_result_2D_trunc = [0, 1, 2, 3, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 18, 
        19, 20, 21, 0, 0, 0, 0, 27, 28, 29, 30, 0, 0, 0, 0, 36, 37, 38, 0, 0, 0, 0, 0, 
        45, 46, 47, 0, 0, 0, 0, 0, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_tensorsiation_2D_trunc = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
        (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), 
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), 
        (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), 
        (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), 
        (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), 
        (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (8, 0), (8, 1), 
        (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), 
        (9, 5), (9, 6), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6)]

    working = [
        jnp.array_equal(test_2D[0][0][0], expected_result_2D) , 
        jnp.array_equal(test_2D[1], expected_tensorsiation_2D) ,
        jnp.array_equal(test_2D_full[0][0][0], expected_result_2D_full) , 
        jnp.array_equal(test_2D_full[1], expected_tensorsiation_2D_full),
        jnp.array_equal(test_2D_trunc[0][0][0], expected_result_2D_trunc) , 
        jnp.array_equal(test_2D_trunc[1], expected_tensorsiation_2D_trunc) , 
    ]
    print(working)
    return all(working)


def unit_test_extension_nD():
    def run_test(lazy_tensorisation, max_lazy_tensorisation, trunc_size):
        ineqs_params = [ x - 1 - trunc_size[i] for x, i in 
            zip(lazy_tensorisation, range(len(lazy_tensorisation)))
        ] # -1 because tensorisation starts at 0
        full_ineq_params = [x + b for x, b in zip(ineqs_params, 
            trunc_size)
        ]
        full_inequalities = generate_rec_ineqs(full_ineq_params)
        print()
        up = trunc_size
        down= trunc_size
        options = Options(
            tensorisation=max_lazy_tensorisation, trunc_size=trunc_size, 
            inequalities = [[None, full_ineq_params[i], up[i], down[i]] 
            for i in range(len(lazy_tensorisation))]
        )
        product = prod([a for a in lazy_tensorisation])
        objs = [jnp.arange(0, product**2).reshape(product, product)]
        # print(
        #     "square ineq: ", ineqs_params, "\ntrunc size", trunc_size, 
        #     "\nmax tensorisation", max_lazy_tensorisation,
        #     "\ntensorisation init:", tensorisation_maker(lazy_tensorisation), 
        #     "\nobj initial:", objs[0][0]
        # )
        temp = extension_nD(objs, options, full_inequalities)
        # print("extension", temp[0][0][0])
        return temp
    
    lazy_tensorisation_1D = [7]
    max_lazy_tensorisation_1D = [100]
    trunc_size_1D = [4]
    lazy_tensorisation_2D = [7,4]
    max_lazy_tensorisation_2D = [100,100]
    trunc_size_2D = [4,2]
    max_lazy_tensorisation_1D_short = [8]
    max_lazy_tensorisation_2D_short = [9,10]

    # simple test
    ext_1D = run_test(lazy_tensorisation_1D, max_lazy_tensorisation_1D, trunc_size_1D)
    ext_2D = run_test(lazy_tensorisation_2D, max_lazy_tensorisation_2D, trunc_size_2D)
    # a test when we extend bu we are stopped by the boundary of the max_tensorisation
    ext_1D_max = run_test(lazy_tensorisation_1D, max_lazy_tensorisation_1D_short, 
        trunc_size_1D
    )
    ext_2D_max = run_test(lazy_tensorisation_2D, max_lazy_tensorisation_2D_short, 
        trunc_size_2D
    )

    expected_first_line_2D =  jnp.array(
        [0., 1., 2., 3., 0., 0., 4., 5., 6., 7., 0., 0., 8., 9., 10., 11., 0., 0.,
        12., 13., 14., 15., 0., 0., 16., 17., 18., 19., 0., 0., 20., 21., 22., 23.,
        0., 0., 24., 25., 26., 27., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    expected_first_line_2D_max = jnp.array(
        [0., 1., 2., 3., 0., 0., 4., 5., 6., 7., 0., 0.,
        8., 9., 10., 11., 0., 0., 12., 13., 14., 15., 0., 0.,
        16., 17., 18., 19., 0., 0., 20., 21., 22., 23., 0., 0.,
        24., 25., 26., 27., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.]
    ) # to check that near the max_extension it's also okay

    results = [
        jnp.array_equal(expected_first_line_2D, ext_2D[0][0][0]) ,
        jnp.array_equal(expected_first_line_2D_max, ext_2D_max[0][0][0])
    ]
    print(results)
    return all(results)



# === degree_guesser unit tests
def unit_test_degree_guesser_nD():
    n_a=5
    n_b=5
    n_c=6
    tensorisation = (n_a, n_b, n_c)

    a_a = destroy(n_a)
    identity_a = jnp.identity(n_a)
    a_b = destroy(n_b)
    a_c = destroy(n_c)
    identity_b = jnp.identity(n_b)
    identity_c = jnp.identity(n_c)
    t_a = tensor(a_a,identity_b,identity_c) #tensorial operations, verified to be in the logical format a_a (x) identity_b
    t_b = tensor(identity_a,a_b,identity_c)
    t_c = tensor(identity_a,identity_b,a_c)
    obj = (
        t_a@t_b@t_a@t_a@t_a@t_a@t_a@t_a+ t_b@t_a + 
        tensor(identity_a,identity_b,identity_c) + t_b@t_a@t_a
        +t_c@t_a@t_a@t_a +dag(t_b)@dag(t_b)@t_c@t_c
    )
    obj2 = (
        (t_a@t_a)@dag(t_b) + 
    dag(t_a@t_a)@t_b - 
    t_b -
    dag(t_b)
    )
    ide = tensor(identity_a,identity_b,identity_c)
    objdeg = degree_guesser_nD(obj, list(tensorisation))
    obj2deg = degree_guesser_nD(obj2, list(tensorisation))
    idedeg = degree_guesser_nD(ide, list(tensorisation))
    if (
        jnp.array_equal(objdeg,[3,2,2]) and
        jnp.array_equal(idedeg,[0,0,0]) and
        jnp.array_equal(obj2deg,[2,1,0])
    ):
        return True
    else:
        print(
            f'{objdeg} not [3,2,2] ?', 
            f'{obj2deg} not [2,1,0] ?', 
            f'{idedeg} not [0,0,0] ?'
        )
        return False



# === mesolve_fcts unit tests
def unit_test_mesolve_estimator_init():
    def run_mesolve_estimator_init(lazy_tensorisation):
        tsave = jnp.linspace(0, 1 , 100)
        product_rho = prod([a for a in lazy_tensorisation])
        H = jnp.arange(0, product_rho**(2)).reshape(product_rho, product_rho)
        Ls = [jnp.arange(1, product_rho**(2) + 1).reshape(product_rho, product_rho),jnp.arange(2, product_rho**(2) + 2).reshape(product_rho, product_rho)]
        options = Options(estimator=True, tensorisation=lazy_tensorisation, reshaping=False, trunc_size=[1 + i for i in range(len(lazy_tensorisation))]) # we fake a trunc_size
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        # print("H:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        # print("Hred:", res_init[1](0)[0], "\nmask:", res_init[3][0], "\ntensorisation:", res_init[4])
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

    results = [
        jnp.array_equal(expected_Hred_1D, res_init_1D[1](0)) , 
        jnp.array_equal(expected_mask_1D, res_init_1D[3]) , 
        jnp.array_equal(expected_tensorisation_1D, res_init_1D[4]) ,
        jnp.array_equal(expected_mask_2D, res_init_2D[3]) , 
        jnp.array_equal(expected_tensorisation_2D, res_init_2D[4])
    ]
    print(results)
    return all(results)



# === reshapings_y unit tests
def unit_test_reshaping_init():
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
        # print("H:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[4], tsave, atol
        )
        # print("ineq", resh_init[0].inequalities, "\ntrunc size:", 
        #     resh_init[0].trunc_size, "\nH_mod:", resh_init[1](0)[0], "\nHred_mod:", 
        #     resh_init[3](0)[0], "\nmask:", resh_init[6][0], "\ntensorisation:", 
        #     resh_init[7]
        # )
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
        # print("\nH:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[4], tsave, atol)
        n, _ = jnp.array(resh_init[1](0)).shape
        rho = jnp.arange(0,n**2).reshape(n,n) # on refait un rho pour voir l'effet du reshaping
        # print("rho", rho[0])
        # print("objects sizes after initial reshaping (+trunc_size(", options.trunc_size,
        #      ") to add)", resh_init[0].inequalities
        # )
        resh_ext = reshaping_extend(resh_init[0], H, Ls, rho, resh_init[7], tsave[0], 
            resh_init[8])
        # print("H_mod:", resh_ext[1](0)[0], "\nHred_mod:", resh_ext[3](0)[0], "\nmask:", 
        #     resh_ext[6][0], "\nrho", resh_ext[5][0], "\ntensorisation:", resh_ext[7], 
        #     "\nineq", resh_ext[0].inequalities
        # )
        return resh_ext
    
    def run_estimator_on_extension_2D(inequalities=None):
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
        options = Options(
            estimator=True, 
            tensorisation=lazy_tensorisation, 
            reshaping=True, 
            trunc_size=[2*x for x in degree_guesser_nD_list(H, Ls, lazy_tensorisation)],
            inequalities=inequalities
        )
        H = _astimearray(H)
        Ls = [_astimearray(L) for L in Ls]
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 10000 # to force reshaping
        resh_init = reshaping_init(
            options, H, Ls, res_init[1], res_init[2], res_init[3], rho0, res_init[4], 
            tsave, atol
        )
        n, _ = jnp.array(resh_init[1](0)).shape
        # print(
        #     "\nobjects sizes after initial reshaping (+trunc_size to add)", 
        #     resh_init[0].inequalities, "\ntrunc_size:", options.trunc_size, 
        #     "\ntensorisation init", resh_init[7]
        # )
        resh_ext = reshaping_extend(
            resh_init[0], H, Ls, resh_init[5], resh_init[7], tsave[0], resh_init[8]
        )
        # print("\nH_mod:", resh_ext[1](0)[0], "\nHred_mod:", resh_ext[3](0)[0], 
        #       "\nmask:", resh_ext[6][0], "\nrho", resh_ext[5][0], "\ntensorisation:", 
        #       resh_ext[7], "\nineq", resh_ext[0].inequalities
        #     )
        return compute_estimator(
            resh_ext[1], resh_ext[2], resh_ext[3], resh_ext[4], resh_ext[5], 0
        )
    
    # test 1D
    lazy_tensorisation_1D = [10]
    resh_ext_1D = run_extension(lazy_tensorisation_1D)
    # test 2D
    lazy_tensorisation_2D = [7,11]
    resh_ext_2D = run_extension(lazy_tensorisation_2D)
    # checking that extending => we get 0 for the estimator (bcs we extend by at least
    # trunc_size)
    estimator_simple = run_estimator_on_extension_2D()
    def ineq(a, b):
        return a+b
    inequalities = [[ineq, 7, 2, 2]] 
    estimator_diag_ineq = run_estimator_on_extension_2D(inequalities)
    print(estimator_diag_ineq)

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
        estimator_simple==0,
        # estimator_diag_ineq==0 we do not expect 0 ! because we do not extend by trunc
        # size but according to the inequalities
        estimator_diag_ineq==0.17384097
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
        # print("\nH:", H(0)[0])
        res_init = mesolve_estimator_init(options, H, Ls, tsave)
        atol = 1e-6
        resh_init = reshaping_init(options, H, Ls, res_init[1], res_init[2], 
            res_init[3], rho0, res_init[4], tsave, atol)
        n, _ = jnp.array(resh_init[1](0)).shape
        rho = jnp.arange(0,n**2).reshape(n,n) # on refait un rho pour voir l'effet du reshaping
        # print("rho", rho[0])
        # print("objects sizes after initial reshaping (+trunc_size(", options.trunc_size,
        #      ") to add)", resh_init[0].inequalities
        # )
        resh_red = reshapings_reduce(resh_init[0], H, Ls, rho, resh_init[7], tsave[0], 
            resh_init[8])
        # print("H_mod:", resh_red[1](0)[0], "\nHred_mod:", resh_red[3](0)[0], "\nmask:", 
        #     resh_red[6][0], "\nrho", resh_red[5][0], "\ntensorisation:", resh_red[7], 
        #     "\nineq", resh_red[0].inequalities
        # )
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



# === test all unit tests
def run_all_tests():
    tests = [
        unit_test_dicts,
        unit_test_mask,
        unit_test_projection_nD,
        unit_test_extended_tensorisation,
        unit_test_reduction_nD,
        unit_test_extension_nD,
        unit_test_degree_guesser_nD,
        unit_test_mesolve_estimator_init,
        unit_test_reshaping_init,
        unit_test_reshaping_extend,
        unit_test_error_reducing,
        unit_test_reshaping_reduce,
    ]
    all_passed = True
    str = ""
    for test in tests:
        print(f"\ntesting {test.__name__}\n")
        if not test():
            str += f"{test.__name__} failed\n"
            all_passed = False
    print(str)
    return all_passed
