from functools import reduce
from .._utils import cdtype
from jax.lax import while_loop
import jax
import jax.numpy as jnp
import itertools

def degree_guesser_nD_list(L, tensorisation):
    # designed for H and jump_ops specifically. Takes the max in the modes amongst all
    k=[]
    for x in L:             
        k = k + [degree_guesser_nD(x, tensorisation),]
    # we take the max amongst all, it's not optimal for computation time, but it's okay
    k = [max(x) for x in zip(*k)]
    return k

def degree_guesser_nD(M, tensorisation):
    # allows to have the max degree in the different modes for n modes.
    # for a@a@b + a@b it outputs [2,1] for instance
    input_list = []
    # start with all degree at least being zero
    for i in range(0, len(tensorisation)):
        input_list += [[i,0]]
    input_list += degree_guesser_nD_rec(M, 0, tensorisation)
    # sort the list by the first column
    sorted_list = sorted(input_list, key=lambda x: x[0])
    # group the list by the first column using itertools.groupby()
    groups = itertools.groupby(sorted_list, key=lambda x: x[0])
    # find the maximum value in the second column for each group
    max_values = [(k, max(g, key=lambda x: x[1])[1]) for k, g in groups]
    # get rid of the positional index
    second_elements = [sublist[1] for sublist in max_values]

    return second_elements

def degree_guesser_nD_rec(M, deg, tensorisation):
    # For each submatrix of the tensorisation, search the max degree attained by 
    # checking the last column and lines where the subsubmatrices are non identical
    # to zeros. It's done recursively.
    # For the last submatrix, it just need to apply the degree_guesser for 1 dimension
    # (1 mode)
    if deg +1 == len(tensorisation):
        yield treatment(deg,M)
    else:
        prod = product(tensorisation[deg+1:])
        zeros = jnp.zeros((prod,prod), cdtype())
        for i in range(tensorisation[deg]):
            reduced_M_lin = M[
                prod * (tensorisation[deg] - 1):prod * (tensorisation[deg]),
                prod * (tensorisation[deg] - i - 1):prod * (tensorisation[deg] - i)
            ]
            reduced_M_col = M[
                prod * (tensorisation[deg] - i - 1): prod * (tensorisation[deg] - i),
                prod * (tensorisation[deg] - 1): prod * (tensorisation[deg])
            ]
            if not jnp.array_equal(zeros, reduced_M_lin):
                yield [deg, i]
                yield from degree_guesser_nD_rec(reduced_M_lin, deg + 1, tensorisation)
            if not jnp.array_equal(zeros, reduced_M_col):
                yield [deg, i]
                yield from degree_guesser_nD_rec(reduced_M_col, deg + 1, tensorisation)
                
def treatment(deg,M):
    # just to return the result of degree_guesser in the good format
    return list([deg, jnp.max(jnp.array(degree_guesser(M)))])

def product(list):
    # make the product of the number in the list
    return reduce(lambda x, y: x * y, list)

def unit_test_degree_guesser_nD():
    from ..utils.operators import destroy
    from ..utils.utils.general import dag, tensor
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

def degree_guesser_list(H, L):
    # guess the maximum degree to check in a and adag for H and jumps operators
    k = (degree_guesser(H),)
    for x in L:             
        k = k + (degree_guesser(x),)
    # we take the max amongst all, it's not optimal for computation time, but it's okay
    k = max(tuple([max(x) for x in zip(*k)]))
    return k

def degree_guesser(H):
    # guess the degree of a matrix issued from additions 
    # and products of polynomials in a and adag
    taille=jnp.size(H[0])
    dega=taille
    degadag=taille
    cond_fun_a = lambda x: jax.lax.cond(
            x>1, lambda y : ((H[taille][taille-y]==jnp.zeros(1, cdtype())).all()), 
            lambda y : False, x
    )
    cond_fun_b = lambda x: jax.lax.cond(
            x>1, lambda y : ((H[taille-y][taille]==jnp.zeros(1, cdtype())).all()), 
            lambda y : False, x
    )
    body_fun_minus_1 = lambda x: x-1
    dega = while_loop(cond_fun_a, body_fun_minus_1, dega)
    degadag = while_loop(cond_fun_b, body_fun_minus_1, degadag)   
    return dega-1,degadag-1

def unit_test_degree_guesser():
    from ..utils.operators import create
    from ..utils.utils.general import dag
    a=create(6)
    id=jnp.identity(6)
    if degree_guesser(
        a@a@a@a+dag(a)+(a@a+id+dag(a))@(id+dag(a)@dag(a)@dag(a)+id+a@a@a@a)
    )!=(4,4) or degree_guesser(a@a@dag(a)+id)!=(1,0):
        print( degree_guesser(a@a@dag(a)+id))
        return False
    else:
        return True