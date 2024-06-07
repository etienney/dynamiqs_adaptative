from ...utils.operators import create
from ...utils.utils.general import dag
from jax.lax import while_loop
import jax
from ..._utils import cdtype
from functools import partial
import jax.numpy as jnp
import numpy as np

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
    taille=np.size(H[0])
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
    a=create(6)
    id=jnp.identity(6)
    if degree_guesser(
        a@a@a@a+dag(a)+(a@a+id+dag(a))@(id+dag(a)@dag(a)@dag(a)+id+a@a@a@a)
    )!=(4,4) or degree_guesser(a@a@dag(a)+id)!=(1,0):
        print( degree_guesser(a@a@dag(a)+id))
        return False
    else:
        return True
    


# print(unit_test_degree_guesser())

