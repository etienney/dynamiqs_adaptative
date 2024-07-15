import jax.numpy as jnp

def ineq_from_param(f, param):
    """
    Make lambda inegalities from a certain set of parameters.

    Args:
    param: a float.
    f: a function that has len("number of dimensions") inputs and outputs a float.

    Returns:
    lambda_func: a lambda function

    Exemple: def f(i, j): return i+j
             param = 3
             ineq_from_params(f, param) = lambda i, j: i+j <= 3 
    """
    lambda_func = lambda *args: f(*args) <= param
    return lambda_func

def ineq_from_params(listf, params):
    """
    Make a list of ineq_from_param(f, param).
    """
    ineq = []
    num_args = len(params)
    for j in range(num_args):
        ineq.append(ineq_from_param(listf[j], params[j]))
    return ineq

def generate_rec_ineqs(trunc):
    """
    Makes a list of lambda functions [lambda i0, ..., i({len(trunc)}): i{j} < trunc{j}].
    It generates a "rectangular" truncature in n dimensions. 
    """
    lambda_list = []
    num_args = len(trunc)
    listf = [generate_rec_func(j) for j in range(num_args)]
    # Generate inequalities based on trunc values and generated functions
    for j in range(num_args):
        lambda_list.append(ineq_from_param(listf[j], trunc[j]))
    return lambda_list

def generate_rec_func(j):
    """
    Creates a function with `n` arguments that returns the `j-th` argument.
    """
    return lambda *args: args[j]