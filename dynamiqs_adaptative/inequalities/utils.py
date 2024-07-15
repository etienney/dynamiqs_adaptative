import itertools

def ineq_to_tensorisation(inequalities, tensorisation):
    """
    Transform the inequalities into a tensorisation up to max_tensorisation
    """
    new_tensorisation = []
    tensorisation = list(
        itertools.product(*[range(max_dim) for max_dim in tensorisation])
    )
    for tensor in tensorisation:
        # Check if the conditions are satisfied
        if all(ineq(*tensor) for ineq in inequalities):
            new_tensorisation.append(tensor)
    return new_tensorisation

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
             ineq_from_params(param, f, lazy_tensorisation) = lambda i, j: i+j <= 3 
    """
    lambda_func = lambda *args: f(*args) <= param
    return lambda_func

def ineq_from_params(listf, params):
    """
    Make a list of ineq_from_param(param, f).
    """
    ineq = []
    num_args = len(params)
    for j in range(num_args):
        ineq.append(ineq_from_param(listf[j], params[j]))
    return ineq

def indices_cut_by_ineqs(inequalities, previous_tensorisation):
    # dictio will make us able to repertoriate the indices concerned by the inequalities
    dictio=[]
    for i in range(len(previous_tensorisation)):
        # we check if some inequalities aren't verified, if so we will add the line
        # in dictio 
        for inegality in inequalities:
            if not inegality(*tuple(previous_tensorisation[i])):
                dictio.append(i)
                break # to avoid appending the same line multiple times
    return dictio
