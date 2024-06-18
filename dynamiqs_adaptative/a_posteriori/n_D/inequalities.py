def ineq_from_params(param, f, lazy_tensorisation):
    """
    Make lambda inegalities from a certain set of parameters.

    Args:
    param: a float.
    f: a function that has len(param) inputs and outputs a float.
    lazy_tensorisation: the tensorisation used for rho.

    Returns:
    lambda_func: a lambda function

    Exemple: def f(i, j): return i+j
             param = 3
             lazy_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
             ineq_from_params(param, f, lazy_tensorisation) = lambda i, j: i+j < 3
    """
    # Get the number of arguments the function f takes
    num_args = len(lazy_tensorisation[0])
    # Create argument names dynamically
    arg_names = [f'i{i}' for i in range(num_args)]
    lambda_str = f"lambda {', '.join(arg_names)}: f({', '.join(arg_names)}) < param"
    # Create the lambda function
    lambda_func = eval(lambda_str, {'f': f, 'N': param})
    return lambda_func

def generate_rec_ineq(trunc):
    """
    Makes a list of lambda functions [lambda i0, ..., i({len(trunc)}): i{j} < trunc{j}].
    It generates a "rectangular" truncature in n dimensions. 
    """
    # Create simple comparison functions for each element in L
    lambda_list=[]
    num_args = len(trunc)
    arg_names = [f'i{i}' for i in range(num_args)]
    for j in range(num_args):
        lambda_str = f"lambda {', '.join(arg_names)}: i{j} < trunc[{j}]"
        lambda_func = eval(lambda_str, {'trunc': trunc})
        lambda_list.append(lambda_func)
    return lambda_list