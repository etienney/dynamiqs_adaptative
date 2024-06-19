def ineq_from_params(param, f):
    """
    Make lambda inegalities from a certain set of parameters.

    Args:
    param: a float.
    f: a function that has len("number of dimensions") inputs and outputs a float.

    Returns:
    lambda_func: a lambda function

    Exemple: def f(i, j): return i+j
             param = 3
             ineq_from_params(param, f, lazy_tensorisation) = lambda i, j: i+j < 3
    """
    lambda_func = lambda *args: f(*args) < param
    return lambda_func
    # Get the number of arguments the function f takes
    # num_args = len(lazy_tensorisation)
    # Create argument names dynamically
    # arg_names = [f'i{i}' for i in range(num_args)]
    # lambda_str = f"lambda {', '.join(arg_names)}: f({', '.join(arg_names)}) < param"
    # # Create the lambda function
    # lambda_func = eval(lambda_str, {'f': f, 'param': param})

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
        lambda_list.append(ineq_from_params(trunc[j], listf[j]))
    return lambda_list
    # lambda_list=[]
    # num_args = len(trunc)
    # arg_names = [f'i{i}' for i in range(num_args)]
    # listf = [generate_rec_func(j, trunc) for j in range(num_args)]
    # print(listf, trunc)
    # # trunc having the same length as lazy_tensorisation
    # return [ineq_from_params(trunc[j], listf[j], trunc) for j in range(num_args)]

def generate_rec_func(j):
    """
    Creates a function with `n` arguments that returns the `j-th` argument.
    """
    return lambda *args: args[j]
    # n = len(lazy_tensorisation)
    # # Create the function definition as a string
    # func_def = f"def generated_func({', '.join([f'i{i}' for i in range(n)])}):\n"
    # func_def += f"    return i{j}\n"
    # # Create a dictionary to hold the local variables of the exec call
    # local_vars = {}
    # # Execute the function definition string in the context of local_vars
    # exec(func_def, {}, local_vars)
    # # Return the generated function from the local_vars dictionary
    # return local_vars['generated_func']
