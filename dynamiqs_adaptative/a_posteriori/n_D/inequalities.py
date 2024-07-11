from ...options import Options
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
             ineq_from_params(param, f, lazy_tensorisation) = lambda i, j: i+j <= 3 
    """
    lambda_func = lambda *args: f(*args) <= param
    return lambda_func
    # Get the number of arguments the function f takes
    # num_args = len(lazy_tensorisation)
    # Create argument names dynamically
    # arg_names = [f'i{i}' for i in range(num_args)]
    # lambda_str = f"lambda {', '.join(arg_names)}: f({', '.join(arg_names)}) < param"
    # # Create the lambda function
    # lambda_func = eval(lambda_str, {'f': f, 'param': param})

def ineq_from_params(listf, params):
    """
    Make a list of ineq_from_param(param, f).
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

def update_ineq(options, direction):
    """
    Update the current inequalities by changing the parameter.
    direction = 'up' if we wanna extend
    direction = 'down' if we wanna reduce
    """
    tmp_dic=options.__dict__
    new_ineq = options.inequalities[:]
    len_ineq = len(options.inequalities)
    if direction == 'up':
        ineq_params = [options.inequalities[i][1] + options.inequalities[i][2] for i in  
            range(len_ineq)
        ]
    else:
        ineq_params = [options.inequalities[i][1] - options.inequalities[i][3] for i in  
            range(len_ineq)
        ]
    for i in range(len_ineq):
        new_ineq[i][1] = ineq_params[i]
    tmp_dic['inequalities'] = new_ineq
    return Options(**tmp_dic)

def ineq_from_params_list(coeffs, R):
    return lambda *args: sum(coeff * arg for coeff, arg in zip(coeffs, args)) < R

