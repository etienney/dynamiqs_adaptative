import itertools

def ineq_to_tensorisation(inequalities, tensorisation):
    """
    Transform the old tensorisation into a new one respecting the inequalities.
    """
    new_tensorisation = []
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


def reverse_indices(L, x):
    """
    Put the indices not in L up to x.
    reverse_indices([5, 6, 8, 9, 15, 25], 27) = 
    [0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]
    """
    L_set = set(L)
    reversed_list = [i for i in range(x + 1) if i not in L_set]
    return reversed_list


def split_contiguous_indices(indices):
    """
    split_contiguous_indices([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18])
    = [(0, 4), (7, 11), (14, 18)]
    """
    ranges = []
    start = indices[0]
    end = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != end + 1:
            ranges.append((start, end))
            start = indices[i]
        end = indices[i]
    ranges.append((start, end))
    return ranges


def shift_ranges(ranges):
    """
    shift_ranges([(0, 4), (7, 11), (14, 22)]) = [(0, 4), (5, 9), (10, 18)]
    """
    new_ranges = []
    start = 0
    for old_start, old_end in ranges:
        length = old_end - old_start
        new_end = start + length
        new_ranges.append((start, new_end))
        start = new_end + 1
    return new_ranges

