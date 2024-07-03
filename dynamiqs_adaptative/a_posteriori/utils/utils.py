import jax.numpy as jnp
import numpy as np
import jax
import itertools

def to_hashable(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if obj.ndim == 0:  # Handle scalar case
            return obj.item()
        else:
            return tuple(to_hashable(sub_obj) for sub_obj in obj)
    elif isinstance(obj, (list, tuple)):
        return tuple(to_hashable(sub_obj) for sub_obj in obj)
    else:
        return obj

def tuple_to_list(t):
    if isinstance(t, tuple):
        return [tuple_to_list(item) for item in t]
    else:
        return t
    
def find_approx_index(lst, value):
    # find the position in a list containing something close to value of this something
    return min(range(len(lst)), key=lambda i: abs(lst[i] - value))

def new_ts(t0, ts):
    def cond_fun(state):
        return jnp.all(ts[state] < t0)
    def body_fun(state):
        return state + 1

    init_state=0
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    new_ts = jnp.zeros(ts[-1] - final_state, jnp.float32)
    new_ts=new_ts.at[:].set(ts[final_state:])
    return new_ts

def prod(lst):
    # make the product of the number in a list
    product = 1
    for num in lst:
        product *= num
    return product

def find_non_infs_matrices(matrices):
    # output matrices not full of infs in a list of matrices ending with only matrices 
    # full of infs
    filtered_matrices = []
    
    for matrix in matrices:
        if not jnp.isfinite(matrix).all():
            break  # Stop iteration as soon as we encounter a matrix with 'inf' values
        filtered_matrices.append(matrix)
    
    return filtered_matrices

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

def reverse_indices(L, x):
    """
    Put the indices not in L up to x.
    reverse_indices([5, 6, 8, 9, 15, 25], 27) = 
    [0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]
    """
    L_set = set(L)  # Convert list L to a set for O(1) membership testing
    reversed_list = [i for i in range(x + 1) if i not in L_set]
    return reversed_list

def ineq_to_tensorisation(old_inequalities, max_tensorisation):
    """
    Transform the inequalities into a tensorisation up to max_tensorisation
    """
    old_tensorisation = []
    # Iterate over all possible indices within max_tensorisation
    indices = [range(max_dim) for max_dim in max_tensorisation]
    index = 0
    for tensor in itertools.product(*indices):
        # Check if the conditions are satisfied
        if all(ineq(*tensor) for ineq in old_inequalities):
            old_tensorisation.append(tensor)
    return old_tensorisation

def put_together_results(L, k, estimator = False):
    """
    The reshaping outputs some [0, 0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, ...,  1.0] for a 
    jnp.linespace(0,1.0,11) for the time for instance.
    We would like to put together the lists (here in L) and cut the redundant parts,
    and they are k of them per end of lists (apart from the last) (here k = 2)
    the output should be the [0, 0.1, 0.2, 0.3, 0.4, 0.5, ...,  1.0] if estimator
    and [0, 0.1, 0.2] [0.3, ..., 1.0]  if not estimator (concatenate won't work for
    list of matrices)
    """
    Lnew = []
    for i in range(len(L)-1):
        if estimator: Lnew.append(L[i][:-k])
        else: Lnew.extend(L[i][:-k])
    # the last list should be taken as a whole
    if estimator: Lnew.append(L[-1])
    else: Lnew.extend(L[-1])
    if estimator: Lnew = jnp.concatenate(Lnew)
    return Lnew
