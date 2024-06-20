import jax.numpy as jnp
import itertools
from .tensorisation_maker import tensorisation_maker
from .inequalities import generate_rec_ineqs
from ...options import Options
import math

def projection_nD(
        objs, original_tensorisation = None, inequalities = None, _mask = None
):
    """
    create a tensorial projection of some n dimensional matrix "objs" tensorised under 
    "original_tensorisation"  into matrixs "new_objs" projected according to
    some inequalities "inequalities"
    as an example : original_tensorisation can have this 
    ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)) and if applied inequalities 
    [lambda i, j: i <= 2 , lambda i, j: j <= 2]
    on such a matrix matrix_6x6 = np.arange(1, 37).reshape(6, 6)
    the output would be : 
    matrix_6x6 = np.array([[ 1,  2,  0,  4,  5,  0],
                           [ 7,  8,  0, 10, 11,  0],
                           [ 0,  0,  0,  0,  0,  0],
                           [19, 20,  0, 22, 23,  0],
                           [25, 26,  0, 28, 29,  0],
                           [ 0,  0,  0,  0,  0,  0]])
    all objs would undergo the related projection, making zeros the (0,2) and (1,2)
    cols and rows of the objs.
    (the size of the objs would not be changed, but there would be zeros on
    projected lines and columns)
    """

    new_objs = objs
    # if _mask is already known we don't calculate it
    if _mask is None:
        if original_tensorisation is None or inequalities is None:
            raise ValueError(" You have to either give a mask or a tensorisation with some inequalities")
        dictio = dict_nD(original_tensorisation, inequalities)
        _mask = mask(new_objs[0], dictio)
    for i in range(len(new_objs)):
        new_objs[i] = jnp.where(_mask, new_objs[i], 0)

    return new_objs

def reduction_nD(objs, tensorisation, inequalities):
    """
    same as projection_nD but delete lines instead of putting zeros.
    """
    new_objs = []
    # Sort positions in descending order to avoid shifting issues
    dictio = sorted(dict_nD(tensorisation, inequalities), reverse=True)
    for i in range(len(objs)):
        if i ==0:
            tensorisation = recursive_delete(tensorisation, dictio, tens=True)
        new_objs.append(recursive_delete(objs[i], dictio))

    return new_objs, tensorisation

def extension_nD(
        objs, old_tensorisation, max_tensorisation, inequalities, options, bypass = False
):
    """
    Extend the objs up to the max_tensorisation by putting zeros cols and rows in spots
    where the inequalities are satisfied and there is no rows and cols.

    Args:
    objs: list of input matrices (JAX arrays)
    old_inequalities: the current tensorisation of objs is made after those inequalities
    max_tensorisation: maximum tensorisation reachable given the max size of the 
                       matrices H and L given as inputs of the mesolve.
    inequalities: list of functions defining the inequalities on the tensorisation.
                  Each function should take a tuple (i, j, ..., n) as input and return a 
                  boolean.
    bypass: extends directly to the max_tensorisation if True

    Returns:
    objs: list of matrices with zeros placed.
    """
    # old_tensorisation = ineq_to_tensorisation(old_inequalities, max_tensorisation)
    lenobjs = len(objs)
    # Iterate over all possible indices within max_tensorisation 
    # (recreates a lazy_tensorisation)
    indices = [range(max_dim) for max_dim in max_tensorisation]
    lentensor = len(max_tensorisation)
    index = 0
    tensorisation = []
    new_objs = objs # to not modify initial object
    for tensor in itertools.product(*indices):
        # Check if the conditions are satisfied
        if all(ineq(*tensor) for ineq in inequalities) or bypass: 
            for i in range(lentensor):
                if max_tensorisation[i]==(tensor[i] + options.trunc_size[i]):
                    # we reach the max size of the matrices given as inputs
                    print('WARNING the size of the objects you gave is too small to'
                           'warranty a solution accurate up to solver precision')
            if list(tensor) not in old_tensorisation:  # ajoute du * n a la complexité, peut etre a virer
                for i in range(lenobjs):
                    # Extend the matrix by adding zero rows and columns
                    new_objs[i] = add_zeros_line(new_objs[i], index)
            tensorisation.append(list(tensor))
        index += 1
    # sort in the good order the new tensorisation
    # tensorisation = sorted(old_tensorisation, key=lambda x: list(reversed(x)))
    # print(old_tensorisation, tensorisation)
    return new_objs, tensorisation

def dict_nD(original_tensorisation, inequalities):
    # dictio will make us able to repertoriate the indices to suppress
    dictio=[]
    # i compte l'indice sur lequel on se trouve pour l'encodage en machine des positions
    # de la matrice, par exemple (i=0 pour (0,0)) et (i=2 pour (1,0)) dans l'exemple
    # precedent
    for i in range(len(original_tensorisation)):
        # we check if some inequalities aren't verified, if so we will add the line
        # in dictio for it to be cut off later
        for inegality in inequalities:
            if not inegality(*tuple(original_tensorisation[i])):
                dictio.append(i)
                break # to avoid appending the same line multiple times

    return dictio

def mask(obj, dict):
    """
    Put zeros on the line and column of a matrix "obj" by reference to a position "pos".
    Zeros are placed in a contiguous streak.

    Args:
    obj: input matrix.
    pos: position where zeros should be placed.

    Returns:
    obj: matrix with zeros placed in a contiguous streak.
    """
    # Create a mask for the positions to zero out
    mask = jnp.ones_like(obj, dtype=bool)
    for x in dict:
        mask = jnp.where(jnp.arange(obj.shape[0])[:, None] == x, False, mask)  # Zero out row
        mask = jnp.where(jnp.arange(obj.shape[1])[None, :] == x, False, mask)  # Zero out column

    return mask

def add_zeros_line(matrix, k):
    """
    Add a zero row and a zero column at index k to a given matrix.

    Args:
    matrix: input matrix (JAX array)
    k: index where zero row and column should be inserted

    Returns:
    modified_matrix: matrix with zero row and column added
    """
    # Adding zero row
    top_part = matrix[:k, :]
    bottom_part = matrix[k:, :]
    zero_line = jnp.zeros((1, matrix.shape[1]))
    matrix_with_zero_row = jnp.concatenate([top_part, zero_line, bottom_part], axis=0)
    
    # Adding zero column
    left_part = matrix_with_zero_row[:, :k]
    right_part = matrix_with_zero_row[:, k:]
    zero_column = jnp.zeros((matrix_with_zero_row.shape[0], 1))
    modified_matrix = jnp.concatenate([left_part, zero_column, right_part], axis=1)
    
    return modified_matrix

def add_zeros_line_old_insert(matrix, k):
    """
    Add a zero row or column at index k to a given matrix.

    Args:
    matrix: input matrix (JAX array)
    k: index where zero row or column should be inserted

    Returns:
    modified_matrix: matrix with zero row or column added
    """
    _, m = matrix.shape
    zero_line = jnp.zeros((1, m))
    zero_col = jnp.zeros((1, m+1)) # +1 because we just added a line
    modified_matrix = jnp.insert(matrix, k, zero_line, axis=0)
    modified_matrix = jnp.insert(modified_matrix, k, zero_col, axis=1)
    return modified_matrix

def add_zeros_line_jaxed(matrix, k):
    """
    Add a zero row and a zero column at index k to a given matrix.

    Args:
    matrix: input matrix (JAX array)
    k: index where zero row and column should be inserted

    Returns:
    modified_matrix: matrix with zero row and column added
    """
    num_rows, num_cols = matrix.shape
    
    # Create the new matrix with an extra row and column
    new_matrix = jnp.zeros((num_rows + 1, num_cols + 1))
    
    # Fill the parts before and after the insertion point
    new_matrix = new_matrix.at[:k, :k].set(matrix[:k, :k])
    new_matrix = new_matrix.at[:k, k+1:].set(matrix[:k, k:])
    new_matrix = new_matrix.at[k+1:, :k].set(matrix[k:, :k])
    new_matrix = new_matrix.at[k+1:, k+1:].set(matrix[k:, k:])
    
    return new_matrix

def zeros_old(obj, pos):
    # Put zeros on the line and column of a matrix "obj" by reference to a position "pos"
    
    zeros = jnp.zeros(obj.shape[0], dtype=obj.dtype)
    obj = obj.at[pos, :].set(zeros)
    obj = obj.at[:, pos].set(zeros)
    
    return obj

def delete(obj, pos, tens):
    if tens:
        # if we act on the tensorisation , delete the tensors that will be deleted in
        # the matrices concerned by reduction_nd
        del obj[pos]
        return obj
    else:
        # delete the line and col of a matrix "obj" by reference to a position "pos"
        return jnp.delete(jnp.delete(obj, pos, axis=0), pos, axis=1)

def recursive_delete(obj, positions, tens=False):
    if not positions:
        return obj
    # Delete the first position in the list and call recursively for the rest
    return recursive_delete(delete(obj, positions[0], tens), positions[1:], tens)

def ineq_to_tensorisation(old_inequalities, max_tensorisation):
    """
    Transform the old inequalities into an old tensorisation which knowledge is needed
    to compute the n cavities extension.
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


def unit_test_projection_nD():
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    objs = [jnp.arange(1, 37).reshape(6, 6)]
    objs = projection_nD(objs, original_tensorisation, inequalities)
    
    expected_result =   [jnp.array([[1, 2, 0, 4, 5, 0],
                                    [7, 8, 0, 10, 11, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [19, 20, 0, 22, 23, 0],
                                    [25, 26, 0, 28, 29, 0],
                                    [0, 0, 0, 0, 0, 0]])]
                                    
    return jnp.array_equal(objs, expected_result)

def unit_test_reduction_nD():
    def prod(lst):
        product = 1
        for num in lst:
            product *= num
        return product
    lazy_tensorisation = [7,5]
    tensorisation = tensorisation_maker(lazy_tensorisation)
    # print(tensorisation)
    inequalities = generate_rec_ineqs(
        [x//2 - 1 for x in lazy_tensorisation]
    )
    print("square ineq: ", [x//2 - 1 for x in lazy_tensorisation])
    # +1 to account for list starting at zero
    product = prod([a + 1 for a in lazy_tensorisation])
    objs = [jnp.arange(0, product**2).reshape(product, product)]
    temp = reduction_nD(objs, tensorisation, inequalities)
    print("reduction: ", temp)    
    expected_result =   [jnp.array([[  0,   1,   6,   7,  12,  13],
                                    [ 48,  49,  54,  55,  60,  61],
                                    [288, 289, 294, 295, 300, 301],
                                    [336, 337, 342, 343, 348, 349],
                                    [576, 577, 582, 583, 588, 589],
                                    [624, 625, 630, 631, 636, 637]])]
    expected_tensorsiation = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
                                    
    if (
        jnp.array_equal(temp[0], expected_result) and 
        jnp.array_equal(temp[1], expected_tensorsiation)):
        print("working")
        return True
    else:
        print("not working")
        return False

def unit_test_extension_nD():
    def prod(lst):
        product = 1
        for num in lst:
            product *= num
        return product
    lazy_tensorisation = [1,4]
    max_lazy_tensorisation = [100,100]
    old_tensorisation = tensorisation_maker(lazy_tensorisation)
    options = Options(trunc_size=[4,2])
    inequalities = generate_rec_ineqs(
        [x + options.trunc_size[i] for x, i in zip(lazy_tensorisation, range(len(lazy_tensorisation)))]
    )
    print("square ineq: ", [x + options.trunc_size[i] for x, i in zip(lazy_tensorisation, range(len(lazy_tensorisation)))])
    product = prod() # +1 to account for list starting at zero
    objs = [jnp.arange(0, product**2).reshape(product, product)]
    print("obj initial: ", objs[0])
    temp = extension_nD(objs, old_tensorisation, max_lazy_tensorisation, inequalities, options)
    print("extension: ", temp)  
    # np.savetxt('rere.csv', np.array(temp[0])[0])
    expected_result = None # I need to find a way to get it in a hard coded way
    expected_size = prod([a + 1 + b for a, b in zip(lazy_tensorisation, options.trunc_size)])
    real_size = math.sqrt(jnp.array(temp[0]).size)
    expected_tensorsiation = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6]]
    print("""WARNING checks only tensorisation and extensions size""")                             
    return (jnp.array_equal(temp[1], expected_tensorsiation) and float(len(expected_tensorsiation))==real_size)

