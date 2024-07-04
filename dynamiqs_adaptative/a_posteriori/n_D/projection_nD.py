import jax.numpy as jnp
import itertools
from .tensorisation_maker import tensorisation_maker
from .inequalities import generate_rec_ineqs
from ...options import Options
import math
from ..utils.utils import prod, split_contiguous_indices, shift_ranges, ineq_to_tensorisation, reverse_indices
from ..._utils import cdtype

def projection_nD(
    objs, original_tensorisation = None, inequalities = None, options=None, _mask = None
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
        dictio = dict_nD(original_tensorisation, inequalities, options)
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
            tensorisation = delete_tensor_elements(tensorisation, dictio)
        new_objs.append(delete_matrix_elements(objs[i], dictio))

    return new_objs, tensorisation

def extension_nD(
        objs, options
):
    """
    Extend the objs up to the max_tensorisation by putting zeros cols and rows in spots
    where the inequalities are satisfied and there is no rows and cols.
    Extend by options.trunc_size

    Args:
    objs: list of input matrices (JAX arrays)
    options: see options[lien]

    Returns:
    objs: list of matrices with zeros placed.
    """
    # lenobjs = len(objs)
    max_tensorisation = options.tensorisation
    # indices = [range(max_dim) for max_dim in max_tensorisation]
    # inequalities = generate_rec_ineqs([x[1] for x in options.inequalities])
    full_ineq_params = [x[1] + b for x, b in zip(options.inequalities, options.trunc_size)]
    extended_ineq_params = [x[1] + 2 * b for x, b in zip(options.inequalities, options.trunc_size)]
    full_inequalities = generate_rec_ineqs(full_ineq_params)
    extended_inequalities = generate_rec_ineqs(extended_ineq_params)
    # print("ineq:", full_ineq_params, "extended ineq:", extended_ineq_params)
    ineq_to_tensors = ineq_to_tensorisation(extended_inequalities, max_tensorisation)
    # print("tensorisation:", ineq_to_tensors)
    len_new_objs = len(ineq_to_tensors)
    reverse_dictio = reverse_indices(dict_nD(ineq_to_tensors, full_inequalities), len_new_objs - 1)
    # print("reverse dictio:", reverse_dictio)
    new_positions = split_contiguous_indices(reverse_dictio)
    old_positions = shift_ranges(new_positions)
    # print("old positions:",old_positions, "new positions:", new_positions)
    len_pos = len(old_positions)
    new_objs = []
    for obj in objs:
        new_obj = jnp.zeros((len_new_objs, len_new_objs), cdtype())
        for i in range(len_pos):
            for j in range(len_pos):
                new_obj = new_obj.at[new_positions[i][0]:new_positions[i][1]+1, 
                    new_positions[j][0]:new_positions[j][1]+1].set(
                    obj[old_positions[i][0]:old_positions[i][1]+1, 
                    old_positions[j][0]:old_positions[j][1]+1]
                )
        new_objs.append(new_obj)
    return new_objs, ineq_to_tensors

def extension_nD_old(
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
    # (recreates a tensorsiation_maker(lazy_tensorisation))
    indices = [range(max_dim) for max_dim in max_tensorisation]
    lentensor = len(max_tensorisation)
    index = 0
    tensorisation = []
    new_objs = objs # to not modify initial object
    for tensor in itertools.product(*indices):
        # Check if the conditions are satisfied
        if all(ineq(*tensor) for ineq in inequalities) or bypass: 
            for i in range(lentensor):
                if max_tensorisation[i]<(tensor[i] + options.trunc_size[i]):
                    # we reach the max size of the matrices given as inputs
                    print('WARNING the size of the objects you gave is too small to'
                           'warranty a solution accurate up to solver precision')
            if list(tensor) not in old_tensorisation:  # ajoute du * n a la complexité, peut etre a virer
                for i in range(lenobjs):
                    # Extend the matrix by adding zero rows and columns
                    new_objs[i] = add_zeros_line(new_objs[i], index)
                    # print(index)
            tensorisation.append(list(tensor))
        index += 1
    # sort in the good order the new tensorisation
    # tensorisation = sorted(old_tensorisation, key=lambda x: list(reversed(x)))
    # print(old_tensorisation, tensorisation)
    return new_objs, tensorisation

def dict_nD(tensorisation, inequalities, options=None):
    # dictio will make us able to repertoriate the indices to suppress
    dictio=[]
    # i compte l'indice sur lequel on se trouve pour l'encodage en machine des positions
    # de la matrice, par exemple (i=0 pour (0,0)) et (i=2 pour (1,0)) dans l'exemple
    # precedent
    for i in range(len(tensorisation)):
        # we check if some inequalities aren't verified, if so we will add the line
        # in dictio for it to be cut off later
        for inegality in inequalities:
            if options is not None: # not a very optimized thing...
                if any(tensorisation[i][j] > options.tensorisation[j] - 1 - 
                    options.trunc_size[j] for j in range(len(options.tensorisation))
                ):
                    dictio.append(i)
                    break
            if not inegality(*tuple(tensorisation[i])):
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

def delete_tensor_elements(obj, positions):
    # Sort positions in reverse order to avoid reindexing issues during deletion
    for pos in sorted(positions, reverse=True):
        del obj[pos]
    return obj

def delete_matrix_elements(obj, positions):
    # Convert positions to a numpy array for indexing
    positions = jnp.array(positions)
    
    # Create masks to keep track of rows and columns to keep
    mask = jnp.ones(obj.shape[0], dtype=bool)
    if positions.size!=0:
        mask = mask.at[positions].set(False)
    
    # Apply masks to delete rows and columns
    obj = obj[mask][:, mask]
    return obj




def unit_test_projection_nD():
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    objs = [jnp.arange(1, 37).reshape(6, 6)]
    options = Options(tensorisation=[100,100], trunc_size=[4,2])
    res = projection_nD(objs, original_tensorisation, inequalities, options)
    expected_result =   [jnp.array([
        [1, 2, 0, 4, 5, 0],
        [7, 8, 0, 10, 11, 0],
        [0, 0, 0, 0, 0, 0], 
        [19, 20, 0, 22, 23, 0],
        [25, 26, 0, 28, 29, 0],
        [0, 0, 0, 0, 0, 0]])]
    options2 = Options(tensorisation=[2,3], trunc_size=[1,1])
    objs2 = [jnp.arange(1, 37).reshape(6, 6)]# if not put res is also modified next line
    res2 = projection_nD(objs2, original_tensorisation, inequalities, options2)
    expected_result2 = [jnp.array([
       [1, 2, 0, 0, 0, 0],
       [7, 8, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])]
    print(res2)
    return (jnp.array_equal(res, expected_result) and
            jnp.array_equal(res2, expected_result2))

def unit_test_reduction_nD():
    lazy_tensorisation = [7,5]
    tensorisation = tensorisation_maker(lazy_tensorisation)
    # print(tensorisation)
    inequalities = generate_rec_ineqs(
        [(x+1)//2 - 1 for x in lazy_tensorisation]
    ) # /!\ +1 since lazy_tensorisation start counting at 0
    print("square ineq: ", [(x+1)//2 - 1 for x in lazy_tensorisation])
    product = prod([a for a in lazy_tensorisation])
    objs = [jnp.arange(0, product**(2)).reshape(product, product)]
    print("objs :", objs[0])
    temp = reduction_nD(objs, tensorisation, inequalities)
    print("reduction: ", temp)    
    expected_result =   [jnp.array([[  0,   1,   2,   5,   6,   7,  10,  11,  12,  15,  16,  17],
                                    [ 35,  36,  37,  40,  41,  42,  45,  46,  47,  50,  51,  52],
                                    [ 70,  71,  72,  75,  76,  77,  80,  81,  82,  85,  86,  87],
                                    [175, 176, 177, 180, 181, 182, 185, 186, 187, 190, 191, 192],
                                    [210, 211, 212, 215, 216, 217, 220, 221, 222, 225, 226, 227],
                                    [245, 246, 247, 250, 251, 252, 255, 256, 257, 260, 261, 262],
                                    [350, 351, 352, 355, 356, 357, 360, 361, 362, 365, 366, 367],
                                    [385, 386, 387, 390, 391, 392, 395, 396, 397, 400, 401, 402],
                                    [420, 421, 422, 425, 426, 427, 430, 431, 432, 435, 436, 437],
                                    [525, 526, 527, 530, 531, 532, 535, 536, 537, 540, 541, 542],
                                    [560, 561, 562, 565, 566, 567, 570, 571, 572, 575, 576, 577],
                                    [595, 596, 597, 600, 601, 602, 605, 606, 607, 610, 611, 612]])]
    expected_tensorsiation = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]]
                                    
    if (
        jnp.array_equal(temp[0], expected_result) and 
        jnp.array_equal(temp[1], expected_tensorsiation)):
        print("working")
        return True
    else:
        print("not working")
        return False

def unit_test_extension_nD_old():
    lazy_tensorisation = [2,5]
    lazy_tensorisation = [6,4]
    max_lazy_tensorisation = [100,100]
    old_tensorisation = tensorisation_maker(lazy_tensorisation)
    options = Options(trunc_size=[4,2])
    inequalities = generate_rec_ineqs(
        [x + options.trunc_size[i] for x, i in zip(lazy_tensorisation, range(len(lazy_tensorisation)))]
    )# /!\ +1 since lazy_tensorisation start counting at 0
    print("square ineq: ", [x + options.trunc_size[i] for x, i in zip(lazy_tensorisation, range(len(lazy_tensorisation)))])
    product = prod([a for a in lazy_tensorisation])
    objs = [jnp.arange(0, product**2).reshape(product, product)]
    print("obj initial: ", objs[0])
    temp = extension_nD_old(objs, old_tensorisation, max_lazy_tensorisation, inequalities, options)
    # print("extension: ", temp) 
    print("extension first line:", temp[0][0][0]) 
    print("tens:", temp[1])
    # np.savetxt('rere.csv', np.array(temp[0])[0])
    expected_result = None # I need to find a way to get it in a hard coded way
    real_size = math.sqrt(jnp.array(temp[0]).size)
    expected_tensorsiation = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8]]
    print("""WARNING checks only tensorisation and extensions size""")                             
    return (jnp.array_equal(temp[1], expected_tensorsiation) and float(len(expected_tensorsiation))==real_size)

def unit_test_extension_nD():
    def run_test(lazy_tensorisation, max_lazy_tensorisation, trunc_size):
        ineqs_params = [[None, x - 1 - trunc_size[i]] for x, i in zip(lazy_tensorisation, range(len(lazy_tensorisation)))] # -1 because tensorisation starts at 0
        print("square ineq: ", ineqs_params)
        options = Options(tensorisation=max_lazy_tensorisation, trunc_size=trunc_size, inequalities=ineqs_params)
        product = prod([a for a in lazy_tensorisation])
        objs = [jnp.arange(0, product**2).reshape(product, product)]
        print("tensorisation init:", tensorisation_maker(lazy_tensorisation), "\nobj initial:", objs[0], "\nfirst line:", objs[0][0])
        temp = extension_nD(objs, options)
        return temp
    lazy_tensorisation_2D = [7,4]
    max_lazy_tensorisation_2D = [100,100]
    trunc_size_2D = [4,2]
    lazy_tensorisation_1D = [7]
    max_lazy_tensorisation_1D = [100]
    trunc_size_1D = [4]
    ext_2D = run_test(lazy_tensorisation_2D, max_lazy_tensorisation_2D, trunc_size_2D)
    ext_1D = run_test(lazy_tensorisation_1D, max_lazy_tensorisation_1D, trunc_size_1D)
    ext_2D_max = run_test(lazy_tensorisation_2D, [9,10], trunc_size_2D)
    ext_1D_max = run_test(lazy_tensorisation_1D, [8], trunc_size_1D)
    print("extension first line", ext_2D[0][0][0]) 
    print("extension first line with near max", ext_2D_max[0][0][0]) 
    expected_first_line_2D =  jnp.array(
        [0., 1., 2., 3., 0., 0., 4., 5., 6., 7., 0., 0., 8., 9., 10., 11., 0., 0.,
        12., 13., 14., 15., 0., 0., 16., 17., 18., 19., 0., 0., 20., 21., 22., 23.,
        0., 0., 24., 25., 26., 27., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    expected_first_line_2D_max = jnp.array(
        [0., 1., 2., 3., 0., 0., 4., 5., 6., 7., 0., 0.,
        8., 9., 10., 11., 0., 0., 12., 13., 14., 15., 0., 0.,
        16., 17., 18., 19., 0., 0., 20., 21., 22., 23., 0., 0.,
        24., 25., 26., 27., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.]
    ) # to check that near the max_extension it's also okay
    print(expected_first_line_2D_max)
    print(ext_2D_max[0][0][0])
    print("ext_1D:", ext_1D[0])
    print("ext_1D_max:", ext_1D_max[0])
    return (jnp.array_equal(expected_first_line_2D, ext_2D[0][0][0]) and
            jnp.array_equal(expected_first_line_2D_max, ext_2D_max[0][0][0]) 
    )


