import jax.numpy as jnp
import jax
import itertools
from .inequalities import generate_rec_ineqs, update_ineq
from .utils.utils import (
    prod, split_contiguous_indices, shift_ranges, ineq_to_tensorisation, 
    reverse_indices, generate_square_indices_around, eliminate_duplicates, to_hashable,
    cut_over_max_tensorisation, find_contiguous_ranges, add_trunc_size_vectors,
    excluded_numbers, tensorisation_maker
)
from .._utils import cdtype

def mask(obj, pos):
    """
    Prepare positions in a matrix of size obj where we want to act through a mask

    Args:
    obj: input matrix.
    pos: position where zeros should be placed.

    Returns:
    mask: Boolean matrix where False indicates where we will act (putting zeros when
    projecting for instance)
    """
    # Create a mask for the positions to zero out
    mask = jnp.ones_like(obj, dtype=bool)
    for x in pos:
        mask = jnp.where(jnp.arange(obj.shape[0])[:, None] == x, False, mask)  # Zero out row
        mask = jnp.where(jnp.arange(obj.shape[1])[None, :] == x, False, mask)  # Zero out column
    return mask

def projection_nD(
    obj, _mask
):
    """
    Put zeros cols and rows where _mask is True for obj.
    Exemple : 
    _mask = [True, True, False, True, True, False]
            [True, True, False, True, True, False]
            [False, False, False, False, False, False]
            [True, True, False, True, True, False]
            [True, True, False, True, True, False]
            [False, False, False, False, False, False]
    obj = jnp.arange(1, 37).reshape(6, 6)
    res = jnp.array([[ 1,  2,  0,  4,  5,  0],
                    [ 7,  8,  0, 10, 11,  0],
                    [ 0,  0,  0,  0,  0,  0],
                    [19, 20,  0, 22, 23,  0],
                    [25, 26,  0, 28, 29,  0],
                    [ 0,  0,  0,  0,  0,  0]])
    """
    new_obj = obj
    new_obj = jnp.where(_mask, obj, 0)
    return new_obj


def dict_nD(tensorisation, inequalities):
    # dictio will make us able to repertoriate the indices concerned by the inequalities
    dictio=[]
    for i in range(len(tensorisation)):
        # we check if some inequalities aren't verified, if so we will add the line
        # in dictio 
        for inegality in inequalities:
            if not inegality(*tuple(tensorisation[i])):
                dictio.append(i)
                break # to avoid appending the same line multiple times
    return dictio


def red_ext_zeros(objs, tensorisation, inequalities, options):
    new_objs = []
    # Sort positions in descending order to avoid shifting issues
    dictio = sorted(dict_nD_reshapings(tensorisation, inequalities, options), reverse=True)
    tensorisation = delete_tensor_elements(tensorisation, dictio)
    tensorisation = list(to_hashable(tensorisation))
    for i in range(len(objs)):
        new_objs.append(reduction_nD(objs[i], dictio))
    # print(tensorisation)
    ext_tens = extended_tensorisation(None, options, tensorisation)
    # print(tensorisation, ext_tens)
    old_pos, new_pos = find_contiguous_ranges(tensorisation, ext_tens)
    # print(old_pos, new_pos, len(ext_tens), jnp.array(new_objs).shape)
    len_new_objs = len(ext_tens)
    zeros_obj = jnp.zeros((len_new_objs, len_new_objs), cdtype())
    jaxed_extension = jax.jit(lambda objs: extension(objs, new_pos, old_pos, zeros_obj))
    ext = jaxed_extension(new_objs)
    return ext, ext_tens

def red_ext_full(objs, tensorisation, inequalities, options):
    # Sort positions in descending order to avoid shifting issues
    dictio = sorted(dict_nD_reshapings(tensorisation, inequalities, options), reverse=True)
    tensorisation = delete_tensor_elements(tensorisation, dictio)
    tensorisation = list(to_hashable(tensorisation))
    # print(tensorisation)
    ext_tens = extended_tensorisation(None, options, tensorisation)
    maximum_tensorisation = list(itertools.product(*[range(max_dim) 
        for max_dim in options.tensorisation]))
    _, ranges_to_keep = find_contiguous_ranges(ext_tens, maximum_tensorisation
    )
    # print(ext_tens, maximum_tensorisation)
    # print(ranges_to_keep)
    new_objs = []
    for i in range(len(objs)):
        max_size = len(objs[i])
        dict_to_delete = excluded_numbers(ranges_to_keep, max_size)
        new_objs.append(reduction_nD(objs[i], dict_to_delete))
    return new_objs, ext_tens


def extension_nD(
        objs, options, inequalities
):
    """
    Extend the objs up to the max_tensorisation by putting zeros cols and rows in spots
    where the inequalities are satisfied and there is no rows and cols.
    Extend by options.trunc_size

    Args:
    objs: list of input matrices (JAX arrays)
    options: see options[lien]
    inequalities: the new inequalities we need to expand to
    tensorisation: optional tensorisation to put if we cannot guess it from the 
    inequalities

    Returns:
    objs: list of matrices with zeros placed.
    """
    ineq_to_tensors = extended_tensorisation(inequalities, options)
    # print("tensorisation aprés:", ineq_to_tensors)
    len_new_objs = len(ineq_to_tensors)
    reverse_dictio = reverse_indices(dict_nD_reshapings(ineq_to_tensors, inequalities, 
        options), len_new_objs - 1
    )
    new_positions = split_contiguous_indices(reverse_dictio)
    old_positions = shift_ranges(new_positions)
    # print("old positions:",old_positions, "new positions:", new_positions, len_new_objs)
    # print(objs)
    zeros_obj = jnp.zeros((len_new_objs, len_new_objs), cdtype())
    jaxed_extension = jax.jit(lambda objs: extension(objs, new_positions, old_positions, zeros_obj))
    return jaxed_extension(objs), ineq_to_tensors, options


def extension(objs, new_positions, old_positions, zeros_obj):
    """
    Extends a matrix by putting its values at index from old_positions to new_positions,
     and filling the rest with zeros
    """
    len_pos = len(old_positions)
    new_objs = []
    for obj in objs:
        new_obj = zeros_obj[:]
        for i in range(len_pos):
            for j in range(len_pos):
                new_obj = new_obj.at[new_positions[i][0]:new_positions[i][1]+1, 
                    new_positions[j][0]:new_positions[j][1]+1].set(
                    obj[old_positions[i][0]:old_positions[i][1]+1, 
                    old_positions[j][0]:old_positions[j][1]+1]
                )
        new_objs.append(new_obj)
    return new_objs


def dict_nD_reshapings(tensorisation, inequalities, options = None, usage = None):
    from .utils.warnings import check_not_under_truncature, check_in_max_truncature
    # dictio will make us able to repertoriate the indices concerned by the inequalities
    dictio=[]
    # i compte l'indice sur lequel on se trouve pour l'encodage en machine des positions
    # de la matrice, par exemple (i=0 pour (0,0)) et (i=2 pour (1,0)) dans l'exemple
    # precedent
    for i in range(len(tensorisation)):
        # we check if some inequalities aren't verified, if so we will add the line
        # in dictio 
        for inegality in inequalities:
            if ((usage == 'proj' or usage == 'reduce') and 
                check_in_max_truncature(tensorisation[i], options)
            ): # not a very optimized thing...
                dictio.append(i)
                break
            if not inegality(*tuple(tensorisation[i])):
                if options is not None:
                    if ((#(usage == 'proj' or usage == 'reduce') and 
                        not check_not_under_truncature(tensorisation[i], 
                        options.trunc_size))# 2* because We need L_n-L_{n-k} n>=2*k
                    ):
                        break
                dictio.append(i)
                break # to avoid appending the same line multiple times
    return dictio


def mask_jaxed(obj, pos): # in tests was fatser in practice... No ?
    """
    Prepare positions in a matrix of size obj where we want to act through a mask.

    Args:
    obj: input matrix.
    pos: positions where zeros should be placed.

    Returns:
    mask: Boolean matrix where False indicates where we will act (putting zeros when
    projecting for instance).
    """
    n_rows, n_cols = obj.shape
    row_indices = jnp.arange(n_rows)[:, None]
    col_indices = jnp.arange(n_cols)[None, :]
    row_mask = jnp.isin(row_indices, pos)
    col_mask = jnp.isin(col_indices, pos)
    combined_mask = row_mask | col_mask
    return ~combined_mask


def delete_tensor_elements(obj, positions):
    # Sort positions in reverse order to avoid reindexing issues during deletion
    for pos in sorted(positions, reverse=True):
        del obj[pos]
    return obj

def reduction_nD(obj, positions):
    """
    same as projection_nD but delete lines instead of putting zeros.
    """
    # Convert positions to a numpy array for indexing
    positions = jnp.array(positions)
    # Create masks to keep track of rows and columns to keep
    mask = jnp.ones(obj.shape[0], dtype=bool)
    if positions.size!=0:
        mask = mask.at[positions].set(False)
    return obj[mask][:, mask]

def extended_tensorisation(new_ineq, options, tensorisation = None):
    # complexity n*k^2
    if tensorisation is None:
        new_tensorisation = add_trunc_size_vectors(ineq_to_tensorisation(new_ineq, 
            options.tensorisation), options.trunc_size
        )
    else:
        new_tensorisation = tensorisation[:] # copy needed to not modify tensorisation
    L=[]
    len_trunc_size = len(options.trunc_size)
    for tensor in new_tensorisation:
        if any((tensor[i]==0 for i in range(len_trunc_size))):
            for new_tensor in (generate_square_indices_around(tensor, 
                options.trunc_size)):
                L.append(new_tensor)
        else:
            tensor_enlarged_for_truncature = [x + trunc for x, trunc in 
                zip(tensor, options.trunc_size)
            ]
            L.append(tensor_enlarged_for_truncature)
    L = to_hashable(cut_over_max_tensorisation(L, [x- 1 for x in 
        options.tensorisation])
    ) # because new_tensorisation starts at 0
    for tensor in L:
        if tensor not in new_tensorisation:
            new_tensorisation.append(tensor)
    new_tensorisation = sorted(eliminate_duplicates(new_tensorisation))
    return new_tensorisation