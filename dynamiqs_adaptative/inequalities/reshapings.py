from .utils import (
    ineq_to_tensorisation, 
    ineq_from_params,
    reverse_indices,
    split_contiguous_indices,
    shift_ranges
)
from .._utils import cdtype
import jax.numpy as jnp
import jax
import itertools

def downsize_via_ineq(H, rho0, jump_ops, options):
    ineqs , params = zip(*options.inequalities)
    inequalities = ineq_from_params(ineqs, params)
    actual_tensorisation = list(
        itertools.product(*[range(max_dim) for max_dim in options.tensorisation])
    )
    ineq_tensorisation = ineq_to_tensorisation(inequalities, actual_tensorisation)
    indices_to_cut = dict_nD(actual_tensorisation, inequalities)
    H, rho0, *jump_ops = [
        reduction_nD(x, indices_to_cut) for x in ([H]+[rho0]+jump_ops)
    ]
    options.__dict__['inequalities'] = [[None, p] for p in params] # JAX cannot stand lambda arguments when called with options
    return (
        options, actual_tensorisation, ineq_tensorisation, inequalities, H, rho0, 
        jump_ops
    )


def reduction_nD(obj, positions):
    """
    delete rows and cols of the object for said positions.
    """
    # Convert positions to a numpy array for indexing
    positions = jnp.array(positions)
    # Create masks to keep track of rows and columns to keep
    mask = jnp.ones(obj.shape[0], dtype=bool)
    if positions.size!=0:
        mask = mask.at[positions].set(False)
    return obj[mask][:, mask]


def dict_nD(previous_tensorisation, inequalities):
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


def extension_nD(objs, actual_tensorisation, inequalities):
    len_new_objs = len(actual_tensorisation)
    reverse_dictio = reverse_indices(
        dict_nD(actual_tensorisation, inequalities), len_new_objs - 1
    )
    new_positions = split_contiguous_indices(reverse_dictio)
    old_positions = shift_ranges(new_positions)
    zeros_obj = jnp.zeros((len_new_objs, len_new_objs), cdtype())
    jaxed_extension = jax.jit(
        lambda objs: extension(objs, new_positions, old_positions, zeros_obj)
    )
    new_objs = []
    for obj in objs:
        new_objs.append(jaxed_extension(obj))
    return jnp.array(new_objs)


def extension(obj, new_positions, old_positions, zeros_obj):
    """
    Extends a matrix by putting its values at index from old_positions to new_positions,
     and filling the rest with zeros
    """
    len_pos = len(old_positions)
    new_obj = []
    new_obj = zeros_obj[:]
    for i in range(len_pos):
        for j in range(len_pos):
            new_obj = new_obj.at[new_positions[i][0]:new_positions[i][1]+1, 
                new_positions[j][0]:new_positions[j][1]+1].set(
                obj[old_positions[i][0]:old_positions[i][1]+1, 
                old_positions[j][0]:old_positions[j][1]+1]
            )
    return new_obj


