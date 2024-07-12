from .utils import ineq_to_tensorisation, ineq_from_params, indices_cut_by_ineqs
import jax.numpy as jnp
import jax
import itertools

def downsize_via_ineq(H, rho0, jump_ops, options):
    ineqs , params = zip(*options.inequalities)
    inequalities = ineq_from_params(ineqs, params)
    actual_tensorisation = list(
        itertools.product(*[range(max_dim) for max_dim in options.tensorisation])
    )
    tensorisation = ineq_to_tensorisation(inequalities, actual_tensorisation)
    indices_to_cut = indices_cut_by_ineqs(inequalities, actual_tensorisation)
    # fun = jax.jit(lambda x: reduction_nD(x, indices_to_cut))
    H, rho0, *jump_ops = [
        reduction_nD(x, indices_to_cut) for x in ([H]+[rho0]+jump_ops)
    ]
    options.__dict__['inequalities'] = [[None, p] for p in params]
    return (H, rho0, jump_ops)

def reduction_nD(obj, positions):
    """
    delete rows and cols of the object for said positions
    """
    # Convert positions to a numpy array for indexing
    positions = jnp.array(positions)
    # Create masks to keep track of rows and columns to keep
    mask = jnp.ones(obj.shape[0], dtype=bool)
    if positions.size!=0:
        mask = mask.at[positions].set(False)
    return obj[mask][:, mask]

