import jax
import jax.numpy as jnp
from ...time_array import ConstantTimeArray, TimeArray
import itertools
from .utils import prod

def warning_bad_TimeArray(H, jump_ops):
    cdn = (type(H)!=ConstantTimeArray or 
        any([type(jump_ops[i])!=ConstantTimeArray 
        for i in range(len(jump_ops))])
    )
    if (cdn):
        jax.debug.print(
            'WARNING : If your array is not time dependant, beware that '
            'the truncature required to compute the estimator won\'t be '
            'trustworthy. See [link to the article] for more details. '
        )
    return cdn

def warning_size_too_small(tensorisation, trunc_size):
    trunc_size_elements = list(
        itertools.product(*[range(max_dim) for max_dim in trunc_size])
    )
    if not all(elem in tensorisation for elem in trunc_size_elements):
        raise ValueError(f"""Your object is too small for the estimator method to be 
        applied. Trunc_size is: 
        {trunc_size} 
        and tensorisation is:
        {tensorisation}
        Try giving a larger object.
        """)
    
def warning_estimator_tol_reached(solution, options, solver):
    estimator_final = solution.estimator[-1]
    rho_final = solution.states[-1]
    if (estimator_final > options.estimator_rtol * (solver.atol + 
        jnp.linalg.norm(rho_final, ord='nuc') * solver.rtol)
    ):
        jax.debug.print(
            'WARNING : At this truncature of your simulation\'s size, '
            'it\'s not possible to warranty anymore the accuracy of '
            'your results. Try to enlarge the truncature'
        )
        jax.debug.print(
            "estimated error = {err} > {estimator_rtol} * tolerance = {tol}", 
            err = ((estimator_final).real.astype(float)), 
            estimator_rtol = options.estimator_rtol,
            tol = options.estimator_rtol * 
            (solver.atol + jnp.linalg.norm(rho_final, ord='nuc') * solver.rtol)     
        )
    return None

def check_max_reshaping_reached(options, obj: TimeArray):
    return prod(options.tensorisation)==len(obj(0)[0])

def check_not_under_truncature(tensorisation, trunc_size):
    """
    Check if the a certain tensorsation is not under 2 times the truncature to avoid
    problems with the estimator.
    For instance if we have trunc_size = [8,4] and we check the tensorisation [7,1], it 
    returns true, while [8,1] returns False (tensorisation starting at 0)
    """
    return any(tensorisation[j] > max(trunc_size[j] - 1, 0)
        for j in range(len(trunc_size))
    )

def check_in_max_truncature(tensorisation, options):
    """
    Add this tensorisation to the mask for projecting if it's near the max attainable 
    tensorisation. (needed to compute the estimator)
    """
    return any(tensorisation[j] > options.tensorisation[j] - 1 - 
        options.trunc_size[j] for j in range(len(options.tensorisation))
    )