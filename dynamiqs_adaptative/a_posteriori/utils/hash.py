import jax.numpy as jnp
import numpy as np

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