import jax.numpy as jnp
import numpy as np
import jax

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