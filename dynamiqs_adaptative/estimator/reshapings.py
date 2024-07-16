import jax.numpy as jnp

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