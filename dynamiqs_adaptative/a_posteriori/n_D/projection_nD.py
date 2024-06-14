import jax.numpy as jnp

def projection_nD(
        objs, original_tensorisation = None, inequalities = None, _mask = None
):
    # create a tensorial projection of some n dimensional matrix "objs" tensorised under 
    # "original_tensorisation"  into matrixs "new_objs" projected according to
    # some inequalities "inequalities"
    # as an example : original_tensorisation can have this 
    # ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)) and if applied inequalities 
    # [lambda i, j: i <= 2 , lambda i, j: j <= 2]
    # on such a matrix matrix_6x6 = np.arange(1, 37).reshape(6, 6)
    # the output would be : 
    # matrix_6x6 = np.array([[ 1,  2,  0,  4,  5,  0],
    #                        [ 7,  8,  0, 10, 11,  0],
    #                        [ 0,  0,  0,  0,  0,  0],
    #                        [19, 20,  0, 22, 23,  0],
    #                        [25, 26,  0, 28, 29,  0],
    #                        [ 0,  0,  0,  0,  0,  0]])
    # all objs would undergo the related projection, making zeros the (0,2) and (1,2)
    # cols and rows of the objs.
    # (the size of the objs would not be changed, but there would be zeros on
    # projected lines and columns)

    new_objs = objs
    # if dictio is already known we don't calculate it
    if _mask is None:
        if original_tensorisation is None or inequalities is None:
            raise ValueError(" You have to either give a mask or a tensorisation with some inequalities")
        dictio = dict_nD(original_tensorisation, inequalities)
        _mask = mask(new_objs[0], dictio)
    for i in range(len(new_objs)):
        new_objs[i] = jnp.where(_mask, new_objs[i], 0)

    return new_objs

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
                # to avoid appending the same line multiple times
                break

    return dictio

def mask(obj, dict):
    """
    Put zeros on the line and column of a matrix "obj" by reference to a position "pos".
    Zeros are placed in a contiguous streak.

    Args:
    obj: input matrix
    pos: position where zeros should be placed

    Returns:
    obj: matrix with zeros placed in a contiguous streak
    """
    # Create a mask for the positions to zero out
    mask = jnp.ones_like(obj, dtype=bool)
    for x in dict:
        mask = jnp.where(jnp.arange(obj.shape[0])[:, None] == x, False, mask)  # Zero out row
        mask = jnp.where(jnp.arange(obj.shape[1])[None, :] == x, False, mask)  # Zero out column

    return mask

def zeros_old(obj, pos):
    # Put zeros on the line and column of a matrix "obj" by reference to a position "pos"
    
    zeros = jnp.zeros(obj.shape[0], dtype=obj.dtype)
    obj = obj.at[pos, :].set(zeros)
    obj = obj.at[:, pos].set(zeros)
    
    return obj

def reduction_nD(objs, original_tensorisation, inequalities):
    # same as projection_nD but delete lines instead of putting zeros

    new_objs = []
    # Sort positions in descending order to avoid shifting issues
    dictio = sorted(dict_nD(original_tensorisation, inequalities), reverse=True)
    for i in range(len(objs)):
        new_objs.append(recursive_delete(objs[i], dictio))

    return new_objs

def delete(obj,pos):
    # delete the line and col of a matrix "obj" by reference to a position "pos"
    return jnp.delete(jnp.delete(obj, pos, axis=0), pos, axis=1)

def recursive_delete(obj, positions):
    if not positions:
        return obj
    # Delete the first position in the list and call recursively for the rest
    return recursive_delete(delete(obj, positions[0]), positions[1:])

def unit_test_projection_nD():
    original_tensorisation = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
    inequalities = [lambda i, j: i <= 1, lambda i, j: j <= 1]
    objs = [jnp.arange(1, 37).reshape(6, 6)]
    objs = projection_nD(objs, original_tensorisation, inequalities)
    
    expected_result = [jnp.array([[1, 2, 0, 4, 5, 0],
                                    [7, 8, 0, 10, 11, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [19, 20, 0, 22, 23, 0],
                                    [25, 26, 0, 28, 29, 0],
                                    [0, 0, 0, 0, 0, 0]])]
                                    
    return jnp.array_equal(objs, expected_result)

