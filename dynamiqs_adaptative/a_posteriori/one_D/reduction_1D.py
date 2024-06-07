import jax.numpy as jnp
from ..._utils import cdtype

def reduction_1D(obj,npetit,type=0):
        #reduce a JAX matrix living in n*n to a npetit*npetit space
        ####use type=1 for a list of jump_ops
        if type==0:
            new_obj=jnp.zeros((npetit,npetit),cdtype())
            new_obj=new_obj.at[:npetit, :npetit].set(obj[:npetit,:npetit])
            return new_obj
        elif type==1:
            taille=len(obj)
            new_obj=[jnp.zeros((npetit,npetit),cdtype()) for i in range(taille)]
            for i in range(taille):
                new_obj[i]=jnp.zeros((npetit,npetit),cdtype()).at[:npetit,:npetit].set(obj[i][:npetit,:npetit])
            return new_obj
        else:
            raise ValueError('type = 0 for a JAX matrix, 1 for jump_ops (list of JAX matrices)')

def unit_test_reduction_1D():
    obj=jnp.zeros((100,100),cdtype())
    obj_red=jnp.zeros((10,10),cdtype())
    for i in range(10):
        for j in range(10):
            obj_red=obj_red.at[i,j].set(2*i+j)
    for i in range(100):
        for j in range(100):
            obj=obj.at[i,j].set(2*i+j)
    if jnp.linalg.norm(reduction_1D(obj,10)-obj_red)==0:
        return True
    else:
        return False

# print(unit_test_reduction_1D())
