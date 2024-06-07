import jax.numpy as jnp
from ..._utils import cdtype

def extension_1D(obj,ngrand,type=0):
        #extend a JAX matrix living in n*n to a ngrand*ngrand space (filling the diff with 0)
        if type==0:
            n=len(obj[0])
            new_obj=jnp.zeros((ngrand,ngrand),cdtype())
            new_obj=new_obj.at[:n, :n].set(obj[:n,:n])
            return new_obj
        elif type==1:
            taille=len(obj)
            new_obj=jnp.zeros((taille,ngrand,ngrand),cdtype())
            for i in range(taille):
                n=len(obj[i][0])
                new_obj=new_obj.at[i].set(jnp.zeros((ngrand,ngrand),cdtype()).at[:n,:n].set(obj[i]))
            return new_obj
        else:
            raise ValueError('type = 0 for a JAX matrix, 1 for extending mesolve output (list of JAX matrices)')

def unit_test_extension_1D():
    obj_ext=jnp.zeros((100,100),cdtype())
    obj=jnp.zeros((10,10),cdtype())
    for i in range(10):
        for j in range(10):
            obj=obj.at[i,j].set(2*i+j)
    for i in range(10):
        for j in range(10):
            obj_ext=obj_ext.at[i,j].set(2*i+j)
    if jnp.linalg.norm(extension_1D(obj,100)-obj_ext)==0:
        return True
    else:
        return False

# print(unit_test_extension_1D())
