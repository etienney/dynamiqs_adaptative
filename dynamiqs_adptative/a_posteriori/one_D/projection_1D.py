import jax.numpy as jnp

from ..._utils import cdtype

def projection_1D(n,ngrand):
        # the projector on the n first state, in the space ngrand
        P=jnp.zeros((ngrand,ngrand),cdtype())
        for i in range(n):
            P=P.at[i,i].set(1)
        return P
