import jax.numpy as jnp

from ...utils.utils import dag

def lindbladien_nD(rho, H, L):
        # Compute the lindbladian

        original_size = len(H[0])
        len_jump_ops = len(L)

        return (
                complex(0,-1) * (H@rho-rho@H) + 
                sum([L[i] @ rho @ dag(L[i]) - 
                (0.5*jnp.identity(original_size)) @
                (dag(L[i]) @ L[i] @ rho + rho @ dag(L[i]) @ 
                L[i])  for i in range(len_jump_ops) ])
        )
