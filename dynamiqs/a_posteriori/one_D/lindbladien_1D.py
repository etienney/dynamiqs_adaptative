import jax.numpy as jnp

from ...utils.utils import dag
from .projection_1D import projection_1D


def lindbladien_1D(rho, npetit, ngrand, Gjump_ops, GH):
        #Compute L_npetit(rho_npetit) in the space size ngrand
        Pn=projection_1D(npetit,ngrand)
        jump_ops = [Pn @ Gjump_ops[i] @ Pn for i in range(len(Gjump_ops))]
        return (
                complex(0,-1) * (Pn@GH@Pn@rho-rho@Pn@GH@Pn) + 
                sum([jump_ops[i] @ rho @ dag(jump_ops[i]) - (0.5*jnp.identity(ngrand)) @
                (dag(jump_ops[i]) @ jump_ops[i] @ rho + rho @ dag(jump_ops[i]) @ 
                jump_ops[i])  for i in range(len(jump_ops)) ])
        )
