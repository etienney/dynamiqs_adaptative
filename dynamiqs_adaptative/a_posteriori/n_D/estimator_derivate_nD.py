import jax.numpy as jnp
from .lindbladien_nD import lindbladien_nD

def estimator_derivate_simple_nD(
        rho, GLs, Ls, GH, H
    ):
    # compute \dot{e}(t)=(\mathcal{L}_{original_tensorisation}-
    # \mathcal{L}_{reduced_tensorisation})(\rho_{reduced_tensorisation}(t))
    # the reduced tensorisation being given by the set inequalities
    # 'G' indicates the non projected operators
                
    # Ls = jnp.stack([x(t) for x in L])
    # GLs = jnp.stack([x(t) for x in GL])
    L_rho = lindbladien_nD(rho, H, Ls)
    L_rho_reduced = lindbladien_nD(rho, GH, GLs)
    
    return jnp.linalg.norm(L_rho-L_rho_reduced, ord='nuc')