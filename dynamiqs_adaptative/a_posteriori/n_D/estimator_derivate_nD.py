import jax.numpy as jnp
from .lindbladien_nD import lindbladien_nD
from ...utils.utils.general import lindbladian
from ...utils.utils import dag

def estimator_derivate_simple_nD(
        rho, GLs, Ls, GH, H
    ):
    # compute \dot{e}(t)=(\mathcal{L}_{original_tensorisation}-
    # \mathcal{L}_{reduced_tensorisation})(\rho_{reduced_tensorisation}(t))
    # the reduced tensorisation being given by the set inequalities
    # 'G' indicates the non projected operators
                
    # Ls = jnp.stack([x(t) for x in L])
    # GLs = jnp.stack([x(t) for x in GL])
    # L_rho_reduced = lindbladien_nD(rho, H, Ls)
    # L_rho = lindbladien_nD(rho, GH, GLs)
    L_rho_reduced = lindbladian(H, Ls, rho)
    L_rho = lindbladian(GH, GLs, rho)
    
    return jnp.linalg.norm(L_rho-L_rho_reduced, ord='nuc')

def estimator_derivate_opti_nD(drho, GH, GLs, rho):
    # dynamiqs method optimized version of above
    L_rho_reduced = drho
    Lsd = dag(GLs)
    LdL = (Lsd @ GLs).sum(0)
    tmp = (-1j * GH - 0.5 * LdL) @ rho + 0.5 * (GLs @ rho @ Lsd).sum(0)
    L_rho = tmp + dag(tmp)

    return jnp.linalg.norm(L_rho-L_rho_reduced, ord='nuc')