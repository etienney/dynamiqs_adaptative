from .extension_1D import extension_1D
from .lindbladien_1D import lindbladien_1D
from .reduction_1D import reduction_1D
from .calcul_matri_opti_1D import *
from ...utils.utils import dag
import jax.numpy as jnp


def estimator_derivate_simple(rho, L, H, t, N, k):
    # compute \dot{e}(t)=(\mathcal{L}_N-\mathcal{L}_{N-k})(\rho_{N-k}(t))
    
    Ls = jnp.stack([x(t) for x in L])
    L_rho_N = lindbladien_1D(rho,N,N,Ls,H)
    L_rho_N_minus_k = lindbladien_1D(rho,N-k,N,Ls,H)
    return jnp.linalg.norm(L_rho_N-L_rho_N_minus_k,ord='nuc')

def estimator_derivate_opti(rho, L, H, t, N, k):
    # compute \dot{e}(t)=(\mathcal{L}_n-\mathcal{L}_{N-k})(\rho_{N-k}(t)) in an 
    # optimised way

    Ls = jnp.stack([x(t) for x in L])
    len_jumps = len(Ls)
    # necessary reduction of H and Ls to compute the estimator
    # (extension(reduction())=projection())
    Ls_N_minus_k = (
        [extension_1D(reduction_1D(Ls[i],N-k),N) for i in range
        (len_jumps)]
    )
    H_N_minus_k = extension_1D(reduction_1D(H,N-k),N)

    # calcul de (\mathcal{L}_{N+k}-\mathcal{L}_{N})(\rho_N)

    a = matrix_to_line((H-H_N_minus_k),k,N)
    b = jnp.matmul(a,rho)
    total = complex(0,-1)*line_to_matrix(b,k,N)

    # calcul de la partie en \mathcal{L}\rho\mathcal{L}
    for i in range(len_jumps):
        total += 0.5 * (
            line_to_matrix(jnp.matmul(jnp.matmul(matrix_to_line((
            Ls[i] - Ls_N_minus_k[i]),k,N),rho),
            dag(Ls[i])),k,N)
        )
        total += 0.5 * (
            line_to_matrix(jnp.matmul(jnp.matmul(matrix_to_line((
            Ls[i] - Ls_N_minus_k[i]),k,N),rho),
            dag(Ls_N_minus_k[i])),k,N)
        )
    # computation of the anticommutator part
    for i in range(len_jumps):
        # dag(Ls[i]-Ls_N_minus_k[i]) @ Ls_N_minus_k[i])
        # @ rho
        total += -0.5 * (
            line_to_matrix(jnp.matmul(jnp.matmul(matrix_to_line((
            dag(Ls[i] - Ls_N_minus_k[i])),k,N),
            Ls_N_minus_k[i]),rho),k,N)
        )
        # dag(Ls[i]-Ls_N_minus_k[i])
        # @ (Ls[i]-Ls_N_minus_k[i]) @ rho
        total = special_product(Ls,Ls_N_minus_k,rho,k,N-k,total)

    # We take the norm
    return jnp.linalg.norm((total)+dag(total), ord = 'nuc')