import jax.numpy as jnp

from ..._utils import cdtype
from ...utils.utils import dag

def matrix_to_line(matrix,k,n):
    # transform a n*n matrix consisting of zeros apart from k lines of length n
    # into its "line" version, ie suppressing the zeros. it aims at having a more
    # efficient way to compute  A_N-P_n@A_N@P_n with P_n the projector on n<N
    # times an other matrix

    res = jnp.zeros((k,n), cdtype())
    res=res.at[0:k,0:n].set(matrix[n-k:n,0:n])
    return res

def matrix_to_column(matrix,k,n):
    # transform a n*n matrix consisting of zeros apart from k columns of length n
    # into its "column" version, ie suppressing the zeros. it aims at having a more
    # efficient way to compute  dag(\mathcal{L}_{N+k}-\mathcal{L}_{N}) @
    # (\mathcal{L}_{N+k}-\mathcal{L}_{N}) @ rho_N

    res = jnp.zeros((n,k), cdtype())
    res=res.at[0:n,0:k].set(matrix[0:n,n-k:n])
    return res

def line_to_matrix(line,k,n):
    # opposite operation to matrix_to_line

    res = jnp.zeros((n,n), cdtype())
    res=res.at[n-k:n,0:n].set(line[0:k,0:n])
    return res

def special_product(jump_ops_N_plus_k,jump_ops_N,rho_etendu,k,N,total):
    # compute dag(\mathcal{L}_{N+k}-\mathcal{L}_{N}) @
    # (\mathcal{L}_{N+k}-\mathcal{L}_{N}) @ rho_N in an efficient way and adds it to
    # total of size (N+k,N+k)
    len_jumps = len(jump_ops_N_plus_k)
    for i in range(len_jumps):
        substract = jump_ops_N_plus_k[i] - jump_ops_N[i]
        substractd = dag(jump_ops_N_plus_k[i] - jump_ops_N[i])
        line = matrix_to_line((substract),k,N+k)
        # we make (\mathcal{L}_{N+k}-\mathcal{L}_{N}) @ rho_N first
        product1 = jnp.matmul(line,rho_etendu)
        # then we make the full product dag(\mathcal{L}_{N+k}-\mathcal{L}_{N}) @
        # (\mathcal{L}_{N+k}-\mathcal{L}_{N}) @ rho_N
        total = (total.at[0:N+k,0:N+k].set(total[0:N+k,0:N+k] - 0.5 *
            jnp.matmul(matrix_to_column(substractd,k,N+k),product1)[0:N+k,0:N+k])
        )
    return total