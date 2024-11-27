from ..utils.utils import dag
import jax.numpy as jnp

def compute_estimator(H, Ls, Hred, Lsred, rho, t):
    Hred = Hred(t)
    Lsred = jnp.stack([L(t) for L in Lsred])
    Lsd = dag(Lsred)
    LdL = (Lsd @ Lsred).sum(0)
    tmp = (-1j * Hred - 0.5 * LdL) @ rho + 0.5 * (Lsred @ rho @ Lsd).sum(0)
    drhored = tmp + dag(tmp)

    H = H(t)
    Ls = jnp.stack([L(t) for L in Ls])
    Lsd = dag(Ls)
    LdL = (Lsd @ Ls).sum(0)
    tmp = (-1j * H - 0.5 * LdL) @ rho + 0.5 * (Ls @ rho @ Lsd).sum(0)
    drho = tmp + dag(tmp)

    return jnp.linalg.norm(drho-drhored, ord='nuc')
