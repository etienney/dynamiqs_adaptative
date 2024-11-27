from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._checks import check_type_int
from .._utils import cdtype
from .operators import displace
from .utils import tensor, todm

__all__ = [
    'fock',
    'fock_dm',
    'basis',
    'basis_dm',
    'coherent',
    'coherent_dm',
    'ground',
    'excited',
]


def fock(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    r"""Returns the ket of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array_like of shape (...) or (..., len(dim)))_: Fock state number
            for each mode, of integer type. If `dim` is a tuple, the last dimension of
            `number` should match the length of `dim`.

    Returns:
        _(array of shape (..., n, 1))_ Ket of the Fock state or tensor product of Fock
            states, with _n = prod(dims)_.

    Examples:
        Single-mode Fock state $\ket{1}$:
        >>> dq.fock(3, 1)
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)

        Batched single-mode Fock states $\{\ket{0}\!, \ket{1}\!, \ket{2}\}$:
        >>> dq.fock(3, [0, 1, 2])
        Array([[[1.+0.j],
                [0.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [1.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [0.+0.j],
                [1.+0.j]]], dtype=complex64)

        Multi-mode Fock state $\ket{1,0}$:
        >>> dq.fock((3, 2), (1, 0))
        Array([[0.+0.j],
               [0.+0.j],
               [1.+0.j],
               [0.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)

        Batched multi-mode Fock states $\{\ket{0,0}\!, \ket{0,1}\!, \ket{1,1}\!,
        \ket{2,0}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock((3, 2), number).shape
        (4, 6, 1)
    """
    dim = jnp.asarray(dim)
    number = jnp.asarray(number)
    check_type_int(dim, 'dim')
    check_type_int(number, 'number')

    # check if dim is a single value or a tuple
    if dim.ndim > 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # if dim is an integer, convert shapes dim: () -> (1,) and number: (...) -> (..., 1)
    if dim.ndim == 0:
        dim = dim[None]
        number = number[..., None]

    # check if number has shape (..., len(ndim))
    if number.shape[-1] != dim.shape[-1]:
        raise ValueError(
            'Argument `number` must have shape `(...)` or `(..., len(dim))`, but'
            f' has shape number.shape={number.shape}.'
        )

    # check if 0 <= number[..., i] < dim[i] for all i
    if jnp.any(dim - number <= 0):
        raise ValueError(
            'Argument `number` must be in the range [0, dim[i]) for each mode i:'
            ' 0 <= number[..., i] < dim[i].'
        )

    # compute all kets
    _vectorized_fock = jnp.vectorize(_fock, signature='(ndim),(ndim)->(prod_ndim,1)')
    return _vectorized_fock(dim, number)


def _fock(dim: Array, number: Array) -> Array:
    # return the tensor product of Fock states |n0> x |n1> x ... x |nf> where dim has
    # shape (ndim,), number has shape (ndim,) and number = [n0, n1,..., nf]
    # this is the unbatched version of fock()
    idx = 0
    for d, n in zip(dim, number):
        idx = d * idx + n
    ket = jnp.zeros((prod(dim), 1), dtype=cdtype())
    return ket.at[idx].set(1.0)


def fock_dm(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    r"""Returns the density matrix of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array_like of shape (...) or (..., len(dim)))_: Fock state number
            for each mode, of integer type. If `dim` is a tuple, the last dimension of
            `number` should match the length of `dim`.

    Returns:
        _(array of shape (..., n, n))_ Density matrix of the Fock state or tensor
            product of Fock states, with _n = prod(dims)_.

    Examples:
        Single-mode Fock state $\ket{1}$:
        >>> dq.fock_dm(3, 1)
        Array([[0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)

        Batched single-mode Fock states $\{\ket{0}\!, \ket{1}\!, \ket{2}\}$:
        >>> dq.fock_dm(3, [0, 1, 2])
        Array([[[1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
               [[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
               [[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j]]], dtype=complex64)

        Multi-mode Fock state $\ket{1,0}$:
        >>> dq.fock_dm((3, 2), (1, 0))
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)

        Batched multi-mode Fock states $\{\ket{0,0}\!, \ket{0,1}\!, \ket{1,1}\!,
        \ket{2,0}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock_dm((3, 2), number).shape
        (4, 6, 6)
    """
    return todm(fock(dim, number))


def basis(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    """Alias of [`dq.fock()`][dynamiqs.fock]."""
    return fock(dim, number)


def basis_dm(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    """Alias of [`dq.fock_dm()`][dynamiqs.fock_dm]."""
    return fock_dm(dim, number)


def coherent(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the ket of a coherent state or a tensor product of coherent states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array_like of shape (...) or (..., len(dim)))_: Coherent state
            amplitude for each mode. If `dim` is a tuple, the last dimension of
            `alpha` should match the length of `dim`.

    Returns:
        _(array of shape (..., n, 1))_ Ket of the coherent state or tensor product of
            coherent states, with _n = prod(dims)_.

    Examples:
        Single-mode coherent state $\ket{\alpha}$:
        >>> dq.coherent(4, 0.5)
        Array([[0.882+0.j],
               [0.441+0.j],
               [0.156+0.j],
               [0.047+0.j]], dtype=complex64)

        Batched single-mode coherent states $\{\ket{\alpha_0}\!, \ket{\alpha_1}\}$:
        >>> dq.coherent(4, [0.5, 0.5j])
        Array([[[ 0.882+0.j   ],
                [ 0.441+0.j   ],
                [ 0.156+0.j   ],
                [ 0.047+0.j   ]],
        <BLANKLINE>
               [[ 0.882+0.j   ],
                [ 0.   +0.441j],
                [-0.156+0.j   ],
                [ 0.   -0.047j]]], dtype=complex64)

        Multi-mode coherent state $\ket{\alpha}\otimes\ket{\beta}$:
        >>> dq.coherent((2, 3), (0.5, 0.5j))
        Array([[ 0.775+0.j   ],
               [ 0.   +0.386j],
               [-0.146+0.j   ],
               [ 0.423+0.j   ],
               [ 0.   +0.211j],
               [-0.08 +0.j   ]], dtype=complex64)

        Batched multi-mode coherent states $\{\ket{\alpha_0}\otimes\ket{\beta_0}\!,
        \ket{\alpha_1}\otimes\ket{\beta_1}\}$:
        >>> alpha = [(0.5, 0.5j), (0.5j, 0.5)]
        >>> dq.coherent((4, 6), alpha).shape
        (2, 24, 1)
    """
    dim = jnp.asarray(dim)
    alpha = jnp.asarray(alpha)
    check_type_int(dim, 'dim')

    # check if dim is a single value or a tuple
    if dim.ndim > 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # if dim is an integer, convert shapes dim: () -> (1,) and alpha: (...) -> (..., 1)
    if dim.ndim == 0:
        dim = dim[None]
        alpha = alpha[..., None]

    # check if alpha has shape (..., len(ndim))
    if alpha.shape[-1] != dim.shape[-1]:
        raise ValueError(
            'Argument `alpha` must have shape `(...)` or `(..., len(dim))`, but'
            f' has shape alpha.shape={alpha.shape}.'
        )

    # compute all kets
    alpha = alpha.swapaxes(0, -1)  # (len(dim), ...)
    kets = [displace(d, a) @ fock(d, 0) for d, a in zip(dim, alpha)]
    return tensor(*kets)


def coherent_dm(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the density matrix of a coherent state or a tensor product of coherent
    states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array_like of shape (...) or (..., len(dim)))_: Coherent state
            amplitude for each mode. If `dim` is a tuple, the last dimension of
            `alpha` should match the length of `dim`.

    Returns:
        _(array of shape (..., n, n))_ Density matrix of the coherent state or tensor
            product of coherent states, with _n = prod(dims)_.

    Examples:
        Single-mode coherent state $\ket{\alpha}$:
        >>> dq.coherent_dm(4, 0.5)
        Array([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
               [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
               [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
               [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]], dtype=complex64)

        Batched single-mode coherent states $\{\ket{\alpha_0}\!, \ket{\alpha_1}\}$:
        >>> dq.coherent_dm(4, [0.5, 0.5j]).shape
        (2, 4, 4)

        Multi-mode coherent state $\ket{\alpha}\otimes\ket{\beta}$:
        >>> dq.coherent_dm((2, 3), (0.5, 0.5j)).shape
        (6, 6)

        Batched multi-mode coherent states $\{\ket{\alpha_0}\otimes\ket{\beta_0}\!,
        \ket{\alpha_1}\otimes\ket{\beta_1}\}$:
        >>> alpha = [(0.5, 0.5j), (0.5j, 0.5)]
        >>> dq.coherent_dm((4, 6), alpha).shape
        (2, 24, 24)
    """
    return todm(coherent(dim, alpha))


def ground() -> Array:
    r"""Returns the eigenvector with eigenvalue -1 of the Pauli $\sigma_z$ operator.

    It is defined by $\ket{g} = \begin{pmatrix}0\\1\end{pmatrix}$.

    Note:
        This function is named `ground` because $\ket{g}$ is the lower energy state of
        a two-level system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(array of shape (2, 1))_ Ket $\ket{g}$.

    Examples:
        >>> dq.ground()
        Array([[0.+0.j],
               [1.+0.j]], dtype=complex64)
    """
    return jnp.array([[0], [1]], dtype=cdtype())


def excited() -> Array:
    r"""Returns the eigenvector with eigenvalue +1 of the Pauli $\sigma_z$ operator.

    It is defined by $\ket{e} = \begin{pmatrix}1\\0\end{pmatrix}$.

    Note:
        This function is named `excited` because $\ket{e}$ is the higher energy state of
        a two-level-system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(array of shape (2, 1))_ Ket $\ket{e}$.

    Examples:
        >>> dq.excited()
        Array([[1.+0.j],
               [0.+0.j]], dtype=complex64)
    """
    return jnp.array([[1], [0]], dtype=cdtype())
