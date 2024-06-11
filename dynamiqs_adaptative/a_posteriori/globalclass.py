import equinox as eqx
import jax
from jax import Array
from ..time_array import TimeArray, ConstantTimeArray
from jaxtyping import PyTree, ScalarLike


# __all__ = ['Options']


class Globalclass(eqx.Module):
    # Objects needed to be accessed across the code

    projH: TimeArray | None = None
    projL: TimeArray | None = None
    dict: Array | None = None

    def __init__(
        self,
        projH: TimeArray | None = None,
        projL: TimeArray | None = None,
        dict: Array | None = None,

    ):
        
        self.projH = projH
        self.projL = projL
        self.dict = dict


