from importlib.metadata import version

from . import dark
from .options import *
from .plots import *
from .result import *
from .solvers import *
from .time_array import *
from .utils import *
from .estimator import *

# get version from pyproject.toml
__version__ = version(__package__)

# set default matmul precision to 'highest'
set_matmul_precision('highest')
