from importlib.metadata import version

from . import dark
from .options import *
from .plots import *
from .result import *
from .solvers import *
from .time_array import *
from .utils import *

from .a_posteriori import *
from .time_array import TimeArray

# get version from pyproject.toml
__version__ = version(__package__)
