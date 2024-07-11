from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from math import ceil

import matplotlib
import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ['gridplot', 'mplstyle']
# __all__ = [
#     'linmap',
#     'figax',
#     'optional_ax',
#     'gridplot',
#     'mplstyle',
#     'integer_ticks',
#     'sample_cmap',
#     'minorticks_off',
#     'ket_ticks',
#     'bra_ticks',
#     'add_colorbar',
# ]


def linmap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Map $x$ linearly from $[a,b]$ to $[c,d]$."""
    return (x - a) / (b - a) * (d - c) + c


def figax(w: float = 7.0, h: float | None = None, **kwargs) -> tuple[Figure, Axes]:
    """Return a figure with specified width and length."""
    if h is None:
        h = w / 1.6
    return plt.subplots(1, 1, figsize=(w, h), constrained_layout=True, **kwargs)


def optional_ax(func: callable) -> callable:
    """Decorator to build an `Axes` object to pass as an argument to a plot
    function if it wasn't passed by the user.

    Examples:
        Replace
        ```
        def myplot(ax=None):
            if ax is None:
                _, ax = plt.subplots(1, 1)

            # ...
        ```
        by
        ```
        @optax
        def myplot(ax=None):
            # ...
        ```
    """

    @wraps(func)
    def wrapper(  # noqa: ANN202
        *args, ax: Axes | None = None, w: float = 7.0, h: float | None = None, **kwargs
    ):
        if ax is None:
            _, ax = figax(w=w, h=h)
        return func(*args, ax=ax, **kwargs)

    return wrapper


def gridplot(
    n: int,
    nrows: int = 1,
    *,
    w: float = 3.0,
    h: float | None = None,
    sharexy: bool = False,
    **kwargs,
) -> tuple[Figure, Iterable[Axes]]:
    """Returns a figure and an iterator of subplots organised in a grid.

    Warning:
        Documentation redaction in progress.

    Note:
        This method is a shortcut to Matplotlib
        [`plt.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

    Examples:
        For example, to plot six different curves:

        >>> x = jnp.linspace(0, 1, 101)
        >>> ys = [jnp.sin(f * 2 * jnp.pi * x) for f in range(6)]  # (6, 101)

        Replace the usual Matplotlib code

        >>> fig, axs = plt.subplots(
        ...     2, 3, figsize=(3 * 3.0, 2 * 3.0), sharex=True, sharey=True
        ... )
        >>> for i, y in enumerate(ys):
        ...     axs[i // 3][i % 3].plot(x, y)
        [...]
        >>> fig.tight_layout()

        by

        >>> _, axs = dq.gridplot(6, 2, sharexy=True)  # 6 subplots, 2 rows
        >>> for y in ys:
        ...     next(axs).plot(x, y)
        [...]
        >>> renderfig('gridplot')

        ![gridplot](/figs_code/gridplot.png){.fig}
    """
    h = w if h is None else h
    ncols = ceil(n / nrows)
    figsize = (w * ncols, h * nrows)

    if sharexy:
        kwargs['sharex'] = True
        kwargs['sharey'] = True

    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, **kwargs
    )

    return fig, iter(axs.flatten())


colors = {
    'blue': '#0c5dA5',
    'red': '#ff6b6b',
    'turquoise': '#2ec4b6',
    'yellow': '#ffc463',
    'grey': '#9e9e9e',
    'purple': '#845b97',
    'brown': '#c0675c',
    'darkgreen': '#20817d',
    'darkgrey': '#666666',
}


def mplstyle(*, usetex: bool = False):
    r"""Set custom Matplotlib style.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> x = jnp.linspace(0, 2 * jnp.pi, 101)
        >>> ys = [jnp.sin(x), jnp.sin(2 * x), jnp.sin(3 * x)]
        >>> default_mpl_style()

        Before (default Matplotlib style):

        >>> fig, ax = plt.subplots(1, 1)
        >>> for y in ys:
        ...     ax.plot(x, y)
        [...]
        >>> ax.set(xlabel=r'$x$', ylabel=r'$\sin(x)$')
        [...]
        >>> renderfig('mplstyle_before')

        ![mplstyle_before](/figs_code/mplstyle_before.png){.fig}

        After (dynamiqs Matplotlib style):

        >>> dq.mplstyle()
        >>> fig, ax = plt.subplots(1, 1)
        >>> for y in ys:
        ...     ax.plot(x, y)
        [...]
        >>> ax.set(xlabel=r'$x$', ylabel=r'$\sin(x)$')
        [...]
        >>> renderfig('mplstyle_after')

        ![mplstyle_after](/figs_code/mplstyle_after.png){.fig}
    """
    plt.rcParams.update(
        {
            # xtick
            'xtick.direction': 'in',
            'xtick.major.size': 4.5,
            'xtick.minor.size': 2.5,
            'xtick.major.width': 1.0,
            'xtick.labelsize': 12,
            'xtick.minor.visible': True,
            # ytick
            'ytick.direction': 'in',
            'ytick.major.size': 4.5,
            'ytick.minor.size': 2.5,
            'ytick.major.width': 1.0,
            'ytick.labelsize': 12,
            'ytick.minor.visible': True,
            # axes
            'axes.facecolor': 'white',
            'axes.grid': False,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'axes.linewidth': 1.0,
            'axes.prop_cycle': cycler('color', colors.values()),
            # grid
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            # legend
            'legend.frameon': False,
            'legend.fontsize': 12,
            # figure
            'figure.facecolor': 'white',
            'figure.dpi': 72,
            'figure.figsize': (7, 7 / 1.6),
            # other
            'savefig.facecolor': 'white',
            'font.size': 12,
            'scatter.marker': 'o',
            'lines.linewidth': 2.0,
            # fonts
            'text.usetex': usetex,
            'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{braket}',
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            # if usetex=False, matplotlib uses mathtext, for which we choose the STIX
            # font which is designed to blend well with Times
            'mathtext.fontset': 'stix',
        }
    )


def integer_ticks(axis: Axis, n: int, all: bool = True):  # noqa: A002
    if all:
        axis.set_ticks(range(n))
        minorticks_off(axis)
    else:
        # let maptlotlib choose major ticks location but restrict to integers
        axis.set_major_locator(MaxNLocator(integer=True))
        # fix minor ticks to integer locations only
        axis.set_minor_locator(MultipleLocator(1))

    # format major ticks as integer
    axis.set_major_formatter(lambda x, _: f'{int(x)}')


def ket_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$| {label.get_text()} \rangle$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)


def bra_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$\langle {label.get_text()} |$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)


def sample_cmap(name: str, n: int, alpha: float = 1.0) -> np.ndarray:
    sampled_cmap = matplotlib.colormaps[name](np.linspace(0, 1, n))
    sampled_cmap[:, -1] = alpha
    return sampled_cmap


def minorticks_off(axis: Axis):
    axis.set_minor_locator(NullLocator())


def add_colorbar(
    ax: Axes, cmap: str, norm: Normalize, *, size: str = '5%', pad: str = '5%'
) -> Axes:
    # append a new axes on the right with the same height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable=mappable, cax=cax)
    cax.grid(False)
    return cax
