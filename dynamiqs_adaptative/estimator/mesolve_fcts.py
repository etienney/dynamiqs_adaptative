import jax
import jax.numpy as jnp

from ..time_array import ConstantTimeArray
from ..core._utils import _astimearray
from ..options import Options

from .utils.warnings import warning_bad_TimeArray, warning_size_too_small
from .degree_guesser import degree_guesser_nD_list
from .reshapings import mask, projection_nD, dict_nD
from .utils.utils import tensorisation_maker
from .inequalities import generate_rec_ineqs

def mesolve_estimator_init(options, H, jump_ops, tsave):
    # to init the arguments necessary for the estimator and the reshaping
    if options.estimator:
        t0 = tsave[0]
        H0 = H(t0)
        L0 = jnp.stack([L(t0) for L in jump_ops])
        lazy_tensorisation = options.tensorisation
        tmp_dic=options.__dict__
        if lazy_tensorisation is None:
            lazy_tensorisation = [len(H0[0])]
            tmp_dic['tensorisation']=lazy_tensorisation
        tensorisation = tensorisation_maker(lazy_tensorisation)
        if options.trunc_size is None:
            warning_bad_TimeArray(H, jump_ops)
            # Find the truncature needed to compute the estimator
            trunc_size = degree_guesser_nD_list(H0, L0, lazy_tensorisation)
            trunc_size = [2 * x.item() for x in jnp.array(trunc_size)]
            # for the "2 *" see [the article]
            tmp_dic['trunc_size'] = trunc_size
            warning_size_too_small(tensorisation, trunc_size)
        ineq_params = [(a - 1) - b for a, b in 
            zip(lazy_tensorisation, options.trunc_size)
        ] # -1 since list indexing starts at 0
        inequalities = generate_rec_ineqs(ineq_params)
        _mask = mask(H0, dict_nD(tensorisation, inequalities))
        Hred, *Lsred = [projection_nD(x, _mask) for x in [H0] + list(L0)]
        # reconvert to Timearray args
        Hred = _astimearray(Hred)
        Lsred = [_astimearray(L) for L in Lsred]
        # print(Hred, Lsred, jump_ops, type(Hred), type(Lsred), type(jump_ops))
        options = Options(**tmp_dic)
    else:
        # setup empty values
        options, Hred, Lsred, _mask = (
            options, None, None, None
        )

    return options, Hred, _mask, Lsred
