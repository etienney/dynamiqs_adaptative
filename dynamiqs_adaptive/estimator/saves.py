from ..result import Result, Saved, Saved_estimator, MEResult
import jax.numpy as jnp
import numpy as np
import jax
from ..core.abstract_solver import State
from ..estimator.reshapings import projection_nD, extension
from .estimator import compute_estimator
from ..estimator.utils.utils import put_together_results
from .._utils import cdtype
from ..estimator.utils.utils import integrate_euler


def save_estimator(t, y, args):
    # Special save function for computing the estimator's derivative
    H, Ls, Hred, Lsred, _mask, _ = args
    y_true = jnp.array(y.rho)
    rho = projection_nD(y_true, _mask)
    dest = compute_estimator(H, Ls, Hred, Lsred, rho, t)
    est = y.err + dest * (t.astype(cdtype())-y.t)
    t_saved = t.astype(cdtype())
    tmp_dic=y.__dict__
    tmp_dic['rho'] = rho
    tmp_dic['err'] = est
    tmp_dic['t'] = t_saved
    required_params = ['rho', 'derr', 'err', 't']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    rho_saved = State(**filtered_tmp_dic)
    return Saved_estimator(rho_saved, None, None, dest, est, t, None)


def collect_saved_estimator(results):
    # format the output of the dx.diffeqsolve
    tmp_dic=results.__dict__
    corrected_time = results._saved.time
    corrected_time = corrected_time[jnp.isfinite(corrected_time)]
    true_steps = len(corrected_time)
    new_states = results.states.rho[:true_steps]
    new_dest = results.destimator[:true_steps]
    est = results.estimator[:true_steps]
    new_save = Saved_estimator(
        new_states, None, None, new_dest, est, corrected_time, None
    )
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def collect_saved_iteration(results, estimator_all, options):
    tmp_dic=results.__dict__
    corrected_time = results._saved.time
    corrected_time = corrected_time[jnp.isfinite(corrected_time)]
    true_steps = len(corrected_time)
    corrected_states = results.states.rho[:true_steps]
    new_dest = results.destimator[:true_steps]
    new_est = results.estimator[:true_steps]
    inequalities = jnp.array([jnp.array(x) for x in (options.inequalities * true_steps)])
    # print("output estimators (der, int, time)!:", new_dest, est, corrected_time)
    new_save = Saved_estimator(
        jnp.array(corrected_states), None, None, new_dest, new_est, corrected_time, 
        inequalities
    )
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def collect_saved_reshapings_final(
        results, rho_all, estimator_all, time_all, inequalities_all
):
    # print("last esti", estimator_all)
    tmp_dic=results.__dict__
    new_states = jnp.array(put_together_results(rho_all, 2))
    est = put_together_results(estimator_all, 2, True)
    time = put_together_results(time_all, 2, True)
    ineqs = put_together_results(inequalities_all, 2)
    ineqs = reduce_list_reshapings(ineqs)
    new_save = Saved_estimator(new_states, None, None, est, est, time, ineqs)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def reduce_list_reshapings(lst):
    reduced_lst = []
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst) and lst[j][0] == lst[i][0]:
            j += 1
        reduced_lst.append([lst[i][0], lst[i][1], [lst[i][2], lst[j-1][2]]])
        i = j
    return reduced_lst

