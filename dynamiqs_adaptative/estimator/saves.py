from ..result import Result, Saved, Saved_estimator, MEResult
import jax.numpy as jnp
import numpy as np
import jax
from ..core.abstract_solver import State
from ..estimator.reshapings import projection_nD
from .estimator import compute_estimator
from ..estimator.utils.utils import put_together_results


def save_estimator(t, y, args):
    # Special save function for computing the estimator
    H, Ls, Hred, Lsred, _mask = args
    y_true = jnp.array(y.rho)
    rho = projection_nD(y_true, _mask)
    dest = compute_estimator(H, Ls, Hred, Lsred, rho, t)
    # dest = jnp.array(0)
    return Saved_estimator(y, None, None, dest, t)


def collect_saved_estimator(results):
    # format the output of the dx.diffeqsolve
    tmp_dic=results.__dict__
    corrected_time = results._saved.time
    corrected_time = corrected_time[jnp.isfinite(corrected_time)]
    true_steps = len(corrected_time)
    new_states = results.states.rho[:true_steps]
    new_dest = results.estimator[:true_steps]
    est = integrate_euler(new_dest, corrected_time)
    new_save = Saved_estimator(new_states, None, None, est, corrected_time)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def collect_saved_iteration(results, estimator_all):
    tmp_dic=results.__dict__
    corrected_time = results._saved.time
    corrected_time = corrected_time[jnp.isfinite(corrected_time)]
    true_steps = len(corrected_time)
    new_states = results.states.rho[:true_steps]
    new_dest = results.estimator[:true_steps]
    print(estimator_all)
    if len(estimator_all) == 0:
        est = integrate_euler(new_dest, corrected_time)
    else:
        est = integrate_euler(new_dest, corrected_time, estimator_all[-1][-1])
    print("output estimators (der, int, time)!:",new_dest, est, corrected_time)
    new_save = Saved_estimator(new_states, None, None, est, corrected_time)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def collect_saved_reshapings_final(results, rho_all, estimator_all):
    tmp_dic=results.__dict__
    new_states = jnp.array(put_together_results(rho_all, 1))
    est = put_together_results(estimator_all, 1, True)
    new_save = Saved_estimator(new_states, None, None, est, None)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results


def integrate_euler(derivatives, time, constant=0):
    # Initialize the integral array with a constant
    integral = np.zeros_like(time)
    # Perform Euler integration
    integral[0] = constant
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        if dt<0:
            print("raise alarms", time[i], time[i-1])
        integral[i] = integral[i-1] + derivatives[i-1] * dt + constant
    return integral
