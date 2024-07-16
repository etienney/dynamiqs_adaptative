from ..result import Result, Saved, Saved_estimator, MEResult
import jax.numpy as jnp
import numpy as np
import jax
from ..core.abstract_solver import State
from ..estimator.reshapings import projection_nD
from .estimator import compute_estimator


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
    # print("all pos ?", new_dest)
    # print("ordered ?", corrected_time)
    est = integrate_euler(new_dest, corrected_time)
    new_save = Saved_estimator(new_states, None, None, est, corrected_time)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results

def integrate_euler(derivatives, time):
    # Initialize the integral array
    integral = np.zeros_like(time)
    # Perform Euler integration
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        if dt<0:
            print("raise alarms", time[i], time[i-1])
        integral[i] = integral[i-1] + derivatives[i-1] * dt
    return integral
