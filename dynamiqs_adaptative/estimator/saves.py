from ..result import Result, Saved, Saved_estimator, MEResult
import jax.numpy as jnp
import jax

def save_estimator(t, y, args):
    # Special save function for computing the estimator
    return Saved_estimator(y, None, None, None, t)


def collect_saved_estimator(results):
    # format the output of the dx.diffeqsolve
    tmp_dic=results.__dict__
    corrected_time = results._saved.time
    corrected_time = corrected_time[jnp.isfinite(corrected_time)]
    true_steps = len(corrected_time)
    new_states = results.states[:true_steps]
    new_save = Saved_estimator(new_states, None, None, None, corrected_time)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results
