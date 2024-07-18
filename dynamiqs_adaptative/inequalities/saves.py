from .reshapings import extension_nD
from ..result import MEResult, Saved

def collect_save_ineq(results, actual_tensorisation, inequalities):
    tmp_dic=results.__dict__
    new_states = extension_nD(results.states, actual_tensorisation, inequalities)
    new_save = Saved(new_states, None, None)
    tmp_dic['_saved'] = new_save
    required_params = ['tsave', 'solver', 'gradient', 'options', '_saved', 'infos']
    filtered_tmp_dic = {k: tmp_dic[k] for k in required_params}
    results = MEResult(**filtered_tmp_dic)
    return results