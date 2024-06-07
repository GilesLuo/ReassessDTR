from HD4RL.OPE.ImportanceSampling import ImportanceSampling, ImportanceRatio
from HD4RL.OPE.F1score import F1
from HD4RL.OPE.TD import TemporalDifference as TD
from HD4RL.OPE.RMSE import DoseRMSE

class OPE_wrapper:
    def __init__(self, OPE_names, buffers, behavior_policy, value_function, num_actions, gamma=0.99, all_soften=False):
        is_names = []
        F1_names = []
        self.estimators = []
        for OPE_name in OPE_names:
            if OPE_name in ['IS', 'FQE', 'WIS', 'WIS_bootstrap', "WIS_mortality", 'WIS_truncated',
                            'WIS_bootstrap_truncated', "DR", "WDR", "PDDR", "PDWDR"]:
                is_names.append(OPE_name)
            elif OPE_name in ['PatientWiseF1', "SampleWiseF1"]:
                F1_names.append(OPE_name)
            elif OPE_name == "TD":
                self.estimators.append(TD(buffers=buffers, num_actions=num_actions, gamma=gamma))
            elif OPE_name == "ratio":
                self.estimators.append(ImportanceRatio(buffers=buffers, num_action=num_actions, gamma=gamma,
                                                       behavior_policy=behavior_policy, modes=None,
                                                       value_function=None))
            elif OPE_name == "doseRMSE":
                self.estimators.append(DoseRMSE(buffers, num_actions, gamma=0.99))
            else:
                raise NotImplementedError(f"Unknown OPE name {OPE_name}")
        if len(is_names) > 0:
            self.estimators.append(
                ImportanceSampling(buffers=buffers, num_action=num_actions, gamma=gamma,
                                   behavior_policy=behavior_policy, modes=is_names, value_function=value_function,
                                   all_soften=all_soften))
        if len(F1_names) > 0:
            self.estimators.append(F1(buffers=buffers, num_actions=num_actions, mode=F1_names))
        self.OPE_names = [f"{buffer_name}-{OPE_name}" for buffer_name in buffers.keys() for OPE_name in OPE_names]

    def __call__(self, *args, **kwargs):
        results = {}
        for estimator in self.estimators:
            r = estimator.evaluate(*args, **kwargs)
            assert set(r.keys()).intersection(set(results.keys())) == set(), "duplicate keys in different OPEs"
            results.update(r)
        return results

    def keys(self):
        return self.OPE_names

    def align_stack_num(self, stack_num):
        for estimator in self.estimators:
            estimator.align_stack_num(stack_num)
