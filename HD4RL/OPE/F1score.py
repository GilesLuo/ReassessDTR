from sklearn.metrics import f1_score
import torch
from HD4RL.OPE.base import BaseOPE
import numpy as np
from typing import Dict, List, Union
from tqdm import tqdm


class F1(BaseOPE):
    def __init__(self, buffers, num_actions, mode: Union[List] = None, **kwargs):
        super().__init__(buffers, num_actions)
        if set(mode).intersection({None, "PatientWiseF1", "SampleWiseF1"}) == set():
            raise NotImplementedError(f"Unknown mode {mode}")
        self.mode = mode if mode is not None else ["PatientWiseF1", "SampleWiseF1"]

    def evaluate(self, policy) -> Dict[str, float]:
        precomputed_target_probs = self.compute_target_probs(policy)
        p, b = {buffer_name: [] for buffer_name in self.buffers.keys()}, {buffer_name: [] for buffer_name in
                                                                          self.buffers.keys()}
        for buffer_name, buffer in self.buffers.items():
            p[buffer_name] = torch.argmax(precomputed_target_probs[buffer_name], dim=-1).numpy()
            b[buffer_name] = buffer.act

        results = {}

        for buffer_name in tqdm(self.buffers.keys(), desc="computing F1"):
            for mode in self.mode:
                if mode == "SampleWiseF1":
                    results[f"{buffer_name}-{mode}"] = f1_score(p[buffer_name], b[buffer_name], average='micro')
                elif mode == "PatientWiseF1":
                    results[f"{buffer_name}-{mode}"] = self.patient_wise_f1(p[buffer_name], b[buffer_name],
                                                                           self.episode_indices[buffer_name])
        return results

    def patient_wise_f1(self, p, a, episode_indices):
        patient_f1_scores = []

        for indices in episode_indices.values():
            episode_predicted_actions = p[indices]
            episode_observed_actions = a[indices]
            f1 = f1_score(episode_observed_actions, episode_predicted_actions, average='micro')
            patient_f1_scores.append(f1)

        return np.mean(patient_f1_scores)