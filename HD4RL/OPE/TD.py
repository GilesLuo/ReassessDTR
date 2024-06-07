from HD4RL.OPE.base import BaseOPE
import torch
import numpy as np
from HD4RL.utils.data import collate_batch_seq2seq
from typing import Dict


class TemporalDifference(BaseOPE):
    """Class for calculating Temporal Difference (TD) error.
    TD error is negative for optimisation purposes."""

    def __init__(self, buffers, num_actions, gamma=0.99):
        """
        Initialize the TemporalDifference class.

        :param buffers: Replay buffers.
        :param num_actions: Number of actions in the environment.
        :param gamma: Discount factor for future rewards.
        """
        super().__init__(buffers, num_actions)
        self.gamma = gamma

    def compute_td_error(self, policy):
        """
        Compute the Temporal Difference error for all samples in the buffer.

        :param policy: The policy to evaluate.
        :return: A dictionary of TD errors for each buffer.
        """
        all_td_errors = {f"{k}-TD": [] for k in self.buffers.keys()}

        for buffer_name, dataset in self.datasets.items():
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False,
                                                     collate_fn=collate_batch_seq2seq)
            for batch in dataloader:
                batch.to_torch(dtype=torch.float32)
                q_pred = policy(batch).logits
                q_pred = q_pred.gather(1, batch.act.long().to(q_pred.device).unsqueeze(-1)).squeeze(-1)
                rewards = batch.rew

                # Calculate next Q values by SARSA
                act_next = batch.info["act_next"].to(int)
                if len(act_next.shape) == 2:  # stack num > 1
                    act_next = act_next[:, -1]
                mask = act_next < 0  # mask for terminated states
                act_next[act_next < 0] = 0
                q_next = policy(batch, model="model", input="obs_next").logits[np.arange(len(act_next)), act_next]
                q_next[mask] = 0.0

                # TD error calculation
                td_errors = rewards.to(q_pred.device) + self.gamma * q_next - q_pred
                td_errors_mse = torch.mean(td_errors ** 2)
                all_td_errors[f"{buffer_name}-TD"].append(td_errors_mse.item())
        for k in all_td_errors.keys():
            all_td_errors[k] = -np.mean(all_td_errors[k])
        return all_td_errors

    def evaluate(self, policy) -> Dict[str, float]:
        """
        Evaluate the policy using Temporal Difference and return relevant metrics.

        :param policy: The policy to evaluate.
        :return: A dictionary containing TD error statistics.
        """
        return self.compute_td_error(policy)
