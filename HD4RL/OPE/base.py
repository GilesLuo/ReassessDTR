from abc import ABC
from tianshou.data import ReplayBuffer
from typing import Dict, Optional, Any, Union
import numpy as np
import torch
from HD4RL.utils.data import TianshouDataset, collate_batch_seq2seq


class BaseOPE(ABC):
    """Base class for Off-Policy Evaluation (OPE) methods.

    This class provides the basic structure and mandatory methods for OPE.
    Subclasses should implement the specific OPE algorithms.

    all_soften: whether to use soften or not for all actions. If False, use target policy probabilities if available.
    """

    def __init__(self, buffers:Union[ReplayBuffer, Dict[str, ReplayBuffer]], num_actions: int, all_soften: bool = False):

        # Check if buffers is a single ReplayBuffer or a dictionary of ReplayBuffers
        self.all_soften = all_soften
        if self.all_soften:
            print("Using soften for all actions")
        else:
            print("Using target policy probabilities for all actions, if possible")
        if isinstance(buffers, ReplayBuffer):
            self.buffers = {'unnamed': buffers}
        else:
            self.buffers = buffers
        self.stack_num = self.get_stack_num()
        self.num_actions = num_actions
        self.device = None

        # Create episode indices dictionary
        self.episode_indices = {k:{} for k in self.buffers.keys()}  # Map episode number to buffer indices for each buffer
        self.datasets = {}  # Map episode number to dataset for each buffer
        for buffer_name, buffer in self.buffers.items():
            self.datasets[buffer_name] = TianshouDataset(buffer, stack_num=self.stack_num)
            episode_start_indices = self.datasets[buffer_name].episode_start_indices
            episode_lengths = np.diff(episode_start_indices, append=len(buffer))
            for i, (start, length) in enumerate(zip(episode_start_indices, episode_lengths)):
                self.episode_indices[buffer_name][i] = np.arange(start, start + length)

    def compute_target_probs(self, policy):
        """
        Precompute target policy probabilities for all buffers
        """
        policy.eval()
        try:
            self.device = next(policy.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        all_target_probs = {k:[] for k in self.buffers.keys()}

        for buffer_name, dataset in self.datasets.items():
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False,
                                                     collate_fn=collate_batch_seq2seq)
            with torch.no_grad():
                for batch in dataloader:
                    batch.to_torch(dtype=torch.float32)
                    # Compute target policy probabilities
                    output_batch = policy(batch)
                    target_probs = self.to_prob(output_batch)
                    all_target_probs[buffer_name].append(target_probs)

        precomputed_target_probs = {k:torch.cat(v, dim=0) for k, v in all_target_probs.items()}
        return precomputed_target_probs

    def align_stack_num(self, stack_num):
        """
        Align the stack_num of all buffers to the given stack_num
        """
        for buffer in self.buffers.values():
            buffer.stack_num = stack_num
        for dataset in self.datasets.values():
            dataset.stack_num = stack_num
        self.stack_num = stack_num

    def get_stack_num(self):
        stack_nums = [buffer.stack_num for buffer in self.buffers.values()]
        if not all(s == stack_nums[0] for s in stack_nums):
            raise ValueError(f"All buffers must have the same stack_num, we have {stack_nums}")
        return stack_nums[0]

    @staticmethod
    def norm2one(x:torch.Tensor):
        assert np.isclose(x.sum(-1), 1).all()
        return x / x.sum(dim=-1, keepdim=True)

    def to_prob(self, output_batch, soften=0.95):


        def to_torch_tensor(variable):
            if not isinstance(variable, torch.Tensor):
                variable = torch.tensor(variable)
            return variable

        """
        examine whether the values sum to 1, if not, convert to probability. Largest value will be 0.95, and the rest will be (1-0.95)/(num_actions - 1)
        """
        batch_size = output_batch.act.shape[0]
        if not self.all_soften and "prob" in output_batch.keys():
            values = output_batch.prob
        elif not self.all_soften and "probs" in output_batch.keys():
            values = output_batch.probs
        elif not self.all_soften and "log_prob" in output_batch.keys():
            values = torch.exp(output_batch.log_prob)
        else:  # just use the best action
            soften_zero = (1 - soften) / (self.num_actions - 1)
            act = torch.tensor(output_batch.act).to("cpu")
            values = torch.full([batch_size, self.num_actions], soften_zero)
            values.scatter_(-1, act.unsqueeze(-1), soften)
        values = to_torch_tensor(values).to("cpu").detach()
        values = self.norm2one(values)
        return values

    def evaluate(self, policy) -> Dict[str, float]:
        """Evaluate the policy using OPE and return relevant metrics.

        This method can be expanded in subclasses to provide additional
        metrics or results.

        :return: A dictionary containing OPE results and metrics.
        """
        raise NotImplementedError



