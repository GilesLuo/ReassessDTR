import tianshou.data
from tianshou.data import ReplayBuffer
from tianshou.policy import BasePolicy
from typing import Dict, Optional, Union, List
from HD4RL.OPE.base import BaseOPE
import numpy as np
import torch
from HD4RL.utils.data import TianshouDataset, collate_batch_seq2seq
from tqdm import tqdm
from tianshou.data import Batch
from torch.nn.utils.rnn import pad_sequence


class ImportanceSampling(BaseOPE):
    """Importance Sampling based Off-Policy Evaluation.

    This class uses the Importance Sampling method to estimate the expected reward
    of a policy based on observed data and the behavior policy.
    """

    def __init__(self, buffers: Union[ReplayBuffer, Dict[str, ReplayBuffer]], num_action,
                 behavior_policy: Optional[Union[BasePolicy, Dict[str, BasePolicy]]] = None,
                 value_function: Optional[Dict[str, BasePolicy]] = None, gamma: float = 0.99,
                 modes: Optional[Union[str, List[str]]] = None, all_soften=False):
        """
        Initialize the Importance Sampling class.
        Choose modes from ["IS", "WIS", "WIS_bootstrap", "WIS_truncated", "WIS_bootstrap_truncated", "WIS_mortality"].
        WIS_mortality is a special mode that returns the mortality rate converted from WIS_bootstrap_truncated
        :param buffers:
        :param num_action:
        :param behavior_policy:
        :param gamma:
        :param modes:
        """
        super().__init__(buffers, num_action, all_soften)
        # accept behaviour policy in 3 forms:
        # 1. None: use empirical probs 2. single policy for all buffers 3. dict of policies
        if modes is None:
            modes = ["WIS", "WIS_bootstrap", "WIS_truncated", "WIS_bootstrap_truncated", "WIS_mortality"]
        self.modes = modes
        self.value_function = value_function["model"] if value_function is not None else None
        if self.value_function is None and ("DR" in self.modes or "WDR" in self.modes):
            raise ValueError("DoublyRobust needs a value function but None is provided.")
        if self.value_function is not None and ("DR" not in self.modes and "WDR" not in self.modes):
            raise ValueError("DoublyRobust is not in the modes but a value function is provided.")

        self.gamma = gamma
        self.behaviour_policy = behavior_policy["model"]

        if self.behaviour_policy is not None:
            self.behaviour_policy.eval()
        if self.value_function is not None:
            self.value_function.eval()

        self.behavioural_datasets = {k: TianshouDataset(v, stack_num=behavior_policy["stack_num"]) for k, v in
                                     self.buffers.items()} if self.behaviour_policy is not None else None
        self.value_datasets = {k: TianshouDataset(v, stack_num=value_function["stack_num"]) for k, v in
                               self.buffers.items()} if self.value_function is not None else None

        self.data_behavior_probs = self.precompute_probs()
        self.all_q, self.data_q = self.precompute_q_values() if self.value_function is not None else [None, None]

    def evaluate(self, target_policy):
        """Estimate the expected reward using Importance Sampling."""
        raw_ratios, raw_rewards, V = self.get_raw_data(target_policy)

        results = {}
        for buffer_name in self.buffers:
            episode_indices = self.episode_indices[buffer_name]
            with tqdm(self.modes) as pbar:
                for mode in self.modes:
                    pbar.set_description(f"Computing {mode} for {buffer_name}")
                    if mode in ["DR", "WDR"]:
                        results[f"{buffer_name}-{mode}"] = self.compute_dr_estimate(raw_ratios[buffer_name],
                                                                                    raw_rewards[buffer_name],
                                                                                    self.data_q[buffer_name],
                                                                                    V[buffer_name],
                                                                                    episode_indices,
                                                                                    weighted=mode == "WDR")
                    elif mode in ["PDDR", "PDWDR"]:
                        results[f"{buffer_name}-{mode}"] = self.compute_per_step_dr_estimate(raw_ratios[buffer_name],
                                                                                             raw_rewards[buffer_name],
                                                                                             self.data_q[
                                                                                                 buffer_name],
                                                                                             episode_indices,
                                                                                             weighted=mode == "PDWDR")

                    else:
                        if mode == "IS":
                            results[f"{buffer_name}-{mode}"] = self.normed_IS(raw_ratios[buffer_name],
                                                                              raw_rewards[buffer_name],
                                                                              episode_indices, self.gamma)
                            continue
                        weighted = True if "WIS" in mode else False
                        bootstrapping = 100 if "bootstrap" in mode else 1
                        truncation = 1000. if "truncated" in mode else 0.0
                        gamma = 1.0 if "mortality" in mode else self.gamma
                        estimate = np.mean(
                            [self._estimate_once(raw_ratios[buffer_name], raw_rewards[buffer_name], episode_indices,
                                                 bootstrapping > 1, weighted, truncation, gamma) for _ in
                             range(bootstrapping)])
                        results[f"{buffer_name}-{mode}"] = estimate

                        if mode == "WIS_bootstrap" and "WIS_mortality" in self.modes:
                            # Convert the estimate from WIS_bootstrap_truncated to WIS_mortality (placeholder logic)
                            raise NotImplementedError("WIS_mortality is not implemented yet")
                    pbar.update()
        return results

    @staticmethod
    def batch2buffer(buffer_cls, batch: tianshou.data.Batch) -> "ReplayBuffer":
        size = len(batch.obs)
        assert all(len(dset) == size for dset in [batch.obs, batch.act, batch.rew, batch.terminated,
                                                  batch.truncated, batch.done, batch.obs_next]), \
            "Lengths of all batch variables need to be equal."
        buf = buffer_cls(size)
        if size == 0:
            return buf
        buf.set_batch(batch)
        buf._size = size
        return buf

    def precompute_q_values(self, batch_size=1024):
        """
        Precompute q value for all buffers from offline SARSA
        """
        all_q = {}  # all q values for all episodes
        q_data = {}  # q values for all actions taken in the dataset
        for buffer_name, dataset in self.value_datasets.items():
            buffer_q_data = []
            buffer_all_q = []
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                     collate_fn=collate_batch_seq2seq)
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Precomputing q values for {buffer_name}"):
                    batch.to_torch(dtype=torch.float32)
                    obs = batch.obs
                    logits = self.value_function(Batch(obs=obs, info={})).logits
                    buffer_q_data.append(logits[np.arange(len(batch.act)), batch.act.long()].cpu().detach())
                    buffer_all_q.append(logits.cpu().detach())
                q_data[buffer_name] = torch.cat(buffer_q_data, dim=0).to("cpu").numpy()
                all_q[buffer_name] = torch.cat(buffer_all_q, dim=0).to("cpu").numpy()
        return all_q, q_data

    def precompute_probs(self, batch_size=1024):
        """
        Precompute behavior policy probabilities for all buffers
        """

        data_behavior_probs = {}
        if self.behaviour_policy is None:
            for buffer_name, buffer in self.buffers.items():
                data_behavior_probs[buffer_name] = np.ones([buffer.obs.shape[0], ], dtype=np.float32)
            return data_behavior_probs

        for buffer_name, dataset in self.behavioural_datasets.items():
            buffer_data_probs = []
            with torch.no_grad():
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                         collate_fn=collate_batch_seq2seq)
                for batch in tqdm(dataloader, desc=f"Precomputing behavior probabilities for {buffer_name}"):
                    batch.to_torch(dtype=torch.float32)
                    obs = batch.obs
                    behavior_prob = self.behaviour_policy(Batch(obs=obs, info={})).prob
                    buffer_data_probs.append(behavior_prob[np.arange(len(batch.act)), batch.act.long()])
                data_behavior_probs[buffer_name] = torch.cat(buffer_data_probs, dim=0).to("cpu").numpy()
        return data_behavior_probs

    def get_raw_data(self, target_policy):
        """ Compute raw ratios and reward for all buffers"""
        raw_ratios = {}
        raw_rewards = {}
        Vs = {}
        # Pre-compute target probabilities outside the loop
        target_probs_all = self.compute_target_probs(target_policy)

        for buffer_name, buffer in tqdm(self.buffers.items(),
                                        desc="Computing importance sampling for all testing buffers"):
            # Convert behavior probabilities to tensors once
            behavior_probs = self.data_behavior_probs[buffer_name]
            target_probs = target_probs_all[buffer_name].numpy()

            # get V by V= sum_a pi(a|s) Q(s,a)
            if self.all_q is not None:
                V = np.sum(target_probs * self.all_q[buffer_name], axis=1)
            else:
                V = None

            # pi(a_data|s_data)
            target_probs = target_probs[np.arange(len(buffer.act)), buffer.act]

            Vs[buffer_name] = V
            raw_ratios[buffer_name] = target_probs / behavior_probs
            raw_rewards[buffer_name] = buffer.rew
        return raw_ratios, raw_rewards, Vs

    def _estimate_once(self, raw_ratio, raw_reward, episode_indices, bootstrap, weighted, truncation, gamma) -> float:
        if bootstrap:
            sampled_episodes = np.random.choice(list(episode_indices.keys()), len(episode_indices), replace=True)
        else:
            sampled_episodes = list(episode_indices.keys())

        ratios = np.minimum(raw_ratio, truncation) if truncation > 0 else raw_ratio[:]
        all_discounted_returns = []
        all_episode_ratios = []
        for episode in sampled_episodes:
            indices = episode_indices[episode]
            episode_rewards = raw_reward[indices]

            # Apply discount factor to rewards for the episode
            discounts = gamma ** np.arange(len(episode_rewards))
            discounted_return = np.sum(episode_rewards * discounts)

            all_discounted_returns.append(discounted_return)
            all_episode_ratios.append(np.prod(ratios[indices]))
        # Compute the estimate using the ratios and the discounted returns
        estimate = np.sum(np.array(all_episode_ratios) * np.array(all_discounted_returns))
        if weighted:
            estimate /= np.sum(all_episode_ratios)
        return float(estimate)

    def normed_IS(self, raw_ratio, raw_reward, episode_indices, gamma) -> float:
        """

        """
        # ratios = raw_ratio[:]
        # estimate = 0
        # for episode in episode_indices:
        #     indices = episode_indices[episode]
        #     episode_rewards = raw_reward[indices]
        #
        #     # Apply discount factor to rewards for the episode
        #     discounts = gamma ** np.arange(len(episode_rewards))
        #     discounted_return = np.sum(episode_rewards * discounts)
        #     estimate += np.sum(discounted_return * ratios[indices]) / np.sum(ratios)
        # return float(estimate)
        raise NotImplementedError("normed_IS is not implemented yet")

    def compute_dr_estimate(self, raw_ratio, raw_reward, data_q, V, episode_indices, weighted):
        """Compute the Doubly Robust estimate for a buffer using vectorized operations and PyTorch."""
        if weighted:
            raise NotImplementedError("Weighted DR is not implemented yet")
        # Create tensors for each episode's data
        seq_len = np.array([len(indices) for indices in episode_indices.values()])
        ratios_tensors = [torch.tensor(raw_ratio[indices]) for indices in episode_indices.values()]
        rewards_tensors = [torch.tensor(raw_reward[indices]) for indices in episode_indices.values()]
        q_tensors = [torch.tensor(data_q[indices]) for indices in episode_indices.values()]
        V_tensors = [torch.tensor(V[indices]) for indices in episode_indices.values()]

        # Pad the sequences with zeros
        rho = pad_sequence(ratios_tensors, batch_first=True, padding_value=0)
        r = pad_sequence(rewards_tensors, batch_first=True, padding_value=0)
        Q = pad_sequence(q_tensors, batch_first=True, padding_value=0)
        V = pad_sequence(V_tensors, batch_first=True, padding_value=0)

        dr_estimates = torch.zeros_like(r)
        # here i := H+1-t, we want to iterate from H to 0. By definition, V_0 = 0, when t = H+1
        for i in range(r.size(1)):
            dr_last = dr_estimates[:, i - 1] if i > 0 else torch.zeros_like(r[:, 0])
            dr_estimates[:, i] = V[:, i] + rho[:, i] * (r[:, i] + self.gamma * dr_last - Q[:, i])
        # assume the last non-zero element of V is seq_len - 1
        assert (np.array([np.nonzero(V[i]).max() for i in range(len(V))]) == seq_len - 1).all()

        # take the last element of dr_estimate for each episode, since sequence length is variable
        dr_estimates = dr_estimates[np.arange(dr_estimates.size(0)), seq_len - 1]
        # bound max and min values for each episode
        min_dr = torch.tensor([-i for i in seq_len])
        max_dr = torch.zeros(len(V))
        dr_estimates = torch.clamp(dr_estimates, min=min_dr, max=max_dr).mean()

        # Sum the DR estimates for each episode to get the total estimate per episode
        return float(dr_estimates.cpu().numpy())

    def compute_per_step_dr_estimate(self, *args, **kwargs):
        raise NotImplementedError("Per-step DR is not implemented yet")


class ImportanceRatio(ImportanceSampling):
    def __init__(self, buffers: Union[ReplayBuffer, Dict[str, ReplayBuffer]], num_action,
                 behavior_policy: Optional[Union[BasePolicy, Dict[str, BasePolicy]]] = None,
                 value_function: Optional[Dict[str, BasePolicy]] = None, gamma: float = 0.99,
                 modes: Optional[Union[str, List[str]]] = None):
        super().__init__(buffers, num_action, behavior_policy, value_function, gamma, modes)

    def evaluate(self, target_policy):
        raw_ratios, raw_rewards, V = self.get_raw_data(target_policy)
        results = {}
        for buffer_name in self.buffers:
            results[f"{buffer_name}-ratios"] = raw_ratios[buffer_name]
            results[f"{buffer_name}-rewards"] = raw_rewards[buffer_name]
            results[f"{buffer_name}-V"] = V[buffer_name]
        return results
