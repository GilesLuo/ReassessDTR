from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy as TianshouDQNPolicy
import copy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param gym.Space action_space: env's action space.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        assert self.action_type in ["continuous", "discrete"], \
            "Please specify action_space."

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.model(batch.obs, state=state, info=batch.info)
        if self.action_type == "discrete":
            act = logits.max(dim=1)[1].detach().cpu().numpy()
        else:
            act = logits
        probs = F.softmax(logits, dim=-1)
        return Batch(logits=logits, act=act, state=hidden, prob=probs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        self.optim.zero_grad()
        if self.action_type == "continuous":  # regression
            act = self(batch).act
            act_target = to_torch(batch.act, dtype=torch.float32, device=act.device)
        elif self.action_type == "discrete":  # classification
            act = self(batch).logits
            act_target = to_torch(batch.act, dtype=torch.long, device=act.device)
        else:
            raise ValueError("Unsupported action type.")

        loss = self.loss_fn(act, act_target)
        loss.backward()
        self.optim.step()
        if self.action_type == "continuous":  # regression
            return {"loss": loss.item()}
        elif self.action_type == "discrete":  # classification
            return {"loss": loss.item(),
                    "BatchWiseF1": f1_score(act_target.cpu().numpy(), act.argmax(dim=-1).cpu().numpy(),
                                            average="micro")}


class OfflineSARSAPolicy(TianshouDQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        result = self(batch, model="model", input="obs_next")

        act_next = batch.info["act_next"]
        if len(act_next.shape) == 2:  # stack num > 1
            act_next = act_next[:, -1]

        mask = act_next < 0 # mask for terminated states
        act_next[act_next < 0] = 0
        q_next = result.logits[np.arange(len(act_next)), act_next]
        # For terminated states, set Q value to 0
        q_next[mask] = 0.0
        return q_next


class DiscreteIQLPolicy(BasePolicy):
    """
    Implements the Implicit Quantile Learning (IQL) Policy.

    IQLPolicy is a model-free algorithm that aims to learn optimal policies
    by approximating the optimal action-value function (Q-function) without explicitly
    learning the policy. It utilizes a distributional approach to estimate the quantiles
    of the return distribution.

    Attributes:
        policy_network (nn.Module): Neural network model for policy approximation.
        qf1 (nn.Module): First Q-function network.
        qf2 (nn.Module): Second Q-function network.
        vf (nn.Module): V-function network.
        discount (float): Discount factor for future rewards.
        reward_scale (float): Scaling factor for rewards.
        policy_lr (float): Learning rate for the policy optimizer.
        qf_lr (float): Learning rate for the Q-function optimizers.
        tau (float): Coefficient for soft update of target networks.
        target_update_period (int): Frequency of target network updates.
        quantile (float): Quantile used in V-function loss calculation.
        beta (float): Temperature parameter for policy loss calculation.
        clip_score (float, optional): Clipping value for advantage scores in policy loss.
        device (str): Device to which tensors will be moved ('cpu' or 'cuda').
    """

    def __init__(self, policy_network, qf1, qf2, vf,
                 discount=0.99, reward_scale=1.0, policy_lr=1e-3, qf_lr=1e-3,
                 tau=1e-2, target_update_freq=1,
                 quantile=0.5, beta=1.0, clip_score=None,
                 device="cpu", **kwargs):
        super().__init__(**kwargs)
        self._iter = 0
        # Initialize networks
        self.policy_network = policy_network
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf

        # Initialize target networks for Q-functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        # Set up optimizers for each network
        self.policy_optimizer = optim.Adam(policy_network.parameters(), lr=policy_lr)
        self.qf1_optimizer = optim.Adam(qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optim.Adam(qf2.parameters(), lr=qf_lr)
        self.vf_optimizer = optim.Adam(vf.parameters(), lr=qf_lr)

        # Set hyperparameters
        self.discount = discount
        self.reward_scale = reward_scale
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.quantile = quantile
        self.beta = beta
        self.clip_score = clip_score
        self.device = device

        # Loss functions for Q-function and V-function
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

    def forward(self, batch, state=None, model='compute_actions'):
        # todo: does not work with rnn for now, but can be easily extended
        """
        Define forward pass for the policy.

        Args:
            batch (Batch): The batch of data for policy inference.
            state: The state for RNN-based policies, not used in this policy.
            model (str): The mode of operation, e.g., 'compute_actions'.

        Returns:
            Batch: The batch containing computed actions.
            state: The state for RNN-based policies, not used in this policy.
        """
        obs = batch.obs
        if model == 'compute_actions':
            with torch.no_grad():
                logit, state = self.policy_network(obs)
                dist = torch.distributions.Categorical(logits=logit)
                q1_pred_all, _ = self.qf1(obs)
                act = dist.sample()
            return Batch(logits=q1_pred_all, policy_logits=logit, state=state, act=act, prob=dist.probs)
        else:
            raise NotImplementedError

    def learn(self, batch, **kwargs):
        """
        Learning process of the IQLPolicy for discrete action settings.

        Args:
            batch (Batch): The batch of data for policy training.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing loss values.
        """
        batch.to_torch(dtype=torch.float32, device=self.device)
        rewards, terminals, actions = batch.rew, batch.done, batch.act.long()
        obs, next_obs = batch.obs, batch.obs_next

        # Compute Q-function loss for discrete actions
        q1_pred_all, _ = self.qf1(obs)
        q2_pred_all, _ = self.qf2(obs)
        q1_pred = q1_pred_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = q2_pred_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_vf_pred, _ = self.vf(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred.squeeze()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        # Compute V-function loss
        with torch.no_grad():
            q_pred = torch.min(self.target_qf1(obs)[0].gather(1, actions.unsqueeze(-1)).squeeze(-1),
                               self.target_qf2(obs)[0].gather(1, actions.unsqueeze(-1)).squeeze(-1))
        vf_pred, _ = self.vf(obs)
        vf_pred = vf_pred.squeeze()
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        # Compute Policy loss
        policy_logits = self.policy_network(obs)[0]
        policy_logpp = torch.distributions.Categorical(logits=policy_logits).log_prob(actions)
        adv = q_pred - vf_pred
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)
        policy_loss = (-policy_logpp * exp_adv.detach()).mean()

        # Update networks using backpropagation
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Perform soft updates of target networks
        if self._iter % self.target_update_freq == 0:
            self.soft_update(self.qf1, self.target_qf1, self.tau)
            self.soft_update(self.qf2, self.target_qf2, self.tau)

        self._iter += 1

        # Return loss values
        return {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "vf_loss": vf_loss.item(),
            "policy_loss": policy_loss.item()
        }


class CQLDQNPolicy(TianshouDQNPolicy):
    """Implementation of basic CQL-DQN algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> Q-values)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int target_update_freq: the target network update frequency.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param float alpha: the weight for the CQL loss.

    .. seealso::
        Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 0,
            alpha: float = 1.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, **kwargs)
        self.alpha = alpha

    def cql_loss(self, q_values, actions):
        """Computes the CQL loss for a batch of Q-values and actions."""
        actions = to_torch_as(actions, q_values)
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, actions.long().unsqueeze(1))
        return (logsumexp - q_a).mean()

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()

        self.optim.zero_grad()
        q = self(batch).logits
        returns = to_torch_as(batch.returns.flatten(), q)

        # DQN loss
        q_value = q[np.arange(len(q)), batch.act]
        loss = F.mse_loss(q_value, returns)

        # CQL loss
        cql_loss = self.cql_loss(q, batch.act)

        # Combine losses
        total_loss = loss + self.alpha * cql_loss
        total_loss.backward()
        self.optim.step()

        self._iter += 1
        return {
            "loss": total_loss.item(),
            "loss/dqn": loss.item(),
            "loss/cql": cql_loss.item(),
        }
