import os
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

from HD4RL.core.base_obj import RLObjective
from HD4RL.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from HD4RL.utils.network import define_single_network


class DQNObjective(RLObjective):
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma,
                      lr,
                      stack_num,
                      linear,
                      cat_num,

                      # dqn hp
                      n_step,
                      target_update_freq,
                      is_double,
                      use_dueling,
                      icm_lr_scale=0,  # help="use intrinsic curiosity module with this lr scale"
                      icm_reward_scale=0,  # help="scaling factor for intrinsic curiosity reward"
                      icm_forward_loss_weight=0,  # help="weight for the forward model loss in ICM",
                      **kwargs
                      ):
        # define model
        net = define_single_network(self.state_shape, self.action_shape, use_dueling=use_dueling,
                                    use_rnn=stack_num > 1, device=self.device, linear=linear, cat_num=cat_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = DQNPolicy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,  # we will have a separate runner for double dqn
        )
        if icm_lr_scale > 0:
            feature_net = define_single_network(self.state_shape, 256, use_rnn=False, device=self.device)
            action_dim = int(np.prod(self.action_shape))
            feature_dim = feature_net.output_dim
            icm_net = IntrinsicCuriosityModule(
                feature_net.net,
                feature_dim,
                action_dim,
                hidden_sizes=[512],
                device=self.device
            )
            icm_optim = torch.optim.Adam(icm_net.parameters(), lr=lr)
            policy = ICMPolicy(
                policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
                icm_forward_loss_weight
            ).to(self.device)
        return policy

    def run(self, policy,
            eps_test,
            eps_train,
            eps_train_final,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            **kwargs
            ):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 10k steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * \
                      (eps_train - eps_train_final)
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
                stack_num=stack_num
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"],
                                  ignore_obs_next=False,
                                  save_only_last_obs=False,
                                  stack_num=stack_num
                                  )

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        # test train_collector and start filling replay buffer
        print("warm start replay buffer, this may take a while...")
        train_collector.collect(n_step=batch_size * self.meta_param["training_num"])
        # trainer

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        )
        return result




