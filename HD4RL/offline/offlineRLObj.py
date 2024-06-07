import numpy as np
from tianshou.policy import BCQPolicy, DiscreteBCQPolicy
from tianshou.utils.net.discrete import Actor as discreteActor
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import MLP, Actor
from tianshou.utils.net.continuous import VAE, Critic, Perturbation
from HD4RL.core.offpolicyRLObj import DQNObjective
from HD4RL.utils.network import define_single_network, Net
from HD4RL.offline.policy import ImitationPolicy, OfflineSARSAPolicy, DiscreteIQLPolicy, CQLDQNPolicy
import os
import gymnasium as gym
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from DTRGym import buffer_registry
from HD4RL.OPE import OPE_wrapper
from HD4RL.offline.offline_trainer import offline_trainer
from HD4RL.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from HD4RL.utils.data import load_buffer
from HD4RL.core.base_obj import RLObjective


class OfflineRLObjective(RLObjective):
    """
    1. val: OPE, test: OPE
    2. val: OPE, test: online/ online+OPE
    3. val: online, test: online
    """

    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, logger, device,
                 train_buffer_name=None, val_buffer_name=None, test_buffer_keyword=None, OPE_methods=None,
                 OPE_args=None,
                 test_online: bool = False, metric: str = None, early_stopping=10, **kwargs):
        super().__init__(env_name, hparam_space, device, logger=logger, multi_obj=False, early_stopping=early_stopping)
        if OPE_args is None:
            OPE_args = {}
        self.train_buffer_name = train_buffer_name
        try:
            self.train_buffer = load_buffer(buffer_registry.make(self.env_name, self.train_buffer_name))
            init_OPE = True
        except Exception as e:
            print(f"Error loading buffer {self.train_buffer_name}: {e}. This is not a bug if you are using "
                  f"offlineRLObjective as a placeholder for using its methods. All OPE estimators will be disabled.")
            self.train_buffer = None
            init_OPE = False
        if init_OPE:
            self.val_buffer_name = val_buffer_name
            self.test_buffer_keyword = test_buffer_keyword
            self.OPE_methods = OPE_methods
            self.OPE_args = OPE_args
            self.check_mode(val_buffer_name, test_buffer_keyword, OPE_methods, test_online, metric)

            # Validate metric
            self.metric = metric
            if metric is not None and metric not in OPE_methods:
                raise ValueError(f"metric {metric} must be one of the keys in {self.val_OPE.keys()}")

    def check_mode(self, val_buffer_name, test_buffer_keyword, OPE_methods, test_online, metric):
        # Mode 1: "val-online-test-online"
        if val_buffer_name is None and test_buffer_keyword is None and OPE_methods is None and test_online:
            self.setup_mode("val-online-test-online")
            return
        # Mode 2: "val-offline-test-offline"
        if val_buffer_name and test_buffer_keyword and OPE_methods and not test_online:
            self.setup_mode("val-offline-test-offline")
            return
        # Mode 3: "val-offline-test-offline&online"
        if val_buffer_name and test_buffer_keyword and OPE_methods and test_online:
            self.setup_mode("val-offline-test-offline&online")
            return
        # Mode 4: "val-offline-test-online"
        if val_buffer_name and not test_buffer_keyword and OPE_methods and test_online:
            self.setup_mode("val-offline-test-online")
            return
        raise ValueError("Provided arguments do not match any known mode.")

    def setup_mode(self, mode):
        if "DR" in self.OPE_methods or "WDR" in self.OPE_methods:
            if "value_fn" not in self.OPE_args.keys():
                raise ValueError("OPE_args must be provided for DR and WDR")
            value_function = self.OPE_args["value_fn"]
        else:
            value_function = None
        if "IS" in self.OPE_methods or "WIS" in self.OPE_methods or "WIS_bootstrap" in self.OPE_methods or "WIS_truncated" in self.OPE_methods or \
                "WIS_bootstrap_truncated" in self.OPE_methods or "DR" in self.OPE_methods or "WDR" in self.OPE_methods:
            if "behavioural_fn" not in self.OPE_args.keys():
                raise ValueError("OPE_args must be provided for IS and WIS")
            behavior_policy = self.OPE_args["behavioural_fn"]
        else:
            behavior_policy = None
        if mode == "val-online-test-online":
            self.test_online = True
            self.test_in_train = True
            self.val_OPE = None
            self.test_OPE = None
            self.val_buffer = None
        elif mode in ["val-offline-test-offline", "val-offline-test-offline&online"]:
            self.test_online = not (mode == "val-offline-test-offline")
            self.test_in_train = False
            self.val_buffer = {
                self.val_buffer_name: load_buffer(buffer_registry.make(self.env_name, self.val_buffer_name))}
            self.test_buffers = {k: load_buffer(v) for k, v in buffer_registry.make_all(self.env_name,
                                                                                        self.test_buffer_keyword).items()}

            self.val_OPE = OPE_wrapper(self.OPE_methods, buffers=self.val_buffer, behavior_policy=behavior_policy,
                                       value_function=value_function,
                                       num_actions=self.action_shape,
                                       gamma=self.hparam_space.get_search_space()["gamma"]["value"],
                                       all_soften=self.meta_param["all_soften"])
            self.test_OPE = OPE_wrapper(self.OPE_methods, buffers=self.test_buffers, behavior_policy=behavior_policy,
                                        value_function=value_function,
                                        num_actions=self.action_shape,
                                        gamma=self.hparam_space.get_search_space()["gamma"]["value"],
                                        all_soften=self.meta_param["all_soften"])
        elif mode == "val-offline-test-online":
            self.val_buffer = load_buffer(buffer_registry.make(self.env_name, self.val_buffer_name))
            self.test_OPE = None
            self.val_OPE = OPE_wrapper(self.OPE_methods, buffers=self.val_buffer, behavior_policy=behavior_policy,
                                       value_function=value_function,
                                       gamma=self.hparam_space.get_search_space()["gamma"]["value"],
                                       num_actions=self.meta_param["num_actions"],
                                       all_soften=self.meta_param["all_soften"])
            self.test_online = True
            self.test_in_train = False
        elif mode == "ope":
            self.test_online = False
            self.test_in_train = False
            self.val_buffer = None
            self.test_OPE = None
            self.val_OPE = None
        else:
            raise NotImplementedError

    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        ckpt_path = os.path.join(self.log_path, f"checkpoint_epoch{epoch}.pth")
        torch.save({"model": self.policy.state_dict()}, ckpt_path)
        return ckpt_path

    def early_stop_fn(self, counter):
        return False  # no early stopping for re-validation on mew metrics

    def search_once(self, hparams: dict = None, metric="best_reward", log_name=None):
        # first detect whether the log dir exists, if so, try to load the model and only do validation and test
        # otherwise, do the whole training process
        super().search_once(hparams, metric, log_name)

    def run(self,
            policy,
            batch_size,
            cat_num,
            stack_num,
            **kwargs
            ):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        def select_best_fn(result_dict):
            try:
                return result_dict[f"{self.val_buffer_name}-{self.metric}"]
            except KeyError:
                raise ValueError(
                    f"metric {f'{self.val_buffer_name}-{self.metric}'} not found in result_dict {result_dict.keys()}")

        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        self.train_buffer.stack_num = stack_num

        # align stack_num for all buffers
        self.val_OPE.align_stack_num(stack_num)
        if self.test_OPE is not None:
            self.test_OPE.align_stack_num(stack_num)

        if self.test_in_train:
            try:
                self.test_envs.step(self.env.action_space.sample(), id=0)
                self.test_envs.reset(id=0)
                online_collector = Collector(policy, self.test_envs, exploration_noise=False)
            except NotImplementedError:
                raise ValueError("It seems that you are trying to use online testing for an offline dataset with no "
                                 "'ground truth' environment."
                                 "Please use the synthetic environment for online testing or turn off test_in_train ")
        else:
            online_collector = None

        result = offline_trainer(
            policy,
            self.train_buffer,
            online_collector,
            self.val_OPE,
            self.meta_param["epoch"],
            self.meta_param["update_per_epoch"] if self.meta_param["update_per_epoch"] is not None else int(
                len(self.train_buffer._meta) / batch_size),
            self.meta_param["test_num"],
            batch_size,
            logger=self.logger,
            select_best_fn=select_best_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=self.save_checkpoint_fn,
        )
        result = {f"best_epoch-{k}": v for k, v in result.items()}
        # test the policy
        self.policy.load_state_dict(torch.load(os.path.join(self.log_path, "policy.pth")))
        self.policy.eval()
        if self.test_online:
            print('testing online')
            test_env = DummyVectorEnv([lambda: gym.make(self.env_name, n_act=self.meta_param["num_actions"])])
            test_env.seed(self.meta_param["seed"])
            test_collector = Collector(self.policy, test_env, exploration_noise=True)
            result["online_reward"] = test_collector.collect(n_episode=self.meta_param["test_num"], render=False)[
                "rews"]
        # test OPE
        if self.test_OPE is not None:
            result.update({f"{k}": v for k, v in self.test_OPE(self.policy).items()})
        return result


class DiscreteImitationObjective(OfflineRLObjective):
    def define_policy(self, lr, cat_num, linear, stack_num, loss_fn, calibration=False, **kwargs):

        net = define_single_network(self.state_shape, self.action_shape, stack_num > 1, False,
                                    cat_num,
                                    linear, device=self.device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)

        if loss_fn == "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn == "weighted_cross_entropy":
            counts = np.unique(self.train_buffer.act, return_counts=True)[1]
            weights = 1.0 / counts
            weights = weights / weights.sum()  #
            class_weights = torch.FloatTensor(weights).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif "focal" in loss_fn:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return ImitationPolicy(net, optim, action_space=self.action_space, loss_fn=loss_fn)


class OfflineSARSAObjective(OfflineRLObjective):
    def define_policy(self, gamma,
                      lr,
                      cat_num,
                      stack_num,
                      linear,
                      **kwargs, ):
        net = define_single_network(self.state_shape, self.action_shape, stack_num > 1, False, linear=linear,
                                    device=self.device, cat_num=cat_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = OfflineSARSAPolicy(net, optim, gamma).to(self.device)
        return policy

class OfflineDQNObjective(OfflineRLObjective, DQNObjective):

    def define_policy(self, *args, **kwargs):
        return DQNObjective.define_policy(self, *args, **kwargs)

    def run(self, *args, **kwargs):
        return OfflineRLObjective.run(self, *args, **kwargs)


class DiscreteCQLObjective(OfflineRLObjective):
    def define_policy(self, gamma,
                      n_step,
                      target_update_freq,
                      alpha,
                      lr,
                      stack_num,
                      cat_num,
                      linear,
                      **kwargs, ):
        use_rnn = stack_num > 1
        net = define_single_network(self.state_shape, self.action_shape, use_dueling=False, linear=linear,
                                    use_rnn=use_rnn, device=self.device, cat_num=cat_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = CQLDQNPolicy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq,
            alpha=alpha,
        ).to(self.device)
        return policy


class BCQObjective(OfflineRLObjective):
    def define_policy(self, phi, actor_lr, critic_lr,
                      gamma, tau, lmbda, cat_num, linear,
                      **kwargs, ):
        hidden_sizes = [256, 256, 256, 256] if not linear else []
        max_action = self.action_space.high[0]
        net_a = MLP(
            input_dim=self.state_shape + self.action_shape,
            output_dim=self.action_space,
            hidden_sizes=hidden_sizes,
            device=self.device,
            flatten_input=True if cat_num > 1 else False
        )
        actor = Perturbation(
            net_a, max_action=max_action, device=self.device, phi=phi
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        net_c1 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        net_c2 = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=self.device,
            cat_num=cat_num
        )
        critic1 = Critic(net_c1, device=self.device).to(self.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=self.device).to(self.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        # vae
        # output_dim = 0, so the last Module in the encoder is ReLU
        vae_encoder = MLP(
            input_dim=self.state_shape * cat_num + self.action_shape,
            hidden_sizes=hidden_sizes,
            device=self.device,
            flatten_input=True if cat_num > 1 else False
        )
        latent_dim = self.action_space * 2
        vae_decoder = MLP(
            input_dim=self.state_shape * cat_num + latent_dim,
            output_dim=self.action_space,
            hidden_sizes=hidden_sizes,
            device=self.device,
            flatten_input=True if cat_num > 1 else False
        )
        vae = VAE(
            vae_encoder,
            vae_decoder,
            hidden_dim=hidden_sizes[-1],
            latent_dim=latent_dim,
            max_action=max_action,
            device=self.device,
        ).to(self.device)
        vae_optim = torch.optim.Adam(vae.parameters())

        policy = BCQPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            vae,
            vae_optim,
            device=self.device,
            gamma=gamma,
            tau=tau,
            lmbda=lmbda,
        )

        return policy


class DiscreteBCQObjective(OfflineRLObjective):
    def define_policy(self,
                      lr,
                      gamma,
                      n_step,
                      target_update_freq,
                      unlikely_action_threshold,
                      imitation_logits_penalty,
                      cat_num,
                      linear,
                      eps_test=0.001,
                      **kwargs, ):
        hidden_sizes = [256, 256, 256, 256] if not linear else []
        feature_net = define_single_network(self.state_shape, self.action_shape, use_dueling=False, linear=linear,
                                            use_rnn=False, device=self.device, cat_num=cat_num)
        policy_net = Actor(
            feature_net,
            self.action_shape,
            device=self.device,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        imitation_net = Actor(
            feature_net,
            self.action_shape,
            device=self.device,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        actor_critic = ActorCritic(policy_net, imitation_net)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        # define policy
        policy = DiscreteBCQPolicy(
            policy_net, imitation_net, optim, gamma, n_step,
            target_update_freq, eps_test, unlikely_action_threshold,
            imitation_logits_penalty
        )
        return policy


class DiscreteIQLObjective(OfflineRLObjective):
    def define_policy(self, gamma,
                      stack_num,
                      cat_num,
                      linear,
                      target_update_freq,

                      actor_lr,
                      critic_lr,
                      tau,
                      quantile,
                      beta,
                      **kwargs, ):
        # model
        use_rnn = stack_num > 1
        net_a = define_single_network(self.state_shape, self.action_shape, use_dueling=False, linear=linear,
                                      use_rnn=use_rnn, device=self.device, cat_num=cat_num)
        net_qf1 = define_single_network(self.state_shape, self.action_shape, use_dueling=False, linear=linear,
                                        use_rnn=use_rnn, device=self.device, cat_num=cat_num)
        net_qf2 = define_single_network(self.state_shape, self.action_shape, use_dueling=False, linear=linear,
                                        use_rnn=use_rnn, device=self.device, cat_num=cat_num)
        vf = define_single_network(self.state_shape, 1, use_dueling=False, linear=linear,
                                   use_rnn=use_rnn, device=self.device, cat_num=cat_num)
        actor = discreteActor(net_a, self.action_shape, device=self.device, softmax_output=False).to(self.device)

        return DiscreteIQLPolicy(actor, net_qf1, net_qf2, vf,
                                 discount=gamma, reward_scale=1.0, policy_lr=actor_lr, qf_lr=critic_lr,
                                 tau=tau, target_update_freq=target_update_freq,
                                 quantile=quantile, beta=beta, clip_score=100,
                                 device=self.device)
