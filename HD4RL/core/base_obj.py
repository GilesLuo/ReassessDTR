import os
import torch
from tianshou.policy.base import BasePolicy
from HD4RL.core.logger import OfflineWandbLogger
from DTRGym.base import make_env
from HD4RL.utils.misc import set_global_seed
from HD4RL.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
import wandb
import json

class RLObjective:
    def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace,
                 device, logger="tensorboard", multi_obj=False, early_stopping=10
                 ):

        # define high level parameters
        self.env_name, self.hparam_space, self.logger, self.early_stopping = env_name, hparam_space, logger, early_stopping
        self.device = device
        self.multi_obj = multi_obj
        self.meta_param = self.hparam_space.get_meta_params()
        self.search_space = self.hparam_space.get_search_space()
        self.retrain = False  # enable checkpoint saving in retrain otherwise disable

        # define job name for logging
        self.job_name = self.env_name

        # early stopping counter
        self.rew_history = []
        self.early_stopping_counter = 0

        # prepare env
        self.env, self.train_envs, self.test_envs = make_env(env_name, 0,
                                                             self.meta_param["training_num"],
                                                             1,
                                                             # test_num is always 1, we will experiment test multiple times for one test env
                                                             num_actions=self.meta_param["num_actions"])

        state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.state_space = self.env.observation_space
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.action_space = self.env.action_space
        if isinstance(state_shape, (tuple, list)):
            if len(state_shape) > 1:
                raise NotImplementedError("state shape > 1 not supported yet")
            self.state_shape = state_shape[0]
        else:
            self.state_shape = int(state_shape)
        if isinstance(action_shape, (tuple, list)):
            if len(action_shape) > 1:
                raise NotImplementedError("action shape > 1 not supported yet")
            self.action_shape = action_shape
        else:
            self.action_shape = int(action_shape)

    def search_once(self, hparams: dict=None, metric="best_reward", log_name=None):
        if self.logger == "wandb":
            self.logger = OfflineWandbLogger(project="SepsisRL",
                                             name=f"{self.env_name}-{self.hparam_space.algo_name}",
                                             save_interval=1,
                                             config=self.hparam_space.get_meta_params(),
                                             )
        else:
            raise NotImplementedError("Only wandb is supported for search_once")
        hparams = hparams if hparams is not None else wandb.config
        self.retrain = False
        set_global_seed(hparams["seed"])

        # use all hparam combinations as the job name
        if log_name is None:
            trial_name = "-".join([f"{k}{v}" for k, v in hparams.items()
                                   if k in self.search_space.keys()])
            log_name = os.path.join(self.job_name, hparams["algo_name"],
                                    f"{trial_name}-seed{hparams['seed']}")
        else:
            print(f"log name {log_name} is provided, will not use hparams to generate log name")
        log_path = os.path.join(self.meta_param["logdir"], log_name)

        self.log_path = str(log_path)
        print(f"logging to {self.log_path}")
        wandb.log({"model_dir": log_path})
        os.makedirs(self.log_path, exist_ok=True)
        self.policy = self.define_policy(**hparams)

        result = self.run(self.policy, **hparams)
        score = result[metric.replace("test/", "")]

        self.logger.log_test_data(result, step=0)
        history = self.logger.download_history()
        with open(os.path.join(self.log_path, "run_summary.json"), "w") as f:
            json.dump(result, f)
        history.to_csv(os.path.join(self.log_path, "run_history.csv"), index=False)
        if self.multi_obj:
            return score, 1
        else:
            return score

    def early_stop_fn(self, mean_rewards):
        # reach reward threshold
        reach = False
        # reach early stopping
        early_stop = False
        # todo: early stopping is not working for now, in the paper we run all epochs
        return reach or early_stop

    # no checkpoint saving needed
    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if self.retrain:
            ckpt_path = os.path.join(self.log_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": self.policy.state_dict()}, ckpt_path)
            return ckpt_path

    def define_policy(self, *args, **kwargs) -> BasePolicy:
        return NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

