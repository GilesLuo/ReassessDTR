from DTRGym.base import EpisodicEnv
import torch
from DTRGym.utils import DiscreteActionWrapper
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import torch.nn as nn


class MIMIC3SepsisEnv(EpisodicEnv):
    action_space = spaces.Discrete(25)
    def __init__(self, ep_len, generator: torch.nn.Module, mort_predictor: torch.nn.Module, patient_first_obs: np.array,
                 action_encode_fn):
        super().__init__(ep_len, generator, mort_predictor, patient_first_obs, action_encode_fn)
        n_feature = 46
        self.observation_space = spaces.Box(low=np.array([0.0] * n_feature),
                                            high=np.array([1.] * n_feature), shape=(n_feature,), dtype=np.float32)


class MIMIC3SepsisPlaceHolder(gym.Env):
    action_space = spaces.Discrete(25)
    def __init__(self):
        n_feature = 46
        self.observation_space = spaces.Box(low=np.array([0.0] * n_feature),
                                            high=np.array([1.] * n_feature), shape=(n_feature,), dtype=np.float32)

    def seed(self, seed=None):
        return

def create_MIMIC3SepsisSynEnv_discrete(n_act=None, buffer_path=None, generator=None, mort=None, device="cpu"):
    max_t = 19
    n_act = 25
    n_feature = 46
    if generator is None and mort is None and buffer_path is None:
        print("Warning: No generator and mortality predictor provided, using random environment.")
        generator = nn.Linear(n_feature, n_act).to(device)
        mort = nn.Sequential(nn.Linear(n_feature, 1)).to(device)
        patient_first_obs = np.random.rand(1000, n_feature)
    elif generator is None or mort is None or buffer_path is None:
        raise ValueError("Generator, Mortality Predictor and buffer must be all None or all not None.")
    else:
        raise NotImplementedError

    env = MIMIC3SepsisEnv(max_t, generator, mort, patient_first_obs, lambda x: x)
    return env


def create_MIMIC3SepsisEnv_discrete(n_act=None):
    env = MIMIC3SepsisPlaceHolder()
    return env

def create_MIMIC3SepsisOutcomeEnv_discrete(n_act=None):
    env = MIMIC3SepsisPlaceHolder()
    return env

def create_MIMIC3SepsisNEWS2Env_discrete(n_act=None):
    env = MIMIC3SepsisPlaceHolder()
    return env

def create_MIMIC3SepsisSOFAEnv_discrete(n_act=None):
    env = MIMIC3SepsisPlaceHolder()
    return env
