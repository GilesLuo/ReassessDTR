"""
This module contains realizations of different simulating envs.
"""
from gymnasium.envs.registration import register
from .MIMIC3SepsisEnv.env import (MIMIC3SepsisEnv, create_MIMIC3SepsisEnv_discrete, create_MIMIC3SepsisSynEnv_discrete,
                                  create_MIMIC3SepsisOutcomeEnv_discrete, create_MIMIC3SepsisNEWS2Env_discrete,
                                  create_MIMIC3SepsisSOFAEnv_discrete)
import os
import importlib
from pathlib import Path
import pkgutil

"""
There are 5 settings for each env:
1. no pkpd, no state and obs noise, no missing data, 
2. pkpd, no state and obs noise, no missing data,
3. pkpd, small state and obs noise, no missing data,
4. pkpd, large state and obs noise, no missing data,
5. pkpd, large state and obs noise, missing data.

"""
registered_ids = ["AhnChemoEnv-continuous",
                  "AhnChemoEnv-discrete",
                  "AhnChemoEnv-discrete-setting1",
                  "AhnChemoEnv-discrete-setting2",
                  "AhnChemoEnv-discrete-setting3",
                  "AhnChemoEnv-discrete-setting4",
                  "AhnChemoEnv-discrete-setting5",
                  "AhnChemoEnv-continuous-setting1",
                  "AhnChemoEnv-continuous-setting2",
                  "AhnChemoEnv-continuous-setting3",
                  "AhnChemoEnv-continuous-setting4",
                  "AhnChemoEnv-continuous-setting5",
                  "OberstSepsisEnv-discrete",
                  "OberstSepsisEnv-discrete-setting1",
                  "OberstSepsisEnv-discrete-setting2",
                  "OberstSepsisEnv-discrete-setting3",
                  "OberstSepsisEnv-discrete-setting4",
                  "OberstSepsisEnv-discrete-setting5",
                  "GhaffariCancerEnv-discrete",
                  "GhaffariCancerEnv-continuous",
                  "GhaffariCancerEnv-discrete-setting1",
                  "GhaffariCancerEnv-discrete-setting2",
                  "GhaffariCancerEnv-discrete-setting3",
                  "GhaffariCancerEnv-discrete-setting4",
                  "GhaffariCancerEnv-discrete-setting5",
                  "GhaffariCancerEnv-continuous-setting1",
                  "GhaffariCancerEnv-continuous-setting2",
                  "GhaffariCancerEnv-continuous-setting3",
                  "GhaffariCancerEnv-continuous-setting4",
                  "GhaffariCancerEnv-continuous-setting5",
                  "SimGlucoseEnv-discrete",
                  "SimGlucoseEnv-continuous",
                  "SimGlucoseEnv-discrete-setting1",
                  "SimGlucoseEnv-discrete-setting2",
                  "SimGlucoseEnv-discrete-setting3",
                  "SimGlucoseEnv-discrete-setting4",
                  "SimGlucoseEnv-discrete-setting5",
                  "MIMIC3SepsisEnv-discrete",
                  "MIMIC3SepsisOutcomeEnv-discrete",
                  "MIMIC3SepsisNEWS2Env-discrete",
                  "MIMIC3SepsisSOFAEnv-discrete",
                  "MIMIC3SepsisSynEnv-discrete"]

envs = ["AhnChemoEnv",
        "GhaffariCancerEnv",
        "SimGlucoseEnv",
        "OberstSepsisEnv",
        "MIMIC3SepsisEnv",  # offline placeholder
        "MIMIC3SepsisOutcomeEnv",  # offline placeholder
        "MIMIC3SepsisSOFAEnv",  # offline placeholder
        "MIMIC3SepsisNEWS2Env",  # offline placeholder
        "MIMIC3SepsisEnvSyn",  # GAN based simulation
        ]

for registered_id in registered_ids:
    register(
        id=registered_id,
        entry_point=f"DTRGym:create_{registered_id.replace('-', '_')}",
    )


class BufferRegistry:
    def __init__(self):
        self.buffers = {env_name: {} for env_name in envs}

    def register(self, env, name, path):
        self.buffers[env][name] = path

    def auto_register(self):
        root_dir = Path(pkgutil.get_loader("DTRGym").get_filename()).parent
        for env_name in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, env_name)):
                data_dir = os.path.join(root_dir, env_name, "offline_data")
                if os.path.exists(data_dir):
                    for buffer in os.listdir(data_dir):
                        if buffer.endswith(".hdf5"):
                            self.register(env_name, buffer.replace('_buffer', '').replace('.hdf5', ''),
                                          os.path.join(data_dir, buffer))

    def make(self, env_name: str, buffer_name: str):
        env_name = env_name.replace('-continuous', '').replace('-discrete', '')
        if env_name not in self.buffers.keys():
            raise ValueError(f"env {env_name} not registered")
        return self.buffers[env_name][buffer_name]

    def make_all(self, env_name: str, buffer_name: str):
        """
        :param buffer_name: name keyword of the buffer
        :return: all buffers which have the keyword in their name
        """
        env_name = env_name.replace('-continuous', '').replace('-discrete', '')
        if env_name not in self.buffers.keys():
            raise ValueError(f"env {env_name} not registered")

        buffers = {}
        for b in self.buffers[env_name].keys():
            if buffer_name in b:
                buffers[b] = self.buffers[env_name][b]
        return buffers


buffer_registry = BufferRegistry()
buffer_registry.auto_register()
