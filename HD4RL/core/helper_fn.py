import torch
from typing import Union, Dict
from tianshou.policy import BasePolicy
from HD4RL.core.policy import RandomPolicy, AlternatingPolicy, AllMaxPolicy, AllMinPolicy, OccurrenceWeightedPolicy
from tianshou.policy import DQNPolicy, DiscreteBCQPolicy, ImitationPolicy
from HD4RL.core.base_obj import RLObjective

from HD4RL.core.offpolicyRLObj import DQNObjective
from HD4RL.offline.offlineRLObj import (DiscreteCQLObjective, \
                                        DiscreteBCQObjective, OfflineDQNObjective,
                                        DiscreteImitationObjective,
                                        OfflineSARSAObjective, DiscreteIQLObjective)
from HD4RL.offline.policy import OfflineSARSAPolicy, DiscreteIQLPolicy, CQLDQNPolicy
from HD4RL.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from HD4RL.core import offpolicyRLHparams as ophp
from HD4RL.offline import offlineRLHparams as olhp


def policy_load(policy, ckpt_path: str, device: str, is_train: bool = False):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        ckpt = ckpt if ckpt_path.endswith("policy.pth") else ckpt["model"]  # policy.pth and ckpt.pth has different keys
        policy.load_state_dict(ckpt)
    if is_train:
        policy.train()
    else:
        policy.eval()
    return policy


def get_class(type, algo_name, offline) -> Union[OffPolicyRLHyperParameterSpace, RLObjective, str]:
    offpolicyLOOKUP = {
        "dqn": {"hparam": ophp.DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
        "dqn-rnn": {"hparam": ophp.DQNRNNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
        "dqn-obs_cat": {"hparam": ophp.DQNObsCatHyperParams, "policy": DQNPolicy, "obj": DQNObjective,
                        "type": "discrete"},
    }

    offlineLOOKUP = {
        "offlinesarsa": {"hparam": olhp.OfflineSARSAHyperParams, "policy": OfflineSARSAPolicy,
                         "obj": OfflineSARSAObjective,
                         "type": "discrete"},
        "offlinesarsa-rnn": {"hparam": olhp.OfflineSARSARNNHyperParams, "policy": OfflineSARSAPolicy,
                             "obj": OfflineSARSAObjective,
                             "type": "discrete"},
        "offlinesarsa-obs_cat": {"hparam": olhp.OfflineSARSAObsCatHyperParams, "policy": OfflineSARSAPolicy,
                                 "obj": OfflineSARSAObjective,
                                 "type": "discrete"},

        "discrete-imitation": {"hparam": olhp.DiscreteImitationHyperParams, "policy": ImitationPolicy,
                               "obj": DiscreteImitationObjective, "type": "discrete"},
        "discrete-imitation-rnn": {"hparam": olhp.DiscreteImitationRNNHyperParams, "policy": ImitationPolicy,
                                   "obj": DiscreteImitationObjective, "type": "discrete"},
        "discrete-imitation-obs_cat": {"hparam": olhp.DiscreteImitationObsCatHyperParams, "policy": ImitationPolicy,
                                       "obj": DiscreteImitationObjective, "type": "discrete"},

        "dqn": {"hparam": olhp.DQNHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective, "type": "discrete"},
        "dqn-rnn": {"hparam": olhp.DQNRNNHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective,
                    "type": "discrete"},
        "dqn-obs_cat": {"hparam": olhp.DQNObsCatHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective,
                        "type": "discrete"},

        "ddqn": {"hparam": olhp.DDQNHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective, "type": "discrete"},
        "ddqn-rnn": {"hparam": olhp.DDQNRNNHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective,
                     "type": "discrete"},
        "ddqn-obs_cat": {"hparam": olhp.DDQNObsCatHyperParams, "policy": DQNPolicy, "obj": OfflineDQNObjective,
                         "type": "discrete"},

        "discrete-cql": {"hparam": olhp.DiscreteCQLHyperParams, "policy": CQLDQNPolicy,
                         "obj": DiscreteCQLObjective,
                         "type": "discrete"},
        "discrete-cql-obs_cat": {"hparam": olhp.DiscreteCQLObsCatHyperParams, "policy": CQLDQNPolicy,
                                 "obj": DiscreteCQLObjective,
                                 "type": "discrete"},

        "discrete-bcq": {"hparam": olhp.DiscreteBCQHyperParams, "policy": DiscreteBCQPolicy,
                         "obj": DiscreteBCQObjective,
                         "type": "discrete"},
        "discrete-bcq-obs_cat": {"hparam": olhp.DiscreteBCQObsCatHyperParams, "policy": DiscreteBCQPolicy,
                                 "obj": DiscreteBCQObjective,
                                 "type": "discrete"},

        "discrete-iql": {"hparam": olhp.DiscreteIQLHyperparameterSpace, "policy": DiscreteIQLPolicy,
                         "obj": DiscreteIQLObjective,
                         "type": "discrete"},
        "discrete-iql-obs_cat": {"hparam": olhp.DiscreteIQLObsCatHyperparameterSpace, "policy": DiscreteIQLPolicy,
                                 "obj": DiscreteIQLObjective,
                                 "type": "discrete"},
    }

    query = offlineLOOKUP if offline else offpolicyLOOKUP
    return query[algo_name][type]


def get_baseline_policy_class(algo_name: str) -> Union[BasePolicy, Dict[str, BasePolicy]]:
    BASELINE_LOOKUP = {"random": {"policy": RandomPolicy},
                       "max": {"policy": AllMaxPolicy},
                       "min": {"policy": AllMinPolicy},
                       "alt": {"policy": AlternatingPolicy},
                       "weight": {"policy": OccurrenceWeightedPolicy}
                       }
    if algo_name == "all":
        return BASELINE_LOOKUP
    else:
        return BASELINE_LOOKUP[algo_name]["policy"]
