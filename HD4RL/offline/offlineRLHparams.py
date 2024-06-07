from HD4RL.core.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from HD4RL.core.base_hparams import common_hparams


class OfflineRLHyperParameterSpace(OffPolicyRLHyperParameterSpace):
    _meta_hparams = [
        "algo_name",  # name of the algorithm
        "logdir",  # directory to save logs
        "epoch",
        "training_num",
        "test_num",
        "update_per_epoch",  # number of training steps per epoch
        "num_actions",  # number of actions, only used for discrete action space
        "linear",  # whether to use linear layer
        "all_soften",  # whether to use soften best action for all OPEs
    ]

    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
        "batch_size": common_hparams["batch_size"],
        "gamma": common_hparams["gamma"],
    }
    _policy_hparams = {}
    _supported_algos = ()

    def __init__(self, algo_name, logdir, seed, test_num, epoch, update_per_epoch, num_actions=None,
                 linear=False, all_soften=False):
        if algo_name.lower() not in [i.lower() for i in self.__class__._supported_algos]:
            raise NotImplementedError(f"algo_name {algo_name} not supported, support {self.__class__._supported_algos}")
        self.algo_name = algo_name
        self.logdir = logdir
        if seed is not None:
            self.seed = seed
            print(f"seed is {seed}, this will override the seed in the config file. "
                  f"Pls double check the hyperparameter to avoid duplicated search.")
        self.test_num = test_num
        self.epoch = epoch
        self.num_actions = num_actions
        self.linear = linear
        self.update_per_epoch = update_per_epoch
        self.training_num = 1
        self.test_num = 1
        self.all_soften = all_soften


class DummyHparam(OfflineRLHyperParameterSpace):
    _supported_algos = ['dummy', ]
    _policy_hparams = []
    _algo_hparams = []

    def __init__(self):
        super().__init__("dummy", "", None, 1, 1, 0, None, True,
                         all_soften=False)


class DiscreteImitationHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-imitation",)
    _policy_hparams = {
        "lr": common_hparams["lr"],
        "stack_num": 1,
        "cat_num": 1,
        "loss_fn": ["cross_entropy", "weighted_cross_entropy"]
    }


class DiscreteImitationRNNHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-imitation-rnn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],
        "stack_num": common_hparams["stack_num"],
        "cat_num": 1,
        "loss_fn": ["cross_entropy", "weighted_cross_entropy"]
    }


class DiscreteImitationObsCatHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-imitation-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
        "loss_fn": ["cross_entropy", "weighted_cross_entropy"]
    }


class DQNHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("dqn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }


class DQNRNNHyperParams(DQNHyperParams):
    _supported_algos = ("dqn-rnn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": common_hparams["stack_num"],
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }


class DQNObsCatHyperParams(DQNHyperParams):
    _supported_algos = ("dqn-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }


class DDQNHyperParams(DQNHyperParams):
    _supported_algos = ("ddqn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": True,
        "use_dueling": False,
    }


class DDQNRNNHyperParams(DDQNHyperParams):
    _supported_algos = ("ddqn-rnn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": common_hparams["stack_num"],
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": True,
        "use_dueling": False,
    }


class DDQNObsCatHyperParams(DDQNHyperParams):
    _supported_algos = ("ddqn-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": True,
        "use_dueling": False,
    }


class DuelingDQNHyperParams(DQNHyperParams):
    _supported_algos = ("dqn-dueling",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": True,
    }


class DuelingDDQNHyperParams(DQNHyperParams):
    _supported_algos = ("ddqn-dueling",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": True,
        "use_dueling": True,
    }


class OfflineSARSAHyperParams(OfflineRLHyperParameterSpace):
    # FQI hyperparams are standalone since it is used for OPE
    _supported_algos = ("offlinesarsa",)
    _policy_hparams = {"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7],
                       "stack_num": 1,
                       "cat_num": 1,
                       }
    _general_hparams = {
        "batch_size": [256, 512, 1024, 2048, 4096],
        "gamma": common_hparams["gamma"],
    }


class OfflineSARSARNNHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("offlinesarsa-rnn",)
    _policy_hparams = {"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7],
                       "stack_num": common_hparams["stack_num"],
                       "cat_num": 1,
                       }
    _general_hparams = {
        "batch_size": [1024, 2048, 4096],
        "gamma": common_hparams["gamma"],
    }


class OfflineSARSAObsCatHyperParams(OfflineRLHyperParameterSpace):
    # FQI hyperparams are standalone since it is used for OPE
    _supported_algos = ("offlinesarsa-obs_cat",)
    _policy_hparams = {"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7],
                       "stack_num": 1,
                       "cat_num": common_hparams["cat_num"],
                       }
    _general_hparams = {
        "batch_size": [256, 512, 1024, 2048, 4096],
        "gamma": common_hparams["gamma"],
    }


class OfflineSACHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-sac", "sac")
    _policy_hparams = {
        "actor_lr": common_hparams["lr"],
        "critic_lr": common_hparams["lr"],
        "alpha_lr": common_hparams["lr"],
        "n_step": common_hparams["n_step"],
        "tau": common_hparams["tau"],
        "stack_num": 1,
        "cat_num": 1,

    }


class CQLHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("cql",)

    def __init__(self):
        raise NotImplementedError


class DiscreteCQLHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-cql",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "alpha": [0.1, 0.5, 1.],
        "stack_num": 1,
        "cat_num": 1,
    }


class DiscreteCQLObsCatHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-cql-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "alpha": [0.1, 0.5, 1.],
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
    }


class DiscreteCQLRNNHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-cql-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "alpha": [0.1, 0.5, 1.],
        "stack_num": common_hparams["stack_num"],
        "cat_num": 1, }


class BCQHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("bcq",)

    def __init__(self):
        raise NotImplementedError


class DiscreteBCQHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-bcq",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "unlikely_action_threshold": [0.3, 0.5],
        "imitation_logits_penalty": [0.02, 0.1, 0.5],
        "eps_test": common_hparams["eps_test"],
        "stack_num": 1,
        "cat_num": 1,
    }


class DiscreteBCQObsCatHyperParams(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-bcq-obs_cat",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "unlikely_action_threshold": [0.3, 0.5],
        "imitation_logits_penalty": [0.02, 0.1, 0.5],
        "eps_test": common_hparams["eps_test"],
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
    }


class DiscreteIQLHyperparameterSpace(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-iql",)
    _policy_hparams = {
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "actor_lr": common_hparams["lr"],
        "critic_lr": common_hparams["lr"],
        "tau": common_hparams["tau"],
        "quantile": [0.7, 0.9],
        "beta": [0.7, 1.],
        "stack_num": 1,
        "cat_num": 1,
    }


class DiscreteIQLObsCatHyperparameterSpace(OfflineRLHyperParameterSpace):
    _supported_algos = ("discrete-iql-obs_cat",)
    _policy_hparams = {
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "actor_lr": common_hparams["lr"],
        "critic_lr": common_hparams["lr"],
        "tau": common_hparams["tau"],
        "quantile": [0.7, 0.9],
        "beta": [0.7, 1.],
        "stack_num": 1,
        "cat_num": common_hparams["cat_num"],
    }
