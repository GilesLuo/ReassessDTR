
from HD4RL.core.base_hparams import common_hparams
import numpy as np


class OffPolicyRLHyperParameterSpace:
    _meta_hparams = [
        "algo_name",  # name of the algorithm
        "logdir",  # directory to save logs
        "seed",
        "training_num",  # number of training envs
        "test_num",  # number of test envs
        "epoch",
        "step_per_epoch",  # number of training steps per epoch
        "buffer_size",  # size of replay buffer
        "num_actions",  # number of actions, only used for discrete action space
        "linear",  # whether to use linear approximation as network
    ]

    # general hyperparameter search space
    _general_hparams = {
        # general parameters
        "batch_size": common_hparams["batch_size"],
        "step_per_collect": common_hparams["step_per_collect"],  # number of steps per collect. refer to tianshou's doc
        "update_per_step": common_hparams["update_per_step"],
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
    }
    # policy hyperparameter search space
    _policy_hparams = {
        "stack_num": None,  # number of frames to stack, must be specified in the child class
        "cat_num": None,
        # number of obs concatenation, must be specified in the child class, cannot be used with stack_num
    }
    _supported_algos = ()

    def __init__(self,
                 algo_name,  # name of the algorithm
                 logdir,  # directory to save logs
                 seed,
                 training_num,  # number of training envs
                 test_num,  # number of test envs
                 epoch,
                 step_per_epoch,  # number of training steps per epoch
                 buffer_size,  # size of replay buffer
                 num_actions=None,  # number of actions, only used for discrete action space
                 linear=False
                 ):
        if algo_name.lower() not in [i.lower() for i in self.__class__._supported_algos]:
            raise NotImplementedError(f"algo_name {algo_name} not supported, support {self.__class__._supported_algos}")
        self.algo_name = algo_name
        self.logdir = logdir
        self.seed = seed
        self.training_num = training_num
        self.test_num = test_num
        self.epoch = epoch
        self.step_per_epoch = step_per_epoch
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.linear = linear

    def check_illegal(self):
        """
        This function makes sure all hyperparameters are defined.
        all hyperparameters should be defined in _meta_hparams, _general_hparams and _policy_hparams. If not, raise error
        and list the undefined hyperparameters.
        :return: list of undefined hyperparameters
        """
        all_hparams = list(self._meta_hparams) + list(self._general_hparams.keys()) + list(self._policy_hparams.keys())
        undefined_hparams = [h for h in all_hparams if not hasattr(self, h)]
        unknown_hparams = [h for h in self.__dict__() if h not in all_hparams]
        if len(undefined_hparams) > 0:
            printout1 = f"undefined hyperparameters: {undefined_hparams}"
        else:
            printout1 = ""
        if len(unknown_hparams) > 0:
            printout2 = f"unknown hyperparameters: {unknown_hparams}"
        else:
            printout2 = ""
        if len(printout1) > 0 or len(printout2) > 0:
            raise ValueError(f"{printout1}\n{printout2}")

    def get_search_space(self):
        search_space = {}
        search_space.update(self._general_hparams)
        search_space.update(self._policy_hparams)
        space = {}
        for k, v in search_space.items():
            if isinstance(v, (list, tuple)):
                space[k] = {"values": v}
            elif isinstance(v, (int, float, bool, str)):
                space[k] = {"value": v}
            else:
                raise NotImplementedError(f"unsupported type {type(v)} for hyperparameter {k}")
        return space

    def sample(self, mode="first"):
        if mode == "first":
            sample_fn = lambda x: x[0]
        else:
            sample_fn = lambda x: np.random.choice(x)
        search_space = self.get_search_space()
        result = {}
        for k, v in search_space.items():
            if "values" in v:
                result[k] = sample_fn(v["values"])
            elif "value" in v:
                result[k] = v["value"]
            else:
                raise NotImplementedError
        return result

    def get_meta_params(self):
        return {k: getattr(self, k) for k in self._meta_hparams}

    def get_general_params(self):
        return {k: getattr(self, k) for k in self._general_hparams.keys()}

    def get_policy_params(self):
        return {k: getattr(self, k) for k in self._policy_hparams.keys()}

    def get_all_params(self):
        result = {}
        dict_args = [self.get_general_params(), self.get_policy_params(),self.get_meta_params(), ]
        # if args in both general and meta, meta will overwrite general (seed)
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def define_general_hparams(self, trial):
        for name, space in self._general_hparams.items():
            if isinstance(space, list):
                value = trial.suggest_categorical(name, space)
            else:
                value = trial.set_user_attr(name, space)
            setattr(self, name, value)

    def define_policy_hparams(self, trial):
        for name, space in self._policy_hparams.items():
            if isinstance(space, list):
                value = trial.suggest_categorical(name, space)
            else:
                value = trial.set_user_attr(name, space)
            setattr(self, name, value)

    def __call__(self, trial):
        # define meta hparams
        for p in self._meta_hparams:
            trial.set_user_attr(p, getattr(self, p))

        # define general hparams
        meta_hparams = self.get_meta_params()
        general_hparams = self.define_general_hparams(trial)
        policy_hparams = self.define_policy_hparams(trial)
        self.check_illegal()
        result = {}
        dict_args = [meta_hparams, general_hparams, policy_hparams]
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def keys(self):
        return self.__dict__()

    def __dict__(self):
        return {k for k in dir(self) if not k.startswith('__') and not callable(getattr(self, k))}

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for key in dir(self):
            if not key.startswith('__') and not callable(getattr(self, key)):
                yield key, getattr(self, key)

    def __str__(self):
        # This will combine the dict representation with the class's own attributes
        class_attrs = {k: getattr(self, k) for k in dir(self) if
                       not k.startswith('__') and not callable(getattr(self, k))}
        all_attrs = {**self, **class_attrs}
        return str(all_attrs)


class DQNHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("dqn",)
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "eps_test": 0.005,
        "eps_train": 1,
        "eps_train_final": 0.005,
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
        "eps_test": 0.005,
        "eps_train": 1,
        "eps_train_final": 0.005,
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
        "eps_test": 0.005,
        "eps_train": 1,
        "eps_train_final": 0.005,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }


