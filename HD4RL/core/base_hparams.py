

common_hparams = {
    "seed": [6311, 6890, 663, 4242, 8376],
    "lr": [0.01, 0.001, 0.0001, 0.00001],
    "batch_size": [256],
    "stack_num": 3,
    "batch_norm": [True, False],
    "dropout": [0, 0.25, 0.5],
    "target_update_freq": 1000,
    "update_per_step": [0.1, 0.5],
    "update_actor_freq": [1, 5],
    "step_per_collect": [50, 100],
    "n_step": 1,
    "start_timesteps": 0,
    "gamma": 0.99,
    "tau": 0.001,
    "exploration_noise": [0.1, 0.2, 0.5],
    "cat_num": 3,
    "eps_test": 1e-6,
}


def get_common_hparams(use_rnn):
    hp = common_hparams.copy()
    if not use_rnn:
        hp["stack_num"] = 1
    return hp

