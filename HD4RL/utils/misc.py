
import torch
import random
import numpy as np



def to_bool(value):
    valid = {'true': True, 't': True, '1': True,
             'false': False, 'f': False, '0': False,
             }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def set_global_seed(seed):
    # Set seed for Python's built-in random
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for all GPU devices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set seed for torch DataLoader
    def _worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn

