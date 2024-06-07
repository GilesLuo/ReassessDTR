import copy

import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import ActorCritic, MLP
from tqdm import tqdm
from typing import Union, List, Tuple, Optional, Callable, Sequence, Dict, Any
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
import torch
from HD4RL.utils.data import collate_batch_seq2seq
from HD4RL.utils.loss import Entropy
from torch import nn
import torch.nn.functional as F
from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

import torch
import torch.nn as nn
from torch import nn, optim
from torch.nn import functional as F


class CalibratedNet(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(CalibratedNet, self).__init__()
        self.model = copy.deepcopy(model)
        self.device = model.device
        self.temperature = nn.Parameter(torch.ones(1, requires_grad=True, device=self.device) * 1.5)

        self.nll_criterion = nn.CrossEntropyLoss().to(self.device)
        self.entropy_criterion = Entropy().to(self.device)

    def reset_temperature(self):
        self.temperature = nn.Parameter(torch.ones(1, requires_grad=True, device=self.device) * 1.5)

    def forward(self, input, state=None, info={}):
        logits, _ = self.model(input)
        return self.temperature_scale(logits), state

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def grid_search_calibration(self, dataset):
        lrs = [0.1, 0.05, 0.02, 0.01, 0.001, 5e-4, 1e-4, 5e-5, 1e-5]
        best_loss = 1e10
        best_calibrated_model = None

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True,
                                                 collate_fn=collate_batch_seq2seq)
        with torch.no_grad():
            for batch in tqdm(val_loader, "Collecting logits"):
                input = batch["obs"].to(self.device)
                logits, _ = self.model(input)
                logits_list.append(logits)
                labels_list.append(batch["act"])
            logits = torch.cat(logits_list).float().to(self.device)
            labels = torch.cat(labels_list).long().to(self.device)

        # Initial NLL
        best_nll = self.nll_criterion(self.temperature_scale(logits), labels).item()
        print(f'Initial entropy: {self.entropy_criterion(self.temperature_scale(logits)).item():.3f}')
        print(f'Initial NLL: {best_nll:.3f}')

        best_temperature = self.temperature.clone()
        for lr in tqdm(lrs, desc="Grid search on learning rate"):
            calibrated_model = CalibratedNet(self.model)
            loss, temp, entropy = calibrated_model.set_temperature(logits, labels, lr)
            if loss < best_loss:
                best_loss = loss
                best_calibrated_model = copy.deepcopy(calibrated_model)
                print(f'New best temperature: {temp.item():.3f}')
                print(f'New best entropy: {entropy:.3f}')
                print(f'New best NLL: {loss:.3f}')
        return best_calibrated_model

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels, lr):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        """
        self.reset_temperature()

        # Setup optimizer with max_iter=50
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=500, history_size=500)

        def eval():
            optimizer.zero_grad()
            loss = self.nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        # Perform optimization
        optimizer.step(eval)

        # Evaluate the final temperature
        final_nll = self.nll_criterion(self.temperature_scale(logits), labels).item()
        final_entropy = self.entropy_criterion(self.temperature_scale(logits)).item()

        # Return the final metrics
        return final_nll, self.temperature.clone(), final_entropy


class Net(nn.Module):
    def __init__(
            self,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            norm_args: Optional[ArgsType] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            act_args: Optional[ArgsType] = None,
            device: Union[str, int, torch.device] = "cpu",
            softmax: bool = False,
            concat: bool = False,
            num_atoms: int = 1,
            dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
            linear_layer: Type[nn.Linear] = nn.Linear,
            cat_num: int = 1,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.cat_num = cat_num
        input_dim = int(np.prod(state_shape)) * cat_num
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
            dropout: float = 0.0,
            num_atoms: int = 1,
            last_step_only: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            dropout=dropout,
            batch_first=True,
        )
        self.num_atoms = num_atoms
        self.action_dim = int(np.prod(action_shape)) * num_atoms
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, self.action_dim)
        self.use_last_step = last_step_only

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        if self.use_last_step:
            obs = self.fc2(obs[:, -1])
        else:
            obs = self.fc2(obs)

        if self.num_atoms > 1:
            obs = obs.view(obs.shape[0], -1, self.num_atoms)

        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }

def define_single_network(input_shape: int, output_shape: int,
                          use_rnn=False, use_dueling=False, cat_num: int = 1, linear=False,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          ):
    if use_dueling and use_rnn:
        raise NotImplementedError("rnn and dueling are not implemented together")

    if use_dueling:
        if linear:
            dueling_params = ({"hidden_sizes": (), "activation": None},
                              {"hidden_sizes": (), "activation": None})
        else:
            dueling_params = ({"hidden_sizes": (256, 256), "activation": nn.ReLU},
                              {"hidden_sizes": (256, 256), "activation": nn.ReLU})
    else:
        dueling_params = None
    if use_rnn and linear:
        raise NotImplementedError("rnn and linear are not implemented together")
    if not use_rnn:
        net = Net(state_shape=input_shape,
                  action_shape=output_shape,
                  hidden_sizes=(256, 256, 256, 256) if not linear else (),
                  norm_layer=None,
                  norm_args=None,
                  activation=nn.ReLU if not linear else None,
                  act_args=None,
                  device=device,
                  softmax=False,
                  concat=False,
                  num_atoms=1,
                  dueling_param=dueling_params,
                  linear_layer=nn.Linear,
                  cat_num=cat_num).to(device)
    else:
        net = Recurrent(layer_num=3,
                        state_shape=input_shape,
                        action_shape=output_shape,
                        device=device,
                        hidden_layer_size=256,
                        ).to(device)

    return net




if __name__ == "__main__":
    input_shape = 3
    output_shape = 2
    network = define_single_network(input_shape, output_shape,
                                    use_rnn=False, use_dueling=False, cat_num=2, linear=True)
    print(network)
    network = define_single_network(input_shape, output_shape,
                                    use_rnn=False, use_dueling=False, cat_num=1, linear=True)
    print(network)
