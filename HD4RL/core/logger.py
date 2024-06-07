from typing import Any, Callable, Optional, Tuple, Dict

import pandas as pd
import wandb
import torch


class OfflineTensorboardLogger:
    def __init__(self, writer, save_interval):
        self.save_interval = save_interval
        self.last_save_step = -1
        self.writer = writer

    def log_any(self, prefix, collect_result: dict, step: int) -> None:
        for k, v in collect_result.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    self.writer.add_scalar(
                        f"{prefix}/{k}/{k_}", v_, global_step=step)
            else:
                if not isinstance(v, (int, float, torch.Tensor)):
                    raise ValueError(
                        f"we only supports int and float, but {type(v)} is given.")
                else:
                    self.writer.add_scalar(f"{prefix}/{k}", v, global_step=step)
        self.writer.flush()

    def save_data(
            self,
            epoch: int,
            save_checkpoint_fn: Optional[Callable[[int], str]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        return self.log_any("test", collect_result, step)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        return self.log_any("train", collect_result, step)

    def log_val_data(self, collect_result: dict, step: int) -> None:
        return self.log_any("val", collect_result, step)


class OfflineWandbLogger:
    """Weights and Biases logger that sends data to https://wandb.ai/.

    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
        add_scalar operation. Default to True.
    :param str project: W&B project name.
    :param str name: W&B experiment name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: experiment id of W&B experiment to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.
    """

    def __init__(
            self,
            save_interval: int = 1,
            project: Optional[str] = None,
            name: Optional[str] = None,
            entity: Optional[str] = None,
            run_id: Optional[str] = None,
            config: Optional[Dict] = None,
    ) -> None:
        self.save_interval = save_interval
        self.restored = False
        self.last_save_step = -float("inf")
        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            entity=entity,
            config=config,  # type: ignore
        )

    def write(self, data, epoch=None, commit=True) -> None:
        wandb.log(data, epoch, commit=commit)

    def save_data(
            self,
            epoch: int,
            env_step: int,
            gradient_step: int,
            save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        if save_checkpoint_fn and (epoch - self.last_save_step) >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)

    def summary(self, data, commit=True) -> None:
        wandb.summary.update(data, commit=commit)

    def log_any(self, prefix, collect_result: dict, step=None, commit=True) -> None:
        data = {}
        for k, v in collect_result.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    data[f"{prefix}/{k}/{k_}"] = v_
            else:
                if not isinstance(v, (int, float, torch.Tensor)):
                    raise ValueError(
                        f"we only supports int and float, but {type(v)} is given.")
                else:
                    data[f"{prefix}/{k}"] = v
        self.write(data, step, commit=commit)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        # wandb.define_metric("test_step")
        # collect_result["test_step"] = step
        return self.log_any("test", collect_result)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        return self.log_any("train", collect_result, step, commit=False)

    def log_update_data(self, collect_result: dict, step: int) -> None:
        pass  # do not log update data

    def log_val_data(self, collect_result: dict, step: int) -> None:
        return self.log_any("val", collect_result, step)

    def download_history(self)->pd.DataFrame:
        api = wandb.Api()
        run = api.run(f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.wandb_run.id}")
        return run.history()