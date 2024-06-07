from typing import Any, Callable, Dict, Optional, Union, List
import numpy as np
from typing import Tuple
import tqdm
from collections import deque
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from HD4RL.core.logger import OfflineWandbLogger
from HD4RL.OPE.base import BaseOPE
from HD4RL.OPE.utils import OPE_wrapper
from tianshou.utils import (
    DummyTqdm,
    tqdm_config,
)


class OfflineTrainer(BaseTrainer):
    """A modification of Tianshou Offline Trainer.
    Off Policy Evaluation is added as an argument to the policy update function.
    The best policy is saved based on the OPE metric, and then used for testing.
    """

    def __init__(
            self,
            policy: BasePolicy,
            buffer: ReplayBuffer,
            test_collector: Optional[Collector],
            OPE_estimators: Optional[Union[List[BaseOPE], BaseOPE]],
            max_epoch: int,
            update_per_epoch: int,
            episode_per_test: int,
            batch_size: int,
            logger: OfflineWandbLogger,
            select_best_fn: Optional[Callable[[Dict], float]] = None,
            test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
            stop_fn: Optional[Callable[[float], bool]] = None,
            save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
            save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
            resume_from_log: bool = False,
            reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            verbose: bool = True,
            show_progress: bool = True,
            **kwargs: Any,
    ):
        super().__init__(
            learning_type="offline",
            policy=policy,
            buffer=buffer,
            test_collector=test_collector,
            max_epoch=max_epoch,
            update_per_epoch=update_per_epoch,
            step_per_epoch=update_per_epoch,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            **kwargs,
        )
        self.select_best_fn = select_best_fn
        self.OPE_estimators = OPE_estimators
        self.best_val_score = -float('inf')
        self.best_metrics_all = None
        self.early_stop_counter = 0

    def get_OPE(self, policy):
        if isinstance(self.OPE_estimators, OPE_wrapper):
            result_dict = self.OPE_estimators(policy)
        elif self.OPE_estimators is None:
            result_dict = {}
        else:
            raise NotImplementedError("OPE_estimators must be a dict of estimators or a single estimator")
        return result_dict

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            if self.epoch > self.max_epoch or self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()
        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        losses = []
        with progress(
                total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not self.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()

                # Offline training directly from buffer
                result["n/ep"] = len(self.buffer)
                result["n/st"] = int(self.gradient_step)
                t.update()
                l = self.policy_update_fn(data, result)
                t.set_postfix(**data)
                losses.append(l)
            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        self.env_step = self.gradient_step * self.batch_size
        losses = {key: sum(d[key] for d in losses) / len(losses) for key in losses[0].keys()}
        # Validate using OPE after the entire epoch
        val_score_dict = self.get_OPE(self.policy)


        self.logger.log_train_data(losses, self.epoch)
        self.logger.log_val_data(val_score_dict, self.epoch)

        val_metric = self.select_best_fn(val_score_dict)
        if val_metric > self.best_val_score:
            self.best_val_score = val_metric
            self.best_metrics_all = val_score_dict
            self.best_metrics_all["best_epoch"] = self.epoch
            self.save_best_fn(self.policy)
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.stop_fn is not None and self.stop_fn(self.early_stop_counter):
                self.stop_fn_flag = True

        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )

            return self.epoch, epoch_stat, self.best_metrics_all
        else:
            return None

    def policy_update_fn(
            self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ):
        """Perform one off-line policy update."""
        assert self.buffer
        self.gradient_step += 1
        losses = self.policy.update(self.batch_size, self.buffer)
        data.update({"gradient_step": str(self.gradient_step)})
        self.log_update_data(data, losses)
        return losses

    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
        finally:
            self.is_run = False
        return self.best_metrics_all


def offline_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for offline_trainer experiment method.

    It is identical to ``OfflineTrainer(...).experiment()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OfflineTrainer(*args, **kwargs).run()


offline_trainer_iter = OfflineTrainer
