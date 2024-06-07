
import torch
from HD4RL.OPE.base import BaseOPE
import numpy as np
from typing import Dict
from tqdm import tqdm
from HD4RL.utils.data import TianshouDataset, collate_batch_seq2seq


class DoseRMSE(BaseOPE):
    def __init__(self, buffers, num_actions, gamma=0.99, **kwargs):
        super().__init__(buffers, num_actions)
        self.gamma = gamma  # Discount factor
        self.iv_median_doses = [0, 40.0, 93.75, 315.35, 949.8]   # List of median doses for IV
        self.vaso_median_doses = [0, 0.044, 0.15, 0.301, 0.9]  # List of median doses for vasopressors

    def evaluate(self, policy) -> Dict[str, float]:
        iv_diff = {buffer_name: [] for buffer_name in self.buffers.keys()}
        vaso_diff = {buffer_name: [] for buffer_name in self.buffers.keys()}

        for buffer_name, buffer in self.buffers.items():
            dataset = TianshouDataset(buffer, stack_num=self.stack_num)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                                     collate_fn=collate_batch_seq2seq)
            for batch in tqdm(dataloader, desc=f"eval RMSE for {buffer_name}"):
                batch.to_torch(dtype=torch.float32)
                # Compute target policy probabilities
                output_batch = policy(batch)
                act_pred = output_batch.act
                if isinstance(act_pred, torch.Tensor):
                    act_pred = act_pred.cpu().numpy()
                gt_iv = batch.info["input_4hourly"]
                gt_iv = gt_iv if gt_iv.nelement() == len(act_pred) else gt_iv[:, -1]
                gt_vaso = batch.info["max_dose_vaso"]
                gt_vaso = gt_vaso if gt_vaso.nelement() == len(act_pred) else gt_vaso[:, -1]

                gt_iv_idx = batch.info["iv_fluid_action"]
                gt_iv_idx = gt_iv_idx if gt_iv_idx.nelement() == len(act_pred) else gt_iv_idx[:, -1]
                gt_vaso_idx = batch.info["vasopressor_action"]
                gt_vaso_idx = gt_vaso_idx if gt_vaso_idx.nelement() == len(act_pred) else gt_vaso_idx[:, -1]

                # Convert actions to doses
                pred_iv_doses = np.array([self.iv_median_doses[action // 5] for action in act_pred])
                pred_vaso_doses = np.array([self.vaso_median_doses[action % 5] for action in act_pred])

                # Calculate MSE for each prediction
                mse_iv = (gt_iv.cpu().numpy() - pred_iv_doses) ** 2
                mse_vaso = (gt_vaso.cpu().numpy() - pred_vaso_doses) ** 2

                # set the MSE to 0 if the action idx is the same as the gt
                mse_iv[act_pred % 5 == gt_iv_idx] = 0
                mse_vaso[act_pred // 5 == gt_vaso_idx] = 0

                # Store RMSE values
                iv_diff[buffer_name].append(mse_iv)
                vaso_diff[buffer_name].append( mse_vaso)
        # Average the RMSE values across all buffers
        iv_diff = {buffer_name: np.sqrt(np.mean(np.concatenate(diff))) for buffer_name, diff in iv_diff.items()}
        vaso_diff = {buffer_name: np.sqrt(np.mean(np.concatenate(diff))) for buffer_name, diff in vaso_diff.items()}

        return {f"{buffer_name}-iv_RMSE": iv_diff[buffer_name] for buffer_name in self.buffers.keys()} | \
                  {f"{buffer_name}-vaso_RMSE": vaso_diff[buffer_name] for buffer_name in self.buffers.keys()}

    def action2dose(self, action):
        iv = action // 5
        vaso = action % 5
        # Convert to median dose
        return self.iv_median_doses[iv], self.vaso_median_doses[vaso]