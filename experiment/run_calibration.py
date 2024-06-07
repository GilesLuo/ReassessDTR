import torch
import os
from HD4RL.utils.data import load_buffer
from DTRGym import buffer_registry
from HD4RL.utils.data import TianshouDataset, collate_batch_seq2seq
import numpy as np
from tianshou.data import Batch
import argparse
from tqdm import tqdm
from experiment.run_sepsis import load_behavioural_fn
import matplotlib.pyplot as plt
from HD4RL.utils.misc import set_global_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="SepsisRL-ICML-new")
    parser.add_argument("--env_name", type=str, default="MIMIC3SepsisNEWS2Env")
    parser.add_argument('--behavioural_model_path',
                        default="/home/reub0014/projects/SimMedEnv/experiment/saved_models/bc_policy.pth")
    parser.add_argument("--train_buffer", type=str, default="all_train")
    parser.add_argument("--val_buffer", type=str, default="all_val")
    parser.add_argument("--test_buffer_keyword", type=str, default="test", help="keyword to find all test buffer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_save_path", type=str, default="/home/reub0014/projects/"
                                                              "SimMedEnv/experiment/saved_models/calibrated_model.pt")
    parser.add_argument("--plot_save_dir", type=str, default="/home/reub0014/projects/SimMedEnv/experiment/results/calibration_plot")
    args = parser.parse_known_args()[0]

    set_global_seed(0)
    os.makedirs(args.plot_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    raw_fn = load_behavioural_fn(args.project, args.env_name, "discrete-imitation-rnn", args.device,
                                 behavioural_model_path=args.behavioural_model_path,
                                 calibrate=False, )

    calibrated_fn = load_behavioural_fn(args.project, args.env_name, "discrete-imitation-rnn", args.device,
                                        calibrate=True,
                                        behavioural_model_path=args.behavioural_model_path,
                                        calibrated_model_path=args.model_save_path,
                                        val_buffer=load_buffer(
                                            buffer_registry.make(args.env_name, "all_test")))

    test_buffers = {k: load_buffer(v) for k, v in buffer_registry.make_all(args.env_name,
                                                                           args.test_buffer_keyword).items()}

    buffers = {args.train_buffer: load_buffer(buffer_registry.make(args.env_name, args.train_buffer)),
               args.val_buffer: load_buffer(buffer_registry.make(args.env_name, args.val_buffer))}

    buffers = {**buffers, **test_buffers}

    for buffer_name, buffer in buffers.items():
        dataset = TianshouDataset(buffer, stack_num=raw_fn["stack_num"])
        raw_probs = []
        calibrated_probs = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False,
                                                     collate_fn=collate_batch_seq2seq)
            for batch in tqdm(dataloader, desc=buffer_name):
                batch.to_torch(dtype=torch.float32)
                obs = batch.obs
                raw_prob = raw_fn["model"](Batch(obs=obs, info={})).prob
                calibrated_prob = calibrated_fn["model"](Batch(obs=obs, info={})).prob
                raw_probs.append(raw_prob[np.arange(len(batch.act)), batch.act.long()])
                calibrated_probs.append(calibrated_prob[np.arange(len(batch.act)), batch.act.long()])
        raw_probs = torch.cat(raw_probs)
        calibrated_probs = torch.cat(calibrated_probs)
        raw_probs = raw_probs.cpu().numpy()
        calibrated_probs = calibrated_probs.cpu().numpy()

        plt.figure(figsize=(8, 4))
        mean_raw_probs = np.mean(raw_probs)
        mean_calibrated_probs = np.mean(calibrated_probs)
        plt.hist(raw_probs, bins=50, alpha=0.5, label="Uncalibrated", color="green", density=True)
        plt.hist(calibrated_probs, bins=50, alpha=0.5, label="Calibrated", color="red", density=True)

        plt.xlabel("Output Probability", fontsize=22)
        plt.ylabel("Density", fontsize=22)
        plt.yscale("log")
        plt.legend(fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.title(buffer_name.replace('_', ' '), fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(args.plot_save_dir, f"calibration_{buffer_name}.pdf"))
        plt.close()

