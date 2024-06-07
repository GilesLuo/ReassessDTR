import numpy as np
import torch
from HD4RL.core.helper_fn import get_baseline_policy_class
from DTRGym.base import make_env
from tianshou.data import Collector
from HD4RL.utils.data import load_buffer
from DTRGym import buffer_registry
from HD4RL.OPE import OPE_wrapper
from HD4RL.OPE.ImportanceSampling import ImportanceRatio
import pandas as pd
from tqdm import tqdm
from HD4RL.utils.data import TianshouDataset, TianshouEpisodeDataset, collate_batch_seq2seq
from HD4RL.utils.summary import get_sweep, get_best_run
from HD4RL.core.helper_fn import get_class, policy_load
from HD4RL.offline.offlineRLHparams import OfflineRLHyperParameterSpace, DummyHparam
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


def get_ratio_plot(ratio_dict, policy_name, save_dir, max_ratio=1000):
    sns.set()
    # Prepare data for box plot
    data_to_plot = pd.DataFrame([(buffer_name.replace("-ratios", "").replace("test_", "").replace("_", " ").replace(
        "rate", "r").replace("std", ""),
                                  np.clip(ratio, None, max_ratio + 1))
                                 for buffer_name, ratios in ratio_dict.items()
                                 for ratio in ratios],
                                columns=['Patient Cohort', 'Importance Ratio per Episode'])
    counts_over_1000 = data_to_plot[data_to_plot['Importance Ratio per Episode'] > max_ratio].groupby(
        'Patient Cohort').size()

    data_to_plot.sort_values(by=['Patient Cohort'], inplace=True)
    # Plot box plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.boxplot(x='Patient Cohort', y='Importance Ratio per Episode', data=data_to_plot, ax=ax, showfliers=True,
                     fliersize=1)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    for i, buffer_name in enumerate(data_to_plot['Patient Cohort'].unique()):
        count = counts_over_1000.get(buffer_name, 0)
        if count > 0:
            ax.text(i, ax.get_ylim()[1] * 0.6, f'+{count}', ha='center', va='bottom', fontsize=18)
    ax.set_yticks([1e-4, 0.01, 1, 100, max_ratio])
    ax.set_yticklabels(ax.get_yticks(), fontsize=18)
    ax.set_ylim(bottom=1e-4, top=max_ratio)
    ax.set_title(f"{policy_name} Policy - Importance Ratio")
    ax.set_ylabel("Importance Ratio", fontsize=18)
    ax.set_yscale('log')
    plt.xticks(rotation=80, fontsize=18)
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(os.path.join(save_dir, f"{policy_name}-ratio_boxplot.pdf"))
    plt.close(fig)



def identify_upper_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outliers = []
    for value in data:
        if value > Q3 + outlier_step:
            outliers.append(value)
    return outliers

def identify_upper_95quantile(data):
    q = np.percentile(data, 99)
    return list(data[data > q])

def get_ratio_outliers(ratio_dict, policy_name, save_dir, max_ratio=10e10):
    # Prepare data for box plot
    sns.set()
    data_to_plot = pd.DataFrame([(buffer_name.replace("-ratios", "").replace("test_", "").replace("_", " ").replace(
        "rate", "r").replace("std", ""),
                                  np.clip(ratio, None, max_ratio + 1))
                                 for buffer_name, ratios in ratio_dict.items()
                                 for ratio in ratios],
                                columns=['Patient Cohort', 'Importance Ratio per Episode'])
    data_to_plot.sort_values(by=['Patient Cohort'], inplace=True)
    outliers_by_cohort = data_to_plot.groupby('Patient Cohort')['Importance Ratio per Episode'].apply(identify_upper_outliers)
    outliers_data = []
    for cohort, outliers in outliers_by_cohort.items():
        for outlier in outliers:
            outliers_data.append({'Patient Cohort': cohort, 'Outlier': outlier})
    outliers_df = pd.DataFrame(outliers_data)
    fig, ax = plt.subplots(figsize=(10, 10))
    (sns.stripplot(x='Patient Cohort', y='Outlier', log_scale=True, hue='Patient Cohort', size=5,
                  data=outliers_df, ax=ax))
    ax.set_xlabel('Patient Cohort', fontsize=18)
    ax.set_ylabel('Importance Ratio Outlier', fontsize=18)
    plt.xticks(rotation=80,fontsize=18)

    plt.tight_layout()  # Adjust subplot spacing
    # Plot box plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(os.path.join(save_dir, f"{policy_name}-ratio_outliers.pdf"))
    plt.close(fig)


def test_baseline_offline(env_name, policy_name, train_buffer_name, test_buffer_keyword, test_seed,
                          num_actions, OPE_names, gamma, behavioural_fn=None, value_fn=None, plot_ratio=None):
    if policy_name == "all":
        policy_names = ["doctor return", ] + list(get_baseline_policy_class(policy_name).keys())
        result = []
        for p in policy_names:
            result.append(
                test_baseline_offline(env_name, p, train_buffer_name, test_buffer_keyword, test_seed,
                                      num_actions, OPE_names, gamma, behavioural_fn, value_fn, plot_ratio))
        result = pd.concat(result)

        return result
    else:
        scores = []
        if policy_name == "doctor return":
            score = {}
            test_buffers = {k: load_buffer(v) for k, v in
                            buffer_registry.make_all(env_name, test_buffer_keyword).items()}
            for buffer_name, buffer in test_buffers.items():
                score[buffer_name + "-return"] = []
                dataset = TianshouEpisodeDataset(buffer)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                                         collate_fn=collate_batch_seq2seq)
                for batch in dataloader:
                    discount_array = np.where(batch.mask, gamma ** (np.cumsum(batch.mask, axis=1) - 1), 0.0)
                    score[buffer_name + "-return"].append(
                        (batch.rew * np.squeeze(batch.mask * discount_array)).sum(axis=1))
                score[buffer_name + "-return"] = np.concatenate(score[buffer_name + "-return"], axis=0).mean()
            scores.append(score)
            scores = pd.DataFrame(scores)
            scores["policy_name"] = policy_name
            return scores

        env, _, test_envs = make_env(env_name, int(test_seed), 1, 1, num_actions)
        if policy_name == "behavioural":
            behavioural_run = get_sweep("SepsisRL", f"MIMIC3SepsisEnv-all_train-",
                                        "discrete-imitation-obs_cat")
            behavioural_run = get_best_run(behavioural_run, "val/all_val-PatientWiseF1", maximize=True)
            policy = get_class("obj", "discrete-imitation-obs_cat", offline=True)(
                f"{args.env}",
                dummy_hparam, device=args.device,
                logger="wandb",
                train_buffer_name=args.train_buffer, val_buffer_name=args.train_buffer,  # placeholder
                test_buffer_keyword=args.test_buffer_keyword,
                OPE_methods=['PatientWiseF1'],  # placeholder
                metric="PatientWiseF1",  # placeholder
                test_online=False).define_policy(**behavioural_run.to_dict())
            policy = policy_load(policy, os.path.join(behavioural_run["model_dir"], "policy.pth"),
                                 args.device, is_train=False)
        else:
            policy_cls = get_baseline_policy_class(policy_name)
            train_buffer = load_buffer(buffer_registry.make(env_name, train_buffer_name))
            policy = policy_cls(env.action_space, train_buffer)
        test_buffers = {k: load_buffer(v) for k, v in
                        buffer_registry.make_all(env_name, test_buffer_keyword).items()}

        if plot_ratio:
            print("Plotting ratio histogram")
            ratio_estimator = ImportanceRatio(buffers=test_buffers, num_action=num_actions, gamma=gamma,
                                              behavior_policy=behavioural_fn, modes=None,
                                              value_function=None)
            ratio_dict = ratio_estimator.evaluate(policy)
            ratio_dict = {k: v for k, v in ratio_dict.items() if k.endswith("ratios")}
            get_ratio_plot(ratio_dict, policy_name, plot_ratio)
            get_ratio_outliers(ratio_dict, policy_name, plot_ratio)
        OPE = OPE_wrapper(OPE_names=OPE_names, buffers=test_buffers,
                          behavior_policy=behavioural_fn, value_function=value_fn,
                          num_actions=env.action_space.n, all_soften=True if policy_name == "behavioural" else False)
        print(f"Testing {policy_name} policy")
        scores.append(OPE(policy))
        scores = pd.DataFrame(scores)
        scores["policy_name"] = policy_name

        return scores


# Function to format the mean and std values to two decimal places
def format_mean_std(mean, std):
    mean_formatted = "{:.2f}".format(mean)
    std_formatted = "{:.2f}".format(std)
    return f"{mean_formatted} ± {std_formatted}"


if __name__ == "__main__":
    import argparse
    from experiment.run_sepsis import load_behavioural_fn, get_value_fn
    import warnings

    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="SepsisRL-ICML-new")
    parser.add_argument("--env", type=str, default="MIMIC3SepsisOutcomeEnv")
    parser.add_argument("--save_dir", type=str, default="results/naive_baseline/Outcome")
    parser.add_argument("--plot_ratio", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=0.99)
    parser.add_argument("--num_actions", type=int, default=25)
    parser.add_argument("--policy_name", type=str, default="all", choices=["all", "random"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_buffer", type=str, default="all_train")
    parser.add_argument("--test_buffer_keyword", type=str, default="test", help="keyword to find all test buffer")
    parser.add_argument("--OPE_methods", nargs='+', choices=['WIS', 'WIS_bootstrap',
                                                             'WIS_truncated', 'WIS_bootstrap_truncated',
                                                             'PatientWiseF1', "SampleWiseF1", "TD",
                                                             "DR",
                                                             "MCQ"],
                        default=['WIS', 'WIS_bootstrap', 'WIS_truncated', 'WIS_bootstrap_truncated', "DR",
                                 'PatientWiseF1', "SampleWiseF1", "doseRMSE"],
                        help="Select one or more options from the list")

    parser.add_argument('--all_soften', default=False, action='store_true', )
    parser.add_argument('--use_calibrate', default=False, action='store_true', )
    parser.add_argument('--bc_algo', default="discrete-imitation-rnn")
    parser.add_argument('--behavioural_model_path',
                        default="/home/reub0014/projects/SimMedEnv/experiment/saved_models/bc_policy.pth", )
    parser.add_argument('--calibrated_model_path', default="/home/reub0014/projects/"
                                                           "SimMedEnv/experiment/saved_models/calibrated_model.pt")
    parser.add_argument('--value_algo', default="offlinesarsa-rnn")
    parser.add_argument('--value_model_path',
                        default="/home/reub0014/projects/SimMedEnv/experiment/saved_models/Outcome_value_policy.pth")
    args = parser.parse_known_args()[0]
    os.makedirs(args.save_dir, exist_ok=True)
    dummy_hparam = DummyHparam()

    ope_args = {}
    if args.use_calibrate:
        print("Calibrating the model!!!!")
    else:
        print("Using the uncalibrated model!!!")

    # get value function from offline sarsa checkpoint
    if "IS" in args.OPE_methods or "WIS" in args.OPE_methods or "WIS_bootstrap" in args.OPE_methods or \
            "WIS_truncated" in args.OPE_methods or "WIS_bootstrap_truncated" in args.OPE_methods or \
            "DR" in args.OPE_methods or "WDR" in args.OPE_methods or "PDDR" in args.OPE_methods or "PDWDR" in args.OPE_methods:
        ope_args["behavioural_fn"] = load_behavioural_fn(args.project, "MIMIC3SepsisNEWS2Env",
                                                         args.bc_algo, args.device,
                                                         behavioural_model_path=args.behavioural_model_path,
                                                         calibrate=args.use_calibrate,
                                                         calibrated_model_path=args.calibrated_model_path,
                                                         val_buffer=load_buffer(
                                                             buffer_registry.make(args.env, "all_test")))  # placeholder

    if "DR" in args.OPE_methods or "WDR" in args.OPE_methods or "PDDR" in args.OPE_methods or "PDWDR" in args.OPE_methods:
        ope_args["value_fn"] = get_value_fn(args.project, args.env, args.value_algo, args.device,
                                            value_model_path=args.value_model_path)

    args.env += "-discrete"
    np.random.seed(args.seed)

    results = []
    print(f"Testing with seed {args.seed}")
    result = test_baseline_offline(args.env, args.policy_name, args.train_buffer, args.test_buffer_keyword,
                                   args.seed, args.num_actions, args.OPE_methods, args.gamma,
                                   behavioural_fn=ope_args["behavioural_fn"], value_fn=ope_args["value_fn"],
                                   plot_ratio=args.save_dir if args.plot_ratio else None)
    results.append(result)
    results = pd.concat(results)

    results.to_csv(os.path.join(args.save_dir, f"baseline-all.csv"))
    mean_result = results.groupby("policy_name").mean().round(2)
    all_columns = [col for col in mean_result.columns if "all" in col]
    rest_columns = [col for col in mean_result.columns if col not in all_columns]
    all_result = mean_result[all_columns]
    rest_result = mean_result[rest_columns]

    all_result.to_csv(os.path.join(args.save_dir, f"baseline-all-mean.csv"))
    rest_result.to_csv(os.path.join(args.save_dir, f"baseline-stratified-mean.csv"))
    print()
