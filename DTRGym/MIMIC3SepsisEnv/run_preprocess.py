from preprocess_util import ActionTransformer, \
    get_reward, get_obs, split_patient_by_stratified_patients, save_to_buffer, matchID, get_data_from_patient_list, \
    save_data, check_step, check_ID, preprocess_obs
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from DTRGym.icu_scoring import get_NEWS2_score, determine_HRF
from DTRGym.MIMIC3SepsisEnv import visu_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def plot_LOS(train_val_obs, test_obs, save_dir):
    LOS_train_val = train_val_obs.reset_index().groupby("icustay_id")["step"].max()
    LOS_test = test_obs.reset_index().groupby("icustay_id")["step"].max()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].hist(4 * LOS_train_val, bins=12)
    ax[0].set_xlabel("LOS/hours")
    ax[0].set_ylabel("population")
    ax[0].title.set_text('train val LOS')

    ax[1].hist(4 * LOS_test, bins=12)
    ax[1].set_xlabel("LOS/hours")
    ax[1].set_ylabel("population")
    ax[1].title.set_text('test LOS')
    plt.savefig(os.path.join(save_dir, "LOS.png"))
    plt.close()
    return


def reverse_cumulative_sum_with_discount(x):
    reversed_cumsum = np.cumsum(x.iloc[::-1])  # Compute cumulative sum in reverse
    discount_factors = args.gamma ** np.arange(len(x))
    return (reversed_cumsum * discount_factors).iloc[::-1]  # Apply discount and reverse back


def calculate_news2(row):
    is_CVPU = 1 if row["GCS"] < 15 else 0
    score = get_NEWS2_score(respiratory_rate=row["RR"],
                            SpO2=row["SpO2"],
                            on_vent=row["mechvent"],
                            blood_pressure=row["SysBP"],
                            heart_rate=row["HR"],
                            is_CVPU=is_CVPU,
                            temperature=row["Temp_C"],
                            is_AHRF=False)  # assume no HRF
    return score


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_splits", default=[0.7, 0.15, 0.15], nargs='+', type=float)
    parser.add_argument("--norm_obs", default=True, type=bool)
    parser.add_argument("--max_len", default=18, type=int)
    parser.add_argument("--reward_option", default="SOFA", choices=["Outcome", "NEWS2", "SOFA"])

    args = parser.parse_args()

    data_dir = os.path.join(Path(__file__).parent)
    target_dir = os.path.join(Path(__file__).parent.parent, f"MIMIC3Sepsis{args.reward_option}Env", "offline_data")
    plot_dir = os.path.join(target_dir, "plots")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    seed_everything(args.seed)
    sepsis_df = pd.read_csv(os.path.join(data_dir, "mimictable.zip"), compression="zip")
    if args.debug_mode:
        sepsis_df = sepsis_df[sepsis_df["icustayid"].isin(sepsis_df["icustayid"].unique()[:1000])]
    sepsis_df["icustayid"] = sepsis_df["icustayid"].astype(int)
    sepsis_df["bloc"] = sepsis_df["bloc"].astype(int)
    sepsis_df["bloc"] -= 1
    sepsis_df.set_index(["icustayid", "bloc"], inplace=True)
    sepsis_df.index.names = ["icustay_id", "step"]
    sepsis_df["outcome"] = sepsis_df["died_within_48h_of_out_time"]
    print(f"{len(sepsis_df.index.get_level_values('icustay_id').unique())} patients before preprocessing")
    # -----filter discontinuous patients
    # Calculate the maximum step for each patient
    max_steps = sepsis_df.groupby(level='icustay_id').apply(lambda df: df.index.get_level_values('step').max())
    actual_counts = sepsis_df.groupby(level='icustay_id').size()
    expected_counts = max_steps + 1
    discontinuous_patients = expected_counts[expected_counts != actual_counts].index
    sepsis_df = sepsis_df[~sepsis_df.index.get_level_values("icustay_id").isin(discontinuous_patients)]
    sepsis_df.sort_values(["icustay_id", "step"], ascending=True, inplace=True)
    remaining_patients = len(sepsis_df.index.get_level_values('icustay_id').unique())
    print(f"{remaining_patients} patients left after filtering patients with discontinuous bloc")
    del max_steps, actual_counts, expected_counts, discontinuous_patients, remaining_patients

    # ------filter patients >24hrs
    p_gt24 = sepsis_df.reset_index().groupby("icustay_id")["step"].max() >= 5
    sepsis_df = sepsis_df[
        sepsis_df.index.get_level_values("icustay_id").isin([p for p, true in p_gt24.items() if true])]
    assert (sepsis_df.groupby("icustay_id").tail(1).index.get_level_values("step").to_numpy() >= 5).all()
    print(
        f"{len(sepsis_df.index.get_level_values('icustay_id').unique())} patients left after filtering patients<24hr LOS")
    # ------filter patients with 24hr admission but not died

    sepsis_df1 = sepsis_df.groupby("icustay_id").head(args.max_len + 1)
    assert sepsis_df1.index.get_level_values("step").max() == args.max_len
    sepsis_df.sort_values(["icustay_id", "step"], ascending=True, inplace=True)
    print(f"{len(sepsis_df) - len(sepsis_df1)} samples removed for LOS>72 ")
    print("here we take 72+4 hours data, since truncated episodes needs obs_next for the 19th step")
    sepsis_df = sepsis_df1
    del sepsis_df1

    # add termination, truncate and obs_next
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    sepsis_df['terminated'] = False
    sepsis_df['truncated'] = False
    sepsis_df['obs_next'] = False
    for patient_id in tqdm(sepsis_df.index.get_level_values('icustay_id').unique(),
                           desc="adding termination, truncate and obs_next"):
        max_len = sepsis_df.loc[patient_id].index.max()
        if max_len <= args.max_len - 1:
            # Mark the last step as terminated
            sepsis_df.loc[(patient_id, max_len), 'terminated'] = True
        elif max_len == args.max_len:
            # Mark the mth step as truncated
            sepsis_df.loc[(patient_id, args.max_len - 1), 'truncated'] = True
        else:
            raise ValueError(f"Invalid length of patient {patient_id}, something goes wrong?")
    print(f"{sepsis_df['terminated'].sum()} terminated episodes",
          f"{sepsis_df['truncated'].sum()} truncated episodes")
    assert sepsis_df['terminated'].sum() + sepsis_df['truncated'].sum() == len(
        sepsis_df.index.get_level_values('icustay_id').unique())
    # add NEWS2 score
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    print("adding NEWS2 score, this might take a while")
    NEWS2 = []
    sepsis_df["NEWS2"] = sepsis_df["NEWS2"] = sepsis_df.apply(calculate_news2, axis=1)
    sepsis_df['rate_NEWS2'] = sepsis_df.groupby('icustay_id')['NEWS2'].diff().fillna(0)


    # get obs, action and reward
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    if not args.debug_mode:
        visu_data.plot_NEWS2_by_quantiles(sepsis_df, plot_dir, quantiles=[0.25, 0.5, 0.75], method="score")
        visu_data.plot_NEWS2_by_quantiles(sepsis_df, plot_dir, quantiles=[0.25, 0.5, 0.75], method="diff")
        visu_data.plot_NEWS2_by_quantiles(sepsis_df, plot_dir, quantiles=[0.25, 0.5, 0.75], method="rate")

    # get obs
    statics_df, obs, statistics = get_obs(sepsis_df)
    statistics.to_excel(os.path.join(target_dir, "statistics.xlsx"))

    reward = get_reward(sepsis_df, args.reward_option, max_len=args.max_len)
    visu_data.plot_reward(reward, plot_dir)
    trans_a = ActionTransformer(sepsis_df)
    a = trans_a.fit(args.max_len)
    visu_data.plot_action(a, plot_dir)

    print(f"{len(statics_df)} patients left after filtering patients with less than 24 hour admission")
    print(f"static size {statics_df.shape}")
    print(f"obs size {obs.shape}")
    print(f"action size {len(a['action_idx'].unique())}")

    # check length for truncated patients
    assert sepsis_df["truncated"].sum() == len(obs) - len(reward)
    assert len(a) == len(reward)

    # get raw csv
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    statics_norm = statics_df.copy()
    scaler = MinMaxScaler()
    statics_norm[statics_norm.columns] = scaler.fit_transform(statics_norm[statics_norm.columns])
    features = obs.merge(statics_df, on="icustay_id", how="right")
    features_norm = obs.merge(statics_norm, on="icustay_id", how="right")
    scaler = MinMaxScaler()
    features_norm[features_norm.columns] = scaler.fit_transform(features_norm[features_norm.columns])
    features.to_csv(os.path.join(target_dir, "features.csv"))
    features_norm.to_csv(os.path.join(target_dir, "features_norm.csv"))
    a.to_csv(os.path.join(target_dir, "action.csv"))
    outcome = sepsis_df[["died_in_hosp", "died_within_48h_of_out_time", "mortality_90d",
                         "delay_end_of_record_and_discharge_or_death"]].groupby("icustay_id").head(1).reset_index()
    outcome.drop(columns=["step"], inplace=True)
    outcome.to_csv(os.path.join(target_dir, "outcome.csv"))
    del statics_norm, features, features_norm, scaler, outcome
    # compute Q value (monte carlo)
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    print("computing Q value, this might take a while")
    reward['Q'] = reward.groupby(level='icustay_id')['r_sum'].transform(reverse_cumulative_sum_with_discount) \
        if not args.debug_mode else np.nan

    #  split by stratified patient groups
    patient_groups = visu_data.plot_NEWS2_by_value(sepsis_df, plot_dir, values=[-0.4, -0.15, 0, 0.15, 0.4],
                                                   method="rate", split_by_std=True)
    train_indices, val_indices, test_indices = {}, {}, {}

    outcome_df = sepsis_df["outcome"].groupby("icustay_id").head(1).reset_index()
    outcome_df.drop(columns=["step"], inplace=True)
    outcome_df.set_index("icustay_id", inplace=True)

    for group_name, indices in patient_groups.items():
        print(f"patients in group {group_name}: {len(indices)}")
        dataset_df = sepsis_df[sepsis_df.index.get_level_values("icustay_id").isin(indices)]
        dataset_outcome_df = outcome_df[outcome_df.index.isin(indices)]
        alive = dataset_outcome_df[dataset_outcome_df == 0].index
        dead = dataset_outcome_df[dataset_outcome_df == 1].index
        train1, tmp1 = train_test_split(alive, test_size=sum(args.data_splits[-2:]), random_state=args.seed)
        val1, test1 = train_test_split(tmp1, test_size=args.data_splits[-1] / sum(args.data_splits[-2:]),
                                       random_state=args.seed)

        train2, tmp2 = train_test_split(dead, test_size=sum(args.data_splits[-2:]), random_state=args.seed)
        val2, test2 = train_test_split(tmp2, test_size=args.data_splits[-1] / sum(args.data_splits[-2:]),
                                       random_state=args.seed)

        train_indices[group_name] = np.concatenate([train1, train2])
        val_indices[group_name] = np.concatenate([val1, val2])
        test_indices[group_name] = np.concatenate([test1, test2])

    # save all data
    all_train_indices = np.concatenate([train_indices[group_name] for group_name in train_indices.keys()])
    all_val_indices = np.concatenate([val_indices[group_name] for group_name in val_indices.keys()])
    all_test_indices = np.concatenate([test_indices[group_name] for group_name in test_indices.keys()])
    # for OPE, testing set must be included as training set. Val set is used for validation.
    ope_train_indices = np.concatenate([all_train_indices, all_test_indices])

    train_statics, train_obs, _, _ = get_data_from_patient_list(list(all_train_indices), statics_df, obs, a, reward)

    demog_scaler = MinMaxScaler()
    vital_scaler = MinMaxScaler()
    demog_scaler.fit(train_statics)
    vital_scaler.fit(train_obs)

    # save to buffer
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    for dataset, idx in zip(["train", "val", "test", "ope_train"], [all_train_indices, all_val_indices,
                                                                    all_test_indices, ope_train_indices]):
        statics_, obs_, a_, reward_ = get_data_from_patient_list(list(idx), statics_df, obs, a, reward)
        if args.norm_obs:
            statics_ = pd.DataFrame(demog_scaler.transform(statics_), columns=statics_.columns, index=statics_.index)
            obs_ = pd.DataFrame(vital_scaler.transform(obs_), columns=obs_.columns, index=obs_.index)
        save_to_buffer(os.path.join(target_dir, f"all_{dataset}_buffer.hdf5"), sepsis_df, statics_, obs_, reward_,
                       a_)

    for dataset, indices_dict in zip(["train", "val", "test"], [train_indices, val_indices, test_indices]):
        for group_name, idx in indices_dict.items():
            statics_, obs_, a_, reward_ = get_data_from_patient_list(list(idx), statics_df, obs, a, reward)
            if args.norm_obs:
                statics_ = pd.DataFrame(demog_scaler.transform(statics_), columns=statics_.columns, index=statics_.index)
                obs_ = pd.DataFrame(vital_scaler.transform(obs_), columns=obs_.columns, index=obs_.index)
            save_to_buffer(os.path.join(target_dir, f"{dataset}_{group_name}_buffer.hdf5"), sepsis_df, statics_, obs_,
                           reward_, a_)

    # save exclude NEWS2 train data
    exclude_train_indices = np.concatenate([all_train_indices[1:]])
    statics_, obs_, a_, reward_ = get_data_from_patient_list(list(exclude_train_indices), statics_df, obs, a, reward)
    if args.norm_obs:
        statics_ = pd.DataFrame(demog_scaler.transform(statics_), columns=statics_.columns, index=statics_.index)
        obs_ = pd.DataFrame(vital_scaler.transform(obs_), columns=obs_.columns, index=obs_.index)
    save_to_buffer(os.path.join(target_dir, f"train_exclude_1_buffer.hdf5"), sepsis_df, statics_, obs_, reward_,
                   a_)
