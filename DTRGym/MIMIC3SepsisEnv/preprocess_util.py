from copy import deepcopy

import pandas as pd
from scipy.stats import zscore, rankdata
import copy
import itertools
from typing import Any, Dict, List, Tuple, Union
import os
import matplotlib.pyplot as plt
import numpy as np
from tianshou.data import Batch, ReplayBuffer
from tqdm import tqdm


class BaseTransformer:
    _id_key = ("icustay_id", "step")

    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError


def get_obs(mimic_data: pd.DataFrame):
    """
    extract 46 features (no input_4hourly or max_vaso since they are actions!)
    """

    def get_data_statistics(sepsis_df):
        demog_features = ["age", "gender", "re_admission", "Weight_kg", "elixhauser"]
        interv_features = ["mechvent", 'input_4hourly', 'max_dose_vaso']
        other_features = ["died_in_hosp", "mortality_90d"]
        demog_df = sepsis_df[demog_features]
        interv_df = sepsis_df[interv_features]
        other_df = sepsis_df[other_features]

        demog_df = demog_df.groupby("icustay_id").head(1).reset_index().set_index("icustay_id").drop("step", axis=1)
        other_df = other_df.groupby("icustay_id").head(1).reset_index().set_index("icustay_id").drop("step", axis=1)

        # 1 means female for gender, convert to male percentage
        demog_df["gender"] -= 1
        demog_df["gender"] = -demog_df["gender"]

        demog_std_df = demog_df.std()
        demog_std_df.index = [f"{i}_std" for i in demog_std_df.index]
        statistic_df = pd.concat([demog_df.mean(), demog_std_df, interv_df.mean(), other_df.mean()])
        return statistic_df

    mimic_data = mimic_data.copy()
    mimic_data["age"] /= 365
    # preprocessing time-varying features
    colbin = ['mechvent']
    colnorm = ['GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 'Potassium',
               'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb',
               'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3',
               'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance']
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili',
              'INR', 'output_4hourly', "input_total", "output_total"]

    # all_column = []
    # all_column.extend(colbin)
    # all_column.extend(colnorm)
    # all_column.extend(collog)
    # colbin = np.where(np.isin(mimic_data.columns, colbin))[0]
    # colnorm = np.where(np.isin(mimic_data.columns, colnorm))[0]
    # collog = np.where(np.isin(mimic_data.columns, collog))[0]
    # print(mimic_data.columns[46])
    # copy_mimic_table = mimic_data.values.copy()
    # fitted_mimic_data = np.concatenate([copy_mimic_table[:, colbin],
    #                                     copy_mimic_table[:, colnorm],
    #                                     np.log(0.1 + copy_mimic_table[:, collog].astype(float))],
    #                                    axis=1)
    # obs = pd.DataFrame(fitted_mimic_data, columns=all_column, index=mimic_data.index)
    df_log = mimic_data[collog].apply(lambda x: np.log(0.1 + x))
    obs = pd.concat([mimic_data[colbin], mimic_data[colnorm], df_log], axis=1)

    col_demog_norm = ['re_admission', 'gender', 'age', "elixhauser", 'Weight_kg']
    demog_df = mimic_data[col_demog_norm].groupby("icustay_id").head(1)
    demog_df.droplevel("step")
    statistic_df = get_data_statistics(mimic_data)
    return demog_df, obs, statistic_df


def preprocess_obs(demog_df, obs_df):
    # preprocessing demographics features
    demog_df = demog_df.copy()
    demog_df = (demog_df - demog_df.min()) / (demog_df.max() - demog_df.min())
    # preprocessing time-varying features
    obs_df = obs_df.copy()
    obs_df = (obs_df - obs_df.min()) / (obs_df.max() - obs_df.min())
    return demog_df, obs_df


class ActionTransformer(BaseTransformer):
    """
    Since all variables are measured IN the 4 hour interval, we do not know when the action happens before or after
    measurements are taken.
    Here we assume that the action happens at the end of the 4 hour interval, i.e. the action is taken AFTER the measurement.
    """

    def __init__(self, mimic_data: pd.DataFrame):
        super().__init__()
        self.mimic_data = deepcopy(mimic_data)
        self.action_idx = None
        self.action_space = None

    def fit(self, max_len):

        # Define bin edges
        iv_fluid_bins = [1e-10, 50, 180, 530]
        vasopressor_bins = [1e-10, 0.08, 0.22, 0.45]

        # Binning for IV fluids
        iv_fluid_values = self.mimic_data["input_4hourly"].values
        iv_fluid_action = np.digitize(iv_fluid_values, iv_fluid_bins, right=True)

        # Binning for Vasopressors
        vasopressor_values = self.mimic_data["max_dose_vaso"].values
        vasopressor_action = np.digitize(vasopressor_values, vasopressor_bins, right=True)

        iv_median_doses = [0]+[np.median(iv_fluid_values[iv_fluid_action == i]).round(2) for i in range(1, len(iv_fluid_bins) + 1)]
        vasopressor_median_doses = [0]+[np.median(vasopressor_values[vasopressor_action == i]).round(3) for i in
                                    range(1, len(vasopressor_bins) + 1)]

        # Combine actions into a single array
        combined_actions = np.stack((iv_fluid_action, vasopressor_action), axis=-1)

        # Find unique action pairs and create action index
        unique_actions, self.action_idx = np.unique(combined_actions, axis=0, return_inverse=True)
        print("unique actions in order: ", unique_actions)
        print("median doses in order: ", iv_median_doses, vasopressor_median_doses)
        # Construct action data DataFrame
        # Explicitly create a new DataFrame
        action_data = pd.DataFrame({
            "input_4hourly": iv_fluid_values,
            "max_dose_vaso": vasopressor_values,
            "iv_fluid_action" : iv_fluid_action,
            "vasopressor_action": vasopressor_action,
            "action_idx": self.action_idx,
            "median_dose_iv": [iv_median_doses[i] for i in iv_fluid_action],
            "median_dose_vaso": [vasopressor_median_doses[i] for i in vasopressor_action],
        }, index=self.mimic_data.index)

        action_data.reset_index(inplace=True)
        action_data.set_index(["icustay_id", "step"], inplace=True)
        action_data = action_data.groupby("icustay_id").head(max_len)
        return action_data


def get_reward(mimic_data: pd.DataFrame, reward_option, max_len):
    mimic_data = mimic_data.copy()
    reward_data = mimic_data[[]]
    reward_data["r_outcome"] = 0
    reward_data = reward_data.groupby("icustay_id").head(max_len)  # size of reward df should be 6-18

    if reward_option == "Outcome":
        # +100 for survival, -100 for death
        for patientID in tqdm(reward_data.index.get_level_values("icustay_id").unique(), desc="get outcome reward"):
            last_step = reward_data.loc[patientID, :].index.max()
            assert 5 <= last_step <= max_len - 1
            if mimic_data.loc[(patientID, 0), "outcome"]:
                reward_data.loc[(patientID, last_step), "r_outcome"] = -100
            else:
                reward_data.loc[(patientID, last_step), "r_outcome"] = 100
    elif reward_option == "NEWS2":
        # 0 for survival, -1 for death; NEWS2 score is between 0-18, normalize to 0-1
        for patientID in tqdm(reward_data.index.get_level_values("icustay_id").unique(), desc="get outcome reward"):
            last_step = reward_data.loc[patientID, :].index.max()
            assert 5 <= last_step <= max_len - 1
            if mimic_data.loc[(patientID, 0), "outcome"]:
                reward_data.loc[(patientID, last_step), "r_outcome"] = -1  # no penalty for survival
        r_score = (mimic_data["NEWS2"] - 0) / (18 - 0)
        reward_data[f"r_NEWS2"] = - 1 * r_score
    elif reward_option == "SOFA":
        # https://arxiv.org/pdf/1711.09602.pdf
        c0,c1,c2 = -0.025, -0.125, -2
        for patientID in tqdm(reward_data.index.get_level_values("icustay_id").unique(), desc="get outcome reward"):
            last_step = reward_data.loc[patientID, :].index.max()
            assert 5 <= last_step <= max_len - 1
            for step in range(last_step):
                sofa, sofa_next = mimic_data.loc[(patientID, step), "SOFA"], mimic_data.loc[(patientID, step + 1), "SOFA"]
                lactate, lactate_next = mimic_data.loc[(patientID, step), "Arterial_lactate"], mimic_data.loc[(patientID, step + 1), "Arterial_lactate"]
                reward_data.loc[(patientID, step), "r_sofa"] = (c0 * (sofa == sofa_next and sofa_next > 0)
                                                                + c1 * (sofa_next - sofa)
                                                                + c2 * float(np.tanh([lactate_next - lactate])))
            reward_data.loc[(patientID, last_step), "r_sofa"] = c0 * (sofa_next > 0)
            if mimic_data.loc[(patientID, 0), "outcome"]:
                reward_data.loc[(patientID, last_step), "r_outcome"] = -15
            else:
                reward_data.loc[(patientID, last_step), "r_outcome"] = 15
    else:
        raise NotImplementedError
    reward_data["r_sum"] = reward_data.sum(1)
    assert reward_data.index.get_level_values("step").max() <= max_len - 1
    return reward_data



def save_to_buffer(buffer_save_path, sepsis_df, statics, vital, reward, interv):
    replay_buffer = ReplayBuffer(len(reward))
    for icustay_id in tqdm(reward.index.get_level_values("icustay_id").unique(),
                           desc=f"saving data to buffer {buffer_save_path}"):
        max_step = len(reward.loc[icustay_id])
        survived = sepsis_df.loc[(icustay_id, 0), "outcome"] == 0
        for step in range(max_step):
            # prepare buffer value
            terminated = sepsis_df.loc[(icustay_id, step), "terminated"]
            truncated = sepsis_df.loc[(icustay_id, step), "truncated"]
            demog = statics.loc[icustay_id].to_numpy().reshape(-1)
            action = interv.loc[(icustay_id, step), "action_idx"]

            obs = vital.loc[(icustay_id, step), :]
            obs_next = vital.loc[(icustay_id, step), :] if terminated else vital.loc[(icustay_id, step + 1), :] # repeat last obs if terminated to avoid stupid error
            act_next = -1 * np.ones_like(action) if truncated or terminated else interv.loc[(icustay_id, step + 1), "action_idx"]

            r = reward.loc[(icustay_id, step), "r_sum"]
            sofa = sepsis_df.loc[(icustay_id, step), "SOFA"]
            news2 = sepsis_df.loc[(icustay_id, step), "NEWS2"]
            q = reward.loc[(icustay_id, step), "Q"]
            remaining_step = max_step - step - 1

            # adding values
            replay_buffer.add(
                Batch(
                    obs=np.concatenate([demog, obs]),
                    act=action,
                    rew=r,
                    terminated=terminated,
                    truncated=truncated,
                    obs_next=np.concatenate([demog, obs_next]),
                    info={"patientID": icustay_id,
                          "step": step,
                          'Q': np.array(q).reshape(-1),
                          "act": action,
                          "act_next": act_next,
                          "LOS": remaining_step,
                          "alive": 0 if step == max_step - 1 and not survived else 1,
                          "SOFA": sofa,
                          "NEWS2": news2,
                          "iv_fluid_action": interv.loc[(icustay_id, step), "iv_fluid_action"],
                          "vasopressor_action": interv.loc[(icustay_id, step), "vasopressor_action"],
                          "input_4hourly": interv.loc[(icustay_id, step), "input_4hourly"],
                          "max_dose_vaso": interv.loc[(icustay_id, step), "max_dose_vaso"],
                          "median_dose_iv": interv.loc[(icustay_id, step), "median_dose_iv"],
                          "median_dose_vaso": interv.loc[(icustay_id, step), "median_dose_vaso"],
                          }
                )
            )
    assert replay_buffer.done.sum() == len(np.unique(replay_buffer.info["patientID"]))
    replay_buffer.save_hdf5(buffer_save_path)
    assert len(replay_buffer.done.shape) == 1, "done.shape must be 1, otherwise tianshou buffer will fall"
    return replay_buffer


def split_patient_by_stratified_patients(proportions: list, patient_df, outcome_df, plot=False):
    def check_identity(set_to_check, target_list):
        # check identity
        a = set()
        for group in set_to_check:
            a = a.union(group)
        if a != set(target_list):
            raise AssertionError("something going wrong")

    patient_list = patient_df.index.get_level_values('icustay_id').unique().tolist()
    # -----------------------------get patient ID for each group
    _keys = [{True: "deceased", False: "survived"}, {True: "elder", False: "young"},
             {True: "female", False: "male"}]

    deceased_patient_list = outcome_df[outcome_df["mortality_90d"] == 1].index.tolist()
    decease_group = {True: deceased_patient_list, False: [i for i in patient_list if i not in deceased_patient_list]}

    elder_patient_list = patient_df[patient_df["age"] >= 60].index.tolist()
    elder_group = {True: elder_patient_list, False: [i for i in patient_list if i not in elder_patient_list]}

    female_patient_list = patient_df[patient_df["gender"] == 0].index.tolist()
    female_group = {True: female_patient_list, False: [i for i in patient_list if i not in female_patient_list]}

    combo = [list(i) for i in itertools.product([False, True], repeat=3)]
    groups = {}
    for setting in combo:
        group1 = copy.deepcopy(patient_list)
        name = []
        for idx, (sign, group) in enumerate(zip(setting, [decease_group, elder_group, female_group])):
            group1 = [i for i in group1 if i in group[sign]]
            name.append(_keys[idx][sign])
        groups.update({'-'.join(name): group1})
    # --------------------------------
    if plot:
        group_count = {key: len(value) for key, value in groups.items()}
        group_count_bar = {}
        for key, value in groups.items():
            if len(value) > 0:
                group_count_bar.update({key: len(value)})
        fig, axs = plt.subplots(2, figsize=(4, 6))
        plt.title("grouping histogram")
        axs[0].set_title('sub-group percentage')
        patches, texts, autotexts = axs[0].pie(group_count_bar.values(),
                                               labels=group_count_bar.keys(),
                                               autopct='%.2f%%')
        [axs[0].spines[location].set_visible(True) for location in ['top', 'right', 'left', 'bottom']]
        [axs[0].spines[location].set_color('black') for location in ['top', 'right', 'left', 'bottom']]
        from matplotlib import font_manager as fm
        proptease = fm.FontProperties()
        proptease.set_size('xx-small')
        plt.setp(autotexts, fontproperties=proptease)
        plt.setp(texts, fontproperties=proptease)

        axs[1].set(ylabel='number of patients')
        plt.axhline(y=len(patient_list), linewidth=1, color='k')
        plt.bar(group_count.keys(), group_count.values())

        plt.xticks(rotation=270)
        plt.gcf().subplots_adjust(bottom=0.5)
        plt.show()
    check_identity(groups.values(), patient_list)

    split_data = [[] for _ in range(len(proportions))]
    for group in groups.values():
        if group:
            if len(group) >= len(proportions):
                patientID_segments = split_list(list(group), proportions, shuffle=True)
                for idx, seg in enumerate(patientID_segments):
                    split_data[idx].extend(seg)
            else:
                split_data[-1].extend(group)
    check_identity(split_data, patient_list)
    return split_data


def get_data_from_patient_list(p_list, *dfs):
    df_list = []
    for df in dfs:
        df_list.append(df[df.index.get_level_values('icustay_id').isin(set(p_list))])
    return df_list


def split_list(patient_list: list, proportions, shuffle=True):
    patient_list = copy.deepcopy(patient_list)
    split_data = []
    if not 0.9999 < sum(proportions) <= 1.0001:
        raise ValueError(f"proportions {proportions} should sum up to one")
    elif len(patient_list) < len(proportions):
        raise ValueError(f"got {len(patient_list)} data but splitting into {len(proportions)} proportions")
    if shuffle:
        np.random.shuffle(patient_list)
    sizes = [int(len(patient_list) * p) for p in proportions]
    sizes[-1] += len(patient_list) - sum(sizes)
    for idx, size in enumerate(sizes):
        seg_p_list = patient_list[:size]
        patient_list = patient_list[size:]
        split_data.append(seg_p_list)
    return split_data


def minmax(x: pd.DataFrame or pd.Series):  # normalize
    return (x - x.min()) / (x.max() - x.min())


def matchID(target_df, *dfs):
    def matchID_(df):
        # match dataframe icu_stay_ID with static
        return df[df.index.get_level_values('icustay_id').isin(target_df.index.get_level_values('icustay_id'))]

    matched_dfs = []
    for df in dfs:
        matched_dfs.append(matchID_(df))
    return tuple(matched_dfs)


def check_ID(target_df, *dfs):
    # unify ID of vital_signs, medical intervention & statics
    idx = target_df.index.get_level_values('icustay_id')
    for df in dfs:
        if set(df.index.get_level_values('icustay_id')) != set(idx):
            raise ValueError(f"{df} Subject ID pools differ!")


def check_step(*dfs):
    for df in dfs:
        df.sort_values(['icustay_id', 'step'], inplace=True)
        prev_idx = (-1, 0)
        for idx in df.index:
            if not ((idx[1] == 0 and idx[0] != prev_idx[0])
                    or (idx[1] == prev_idx[1] + 1 and idx[0] == prev_idx[0])):
                raise ValueError("df index is not incremental: prev_idx{}, cur_idx{}".format(prev_idx, idx))
            prev_idx = idx


def save_data(raw_df, statics, vital, reward, intervention, outcome,
              patient_list,
              save_dir,
              save_name='unnamed.h5', save_demo=True):
    statics, vital, intervention, reward, outcome, raw_df = get_data_from_patient_list(patient_list, statics, vital,
                                                                                       intervention, reward, outcome,
                                                                                       raw_df)
    check_ID(statics, vital, intervention, reward, outcome, raw_df)

    with pd.HDFStore(os.path.join(save_dir, save_name), 'w', chunk_cache_mem_size=1024 ** 2 * 200) as store:
        store.put('raw', raw_df, format="t")
        store.put('statics', statics, format="t")
        store.put('vital', vital, format="t")
        store.put('interventions', intervention, format="t")
        store.put('reward', reward, format="t")
        store.put('outcome', outcome, format="t")
    print("data saved in " + save_dir)

    if save_demo:
        size = 5
        print("generating demo excel with size " + str(size))
        check_ID(statics, vital, intervention, reward, outcome)
        demo_statics = statics[:size]
        demo_vital, demo_interv, demo_reward, demo_outcome = matchID(demo_statics, vital, intervention, reward, outcome)

        with pd.ExcelWriter(os.path.join(save_dir, 'demo_statics.xlsx'), mode='w', engine='openpyxl') as writer:
            demo_statics.to_excel(writer)
        with pd.ExcelWriter(os.path.join(save_dir, 'demo_vitals.xlsx'), mode='w', engine='openpyxl') as writer:
            demo_vital.to_excel(writer)
        with pd.ExcelWriter(os.path.join(save_dir, 'demo_interventions.xlsx'), mode='w',
                            engine='openpyxl') as writer:
            demo_interv.to_excel(writer)
        with pd.ExcelWriter(os.path.join(save_dir, 'demo_reward.xlsx'), mode='w', engine='openpyxl') as writer:
            demo_reward.to_excel(writer)
        with pd.ExcelWriter(os.path.join(save_dir, 'demo_outcome.xlsx'), mode='w', engine='openpyxl') as writer:
            demo_outcome.to_excel(writer)
