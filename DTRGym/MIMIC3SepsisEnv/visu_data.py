import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec


def plot_action(a, plot_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(a.groupby("icustay_id")["input_4hourly"].mean(),
                a.groupby("icustay_id")["max_dose_vaso"].mean(),
                alpha=0.6, edgecolors="k", linewidth=0.5)
    plt.title("Scatter Plot of Mean vs. Std Deviation of NEWS2 Scores per Patient")
    plt.xlabel("Mean input 4hourly")
    plt.ylabel("Standard Deviation of max dose vaso")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, "dosage_mean.pdf"), dpi=800)

    plt.figure(figsize=(10, 6))
    plt.scatter(a.groupby("icustay_id")["input_4hourly"].mean(),
                a.groupby("icustay_id")["input_4hourly"].std(),
                alpha=0.6, edgecolors="k", linewidth=0.5)
    plt.title("Scatter Plot of Mean vs. Std Deviation of NEWS2 Scores per Patient")
    plt.xlabel("Mean input 4hourly")
    plt.ylabel("Standard Deviation of max dose vaso")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, "input_4hourly_mean_std.pdf"), dpi=800)

def plot_reward(r, plot_dir):
    r_sum = r.groupby("icustay_id")["r_sum"].sum()
    r_outcome = r.groupby("icustay_id")["r_outcome"].sum()
    r_surv = r_sum[r_outcome >= 0]
    r_died = r_sum[r_outcome < 0]
    plt.figure(figsize=(10, 6))
    plt.hist(r_surv, bins=20, alpha=0.5, label='survived')
    plt.hist(r_died, bins=20, alpha=0.5, label='diceased')
    plt.title("Histogram of Reward Sum")
    plt.xlabel("Reward Sum")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, "reward_sum.pdf"), dpi=800)

def plot_NEWS2(sepsis_df, target_dir):
    surv_df = sepsis_df[sepsis_df['died_in_hosp'] == 0]
    died_df = sepsis_df[sepsis_df['died_in_hosp'] == 1]

    rate_surv = surv_df.groupby('icustay_id')['NEWS2'].diff().fillna(0)
    rate_died = died_df.groupby('icustay_id')['NEWS2'].diff().fillna(0)

    surv_end_start_diff = surv_df.groupby('icustay_id')['NEWS2'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    died_end_start_diff = died_df.groupby('icustay_id')['NEWS2'].agg(lambda x: x.iloc[-1] - x.iloc[0])

    # mean and std of NEWS2
    plt.figure(figsize=(10, 6))
    plt.scatter(surv_df.groupby("icustay_id")["NEWS2"].mean(),
                surv_df.groupby("icustay_id")["NEWS2"].std(),
                alpha=0.6, edgecolors="k", linewidth=0.5, color='blue', label='survived')
    plt.scatter(died_df.groupby("icustay_id")["NEWS2"].mean(),
                died_df.groupby("icustay_id")["NEWS2"].std(),
                alpha=0.6, edgecolors="k", linewidth=0.5, color='red', label='died')
    plt.xlabel("Mean NEWS2 Score")
    plt.ylabel("Standard Deviation of NEWS2 Score")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(target_dir, "NEWS2_scatter.pdf"), dpi=800)

    # mean and std of NEWS2 end - start
    plt.figure(figsize=(10, 6))
    plt.hist(surv_end_start_diff, color='blue', alpha=0.5, label='survived')
    plt.hist(died_end_start_diff, color='red', alpha=0.5, label='died')
    plt.xlabel("End-start Difference of NEWS2 Score")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(target_dir, "NEWS2_end-start.pdf"), dpi=800)

    # mean and std of change ratio of NEWS2
    plt.figure(figsize=(10, 6))
    plt.scatter(rate_surv.groupby("icustay_id").mean(),
                rate_surv.groupby("icustay_id").std(),
                alpha=0.6, edgecolors="k", linewidth=0.5, color='blue', label='survived')
    plt.scatter(rate_died.groupby("icustay_id").mean(),
                rate_died.groupby("icustay_id").std(),
                alpha=0.6, edgecolors="k", linewidth=0.5, color='red', label='died')
    plt.xlabel("Mean of NEWS2 Score change rate")
    plt.ylabel("Standard Deviation of NEWS2 Score change rate")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(target_dir, "NEWS2_change_scatter.pdf"), dpi=800)

    # histogram of mean NEWS2
    plt.figure(figsize=(10, 6))
    plt.hist(sepsis_df.groupby("icustay_id")["NEWS2"].mean(), bins=40)
    plt.axvline(sepsis_df.groupby("icustay_id")["NEWS2"].mean().quantile(0.10), color='red', linestyle='--',
                linewidth=1, label='10% Quantile')
    plt.axvline(sepsis_df.groupby("icustay_id")["NEWS2"].mean().quantile(0.25), color='green', linestyle='--',
                linewidth=1, label='25% Quantile')
    plt.axvline(sepsis_df.groupby("icustay_id")["NEWS2"].mean().quantile(0.75), color='blue', linestyle='--',
                linewidth=1, label='75% Quantile')
    plt.axvline(sepsis_df.groupby("icustay_id")["NEWS2"].mean().quantile(0.90), color='purple', linestyle='--',
                linewidth=1, label='90% Quantile')
    plt.title("Scatter Plot of Mean vs. Std Deviation of NEWS2 Scores per Patient")
    plt.xlabel("Mean NEWS2 Score")
    plt.ylabel("Number of Patients")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(target_dir, "NEWS2_mean_hist.pdf"), dpi=800)

    # NEWS2 score vs time
    filtered_ids = surv_end_start_diff.loc[lambda x: x <= -4].index
    filtered_df = sepsis_df[sepsis_df.index.get_level_values('icustay_id').isin(filtered_ids)]

    surv_agg_df = surv_df.groupby('step')['NEWS2'].agg(['mean', 'std']).reset_index().fillna(0)
    died_agg_df = died_df.groupby('step')['NEWS2'].agg(['mean', 'std']).reset_index().fillna(0)
    agg_sub_df = filtered_df.groupby('step')['NEWS2'].agg(['mean', 'std']).reset_index().fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(surv_agg_df['step'].tolist(), surv_agg_df['mean'], label='survived', color='blue')
    plt.fill_between(surv_agg_df['step'].tolist(), (surv_agg_df['mean'] - surv_agg_df['std'] / 2).tolist(),
                     (surv_agg_df['mean'] + surv_agg_df['std'] / 2).tolist(), color='blue',
                     alpha=0.2, label='survived')
    plt.plot(died_agg_df['step'].tolist(), died_agg_df['mean'], label='died', color='green')
    plt.fill_between(died_agg_df['step'].tolist(), (died_agg_df['mean'] - died_agg_df['std'] / 2).tolist(),
                     (died_agg_df['mean'] + died_agg_df['std'] / 2).tolist(), color='green',
                     alpha=0.2, label='died')

    plt.plot(surv_agg_df['step'].tolist(), agg_sub_df['mean'], label='NEWS2_diff <= -4', color='red')
    plt.fill_between(surv_agg_df['step'].tolist(), (agg_sub_df['mean'] - agg_sub_df['std'] / 2).tolist(),
                     (agg_sub_df['mean'] + agg_sub_df['std'] / 2).tolist(),
                     color='red', alpha=0.2, label='NEWS2_diff <= -4')
    plt.title("NEWS2 Score Changing w.r.t. Time")
    plt.xticks(range(0, 19, 2))
    plt.xlabel("Step")
    plt.ylabel("NEWS2 Score")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "NEWS2_time.pdf"), dpi=800)


def segregate_data(data, q_values):
    segments = []
    for q_val in q_values:
        segment = data[data <= q_val]
        segments.append(segment)
        data = data.drop(segment.index)
    segments.append(data)
    return segments


def plot_data(ax1, ax2, data_means, data_std, q, colors, legend_labels):
    # Histogram
    ax1.hist(data_means, bins=30, stacked=True, color=colors, alpha=0.7, edgecolor='k')
    ax1.set_xlabel("Mean")
    ax1.set_ylabel("Count")

    # Scatter plot
    if ax2 is not None:
        ax2.set_xlabel("Mean")
        ax2.set_ylabel("Std")
        for i in range(len(data_means)):
            ax2.scatter(data_means[i], data_std[i],
                        c=colors[i], alpha=0.6, edgecolors="k", linewidth=0.5)

        # Creating custom legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], markersize=10,
                       markerfacecolor=colors[i])
            for i in range(len(q) + 1)]
        ax2.legend(handles=handles, title="Quantiles of data mean")


def plot_NEWS2_by_quantiles(sepsis_df, target_dir, quantiles, method="rate"):
    # Data pre-processing

    diff = sepsis_df.groupby('icustay_id')['NEWS2'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    rate = sepsis_df.groupby('icustay_id')['NEWS2'].diff()

    news2_means = sepsis_df.groupby('icustay_id')['NEWS2'].mean()
    news2_std = sepsis_df.groupby('icustay_id')['NEWS2'].std()
    rate_means = rate.groupby('icustay_id').mean()
    rate_std = rate.groupby('icustay_id').std()
    diff_means = diff.groupby('icustay_id').mean()

    colors = cm.rainbow(np.linspace(0, 1, len(quantiles) + 1))
    legend_labels = [f'<= {quantile * 100:.0f}th Quantile' for quantile in quantiles] + [f'> {quantiles[-1] * 100:.0f}th Quantile']
    if method == "rate":
        q_values = [rate_means.quantile(q) for q in quantiles]
        rate_means_segments = segregate_data(rate_means, q_values)
        rate_std_segments = [rate_std[segment.index] for segment in rate_means_segments]
        news2_means_segments = [news2_means[segment.index] for segment in rate_means_segments]
        news2_std_segments = [news2_std[segment.index] for segment in rate_means_segments]
        diff_mean_segments = [diff_means[segment.index] for segment in rate_means_segments]

    elif method == "score":
        q_values = [news2_means.quantile(q) for q in quantiles]
        news2_means_segments = segregate_data(news2_means, q_values)
        news2_std_segments = [news2_std[segment.index] for segment in news2_means_segments]
        rate_means_segments = [rate_means[segment.index] for segment in news2_means_segments]
        rate_std_segments = [rate_std[segment.index] for segment in news2_means_segments]
        diff_mean_segments = [diff_means[segment.index] for segment in news2_means_segments]
    elif method == "diff":
        q_values = [diff_means.quantile(q) for q in quantiles]
        diff_mean_segments = segregate_data(diff_means, q_values)
        news2_means_segments = [news2_means[segment.index] for segment in diff_mean_segments]
        news2_std_segments = [news2_std[segment.index] for segment in diff_mean_segments]
        rate_means_segments = [rate_means[segment.index] for segment in diff_mean_segments]
        rate_std_segments = [rate_std[segment.index] for segment in diff_mean_segments]

    else:
        raise NotImplementedError

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(4, 2)

    ax1 = fig.add_subplot(gs[0, 0])  # Top plot spanning both columns
    ax2 = fig.add_subplot(gs[0, 1])  # Top plot spanning both columns
    ax3 = fig.add_subplot(gs[1, 0])  # Middle left plot
    ax4 = fig.add_subplot(gs[1, 1])  # Middle right plot
    ax5 = fig.add_subplot(gs[2, :])  # Bottom plot spanning both columns
    plot_data(ax1, ax2, news2_means_segments, news2_std_segments, quantiles, colors, legend_labels)
    ax1.set_title("Histogram of NEWS2 Score Mean")
    ax2.set_title("Scatter of NEWS2 Score Mean vs Std")
    plot_data(ax3, ax4, rate_means_segments, rate_std_segments, quantiles, colors, legend_labels)
    ax3.set_title("Histogram of NEWS2 Score Change Rate Mean")
    ax4.set_title("Scatter of NEWS2 Score Change Rate Mean vs Std")
    plot_data(ax5, None, diff_mean_segments, None, quantiles, colors, legend_labels)
    ax5.set_title("Histogram of NEWS2 Score Difference Mean")
    plt.title(f"Data Split by {method} Quantile Mean")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, f"NEWS2_quantile_{method}.pdf"), dpi=800)
    plt.close(fig)
    return [df.index for df in news2_means_segments]


def plot_NEWS2_by_value(sepsis_df, target_dir, values, method="rate", split_by_std=True, plot=True):
    diff = sepsis_df.groupby('icustay_id')['NEWS2'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    rate = sepsis_df.groupby('icustay_id')['NEWS2'].diff()

    news2_means = sepsis_df.groupby('icustay_id')['NEWS2'].mean()
    news2_std = sepsis_df.groupby('icustay_id')['NEWS2'].std()
    rate_means = rate.groupby('icustay_id').mean()
    rate_std = rate.groupby('icustay_id').std()
    diff_means = diff.groupby('icustay_id').mean()

    colors = cm.rainbow(np.linspace(0, 1, len(values) + 1))
    legend_labels = [f'<= {v}' for v in values] + [
        f'> {values[-1]}']
    if method == "rate":
        mean_sep_df = rate_means.copy()
        std_sep_df = rate_std.copy()

    elif method == "score":
        mean_sep_df = news2_means.copy()
        std_sep_df = news2_std.copy()
    elif method == "diff":
        mean_sep_df = diff_means.copy()
        std_sep_df = None
        if split_by_std:
            raise NotImplementedError("Split by std not implemented for diff method")
    else:
        raise NotImplementedError
    segments = segregate_data(mean_sep_df, values)
    indices = [seg.index for seg in segments]
    rate_means_segments = [rate_means[idx] for idx in indices]
    rate_std_segments = [rate_std[idx] for idx in indices]
    news2_means_segments = [news2_means[idx] for idx in indices]
    news2_std_segments = [news2_std[idx] for idx in indices]
    diff_mean_segments = [diff_means[idx] for idx in indices]

    indices_dict = {}
    for i in range(len(indices)):
        index = indices[i]
        v_ = str(values[i -1]) if i > 0 else ""
        v = str(values[i]) if i < len(indices) - 1 else ""
        indices_dict[f"{method}_{v_}-{v}"] = index

    if split_by_std:   #     for each segment, split by std median
        indices_dict_ = {}
        for k,v in indices_dict.items():
            segment = std_sep_df[v]
            median = segment.median()
            low_std = segment[segment <= median]
            high_std = segment[segment > median]
            indices_dict_[f"{k}_low_std"] = low_std.index
            indices_dict_[f"{k}_high_std"] = high_std.index
        indices_dict = indices_dict_
    if plot:
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(4, 2)

        ax1 = fig.add_subplot(gs[0, 0])  # Top plot spanning both columns
        ax2 = fig.add_subplot(gs[0, 1])  # Top plot spanning both columns
        ax3 = fig.add_subplot(gs[1, 0])  # Middle left plot
        ax4 = fig.add_subplot(gs[1, 1])  # Middle right plot
        ax5 = fig.add_subplot(gs[2, :])  # Bottom plot spanning both columns
        plot_data(ax1, ax2, news2_means_segments, news2_std_segments, values, colors, legend_labels)
        ax1.set_title("Histogram of NEWS2 Score Mean")
        ax2.set_title("Scatter of NEWS2 Score Mean vs Std")
        plot_data(ax3, ax4, rate_means_segments, rate_std_segments, values, colors, legend_labels)
        ax3.set_title("Histogram of NEWS2 Score Change Rate Mean")
        ax4.set_title("Scatter of NEWS2 Score Change Rate Mean vs Std")
        plot_data(ax5, None, diff_mean_segments, None, values, colors, legend_labels)
        ax5.set_title("Histogram of NEWS2 Score Difference Mean")
        plt.title(f"Data Split by {method} Value Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"NEWS2_value_{method}.pdf"), dpi=800)
        plt.close(fig)
    return indices_dict
