import json

import wandb
import pandas as pd
import os
import re
import numpy as np


def get_sweep(project_path, task_name, algo_name):
    wandb.login()
    api = wandb.Api()
    sweeps = api.project(project_path, entity="gilesluo").sweeps()
    if len(sweeps) == 0:
        raise ValueError(f"No sweep found for project {project_path}")
    sweep_ids = []
    all_sweep_names = [sweep.config["name"] for sweep in sweeps]
    for sweep in sweeps:
        sweep_id = sweep.id
        if sweep.config["name"].replace(task_name, "") != algo_name:
            continue
        sweep_ids.append(sweep_id)

    if len(sweep_ids) == 0:
        raise ValueError(f"No sweep found for project {project_path} algorithm {task_name}{algo_name}, \n"
                         f"all sweeps are {all_sweep_names}")
    elif len(sweep_ids) > 1:
        raise ValueError(f"Multiple sweeps found for {algo_name}. Pls check")

    sweep_data = []
    runs = api.runs(project_path, {"sweep": sweep_ids[0]})
    for run in runs:
        # Extract the desired data from each experiment
        sweep_data.append(dict(**run.summary, **run.config,
                               **{"config_columns": list(run.config.keys())}))  # add experiment logdir to base obj
    # Create a DataFrame for the current sweep
    df = pd.DataFrame(sweep_data)
    return df


def get_sweeps(project_path, task_name, rl_algos, save_dir=None):
    for algo_name in rl_algos:
        df = get_sweep(project_path, task_name, algo_name)
        if save_dir is not None:
            df.to_csv(os.path.join(save_dir, f'{algo_name}.csv'), index=False)


def get_best_run(sweep_df, metric, maximize=True):
    if maximize:
        best_run = sweep_df.loc[sweep_df[metric].idxmax()]
    else:
        best_run = sweep_df.loc[sweep_df[metric].idxmin()]
    return best_run

