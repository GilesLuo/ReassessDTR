import wandb
import pandas as pd
import os
from tqdm import tqdm
import time

def clone_stopped_sweep(project_name, new_sweep_id, stopped_sweep_id, debug=True):
    def clone_run():
        wandb.init()
        if dict(wandb.config) in runs_config:
            idx = runs_config.index(dict(wandb.config))
            for _, row in tqdm(runs_history[idx].iterrows(), desc="log metrics"):
                wandb.log(row.to_dict())
                time.sleep(0.8)
        else:
            print(dict(wandb.config))
            print(dict(wandb.config) in runs_config)
            print("===" * 10)
            raise ValueError("Config not found in the stopped sweep")
        wandb.run.finish()

    wandb.login()
    api = wandb.Api()
    stopped_sweep = api.sweep(f"{project_name}/{stopped_sweep_id}")

    new_sweep_config = stopped_sweep.config

    # record all runs in the stopped sweep
    runs_config = []
    runs_history = []
    runs = stopped_sweep.runs
    for run in tqdm(runs, desc="fetch experiment results"):
        if run.state in ['finished']:
            # Log metrics from the old experiment
            runs_config.append(run.config)
            runs_history.append(run.history())

    if debug:
        clone_run()
    else:
        if new_sweep_id is None:
            new_sweep_id = wandb.sweep(new_sweep_config, project=project_name)

        wandb.agent(sweep_id=new_sweep_id, function=clone_run, project=project_name, entity="gilesluo")


# Example usage
clone_stopped_sweep("SepsisRL", "7rsknrnk","q5u2jjko", debug=False)
