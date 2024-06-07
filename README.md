# [ICML2024] Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination
<div align="center">
  <a href="https://arxiv.org/pdf/2405.18556">[paper]</a>
</div>


In the rapidly changing healthcare landscape, the implementation of offline reinforcement learning (RL) in dynamic treatment regimes (DTRs) presents a mix of unprecedented opportunities and challenges. This position paper offers a critical examination of the current status of offline RL in the context of DTRs. 

## What you should do before running the code
1. Get access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/);
2. Deploy MIMIC-III database using postgres locally. Remember the database name and password;
3. Modify the directory in each bash script (e.g., PYTHONPATH and log dir);
4. Prepare a wandb account;


## Package Installation

To replicate our experiments and results, please follow the steps below to set up the environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GilesLuo/ReassessDTR.git
   cd ReassessDTR
   ```

2. **Create a conda environment:**

   ```bash
   conda create --name ReassessDTR python=3.9
   conda activate ReassessDTR
   ```

3. **Install the required packages:**

   ```bash
   conda create --name ReassessDTR --file ./environment.yml
   ```
4. **Re-install pytorch if CUDA is not available :**
   
   You might need to reinstall pytorch if torch.cuda.is_available() returns False. This is because tianshou may overwrite your pytorch with an incompatible version. You may use any pytorch>2.0.0, as long as it is capable with your CUDA version. For example:
   ```bash
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
   ```
## Running the Code

To run the code and reproduce the experiments, please follow these steps:

**Modify directory paths:**

   Before running the code, please modify the directory paths in all files in `ReassessDTR/experiments/scripts` to match your local environment.
### 0. **Data Access**
   `ReassessDTR/DTRGym/MIMIC3SepsisEnv/mimic3table.zip` provides a template data file. You will need to replace it by the real MIMIC data.
   IMPORTANT: Before running the data generation code, please ensure that you are permitted to access MIMIC-III Clinical Database. This includes becoming a credentialed user, complete necessary training, and sign the data use agreement for the MIMIC-III project. 

   Due to data security restriction, the data will not be directly provided in this repo. However, you can access it by following step 0,1,2 and 3 in https://github.com/uribyul/py_ai_clinician, you will get a 'mimic3table.csv' file. **Please zip it and replace the 'mimic3table.zip' provided in the repo.**

   For your convenience, the data curation file has been provided. More instructions is given in `ReassessDTR/DTRGym/MIMIC3SepsisEnv/preprocess_MIMIC3Sepsis.md`.
### 1. **Preprocess the dataset:**

   ```bash
   cd ReassessDTR/experiments/scripts
   sh ./step_0_gen_data.sh
   ```
   This will generate 3 folders in `ReassessDTR/DTRGym`containing all train/val/test sets for later experiments.

### 2. **Train value estimator and behavior policy:**

   ```bash
   cd ReassessDTR/experiments/scripts
   sh ./step_1_ope_models_sweep.sh
   ```
   You will need to replace the `--algo_name` flag with the desired algorithm (e.g., `--algo_name "offlinesarsa-rnn",`). We provide 3 types of observations, i.e., single-step, multi-step stacked (RNN style) and multi-step concatenated (MLP style).
In the paper we use RNN style observations for value estimator and behavior policy; and we use multi-step concatenated obs for target policies.

### 3. **Move best value estimator and behavior policy**

   After hyperparameter sweep, please make sure you copy the best value estimator and behavior policy to
    `ReassessDTR/experiment/saved_models/`.

   After you sweep the behavior and value models for all reward settings, the following files should be in `saved_models/`:
   `NEWS2_value_policy.pth`
   `Outcome_value_policy.pth`
   `SOFA_value_policy.pth`
   `bc_policy.pth`

### 4. **Run baseline analysis and calibration**

   ```bash
   cd ReassessDTR/experiments/scripts
   sh step_3_run_calibration.sh
   sh step_4_run_baselines.sh
   ```
   **You will need to use your own wandb account for training models.**
   Feel free to modify the environment variable to run the experiments on different datasets.
### 5. **Sweep RL algorithms**

   ```bash
   cd ReassessDTR/experiments/scripts
   sh step_5_rl_sweep.sh
   ```
   The `env` variable should be one of the following: `NEWS2`, `Outcome`, `SOFA`.
   You will need to replace the `--algo_name` flag with the desired algorithm (e.g., `--algo_name "cql-obs_cat"`). All available algorithms are listed in the `ReassessDTR/experiments/run_sepsis.py` file.

## Citation

This repository accompanies our ICML 2024 accepted paper titled **"Position: Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination"**. For any questions or discussions, feel free to open an issue or contact us directly.
```
@article{luo2024reinforcement,
  title={Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination},
  author={Luo, Zhiyao and Pan, Yangchen and Watkinson, Peter and Zhu, Tingting},
  journal={arXiv preprint arXiv:2405.18556},
  year={2024}
}
```

## Acknowledgements

We invite the community to engage with our work and contribute to further advancements in the reliable development of RL-based dynamic treatment regimes.

**Code contributors:** Zhiyao Luo
**Contact:** zhiyao.luo@eng.ox.ac.uk