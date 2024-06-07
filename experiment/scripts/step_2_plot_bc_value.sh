source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/" # change this to your path
conda activate ReassessDTR


python ../plot_bc_value.py \
  --plot_save_dir "/home/reub0014/projects/SimMedEnv/experiment/results/plot_bc_value" \
  --behavioural_model_path "/home/reub0014/projects/SimMedEnv/experiment/saved_models/bc_policy.pth" \
  --value_model_dir "/home/reub0014/projects/SimMedEnv/experiment/saved_models" \
  --envs "SOFA" "Outcome" "NEWS2"