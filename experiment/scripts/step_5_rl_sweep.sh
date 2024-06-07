source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/" # change this to your path
conda activate ReassessDTR

# 6311 6890 663 4242 8376
env = "NEWS2"
python /home/reub0014/projects/SimMedEnv/experiment/run_sepsis.py \
      --project SepsisRL-ICML-new \
      --env "MIMIC3Sepsis${env}Env" \
      --algo_name dqn-obs_cat\
      --linear \
      --role agent \
      --train_buffer all_train\
      --val_buffer all_val \
      --test_buffer_keyword test \
      --sweep_id 5c0rhvw7\
      --epoch 50 \
      --OPE_methods 'WIS' 'WIS_bootstrap' 'WIS_truncated' 'WIS_bootstrap_truncated' 'PatientWiseF1' "SampleWiseF1" "DR" "doseRMSE"\
      --OPE_metric 'WIS_bootstrap_truncated' \
      --goal maximize \
      --logdir "/home/reub0014/projects/SimMedEnv/experiment/train_log" \
      --value_algo offlinesarsa-rnn \
      --value_model_path "/home/reub0014/projects/SimMedEnv/experiment/saved_models/${env}_value_policy.pth" \


