source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/"   # change this to your path
conda activate ReassessDTR
python /home/reub0014/projects/SimMedEnv/run/run_sepsis.py \
      --algo_name discrete-imitation-rnn\
      --nonlinear \
      --train_buffer all_ope_train\
      --val_buffer all_val \
      --test_buffer_keyword all_val \
      --role sweep\
      --sweep_id oubvvsug\
      --epoch 80 \
      --OPE_methods 'PatientWiseF1' "SampleWiseF1" \
      --OPE_metric 'PatientWiseF1' \
      --goal maximize \
      --logdir "/home/reub0014/projects/ReassessDTR/experiment/runs"  # change this to your path


