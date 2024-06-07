source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/" # change this to your path
conda activate ReassessDTR

for calibrate in false true; do
  for env in SOFA Outcome NEWS2; do
    echo model calibration is: $calibrate
    if [ "$calibrate" = true ]; then
      name="Calibrated${env}"
      calibrate_arg="--use_calibrate"

    else
      name="$env"
      calibrate_arg=""
    fi

    python ../run_naive_baseline.py \
      --env "MIMIC3Sepsis${env}Env" \
      --save_dir "/home/reub0014/projects/SimMedEnv/experiment/results/naive_baseline/${name}" \
      --value_model_path "/home/reub0014/projects/SimMedEnv/experiment/saved_models/${env}_value_policy.pth" \
      --behavioural_model_path "/home/reub0014/projects/SimMedEnv/experiment/saved_models/bc_policy.pth" \
      $calibrate_arg
  done
done

wait