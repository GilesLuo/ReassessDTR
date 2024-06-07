source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/"
conda activate ReassessDTR

python ../run_calibration.py \
      --project SepsisRL-ICML-new \
      --model_save_path "/home/reub0014/projects/SimMedEnv/experiment/saved_models/calibrated_model.pt"\
      --plot_save_dir "/home/reub0014/projects/SimMedEnv/experiment/results/calibration"