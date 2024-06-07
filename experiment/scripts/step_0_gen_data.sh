source ~/anaconda3/etc/profile.d/conda.sh
conda activate ReassessDTR
export PYTHONPATH="/home/reub0014/projects/ReassessDTR/"  # change this to your path
# feel free to disable & to run sequentially
python ../../DTRGym/MIMIC3SepsisEnv/run_preprocess.py --reward_option Outcome &
python ../../DTRGym/MIMIC3SepsisEnv/run_preprocess.py --reward_option SOFA &
python ../../DTRGym/MIMIC3SepsisEnv/run_preprocess.py --reward_option NEWS2