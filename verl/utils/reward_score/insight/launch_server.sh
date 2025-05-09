eval "$(conda shell.bash hook)"
conda activate verl

export CUDA_VISIBLE_DEVICES=7
python /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/api_server.py