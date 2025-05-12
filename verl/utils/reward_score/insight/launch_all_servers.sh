# Use with nginx
eval "$(conda shell.bash hook)"
conda activate verl

export CUDA_VISIBLE_DEVICES=7
python /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/api_server.py --port 10000 &

export CUDA_VISIBLE_DEVICES=6
python /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/api_server.py --port 10001 &

export CUDA_VISIBLE_DEVICES=5
python /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/api_server.py --port 10002 &

export CUDA_VISIBLE_DEVICES=4
python /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/api_server.py --port 10003 &

wait