context_length=$1
checkpoint_number=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu nohup python /home/cmu/deepscaler/data_preprocess/pass_at_k.py --model deepscaler-1.5b-${context_length}k --checkpoint_number $checkpoint_number > /home/cmu/deepscaler/logs/${context_length}k-$checkpoint_number.log 2>&1 &