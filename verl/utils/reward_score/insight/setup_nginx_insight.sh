#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate verl

# Create necessary Nginx directories in home directory
mkdir -p ~/nginx/run
mkdir -p ~/nginx/logs
mkdir -p ~/nginx/cache

# Start Nginx with our configuration
nginx -c /iris/u/asap7772/verl-stable/verl/utils/reward_score/insight/nginx.conf -p ~/nginx/

echo "Load balanced vLLM servers are running!"
echo "Access the API through port 8000" 