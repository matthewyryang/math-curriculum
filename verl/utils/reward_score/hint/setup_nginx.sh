#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate zero

# Create necessary Nginx directories in home directory
mkdir -p ~/nginx/run
mkdir -p ~/nginx/logs
mkdir -p ~/nginx/cache

# Start Nginx with our configuration
nginx -c /home/anikait.singh/verl-stable/verl/utils/reward_score/hint/nginx.conf -p ~/nginx/

echo "Load balanced vLLM servers are running!"
echo "Access the API through port 8000" 