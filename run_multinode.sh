#!/bin/bash
#SBATCH --job-name=ray-multi
#SBATCH --partition=flame # Or your desired partition
#SBATCH --nodes=2           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node (for ray start)
#SBATCH --gres=gpu:8        # 8 GPUs per node
#SBATCH --cpus-per-task=96  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=1024G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --output=slurm-ray-%j.out
#SBATCH --error=slurm-ray-%j.err  # Good practice for separate error logs
#SBATCH --qos=flame-t2_g1_qos


# --- Configuration ---
RAY_PORT=6379            # Default Ray port 6379
RAY_DASHBOARD_PORT=8265 # Default Ray dashboard port 8265
# Define the absolute path to the working directory for the job
JOB_WORKING_DIR="/home/asetlur/math-curriculum"
# Define the script to run *relative to the working directory*
# JOB_SCRIPT_NAME="$JOB_WORKING_DIR/scripts/grpo/grpo_16k.sh"
JOB_SCRIPT_NAME="$JOB_WORKING_DIR/scripts/grpo/grpo_24k.sh"
# JOB_SCRIPT_NAME="$JOB_WORKING_DIR/scripts/grpo/grpo_32k.sh"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" # Verify Slurm is parsing --gres correctly
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"

# Get the list of nodes allocated to the job
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
worker_nodes=("${nodes_array[@]:1}") # All nodes except the first

# Get the IP address of the head node.
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo $head_node_ip | awk '{print $1}') # Clean up potential extra output

# Validate IP address was obtained
if [ -z "$head_node_ip" ]; then
    echo "ERROR: Failed to obtain head node IP address."
    exit 1
fi

echo "--------------------"
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "Worker Nodes: ${worker_nodes[@]}"
echo "--------------------"

# --- Start Ray Head Node ---
echo "Starting Ray head node on $head_node..."
# Use SLURM_CPUS_PER_TASK and SLURM_GPUS_ON_NODE which Slurm should set based on --cpus-per-task and --gres
# Note: --gres=gpu:8 with --ntasks-per-node=1 should result in SLURM_GPUS_ON_NODE=8
srun --export=ALL --nodes=1 --ntasks=1 -w "$head_node" \
    /project/flame/asetlur/miniconda3/envs/verl/bin/ray start --head --node-ip-address="$head_node_ip" --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT \
    --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_ON_NODE \
    --block &
head_pid=$! # Store PID if needed for explicit kill later (optional)
echo "Ray head node PID: $head_pid"
sleep 15 # Increased sleep time for head node initialization

# --- Start Ray Worker Nodes ---
echo "Starting Ray worker nodes..."
worker_pids=()
for worker_node in "${worker_nodes[@]}"; do
    echo "Starting worker on $worker_node"
    srun --export=ALL --nodes=1 --ntasks=1 -w "$worker_node" \
        /project/flame/asetlur/miniconda3/envs/verl/bin/ray start --address="$head_node_ip:$RAY_PORT" \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_ON_NODE \
        --block &
    worker_pids+=($!) # Store worker PIDs (optional)
    # Optional: sleep briefly between starting workers if needed
    # sleep 5
done
echo "Ray worker PIDs: ${worker_pids[@]}"

# Wait a bit longer to ensure workers have connected and registered
echo "Waiting for cluster to form completely..."
sleep 20 # Increased sleep time, adjust as needed

# --- Optional: Check Cluster Status ---
echo "Checking Ray cluster status..."
# Run status check directly, as srun might still fail here if nodes are busy initializing
/project/flame/asetlur/miniconda3/envs/verl/bin/ray status || echo "WARNING: Ray status check failed or cluster not fully ready yet."
sleep 5

# --- Submit Ray Job ---
echo "Submitting Ray job: $JOB_SCRIPT_NAME from $JOB_WORKING_DIR"
# *** REMOVED srun wrapper ***
# Execute ray job submit directly from the main script context
# Ensure the JOB_SCRIPT_NAME is executable and exists within JOB_WORKING_DIR

# srun --export=ALL --nodes=1 --ntasks=1 -w "$head_node" \
#     cd $JOB_WORKING_DIR & bash $JOB_SCRIPT_NAME \
#     --block &

# job_submit_status=$(srun --export=ALL --nodes=1 --ntasks=1 -w "$head_node" \
#     /project/flame/asetlur/miniconda3/envs/verl/bin/ray job submit --address="http://127.0.0.1:8265"  --no-wait --runtime-env $JOB_WORKING_DIR/runtime_env.yaml sh -c "cd $JOB_WORKING_DIR && exec bash $JOB_SCRIPT_NAME")

/project/flame/asetlur/miniconda3/envs/verl/bin/ray job submit --address="http://$head_node_ip:$RAY_DASHBOARD_PORT" \
  --no-wait \
  --runtime-env $JOB_WORKING_DIR/runtime_env.yaml \
  -- sh -c "exec bash $JOB_SCRIPT_NAME"

# Check the exit status of job submission
# job_submit_status=$?
if [ $job_submit_status -ne 0 ]; then
    echo "ERROR: Ray job submission command failed with status $job_submit_status!"
    # Optionally, kill the ray processes if submission fails
    # echo "Attempting to stop Ray cluster due to submission failure..."
    # srun --nodes=1 --ntasks=1 -w "$head_node" ray stop || echo "Ray stop command failed"
    # kill $head_pid ${worker_pids[@]} 2>/dev/null
    # exit 1 # Exit the Slurm script
else
    echo "Ray job submission command executed successfully (exit status 0)."
    echo "Use 'ray job list' or the dashboard http://$head_node_ip:$RAY_DASHBOARD_PORT to check job status."
fi

# --- Keep Slurm Job Alive ---
echo "Ray cluster is running."
echo "Slurm job will remain active until Ray processes finish or time limit is reached."
# The `wait` command waits for background jobs started *by this script* (the ray start commands)

wait

echo "Ray processes finished or Slurm job ended."
# --- Optional Cleanup ---
# echo "Stopping Ray on head node..."
# ray stop # Can try stopping directly here too
# sleep 5

echo "Slurm script finished."
