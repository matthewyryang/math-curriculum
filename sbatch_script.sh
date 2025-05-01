#!/bin/bash
#SBATCH --job-name=generate-data
#SBATCH --partition=flame # Or your desired partition
#SBATCH --nodes=1           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node (for ray start)
#SBATCH --gres=gpu:8        # 8 GPUs per node
#SBATCH --cpus-per-task=16  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=512G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --output=slurm-ray-%j.out
#SBATCH --error=slurm-ray-%j.err  # Good practice for separate error logs
#SBATCH --qos=flame-t2_g1_qos
#SBATCH --account=aviralku

JOB_WORKING_DIR="/home/asetlur/math-curriculum"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" # Verify Slurm is parsing --gres correctly
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"


cd $JOB_WORKING_DIR

python convert_fsdp_to_hf.py /project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-q1.5sft16k-cr0.3-dualclip-bs32/global_step_300/actor/ /project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-q1.5sft16k-cr0.3-dualclip-bs32/global_step_300/actor/huggingface/ /project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-q1.5sft16k-cr0.3-dualclip-bs32/global_step_300/actor/hf-format/
#python generate_sft_data_from_openthoughts.py
