#!/bin/bash
#SBATCH --job-name=generate-data
#SBATCH --partition=flame # Or your desired partition
#SBATCH --nodes=1           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node (for ray start)
#SBATCH --gres=gpu:1        # 8 GPUs per node
#SBATCH --cpus-per-task=16  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=512G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --output=slurm-ray-%j.out
#SBATCH --error=slurm-ray-%j.err  # Good practice for separate error logs
#SBATCH --qos=flame-t2_g1_qos
#SBATCH --account=aviralku

JOB_WORKING_DIR="/home/asetlur/math-curriculum"
# MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3-medhard-crh0.5/global_step_100/actor/"
# MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easy-crh0.5l0.2-ent0.002/global_step_100/actor/"
# MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3-medhard-crh0.5-minibs64/global_step_100/actor/"
MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/12klen-qwen3base-easymed-crh0.5l0.2-ent0.002/global_step_100/actor"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" # Verify Slurm is parsing --gres correctly
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"


cd $JOB_WORKING_DIR

mkdir -p $MODEL_PATH/huggingface
cp /project/flame/asetlur/hub/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/*.json $MODEL_PATH/huggingface/ 

python convert_fsdp_to_hf.py $MODEL_PATH $MODEL_PATH/huggingface $MODEL_PATH/hf-format 8
# python generate_sft_data_from_openthoughts.py
