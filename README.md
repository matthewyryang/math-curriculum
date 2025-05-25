# Setup
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl
conda install nvidia/label/cuda-12.6.0::cuda-nvcc
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/
pip install uv

# Install verl
git clone https://github.com/volcengine/verl.git
cd verl
uv pip install -e .

# Install the latest stable version of vLLM
uv pip install vllm==0.8.4

# Install flash-attn
uv pip install flash-attn --no-build-isolation

uv pip install seaborn
uv pip install tensordict==0.6.2
uv pip install liger-kernel
```

# dataset setup

```
mkdir -p data
python examples/data_preprocess/math/generate_dataset.py --local_dir "./data" --split train
python examples/data_preprocess/math/generate_dataset.py --local_dir "./data" --split test
```

# e3 uses a coupled curriculum
- First train on easy problems in the training mixture, with a max sequence length of 8192  
```
bash scripts/grpo/grpo_8k.sh
```
The above can be run by spawning a ray job on a single H100 node with 8 cards: run ```sbatch run_singlenode.sh``` 

- Then pick a checkpoint from the previous run on easy problems and train on medium and/or hard problems, with a max sequence length of 16384  
```
bash scripts/grpo/grpo_16k.sh
```
The above can be run by spawning a ray job on four H100 node with 8 cards: run ```sbatch run_multinode.sh``` 