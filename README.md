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