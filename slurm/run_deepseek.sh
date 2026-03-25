#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=/h/439/snasiri/ece1508/results/deepseek_%j.log
#SBATCH --exclude=gpunode4,gpunode5,gpunode7

# Redirect caches to /tmp to avoid home directory quota issues
export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

# Set up a temporary venv in /tmp with all required deps
TMPVENV=/tmp/venv_$USER
python3 -m venv $TMPVENV
source $TMPVENV/bin/activate
pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -e /h/439/snasiri/ece1508/.[dev,slm] --no-deps
pip install --no-cache-dir transformers accelerate peft safetensors

cd /h/439/snasiri/ece1508/llm_evaluation

mkdir -p ../results

python mc_eval.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --backend huggingface \
    --split test
