#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=/h/439/snasiri/ece1508/results/smollm2_%j.log
#SBATCH --exclude=gpunode4,gpunode5,gpunode7

# Redirect caches to /tmp to avoid home directory quota issues
export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

# Set up a temporary venv in /tmp with all required deps
TMPVENV=/tmp/venv_$USER
rm -rf $TMPVENV
python3 -m venv $TMPVENV
source $TMPVENV/bin/activate
pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -e /h/439/snasiri/ece1508/.[dev]
pip install --no-cache-dir transformers accelerate peft safetensors

cd /h/439/snasiri/ece1508/llm_evaluation

mkdir -p ../results

python mc_eval.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --backend huggingface \
    --split test
