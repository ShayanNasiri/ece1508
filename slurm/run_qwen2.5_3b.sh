#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=/h/439/snasiri/ece1508/results/qwen2.5_3b_%j.log
#SBATCH --exclude=gpunode4,gpunode5,gpunode7

export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

TMPVENV=/tmp/venv_$USER
rm -rf $TMPVENV
python3 -m venv $TMPVENV
source $TMPVENV/bin/activate
pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -e /h/439/snasiri/ece1508/.[dev]
pip install --no-cache-dir transformers accelerate peft safetensors

cd /h/439/snasiri/ece1508/llm_evaluation

python mc_eval.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --backend huggingface \
    --split test
