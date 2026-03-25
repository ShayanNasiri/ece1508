#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=/h/439/snasiri/ece1508/results/smollm2_finetuned_%j.log

# Redirect caches to /tmp to avoid home directory quota issues
export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

# Set up a temporary venv in /tmp with all required deps
TMPVENV=/tmp/venv_$USER
python3 -m venv $TMPVENV
source $TMPVENV/bin/activate
pip install --no-cache-dir -e /h/439/snasiri/ece1508/.[dev,slm]

cd /h/439/snasiri/ece1508/llm_evaluation

mkdir -p ../results

python mc_eval.py \
    --model /h/439/snasiri/ece1508/outputs/smollm2_recipe_mpr/final \
    --backend huggingface \
    --split test
