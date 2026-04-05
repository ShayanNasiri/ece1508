#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --output=/h/439/snasiri/ece1508/results/finetune_qwen2.5_3b_v2_%j.log
#SBATCH --exclude=gpunode4,gpunode5,gpunode7

export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

TMPVENV=/tmp/venv_$USER
rm -rf $TMPVENV
python3 -m venv $TMPVENV
source $TMPVENV/bin/activate
pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -e /h/439/snasiri/ece1508/."[slm]"

cd /h/439/snasiri/ece1508

python finetuning/finetune.py \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --num-train-epochs 2 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.1 \
    --lora-r 8 \
    --lora-alpha 16 \
    --output-dir "outputs/run_J2_qwen3b_conservative"
