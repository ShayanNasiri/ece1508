#!/bin/bash
#SBATCH --job-name=smollm2-135m-ft
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurm/common_env.sh"
bash "${REPO_ROOT}/slurm/bootstrap_train_env.sh"
source "${RECIPE_MPR_QA_VENV_DIR}/bin/activate"
source "${REPO_ROOT}/slurm/common_env.sh"

OUTPUT_ROOT="${RECIPE_MPR_QA_OUTPUT_ROOT:-}"
CMD=(python -m recipe_mpr_qa.cli train-slm --config configs/slm_smollm2_135m_finetune_original_highcap.toml)
if [ -n "${OUTPUT_ROOT}" ]; then
  CMD+=(--output-dir "${OUTPUT_ROOT}")
fi
"${CMD[@]}"
python "${REPO_ROOT}/slurm/prune_run_checkpoints.py" \
  --artifacts-root "${OUTPUT_ROOT:-artifacts/runs}" \
  --run-id "smollm2-135m-finetune-original-highcap"
