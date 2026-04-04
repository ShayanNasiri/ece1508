#!/bin/bash
#SBATCH --job-name=smollm2-360m-loglik
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurm/common_env.sh"
bash "${REPO_ROOT}/slurm/bootstrap_train_env.sh"
source "${RECIPE_MPR_QA_VENV_DIR}/bin/activate"
source "${REPO_ROOT}/slurm/common_env.sh"

CMD=(python -m recipe_mpr_qa.cli evaluate-slm --config configs/slm_smollm2_360m_baseline_loglikelihood.toml)
if [ -n "${RECIPE_MPR_QA_OUTPUT_ROOT:-}" ]; then
  CMD+=(--output-dir "${RECIPE_MPR_QA_OUTPUT_ROOT}")
fi
"${CMD[@]}"
