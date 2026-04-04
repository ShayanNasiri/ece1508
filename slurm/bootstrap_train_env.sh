#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

STAMP_FILE="${RECIPE_MPR_QA_VENV_DIR}/.recipe_mpr_qa_train_ready"

if [ ! -x "${RECIPE_MPR_QA_VENV_DIR}/bin/python" ]; then
  python3 -m venv "${RECIPE_MPR_QA_VENV_DIR}"
fi

source "${RECIPE_MPR_QA_VENV_DIR}/bin/activate"

if [ ! -f "${STAMP_FILE}" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch
  python -m pip install --no-cache-dir \
    "transformers==4.46.3" \
    "peft==0.13.2" \
    "datasets==2.21.0" \
    "accelerate==0.34.2" \
    "requests==2.32.3"
  touch "${STAMP_FILE}"
fi

cd "${REPO_ROOT}"
