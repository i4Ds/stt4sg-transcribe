#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=srg_ap_sentence_ch_gemma4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --output=logs/srg_ap_sentence_ch_gemma4_%j.out
#SBATCH --error=logs/srg_ap_sentence_ch_gemma4_%j.err

set -euo pipefail

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "${CONDA_BASE}" ]; then
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="${HOME}/miniconda3"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="${HOME}/anaconda3"
    fi
fi
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate stt4sg-transcribe
else
    echo "Unable to initialize conda. Check conda installation."
    exit 1
fi

cd /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe
mkdir -p logs

INPUT_MANIFEST="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed/manifest_combined_sliding.jsonl}"
OUTPUT_JSONL="${2:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed/manifest_sentence_ch_gemma.jsonl}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "INPUT_MANIFEST=${INPUT_MANIFEST}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "MODEL_ID=${MODEL_ID}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "DTYPE=${DTYPE}"
echo "ATTN_IMPL=${ATTN_IMPL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

[ -f "${INPUT_MANIFEST}" ] || { echo "Missing input manifest: ${INPUT_MANIFEST}"; exit 1; }

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

/usr/bin/time -v python -u transcribe_manifest_sentence_ch_gemma.py \
    --input "${INPUT_MANIFEST}" \
    --output "${OUTPUT_JSONL}" \
    --model-id "${MODEL_ID}" \
    --dtype "${DTYPE}" \
    --attn-implementation "${ATTN_IMPL}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --log-level INFO

echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
