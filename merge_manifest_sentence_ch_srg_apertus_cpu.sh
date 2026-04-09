#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_ap_merge_sentence_ch
#SBATCH --mem=32G
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --output=logs/srg_ap_merge_sentence_ch_%j.out
#SBATCH --error=logs/srg_ap_merge_sentence_ch_%j.err

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

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed}"
BASE_MANIFEST="${2:-${ROOT_DIR}/manifest_final_sliding.jsonl}"
SENTENCE_CH_JSONL="${3:-${ROOT_DIR}/manifest_sentence_ch_gemma.jsonl}"
OUTPUT_JSONL="${4:-${ROOT_DIR}/manifest_final_sliding_with_sentence_ch.jsonl}"

[ -f "${BASE_MANIFEST}" ] || { echo "Missing base manifest: ${BASE_MANIFEST}"; exit 1; }
[ -f "${SENTENCE_CH_JSONL}" ] || { echo "Missing sentence_ch JSONL: ${SENTENCE_CH_JSONL}"; exit 1; }

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "BASE_MANIFEST=${BASE_MANIFEST}"
echo "SENTENCE_CH_JSONL=${SENTENCE_CH_JSONL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"

python merge_manifest_sentence_ch.py \
    "${BASE_MANIFEST}" \
    "${SENTENCE_CH_JSONL}" \
    --output "${OUTPUT_JSONL}"

echo "END=$(date --iso-8601=seconds)"
