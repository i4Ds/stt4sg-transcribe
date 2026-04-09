#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=apr_srg_ap_h200_v2
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --out=logs/apr_srg_ap_h200_v2_%j.out
#SBATCH --error=logs/apr_srg_ap_h200_v2_%j.err

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
MANIFEST="${ROOT_DIR}/manifest.jsonl"
OUT="${ROOT_DIR}/manifest.tagged.h200.v2.sliding.jsonl"

[ -f "${MANIFEST}" ] || { echo "Manifest not found: ${MANIFEST}"; exit 1; }

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT=${OUT}"

python -u audio_pattern_recognition.py "${MANIFEST}" \
    --output "${OUT}" \
    --batch-size 1024 \
    --frame-seconds 1 \
    --frame-hop 0.5 \
    --context-seconds 2 \
    --aggregation-window-frames 1 \
    --min-prob 0.25 \
    --round 3 \
    --minimal-output

echo "END=$(date --iso-8601=seconds)"
