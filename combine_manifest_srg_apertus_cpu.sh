#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=srg_ap_manifest_combine
#SBATCH --mem=64G
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --output=logs/srg_ap_manifest_combine_%j.out
#SBATCH --error=logs/srg_ap_manifest_combine_%j.err

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
EMOTION="${ROOT_DIR}/manifest.emotion.jsonl"
DIALECT="${ROOT_DIR}/manifest_speaker_dialects.jsonl"
TAGGED="${ROOT_DIR}/manifest.tagged.h200.v2.sliding.jsonl"
COMBINED_OUT="${ROOT_DIR}/manifest_combined_sliding.jsonl"
FINAL_OUT="${ROOT_DIR}/manifest_final_sliding.jsonl"
FINAL_MISSING_CSV="${ROOT_DIR}/manifest_final_missing_sliding.csv"

[ -f "${MANIFEST}" ] || { echo "Missing file: ${MANIFEST}"; exit 1; }
[ -f "${EMOTION}" ] || { echo "Missing file: ${EMOTION}"; exit 1; }
[ -f "${DIALECT}" ] || { echo "Missing file: ${DIALECT}"; exit 1; }
[ -f "${TAGGED}" ] || { echo "Missing file: ${TAGGED}"; exit 1; }

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "ROOT_DIR=${ROOT_DIR}"

python combine_manifests.py \
    "${MANIFEST}" \
    "${EMOTION}" \
    "${DIALECT}" \
    "${TAGGED}" \
    --output "${COMBINED_OUT}"

python create_final_manifest.py \
    "${COMBINED_OUT}" \
    --output "${FINAL_OUT}" \
    --missing-report-csv "${FINAL_MISSING_CSV}"

echo "COMBINED_OUT=${COMBINED_OUT}"
echo "FINAL_OUT=${FINAL_OUT}"
echo "FINAL_MISSING_CSV=${FINAL_MISSING_CSV}"
echo "END=$(date --iso-8601=seconds)"
