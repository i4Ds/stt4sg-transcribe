#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=yt_manifest_combine
#SBATCH --mem=64G
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/yt_manifest_combine_%j.out
#SBATCH --error=logs/yt_manifest_combine_%j.err

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

ROOT_DIR="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed"
MANIFEST="${ROOT_DIR}/manifest.jsonl"
EMOTION="${ROOT_DIR}/manifest.emotion.jsonl"
OMNI="${ROOT_DIR}/manifest.omni.all.mixed50.ctx.zs.jsonl"
DIALECT="${ROOT_DIR}/manifest_with_speaker_dialect.jsonl"
TAGGED="${ROOT_DIR}/manifest.tagged.sed.h200.v2.fullclip.jsonl"
OUT="${ROOT_DIR}/manifest_combined.jsonl"

[ -f "${MANIFEST}" ] || { echo "Missing file: ${MANIFEST}"; exit 1; }
[ -f "${EMOTION}" ] || { echo "Missing file: ${EMOTION}"; exit 1; }
[ -f "${OMNI}" ] || { echo "Missing file: ${OMNI}"; exit 1; }
[ -f "${DIALECT}" ] || { echo "Missing file: ${DIALECT}"; exit 1; }
[ -f "${TAGGED}" ] || { echo "Missing file: ${TAGGED}"; exit 1; }

echo "== Combining manifest with emotion/omni/dialect/tag data =="

python combine_manifests.py \
    "${MANIFEST}" \
    "${EMOTION}" \
    "${OMNI}" \
    "${DIALECT}" \
    "${TAGGED}" \
    --output "${OUT}"

echo "Done. Output: ${OUT}"
