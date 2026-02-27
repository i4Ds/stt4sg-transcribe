#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=apr_youtube_sed_h200_v2
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --out=logs/apr_youtube_sed_h200_v2_%j.out
#SBATCH --error=logs/apr_youtube_sed_h200_v2_%j.err

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
OUT="${ROOT_DIR}/manifest.tagged.sed.h200.v2.fullclip.jsonl"

[ -f "${MANIFEST}" ] || { echo "Manifest not found: ${MANIFEST}"; exit 1; }
[ ! -f "${OUT}" ] || { echo "Refusing to overwrite existing output: ${OUT}"; exit 1; }

echo "== Running SED APR v2 on full manifest (full-clip inference) =="

python audio_pattern_recognition_sed.py "${MANIFEST}" \
    --output "${OUT}" \
    --batch-size 1024 \
    --top-k 5 \
    --min-prob 0.05 \
    --round 3 \
    --minimal-output

echo "Done. Output: ${OUT}"
