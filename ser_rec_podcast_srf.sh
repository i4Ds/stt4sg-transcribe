#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name ser_emotion_recognition
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --nodelist=calc-g-006
#SBATCH --out=logs/ser_emotion_recognition%j.out
#SBATCH --error=logs/ser_emotion_recognition%j.err

set -euo pipefail

# Ensure conda environment is available
# Try to initialize conda in non-interactive shells
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

python -V

SER_CACHE_DIR="${SER_CACHE_DIR:-$HOME/.cache/huggingface}"
LOCAL_ONLY_FLAG=""
if [ "${SER_LOCAL_FILES_ONLY:-0}" = "1" ]; then
    LOCAL_ONLY_FLAG="--local-files-only"
fi

ROOT_DIR="/mnt/nas05/data02/vincenzo/podcast_data/srf/processed"

for d in "${ROOT_DIR}"/*; do
    [ -d "${d}" ] || continue
    manifest="${d}/manifest.jsonl"
    [ -f "${manifest}" ] || continue

    output="${d}/manifest.emotion.jsonl"
    if [ -f "${output}" ]; then
        echo "Skipping (exists): ${output}"
        continue
    fi

    echo "SER: ${manifest}"
    python ser.py "${manifest}" \
        --output "${output}" \
        --cache-dir "${SER_CACHE_DIR}" \
        ${LOCAL_ONLY_FLAG} \
        --batch-size 8 \
        --chunk-seconds 10 \
        --min-seconds 0.2 \
        --slim-output \
        --framewise \
        --frame-seconds 2.0 \
        --frame-hop 1.0
done
