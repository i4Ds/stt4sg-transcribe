#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name apr_youtube
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx3080
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --out=logs/apr_youtube_%j.out
#SBATCH --error=logs/apr_youtube_%j.err

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

ROOT_DIR="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed"

manifest="${ROOT_DIR}/manifest.jsonl"
[ -f "${manifest}" ] || { echo "Manifest not found: ${manifest}"; exit 1; }

dir="$(dirname "${manifest}")"
apr_output="${dir}/manifest.tagged.jsonl"

echo "APR: ${manifest}"
python audio_pattern_recognition.py "${manifest}" \
    --output "${apr_output}" \
    --batch-size 16 \
    --round 3 \
    --minimal-output
