#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_apertus_v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --nodelist=calc-g-002,calc-g-003,calc-g-004
#SBATCH --out=logs/srg_apertus_v2_%j.out
#SBATCH --error=logs/srg_apertus_v2_%j.err

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

ROOT="/mnt/nas05/data01/vincenzo/SRG_apertus"
[ -d "${ROOT}" ] || { echo "Input path not found: ${ROOT}"; exit 1; }

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Input root: ${ROOT}"
echo "Model: whisper-large-v2"

python -u batch_transcribe.py "${ROOT}" \
    --model large-v2 \
    --device cuda \
    --compute-type float16 \
    --srt-only \
    --srt-in-place \
    --skip-existing \
    --vad-method silero \
    --vad-params '{"threshold": 0.5, "neg_threshold": 0.365}' \
    --no-logs \
    --add_lock


python -u batch_transcribe.py "${ROOT}" \
    --model large-v2 \
    --device cuda \
    --compute-type float16 \
    --srt-only \
    --srt-in-place \
    --skip-existing \
    --vad-method silero \
    --vad-params '{"threshold": 0.5, "neg_threshold": 0.365}' \
    --no-logs \
    --add_lock
