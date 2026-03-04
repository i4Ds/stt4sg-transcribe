#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=omni_youtube_h200
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --output=logs/omni_youtube_h200_%j.out
#SBATCH --error=logs/omni_youtube_h200_%j.err

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

python -V
nvidia-smi || true

ROOT_DIR="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed"
MANIFEST="${ROOT_DIR}/manifest_with_speaker_dialect.jsonl"
HF_DATASET="/home2/vincenzo/stt4sg-transcribe/swissdial_hu"
OUT="${ROOT_DIR}/manifest.omni.all.mixed50.ctx.zs.jsonl"

[ -f "${MANIFEST}" ] || { echo "Manifest not found: ${MANIFEST}"; exit 1; }
[ -d "${HF_DATASET}" ] || { echo "HF dataset path not found: ${HF_DATASET}"; exit 1; }

echo "== Running Omni ASR on full manifest =="
echo "Manifest: ${MANIFEST}"
echo "HF data:   ${HF_DATASET}"
echo "Output:    ${OUT}"

python omni_context_transcribe.py "${MANIFEST}" \
    --hf-dataset-path "${HF_DATASET}" \
    --model-card omniASR_LLM_7B_ZS \
    --default-dialect ZH \
    --dialect-aliases ZH=zh \
    --batch-size 800 \
    --context-size 10 \
    --context-number-ratio 0.5 \
    --output "${OUT}" \
    --log-level INFO

echo "Done. Output: ${OUT}"
