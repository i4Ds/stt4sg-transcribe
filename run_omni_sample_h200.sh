#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=omni_sample
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --output=logs/omni_sample_%j.out
#SBATCH --error=logs/omni_sample_%j.err

set -euo pipefail
cd /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate stt4sg-transcribe
fi

MANIFEST="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.jsonl"
HF_DATASET="/home2/vincenzo/stt4sg-transcribe/swissdial_hu"
OUT="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.omni.simple.sample.jsonl"

python omni_context_transcribe.py "${MANIFEST}" \
  --hf-dataset-path "${HF_DATASET}" \
  --model-card omniASR_LLM_7B_ZS \
  --default-dialect ZH \
  --dialect-aliases ZH=zh \
  --batch-size 8 \
  --context-size 10 \
  --limit 8 \
  --output "${OUT}" \
  --log-level INFO
