#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=omni_finanz_zh
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --output=logs/omni_finanz_zh_%j.out
#SBATCH --error=logs/omni_finanz_zh_%j.err

set -euo pipefail
cd /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate stt4sg-transcribe
fi

python omni_context_transcribe.py \
  /mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.finanz_fabio.jsonl \
  --hf-dataset-path /home2/vincenzo/stt4sg-transcribe/swissdial_hu \
  --model-card omniASR_LLM_7B_ZS \
  --default-dialect ZH \
  --dialect-aliases ZH=zh \
  --batch-size 160 \
  --context-size 10 \
  --context-number-like \
  --output /mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.omni.finanz_fabio.zh.numctx.jsonl \
  --log-level INFO
