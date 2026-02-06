#!/bin/bash
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name srf_srt_batch
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --nodelist=calc-g-006
#SBATCH --out=logs/srf_srt_batch%j.out
#SBATCH --error=logs/srf_srt_batch%j.err

ROOT="/mnt/nas05/data01/vincenzo/SRG_data_v4"

for d in "$ROOT"/*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  model="large-v2"

  case "$name" in
    "19h30"|"Couleurs locales"|"Il Quotidiano"|"telegiornale")
      model="large-v3"
      ;;
  esac

  echo "Transcribing: $name (model=$model)"
  python batch_transcribe.py "$d" \
    --model "$model" \
    --srt-only \
    --srt-in-place \
    --skip-existing \
    --vad-method silero \
    --vad-params '{"threshold": 0.5, "neg_threshold": 0.365}' \
    --no-logs
done
