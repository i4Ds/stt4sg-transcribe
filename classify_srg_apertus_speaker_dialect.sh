#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_ap_speaker_dialect
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --output=logs/srg_ap_speaker_dialect_%j.out
#SBATCH --error=logs/srg_ap_speaker_dialect_%j.err

set -eu

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed}"
MANIFEST="${ROOT_DIR}/manifest.jsonl"

[ -f "${MANIFEST}" ] || { echo "Manifest not found: ${MANIFEST}"; exit 1; }

mkdir -p /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe/logs

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
echo "MANIFEST=${MANIFEST}"

cd /mnt/nas05/clusterdata01/home2/vincenzo/SwissGPC
uv run python -m src.classification_i4ds.classify_manifest_speaker_dialect \
  --manifest "${MANIFEST}" \
  --feature-source audio \
  --device auto \
  --vote-mode prob_duration

echo "END=$(date --iso-8601=seconds)"
