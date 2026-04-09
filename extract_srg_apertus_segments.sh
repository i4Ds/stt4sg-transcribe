#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_ap_extract_segments
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/srg_ap_extract_segments_%j.out
#SBATCH --error=logs/srg_ap_extract_segments_%j.err

set -eu

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus}"
OUT_DIR="${2:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed}"

cd /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe
mkdir -p logs

export STT4SG_ORT_INTRA_OP_THREADS=1
export STT4SG_ORT_INTER_OP_THREADS=1

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"

conda run --no-capture-output -n stt4sg-transcribe python extract_segments.py "${ROOT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --min-purity 0.99 \
    --min-coverage 0.9 \
    --min-duration 2.0 \
    --max-duration 15.0 \
    --min-avg-logprob -0.5 \
    --frame-ms 10 \
    --cut-pad-start-ms 25 \
    --cut-pad-end-ms 200 \
    --no-summary \
    --max-pause 3 \
    --max-non-main-time 0.2 \
    --reject-clipped \
    --clip-sample-threshold 0.999 \
    --max-clip-ratio 0.002 \
    --min-dnsmos-bak 3.4 \
    --min-dnsmos-sig 3.3

echo "END=$(date --iso-8601=seconds)"
