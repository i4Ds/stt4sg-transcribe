#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=stt4sg_350_hf_dataset
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/stt4sg_350_hf_dataset_%j.out
#SBATCH --error=logs/stt4sg_350_hf_dataset_%j.err

set -eu

ORIGINAL_ROOT="${1:-/mnt/nas05/data01/vincenzo/stt4sg_data/stt4sg-350_v2.1}"
REGENERATED_ROOT="${2:-/home2/vincenzo/stt4sg-transcribe/stt4sg-350}"
OUT_DIR="${3:-outputs/hf_stt4sg_350_${SLURM_JOB_ID:-local}}"
REPO_ID="${REPO_ID:-i4ds/stt4sg-350}"
PRIVATE_FLAG="${PRIVATE_FLAG:---private}"
UPLOAD="${UPLOAD:-1}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1GB}"
NUM_PROC="${NUM_PROC:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MAX_ROWS_PER_SPLIT="${MAX_ROWS_PER_SPLIT:-}"

mkdir -p logs

echo "ORIGINAL_ROOT=${ORIGINAL_ROOT}"
echo "REGENERATED_ROOT=${REGENERATED_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "UPLOAD=${UPLOAD}"
echo "PRIVATE_FLAG=${PRIVATE_FLAG}"
echo "MAX_SHARD_SIZE=${MAX_SHARD_SIZE}"
echo "NUM_PROC=${NUM_PROC}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

UPLOAD_ARGS=""
if [ "${UPLOAD}" = "1" ]; then
    UPLOAD_ARGS="--upload"
fi

MAX_ROWS_ARGS=""
if [ -n "${MAX_ROWS_PER_SPLIT}" ]; then
    MAX_ROWS_ARGS="--max-rows-per-split ${MAX_ROWS_PER_SPLIT}"
fi

/usr/bin/time -v conda run --no-capture-output -n stt4sg-transcribe python build_hf_stt4sg_350_dataset.py \
    --original-root "${ORIGINAL_ROOT}" \
    --regenerated-root "${REGENERATED_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    ${PRIVATE_FLAG} \
    ${UPLOAD_ARGS} \
    ${MAX_ROWS_ARGS} \
    --max-shard-size "${MAX_SHARD_SIZE}" \
    --num-proc "${NUM_PROC}" \
    --log-level "${LOG_LEVEL}"

echo "END=$(date --iso-8601=seconds)"
echo "Build complete."
echo "README: ${OUT_DIR}/README.md"
echo "Summary: ${OUT_DIR}/build_summary.json"
echo "Stats dir: ${OUT_DIR}/stats"
