#!/bin/sh
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_apertus_hf_dataset
#SBATCH --mem=48G
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/srg_apertus_hf_dataset_%j.out
#SBATCH --error=logs/srg_apertus_hf_dataset_%j.err

set -eu

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus}"
OUT_DIR="${2:-outputs/hf_srg_apertus_data_${SLURM_JOB_ID:-local}}"
REPO_ID="${REPO_ID:-i4ds/SRG_apertus_data}"
PRIVATE_FLAG="${PRIVATE_FLAG:---private}"
UPLOAD="${UPLOAD:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MIN_SECONDS="${MIN_SECONDS:-5}"
MAX_SECONDS="${MAX_SECONDS:-50}"
MAX_FILES="${MAX_FILES:-}"
MAX_ROWS="${MAX_ROWS:-}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-${HF_TOKEN:-}}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "UPLOAD=${UPLOAD}"
echo "PRIVATE_FLAG=${PRIVATE_FLAG}"
echo "MIN_SECONDS=${MIN_SECONDS}"
echo "MAX_SECONDS=${MAX_SECONDS}"
echo "MAX_FILES=${MAX_FILES}"
echo "MAX_ROWS=${MAX_ROWS}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

EXTRA_ARGS=""
if [ "${UPLOAD}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --upload"
    if [ -n "${HF_TOKEN_VALUE}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --token ${HF_TOKEN_VALUE}"
    fi
fi

if [ -n "${MAX_FILES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-files ${MAX_FILES}"
fi

if [ -n "${MAX_ROWS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-rows ${MAX_ROWS}"
fi

/usr/bin/time -v conda run --no-capture-output -n stt4sg-transcribe python build_hf_srg_apertus_dataset.py \
    "${ROOT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    ${PRIVATE_FLAG} \
    --min-seconds "${MIN_SECONDS}" \
    --max-seconds "${MAX_SECONDS}" \
    ${EXTRA_ARGS} \
    --log-level "${LOG_LEVEL}"

echo "END=$(date --iso-8601=seconds)"
echo "Build complete."
echo "Rows: ${OUT_DIR}/dataset_rows.jsonl"
echo "Summary: ${OUT_DIR}/summary.json"
echo "README: ${OUT_DIR}/README.md"
