#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=srg_hf_dataset_full
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/srg_hf_dataset_full_%j.out
#SBATCH --error=logs/srg_hf_dataset_full_%j.err

set -eu

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_data_v4}"
OUT_DIR="${2:-outputs/hf_srg_data_v4_full_${SLURM_JOB_ID:-local}}"
REPO_ID="${REPO_ID:-i4ds/SRG_V4}"
PRIVATE_FLAG="${PRIVATE_FLAG:---private}"
UPLOAD="${UPLOAD:-1}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
# If set, this is passed explicitly as --token to the uploader.
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-${HF_TOKEN:-}}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "UPLOAD=${UPLOAD}"
echo "SKIP_PLOTS=${SKIP_PLOTS}"
echo "PRIVATE_FLAG=${PRIVATE_FLAG}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

UPLOAD_ARGS=""
if [ "${UPLOAD}" = "1" ]; then
    UPLOAD_ARGS="--upload"
    if [ -n "${HF_TOKEN_VALUE}" ]; then
        UPLOAD_ARGS="${UPLOAD_ARGS} --token ${HF_TOKEN_VALUE}"
    fi
fi

PLOT_ARGS=""
if [ "${SKIP_PLOTS}" = "1" ]; then
    PLOT_ARGS="--skip-plots"
fi

/usr/bin/time -v conda run --no-capture-output -n stt4sg-transcribe python build_hf_srg_dataset.py \
    "${ROOT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    ${PRIVATE_FLAG} \
    ${PLOT_ARGS} \
    ${UPLOAD_ARGS} \
    --log-level "${LOG_LEVEL}"

echo "END=$(date --iso-8601=seconds)"
echo "Full build complete."
echo "Rows: ${OUT_DIR}/dataset_rows.jsonl"
echo "Summary: ${OUT_DIR}/summary.json"
echo "README: ${OUT_DIR}/README.md"
