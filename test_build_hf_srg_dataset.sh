#!/bin/sh
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=srg_hf_dataset_test
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --output=logs/srg_hf_dataset_test_%j.out
#SBATCH --error=logs/srg_hf_dataset_test_%j.err

set -eu

ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_data_v4}"
OUT_DIR="${2:-outputs/hf_srg_data_v4_test_${SLURM_JOB_ID:-local}}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"

# Optional thread caps for dependencies that honor OMP/MKL.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

/usr/bin/time -v conda run --no-capture-output -n stt4sg-transcribe python build_hf_srg_dataset.py \
    "${ROOT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --log-level INFO

echo "END=$(date --iso-8601=seconds)"
echo "Smoke test complete. Inspect summary: ${OUT_DIR}/summary.json"
