#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="${1:-/mnt/nas05/data01/vincenzo/SRG_apertus}"
PROCESSED_DIR="${2:-/mnt/nas05/data01/vincenzo/SRG_apertus/processed}"

cd "${SCRIPT_DIR}"
mkdir -p logs

extract_job="$(sbatch --parsable "${SCRIPT_DIR}/extract_srg_apertus_segments.sh" "${ROOT_DIR}" "${PROCESSED_DIR}")"
ser_job="$(sbatch --parsable --dependency=afterok:${extract_job} "${SCRIPT_DIR}/ser_rec_srg_apertus.sh" "${PROCESSED_DIR}")"
apr_job="$(sbatch --parsable --dependency=afterok:${extract_job} "${SCRIPT_DIR}/apr_rec_srg_apertus_h200_auto_v2.sh" "${PROCESSED_DIR}")"
dialect_job="$(sbatch --parsable --dependency=afterok:${extract_job} "${SCRIPT_DIR}/classify_srg_apertus_speaker_dialect.sh" "${PROCESSED_DIR}")"
combine_job="$(sbatch --parsable --dependency=afterok:${ser_job}:${apr_job}:${dialect_job} "${SCRIPT_DIR}/combine_manifest_srg_apertus_cpu.sh" "${PROCESSED_DIR}")"
gemma_job="$(sbatch --parsable --dependency=afterok:${combine_job} "${SCRIPT_DIR}/transcribe_manifest_sentence_ch_gemma_srg_apertus_h200.sh" "${PROCESSED_DIR}/manifest_combined_sliding.jsonl" "${PROCESSED_DIR}/manifest_sentence_ch_gemma.jsonl")"
merge_sentence_ch_job="$(sbatch --parsable --dependency=afterok:${gemma_job} "${SCRIPT_DIR}/merge_manifest_sentence_ch_srg_apertus_cpu.sh" "${PROCESSED_DIR}" "${PROCESSED_DIR}/manifest_final_sliding.jsonl" "${PROCESSED_DIR}/manifest_sentence_ch_gemma.jsonl" "${PROCESSED_DIR}/manifest_final_sliding_with_sentence_ch.jsonl")"

echo "extract_job=${extract_job}"
echo "ser_job=${ser_job}"
echo "apr_job=${apr_job}"
echo "dialect_job=${dialect_job}"
echo "combine_job=${combine_job}"
echo "gemma_job=${gemma_job}"
echo "merge_sentence_ch_job=${merge_sentence_ch_job}"
echo "processed_dir=${PROCESSED_DIR}"
