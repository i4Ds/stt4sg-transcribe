#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=omni_asr_dbg
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --output=logs/omni_asr_dbg_%j.out
#SBATCH --error=logs/omni_asr_dbg_%j.err

set -euo pipefail

cd /home2/vincenzo/stt4sg-transcribe
mkdir -p logs

# Conda bootstrap (same pattern as your other sbatch scripts)
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "${CONDA_BASE}" ]; then
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="${HOME}/miniconda3"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="${HOME}/anaconda3"
    fi
fi
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate stt4sg-transcribe
else
    echo "Unable to initialize conda. Check conda installation."
    exit 1
fi

python -V
nvidia-smi || true

MANIFEST="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.jsonl"
HF_DATASET="/home2/vincenzo/stt4sg-transcribe/swissdial_hu"

if [ ! -f "${MANIFEST}" ]; then
    echo "Manifest not found: ${MANIFEST}"
    exit 1
fi
if [ ! -d "${HF_DATASET}" ]; then
    echo "HF dataset path not found: ${HF_DATASET}"
    exit 1
fi

# Quick dependency check so failures are explicit in logs.
python - <<'PY'
import importlib
mods = ["omnilingual_asr", "datasets"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python modules: {missing}")
print("Dependency check ok")
PY

BATCH_SIZES=(8 16 32 64 96 128 160 192 224 256 320 384 448 512 640 768)
LAST_OK=0
LIMIT=0
CONTEXT=10

for BS in "${BATCH_SIZES[@]}"; do
    OUT="/tmp/manifest.omni.debug.bs${BS}.jsonl"
    rm -f "${OUT}"
    LIMIT=${BS}

    echo "============================================================"
    echo "Testing batch size: ${BS}"
    echo "============================================================"

    set +e
    python omni_context_transcribe.py "${MANIFEST}" \
      --hf-dataset-path "${HF_DATASET}" \
      --model-card omniASR_LLM_7B_ZS \
      --default-dialect ZH \
      --dialect-aliases ZH=zh \
      --batch-size "${BS}" \
      --context-size "${CONTEXT}" \
      --limit "${LIMIT}" \
      --output "${OUT}" \
      --log-level INFO
    RC=$?
    set -e

    if [ ${RC} -eq 0 ]; then
        LAST_OK=${BS}
        echo "PASS batch_size=${BS}"
    else
        echo "FAIL batch_size=${BS} (exit=${RC})"
        break
    fi

done

echo "RESULT: last successful batch size = ${LAST_OK}"
