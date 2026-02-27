#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=apr_youtube_h200_auto
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --out=logs/apr_youtube_h200_auto_%j.out
#SBATCH --error=logs/apr_youtube_h200_auto_%j.err

set -euo pipefail

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

cd /mnt/nas05/clusterdata01/home2/vincenzo/stt4sg-transcribe
mkdir -p logs

ROOT_DIR="/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed"
MANIFEST="${ROOT_DIR}/manifest.jsonl"
OUT="${ROOT_DIR}/manifest.tagged.h200.ctx1p5.auto_bs.jsonl"

[ -f "${MANIFEST}" ] || { echo "Manifest not found: ${MANIFEST}"; exit 1; }

echo "== Probing max stable APR batch size on H200 =="
BEST_BS="$(
python - <<'PY'
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
device = "cuda" if torch.cuda.is_available() else "cpu"
sr = 16000
context_seconds = 1.5
context_samples = int(round(context_seconds * sr))
batch_candidates = [1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 5120, 6144, 8192]

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
model.eval()

best = 0
for bs in batch_candidates:
    try:
        # Repeat a context-sized dummy chunk to stress the same code path.
        contexts = [np.zeros((context_samples,), dtype=np.float32) for _ in range(bs)]
        inputs = feature_extractor(contexts, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs).logits
        best = bs
        print(f"PASS bs={bs}", flush=True)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            print(f"OOM bs={bs}", flush=True)
            break
        raise
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()

print(best)
PY
)"
BEST_BS="$(echo "${BEST_BS}" | tail -n 1 | tr -d '[:space:]')"
if [ -z "${BEST_BS}" ] || [ "${BEST_BS}" = "0" ]; then
    BEST_BS=64
fi

echo "Selected batch size: ${BEST_BS}"
echo "== Running APR on full manifest =="

python audio_pattern_recognition.py "${MANIFEST}" \
    --output "${OUT}" \
    --batch-size "${BEST_BS}" \
    --frame-seconds 0.25 \
    --frame-hop 0.125 \
    --context-seconds 1.5 \
    --round 3 \
    --minimal-output

echo "Done. Output: ${OUT}"
