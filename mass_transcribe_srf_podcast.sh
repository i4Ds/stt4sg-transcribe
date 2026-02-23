#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name srf_podcast_diarize
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx3080
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --out=logs/srf_podcast_diarize%j.out
#SBATCH --error=logs/srf_podcast_diarize%j.err

python batch_transcribe.py /mnt/nas05/data02/vincenzo/podcast_data/srf/ \
    --diarization \
    --vad-method silero \
    --vad-params '{"threshold": 0.5, "neg_threshold": 0.365}' \
    --no-logs \
    --skip-if-exist \
    --tqdm \
    --add_lock

