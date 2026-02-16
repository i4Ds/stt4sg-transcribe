#!/bin/sh
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name yt_podcast_diarize
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --nodelist=calc-g-006
#SBATCH --out=logs/yt_podcast_diarize%j.out
#SBATCH --error=logs/yt_podcast_diarize%j.err

python batch_transcribe.py /mnt/nas05/data02/vincenzo/podcast_data/youtube/ \
    --diarization \
    --vad-method silero \
    --vad-params '{"threshold": 0.5, "neg_threshold": 0.365}' \
    --no-logs \
    --skip-if-exist \
    --tqdm
