#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=yt_podcast_extract_segments
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --partition=top6
#SBATCH --nodes=1
#SBATCH --output=logs/yt_podcast_extract_segments_%j.out
#SBATCH --error=logs/yt_podcast_extract_segments_%j.err

python extract_segments.py /mnt/nas05/data02/vincenzo/podcast_data/youtube \
    --output-dir /mnt/nas05/data02/vincenzo/podcast_data/youtube/processed \
    --min-purity 0.99 \
    --min-coverage 0.9 \
    --min-duration 3.0 \
    --max-duration 30.0 \
    --min-avg-logprob -0.5 \
    --no-summary \
    --max-pause 3 \
    --max-non-main-time 0.2
