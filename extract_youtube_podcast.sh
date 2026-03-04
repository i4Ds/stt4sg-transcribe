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

export STT4SG_ORT_INTRA_OP_THREADS=1
export STT4SG_ORT_INTER_OP_THREADS=1

conda run --no-capture-output -n stt4sg-transcribe python extract_segments.py /mnt/nas05/data02/vincenzo/podcast_data/youtube \
    --output-dir /mnt/nas05/data02/vincenzo/podcast_data/youtube/processed \
    --min-purity 0.99 \
    --min-coverage 0.9 \
    --min-duration 2.0 \
    --max-duration 15.0 \
    --min-avg-logprob -0.5 \
    --frame-ms 10 \
    --cut-pad-start-ms 25 \
    --cut-pad-end-ms 200 \
    --no-summary \
    --max-pause 3 \
    --max-non-main-time 0.2 \
    --reject-clipped \
    --clip-sample-threshold 0.999 \
    --max-clip-ratio 0.002 \
    --min-dnsmos-bak 3.4 \
    --min-dnsmos-sig 3.3
