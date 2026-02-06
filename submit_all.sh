#!/bin/bash
set -euo pipefail

jid_srf=$(sbatch --parsable mass_transcribe_srf.sh)
jid_srfpod=$(sbatch --parsable mass_transcribe_srf_podcast.sh)
jid_yt=$(sbatch --parsable mass_transcribe_youtube_podcast.sh)

echo "Submitted:"
echo "  mass_transcribe_srf.sh: ${jid_srf}"
echo "  mass_transcribe_srf_podcast.sh: ${jid_srfpod}"
echo "  mass_transcribe_youtube_podcast.sh: ${jid_yt}"

sbatch --dependency=afterok:${jid_yt} ser_rec_podcast_youtube.sh
sbatch --dependency=afterok:${jid_srfpod} ser_rec_podcast_srf.sh

echo "Submitted SER jobs with dependency on ${jid_yt}"
