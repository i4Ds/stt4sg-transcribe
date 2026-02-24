#!/bin/bash
set -euo pipefail

for script in extract_youtube_podcast.sh ser_rec_podcast_youtube.sh apr_rec_podcast_youtube.sh; do
    [ -f "${script}" ] || { echo "Missing script: ${script}" >&2; exit 1; }
done

extract_job_id="$(sbatch --parsable extract_youtube_podcast.sh)"
ser_job_id="$(
    sbatch --parsable --dependency=afterok:${extract_job_id} ser_rec_podcast_youtube.sh
)"
apr_job_id="$(
    sbatch --parsable --dependency=afterok:${extract_job_id} apr_rec_podcast_youtube.sh
)"

echo "Submitted extract job: ${extract_job_id}"
echo "Submitted SER job (after extract success): ${ser_job_id}"
echo "Submitted APR job (after extract success): ${apr_job_id}"
