#!/bin/bash
# Test transcription with all supported VAD methods

AUDIO_FILE="/root/projects/audio-quality-eval/benchmarks/67211ecd520c24211abedff0-n.ferrari-10min/67211ecd520c24211abedff0-n.ferrari-10min.mp3"
MODEL="i4ds/crimson-music-1"
BASENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')

# List of supported VAD methods
VAD_METHODS=("pyannote" "silero" "speechbrain" "nemo")

export LD_LIBRARY_PATH="$(python3 -c 'import importlib.util,os;pkgs=["nvidia.cublas","nvidia.cudnn"];print(":".join([os.path.join((spec.submodule_search_locations[0] if (spec:=importlib.util.find_spec(p)) and spec.submodule_search_locations else os.path.dirname(spec.origin)),"lib") for p in pkgs]))'):$LD_LIBRARY_PATH"

for vad in "${VAD_METHODS[@]}"; do
    echo "=========================================="
    echo "Testing VAD method: $vad"
    echo "=========================================="
    uv run main.py "$AUDIO_FILE" --model "$MODEL" --vad-method "$vad" --no-alignment -o "outputs/srt/${BASENAME}_${vad}.srt" --log-progress
    echo ""
done

echo "All VAD tests completed."