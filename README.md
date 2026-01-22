# STT4SG Transcribe

Speech-to-Text transcription tool for Swiss German and other languages using:
- **Faster-Whisper** for fast ASR transcription
- **PyAnnote** for Voice Activity Detection (VAD) and speaker diarization
- **CTC Forced Alignment** for word-level timestamps using wav2vec2 models

## Features

- 🎤 **Speaker Diarization**: Automatically identify different speakers in audio
- 📝 **Word-level Timestamps**: Precise timing for each word via CTC alignment
- 🔊 **Voice Activity Detection**: Segment audio by speech regions
- 📊 **Detailed Logging**: Comprehensive logs for avg_logprob, alignment scores, speaker info
- 🎬 **SRT Output**: Standard subtitle format with optional speaker labels

## Installation

```bash
# Clone the repository
git clone https://github.com/i4Ds/stt4sg-transcribe
cd stt4sg-transcribe

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### PyAnnote Authentication

PyAnnote models require a HuggingFace token. Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="your-huggingface-token"
```

Or pass it via `--hf-token` argument.

## Usage

### Basic Usage

```bash
# Transcribe with default settings (VAD on, diarization off)
uv run main.py audio.mp3

# Specify language
uv run main.py audio.mp3 -l de

# Custom output path
uv run main.py audio.mp3 -o output.srt
```

### Options

```bash
# Enable speaker diarization
uv run main.py audio.mp3 --diarization

# Disable VAD
uv run main.py audio.mp3 --no-vad

# Disable CTC alignment
uv run main.py audio.mp3 --no-alignment

# Specify number of speakers (requires diarization)
uv run main.py audio.mp3 --diarization -n 2

# Use a specific Whisper model
uv run main.py audio.mp3 -m medium

# Use CPU instead of GPU
uv run main.py audio.mp3 --device cpu
```

### Full Options

```
usage: main.py [-h] [-o OUTPUT] [--output-dir OUTPUT_DIR] [-m MODEL] [-l LANGUAGE]
               [--task {transcribe,translate}] [--log-progress]
               [--no-vad] [--vad-method VAD_METHOD] [--vad-params VAD_PARAMS]
               [--diarization] [--diarization-method DIARIZATION_METHOD]
               [--diarization-params DIARIZATION_PARAMS] [-n NUM_SPEAKERS]
               [--min-speakers MIN_SPEAKERS] [--max-speakers MAX_SPEAKERS]
               [--no-alignment] [--alignment-model ALIGNMENT_MODEL]
               [--batch-size BATCH_SIZE]
               [--device {cuda,cpu}] [--compute-type {float16,float32,int8}]
               [--no-speaker-labels] [--no-logs] [--hf-token HF_TOKEN]
               audio_path

Arguments:
  audio_path                    Path to audio file
  -o, --output OUTPUT           Output SRT path (default: outputs/srt/<filename>.srt)
  --output-dir OUTPUT_DIR       Output folder for SRTs (default: outputs/srt)
  -m, --model MODEL             Whisper model (default: large-v3)
  -l, --language LANGUAGE       Language code (auto-detect if not specified)
  --task {transcribe,translate} Task to perform (default: transcribe)
  --log-progress                Log transcription progress

Diarization:
  --no-vad                      Disable VAD (on by default)
  --vad-method VAD_METHOD       VAD method: pyannote | speechbrain | nemo | silero (default: pyannote)
  --vad-params VAD_PARAMS       JSON dict of VAD parameters
  --diarization                 Enable speaker diarization (implies VAD)
  --diarization-method METHOD   Diarization method: pyannote | nemo | speechbrain (default: pyannote)
  --diarization-params PARAMS   JSON dict of diarization parameters
  -n, --num-speakers NUM        Number of speakers
  --min-speakers MIN            Minimum number of speakers (default: 2)
  --max-speakers MAX            Maximum number of speakers

Alignment:
  --no-alignment                Disable CTC alignment
  --alignment-model MODEL       Custom alignment model

Performance:
  --batch-size BATCH_SIZE       Batch size for transcription (enables batched inference and **disables** condition_on_previous_text)

Device:
  --device {cuda,cpu}           Device to use for processing
  --compute-type TYPE           Compute type: float16 | float32 | int8

Output:
  --no-speaker-labels           Disable speaker labels in SRT output
  --no-logs                     Don't save log files

Auth:
  --hf-token HF_TOKEN           HuggingFace token (or set HF_TOKEN env)
```

## Output Files

### SRT Output
- Default: `outputs/srt/<filename>.srt`
- Custom path via `-o` argument

### Log Files
All logs are saved in `logs/timestamps/` with timestamps:
- `*_vad_diarization.json`: Speech segments and speaker info
- `*_transcription.json`: Transcription with avg_logprob, no_speech_prob
- `*_alignment.json`: Word-level timestamps with alignment confidence scores
- `*_speaker_alignment.json`: Speaker assignments per segment

## Pipeline Steps

1. **VAD/Diarization** (PyAnnote): Detect speech segments (diarization only if enabled)
2. **Transcription** (Faster-Whisper): Transcribe using speech segment timestamps
3. **CTC Alignment** (wav2vec2): Generate word-level timestamps
4. **Speaker Assignment**: Map speakers to transcript segments
5. **SRT Generation**: Create subtitle file with speaker labels

## Programmatic Usage

```python
from pipeline import TranscriptionPipeline, TranscriptionConfig

config = TranscriptionConfig(
    whisper_model="large-v3",
    language="de",
    use_vad=True,
    use_diarization=False,
    use_alignment=True,
)

pipeline = TranscriptionPipeline(config)
result = pipeline.transcribe("audio.mp3")

print(result["srt_content"])
```

---


# CUDA / cuDNN Setup Guide

When running PyTorch, Faster-Whisper, or similar GPU libraries, you may encounter errors such as:

```
Unable to load libcudnn_cnn.so.9*
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
```

This usually means either:

* cuDNN 9.x is not installed, or
* The libraries are installed but not visible in your `LD_LIBRARY_PATH`. Add them to your library path:

```bash
export LD_LIBRARY_PATH="$(python3 -c 'import importlib.util,os;pkgs=["nvidia.cublas","nvidia.cudnn"];print(":".join([os.path.join((spec.submodule_search_locations[0] if (spec:=importlib.util.find_spec(p)) and spec.submodule_search_locations else os.path.dirname(spec.origin)),"lib") for p in pkgs]))'):$LD_LIBRARY_PATH"
```
or if you use fish:

````fish
set -x LD_LIBRARY_PATH (python3 -c 'import importlib.util,os;pkgs=["nvidia.cublas","nvidia.cudnn"];print(":".join([os.path.join((spec.submodule_search_locations[0] if (spec:=importlib.util.find_spec(p)) and spec.submodule_search_locations else os.path.dirname(spec.origin)),"lib") for p in pkgs]))') $LD_LIBRARY_PATH
````

## Maintainer
- [@kenfus](https://github.com/kenfus)
