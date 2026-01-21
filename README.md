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
python main.py audio.mp3

# Specify language
python main.py audio.mp3 -l de

# Custom output path
python main.py audio.mp3 -o output.srt
```

### Options

```bash
# Enable speaker diarization
python main.py audio.mp3 --diarization

# Disable VAD
python main.py audio.mp3 --no-vad

# Disable CTC alignment
python main.py audio.mp3 --no-alignment

# Specify number of speakers (requires diarization)
python main.py audio.mp3 --diarization -n 2

# Use a specific Whisper model
python main.py audio.mp3 -m medium

# Use CPU instead of GPU
python main.py audio.mp3 --device cpu
```

### Full Options

```
usage: main.py [-h] [-o OUTPUT] [-m MODEL] [-l LANGUAGE] [--task {transcribe,translate}]
               [--no-vad] [--diarization] [-n NUM_SPEAKERS] [--min-speakers MIN_SPEAKERS]
               [--max-speakers MAX_SPEAKERS] [--no-alignment] [--alignment-model ALIGNMENT_MODEL]
               [--device {cuda,cpu}] [--compute-type {float16,float32,int8}]
               [--no-speaker-labels] [--no-logs] [--hf-token HF_TOKEN]
               audio_path
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

---

## Installation Methods

### 1. Use NVIDIA Docker

The official NVIDIA CUDA Docker images come with cuBLAS and cuDNN preinstalled:

```
docker pull nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

### 2. Install with pip (Linux only)

You can install the required libraries directly from PyPI:

```bash
pip install nvidia-cublas-cu12
pip install nvidia-cudnn-cu12==9.*
```

Then add them to your library path:

```bash
export LD_LIBRARY_PATH="$(python3 -c 'import importlib.util,os;pkgs=["nvidia.cublas","nvidia.cudnn"];print(":".join([os.path.join((spec.submodule_search_locations[0] if (spec:=importlib.util.find_spec(p)) and spec.submodule_search_locations else os.path.dirname(spec.origin)),"lib") for p in pkgs]))'):$LD_LIBRARY_PATH"
```

### 3. Manual Download

For Windows and Linux, precompiled CUDA/cuDNN archives are available (e.g. from NVIDIA or community distributions).
Unpack the archive and place the `.so` / `.dll` files in a directory that is part of your `PATH` (Linux) or `%PATH%` (Windows).

---

## Notes

* Ensure your CUDA and cuDNN versions match what your PyTorch wheel expects.
* If libraries are installed but still not found, verify `LD_LIBRARY_PATH` is correctly pointing to their directories.
* If using cuDNN < 9, some packages (like `ctranslate2`) may require a downgrade.


## Install the required Python packages; 
```bash
pip install -r requirements.txt
```

## Usage
In the `transcribe.py` file, update the `folder_path` variable to the path of the folder containing and pass the model name or path to the `AudioTranscriber` class. Then, run the script:
```bash
python transcribe.py
```

## Maintainer
- [@kenfus](https://github.com/kenfus)
