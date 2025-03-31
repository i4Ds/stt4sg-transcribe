# WhisperX Transcription Tool

This tool uses WhisperX to transcribe all audio and video files in a folder. Additionally, it offers guidelines in CUDA and CUDNN version management.

## Installation
1. CUDA and PyTorch installation: Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the correct version of CUDA and PyTorch for your system.
2. Find out your CUDA version with `nvidia-smi` and CUDNN with `python get_cudnn_version.py`.
3. With your CUDA and CUDNN version, head over to [faster-whisper | Requirements](https://github.com/SYSTRAN/faster-whisper), check for the correct version of `ctranslate2` and install it.
4. Install the required Python packages; 
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