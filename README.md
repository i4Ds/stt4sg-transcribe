# WhisperX Transcription Tool

This tool uses WhisperX to transcribe all audio and video files in a folder. Additionally, it offers a guide for installation and CUDA and CUDNN version management.


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
