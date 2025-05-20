import numpy as np
import pysubs2
import torch
import os
import whisperx
import torchaudio
from moviepy import VideoFileClip
import json
import glob
import os
import mimetypes
from tqdm.auto import tqdm


def align_with_whisperx(list_of_dict, audio, device, language):
    model_a, metadata = whisperx.load_align_model(language_code=language, device=None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    aligned_result = whisperx.align(
        list_of_dict,
        model_a,
        metadata,
        audio.astype(np.float32),
        device,
        return_char_alignments=False,
    )
    return aligned_result


class AudioTranscriber:
    def __init__(
        self,
        transcribe_model="large-v3",
        language="de",
        align_model=None,
        device=None,
        compute_type=None,
        sr_rate=16000,
    ):
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.compute_type = (
            compute_type
            if compute_type
            else "float16" if self.device == "cuda" else "float32"
        )
        self.transcribe_model = whisperx.load_model(
            whisper_arch=transcribe_model,
            device=self.device,
            device_index=0,
            compute_type=self.compute_type,
            asr_options=
            {
                "beam_size": 5,
                "best_of": 5,
                "log_prob_threshold": -1.0,
            },
        )
        # Handle 128 mel spectrogram for v3
        if "v3" in transcribe_model:
            self.transcribe_model.model.feat_kwargs["feature_size"] = 128
        self.language = language
        self.align_model, self.metadata = (
            align_model
            if align_model
            else whisperx.load_align_model(
                device=self.device,
                language_code=self.language,
            )
        )
        self.sr_rate = sr_rate
        self.transcription_result = None

    def video_to_mp3(self, path):
        # Load the video file
        video = VideoFileClip(path)

        # Extract the audio
        audio = video.audio

        # Mp3 path
        file_path, _ = os.path.splitext(path)
        mp3_path = file_path + ".mp3"

        # Write the audio to an MP3 file
        audio.write_audiofile(mp3_path)

        return mp3_path

    def load_audio_to_numpy(self, audio_path):
        audio_array, sr = torchaudio.load(audio_path)
        if sr != self.sr_rate:
            audio_array = torchaudio.transforms.Resample(sr, self.sr_rate)(audio_array)
        return torch.mean(audio_array, dim=0).numpy()

    def transcribe(self, audio, align=True, batch_size=8):
        self.transcription_result = self.transcribe_model.transcribe(
            audio, batch_size=batch_size, language=self.language
        )

        if align:
            self.transcription_result = whisperx.align(
                self.transcription_result["segments"],
                self.align_model,
                self.metadata,
                audio,
                device=self.device,
                return_char_alignments=False,
            )

        self.pysubs2 = pysubs2.load_from_whisper(self.transcription_result["segments"])
        self.text = " ".join([segment["text"] for segment in self.transcription_result["segments"]])

    def save_subs_to_srt(self, srt_path):
        if self.transcription_result is None:
            raise Exception("Nothing to save. Did you transcribe?")

        self.pysubs2.save(srt_path, format_="srt")

    def subs_to_srt_string(self):
        if self.transcription_result is None:
            raise Exception("Nothing to save. Did you transcribe?")

        return self.pysubs2.to_string(format_="srt")
    
    def save_as_txt(self, txt_path):
        if self.transcription_result is None:
            raise Exception("Nothing to save. Did you transcribe?")
        with open(txt_path, "w") as f:
            f.write(self.text)


if __name__ == "__main__":
    # Model path can either be a repository on huggingface, a model version or a path to a local model folder
    MODEL_PATH = "whisper-large-v3-srg-v2-full-mc-de-sg-corpus"
    # Folder path is the folder where the media files are located. It will find all media files in the folder (including subfolders)
    # If it's a local model, the folder has to contain the model.bin, vocab files, config files etc. 
    FOLDER_PATH = "test"


    # ----------------------------------------------------------------
    # Initialize the transcriber
    at = AudioTranscriber(
        transcribe_model="whisper-large-v3-srg-v2-full-mc-de-sg-corpus",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ----------------------------------------------------------------
    # Process the folder
    print(f"Processing folder: {FOLDER_PATH}")
    # Find all media files in the folder (including subfolders)
    media_files = []
    for file_path in glob.iglob(os.path.join(FOLDER_PATH, "**"), recursive=True):
        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith(("audio/", "video/")):
                media_files.append(file_path)

    # Process each media file
    for path in tqdm(media_files):
        try:
            at.transcribe(path)
            srt_path = os.path.splitext(path)[0] + ".srt"
            at.save_subs_to_srt(srt_path)
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")

    print("Processing complete.")
