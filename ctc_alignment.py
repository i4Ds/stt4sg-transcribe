"""
CTC-based Forced Alignment for Word-level Timestamps.

Uses wav2vec2 models and CTC forced alignment algorithm.
Based on WhisperX alignment implementation.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


@dataclass
class AlignedWord:
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AlignedSegment:
    text: str
    start: float
    end: float
    words: List[AlignedWord]
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None
    
    def to_dict(self) -> dict:
        d = {"text": self.text, "start": self.start, "end": self.end, "words": [w.to_dict() for w in self.words]}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.avg_logprob is not None:
            d["avg_logprob"] = self.avg_logprob
        return d


@dataclass
class AlignmentResult:
    segments: List[AlignedSegment]
    word_segments: List[AlignedWord]
    
    def to_dict(self) -> dict:
        return {"segments": [s.to_dict() for s in self.segments], "word_segments": [w.to_dict() for w in self.word_segments]}


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class CharSegment:
    label: str
    start: int
    end: int
    score: float


class CTCAligner:
    """CTC-based forced alignment using wav2vec2 models."""
    
    def __init__(self, language: str = "de", device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_name: Optional[str] = None, model_dir: Optional[str] = None):
        self.language = language
        self.device = device
        self.model_name = model_name
        self.model_dir = model_dir
        self._model = None
        self._metadata = None
        
    def _load_model(self):
        if self._model is not None:
            return
        
        model_name = self.model_name
        if model_name is None:
            if self.language in DEFAULT_ALIGN_MODELS_TORCH:
                model_name = DEFAULT_ALIGN_MODELS_TORCH[self.language]
            elif self.language in DEFAULT_ALIGN_MODELS_HF:
                model_name = DEFAULT_ALIGN_MODELS_HF[self.language]
            else:
                raise ValueError(f"No default alignment model for language: {self.language}")
        
        logger.info(f"Loading alignment model: {model_name}")
        
        if model_name in torchaudio.pipelines.__all__:
            bundle = torchaudio.pipelines.__dict__[model_name]
            self._model = bundle.get_model(dl_kwargs={"model_dir": self.model_dir}).to(self.device)
            labels = bundle.get_labels()
            align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
            pipeline_type = "torchaudio"
        else:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=self.model_dir)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=self.model_dir).to(self.device)
            align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}
            pipeline_type = "huggingface"
        
        self._metadata = {"language": self.language, "dictionary": align_dictionary, "type": pipeline_type}
        logger.info(f"Alignment model loaded: {pipeline_type}")
    
    def align(self, segments: List[dict], audio: Union[str, Path, np.ndarray, torch.Tensor]) -> AlignmentResult:
        """Align transcript segments to audio using CTC forced alignment."""
        self._load_model()
        
        if isinstance(audio, (str, Path)):
            audio = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        max_duration = audio.shape[1] / SAMPLE_RATE
        model_dictionary = self._metadata["dictionary"]
        model_lang = self._metadata["language"]
        model_type = self._metadata["type"]
        
        aligned_segments: List[AlignedSegment] = []
        all_words: List[AlignedWord] = []
        
        for segment in segments:
            t1, t2 = segment["start"], segment["end"]
            text = segment["text"]
            speaker = segment.get("speaker")
            avg_logprob = segment.get("avg_logprob")
            
            if t1 >= max_duration:
                aligned_segments.append(AlignedSegment(text=text, start=t1, end=t2, words=[], speaker=speaker, avg_logprob=avg_logprob))
                continue
            
            num_leading = len(text) - len(text.lstrip())
            num_trailing = len(text) - len(text.rstrip())
            
            # Build clean character list for alignment
            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                char_ = char.lower().replace(" ", "|") if model_lang not in LANGUAGES_WITHOUT_SPACES else char.lower()
                if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                    continue
                clean_char.append(char_ if char_ in model_dictionary else '*')
                clean_cdx.append(cdx)
            
            if not clean_char:
                aligned_segments.append(AlignedSegment(text=text, start=t1, end=t2, words=[], speaker=speaker, avg_logprob=avg_logprob))
                continue
            
            text_clean = "".join(clean_char)
            tokens = [model_dictionary.get(c, -1) for c in text_clean]
            
            f1, f2 = int(t1 * SAMPLE_RATE), int(t2 * SAMPLE_RATE)
            waveform_segment = audio[:, f1:f2]
            
            if waveform_segment.shape[-1] < 400:
                lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(self.device)
                waveform_segment = torch.nn.functional.pad(waveform_segment, (0, 400 - waveform_segment.shape[-1]))
            else:
                lengths = None
            
            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions, _ = self._model(waveform_segment.to(self.device), lengths=lengths)
                else:
                    emissions = self._model(waveform_segment.to(self.device)).logits
                emissions = torch.log_softmax(emissions, dim=-1)
            
            emission = emissions[0].cpu().detach()
            blank_id = next((code for char, code in model_dictionary.items() if char in ['[pad]', '<pad>']), 0)
            
            trellis = self._get_trellis(emission, tokens, blank_id)
            path = self._backtrack(trellis, emission, tokens, blank_id)
            
            if path is None:
                aligned_segments.append(AlignedSegment(text=text, start=t1, end=t2, words=[], speaker=speaker, avg_logprob=avg_logprob))
                continue
            
            char_segments = self._merge_repeats(path, text_clean)
            ratio = (t2 - t1) / (trellis.size(0) - 1)
            
            # Assign timestamps to characters
            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in clean_cdx:
                    char_seg = char_segments[clean_cdx.index(cdx)]
                    start, end = round(char_seg.start * ratio + t1, 3), round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)
                char_segments_arr.append({"char": char, "start": start, "end": end, "score": score, "word_idx": word_idx})
                if model_lang in LANGUAGES_WITHOUT_SPACES or cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1
            
            # Group into words
            char_df = pd.DataFrame(char_segments_arr)
            words = []
            for word_idx in char_df["word_idx"].unique():
                word_chars = char_df[char_df["word_idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if not word_text:
                    continue
                word_chars_no_space = word_chars[word_chars["char"] != " "]
                word = AlignedWord(
                    word=word_text,
                    start=None if pd.isna(s := word_chars_no_space["start"].min()) else s,
                    end=None if pd.isna(e := word_chars_no_space["end"].max()) else e,
                    score=None if pd.isna(sc := word_chars_no_space["score"].mean()) else round(sc, 3)
                )
                words.append(word)
                all_words.append(word)
            
            words = self._interpolate_word_timestamps(words, t1, t2)
            aligned_segments.append(AlignedSegment(text=text, start=t1, end=t2, words=words, speaker=speaker, avg_logprob=avg_logprob))
        
        return AlignmentResult(segments=aligned_segments, word_segments=all_words)
    
    def _get_trellis(self, emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        num_frame, num_tokens = emission.size(0), len(tokens)
        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1:, 0] = float("inf")
        
        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + self._get_wildcard_emission(emission[t], tokens[1:], blank_id)
            )
        return trellis
    
    def _get_wildcard_emission(self, frame_emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
        wildcard_mask = (tokens == -1)
        regular_scores = frame_emission[tokens.clamp(min=0).long()]
        max_valid = frame_emission.clone()
        max_valid[blank_id] = float('-inf')
        return torch.where(wildcard_mask, max_valid.max(), regular_scores)
    
    def _backtrack(self, trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int) -> Optional[List[Point]]:
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = [Point(j, t, emission[t, blank_id].exp().item())]
        
        while j > 0:
            if t <= 0:
                return None
            p_stay = emission[t - 1, blank_id]
            p_change = self._get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]
            stayed, changed = trellis[t - 1, j] + p_stay, trellis[t - 1, j - 1] + p_change
            t -= 1
            if changed > stayed:
                j -= 1
            path.append(Point(j, t, (p_change if changed > stayed else p_stay).exp().item()))
        
        while t > 0:
            path.append(Point(j, t - 1, emission[t - 1, blank_id].exp().item()))
            t -= 1
        return path[::-1]
    
    def _merge_repeats(self, path: List[Point], transcript: str) -> List[CharSegment]:
        i1, i2, segments = 0, 0, []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(CharSegment(transcript[path[i1].token_index], path[i1].time_index, path[i2 - 1].time_index + 1, score))
            i1 = i2
        return segments
    
    def _interpolate_word_timestamps(self, words: List[AlignedWord], seg_start: float, seg_end: float) -> List[AlignedWord]:
        for i, word in enumerate(words):
            if word.start is None:
                prev_end = next((words[j].end for j in range(i - 1, -1, -1) if words[j].end), seg_start)
                next_start = next((words[j].start for j in range(i + 1, len(words)) if words[j].start), seg_end)
                word.start, word.end = prev_end, next_start
        return words


def save_alignment_log(result: AlignmentResult, output_path: Union[str, Path], audio_path: Optional[str] = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_data = result.to_dict()
    if audio_path:
        log_data["audio_file"] = str(audio_path)
    
    log_data["segment_confidences"] = [
        {"text": seg.text, "start": seg.start, "end": seg.end,
         "avg_alignment_score": sum(w.score for w in seg.words if w.score) / len([w for w in seg.words if w.score]) if any(w.score for w in seg.words) else None,
         "word_count": len(seg.words)}
        for seg in result.segments
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved alignment log to {output_path}")


def load_srt_segments(srt_path: Union[str, Path]) -> List[dict]:
    """
    Load segments from an SRT file for alignment.
    
    Args:
        srt_path: Path to the SRT file
        
    Returns:
        List of segment dicts with 'text', 'start', 'end'
    """
    import re
    import pysubs2
    
    subs = pysubs2.load(str(srt_path))
    segments = []
    
    for event in subs:
        text = event.text.strip()
        text = re.sub(r'^\[SPEAKER_\d+\]\s*', '', text)
        text = re.sub(r'^\[[^\]]+\]\s*', '', text)
        
        if not text:
            continue
            
        segments.append({
            "text": text,
            "start": event.start / 1000.0,
            "end": event.end / 1000.0,
        })
    
    return segments


def align_srt_to_audio(
    audio_path: Union[str, Path],
    srt_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    language: str = "de",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_name: Optional[str] = None,
) -> AlignmentResult:
    """
    Align an existing SRT file to audio using CTC forced alignment.
    
    Args:
        audio_path: Path to the audio file
        srt_path: Path to the SRT file with transcription
        output_path: Optional path to save the aligned SRT
        language: Language code for alignment model
        device: Device to use (cuda/cpu)
        model_name: Optional custom alignment model
        
    Returns:
        AlignmentResult with word-level timestamps
    """

    segments = load_srt_segments(srt_path)
    logger.info(f"Loaded {len(segments)} segments from {srt_path}")
    
    aligner = CTCAligner(language=language, device=device, model_name=model_name)
    result = aligner.align(segments, audio_path)
    
    if output_path:
        from srt_formatter import segments_to_srt
        aligned_segments = [seg.to_dict() for seg in result.segments]
        segments_to_srt(aligned_segments, output_path, include_speaker=False)
        logger.info(f"Saved aligned SRT to {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Align SRT to audio using CTC forced alignment")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("srt_path", help="Path to SRT file")
    parser.add_argument("-o", "--output", help="Output aligned SRT path")
    parser.add_argument("-l", "--language", default="de", help="Language code (default: de)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--model", help="Custom alignment model name")
    parser.add_argument("--save-log", help="Path to save alignment log JSON")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    result = align_srt_to_audio(
        args.audio_path,
        args.srt_path,
        args.output,
        args.language,
        device,
        args.model,
    )
    
    if args.save_log:
        save_alignment_log(result, args.save_log, args.audio_path)
    
    print(f"\nAligned {len(result.segments)} segments with {len(result.word_segments)} words")
    if args.output:
        print(f"Output: {args.output}")
