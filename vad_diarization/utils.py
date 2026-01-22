"""
Utility functions for VAD and Diarization.

This module contains:
- merge_close_segments: Merge segments that are closer than threshold- generate_chunk_timestamps: Generate fixed-length chunk timestamps for audio- get_device: Get actual device, falling back to CPU if CUDA unavailable
- parse_rttm: Parse RTTM file and return list of speech segments
- save_vad_diarization_log: Save VAD/diarization results to a JSON log file
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch

from .base import SpeechSegment, DiarizationResult

logger = logging.getLogger(__name__)


def merge_close_segments(
    segments: List[SpeechSegment],
    threshold: float
) -> List[SpeechSegment]:
    """Merge segments that are closer than threshold."""
    if not segments:
        return segments
    
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start - merged[-1].end <= threshold:
            merged[-1] = SpeechSegment(
                start=merged[-1].start,
                end=seg.end,
                speaker=merged[-1].speaker
            )
        else:
            merged.append(seg)
    
    return merged


def split_long_segments(
    segments: List[SpeechSegment],
    max_duration: float = 30.0
) -> List[SpeechSegment]:
    """
    Split segments longer than max_duration into smaller chunks.
    
    Args:
        segments: List of speech segments
        max_duration: Maximum allowed segment duration in seconds (default: 30.0)
        
    Returns:
        List of segments, each with duration <= max_duration
    """
    if not segments or max_duration <= 0:
        return segments
    
    result = []
    for seg in segments:
        if seg.duration <= max_duration:
            result.append(seg)
        else:
            # Split the segment into chunks of max_duration
            current_start = seg.start
            while current_start < seg.end:
                chunk_end = min(current_start + max_duration, seg.end)
                result.append(SpeechSegment(
                    start=current_start,
                    end=chunk_end,
                    speaker=seg.speaker
                ))
                current_start = chunk_end
    
    return result


def generate_chunk_timestamps(duration: float, chunk_length: float = 30.0) -> list[dict]:
    """
    Generate clip timestamps for fixed-length chunks.
    
    Used for batched inference without VAD - splits audio into chunk_length segments.
    
    Args:
        duration: Total audio duration in seconds
        chunk_length: Length of each chunk in seconds (default 30.0)
    
    Returns:
        List of dicts with 'start' and 'end' keys in seconds
    """
    timestamps = []
    start = 0.01 # slight offset to avoid zero start time issues of faster-whisper batched inference
    while start < duration:
        end = min(start + chunk_length, duration)
        timestamps.append({"start": start, "end": end})
        start = end
    return timestamps

def get_device(requested_device: str) -> str:
    """Get actual device, falling back to CPU if CUDA unavailable."""
    if requested_device == "cuda":
        try:
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return requested_device


def parse_rttm(rttm_path: Union[str, Path], uri: Optional[str] = None) -> List[SpeechSegment]:
    """Parse RTTM file and return list of speech segments."""
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            rec_id = parts[1]
            if uri and rec_id != uri:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append(SpeechSegment(start=start, end=start + dur, speaker=speaker))
    segments.sort(key=lambda s: s.start)
    return segments


def save_vad_diarization_log(
    result: DiarizationResult,
    output_path: Union[str, Path],
    audio_path: Optional[str] = None
):
    """
    Save VAD/diarization results to a JSON log file.
    
    Args:
        result: DiarizationResult to save
        output_path: Path to save the log file
        audio_path: Optional audio file path to include in log
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_data = result.to_dict()
    if audio_path:
        log_data["audio_file"] = str(audio_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved VAD/diarization log to {output_path}")
