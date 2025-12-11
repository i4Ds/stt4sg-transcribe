"""
SRT file generation using pysubs2.

Simple wrapper around pysubs2 for creating SRT files from transcription segments.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pysubs2

logger = logging.getLogger(__name__)


def segments_to_srt(
    segments: List[dict],
    output_path: Optional[Union[str, Path]] = None,
    include_speaker: bool = True,
) -> str:
    """
    Convert transcription segments to SRT format using pysubs2.
    
    Args:
        segments: List of segment dicts with 'text', 'start', 'end', optional 'speaker'
        output_path: Optional path to save SRT file
        include_speaker: Whether to include speaker labels in output
        
    Returns:
        SRT format string
    """
    subs = pysubs2.SSAFile()
    
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        
        start_ms = int(segment.get("start", 0) * 1000)
        end_ms = int(segment.get("end", 0) * 1000)
        speaker = segment.get("speaker")
        
        if include_speaker and speaker:
            text = f"[{speaker}] {text}"
        
        event = pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text)
        subs.append(event)
    
    srt_content = subs.to_string("srt")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path))
        logger.info(f"Saved SRT file to {output_path}")
    
    return srt_content


def save_transcription_log(
    segments: List[dict],
    output_path: Union[str, Path],
    audio_path: Optional[str] = None,
    language: Optional[str] = None,
):
    """Save transcription details to JSON log file."""
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_data = {
        "audio_file": str(audio_path) if audio_path else None,
        "language": language,
        "segments": [
            {
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "speaker": seg.get("speaker"),
            }
            for seg in segments
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved transcription log to {output_path}")
