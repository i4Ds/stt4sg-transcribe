"""
Voice Activity Detection and Speaker Diarization using PyAnnote.

This module provides functionality to:
1. Detect speech segments using PyAnnote VAD
2. Perform speaker diarization to identify different speakers
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment, Timeline

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """Represents a speech segment with optional speaker label."""
    start: float
    end: float
    speaker: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiarizationResult:
    """Result of diarization containing segments and speaker info."""
    segments: List[SpeechSegment]
    num_speakers: int
    
    def get_clip_timestamps(self) -> List[float]:
        """Return timestamps as flat list for faster-whisper clip_timestamps parameter."""
        timestamps = []
        for seg in self.segments:
            timestamps.extend([seg.start, seg.end])
        return timestamps
    
    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "num_speakers": self.num_speakers
        }


class PyAnnoteVAD:
    """Voice Activity Detection using PyAnnote."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize PyAnnote VAD.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy load the VAD pipeline."""
        if self._pipeline is None:
            logger.info("Loading PyAnnote VAD model...")
            # Use the segmentation model for VAD
            model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=self.use_auth_token
            )
            self._pipeline = VoiceActivityDetection(segmentation=model)
            self._pipeline.to(torch.device(self.device))
            
            # Set hyperparameters for VAD
            HYPER_PARAMETERS = {
                "min_duration_on": 0.0,  # minimum duration of speech
                "min_duration_off": 0.0,  # minimum duration of silence
            }
            self._pipeline.instantiate(HYPER_PARAMETERS)
            logger.info("PyAnnote VAD model loaded successfully")
        return self._pipeline
    
    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio_path: Path to audio file
            min_duration: Minimum segment duration in seconds
            merge_threshold: Merge segments closer than this threshold
            
        Returns:
            List of SpeechSegment objects
        """
        pipeline = self._load_pipeline()
        
        logger.info(f"Running VAD on {audio_path}")
        vad_result = pipeline(str(audio_path))
        
        # Convert to SpeechSegment objects
        # VAD pipeline returns Annotation with (Segment, label) tuples
        segments = []
        for item in vad_result.itertracks():
            # itertracks yields (Segment, track_id) or just Segment
            if isinstance(item, tuple):
                segment = item[0]
            else:
                segment = item
            
            if segment.duration >= min_duration:
                segments.append(SpeechSegment(
                    start=segment.start,
                    end=segment.end
                ))
        
        # Merge close segments
        if merge_threshold > 0 and len(segments) > 1:
            segments = self._merge_close_segments(segments, merge_threshold)
        
        logger.info(f"Detected {len(segments)} speech segments")
        return segments
    
    def _merge_close_segments(
        self,
        segments: List[SpeechSegment],
        threshold: float
    ) -> List[SpeechSegment]:
        """Merge segments that are closer than threshold."""
        if not segments:
            return segments
        
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.start - merged[-1].end <= threshold:
                # Merge with previous segment
                merged[-1] = SpeechSegment(
                    start=merged[-1].start,
                    end=seg.end,
                    speaker=merged[-1].speaker
                )
            else:
                merged.append(seg)
        
        return merged


class PyAnnoteDiarization:
    """Speaker Diarization using PyAnnote."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize PyAnnote Diarization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            logger.info("Loading PyAnnote diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.use_auth_token
            )
            self._pipeline.to(torch.device(self.device))
            logger.info("PyAnnote diarization pipeline loaded successfully")
        return self._pipeline
    
    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            DiarizationResult with speaker-labeled segments
        """
        pipeline = self._load_pipeline()
        
        logger.info(f"Running diarization on {audio_path}")
        
        # Build kwargs for pipeline
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        
        diarization: Annotation = pipeline(str(audio_path), **kwargs)
        
        # Convert to SpeechSegment objects with speaker labels
        segments = []
        speakers_set = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeechSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker
            ))
            speakers_set.add(speaker)
        
        # Sort by start time
        segments.sort(key=lambda x: x.start)
        
        logger.info(f"Diarization complete: {len(segments)} segments, {len(speakers_set)} speakers")
        
        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers_set)
        )


class CombinedVADDiarization:
    """Combined VAD and Diarization for the full pipeline."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize combined VAD and Diarization.
        
        Args:
            device: Device to run the models on
            use_auth_token: HuggingFace auth token
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self.vad = PyAnnoteVAD(device=device, use_auth_token=use_auth_token)
        self.diarization = PyAnnoteDiarization(device=device, use_auth_token=use_auth_token)
    
    def process(
        self,
        audio_path: Union[str, Path],
        use_diarization: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        vad_min_duration: float = 0.5,
        vad_merge_threshold: float = 0.3,
    ) -> DiarizationResult:
        """
        Process audio with VAD and optionally diarization.
        
        Args:
            audio_path: Path to audio file
            use_diarization: Whether to perform speaker diarization
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            vad_min_duration: Minimum VAD segment duration
            vad_merge_threshold: Merge VAD segments closer than this
            
        Returns:
            DiarizationResult with segments (with or without speaker labels)
        """
        if use_diarization:
            # Full diarization (includes VAD internally)
            return self.diarization.diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        else:
            # VAD only
            segments = self.vad.detect_speech(
                audio_path,
                min_duration=vad_min_duration,
                merge_threshold=vad_merge_threshold
            )
            return DiarizationResult(
                segments=segments,
                num_speakers=0  # Unknown without diarization
            )


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
