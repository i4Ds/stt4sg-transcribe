"""
Base classes and data structures for VAD and Diarization.

This module contains:
- SpeechSegment: Represents a speech segment with optional speaker label
- DiarizationResult: Result of diarization containing segments and speaker info
- VADProvider: Abstract base class for VAD providers
- DiarizationProvider: Abstract base class for Diarization providers
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

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
        """Return timestamps as flat list for faster-whisper WhisperModel.transcribe()."""
        timestamps = []
        for seg in self.segments:
            timestamps.extend([seg.start, seg.end])
        return timestamps
    
    def get_clip_timestamps_dict(self) -> List[Dict[str, float]]:
        """Return timestamps as list of dicts for faster-whisper BatchedInferencePipeline.transcribe()."""
        return [{"start": seg.start, "end": seg.end} for seg in self.segments]
    
    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "num_speakers": self.num_speakers
        }


# =============================================================================
# Abstract Base Classes
# =============================================================================

class VADProvider(ABC):
    """Abstract base class for Voice Activity Detection providers."""
    
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize VAD provider with uniform signature.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        self.device = device
        self.params = params or {}
        self.use_auth_token = use_auth_token
    
    @abstractmethod
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
        pass
    
    def _post_process_segments(
        self,
        segments: List[SpeechSegment],
        min_duration: float,
        merge_threshold: float,
        max_duration: float = 30.0,
    ) -> List[SpeechSegment]:
        """Common post-processing: merge close segments, filter by duration, and split long segments."""
        from .utils import merge_close_segments, split_long_segments
        
        if merge_threshold > 0 and len(segments) > 1:
            segments = merge_close_segments(segments, merge_threshold)
        segments = [s for s in segments if s.duration >= min_duration]
        if max_duration > 0 and len(segments) > 0:
            segments = split_long_segments(segments, max_duration)
        return segments


class DiarizationProvider(ABC):
    """Abstract base class for Speaker Diarization providers."""
    
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize Diarization provider with uniform signature.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        self.device = device
        self.params = params or {}
        self.use_auth_token = use_auth_token
    
    @abstractmethod
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
        pass
