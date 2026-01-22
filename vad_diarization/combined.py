"""
Combined VAD and Diarization pipeline.

This module contains:
- CombinedVADDiarization: Combined VAD and Diarization for the full pipeline
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import DiarizationProvider, DiarizationResult, SpeechSegment, VADProvider
from .vad import VADFactory
from .diarization import DiarizationFactory

logger = logging.getLogger(__name__)


class CombinedVADDiarization:
    """Combined VAD and Diarization for the full pipeline."""
    
    def __init__(
        self,
        device: str = "cuda",
        use_auth_token: Optional[str] = None,
        vad_method: str = "pyannote",
        vad_params: Optional[Dict[str, Any]] = None,
        diarization_method: str = "pyannote",
        diarization_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize combined VAD and Diarization.
        
        Args:
            device: Device to run the models on
            use_auth_token: HuggingFace auth token
            vad_method: VAD provider name
            vad_params: VAD provider parameters
            diarization_method: Diarization provider name
            diarization_params: Diarization provider parameters
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self.vad_method = vad_method
        self.vad_params = vad_params or {}
        self.diarization_method = diarization_method
        self.diarization_params = diarization_params or {}
        self._vad_provider: Optional[VADProvider] = None
        self._diarization_provider: Optional[DiarizationProvider] = None
    
    def process(
        self,
        audio_path: Union[str, Path],
        use_vad: bool = True,
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
            use_vad: Whether to perform VAD
            use_diarization: Whether to perform speaker diarization
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            vad_min_duration: Minimum VAD segment duration
            vad_merge_threshold: Merge VAD segments closer than this
            
        Returns:
            DiarizationResult with segments (with or without speaker labels)
        """
        vad_segments: List[SpeechSegment] = []
        diar_segments: List[SpeechSegment] = []
        num_found_speakers = 0

        if use_vad:
            vad_provider = self._get_vad_provider()
            vad_segments = vad_provider.detect_speech(
                audio_path,
                min_duration=vad_min_duration,
                merge_threshold=vad_merge_threshold,
            )

        if use_diarization:
            provider = self._get_diarization_provider()
            diarization = provider.diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diar_segments = diarization.segments
            num_found_speakers = diarization.num_speakers

        if use_vad and use_diarization:
            segments = self._assign_speakers_to_segments(vad_segments, diar_segments)
            return DiarizationResult(segments=segments, num_speakers=num_found_speakers)

        if use_vad:
            return DiarizationResult(segments=vad_segments, num_speakers=0)

        return DiarizationResult(segments=diar_segments, num_speakers=num_found_speakers)

    def _get_vad_provider(self) -> VADProvider:
        """Get or create VAD provider instance."""
        if self._vad_provider is None:
            self._vad_provider = VADFactory.create(
                method=self.vad_method,
                device=self.device,
                params=self.vad_params,
                use_auth_token=self.use_auth_token,
            )
        return self._vad_provider

    def _get_diarization_provider(self) -> DiarizationProvider:
        """Get or create Diarization provider instance."""
        if self._diarization_provider is None:
            self._diarization_provider = DiarizationFactory.create(
                method=self.diarization_method,
                device=self.device,
                params=self.diarization_params,
                use_auth_token=self.use_auth_token,
            )
        return self._diarization_provider

    @staticmethod
    def _assign_speakers_to_segments(
        vad_segments: List[SpeechSegment],
        diar_segments: List[SpeechSegment],
    ) -> List[SpeechSegment]:
        """Assign speaker labels from diarization to VAD segments based on overlap."""
        if not vad_segments:
            return vad_segments

        for seg in vad_segments:
            best_speaker, best_overlap = None, 0.0
            for diar_seg in diar_segments:
                overlap = max(0.0, min(seg.end, diar_seg.end) - max(seg.start, diar_seg.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.speaker
            seg.speaker = best_speaker
        return vad_segments
