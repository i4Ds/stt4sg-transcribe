"""
Voice Activity Detection providers.

This module contains VAD implementations:
- PyAnnoteVAD: Using PyAnnote
- SileroVAD: Using Silero (via faster-whisper)
- SpeechBrainVAD: Using SpeechBrain
- NemoVAD: Using NeMo
- VADFactory: Factory for creating VAD provider instances
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .base import VADProvider, SpeechSegment
from .utils import get_device

logger = logging.getLogger(__name__)


# =============================================================================
# VAD Implementations
# =============================================================================

class PyAnnoteVAD(VADProvider):
    """Voice Activity Detection using PyAnnote."""
    
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize PyAnnote VAD.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters (unused, for signature consistency)
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy load the VAD pipeline."""
        if self._pipeline is None:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
            
            logger.info("Loading PyAnnote VAD model...")
            model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=self.use_auth_token
            )
            self._pipeline = VoiceActivityDetection(segmentation=model)
            self._pipeline.to(torch.device(self.device))
            
            HYPER_PARAMETERS = {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
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
        pipeline = self._load_pipeline()
        
        logger.info(f"Running VAD on {audio_path}")
        vad_result = pipeline(str(audio_path))
        
        segments = []
        for item in vad_result.itertracks():
            if isinstance(item, tuple):
                segment = item[0]
            else:
                segment = item
            segments.append(SpeechSegment(start=segment.start, end=segment.end))
        
        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info(f"Detected {len(segments)} speech segments (PyAnnote)")
        return segments


class SileroVAD(VADProvider):
    """Voice Activity Detection using Silero (via faster-whisper)."""

    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize Silero VAD.
        
        Args:
            device: Device parameter (ignored - Silero runs on CPU via faster-whisper)
            params: VAD options passed to faster-whisper VadOptions
            use_auth_token: HuggingFace auth token (unused, for signature consistency)
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        
        audio = decode_audio(str(audio_path), sampling_rate=16000)
        
        if isinstance(self.params, VadOptions):
            vad_options = self.params
        else:
            vad_options = VadOptions(**self.params) if self.params else VadOptions()

        speech_timestamps = get_speech_timestamps(audio, vad_options)
        segments = []
        for ts in speech_timestamps:
            start = float(ts["start"]) / 16000.0
            end = float(ts["end"]) / 16000.0
            segments.append(SpeechSegment(start=start, end=end))

        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info(f"Detected {len(segments)} speech segments (Silero)")
        return segments


class SpeechBrainVAD(VADProvider):
    """Voice Activity Detection using SpeechBrain."""

    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize SpeechBrain VAD.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters (vad_model, vad_savedir, thresholds, etc.)
            use_auth_token: HuggingFace auth token (unused, for signature consistency)
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        try:
            from speechbrain.inference.VAD import VAD
        except Exception as exc:
            raise RuntimeError("SpeechBrain VAD requires speechbrain and torchaudio") from exc

        audio_path = Path(audio_path)
        run_device = get_device(self.device)
        run_opts = {"device": run_device} if run_device else None

        vad_model = self.params.get("vad_model", "speechbrain/vad-crdnn-libriparty")
        vad = VAD.from_hparams(
            source=vad_model,
            savedir=self.params.get("vad_savedir", "pretrained_sb_vad"),
            run_opts=run_opts,
        )
        speech_segs = vad.get_speech_segments(
            str(audio_path),
            activation_th=self.params.get("vad_threshold", 0.5),
            deactivation_th=self.params.get("vad_deactivation_th", 0.25),
            len_th=self.params.get("min_speech_duration", 0.2),
            close_th=self.params.get("min_silence_duration", 0.2),
        )
        if len(speech_segs) == 0:
            raise RuntimeError("No speech detected by SpeechBrain VAD")

        segments = [SpeechSegment(start=float(s), end=float(e)) for (s, e) in speech_segs]
        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info(f"Detected {len(segments)} speech segments (SpeechBrain)")
        return segments


class NemoVAD(VADProvider):
    """Voice Activity Detection using NeMo (via clustering diarizer VAD)."""

    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize NeMo VAD.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters passed to NeMo diarizer
            use_auth_token: HuggingFace auth token (unused, for signature consistency)
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def detect_speech(
        self,
        audio_path: Union[str, Path],
        min_duration: float = 0.5,
        merge_threshold: float = 0.3,
    ) -> List[SpeechSegment]:
        # Import here to avoid circular dependency
        from .diarization import NemoClusteringDiarization
        
        diarizer = NemoClusteringDiarization(
            device=self.device,
            params=self.params,
            use_auth_token=self.use_auth_token,
        )
        diarization = diarizer.diarize(audio_path, num_speakers=1, min_speakers=1, max_speakers=1)
        segments = [SpeechSegment(start=s.start, end=s.end) for s in diarization.segments]

        segments = self._post_process_segments(segments, min_duration, merge_threshold)
        logger.info(f"Detected {len(segments)} speech segments (NeMo)")
        return segments


# =============================================================================
# VAD Factory
# =============================================================================

class VADFactory:
    """Factory for creating VAD provider instances."""
    
    _providers: Dict[str, type] = {
        "pyannote": PyAnnoteVAD,
        "silero": SileroVAD,
        "speechbrain": SpeechBrainVAD,
        "nemo": NemoVAD,
    }
    
    @classmethod
    def create(
        cls,
        method: str,
        device: str = "cuda",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ) -> VADProvider:
        """
        Create a VAD provider instance.
        
        Args:
            method: VAD method name ('pyannote', 'silero', 'speechbrain', 'nemo')
            device: Device to run on
            params: Provider-specific parameters
            use_auth_token: HuggingFace token
            
        Returns:
            VADProvider instance
        """
        method = method.lower()
        if method not in cls._providers:
            raise ValueError(f"Unknown VAD method: {method}. Available: {list(cls._providers.keys())}")
        
        provider_cls = cls._providers[method]
        return provider_cls(device=device, params=params, use_auth_token=use_auth_token)
    
    @classmethod
    def register(cls, name: str, provider_cls: type):
        """Register a custom VAD provider."""
        cls._providers[name.lower()] = provider_cls
