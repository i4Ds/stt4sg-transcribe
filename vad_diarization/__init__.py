"""
Voice Activity Detection and Speaker Diarization package.

This package provides functionality to:
1. Detect speech segments using various VAD backends (PyAnnote, Silero, SpeechBrain, NeMo)
2. Perform speaker diarization to identify different speakers (PyAnnote, NeMo, SpeechBrain)

Public API:
- CombinedVADDiarization: Combined VAD and Diarization pipeline
- DiarizationResult: Result of diarization containing segments and speaker info
- SpeechSegment: Represents a speech segment with optional speaker label
- save_vad_diarization_log: Save VAD/diarization results to a JSON log file

For advanced usage:
- VADProvider, DiarizationProvider: Abstract base classes
- VADFactory, DiarizationFactory: Factories for creating provider instances
- Individual providers: PyAnnoteVAD, SileroVAD, SpeechBrainVAD, NemoVAD,
                        PyAnnoteDiarization, NemoClusteringDiarization, SpeechBrainDiarization
"""

# Core data structures
from .base import (
    SpeechSegment,
    DiarizationResult,
    VADProvider,
    DiarizationProvider,
)

# Utility functions
from .utils import save_vad_diarization_log, generate_chunk_timestamps

# VAD providers and factory
from .vad import (
    PyAnnoteVAD,
    SileroVAD,
    SpeechBrainVAD,
    NemoVAD,
    VADFactory,
)

# Diarization providers and factory
from .diarization import (
    PyAnnoteDiarization,
    NemoClusteringDiarization,
    SpeechBrainDiarization,
    DiarizationFactory,
)

# Combined pipeline
from .combined import CombinedVADDiarization

__all__ = [
    # Core data structures
    "SpeechSegment",
    "DiarizationResult",
    "VADProvider",
    "DiarizationProvider",
    # Utility functions
    "save_vad_diarization_log",
    "generate_chunk_timestamps",
    # VAD
    "PyAnnoteVAD",
    "SileroVAD",
    "SpeechBrainVAD",
    "NemoVAD",
    "VADFactory",
    # Diarization
    "PyAnnoteDiarization",
    "NemoClusteringDiarization",
    "SpeechBrainDiarization",
    "DiarizationFactory",
    # Combined
    "CombinedVADDiarization",
]
