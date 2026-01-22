"""
Speaker Diarization providers.

This module contains Diarization implementations:
- PyAnnoteDiarization: Using PyAnnote
- NemoClusteringDiarization: Using NeMo ClusteringDiarizer
- SpeechBrainDiarization: Using SpeechBrain
- DiarizationFactory: Factory for creating Diarization provider instances
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .base import DiarizationProvider, DiarizationResult, SpeechSegment
from .utils import get_device, parse_rttm

logger = logging.getLogger(__name__)


# =============================================================================
# Diarization Implementations
# =============================================================================

class PyAnnoteDiarization(DiarizationProvider):
    """Speaker Diarization using PyAnnote."""
    
    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize PyAnnote Diarization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters. Supports:
                - pipeline_name: PyAnnote pipeline name (default: "pyannote/speaker-diarization-3.1")
                - pipeline_params: Additional parameters passed to the pipeline
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            
            pipeline_name = self.params.get("pipeline_name", "pyannote/speaker-diarization-3.1")
            
            logger.info("Loading PyAnnote diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                pipeline_name,
                token=self.use_auth_token
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
        
        # Add pipeline_params from self.params
        pipeline_params = self.params.get("pipeline_params")
        if pipeline_params:
            kwargs.update(pipeline_params)
        
        diarization_output = pipeline(str(audio_path), **kwargs)
        diarization = diarization_output.speaker_diarization
        
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


class NemoClusteringDiarization(DiarizationProvider):
    """Offline diarization using NVIDIA NeMo's ClusteringDiarizer."""

    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize NeMo Clustering Diarization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters (workdir, vad_model, speaker_model, etc.)
            use_auth_token: HuggingFace auth token (unused, for signature consistency)
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        try:
            import torch
            from omegaconf import OmegaConf
            from nemo.collections.asr.models import ClusteringDiarizer
        except Exception as exc:
            raise RuntimeError("NeMo diarization requires nemo_toolkit[asr] and omegaconf") from exc

        audio_path = Path(audio_path)
        workdir = Path(self.params.get("workdir", audio_path.parent / "nemo_out"))
        workdir.mkdir(parents=True, exist_ok=True)

        manifest_path = workdir / "manifest.jsonl"
        entry = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        min_spk = min_speakers if min_speakers is not None else int(self.params.get("min_speakers", 1))
        max_spk = max_speakers if max_speakers is not None else int(self.params.get("max_speakers", 10))

        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "batch_size": 1,
                "num_workers": 0,
                "verbose": False,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(workdir / "pred_rttms"),
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": self.params.get("vad_model", "vad_marblenet"),
                        "parameters": self.params.get(
                            "vad_params",
                            {
                                "onset": 0.8,
                                "offset": 0.6,
                                "pad_offset": -0.05,
                                "window_length_in_sec": 0.15,
                                "shift_length_in_sec": 0.01,
                                "smoothing": False,
                                "overlap": 0.875,
                            },
                        ),
                    },
                    "speaker_embeddings": {
                        "model_path": self.params.get("speaker_model", "titanet_large"),
                        "parameters": self.params.get(
                            "speaker_params",
                            {
                                "save_embeddings": False,
                                "window_length_in_sec": [1.5, 1.0, 0.5, 0.25],
                                "shift_length_in_sec": [0.75, 0.5, 0.25, 0.125],
                                "multiscale_weights": [1.0, 1.0, 1.0, 1.2],
                            },
                        ),
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": num_speakers is not None,
                            "max_num_speakers": max_spk,
                            "min_num_speakers": min_spk,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 60,
                            "embeddings_per_chunk": 10000,
                            "chunk_cluster_count": 80,
                        }
                    },
                },
                "device": self.device,
            }
        )
        cfg_overrides = self.params.get("cfg_overrides")
        if cfg_overrides:
            cfg = OmegaConf.merge(cfg, cfg_overrides)

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        rttm_dir = Path(cfg.diarizer.out_dir)
        rttm_files = list(rttm_dir.rglob("*.rttm"))
        if not rttm_files:
            raise RuntimeError(f"No RTTM files found in {rttm_dir}")
        target_stem = audio_path.stem
        rttm_path = next((p for p in rttm_files if p.stem == target_stem), rttm_files[0])
        segments = parse_rttm(rttm_path, uri=target_stem)
        speakers_set = {s.speaker for s in segments if s.speaker}
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))


class SpeechBrainDiarization(DiarizationProvider):
    """SpeechBrain diarization baseline: VAD -> embeddings -> spectral clustering."""

    def __init__(
        self,
        device: str = "cpu",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize SpeechBrain Diarization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            params: Provider-specific parameters (vad_model, speaker_model, clustering_method, etc.)
            use_auth_token: HuggingFace auth token (unused, for signature consistency)
        """
        super().__init__(device=device, params=params, use_auth_token=use_auth_token)

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        try:
            import torchaudio  # noqa: F401 - required for dependency check
            from speechbrain.inference.VAD import VAD
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.processing import diarization as diar
        except Exception as exc:
            raise RuntimeError("SpeechBrain diarization requires speechbrain, torchaudio, numpy") from exc

        audio_path = Path(audio_path)
        run_device = get_device(self.device)
        run_opts = {"device": run_device} if run_device else None

        # Load audio
        wav, sr = self._load_mono_16k(audio_path)

        # Run VAD
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

        # Load speaker encoder
        spk_model = self.params.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")
        spk = EncoderClassifier.from_hparams(
            source=spk_model,
            savedir=self.params.get("speaker_savedir", "pretrained_sb_ecapa"),
            run_opts=run_opts,
        )

        # Extract embeddings
        win = float(self.params.get("win", 1.5))
        hop = float(self.params.get("hop", 0.75))
        min_spk = min_speakers if min_speakers is not None else int(self.params.get("min_spk", 2))
        max_spk = max_speakers if max_speakers is not None else int(self.params.get("max_spk", 10))
        pval = float(self.params.get("pval", 0.30))

        embs, seg_meta = self._extract_embeddings(wav, sr, speech_segs, spk, win, hop)

        if len(embs) < max(2, min_spk):
            raise RuntimeError(f"Too few segments ({len(embs)}) to cluster")

        # Cluster embeddings
        X = np.stack(embs, axis=0)
        clustering_method = self.params.get("clustering_method", "Spec_Clust_unorm")
        clustering_cls = getattr(diar, clustering_method, None)
        if clustering_cls is None:
            raise RuntimeError(f"Unknown SpeechBrain clustering method: {clustering_method}")
        clust = clustering_cls(min_num_spkrs=min_spk, max_num_spkrs=max_spk)
        oracle_k = num_speakers if num_speakers is not None else min(max_spk, max(min_spk, 2))
        clust.do_spec_clust(X, k_oracle=oracle_k, p_val=pval)
        labels = clust.labels_.astype(int)

        segments = [
            SpeechSegment(start=start, end=end, speaker=f"spk{lab}")
            for (start, end), lab in zip(seg_meta, labels)
        ]
        speakers_set = {s.speaker for s in segments if s.speaker}
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))

    @staticmethod
    def _load_mono_16k(path: Path):
        """Load audio as mono 16kHz."""
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        return wav, sr

    def _extract_embeddings(self, wav, sr, speech_segs, spk_encoder, win, hop):
        """Extract speaker embeddings from speech segments."""
        embs = []
        seg_meta = []
        wav_np = wav.squeeze(0).numpy()
        
        for (s, e) in speech_segs:
            s = float(s)
            e = float(e)
            t = s
            while t + win <= e + 1e-6:
                s2 = t
                e2 = t + win
                a = int(round(s2 * sr))
                b = int(round(e2 * sr))
                chunk = torch.from_numpy(wav_np[a:b].astype(np.float32))[None, :]
                with torch.no_grad():
                    emb = spk_encoder.encode_batch(chunk).squeeze(0).squeeze(0).cpu().numpy()
                embs.append(emb)
                seg_meta.append((s2, e2))
                t += hop
        
        return embs, seg_meta


# =============================================================================
# Diarization Factory
# =============================================================================

class DiarizationFactory:
    """Factory for creating Diarization provider instances."""
    
    _providers: Dict[str, type] = {
        "pyannote": PyAnnoteDiarization,
        "nemo": NemoClusteringDiarization,
        "speechbrain": SpeechBrainDiarization,
    }
    
    @classmethod
    def create(
        cls,
        method: str,
        device: str = "cuda",
        params: Optional[Dict[str, Any]] = None,
        use_auth_token: Optional[str] = None,
    ) -> DiarizationProvider:
        """
        Create a Diarization provider instance.
        
        Args:
            method: Diarization method name ('pyannote', 'nemo', 'speechbrain')
            device: Device to run on
            params: Provider-specific parameters
            use_auth_token: HuggingFace token
            
        Returns:
            DiarizationProvider instance
        """
        method = method.lower()
        if method not in cls._providers:
            raise ValueError(f"Unknown diarization method: {method}. Available: {list(cls._providers.keys())}")
        
        provider_cls = cls._providers[method]
        return provider_cls(device=device, params=params, use_auth_token=use_auth_token)
    
    @classmethod
    def register(cls, name: str, provider_cls: type):
        """Register a custom Diarization provider."""
        cls._providers[name.lower()] = provider_cls
