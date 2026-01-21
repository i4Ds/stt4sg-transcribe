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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import VoiceActivityDetection

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
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        pipeline_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PyAnnote Diarization.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            use_auth_token: HuggingFace auth token for accessing gated models
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self.pipeline_name = pipeline_name
        self.pipeline_params = pipeline_params or {}
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            logger.info("Loading PyAnnote diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                self.pipeline_name,
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
        pipeline_params: Optional[Dict[str, Any]] = None,
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
        if pipeline_params is None:
            pipeline_params = self.pipeline_params
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


class CombinedVADDiarization:
    """Combined VAD and Diarization for the full pipeline."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_auth_token: Optional[str] = None,
        diarization_method: str = "pyannote",
        diarization_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize combined VAD and Diarization.
        
        Args:
            device: Device to run the models on
            use_auth_token: HuggingFace auth token
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self.diarization_method = diarization_method
        self.diarization_params = diarization_params or {}
        self.vad = PyAnnoteVAD(device=device, use_auth_token=use_auth_token)
        self._diarization_provider = None
    
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
            # Full diarization (includes VAD internally for most providers)
            provider = self._get_diarization_provider()
            return provider.diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
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

    def _get_diarization_provider(self):
        if self._diarization_provider is not None:
            return self._diarization_provider

        method = (self.diarization_method or "pyannote").lower()
        params = self.diarization_params or {}
        if method == "pyannote":
            pipeline_name = params.get("pipeline_name", "pyannote/speaker-diarization-3.1")
            pipeline_params = params.get("pipeline_params")
            self._diarization_provider = PyAnnoteDiarization(
                device=self.device,
                use_auth_token=self.use_auth_token,
                pipeline_name=pipeline_name,
                pipeline_params=pipeline_params,
            )
        elif method == "nemo":
            self._diarization_provider = NemoClusteringDiarization(
                device=self.device,
                diarization_params=params,
            )
        elif method == "speechbrain":
            self._diarization_provider = SpeechBrainDiarization(
                device=self.device,
                diarization_params=params,
            )
        else:
            raise ValueError(f"Unknown diarization method: {self.diarization_method}")
        return self._diarization_provider


def _parse_rttm(rttm_path: Union[str, Path], uri: Optional[str] = None) -> List[SpeechSegment]:
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


class NemoClusteringDiarization:
    """Offline diarization using NVIDIA NeMo's ClusteringDiarizer."""

    def __init__(self, device: str, diarization_params: Optional[Dict[str, Any]] = None):
        self.device = device
        self.params = diarization_params or {}

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
        segments = _parse_rttm(rttm_path, uri=target_stem)
        speakers_set = {s.speaker for s in segments if s.speaker}
        return DiarizationResult(segments=segments, num_speakers=len(speakers_set))


class SpeechBrainDiarization:
    """SpeechBrain diarization baseline: VAD -> embeddings -> spectral clustering."""

    def __init__(self, device: str, diarization_params: Optional[Dict[str, Any]] = None):
        self.device = device
        self.params = diarization_params or {}

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        try:
            import numpy as np
            import torchaudio
            from speechbrain.inference.VAD import VAD
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.processing import diarization as diar
        except Exception as exc:
            raise RuntimeError("SpeechBrain diarization requires speechbrain, torchaudio, numpy") from exc

        def load_mono_16k(path: Path):
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                sr = 16000
            return wav, sr

        def torch_from_np(x):
            import torch
            return torch.from_numpy(x.astype(np.float32))

        class np_no_grad:
            def __enter__(self):
                import torch
                self._ctx = torch.no_grad()
                return self._ctx.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return self._ctx.__exit__(exc_type, exc, tb)

        audio_path = Path(audio_path)
        uri = self.params.get("uri", audio_path.stem)

        wav, sr = load_mono_16k(audio_path)
        run_device = self.device
        if run_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    run_device = "cpu"
            except Exception:
                run_device = "cpu"
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

        spk_model = self.params.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")
        spk = EncoderClassifier.from_hparams(
            source=spk_model,
            savedir=self.params.get("speaker_savedir", "pretrained_sb_ecapa"),
            run_opts=run_opts,
        )

        win = float(self.params.get("win", 1.5))
        hop = float(self.params.get("hop", 0.75))
        min_spk = min_speakers if min_speakers is not None else int(self.params.get("min_spk", 2))
        max_spk = max_speakers if max_speakers is not None else int(self.params.get("max_spk", 10))
        pval = float(self.params.get("pval", 0.30))

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
                chunk = torch_from_np(wav_np[a:b])[None, :]
                with np_no_grad():
                    emb = spk.encode_batch(chunk).squeeze(0).squeeze(0).cpu().numpy()
                embs.append(emb)
                seg_meta.append((s2, e2))
                t += hop

        if len(embs) < max(2, min_spk):
            raise RuntimeError(f"Too few segments ({len(embs)}) to cluster")

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
