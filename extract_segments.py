#!/usr/bin/env python3
"""
Post-processing script to extract and filter audio segments from transcription JSONs.

Steps:
1. Glob all JSON files from batch transcription
2. Filter segments by quality criteria (purity, duration, logprob, coverage)
3. Merge consecutive segments from same speaker if pause < threshold and result < max_duration
4. Cut audio segments to output folder
5. Create JSONL manifest with all metadata

Usage:
    python extract_segments.py /path/to/json_folder --output-dir /path/to/output \
        --min-purity 0.95 --min-duration 2.0 --max-duration 15.0
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import librosa
import numpy as np
from pydub import AudioSegment

logger = logging.getLogger("extract_segments")
_DNSMOS_SCORER: Optional["DNSMOSScorer"] = None


@dataclass
class FilterConfig:
    """Configuration for segment filtering."""

    min_duration: float = 2.0
    max_duration: float = 15.0
    min_purity: float = 0.95
    min_coverage: float = 0.5
    max_avg_logprob: float = 0.0  # logprob is negative, closer to 0 is better
    min_avg_logprob: float = -1.0  # filter out very bad segments

    # Merging settings
    merge_same_speaker: bool = True
    max_pause_for_merge: float = 0.2  # seconds
    # Maximum allowable total time (seconds) of non-main speakers overlapping
    # a segment. If the sum of all other speakers' overlap time exceeds this
    # threshold the segment is rejected.
    max_non_main_time: float = 0.5
    # Post-cut audio checks
    reject_clipped: bool = False
    clip_sample_threshold: float = 0.999
    max_clip_ratio: float = 0.002
    # DNSMOS gates (optional)
    min_dnsmos_sig: Optional[float] = None
    min_dnsmos_bak: Optional[float] = None
    dnsmos_sig_bak_ovr_model: Optional[str] = None


@dataclass
class SegmentInfo:
    """Information about a single segment."""

    # Source info
    source_audio: str
    source_json: str

    # Timing
    start: float
    end: float
    duration: float

    # Content
    text: str
    speaker: Optional[str] = None

    # Quality metrics
    purity: Optional[float] = None
    coverage: Optional[float] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None

    # Speaker overlap details
    speaker_overlaps: Optional[Dict[str, float]] = None
    # Total time (s) of non-main speakers overlapping this segment
    non_main_time: Optional[float] = None
    # Word-level timestamps normalized to segment-local time
    words: Optional[List[Dict[str, Any]]] = None

    # Merge info
    is_merged: bool = False
    merge_count: int = 1
    original_segments: List[Dict] = field(default_factory=list)
    source_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL output. audio_path added separately as first key."""
        d = {
            # audio_path will be inserted first when writing JSONL
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "speaker": self.speaker,
            "purity": self.purity,
            "coverage": self.coverage,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "compression_ratio": self.compression_ratio,
            "speaker_overlaps": self.speaker_overlaps,
            "non_main_time": self.non_main_time,
            "words": self.words,
            "is_merged": self.is_merged,
            "merge_count": self.merge_count,
            "source_metrics": self.source_metrics,
            # source_audio at the end
            "source_audio": self.source_audio,
            "source_json": self.source_json,
        }
        return {k: v for k, v in d.items() if v is not None}


def iter_json_files(root: Path) -> Iterable[Path]:
    """Recursively find all JSON files."""
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def _safe_filename_token(value: Any, max_len: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    token = token.strip("._-")
    if not token:
        token = "na"
    return token[:max_len]


def _download_file(url: str, target_path: Path) -> None:
    """Download file to target path atomically."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)  # nosec B310
    tmp_path.replace(target_path)


def _ensure_dnsmos_model(model_dir: Path) -> Path:
    """
    Ensure DNSMOS ONNX models are present locally.

    Sources:
      - https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
    """
    sig_bak_ovr_model = model_dir / "sig_bak_ovr.onnx"

    if not sig_bak_ovr_model.exists():
        logger.info("Downloading DNSMOS model: %s", sig_bak_ovr_model)
        _download_file(
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
            sig_bak_ovr_model,
        )
    return sig_bak_ovr_model


class DNSMOSScorer:
    """DNSMOS scorer backed by ONNX models from Microsoft's DNS-Challenge."""

    sample_rate = 16000
    input_length = 9.01

    @staticmethod
    def _ort_thread_config() -> tuple[int, int]:
        """
        Return (intra_op_threads, inter_op_threads) for ONNXRuntime sessions.

        Defaults to 1/1 to avoid ORT auto-affinity issues on some SLURM nodes.
        Override with env vars:
          - STT4SG_ORT_INTRA_OP_THREADS or ORT_INTRA_OP_NUM_THREADS
          - STT4SG_ORT_INTER_OP_THREADS or ORT_INTER_OP_NUM_THREADS
        """
        intra_raw = os.getenv("STT4SG_ORT_INTRA_OP_THREADS") or os.getenv(
            "ORT_INTRA_OP_NUM_THREADS"
        )
        inter_raw = os.getenv("STT4SG_ORT_INTER_OP_THREADS") or os.getenv(
            "ORT_INTER_OP_NUM_THREADS"
        )
        try:
            intra = max(1, int(intra_raw)) if intra_raw else 1
        except Exception:
            intra = 1
        try:
            inter = max(1, int(inter_raw)) if inter_raw else 1
        except Exception:
            inter = 1
        return intra, inter

    def __init__(self, sig_bak_ovr_model: Path):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is required for DNSMOS filtering. "
                "Install it before enabling --min-dnsmos-* filters."
            ) from exc

        providers = ["CPUExecutionProvider"]
        intra_threads, inter_threads = self._ort_thread_config()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sig_bak_ovr_sess = ort.InferenceSession(
            str(sig_bak_ovr_model),
            sess_options=sess_options,
            providers=providers,
        )
        self.sig_bak_ovr_input = self.sig_bak_ovr_sess.get_inputs()[0].name
        logger.info(
            "DNSMOS ORT threads: intra_op=%d inter_op=%d", intra_threads, inter_threads
        )

        # Non-personalized polynomial calibration from dnsmos_local.py.
        self._sig_poly = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self._bak_poly = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    @staticmethod
    def _to_mono_1d(y: np.ndarray) -> np.ndarray:
        """Normalize waveform shape to (num_samples,) robustly."""
        arr = np.asarray(y)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported waveform rank: {arr.ndim}")

        if 1 in arr.shape:
            return arr.reshape(-1).astype(np.float32, copy=False)

        # Handle both (channels, samples) and (samples, channels).
        if arr.shape[0] <= 8 and arr.shape[1] > 8:
            mono = arr.mean(axis=0)
        elif arr.shape[1] <= 8 and arr.shape[0] > 8:
            mono = arr.mean(axis=1)
        else:
            raise ValueError(
                f"Ambiguous 2D waveform shape {arr.shape}; expected channels axis <= 8"
            )
        return mono.astype(np.float32, copy=False)

    @staticmethod
    def _center_window(y: np.ndarray, samples_needed: int) -> np.ndarray:
        """Take centered window after repeating if needed."""
        if y.size == 0:
            raise ValueError("empty waveform")
        if y.shape[0] < samples_needed:
            repeats = int(np.ceil(samples_needed / y.shape[0])) + 2
            y = np.tile(y, repeats)
        start = (y.shape[0] - samples_needed) // 2
        end = start + samples_needed
        return y[start:end]

    def score_waveform(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        y = self._to_mono_1d(y)
        if y.size == 0:
            raise ValueError("empty waveform")

        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        peak = float(np.max(np.abs(y)))
        if peak > 1.0:
            y = y / peak

        samples_needed = int(self.input_length * sr)
        segment = self._center_window(y, samples_needed).astype(np.float32, copy=False)
        if segment.shape[0] != samples_needed:
            raise RuntimeError("DNSMOS failed: invalid segment length after center crop")

        raw_sig, raw_bak, _raw_ovr = self.sig_bak_ovr_sess.run(
            None, {self.sig_bak_ovr_input: np.array([segment], dtype=np.float32)}
        )[0][0]
        sig_mos = float(self._sig_poly(raw_sig))
        bak_mos = float(self._bak_poly(raw_bak))
        return {
            "dnsmos_sig": round(sig_mos, 4),
            "dnsmos_bak": round(bak_mos, 4),
        }


def _get_dnsmos_scorer(config: FilterConfig) -> Optional[DNSMOSScorer]:
    """Lazy-load DNSMOS scorer once per process."""
    if config.min_dnsmos_sig is None and config.min_dnsmos_bak is None:
        return None

    if not config.dnsmos_sig_bak_ovr_model:
        raise RuntimeError("DNSMOS models are not configured")

    global _DNSMOS_SCORER
    if _DNSMOS_SCORER is None:
        _DNSMOS_SCORER = DNSMOSScorer(
            sig_bak_ovr_model=Path(config.dnsmos_sig_bak_ovr_model),
        )
    return _DNSMOS_SCORER


def load_segments_from_json(json_path: Path) -> tuple[str, List[Dict]]:
    """Load segments with purity/coverage from batch_transcribe JSON output."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_file = data.get("audio_file", "")

    # batch_transcribe.py outputs segments at root level with all metrics included
    segments = data.get("segments")
    if not segments:
        raise ValueError(
            f"Missing segments in {json_path}. "
            "Expected 'segments' array at root level."
        )

    # Check that segments have purity/coverage (from diarization)
    first_seg = segments[0] if segments else {}
    if first_seg.get("purity") is None:
        raise ValueError(
            f"Segments missing purity in {json_path}. "
            "Run with --diarization to populate purity/coverage for filtering."
        )

    return audio_file, segments


def passes_quality_filter(seg: Dict, config: FilterConfig) -> tuple[bool, str]:
    """
    Check if a segment passes purity/coverage quality filters (PRE-merge).

    Returns (passed, reason) - reason is empty string if passed.
    """
    # Purity filter
    purity = seg.get("purity")
    if purity is not None and purity < config.min_purity:
        return False, f"purity {purity:.2f} < {config.min_purity}"

    # Coverage filter
    coverage = seg.get("coverage")
    if coverage is not None and coverage < config.min_coverage:
        return False, f"coverage {coverage:.2f} < {config.min_coverage}"

    # Logprob filter (negative values, closer to 0 is better)
    avg_logprob = seg.get("avg_logprob")
    if avg_logprob is not None:
        if avg_logprob < config.min_avg_logprob:
            return False, f"avg_logprob {avg_logprob:.2f} < {config.min_avg_logprob}"
        if avg_logprob > config.max_avg_logprob:
            return False, f"avg_logprob {avg_logprob:.2f} > {config.max_avg_logprob}"

    # Must have text
    if not seg.get("text", "").strip():
        return False, "empty text"

    # Cheap pre-merge overlap guard (same threshold re-checked post-merge too).
    non_main_time = _compute_non_main_time(seg)
    if config.max_non_main_time is not None and non_main_time is not None:
        if non_main_time > config.max_non_main_time:
            return (
                False,
                f"non_main_time {non_main_time:.2f}s > {config.max_non_main_time}s",
            )

    return True, ""


def _compute_non_main_time(seg: Dict[str, Any]) -> Optional[float]:
    """Compute total overlap duration from speakers other than the main speaker."""
    overlaps = seg.get("speaker_overlaps") or {}
    if not overlaps:
        return None
    main = seg.get("speaker")
    try:
        return float(
            sum(float(t) for s, t in overlaps.items() if s != main and t is not None)
        )
    except Exception:
        return None


def passes_non_main_overlap_filter(
    seg: Dict[str, Any], config: FilterConfig
) -> tuple[bool, str]:
    """Apply max_non_main_time constraint after merging."""
    non_main_time = _compute_non_main_time(seg)
    if config.max_non_main_time is None or non_main_time is None:
        return True, ""
    if non_main_time > config.max_non_main_time:
        return (
            False,
            f"non_main_time {non_main_time:.2f}s > {config.max_non_main_time}s",
        )
    return True, ""


def normalize_words_to_segment(seg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Normalize word start/end to segment-local timestamps."""
    words = seg.get("words")
    if not isinstance(words, list) or not words:
        return None

    seg_start = float(seg.get("start", 0.0) or 0.0)
    seg_duration = float(seg.get("duration", seg.get("end", 0.0) - seg_start) or 0.0)
    out: List[Dict[str, Any]] = []

    for word in words:
        if not isinstance(word, dict):
            continue
        item = dict(word)

        w_start = item.get("start")
        w_end = item.get("end")
        if w_start is not None:
            rel_start = max(0.0, float(w_start) - seg_start)
            if seg_duration > 0:
                rel_start = min(rel_start, seg_duration)
            item["start"] = round(rel_start, 4)
        if w_end is not None:
            rel_end = max(0.0, float(w_end) - seg_start)
            if seg_duration > 0:
                rel_end = min(rel_end, seg_duration)
            item["end"] = round(rel_end, 4)
        out.append(item)

    return out or None


def passes_duration_filter(seg: Dict, config: FilterConfig) -> tuple[bool, str]:
    """
    Check if a segment passes duration filters (POST-merge).

    Returns (passed, reason) - reason is empty string if passed.
    """
    duration = seg.get("duration", seg.get("end", 0) - seg.get("start", 0))

    if duration < config.min_duration:
        return False, f"duration {duration:.2f}s < {config.min_duration}s"
    if duration > config.max_duration:
        return False, f"duration {duration:.2f}s > {config.max_duration}s"

    return True, ""


def attach_diarization_metrics(
    transcription_segments: List[Dict], diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Attach diarization-derived fields (speaker, purity, coverage, overlaps) to
    transcription segments using time overlap.
    """
    diar_sorted = sorted(
        diarization_segments, key=lambda s: (s.get("start", 0), s.get("end", 0))
    )
    out = []

    for seg in transcription_segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        best = None
        best_overlap = 0.0
        for dseg in diar_sorted:
            dstart = dseg.get("start", 0)
            dend = dseg.get("end", 0)
            if dend <= start:
                continue
            if dstart >= end:
                break
            overlap = max(0.0, min(end, dend) - max(start, dstart))
            if overlap > best_overlap:
                best_overlap = overlap
                best = dseg

        merged = dict(seg)
        if best and best_overlap > 0:
            for key in ("speaker", "purity", "coverage", "speaker_overlaps"):
                if key in best:
                    merged[key] = best.get(key)
        out.append(merged)

    return out


def can_merge(seg1: Dict, seg2: Dict, config: FilterConfig) -> bool:
    """Check if two consecutive segments can be merged."""
    if not config.merge_same_speaker:
        return False

    # Must be same speaker
    if seg1.get("speaker") != seg2.get("speaker"):
        return False

    # Check pause duration
    pause = seg2.get("start", 0) - seg1.get("end", 0)
    if pause > config.max_pause_for_merge:
        return False
    if pause < 0:
        # Overlapping segments - don't merge
        return False

    # Check resulting duration
    merged_duration = seg2.get("end", 0) - seg1.get("start", 0)
    if merged_duration > config.max_duration:
        return False

    return True


def merge_segments(seg1: Dict, seg2: Dict) -> Dict:
    """Merge two segments into one, averaging metrics."""
    dur1 = seg1.get("end", 0) - seg1.get("start", 0)
    dur2 = seg2.get("end", 0) - seg2.get("start", 0)
    total_dur = dur1 + dur2

    # Weighted average for metrics
    def weighted_avg(v1, v2):
        if v1 is None and v2 is None:
            return None
        if v1 is None:
            return v2
        if v2 is None:
            return v1
        return (v1 * dur1 + v2 * dur2) / total_dur

    # Merge speaker_overlaps
    overlaps1 = seg1.get("speaker_overlaps", {}) or {}
    overlaps2 = seg2.get("speaker_overlaps", {}) or {}
    merged_overlaps = {}
    for speaker in set(overlaps1.keys()) | set(overlaps2.keys()):
        merged_overlaps[speaker] = overlaps1.get(speaker, 0) + overlaps2.get(speaker, 0)

    # Merge words
    words1 = seg1.get("words", []) or []
    words2 = seg2.get("words", []) or []
    merged_words = words1 + words2

    merged = {
        "start": seg1.get("start"),
        "end": seg2.get("end"),
        "duration": seg2.get("end", 0) - seg1.get("start", 0),
        "text": (seg1.get("text", "") + " " + seg2.get("text", "")).strip(),
        "speaker": seg1.get("speaker"),
        "purity": weighted_avg(seg1.get("purity"), seg2.get("purity")),
        "coverage": weighted_avg(seg1.get("coverage"), seg2.get("coverage")),
        "avg_logprob": weighted_avg(seg1.get("avg_logprob"), seg2.get("avg_logprob")),
        "no_speech_prob": weighted_avg(
            seg1.get("no_speech_prob"), seg2.get("no_speech_prob")
        ),
        "compression_ratio": weighted_avg(
            seg1.get("compression_ratio"), seg2.get("compression_ratio")
        ),
        "speaker_overlaps": merged_overlaps if merged_overlaps else None,
        "words": merged_words if merged_words else None,
        "_is_merged": True,
        "_merge_count": seg1.get("_merge_count", 1) + seg2.get("_merge_count", 1),
        "_original_segments": seg1.get("_original_segments", [seg1])
        + seg2.get("_original_segments", [seg2]),
    }

    # Round numeric values
    for key in [
        "purity",
        "coverage",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
    ]:
        if merged[key] is not None:
            merged[key] = round(merged[key], 4)

    return merged


def merge_consecutive_segments(
    segments: List[Dict], config: FilterConfig
) -> List[Dict]:
    """Merge consecutive segments from same speaker if criteria met."""
    if not segments or not config.merge_same_speaker:
        return segments

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.get("start", 0))

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        if can_merge(current, next_seg, config):
            current = merge_segments(current, next_seg)
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)
    return merged


def cut_audio_segment(
    audio: AudioSegment,
    start: float,
    end: float,
    output_path: Path,
    format: str,
    frame_ms: int,
    cut_pad_start_ms: int,
    cut_pad_end_ms: int,
) -> tuple[bool, Optional[AudioSegment]]:
    """
    Cut a segment from audio and save to file.

    Uses floor for start and ceil for end, rounding to frame boundaries.
    Applies explicit asymmetric padding before rounding so trailing speech
    is less likely to be cut off.

    Args:
        audio: Source audio
        start: Start time in seconds
        end: End time in seconds
        output_path: Where to save the segment
        format: Output format (flac, wav, mp3)
        frame_ms: Frame size in ms for rounding
        cut_pad_start_ms: Extra padding (ms) added before start
        cut_pad_end_ms: Extra padding (ms) added after end
    """
    import math

    try:
        # Convert to ms
        start_ms = start * 1000 - cut_pad_start_ms
        end_ms = end * 1000 + cut_pad_end_ms

        # Round START down to nearest frame boundary
        start_ms = int(math.floor(start_ms / frame_ms) * frame_ms)
        start_ms = max(0, start_ms)

        # Round END up to nearest frame boundary
        end_ms = int(math.ceil(end_ms / frame_ms) * frame_ms)
        end_ms = min(len(audio), end_ms)

        segment = audio[start_ms:end_ms]
        segment.export(str(output_path), format=format)
        return True, segment
    except Exception as e:
        logger.error(f"Failed to cut segment {start}-{end}: {e}")
        return False, None


def _audiosegment_to_mono_float32(audio: AudioSegment) -> tuple[np.ndarray, int]:
    """Convert pydub AudioSegment to mono float32 waveform in [-1, 1]."""
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    scale = float(1 << (8 * audio.sample_width - 1))
    y = samples.astype(np.float32, copy=False) / max(scale, 1.0)
    y = np.clip(y, -1.0, 1.0)
    return y, int(audio.frame_rate)


def _compute_basic_audio_metrics(
    y: np.ndarray,
    sr: int,
    clip_sample_threshold: float,
) -> Dict[str, Any]:
    """Compute lightweight post-cut waveform checks."""
    if y.size == 0:
        return {"audio_empty": True, "sample_rate": int(sr), "num_samples": 0}

    clip_ratio = float(np.mean(np.abs(y) >= clip_sample_threshold))
    return {
        "audio_empty": False,
        "sample_rate": int(sr),
        "num_samples": int(y.size),
        "clip_sample_threshold": round(float(clip_sample_threshold), 6),
        "clip_ratio": round(clip_ratio, 8),
    }


def _passes_post_cut_filter(
    metrics: Dict[str, Any], config: FilterConfig
) -> tuple[bool, str]:
    """Apply optional post-cut DSP and DNSMOS gates."""
    if metrics.get("audio_empty"):
        return False, "empty audio after cutting"

    clip_ratio = metrics.get("clip_ratio")
    if config.reject_clipped:
        if clip_ratio is None:
            return False, "clip_ratio missing"
        if clip_ratio > config.max_clip_ratio:
            return (
                False,
                f"clip_ratio {clip_ratio:.6f} > {config.max_clip_ratio} "
                f"(threshold={config.clip_sample_threshold})",
            )

    sig = metrics.get("dnsmos_sig")
    if config.min_dnsmos_sig is not None:
        if sig is None:
            return False, "dnsmos_sig missing"
        if sig < config.min_dnsmos_sig:
            return False, f"dnsmos_sig {sig:.2f} < {config.min_dnsmos_sig}"

    bak = metrics.get("dnsmos_bak")
    if config.min_dnsmos_bak is not None:
        if bak is None:
            return False, "dnsmos_bak missing"
        if bak < config.min_dnsmos_bak:
            return False, f"dnsmos_bak {bak:.2f} < {config.min_dnsmos_bak}"

    return True, ""


def process_json_file(
    json_path: Path,
    config: FilterConfig,
    output_dir: Path,
    input_root: Path,
    audio_format: str,
    dry_run: bool,
    frame_ms: int,
    cut_pad_start_ms: int,
    cut_pad_end_ms: int,
) -> tuple[List[Dict], List[Dict], Dict]:
    """
    Process a single JSON file and extract segments.

    Args:
        json_path: Path to the JSON file
        config: Filter configuration
        output_dir: Base output directory
        input_root: Root input directory (for calculating relative paths)
        audio_format: Output audio format
        dry_run: If True, don't actually cut audio
        frame_ms: Frame size for rounding timestamps
        cut_pad_start_ms: Extra padding (ms) before segment start
        cut_pad_end_ms: Extra padding (ms) after segment end

    Returns:
        (accepted_segments, rejected_segments, input_stats) where input_stats contains:
            - segments_in: number of segments loaded from JSON
            - duration_in: total duration of input segments
    """

    audio_file, segments = load_segments_from_json(json_path)
    logger.debug(f"Loaded {len(segments)} segments from {json_path}")

    # Track input stats for reporting
    input_stats = {
        "segments_in": len(segments),
        "duration_in": sum(
            seg.get("duration", seg.get("end", 0) - seg.get("start", 0))
            for seg in segments
        ),
    }

    rejected = []

    if not segments:
        logger.warning(f"No segments found in {json_path}")
        return [], [], input_stats

    if not audio_file or not Path(audio_file).exists():
        logger.error(f"Audio file not found: {audio_file}")
        return [], [], input_stats

    # Segments already have purity/coverage from batch_transcribe.py
    # No need to attach_diarization_metrics - they're already included

    # Step 1: Filter by quality (purity/coverage/logprob) ONLY - NOT duration!
    # This allows short segments to be merged before duration check
    quality_passed = []
    for seg in segments:
        passed, reason = passes_quality_filter(seg, config)
        if passed:
            quality_passed.append(seg)
        else:
            rejected.append(
                {
                    "stage": "quality_filter",
                    "reason": reason,
                    "source_audio": audio_file,
                    "source_json": str(json_path),
                    "segment": {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": seg.get(
                            "duration", seg.get("end", 0) - seg.get("start", 0)
                        ),
                        "text": seg.get("text", "")[:100],
                        "speaker": seg.get("speaker"),
                        "purity": seg.get("purity"),
                        "coverage": seg.get("coverage"),
                        "avg_logprob": seg.get("avg_logprob"),
                        "non_main_time": _compute_non_main_time(seg),
                    },
                }
            )
    logger.info(f"  Quality filter: {len(segments)} -> {len(quality_passed)} segments")

    if not quality_passed:
        return [], rejected, input_stats

    # Step 2: Merge consecutive segments from same speaker
    if config.merge_same_speaker:
        merged = merge_consecutive_segments(quality_passed, config)
        logger.info(f"  After merging: {len(merged)} segments")
    else:
        merged = quality_passed

    # Step 3: Filter by non-main overlap AFTER merging
    overlap_passed = []
    for seg in merged:
        passed, reason = passes_non_main_overlap_filter(seg, config)
        if passed:
            overlap_passed.append(seg)
        else:
            rejected.append(
                {
                    "stage": "overlap_filter",
                    "reason": reason,
                    "source_audio": audio_file,
                    "source_json": str(json_path),
                    "segment": {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": seg.get(
                            "duration", seg.get("end", 0) - seg.get("start", 0)
                        ),
                        "text": seg.get("text", "")[:100],
                        "speaker": seg.get("speaker"),
                        "purity": seg.get("purity"),
                        "coverage": seg.get("coverage"),
                        "non_main_time": _compute_non_main_time(seg),
                        "is_merged": seg.get("_is_merged", False),
                        "merge_count": seg.get("_merge_count", 1),
                    },
                }
            )
    logger.info(
        "  Overlap filter: %d -> %d segments", len(merged), len(overlap_passed)
    )

    if not overlap_passed:
        return [], rejected, input_stats

    # Step 4: Filter by duration AFTER merging/overlap filtering
    final = []
    for seg in overlap_passed:
        passed, reason = passes_duration_filter(seg, config)
        if passed:
            final.append(seg)
        else:
            rejected.append(
                {
                    "stage": "duration_filter",
                    "reason": reason,
                    "source_audio": audio_file,
                    "source_json": str(json_path),
                    "segment": {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "duration": seg.get(
                            "duration", seg.get("end", 0) - seg.get("start", 0)
                        ),
                        "text": seg.get("text", "")[:100],
                        "speaker": seg.get("speaker"),
                        "purity": seg.get("purity"),
                        "coverage": seg.get("coverage"),
                        "non_main_time": _compute_non_main_time(seg),
                        "is_merged": seg.get("_is_merged", False),
                        "merge_count": seg.get("_merge_count", 1),
                    },
                }
            )
    logger.info(f"  Duration filter: {len(overlap_passed)} -> {len(final)} segments")

    if not final:
        return [], rejected, input_stats

    # Create output subdirectory preserving folder structure
    # e.g., input: srf/Trüffelschweine/ep1.mp3 -> output: processed/audio/Trüffelschweine/ep1/
    source_stem = Path(audio_file).stem
    try:
        # Get relative path from input root to the audio file's parent
        rel_parent = Path(audio_file).parent.relative_to(input_root)
        segment_dir = output_dir / "audio" / rel_parent / source_stem
    except ValueError:
        # Fallback if audio_file is not under input_root
        segment_dir = output_dir / "audio" / source_stem
    if not dry_run:
        segment_dir.mkdir(parents=True, exist_ok=True)

    # Sort final segments by start time for proper cutting
    final = sorted(final, key=lambda s: s.get("start", 0))

    pending: List[tuple[Dict, Path]] = []
    for seg in final:
        start_str = f"{seg['start']:.2f}".replace(".", "_")
        end_str = f"{seg['end']:.2f}".replace(".", "_")
        segment_filename = f"{source_stem}_{start_str}-{end_str}.{audio_format}"
        segment_path = segment_dir / segment_filename
        pending.append((seg, segment_path))

    # Load audio once for pending segments
    audio = None
    if not dry_run:
        try:
            audio = AudioSegment.from_file(audio_file)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_file}: {e}")
            return [], rejected, input_stats

    # Process each pending segment
    results = []
    for seg, segment_path in pending:
        segment_filename = segment_path.name

        # Cut audio with frame-aligned boundaries (Silero uses 32ms frames)
        segment_metrics = None
        cut_segment = None
        if not dry_run and audio:
            success, cut_segment = cut_audio_segment(
                audio,
                start=seg["start"],
                end=seg["end"],
                output_path=segment_path,
                format=audio_format,
                frame_ms=frame_ms,
                cut_pad_start_ms=cut_pad_start_ms,
                cut_pad_end_ms=cut_pad_end_ms,
            )
            if not success:
                continue
            if cut_segment is None:
                continue

            y, sr = _audiosegment_to_mono_float32(cut_segment)
            segment_metrics = _compute_basic_audio_metrics(
                y,
                sr,
                clip_sample_threshold=config.clip_sample_threshold,
            )
            segment_metrics["max_clip_ratio"] = round(float(config.max_clip_ratio), 8)
            segment_metrics["is_clipped"] = bool(
                segment_metrics.get("clip_ratio", 0.0) > config.max_clip_ratio
            )

            dnsmos_scorer = _get_dnsmos_scorer(config)
            if dnsmos_scorer is not None:
                try:
                    segment_metrics.update(dnsmos_scorer.score_waveform(y, sr))
                except Exception as exc:
                    raise RuntimeError(
                        f"DNSMOS scoring failed for {segment_path}: {exc}"
                    ) from exc

            passed_post, reason_post = _passes_post_cut_filter(segment_metrics, config)
            if not passed_post:
                try:
                    if segment_path.exists():
                        segment_path.unlink()
                except Exception as exc:
                    logger.warning(
                        "Failed to delete rejected segment file %s: %s",
                        segment_path,
                        exc,
                    )
                rejected.append(
                    {
                        "stage": "post_cut_filter",
                        "reason": reason_post,
                        "source_audio": audio_file,
                        "source_json": str(json_path),
                        "segment": {
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "duration": seg.get(
                                "duration", seg.get("end", 0) - seg.get("start", 0)
                            ),
                            "text": seg.get("text", "")[:100],
                            "speaker": seg.get("speaker"),
                            "clip_ratio": segment_metrics.get("clip_ratio"),
                            "clip_sample_threshold": segment_metrics.get(
                                "clip_sample_threshold"
                            ),
                            "max_clip_ratio": segment_metrics.get("max_clip_ratio"),
                            "is_clipped": segment_metrics.get("is_clipped"),
                            "dnsmos_sig": segment_metrics.get("dnsmos_sig"),
                            "dnsmos_bak": segment_metrics.get("dnsmos_bak"),
                        },
                    }
                )
                continue

        # Create segment info
        info = SegmentInfo(
            source_audio=audio_file,
            source_json=str(json_path),
            start=seg["start"],
            end=seg["end"],
            duration=seg.get("duration", seg["end"] - seg["start"]),
            text=seg.get("text", ""),
            speaker=seg.get("speaker"),
            purity=seg.get("purity"),
            coverage=seg.get("coverage"),
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
            compression_ratio=seg.get("compression_ratio"),
            speaker_overlaps=seg.get("speaker_overlaps"),
            non_main_time=_compute_non_main_time(seg),
            words=normalize_words_to_segment(seg),
            is_merged=seg.get("_is_merged", False),
            merge_count=seg.get("_merge_count", 1),
            source_metrics=segment_metrics,
        )

        # Add output path to dict for JSONL - audio_path as FIRST key
        info_dict = info.to_dict()
        audio_path_value = (
            str(segment_path.resolve())
            if not dry_run
            else f"<dry-run>/{segment_filename}"
        )
        # Create new dict with audio_path first, then text, then rest
        ordered_dict = {"audio_path": audio_path_value}
        ordered_dict.update(info_dict)

        results.append(ordered_dict)

    return results, rejected, input_stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract and filter audio segments from transcription JSONs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument("input_dir", help="Directory containing JSON files to process")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for segments and JSONL"
    )
    parser.add_argument(
        "--audio-format",
        default="flac",
        choices=["wav", "mp3", "flac"],
        help="Output audio format",
    )

    # Duration filters
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum segment duration (seconds)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum segment duration (seconds)",
    )

    # Quality filters
    parser.add_argument(
        "--min-purity", type=float, default=0.95, help="Minimum speaker purity (0-1)"
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.9, help="Minimum speaker coverage (0-1)"
    )
    parser.add_argument(
        "--min-avg-logprob",
        type=float,
        default=-0.5,
        help="Minimum avg_logprob (negative, closer to 0 is better)",
    )
    parser.add_argument(
        "--max-avg-logprob",
        type=float,
        default=0.0,
        help="Maximum avg_logprob (should be <= 0)",
    )

    # Merging options
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Disable merging of consecutive segments",
    )
    parser.add_argument(
        "--max-pause",
        type=float,
        default=0.2,
        help="Maximum pause between segments to merge (seconds)",
    )

    parser.add_argument(
        "--max-non-main-time",
        type=float,
        default=0.5,
        help="Maximum total time (s) of non-main speakers overlapping a segment",
    )
    parser.add_argument(
        "--reject-clipped",
        action="store_true",
        help="Enable clip-ratio rejection on cut audio",
    )
    parser.add_argument(
        "--clip-sample-threshold",
        type=float,
        default=0.999,
        help="Absolute sample level counted as clipped for clip-ratio",
    )
    parser.add_argument(
        "--max-clip-ratio",
        type=float,
        default=0.002,
        help="Maximum allowed clipped-sample ratio when --reject-clipped is enabled",
    )
    parser.add_argument(
        "--min-dnsmos-sig",
        type=float,
        default=None,
        help="Optional minimum DNSMOS SIG score (post-cut filter)",
    )
    parser.add_argument(
        "--min-dnsmos-bak",
        type=float,
        default=None,
        help="Optional minimum DNSMOS BAK score (post-cut filter)",
    )
    parser.add_argument(
        "--dnsmos-model-dir",
        type=Path,
        default=Path.home() / ".cache" / "dnsmos",
        help="Directory for DNSMOS ONNX models (auto-downloaded if missing)",
    )
    parser.add_argument(
        "--dnsmos-sig-bak-ovr-model",
        type=Path,
        default=None,
        help="Path to DNSMOS sig_bak_ovr.onnx (overrides auto-download)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually cut audio, just show what would be done",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=32,
        help="Frame size in ms for rounding timestamps (Silero uses 32ms)",
    )
    parser.add_argument(
        "--cut-pad-start-ms",
        type=int,
        default=25,
        help="Extra padding in ms before each cut segment start",
    )
    parser.add_argument(
        "--cut-pad-end-ms",
        type=int,
        default=200,
        help="Extra padding in ms after each cut segment end",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of JSON files to process"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes (set 1 to disable parallelism)",
    )
    parser.add_argument(
        "--no-summary",
        "--quiet",
        dest="summary",
        action="store_false",
        help="Suppress end-of-run summary output",
    )
    parser.set_defaults(summary=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clip_sample_threshold <= 0.0 or args.clip_sample_threshold > 1.0:
        raise SystemExit("--clip-sample-threshold must be in (0, 1].")
    if args.max_clip_ratio < 0.0 or args.max_clip_ratio > 1.0:
        raise SystemExit("--max-clip-ratio must be in [0, 1].")

    dnsmos_sig_bak_ovr_model: Optional[Path] = None
    dnsmos_enabled = args.min_dnsmos_sig is not None or args.min_dnsmos_bak is not None
    if dnsmos_enabled and not args.dry_run:
        if args.dnsmos_sig_bak_ovr_model:
            dnsmos_sig_bak_ovr_model = args.dnsmos_sig_bak_ovr_model.resolve()
        else:
            dnsmos_sig_bak_ovr_model = _ensure_dnsmos_model(args.dnsmos_model_dir.resolve())

        if not dnsmos_sig_bak_ovr_model.exists():
            raise SystemExit("DNSMOS model file is missing after setup.")

    # Build filter config
    config = FilterConfig(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_purity=args.min_purity,
        min_coverage=args.min_coverage,
        min_avg_logprob=args.min_avg_logprob,
        max_avg_logprob=args.max_avg_logprob,
        merge_same_speaker=args.merge,
        max_pause_for_merge=args.max_pause,
        max_non_main_time=args.max_non_main_time,
        reject_clipped=args.reject_clipped,
        clip_sample_threshold=args.clip_sample_threshold,
        max_clip_ratio=args.max_clip_ratio,
        min_dnsmos_sig=args.min_dnsmos_sig,
        min_dnsmos_bak=args.min_dnsmos_bak,
        dnsmos_sig_bak_ovr_model=(
            str(dnsmos_sig_bak_ovr_model) if dnsmos_sig_bak_ovr_model else None
        ),
    )

    logger.info(
        f"Filter config: min_dur={config.min_duration}s, max_dur={config.max_duration}s, "
        f"min_purity={config.min_purity}, min_coverage={config.min_coverage}, "
        f"max_non_main_time={config.max_non_main_time}s"
    )
    logger.info(
        f"Merge config: enabled={config.merge_same_speaker}, max_pause={config.max_pause_for_merge}s, "
        f"max_dur={config.max_duration}s"
    )
    if config.reject_clipped:
        logger.info(
            "Clip filter enabled: clip_ratio <= %s (sample_threshold=%s)",
            config.max_clip_ratio,
            config.clip_sample_threshold,
        )
    if dnsmos_enabled:
        logger.info(
            "DNSMOS filters enabled: min_sig=%s, min_bak=%s",
            config.min_dnsmos_sig,
            config.min_dnsmos_bak,
        )

    # Find all JSON files
    json_files = sorted(iter_json_files(input_dir))
    if args.limit:
        json_files = json_files[: args.limit]

    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be created")

    # Process all files
    stats = {
        "total_json_files": len(json_files),
        "processed_files": 0,
        "total_segments_in": 0,
        "total_segments_out": 0,
        "total_rejected": 0,
        "total_duration_in": 0.0,
        "total_duration_out": 0.0,
        "speakers": {},
        "rejection_reasons": {},
        "rejected_samples_written": 0,
    }

    jsonl_path = output_dir / "manifest.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    rejected_samples_dir = output_dir / "_rejected_samples"
    rejected_samples_dir.mkdir(parents=True, exist_ok=True)
    rejected_sample_prob = 0.01
    rejected_sample_limit = 50
    rejected_sample_min_duration = float(config.min_duration)
    rejected_sample_count = 0
    rejected_sample_rng = random.Random()
    rejected_audio_cache: Dict[str, AudioSegment] = {}
    manifest_mode = "w"
    logger.info(
        "Writing manifest incrementally to %s (%s mode)",
        jsonl_path,
        "append" if manifest_mode == "a" else "write",
    )
    logger.info(f"Writing rejected segments incrementally to {rejected_path}")
    logger.info(
        "Writing sampled rejected audio to %s (p=%.2f, max=%d)",
        rejected_samples_dir,
        rejected_sample_prob,
        rejected_sample_limit,
    )
    logger.info(
        "Rejected sample criteria: reason in {avg_logprob, dnsmos_*}, duration >= %.2fs",
        rejected_sample_min_duration,
    )

    max_workers = int(args.workers)
    if max_workers > 1:
        logger.info(f"Parallel processing enabled with {max_workers} workers")
    else:
        logger.info("Parallel processing disabled (workers=1)")

    with (
        open(jsonl_path, manifest_mode, encoding="utf-8") as manifest_f,
        open(rejected_path, "w", encoding="utf-8") as rejected_f,
    ):
        def maybe_write_rejected_sample(rej: Dict[str, Any]) -> Optional[str]:
            nonlocal rejected_sample_count
            if rejected_sample_count >= rejected_sample_limit:
                return None

            stage = str(rej.get("stage") or "")
            reason = str(rej.get("reason") or "")
            reason_l = reason.lower()
            is_avg_logprob_reject = stage == "quality_filter" and "avg_logprob" in reason_l
            is_dnsmos_reject = stage == "post_cut_filter" and "dnsmos_" in reason_l
            if not (is_avg_logprob_reject or is_dnsmos_reject):
                return None

            source_audio = rej.get("source_audio")
            seg = rej.get("segment") if isinstance(rej.get("segment"), dict) else None
            if not source_audio or seg is None:
                return None

            try:
                start = float(seg.get("start"))
                end = float(seg.get("end"))
            except Exception:
                return None
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                return None
            seg_duration = float(seg.get("duration", end - start) or (end - start))
            if seg_duration < rejected_sample_min_duration:
                return None
            if rejected_sample_rng.random() >= rejected_sample_prob:
                return None

            source_path = Path(str(source_audio))
            if not source_path.exists():
                return None
            source_key = str(source_path.resolve())

            audio = rejected_audio_cache.get(source_key)
            if audio is None:
                try:
                    audio = AudioSegment.from_file(source_key)
                except Exception as exc:
                    logger.warning(
                        "Failed to load source audio for rejected sample %s: %s",
                        source_key,
                        exc,
                    )
                    return None
                rejected_audio_cache[source_key] = audio

            stage = _safe_filename_token(stage or "rejected", max_len=24)
            source_stem = _safe_filename_token(source_path.stem, max_len=48)
            start_str = f"{start:.2f}".replace(".", "_")
            end_str = f"{end:.2f}".replace(".", "_")
            sample_idx = rejected_sample_count + 1
            sample_name = (
                f"rej_{sample_idx:03d}_{stage}_{source_stem}_{start_str}-{end_str}."
                f"{args.audio_format}"
            )
            sample_path = rejected_samples_dir / sample_name

            success, _ = cut_audio_segment(
                audio=audio,
                start=start,
                end=end,
                output_path=sample_path,
                format=args.audio_format,
                frame_ms=args.frame_ms,
                cut_pad_start_ms=args.cut_pad_start_ms,
                cut_pad_end_ms=args.cut_pad_end_ms,
            )
            if not success or not sample_path.exists():
                return None

            rejected_sample_count += 1
            stats["rejected_samples_written"] = rejected_sample_count
            return str(sample_path.resolve())

        def persist_and_update(
            segments: List[Dict], rejected: List[Dict], input_stats: Dict[str, Any]
        ) -> None:
            written_segments = 0
            for seg in segments:
                manifest_f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                written_segments += 1
            for rej in rejected:
                rej_out = dict(rej)
                sample_audio_path = maybe_write_rejected_sample(rej_out)
                if sample_audio_path:
                    rej_out["sample_audio_path"] = sample_audio_path
                rejected_f.write(json.dumps(rej_out, ensure_ascii=False) + "\n")
            manifest_f.flush()
            rejected_f.flush()

            stats["processed_files"] += 1
            stats["total_segments_in"] += input_stats["segments_in"]
            stats["total_duration_in"] += input_stats["duration_in"]
            stats["total_segments_out"] += written_segments
            stats["total_rejected"] += len(rejected)

            for seg in segments:
                stats["total_duration_out"] += seg.get("duration", 0)
                speaker = seg.get("speaker", "UNKNOWN")
                stats["speakers"][speaker] = stats["speakers"].get(speaker, 0) + 1

            for rej in rejected:
                reason = rej.get("reason", "unknown")
                stats["rejection_reasons"][reason] = (
                    stats["rejection_reasons"].get(reason, 0) + 1
                )

        if max_workers == 1:
            for json_path in json_files:
                logger.info(f"Processing: {json_path}")
                try:
                    result = process_json_file(
                        json_path,
                        config,
                        output_dir,
                        input_dir,
                        args.audio_format,
                        args.dry_run,
                        frame_ms=args.frame_ms,
                        cut_pad_start_ms=args.cut_pad_start_ms,
                        cut_pad_end_ms=args.cut_pad_end_ms,
                    )
                except Exception as e:
                    logger.exception(f"Error processing {json_path}: {e}")
                    if dnsmos_enabled:
                        raise
                    continue
                segments, rejected, input_stats = result
                persist_and_update(segments, rejected, input_stats)
        else:
            future_to_path = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                for json_path in json_files:
                    future = executor.submit(
                        process_json_file,
                        json_path,
                        config,
                        output_dir,
                        input_dir,
                        args.audio_format,
                        args.dry_run,
                        args.frame_ms,
                        args.cut_pad_start_ms,
                        args.cut_pad_end_ms,
                    )
                    future_to_path[future] = json_path

                for future in concurrent.futures.as_completed(future_to_path):
                    json_path = future_to_path[future]
                    logger.info(f"Completed: {json_path}")
                    try:
                        segments, rejected, input_stats = future.result()
                    except Exception as e:
                        logger.exception(f"Error processing {json_path}: {e}")
                        if dnsmos_enabled:
                            raise
                        continue
                    persist_and_update(segments, rejected, input_stats)

    logger.info(f"Wrote manifest to {jsonl_path}")
    logger.info(f"Wrote rejected segments to {rejected_path}")

    # Calculate dropped stats
    dropped_duration = stats["total_duration_in"] - stats["total_duration_out"]
    dropped_pct = (
        (dropped_duration / stats["total_duration_in"] * 100)
        if stats["total_duration_in"] > 0
        else 0
    )
    dropped_hours = dropped_duration / 3600

    # Write stats
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **stats,
                "total_duration_in_hours": round(stats["total_duration_in"] / 3600, 2),
                "total_duration_out_hours": round(
                    stats["total_duration_out"] / 3600, 2
                ),
                "dropped_duration": round(dropped_duration, 1),
                "dropped_duration_hours": round(dropped_hours, 2),
                "dropped_percentage": round(dropped_pct, 1),
                "config": {
                    "min_duration": config.min_duration,
                    "max_duration": config.max_duration,
                    "min_purity": config.min_purity,
                    "min_coverage": config.min_coverage,
                    "min_avg_logprob": config.min_avg_logprob,
                    "max_avg_logprob": config.max_avg_logprob,
                    "merge_same_speaker": config.merge_same_speaker,
                    "max_non_main_time": config.max_non_main_time,
                    "max_pause_for_merge": config.max_pause_for_merge,
                    "max_merged_duration": config.max_duration,
                    "reject_clipped": config.reject_clipped,
                    "clip_sample_threshold": config.clip_sample_threshold,
                    "max_clip_ratio": config.max_clip_ratio,
                    "min_dnsmos_sig": config.min_dnsmos_sig,
                    "min_dnsmos_bak": config.min_dnsmos_bak,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Summary
    if args.summary:
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(
            f"JSON files processed: {stats['processed_files']}/{stats['total_json_files']}"
        )
        print(f"\nSegments:")
        print(
            f"  Input:     {stats['total_segments_in']} ({stats['total_duration_in']:.1f}s = {stats['total_duration_in']/3600:.2f} hours)"
        )
        print(
            f"  Output:    {stats['total_segments_out']} ({stats['total_duration_out']:.1f}s = {stats['total_duration_out']/3600:.2f} hours)"
        )
        print(f"  Rejected:  {stats['total_rejected']}")
        print(f"\nDropped: {dropped_pct:.1f}% ({dropped_hours:.2f} hours)")
        print(f"\nSpeakers: {len(stats['speakers'])}")
        for speaker, count in sorted(stats["speakers"].items()):
            print(f"  - {speaker}: {count} segments")
        if stats["rejection_reasons"]:
            print(f"\nRejection reasons:")
            for reason, count in sorted(
                stats["rejection_reasons"].items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason}: {count}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Manifest: {jsonl_path}")
        print(f"Rejected: {rejected_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
