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
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydub import AudioSegment

logger = logging.getLogger("extract_segments")


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

    # Word-level info (optional)
    words: Optional[List[Dict]] = None
    avg_word_score: Optional[float] = None

    # Merge info
    is_merged: bool = False
    merge_count: int = 1
    original_segments: List[Dict] = field(default_factory=list)

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
            "avg_word_score": self.avg_word_score,
            "is_merged": self.is_merged,
            "merge_count": self.merge_count,
            # source_audio at the end
            "source_audio": self.source_audio,
            "source_json": self.source_json,
        }
        # Don't include words in JSONL to keep it compact
        return {k: v for k, v in d.items() if v is not None}


def iter_json_files(root: Path) -> Iterable[Path]:
    """Recursively find all JSON files."""
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


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

    # Check total overlap time from non-main speakers (if available)
    overlaps = seg.get("speaker_overlaps") or {}
    if overlaps and config.max_non_main_time is not None:
        main = seg.get("speaker")
        try:
            non_main_time = sum(
                float(t) for s, t in overlaps.items() if s != main and t is not None
            )
        except Exception:
            non_main_time = 0.0

        if non_main_time > config.max_non_main_time:
            return (
                False,
                f"non_main_time {non_main_time:.2f}s > {config.max_non_main_time}s",
            )

    return True, ""


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


def calculate_avg_word_score(seg: Dict) -> Optional[float]:
    """Calculate average word alignment score from segment."""
    words = seg.get("words", [])
    if not words:
        return None

    scores = [
        w.get("score") or w.get("probability")
        for w in words
        if w.get("score") or w.get("probability")
    ]
    if not scores:
        return None

    return round(sum(scores) / len(scores), 4)


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
    format: str = "flac",
    frame_ms: int = 10,
) -> bool:
    """
    Cut a segment from audio and save to file.

    Uses floor for start and ceil for end, rounding to frame boundaries.
    Silero VAD uses 400ms frames by default (speech_pad_ms), so we round
    END up to the nearest frame boundary to avoid cutting off audio.

    Args:
        audio: Source audio
        start: Start time in seconds
        end: End time in seconds
        output_path: Where to save the segment
        format: Output format (flac, wav, mp3)
        frame_ms: Frame size in ms for rounding (default: 400 for Silero)
    """
    import math

    try:
        # Convert to ms
        start_ms = start * 1000
        end_ms = end * 1000

        # Round START down to nearest frame boundary
        start_ms = int(math.floor(start_ms / frame_ms) * frame_ms)
        start_ms = max(0, start_ms)

        # Round END up to nearest frame boundary
        end_ms = int(math.ceil(end_ms / frame_ms) * frame_ms)
        end_ms = min(len(audio), end_ms)

        segment = audio[start_ms:end_ms]
        segment.export(str(output_path), format=format)
        return True
    except Exception as e:
        logger.error(f"Failed to cut segment {start}-{end}: {e}")
        return False


def process_json_file(
    json_path: Path,
    config: FilterConfig,
    output_dir: Path,
    input_root: Path,
    audio_format: str = "flac",
    dry_run: bool = False,
    frame_ms: int = 10,
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
        return [], []

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
            # compute non_main_time for logging/storage
            overlaps = seg.get("speaker_overlaps") or {}
            main = seg.get("speaker")
            try:
                non_main_time = sum(
                    float(t) for s, t in overlaps.items() if s != main and t is not None
                )
            except Exception:
                non_main_time = None
            rejected.append(
                {
                    "stage": "quality_filter",
                    "reason": reason,
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
                        "non_main_time": non_main_time,
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

    # Step 3: Filter by duration AFTER merging
    final = []
    for seg in merged:
        passed, reason = passes_duration_filter(seg, config)
        if passed:
            final.append(seg)
        else:
            rejected.append(
                {
                    "stage": "duration_filter",
                    "reason": reason,
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
                        "is_merged": seg.get("_is_merged", False),
                        "merge_count": seg.get("_merge_count", 1),
                    },
                }
            )
    logger.info(f"  Duration filter: {len(merged)} -> {len(final)} segments")

    if not final:
        return [], rejected, input_stats

    # Load audio once for all segments
    audio = None
    if not dry_run:
        try:
            audio = AudioSegment.from_file(audio_file)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_file}: {e}")
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

    # Process each segment
    results = []
    for i, seg in enumerate(final):
        # Generate output filename
        start_str = f"{seg['start']:.2f}".replace(".", "_")
        end_str = f"{seg['end']:.2f}".replace(".", "_")
        segment_filename = f"{source_stem}_{start_str}-{end_str}.{audio_format}"
        segment_path = segment_dir / segment_filename

        # Determine cut_end: use START of NEXT segment if available
        # This avoids cutting off trailing audio between segments
        if i + 1 < len(final):
            next_start = final[i + 1].get("start", seg["end"])
            # Use next segment's start, but cap at a reasonable max (e.g., +2s)
            max_end = seg["end"] + 1.0
            cut_end = min(next_start, max_end)
        else:
            # Last segment: use its own end timestamp
            cut_end = seg["end"]

        # Cut audio with frame-aligned boundaries (Silero uses 400ms frames)
        if not dry_run and audio:
            success = cut_audio_segment(
                audio,
                seg["start"],
                cut_end,
                segment_path,
                audio_format,
                frame_ms=frame_ms,
            )
            if not success:
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
            non_main_time=(
                sum(float(t) for s, t in (seg.get("speaker_overlaps") or {}).items() if s != seg.get("speaker") and t is not None)
                if seg.get("speaker_overlaps")
                else None
            ),
            words=seg.get("words"),
            avg_word_score=calculate_avg_word_score(seg),
            is_merged=seg.get("_is_merged", False),
            merge_count=seg.get("_merge_count", 1),
        )

        # Add output path to dict for JSONL - audio_path as FIRST key
        info_dict = info.to_dict()
        audio_path_value = (
            str(segment_path) if not dry_run else f"<dry-run>/{segment_filename}"
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

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually cut audio, just show what would be done",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=400,
        help="Frame size in ms for rounding timestamps (Silero uses 400ms)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of JSON files to process"
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

    # Find all JSON files
    json_files = sorted(iter_json_files(input_dir))
    if args.limit:
        json_files = json_files[: args.limit]

    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be created")

    # Process all files
    all_segments = []
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
    }

    all_rejected = []

    for json_path in json_files:
        logger.info(f"Processing: {json_path}")

        try:
            segments, rejected, input_stats = process_json_file(
                json_path,
                config,
                output_dir,
                input_dir,
                args.audio_format,
                args.dry_run,
                frame_ms=args.frame_ms,
            )

            all_segments.extend(segments)
            all_rejected.extend(rejected)
            stats["processed_files"] += 1
            stats["total_segments_in"] += input_stats["segments_in"]
            stats["total_duration_in"] += input_stats["duration_in"]
            stats["total_segments_out"] += len(segments)
            stats["total_rejected"] += len(rejected)

            for seg in segments:
                stats["total_duration_out"] += seg.get("duration", 0)
                speaker = seg.get("speaker", "UNKNOWN")
                stats["speakers"][speaker] = stats["speakers"].get(speaker, 0) + 1

            # Track rejection reasons
            for rej in rejected:
                reason = rej.get("reason", "unknown")
                stats["rejection_reasons"][reason] = (
                    stats["rejection_reasons"].get(reason, 0) + 1
                )

        except Exception as e:
            logger.exception(f"Error processing {json_path}: {e}")

    # Write JSONL manifest
    jsonl_path = output_dir / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    logger.info(f"Wrote manifest to {jsonl_path}")

    # Write rejected segments JSONL
    rejected_path = output_dir / "rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as f:
        for rej in all_rejected:
            f.write(json.dumps(rej, ensure_ascii=False) + "\n")

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
