#!/usr/bin/env python3
"""
Batch transcription script.

Scans a folder recursively, transcribes all audio files using the pipeline,
and writes JSON output with full metrics (purity, logprob, alignment scores, etc.).

No segments are dropped - all data is preserved with quality metrics for filtering.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pipeline import TranscriptionConfig, TranscriptionPipeline

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger("batch_transcribe")


def normalize_extensions(exts: str) -> List[str]:
    """Normalize extension string to list of lowercase extensions with dots."""
    result = []
    for raw in exts.split(","):
        e = raw.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = f".{e}"
        result.append(e)
    return result


def iter_audio_files(root: Path, extensions: List[str]) -> Iterable[Path]:
    """Recursively iterate over audio files in a directory."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def process_file(
    audio_path: Path,
    pipeline: TranscriptionPipeline,
    output_path: Path,
    srt_output_path: Optional[Path] = None,
    save_logs: bool = True,
) -> Dict:
    """
    Process a single audio file through the pipeline.

    Returns:
        Full result dict with transcription, alignment, diarization, and metrics.
    """
    result = pipeline.transcribe(
        audio_path, output_path=srt_output_path, save_logs=save_logs
    )

    # Build output with all metrics preserved
    output = {
        "audio_file": str(audio_path),
        "run_id": result.get("run_id"),
        "config": result.get("config"),
        "transcription": {
            "language": result["transcription"]["language"],
            "language_probability": result["transcription"]["language_probability"],
            "duration": result["transcription"]["duration"],
        },
        "segments": [],
        "statistics": {},
    }

    # Process final segments with all metrics
    segments = result.get("final_segments", [])
    speakers = {}
    total_purity_weighted = 0.0
    total_duration = 0.0

    for seg in segments:
        seg_duration = seg.get("end", 0) - seg.get("start", 0)
        total_duration += seg_duration

        purity = seg.get("purity", 1.0)
        total_purity_weighted += purity * seg_duration

        speaker = seg.get("speaker")
        if speaker:
            speakers.setdefault(speaker, {"duration": 0, "count": 0})
            speakers[speaker]["duration"] += seg_duration
            speakers[speaker]["count"] += 1

        output["segments"].append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "duration": round(seg_duration, 4),
                "text": seg.get("text", ""),
                "speaker": speaker,
                "purity": seg.get("purity"),
                "coverage": seg.get("coverage"),
                "speaker_overlaps": seg.get("speaker_overlaps"),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                # Include word-level data if available
                "words": seg.get("words"),
            }
        )

    # Calculate statistics
    output["statistics"] = {
        "num_segments": len(segments),
        "total_duration": round(total_duration, 2),
        "avg_purity": (
            round(total_purity_weighted / total_duration, 4)
            if total_duration > 0
            else 0
        ),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "high_purity_segments": sum(1 for s in segments if s.get("purity", 0) >= 0.95),
        "high_purity_duration": round(
            sum(
                s.get("end", 0) - s.get("start", 0)
                for s in segments
                if s.get("purity", 0) >= 0.95
            ),
            2,
        ),
    }

    # Add log directory if available
    if result.get("log_dir"):
        output["log_dir"] = result["log_dir"]

    # Save SRT path
    output["srt_path"] = result.get("srt_path")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcription with full metrics (purity, alignment, etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument("input_dir", help="Root folder to scan recursively")
    parser.add_argument(
        "--output-dir", help="Output folder for JSON files (default: next to audio)"
    )
    parser.add_argument(
        "--output-rel-root", help="Base path for relative output layout"
    )
    parser.add_argument("--extensions", default=".wav,.mp3,.flac,.m4a,.ogg,.opus,.aac")
    parser.add_argument("--limit", type=int, help="Max number of files to process")
    parser.add_argument(
        "--dry-run", action="store_true", help="List files without processing"
    )
    parser.add_argument(
        "--skip-existing",
        "--skip-if-exist",
        dest="skip_existing",
        action="store_true",
        help="Skip if JSON output exists",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Show progress bar with ETA (requires tqdm)",
    )
    parser.add_argument(
        "--add_lock",
        action="store_true",
        help="Create lock file to prevent concurrent processing",
    )

    # Model
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model")
    parser.add_argument(
        "-l", "--language", help="Language code (auto-detect if omitted)"
    )
    parser.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe"
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, help="Use batched inference")
    parser.add_argument("--log-progress", action="store_true")

    # VAD
    parser.add_argument(
        "--no-vad", dest="use_vad", action="store_false", help="Disable VAD"
    )
    parser.add_argument("--vad-method", default="silero")
    parser.add_argument("--vad-params", help="JSON dict of VAD params")
    parser.add_argument("--vad-min-duration", type=float, default=0.5)
    parser.add_argument("--vad-merge-threshold", type=float, default=0.3)

    # Diarization
    parser.add_argument(
        "--diarization", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument("--diarization-method", default="pyannote")
    parser.add_argument("--diarization-params", help="JSON dict of diarization params")
    parser.add_argument("-n", "--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int)

    # Alignment
    parser.add_argument(
        "--no-alignment", action="store_true", help="Disable CTC alignment"
    )
    parser.add_argument("--alignment-model", help="Custom alignment model")

    # Device
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", choices=["float16", "float32", "int8"])

    # Output options
    parser.add_argument(
        "--no-srt", action="store_true", help="Don't generate SRT files"
    )
    parser.add_argument(
        "--no-logs", action="store_true", help="Don't save detailed log files"
    )
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="Only write SRT files (skip JSON output)",
    )
    parser.add_argument(
        "--srt-in-place",
        action="store_true",
        help="Write SRT next to each audio file",
    )

    # Auth
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input path not found: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rel_root = Path(args.output_rel_root) if args.output_rel_root else input_dir
    extensions = normalize_extensions(args.extensions)

    # Parse JSON params
    vad_params = json.loads(args.vad_params) if args.vad_params else None
    diar_params = (
        json.loads(args.diarization_params) if args.diarization_params else None
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Build config
    config = TranscriptionConfig(
        whisper_model=args.model,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        log_progress=args.log_progress,
        use_vad=args.use_vad,
        vad_method=args.vad_method,
        vad_params=vad_params,
        vad_min_duration=args.vad_min_duration,
        vad_merge_threshold=args.vad_merge_threshold,
        use_diarization=args.diarization,
        diarization_method=args.diarization_method,
        diarization_params=diar_params,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        use_alignment=not args.no_alignment,
        alignment_model=args.alignment_model,
        include_speaker_labels=args.diarization,
        hf_token=hf_token,
    )

    if args.device:
        config.device = args.device
    if args.compute_type:
        config.compute_type = args.compute_type
    elif config.device == "cpu":
        config.compute_type = "float32"

    # Find files
    files = sorted(iter_audio_files(input_dir, extensions))

    # Random shuffle for better load distribution if processing in parallel
    # across multiple machines or processes. Comment out if you want deterministic order.
    import random

    random.shuffle(files)

    if args.limit:
        files = files[: args.limit]

    logger.info(f"Found {len(files)} audio files under {input_dir}")

    if args.dry_run:
        for path in files:
            print(path)
        return

    # Create pipeline (reused for all files)
    pipeline = TranscriptionPipeline(config)

    errors: List[str] = []
    file_iter: Iterable[Path] = files
    if args.tqdm:
        if tqdm is None:
            logger.warning("tqdm not installed; progress bar disabled")
        else:
            file_iter = tqdm(files, desc="Transcribing", unit="file")

    for audio_path in file_iter:
        srt_output_path: Optional[Path] = None
        if args.srt_in_place:
            srt_output_path = audio_path.with_suffix(".srt")

        if args.srt_only:
            if args.skip_existing:
                default_srt_path = Path("outputs/srt") / f"{audio_path.stem}.srt"
                srt_path = srt_output_path or default_srt_path
                if srt_path.exists():
                    logger.info(f"Skipping existing: {srt_path}")
                    continue

            logger.info(f"Processing: {audio_path}")
            try:
                lock_path = None
                if args.add_lock:
                    lock_path = audio_path.with_suffix(".lock")
                    try:
                        # Atomic create; fail if already exists
                        with open(lock_path, "x", encoding="utf-8") as f:
                            f.write(f"pid:{os.getpid()}\nstarted:{time.time()}\n")
                    except FileExistsError:
                        logger.info(f"Skipping locked file: {audio_path}")
                        continue

                try:
                    result = pipeline.transcribe(
                        audio_path,
                        output_path=srt_output_path,
                        save_logs=not args.no_logs,
                    )
                    logger.info(f"Wrote: {result['srt_path']}")
                finally:
                    if args.add_lock and lock_path is not None and lock_path.exists():
                        try:
                            lock_path.unlink()
                        except Exception:
                            logger.exception(f"Failed to remove lock: {lock_path}")
            except Exception as exc:
                logger.exception(f"Failed: {audio_path}")
                errors.append(f"{audio_path}: {exc}")
            continue

        # Determine output path
        if output_dir:
            try:
                rel_path = audio_path.relative_to(rel_root)
            except ValueError:
                rel_path = Path(audio_path.name)
            json_path = output_dir / rel_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            json_path = audio_path.with_suffix(".json")

        if args.skip_existing and json_path.exists():
            logger.info(f"Skipping existing: {json_path}")
            continue

        logger.info(f"Processing: {audio_path}")
        try:
            result = process_file(
                audio_path,
                pipeline,
                json_path,
                srt_output_path=srt_output_path,
                save_logs=not args.no_logs,
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Wrote: {json_path}")
            logger.info(
                f"  Segments: {result['statistics']['num_segments']}, "
                f"Speakers: {result['statistics']['num_speakers']}, "
                f"Avg purity: {result['statistics']['avg_purity']:.2%}"
            )

        except Exception as exc:
            logger.exception(f"Failed: {audio_path}")
            errors.append(f"{audio_path}: {exc}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Processed {len(files) - len(errors)} / {len(files)} files")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
