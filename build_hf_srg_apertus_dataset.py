#!/usr/bin/env python3
"""Build and optionally upload the SRG_apertus speaker-chunk dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("build_hf_srg_apertus_dataset")

DEFAULT_ROOT = "/mnt/nas05/data01/vincenzo/SRG_apertus"
DEFAULT_OUTPUT = "outputs/hf_srg_apertus_data"
DEFAULT_REPO_ID = "i4ds/SRG_apertus_data"
DEFAULT_LANGUAGE = "de"
DEFAULT_MIN_SECONDS = 5.0
DEFAULT_MAX_SECONDS = 50.0
TARGET_SAMPLE_RATE = 16000

CONFIG_KEYS = [
    "whisper_model",
    "device",
    "compute_type",
    "use_vad",
    "use_diarization",
    "vad_method",
    "vad_params",
    "diarization_method",
    "diarization_params",
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "vad_min_duration",
    "vad_merge_threshold",
    "language",
    "task",
    "beam_size",
    "batch_size",
    "word_timestamps",
    "log_progress",
    "use_alignment",
    "alignment_model",
    "include_speaker_labels",
]

WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


@dataclass
class SegmentPiece:
    start: float
    end: float
    text: str
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SRG_apertus speaker-consistent 5s-50s chunks and optionally "
            "upload them to Hugging Face."
        )
    )
    parser.add_argument("root_dir", nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--token", default=None, help="HF token (or use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/upload a private dataset repo (default: true).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = WHITESPACE_RE.sub(" ", text).strip()
    return SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _join_texts(parts: list[str]) -> str:
    return _normalize_text(" ".join(part for part in parts if part))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _collect_series_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def _load_episode_info(series_dir: Path) -> dict[str, dict[str, str]]:
    info_path = series_dir / "episode_info.csv"
    if not info_path.exists():
        return {}

    episode_info: dict[str, dict[str, str]] = {}
    with info_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = (row.get("Episode ID") or "").strip()
            if episode_id:
                episode_info[episode_id] = row
                continue

            audio_path = (row.get("Audio Path") or "").strip()
            if audio_path:
                episode_info[Path(audio_path).stem] = row
    return episode_info


def _extract_processing_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        return {}
    return {key: config.get(key) for key in CONFIG_KEYS}


def _iter_valid_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        speaker = _normalize_text(str(raw_segment.get("speaker") or ""))
        text = _normalize_text(str(raw_segment.get("text") or ""))
        start = _safe_float(raw_segment.get("start"))
        end = _safe_float(raw_segment.get("end"))
        if not speaker or not text or start is None or end is None or end <= start:
            continue
        cleaned.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "words": raw_segment.get("words"),
            }
        )
    return cleaned


def _split_segment_by_words(segment: dict[str, Any], max_seconds: float) -> list[SegmentPiece]:
    words = segment.get("words")
    if not isinstance(words, list) or not words:
        return [
            SegmentPiece(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                speaker=segment["speaker"],
            )
        ]

    pieces: list[SegmentPiece] = []
    current_words: list[tuple[float, float, str]] = []

    for raw_word in words:
        if not isinstance(raw_word, dict):
            continue
        word_text = _normalize_text(str(raw_word.get("word") or ""))
        start = _safe_float(raw_word.get("start"))
        end = _safe_float(raw_word.get("end"))
        if not word_text or start is None or end is None or end <= start:
            continue

        if not current_words:
            current_words.append((start, end, word_text))
            continue

        current_duration = end - current_words[0][0]
        if current_duration <= max_seconds:
            current_words.append((start, end, word_text))
            continue

        pieces.append(
            SegmentPiece(
                start=current_words[0][0],
                end=current_words[-1][1],
                text=_join_texts([word[2] for word in current_words]),
                speaker=segment["speaker"],
            )
        )
        current_words = [(start, end, word_text)]

    if current_words:
        pieces.append(
            SegmentPiece(
                start=current_words[0][0],
                end=current_words[-1][1],
                text=_join_texts([word[2] for word in current_words]),
                speaker=segment["speaker"],
            )
        )

    if not pieces:
        return [
            SegmentPiece(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                speaker=segment["speaker"],
            )
        ]

    return pieces


def _segment_to_pieces(segment: dict[str, Any], max_seconds: float) -> list[SegmentPiece]:
    duration = segment["end"] - segment["start"]
    if duration <= max_seconds:
        return [
            SegmentPiece(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                speaker=segment["speaker"],
            )
        ]

    pieces = _split_segment_by_words(segment, max_seconds=max_seconds)
    oversized_without_words = [
        piece
        for piece in pieces
        if piece.duration > max_seconds + 1e-6 and piece.text == segment["text"]
    ]
    if oversized_without_words:
        LOGGER.warning(
            "Skipping oversized segment without usable word timestamps: %.2fs speaker=%s text=%r",
            duration,
            segment["speaker"],
            segment["text"][:120],
        )
        return []
    return pieces


def _build_same_speaker_runs(
    segments: list[dict[str, Any]], max_seconds: float
) -> list[list[SegmentPiece]]:
    runs: list[list[SegmentPiece]] = []
    current_run: list[SegmentPiece] = []
    current_speaker = ""

    for segment in segments:
        pieces = _segment_to_pieces(segment, max_seconds=max_seconds)
        for piece in pieces:
            if not current_run or piece.speaker == current_speaker:
                current_run.append(piece)
                current_speaker = piece.speaker
                continue

            runs.append(current_run)
            current_run = [piece]
            current_speaker = piece.speaker

    if current_run:
        runs.append(current_run)
    return runs


def _chunk_run(
    run: list[SegmentPiece], min_seconds: float, max_seconds: float
) -> list[list[SegmentPiece]]:
    if not run:
        return []

    raw_chunks: list[list[SegmentPiece]] = []
    current: list[SegmentPiece] = []

    for piece in run:
        if not current:
            current = [piece]
            continue

        projected_duration = piece.end - current[0].start
        if projected_duration <= max_seconds:
            current.append(piece)
            continue

        raw_chunks.append(current)
        current = [piece]

    if current:
        raw_chunks.append(current)

    final_chunks: list[list[SegmentPiece]] = []
    for index, chunk in enumerate(raw_chunks):
        duration = chunk[-1].end - chunk[0].start
        if duration >= min_seconds:
            final_chunks.append(chunk)
            continue

        if final_chunks:
            merged_duration = chunk[-1].end - final_chunks[-1][0].start
            if merged_duration <= max_seconds:
                final_chunks[-1].extend(chunk)
                continue

        if index + 1 < len(raw_chunks):
            next_chunk = raw_chunks[index + 1]
            merged_duration = next_chunk[-1].end - chunk[0].start
            if merged_duration <= max_seconds:
                next_chunk[:0] = chunk
                continue

        LOGGER.debug(
            "Dropping short speaker chunk %.2fs for %s", duration, chunk[0].speaker
        )

    return final_chunks


def _source_relative_path(root_dir: Path, audio_path: Path) -> str:
    return audio_path.relative_to(root_dir).as_posix()


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "unknown"


def _write_chunk_audio(
    waveform: Any,
    sample_rate: int,
    chunk_start: float,
    chunk_end: float,
    destination: Path,
) -> None:
    import torch
    import torchaudio

    start_frame = max(0, int(round(chunk_start * sample_rate)))
    end_frame = min(waveform.shape[-1], int(round(chunk_end * sample_rate)))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid chunk frame range: {start_frame} >= {end_frame}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_waveform = waveform[:, start_frame:end_frame].detach().cpu()
    if chunk_waveform.dtype != torch.float32:
        chunk_waveform = chunk_waveform.to(torch.float32)
    torchaudio.save(str(destination), chunk_waveform, sample_rate, format="wav")


def _build_readme(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    series_rows = summary["series"]
    processing = summary["processing"]

    lines = [
        "# SRG_apertus_data",
        "",
        "Speaker-consistent German audio chunks from SRG Apertus broadcasts.",
        "",
        "## Dataset Overview",
        "",
        "- One row per diarization-consistent audio chunk.",
        "- Each chunk contains only one speaker within one source file.",
        "- Chunk duration is constrained to 5s-50s.",
        "- `text` is the concatenated transcription for the full chunk.",
        "- `speaker_id` is only meaningful within a single source file.",
        "- Source metadata is joined from each series `episode_info.csv`.",
        "",
        "## Columns",
        "",
        "- `audio`: chunk audio",
        "- `language`: fixed language tag (`de`)",
        "- `text`: chunk transcript",
        "- `speaker_id`: per-file speaker label from diarization",
        "- `audio_file`: original source audio file name",
        "- `source_audio_path`: relative source path inside the dataset tree",
        "- `series`: broadcast series name",
        "- `date`: episode datetime from `episode_info.csv`",
        "- `episode_id`, `episode_name`",
        "- `chunk_index`, `chunk_start_seconds`, `chunk_end_seconds`, `chunk_duration_seconds`",
        "",
        "## Construction",
        "",
        f"- Source files processed: {totals['source_files']}",
        f"- Dataset rows: {totals['rows']}",
        f"- Total chunk hours: {totals['chunk_hours']:.2f}",
        f"- Min chunk length: {processing['min_seconds']:.1f}s",
        f"- Max chunk length: {processing['max_seconds']:.1f}s",
        "",
        "### Transcription And Diarization Settings",
        "",
        f"- Transcription model: `{processing['config'].get('whisper_model', '')}`",
        f"- Transcription device: `{processing['config'].get('device', '')}`",
        f"- Compute type: `{processing['config'].get('compute_type', '')}`",
        f"- Task: `{processing['config'].get('task', '')}`",
        f"- Beam size: `{processing['config'].get('beam_size', '')}`",
        f"- Word timestamps: `{processing['config'].get('word_timestamps', '')}`",
        f"- Alignment enabled: `{processing['config'].get('use_alignment', '')}`",
        f"- Alignment model override: `{processing['config'].get('alignment_model', '')}`",
        f"- VAD enabled: `{processing['config'].get('use_vad', '')}`",
        f"- VAD method: `{processing['config'].get('vad_method', '')}`",
        f"- VAD params: `{json.dumps(processing['config'].get('vad_params', {}), ensure_ascii=False)}`",
        f"- VAD min duration: `{processing['config'].get('vad_min_duration', '')}`",
        f"- VAD merge threshold: `{processing['config'].get('vad_merge_threshold', '')}`",
        f"- Diarization enabled: `{processing['config'].get('use_diarization', '')}`",
        f"- Diarization method: `{processing['config'].get('diarization_method', '')}`",
        f"- Diarization params override: `{processing['config'].get('diarization_params', '')}`",
        f"- Min speakers setting: `{processing['config'].get('min_speakers', '')}`",
        f"- Speaker labels included: `{processing['config'].get('include_speaker_labels', '')}`",
        "",
        "## Series Coverage",
        "",
        "| Series | Files | Rows | Hours |",
        "|---|---:|---:|---:|",
    ]

    for row in series_rows:
        lines.append(
            f"| {row['series']} | {row['source_files']} | {row['rows']} | {row['chunk_hours']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'ds = load_dataset("{summary["repo_id"]}", split="train")',
            "sample = ds[0]",
            'print(sample["text"])',
            'print(sample["speaker_id"], sample["audio_file"])',
            "```",
            "",
            "## Acknowledgements",
            "",
            "This dataset was prepared with the support of the Institute for Data Science at FHNW.",
            "",
            "## Notes",
            "",
            "- Speaker labels are local to each source file and are not global identities.",
            "- Short leftovers below the minimum duration may be dropped when they cannot be merged without exceeding 50s.",
            f"- Generated at: {summary['generated_at_utc']}",
        ]
    )

    return "\n".join(lines) + "\n"


def _upload_to_hub(
    rows_path: Path,
    repo_id: str,
    token: str | None,
    private: bool,
    readme_path: Path,
    summary_path: Path,
) -> None:
    try:
        from datasets import Audio, Dataset, Features, Value
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for upload. Install with: pip install datasets huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    features = Features(
        {
            "id": Value("string"),
            "audio": Value("string"),
            "language": Value("string"),
            "text": Value("string"),
            "speaker_id": Value("string"),
            "audio_file": Value("string"),
            "source_audio_path": Value("string"),
            "series": Value("string"),
            "date": Value("string"),
            "episode_id": Value("string"),
            "episode_name": Value("string"),
            "chunk_index": Value("int32"),
            "chunk_start_seconds": Value("float32"),
            "chunk_end_seconds": Value("float32"),
            "chunk_duration_seconds": Value("float32"),
        }
    )

    dataset = Dataset.from_json(str(rows_path), features=features)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    dataset.push_to_hub(repo_id=repo_id, private=private, token=token)
    LOGGER.info("Uploaded dataset rows to hf://datasets/%s", repo_id)

    for path, path_in_repo in [
        (readme_path, "README.md"),
        (summary_path, "summary.json"),
    ]:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    LOGGER.info("Uploaded README and summary to hf://datasets/%s", repo_id)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if args.min_seconds <= 0 or args.max_seconds <= 0:
        LOGGER.error("Chunk durations must be positive.")
        return 1
    if args.min_seconds > args.max_seconds:
        LOGGER.error("--min-seconds cannot be larger than --max-seconds.")
        return 1

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        LOGGER.error("Root folder does not exist: %s", root_dir)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    rows_path = output_dir / "dataset_rows.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        LOGGER.error("Missing audio dependencies: %s", exc)
        return 1

    canonical_config: dict[str, Any] | None = None
    config_mismatch_count = 0
    total_source_files = 0
    total_rows = 0
    total_chunk_seconds = 0.0
    skipped_files_no_segments = 0
    skipped_rows_short = 0
    series_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"source_files": 0, "rows": 0, "chunk_seconds": 0.0}
    )
    speaker_rows: Counter[str] = Counter()

    with rows_path.open("w", encoding="utf-8") as rows_handle:
        for series_dir in _collect_series_dirs(root_dir):
            episode_info = _load_episode_info(series_dir)
            json_paths = sorted(series_dir.glob("*.json"))

            for json_index, json_path in enumerate(json_paths, start=1):
                if args.max_files is not None and total_source_files >= args.max_files:
                    break

                source_payload = json.loads(json_path.read_text(encoding="utf-8"))
                source_config = _extract_processing_config(source_payload)
                if canonical_config is None:
                    canonical_config = source_config
                elif source_config != canonical_config:
                    config_mismatch_count += 1

                audio_path = series_dir / f"{json_path.stem}.mp3"
                if not audio_path.exists():
                    LOGGER.warning("Missing source audio for %s", json_path)
                    continue

                segments = _iter_valid_segments(source_payload)
                if not segments:
                    skipped_files_no_segments += 1
                    continue

                waveform, sample_rate = torchaudio.load(str(audio_path))
                if waveform.ndim != 2:
                    LOGGER.warning("Unexpected waveform shape for %s", audio_path)
                    continue

                waveform = waveform.mean(dim=0, keepdim=True)
                if sample_rate != TARGET_SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, TARGET_SAMPLE_RATE
                    )
                    sample_rate = TARGET_SAMPLE_RATE
                waveform = waveform.to(torch.float32)

                episode_meta = episode_info.get(json_path.stem, {})
                source_relative_path = _source_relative_path(root_dir, audio_path)
                same_speaker_runs = _build_same_speaker_runs(
                    segments, max_seconds=args.max_seconds
                )
                file_chunk_index = 0
                kept_any_chunk = False

                for run in same_speaker_runs:
                    for chunk in _chunk_run(
                        run,
                        min_seconds=args.min_seconds,
                        max_seconds=args.max_seconds,
                    ):
                        if args.max_rows is not None and total_rows >= args.max_rows:
                            break

                        chunk_start = chunk[0].start
                        chunk_end = chunk[-1].end
                        chunk_duration = round(chunk_end - chunk_start, 3)
                        if chunk_duration < args.min_seconds:
                            skipped_rows_short += 1
                            continue

                        speaker_id = chunk[0].speaker
                        text = _join_texts([piece.text for piece in chunk])
                        if not text:
                            skipped_rows_short += 1
                            continue

                        chunk_file_name = (
                            f"{json_path.stem}__{speaker_id}__chunk_{file_chunk_index:05d}.wav"
                        )
                        chunk_path = (
                            chunks_dir
                            / _sanitize_name(series_dir.name)
                            / json_path.stem
                            / chunk_file_name
                        )
                        _write_chunk_audio(
                            waveform=waveform,
                            sample_rate=sample_rate,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            destination=chunk_path,
                        )

                        row = {
                            "id": f"{series_dir.name}/{json_path.stem}/{speaker_id}/{file_chunk_index:05d}",
                            "audio": str(chunk_path),
                            "language": args.language,
                            "text": text,
                            "speaker_id": speaker_id,
                            "audio_file": audio_path.name,
                            "source_audio_path": source_relative_path,
                            "series": series_dir.name,
                            "date": (episode_meta.get("Date") or "").strip(),
                            "episode_id": json_path.stem,
                            "episode_name": (episode_meta.get("Episode Name") or "").strip(),
                            "chunk_index": file_chunk_index,
                            "chunk_start_seconds": chunk_start,
                            "chunk_end_seconds": chunk_end,
                            "chunk_duration_seconds": chunk_duration,
                        }
                        rows_handle.write(
                            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        )

                        total_rows += 1
                        total_chunk_seconds += chunk_duration
                        series_stats[series_dir.name]["rows"] += 1
                        series_stats[series_dir.name]["chunk_seconds"] += chunk_duration
                        speaker_rows[f"{json_path.stem}:{speaker_id}"] += 1
                        file_chunk_index += 1
                        kept_any_chunk = True

                    if args.max_rows is not None and total_rows >= args.max_rows:
                        break

                total_source_files += 1
                series_stats[series_dir.name]["source_files"] += 1

                if json_index % args.log_every == 0:
                    LOGGER.info(
                        "Processed %s files in %s (%s rows total)",
                        json_index,
                        series_dir.name,
                        total_rows,
                    )

                if not kept_any_chunk:
                    LOGGER.debug("No valid chunks kept for %s", json_path)

                if args.max_rows is not None and total_rows >= args.max_rows:
                    break

            if args.max_files is not None and total_source_files >= args.max_files:
                break
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

    summary = {
        "repo_id": args.repo_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "source_files": total_source_files,
            "rows": total_rows,
            "chunk_seconds": round(total_chunk_seconds, 3),
            "chunk_hours": round(total_chunk_seconds / 3600.0, 3),
            "unique_file_speakers": len(speaker_rows),
            "skipped_files_no_segments": skipped_files_no_segments,
            "skipped_rows_short": skipped_rows_short,
            "config_mismatch_count": config_mismatch_count,
        },
        "series": [
            {
                "series": series_name,
                "source_files": stats["source_files"],
                "rows": stats["rows"],
                "chunk_seconds": round(stats["chunk_seconds"], 3),
                "chunk_hours": round(stats["chunk_seconds"] / 3600.0, 3),
            }
            for series_name, stats in sorted(series_stats.items())
        ],
        "processing": {
            "language": args.language,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "config": canonical_config or {},
        },
    }

    _write_json(summary_path, summary)
    readme_path.write_text(_build_readme(summary), encoding="utf-8")

    LOGGER.info("Wrote rows: %s", rows_path)
    LOGGER.info("Wrote summary: %s", summary_path)
    LOGGER.info("Wrote README: %s", readme_path)

    if args.upload:
        _upload_to_hub(
            rows_path=rows_path,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            readme_path=readme_path,
            summary_path=summary_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
