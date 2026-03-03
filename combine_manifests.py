#!/usr/bin/env python3
"""Manifest combiner.

Joins by audio_path and writes only:
- emotion_frames
- omni_text
- audio_tag_topk
- audio_tag_frames
- dialect_segment
- dialect_segment_name
- dialect_speaker_majority
- dialect_speaker_majority_name

Default mode: strict fail-fast.
Optional mode: skip incomplete rows and write a CSV report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine manifest JSONL files.")
    parser.add_argument("manifest", type=Path, help="Base manifest JSONL path")
    parser.add_argument("emotion", type=Path, help="Emotion JSONL path")
    parser.add_argument("omni", type=Path, help="Omni translation JSONL path")
    parser.add_argument("dialect", type=Path, help="Dialect JSONL path")
    parser.add_argument("tagged", type=Path, help="Audio tags JSONL path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("manifest_combined.jsonl"),
        help="Output JSONL path (default: manifest_combined.jsonl)",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip rows with missing data instead of failing.",
    )
    parser.add_argument(
        "--missing-report-csv",
        type=Path,
        default=Path("manifest_combined_missing.csv"),
        help="CSV report path for skipped rows (used with --skip-incomplete).",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_num}")
            yield row


def resolve_output_path(manifest_path: Path, output_arg: Path) -> Path:
    if output_arg.is_absolute():
        return output_arg
    return manifest_path.parent / output_arg


def resolve_report_path(manifest_path: Path, report_arg: Path) -> Path:
    if report_arg.is_absolute():
        return report_arg
    return manifest_path.parent / report_arg


def build_emotion_index(path: Path) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in read_jsonl(path):
        out[row["audio_path"]] = row["emotion_frames"]
    return out


def build_omni_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        out[row["audio_path"]] = row["omni_text"]
    return out


def build_tag_index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        payload: dict[str, Any] = {}
        if "audio_tag_topk" in row and _is_present(row["audio_tag_topk"]):
            payload["audio_tag_topk"] = row["audio_tag_topk"]
        if "audio_tag_frames" in row and _is_present(row["audio_tag_frames"]):
            payload["audio_tag_frames"] = row["audio_tag_frames"]
        out[row["audio_path"]] = payload
    return out


def _is_present(value: Any) -> bool:
    return value is not None


def _missing_csv_row(
    *,
    line_num: int,
    audio_path: str,
    source_audio: str,
    speaker: str,
    missing: list[str],
) -> dict[str, str]:
    return {
        "manifest_line": str(line_num),
        "audio_path": audio_path,
        "source_audio": source_audio,
        "speaker": speaker,
        "missing_count": str(len(missing)),
        "missing_fields": ";".join(missing),
    }


def build_dialect_indexes(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_audio: dict[str, dict[str, Any]] = {}
    by_source_speaker: dict[tuple[str, str], dict[str, Any]] = {}

    for row in read_jsonl(path):
        # Sentence-level dialect rows (preferred).
        if "audio_path" in row:
            by_audio[row["audio_path"]] = {
                "dialect_segment": row["dialect_segment"],
                "dialect_segment_name": row["dialect_segment_name"],
                "dialect_speaker_majority": row["dialect_speaker_majority"],
                "dialect_speaker_majority_name": row["dialect_speaker_majority_name"],
            }
            continue

        # Speaker-level rows fallback.
        payload = {
            "dialect_segment": row["speaker_dialect"],
            "dialect_segment_name": row["speaker_dialect_name"],
            "dialect_speaker_majority": row["speaker_dialect"],
            "dialect_speaker_majority_name": row["speaker_dialect_name"],
        }
        by_source_speaker[(row["source_audio"], row["speaker"])] = payload

    return by_audio, by_source_speaker


def main() -> int:
    args = parse_args()

    for path in (args.manifest, args.emotion, args.omni, args.dialect, args.tagged):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    emotion_by_audio = build_emotion_index(args.emotion)
    omni_by_audio = build_omni_index(args.omni)
    tags_by_audio = build_tag_index(args.tagged)
    dialect_by_audio, dialect_by_source_speaker = build_dialect_indexes(args.dialect)

    output_path = resolve_output_path(args.manifest, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = resolve_report_path(args.manifest, args.missing_report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    written_rows = 0
    skipped_rows = 0
    skipped_report: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in manifest {args.manifest}:{line_num}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in manifest {args.manifest}:{line_num}")

            rows += 1
            merged = dict(row)
            akey = merged["audio_path"]
            missing: list[str] = []

            emotion_val = emotion_by_audio.get(akey)
            if not _is_present(emotion_val):
                missing.append("emotion_frames")
            else:
                merged["emotion_frames"] = emotion_val

            omni_val = omni_by_audio.get(akey)
            if not _is_present(omni_val):
                missing.append("omni_text")
            else:
                merged["omni_text"] = omni_val

            tags_payload = tags_by_audio.get(akey, {})
            has_topk = _is_present(tags_payload.get("audio_tag_topk"))
            has_frames = _is_present(tags_payload.get("audio_tag_frames"))
            if not (has_topk or has_frames):
                missing.append("audio_tag_topk_or_audio_tag_frames")
            else:
                if has_topk:
                    merged["audio_tag_topk"] = tags_payload["audio_tag_topk"]
                if has_frames:
                    merged["audio_tag_frames"] = tags_payload["audio_tag_frames"]

            dialect_payload = dialect_by_audio.get(akey)
            if dialect_payload is None:
                source_audio = merged.get("source_audio")
                speaker = merged.get("speaker")
                if _is_present(source_audio) and _is_present(speaker):
                    dialect_payload = dialect_by_source_speaker.get((source_audio, speaker))
                else:
                    if not _is_present(source_audio):
                        missing.append("source_audio")
                    if not _is_present(speaker):
                        missing.append("speaker")

            if dialect_payload is None:
                missing.extend(
                    [
                        "dialect_segment",
                        "dialect_segment_name",
                        "dialect_speaker_majority",
                        "dialect_speaker_majority_name",
                    ]
                )
            else:
                for key in (
                    "dialect_segment",
                    "dialect_segment_name",
                    "dialect_speaker_majority",
                    "dialect_speaker_majority_name",
                ):
                    value = dialect_payload.get(key)
                    if not _is_present(value):
                        missing.append(key)
                    else:
                        merged[key] = value

            if missing:
                # Always drop rows that have neither legacy nor framewise tags.
                # This keeps output manifests compatible with downstream browser tooling.
                force_skip = "audio_tag_topk_or_audio_tag_frames" in missing
                if (not args.skip_incomplete) and (not force_skip):
                    raise KeyError(
                        f"Missing data for audio_path='{akey}' at manifest line {line_num}: {', '.join(missing)}"
                    )
                skipped_rows += 1
                skipped_report.append(
                    _missing_csv_row(
                        line_num=line_num,
                        audio_path=str(akey),
                        source_audio=str(merged.get("source_audio", "")),
                        speaker=str(merged.get("speaker", "")),
                        missing=missing,
                    )
                )
                continue

            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written_rows += 1

    if args.skip_incomplete:
        with report_path.open("w", encoding="utf-8", newline="") as rep:
            writer = csv.DictWriter(
                rep,
                fieldnames=[
                    "manifest_line",
                    "audio_path",
                    "source_audio",
                    "speaker",
                    "missing_count",
                    "missing_fields",
                ],
            )
            writer.writeheader()
            for record in skipped_report:
                writer.writerow(record)
        print(f"Missing report CSV: {report_path}")
        print(f"Rows skipped (incomplete): {skipped_rows}")

    print(f"Wrote: {output_path}")
    print(f"Rows processed: {rows}")
    print(f"Rows written: {written_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
