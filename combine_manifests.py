#!/usr/bin/env python3
"""Strict manifest combiner (fail-fast).

Joins by audio_path and writes only:
- emotion_frames
- omni_text
- audio_tag_topk
- dialect_segment
- dialect_segment_name
- dialect_speaker_majority
- dialect_speaker_majority_name

Any missing key/mapping raises and stops the run.
"""

from __future__ import annotations

import argparse
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
        out[row["audio_path"]] = row["audio_tag_topk"]
    return out


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

    rows = 0
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

            merged["emotion_frames"] = emotion_by_audio[akey]
            merged["omni_text"] = omni_by_audio[akey]
            merged["audio_tag_topk"] = tags_by_audio[akey]

            try:
                merged.update(dialect_by_audio[akey])
            except KeyError:
                merged.update(dialect_by_source_speaker[(merged["source_audio"], merged["speaker"])])

            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"Wrote: {output_path}")
    print(f"Rows processed: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
