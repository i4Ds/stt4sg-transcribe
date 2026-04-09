#!/usr/bin/env python3
"""Merge a sentence_ch JSONL keyed by audio_path back into a manifest JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sentence_ch predictions into a base manifest JSONL."
    )
    parser.add_argument("manifest", type=Path, help="Base manifest JSONL.")
    parser.add_argument("sentence_ch", type=Path, help="JSONL with audio_path + sentence_ch.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output merged JSONL path.",
    )
    parser.add_argument(
        "--audio-field",
        default="audio_path",
        help="Shared key field used to match rows.",
    )
    parser.add_argument(
        "--sentence-field",
        default="sentence_ch",
        help="Field name to merge from the sentence_ch JSONL.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_num}")
            yield payload


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if not args.sentence_ch.exists():
        raise FileNotFoundError(f"Missing sentence_ch JSONL: {args.sentence_ch}")

    sentence_by_audio: dict[str, Any] = {}
    for row in iter_jsonl(args.sentence_ch):
        audio_path = row.get(args.audio_field)
        sentence = row.get(args.sentence_field)
        if isinstance(audio_path, str) and audio_path and isinstance(sentence, str):
            sentence_by_audio[audio_path] = sentence

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    matched = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(args.manifest):
            processed += 1
            audio_path = row.get(args.audio_field)
            if isinstance(audio_path, str):
                sentence = sentence_by_audio.get(audio_path)
                if isinstance(sentence, str):
                    row[args.sentence_field] = sentence
                    matched += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output}")
    print(f"Rows processed: {processed}")
    print(f"Rows matched: {matched}")
    print(f"Rows missing sentence_ch: {processed - matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
