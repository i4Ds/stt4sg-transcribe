#!/usr/bin/env python3
"""Merge SER and APR JSONL outputs back into the base manifest JSONL."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_num, json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping %s line %d: %s", path, line_num, exc)


def _audio_key(entry: Dict) -> Optional[str]:
    for k in ("audio_path", "audio_filepath", "path", "audio"):
        value = entry.get(k)
        if value:
            return str(value)
    return None


def _index_by_audio_path(path: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for _, entry in _iter_jsonl(path):
        key = _audio_key(entry)
        if not key:
            continue
        if key in index:
            logger.warning("Duplicate key '%s' in %s; last value wins", key, path)
        index[key] = entry
    return index


def _copy_selected(source: Dict, target: Dict, fields: Iterable[str]) -> None:
    for field in fields:
        if field in source:
            target[field] = source[field]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge SER and APR JSONL outputs.")
    parser.add_argument("manifest", type=Path, help="Base manifest JSONL")
    parser.add_argument("--ser", type=Path, required=True, help="SER output JSONL")
    parser.add_argument("--apr", type=Path, required=True, help="APR output JSONL")
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Merged output JSONL"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    for p in (args.manifest, args.ser, args.apr):
        if not p.exists():
            logger.error("File not found: %s", p)
            return 1

    ser_map = _index_by_audio_path(args.ser)
    apr_map = _index_by_audio_path(args.apr)

    matched_ser = 0
    matched_apr = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as outfile:
        for _, base in _iter_jsonl(args.manifest):
            total += 1
            key = _audio_key(base)

            if key and key in ser_map:
                _copy_selected(
                    ser_map[key],
                    base,
                    ("emotion_tagging", "emotion", "emotion_frames"),
                )
                matched_ser += 1

            if key and key in apr_map:
                _copy_selected(
                    apr_map[key],
                    base,
                    (
                        "audio_tagging",
                        "audio_tags_source",
                        "audio_tags",
                        "audio_tag_frames",
                        "audio_tag_frames_raw",
                    ),
                )
                matched_apr += 1

            outfile.write(json.dumps(base, ensure_ascii=False) + "\n")

    logger.info("Merged output written to %s", args.output)
    logger.info("Manifest rows: %d", total)
    logger.info("Rows matched with SER: %d", matched_ser)
    logger.info("Rows matched with APR: %d", matched_apr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
