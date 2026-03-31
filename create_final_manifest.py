#!/usr/bin/env python3
"""Create the final training manifest from a raw combined JSONL.

This step:
- derives sentence-level emotion from `emotion_frames`
- derives filtered canonical non-speech `tags` from `audio_tag_frames`
- injects tags into `text` using word timings when available
- emits a compact output schema:
  - audio_path
  - base_audio_path
  - text
  - emotion
  - dialect_tag
  - tags
  - dnsmos_sig
  - dnsmos_bak
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from combine_manifests import (
    _missing_csv_row,
    derive_overall_emotion,
    derive_tag_events,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final training manifest JSONL.")
    parser.add_argument("input", type=Path, help="Combined raw manifest JSONL.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("manifest_final.jsonl"),
        help="Output JSONL path (default: manifest_final.jsonl).",
    )
    parser.add_argument(
        "--missing-report-csv",
        type=Path,
        default=Path("manifest_final_missing.csv"),
        help="CSV report path for skipped rows.",
    )
    return parser.parse_args(argv)


def resolve_output_path(input_path: Path, output_arg: Path) -> Path:
    return output_arg if output_arg.is_absolute() else input_path.parent / output_arg


def resolve_report_path(input_path: Path, report_arg: Path) -> Path:
    return report_arg if report_arg.is_absolute() else input_path.parent / report_arg


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_dnsmos(row: dict[str, Any]) -> tuple[float | None, float | None]:
    source_metrics = row.get("source_metrics")
    if not isinstance(source_metrics, dict):
        return None, None
    sig = _as_float(source_metrics.get("dnsmos_sig"))
    bak = _as_float(source_metrics.get("dnsmos_bak"))
    return sig, bak


def _extract_dialect_tag(row: dict[str, Any]) -> str | None:
    for key in ("dialect_tag", "dialect", "dialect_name"):
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def _extract_base_audio_path(row: dict[str, Any]) -> str | None:
    value = row.get("source_audio")
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _normalized_word_tokens(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        token = word.get("word")
        if not isinstance(token, str):
            continue
        token = token.strip()
        if not token:
            continue
        tokens.append(
            {
                "word": token,
                "start": _as_float(word.get("start")),
                "end": _as_float(word.get("end")),
            }
        )
    return tokens


def _insertion_index(tokens: list[dict[str, Any]], event: dict[str, Any]) -> int:
    event_from = _as_float(event.get("from"))
    event_to = _as_float(event.get("to"))
    if event_from is None:
        return len(tokens)
    if event_to is None:
        event_to = event_from
    event_mid = (event_from + event_to) / 2.0

    for idx, token in enumerate(tokens):
        start = token.get("start")
        if start is not None and start >= event_mid:
            return idx

    for idx, token in enumerate(tokens):
        end = token.get("end")
        if end is not None and end >= event_from:
            return idx + 1

    return len(tokens)


def inject_tags_into_text(
    text: str,
    words: list[dict[str, Any]] | None,
    tags: list[dict[str, Any]],
) -> str:
    if not tags:
        return text.strip()

    normalized_tags = sorted(
        (
            tag
            for tag in tags
            if isinstance(tag, dict) and isinstance(tag.get("tag"), str)
        ),
        key=lambda x: (
            _as_float(x.get("from")) if _as_float(x.get("from")) is not None else float("inf"),
            _as_float(x.get("to")) if _as_float(x.get("to")) is not None else float("inf"),
            str(x.get("tag")),
        ),
    )
    if not normalized_tags:
        return text.strip()

    if not isinstance(words, list):
        suffix = " ".join(tag["tag"] for tag in normalized_tags)
        return f"{text.strip()} {suffix}".strip()

    tokens = _normalized_word_tokens(words)
    if not tokens:
        suffix = " ".join(tag["tag"] for tag in normalized_tags)
        return f"{text.strip()} {suffix}".strip()

    inserts: dict[int, list[str]] = {}
    for tag in normalized_tags:
        idx = _insertion_index(tokens, tag)
        inserts.setdefault(idx, []).append(tag["tag"])

    pieces: list[str] = []
    for idx, token in enumerate(tokens):
        pieces.extend(inserts.get(idx, []))
        pieces.append(token["word"])
    pieces.extend(inserts.get(len(tokens), []))
    return " ".join(pieces).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    output_path = resolve_output_path(args.input, args.output)
    report_path = resolve_report_path(args.input, args.missing_report_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    skipped = 0
    skipped_report: list[dict[str, str]] = []

    with args.input.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line_num, line in enumerate(infile, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {args.input}:{line_num}")

            processed += 1
            missing: list[str] = []

            audio_path = row.get("audio_path")
            text = row.get("text")
            if not isinstance(audio_path, str) or not audio_path:
                missing.append("audio_path")
            if not isinstance(text, str) or not text.strip():
                missing.append("text")

            emotion_frames = row.get("emotion_frames")
            overall = (
                derive_overall_emotion(emotion_frames)
                if isinstance(emotion_frames, list) and emotion_frames
                else None
            )
            if overall is None:
                missing.append("overall_emotion")

            tag_frames = row.get("audio_tag_frames")
            tags = (
                derive_tag_events(tag_frames)
                if isinstance(tag_frames, list) and tag_frames
                else []
            )

            if missing:
                skipped += 1
                skipped_report.append(
                    _missing_csv_row(
                        line_num=line_num,
                        audio_path=str(audio_path or ""),
                        source_audio=str(row.get("source_audio", "")),
                        speaker=str(row.get("speaker", "")),
                        missing=missing,
                    )
                )
                continue

            emotion_label, _, _ = overall
            final_text = inject_tags_into_text(text, row.get("words"), tags)
            dialect_tag = _extract_dialect_tag(row)
            base_audio_path = _extract_base_audio_path(row)
            dnsmos_sig, dnsmos_bak = _extract_dnsmos(row)
            payload = {
                "audio_path": audio_path,
                "text": final_text,
                "emotion": emotion_label.upper(),
                "tags": tags,
            }
            if base_audio_path is not None:
                payload["base_audio_path"] = base_audio_path
            if dialect_tag is not None:
                payload["dialect_tag"] = dialect_tag
            if dnsmos_sig is not None:
                payload["dnsmos_sig"] = round(dnsmos_sig, 4)
            if dnsmos_bak is not None:
                payload["dnsmos_bak"] = round(dnsmos_bak, 4)
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

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

    print(f"Wrote: {output_path}")
    print(f"Rows processed: {processed}")
    print(f"Rows written: {written}")
    print(f"Rows skipped: {skipped}")
    print(f"Missing report CSV: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
