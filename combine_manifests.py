#!/usr/bin/env python3
"""Lean combiner for processed podcast manifests.

Merges rows by `audio_path` from:
- base manifest.jsonl
- manifest.tagged.sed.h200.fullclip.jsonl (time-based tags via audio_tag_topk)
- manifest.emotion.jsonl (time-based emotions via emotion_frames)
- manifest_with_speaker_dialect.jsonl (per-sentence + speaker-majority dialect)
- manifest.omni*.jsonl (omni transcription)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine processed JSONL manifests.")
    parser.add_argument(
        "processed_dir", type=Path, help="Folder that contains manifest*.jsonl"
    )
    parser.add_argument("--base", default="manifest.jsonl", help="Base manifest file")
    parser.add_argument(
        "--output", default="manifest_combined.jsonl", help="Combined output file"
    )
    parser.add_argument(
        "--sample-output",
        default="manifest_combined.sample.jsonl",
        help="Random sample output file",
    )
    parser.add_argument(
        "--sample-size", type=int, default=120, help="Random sample size"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def audio_key(row: dict[str, Any]) -> str | None:
    value = row.get("audio_path")
    if isinstance(value, str) and value:
        return value
    return None


def load_index(path: Path, extractor: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = audio_key(row)
        if not key:
            continue
        payload = extractor(row)
        if payload:
            index[key] = payload
    return index


def tag_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(row.get("audio_tags_source"), str):
        out["audio_tags_source"] = row["audio_tags_source"]
    if isinstance(row.get("audio_tag_topk"), dict):
        out["audio_tag_topk"] = row["audio_tag_topk"]
    return out


def emotion_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(row.get("emotion_tags_source"), str):
        out["emotion_tags_source"] = row["emotion_tags_source"]
    if isinstance(row.get("emotion_frames"), list):
        out["emotion_frames"] = row["emotion_frames"]
    return out


def dialect_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "dialect_segment",
        "dialect_segment_name",
        "dialect_speaker_majority",
        "dialect_speaker_majority_name",
    ):
        if key in row:
            out[key] = row[key]
    return out


def omni_variant_name(path: Path) -> str:
    name = path.name
    if name.startswith("manifest.omni") and name.endswith(".jsonl"):
        middle = name[len("manifest.omni") : -len(".jsonl")].lstrip(".")
        return middle or "default"
    return path.stem


def omni_payload(row: dict[str, Any], source_manifest: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("omni_text", "dialect", "de_text", "ch_text"):
        if key in row:
            out[key] = row[key]
    if out:
        out["source_manifest"] = source_manifest
    return out


def load_omni_index(processed_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    omni_files = sorted(
        p
        for p in processed_dir.glob("manifest.omni*.jsonl")
        if ".config." not in p.name
    )
    by_audio: dict[str, dict[str, dict[str, Any]]] = {}
    for path in omni_files:
        variant = omni_variant_name(path)
        for row in read_jsonl(path):
            key = audio_key(row)
            if not key:
                continue
            payload = omni_payload(row, source_manifest=path.name)
            if not payload:
                continue
            by_audio.setdefault(key, {})[variant] = payload
    return by_audio


def preferred_omni_variant(
    variants: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    if not variants:
        return None
    for variant in sorted(
        variants.keys(), key=lambda x: (0 if x == "default" else 1, x)
    ):
        payload = variants.get(variant)
        if isinstance(payload, dict):
            return variant, payload
    return None


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    processed_dir = args.processed_dir.expanduser().resolve()
    if not processed_dir.is_dir():
        print(f"Error: not a directory: {processed_dir}")
        return 1

    base_manifest = resolve_path(processed_dir, args.base)
    output_manifest = resolve_path(processed_dir, args.output)
    sample_manifest = resolve_path(processed_dir, args.sample_output)

    if not base_manifest.exists():
        print(f"Error: base manifest not found: {base_manifest}")
        return 1

    tagged = load_index(
        processed_dir / "manifest.tagged.sed.h200.fullclip.jsonl",
        tag_payload,
    )
    emotion = load_index(processed_dir / "manifest.emotion.jsonl", emotion_payload)
    dialect = load_index(
        processed_dir / "manifest_with_speaker_dialect.jsonl", dialect_payload
    )
    omni = load_omni_index(processed_dir)

    sample_size = max(0, int(args.sample_size))
    sample_rows: list[dict[str, Any]] = []
    total = 0

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if sample_size > 0:
        sample_manifest.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", encoding="utf-8") as out:
        for row in read_jsonl(base_manifest):
            total += 1
            merged = dict(row)
            key = audio_key(merged)

            if key:
                if key in tagged:
                    merged.update(tagged[key])
                if key in emotion:
                    merged.update(emotion[key])
                if key in dialect:
                    merged.update(dialect[key])
                if key in omni:
                    variants = omni[key]
                    merged["omni_variants"] = variants
                    preferred = preferred_omni_variant(variants)
                    if preferred:
                        variant_name, payload = preferred
                        merged["omni_variant"] = variant_name
                        if isinstance(payload.get("omni_text"), str):
                            merged["omni_text"] = payload["omni_text"]
                        if isinstance(payload.get("dialect"), str):
                            merged["omni_dialect"] = payload["dialect"]

            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

            if sample_size > 0:
                if len(sample_rows) < sample_size:
                    sample_rows.append(merged)
                else:
                    r = random.randint(0, total - 1)
                    if r < sample_size:
                        sample_rows[r] = merged

    if sample_size > 0:
        with sample_manifest.open("w", encoding="utf-8") as f:
            for row in sample_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {output_manifest}")
    if sample_size > 0:
        print(f"Wrote sample: {sample_manifest} ({len(sample_rows)} rows)")
    print(f"Rows processed: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
