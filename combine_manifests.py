#!/usr/bin/env python3
"""Combine processed manifests into one raw merged JSONL.

This step performs no training-oriented filtering. It keeps the base manifest
rows and attaches any matching annotation payloads that can be found:

- `emotion_frames`
- `audio_tag_frames`
- `dialect`
- `dialect_code`
- `dialect_name`
- `dialect_source`

Downstream scripts can then derive filtered fields such as canonical `tags`,
inline-tagged `text`, or sentence-level `emotion`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TAG_DISPLAY_MIN_START_SECONDS = 2.0

DEFAULT_TAG_MIN_DURATION_S = {
    "<speech>": 0.15,
    "<laugh>": 0.30,
    "<chuckle>": 0.25,
    "<sigh>": 0.12,
    "<cough>": 0.08,
    "<sniffle>": 0.08,
    "<groan>": 0.15,
    "<yawn>": 0.20,
    "<gasp>": 0.10,
}

TAG_GROUP_LAUGHTER = "Laughter"
TAG_GROUP_CHUCKLE = "Chuckle/Giggle"
TAG_GROUP_BREATHING = "Breathing"
TAG_GROUP_SIGH = "Sigh"

GROUP_MIN_PROB = {
    TAG_GROUP_SIGH: 0.4,
    TAG_GROUP_LAUGHTER: 0.45,
    TAG_GROUP_CHUCKLE: 0.3,
    TAG_GROUP_BREATHING: 0.4,
}

GROUP_MIN_DURATION_S = {
    TAG_GROUP_SIGH: DEFAULT_TAG_MIN_DURATION_S["<sigh>"],
    TAG_GROUP_LAUGHTER: 1.2,
    TAG_GROUP_CHUCKLE: DEFAULT_TAG_MIN_DURATION_S["<chuckle>"],
    TAG_GROUP_BREATHING: 0.0,
}

CANONICAL_TAG_ORDER = [
    "<speech>",
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine processed manifest JSONL files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help=(
            "Either: manifest emotion dialect tagged "
            "or legacy: manifest emotion omni dialect tagged"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("manifest_combined.jsonl"),
        help="Output JSONL path (default: manifest_combined.jsonl).",
    )
    args = parser.parse_args(argv)

    if len(args.inputs) == 4:
        manifest, emotion, dialect, tagged = args.inputs
        omni = None
    elif len(args.inputs) == 5:
        manifest, emotion, omni, dialect, tagged = args.inputs
    else:
        parser.error(
            "Expected 4 inputs (manifest emotion dialect tagged) "
            "or 5 inputs for legacy mode (manifest emotion omni dialect tagged)."
        )

    args.manifest = Path(manifest)
    args.emotion = Path(emotion)
    args.omni = Path(omni) if omni else None
    args.dialect = Path(dialect)
    args.tagged = Path(tagged)
    return args


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
    return output_arg if output_arg.is_absolute() else manifest_path.parent / output_arg


def resolve_report_path(manifest_path: Path, report_arg: Path) -> Path:
    return report_arg if report_arg.is_absolute() else manifest_path.parent / report_arg


def _is_present(value: Any) -> bool:
    return value is not None and value != ""


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def _path_suffixes(value: str) -> list[str]:
    path = Path(value)
    parts = path.parts
    out = [str(path)]
    for depth in (1, 2, 3, 4):
        if len(parts) >= depth:
            out.append("/".join(parts[-depth:]))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_audio_index(rows: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for audio_path, payload in rows.items():
        for alias in _path_suffixes(audio_path):
            out.setdefault(alias, payload)
    return out


def _lookup_audio(index: dict[str, Any], audio_path: str) -> Any:
    for alias in _path_suffixes(audio_path):
        payload = index.get(alias)
        if payload is not None:
            return payload
    return None


def build_emotion_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        audio_path = row.get("audio_path")
        frames = row.get("emotion_frames")
        if isinstance(audio_path, str) and isinstance(frames, list):
            out[audio_path] = frames
    return out


def build_tag_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        audio_path = row.get("audio_path")
        frames = row.get("audio_tag_frames")
        if isinstance(audio_path, str) and isinstance(frames, list):
            out[audio_path] = frames
    return out


def build_dialect_indexes(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_audio: dict[str, dict[str, Any]] = {}
    by_source_speaker: dict[tuple[str, str], dict[str, Any]] = {}

    for row in read_jsonl(path):
        if "audio_path" in row and isinstance(row.get("audio_path"), str):
            by_audio[row["audio_path"]] = {
                "dialect_code": row.get("dialect_segment")
                or row.get("dialect_speaker_majority"),
                "dialect_name": row.get("dialect_segment_name")
                or row.get("dialect_speaker_majority_name"),
                "dialect_source": "segment",
            }
            continue

        source_audio = row.get("source_audio")
        speaker = row.get("speaker")
        if not isinstance(source_audio, str) or not isinstance(speaker, str):
            continue
        by_source_speaker[(source_audio, speaker)] = {
            "dialect_code": row.get("speaker_dialect"),
            "dialect_name": row.get("speaker_dialect_name"),
            "dialect_source": "speaker",
        }

    return by_audio, by_source_speaker


def _normalize_tag_label(label: str) -> str | None:
    raw = label.strip().lower()
    if not raw:
        return None
    if raw in CANONICAL_TAG_ORDER:
        return raw
    if raw in {
        "speech",
        "male speech, man speaking",
        "female speech, woman speaking",
        "child speech, kid speaking",
        "conversation",
        "narration, monologue",
        "babbling",
        "speech synthesizer",
        "shout",
        "screaming",
        "whispering",
        "singing",
        "humming",
    }:
        return "<speech>"
    if raw in {"laughter", "baby laughter", "belly laugh"}:
        return "<laugh>"
    if raw in {"giggle", "snicker", "chuckle, chortle"}:
        return "<laugh>"
    if raw in {"sigh"}:
        return "<sigh>"
    if raw in {"cough", "throat clearing"}:
        return "<cough>"
    if raw in {"sniff", "sneeze"}:
        return "<sniffle>"
    if raw in {"groan", "grunt", "wail, moan", "crying, sobbing"}:
        return "<groan>"
    if raw in {"gasp", "pant", "snort", "wheeze", "breathing", "snoring"}:
        return "<gasp>"
    return None


def _tag_group_from_raw_label(label: str) -> str | None:
    raw = label.strip().lower()
    if raw in {"laughter", "baby laughter", "belly laugh"}:
        return TAG_GROUP_LAUGHTER
    if raw in {"snicker", "chuckle, chortle", "giggle"}:
        return TAG_GROUP_CHUCKLE
    if raw == "sigh":
        return TAG_GROUP_SIGH
    if raw in {
        "gasp",
        "pant",
        "snort",
        "wheeze",
        "breathing",
        "snoring",
        "cough",
        "throat clearing",
        "sniff",
        "sneeze",
        "groan",
        "grunt",
        "wail, moan",
        "crying, sobbing",
    }:
        return TAG_GROUP_BREATHING
    return None


def _merge_group_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: (x["start"], x["end"]))
    merged = [dict(rows[0])]
    for row in rows[1:]:
        prev = merged[-1]
        if row["start"] <= prev["end"] + 0.05:
            prev["end"] = max(prev["end"], row["end"])
            prev["score"] = max(prev["score"], row["score"])
            prev_raw = prev.setdefault("raw_scores", {})
            for label, score in row.get("raw_scores", {}).items():
                old = _as_float(prev_raw.get(label))
                prev_raw[label] = score if old is None else max(old, score)
        else:
            merged.append(dict(row))
    return merged


def _finalize_group_event(group: str, row: dict[str, Any]) -> dict[str, Any] | None:
    start = _as_float(row.get("start"))
    end = _as_float(row.get("end"))
    score = _as_float(row.get("score"))
    raw_scores = row.get("raw_scores")
    if start is None or end is None or score is None or not isinstance(raw_scores, dict):
        return None
    if start < TAG_DISPLAY_MIN_START_SECONDS:
        return None
    if score < GROUP_MIN_PROB[group]:
        return None
    if (end - start) < GROUP_MIN_DURATION_S[group]:
        return None

    raw_top = sorted(
        (
            (label, float(prob))
            for label, prob in raw_scores.items()
            if isinstance(label, str) and _as_float(prob) is not None
        ),
        key=lambda x: (-x[1], x[0]),
    )
    if not raw_top:
        return None

    if group in {TAG_GROUP_LAUGHTER, TAG_GROUP_CHUCKLE}:
        canonical = "<laugh>"
    elif group == TAG_GROUP_SIGH:
        canonical = "<sigh>"
    else:
        canonical = _normalize_tag_label(raw_top[0][0])

    if canonical is None or canonical == "<speech>":
        return None

    min_duration = DEFAULT_TAG_MIN_DURATION_S.get(canonical, 0.0)
    if (end - start) < min_duration:
        return None

    return {
        "from": round(start, 3),
        "to": round(end, 3),
        "tag": canonical,
        "score": round(score, 4),
    }


def _merge_canonical_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    events = sorted(events, key=lambda x: (x["from"], x["to"], x["tag"]))
    merged = [dict(events[0])]
    for event in events[1:]:
        prev = merged[-1]
        if event["tag"] == prev["tag"] and event["from"] <= prev["to"] + 0.05:
            prev["to"] = max(prev["to"], event["to"])
            prev["score"] = round(max(prev["score"], event["score"]), 4)
        else:
            merged.append(dict(event))
    return merged


def derive_tag_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        TAG_GROUP_SIGH: [],
        TAG_GROUP_LAUGHTER: [],
        TAG_GROUP_CHUCKLE: [],
        TAG_GROUP_BREATHING: [],
    }

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        start = _as_float(frame.get("start"))
        end = _as_float(frame.get("end"))
        tags = frame.get("audio_tags")
        if start is None or end is None or not isinstance(tags, dict):
            continue
        if end < start:
            start, end = end, start

        group_scores: dict[str, float] = {}
        group_raw_scores: dict[str, dict[str, float]] = {}
        for raw_label, raw_prob in tags.items():
            if not isinstance(raw_label, str):
                continue
            prob = _as_float(raw_prob)
            if prob is None:
                continue
            group = _tag_group_from_raw_label(raw_label)
            if group is None:
                continue
            old = group_scores.get(group)
            group_scores[group] = prob if old is None else max(old, prob)
            current = group_raw_scores.setdefault(group, {})
            prev_prob = current.get(raw_label)
            current[raw_label] = prob if prev_prob is None else max(prev_prob, prob)

        for group, score in group_scores.items():
            grouped[group].append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "score": float(score),
                    "raw_scores": group_raw_scores.get(group, {}),
                }
            )

    events: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        for row in _merge_group_events(rows):
            event = _finalize_group_event(group, row)
            if event is not None:
                events.append(event)

    return _merge_canonical_events(events)


def derive_overall_emotion(
    frames: list[dict[str, Any]],
) -> tuple[str, float, float | None] | None:
    duration_by_label: dict[str, float] = {}
    score_weighted_sum: dict[str, float] = {}
    score_weighted_dur: dict[str, float] = {}
    total = 0.0

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        start = _as_float(frame.get("start"))
        end = _as_float(frame.get("end"))
        emotion = frame.get("emotion")
        if start is None or end is None or not isinstance(emotion, dict):
            continue
        label = emotion.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        label = label.strip()
        total += dur
        duration_by_label[label] = duration_by_label.get(label, 0.0) + dur
        score = _as_float(emotion.get("confidence"))
        if score is not None:
            score_weighted_sum[label] = score_weighted_sum.get(label, 0.0) + score * dur
            score_weighted_dur[label] = score_weighted_dur.get(label, 0.0) + dur

    if not duration_by_label or total <= 0:
        return None

    label = max(duration_by_label, key=duration_by_label.get)
    ratio = duration_by_label[label] / total
    score = None
    if score_weighted_dur.get(label, 0.0) > 0:
        score = score_weighted_sum[label] / score_weighted_dur[label]
    return label, ratio, score


def resolve_dialect_payload(
    row: dict[str, Any],
    *,
    by_audio: dict[str, dict[str, Any]],
    by_audio_alias: dict[str, dict[str, Any]],
    by_source_speaker: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    audio_path = row.get("audio_path")
    if isinstance(audio_path, str):
        payload = by_audio.get(audio_path) or _lookup_audio(by_audio_alias, audio_path)
        if payload is not None:
            return payload

    source_audio = row.get("source_audio")
    speaker = row.get("speaker")
    if isinstance(source_audio, str) and isinstance(speaker, str):
        payload = by_source_speaker.get((source_audio, speaker))
        if payload is not None:
            return payload

        source_name = Path(source_audio).name
        for (candidate_source, candidate_speaker), payload in by_source_speaker.items():
            if candidate_speaker == speaker and Path(candidate_source).name == source_name:
                return payload

    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in (args.manifest, args.emotion, args.dialect, args.tagged):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    emotion_by_audio = build_emotion_index(args.emotion)
    emotion_by_audio_alias = _build_audio_index(emotion_by_audio)
    tags_by_audio = build_tag_index(args.tagged)
    tags_by_audio_alias = _build_audio_index(tags_by_audio)
    dialect_by_audio, dialect_by_source_speaker = build_dialect_indexes(args.dialect)
    dialect_by_audio_alias = _build_audio_index(dialect_by_audio)

    output_path = resolve_output_path(args.manifest, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    written_rows = 0

    with args.manifest.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line_num, line in enumerate(infile, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in manifest {args.manifest}:{line_num}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in manifest {args.manifest}:{line_num}")

            rows += 1
            merged = dict(row)

            audio_path = merged.get("audio_path")

            emotion_frames = None
            if isinstance(audio_path, str):
                emotion_frames = emotion_by_audio.get(audio_path) or _lookup_audio(
                    emotion_by_audio_alias, audio_path
                )
            if isinstance(emotion_frames, list) and emotion_frames:
                merged["emotion_frames"] = emotion_frames

            tag_frames = None
            if isinstance(audio_path, str):
                tag_frames = tags_by_audio.get(audio_path) or _lookup_audio(
                    tags_by_audio_alias, audio_path
                )
            if isinstance(tag_frames, list) and tag_frames:
                merged["audio_tag_frames"] = tag_frames

            dialect_payload = resolve_dialect_payload(
                merged,
                by_audio=dialect_by_audio,
                by_audio_alias=dialect_by_audio_alias,
                by_source_speaker=dialect_by_source_speaker,
            )
            if dialect_payload is not None:
                dialect_code = dialect_payload.get("dialect_code")
                dialect_name = dialect_payload.get("dialect_name")
                if _is_present(dialect_code):
                    merged["dialect_code"] = dialect_code
                if _is_present(dialect_name):
                    merged["dialect"] = dialect_name
                    merged["dialect_name"] = dialect_name
                source = dialect_payload.get("dialect_source")
                if _is_present(source):
                    merged["dialect_source"] = source

            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"Wrote: {output_path}")
    print(f"Rows processed: {rows}")
    print(f"Rows written: {written_rows}")
    if args.omni is not None:
        print(f"Legacy omni input ignored: {args.omni}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
