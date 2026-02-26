#!/usr/bin/env python3
"""Simple manifest combiner for processed podcast data."""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any


def extract_string_list_from_module(module_path: Path, var_name: str) -> list[str]:
    if not module_path.exists():
        return []
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except Exception:
        return []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != var_name:
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            continue
        out: list[str] = []
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
        return out
    return []


def load_allowed_non_speech_tags() -> set[str]:
    apr_path = Path(__file__).resolve().with_name("audio_pattern_recognition.py")
    speech = extract_string_list_from_module(apr_path, "SPEECH_LABELS")
    human = extract_string_list_from_module(apr_path, "HUMAN_VOICE_LABELS")
    respiratory = extract_string_list_from_module(apr_path, "RESPIRATORY_LABELS")
    speech_labels = [x.strip().lower() for x in speech if isinstance(x, str) and x.strip()]
    non_speech_labels = [x.strip().lower() for x in (human + respiratory) if isinstance(x, str) and x.strip()]
    return set(speech_labels + non_speech_labels)


ALLOWED_NON_SPEECH_TAGS = load_allowed_non_speech_tags()
SPEECH_TAGS = {
    x.strip().lower()
    for x in extract_string_list_from_module(
        Path(__file__).resolve().with_name("audio_pattern_recognition.py"), "SPEECH_LABELS"
    )
    if isinstance(x, str) and x.strip()
}


def is_allowed_non_speech_label(label: str) -> bool:
    return label.strip().lower() in ALLOWED_NON_SPEECH_TAGS


def is_speech_label(label: str) -> bool:
    return label.strip().lower() in SPEECH_TAGS


def top3_allowed_from_scores(scores: dict[str, Any]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for label, score in scores.items():
        if not isinstance(label, str):
            continue
        score_f = as_float(score)
        if score_f is None:
            continue
        if not is_allowed_non_speech_label(label):
            continue
        pairs.append((label, score_f))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:3]


def build_non_speech_top3_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_events: list[dict[str, Any]] = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        start = as_float(frame.get("start"))
        end = as_float(frame.get("end"))
        tags = frame.get("audio_tags")
        if start is None or end is None or not isinstance(tags, dict):
            continue

        all_pairs = []
        for label, score in tags.items():
            if not isinstance(label, str):
                continue
            score_f = as_float(score)
            if score_f is None:
                continue
            all_pairs.append((label, score_f))
        if not all_pairs:
            continue

        top3 = top3_allowed_from_scores(tags)
        if not top3:
            continue

        display_label, display_score = top3[0]
        for lbl, prob in top3:
            if not is_speech_label(lbl):
                display_label, display_score = lbl, prob
                break

        raw_events.append(
            {
                "start": min(start, end),
                "end": max(start, end),
                "label": display_label,
                "score": display_score,
                "top_label": top3[0][0],
                "_scores": {lbl: prob for lbl, prob in top3},
            }
        )

    if not raw_events:
        return []

    raw_events.sort(key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []
    for event in raw_events:
        if (
            merged
            and event["label"] == merged[-1]["label"]
            and event["start"] <= merged[-1]["end"] + 0.05
        ):
            prev = merged[-1]
            prev["end"] = max(prev["end"], event["end"])
            prev["score"] = max(float(prev.get("score", 0.0)), float(event.get("score", 0.0)))
            prev_scores = prev.get("_scores", {})
            cur_scores = event.get("_scores", {})
            for lbl, prob in cur_scores.items():
                old = as_float(prev_scores.get(lbl))
                prev_scores[lbl] = prob if old is None else max(old, prob)
        else:
            merged.append(event)

    final_events: list[dict[str, Any]] = []
    for event in merged:
        score_map = event.get("_scores")
        if not isinstance(score_map, dict):
            continue
        top3 = top3_allowed_from_scores(score_map)
        if not top3:
            continue
        final_events.append(
            {
                "start": round(float(event["start"]), 3),
                "end": round(float(event["end"]), 3),
                "label": event["label"],
                "score": round(float(event["score"]), 4),
                "top_label": top3[0][0],
                "top3": [[lbl, round(float(prob), 4)] for lbl, prob in top3],
            }
        )
    return final_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine manifest.jsonl with emotion/tagged/dialect/omni manifests."
    )
    parser.add_argument("processed_dir", type=Path, help="Folder that contains manifest*.jsonl")
    parser.add_argument("--base", default="manifest.jsonl", help="Base manifest file")
    parser.add_argument("--output", default="manifest_combined.jsonl", help="Combined output file")
    parser.add_argument(
        "--sample-output",
        default="manifest_combined.sample.jsonl",
        help="Random sample output file",
    )
    parser.add_argument("--sample-size", type=int, default=120, help="Random sample size")
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


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def audio_key(row: dict[str, Any]) -> str | None:
    value = row.get("audio_path")
    if isinstance(value, str) and value:
        return value
    return None


def collapse_timeline(items: list[dict[str, Any]], merge_gap: float = 0.05) -> list[dict[str, Any]]:
    cleaned = []
    for item in items:
        start = as_float(item.get("start"))
        end = as_float(item.get("end"))
        label = item.get("label")
        if start is None or end is None or not isinstance(label, str) or not label.strip():
            continue
        if end < start:
            start, end = end, start
        out = {"start": round(start, 3), "end": round(end, 3), "label": label.strip()}
        score = as_float(item.get("score"))
        if score is not None:
            out["score"] = round(score, 4)
        cleaned.append(out)

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    merged = [cleaned[0]]
    for item in cleaned[1:]:
        prev = merged[-1]
        if item["label"] == prev["label"] and item["start"] <= prev["end"] + merge_gap:
            prev["end"] = max(prev["end"], item["end"])
            prev_score = as_float(prev.get("score"))
            item_score = as_float(item.get("score"))
            if item_score is not None:
                prev["score"] = item_score if prev_score is None else round(max(prev_score, item_score), 4)
        else:
            merged.append(item)
    return merged


def tag_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(row.get("audio_tags_source"), str):
        out["audio_tags_source"] = row["audio_tags_source"]
    if isinstance(row.get("audio_tags"), dict):
        out["audio_tags"] = row["audio_tags"]

    frames = row.get("audio_tag_frames")
    timeline = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            label = frame.get("top_label")
            score = None
            tags = frame.get("audio_tags")
            if isinstance(tags, dict) and isinstance(label, str):
                score = tags.get(label)
            timeline.append(
                {"start": frame.get("start"), "end": frame.get("end"), "label": label, "score": score}
            )
    collapsed = collapse_timeline(timeline)
    if collapsed:
        out["audio_tag_timeline"] = collapsed

    if isinstance(frames, list):
        top3_frames = build_non_speech_top3_frames(frames)
        if top3_frames:
            out["audio_tag_top3_frames"] = top3_frames
    return out


def emotion_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(row.get("emotion_tags_source"), str):
        out["emotion_tags_source"] = row["emotion_tags_source"]

    frames = row.get("emotion_frames")
    timeline = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            emotion = frame.get("emotion")
            if not isinstance(emotion, dict):
                continue
            timeline.append(
                {
                    "start": frame.get("start"),
                    "end": frame.get("end"),
                    "label": emotion.get("label"),
                    "score": emotion.get("confidence"),
                }
            )
    collapsed = collapse_timeline(timeline)
    if collapsed:
        out["emotion_timeline"] = collapsed
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


def load_index(path: Path, extractor) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = audio_key(row)
        if not key:
            continue
        payload = extractor(row)
        if payload:
            index[key] = payload
    return index


def load_omni_index(processed_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    omni_files = sorted(
        p for p in processed_dir.glob("manifest.omni*.jsonl") if ".config." not in p.name
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


def preferred_omni_variant(variants: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    if not variants:
        return None
    for variant in sorted(variants.keys(), key=lambda x: (0 if x == "default" else 1, x)):
        payload = variants.get(variant)
        if isinstance(payload, dict):
            return variant, payload
    return None


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


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

    tagged = load_index(processed_dir / "manifest.tagged.jsonl", tag_payload)
    emotion = load_index(processed_dir / "manifest.emotion.jsonl", emotion_payload)
    dialect = load_index(processed_dir / "manifest_with_speaker_dialect.jsonl", dialect_payload)
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
