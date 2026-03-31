#!/usr/bin/env python3
"""Quick Gradio browser for combined JSONL manifests with audio playback."""

from __future__ import annotations

import argparse
import ast
import html
import json
import random
from pathlib import Path
from typing import Any

import gradio as gr

from combine_manifests import (
    CANONICAL_TAG_ORDER as COMBINE_CANONICAL_TAG_ORDER,
    DEFAULT_TAG_MIN_DURATION_S as COMBINE_DEFAULT_TAG_MIN_DURATION_S,
    GROUP_MIN_DURATION_S as COMBINE_GROUP_MIN_DURATION_S,
    GROUP_MIN_PROB as COMBINE_GROUP_MIN_PROB,
    TAG_DISPLAY_MIN_START_SECONDS as COMBINE_TAG_DISPLAY_MIN_START_SECONDS,
    TAG_GROUP_BREATHING as COMBINE_TAG_GROUP_BREATHING,
    TAG_GROUP_CHUCKLE as COMBINE_TAG_GROUP_CHUCKLE,
    TAG_GROUP_LAUGHTER as COMBINE_TAG_GROUP_LAUGHTER,
    TAG_GROUP_SIGH as COMBINE_TAG_GROUP_SIGH,
    _normalize_tag_label as combine_normalize_tag_label,
    _tag_group_from_raw_label as combine_tag_group_from_raw_label,
)
from create_final_manifest import inject_tags_into_text

DEFAULT_MANIFEST = "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_combined_sliding_sample.jsonl"
TAG_DISPLAY_MIN_START_SECONDS = COMBINE_TAG_DISPLAY_MIN_START_SECONDS

RAW_JSON_KEYS = [
    "audio_path",
    "base_audio_path",
    "text",
    "tags",
    "dialect_tag",
    "dialect",
    "dialect_name",
    "dialect_code",
    "dialect_source",
    "start",
    "end",
    "duration",
    "speaker",
    "purity",
    "coverage",
    "avg_logprob",
    "speaker_overlaps",
    "non_main_time",
    "avg_word_score",
    "is_merged",
    "merge_count",
    "words",
    "emotion_frames",
    "audio_tag_frames",
    "source_metrics",
    "source_audio",
    "source_json",
]


def _select_raw_json_fields(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in RAW_JSON_KEYS:
        if key in record:
            out[key] = record[key]
    return out


def _extract_string_list_from_module(module_path: Path, var_name: str) -> list[str]:
    if not module_path.exists():
        return []
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except Exception:
        return []

    def _dedupe_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _eval_str_list(node: ast.AST, env: dict[str, list[str]]) -> list[str] | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            out: list[str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(elt.value)
                else:
                    return None
            return out

        if isinstance(node, ast.Name):
            value = env.get(node.id)
            return None if value is None else list(value)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _eval_str_list(node.left, env)
            right = _eval_str_list(node.right, env)
            if left is None or right is None:
                return None
            return left + right

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_dedupe_keep_order" and len(node.args) == 1:
                arg = _eval_str_list(node.args[0], env)
                if arg is None:
                    return None
                return _dedupe_keep_order(arg)
        return None

    env: dict[str, list[str]] = {}
    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None

        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                target_name = node.target.id
                value_node = node.value

        if target_name is None or value_node is None:
            continue

        value = _eval_str_list(value_node, env)
        if value is None:
            continue
        env[target_name] = value

    return env.get(var_name, [])


def _load_allowed_non_speech_tags() -> set[str]:
    apr_path = Path(__file__).resolve().with_name("audio_pattern_recognition.py")
    speech = _extract_string_list_from_module(apr_path, "SPEECH_LABELS")
    human = _extract_string_list_from_module(apr_path, "HUMAN_VOICE_LABELS")
    respiratory = _extract_string_list_from_module(apr_path, "RESPIRATORY_LABELS")
    return {
        label.strip().lower()
        for label in (speech + human + respiratory)
        if isinstance(label, str) and label.strip()
    }


ALLOWED_NON_SPEECH_TAGS = _load_allowed_non_speech_tags()
SPEECH_TAGS = {
    label.strip().lower()
    for label in _extract_string_list_from_module(
        Path(__file__).resolve().with_name("audio_pattern_recognition.py"),
        "SPEECH_LABELS",
    )
    if isinstance(label, str) and label.strip()
}


def _load_topk_label_vocab() -> list[str]:
    sed_path = Path(__file__).resolve().with_name("audio_pattern_recognition_sed.py")
    labels = _extract_string_list_from_module(sed_path, "NON_SPEECH_TARGET_LABELS")
    if not labels:
        human = _extract_string_list_from_module(sed_path, "HUMAN_VOICE_LABELS")
        respiratory = _extract_string_list_from_module(sed_path, "RESPIRATORY_LABELS")
        labels = human + respiratory
    if not labels:
        labels = _extract_string_list_from_module(sed_path, "TARGET_LABELS")
    return [x for x in labels if isinstance(x, str) and x.strip()]


TOPK_LABEL_VOCAB = _load_topk_label_vocab()

CANONICAL_TAG_ORDER = list(COMBINE_CANONICAL_TAG_ORDER)
DEFAULT_TAG_MIN_DURATION_S = dict(COMBINE_DEFAULT_TAG_MIN_DURATION_S)
DEFAULT_GROUP_MIN_PROB = dict(COMBINE_GROUP_MIN_PROB)
DEFAULT_GROUP_MIN_DURATION_S = dict(COMBINE_GROUP_MIN_DURATION_S)

AGG_TAG_SLIDERS: list[tuple[str, str]] = [
    ("<speech>", "Speech"),
    ("<laugh>", "Laughter"),
    ("<chuckle>", "Chuckle"),
    ("<sigh>", "Sigh"),
    ("<cough>", "Cough"),
    ("<sniffle>", "Sniffle"),
    ("<groan>", "Groan"),
    ("<yawn>", "Yawn"),
    ("<gasp>", "Gasp/Breathing"),
]

DISPLAY_TAG_LABEL = {
    "<speech>": "Speech",
    "<laugh>": "Laughter",
    "<chuckle>": "Chuckle",
    "<sigh>": "Sigh",
    "<cough>": "Cough",
    "<sniffle>": "Sniffle",
    "<groan>": "Groan",
    "<yawn>": "Yawn",
    "<gasp>": "Gasp/Breathing",
}

TAG_GROUP_LAUGHTER = COMBINE_TAG_GROUP_LAUGHTER
TAG_GROUP_CHUCKLE = COMBINE_TAG_GROUP_CHUCKLE
TAG_GROUP_BREATHING = COMBINE_TAG_GROUP_BREATHING
TAG_GROUP_SIGH = COMBINE_TAG_GROUP_SIGH


def _tag_group_from_raw_label(label: str) -> str | None:
    return combine_tag_group_from_raw_label(label)


def _pretty_tag_label(label: str) -> str:
    canonical = _normalize_tag_label(label)
    if canonical is None:
        return label
    return DISPLAY_TAG_LABEL.get(canonical, canonical)


def _normalize_tag_label(label: str) -> str | None:
    return combine_normalize_tag_label(label)


def _canonical_rank(tag: str) -> int:
    try:
        return CANONICAL_TAG_ORDER.index(tag)
    except ValueError:
        return len(CANONICAL_TAG_ORDER)


def _min_duration_for_tag(
    label: str,
    *,
    scale: float,
    floor: float,
    overrides: dict[str, float] | None = None,
) -> float:
    normalized = _normalize_tag_label(label)
    key = normalized if normalized is not None else label.strip().lower()
    base = DEFAULT_TAG_MIN_DURATION_S.get(key, 0.15)
    if overrides and key in overrides:
        base = float(overrides[key])
    return max(float(floor), float(base) * float(scale))


class ManifestStore:
    """In-memory store for fast random row sampling."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.eligible_row_ids: list[int] = []
        self.total_rows = 0
        self.total_eligible_rows = 0

    def build(self, *, require_annotations: bool = False) -> int:
        self.rows.clear()
        self.eligible_row_ids.clear()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    row_id = len(self.rows)
                    self.rows.append(row)
                    if (not require_annotations) or _row_has_emotion_and_tags(row):
                        self.eligible_row_ids.append(row_id)
        self.total_rows = len(self.rows)
        self.total_eligible_rows = len(self.eligible_row_ids)
        return self.total_rows

    def get_row(self, row_id: int) -> dict[str, Any]:
        if row_id < 0 or row_id >= self.total_rows:
            raise IndexError(f"row_id out of bounds: {row_id}")
        return self.rows[row_id]

    def random_row(self) -> tuple[int, dict[str, Any]] | None:
        if self.total_eligible_rows <= 0:
            return None
        row_id = random.choice(self.eligible_row_ids)
        return row_id, self.rows[row_id]


def _row_has_emotion_and_tags(record: dict[str, Any]) -> bool:
    has_emotion = isinstance(record.get("emotion_timeline"), list) and bool(
        record.get("emotion_timeline")
    )
    if not has_emotion:
        has_emotion = isinstance(record.get("emotion_frames"), list) and bool(
            record.get("emotion_frames")
        )

    has_tags = isinstance(record.get("audio_tag_top3_frames"), list) and bool(
        record.get("audio_tag_top3_frames")
    )
    if not has_tags:
        topk = record.get("audio_tag_topk")
        has_tags = (
            isinstance(topk, dict)
            and isinstance(topk.get("top_idx"), list)
            and bool(topk.get("top_idx"))
        )
    if not has_tags:
        has_tags = isinstance(record.get("audio_tag_events"), list) and bool(
            record.get("audio_tag_events")
        )
    if not has_tags:
        has_tags = isinstance(record.get("audio_tag_frames"), list) and bool(
            record.get("audio_tag_frames")
        )
    if not has_tags:
        has_tags = isinstance(record.get("audio_tag_timeline"), list) and bool(
            record.get("audio_tag_timeline")
        )
    return has_emotion and has_tags


def _derive_podcast_and_title(record: dict[str, Any]) -> tuple[str, str]:
    source_audio = record.get("source_audio")
    if isinstance(source_audio, str) and source_audio:
        source_path = Path(source_audio)
        return source_path.parent.name, source_path.stem

    audio_path = record.get("audio_path")
    if isinstance(audio_path, str) and audio_path:
        segment_path = Path(audio_path)
        title = segment_path.parent.name
        podcast = (
            segment_path.parent.parent.name if len(segment_path.parents) > 1 else ""
        )
        return podcast, title
    return "", ""


def _fmt_time(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "-"


def _fmt_clock(value: Any) -> str:
    sec = _as_float(value)
    if sec is None:
        return "-"
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collapse_timeline(
    items: list[dict[str, Any]],
    *,
    merge_gap: float = 0.05,
) -> list[dict[str, Any]]:
    cleaned = []
    for item in items:
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        label = item.get("label")
        if (
            start is None
            or end is None
            or not isinstance(label, str)
            or not label.strip()
        ):
            continue
        if end < start:
            start, end = end, start
        score = _as_float(item.get("score"))
        row: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label.strip(),
        }
        if score is not None:
            row["score"] = round(score, 4)
        cleaned.append(row)

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    merged = [cleaned[0]]
    for item in cleaned[1:]:
        prev = merged[-1]
        if item["label"] == prev["label"] and item["start"] <= prev["end"] + merge_gap:
            prev["end"] = max(prev["end"], item["end"])
            prev_score = _as_float(prev.get("score"))
            item_score = _as_float(item.get("score"))
            if item_score is not None:
                prev["score"] = (
                    item_score
                    if prev_score is None
                    else round(max(prev_score, item_score), 4)
                )
        else:
            merged.append(item)
    return merged


def _tag_timeline(record: dict[str, Any]) -> list[dict[str, Any]]:
    existing = record.get("audio_tag_timeline")
    if isinstance(existing, list):
        return _collapse_timeline(existing)

    frames = record.get("audio_tag_frames")
    if not isinstance(frames, list):
        return []

    timeline_rows = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        label = frame.get("top_label")
        score = None
        frame_tags = frame.get("audio_tags")
        if isinstance(frame_tags, dict) and isinstance(label, str):
            score = frame_tags.get(label)
        timeline_rows.append(
            {
                "start": frame.get("start"),
                "end": frame.get("end"),
                "label": label,
                "score": score,
            }
        )
    return _collapse_timeline(timeline_rows)


def _emotion_timeline(record: dict[str, Any]) -> list[dict[str, Any]]:
    existing = record.get("emotion_timeline")
    if isinstance(existing, list):
        return _collapse_timeline(existing)

    frames = record.get("emotion_frames")
    if not isinstance(frames, list):
        return []

    timeline_rows = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        emotion = frame.get("emotion")
        if not isinstance(emotion, dict):
            continue
        timeline_rows.append(
            {
                "start": frame.get("start"),
                "end": frame.get("end"),
                "label": emotion.get("label"),
                "score": emotion.get("confidence"),
            }
        )
    return _collapse_timeline(timeline_rows)


def _segment_duration(
    record: dict[str, Any], timelines: list[list[dict[str, Any]]]
) -> float:
    duration = _as_float(record.get("duration"))
    if duration is not None and duration > 0:
        return duration

    start = _as_float(record.get("start"))
    end = _as_float(record.get("end"))
    if start is not None and end is not None and end > start:
        return end - start

    max_end = 0.0
    for timeline in timelines:
        for row in timeline:
            row_end = _as_float(row.get("end"))
            if row_end is not None:
                max_end = max(max_end, row_end)
    return max_end


def _overall_emotion(
    emotion_timeline: list[dict[str, Any]],
) -> tuple[str, float, float | None] | None:
    if not emotion_timeline:
        return None
    duration_by_label: dict[str, float] = {}
    score_weighted_sum: dict[str, float] = {}
    score_weighted_dur: dict[str, float] = {}
    total = 0.0

    for row in emotion_timeline:
        label = row.get("label")
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        if not isinstance(label, str) or start is None or end is None:
            continue
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        total += dur
        duration_by_label[label] = duration_by_label.get(label, 0.0) + dur
        score = _as_float(row.get("score"))
        if score is not None:
            score_weighted_sum[label] = score_weighted_sum.get(label, 0.0) + score * dur
            score_weighted_dur[label] = score_weighted_dur.get(label, 0.0) + dur

    if not duration_by_label or total <= 0:
        return None
    label = max(duration_by_label, key=duration_by_label.get)
    ratio = duration_by_label[label] / total
    score = None
    if label in score_weighted_sum and score_weighted_dur.get(label, 0.0) > 0:
        score = score_weighted_sum[label] / score_weighted_dur[label]
    return label, ratio, score


def _is_allowed_tag(label: str) -> bool:
    return _normalize_tag_label(label) is not None


def _is_speech_tag(label: str) -> bool:
    return _normalize_tag_label(label) == "<speech>"


def _emotion_color(label: str) -> str:
    color_map = {
        "Happy": "#f5c542",
        "Sad": "#5aa0ff",
        "Neutral": "#9ca3af",
        "Angry": "#ef4444",
        "Disgusted": "#22c55e",
        "Fearful": "#a855f7",
        "Surprised": "#f97316",
    }
    return color_map.get(label, "#60a5fa")


def _tag_color(label: str) -> str:
    if label == TAG_GROUP_CHUCKLE:
        return "#fb923c"
    if label == TAG_GROUP_LAUGHTER:
        return "#f59e0b"
    if label == TAG_GROUP_SIGH:
        return "#22c55e"
    if label == TAG_GROUP_BREATHING:
        return "#3b82f6"

    canonical = _normalize_tag_label(label) or label
    fixed = {
        "<speech>": "#64748b",
        "<laugh>": "#f59e0b",
        "<chuckle>": "#f97316",
        "<sigh>": "#22c55e",
        "<cough>": "#ef4444",
        "<sniffle>": "#14b8a6",
        "<groan>": "#8b5cf6",
        "<yawn>": "#eab308",
        "<gasp>": "#3b82f6",
    }
    if canonical in fixed:
        return fixed[canonical]

    palette = [
        "#2563eb",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#d97706",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#0f766e",
        "#7c2d12",
    ]
    idx = sum(ord(c) for c in canonical) % len(palette)
    return palette[idx]


def _top3_allowed_from_scores(scores: dict[str, Any]) -> list[tuple[str, float]]:
    grouped: dict[str, float] = {}
    for label, score in scores.items():
        if not isinstance(label, str):
            continue
        score_f = _as_float(score)
        if score_f is None:
            continue
        canonical = _normalize_tag_label(label)
        if canonical is None:
            continue
        old = grouped.get(canonical)
        grouped[canonical] = score_f if old is None else max(old, score_f)

    pairs = sorted(grouped.items(), key=lambda x: (-x[1], _canonical_rank(x[0]), x[0]))
    return pairs[:3]


def _non_speech_tag_events_from_frames(
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_events: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        start = _as_float(frame.get("start"))
        end = _as_float(frame.get("end"))
        tags = frame.get("audio_tags")
        if start is None or end is None or not isinstance(tags, dict):
            continue
        all_pairs = []
        for label, score in tags.items():
            if not isinstance(label, str):
                continue
            score_f = _as_float(score)
            if score_f is None:
                continue
            all_pairs.append((label, score_f))
        if not all_pairs:
            continue
        allowed_top3 = _top3_allowed_from_scores(tags)
        if not allowed_top3:
            continue

        display_label, display_score = allowed_top3[0]
        for lbl, prob in allowed_top3:
            if not _is_speech_tag(lbl):
                display_label, display_score = lbl, prob
                break

        raw_events.append(
            {
                "start": min(start, end),
                "end": max(start, end),
                "label": display_label,
                "score": display_score,
                "top_label": allowed_top3[0][0],
                "_scores": {lbl: prob for lbl, prob in allowed_top3},
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
            prev_scores = prev.get("_scores", {})
            cur_scores = event.get("_scores", {})
            for lbl, prob in cur_scores.items():
                old = _as_float(prev_scores.get(lbl))
                prev_scores[lbl] = prob if old is None else max(old, prob)
            prev["score"] = max(
                float(prev.get("score", 0.0)), float(event.get("score", 0.0))
            )
        else:
            merged.append(event)

    final_events: list[dict[str, Any]] = []
    for event in merged:
        score_map = event.get("_scores")
        if not isinstance(score_map, dict):
            continue
        top3 = _top3_allowed_from_scores(score_map)
        if not top3:
            continue
        final_events.append(
            {
                "start": round(float(event["start"]), 3),
                "end": round(float(event["end"]), 3),
                "label": str(event["label"]),
                "score": round(float(event["score"]), 4),
                "top_label": top3[0][0],
                "top3": [(lbl, round(prob, 4)) for lbl, prob in top3],
            }
        )
    return final_events


def _load_tag_events(
    record: dict[str, Any], min_tag_prob: float = 0.5
) -> list[dict[str, Any]]:
    min_tag_prob = max(0.0, min(1.0, float(min_tag_prob)))
    topk = record.get("audio_tag_topk")
    if isinstance(topk, dict):
        idx_rows = topk.get("top_idx")
        prob_rows = topk.get("top_prob")
        if isinstance(idx_rows, list) and isinstance(prob_rows, list):
            row_count = min(len(idx_rows), len(prob_rows))
            duration = _as_float(record.get("duration"))
            if duration is None or duration <= 0:
                seg_start = _as_float(record.get("start"))
                seg_end = _as_float(record.get("end"))
                if (
                    seg_start is not None
                    and seg_end is not None
                    and seg_end > seg_start
                ):
                    duration = seg_end - seg_start
            if duration is None or duration <= 0:
                duration = float(max(row_count, 1)) * 0.02
            step = duration / float(max(row_count, 1))

            events = []
            for i in range(row_count):
                idx_row = idx_rows[i]
                prob_row = prob_rows[i]
                if not isinstance(idx_row, list) or not isinstance(prob_row, list):
                    continue
                raw_pairs: list[tuple[str, float]] = []
                ranked_pairs: list[tuple[str, float]] = []
                grouped: dict[str, float] = {}
                for raw_idx, raw_prob in zip(idx_row, prob_row):
                    if not isinstance(raw_idx, int):
                        continue
                    if raw_idx < 0 or raw_idx >= len(TOPK_LABEL_VOCAB):
                        continue
                    prob = _as_float(raw_prob)
                    if prob is None:
                        continue
                    if prob < min_tag_prob:
                        continue
                    raw_label = TOPK_LABEL_VOCAB[raw_idx]
                    canonical = _normalize_tag_label(raw_label)
                    if canonical is None:
                        continue
                    raw_pairs.append((raw_label, prob))
                    old = grouped.get(canonical)
                    grouped[canonical] = prob if old is None else max(old, prob)

                raw_pairs = sorted(raw_pairs, key=lambda x: (-x[1], x[0]))
                raw_top3 = raw_pairs[:3]
                ranked_pairs = sorted(
                    grouped.items(),
                    key=lambda x: (-x[1], _canonical_rank(x[0]), x[0]),
                )
                clean_top3 = ranked_pairs[:3]
                if not clean_top3:
                    continue
                display_label, display_score = clean_top3[0]
                for lbl, prob in clean_top3:
                    if not _is_speech_tag(lbl):
                        display_label, display_score = lbl, prob
                        break
                pretty_top3 = [
                    (_pretty_tag_label(lbl), prob) for lbl, prob in clean_top3
                ]
                events.append(
                    {
                        "start": round(float(i) * step, 3),
                        "end": round(float(i + 1) * step, 3),
                        "label": _pretty_tag_label(display_label),
                        "score": round(float(display_score), 4),
                        "top3": [
                            (lbl, round(float(prob), 4)) for lbl, prob in pretty_top3
                        ],
                        "raw_top3": [
                            (lbl, round(float(prob), 4)) for lbl, prob in raw_top3
                        ],
                    }
                )
            if events:
                return events

    if isinstance(record.get("audio_tag_events"), list):
        events = []
        for row in record["audio_tag_events"]:
            if not isinstance(row, dict):
                continue
            label = row.get("label")
            if not isinstance(label, str):
                continue
            score = _as_float(row.get("score_mean"))
            if score is None:
                score = _as_float(row.get("score_max"))
            if score is not None and score < min_tag_prob:
                continue
            score_out = round(float(score), 4) if score is not None else 0.0
            events.append(
                {
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "label": label.strip(),
                    "score": score_out,
                    "top3": [(label.strip(), score_out)],
                    "raw_top3": [(label.strip(), score_out)],
                }
            )
        if events:
            return events

    if isinstance(record.get("audio_tag_top3_frames"), list):
        events = []
        for row in record["audio_tag_top3_frames"]:
            if not isinstance(row, dict):
                continue
            top3 = row.get("top3")
            grouped: dict[str, float] = {}
            if isinstance(top3, list):
                for pair in top3:
                    if (
                        isinstance(pair, (list, tuple))
                        and len(pair) == 2
                        and isinstance(pair[0], str)
                    ):
                        prob = _as_float(pair[1])
                        if prob is None:
                            continue
                        if prob < min_tag_prob:
                            continue
                        label = pair[0].strip()
                        if not label:
                            continue
                        old = grouped.get(label)
                        grouped[label] = prob if old is None else max(old, prob)
            clean_top3 = sorted(
                grouped.items(),
                key=lambda x: (-x[1], x[0]),
            )[:3]
            if clean_top3:
                display_label, display_score = clean_top3[0]
                for lbl, prob in clean_top3:
                    if not _is_speech_tag(lbl):
                        display_label, display_score = lbl, prob
                        break
                events.append(
                    {
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "label": display_label,
                        "score": round(float(display_score), 4),
                        "top3": [
                            (lbl, round(float(prob), 4)) for lbl, prob in clean_top3[:3]
                        ],
                        "raw_top3": [
                            (lbl, round(float(prob), 4)) for lbl, prob in clean_top3[:3]
                        ],
                    }
                )
        if events:
            return events

    if isinstance(record.get("audio_tag_frames"), list):
        return _non_speech_tag_events_from_frames(record["audio_tag_frames"])

    # Fallback: use collapsed timeline if full frames are unavailable.
    fallback = []
    for row in _tag_timeline(record):
        label = str(row.get("label", ""))
        clean = label.strip()
        if not clean:
            continue
        score = round(float(row.get("score", 0.0) or 0.0), 4)
        if score < min_tag_prob:
            continue
        fallback.append(
            {
                "start": row.get("start"),
                "end": row.get("end"),
                "label": clean,
                "score": score,
                "top3": [(clean, score)],
                "raw_top3": [(clean, score)],
            }
        )
    if fallback:
        return fallback

    return []


def _merge_group_events(
    rows: list[dict[str, Any]], *, merge_gap: float = 0.05
) -> list[dict[str, Any]]:
    if not rows:
        return []

    cleaned = []
    for row in rows:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        label = row.get("label")
        score = _as_float(row.get("score"))
        raw_top3 = row.get("raw_top3")
        if (
            start is None
            or end is None
            or not isinstance(label, str)
            or score is None
            or not isinstance(raw_top3, list)
        ):
            continue
        if end < start:
            start, end = end, start
        cleaned.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "raw_label": row.get("raw_label"),
                "score": score,
                "raw_top3": raw_top3,
            }
        )

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []
    for row in cleaned:
        if (
            merged
            and row["label"] == merged[-1]["label"]
            and row["start"] <= merged[-1]["end"] + merge_gap
        ):
            prev = merged[-1]
            prev["end"] = max(prev["end"], row["end"])
            prev["_score_sum"] = float(prev.get("_score_sum", prev["score"])) + float(
                row["score"]
            )
            prev["_score_n"] = int(prev.get("_score_n", 1)) + 1
        else:
            merged.append(
                {
                    **row,
                    "_score_sum": float(row["score"]),
                    "_score_n": 1,
                }
            )

    out: list[dict[str, Any]] = []
    for row in merged:
        n = max(1, int(row.get("_score_n", 1)))
        avg_score = float(row.get("_score_sum", row["score"])) / n
        out.append(
            {
                "start": row["start"],
                "end": row["end"],
                "label": row["label"],
                "raw_label": row.get("raw_label"),
                "score": round(avg_score, 4),
                "raw_top3": row["raw_top3"],
            }
        )
    return out


def _group_tag_events(
    record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        TAG_GROUP_SIGH: [],
        TAG_GROUP_LAUGHTER: [],
        TAG_GROUP_CHUCKLE: [],
        TAG_GROUP_BREATHING: [],
    }

    frames = record.get("audio_tag_frames")
    if not isinstance(frames, list):
        return out
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
            prev_prob = _as_float(current.get(raw_label))
            current[raw_label] = prob if prev_prob is None else max(prev_prob, prob)

        for group, score in group_scores.items():
            raw_scores = group_raw_scores.get(group, {})
            raw_top3 = sorted(
                (
                    (label, float(prob))
                    for label, prob in raw_scores.items()
                    if isinstance(label, str) and _as_float(prob) is not None
                ),
                key=lambda x: (-x[1], x[0]),
            )[:3]
            raw_label = raw_top3[0][0] if raw_top3 else group
            out[group].append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "label": group,
                    "raw_label": raw_label,
                    "score": round(float(score), 4),
                    "raw_scores": raw_scores,
                    "raw_top3": [(label, round(prob, 4)) for label, prob in raw_top3],
                }
            )
    return out


def _cluster_group_rows(
    rows: list[dict[str, Any]], *, merge_gap: float = 0.05
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        score = _as_float(row.get("score"))
        raw_scores = row.get("raw_scores")
        if start is None or end is None or score is None or not isinstance(raw_scores, dict):
            continue
        if end < start:
            start, end = end, start
        cleaned.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(score, 4),
                "raw_scores": dict(raw_scores),
                "row": row,
            }
        )

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    clusters: list[dict[str, Any]] = []
    for row in cleaned:
        if not clusters or row["start"] > clusters[-1]["end"] + merge_gap:
            clusters.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "score": row["score"],
                    "raw_scores": dict(row["raw_scores"]),
                    "peak_row": row["row"],
                }
            )
            continue

        cluster = clusters[-1]
        cluster["end"] = max(cluster["end"], row["end"])
        cluster["score"] = max(float(cluster["score"]), row["score"])
        cluster_raw_scores = cluster["raw_scores"]
        for label, prob in row["raw_scores"].items():
            old = _as_float(cluster_raw_scores.get(label))
            cluster_raw_scores[label] = prob if old is None else max(old, prob)
        peak_score = _as_float(cluster["peak_row"].get("score"))
        if peak_score is None or row["score"] >= peak_score:
            cluster["peak_row"] = row["row"]
    return clusters


def _display_group_events(
    rows: list[dict[str, Any]],
    *,
    group: str,
    min_prob: float,
    min_duration_s: float,
    min_start_seconds: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        start = _as_float(row.get("start"))
        score = _as_float(row.get("score"))
        if start is None or score is None:
            continue
        if start < min_start_seconds:
            continue
        if score < min_prob:
            continue
        filtered.append(row)

    clusters = _cluster_group_rows(filtered)
    out: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_start = _as_float(cluster.get("start"))
        cluster_end = _as_float(cluster.get("end"))
        peak_row = cluster.get("peak_row")
        if cluster_start is None or cluster_end is None or not isinstance(peak_row, dict):
            continue
        if (cluster_end - cluster_start) < min_duration_s:
            continue

        if group in {TAG_GROUP_LAUGHTER, TAG_GROUP_CHUCKLE}:
            event_start = cluster_start
            event_end = cluster_end
            event_score = _as_float(cluster.get("score"))
            raw_scores = cluster.get("raw_scores")
        else:
            event_start = _as_float(peak_row.get("start"))
            event_end = _as_float(peak_row.get("end"))
            event_score = _as_float(peak_row.get("score"))
            raw_scores = peak_row.get("raw_scores")

        if (
            event_start is None
            or event_end is None
            or event_score is None
            or not isinstance(raw_scores, dict)
        ):
            continue

        raw_top3 = sorted(
            (
                (label, float(prob))
                for label, prob in raw_scores.items()
                if isinstance(label, str) and _as_float(prob) is not None
            ),
            key=lambda x: (-x[1], x[0]),
        )[:3]
        raw_label = raw_top3[0][0] if raw_top3 else group
        out.append(
            {
                "start": round(event_start, 3),
                "end": round(event_end, 3),
                "label": group,
                "raw_label": raw_label,
                "score": round(event_score, 4),
                "raw_scores": raw_scores,
                "raw_top3": [(label, round(prob, 4)) for label, prob in raw_top3],
            }
        )
    return out


def _merge_canonical_tag_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _derive_canonical_tag_events(
    record: dict[str, Any],
    *,
    min_start_seconds: float,
    sigh_min_prob: float,
    sigh_min_duration_s: float,
    laughter_min_prob: float,
    laughter_min_duration_s: float,
    chuckle_min_prob: float,
    chuckle_min_duration_s: float,
    breathing_min_prob: float,
    breathing_min_duration_s: float,
) -> list[dict[str, Any]]:
    grouped = _group_tag_events(record)
    events: list[dict[str, Any]] = []
    thresholds = {
        TAG_GROUP_SIGH: (float(sigh_min_prob), float(sigh_min_duration_s)),
        TAG_GROUP_LAUGHTER: (float(laughter_min_prob), float(laughter_min_duration_s)),
        TAG_GROUP_CHUCKLE: (float(chuckle_min_prob), float(chuckle_min_duration_s)),
        TAG_GROUP_BREATHING: (
            float(breathing_min_prob),
            float(breathing_min_duration_s),
        ),
    }

    for group, rows in grouped.items():
        min_prob, min_duration_s = thresholds[group]
        filtered = _display_group_events(
            rows,
            group=group,
            min_prob=min_prob,
            min_duration_s=min_duration_s,
            min_start_seconds=min_start_seconds,
        )
        for row in filtered:
            start = _as_float(row.get("start"))
            end = _as_float(row.get("end"))
            score = _as_float(row.get("score"))
            raw_top3 = row.get("raw_top3")
            if (
                start is None
                or end is None
                or score is None
                or not isinstance(raw_top3, list)
            ):
                continue
            if group in {TAG_GROUP_LAUGHTER, TAG_GROUP_CHUCKLE}:
                canonical = "<laugh>"
            elif group == TAG_GROUP_SIGH:
                canonical = "<sigh>"
            else:
                if not raw_top3:
                    continue
                first_label = raw_top3[0][0]
                if not isinstance(first_label, str):
                    continue
                canonical = _normalize_tag_label(first_label)

            if canonical is None or canonical == "<speech>":
                continue
            canonical_min_duration_s = DEFAULT_TAG_MIN_DURATION_S.get(canonical, 0.0)
            if (end - start) < canonical_min_duration_s:
                continue
            events.append(
                {
                    "from": round(start, 3),
                    "to": round(end, 3),
                    "tag": canonical,
                    "score": round(score, 4),
                }
            )

    return _merge_canonical_tag_events(events)


def _filter_group_events(
    rows: list[dict[str, Any]],
    *,
    min_prob: float,
    min_duration_s: float = 0.0,
    min_start_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    min_prob = max(0.0, min(1.0, float(min_prob)))
    min_duration_s = max(0.0, float(min_duration_s))
    min_start_seconds = max(0.0, float(min_start_seconds))
    kept: list[dict[str, Any]] = []
    for row in rows:
        score = _as_float(row.get("score"))
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        if score is None:
            continue
        if score < min_prob:
            continue
        if start is None or end is None:
            continue
        if start < min_start_seconds:
            continue
        if (end - start) < min_duration_s:
            continue
        kept.append(row)
    return kept


def _svg_track_rects(
    rows: list[dict[str, Any]],
    *,
    total: float,
    y: float,
    height: float,
    x0: float,
    width: float,
    color_fn,
    lane_name: str,
    show_label: bool = True,
    squish_px: float = 0.0,
) -> tuple[str, str]:
    base_out = []
    hover_out = []
    segments: list[dict[str, Any]] = []
    for row in rows:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        label = row.get("label")
        if start is None or end is None or not isinstance(label, str):
            continue
        if end < start:
            start, end = end, start
        start = max(0.0, min(total, start))
        end = max(0.0, min(total, end))
        if total <= 0:
            continue
        x = x0 + (start / total) * width
        w = max(1.0, ((end - start) / total) * width)
        color = color_fn(label)
        score = _as_float(row.get("score"))
        title_bits = [lane_name, f"{_fmt_clock(start)} -> {_fmt_clock(end)}"]
        if score is not None:
            title_bits.append(f"{label}: {score:.3f}")
        else:
            title_bits.append(label)
        raw_top3 = row.get("raw_top3")
        if isinstance(raw_top3, list) and raw_top3:
            raw_bits = []
            for pair in raw_top3[:3]:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                raw_label = pair[0]
                raw_prob = _as_float(pair[1])
                if not isinstance(raw_label, str) or raw_prob is None:
                    continue
                raw_bits.append(f"{raw_label} {raw_prob:.3f}")
            if raw_bits:
                title_bits.append(f"Raw: {', '.join(raw_bits)}")
        segments.append(
            {
                "x": x,
                "w": w,
                "start": start,
                "label": label,
                "score": score,
                "title_bits": title_bits,
                "color": color,
            }
        )

    if not segments:
        return "", ""

    segments.sort(key=lambda s: (s["x"], s["x"] + s["w"]))
    if squish_px > 0:
        for i in range(len(segments) - 1):
            left = segments[i]
            right = segments[i + 1]
            left_end = left["x"] + left["w"]
            gap = right["x"] - left_end
            if 0.0 < gap <= squish_px:
                delta = gap / 2.0
                left["w"] = left["w"] + delta
                right["x"] = right["x"] - delta
                right["w"] = right["w"] + delta

    for seg in segments:
        x = float(seg["x"])
        w = max(1.0, float(seg["w"]))
        start = float(seg["start"])
        label = str(seg["label"])
        score = _as_float(seg.get("score"))
        title_bits = seg["title_bits"]
        color = str(seg["color"])

        title = html.escape("\n".join(title_bits), quote=True)
        tooltip_lines = [line for line in title_bits if isinstance(line, str)]
        max_chars = max((len(line) for line in tooltip_lines), default=10)
        tip_w = min(420.0, max(140.0, 18.0 + 6.4 * max_chars))
        tip_h = 14.0 + 14.0 * max(1, len(tooltip_lines))
        tip_x = x + 6.0
        if tip_x + tip_w > x0 + width:
            tip_x = x0 + width - tip_w - 4.0
        if tip_x < x0 + 2.0:
            tip_x = x0 + 2.0
        tip_y = y - tip_h - 4.0
        if tip_y < 2.0:
            tip_y = y + height + 4.0
        tip_text = "".join(
            (
                f'<tspan x="{tip_x + 8.0:.2f}"'
                f' dy="{0 if idx == 0 else 14}">{html.escape(line)}</tspan>'
            )
            for idx, line in enumerate(tooltip_lines)
        )
        if show_label and w >= 84 and score is not None:
            text = html.escape(f"{label} {score:.2f}")
        elif show_label and w >= 56:
            text = html.escape(label)
        else:
            text = ""
        base_out.append(
            f'<g class="mb-seg-group"><rect class="mb-seg-fill" '
            f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height:.2f}" '
            f'rx="1.5" fill="{color}" opacity="0.95"></rect>'
            f'<text x="{x + 4:.2f}" y="{y + height * 0.68:.2f}" '
            f'class="mb-seg-text" font-size="10" font-weight="600">{text}</text></g>'
        )
        hover_out.append(
            f'<g class="mb-seg-group"><rect class="mb-hover-seg" '
            f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height:.2f}" '
            f'rx="1.5" fill="rgba(0,0,0,0)"><title>{title}</title></rect>'
            f'<g class="mb-hover-tip"><rect x="{tip_x:.2f}" y="{tip_y:.2f}" '
            f'width="{tip_w:.2f}" height="{tip_h:.2f}" rx="6"></rect>'
            f'<text x="{tip_x + 8.0:.2f}" y="{tip_y + 14.0:.2f}">{tip_text}</text></g>'
            f"</g>"
        )
    return "".join(base_out), "".join(hover_out)


def _tag_rank_rows(
    tag_events: list[dict[str, Any]],
    rank_idx: int,
    *,
    min_duration_s: float = 0.0,
    per_tag_scale: float = 1.0,
    per_tag_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in tag_events:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        top3 = row.get("top3")
        if start is None or end is None or not isinstance(top3, list):
            continue
        if rank_idx >= len(top3):
            continue
        pair = top3[rank_idx]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        label, prob = pair[0], _as_float(pair[1])
        if not isinstance(label, str) or prob is None:
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "score": prob,
                "top3": top3,
                "raw_top3": row.get("raw_top3"),
            }
        )
    merged = _merge_adjacent_tag_rows(out)
    min_duration_s = max(0.0, float(min_duration_s))
    per_tag_scale = max(0.0, float(per_tag_scale))
    kept = []
    for row in merged:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        label = row.get("label")
        if start is None or end is None:
            continue
        if not isinstance(label, str):
            continue
        min_needed = _min_duration_for_tag(
            label,
            scale=per_tag_scale,
            floor=min_duration_s,
            overrides=per_tag_overrides,
        )
        if (end - start) >= min_needed:
            kept.append(row)
    return kept


def _merge_adjacent_tag_rows(
    rows: list[dict[str, Any]], *, merge_gap: float = 0.05
) -> list[dict[str, Any]]:
    if not rows:
        return []

    cleaned = []
    for row in rows:
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        label = row.get("label")
        score = _as_float(row.get("score"))
        if start is None or end is None or not isinstance(label, str):
            continue
        if end < start:
            start, end = end, start
        cleaned.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "score": 0.0 if score is None else score,
                "raw_top3": row.get("raw_top3"),
            }
        )

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []
    for row in cleaned:
        dur = max(0.0, row["end"] - row["start"])
        if not merged:
            merged.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "label": row["label"],
                    "_score_max": row["score"],
                    "_raw_top3": row.get("raw_top3"),
                }
            )
            continue

        prev = merged[-1]
        if row["label"] == prev["label"] and row["start"] <= prev["end"] + merge_gap:
            prev["end"] = max(prev["end"], row["end"])
            if row["score"] >= float(prev["_score_max"]):
                prev["_raw_top3"] = row.get("raw_top3")
            prev["_score_max"] = max(float(prev["_score_max"]), row["score"])
        else:
            merged.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "label": row["label"],
                    "_score_max": row["score"],
                    "_raw_top3": row.get("raw_top3"),
                }
            )

    out: list[dict[str, Any]] = []
    for row in merged:
        score_max = float(row.get("_score_max", 0.0))
        out.append(
            {
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "label": str(row["label"]),
                "score": round(score_max, 4),
                "raw_top3": row.get("_raw_top3"),
            }
        )
    return out


def _fmt_timeline_html(
    record: dict[str, Any],
    min_start_seconds: float = TAG_DISPLAY_MIN_START_SECONDS,
    sigh_min_prob: float = 0.0,
    sigh_min_duration_s: float = 0.0,
    laughter_min_prob: float = 0.0,
    laughter_min_duration_s: float = 0.0,
    chuckle_min_prob: float = 0.0,
    chuckle_min_duration_s: float = 0.0,
    breathing_min_prob: float = 0.0,
    breathing_min_duration_s: float = 0.0,
) -> str:
    emotion_timeline = _emotion_timeline(record)
    grouped_tag_events = _group_tag_events(record)
    derived_tags = _derive_canonical_tag_events(
        record,
        min_start_seconds=min_start_seconds,
        sigh_min_prob=sigh_min_prob,
        sigh_min_duration_s=sigh_min_duration_s,
        laughter_min_prob=laughter_min_prob,
        laughter_min_duration_s=laughter_min_duration_s,
        chuckle_min_prob=chuckle_min_prob,
        chuckle_min_duration_s=chuckle_min_duration_s,
        breathing_min_prob=breathing_min_prob,
        breathing_min_duration_s=breathing_min_duration_s,
    )
    laughter_rows = _display_group_events(
        grouped_tag_events.get(TAG_GROUP_LAUGHTER, []),
        group=TAG_GROUP_LAUGHTER,
        min_prob=laughter_min_prob,
        min_duration_s=laughter_min_duration_s,
        min_start_seconds=min_start_seconds,
    )
    sigh_rows = _display_group_events(
        grouped_tag_events.get(TAG_GROUP_SIGH, []),
        group=TAG_GROUP_SIGH,
        min_prob=sigh_min_prob,
        min_duration_s=sigh_min_duration_s,
        min_start_seconds=min_start_seconds,
    )
    chuckle_rows = _display_group_events(
        grouped_tag_events.get(TAG_GROUP_CHUCKLE, []),
        group=TAG_GROUP_CHUCKLE,
        min_prob=chuckle_min_prob,
        min_duration_s=chuckle_min_duration_s,
        min_start_seconds=min_start_seconds,
    )
    breathing_rows = _display_group_events(
        grouped_tag_events.get(TAG_GROUP_BREATHING, []),
        group=TAG_GROUP_BREATHING,
        min_prob=breathing_min_prob,
        min_duration_s=breathing_min_duration_s,
        min_start_seconds=min_start_seconds,
    )
    total = _segment_duration(
        record,
        [emotion_timeline, laughter_rows, sigh_rows, chuckle_rows, breathing_rows],
    )
    if total <= 0:
        total = 1.0

    overall = _overall_emotion(emotion_timeline)
    if overall:
        label, ratio, score = overall
        if score is None:
            overall_html = (
                f"<b>Overall sentence emotion:</b> "
                f"<span class='mb-pill'>{html.escape(label)}</span> "
                f"({ratio * 100:.1f}% of segment)"
            )
        else:
            overall_html = (
                f"<b>Overall sentence emotion:</b> "
                f"<span class='mb-pill'>{html.escape(label)}</span> "
                f"({ratio * 100:.1f}% of segment, avg conf {score:.3f})"
            )
    else:
        overall_html = "<b>Overall sentence emotion:</b> not available yet"

    transcript = record.get("text")
    transcript = transcript.strip() if isinstance(transcript, str) else ""
    tagged_text = (
        inject_tags_into_text(transcript, record.get("words"), derived_tags)
        if transcript
        else ""
    )
    if derived_tags:
        tag_summary = " ".join(
            (
                f"<span class='mb-pill'>{html.escape(tag['tag'])} "
                f"{_fmt_clock(tag.get('from'))}-{_fmt_clock(tag.get('to'))} "
                f"{float(tag['score']):.2f}</span>"
            )
            for tag in derived_tags
            if isinstance(tag, dict)
            and isinstance(tag.get("tag"), str)
            and _as_float(tag.get("score")) is not None
        )
    else:
        tag_summary = (
            "<span class='mb-empty'>no tags at current thresholds; "
            "move min-prob sliders left, reduce min duration, or lower the early-cutoff slider</span>"
        )

    if tagged_text:
        tagged_text_html = html.escape(tagged_text)
    else:
        tagged_text_html = "<span class='mb-empty'>no transcript text</span>"

    container_id = f"mb-{random.randrange(1_000_000_000):x}"

    svg_w = 1000.0
    svg_h = 200.0
    x0 = 0.0
    xw = 1000.0
    emo_y = 26.0
    laugh_y = 60.0
    sigh_y = 94.0
    chuckle_y = 128.0
    breath_y = 162.0
    track_h = 14.0

    emotion_rects, emotion_hover = _svg_track_rects(
        emotion_timeline,
        total=total,
        y=emo_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_emotion_color,
        lane_name="Emotion",
        squish_px=0.0,
    )
    tag1_rects, tag1_hover = _svg_track_rects(
        laughter_rows,
        total=total,
        y=laugh_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name=TAG_GROUP_LAUGHTER,
        show_label=False,
        squish_px=2.2,
    )
    sigh_rects, sigh_hover = _svg_track_rects(
        sigh_rows,
        total=total,
        y=sigh_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name=TAG_GROUP_SIGH,
        show_label=False,
        squish_px=2.2,
    )
    chuckle_rects, chuckle_hover = _svg_track_rects(
        chuckle_rows,
        total=total,
        y=chuckle_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name=TAG_GROUP_CHUCKLE,
        show_label=False,
        squish_px=2.2,
    )
    tag2_rects, tag2_hover = _svg_track_rects(
        breathing_rows,
        total=total,
        y=breath_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name=TAG_GROUP_BREATHING,
        show_label=False,
        squish_px=2.2,
    )
    return f"""
<style>
.mb-wrap {{
  --mb-bg: #fafafa;
  --mb-border: #d4d4d8;
  --mb-text: #18181b;
  --mb-muted: #52525b;
  --mb-pill-bg: #f4f4f5;
  --mb-pill-border: #d4d4d8;
  --mb-list-bg: #ffffff;
  --mb-list-border: #d4d4d8;
  --mb-list-text: #18181b;
  width: 100%;
  box-sizing: border-box;
  padding: 0;
  margin: 0;
  border: 0;
  background: transparent;
  color: var(--mb-text);
}}
@media (prefers-color-scheme: dark) {{
  .mb-wrap {{
    --mb-bg: #18181b;
    --mb-border: #3f3f46;
    --mb-text: #f4f4f5;
    --mb-muted: #d4d4d8;
    --mb-pill-bg: #27272a;
    --mb-pill-border: #52525b;
    --mb-list-bg: #18181b;
    --mb-list-border: #3f3f46;
    --mb-list-text: #f4f4f5;
  }}
}}
.dark .mb-wrap,
[data-theme="dark"] .mb-wrap {{
  --mb-bg: #18181b;
  --mb-border: #3f3f46;
  --mb-text: #f4f4f5;
  --mb-muted: #d4d4d8;
  --mb-pill-bg: #27272a;
  --mb-pill-border: #52525b;
  --mb-list-bg: #18181b;
  --mb-list-border: #3f3f46;
  --mb-list-text: #f4f4f5;
}}
.mb-head {{
  font-size: 13px;
  margin: 0 0 6px 0;
  padding: 0 2px;
}}
.mb-chart {{
  position: relative;
  width: 100%;
  margin: 0;
  box-sizing: border-box;
}}
.mb-hover-seg {{
  cursor: pointer;
}}
.mb-seg-group {{
  isolation: isolate;
}}
.mb-hover-seg:hover {{
  stroke: #111827;
  stroke-width: 1.2;
}}
.dark .mb-hover-seg:hover,
[data-theme="dark"] .mb-hover-seg:hover {{
  stroke: #f4f4f5;
  stroke-width: 1.2;
}}
.mb-hover-tip {{
  opacity: 0;
  pointer-events: none;
}}
.mb-seg-group:hover .mb-hover-tip {{
  opacity: 1;
}}
.mb-hover-tip rect {{
  fill: #ffffff;
  stroke: #a1a1aa;
  stroke-width: 1;
}}
.mb-hover-tip text {{
  fill: #111827;
  font-size: 11px;
  font-weight: 600;
}}
@media (prefers-color-scheme: dark) {{
  .mb-hover-tip rect {{
    fill: #18181b;
    stroke: #52525b;
  }}
  .mb-hover-tip text {{
    fill: #f4f4f5;
  }}
}}
.dark .mb-hover-tip rect,
[data-theme="dark"] .mb-hover-tip rect {{
  fill: #18181b;
  stroke: #52525b;
}}
.dark .mb-hover-tip text,
[data-theme="dark"] .mb-hover-tip text {{
  fill: #f4f4f5;
}}
.mb-lane-label {{
  fill: #52525b;
}}
@media (prefers-color-scheme: dark) {{
  .mb-lane-label {{
    fill: #d4d4d8;
  }}
}}
.dark .mb-lane-label,
[data-theme="dark"] .mb-lane-label {{
  fill: #d4d4d8;
}}
.mb-time-label {{
  fill: #6b7280;
}}
@media (prefers-color-scheme: dark) {{
  .mb-time-label {{
    fill: #a1a1aa;
  }}
}}
.dark .mb-time-label,
[data-theme="dark"] .mb-time-label {{
  fill: #a1a1aa;
}}
.mb-svg-bg {{
  fill: #e4e4e7;
}}
@media (prefers-color-scheme: dark) {{
  .mb-svg-bg {{
    fill: #3f3f46;
  }}
}}
.dark .mb-svg-bg,
[data-theme="dark"] .mb-svg-bg {{
  fill: #3f3f46;
}}
.mb-scale-line {{
  stroke: #a1a1aa;
}}
@media (prefers-color-scheme: dark) {{
  .mb-scale-line {{
    stroke: #71717a;
  }}
}}
.dark .mb-scale-line,
[data-theme="dark"] .mb-scale-line {{
  stroke: #71717a;
}}
.mb-seg-text {{
  fill: #111827;
  pointer-events: none;
}}
@media (prefers-color-scheme: dark) {{
  .mb-seg-text {{
    fill: #f4f4f5;
  }}
}}
.dark .mb-seg-text,
[data-theme="dark"] .mb-seg-text {{
  fill: #f4f4f5;
}}
.mb-svg {{
  display: block;
  width: 100%;
  box-sizing: border-box;
  margin: 0;
  border: 1px solid var(--mb-border);
  border-radius: 6px;
  background: var(--mb-list-bg);
}}
.mb-pill {{
  display: inline-block;
  border-radius: 999px;
  padding: 1px 9px;
  margin: 0 4px;
  font-size: 12px;
  font-weight: 700;
  color: var(--mb-text);
  background: var(--mb-pill-bg);
  border: 1px solid var(--mb-pill-border);
}}
.mb-subhead {{
  font-size: 13px;
  line-height: 1.45;
  margin: 0 0 8px 0;
  padding: 0 2px;
}}
.mb-tagged-text {{
  margin: 6px 0 8px 0;
  padding: 8px 10px;
  border: 1px solid var(--mb-border);
  border-radius: 6px;
  background: var(--mb-list-bg);
  font-size: 13px;
  line-height: 1.45;
  white-space: normal;
}}
.mb-empty {{
  color: var(--mb-muted);
  font-style: italic;
}}
</style>
<div class="mb-wrap" id="{container_id}">
  <div class="mb-head">{overall_html}</div>
  <div class="mb-subhead"><b>Derived tags:</b> {tag_summary}</div>
  <div class="mb-subhead"><b>Tagged text preview:</b></div>
  <div class="mb-tagged-text">{tagged_text_html}</div>
  <div class="mb-chart">
    <svg class="mb-svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">
      <line class="mb-scale-line" x1="{x0:.1f}" y1="{emo_y - 10:.1f}" x2="{x0 + xw:.1f}" y2="{emo_y - 10:.1f}" stroke-width="1"/>
      <text class="mb-time-label" x="{x0 + 2:.1f}" y="{emo_y - 14:.1f}" font-size="11">{_fmt_clock(0.0)}</text>
      <text class="mb-time-label" x="{x0 + xw - 64:.1f}" y="{emo_y - 14:.1f}" font-size="11">{_fmt_clock(total)}</text>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{emo_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{laugh_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{sigh_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{chuckle_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{breath_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{emo_y + track_h - 6:.1f}" font-size="10" font-weight="700">Emotion</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{laugh_y + track_h - 6:.1f}" font-size="10" font-weight="700">{TAG_GROUP_LAUGHTER}</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{sigh_y + track_h - 6:.1f}" font-size="10" font-weight="700">{TAG_GROUP_SIGH}</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{chuckle_y + track_h - 6:.1f}" font-size="10" font-weight="700">{TAG_GROUP_CHUCKLE}</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{breath_y + track_h - 6:.1f}" font-size="10" font-weight="700">Breathing/Gasp+</text>
      {emotion_rects}
      {tag1_rects}
      {sigh_rects}
      {chuckle_rects}
      {tag2_rects}
      {emotion_hover}
      {tag1_hover}
      {sigh_hover}
      {chuckle_hover}
      {tag2_hover}
    </svg>
  </div>
</div>
"""


def _fmt_dialect_markdown(record: dict[str, Any]) -> str:
    lines = ["### Dialect"]
    dialect_name = (
        record.get("dialect_tag") or record.get("dialect") or record.get("dialect_name")
    )
    dialect_code = record.get("dialect_code")
    dialect_source = record.get("dialect_source")

    if (
        dialect_name is not None
        or dialect_code is not None
        or dialect_source is not None
    ):
        lines.append(
            f"<div style='font-size: 1.12rem; font-weight: 800; line-height: 1.35;'>"
            f"{dialect_name or '-'}"
            f"{f' ({dialect_code})' if dialect_code is not None else ''}"
            f"</div>"
        )
        if dialect_source is not None:
            lines.append(f"Source: `{dialect_source}`")
    else:
        segment_name = record.get("dialect_segment_name")
        segment_code = record.get("dialect_segment")
        speaker_name = record.get("dialect_speaker_majority_name")
        speaker_code = record.get("dialect_speaker_majority")
        if (
            segment_name is not None
            or segment_code is not None
            or speaker_name is not None
            or speaker_code is not None
        ):
            lines.append("Sentence-level prediction:")
            lines.append(
                f"<div style='font-size: 1.06rem; font-weight: 700; line-height: 1.35;'>"
                f"{segment_name or '-'} ({segment_code or '-'})"
                f"</div>"
            )
            lines.append("")
            lines.append("Speaker majority over all sentences:")
            lines.append(
                f"<div style='font-size: 1.18rem; font-weight: 800; line-height: 1.35;'>"
                f"{speaker_name or '-'} ({speaker_code or '-'})"
                f"</div>"
            )
        else:
            lines.append("_Not available yet._")
    return "\n".join(lines)


def _fmt_omni_markdown(record: dict[str, Any]) -> str:
    lines = ["### Omni Transcription"]

    omni_text = record.get("omni_text")
    if isinstance(omni_text, str) and omni_text:
        lines.append(omni_text)
    else:
        lines.append("_Not available yet._")
    return "\n".join(lines)


def _fmt_record_summary(record: dict[str, Any]) -> str:
    podcast, title = _derive_podcast_and_title(record)
    base_text = record.get("text")
    base_audio_path = record.get("base_audio_path") or record.get("source_audio")
    transcript = (
        base_text
        if isinstance(base_text, str) and base_text.strip()
        else "_Missing transcript text._"
    )
    lines = [
        "### Transcript",
        transcript,
        "",
        _fmt_omni_markdown(record),
        "",
        _fmt_dialect_markdown(record),
        "",
        "### Segment",
        f"- Podcast: {podcast or 'Unknown Podcast'}",
        f"- Episode: {title or 'Unknown Title'}",
        f"- Speaker: {record.get('speaker', '-')}",
        f"- Time: {_fmt_clock(record.get('start'))} -> {_fmt_clock(record.get('end'))}",
        f"- Base audio: {base_audio_path or '-'}",
    ]
    return "\n".join(lines)


def create_app(default_manifest: str = DEFAULT_MANIFEST) -> gr.Blocks:
    browser: dict[str, Any] = {
        "store": None,
        "total_rows": 0,
        "current_row_id": None,
    }
    fixed_manifest = Path(default_manifest).expanduser()
    initial_min_start_seconds = TAG_DISPLAY_MIN_START_SECONDS
    initial_sigh_min_prob = DEFAULT_GROUP_MIN_PROB[TAG_GROUP_SIGH]
    initial_sigh_min_duration_s = DEFAULT_GROUP_MIN_DURATION_S[TAG_GROUP_SIGH]
    initial_laughter_min_prob = DEFAULT_GROUP_MIN_PROB[TAG_GROUP_LAUGHTER]
    initial_laughter_min_duration_s = DEFAULT_GROUP_MIN_DURATION_S[TAG_GROUP_LAUGHTER]
    initial_chuckle_min_prob = DEFAULT_GROUP_MIN_PROB[TAG_GROUP_CHUCKLE]
    initial_chuckle_min_duration_s = DEFAULT_GROUP_MIN_DURATION_S[TAG_GROUP_CHUCKLE]
    initial_breathing_min_prob = DEFAULT_GROUP_MIN_PROB[TAG_GROUP_BREATHING]
    initial_breathing_min_duration_s = DEFAULT_GROUP_MIN_DURATION_S[TAG_GROUP_BREATHING]

    def _row_bundle(
        record: dict[str, Any],
        min_start_seconds: float,
        sigh_min_prob: float,
        sigh_min_duration_s: float,
        laughter_min_prob: float,
        laughter_min_duration_s: float,
        chuckle_min_prob: float,
        chuckle_min_duration_s: float,
        breathing_min_prob: float,
        breathing_min_duration_s: float,
    ):
        audio = str(record.get("audio_path", ""))
        if not audio or not Path(audio).exists():
            audio = None
        return (
            audio,
            _fmt_timeline_html(
                record,
                min_start_seconds=min_start_seconds,
                sigh_min_prob=sigh_min_prob,
                sigh_min_duration_s=sigh_min_duration_s,
                laughter_min_prob=laughter_min_prob,
                laughter_min_duration_s=laughter_min_duration_s,
                chuckle_min_prob=chuckle_min_prob,
                chuckle_min_duration_s=chuckle_min_duration_s,
                breathing_min_prob=breathing_min_prob,
                breathing_min_duration_s=breathing_min_duration_s,
            ),
            _fmt_record_summary(record),
            json.dumps(_select_raw_json_fields(record), ensure_ascii=False, indent=2),
        )

    def _id_markdown(row_id: int | None, total_rows: int) -> str:
        if row_id is None or total_rows <= 0:
            return "Current ID: -"
        return f"Current ID: **{row_id}** / {max(total_rows - 1, 0)}"

    def _empty_outputs(summary: str = "Manifest not loaded."):
        return (
            _id_markdown(None, int(browser.get("total_rows", 0))),
            None,
            "<div>No timeline available.</div>",
            summary,
            "",
            None,
        )

    def _render_row(
        row_id: int,
        min_start_seconds: float,
        sigh_min_prob: float,
        sigh_min_duration_s: float,
        laughter_min_prob: float,
        laughter_min_duration_s: float,
        chuckle_min_prob: float,
        chuckle_min_duration_s: float,
        breathing_min_prob: float,
        breathing_min_duration_s: float,
    ):
        store = browser.get("store")
        total_rows = int(browser.get("total_rows", 0))
        if store is None or total_rows <= 0:
            return _empty_outputs("Manifest missing.")

        row_id = max(0, min(int(row_id), total_rows - 1))
        browser["current_row_id"] = row_id
        row = store.get_row(row_id)
        audio, timeline_html, summary, raw = _row_bundle(
            row,
            min_start_seconds,
            sigh_min_prob,
            sigh_min_duration_s,
            laughter_min_prob,
            laughter_min_duration_s,
            chuckle_min_prob,
            chuckle_min_duration_s,
            breathing_min_prob,
            breathing_min_duration_s,
        )
        return (
            _id_markdown(row_id, total_rows),
            audio,
            timeline_html,
            summary,
            raw,
            row_id,
        )

    initial_audio = None
    initial_timeline = "<div>No timeline available.</div>"
    initial_summary = "Manifest not loaded."
    initial_raw = ""
    initial_current_id_md = _id_markdown(None, 0)
    initial_jump_value: int | None = None

    if not fixed_manifest.exists():
        initial_summary = f"Manifest not found: `{fixed_manifest}`"
    else:
        store = ManifestStore(fixed_manifest)
        total = store.build(require_annotations=False)
        browser["store"] = store
        browser["total_rows"] = total
        if total == 0:
            initial_summary = "Manifest is empty."
        else:
            browser["current_row_id"] = 0
            initial_audio, initial_timeline, initial_summary, initial_raw = _row_bundle(
                store.get_row(0),
                initial_min_start_seconds,
                initial_sigh_min_prob,
                initial_sigh_min_duration_s,
                initial_laughter_min_prob,
                initial_laughter_min_duration_s,
                initial_chuckle_min_prob,
                initial_chuckle_min_duration_s,
                initial_breathing_min_prob,
                initial_breathing_min_duration_s,
            )
            initial_current_id_md = _id_markdown(0, total)
            initial_jump_value = 0

    with gr.Blocks(
        title="Manifest Audio Browser",
        css="""
#segment-audio audio { max-height: 42px; }
#segment-audio .wrap { min-height: 52px !important; }
""",
    ) as app:
        gr.Markdown("## Manifest Audio Browser")
        with gr.Row():
            prev_btn = gr.Button("Previous ID")
            with gr.Column():
                next_btn = gr.Button("Next ID", variant="primary")
                random_btn = gr.Button("Random")
            jump_id_input = gr.Number(
                label="Jump to ID",
                value=initial_jump_value,
                precision=0,
            )
            jump_btn = gr.Button("Jump")

        current_id_md = gr.Markdown(initial_current_id_md)
        with gr.Row():
            min_start_slider = gr.Slider(
                minimum=0.0,
                maximum=8.0,
                step=0.05,
                value=initial_min_start_seconds,
                label="Hide events starting before (s)",
            )
            sigh_prob_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=initial_sigh_min_prob,
                label="Sigh min prob (lower = more)",
            )
            sigh_duration_slider = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                value=initial_sigh_min_duration_s,
                label="Sigh min duration (s)",
            )
        with gr.Row():
            laughter_prob_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=initial_laughter_min_prob,
                label="Laughter min prob (lower = more)",
            )
            laughter_duration_slider = gr.Slider(
                minimum=0.0,
                maximum=6.0,
                step=0.05,
                value=initial_laughter_min_duration_s,
                label="Laughter min duration (s)",
            )
            chuckle_prob_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=initial_chuckle_min_prob,
                label="Chuckle/Giggle min prob (lower = more)",
            )
            chuckle_duration_slider = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                value=initial_chuckle_min_duration_s,
                label="Chuckle/Giggle min duration (s)",
            )
        with gr.Row():
            breathing_prob_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=initial_breathing_min_prob,
                label="Breathing/Gasp+ min prob (lower = more)",
            )
            breathing_duration_slider = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                step=0.05,
                value=initial_breathing_min_duration_s,
                label="Breathing/Gasp+ min duration (s)",
            )

        audio_player = gr.Audio(
            label="Segment Audio",
            type="filepath",
            value=initial_audio,
            elem_id="segment-audio",
        )
        timeline_md = gr.HTML(initial_timeline)
        summary_md = gr.Markdown(initial_summary)
        raw_json = gr.Code(label="Raw record JSON", language="json", value=initial_raw)
        timeline_inputs = [
            min_start_slider,
            sigh_prob_slider,
            sigh_duration_slider,
            laughter_prob_slider,
            laughter_duration_slider,
            chuckle_prob_slider,
            chuckle_duration_slider,
            breathing_prob_slider,
            breathing_duration_slider,
        ]

        def go_prev(
            min_start_seconds: float,
            sigh_min_prob: float,
            sigh_min_duration_s: float,
            laughter_min_prob: float,
            laughter_min_duration_s: float,
            chuckle_min_prob: float,
            chuckle_min_duration_s: float,
            breathing_min_prob: float,
            breathing_min_duration_s: float,
        ):
            total_rows = int(browser.get("total_rows", 0))
            if total_rows <= 0:
                return _empty_outputs("Manifest is empty.")
            current = browser.get("current_row_id")
            current_id = int(current) if isinstance(current, int) else 0
            return _render_row(
                current_id - 1,
                min_start_seconds,
                sigh_min_prob,
                sigh_min_duration_s,
                laughter_min_prob,
                laughter_min_duration_s,
                chuckle_min_prob,
                chuckle_min_duration_s,
                breathing_min_prob,
                breathing_min_duration_s,
            )

        def go_next(
            min_start_seconds: float,
            sigh_min_prob: float,
            sigh_min_duration_s: float,
            laughter_min_prob: float,
            laughter_min_duration_s: float,
            chuckle_min_prob: float,
            chuckle_min_duration_s: float,
            breathing_min_prob: float,
            breathing_min_duration_s: float,
        ):
            total_rows = int(browser.get("total_rows", 0))
            if total_rows <= 0:
                return _empty_outputs("Manifest is empty.")
            current = browser.get("current_row_id")
            current_id = int(current) if isinstance(current, int) else 0
            return _render_row(
                current_id + 1,
                min_start_seconds,
                sigh_min_prob,
                sigh_min_duration_s,
                laughter_min_prob,
                laughter_min_duration_s,
                chuckle_min_prob,
                chuckle_min_duration_s,
                breathing_min_prob,
                breathing_min_duration_s,
            )

        def jump_to_id(
            jump_row_id: float | int | None,
            min_start_seconds: float,
            sigh_min_prob: float,
            sigh_min_duration_s: float,
            laughter_min_prob: float,
            laughter_min_duration_s: float,
            chuckle_min_prob: float,
            chuckle_min_duration_s: float,
            breathing_min_prob: float,
            breathing_min_duration_s: float,
        ):
            total_rows = int(browser.get("total_rows", 0))
            if total_rows <= 0:
                return _empty_outputs("Manifest is empty.")
            if jump_row_id is None:
                current = browser.get("current_row_id")
                target = int(current) if isinstance(current, int) else 0
            else:
                target = int(jump_row_id)
            return _render_row(
                target,
                min_start_seconds,
                sigh_min_prob,
                sigh_min_duration_s,
                laughter_min_prob,
                laughter_min_duration_s,
                chuckle_min_prob,
                chuckle_min_duration_s,
                breathing_min_prob,
                breathing_min_duration_s,
            )

        def go_random(
            min_start_seconds: float,
            sigh_min_prob: float,
            sigh_min_duration_s: float,
            laughter_min_prob: float,
            laughter_min_duration_s: float,
            chuckle_min_prob: float,
            chuckle_min_duration_s: float,
            breathing_min_prob: float,
            breathing_min_duration_s: float,
        ):
            total_rows = int(browser.get("total_rows", 0))
            if total_rows <= 0:
                return _empty_outputs("Manifest is empty.")
            target = random.randrange(total_rows)
            return _render_row(
                target,
                min_start_seconds,
                sigh_min_prob,
                sigh_min_duration_s,
                laughter_min_prob,
                laughter_min_duration_s,
                chuckle_min_prob,
                chuckle_min_duration_s,
                breathing_min_prob,
                breathing_min_duration_s,
            )

        def refresh_timeline(
            min_start_seconds: float,
            sigh_min_prob: float,
            sigh_min_duration_s: float,
            laughter_min_prob: float,
            laughter_min_duration_s: float,
            chuckle_min_prob: float,
            chuckle_min_duration_s: float,
            breathing_min_prob: float,
            breathing_min_duration_s: float,
        ):
            store = browser.get("store")
            row_id = browser.get("current_row_id")
            if store is None or not isinstance(row_id, int):
                return "<div>No timeline available.</div>"
            row = store.get_row(row_id)
            return _fmt_timeline_html(
                row,
                min_start_seconds=min_start_seconds,
                sigh_min_prob=sigh_min_prob,
                sigh_min_duration_s=sigh_min_duration_s,
                laughter_min_prob=laughter_min_prob,
                laughter_min_duration_s=laughter_min_duration_s,
                chuckle_min_prob=chuckle_min_prob,
                chuckle_min_duration_s=chuckle_min_duration_s,
                breathing_min_prob=breathing_min_prob,
                breathing_min_duration_s=breathing_min_duration_s,
            )

        prev_btn.click(
            go_prev,
            inputs=timeline_inputs,
            outputs=[
                current_id_md,
                audio_player,
                timeline_md,
                summary_md,
                raw_json,
                jump_id_input,
            ],
        )
        next_btn.click(
            go_next,
            inputs=timeline_inputs,
            outputs=[
                current_id_md,
                audio_player,
                timeline_md,
                summary_md,
                raw_json,
                jump_id_input,
            ],
        )
        random_btn.click(
            go_random,
            inputs=timeline_inputs,
            outputs=[
                current_id_md,
                audio_player,
                timeline_md,
                summary_md,
                raw_json,
                jump_id_input,
            ],
        )
        jump_btn.click(
            jump_to_id,
            inputs=[jump_id_input, *timeline_inputs],
            outputs=[
                current_id_md,
                audio_player,
                timeline_md,
                summary_md,
                raw_json,
                jump_id_input,
            ],
        )
        for slider in timeline_inputs:
            slider.change(
                refresh_timeline, inputs=timeline_inputs, outputs=[timeline_md]
            )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast Gradio browser for combined JSONL manifests with audio playback."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Default manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable Gradio share link (default: false). Use --share to enable.",
    )
    parser.add_argument(
        "--allow-path",
        action="append",
        default=[],
        help=(
            "Additional filesystem path to allow Gradio to serve. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(default_manifest=args.manifest)
    manifest_path = Path(args.manifest).expanduser().resolve()
    allowed_paths = {
        str(Path.cwd().resolve()),
        str(manifest_path.parent),
        str(manifest_path.parent.parent),
    }
    for extra in args.allow_path:
        allowed_paths.add(str(Path(extra).expanduser().resolve()))

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=sorted(allowed_paths),
    )


if __name__ == "__main__":
    main()
