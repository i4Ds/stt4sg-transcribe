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

DEFAULT_MANIFEST = (
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_combined.jsonl"
)

TIMELINE_SEEK_HEAD = """
<script>
(() => {
  window.__mbBindTimelineSeek = () => {
    if (window.__mbTimelineSeekBound) return;
    const findAudioElement = () => {
      const direct = document.querySelector("#segment_audio_player audio");
      if (direct) return direct;
      const queue = [document.documentElement];
      while (queue.length) {
        const node = queue.shift();
      if (!node) continue;
      if (node.querySelector) {
        const a = node.querySelector("audio");
        if (a) return a;
      }
      const children = node.children || [];
      for (const child of children) queue.push(child);
      if (node.shadowRoot) queue.push(node.shadowRoot);
    }
    return null;
      };
    };

    const seekAudio = (audio, seek) => {
      const apply = () => {
        const maxT = Number.isFinite(audio.duration) && audio.duration > 0
          ? Math.max(0, audio.duration - 0.05)
          : seek;
        audio.currentTime = Math.max(0, Math.min(seek, maxT));
        const p = audio.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      };
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        apply();
        return;
      }
      audio.addEventListener("loadedmetadata", apply, { once: true });
      audio.load?.();
    };

    const onPointer = (event) => {
      const node = event.target && event.target.closest
        ? event.target.closest(".mb-hover-seg")
        : null;
      if (!node) return;
      const raw = node.getAttribute("data-start");
      const parsed = Number.parseFloat(raw || "0");
      const seek = Math.max(0, (Number.isFinite(parsed) ? parsed : 0) - 0.2);
      const audio = findAudioElement();
      if (!audio) return;
      try {
        seekAudio(audio, seek);
      } catch (_err) {}
      event.preventDefault();
      event.stopPropagation();
    };
    document.addEventListener("pointerdown", onPointer, true);
    window.__mbTimelineSeekBound = true;
  };
  window.__mbBindTimelineSeek();
})();
</script>
"""

TIMELINE_SEEK_JS = """
() => {
  if (typeof window.__mbBindTimelineSeek === "function") {
    window.__mbBindTimelineSeek();
  }
}
"""


def _extract_string_list_from_module(module_path: Path, var_name: str) -> list[str]:
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


def _normalize_tag_label(label: str) -> str | None:
    raw = label.strip().lower()
    if not raw:
        return None
    if raw in CANONICAL_TAG_ORDER:
        return raw

    if raw in SPEECH_TAGS:
        return "<speech>"

    if raw in {"laughter", "baby laughter", "giggle", "belly laugh"}:
        return "<laugh>"
    if raw in {"snicker", "chuckle, chortle"}:
        return "<chuckle>"
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

    # Keep these vocal events under speech bucket in browser view.
    if raw in {"shout", "screaming", "whispering", "singing", "humming"}:
        return "<speech>"

    # yawn is part of target UI vocabulary, but no current source label maps to it.
    return None


def _canonical_rank(tag: str) -> int:
    try:
        return CANONICAL_TAG_ORDER.index(tag)
    except ValueError:
        return len(CANONICAL_TAG_ORDER)


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
        has_tags = isinstance(record.get("audio_tag_frames"), list) and bool(
            record.get("audio_tag_frames")
        )
    if not has_tags:
        has_tags = isinstance(record.get("audio_tag_timeline"), list) and bool(
            record.get("audio_tag_timeline")
        )
    return has_emotion and has_tags


class TaggedManifestIndex:
    """Byte-offset index for looking up frame-level tag rows by audio_path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset_by_audio: dict[str, int] = {}

    def build(self) -> int:
        self.offset_by_audio.clear()
        if not self.path.exists():
            return 0
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                audio_path = row.get("audio_path")
                if isinstance(audio_path, str) and audio_path:
                    self.offset_by_audio[audio_path] = offset
        return len(self.offset_by_audio)

    def get_row(self, audio_path: str) -> dict[str, Any] | None:
        offset = self.offset_by_audio.get(audio_path)
        if offset is None:
            return None
        with self.path.open("rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8").strip()
        if not line:
            return None
        try:
            row = json.loads(line)
        except Exception:
            return None
        return row if isinstance(row, dict) else None


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
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


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
    record: dict[str, Any], tagged_index: TaggedManifestIndex | None
) -> list[dict[str, Any]]:
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
                        canonical = _normalize_tag_label(pair[0])
                        if canonical is None:
                            continue
                        old = grouped.get(canonical)
                        grouped[canonical] = prob if old is None else max(old, prob)
            clean_top3 = sorted(
                grouped.items(),
                key=lambda x: (-x[1], _canonical_rank(x[0]), x[0]),
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
                        "top3": [(lbl, round(float(prob), 4)) for lbl, prob in clean_top3[:3]],
                    }
                )
        if events:
            return events

    if isinstance(record.get("audio_tag_frames"), list):
        return _non_speech_tag_events_from_frames(record["audio_tag_frames"])

    if tagged_index is not None:
        audio_path = record.get("audio_path")
        if isinstance(audio_path, str) and audio_path:
            tagged_row = tagged_index.get_row(audio_path)
            if tagged_row and isinstance(tagged_row.get("audio_tag_frames"), list):
                return _non_speech_tag_events_from_frames(
                    tagged_row["audio_tag_frames"]
                )

    # Fallback: use collapsed timeline if full frames are unavailable.
    fallback = []
    for row in _tag_timeline(record):
        label = str(row.get("label", ""))
        canonical = _normalize_tag_label(label)
        if canonical is None:
            continue
        score = round(float(row.get("score", 0.0) or 0.0), 4)
        fallback.append(
            {
                "start": row.get("start"),
                "end": row.get("end"),
                "label": canonical,
                "score": score,
                "top3": [(canonical, score)],
            }
        )
    return fallback


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
            f'<g class="mb-seg-group"><rect class="mb-hover-seg" data-start="{start:.3f}" '
            f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height:.2f}" '
            f'rx="1.5" fill="rgba(0,0,0,0)"><title>{title}</title></rect>'
            f'<g class="mb-hover-tip"><rect x="{tip_x:.2f}" y="{tip_y:.2f}" '
            f'width="{tip_w:.2f}" height="{tip_h:.2f}" rx="6"></rect>'
            f'<text x="{tip_x + 8.0:.2f}" y="{tip_y + 14.0:.2f}">{tip_text}</text></g>'
            f"</g>"
        )
    return "".join(base_out), "".join(hover_out)


def _tag_rank_rows(
    tag_events: list[dict[str, Any]], rank_idx: int
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
            }
        )
    return _merge_adjacent_tag_rows(out)


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
                    "_score_sum": row["score"] * dur,
                    "_dur_sum": dur,
                }
            )
            continue

        prev = merged[-1]
        if row["label"] == prev["label"] and row["start"] <= prev["end"] + merge_gap:
            prev["end"] = max(prev["end"], row["end"])
            prev["_score_sum"] += row["score"] * dur
            prev["_dur_sum"] += dur
        else:
            merged.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "label": row["label"],
                    "_score_sum": row["score"] * dur,
                    "_dur_sum": dur,
                }
            )

    out: list[dict[str, Any]] = []
    for row in merged:
        dur_sum = float(row.get("_dur_sum", 0.0))
        score_sum = float(row.get("_score_sum", 0.0))
        avg = score_sum / dur_sum if dur_sum > 0 else 0.0
        out.append(
            {
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "label": str(row["label"]),
                "score": round(avg, 4),
            }
        )
    return out


def _fmt_timeline_html(
    record: dict[str, Any], tagged_index: TaggedManifestIndex | None
) -> str:
    emotion_timeline = _emotion_timeline(record)
    tag_events = _load_tag_events(record, tagged_index)
    total = _segment_duration(record, [emotion_timeline, tag_events])
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

    tag_rank1 = _tag_rank_rows(tag_events, 0)
    tag_rank2 = _tag_rank_rows(tag_events, 1)
    tag_rank3 = _tag_rank_rows(tag_events, 2)
    container_id = f"mb-{random.randrange(1_000_000_000):x}"

    svg_w = 1000.0
    svg_h = 208.0
    x0 = 0.0
    xw = 1000.0
    emo_y = 30.0
    tag1_y = 72.0
    tag2_y = 114.0
    tag3_y = 156.0
    track_h = 22.0

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
        tag_rank1,
        total=total,
        y=tag1_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name="Tag #1",
        squish_px=2.2,
    )
    tag2_rects, tag2_hover = _svg_track_rects(
        tag_rank2,
        total=total,
        y=tag2_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name="Tag #2",
        squish_px=2.2,
    )
    tag3_rects, tag3_hover = _svg_track_rects(
        tag_rank3,
        total=total,
        y=tag3_y,
        height=track_h,
        x0=x0,
        width=xw,
        color_fn=_tag_color,
        lane_name="Tag #3",
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
  font-size: 14px;
  margin: 0 0 8px 0;
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
  border-radius: 8px;
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
</style>
<div class="mb-wrap" id="{container_id}">
  <div class="mb-head">{overall_html}</div>
  <div class="mb-chart">
    <svg class="mb-svg" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">
      <line class="mb-scale-line" x1="{x0:.1f}" y1="{emo_y - 10:.1f}" x2="{x0 + xw:.1f}" y2="{emo_y - 10:.1f}" stroke-width="1"/>
      <text class="mb-time-label" x="{x0 + 2:.1f}" y="{emo_y - 14:.1f}" font-size="11">{_fmt_clock(0.0)}</text>
      <text class="mb-time-label" x="{x0 + xw - 64:.1f}" y="{emo_y - 14:.1f}" font-size="11">{_fmt_clock(total)}</text>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{emo_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{tag1_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{tag2_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <rect class="mb-svg-bg" x="{x0:.2f}" y="{tag3_y:.2f}" width="{xw:.2f}" height="{track_h:.2f}" rx="4"></rect>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{emo_y + track_h - 6:.1f}" font-size="10" font-weight="700">Emotion</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{tag1_y + track_h - 6:.1f}" font-size="10" font-weight="700">Tag #1</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{tag2_y + track_h - 6:.1f}" font-size="10" font-weight="700">Tag #2</text>
      <text class="mb-lane-label" x="{x0 + 6:.1f}" y="{tag3_y + track_h - 6:.1f}" font-size="10" font-weight="700">Tag #3</text>
      {emotion_rects}
      {tag1_rects}
      {tag2_rects}
      {tag3_rects}
      {emotion_hover}
      {tag1_hover}
      {tag2_hover}
      {tag3_hover}
    </svg>
  </div>
</div>
"""


def _fmt_dialect_markdown(record: dict[str, Any]) -> str:
    lines = ["### Dialect"]
    speaker_name = record.get("dialect_speaker_majority_name")
    speaker_code = record.get("dialect_speaker_majority")

    has_speaker = speaker_name is not None or speaker_code is not None

    if has_speaker:
        lines.append("Speaker majority dialect")
        lines.append(
            f"<div style='font-size: 1.18rem; font-weight: 800; line-height: 1.35;'>"
            f"{speaker_name or '-'} ({speaker_code or '-'})"
            f"</div>"
        )
        lines.append(
            "  Majority-vote dialect across multiple segments spoken by the same speaker."
        )
    if not has_speaker:
        lines.append("_Not available yet._")

    omni_dialect = record.get("omni_dialect")
    if isinstance(omni_dialect, str) and omni_dialect:
        lines.append(f"- Omni dialect: {omni_dialect}")
    return "\n".join(lines)


def _fmt_omni_markdown(record: dict[str, Any]) -> str:
    lines = ["### Omni Transcription"]

    omni_text = record.get("omni_text")
    if isinstance(omni_text, str) and omni_text:
        variant = record.get("omni_variant", "default")
        lines.append(f"- Selected variant: `{variant}`")
        lines.append(omni_text)
    else:
        lines.append("_Not available yet._")

    variants = record.get("omni_variants")
    if isinstance(variants, dict) and variants:
        lines.append("#### Variants")
        for variant_name in sorted(variants):
            payload = variants.get(variant_name)
            if not isinstance(payload, dict):
                continue
            variant_text = payload.get("omni_text")
            if not isinstance(variant_text, str) or not variant_text:
                continue
            lines.append(f"- `{variant_name}`: {variant_text}")
    return "\n".join(lines)


def _fmt_record_summary(record: dict[str, Any]) -> str:
    podcast, title = _derive_podcast_and_title(record)
    base_text = record.get("text")
    transcript = (
        base_text
        if isinstance(base_text, str) and base_text.strip()
        else "_Missing transcript text._"
    )
    lines = [
        "### Transcript",
        transcript,
        "",
        _fmt_dialect_markdown(record),
        "",
        _fmt_omni_markdown(record),
        "",
        "### Segment",
        f"- Podcast: {podcast or 'Unknown Podcast'}",
        f"- Episode: {title or 'Unknown Title'}",
        f"- Speaker: {record.get('speaker', '-')}",
        f"- Time: {_fmt_clock(record.get('start'))} -> {_fmt_clock(record.get('end'))}",
    ]
    return "\n".join(lines)


def create_app(default_manifest: str = DEFAULT_MANIFEST) -> gr.Blocks:
    browser: dict[str, Any] = {
        "store": None,
        "total_rows": 0,
        "eligible_rows": 0,
        "tagged_index": None,
    }
    fixed_manifest = Path(default_manifest).expanduser()
    tagged_manifest_path = fixed_manifest.with_name("manifest.tagged.jsonl")

    def _row_bundle(record: dict[str, Any]):
        audio = str(record.get("audio_path", ""))
        if not audio or not Path(audio).exists():
            audio = None
        tagged_index = browser.get("tagged_index")
        return (
            audio,
            _fmt_timeline_html(record, tagged_index),
            _fmt_record_summary(record),
            json.dumps(record, ensure_ascii=False, indent=2),
        )

    initial_status = ""
    initial_audio = None
    initial_timeline = "<div>No timeline available.</div>"
    initial_summary = "Manifest not loaded."
    initial_raw = ""

    if not fixed_manifest.exists():
        initial_status = f"Manifest not found: `{fixed_manifest}`"
    else:
        store = ManifestStore(fixed_manifest)
        total = store.build(require_annotations=True)
        browser["store"] = store
        browser["total_rows"] = total
        browser["eligible_rows"] = store.total_eligible_rows
        tagged_index = TaggedManifestIndex(tagged_manifest_path)
        tagged_index.build()
        browser["tagged_index"] = tagged_index
        if total == 0:
            initial_status = "Manifest is empty."
            initial_summary = "Manifest is empty."
        elif store.total_eligible_rows == 0:
            initial_status = (
                f"Loaded **{total:,}** rows, but none have both tags and emotions."
            )
            initial_summary = "No eligible row found."
        else:
            initial_status = (
                f"Loaded **{total:,}** rows. "
                f"Eligible (tags+emotions): **{store.total_eligible_rows:,}**."
            )
            picked = store.random_row()
            if picked is not None:
                _, row = picked
                initial_audio, initial_timeline, initial_summary, initial_raw = (
                    _row_bundle(row)
                )

    with gr.Blocks(title="Manifest Audio Browser") as app:
        gr.Markdown("## Manifest Audio Browser")
        random_btn = gr.Button("New Random Sample", variant="primary")

        status_md = gr.Markdown(initial_status)
        audio_player = gr.Audio(
            label="Segment Audio",
            type="filepath",
            value=initial_audio,
            elem_id="segment_audio_player",
        )
        timeline_md = gr.HTML(initial_timeline)
        summary_md = gr.Markdown(initial_summary)
        raw_json = gr.Code(label="Raw record JSON", language="json", value=initial_raw)

        def refresh_sample():
            store = browser.get("store")
            total_rows = int(browser.get("total_rows", 0))
            eligible_rows = int(browser.get("eligible_rows", 0))
            if store is None:
                return (
                    "Manifest unavailable.",
                    None,
                    "<div>No timeline available.</div>",
                    "Manifest missing.",
                    "",
                )
            picked = store.random_row()
            if picked is None:
                return (
                    "No eligible sample with both tags and emotions.",
                    None,
                    "<div>No timeline available.</div>",
                    "No row selected.",
                    "",
                )
            _, row = picked
            audio, timeline_html, summary, raw = _row_bundle(row)
            return (
                f"Loaded **{total_rows:,}** rows. "
                f"Eligible (tags+emotions): **{eligible_rows:,}**.",
                audio,
                timeline_html,
                summary,
                raw,
            )

        random_btn.click(
            refresh_sample,
            inputs=[],
            outputs=[status_md, audio_player, timeline_md, summary_md, raw_json],
            js=TIMELINE_SEEK_JS,
        )
        app.load(fn=None, inputs=None, outputs=None, js=TIMELINE_SEEK_JS)

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
        head=TIMELINE_SEEK_HEAD,
    )


if __name__ == "__main__":
    main()
