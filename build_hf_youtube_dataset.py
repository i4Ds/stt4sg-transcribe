#!/usr/bin/env python3
"""Build and optionally upload a YouTube podcast dataset to Hugging Face."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("build_hf_youtube_dataset")

DEFAULT_ROOT = "/mnt/nas05/data02/vincenzo/podcast_data/youtube"
DEFAULT_OUTPUT = "outputs/hf_youtube_podcast"
DEFAULT_REPO_ID = "i4ds/Youtube_Podcast"
DEFAULT_EXCLUDES = ["processed", "_previous", "Trüffelschweine_test"]
SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")


@dataclass
class SeriesStats:
    name: str
    json_files: int = 0
    audio_files: int = 0
    paired_files: int = 0
    episodes_used: int = 0
    total_duration_sec: float = 0.0
    total_segments: int = 0
    parse_errors: int = 0
    missing_audio_for_json: int = 0

    @property
    def hours(self) -> float:
        return self.total_duration_sec / 3600.0

    @property
    def status(self) -> str:
        if self.json_files == 0 and self.audio_files == 0:
            return "empty"
        if self.paired_files == self.json_files == self.audio_files and self.paired_files > 0:
            return "ready"
        if self.paired_files > 0:
            return "partial"
        return "unpaired"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a HF dataset from podcast folders with columns: audio, text, srt, "
            "segments, transcription."
        )
    )
    parser.add_argument("root_dir", nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude. Can be passed multiple times.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--token", default=None, help="HF token (or use HF_TOKEN env var)")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/upload private dataset repo (default: true).",
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms -= hours * 3_600_000
    minutes = total_ms // 60_000
    total_ms -= minutes * 60_000
    secs = total_ms // 1000
    millis = total_ms - secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        start = _to_float(seg.get("start"))
        end = _to_float(seg.get("end"))
        text = seg.get("text")
        if start is None or end is None:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        if end < start:
            start, end = end, start
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
        idx += 1
    return "\n".join(lines).strip() + ("\n" if lines else "")


def normalize_segments(raw_segments: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []

    normalized: list[dict[str, Any]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        start = _to_float(seg.get("start"))
        end = _to_float(seg.get("end"))
        text = seg.get("text")
        if start is None or end is None:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        if end < start:
            start, end = end, start
        duration = _to_float(seg.get("duration"))
        if duration is None:
            duration = max(0.0, end - start)

        out: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "text": text.strip(),
        }
        speaker = seg.get("speaker")
        if isinstance(speaker, str) and speaker:
            out["speaker"] = speaker
        normalized.append(out)
    return normalized


def _resolve_audio_path(json_path: Path, payload: dict[str, Any]) -> Path | None:
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    audio_from_json = payload.get("audio_file")
    if isinstance(audio_from_json, str) and audio_from_json:
        candidate = Path(audio_from_json).expanduser()
        if not candidate.is_absolute():
            candidate = (json_path.parent / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


def _canonical_config(config: dict[str, Any]) -> str:
    cleaned = dict(config)
    cleaned.pop("hf_token", None)
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)


def _duration_from_payload(
    transcription: dict[str, Any] | None,
    segments: list[dict[str, Any]],
) -> float:
    if isinstance(transcription, dict):
        dur = _to_float(transcription.get("duration"))
        if dur is not None and dur > 0:
            return dur
    max_end = 0.0
    for seg in segments:
        end = _to_float(seg.get("end"))
        if end is not None:
            max_end = max(max_end, end)
    return max_end


def _collect_series_dirs(root_dir: Path, excluded_names: set[str]) -> list[Path]:
    return sorted(
        p
        for p in root_dir.iterdir()
        if p.is_dir() and p.name not in excluded_names and not p.name.startswith(".")
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_readme(summary: dict[str, Any], include_plot: bool) -> str:
    totals = summary["totals"]
    rows = summary["series"]
    config_info = summary["config"]

    lines = [
        "# YouTube Podcast Dataset",
        "",
        "Swiss-German/Swiss podcast audio with segment-level transcripts and generated SRT.",
        "",
        "## Dataset Columns",
        "",
        "- `audio`: episode audio file",
        "- `text`: full episode text (`\"\\n\".join(segment.text)`) ",
        "- `srt`: segment-level SRT content",
        "- `segments`: minimal per-segment JSON (`start`, `end`, `duration`, `text`, `speaker`)",
        "- `transcription`: top-level transcription metadata from source JSON",
        "- `series`, `episode`, `id`, `duration_seconds`, `num_segments`",
        "",
        "## Scope",
        "",
        f"- Root: `{summary['root_dir']}`",
        f"- Excluded folders: {', '.join(summary['excluded_dirs'])}",
        f"- Series included: {totals['series_count']}",
        f"- Episodes included: {totals['episodes_included']}",
        f"- Total duration: {totals['total_hours']:.2f} hours",
        "",
        "## Processing Status By Podcast Series",
        "",
        "| Series | JSON | MP3 | Paired | Included | Hours | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['series']} | {row['json_files']} | {row['audio_files']} | "
            f"{row['paired_files']} | {row['episodes_used']} | {row['hours']:.2f} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Shared Transcription Config",
            "",
            (
                f"Detected {config_info['variant_count']} config variant(s). "
                f"Most common appears in {config_info['top_count']}/{totals['episodes_included']} episodes."
            ),
            "",
        ]
    )

    if config_info["top_config"] is not None:
        lines.extend(
            [
                "```json",
                json.dumps(config_info["top_config"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- SRT files are generated from `segments` timestamps only (no word-level timestamps).",
            "- Config is documented once here to avoid duplication in each episode row.",
            f"- Generated at: {summary['generated_at_utc']}",
        ]
    )

    if include_plot:
        lines.extend(["", "## Duration Plot", "", "![Series Hours](series_hours.png)"])

    return "\n".join(lines) + "\n"


def _save_plot(series_rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not installed; skipping plot.")
        return False

    plot_rows = [row for row in series_rows if row["hours"] > 0]
    if not plot_rows:
        LOGGER.warning("No non-zero durations available for plot.")
        return False

    plot_rows.sort(key=lambda x: x["hours"], reverse=True)

    labels = [row["series"] for row in plot_rows]
    values = [float(row["hours"]) for row in plot_rows]
    max_hours = max(values) if values else 1.0

    fig_h = max(4.5, 1.6 + 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=180)
    colors = [
        plt.cm.Blues(0.45 + (0.45 * i / max(1, len(values) - 1)))
        for i in range(len(values))
    ]
    bars = ax.barh(labels, values, color=colors, edgecolor="#1f3552", linewidth=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Hours")
    ax.set_title("Audio Hours per Podcast Series", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    label_offset = max_hours * 0.01
    for bar, hours in zip(bars, values):
        ax.text(
            hours + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{hours:.2f}h",
            va="center",
            ha="left",
            fontsize=9,
            color="#243447",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved plot: %s", output_path)
    return True


def _upload_to_hub(
    rows: list[dict[str, Any]],
    repo_id: str,
    private: bool,
    token: str | None,
    readme_path: Path,
    summary_path: Path,
    plot_path: Path | None,
) -> None:
    try:
        from datasets import Audio, Dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for upload. Install with: pip install datasets huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio())
    dataset.push_to_hub(repo_id=repo_id, private=private, token=token)
    LOGGER.info("Uploaded dataset rows to hf://datasets/%s", repo_id)

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="summary.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    if plot_path and plot_path.exists():
        api.upload_file(
            path_or_fileobj=str(plot_path),
            path_in_repo="series_hours.png",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    LOGGER.info("Uploaded README/summary assets to hf://datasets/%s", repo_id)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        LOGGER.error("Root folder does not exist: %s", root_dir)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_root = output_dir / "srt"
    json_subset_root = output_dir / "json_subset"
    rows_path = output_dir / "dataset_rows.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    plot_path = output_dir / "series_hours.png"

    excluded = set(DEFAULT_EXCLUDES)
    excluded.update(args.exclude_dir)

    series_dirs = _collect_series_dirs(root_dir, excluded)
    LOGGER.info("Series folders included: %d", len(series_dirs))

    rows: list[dict[str, Any]] = []
    series_stats: dict[str, SeriesStats] = {}
    config_counter: Counter[str] = Counter()
    config_lookup: dict[str, dict[str, Any]] = {}
    episodes_seen = 0

    for series_dir in series_dirs:
        stats = SeriesStats(name=series_dir.name)
        stats.json_files = len(list(series_dir.glob("*.json")))
        stats.audio_files = sum(len(list(series_dir.glob(f"*{ext}"))) for ext in SUPPORTED_AUDIO_EXTENSIONS)
        series_stats[series_dir.name] = stats

        for json_path in sorted(series_dir.glob("*.json")):
            if args.max_episodes is not None and episodes_seen >= args.max_episodes:
                break

            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                stats.parse_errors += 1
                LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, exc)
                continue

            audio_path = _resolve_audio_path(json_path, payload)
            if audio_path is None:
                stats.missing_audio_for_json += 1
                LOGGER.warning("Skipping %s (no matching audio file found)", json_path)
                continue
            stats.paired_files += 1

            config = payload.get("config")
            if isinstance(config, dict):
                canonical = _canonical_config(config)
                config_counter[canonical] += 1
                config_lookup.setdefault(canonical, json.loads(canonical))

            segments = normalize_segments(payload.get("segments"))
            transcription = payload.get("transcription")
            if not isinstance(transcription, dict):
                transcription = {}

            duration_seconds = _duration_from_payload(transcription, segments)
            stats.total_duration_sec += duration_seconds
            stats.total_segments += len(segments)
            stats.episodes_used += 1

            srt_text = segments_to_srt(segments)
            combined_text = "\n".join(seg["text"] for seg in segments)

            srt_path = srt_root / series_dir.name / f"{json_path.stem}.srt"
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text(srt_text, encoding="utf-8")

            json_subset = {
                "transcription": transcription,
                "segments": segments,
            }
            json_subset_path = json_subset_root / series_dir.name / f"{json_path.stem}.json"
            _write_json(json_subset_path, json_subset)

            row = {
                "id": f"{series_dir.name}/{json_path.stem}",
                "series": series_dir.name,
                "episode": json_path.stem,
                "audio": str(audio_path),
                "text": combined_text,
                "srt": srt_text,
                "segments": segments,
                "transcription": transcription,
                "duration_seconds": round(duration_seconds, 3),
                "num_segments": len(segments),
            }
            rows.append(row)
            episodes_seen += 1

    with rows_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote rows JSONL: %s (%d rows)", rows_path, len(rows))

    series_rows = []
    total_hours = 0.0
    for name in sorted(series_stats):
        stat = series_stats[name]
        total_hours += stat.hours
        series_rows.append(
            {
                "series": stat.name,
                "json_files": stat.json_files,
                "audio_files": stat.audio_files,
                "paired_files": stat.paired_files,
                "episodes_used": stat.episodes_used,
                "hours": round(stat.hours, 3),
                "segments": stat.total_segments,
                "parse_errors": stat.parse_errors,
                "missing_audio_for_json": stat.missing_audio_for_json,
                "status": stat.status,
            }
        )

    top_config = None
    top_count = 0
    if config_counter:
        top_key, top_count = config_counter.most_common(1)[0]
        top_config = config_lookup[top_key]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root_dir": str(root_dir),
        "excluded_dirs": sorted(excluded),
        "totals": {
            "series_count": len(series_dirs),
            "episodes_included": len(rows),
            "total_hours": round(total_hours, 3),
        },
        "config": {
            "variant_count": len(config_counter),
            "top_count": top_count,
            "top_config": top_config,
        },
        "series": series_rows,
    }
    _write_json(summary_path, summary)
    LOGGER.info("Wrote summary: %s", summary_path)

    include_plot = False
    if not args.skip_plot:
        include_plot = _save_plot(series_rows, plot_path)

    readme_text = _build_readme(summary, include_plot=include_plot)
    _write_readme(readme_path, readme_text)
    LOGGER.info("Wrote README: %s", readme_path)

    if args.upload:
        token = args.token or os.getenv("HF_TOKEN")
        if not token:
            LOGGER.info("No explicit HF token provided; using cached huggingface auth.")
        _upload_to_hub(
            rows=rows,
            repo_id=args.repo_id,
            private=bool(args.private),
            token=token,
            readme_path=readme_path,
            summary_path=summary_path,
            plot_path=plot_path if include_plot else None,
        )

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
