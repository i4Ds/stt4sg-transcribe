#!/usr/bin/env python3
"""Quick Gradio browser for large JSONL manifests with audio playback."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import gradio as gr

DEFAULT_MANIFEST = (
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.jsonl"
)


class ManifestIndex:
    """Byte-offset index for fast random access to JSONL rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self.total_rows = 0

    def build(self) -> int:
        self.offsets.clear()
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.total_rows = len(self.offsets)
        return self.total_rows

    def get_row(self, row_id: int) -> dict[str, Any]:
        if row_id < 0 or row_id >= self.total_rows:
            raise IndexError(f"row_id out of bounds: {row_id}")
        with self.path.open("rb") as f:
            f.seek(self.offsets[row_id])
            raw = f.readline().decode("utf-8")
        return json.loads(raw)


def _derive_podcast_and_title(record: dict[str, Any]) -> tuple[str, str]:
    source_audio = record.get("source_audio")
    if isinstance(source_audio, str) and source_audio:
        source_path = Path(source_audio)
        return source_path.parent.name, source_path.stem

    audio_path = record.get("audio_path")
    if isinstance(audio_path, str) and audio_path:
        segment_path = Path(audio_path)
        title = segment_path.parent.name
        podcast = segment_path.parent.parent.name if len(segment_path.parents) > 1 else ""
        return podcast, title
    return "", ""


def _fmt_time(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "-"


def _row_label(row_id: int, record: dict[str, Any]) -> str:
    podcast, title = _derive_podcast_and_title(record)
    speaker = str(record.get("speaker", "-"))
    start = _fmt_time(record.get("start"))
    end = _fmt_time(record.get("end"))
    duration = _fmt_time(record.get("duration"))
    return (
        f"{row_id} | {podcast or 'Unknown Podcast'} | {title or 'Unknown Title'} | "
        f"{speaker} | {start} -> {end} ({duration})"
    )


def _sample_row_ids(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    n = max(1, min(n, total))
    return random.sample(range(total), n)


def create_app(default_manifest: str = DEFAULT_MANIFEST) -> gr.Blocks:
    browser: dict[str, Any] = {"index": None}

    with gr.Blocks(title="Manifest Audio Browser") as app:
        gr.Markdown("## Manifest Audio Browser")

        with gr.Row():
            manifest_path = gr.Textbox(
                label="Manifest path",
                value=default_manifest,
                lines=1,
            )
            sample_size = gr.Slider(
                label="Random sample size",
                minimum=10,
                maximum=1000,
                value=120,
                step=10,
            )

        with gr.Row():
            load_btn = gr.Button("Load Manifest", variant="primary")
            random_btn = gr.Button("New Random Sample")

        status_md = gr.Markdown("Load a manifest to start browsing.")
        row_choices = gr.Dropdown(
            label="Sampled rows",
            choices=[],
            value=None,
            interactive=True,
        )
        audio_player = gr.Audio(label="Segment Audio", type="filepath")
        raw_json = gr.Code(label="Raw record JSON", language="json")

        def load_manifest(path: str, n: int):
            manifest = Path(path).expanduser()
            if not manifest.exists():
                browser["index"] = None
                return (
                    f"Manifest not found: `{manifest}`",
                    gr.update(choices=[], value=None),
                    None,
                    "",
                )

            idx = ManifestIndex(manifest)
            total = idx.build()
            browser["index"] = idx

            if total == 0:
                return (
                    f"Loaded `{manifest}` but it contains 0 records.",
                    gr.update(choices=[], value=None),
                    None,
                    "",
                )

            sampled_ids = _sample_row_ids(total, int(n))
            options = []
            for row_id in sampled_ids:
                record = idx.get_row(row_id)
                label = _row_label(row_id, record)
                options.append((label, row_id))

            first_row = sampled_ids[0]
            first_record = idx.get_row(first_row)
            first_audio = str(first_record.get("audio_path", ""))
            if not first_audio or not Path(first_audio).exists():
                first_audio = None

            return (
                f"Loaded `{manifest}` with **{total:,}** rows. Sampled **{len(sampled_ids)}** rows.",
                gr.update(choices=options, value=first_row),
                first_audio,
                json.dumps(first_record, ensure_ascii=False, indent=2),
            )

        def refresh_sample(n: int):
            idx = browser.get("index")
            if idx is None:
                return (
                    "Load a manifest first.",
                    gr.update(choices=[], value=None),
                    None,
                    "",
                )

            sampled_ids = _sample_row_ids(idx.total_rows, int(n))
            options = []
            for row_id in sampled_ids:
                record = idx.get_row(row_id)
                label = _row_label(row_id, record)
                options.append((label, row_id))

            first_row = sampled_ids[0]
            first_record = idx.get_row(first_row)
            first_audio = str(first_record.get("audio_path", ""))
            if not first_audio or not Path(first_audio).exists():
                first_audio = None

            return (
                f"Resampled **{len(sampled_ids)}** rows from **{idx.total_rows:,}** total.",
                gr.update(choices=options, value=first_row),
                first_audio,
                json.dumps(first_record, ensure_ascii=False, indent=2),
            )

        def select_row(row_id: int | None):
            idx = browser.get("index")
            if idx is None or row_id is None:
                return None, ""
            record = idx.get_row(int(row_id))
            audio = str(record.get("audio_path", ""))
            if not audio or not Path(audio).exists():
                audio = None
            return (
                audio,
                json.dumps(record, ensure_ascii=False, indent=2),
            )

        load_btn.click(
            load_manifest,
            inputs=[manifest_path, sample_size],
            outputs=[status_md, row_choices, audio_player, raw_json],
        )
        random_btn.click(
            refresh_sample,
            inputs=[sample_size],
            outputs=[status_md, row_choices, audio_player, raw_json],
        )
        row_choices.change(
            select_row,
            inputs=[row_choices],
            outputs=[audio_player, raw_json],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast Gradio browser for JSONL manifests with audio playback."
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
        help="Enable Gradio share link",
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
