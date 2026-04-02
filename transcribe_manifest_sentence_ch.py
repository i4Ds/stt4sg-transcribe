#!/usr/bin/env python3
"""Transcribe manifest audio snippets to Swiss German and write sentence_ch JSONL."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import backoff
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    UnprocessableEntityError,
)

LOGGER = logging.getLogger("transcribe_manifest_sentence_ch")

DEFAULT_INPUT = Path(
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_combined_sliding_sample.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_sentence_ch_sample.jsonl"
)
DEFAULT_BASE_URL = (
    "https://ws-ef97yffwb7pl5qzh.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL = "qwen3.5-omni-plus"
DEFAULT_PROMPT = (
    "Transkribiere wortgetreu ins Schweizerdeutsche; exakt wie gesprochen. "
    "Antworte nur mit der Transkription. "
    "Keine Einleitung, keine Erklärungen, keine Übersetzung, keine Klammern, "
    "keine Labels. "
    "Behalte Dialekt, Füllwörter, Wiederholungen und gesprochene Formen exakt bei. "
    "Wenn nichts Verständliches gesagt wird, antworte mit einer leeren Zeichenkette."
)
SUPPORTED_FORMATS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".m4a": "m4a",
    ".aac": "aac",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".opus": "ogg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSONL manifest with audio_path, transcribe each clip to verbatim "
            "Swiss German, and write a separate JSONL with audio_path and sentence_ch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite output file instead of appending and skipping completed rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between requests to throttle if needed.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry failed API calls this many times before giving up.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP timeout passed to the OpenAI client.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            yield payload


def load_completed_audio_paths(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    for row in iter_jsonl(output_path):
        audio_path = row.get("audio_path")
        if isinstance(audio_path, str) and audio_path:
            completed.add(audio_path)
    return completed


def audio_format_for_path(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    fmt = SUPPORTED_FORMATS.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unsupported audio suffix {audio_path.suffix!r} for {audio_path}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return fmt


def audio_to_base64(audio_path: Path) -> tuple[str, str]:
    fmt = audio_format_for_path(audio_path)
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return encoded, fmt


def extract_text_delta(delta: Any) -> str:
    if delta is None:
        return ""

    if isinstance(delta, str):
        return delta

    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                out.append(text)
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    out.append(maybe_text)
        return "".join(out)

    if isinstance(delta, dict):
        maybe_content = delta.get("content")
        if isinstance(maybe_content, str):
            return maybe_content
        if isinstance(maybe_content, list):
            out: list[str] = []
            for item in maybe_content:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    maybe_text = item.get("text")
                    if isinstance(maybe_text, str):
                        out.append(maybe_text)
            return "".join(out)
    return ""


def normalize_transcript(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _should_give_up(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            BadRequestError,
            UnprocessableEntityError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code < 500 and status_code not in {408, 409, 429}
    return False


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    prompt: str,
    model: str,
    max_retries: int,
) -> str:
    encoded_audio, audio_format = audio_to_base64(audio_path)

    def _log_backoff(details: dict[str, Any]) -> None:
        exc = details.get("exception")
        LOGGER.warning(
            "Attempt %s/%s failed for %s: %s. Retrying in %.1fs.",
            details["tries"],
            max_retries,
            audio_path,
            exc,
            details["wait"],
        )

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APITimeoutError, APIStatusError),
        max_tries=max_retries,
        giveup=_should_give_up,
        jitter=backoff.full_jitter,
        on_backoff=_log_backoff,
    )
    def _request() -> str:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein ASR-System fuer Schweizerdeutsch. "
                        "Gib nur die wortgetreue Transkription zurueck."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": audio_format,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )

        parts: list[str] = []
        for chunk in stream:
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                text = extract_text_delta(delta)
                if text:
                    parts.append(text)

        return normalize_transcript("".join(parts))

    return _request()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv(Path(__file__).resolve().with_name(".env"))

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key in environment variable {args.api_key_env!r}. "
            "Set it in the environment or in .env."
        )

    if not args.input.exists():
        raise SystemExit(f"Input manifest not found: {args.input}")

    records = list(iter_jsonl(args.input))
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        LOGGER.info("No records selected from %s", args.input)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_completed_audio_paths(args.output)
    output_mode = "w" if args.overwrite else "a"

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout_seconds,
    )

    processed = 0
    skipped = 0
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for index, row in enumerate(records, start=1):
            audio_path_raw = row.get("audio_path")
            if not isinstance(audio_path_raw, str) or not audio_path_raw:
                LOGGER.warning("Skipping row %s without audio_path", index)
                skipped += 1
                continue

            if audio_path_raw in completed:
                LOGGER.info("Skipping already completed: %s", audio_path_raw)
                skipped += 1
                continue

            audio_path = Path(audio_path_raw)
            if not audio_path.exists():
                LOGGER.warning("Skipping missing audio file: %s", audio_path)
                skipped += 1
                continue

            LOGGER.info("Transcribing %s/%s: %s", index, len(records), audio_path)
            sentence_ch = transcribe_audio(
                client=client,
                audio_path=audio_path,
                prompt=args.prompt,
                model=args.model,
                max_retries=args.max_retries,
            )

            payload = {
                "audio_path": str(audio_path),
                "sentence_ch": sentence_ch,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

            completed.add(audio_path_raw)
            processed += 1
            LOGGER.info("Saved transcript with %s characters", len(sentence_ch))

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    LOGGER.info(
        "Finished. processed=%s skipped=%s output=%s",
        processed,
        skipped,
        args.output,
    )


if __name__ == "__main__":
    main()
