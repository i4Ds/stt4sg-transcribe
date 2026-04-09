#!/usr/bin/env python3
"""Transcribe manifest audio snippets with Gemma 4 and write sentence_ch JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

LOGGER = logging.getLogger("transcribe_manifest_sentence_ch_gemma")

DEFAULT_INPUT = Path(
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_combined_sliding.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_sentence_ch_gemma.jsonl"
)
DEFAULT_ERRORS_OUTPUT = Path(
    "/mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest_sentence_ch_gemma.errors.jsonl"
)
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_OUTPUT_KEY = "sentence_ch"
DEFAULT_PROMPT = "Transcribe the following speech segment in its original language (Swiss-German). Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSONL manifest with audio_path, transcribe each clip with Gemma 4, "
            "and write a separate JSONL with audio_path and sentence_ch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--audio-field", default="audio_path")
    parser.add_argument("--output-key", default=DEFAULT_OUTPUT_KEY)
    parser.add_argument("--errors-output", type=Path, default=DEFAULT_ERRORS_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite output file instead of appending and skipping completed rows.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per clip.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to from_pretrained().",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="torch_dtype passed to from_pretrained(); use auto, bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation, for example sdpa or flash_attention_2.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Optional HF token env var for gated model access.",
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
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {path}"
                )
            yield payload


def load_completed_audio_paths(output_path: Path, audio_field: str) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    for row in iter_jsonl(output_path):
        audio_path = row.get(audio_field)
        if isinstance(audio_path, str) and audio_path:
            completed.add(audio_path)
    return completed


def resolve_torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def build_messages(audio_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def patch_gemma4_audio_placeholder_count(processor: AutoProcessor) -> None:
    compute_fn = getattr(processor, "_compute_audio_num_tokens", None)
    audio_seq_length = getattr(processor, "audio_seq_length", None)
    if compute_fn is None or not callable(compute_fn):
        return
    if getattr(processor, "_stt4sg_audio_token_patch", False):
        return

    def _patched(audio_waveform: Any, sampling_rate: int) -> int:
        tokens = compute_fn(audio_waveform, sampling_rate)
        if isinstance(tokens, int) and tokens > 0:
            if isinstance(audio_seq_length, int):
                return min(tokens + 1, audio_seq_length)
            return tokens + 1
        return tokens

    processor._compute_audio_num_tokens = _patched  # type: ignore[method-assign]
    processor._stt4sg_audio_token_patch = True


def parse_generated_text(processor: AutoProcessor, output_tokens: torch.Tensor) -> str:
    decoded_raw = processor.decode(output_tokens, skip_special_tokens=False)
    parsed = None
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(decoded_raw)
        except Exception:
            parsed = None

    if isinstance(parsed, str):
        return parsed.replace("\n", " ").strip()
    if isinstance(parsed, dict):
        for key in ("text", "response", "content"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.replace("\n", " ").strip()
    if isinstance(parsed, list):
        parts = [item for item in parsed if isinstance(item, str) and item.strip()]
        if parts:
            return " ".join(parts).strip()

    return (
        processor.decode(output_tokens, skip_special_tokens=True)
        .replace("\n", " ")
        .strip()
    )


def transcribe_audio(
    *,
    processor: AutoProcessor,
    model: AutoModelForMultimodalLM,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(audio_path=audio_path, prompt=prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_tokens = outputs[0][input_len:]
    return parse_generated_text(processor=processor, output_tokens=output_tokens)


def is_audio_token_mismatch_error(exc: Exception) -> bool:
    return isinstance(
        exc, ValueError
    ) and "Audio features and audio tokens do not match" in str(exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv(Path(__file__).resolve().with_name(".env"))

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
    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        set()
        if args.overwrite
        else load_completed_audio_paths(args.output, args.audio_field)
    )
    output_mode = "w" if args.overwrite else "a"
    errors_mode = "w" if args.overwrite else "a"

    hf_token = os.getenv(args.hf_token_env) or None
    dtype = resolve_torch_dtype(args.dtype)

    LOGGER.info("Loading processor for %s", args.model_id)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        token=hf_token,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "token": hf_token,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    LOGGER.info("Loading model %s", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    processed = 0
    skipped = 0
    failed = 0
    with args.output.open(output_mode, encoding="utf-8") as handle:
        with args.errors_output.open(errors_mode, encoding="utf-8") as err_handle:
            for index, row in enumerate(records, start=1):
                audio_path_raw = row.get(args.audio_field)
                if not isinstance(audio_path_raw, str) or not audio_path_raw:
                    LOGGER.warning(
                        "Skipping row %s without %s", index, args.audio_field
                    )
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
                try:
                    transcript = transcribe_audio(
                        processor=processor,
                        model=model,
                        audio_path=str(audio_path),
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    failed += 1
                    error_payload = {
                        args.audio_field: str(audio_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    err_handle.write(
                        json.dumps(error_payload, ensure_ascii=False) + "\n"
                    )
                    err_handle.flush()
                    if is_audio_token_mismatch_error(exc):
                        LOGGER.warning(
                            "Skipping audio token mismatch for %s: %s", audio_path, exc
                        )
                        continue
                    raise

                payload = {
                    args.audio_field: str(audio_path),
                    args.output_key: transcript,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()

                completed.add(audio_path_raw)
                processed += 1
                LOGGER.info("Saved transcript with %s characters", len(transcript))

    LOGGER.info(
        "Finished. processed=%s skipped=%s failed=%s output=%s errors=%s",
        processed,
        skipped,
        failed,
        args.output,
        args.errors_output,
    )


if __name__ == "__main__":
    main()
