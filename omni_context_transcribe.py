#!/usr/bin/env python3
"""Dialect-aware omnilingual-asr transcription with HF in-context examples."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.compute as pc

LOGGER = logging.getLogger("omni_context_transcribe")


DIALECT_NAME_ALIASES: Dict[str, str] = {
    # Swissdial client_id codes
    "be": "be",
    "bs": "bs",
    "zh": "zh",
    "lu": "lu",
    "sg": "sg",
    "ag": "ag",
    "gr": "gr",
    "vs": "vs",
    # Spoken-dialect names in manifest_with_speaker_dialect.jsonl
    "bern": "be",
    "basel": "bs",
    "zurich": "zh",
    "zürich": "zh",
    "innerschweiz": "lu",
    "zentralschweiz": "lu",
    "ostschweiz": "sg",
    "aargau": "ag",
    "graubunden": "gr",
    "graubünden": "gr",
    "wallis": "vs",
    # Non-swissdial language labels; these do not match Swissdial dialect pools.
    "deutsch": "de",
    "hochdeutsch": "de",
    "englisch": "en",
    "französisch": "fr",
    "italienisch": "it",
}


@dataclass
class ManifestRecord:
    line_number: int
    payload: Dict[str, Any]
    audio_input: str
    dialect: str


@dataclass
class ContextRef:
    split: str
    row_idx: int
    dialect: str
    text: str
    filter_text: str
    audio_path: Optional[str]


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_aliases(raw: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if not raw:
        return aliases
    for part in _split_csv(raw):
        if "=" not in part:
            raise ValueError(
                f"Invalid alias entry '{part}'. Use format SRC=DST, e.g. BS=bs"
            )
        src, dst = part.split("=", 1)
        aliases[src.strip().lower()] = dst.strip().lower()
    return aliases


def _normalize_dialect(value: Any, aliases: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    normalized_ascii = (
        unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    )
    if lowered in aliases:
        return aliases[lowered]
    if normalized_ascii in aliases:
        return aliases[normalized_ascii]
    return lowered


def _resolve_path(raw_path: str, base_dir: Path) -> str:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _iter_manifest_records(
    manifest_path: Path,
    audio_fields: List[str],
    dialect_fields: List[str],
    aliases: Dict[str, str],
    default_dialect: Optional[str],
    skip_missing_dialect: bool,
) -> Iterable[ManifestRecord]:
    default_norm = _normalize_dialect(default_dialect, aliases)
    base_dir = manifest_path.parent.resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping line %d: JSON decode error: %s", line_no, exc)
                continue

            audio_input: Optional[str] = None
            for field in audio_fields:
                if payload.get(field):
                    audio_input = _resolve_path(str(payload[field]), base_dir)
                    break
            if not audio_input:
                LOGGER.warning(
                    "Skipping line %d: missing audio field in %s", line_no, audio_fields
                )
                continue

            dialect_value = None
            for field in dialect_fields:
                if payload.get(field) is not None:
                    dialect_value = payload[field]
                    break

            dialect = _normalize_dialect(dialect_value, aliases) or default_norm
            if not dialect:
                if skip_missing_dialect:
                    LOGGER.debug("Skipping line %d: missing dialect", line_no)
                    continue
                raise ValueError(
                    f"Line {line_no} has no dialect (checked {dialect_fields}) and no --default-dialect"
                )

            yield ManifestRecord(
                line_number=line_no,
                payload=payload,
                audio_input=audio_input,
                dialect=dialect,
            )


def _build_context_index(
    hf_dataset_path: Path,
    splits: List[str],
    dialect_fields: List[str],
    text_fields: List[str],
    filter_text_fields: List[str],
    audio_field: str,
    aliases: Dict[str, str],
    context_text_regex: Optional[str] = None,
) -> tuple[Any, Dict[str, List[ContextRef]], str, str, str]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `datasets`. Install it first (e.g. `uv add datasets`)."
        ) from exc

    dataset_dict = load_from_disk(str(hf_dataset_path))

    index: Dict[str, List[ContextRef]] = defaultdict(list)
    chosen_dialect_field: Optional[str] = None
    chosen_text_field: Optional[str] = None
    chosen_filter_text_field: Optional[str] = None

    text_pattern = re.compile(context_text_regex) if context_text_regex else None

    for split in splits:
        if split not in dataset_dict:
            raise ValueError(
                f"Split '{split}' not found in dataset. Available: {list(dataset_dict.keys())}"
            )

        ds_split = dataset_dict[split]
        column_names = set(ds_split.column_names)

        dialect_field = next((x for x in dialect_fields if x in column_names), None)
        if dialect_field is None:
            raise ValueError(
                f"Could not find dialect field in split '{split}'. Tried: {dialect_fields}. Found: {ds_split.column_names}"
            )

        text_field = next((x for x in text_fields if x in column_names), None)
        if text_field is None:
            raise ValueError(
                f"Could not find context text field in split '{split}'. Tried: {text_fields}. Found: {ds_split.column_names}"
            )
        filter_text_field = next((x for x in filter_text_fields if x in column_names), None)
        if filter_text_field is None:
            filter_text_field = text_field

        if audio_field not in column_names:
            raise ValueError(
                f"Could not find audio field '{audio_field}' in split '{split}'. Found: {ds_split.column_names}"
            )

        chosen_dialect_field = chosen_dialect_field or dialect_field
        chosen_text_field = chosen_text_field or text_field
        chosen_filter_text_field = chosen_filter_text_field or filter_text_field

        table = ds_split._data.table
        dialect_col = table[dialect_field]
        text_col = table[text_field]
        filter_text_col = table[filter_text_field]

        # Access nested audio.path without materializing waveform arrays.
        path_col = None
        try:
            path_col = pc.struct_field(table[audio_field], "path")
        except Exception:
            path_col = None

        for i in range(len(table)):
            dialect = _normalize_dialect(dialect_col[i].as_py(), aliases)
            if not dialect:
                continue
            text = text_col[i].as_py()
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            filter_text = filter_text_col[i].as_py()
            if filter_text is None:
                continue
            filter_text = str(filter_text).strip()
            if not filter_text:
                continue
            if text_pattern and not text_pattern.search(filter_text):
                continue
            audio_path = None
            if path_col is not None:
                audio_path = path_col[i].as_py()
            index[dialect].append(
                ContextRef(
                    split=split,
                    row_idx=i,
                    dialect=dialect,
                    text=text,
                    filter_text=filter_text,
                    audio_path=audio_path,
                )
            )

    if (
        chosen_dialect_field is None
        or chosen_text_field is None
        or chosen_filter_text_field is None
    ):
        raise RuntimeError("No valid splits were indexed.")

    return (
        dataset_dict,
        index,
        chosen_dialect_field,
        chosen_text_field,
        chosen_filter_text_field,
    )


def _resolve_hf_audio_path(
    raw_path: Optional[str],
    candidate_roots: List[Path],
) -> Optional[str]:
    if not raw_path:
        return None

    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return str(path)

    for root in candidate_roots:
        cand = (root / raw_path).resolve()
        if cand.exists():
            return str(cand)

    return None


def _build_context_examples(
    refs: List[ContextRef],
    dataset_dict: Any,
    hf_audio_field: str,
    context_audio_mode: str,
    hf_audio_roots: List[Path],
    use_omni_types: bool = True,
) -> tuple[List[Any], List[Dict[str, Any]]]:
    ContextExample = None
    if use_omni_types:
        try:
            from omnilingual_asr.models.inference.pipeline import ContextExample
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `omnilingual-asr`. Install it first."
            ) from exc

    examples: List[Any] = []
    metadata: List[Dict[str, Any]] = []

    for ref in refs:
        row = dataset_dict[ref.split][ref.row_idx]
        audio = row[hf_audio_field]

        resolved_path = _resolve_hf_audio_path(ref.audio_path, hf_audio_roots)
        audio_input: Any = None

        if context_audio_mode == "path":
            if not resolved_path:
                raise FileNotFoundError(
                    f"Context row {ref.split}[{ref.row_idx}] has unresolved audio path '{ref.audio_path}'."
                )
            audio_input = resolved_path
        else:
            arr = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate") if isinstance(audio, dict) else None
            if arr is not None and sr is not None:
                audio_input = {"waveform": arr, "sample_rate": int(sr)}
            elif resolved_path:
                audio_input = resolved_path
            else:
                raise ValueError(
                    f"Context row {ref.split}[{ref.row_idx}] has neither decoded waveform nor a resolvable path."
                )

        if use_omni_types:
            examples.append(ContextExample(audio_input, ref.text))
        else:
            examples.append({"audio": audio_input, "text": ref.text})
        metadata.append(
            {
                "split": ref.split,
                "row_idx": ref.row_idx,
                "dialect": ref.dialect,
                "text": ref.text,
                "audio_path": resolved_path or ref.audio_path,
            }
        )

    return examples, metadata


def _read_resume_lines(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            line_no = obj.get("manifest_line")
            if isinstance(line_no, int):
                done.add(line_no)
    return done


def _chunked(
    items: List[ManifestRecord], batch_size: int
) -> Iterable[List[ManifestRecord]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sample_context_refs(
    rng: random.Random,
    pool: List[ContextRef],
    number_pool: List[ContextRef],
    context_size: int,
    number_ratio: float,
    previous_keys: Optional[set[tuple[str, int]]] = None,
) -> List[ContextRef]:
    k = min(context_size, len(pool))
    if k <= 0:
        return []

    effective_ratio = min(max(number_ratio, 0.0), 1.0)
    target_num = min(int(round(k * effective_ratio)), len(number_pool), k)
    if effective_ratio > 0.0 and target_num == 0 and number_pool:
        target_num = 1

    selected: List[ContextRef] = []
    for attempt in range(4):
        selected.clear()
        selected_keys: set[tuple[str, int]] = set()

        if target_num > 0:
            nums = rng.sample(number_pool, k=target_num)
            selected.extend(nums)
            selected_keys.update((x.split, x.row_idx) for x in nums)

        remaining = k - len(selected)
        if remaining > 0:
            candidates = [x for x in pool if (x.split, x.row_idx) not in selected_keys]
            selected.extend(rng.sample(candidates, k=remaining))

        rng.shuffle(selected)
        current_keys = {(x.split, x.row_idx) for x in selected}
        if previous_keys is None or current_keys != previous_keys or len(pool) <= k:
            break

    return list(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a manifest with Omni ASR using dialect-matched HF context examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("manifest", type=Path, help="Path to input manifest.jsonl")
    parser.add_argument(
        "--hf-dataset-path",
        type=Path,
        default=Path("all"),
        help="Path to local Hugging Face dataset directory (load_from_disk format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path (default: <manifest>.omni.jsonl)",
    )

    parser.add_argument("--model-card", default="omniASR_LLM_7B_ZS")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--context-size",
        type=int,
        default=10,
        help="Number of context examples to sample per dialect batch (1..10 recommended for ZS model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible context sampling/shuffling",
    )
    parser.add_argument(
        "--shuffle-within-dialect",
        action="store_true",
        help="Shuffle samples inside each dialect group before batching",
    )

    parser.add_argument(
        "--manifest-audio-fields",
        default="audio_path,path,audio_file",
        help="Comma-separated candidate audio-path fields in manifest",
    )
    parser.add_argument(
        "--manifest-dialect-fields",
        default="dialect_speaker_majority_name,dialect_speaker_majority,dialect_segment_name,dialect_segment,dialect,dialect_tag,client_id",
        help="Comma-separated candidate dialect fields in manifest",
    )
    parser.add_argument(
        "--default-dialect",
        help="Fallback dialect used when manifest entries have no dialect field",
    )
    parser.add_argument(
        "--skip-missing-dialect",
        action="store_true",
        help="Skip manifest rows that have no dialect and no --default-dialect",
    )
    parser.add_argument(
        "--dialect-aliases",
        default="",
        help="Comma-separated aliases SRC=DST, e.g. BS=bs,ZH=zh",
    )

    parser.add_argument(
        "--hf-splits",
        default="train,validation,test",
        help="Comma-separated HF splits to use as context source",
    )
    parser.add_argument(
        "--hf-dialect-fields",
        default="client_id,dialect,dialect_tag",
        help="Comma-separated candidate dialect fields in HF dataset",
    )
    parser.add_argument(
        "--hf-text-fields",
        default="sentence_ch,sentence_de,text,sentence,transcript",
        help="Comma-separated candidate transcript fields in HF dataset used as context text passed to Omni",
    )
    parser.add_argument(
        "--hf-filter-text-fields",
        default="sentence_de,sentence_ch,text,sentence,transcript",
        help="Comma-separated candidate transcript fields used for context filtering/number detection",
    )
    parser.add_argument(
        "--context-text-regex",
        help="Optional regex; only HF context texts matching it are eligible",
    )
    parser.add_argument(
        "--context-number-like",
        action="store_true",
        help="Prefer number-like HF context texts. If --context-number-ratio is not set, this implies ratio=1.0",
    )
    parser.add_argument(
        "--context-number-ratio",
        type=float,
        default=0.0,
        help="Target fraction of context examples per batch that should be number-like (0.0..1.0)",
    )
    parser.add_argument(
        "--hf-audio-field",
        default="audio",
        help="Audio column name in HF dataset",
    )
    parser.add_argument(
        "--context-audio-mode",
        choices=["waveform", "path"],
        default="waveform",
        help="How to pass context audio into ContextExample",
    )
    parser.add_argument(
        "--hf-audio-roots",
        default="",
        help="Comma-separated roots to resolve relative HF audio paths (used when --context-audio-mode=path)",
    )

    parser.add_argument(
        "--allow-global-context-fallback",
        action="store_true",
        help="If a dialect has no context pool, sample from all dialects",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping manifest lines already present in output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many manifest rows after filtering/resume",
    )
    parser.add_argument(
        "--save-context-in-output",
        action="store_true",
        help="Save the sampled in-context examples (dialect/text) into each output row",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do all indexing/sampling/batching but skip model inference",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    save_context_in_output = args.save_context_in_output or (args.limit is not None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.context_size < 1:
        raise ValueError("--context-size must be >= 1")

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    hf_dataset_path = args.hf_dataset_path.expanduser().resolve()
    if not hf_dataset_path.exists():
        raise FileNotFoundError(f"HF dataset path not found: {hf_dataset_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else manifest_path.with_suffix(".omni.jsonl")
    )

    aliases = dict(DIALECT_NAME_ALIASES)
    aliases.update(_parse_aliases(args.dialect_aliases))
    manifest_audio_fields = _split_csv(args.manifest_audio_fields)
    manifest_dialect_fields = _split_csv(args.manifest_dialect_fields)

    hf_splits = _split_csv(args.hf_splits)
    hf_dialect_fields = _split_csv(args.hf_dialect_fields)
    hf_text_fields = _split_csv(args.hf_text_fields)
    hf_filter_text_fields = _split_csv(args.hf_filter_text_fields)

    number_like_regex = (
        r"(?i)(\d|\b(eis|ein|eine|eins|zwei|zwöi|zwo|drü|drei|vier|füf|fünf|"
        r"sechs|sibe|sieben|acht|nün|neun|zäh|zehn|elf|zwölf|hundert|tausend)\b)"
    )
    context_text_regex = args.context_text_regex
    context_number_ratio = args.context_number_ratio
    if args.context_number_like and context_number_ratio <= 0.0:
        context_number_ratio = 1.0
    if context_number_ratio < 0.0 or context_number_ratio > 1.0:
        raise ValueError("--context-number-ratio must be in [0.0, 1.0]")
    number_pattern = re.compile(number_like_regex)

    hf_audio_roots = [
        Path(x).expanduser().resolve() for x in _split_csv(args.hf_audio_roots)
    ]
    if not hf_audio_roots:
        hf_audio_roots = [hf_dataset_path]

    done_lines = _read_resume_lines(output_path) if args.resume else set()
    if done_lines:
        LOGGER.info(
            "Resume enabled: %d lines already done in %s", len(done_lines), output_path
        )

    manifest_records = [
        x
        for x in _iter_manifest_records(
            manifest_path=manifest_path,
            audio_fields=manifest_audio_fields,
            dialect_fields=manifest_dialect_fields,
            aliases=aliases,
            default_dialect=args.default_dialect,
            skip_missing_dialect=args.skip_missing_dialect,
        )
        if x.line_number not in done_lines
    ]
    if args.limit is not None:
        manifest_records = manifest_records[: max(args.limit, 0)]

    if not manifest_records:
        LOGGER.warning("No manifest rows to process.")
        return

    rng = random.Random(args.seed)

    (
        dataset_dict,
        context_index,
        hf_dialect_col,
        hf_text_col,
        hf_filter_text_col,
    ) = _build_context_index(
        hf_dataset_path=hf_dataset_path,
        splits=hf_splits,
        dialect_fields=hf_dialect_fields,
        text_fields=hf_text_fields,
        filter_text_fields=hf_filter_text_fields,
        audio_field=args.hf_audio_field,
        aliases=aliases,
        context_text_regex=context_text_regex,
    )

    all_context_refs = [x for refs in context_index.values() for x in refs]
    if not all_context_refs:
        raise RuntimeError("No usable context rows found in HF dataset.")
    number_context_index: Dict[str, List[ContextRef]] = defaultdict(list)
    for dialect, refs in context_index.items():
        number_context_index[dialect] = [
            x for x in refs if number_pattern.search(x.filter_text)
        ]

    grouped_manifest: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for rec in manifest_records:
        grouped_manifest[rec.dialect].append(rec)

    dialects = sorted(grouped_manifest.keys())

    LOGGER.info("Manifest rows to process: %d", len(manifest_records))
    LOGGER.info(
        "Dialect groups in manifest: %s",
        ", ".join(f"{d}:{len(grouped_manifest[d])}" for d in dialects),
    )
    LOGGER.info("HF context dialect field: %s", hf_dialect_col)
    LOGGER.info("HF context text field: %s", hf_text_col)
    LOGGER.info("HF filter text field: %s", hf_filter_text_col)
    LOGGER.info(
        "HF context pools: %d dialects, %d rows",
        len(context_index),
        len(all_context_refs),
    )
    if context_number_ratio > 0.0:
        total_number_like = sum(len(v) for v in number_context_index.values())
        LOGGER.info(
            "Number-like context target ratio: %.2f (%d number-like rows available)",
            context_number_ratio,
            total_number_like,
        )

    pipeline = None
    if not args.dry_run:
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `omnilingual-asr`. Install it first."
            ) from exc
        pipeline = ASRInferencePipeline(model_card=args.model_card)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    previous_context_keys_by_dialect: Dict[str, set[tuple[str, int]]] = {}
    output_mode = "a" if args.resume else "w"
    with output_path.open(output_mode, encoding="utf-8") as out_f:
        for dialect in dialects:
            records = grouped_manifest[dialect]
            if args.shuffle_within_dialect:
                rng.shuffle(records)

            dialect_pool = context_index.get(dialect)
            if not dialect_pool:
                if not args.allow_global_context_fallback:
                    LOGGER.warning(
                        "Skipping dialect '%s': no matching context examples. Use --allow-global-context-fallback to fallback.",
                        dialect,
                    )
                    continue
                dialect_pool = all_context_refs
                LOGGER.warning(
                    "Dialect '%s' not found in HF context pool. Falling back to global random contexts.",
                    dialect,
                )
            number_pool = number_context_index.get(dialect, [])
            if dialect_pool is all_context_refs:
                number_pool = [
                    x for x in all_context_refs if number_pattern.search(x.filter_text)
                ]

            for batch in _chunked(records, args.batch_size):
                sampled_refs = _sample_context_refs(
                    rng=rng,
                    pool=dialect_pool,
                    number_pool=number_pool,
                    context_size=args.context_size,
                    number_ratio=context_number_ratio,
                    previous_keys=previous_context_keys_by_dialect.get(dialect),
                )
                previous_context_keys_by_dialect[dialect] = {
                    (x.split, x.row_idx) for x in sampled_refs
                }
                number_context_count = sum(
                    1 for x in sampled_refs if number_pattern.search(x.filter_text)
                )

                context_examples, context_meta = _build_context_examples(
                    refs=sampled_refs,
                    dataset_dict=dataset_dict,
                    hf_audio_field=args.hf_audio_field,
                    context_audio_mode=args.context_audio_mode,
                    hf_audio_roots=hf_audio_roots,
                    use_omni_types=not args.dry_run,
                )

                audio_inputs = [x.audio_input for x in batch]

                if args.dry_run:
                    transcriptions = [None] * len(audio_inputs)
                else:
                    assert pipeline is not None
                    context_batch = [context_examples for _ in range(len(audio_inputs))]
                    transcriptions = pipeline.transcribe_with_context(
                        audio_inputs,
                        context_examples=context_batch,
                        batch_size=len(audio_inputs),
                    )

                for rec, transcription in zip(batch, transcriptions):
                    src_text = rec.payload.get("text")
                    row = {
                        "audio_path": rec.audio_input,
                        "dialect": rec.dialect,
                        "text": src_text,
                        "omni_text": transcription,
                    }
                    if save_context_in_output:
                        row["context_examples"] = [
                            {"dialect": m["dialect"], "text": m["text"]}
                            for m in context_meta
                        ]
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed += 1

                out_f.flush()
                LOGGER.info(
                    "Processed %d/%d rows (dialect=%s, batch=%d, context=%d, number_context=%d)",
                    processed,
                    len(manifest_records),
                    dialect,
                    len(batch),
                    len(context_examples),
                    number_context_count,
                )

    LOGGER.info("Done. Wrote %d rows to %s", processed, output_path)


if __name__ == "__main__":
    main()
