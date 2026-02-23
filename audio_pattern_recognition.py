"""
Framewise AudioSet tagging using AST (Hugging Face).

This script reads a JSONL manifest and appends frame-level tags only.
No event extraction and no threshold-based laughter logic.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Ontology-aligned target labels.
SPEECH_LABELS = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
]

HUMAN_VOICE_LABELS = [
    "Shout",
    "Screaming",
    "Whispering",
    "Laughter",
    "Baby laughter",
    "Giggle",
    "Snicker",
    "Belly laugh",
    "Chuckle, chortle",
    "Crying, sobbing",
    "Wail, moan",
    "Sigh",
    "Singing",
    "Humming",
    "Groan",
    "Grunt",
]

RESPIRATORY_LABELS = [
    "Breathing",
    "Wheeze",
    "Snoring",
    "Gasp",
    "Pant",
    "Snort",
    "Cough",
    "Throat clearing",
    "Sneeze",
    "Sniff",
]


def _dedupe_keep_order(labels: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


TARGET_LABELS = _dedupe_keep_order(
    SPEECH_LABELS + HUMAN_VOICE_LABELS + RESPIRATORY_LABELS
)


def _round_probs(
    probs: Dict[str, float], digits: Optional[int], min_prob: float
) -> Dict[str, float]:
    filtered = {k: float(v) for k, v in probs.items() if float(v) >= min_prob}
    if digits is None:
        return filtered
    return {k: round(v, digits) for k, v in filtered.items()}


def _find_audio_path(entry: Dict, segment: Optional[Dict]) -> Optional[Path]:
    candidates = []
    if segment:
        candidates.extend(
            [
                segment.get("audio_filepath"),
                segment.get("audio_path"),
                segment.get("path"),
                segment.get("audio"),
            ]
        )
    candidates.extend(
        [
            entry.get("audio_filepath"),
            entry.get("audio_path"),
            entry.get("path"),
            entry.get("audio"),
        ]
    )
    for value in candidates:
        if value:
            return Path(value)
    return None


def _resolve_audio_path(
    entry: Dict,
    segment: Optional[Dict],
    base_dir: Path,
) -> Optional[Path]:
    audio_path = _find_audio_path(entry, segment)
    if audio_path is None:
        return None
    if not audio_path.is_absolute():
        audio_path = (base_dir / audio_path).resolve()
    else:
        audio_path = audio_path.resolve()
    return audio_path


def _collect_audio_keys(entry: Dict, base_dir: Path) -> List[str]:
    keys: List[str] = []
    seen = set()
    segments: Optional[List[Dict]] = None
    if isinstance(entry.get("final_segments"), list):
        segments = entry["final_segments"]
    elif isinstance(entry.get("segments"), list):
        segments = entry["segments"]
    if segments is None:
        segments = [entry]

    for segment in segments:
        resolved = _resolve_audio_path(entry, segment, base_dir)
        if resolved is None:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _load_processed_audio_keys(output_path: Path, base_dir: Path) -> set[str]:
    processed: set[str] = set()
    if not output_path.exists():
        return processed

    with open(output_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Ignoring output line %d: JSON decode error: %s", line_num, exc
                )
                continue
            for key in _collect_audio_keys(entry, base_dir):
                processed.add(key)
    return processed


def _write_run_config(output_path: Path, config: Dict[str, object]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_path = output_path.with_name(f"{output_path.stem}.config.{ts}.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as cfg:
        json.dump(config, cfg, ensure_ascii=False, indent=2)
        cfg.write("\n")
    return config_path


def _load_audio(
    audio_path: Path,
    target_sr: int,
    cache: Dict[Path, Tuple[np.ndarray, int]],
) -> Tuple[np.ndarray, int]:
    if audio_path in cache:
        return cache[audio_path]

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.abs().max() > 1.0:
        waveform = waveform.float() / 32768.0

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(
            waveform
        )
        sr = target_sr

    audio = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    cache[audio_path] = (audio, sr)
    return audio, sr


def _resolve_label_indices(
    id2label: Dict[int, str],
    target_labels: Sequence[str],
) -> Tuple[List[int], List[str]]:
    label_to_idx = {label: idx for idx, label in id2label.items()}
    indices: List[int] = []
    names: List[str] = []
    for label in target_labels:
        idx = label_to_idx.get(label)
        if idx is None:
            logger.warning("Label '%s' not found in model labels; skipping.", label)
            continue
        indices.append(idx)
        names.append(label)
    return indices, names


def _extract_centered_context(
    audio: np.ndarray, center_sample: int, context_samples: int
) -> np.ndarray:
    out = np.zeros((context_samples,), dtype=np.float32)
    start = center_sample - (context_samples // 2)
    end = start + context_samples

    src_start = max(0, start)
    src_end = min(audio.shape[0], end)
    if src_end <= src_start:
        return out

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = audio[src_start:src_end]
    return out


def _build_frame_starts(
    total_samples: int, frame_samples: int, hop_samples: int
) -> List[int]:
    if total_samples <= 0 or total_samples <= frame_samples:
        return [0]

    starts = list(range(0, total_samples - frame_samples + 1, hop_samples))
    final_start = total_samples - frame_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _predict_framewise(
    model,
    feature_extractor,
    audio: np.ndarray,
    sr: int,
    frame_seconds: float,
    frame_hop: float,
    context_seconds: float,
    batch_size: int,
    device: str,
) -> List[Tuple[float, float, np.ndarray]]:
    if audio.size == 0:
        return []

    frame_samples = max(int(round(frame_seconds * sr)), 1)
    hop_samples = max(int(round(frame_hop * sr)), 1)
    context_samples = max(int(round(context_seconds * sr)), frame_samples)

    starts = _build_frame_starts(audio.shape[0], frame_samples, hop_samples)

    frames: List[Tuple[float, float, np.ndarray]] = []
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        contexts: List[np.ndarray] = []
        intervals: List[Tuple[float, float]] = []

        for start in batch_starts:
            end = min(start + frame_samples, audio.shape[0])
            center = start + ((end - start) // 2)
            contexts.append(
                _extract_centered_context(
                    audio, center_sample=center, context_samples=context_samples
                )
            )
            intervals.append((start / sr, end / sr))

        inputs = feature_extractor(contexts, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = (
            torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
        )

        for (t0, t1), frame_probs in zip(intervals, probs):
            frames.append((t0, t1, frame_probs))

    return frames


def _tag_audio(
    audio: np.ndarray,
    sr: int,
    model,
    feature_extractor,
    label_indices: Sequence[int],
    label_names: Sequence[str],
    round_digits: Optional[int],
    frame_seconds: float,
    frame_hop: float,
    context_seconds: float,
    batch_size: int,
    device: str,
    save_raw_frames: bool,
    raw_top_k: int,
    min_prob: float,
    segment_start: float = 0.0,
) -> Dict[str, object]:
    frames = _predict_framewise(
        model=model,
        feature_extractor=feature_extractor,
        audio=audio,
        sr=sr,
        frame_seconds=frame_seconds,
        frame_hop=frame_hop,
        context_seconds=context_seconds,
        batch_size=batch_size,
        device=device,
    )

    if not frames:
        result: Dict[str, object] = {
            "audio_tags_source": "framewise_max",
            "audio_tags": _round_probs(
                {label: 0.0 for label in label_names}, round_digits, min_prob
            ),
            "audio_tag_frames": [],
        }
        if save_raw_frames:
            result["audio_tag_frames_raw"] = []
        return result

    selected_probs = np.stack(
        [[float(probs[idx]) for idx in label_indices] for _, _, probs in frames],
        axis=0,
    )
    max_probs = selected_probs.max(axis=0)

    frame_entries = []
    raw_entries = []
    for t0, t1, probs in frames:
        frame_tags = {
            label: float(probs[idx]) for label, idx in zip(label_names, label_indices)
        }
        filtered_frame_tags = _round_probs(frame_tags, round_digits, min_prob)
        if filtered_frame_tags:
            best_idx = int(np.argmax(probs))
            frame_entries.append(
                {
                    "start": segment_start + t0,
                    "end": segment_start + t1,
                    "top_label": model.config.id2label[best_idx],
                    "audio_tags": filtered_frame_tags,
                }
            )

        if save_raw_frames:
            top_idx = np.argsort(-probs)[: max(raw_top_k, 1)]
            top_labels = [
                {"label": model.config.id2label[int(i)], "score": float(probs[int(i)])}
                for i in top_idx
                if float(probs[int(i)]) >= min_prob
            ]
            if round_digits is not None:
                for item in top_labels:
                    item["score"] = round(item["score"], round_digits)
            if top_labels:
                raw_entries.append(
                    {
                        "start": segment_start + t0,
                        "end": segment_start + t1,
                        "top_labels": top_labels,
                    }
                )

    result = {
        "audio_tags_source": "framewise_max",
        "audio_tags": _round_probs(
            {label: float(value) for label, value in zip(label_names, max_probs)},
            round_digits,
            min_prob,
        ),
        "audio_tag_frames": frame_entries,
    }

    if save_raw_frames:
        result["audio_tag_frames_raw"] = raw_entries

    return result


def _tag_entry(
    entry: Dict,
    model,
    feature_extractor,
    label_indices: Sequence[int],
    label_names: Sequence[str],
    base_dir: Path,
    sample_rate: int,
    frame_seconds: float,
    frame_hop: float,
    context_seconds: float,
    batch_size: int,
    round_digits: Optional[int],
    save_raw_frames: bool,
    raw_top_k: int,
    min_prob: float,
    cache_audio: bool,
    audio_cache: Dict[Path, Tuple[np.ndarray, int]],
    device: str,
) -> Dict:
    segments_key = None
    segments = None
    if isinstance(entry.get("final_segments"), list):
        segments_key = "final_segments"
        segments = entry["final_segments"]
    elif isinstance(entry.get("segments"), list):
        segments_key = "segments"
        segments = entry["segments"]

    if segments is None:
        segments = [entry]

    for segment in segments:
        audio_path = _resolve_audio_path(entry, segment, base_dir)
        if audio_path is None:
            logger.warning("No audio path found for entry; skipping tagging.")
            continue

        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            continue

        cache = audio_cache if cache_audio else {}
        audio, sr = _load_audio(audio_path, target_sr=sample_rate, cache=cache)

        segment.update(
            _tag_audio(
                audio=audio,
                sr=sr,
                model=model,
                feature_extractor=feature_extractor,
                label_indices=label_indices,
                label_names=label_names,
                round_digits=round_digits,
                frame_seconds=frame_seconds,
                frame_hop=frame_hop,
                context_seconds=context_seconds,
                batch_size=batch_size,
                device=device,
                save_raw_frames=save_raw_frames,
                raw_top_k=raw_top_k,
                min_prob=min_prob,
                segment_start=0.0,
            )
        )

    if segments_key:
        entry[segments_key] = segments
    return entry


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_num, json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: JSON decode error: %s", line_num, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Framewise AudioSet tagging with AST")
    parser.add_argument("manifest", type=Path, help="Path to manifest.jsonl")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL path")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Inference device (default: auto)"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--frame-hop", type=float, default=0.125)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--round", type=int, default=3, help="Use -1 to disable")
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument("--raw-top-k", type=int, default=10)
    parser.add_argument("--minimal-output", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.05,
        help="Only keep tag probabilities >= this value (default: 0.05)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1
    if args.frame_seconds <= 0 or args.frame_hop <= 0 or args.context_seconds <= 0:
        logger.error("--frame-seconds, --frame-hop and --context-seconds must be > 0")
        return 1
    if args.min_prob < 0.0 or args.min_prob > 1.0:
        logger.error("--min-prob must be between 0.0 and 1.0")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model: %s", args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    model_labels = model.config.id2label
    label_indices, label_names = _resolve_label_indices(model_labels, TARGET_LABELS)
    if not label_indices:
        logger.error("No target labels resolved against model labels. Aborting.")
        return 1

    round_digits = None if args.round < 0 else args.round
    output_path = args.output or args.manifest.with_suffix(".tagged.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = args.manifest.parent.resolve()
    processed_audio_keys = _load_processed_audio_keys(output_path, base_dir)
    if output_path.exists():
        logger.info(
            "Output file already exists: %s (%d processed audio paths)",
            output_path,
            len(processed_audio_keys),
        )
    else:
        logger.info("Output file will be created: %s", output_path)

    config_path = _write_run_config(
        output_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "audio_pattern_recognition",
            "manifest": str(args.manifest.resolve()),
            "output_jsonl": str(output_path.resolve()),
            "model_id": args.model_id,
            "sample_rate": args.sample_rate,
            "frame_seconds": args.frame_seconds,
            "frame_hop": args.frame_hop,
            "context_seconds": args.context_seconds,
            "batch_size": args.batch_size,
            "round_digits": round_digits,
            "min_prob": args.min_prob,
            "device": device,
            "target_labels": label_names,
        },
    )
    logger.info("Run config written to %s", config_path)

    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

    with open(output_path, "a", encoding="utf-8") as outfile:
        for _, entry in _iter_jsonl(args.manifest):
            entry_audio_keys = _collect_audio_keys(entry, base_dir)
            if entry_audio_keys and all(
                key in processed_audio_keys for key in entry_audio_keys
            ):
                continue

            tagged = _tag_entry(
                entry=entry,
                model=model,
                feature_extractor=feature_extractor,
                label_indices=label_indices,
                label_names=label_names,
                base_dir=base_dir,
                sample_rate=args.sample_rate,
                frame_seconds=args.frame_seconds,
                frame_hop=args.frame_hop,
                context_seconds=args.context_seconds,
                batch_size=args.batch_size,
                round_digits=round_digits,
                save_raw_frames=args.save_raw_frames,
                raw_top_k=args.raw_top_k,
                min_prob=args.min_prob,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
                device=device,
            )

            if args.minimal_output:
                slim = {
                    "audio_path": tagged.get("audio_path")
                    or tagged.get("audio_filepath")
                    or tagged.get("path")
                    or tagged.get("audio"),
                    "text": tagged.get("text"),
                    "audio_tags_source": tagged.get("audio_tags_source"),
                    "audio_tags": tagged.get("audio_tags"),
                    "audio_tag_frames": tagged.get("audio_tag_frames"),
                }
                if "audio_tag_frames_raw" in tagged:
                    slim["audio_tag_frames_raw"] = tagged.get("audio_tag_frames_raw")
                outfile.write(json.dumps(slim, ensure_ascii=False) + "\n")
            else:
                outfile.write(json.dumps(tagged, ensure_ascii=False) + "\n")
            processed_audio_keys.update(entry_audio_keys)

    logger.info("Tagged manifest written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
