"""
Speech Emotion Recognition (SER) tagging using MERaLiON-SER-v1.

Reads a JSONL manifest and appends framewise emotion predictions per segment.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)


EMO_LABELS = [
    "Neutral",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgusted",
    "Surprised",
]


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
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    audio = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    cache[audio_path] = (audio, sr)
    return audio, sr


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_val = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits - max_val)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-9, None)


def _infer_batch(
    model,
    processor,
    wavs: List[np.ndarray],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if not wavs:
        return np.zeros((0, len(EMO_LABELS)), dtype=np.float32), np.zeros((0, 3))

    inputs = processor(
        wavs,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=True,
    )
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
        if k in ("input_features", "attention_mask")
    }
    with torch.inference_mode():
        out = model(**inputs)
    logits = out["logits"].detach().cpu().numpy()
    dims = out.get("dims")
    if dims is None:
        dims_arr = np.zeros((logits.shape[0], 3), dtype=np.float32)
    else:
        dims_arr = dims.detach().cpu().numpy()
    return logits, dims_arr


def _predict_framewise(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    frame_seconds: float,
    hop_seconds: float,
    context_seconds: float,
    batch_size: int,
    min_seconds: float,
    device: str,
) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
    if audio.size == 0:
        return []

    frame_samples = max(int(round(frame_seconds * sr)), 1)
    hop_samples = max(int(round(hop_seconds * sr)), 1)
    context_samples = max(int(round(context_seconds * sr)), frame_samples)
    min_samples = int(min_seconds * sr)
    if audio.shape[0] <= frame_samples:
        starts = [0]
    else:
        starts = list(range(0, audio.shape[0] - frame_samples + 1, hop_samples))
        final_start = audio.shape[0] - frame_samples
        if starts[-1] != final_start:
            starts.append(final_start)

    windows: List[np.ndarray] = []
    times: List[Tuple[float, float]] = []
    for start in starts:
        end = min(start + frame_samples, audio.shape[0])
        if (end - start) < min_samples:
            continue

        center = start + ((end - start) // 2)
        context_start = center - (context_samples // 2)
        context_end = context_start + context_samples
        ctx = np.zeros((context_samples,), dtype=np.float32)
        src_start = max(0, context_start)
        src_end = min(audio.shape[0], context_end)
        if src_end > src_start:
            dst_start = src_start - context_start
            dst_end = dst_start + (src_end - src_start)
            ctx[dst_start:dst_end] = audio[src_start:src_end]

        windows.append(ctx)
        times.append((start / sr, end / sr))

    if not windows:
        return []

    probs_list: List[np.ndarray] = []
    dims_list: List[np.ndarray] = []
    for idx in range(0, len(windows), batch_size):
        batch = windows[idx : idx + batch_size]
        logits, dims = _infer_batch(model, processor, batch, device)
        probs_list.append(_softmax(logits))
        dims_list.append(dims)

    probs_arr = np.concatenate(probs_list, axis=0)
    dims_arr = np.concatenate(dims_list, axis=0)
    results = []
    for (start, end), probs, dims in zip(times, probs_arr, dims_arr):
        results.append((start, end, probs, dims))
    return results


def _round_probs(
    probs: Dict[str, float], digits: Optional[int], min_prob: float
) -> Dict[str, float]:
    filtered = {k: float(v) for k, v in probs.items() if float(v) >= min_prob}
    if digits is None:
        return filtered
    return {k: round(v, digits) for k, v in filtered.items()}


def _round_list(values: Iterable[float], digits: Optional[int]) -> List[float]:
    if digits is None:
        return [float(v) for v in values]
    return [round(float(v), digits) for v in values]


def _tag_segment(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    frame_seconds: float,
    hop_seconds: float,
    context_seconds: float,
    min_prob: float,
    segment_start: Optional[float],
    device: str,
) -> Dict[str, object]:
    frames = _predict_framewise(
        model,
        processor,
        audio,
        sr,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        context_seconds=context_seconds,
        batch_size=batch_size,
        min_seconds=min_seconds,
        device=device,
    )

    frame_list = []
    for start, end, f_probs, f_dims in frames:
        filtered_probs = _round_probs(
            {label: float(prob) for label, prob in zip(EMO_LABELS, f_probs)},
            round_digits,
            min_prob,
        )
        if not filtered_probs:
            continue
        f_idx = int(np.argmax(f_probs)) if f_probs.size else 0
        f_label = EMO_LABELS[f_idx] if EMO_LABELS else "unknown"
        frame_list.append(
            {
                "start": (segment_start or 0.0) + start,
                "end": (segment_start or 0.0) + end,
                "emotion": {
                    "label": f_label,
                    "confidence": float(np.max(f_probs)) if f_probs.size else 0.0,
                    "probs": filtered_probs,
                    "vad": _round_list(
                        f_dims.tolist() if hasattr(f_dims, "tolist") else f_dims,
                        round_digits,
                    ),
                },
            }
        )

    return {
        "emotion_tags_source": "framewise_meralion_ser",
        "emotion_frames": frame_list,
    }


def _tag_entry(
    entry: Dict,
    model,
    processor,
    base_dir: Path,
    sample_rate: int,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    frame_seconds: float,
    hop_seconds: float,
    context_seconds: float,
    min_prob: float,
    cache_audio: bool,
    audio_cache: Dict[Path, Tuple[np.ndarray, int]],
    device: str,
    slim_output: bool,
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

    results = []
    for segment in segments:
        audio_path = _resolve_audio_path(entry, segment, base_dir)
        if audio_path is None:
            logger.warning("No audio path found for entry; skipping tagging.")
            continue
        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            continue

        cache = audio_cache if cache_audio else {}
        audio, sr = _load_audio(audio_path, sample_rate, cache)
        # Always use the full audio_path; manifest timestamps are ignored.
        segment_start_for_frames = None
        segment_audio = audio
        tags = _tag_segment(
            model,
            processor,
            segment_audio,
            sr,
            batch_size=batch_size,
            min_seconds=min_seconds,
            round_digits=round_digits,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            context_seconds=context_seconds,
            min_prob=min_prob,
            segment_start=segment_start_for_frames,
            device=device,
        )
        if slim_output:
            slim_entry = {
                "audio_path": str(audio_path),
                "text": segment.get("text") or entry.get("text"),
            }
            slim_entry.update(tags)
            results.append(slim_entry)
        else:
            segment.update(tags)

    if slim_output:
        return results

    if segments_key:
        entry[segments_key] = segments
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition using MERaLiON-SER-v1",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSONL path (default: <manifest>.emotion.jsonl)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="MERaLiON/MERaLiON-SER-v1",
        help="Model repository (default: MERaLiON/MERaLiON-SER-v1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache directory (default: HF cache)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local files from cache (no downloads)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for inference (default: 16000)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.2,
        help="Minimum segment length to tag (default: 0.2s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Round probabilities and VAD to N digits (default: 4)",
    )
    parser.add_argument(
        "--frame-seconds",
        type=float,
        default=1.0,
        help="Frame length in seconds for framewise mode (default: 1.0)",
    )
    parser.add_argument(
        "--frame-hop",
        type=float,
        default=0.5,
        help="Frame hop in seconds for framewise mode (default: 0.5)",
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=4.0,
        help=(
            "Context window in seconds for framewise mode; each frame is inferred "
            "from centered context (default: 4.0)"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching full audio files in memory",
    )
    parser.add_argument(
        "--slim-output",
        action="store_true",
        help="Write only audio_path, text, and emotion fields per line",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.05,
        help="Only keep emotion probabilities >= this value (default: 0.05)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1
    if args.min_prob < 0.0 or args.min_prob > 1.0:
        logger.error("--min-prob must be between 0.0 and 1.0")
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import AutoModelForAudioClassification, AutoProcessor
    except Exception as exc:
        logger.error("transformers is required. Install it (pip install transformers).")
        logger.error("Import error: %s", exc)
        return 1

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    logger.info("Loading SER processor: %s", args.repo)
    processor = AutoProcessor.from_pretrained(
        args.repo,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    logger.info("Loading SER model: %s", args.repo)
    model = AutoModelForAudioClassification.from_pretrained(
        args.repo,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    round_digits = None if args.round < 0 else args.round

    output_path = args.output
    if output_path is None:
        output_path = args.manifest.with_suffix(".emotion.jsonl")
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
            "tool": "ser",
            "manifest": str(args.manifest.resolve()),
            "output_jsonl": str(output_path.resolve()),
            "repo": args.repo,
            "sample_rate": args.sample_rate,
            "frame_seconds": args.frame_seconds,
            "frame_hop": args.frame_hop,
            "context_seconds": args.context_seconds,
            "min_seconds": args.min_seconds,
            "batch_size": args.batch_size,
            "round_digits": round_digits,
            "min_prob": args.min_prob,
            "device": device,
            "labels": EMO_LABELS,
        },
    )
    logger.info("Run config written to %s", config_path)

    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}
    total_lines = sum(1 for _ in args.manifest.open("r", encoding="utf-8"))
    with (
        open(args.manifest, "r", encoding="utf-8") as infile,
        open(output_path, "a", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(tqdm(infile, total=total_lines), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: JSON decode error: %s", line_num, exc)
                continue

            entry_audio_keys = _collect_audio_keys(entry, base_dir)
            if entry_audio_keys and all(
                key in processed_audio_keys for key in entry_audio_keys
            ):
                continue

            tagged = _tag_entry(
                entry,
                model,
                processor,
                base_dir=base_dir,
                sample_rate=args.sample_rate,
                batch_size=args.batch_size,
                min_seconds=args.min_seconds,
                round_digits=round_digits,
                frame_seconds=args.frame_seconds,
                hop_seconds=args.frame_hop,
                context_seconds=args.context_seconds,
                min_prob=args.min_prob,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
                device=device,
                slim_output=args.slim_output,
            )
            wrote_any = False
            if args.slim_output:
                for slim in tagged:
                    outfile.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    wrote_any = True
            else:
                outfile.write(json.dumps(tagged, ensure_ascii=False) + "\n")
                wrote_any = True
            if wrote_any:
                processed_audio_keys.update(entry_audio_keys)

    logger.info("Emotion-tagged manifest written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
