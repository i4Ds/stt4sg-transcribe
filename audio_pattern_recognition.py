"""
Audio pattern recognition for AudioSet tags using PANNs inference.

Reads a JSONL manifest and appends tag probabilities for target labels.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


TARGET_LABELS = [
    "Speech",
    "Music",
    "Laughter",
    "Cough",
    "Sneeze",
    "Breathing",
]


def _normalize_label(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in label).strip()


def _resolve_label_indices(
    available_labels: List[str], target_labels: List[str]
) -> Tuple[List[int], List[str]]:
    norm_map = {_normalize_label(lbl): lbl for lbl in available_labels}
    resolved_indices: List[int] = []
    resolved_labels: List[str] = []

    for target in target_labels:
        norm_target = _normalize_label(target)
        if norm_target in norm_map:
            label = norm_map[norm_target]
            resolved_indices.append(available_labels.index(label))
            resolved_labels.append(label)
            continue

        # Fallback: substring match
        matches = [
            lbl
            for lbl in available_labels
            if norm_target in _normalize_label(lbl)
            or _normalize_label(lbl) in norm_target
        ]
        if len(matches) == 1:
            label = matches[0]
            resolved_indices.append(available_labels.index(label))
            resolved_labels.append(label)
            continue

        logger.warning(
            "Target label '%s' not found in available labels. Skipping.",
            target,
        )

    return resolved_indices, resolved_labels


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


def _batch_infer(model, audio_list: List[np.ndarray]) -> np.ndarray:
    if not audio_list:
        return np.zeros((0, 0), dtype=np.float32)

    max_len = max(arr.shape[0] for arr in audio_list)
    batch = np.zeros((len(audio_list), max_len), dtype=np.float32)
    for idx, arr in enumerate(audio_list):
        batch[idx, : arr.shape[0]] = arr

    with torch.no_grad():
        clipwise_output, _ = model.inference(batch)
    return clipwise_output


def _predict_clipwise(
    model,
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((model.num_classes,), dtype=np.float32)

    chunk_samples = int(chunk_seconds * sr)
    min_samples = int(min_seconds * sr)
    if chunk_samples <= 0 or audio.shape[0] <= chunk_samples:
        outputs = _batch_infer(model, [audio])
        return outputs[0]

    chunks: List[np.ndarray] = []
    weights: List[float] = []
    for start in range(0, audio.shape[0], chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if chunk.shape[0] < min_samples:
            continue
        chunks.append(chunk)
        weights.append(float(chunk.shape[0]))

    if not chunks:
        outputs = _batch_infer(model, [audio])
        return outputs[0]

    outputs_list: List[np.ndarray] = []
    for idx in range(0, len(chunks), batch_size):
        batch = chunks[idx : idx + batch_size]
        outputs_list.append(_batch_infer(model, batch))

    outputs = np.concatenate(outputs_list, axis=0)
    weights_arr = np.array(weights, dtype=np.float32)
    weighted = (outputs * weights_arr[:, None]).sum(axis=0) / max(
        weights_arr.sum(), 1.0
    )
    return weighted


def _aggregate_frames(
    frames: List[Tuple[float, float, np.ndarray]],
    frame_seconds: float,
    hop_seconds: float,
) -> List[Tuple[float, float, np.ndarray]]:
    if not frames:
        return []
    if frame_seconds <= 0 or hop_seconds <= 0:
        return frames

    raw_hop = frames[0][1] - frames[0][0]
    if raw_hop <= 0:
        return frames

    window = max(int(round(frame_seconds / raw_hop)), 1)
    hop = max(int(round(hop_seconds / raw_hop)), 1)

    probs = np.stack([f[2] for f in frames], axis=0)
    results = []
    for start_idx in range(0, len(frames) - window + 1, hop):
        end_idx = start_idx + window
        start_t = frames[start_idx][0]
        end_t = frames[end_idx - 1][1]
        avg = probs[start_idx:end_idx].mean(axis=0)
        results.append((start_t, end_t, avg))
    return results


def _predict_framewise(
    sed_model,
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    frame_seconds: float,
    hop_seconds: float,
    min_seconds: float,
) -> Tuple[
    List[Tuple[float, float, np.ndarray]], List[Tuple[float, float, np.ndarray]]
]:
    if audio.size == 0:
        return [], []

    min_samples = int(min_seconds * sr)
    if audio.shape[0] < min_samples:
        return [], []

    chunk_samples = int(chunk_seconds * sr)
    if chunk_samples <= 0:
        chunk_samples = audio.shape[0]

    raw_results: List[Tuple[float, float, np.ndarray]] = []
    for start in range(0, audio.shape[0], chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if chunk.shape[0] < min_samples:
            continue
        chunk_batch = chunk[None, :]
        with torch.no_grad():
            framewise_output = sed_model.inference(chunk_batch)

        framewise = framewise_output[0]
        frames = framewise.shape[0]
        if frames == 0:
            continue

        chunk_duration = chunk.shape[0] / max(sr, 1)
        frame_hop = chunk_duration / frames
        raw_frames: List[Tuple[float, float, np.ndarray]] = []
        for idx in range(frames):
            t0 = (start / sr) + idx * frame_hop
            t1 = (start / sr) + (idx + 1) * frame_hop
            raw_frames.append((t0, t1, framewise[idx]))

        raw_results.extend(raw_frames)

    agg_results = _aggregate_frames(raw_results, frame_seconds, hop_seconds)
    return raw_results, agg_results


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


def _segment_bounds(segment: Dict) -> Tuple[Optional[float], Optional[float]]:
    if "start" in segment and "end" in segment:
        return segment.get("start"), segment.get("end")
    if "offset" in segment and "duration" in segment:
        offset = segment.get("offset")
        duration = segment.get("duration")
        if offset is not None and duration is not None:
            return float(offset), float(offset) + float(duration)
    return None, None


def _round_probs(probs: Dict[str, float], digits: Optional[int]) -> Dict[str, float]:
    if digits is None:
        return probs
    return {k: round(v, digits) for k, v in probs.items()}


def _tag_segment(
    model,
    sed_model,
    audio: np.ndarray,
    sr: int,
    label_indices: List[int],
    label_names: List[str],
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    framewise: bool,
    frame_seconds: float,
    hop_seconds: float,
    segment_start: Optional[float],
    save_raw_frames: bool,
) -> Dict[str, object]:
    clipwise = _predict_clipwise(
        model,
        audio,
        sr,
        chunk_seconds=chunk_seconds,
        batch_size=batch_size,
        min_seconds=min_seconds,
    )
    tag_probs = {
        label: float(clipwise[idx]) for label, idx in zip(label_names, label_indices)
    }
    result = {"audio_tags": _round_probs(tag_probs, round_digits)}

    if framewise and sed_model is not None:
        raw_frames, frames = _predict_framewise(
            sed_model,
            audio,
            sr,
            chunk_seconds=chunk_seconds,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            min_seconds=min_seconds,
        )
        frame_list = []
        for start, end, probs in frames:
            frame_tags = {
                label: float(probs[idx])
                for label, idx in zip(label_names, label_indices)
            }
            frame_list.append(
                {
                    "start": (segment_start or 0.0) + start,
                    "end": (segment_start or 0.0) + end,
                    "audio_tags": _round_probs(frame_tags, round_digits),
                }
            )
        result["audio_tag_frames"] = frame_list

        if save_raw_frames:
            raw_list = []
            for start, end, probs in raw_frames:
                frame_tags = {
                    label: float(probs[idx])
                    for label, idx in zip(label_names, label_indices)
                }
                raw_list.append(
                    {
                        "start": (segment_start or 0.0) + start,
                        "end": (segment_start or 0.0) + end,
                        "audio_tags": _round_probs(frame_tags, round_digits),
                    }
                )
            result["audio_tag_frames_raw"] = raw_list

    return result


def _tag_entry(
    entry: Dict,
    model,
    sed_model,
    label_indices: List[int],
    label_names: List[str],
    base_dir: Path,
    sample_rate: int,
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    framewise: bool,
    frame_seconds: float,
    hop_seconds: float,
    save_raw_frames: bool,
    cache_audio: bool,
    audio_cache: Dict[Path, Tuple[np.ndarray, int]],
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
        audio_path = _find_audio_path(entry, segment)
        if audio_path is None:
            logger.warning("No audio path found for entry; skipping tagging.")
            continue
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()
        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            continue

        cache = audio_cache if cache_audio else {}
        audio, sr = _load_audio(audio_path, sample_rate, cache)
        # Audio clips in this dataset are already cut. Always tag the full clip.
        segment_audio = audio
        frame_offset = None
        tags = _tag_segment(
            model,
            sed_model,
            segment_audio,
            sr,
            label_indices,
            label_names,
            chunk_seconds=chunk_seconds,
            batch_size=batch_size,
            min_seconds=min_seconds,
            round_digits=round_digits,
            framewise=framewise,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            segment_start=frame_offset,
            save_raw_frames=save_raw_frames,
        )
        segment.update(tags)

    if segments_key:
        entry[segments_key] = segments
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio pattern recognition using PANNs AudioSet tagging",
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
        help="Output JSONL path (default: <manifest>.tagged.jsonl)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for PANNs (default: panns_inference default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Sample rate for inference (default: 32000)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Chunk length for long segments (default: 10s)",
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
        help="Round tag probabilities to N digits (use -1 to disable)",
    )
    parser.add_argument(
        "--framewise",
        action="store_true",
        help="Enable framewise tagging using PANNs SoundEventDetection",
    )
    parser.add_argument(
        "--frame-seconds",
        type=float,
        default=2.0,
        help="Aggregate framewise tags into windows (default: 2.0s, set 0 to keep raw)",
    )
    parser.add_argument(
        "--frame-hop",
        type=float,
        default=1.0,
        help="Hop size in seconds for framewise aggregation (default: 1.0s)",
    )
    parser.add_argument(
        "--save-raw-frames",
        action="store_true",
        help="Also store full-resolution SED frames in audio_tag_frames_raw",
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Write a slim JSONL with only audio_path, text, and tagging outputs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching full audio files in memory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    try:
        from panns_inference import (
            AudioTagging,
            SoundEventDetection,
            labels as panns_labels,
        )
    except Exception as exc:
        logger.error(
            "panns_inference is required. Install it (pip install panns-inference)."
        )
        logger.error("Import error: %s", exc)
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AudioTagging(checkpoint_path=args.checkpoint, device=device)
    if not hasattr(model, "num_classes"):
        model.num_classes = len(panns_labels)

    sed_model = None
    if args.framewise:
        sed_model = SoundEventDetection(checkpoint_path=args.checkpoint, device=device)
        if not hasattr(sed_model, "num_classes"):
            sed_model.num_classes = len(panns_labels)

    label_indices, label_names = _resolve_label_indices(panns_labels, TARGET_LABELS)
    if not label_indices:
        logger.error("None of the target labels were resolved. Aborting.")
        return 1

    round_digits = None if args.round < 0 else args.round

    output_path = args.output
    if output_path is None:
        output_path = args.manifest.with_suffix(".tagged.jsonl")

    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

    with (
        open(args.manifest, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: JSON decode error: %s", line_num, exc)
                continue

            entry.setdefault(
                "audio_tagging",
                {
                    "labels": label_names,
                    "sample_rate": args.sample_rate,
                    "checkpoint": args.checkpoint,
                    "chunk_seconds": args.chunk_seconds,
                    "framewise": args.framewise,
                    "frame_seconds": args.frame_seconds if args.framewise else None,
                    "frame_hop": args.frame_hop if args.framewise else None,
                    "frame_mode": "sed" if args.framewise else None,
                    "save_raw_frames": args.save_raw_frames if args.framewise else None,
                },
            )

            tagged = _tag_entry(
                entry,
                model,
                sed_model,
                label_indices,
                label_names,
                base_dir=args.manifest.parent,
                sample_rate=args.sample_rate,
                chunk_seconds=args.chunk_seconds,
                batch_size=args.batch_size,
                min_seconds=args.min_seconds,
                round_digits=round_digits,
                framewise=args.framewise,
                frame_seconds=args.frame_seconds,
                hop_seconds=args.frame_hop,
                save_raw_frames=args.save_raw_frames,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
            )
            if args.minimal_output:
                slim = {
                    "audio_path": tagged.get("audio_path")
                    or tagged.get("audio_filepath")
                    or tagged.get("path")
                    or tagged.get("audio"),
                    "text": tagged.get("text"),
                    "audio_tagging": tagged.get("audio_tagging"),
                    "audio_tags": tagged.get("audio_tags"),
                    "audio_tag_frames": tagged.get("audio_tag_frames"),
                }
                if "audio_tag_frames_raw" in tagged:
                    slim["audio_tag_frames_raw"] = tagged.get("audio_tag_frames_raw")
                outfile.write(json.dumps(slim, ensure_ascii=False) + "\n")
            else:
                outfile.write(json.dumps(tagged, ensure_ascii=False) + "\n")

    logger.info("Tagged manifest written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
