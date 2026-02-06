"""
Speech Emotion Recognition (SER) tagging using MERaLiON-SER-v1.

Reads a JSONL manifest and appends emotion predictions per segment.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

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


def _predict_clipwise(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.zeros((len(EMO_LABELS),), dtype=np.float32), np.zeros((3,))

    chunk_samples = int(chunk_seconds * sr)
    min_samples = int(min_seconds * sr)
    if chunk_samples <= 0 or audio.shape[0] <= chunk_samples:
        logits, dims = _infer_batch(model, processor, [audio], device)
        probs = _softmax(logits)[0]
        return probs, dims[0]

    chunks: List[np.ndarray] = []
    weights: List[float] = []
    for start in range(0, audio.shape[0], chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if chunk.shape[0] < min_samples:
            continue
        chunks.append(chunk)
        weights.append(float(chunk.shape[0]))

    if not chunks:
        logits, dims = _infer_batch(model, processor, [audio], device)
        probs = _softmax(logits)[0]
        return probs, dims[0]

    probs_list: List[np.ndarray] = []
    dims_list: List[np.ndarray] = []
    for idx in range(0, len(chunks), batch_size):
        batch = chunks[idx : idx + batch_size]
        logits, dims = _infer_batch(model, processor, batch, device)
        probs_list.append(_softmax(logits))
        dims_list.append(dims)

    probs_arr = np.concatenate(probs_list, axis=0)
    dims_arr = np.concatenate(dims_list, axis=0)
    weights_arr = np.array(weights, dtype=np.float32)
    norm = max(weights_arr.sum(), 1.0)
    probs = (probs_arr * weights_arr[:, None]).sum(axis=0) / norm
    dims = (dims_arr * weights_arr[:, None]).sum(axis=0) / norm
    return probs, dims


def _predict_framewise(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    frame_seconds: float,
    hop_seconds: float,
    batch_size: int,
    min_seconds: float,
    device: str,
) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
    if audio.size == 0:
        return []

    frame_samples = int(frame_seconds * sr)
    hop_samples = int(hop_seconds * sr)
    min_samples = int(min_seconds * sr)
    if frame_samples <= 0 or hop_samples <= 0:
        return []

    windows: List[np.ndarray] = []
    times: List[Tuple[float, float]] = []
    for start in range(0, audio.shape[0], hop_samples):
        end = start + frame_samples
        chunk = audio[start:end]
        if chunk.shape[0] < min_samples:
            continue
        windows.append(chunk)
        times.append((start / sr, min(end, audio.shape[0]) / sr))

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


def _round_probs(probs: Dict[str, float], digits: Optional[int]) -> Dict[str, float]:
    if digits is None:
        return probs
    return {k: round(v, digits) for k, v in probs.items()}


def _round_list(values: Iterable[float], digits: Optional[int]) -> List[float]:
    if digits is None:
        return [float(v) for v in values]
    return [round(float(v), digits) for v in values]


def _tag_segment(
    model,
    processor,
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    framewise: bool,
    frame_seconds: float,
    hop_seconds: float,
    segment_start: Optional[float],
    device: str,
) -> Dict[str, object]:
    probs, dims = _predict_clipwise(
        model,
        processor,
        audio,
        sr,
        chunk_seconds=chunk_seconds,
        batch_size=batch_size,
        min_seconds=min_seconds,
        device=device,
    )
    emo_idx = int(np.argmax(probs)) if probs.size else 0
    emo_label = EMO_LABELS[emo_idx] if EMO_LABELS else "unknown"
    tag_probs = {label: float(prob) for label, prob in zip(EMO_LABELS, probs)}
    result = {
        "emotion": {
            "label": emo_label,
            "confidence": float(np.max(probs)) if probs.size else 0.0,
            "probs": _round_probs(tag_probs, round_digits),
            "vad": _round_list(dims.tolist() if hasattr(dims, "tolist") else dims, round_digits),
        }
    }

    if framewise:
        frames = _predict_framewise(
            model,
            processor,
            audio,
            sr,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            batch_size=batch_size,
            min_seconds=min_seconds,
            device=device,
        )
        frame_list = []
        for start, end, f_probs, f_dims in frames:
            f_idx = int(np.argmax(f_probs)) if f_probs.size else 0
            f_label = EMO_LABELS[f_idx] if EMO_LABELS else "unknown"
            frame_list.append(
                {
                    "start": (segment_start or 0.0) + start,
                    "end": (segment_start or 0.0) + end,
                    "emotion": {
                        "label": f_label,
                        "confidence": float(np.max(f_probs)) if f_probs.size else 0.0,
                        "probs": _round_probs(
                            {label: float(prob) for label, prob in zip(EMO_LABELS, f_probs)},
                            round_digits,
                        ),
                        "vad": _round_list(
                            f_dims.tolist() if hasattr(f_dims, "tolist") else f_dims,
                            round_digits,
                        ),
                    },
                }
            )
        result["emotion_frames"] = frame_list

    return result


def _tag_entry(
    entry: Dict,
    model,
    processor,
    base_dir: Path,
    sample_rate: int,
    chunk_seconds: float,
    batch_size: int,
    min_seconds: float,
    round_digits: Optional[int],
    framewise: bool,
    frame_seconds: float,
    hop_seconds: float,
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
        # Always use the full audio_path; manifest timestamps are ignored.
        segment_start_for_frames = None
        segment_audio = audio
        tags = _tag_segment(
            model,
            processor,
            segment_audio,
            sr,
            chunk_seconds=chunk_seconds,
            batch_size=batch_size,
            min_seconds=min_seconds,
            round_digits=round_digits,
            framewise=framewise,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
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
        help="Round probabilities and VAD to N digits (default: 4)",
    )
    parser.add_argument(
        "--framewise",
        action="store_true",
        help="Enable framewise tagging with sliding windows",
    )
    parser.add_argument(
        "--frame-seconds",
        type=float,
        default=2.0,
        help="Frame length in seconds for framewise mode (default: 2.0)",
    )
    parser.add_argument(
        "--frame-hop",
        type=float,
        default=1.0,
        help="Frame hop in seconds for framewise mode (default: 1.0)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import AutoModelForAudioClassification, AutoProcessor
    except Exception as exc:
        logger.error(
            "transformers is required. Install it (pip install transformers)."
        )
        logger.error("Import error: %s", exc)
        return 1

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    logger.info("Loading SER processor: %s", args.repo)
    processor = AutoProcessor.from_pretrained(
        args.repo, cache_dir=cache_dir, local_files_only=args.local_files_only
    )
    logger.info("Loading SER model: %s", args.repo)
    model = AutoModelForAudioClassification.from_pretrained(
        args.repo,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    output_path = args.output
    if output_path is None:
        output_path = args.manifest.with_suffix(".emotion.jsonl")

    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

    with open(args.manifest, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping line %d: JSON decode error: %s", line_num, exc
                )
                continue

            entry.setdefault(
                "emotion_tagging",
                {
                    "repo": args.repo,
                    "sample_rate": args.sample_rate,
                    "chunk_seconds": args.chunk_seconds,
                    "framewise": args.framewise,
                    "frame_seconds": args.frame_seconds if args.framewise else None,
                    "frame_hop": args.frame_hop if args.framewise else None,
                    "labels": EMO_LABELS,
                },
            )

            tagged = _tag_entry(
                entry,
                model,
                processor,
                base_dir=args.manifest.parent,
                sample_rate=args.sample_rate,
                chunk_seconds=args.chunk_seconds,
                batch_size=args.batch_size,
                min_seconds=args.min_seconds,
                round_digits=args.round,
                framewise=args.framewise,
                frame_seconds=args.frame_seconds,
                hop_seconds=args.frame_hop,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
                device=device,
                slim_output=args.slim_output,
            )
            if args.slim_output:
                for slim in tagged:
                    outfile.write(json.dumps(slim, ensure_ascii=False) + "\n")
            else:
                outfile.write(json.dumps(tagged, ensure_ascii=False) + "\n")

    logger.info("Emotion-tagged manifest written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
