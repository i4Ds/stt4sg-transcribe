"""
Framewise AudioSet tagging using CNN8RNN SED (Hugging Face).

This script mirrors `audio_pattern_recognition.py` output schema but uses
`wsntxxn/cnn8rnn-audioset-sed` for framewise probabilities.
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
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "wsntxxn/cnn8rnn-audioset-sed"

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
NON_SPEECH_TARGET_LABELS = _dedupe_keep_order(HUMAN_VOICE_LABELS + RESPIRATORY_LABELS)


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


def _resolve_audio_path(entry: Dict, segment: Optional[Dict], base_dir: Path) -> Optional[Path]:
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
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr

    audio = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    cache[audio_path] = (audio, sr)
    return audio, sr


def _get_framewise_tensor(output_obj: object) -> torch.Tensor:
    if isinstance(output_obj, dict):
        fw = output_obj.get("framewise_output")
    else:
        fw = getattr(output_obj, "framewise_output", None)
    if fw is None:
        raise RuntimeError("Model output has no framewise_output.")
    if not isinstance(fw, torch.Tensor):
        fw = torch.as_tensor(fw)
    return fw


def _normalize_framewise(fw: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Expected either [B, C, T] or [B, T, C].
    if fw.ndim != 3:
        raise RuntimeError(f"Unexpected framewise_output shape: {tuple(fw.shape)}")
    if fw.shape[1] == num_classes:
        out = fw
    elif fw.shape[2] == num_classes:
        out = fw.transpose(1, 2)
    else:
        raise RuntimeError(
            f"Could not locate class dimension in framewise_output shape {tuple(fw.shape)} with num_classes={num_classes}"
        )
    return out


def _predict_framewise(
    model,
    audio: np.ndarray,
    sr: int,
    device: str,
    num_classes: int,
) -> List[Tuple[float, float, np.ndarray]]:
    return _predict_framewise_batch(
        model=model,
        audios=[audio],
        sr=sr,
        device=device,
        num_classes=num_classes,
    )[0]


def _predict_framewise_batch(
    model,
    audios: Sequence[np.ndarray],
    sr: int,
    device: str,
    num_classes: int,
) -> List[List[Tuple[float, float, np.ndarray]]]:
    if not audios:
        return []
    if all(int(a.shape[0]) <= 0 for a in audios):
        return [[] for _ in audios]

    lengths = [int(a.shape[0]) for a in audios]
    max_len = max(lengths) if lengths else 0
    if max_len <= 0:
        return [[] for _ in audios]

    wav_tensors = [torch.as_tensor(a, dtype=torch.float32) for a in audios]
    wav = torch.nn.utils.rnn.pad_sequence(wav_tensors, batch_first=True).to(device)
    with torch.no_grad():
        out = model(waveform=wav)
    fw = _normalize_framewise(_get_framewise_tensor(out), num_classes=num_classes)  # [B, C, T]
    if torch.min(fw) < 0.0 or torch.max(fw) > 1.0:
        fw = torch.sigmoid(fw)
    probs_all = fw.transpose(1, 2).detach().cpu().numpy().astype(np.float32, copy=False)  # [B, T, C]

    out_frames: List[List[Tuple[float, float, np.ndarray]]] = []
    total_steps = int(probs_all.shape[1])
    for idx, length in enumerate(lengths):
        if length <= 0:
            out_frames.append([])
            continue
        clip_dur = float(length) / float(sr)
        clip_steps = max(1, int(round((float(length) / float(max_len)) * float(total_steps))))
        clip_steps = min(clip_steps, total_steps)
        probs = probs_all[idx, :clip_steps, :]
        step_dur = clip_dur / float(max(clip_steps, 1))
        frames: List[Tuple[float, float, np.ndarray]] = []
        for i, row in enumerate(probs):
            t0 = i * step_dur
            t1 = min((i + 1) * step_dur, clip_dur)
            frames.append((t0, t1, row))
        out_frames.append(frames)
    return out_frames


def _resolve_label_indices(classes: Sequence[str], target_labels: Sequence[str]) -> Tuple[List[int], List[str]]:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    indices: List[int] = []
    names: List[str] = []
    for label in target_labels:
        idx = label_to_idx.get(label)
        if idx is None:
            logger.warning("Label '%s' not found in model classes; skipping.", label)
            continue
        indices.append(idx)
        names.append(label)
    return indices, names


def _aggregate_dominant_events(
    frames: Sequence[Tuple[float, float, np.ndarray]],
    selected_probs: np.ndarray,
    label_names: Sequence[str],
    min_prob: float,
    merge_gap_s: float,
    min_duration_s: float,
    round_digits: Optional[int],
    segment_start: float,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    def _finalize(event: Optional[Dict[str, object]]) -> None:
        if event is None:
            return
        duration = float(event["end"]) - float(event["start"])
        if duration < min_duration_s:
            return
        score_mean = float(event["_score_sum"]) / max(int(event["_count"]), 1)
        out = {
            "label": event["label"],
            "start": float(event["start"]),
            "end": float(event["end"]),
            "score_max": float(event["score_max"]),
            "score_mean": score_mean,
            "num_frames": int(event["_count"]),
        }
        if round_digits is not None:
            out["start"] = round(out["start"], round_digits)
            out["end"] = round(out["end"], round_digits)
            out["score_max"] = round(out["score_max"], round_digits)
            out["score_mean"] = round(out["score_mean"], round_digits)
        events.append(out)

    for (t0, t1, _), row in zip(frames, selected_probs):
        idx = int(np.argmax(row))
        score = float(row[idx])
        label: Optional[str] = label_names[idx] if score >= min_prob else None

        abs_t0 = segment_start + float(t0)
        abs_t1 = segment_start + float(t1)

        if label is None:
            _finalize(current)
            current = None
            continue

        if (
            current is not None
            and current["label"] == label
            and abs_t0 <= float(current["end"]) + merge_gap_s
        ):
            current["end"] = max(float(current["end"]), abs_t1)
            current["score_max"] = max(float(current["score_max"]), score)
            current["_score_sum"] = float(current["_score_sum"]) + score
            current["_count"] = int(current["_count"]) + 1
            continue

        _finalize(current)
        current = {
            "label": label,
            "start": abs_t0,
            "end": abs_t1,
            "score_max": score,
            "_score_sum": score,
            "_count": 1,
        }

    _finalize(current)
    return events


def _build_topk_frame_matrix(
    frames: Sequence[Tuple[float, float, np.ndarray]],
    selected_probs: np.ndarray,
    *,
    top_k: int,
    round_digits: Optional[int],
) -> Dict[str, object]:
    top_idx_rows: List[List[int]] = []
    top_prob_rows: List[List[float]] = []

    if selected_probs.ndim != 2:
        return {
            "k": int(max(1, top_k)),
            "top_idx": top_idx_rows,
            "top_prob": top_prob_rows,
        }

    num_labels = int(selected_probs.shape[1])
    k = int(max(1, min(top_k, num_labels))) if num_labels > 0 else 0
    if k <= 0:
        return {
            "k": 0,
            "top_idx": top_idx_rows,
            "top_prob": top_prob_rows,
        }

    for (_, _, _), row in zip(frames, selected_probs):
        idx = np.argpartition(-row, kth=k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        probs = [float(row[int(i)]) for i in idx]
        if round_digits is not None:
            probs = [round(p, round_digits) for p in probs]
        top_idx_rows.append([int(i) for i in idx.tolist()])
        top_prob_rows.append(probs)

    return {
        "k": int(k),
        "top_idx": top_idx_rows,
        "top_prob": top_prob_rows,
    }


def _tag_audio(
    audio: np.ndarray,
    sr: int,
    model,
    label_indices: Sequence[int],
    label_names: Sequence[str],
    classes: Sequence[str],
    round_digits: Optional[int],
    device: str,
    save_raw_frames: bool,
    raw_top_k: int,
    top_k: int,
    min_prob: float,
    segment_start: float = 0.0,
) -> Dict[str, object]:
    frames = _predict_framewise(
        model=model,
        audio=audio,
        sr=sr,
        device=device,
        num_classes=len(classes),
    )

    if not frames:
        result: Dict[str, object] = {
            "audio_tags_source": "sed_framewise_center",
            "audio_tag_topk": {
                **_build_topk_frame_matrix(
                    [],
                    np.zeros((0, len(label_names)), dtype=np.float32),
                    top_k=top_k,
                    round_digits=round_digits,
                ),
            },
        }
        if save_raw_frames:
            result["audio_tag_frames_raw"] = []
        return result

    selected_probs = np.stack(
        [[float(probs[idx]) for idx in label_indices] for _, _, probs in frames],
        axis=0,
    )
    raw_entries = []
    for t0, t1, probs in frames:
        if save_raw_frames:
            top_idx = np.argsort(-probs)[: max(raw_top_k, 1)]
            top_labels = [
                {"label": classes[int(i)], "score": float(probs[int(i)])}
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
        "audio_tags_source": "sed_framewise_center",
        "audio_tag_topk": {
            **_build_topk_frame_matrix(
                frames,
                selected_probs,
                top_k=top_k,
                round_digits=round_digits,
            ),
        },
    }
    if save_raw_frames:
        result["audio_tag_frames_raw"] = raw_entries
    return result


def _tag_audio_from_frames(
    frames: List[Tuple[float, float, np.ndarray]],
    label_indices: Sequence[int],
    label_names: Sequence[str],
    classes: Sequence[str],
    round_digits: Optional[int],
    save_raw_frames: bool,
    raw_top_k: int,
    top_k: int,
    min_prob: float,
    segment_start: float = 0.0,
) -> Dict[str, object]:
    if not frames:
        result: Dict[str, object] = {
            "audio_tags_source": "sed_framewise_center",
            "audio_tag_topk": {
                **_build_topk_frame_matrix(
                    [],
                    np.zeros((0, len(label_names)), dtype=np.float32),
                    top_k=top_k,
                    round_digits=round_digits,
                ),
            },
        }
        if save_raw_frames:
            result["audio_tag_frames_raw"] = []
        return result

    selected_probs = np.stack(
        [[float(probs[idx]) for idx in label_indices] for _, _, probs in frames],
        axis=0,
    )
    raw_entries = []
    for t0, t1, probs in frames:
        if save_raw_frames:
            top_idx = np.argsort(-probs)[: max(raw_top_k, 1)]
            top_labels = [
                {"label": classes[int(i)], "score": float(probs[int(i)])}
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
        "audio_tags_source": "sed_framewise_center",
        "audio_tag_topk": {
            **_build_topk_frame_matrix(
                frames,
                selected_probs,
                top_k=top_k,
                round_digits=round_digits,
            ),
        },
    }
    if save_raw_frames:
        result["audio_tag_frames_raw"] = raw_entries
    return result


def _tag_entry_batch(
    entries: Sequence[Dict],
    model,
    label_indices: Sequence[int],
    label_names: Sequence[str],
    classes: Sequence[str],
    base_dir: Path,
    sample_rate: int,
    round_digits: Optional[int],
    save_raw_frames: bool,
    raw_top_k: int,
    top_k: int,
    min_prob: float,
    batch_size: int,
    cache_audio: bool,
    audio_cache: Dict[Path, Tuple[np.ndarray, int]],
    device: str,
) -> List[Dict]:
    tasks: List[Dict[str, object]] = []
    for entry in entries:
        segments = None
        if isinstance(entry.get("final_segments"), list):
            segments = entry["final_segments"]
        elif isinstance(entry.get("segments"), list):
            segments = entry["segments"]
        if segments is None:
            segments = [entry]
        for segment in segments:
            audio_path = _resolve_audio_path(entry, segment, base_dir)
            if audio_path is None or not audio_path.exists():
                continue
            cache = audio_cache if cache_audio else {}
            audio, sr = _load_audio(audio_path, target_sr=sample_rate, cache=cache)
            tasks.append({"segment": segment, "audio": audio, "sr": sr})

    if not tasks:
        return list(entries)

    eff_bs = max(1, int(batch_size))
    for start in range(0, len(tasks), eff_bs):
        chunk = tasks[start : start + eff_bs]
        audios = [t["audio"] for t in chunk]
        frames_batch = _predict_framewise_batch(
            model=model,
            audios=audios,
            sr=sample_rate,
            device=device,
            num_classes=len(classes),
        )
        for task, frames in zip(chunk, frames_batch):
            segment = task["segment"]
            segment.update(
                _tag_audio_from_frames(
                    frames=frames,
                    label_indices=label_indices,
                    label_names=label_names,
                    classes=classes,
                    round_digits=round_digits,
                    save_raw_frames=save_raw_frames,
                    raw_top_k=raw_top_k,
                    top_k=top_k,
                    min_prob=min_prob,
                    segment_start=0.0,
                )
            )
    return list(entries)


def _tag_entry(
    entry: Dict,
    model,
    label_indices: Sequence[int],
    label_names: Sequence[str],
    classes: Sequence[str],
    base_dir: Path,
    sample_rate: int,
    round_digits: Optional[int],
    save_raw_frames: bool,
    raw_top_k: int,
    top_k: int,
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
                label_indices=label_indices,
                label_names=label_names,
                classes=classes,
                round_digits=round_digits,
                device=device,
                save_raw_frames=save_raw_frames,
                raw_top_k=raw_top_k,
                top_k=top_k,
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
    parser = argparse.ArgumentParser(description="Framewise AudioSet tagging with CNN8RNN SED")
    parser.add_argument("manifest", type=Path, help="Path to manifest.jsonl")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL path")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Inference device (default: auto)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--round", type=int, default=3, help="Use -1 to disable")
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument("--raw-top-k", type=int, default=10)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Always keep Top-K labels per frame in audio_tag_topk (default: 5).",
    )
    parser.add_argument(
        "--include-speech-tags",
        action="store_true",
        help="Include speech-like labels. Default keeps only non-speech labels.",
    )
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
    if args.min_prob < 0.0 or args.min_prob > 1.0:
        logger.error("--min-prob must be between 0.0 and 1.0")
        return 1
    if args.batch_size < 1:
        logger.error("--batch-size must be >= 1")
        return 1
    if args.top_k < 1:
        logger.error("--top-k must be >= 1")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model: %s", args.model_id)
    # Build model from config first (no meta-device init), then load checkpoint.
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    ckpt_path = hf_hub_download(repo_id=args.model_id, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if hasattr(model, "classes"):
        classes: Sequence[str] = model.classes
    else:
        classes_path = hf_hub_download(repo_id=args.model_id, filename="classes.txt")
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if not classes:
            logger.error("No classes available from model or classes.txt.")
            return 1
    target_labels = TARGET_LABELS if args.include_speech_tags else NON_SPEECH_TARGET_LABELS
    label_indices, label_names = _resolve_label_indices(classes, target_labels)
    if not label_indices:
        logger.error("No target labels resolved against model classes. Aborting.")
        return 1

    round_digits = None if args.round < 0 else args.round
    output_path = args.output or args.manifest.with_suffix(".tagged.sed.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = args.manifest.parent.resolve()
    processed_audio_keys = _load_processed_audio_keys(output_path, base_dir)

    config_path = _write_run_config(
        output_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "audio_pattern_recognition_sed",
            "manifest": str(args.manifest.resolve()),
            "output_jsonl": str(output_path.resolve()),
            "model_id": args.model_id,
            "sample_rate": args.sample_rate,
            "full_clip_inference": True,
            "batch_size": args.batch_size,
            "round_digits": round_digits,
            "min_prob": args.min_prob,
            "top_k": args.top_k,
            "include_speech_tags": bool(args.include_speech_tags),
            "device": device,
            "target_labels": label_names,
        },
    )
    logger.info("Run config written to %s", config_path)

    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

    pending_entries: List[Dict] = []
    pending_keys: List[List[str]] = []

    with open(output_path, "a", encoding="utf-8") as outfile:
        for _, entry in _iter_jsonl(args.manifest):
            entry_audio_keys = _collect_audio_keys(entry, base_dir)
            if entry_audio_keys and all(key in processed_audio_keys for key in entry_audio_keys):
                continue
            pending_entries.append(entry)
            pending_keys.append(entry_audio_keys)
            if len(pending_entries) < args.batch_size:
                continue

            tagged_batch = _tag_entry_batch(
                entries=pending_entries,
                model=model,
                label_indices=label_indices,
                label_names=label_names,
                classes=classes,
                base_dir=base_dir,
                sample_rate=args.sample_rate,
                round_digits=round_digits,
                save_raw_frames=args.save_raw_frames,
                raw_top_k=args.raw_top_k,
                top_k=args.top_k,
                min_prob=args.min_prob,
                batch_size=args.batch_size,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
                device=device,
            )
            for tagged, entry_audio_keys in zip(tagged_batch, pending_keys):
                if args.minimal_output:
                    slim = {
                        "audio_path": tagged.get("audio_path")
                        or tagged.get("audio_filepath")
                        or tagged.get("path")
                        or tagged.get("audio"),
                        "text": tagged.get("text"),
                        "audio_tags_source": tagged.get("audio_tags_source"),
                        "audio_tag_topk": tagged.get("audio_tag_topk"),
                    }
                    if "audio_tag_frames_raw" in tagged:
                        slim["audio_tag_frames_raw"] = tagged.get("audio_tag_frames_raw")
                    outfile.write(json.dumps(slim, ensure_ascii=False) + "\n")
                else:
                    outfile.write(json.dumps(tagged, ensure_ascii=False) + "\n")
                processed_audio_keys.update(entry_audio_keys)
            pending_entries = []
            pending_keys = []

        if pending_entries:
            tagged_batch = _tag_entry_batch(
                entries=pending_entries,
                model=model,
                label_indices=label_indices,
                label_names=label_names,
                classes=classes,
                base_dir=base_dir,
                sample_rate=args.sample_rate,
                round_digits=round_digits,
                save_raw_frames=args.save_raw_frames,
                raw_top_k=args.raw_top_k,
                top_k=args.top_k,
                min_prob=args.min_prob,
                batch_size=args.batch_size,
                cache_audio=not args.no_cache,
                audio_cache=audio_cache,
                device=device,
            )
            for tagged, entry_audio_keys in zip(tagged_batch, pending_keys):
                if args.minimal_output:
                    slim = {
                        "audio_path": tagged.get("audio_path")
                        or tagged.get("audio_filepath")
                        or tagged.get("path")
                        or tagged.get("audio"),
                        "text": tagged.get("text"),
                        "audio_tags_source": tagged.get("audio_tags_source"),
                        "audio_tag_topk": tagged.get("audio_tag_topk"),
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
