"""
Main Transcription Pipeline.

Combines:
1. PyAnnote VAD/Diarization
2. Faster-Whisper transcription
3. CTC forced alignment
4. SRT output generation (via pysubs2)

Logs saved to one folder per run in logs/timestamps/<run_id>/
"""

import json
import logging
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel, BatchedInferencePipeline

from vad_diarization import (
    CombinedVADDiarization,
    DiarizationResult,
    save_vad_diarization_log,
    generate_chunk_timestamps,
)
from ctc_alignment import CTCAligner, save_alignment_log
from srt_formatter import segments_to_srt, save_transcription_log

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_16k_wav(
    audio_path: Union[str, Path], target_sr: int = 16000
) -> Optional[Path]:
    """
    Ensure audio is 16kHz mono WAV for pyannote using pydub.

    Returns a temp WAV path if conversion is needed, otherwise None.
    """
    audio_path = Path(audio_path)
    audio = AudioSegment.from_file(audio_path)

    needs_resample = audio.frame_rate != target_sr
    needs_mono = audio.channels != 1
    is_wav = audio_path.suffix.lower() == ".wav"

    if not (needs_resample or needs_mono or not is_wav):
        logger.info(f"Audio already {target_sr}Hz mono WAV; no resample needed")
        return None

    temp_dir = Path(tempfile.gettempdir()) / "stt4sg_transcribe"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{audio_path.stem}_16khz.wav"

    logger.info(f"Resampling to {target_sr}Hz mono WAV: {audio_path} -> {temp_path}")
    if needs_resample:
        audio = audio.set_frame_rate(target_sr)
    if needs_mono:
        audio = audio.set_channels(1)

    audio.export(temp_path, format="wav")
    return temp_path


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription pipeline."""

    whisper_model: str = "large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"

    use_vad: bool = True
    use_diarization: bool = False
    vad_method: str = "silero"
    vad_params: Optional[Dict[str, Any]] = None
    diarization_method: str = "pyannote"
    diarization_params: Optional[Dict[str, Any]] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    vad_min_duration: float = 0.5
    vad_merge_threshold: float = 0.3

    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    batch_size: Optional[int] = None
    word_timestamps: bool = True
    log_progress: bool = False

    use_alignment: bool = True
    alignment_model: Optional[str] = None

    generate_srt: bool = True
    include_speaker_labels: bool = True
    hf_token: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptionPipeline:
    """Main transcription pipeline combining all components."""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._whisper_model = None
        self._vad_diarization = None
        self._aligner = None
        self.output_dir = Path("outputs/srt")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def whisper_model(self) -> WhisperModel:
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self._whisper_model = WhisperModel(
                self.config.whisper_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            logger.info("Whisper model loaded")
        return self._whisper_model

    @property
    def batched_pipeline(self) -> BatchedInferencePipeline:
        """Returns a BatchedInferencePipeline wrapping the WhisperModel."""
        return BatchedInferencePipeline(model=self.whisper_model)

    @property
    def vad_diarization(self) -> CombinedVADDiarization:
        if self._vad_diarization is None:
            self._vad_diarization = CombinedVADDiarization(
                device=self.config.device,
                use_auth_token=self.config.hf_token,
                vad_method=self.config.vad_method,
                vad_params=self.config.vad_params,
                diarization_method=self.config.diarization_method,
                diarization_params=self.config.diarization_params,
            )
        return self._vad_diarization

    def get_aligner(self, language: str) -> CTCAligner:
        if self._aligner is None or self._aligner.language != language:
            self._aligner = CTCAligner(
                language=language,
                device=self.config.device,
                model_name=self.config.alignment_model,
            )
        return self._aligner

    def transcribe(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_logs: bool = True,
    ) -> Dict:
        """
        Run the full transcription pipeline.

        Pipeline order: VAD → Transcription → Alignment → Diarization → Speaker/Purity calculation
        """
        audio_path = Path(audio_path)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = audio_path.stem

        output_path = Path(output_path) if output_path else None

        # Create run-specific log folder
        if output_path:
            output_root = output_path.parent
            run_log_dir = output_root / "logs" / f"{base_name}_{run_id}"
        else:
            run_log_dir = Path("logs/timestamps") / f"{base_name}_{run_id}"
        if save_logs:
            run_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting transcription: {audio_path}")
        results = {
            "audio_file": str(audio_path),
            "run_id": run_id,
            "config": self.config.to_dict(),
        }

        vad_segments = None
        diarization_result = None
        temp_audio_path = None

        # Step 1: VAD (runs FIRST to get speech segments for transcription)
        if self.config.use_vad:
            logger.info("Step 1: Voice Activity Detection...")
            temp_audio_path = ensure_16k_wav(audio_path, target_sr=16000)
            working_audio_path = temp_audio_path if temp_audio_path else audio_path

            vad_provider = self.vad_diarization._get_vad_provider()
            vad_segments = vad_provider.detect_speech(
                working_audio_path,
                min_duration=self.config.vad_min_duration,
                merge_threshold=self.config.vad_merge_threshold,
            )
            results["vad_segments"] = [
                {"start": s.start, "end": s.end} for s in vad_segments
            ]
            logger.info(f"VAD detected {len(vad_segments)} speech segments")
            if save_logs:
                self._save_vad_log(
                    vad_segments, run_log_dir / "vad.json", str(audio_path)
                )

        # Step 2: Transcription (use VAD segments as clip_timestamps if available)
        logger.info("Step 2: Transcription...")
        transcribe_kwargs = {
            "language": self.config.language,
            "task": self.config.task,
            "beam_size": self.config.beam_size,
            "word_timestamps": self.config.word_timestamps,
            "log_progress": self.config.log_progress,
        }

        use_batched = self.config.batch_size is not None

        # Use VAD segments as clip_timestamps for transcription
        # Note: BatchedInferencePipeline uses list of dicts, WhisperModel uses flat list of floats
        if vad_segments and len(vad_segments) > 0:
            if use_batched:
                clip_timestamps = [
                    {"start": s.start, "end": s.end} for s in vad_segments
                ]
            else:
                # WhisperModel expects flat list: [start1, end1, start2, end2, ...]
                clip_timestamps = []
                for s in vad_segments:
                    clip_timestamps.extend([s.start, s.end])
            transcribe_kwargs["clip_timestamps"] = clip_timestamps
            logger.info(f"Using {len(vad_segments)} VAD segments as clip_timestamps")
        elif use_batched:
            # Fallback to fixed chunks if no VAD but batched inference requested
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            clip_timestamps = generate_chunk_timestamps(duration)
            clip_timestamps[-1]["end"] = min(
                duration,
                max(
                    clip_timestamps[-1]["start"] + 0.1,
                    clip_timestamps[-1]["end"] - 0.01,
                ),
            )
            transcribe_kwargs["clip_timestamps"] = clip_timestamps
            logger.info(f"Using {len(clip_timestamps)} fixed 30s chunks (no VAD)")
        else:
            transcribe_kwargs["vad_filter"] = False

        if use_batched:
            logger.info(
                f"Using batched inference with batch_size={self.config.batch_size}"
            )
            transcribe_kwargs["batch_size"] = self.config.batch_size
            pipeline = self.batched_pipeline
        else:
            pipeline = self.whisper_model

        segments_gen, info = pipeline.transcribe(str(audio_path), **transcribe_kwargs)

        segments = []
        for seg in segments_gen:
            seg_dict = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
            }
            if seg.words:
                seg_dict["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]
            segments.append(seg_dict)

        transcription_result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }
        results["transcription"] = transcription_result
        if save_logs:
            save_transcription_log(
                segments,
                run_log_dir / "transcription.json",
                str(audio_path),
                info.language,
            )

        # Step 3: CTC Alignment
        alignment_result = None
        if self.config.use_alignment and info.language:
            logger.info("Step 3: CTC alignment...")
            try:
                aligner = self.get_aligner(info.language)
                alignment_result = aligner.align(
                    [
                        {
                            "text": s["text"],
                            "start": s["start"],
                            "end": s["end"],
                            "avg_logprob": s.get("avg_logprob"),
                        }
                        for s in segments
                    ],
                    audio_path,
                )
                results["alignment"] = alignment_result.to_dict()
                if save_logs:
                    save_alignment_log(
                        alignment_result,
                        run_log_dir / "alignment.json",
                        str(audio_path),
                    )
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")

        # Step 4: Diarization (runs AFTER transcription to assign speakers)
        if self.config.use_diarization:
            logger.info("Step 4: Speaker Diarization...")
            if temp_audio_path is None:
                temp_audio_path = ensure_16k_wav(audio_path, target_sr=16000)
            working_audio_path = temp_audio_path if temp_audio_path else audio_path

            diar_provider = self.vad_diarization._get_diarization_provider()
            diarization_result = diar_provider.diarize(
                working_audio_path,
                num_speakers=self.config.num_speakers,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            results["diarization"] = diarization_result.to_dict()
            if save_logs:
                self._save_diarization_log(
                    diarization_result,
                    run_log_dir / "diarization.json",
                    str(audio_path),
                )

        # Cleanup temp audio
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.debug(f"Cleaned up temp audio: {temp_audio_path}")

        # Step 5: Assign speakers and calculate purity
        if diarization_result and self.config.use_diarization:
            logger.info("Step 5: Speaker assignment and purity calculation...")
            final_segments = self._assign_speakers_with_purity(
                alignment_result.segments if alignment_result else segments,
                diarization_result,
            )
            if save_logs:
                self._save_speaker_log(
                    final_segments, run_log_dir / "speaker_alignment.json"
                )
        else:
            final_segments = (
                [s.to_dict() for s in alignment_result.segments]
                if alignment_result
                else segments
            )

        results["final_segments"] = final_segments

        if self.config.generate_srt:
            # Step 6: Generate SRT
            logger.info("Step 6: Generating SRT...")
            output_path = (
                output_path if output_path else self.output_dir / f"{base_name}.srt"
            )
            srt_content = segments_to_srt(
                final_segments,
                output_path,
                include_speaker=self.config.include_speaker_labels
                and self.config.use_diarization,
            )
            results["srt_path"] = str(output_path)
            results["srt_content"] = srt_content
        else:
            logger.info("Step 6: Skipping SRT generation")
            results["srt_path"] = None
            results["srt_content"] = None
        results["log_dir"] = str(run_log_dir) if save_logs else None

        if results["srt_path"]:
            logger.info(f"Done! SRT: {results['srt_path']}")
        else:
            logger.info("Done! No SRT generated")
        if save_logs:
            logger.info(f"Logs: {run_log_dir}")

        return results

    def _assign_speakers_with_purity(
        self, segments, diarization_result: DiarizationResult
    ) -> List[Dict]:
        """
        Assign speakers to segments and calculate purity.

        Purity = time covered by dominant speaker / total segment duration
        A segment with purity < 1.0 means multiple speakers are present.
        """
        if not diarization_result.segments:
            return (
                segments
                if isinstance(segments[0], dict)
                else [s.to_dict() for s in segments]
            )

        final_segments = []
        for seg in segments:
            seg_dict = seg.to_dict() if hasattr(seg, "to_dict") else dict(seg)
            seg_start, seg_end = seg_dict.get("start", 0), seg_dict.get("end", 0)
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                seg_dict["speaker"] = None
                seg_dict["purity"] = 0.0
                seg_dict["speaker_overlaps"] = {}
                final_segments.append(seg_dict)
                continue

            # Calculate overlap with each speaker
            speaker_overlaps = {}
            for diar_seg in diarization_result.segments:
                overlap = max(
                    0, min(seg_end, diar_seg.end) - max(seg_start, diar_seg.start)
                )
                if overlap > 0:
                    speaker = diar_seg.speaker or "UNKNOWN"
                    speaker_overlaps[speaker] = (
                        speaker_overlaps.get(speaker, 0) + overlap
                    )

            if not speaker_overlaps:
                seg_dict["speaker"] = None
                seg_dict["purity"] = 0.0
                seg_dict["speaker_overlaps"] = {}
            else:
                # Find dominant speaker
                dominant_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                dominant_overlap = speaker_overlaps[dominant_speaker]
                total_speaker_time = sum(speaker_overlaps.values())

                # Purity = dominant speaker time / total speaker time in segment
                # (not segment duration, since there might be gaps)
                purity = (
                    dominant_overlap / total_speaker_time
                    if total_speaker_time > 0
                    else 0.0
                )

                # Coverage = how much of segment is covered by any speaker
                coverage = (
                    min(total_speaker_time / seg_duration, 1.0)
                    if seg_duration > 0
                    else 0.0
                )

                seg_dict["speaker"] = dominant_speaker
                seg_dict["purity"] = round(purity, 4)
                seg_dict["coverage"] = round(coverage, 4)
                seg_dict["speaker_overlaps"] = {
                    k: round(v, 4) for k, v in speaker_overlaps.items()
                }

            final_segments.append(seg_dict)
        return final_segments

    def _save_vad_log(self, segments: List, output_path: Path, audio_path: str):
        """Save VAD segments to JSON log file."""
        log_data = {
            "audio_file": audio_path,
            "num_segments": len(segments),
            "total_speech_duration": sum(s.end - s.start for s in segments),
            "segments": [
                {"start": s.start, "end": s.end, "duration": s.end - s.start}
                for s in segments
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved VAD log to {output_path}")

    def _save_diarization_log(
        self, result: DiarizationResult, output_path: Path, audio_path: str
    ):
        """Save diarization results to JSON log file."""
        speakers = {}
        for seg in result.segments:
            speaker = seg.speaker or "UNKNOWN"
            duration = seg.end - seg.start
            speakers.setdefault(speaker, {"total_duration": 0, "segment_count": 0})
            speakers[speaker]["total_duration"] += duration
            speakers[speaker]["segment_count"] += 1

        log_data = {
            "audio_file": audio_path,
            "num_speakers": result.num_speakers,
            "speaker_statistics": speakers,
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in result.segments
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved diarization log to {output_path}")

    def _assign_speakers(
        self, segments, diarization_result: DiarizationResult
    ) -> List[Dict]:
        if not diarization_result.segments:
            return (
                segments
                if isinstance(segments[0], dict)
                else [s.to_dict() for s in segments]
            )

        final_segments = []
        for seg in segments:
            seg_dict = seg.to_dict() if hasattr(seg, "to_dict") else dict(seg)
            seg_start, seg_end = seg_dict.get("start", 0), seg_dict.get("end", 0)

            best_speaker, best_overlap = None, 0
            for diar_seg in diarization_result.segments:
                overlap = max(
                    0, min(seg_end, diar_seg.end) - max(seg_start, diar_seg.start)
                )
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, diar_seg.speaker

            seg_dict["speaker"] = best_speaker
            final_segments.append(seg_dict)
        return final_segments

    def _save_speaker_log(self, segments: List[Dict], output_path: Path):
        speakers = {}
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            duration = seg.get("end", 0) - seg.get("start", 0)
            speakers.setdefault(speaker, {"total_duration": 0, "segment_count": 0})
            speakers[speaker]["total_duration"] += duration
            speakers[speaker]["segment_count"] += 1

        log_data = {
            "segments": [
                {
                    "text": s.get("text"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "speaker": s.get("speaker"),
                }
                for s in segments
            ],
            "speaker_statistics": speakers,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved speaker log to {output_path}")


def transcribe_file(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    whisper_model: str = "large-v3",
    language: Optional[str] = None,
    use_vad: bool = True,
    use_diarization: bool = False,
    vad_method: str = "silero",
    vad_params: Optional[Dict[str, Any]] = None,
    diarization_method: str = "pyannote",
    diarization_params: Optional[Dict[str, Any]] = None,
    num_speakers: Optional[int] = None,
    use_alignment: bool = True,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Convenience function to transcribe a single audio file."""
    config = TranscriptionConfig(
        whisper_model=whisper_model,
        language=language,
        use_vad=use_vad,
        use_diarization=use_diarization,
        vad_method=vad_method,
        vad_params=vad_params,
        diarization_method=diarization_method,
        diarization_params=diarization_params,
        num_speakers=num_speakers,
        use_alignment=use_alignment,
        hf_token=hf_token,
    )
    if device:
        config.device = device
        config.compute_type = "float16" if device == "cuda" else "float32"

    return TranscriptionPipeline(config).transcribe(audio_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio with optional speaker diarization"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output SRT path")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model")
    parser.add_argument("-l", "--language", help="Language code")
    parser.add_argument("--no-vad", dest="vad", action="store_false")
    parser.add_argument("--vad-method", default="silero")
    parser.add_argument("--vad-params", help="JSON dict of VAD params")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--diarization-method", default="pyannote")
    parser.add_argument("--diarization-params", help="JSON dict of diarization params")
    parser.add_argument("--no-alignment", action="store_true")
    parser.add_argument("-n", "--num-speakers", type=int)
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--device", choices=["cuda", "cpu"])

    args = parser.parse_args()

    diar_params = None
    if args.diarization_params:
        diar_params = json.loads(args.diarization_params)
    vad_params = None
    if args.vad_params:
        vad_params = json.loads(args.vad_params)

    result = transcribe_file(
        args.audio_path,
        args.output,
        args.model,
        args.language,
        args.vad,
        args.diarization,
        args.vad_method,
        vad_params,
        args.diarization_method,
        diar_params,
        args.num_speakers,
        not args.no_alignment,
        args.hf_token,
        args.device,
    )

    print(f"\nDone! SRT: {result['srt_path']}")
    print(
        f"Language: {result['transcription']['language']}, Segments: {len(result['final_segments'])}"
    )
    if result.get("log_dir"):
        print(f"Logs: {result['log_dir']}")
