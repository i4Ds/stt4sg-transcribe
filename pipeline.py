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
from faster_whisper import WhisperModel

from vad_diarization import CombinedVADDiarization, DiarizationResult, save_vad_diarization_log
from ctc_alignment import CTCAligner, save_alignment_log
from srt_formatter import segments_to_srt, save_transcription_log

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_16k_wav(audio_path: Union[str, Path], target_sr: int = 16000) -> Optional[Path]:
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
    word_timestamps: bool = True
    
    use_alignment: bool = True
    alignment_model: Optional[str] = None
    
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
                self.config.whisper_model, device=self.config.device, compute_type=self.config.compute_type
            )
            logger.info("Whisper model loaded")
        return self._whisper_model
    
    @property
    def vad_diarization(self) -> CombinedVADDiarization:
        if self._vad_diarization is None:
            self._vad_diarization = CombinedVADDiarization(
                device=self.config.device,
                use_auth_token=self.config.hf_token,
                diarization_method=self.config.diarization_method,
                diarization_params=self.config.diarization_params,
            )
        return self._vad_diarization
    
    def get_aligner(self, language: str) -> CTCAligner:
        if self._aligner is None or self._aligner.language != language:
            self._aligner = CTCAligner(language=language, device=self.config.device, model_name=self.config.alignment_model)
        return self._aligner
    
    def transcribe(self, audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, save_logs: bool = True) -> Dict:
        """Run the full transcription pipeline."""
        audio_path = Path(audio_path)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = audio_path.stem
        
        # Create run-specific log folder
        run_log_dir = Path("logs/timestamps") / f"{base_name}_{run_id}"
        if save_logs:
            run_log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting transcription: {audio_path}")
        results = {"audio_file": str(audio_path), "run_id": run_id, "config": self.config.to_dict()}
        
        # Resample audio to 16kHz mono WAV for pyannote if needed
        temp_audio_path = None
        working_audio_path = audio_path
        if self.config.use_vad or self.config.use_diarization:
            temp_audio_path = ensure_16k_wav(audio_path, target_sr=16000)
            if temp_audio_path:
                working_audio_path = temp_audio_path
        
        try:
            # Step 1: VAD/Diarization
            diarization_result = None
            if self.config.use_vad or self.config.use_diarization:
                logger.info("Step 1: VAD/Diarization...")
                diarization_result = self.vad_diarization.process(
                    working_audio_path, use_diarization=self.config.use_diarization,
                    num_speakers=self.config.num_speakers, min_speakers=self.config.min_speakers,
                    max_speakers=self.config.max_speakers, vad_min_duration=self.config.vad_min_duration,
                    vad_merge_threshold=self.config.vad_merge_threshold
                )
                results["vad_diarization"] = diarization_result.to_dict()
                if save_logs:
                    save_vad_diarization_log(diarization_result, run_log_dir / "vad_diarization.json", str(audio_path))
            
            # Step 2: Transcription (use original audio for whisper - it handles any format)
            logger.info("Step 2: Transcription...")
            transcribe_kwargs = {
                "language": self.config.language,
                "task": self.config.task,
                "beam_size": self.config.beam_size,
                "word_timestamps": self.config.word_timestamps,
            }
            if diarization_result and diarization_result.segments:
                clip_timestamps = diarization_result.get_clip_timestamps()
                if clip_timestamps:
                    transcribe_kwargs["clip_timestamps"] = clip_timestamps
                    logger.info(f"Using {len(diarization_result.segments)} speech segments")
            
            segments_gen, info = self.whisper_model.transcribe(str(audio_path), **transcribe_kwargs)
        finally:
            # Clean up temp file
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()
                logger.debug(f"Cleaned up temp audio: {temp_audio_path}")
        
        segments = []
        for seg in segments_gen:
            seg_dict = {
                "id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob, "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob
            }
            if seg.words:
                seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
            segments.append(seg_dict)
        
        transcription_result = {"language": info.language, "language_probability": info.language_probability, "duration": info.duration, "segments": segments}
        results["transcription"] = transcription_result
        if save_logs:
            save_transcription_log(segments, run_log_dir / "transcription.json", str(audio_path), info.language)
        
        # Step 3: CTC Alignment
        alignment_result = None
        if self.config.use_alignment and info.language:
            logger.info("Step 3: CTC alignment...")
            try:
                aligner = self.get_aligner(info.language)
                alignment_result = aligner.align([{"text": s["text"], "start": s["start"], "end": s["end"], "avg_logprob": s.get("avg_logprob")} for s in segments], audio_path)
                results["alignment"] = alignment_result.to_dict()
                if save_logs:
                    save_alignment_log(alignment_result, run_log_dir / "alignment.json", str(audio_path))
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")
        
        # Step 4: Assign speakers
        if diarization_result and self.config.use_diarization:
            logger.info("Step 4: Speaker assignment...")
            final_segments = self._assign_speakers(
                alignment_result.segments if alignment_result else segments,
                diarization_result
            )
            if save_logs:
                self._save_speaker_log(final_segments, run_log_dir / "speaker_alignment.json")
        else:
            final_segments = [s.to_dict() for s in alignment_result.segments] if alignment_result else segments
        
        results["final_segments"] = final_segments
        
        # Step 5: Generate SRT
        logger.info("Step 5: Generating SRT...")
        output_path = Path(output_path) if output_path else self.output_dir / f"{base_name}.srt"
        srt_content = segments_to_srt(final_segments, output_path, include_speaker=self.config.include_speaker_labels and self.config.use_diarization)
        results["srt_path"] = str(output_path)
        results["srt_content"] = srt_content
        results["log_dir"] = str(run_log_dir) if save_logs else None
        
        logger.info(f"Done! SRT: {output_path}")
        if save_logs:
            logger.info(f"Logs: {run_log_dir}")
        
        return results
    
    def _assign_speakers(self, segments, diarization_result: DiarizationResult) -> List[Dict]:
        if not diarization_result.segments:
            return segments if isinstance(segments[0], dict) else [s.to_dict() for s in segments]
        
        final_segments = []
        for seg in segments:
            seg_dict = seg.to_dict() if hasattr(seg, 'to_dict') else dict(seg)
            seg_start, seg_end = seg_dict.get("start", 0), seg_dict.get("end", 0)
            
            best_speaker, best_overlap = None, 0
            for diar_seg in diarization_result.segments:
                overlap = max(0, min(seg_end, diar_seg.end) - max(seg_start, diar_seg.start))
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
            "segments": [{"text": s.get("text"), "start": s.get("start"), "end": s.get("end"), "speaker": s.get("speaker")} for s in segments],
            "speaker_statistics": speakers
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved speaker log to {output_path}")


def transcribe_file(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    whisper_model: str = "large-v3",
    language: Optional[str] = None,
    use_vad: bool = True,
    use_diarization: bool = False,
    diarization_method: str = "pyannote",
    diarization_params: Optional[Dict[str, Any]] = None,
    num_speakers: Optional[int] = None,
    use_alignment: bool = True,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Convenience function to transcribe a single audio file."""
    config = TranscriptionConfig(
        whisper_model=whisper_model, language=language,
        use_vad=use_vad, use_diarization=use_diarization,
        diarization_method=diarization_method, diarization_params=diarization_params,
        num_speakers=num_speakers,
        use_alignment=use_alignment, hf_token=hf_token
    )
    if device:
        config.device = device
        config.compute_type = "float16" if device == "cuda" else "float32"
    
    return TranscriptionPipeline(config).transcribe(audio_path, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker diarization")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output SRT path")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model")
    parser.add_argument("-l", "--language", help="Language code")
    parser.add_argument("--no-vad", dest="vad", action="store_false")
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

    result = transcribe_file(
        args.audio_path, args.output, args.model, args.language,
        args.vad, args.diarization, args.diarization_method, diar_params,
        args.num_speakers, not args.no_alignment,
        args.hf_token, args.device
    )
    
    print(f"\nDone! SRT: {result['srt_path']}")
    print(f"Language: {result['transcription']['language']}, Segments: {len(result['final_segments'])}")
    if result.get('log_dir'):
        print(f"Logs: {result['log_dir']}")
