"""
STT4SG Transcribe - Main Entry Point

Speech-to-Text using Faster-Whisper + PyAnnote + CTC alignment.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pipeline import TranscriptionConfig, TranscriptionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with optional speaker diarization and word-level alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.mp3
  python main.py audio.mp3 -l de -o output.srt
  python main.py audio.mp3 --no-vad
  python main.py audio.mp3 --diarization -n 2 -m medium
        """
    )
    
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output SRT path (default: outputs/srt/<filename>.srt)")
    parser.add_argument("--output-dir", help="Output folder for SRTs (default: outputs/srt)")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model (default: large-v3)")
    parser.add_argument("-l", "--language", help="Language code (auto-detect if not specified)")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    
    diar = parser.add_argument_group("Diarization")
    diar.add_argument("--no-vad", dest="vad", action="store_false", help="Disable VAD (on by default)")
    diar.add_argument("--vad-method", default="pyannote", help="pyannote | speechbrain | nemo | silero")
    diar.add_argument("--vad-params", help="JSON dict of VAD params")
    diar.add_argument("--diarization", action="store_true", help="Enable speaker diarization (implies VAD)")
    diar.add_argument("--diarization-method", default="pyannote", help="pyannote | nemo | speechbrain")
    diar.add_argument("--diarization-params", help="JSON dict of diarization params")
    diar.add_argument("-n", "--num-speakers", type=int, help="Number of speakers")
    diar.add_argument("--min-speakers", type=int, default=2)
    diar.add_argument("--max-speakers", type=int)
    
    align = parser.add_argument_group("Alignment")
    align.add_argument("--no-alignment", action="store_true", help="Disable CTC alignment")
    align.add_argument("--alignment-model", help="Custom alignment model")
    
    device = parser.add_argument_group("Device")
    device.add_argument("--device", choices=["cuda", "cpu"])
    device.add_argument("--compute-type", choices=["float16", "float32", "int8"])
    
    output = parser.add_argument_group("Output")
    output.add_argument("--no-speaker-labels", action="store_true")
    output.add_argument("--no-logs", action="store_true", help="Don't save log files")
    
    auth = parser.add_argument_group("Auth")
    auth.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env)")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)
    
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    diar_params = None
    if args.diarization_params:
        diar_params = json.loads(args.diarization_params)
    vad_params = None
    if args.vad_params:
        vad_params = json.loads(args.vad_params)

    config = TranscriptionConfig(
        whisper_model=args.model,
        language=args.language,
        task=args.task,
        use_vad=args.vad or args.diarization,
        use_diarization=args.diarization,
        vad_method=args.vad_method,
        vad_params=vad_params,
        diarization_method=args.diarization_method,
        diarization_params=diar_params,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        use_alignment=not args.no_alignment,
        alignment_model=args.alignment_model,
        include_speaker_labels=not args.no_speaker_labels,
        hf_token=hf_token,
    )
    
    if args.device:
        config.device = args.device
    if args.compute_type:
        config.compute_type = args.compute_type
    elif config.device == "cpu":
        config.compute_type = "float32"
    
    logger.info(f"Transcribing: {audio_path}")
    logger.info(f"Model: {config.whisper_model}, Device: {config.device}")
    logger.info(
        f"VAD: {'on' if config.use_vad else 'off'} ({config.vad_method}), "
        f"Diarization: {'on' if config.use_diarization else 'off'} ({config.diarization_method}), "
        f"Alignment: {'on' if config.use_alignment else 'off'}"
    )
    
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{audio_path.stem}.srt"

    try:
        pipeline = TranscriptionPipeline(config)
        result = pipeline.transcribe(audio_path, output_path, save_logs=not args.no_logs)
        
        print("\n" + "="*50)
        print("TRANSCRIPTION COMPLETE")
        print("="*50)
        print(f"Audio: {audio_path}")
        print(f"Language: {result['transcription']['language']}")
        print(f"Duration: {result['transcription']['duration']:.1f}s")
        print(f"Segments: {len(result['final_segments'])}")
        print(f"SRT: {result['srt_path']}")
        if result.get('log_dir'):
            print(f"Logs: {result['log_dir']}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
