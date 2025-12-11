"""
STT4SG Transcribe - Main Entry Point

Speech-to-Text using Faster-Whisper + PyAnnote + CTC alignment.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from pipeline import TranscriptionConfig, TranscriptionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization and word-level alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.mp3
  python main.py audio.mp3 -l de -o output.srt
  python main.py audio.mp3 --vad
  python main.py audio.mp3 --diarization -n 2 -m medium
        """
    )
    
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output SRT path (default: outputs/srt/<filename>.srt)")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model (default: large-v3)")
    parser.add_argument("-l", "--language", help="Language code (auto-detect if not specified)")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    
    diar = parser.add_argument_group("Diarization")
    diar.add_argument("--vad", action="store_true", help="Enable VAD (off by default)")
    diar.add_argument("--diarization", action="store_true", help="Enable speaker diarization (implies --vad)")
    diar.add_argument("-n", "--num-speakers", type=int, help="Number of speakers")
    diar.add_argument("--min-speakers", type=int)
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
    
    config = TranscriptionConfig(
        whisper_model=args.model,
        language=args.language,
        task=args.task,
        use_vad=args.vad or args.diarization,
        use_diarization=args.diarization,
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
    logger.info(f"Diarization: {'on' if config.use_diarization else 'off'}, Alignment: {'on' if config.use_alignment else 'off'}")
    
    try:
        pipeline = TranscriptionPipeline(config)
        result = pipeline.transcribe(audio_path, args.output, save_logs=not args.no_logs)
        
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
