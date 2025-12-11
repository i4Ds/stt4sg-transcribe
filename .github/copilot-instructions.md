# Important
activate enviroment first: source ./.venv/bin/activate or use the following python: `/home/kenfus/stt4sg-transcribe/.venv/bin/python`

# GitHub Copilot Instructions
A test file to run a transcription can be found at `test/97_Brugg.mp3` and it's transcription at `test/97_Brugg.srt`
Test your implementation with that file. 

# Goal
Your goal is to mirror what WhisperX does but only with faster-whisper. Faster-whisper contains all functionality already, except.

- PyAnnote for VAD and speaker diarization.
- Forced alignment with classic ctc segmentation.


For this, example script from WhisperX is in `alignment.py`. For the VAD via pyannote you need to implement it from scratch; Basically, use pyannote to get speech segments and speaker diarization, then pass them to the transcribe function of faster-whisper (see `transcribe.py`) as `clip_timestamps`.

So to transcribe it would then be (separate functios, for now:)
1. Get speech segments and speaker diarization via pyannote
2. Pass the segments to faster-whisper's transcribe function as `clip_timestamps`
3. Collect the results and format them as srt.
4. Alignment via ctc segmentation (see `alignment.py` for reference implementation).
5. Format the final results as srt with speaker labels. 


Make sure that logs etc are saved, such as avg_logprob, speech timestamps, speaker alignment, alignment confidence etc is all saved in /logs/timestamps/* in separate files, such as avg logprob per segment, speaker diarization per segment, ctc alignment confidence per segment etc.

The SRT output should be saved in /outputs/srt/<filename>.srt or to a path specified by the user.

# Folder structure
- /logs/timestamps/: All logs such as avg logprob, speaker diarization, c
tc alignment confidence etc.
- /outputs/srt/: Final srt outputs.

The rest should follow the uv standard; so no src folder or so; dependencies are added via "uv add <package>" after activating the enviroment `./.venv/bin/activate`.