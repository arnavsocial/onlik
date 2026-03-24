# Project: Offline AI Filler Word Remover & Auto-Captioner

## Goal
Build a desktop-based Python application that processes large video/audio files (GBs) entirely offline to:
1. Automatically remove filler words ("um", "uh", "like").
2. Remove background noise/enhance audio quality.
3. Generate and burn-in accurate captions.

## Tech Stack (Offline-First)
- Language: Python 3.10+
- Media Handling: FFmpeg (via ffmpeg-python)
- Audio Enhancement: DeepFilterNet / Fish Audio S2 (Vocal Separation)
- Transcription: WhisperX (Forced Alignment for millisecond timestamps)
- UI: Streamlit or PyQt6

## Processing Pipeline
1. Audio Extraction: Extract .wav from source video using FFmpeg.
2. AI Cleaning: Apply DeepFilterNet to the .wav to isolate speech.
3. STT Processing: Use WhisperX on cleaned audio to get a JSON of word-level timestamps.
4. Edit Logic: Generate an FFmpeg filter string that excludes timestamps identified as filler words or silence.
5. Captioning: Convert the adjusted WhisperX output into .srt format.
6. Rendering: Use FFmpeg concat or filter_complex to render the final video with captions burned in.

## Constraints
- Zero Cloud: No external APIs (Deepgram, OpenAI API, etc.).
- Large Files: Must handle 1GB+ files without memory crashes by processing audio/video separately.
- Precision: Must use "Forced Alignment" (WhisperX) to avoid choppy cuts.
# Blueprint: Offline AI Video Editor (2026)

## 1. Project Overview
A professional, 100% offline desktop application built in Python to process large (1GB+) video/audio files. It uses local AI models and FFmpeg to automate three specific features:
- Feature A: Automatic Filler Word Removal (ums, uhs, er).
- Feature B: Audio Cleaning/Background Noise Suppression.
- Feature C: One-click Auto-Captions (burned-in or SRT).

## 2. Technical Stack
- Framework: PySide6 (Qt) for a responsive, multi-threaded GUI.
- Audio Cleaning: DeepFilterNet (Local AI noise suppression).
- Transcription/Alignment: WhisperX (provides millisecond-level word timestamps).
- Media Engine: FFmpeg (Handles all cutting and stitching to avoid memory issues).

## 3. The Pipeline Logic
1. Extraction: Python uses FFmpeg to extract audio from video.
2. Cleaning: AI filters the audio track to remove noise.
3. Mapping: WhisperX generates a JSON "map" of every word and silence.
4. Pruning: A script identifies "um" timestamps and generates an FFmpeg "keep list."
5. Assembly: FFmpeg slices the original large video and merges the "good" parts.
6. Captioning: The JSON map is converted to SRT and overlaid on the final video.

## 4. Performance Constraints
- Multi-threading: AI/FFmpeg must run on a QThread to prevent UI freezing.
- Memory Management: The video itself is never loaded into RAM; only FFmpeg manipulates the file stream.
- Local Assets: App must bundle ffmpeg.exe and model weights (.pt or .onnx files) for total offline usage.

## 5. Development Status
- Conceptual pipeline finalized.
- GUI framework selected (PySide6).
- Offline cleaning tool selected (DeepFilterNet).
- Transcription tool selected (WhisperX).