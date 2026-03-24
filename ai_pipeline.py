import os
import json
import subprocess
import shutil
import sys
import types

# ------------------------------------------------------------------
# Monkey-patch missing submodule in PyTorch 2.0.1 to satisfy pytorch-lightning
# ------------------------------------------------------------------
if 'torch.utils.flop_counter' not in sys.modules:
    mod = types.ModuleType('torch.utils.flop_counter')
    mod.FlopCounterMode = object
    sys.modules['torch.utils.flop_counter'] = mod

# Mock torch.compiler for PyTorch 2.0.1 compatibility with decorators
import torch
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(func):
            return func
    torch.compiler = DummyCompiler()
    sys.modules['torch.compiler'] = torch.compiler

# Mock missing function in torch.optim.swa_utils for pytorch-lightning
import torch.optim.swa_utils
if not hasattr(torch.optim.swa_utils, "get_ema_avg_fn"):
    def get_ema_avg_fn(decay=0.999):
        return lambda averaged_param, model_param, num_averaged: None
    torch.optim.swa_utils.get_ema_avg_fn = get_ema_avg_fn


def _find_bin(name):
    """Find binary — prefer bundled copy in bin/ffmpeg/."""
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "ffmpeg", name)
    if os.path.isfile(bundled):
        return bundled
    return name.replace(".exe", "")  # fall back to PATH


FFMPEG = _find_bin("ffmpeg.exe")
FFPROBE = _find_bin("ffprobe.exe")


class MediaProcessor:
    """
    100% offline media processing pipeline.
    Supports both video and audio-only inputs.
    Uses GPU (CUDA) when available for AI inference.
    All AI dependencies are lazy-loaded so the app starts instantly.
    """

    TOTAL_STEPS = 7

    def __init__(self, device=None, progress_callback=None):
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self._log = progress_callback or (lambda msg, pct=None: print(f"[Pipeline] {msg}"))
        self._step = 0

        gpu_info = ""
        if self.device == "cuda":
            gpu_info = f" ({torch.cuda.get_device_name(0)})"
        self._log(f"Device: {self.device.upper()}{gpu_info}")

    def _progress(self, msg):
        self._step += 1
        pct = min(int((self._step / self.TOTAL_STEPS) * 100), 99)
        self._log(msg, pct)

    def _lazy_load_deepfilter(self):
        if not hasattr(self, "df_model"):
            from df.enhance import init_df
            import df.utils
            # Monkey-patch to fix crash on systems without Git installed
            df.utils.get_git_root = lambda: None
            df.utils.get_commit_hash = lambda: None
            df.utils.get_branch_name = lambda: None

            self._log("Loading DeepFilterNet model...")
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DeepFilterNet3")
            self.df_model, self.df_state, _ = init_df(model_base_dir=base_dir)

    def _lazy_load_whisper(self):
        if not hasattr(self, "whisper_model"):
            import sys
            import types
            # Mock missing submodule in PyTorch 2.0.1 to satisfy pytorch-lightning
            if 'torch.utils.flop_counter' not in sys.modules:
                mod = types.ModuleType('torch.utils.flop_counter')
                mod.FlopCounterMode = object
                sys.modules['torch.utils.flop_counter'] = mod

            import whisperx
            self._log(f"Loading WhisperX on {self.device.upper()}...")
            self.whisper_model = whisperx.load_model(
                "base", self.device, compute_type=self.compute_type
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def has_video_stream(filepath):
        """Returns True if file contains a video stream."""
        try:
            import ffmpeg as ff
            probe = ff.probe(filepath, cmd=FFPROBE)
            return any(s["codec_type"] == "video" for s in probe.get("streams", []))
        except Exception:
            return False

    @staticmethod
    def get_duration(filepath):
        import ffmpeg as ff
        probe = ff.probe(filepath, cmd=FFPROBE)
        return float(probe["format"]["duration"])

    # ------------------------------------------------------------------
    # Step: Extract audio
    # ------------------------------------------------------------------
    def extract_audio(self, input_path, output_wav):
        self._progress("Extracting audio...")
        cmd = [FFMPEG, "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_wav]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{r.stderr[-400:]}")

    # ------------------------------------------------------------------
    # Step: Clean audio (DeepFilterNet — GPU accelerated)
    # ------------------------------------------------------------------
    def clean_audio(self, input_wav, output_wav):
        from df.enhance import enhance, load_audio, save_audio
        self._lazy_load_deepfilter()
        self._progress("Cleaning audio (DeepFilterNet)...")
        audio, _ = load_audio(input_wav, sr=self.df_state.sr())
        enhanced = enhance(self.df_model, self.df_state, audio)
        save_audio(output_wav, enhanced, self.df_state.sr())

    # ------------------------------------------------------------------
    # Step: Transcribe & Align (WhisperX — GPU accelerated)
    # ------------------------------------------------------------------
    def transcribe_and_align(self, audio_path):
        import whisperx
        self._lazy_load_whisper()
        self._progress("Transcribing with WhisperX...")
        audio = whisperx.load_audio(audio_path)
        result = self.whisper_model.transcribe(
            audio, batch_size=16,
            initial_prompt="Um, uh, well, like, you know, this is a transcript with filler words."
        )
        self._progress("Running Forced Alignment...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
        aligned = whisperx.align(result["segments"], model_a, metadata, audio, self.device)
        return aligned["segments"]

    # ------------------------------------------------------------------
    # Filler detection
    # ------------------------------------------------------------------
    def detect_fillers(self, word_map, filler_list=None):
        if filler_list is None:
            filler_list = ["um", "uh", "er", "ah", "like"]
        cuts = []
        for seg in word_map:
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue
                if w["word"].lower().strip(".,!? ") in filler_list:
                    cuts.append({"start": w["start"], "end": w["end"], "word": w["word"]})
        self._log(f"Found {len(cuts)} filler(s).")
        return cuts

    # ------------------------------------------------------------------
    # Segment generation
    # ------------------------------------------------------------------
    def generate_segments(self, cuts, duration):
        keep = []
        t = 0.0
        for c in cuts:
            if c["start"] > t:
                keep.append((t, c["start"]))
            t = c["end"]
        if t < duration:
            keep.append((t, duration))
        return keep

    # ------------------------------------------------------------------
    # Render trimmed media (synced A/V via filter_complex)
    # ------------------------------------------------------------------
    def render_trimmed(self, input_path, segments, output_path, is_video=True):
        self._progress(f"Rendering trimmed {'video' if is_video else 'audio'} ({len(segments)} segments)...")
        if not segments:
            raise RuntimeError("Nothing to render — entire file was filler?")

        parts = []
        for i, (s, e) in enumerate(segments):
            if is_video:
                parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];")
                parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}];")
            else:
                parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}];")

        n = len(segments)
        if is_video:
            vc = "".join(f"[v{i}]" for i in range(n))
            ac = "".join(f"[a{i}]" for i in range(n))
            filt = "".join(parts) + f"{vc}concat=n={n}:v=1:a=0[v_out];{ac}concat=n={n}:v=0:a=1[a_out]"
            cmd = [FFMPEG, "-y", "-i", input_path, "-filter_complex", filt,
                   "-map", "[v_out]", "-map", "[a_out]", output_path]
        else:
            ac = "".join(f"[a{i}]" for i in range(n))
            filt = "".join(parts) + f"{ac}concat=n={n}:v=0:a=1[a_out]"
            cmd = [FFMPEG, "-y", "-i", input_path, "-filter_complex", filt,
                   "-map", "[a_out]", output_path]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed:\n{r.stderr[-400:]}")

    # ------------------------------------------------------------------
    # SRT generation (sentence-based, punctuation-aware)
    # ------------------------------------------------------------------
    def generate_srt(self, word_map, cuts=None, srt_path="output.srt"):
        self._progress("Generating sentence-based subtitles...")
        cut_tuples = [(c["start"], c["end"]) for c in (cuts or [])]

        def shift(ts):
            rm = 0.0
            for cs, ce in cut_tuples:
                if ts >= ce:  rm += ce - cs
                elif ts > cs: rm += ts - cs
            return max(0.0, ts - rm)

        words = []
        for seg in word_map:
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue
                if cuts and any(abs(w["start"] - c["start"]) < 0.01 and abs(w["end"] - c["end"]) < 0.01 for c in cuts):
                    continue
                words.append(w)

        sentences, buf = [], []
        for w in words:
            buf.append(w)
            if w["word"].strip()[-1:] in ".?!;:":
                sentences.append(buf)
                buf = []
        if buf:
            sentences.append(buf)

        idx = 1
        with open(srt_path, "w", encoding="utf-8") as f:
            for sw in sentences:
                if not sw: continue
                t1, t2 = shift(sw[0]["start"]), shift(sw[-1]["end"])
                txt = " ".join(w["word"] for w in sw).strip()
                if txt:
                    f.write(f"{idx}\n{_fmt_ts(t1)} --> {_fmt_ts(t2)}\n{txt}\n\n")
                    idx += 1
        self._log(f"SRT: {idx - 1} subtitle entries.")

    # ------------------------------------------------------------------
    # Burn captions (Montserrat font)
    # ------------------------------------------------------------------
    def burn_captions(self, input_video, srt_path, output_video):
        self._progress("Burning captions (Montserrat)...")
        safe_srt = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
        style = "FontName=Montserrat,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=30"
        cmd = [FFMPEG, "-y", "-i", input_video,
               "-vf", f"subtitles='{safe_srt}':force_style='{style}'",
               output_video]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg captioning failed:\n{r.stderr[-400:]}")

    # ------------------------------------------------------------------
    # Full pipeline (toggle-aware, audio/video aware)
    # ------------------------------------------------------------------
    def run_full_pipeline(self, input_path, output_dir,
                          enhance_voice=True, trim_fillers=True,
                          add_captions=False, export_srt=False):
        is_video = self.has_video_stream(input_path)
        ext = os.path.splitext(input_path)[1] or (".mp4" if is_video else ".wav")

        os.makedirs(output_dir, exist_ok=True)
        temp = os.path.join(output_dir, "temp")
        os.makedirs(temp, exist_ok=True)

        steps = 1
        if enhance_voice: steps += 1
        if trim_fillers or add_captions or export_srt: steps += 2
        if trim_fillers: steps += 1
        if add_captions and is_video: steps += 2
        elif export_srt: steps += 1
        self.TOTAL_STEPS = max(steps, 1)
        self._step = 0

        raw_wav = os.path.join(temp, "audio.wav")
        clean_wav = os.path.join(temp, "clean.wav")

        needs_audio = enhance_voice or trim_fillers or add_captions or export_srt
        if needs_audio:
            self.extract_audio(input_path, raw_wav)
        whisper_input = raw_wav

        if enhance_voice:
            self.clean_audio(raw_wav, clean_wav)
            whisper_input = clean_wav

        word_map = None
        cuts = []
        if trim_fillers or add_captions or export_srt:
            word_map = self.transcribe_and_align(whisper_input)

        current_media = input_path
        if trim_fillers and word_map:
            dur = self.get_duration(input_path)
            cuts = self.detect_fillers(word_map)
            segs = self.generate_segments(cuts, dur)
            trimmed = os.path.join(temp, f"trimmed{ext}")
            self.render_trimmed(input_path, segs, trimmed, is_video=is_video)
            current_media = trimmed

        srt_out = None
        if (add_captions or export_srt) and word_map:
            srt_path = os.path.join(temp, "output.srt")
            self.generate_srt(word_map, cuts if trim_fillers else None, srt_path)
            srt_out = srt_path

        final_media = os.path.join(output_dir, f"final{ext}")
        if add_captions and is_video and srt_out:
            self.burn_captions(current_media, srt_out, final_media)
        elif current_media != input_path:
            shutil.copy2(current_media, final_media)
        else:
            shutil.copy2(input_path, final_media)

        final_srt = None
        if export_srt and srt_out:
            final_srt = os.path.join(output_dir, "captions.srt")
            shutil.copy2(srt_out, final_srt)

        self._log("Pipeline complete!", 100)
        return {"media": final_media, "srt": final_srt}

    # ------------------------------------------------------------------
    # Caption-only export
    # ------------------------------------------------------------------
    def export_captions_only(self, input_path, output_srt):
        self.TOTAL_STEPS = 3
        self._step = 0
        tmp_wav = output_srt + ".tmp.wav"
        try:
            self.extract_audio(input_path, tmp_wav)
            word_map = self.transcribe_and_align(tmp_wav)
            self.generate_srt(word_map, cuts=None, srt_path=output_srt)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        self._log("Caption export complete!", 100)
        return output_srt


def _fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"