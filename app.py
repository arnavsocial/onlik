"""
Offline AI Video Editor — Desktop GUI
Apple / Google Material 3 inspired design snippet.
Fully custom sliding toggle switches based on QAbstractButton.
"""
import sys
import os
import types

# ------------------------------------------------------------------
# Monkey-patch PyTorch 2.0.1 compatibility for downstream libraries
# ------------------------------------------------------------------
if 'torch.utils.flop_counter' not in sys.modules:
    mod = types.ModuleType('torch.utils.flop_counter')
    mod.FlopCounterMode = object
    sys.modules['torch.utils.flop_counter'] = mod

import torch
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(func): return func
    torch.compiler = DummyCompiler()
    sys.modules['torch.compiler'] = torch.compiler

import torch.optim.swa_utils
if not hasattr(torch.optim.swa_utils, "get_ema_avg_fn"):
    def get_ema_avg_fn(decay=0.999):
        return lambda averaged_param, model_param, num_averaged: None
    torch.optim.swa_utils.get_ema_avg_fn = get_ema_avg_fn

# ------------------------------------------------------------------
# Mock missing arguments in faster-whisper TranscriptionOptions
# ------------------------------------------------------------------
import faster_whisper.transcribe
original_init = faster_whisper.transcribe.TranscriptionOptions.__init__

def patched_init(self, **kwargs):
    defaults = {
        'beam_size': 5, 'best_of': 5, 'patience': 1.0, 
        'length_penalty': 1.0, 'repetition_penalty': 1.0, 
        'no_repeat_ngram_size': 0, 'log_prob_threshold': -1.0, 
        'no_speech_threshold': 0.6, 'compression_ratio_threshold': 2.4, 
        'condition_on_previous_text': True, 'prompt_reset_on_temperature': 0.5, 
        'temperatures': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        'initial_prompt': None, 'prefix': None, 
        'suppress_blank': True, 'suppress_tokens': [-1], 
        'without_timestamps': False, 'max_initial_timestamp': 1.0, 
        'word_timestamps': False, 'prepend_punctuations': '"\'“¿([{-', 
        'append_punctuations': '"\'”?.!,;)]}-', 
        'multilingual': False, 'max_new_tokens': 256, 
        'clip_timestamps': '0', 'hallucination_silence_threshold': None, 
        'hotwords': ''
    }
    defaults.update(kwargs)
    return original_init(self, **defaults)

faster_whisper.transcribe.TranscriptionOptions.__init__ = patched_init
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QFrame,
    QGraphicsDropShadowEffect, QSizePolicy, QAbstractButton, QGridLayout
)
from PySide6.QtCore import (
    QThread, Signal, Qt, QSize, QPropertyAnimation, QEasingCurve,
    Property, QPoint, QRect, QRectF
)
from PySide6.QtGui import QFont, QColor, QPainter, QPainterPath

from ai_pipeline import MediaProcessor

# ── Design Tokens ──
COLORS = {
    "bg":         "#f5f5f7",       # Light mode macOS background
    "bg_dark":    "#0f0b1a",       # Dark mode background
    "surface":    "#1a1428",       # Dark elevated card
    "border":     "#2e2545",       # Card border
    "accent":     "#9b5de5",       # Brand purple
    "accent_hi":  "#b882fe",       # Hover purple
    "accent_dim": "#7e40c4",       # Pressed purple
    "text":       "#ffffff",       # Main text
    "text2":      "#b5aac5",       # Secondary text
    "success":    "#34c759",       # Apple success green
    "error":      "#ff3b30",       # Apple red
    "toggle_off": "#39314b",       # Toggle track (off)
    "thumb":      "#ffffff"        # Toggle thumb
}
FONT_FAMILY = "SF Pro Display, Inter, Roboto, sans-serif"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"}

# ═══════════════════════════════════════════════════════════════════════
# Custom UI Components (Apple/Google Style)
# ═══════════════════════════════════════════════════════════════════════
class ToggleSwitch(QAbstractButton):
    """
    A custom sliding toggle switch mimicking iOS / Material 3 toggles.
    Animates the thumb sliding and color fading.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(50, 28)
        self._thumb_pos = 2
        self._anim = QPropertyAnimation(self, b"thumbPos", self)
        self._anim.setEasingCurve(QEasingCurve.InOutCirc)
        self._anim.setDuration(150)
        self.toggled.connect(self._start_animation)

    def _get_thumb_pos(self):
        return self._thumb_pos

    def _set_thumb_pos(self, pos):
        self._thumb_pos = pos
        self.update()

    thumbPos = Property(float, _get_thumb_pos, _set_thumb_pos)

    def _start_animation(self, checked):
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(24 if checked else 2)
        self._anim.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        track_rect = self.rect()
        thumb_rect = QRectF(self._thumb_pos, 2, 24, 24)

        # Draw Track
        track_color = QColor(COLORS["accent"]) if self.isChecked() else QColor(COLORS["toggle_off"])
        if not self.isEnabled():
            track_color = QColor("#222222")

        p.setPen(Qt.NoPen)
        p.setBrush(track_color)
        p.drawRoundedRect(track_rect, 14, 14)

        # Draw Thumb
        thumb_color = QColor(COLORS["thumb"])
        if not self.isEnabled():
            thumb_color = QColor("#666666")

        p.setBrush(thumb_color)
        p.drawEllipse(thumb_rect)
        p.end()

class Card(QFrame):
    """Elevated surface with rounded corners and drop shadow."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            Card {{
                background-color: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 16px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

class SettingRow(QWidget):
    """Combines a title, description, and the native ToggleSwitch."""
    def __init__(self, title, desc, checked=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text']};")
        
        self.lbl_desc = QLabel(desc)
        self.lbl_desc.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        
        text_layout.addWidget(self.lbl_title)
        text_layout.addWidget(self.lbl_desc)
        
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(checked)
        
        layout.addLayout(text_layout, stretch=1)
        layout.addWidget(self.toggle, alignment=Qt.AlignRight | Qt.AlignVCenter)


# ═══════════════════════════════════════════════════════════════════════
# Workers
# ═══════════════════════════════════════════════════════════════════════
class PipelineWorker(QThread):
    progress_text = Signal(str)
    progress_pct = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path, out_dir, enhance, trim, captions, export_srt):
        super().__init__()
        self.path, self.out_dir = path, out_dir
        self.enhance, self.trim = enhance, trim
        self.captions, self.export_srt = captions, export_srt

    def _cb(self, msg, pct=None):
        self.progress_text.emit(msg)
        if pct is not None:
            self.progress_pct.emit(pct)

    def run(self):
        try:
            p = MediaProcessor(progress_callback=self._cb)
            r = p.run_full_pipeline(
                self.path, self.out_dir,
                enhance_voice=self.enhance, trim_fillers=self.trim,
                add_captions=self.captions, export_srt=self.export_srt,
            )
            self.finished.emit(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class CaptionWorker(QThread):
    progress_text = Signal(str)
    progress_pct = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, path, srt_path):
        super().__init__()
        self.path, self.srt_path = path, srt_path

    def _cb(self, msg, pct=None):
        self.progress_text.emit(msg)
        if pct is not None:
            self.progress_pct.emit(pct)

    def run(self):
        try:
            p = MediaProcessor(progress_callback=self._cb)
            p.export_captions_only(self.path, self.srt_path)
            self.finished.emit(self.srt_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Video Editor")
        self.setMinimumSize(600, 680)
        self.input_path = None
        self.is_video = False
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(f"background-color: {COLORS['bg_dark']}; font-family: {FONT_FAMILY};")

        root = QVBoxLayout(central)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(24)

        # ── Header ──
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)
        
        title = QLabel("AI Video Editor")
        title.setStyleSheet(f"font-size: 32px; font-weight: 800; color: {COLORS['text']}; letter-spacing: -0.5px;")
        
        subtitle = QLabel("DeepFilterNet · WhisperX · FFmpeg")
        subtitle.setStyleSheet(f"font-size: 13px; font-weight: 500; color: {COLORS['accent_hi']};")
        
        desc = QLabel("100% Offline processing. Render lossless quality directly from your machine.")
        desc.setStyleSheet(f"font-size: 14px; color: {COLORS['text2']};")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(desc)
        root.addLayout(header_layout)

        # ── File Selection Card ──
        self.file_card = Card()
        fc_layout = QVBoxLayout(self.file_card)
        fc_layout.setContentsMargins(24, 24, 24, 24)

        file_row = QHBoxLayout()
        self.lbl_file = QLabel("Drop a video or audio file here")
        self.lbl_file.setStyleSheet(f"font-size: 15px; color: {COLORS['text2']}; font-weight: 500;")
        
        btn_browse = QPushButton("Browse")
        btn_browse.setCursor(Qt.PointingHandCursor)
        btn_browse.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['toggle_off']};
                color: {COLORS['text']};
                border-radius: 16px;
                padding: 8px 20px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: {COLORS['border']}; }}
        """)
        btn_browse.clicked.connect(self._browse)
        
        file_row.addWidget(self.lbl_file, stretch=1)
        file_row.addWidget(btn_browse)
        fc_layout.addLayout(file_row)
        root.addWidget(self.file_card)

        # ── Settings Card ──
        opts_card = Card()
        opts_layout = QVBoxLayout(opts_card)
        opts_layout.setContentsMargins(24, 16, 24, 16)
        opts_layout.setSpacing(12)

        self.row_enhance = SettingRow(
            "Studio Voice Enhance", 
            "Removes background noise and echo using DeepFilterNet."
        )
        self.row_trim = SettingRow(
            "Auto Filler Trimming", 
            "Seamlessly cuts out 'um', 'uh', 'er' and 'ah'."
        )
        self.row_captions = SettingRow(
            "Burn Captions", 
            "Generates Montserrat sentence-based subtitles on the video.",
            checked=False
        )

        opts_layout.addWidget(self.row_enhance)
        
        # Add subtle dividers inside the card
        def make_divider():
            line = QFrame()
            line.setFixedHeight(1)
            line.setStyleSheet(f"background-color: {COLORS['border']};")
            return line

        opts_layout.addWidget(make_divider())
        opts_layout.addWidget(self.row_trim)
        opts_layout.addWidget(make_divider())
        opts_layout.addWidget(self.row_captions)
        
        root.addWidget(opts_card)

        # ── Primary Actions ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_process = QPushButton("Process Media")
        self.btn_process.setCursor(Qt.PointingHandCursor)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['text']};
                border-radius: 12px;
                padding: 16px;
                font-weight: 700;
                font-size: 15px;
            }}
            QPushButton:hover {{ background-color: {COLORS['accent_hi']}; }}
            QPushButton:disabled {{ background-color: {COLORS['toggle_off']}; color: #666; }}
            QPushButton:pressed {{ background-color: {COLORS['accent_dim']}; }}
        """)
        self.btn_process.clicked.connect(self._process)
        
        self.btn_export = QPushButton("Export .srt")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 2px solid {COLORS['accent']};
                color: {COLORS['accent']};
                border-radius: 12px;
                padding: 14px;
                font-weight: 700;
                font-size: 14px;
            }}
            QPushButton:hover {{ background-color: {COLORS['toggle_off']}; }}
            QPushButton:disabled {{ border-color: {COLORS['toggle_off']}; color: #666; }}
            QPushButton:pressed {{ background-color: {COLORS['border']}; }}
        """)
        self.btn_export.clicked.connect(self._export)

        btn_layout.addWidget(self.btn_export, stretch=1)
        btn_layout.addWidget(self.btn_process, stretch=2)
        root.addLayout(btn_layout)

        # ── Status & Progress ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6) # native slim look
        self.progress.setVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['toggle_off']};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        root.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(f"font-size: 13px; font-weight: 500; color: {COLORS['text2']};")
        root.addWidget(self.lbl_status)

        root.addStretch()

    # ── Slots ──
    def _browse(self):
        f = "Media Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wav *.mp3 *.flac *.aac *.ogg *.m4a)"
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", f)
        if not path:
            return
        self.input_path = path
        ext = os.path.splitext(path)[1].lower()
        self.is_video = ext in VIDEO_EXTS

        name = os.path.basename(path)
        icon = "🎬" if self.is_video else "🎵"
        self.lbl_file.setText(f"{icon}  {name}")
        self.lbl_file.setStyleSheet(f"font-size: 15px; color: {COLORS['text']}; font-weight: 700;")
        
        self.btn_process.setEnabled(True)
        self.btn_export.setEnabled(True)

        self.row_captions.toggle.setEnabled(self.is_video)
        if not self.is_video:
            self.row_captions.toggle.setChecked(False)

        self.lbl_status.setText("Ready to process.")
        self.lbl_status.setStyleSheet(f"color: {COLORS['text2']}; font-size: 13px; font-weight: 500;")

    def _process(self):
        if not self.input_path:
            return
        opts = [self.row_enhance.toggle.isChecked(), self.row_trim.toggle.isChecked(), self.row_captions.toggle.isChecked()]
        if not any(opts):
            self.lbl_status.setText("Enable at least one option.")
            self.lbl_status.setStyleSheet(f"color: {COLORS['error']}; font-size: 13px; font-weight: 500;")
            return
            
        out = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out:
            return

        self._busy(True, "Initializing pipeline...")
        self.worker = PipelineWorker(
            self.input_path, out,
            enhance=self.row_enhance.toggle.isChecked(),
            trim=self.row_trim.toggle.isChecked(),
            captions=self.row_captions.toggle.isChecked(),
            export_srt=self.row_captions.toggle.isChecked(),
        )
        self.worker.progress_text.connect(self._log)
        self.worker.progress_pct.connect(self.progress.setValue)
        self.worker.finished.connect(self._done)
        self.worker.error.connect(self._err)
        self.worker.start()

    def _export(self):
        if not self.input_path:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Caption", "captions.srt", "SRT (*.srt)")
        if not p:
            return
        self._busy(True, "Transcribing natively...")
        self.worker = CaptionWorker(self.input_path, p)
        self.worker.progress_text.connect(self._log)
        self.worker.progress_pct.connect(self.progress.setValue)
        self.worker.finished.connect(lambda sp: self._done({"media": None, "srt": sp}))
        self.worker.error.connect(self._err)
        self.worker.start()

    def _log(self, text):
        self.lbl_status.setText(text)

    def _done(self, result):
        parts = []
        if result.get("media"): parts.append("Video Saved")
        if result.get("srt"):   parts.append("Captions Saved")
        msg = " ✓ " + " & ".join(parts) + "!"
        
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px; font-weight: 600;")
        self.progress.setValue(100)
        self._busy(False)

    def _err(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        self.lbl_status.setStyleSheet(f"color: {COLORS['error']}; font-size: 13px; font-weight: 500;")
        self._busy(False)

    def _busy(self, b, msg=""):
        self.btn_process.setEnabled(not b)
        self.btn_export.setEnabled(not b)
        self.progress.setVisible(b or self.progress.value() == 100)
        if b:
            self.progress.setValue(0)
            self.lbl_status.setStyleSheet(f"color: {COLORS['accent_hi']}; font-size: 13px; font-weight: 500;")
            if msg:
                self.lbl_status.setText(msg)


if __name__ == "__main__":
    import ctypes
    # Enforce exact macOS/iOS text aliasing logic in Windows standard PySide engine
    if sys.platform == 'win32':
        myappid = 'com.antigravity.offline.editor'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        
    app = QApplication(sys.argv)
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
