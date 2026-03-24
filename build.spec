# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Offline AI Video Editor
# Build command: pyinstaller build.spec

import os

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(project_dir, 'app.py')],
    pathex=[project_dir],
    binaries=[
        # Bundle FFmpeg executables
        (os.path.join(project_dir, 'bin', 'ffmpeg', 'ffmpeg.exe'), os.path.join('bin', 'ffmpeg')),
        (os.path.join(project_dir, 'bin', 'ffmpeg', 'ffprobe.exe'), os.path.join('bin', 'ffmpeg')),
    ],
    datas=[],
    hiddenimports=[
        'whisperx',
        'torch',
        'torchaudio',
        'torchaudio.lib',
        'torchaudio.lib.libtorchaudio',
        'torchaudio.backend',
        'torchaudio.backend.soundfile_backend',
        'torchaudio.pipelines',
        'torchaudio.models',
        'torchaudio.utils',
        'soundfile',
        'df',
        'df.enhance',
        'df.model',
        'PySide6',
        'ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorboard', 'torch.utils.tensorboard'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OfflineAIVideoEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True during debugging to capture tracebacks
    icon=None,      # Add an .ico file here if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OfflineAIVideoEditor',
)
