# -*- mode: python ; coding: utf-8 -*-
"""Сборка SpanVerify в один исполняемый файл (PyInstaller).

    pyinstaller spanverify.spec --noconfirm

Результат: dist/spanverify(.exe). Конфигурация и калибратор кладутся рядом
с исполняемым файлом (папка config/), чтобы порог можно было менять без
пересборки. Если их нет, приложение стартует со значениями по умолчанию.
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "config"), "config"),
    (str(ROOT / "data"), "data"),
]
hiddenimports = [
    "spanverify.backends.surrogate",
    "spanverify.backends.hf",
]

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # torch/transformers подключаются только если установлены и указаны явно;
    # для демо-сборки их вырезаем, чтобы бинарник остался компактным.
    excludes=["torch", "transformers", "streamlit", "pandas", "matplotlib", "numpy"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="spanverify",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

if sys.platform == "win32":
    # Консольное окно полезно: в нём печатается адрес интерфейса и логи.
    exe.console = True
