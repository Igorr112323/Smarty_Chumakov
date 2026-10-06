# -*- mode: python ; coding: utf-8 -*-
"""Сборка SpanVerify в один исполняемый файл (PyInstaller).

    pyinstaller spanverify.spec --noconfirm

Результат: ``dist/spanverify(.exe)`` — один файл, консольное приложение.

Что важно в этой сборке (Приложение В мастер-промта):

* ``console=True`` — виден лог запуска; ``strip`` включён только вне Windows:
  стрип PE-файлов на Windows приводил к отказу загрузки Python DLL
  (``Invalid access to memory location``) на сборочном раннере, поэтому там
  стрип отключён — размером жертвуем ради работоспособности;
* ``name="spanverify"`` — имя файла совпадает с именем в релизе;
* ``datas`` включает ``config/`` и ``data/`` — обученные веса, калибратор и
  демонстрационный корпус едут внутри файла, поэтому .exe работает сам по себе
  (файл ``config/weights.json`` рядом с .exe, если он есть, имеет приоритет);
* ``hiddenimports`` — модули, которые PyInstaller не видит за ленивыми
  импортами (режим ``hf`` подключается только при наличии torch);
* ``excludes`` — тяжёлые пакеты, которые .exe не использует. Если torch
  установлен в среде сборки, без исключения бинарник вырос бы до гигабайт.

Сборка выполняется только на Windows в GitHub Actions, поэтому .exe всегда
проверяется смоук-тестом до публикации релиза.
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
    "spanverify.cli",
    "spanverify.server",
    "spanverify.webui",
    "spanverify.engine",
    "spanverify.features",
    "spanverify.train",
    "spanverify.dataset",
    "spanverify.detector",
    "spanverify.backends.surrogate",
    "spanverify.backends.hf",
]

excludes = [
    "torch", "transformers", "faiss", "streamlit", "tkinter", "unittest",
    "pydoc", "pytest", "setuptools", "pip", "matplotlib", "pandas", "scipy",
    "PIL", "IPython", "notebook", "numpy.testing",
]

a = Analysis(
    [str(ROOT / "scripts" / "exe_launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    strip=sys.platform != "win32",
    upx=False,
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
