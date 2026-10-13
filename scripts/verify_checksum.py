#!/usr/bin/env python3
"""Сверка контрольной суммы поставки перед установкой (ТЗ п. 4.2).

Программа распространяется копированием: файл переносится на машину заказчика
вместе с файлом сумм ``SHA256SUMS.txt``, который формирует автоматическая сборка
(``.github/workflows/build-and-release-exe.yml``). Перед запуском суммы
сверяются — так подтверждается, что файл не подменён и перенесён без потерь.

Примеры::

    python scripts/verify_checksum.py spanverify.exe            # сверить с SHA256SUMS.txt
    python scripts/verify_checksum.py spanverify.pyz --sums SHA256SUMS.txt
    python scripts/verify_checksum.py --self-test               # проверить сам инструмент
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

DEFAULT_SUMS = "SHA256SUMS.txt"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    """Прочитать формат ``<сумма>  <имя файла>`` (как печатает sha256sum)."""
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].lower(), parts[1].strip().lstrip("*")
        sums[Path(name).name] = digest
    return sums


def verify(target: Path, sums_file: Path) -> int:
    if not target.is_file():
        print(f"файл не найден: {target}")
        return 1
    if not sums_file.is_file():
        print(f"файл сумм не найден: {sums_file}")
        return 1
    sums = parse_sums(sums_file)
    expected = sums.get(target.name)
    if expected is None:
        print(f"в {sums_file} нет строки для {target.name}")
        return 1
    actual = sha256_of(target)
    if actual == expected:
        print(f"сумма совпала: {target.name}\n  ожидалось {expected}\n  получено  {actual}")
        return 0
    print(
        f"СУММА НЕ СОВПАЛА: {target.name}\n  ожидалось {expected}\n  получено  {actual}\n"
        "файл повреждён или подменён — запускать нельзя"
    )
    return 1


def self_test() -> int:
    """Проверка самого инструмента: исправный файл принимается, подменённый — нет."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "spanverify.pyz"
        good.write_bytes(b"spanverify" * 100)
        digest = sha256_of(good)
        (root / DEFAULT_SUMS).write_text(f"{digest}  spanverify.pyz\n", encoding="utf-8")
        if verify(good, root / DEFAULT_SUMS) != 0:
            print("самопроверка не пройдена: исправный файл отвергнут")
            return 1
        bad = root / "spanverify.pyz"
        bad.write_bytes(b"spanverify" * 99 + b"tampered")
        if verify(bad, root / DEFAULT_SUMS) == 0:
            print("самопроверка не пройдена: подменённый файл принят")
            return 1
    print("самопроверка пройдена: исправный файл принят, подменённый отвергнут")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Сверка контрольной суммы поставки")
    parser.add_argument("target", nargs="?", help="файл поставки (spanverify.exe, spanverify.pyz, …)")
    parser.add_argument("--sums", default=DEFAULT_SUMS, help="файл сумм (по умолчанию SHA256SUMS.txt)")
    parser.add_argument("--self-test", action="store_true", help="проверить сам инструмент и выйти")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.target:
        parser.error("укажите файл поставки или --self-test")
    return verify(Path(args.target), Path(args.sums))


if __name__ == "__main__":
    sys.exit(main())
