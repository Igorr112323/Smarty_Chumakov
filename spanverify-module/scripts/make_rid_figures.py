#!/usr/bin/env python3
"""Сгенерировать фигуры для материалов заявки на полезную модель.

Фигуры строятся **из кода**: блоки и связи берутся из реальной структуры
пакета ``spanverify``, поэтому схема не расходится с программой. Результат —
векторные файлы SVG (без растровых вставок, масштабируются без потери).

Запуск:

    python scripts/make_rid_figures.py --out ../docs/РИД_полезная_модель/фигуры

Каждая фигура — отдельный файл ``fig1.svg``, ``fig2.svg`` и так далее.
Перечень фигур и нумерация соответствуют документу
``docs/РИД_полезная_модель/Комплект.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Блоки устройства: (идентификатор, подпись, модуль программы).
# Взято из состава пакета: каждый блок существует в коде.
BLOCKS: tuple[tuple[str, str], ...] = (
    ("1", "блок ввода документа-источника и ответа"),
    ("2", "блок токенизации с сохранением символьных смещений"),
    ("3", "блок нормализации чисел, единиц измерения и дат"),
    ("4", "блок вычисления признаков токенов"),
    ("5", "блок привязки числа к объекту документа"),
    ("6", "блок свёртки признаков в оценку риска"),
    ("7", "блок калибровки и порога решения"),
    ("8", "блок выделения спорных фрагментов и сужения их границ"),
    ("9", "блок оценки доли участия ИИ"),
    ("10", "блок вывода результата"),
)

# Связи между блоками (откуда → куда).
EDGES: tuple[tuple[str, str, str], ...] = (
    ("1", "2", "документ и ответ"),
    ("2", "3", "токены"),
    ("3", "4", "нормализованные токены"),
    ("4", "5", "признаки и числа"),
    ("5", "6", "признанные и спорные числа"),
    ("6", "7", "риск токена"),
    ("7", "8", "порог решения"),
    ("2", "9", "токены ответа"),
    ("8", "10", "фрагменты со смещениями"),
    ("9", "10", "доля участия ИИ"),
    ("7", "10", "вердикт и оценка риска"),
)

MODULES: tuple[tuple[str, str], ...] = (
    ("1", "cli.py, server.py, webui.py"),
    ("2", "core.py"),
    ("3", "normalize.py"),
    ("4", "features.py, backends/hf.py"),
    ("5", "features.py::number_attribution"),
    ("6", "features.py::combine"),
    ("7", "calibration.py, train.py"),
    ("8", "engine.py::_shrink_to_clause"),
    ("9", "participation.py, detector.py"),
    ("10", "engine.py::VerificationResult"),
)

WIDTH = 900
BLOCK_W = 250
BLOCK_H = 46
GAP_Y = 26
MARGIN = 60


def _block_x(index: int, column: int) -> int:
    return MARGIN + column * (BLOCK_W + 240)


def _block_y(index: int) -> int:
    return MARGIN + 40 + index * (BLOCK_H + GAP_Y)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def figure_structure() -> str:
    """Фигура 1: общая структура устройства (блоки и связи)."""
    height = MARGIN + 60 + len(BLOCKS) * (BLOCK_H + GAP_Y)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="34" font-size="17" font-weight="bold">'
        "Фиг. 1. Устройство проверки ответа по документу-источнику</text>",
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#33475b"/></marker></defs>',
    ]
    positions: dict[str, tuple[int, int, int, int]] = {}
    for index, (number, caption) in enumerate(BLOCKS):
        x = MARGIN
        y = _block_y(index)
        positions[number] = (x, y, x + BLOCK_W, y + BLOCK_H)
        lines.append(
            f'<rect x="{x}" y="{y}" width="{BLOCK_W}" height="{BLOCK_H}" rx="6" '
            f'fill="#eef3f8" stroke="#33475b" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text x="{x + 12}" y="{y + 21}" font-size="13" font-weight="bold">{number}. ' f"{_escape(caption)}</text>"
        )
        module = dict(MODULES).get(number, "")
        lines.append(f'<text x="{x + 12}" y="{y + 37}" font-size="10.5" fill="#5b6b7c">{_escape(module)}</text>')

    for source, target, label in EDGES:
        sx1, sy1, sx2, sy2 = positions[source]
        tx1, ty1, _tx2, _ty2 = positions[target]
        start_x = sx2
        start_y = (sy1 + sy2) // 2
        end_x = tx1 - 6
        end_y = (ty1 + ty1 + BLOCK_H) // 2
        mid_x = start_x + 90
        lines.append(
            f'<path d="M {start_x} {start_y} L {mid_x} {start_y} L {mid_x} {end_y} L {end_x} {end_y}" '
            f'fill="none" stroke="#33475b" stroke-width="1.2" marker-end="url(#arrow)"/>'
        )
        if label:
            lines.append(
                f'<text x="{mid_x + 8}" y="{(start_y + end_y) // 2 - 4}" font-size="10" '
                f'fill="#5b6b7c">{_escape(label)}</text>'
            )
    lines.append("</svg>")
    return "\n".join(lines)


def figure_pipeline() -> str:
    """Фигура 2: порядок работы (последовательность этапов)."""
    steps = (
        ("А", "приём документа-источника и ответа"),
        ("Б", "токенизация ответа с сохранением смещений"),
        ("В", "нормализация чисел, единиц измерения, дат"),
        ("Г", "вычисление признаков каждого токена"),
        ("Д", "привязка числа ответа к объекту документа"),
        ("Е", "свёртка признаков в оценку риска токена"),
        ("Ж", "калибровка оценки и порог решения"),
        ("З", "выделение спорных фрагментов и сужение границ до клаузы"),
        ("И", "оценка доли участия ИИ"),
        ("К", "выдача вердикта, фрагментов со смещениями и оценки риска"),
    )
    height = MARGIN + 40 + len(steps) * 52
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="34" font-size="17" font-weight="bold">Фиг. 2. Порядок работы устройства</text>',
    ]
    for index, (letter, caption) in enumerate(steps):
        y = MARGIN + 30 + index * 52
        lines.append(f'<circle cx="86" cy="{y + 17}" r="15" fill="#eef3f8" stroke="#33475b" stroke-width="1.4"/>')
        lines.append(
            f'<text x="86" y="{y + 22}" font-size="13" font-weight="bold" text-anchor="middle">{letter}</text>'
        )
        lines.append(
            f'<rect x="118" y="{y}" width="700" height="34" rx="5" fill="#f7f9fb" '
            f'stroke="#c3ced9" stroke-width="1"/>'
        )
        lines.append(f'<text x="132" y="{y + 22}" font-size="12.5">{_escape(caption)}</text>')
        if index:
            lines.append(
                f'<path d="M 86 {y - 18} L 86 {y - 2}" stroke="#33475b" stroke-width="1.2" '
                f'marker-end="url(#arrow)"/>'
            )
    lines.insert(
        3,
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="4" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#33475b"/></marker></defs>',
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Фигуры для материалов заявки на полезную модель")
    parser.add_argument("--out", type=Path, default=Path("../docs/РИД_полезная_модель/фигуры"))
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig1_structure.svg": figure_structure(),
        "fig2_pipeline.svg": figure_pipeline(),
    }
    for name, content in figures.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        print(f"{path}: {len(content)} байт")
    print(f"Фигур: {len(figures)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
