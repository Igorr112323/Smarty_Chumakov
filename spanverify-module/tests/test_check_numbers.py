"""Тесты для scripts/check_numbers.py — сверка чисел в документах."""
import sys
from pathlib import Path

# Добавляем scripts в путь
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

import check_numbers  # noqa: E402


def test_coverage_pattern_decimal():
    """Проверяет, что регулярное выражение ловит десятичные проценты покрытия."""
    pattern = check_numbers.TOLERANT_PATTERNS[0][0]

    # Тестовые случаи
    test_cases = [
        ("покрытие 58.89 %", "58.89"),
        ("покрытие 87.82 %", "87.82"),
        ("покрытие 88 %", "88"),
        ("покрытием 58.89%", "58.89"),
        ("покрытие 100 %", "100"),
        ("покрытие 0 %", "0"),
    ]

    for text, expected in test_cases:
        match = pattern.search(text)
        assert match is not None, f"Pattern should match '{text}'"
        # Нормализуем запятую в точку для сравнения
        captured = match.group(1).replace(",", ".")
        assert captured == expected, f"Expected {expected}, got {captured} in '{text}'"


def test_coverage_pattern_no_false_positives():
    """Проверяет, что регулярное выражение не ловит нерелевантные числа."""
    pattern = check_numbers.TOLERANT_PATTERNS[0][0]

    # Не должно ловить
    negative_cases = [
        "покрытие не измерено",
        "тестов 629",
        "F1 0.85",
        "689 токенов",
    ]

    for text in negative_cases:
        match = pattern.search(text)
        assert match is None, f"Pattern should NOT match '{text}'"


def test_trap_wrong_coverage(tmp_path):
    """Тест-ловушка: неверное покрытие должно отличаться от правильного."""
    # Создаём временный документ с НЕВЕРНЫМ покрытием
    doc = tmp_path / "probe.md"
    doc.write_text("Покрытие кода: 58.89 %\n", encoding="utf-8")

    # Извлекаем покрытие из документа
    pattern = check_numbers.TOLERANT_PATTERNS[0][0]
    doc_text = doc.read_text(encoding="utf-8")
    match = pattern.search(doc_text)

    assert match is not None, "Pattern should match coverage in doc"
    doc_coverage = float(match.group(1).replace(",", "."))

    # Проверяем, что это НЕ равно правильному значению
    correct_coverage = 87.17
    assert doc_coverage != correct_coverage, (
        f"Trap test: doc coverage ({doc_coverage}) should differ from correct ({correct_coverage})"
    )


def test_correct_coverage_value(tmp_path):
    """Тест-ловушка: верное покрытие должно совпадать с правильным."""
    # Создаём временный документ с ВЕРНЫМ покрытием
    doc = tmp_path / "probe.md"
    doc.write_text("Покрытие кода: 87.17 %\n", encoding="utf-8")

    # Извлекаем покрытие из документа
    pattern = check_numbers.TOLERANT_PATTERNS[0][0]
    doc_text = doc.read_text(encoding="utf-8")
    match = pattern.search(doc_text)

    assert match is not None, "Pattern should match coverage in doc"
    doc_coverage = float(match.group(1).replace(",", "."))

    # Проверяем, что это равно правильному значению
    correct_coverage = 87.17
    assert doc_coverage == correct_coverage, (
        f"Doc coverage ({doc_coverage}) should match correct ({correct_coverage})"
    )
