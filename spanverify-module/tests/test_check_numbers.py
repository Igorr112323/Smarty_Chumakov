"""Тесты для scripts/check_numbers.py — сверка чисел в документах."""
import json
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


def _load_real_metrics():
    """Загружает реальный METRICS.json для тестов."""
    metrics_path = Path(__file__).parent.parent / "reports" / "METRICS.json"
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def test_trap_wrong_coverage(tmp_path):
    """Тест-ловушка: подставной документ с неверным покрытием обязан дать проблему."""
    # Загружаем реальный METRICS.json
    metrics = _load_real_metrics()
    real_coverage = metrics["tests"]["coverage_percent"]

    # Создаём документ с НЕВЕРНЫМ покрытием (сильно отличается от реального)
    wrong_coverage = 50.0  # Заведомо неверное значение
    doc = tmp_path / "probe.md"
    doc.write_text(f"Покрытие кода: {wrong_coverage} %\n", encoding="utf-8")

    # Вызываем реальную функцию check_docs
    # Используем реальный repo_root, чтобы не было ошибок с отсутствующими файлами
    real_repo_root = Path(__file__).parent.parent.parent
    problems = check_numbers.check_docs([doc], metrics, repo_root=real_repo_root)

    # Должна быть проблема с покрытием
    assert len(problems) > 0, f"Expected coverage problem, got no problems"
    # Проверяем, что проблема связана с покрытием (50.0 не должно быть в allowed)
    coverage_problems = [
        p for p in problems
        if "coverage" in p.lower() or "50" in p or str(real_coverage) in p
    ]
    assert len(coverage_problems) > 0, (
        f"Expected coverage problem for {wrong_coverage} vs {real_coverage}, got: {problems}"
    )


def test_correct_coverage_is_silent(tmp_path):
    """Верное покрытие — ни одной проблемы с покрытием (защита от ложных срабатываний)."""
    # Загружаем реальный METRICS.json
    metrics = _load_real_metrics()
    real_coverage = metrics["tests"]["coverage_percent"]

    # Создаём документ с ПРАВИЛЬНЫМ покрытием
    doc = tmp_path / "probe.md"
    doc.write_text(f"Покрытие кода: {real_coverage} %\n", encoding="utf-8")

    # Вызываем реальную функцию check_docs
    real_repo_root = Path(__file__).parent.parent.parent
    problems = check_numbers.check_docs([doc], metrics, repo_root=real_repo_root)

    # Не должно быть проблем именно с этим файлом probe.md (могут быть другие проблемы с ИТОГ.md и т.д.)
    probe_problems = [p for p in problems if "probe.md" in p]
    assert len(probe_problems) == 0, (
        f"Expected no problems for probe.md with correct coverage {real_coverage}, got: {probe_problems}"
    )
