"""Критерии качества: договорный ориентир и внутренний ориентир проекта.

Договор № ____ГУ/2026 (проект 117812), ТЗ п. 4.1, требование 6:

    целевой ориентир — F1 ≥ 0.50 при FPR ≤ 0.40 на отложенной по документам
    части набора из 1200 пар.

Внутренний (более жёсткий) ориентир проекта — F1 ≥ 0.90 при FPR ≤ 0.10 из
мастер-промта. Он остаётся в отчётах как отдельная строка и никогда не
подменяет договорный: подмена целевого значения запрещена условиями задачи.
Оба порога живут здесь, чтобы командная строка, отчёты и контрактные проверки
считали «достигнут/не достигнут» одинаково.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "CONTRACT_GATE",
    "CONTRACT_GATE_SOURCE",
    "FAMILIES",
    "INTERNAL_TARGET",
    "INTERNAL_TARGET_SOURCE",
    "gate_report",
    "gate_status",
]

#: Договорный ориентир (п. 4.1 треб. 6). Менять значения нельзя.
CONTRACT_GATE = {"f1_min": 0.50, "fpr_max": 0.40}
CONTRACT_GATE_SOURCE = "договор № ____ГУ/2026, проект 117812, ТЗ п. 4.1 треб. 6"

#: Внутренний ориентир проекта (мастер-промт): строже договорного, не приёмка.
INTERNAL_TARGET = {"f1_min": 0.90, "fpr_max": 0.10}
INTERNAL_TARGET_SOURCE = "мастер-промт проекта (внутренний ориентир, не приёмка)"

#: Группы метрик требования 5: токены, ответы, фрагменты.
FAMILIES = ("tokens", "spans", "answers")

STATUS_REACHED = "достигнут"
STATUS_NOT_REACHED = "НЕ достигнут"


def gate_status(metrics: Mapping[str, Any], gate: Mapping[str, float]) -> str:
    """Статус «достигнут / НЕ достигнут» для одной группы метрик."""
    f1 = metrics.get("f1")
    fpr = metrics.get("fpr")
    if f1 is None or fpr is None:
        return "не измерено"
    if float(f1) >= float(gate["f1_min"]) and float(fpr) <= float(gate["fpr_max"]):
        return STATUS_REACHED
    return STATUS_NOT_REACHED


def gate_report(metrics: Mapping[str, Any], family: str = "tokens") -> dict[str, Any]:
    """Отчёт по обоим ориентирам для одной группы метрик.

    ``metrics`` — блок метрик из ``Verifier.evaluate`` (``tokens``/``spans``/
    ``answers``). Для группы ``spans`` F1 берётся из строгого сопоставления с
    порогом IoU 0.5 (требование 5), FPR для фрагментов не определён — он
    считается по токенам и дублируется в отчёте с явной оговоркой.
    """
    if family not in FAMILIES:
        raise ValueError(f"неизвестная группа метрик: {family!r}, ожидается одна из {FAMILIES}")
    f1 = metrics.get("f1")
    fpr = metrics.get("fpr")
    if family == "spans" and fpr is None:
        # FPR фрагментов не определён (ложное срабатывание — это помеченный
        # токен), поэтому в отчёт идёт токенный FPR с явным именем поля.
        fpr = metrics.get("fpr_tokens")
    measured = {"f1": f1, "fpr": fpr}
    usable = f1 is not None and fpr is not None
    return {
        "family": family,
        "contract": {
            "gate": dict(CONTRACT_GATE),
            "source": CONTRACT_GATE_SOURCE,
            "f1": f1,
            "fpr": fpr,
            "status": gate_status(measured, CONTRACT_GATE) if usable else "не измерено",
        },
        "internal": {
            "gate": dict(INTERNAL_TARGET),
            "source": INTERNAL_TARGET_SOURCE,
            "f1": f1,
            "fpr": fpr,
            "status": gate_status(measured, INTERNAL_TARGET) if usable else "не измерено",
        },
    }
