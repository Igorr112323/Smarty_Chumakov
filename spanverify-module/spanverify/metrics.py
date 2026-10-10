"""Метрики Prometheus для HTTP-сервиса (задача 4, промышленный контур).

Реестр собран вручную, без ``prometheus_client``: сервис по контракту запускается
стандартной библиотекой и из собранного .exe, и каждая лишняя зависимость — это
риск, что бинарник не соберётся или не стартует (тот же довод, что у выбора
``http.server`` в :mod:`spanverify.server`). Формат ответа — текстовая экспозиция
Prometheus 1.0.0 (``# HELP``/``# TYPE`` + строки метрик), которую читают и
``promtool``, и сборщик.

Имена метрик закреплены контрактом:

* ``spanverify_requests_total`` — счётчик запросов по маршруту, коду ответа и
  режиму;
* ``spanverify_latency_seconds_bucket`` — гистограмма длительности обработки
  (бакеты покрывают и демо-режим на десятки миллисекунд, и режим ``hf`` на
  секунды);
* ``spanverify_verdict_total`` — счётчик выданных вердиктов;
* ``spanverify_model_mode`` — gauge с активным режимом (значение 1), чтобы
  дашборд видел подмену режима без опроса /health;
* ``spanverify_uptime_seconds`` — время жизни процесса.

Отдельных метрик по токенам нет сознательно: метка «токен» означала бы cardinality
по длине ответа, а Prometheus такие ряды храня́т плохо.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "BUCKETS_SECONDS",
    "Metrics",
    "format_prometheus",
    "global_metrics",
    "metric_names",
]

# Бакеты длительности. Средние границы выбраны так, чтобы p95 демо-режима
# (десятки миллисекунд) и hf-режима (сотни миллисекунд — секунды) попадали в
# разные интервалы: иначе алерт «p95 > 500 мс» было бы невозможно посчитать.
BUCKETS_SECONDS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

METRIC_HELP = {
    "spanverify_requests_total": "Запросов к HTTP-сервису, по маршруту, коду и режиму",
    "spanverify_latency_seconds": "Длительность обработки запроса, секунды (гистограмма)",
    "spanverify_verdict_total": "Выданные вердикты, по значению вердикта и режиму",
    "spanverify_model_mode": "Активный режим движка (значение 1)",
    "spanverify_uptime_seconds": "Время жизни процесса сервиса, секунды",
}

_LABEL_ORDER = ("route", "status", "mode", "verdict", "le")


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: Iterable[tuple[str, str]]) -> str:
    items = [(name, value) for name, value in pairs if value != ""]
    if not items:
        return ""
    return "{" + ",".join(f'{name}="{_escape(value)}"' for name, value in items) + "}"


def _fmt(value: float) -> str:
    """Число в формате Prometheus: без NaN/Inf, без лишнего «.0» у целых."""
    if math.isnan(value) or math.isinf(value):
        return "NaN"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(round(float(value), 6))


@dataclass
class Metrics:
    """Счётчики, гистограмма и режим; потокобезопасно (сервер многопоточный)."""

    started_at: float = field(default_factory=time.time)
    mode: str = "demo"
    version: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _requests: dict[tuple[str, str, str], float] = field(default_factory=dict)
    _verdicts: dict[tuple[str, str], float] = field(default_factory=dict)
    _latency_buckets: dict[str, dict[float, float]] = field(default_factory=dict)
    _latency_count: dict[str, float] = field(default_factory=dict)
    _latency_sum: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------- сбор событий

    def observe(self, route: str, status: int, latency_s: float, verdict: str = "", mode: str | None = None) -> None:
        """Записать один запрос: счётчик, длительность и (если есть) вердикт."""
        mode = self.mode if mode is None else mode
        latency = max(0.0, float(latency_s)) if math.isfinite(float(latency_s)) else 0.0
        with self._lock:
            self.mode = mode
            key = (route, str(int(status)), mode)
            self._requests[key] = self._requests.get(key, 0.0) + 1.0
            if verdict:
                vkey = (verdict, mode)
                self._verdicts[vkey] = self._verdicts.get(vkey, 0.0) + 1.0
            buckets = self._latency_buckets.setdefault(mode, {edge: 0.0 for edge in BUCKETS_SECONDS})
            for edge in BUCKETS_SECONDS:
                if latency <= edge:
                    buckets[edge] += 1.0
            self._latency_count[mode] = self._latency_count.get(mode, 0.0) + 1.0
            self._latency_sum[mode] = self._latency_sum.get(mode, 0.0) + latency

    # ------------------------------------------------------------------ снимок

    def snapshot(self) -> dict[str, object]:
        """Числа для /health и тестов: то же, что видит сборщик метрик."""
        with self._lock:
            requests = sum(self._requests.values())
            by_route: dict[str, float] = {}
            for (route, _status, _mode), value in self._requests.items():
                by_route[route] = by_route.get(route, 0.0) + value
            p95 = {
                item: _percentile(self._latency_buckets.get(item, {}), self._latency_count.get(item, 0.0), 0.95)
                for item in self._latency_count
            }
            return {
                "requests_total": int(requests),
                "by_route": {route: int(value) for route, value in sorted(by_route.items())},
                "verdicts": {verdict: int(value) for (verdict, _mode), value in sorted(self._verdicts.items())},
                "latency_sum_s": {mode: round(value, 4) for mode, value in sorted(self._latency_sum.items())},
                "latency_count": {mode: int(value) for mode, value in sorted(self._latency_count.items())},
                "latency_p95_s": p95,
                "uptime_s": round(time.time() - self.started_at, 1),
                "mode": self.mode,
            }

    def p95_seconds(self, mode: str | None = None) -> float | None:
        """p95 по бакетам (аппроксимация, линейная внутри бакета) или None."""
        with self._lock:
            modes = [self.mode] if mode is None else [mode]
            for item in modes:
                if item in self._latency_count:
                    return _percentile(self._latency_buckets.get(item, {}), self._latency_count[item], 0.95)
            values = sorted(
                _percentile(buckets, self._latency_count.get(item, 0.0), 0.95)
                for item, buckets in self._latency_buckets.items()
                if self._latency_count.get(item)
            )
            return values[-1] if values else None

    # -------------------------------------------------------------- текст ответа

    def render(self) -> str:
        """Текстовая экспозиция Prometheus 1.0.0."""
        lines: list[str] = []
        with self._lock:
            for name in ("spanverify_requests_total", "spanverify_verdict_total"):
                lines.append(f"# HELP {name} {METRIC_HELP[name]}")
                lines.append(f"# TYPE {name} counter")
            for (route, status, mode), value in sorted(self._requests.items()):
                labels = _labels((("route", route), ("status", status), ("mode", mode)))
                lines.append(f"spanverify_requests_total{labels} {_fmt(value)}")
            for (verdict, mode), value in sorted(self._verdicts.items()):
                labels = _labels((("verdict", verdict), ("mode", mode)))
                lines.append(f"spanverify_verdict_total{labels} {_fmt(value)}")

            lines.append(f"# HELP spanverify_latency_seconds {METRIC_HELP['spanverify_latency_seconds']}")
            lines.append("# TYPE spanverify_latency_seconds histogram")
            for mode in sorted(self._latency_count):
                # observe() копит бакеты уже кумулятивно (каждый край, накрывающий
                # длительность), поэтому здесь числа выводятся как есть: повторное
                # суммирование удвоило бы каждый бакет.
                buckets = self._latency_buckets.get(mode, {})
                total = self._latency_count.get(mode, 0.0)
                for edge in BUCKETS_SECONDS:
                    labels = _labels((("mode", mode), ("le", _fmt(edge))))
                    lines.append(f"spanverify_latency_seconds_bucket{labels} {_fmt(buckets.get(edge, 0.0))}")
                labels_inf = _labels((("mode", mode), ("le", "+Inf")))
                lines.append(f"spanverify_latency_seconds_bucket{labels_inf} {_fmt(total)}")
                labels_mode = _labels((("mode", mode),))
                lines.append(f"spanverify_latency_seconds_sum{labels_mode} {_fmt(self._latency_sum.get(mode, 0.0))}")
                lines.append(f"spanverify_latency_seconds_count{labels_mode} {_fmt(total)}")

            lines.append(f"# HELP spanverify_model_mode {METRIC_HELP['spanverify_model_mode']}")
            lines.append("# TYPE spanverify_model_mode gauge")
            lines.append(f'spanverify_model_mode{{mode="{_escape(self.mode)}",version="{_escape(self.version)}"}} 1')
            lines.append(f"# HELP spanverify_uptime_seconds {METRIC_HELP['spanverify_uptime_seconds']}")
            lines.append("# TYPE spanverify_uptime_seconds gauge")
            lines.append(f"spanverify_uptime_seconds {time.time() - self.started_at:.1f}")
        return "\n".join(lines) + "\n"


def _percentile(buckets: dict[float, float], total: float, quantile: float) -> float | None:
    """Квантиль по кумулятивным бакетам: ищем первый бакет, накрывающий долю.

    ``buckets`` — кумулятивные счётчики (как в текстовой экспозиции Prometheus),
    поэтому число наблюдений внутри бакета — разность соседних кумулятив.
    """
    if not buckets or not total:
        return None
    target = quantile * total
    previous_cumulative = 0.0
    previous_edge = 0.0
    for edge in BUCKETS_SECONDS:
        cumulative = buckets.get(edge, 0.0)
        count_in_bucket = cumulative - previous_cumulative
        previous_cumulative = cumulative
        if cumulative >= target:
            if count_in_bucket <= 0:
                return float(edge)
            # Линейная интерполяция внутри бакета: без неё p95 всегда равен
            # границе бакета, и 0,49 с с 0,06 с неразличимы — алерт по p95 врёт.
            share = min(1.0, max(0.0, (target - (cumulative - count_in_bucket)) / count_in_bucket))
            return round(previous_edge + (edge - previous_edge) * share, 6)
        previous_edge = edge
    return float(max(BUCKETS_SECONDS))


def format_prometheus(metrics: Metrics) -> str:
    """Тот же текст, что отдаёт /metrics (функция — для тестов и переиспользования)."""
    return metrics.render()


def metric_names() -> tuple[str, ...]:
    """Имена метрик контракта — их проверяет тест и документация."""
    return (
        "spanverify_requests_total",
        "spanverify_latency_seconds_bucket",
        "spanverify_verdict_total",
        "spanverify_model_mode",
        "spanverify_uptime_seconds",
    )


_GLOBAL = Metrics()


def global_metrics() -> Metrics:
    """Единственный реестр процесса: его наполняет HTTP-слой."""
    return _GLOBAL


def _reset_global_for_tests(mode: str = "demo", version: str = "") -> Metrics:
    """Сброс реестра между тестами (не часть публичного контракта)."""
    global _GLOBAL  # noqa: PLW0603 - намеренная замена единственного реестра процесса
    _GLOBAL = Metrics(mode=mode, version=version)
    return _GLOBAL
