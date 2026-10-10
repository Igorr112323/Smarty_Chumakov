"""Сверка openapi.yaml с кодом сервиса (задача 4).

`openapi-spec-validator` проверяет только синтаксис схемы. Опасность другого рода:
описание расходится с реализацией — маршрут добавили в `server.py`, а в контракте
его нет (или наоборот), поле ответа переименовали, метрику переименовали. Для
заказчика это хуже, чем отсутствие схемы: контракт начинает врать.

Скрипт сравнивает три вещи, каждую — с живым кодом:

1. маршруты `paths` против разбора путей в `spanverify/server.py`;
2. поля `VerifyResult` против ключей, которые реально возвращает
   `Verifier.verify(...).to_dict()`;
3. имена метрик в описании `/metrics` против `spanverify.metrics.metric_names()`.

Возвращает 0, если расхождений нет; иначе печатает список и возвращает 1.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "openapi.yaml"
SERVER_PATH = ROOT / "spanverify" / "server.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415 - зависимость нужна только этому скрипту
    except ImportError as exc:  # pragma: no cover - окружение без pyyaml
        raise SystemExit("нужен pyyaml: python -m pip install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def routes_in_code(path: Path) -> set[str]:
    """Маршруты, которые разбирает обработчик: строковые литералы сравнений path."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    pattern = re.compile(r"^[a-z0-9_./-]+$")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if pattern.match(value) and (value.startswith("/") or value in {"index.html", "favicon.ico"}):
                found.add("/" + value.lstrip("/"))
    # Веб-интерфейс и иконка — не часть API-контракта; /verify — исторический
    # псевдоним /v1/verify, он тоже обязан быть описан.
    return {item for item in found if item.startswith("/v1/") or item in {"/health", "/metrics", "/verify"}}


def result_keys() -> set[str]:
    """Ключи ответа движка на реальном вызове (демо-режим, без весов)."""
    from spanverify.engine import Verifier

    verifier = Verifier(mode="demo", weights_path=None)
    result = verifier.verify("Срок хранения 3 года.", "Регламент: срок хранения 10 лет.", with_tokens=True)
    return set(result.to_dict(with_tokens=True))


def metric_names() -> set[str]:
    from spanverify.metrics import metric_names as contract

    return set(contract())


def check(spec_path: Path = SPEC_PATH, server_path: Path = SERVER_PATH) -> list[str]:
    """Список расхождений; пустой список — контракт и код совпадают."""
    problems: list[str] = []
    if not spec_path.is_file():
        return [f"нет файла {spec_path}"]
    spec = _load_yaml(spec_path)
    if not isinstance(spec, dict) or spec.get("openapi", "").startswith("2."):
        return ["в openapi.yaml ожидалась версия 3.x"]

    documented = set(spec.get("paths") or {})
    implemented = routes_in_code(server_path)
    for route in sorted(implemented - documented):
        problems.append(f"маршрут {route} есть в server.py, но не описан в openapi.yaml")
    for route in sorted(documented - implemented):
        problems.append(f"маршрут {route} описан в openapi.yaml, но не обрабатывается в server.py")

    schemas = (spec.get("components") or {}).get("schemas") or {}
    verify_schema = (schemas.get("VerifyResult") or {}).get("properties") or {}
    documented_fields = set(verify_schema)
    if not documented_fields:
        problems.append("в схеме VerifyResult нет ни одного поля")
    actual = result_keys()
    # Поля, которые движок отдаёт, но контракт не описывает, — это обещание
    # «стабильности» для клиента, которого нет: их обязано описать или удалить.
    for field_name in sorted(actual - documented_fields):
        problems.append(f"поле ответа {field_name!r} отдаётся движком, но не описано в VerifyResult")
    for field_name in sorted(documented_fields - actual):
        problems.append(f"поле {field_name!r} описано в VerifyResult, но движок его не возвращает")
    required = set(((schemas.get("VerifyResult") or {}).get("required")) or [])
    for field_name in sorted(required - actual):
        problems.append(f"поле {field_name!r} помечено required, но отсутствует в реальном ответе")

    request_schema = (schemas.get("VerifyRequest") or {}).get("properties") or {}
    for field_name in ("answer", "context", "with_tokens"):
        if field_name not in request_schema:
            problems.append(f"поле запроса {field_name!r} не описано в VerifyRequest")

    metrics_text = json.dumps(spec, ensure_ascii=False)
    for name in sorted(metric_names()):
        if name not in metrics_text:
            problems.append(f"метрика {name} не упомянута в openapi.yaml (раздел /metrics)")

    info = spec.get("info") or {}
    for key in ("title", "version", "description"):
        if not str(info.get(key) or "").strip():
            problems.append(f"в info нет обязательного поля {key!r}")
    security_schemes = (spec.get("components") or {}).get("securitySchemes") or {}
    if "ApiKeyAuth" not in security_schemes:
        problems.append("нет схемы безопасности ApiKeyAuth (X-API-Key)")
    health = ((spec.get("paths") or {}).get("/health") or {}).get("get") or {}
    if health.get("security") != []:
        problems.append("/health обязан быть с security: [] — пробы живости ходят без ключа")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сверка openapi.yaml с кодом сервиса")
    parser.add_argument("--spec", default=str(SPEC_PATH))
    parser.add_argument("--server", default=str(SERVER_PATH))
    parser.add_argument("--json", action="store_true", help="расхождения в JSON (для CI-сводки)")
    args = parser.parse_args(argv)

    problems = check(Path(args.spec), Path(args.server))
    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems}, ensure_ascii=False, indent=2))
    elif problems:
        print(f"Расхождения контракта и кода ({len(problems)}):")
        for item in problems:
            print(f"  - {item}")
    else:
        routes = sorted(set((_load_yaml(Path(args.spec)) or {}).get("paths") or {}))
        print(f"Контракт сходится с кодом: маршрутов {len(routes)}, метрик {len(metric_names())}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
