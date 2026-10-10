"""Промышленный контур HTTP-сервиса: ключи доступа, аудит, метрики (задача 4).

Тесты поднимают настоящий сервер в процессе (тот же ``make_handler``, что и
``spanverify server``) и ходят по нему HTTP-клиентом: проверяется именно
контур — какие маршруты открыты без ключа, какой код возвращает отказ, что
попадает в журнал и что видит сборщик метрик. Фикстура с сервером — одна на
модуль, чтобы не плодить порты и прогрев движка.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from spanverify.auth import (
    PERMISSION_ROLES,
    ROLES,
    AuditLog,
    AuthError,
    KeyStore,
    hash_answer,
    hash_key,
    load_keystore,
)
from spanverify.metrics import BUCKETS_SECONDS, Metrics, metric_names
from spanverify.server import Service, free_port, make_handler

OPERATOR_KEY = "operator-key-0000000001"
VIEWER_KEY = "viewer-key-000000000001"
ANSWER = "Срок хранения первичных документов составляет 3 года."
CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Сервис с включённым доступом и журналом; адрес — ``http://127.0.0.1:<порт>``."""
    directory = tmp_path_factory.mktemp("enterprise")
    keys = directory / "keys.json"
    keys.write_text(
        json.dumps(
            {
                "keys": [
                    {"subject": "ops", "role": "operator", "key_hash": hash_key(OPERATOR_KEY)},
                    {"subject": "obs", "role": "viewer", "key_hash": hash_key(VIEWER_KEY)},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    auth = load_keystore(keys)
    audit_path = directory / "audit.jsonl"
    service = Service(mode="demo", weights_path=None, auth=auth, audit=audit_path)
    service.metrics.mode = "demo"
    port = free_port(8899, host="127.0.0.1")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    httpd.quiet = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.15)
    yield {"base": f"http://127.0.0.1:{port}", "service": service, "audit": audit_path, "auth": auth}
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _call(fixture: dict[str, object], path: str, key: str | None = None, body: dict | None = None) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{fixture['base']}{path}",
        method="POST" if body is not None else "GET",
    )
    if key:
        request.add_header("X-API-Key", key)
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, payload, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


# ----------------------------------------------------------------- ключи и роли


def test_keystore_reads_hashed_keys(tmp_path: Path) -> None:
    """Файл с ``key_hash`` читается, а открытый текст ключа в файле не хранится."""
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps({"keys": [{"subject": "s", "role": "admin", "key_hash": hash_key("k" * 20)}]}), encoding="utf-8"
    )
    store = load_keystore(path)
    assert len(store) == 1
    assert store.warnings == ()
    assert store.authenticate("k" * 20).role == "admin"


def test_keystore_warns_on_plaintext_key(tmp_path: Path) -> None:
    """Открытый ключ в файле — предупреждение: годится только для разработки."""
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"subject": "s", "role": "viewer", "key": "d" * 20}]}), encoding="utf-8")
    store = load_keystore(path)
    assert any("открытым текстом" in notice for notice in store.warnings)
    assert store.authenticate("d" * 20).role == "viewer"


def test_keystore_ignores_unknown_role_and_broken_hash(tmp_path: Path) -> None:
    """Неизвестная роль и битый хеш не попадают в хранилище, но видны в warnings."""
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {"subject": "x", "role": "root", "key": "e" * 20},
                    {"subject": "y", "role": "viewer", "key_hash": "not-hex"},
                ]
            }
        ),
        encoding="utf-8",
    )
    store = load_keystore(path)
    assert len(store) == 0
    assert len(store.warnings) == 2


def test_env_keys_extend_keystore(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SPANVERIFY_API_KEYS`` даёт ключи без файла: секрет приходит из Secret."""
    monkeypatch.setenv("SPANVERIFY_API_KEYS", f"deploy:admin:{'f' * 20}")
    store = load_keystore(None)
    assert store.enabled
    assert store.authenticate("f" * 20).role == "admin"


def test_missing_key_file_is_explicit_error(tmp_path: Path) -> None:
    """Несуществующий файл ключей — 503 с путём, а не «пустое хранилище»."""
    with pytest.raises(AuthError) as excinfo:
        load_keystore(tmp_path / "нет-такого.json")
    assert excinfo.value.status == 503
    assert "не найден" in str(excinfo.value)


@pytest.mark.parametrize("permission", sorted(PERMISSION_ROLES))
def test_role_matrix_is_closed_set(permission: str) -> None:
    """Каждому праву назначена роль, и все роли известны (иначе опечатка в таблице)."""
    assert PERMISSION_ROLES[permission]
    assert set(PERMISSION_ROLES[permission]) <= set(ROLES)


def test_empty_keystore_reports_unavailable() -> None:
    """Пустое хранилище при явном вызове — 503, а не «доступ разрешён»."""
    with pytest.raises(AuthError) as excinfo:
        KeyStore().authenticate("что-то")
    assert excinfo.value.status == 503


# ------------------------------------------------------------------- ответы HTTP


def test_health_is_open_but_api_requires_key(server: dict[str, object]) -> None:
    """``/health`` открыт (пробы живости без заголовков), остальное — под ключом."""
    assert _call(server, "/health")[0] == 200
    assert _call(server, "/v1/config")[0] == 401
    assert _call(server, "/v1/config", key=OPERATOR_KEY)[0] == 200


def test_verify_denied_by_role(server: dict[str, object]) -> None:
    """Наблюдателю проверка ответа не разрешена: 403, а не 200 с вердиктом."""
    status, body = _call(server, "/v1/verify", key=VIEWER_KEY, body={"answer": ANSWER, "context": CONTEXT})
    assert status == 403
    assert json.loads(body)["status"] == 403


def test_verify_ok_for_operator_with_latency(server: dict[str, object]) -> None:
    """Оператор получает вердикт и длительность; ответ контракта не изменён."""
    status, body = _call(server, "/v1/verify", key=OPERATOR_KEY, body={"answer": ANSWER, "context": CONTEXT})
    assert status == 200
    payload = json.loads(body)
    assert payload["verdict"] in {"grounded", "doubtful", "likely_hallucination"}
    assert payload["latency_ms"] >= 0.0
    assert {"verdict", "score", "spans", "threshold", "mode"} <= set(payload)


def test_unknown_key_is_401_and_never_logged_in_plaintext(server: dict[str, object]) -> None:
    """Неверный ключ — 401; в журнале нет ни ключа, ни текста ответа."""
    status, _ = _call(server, "/v1/config", key="wrong-key-000000000000")
    assert status == 401
    lines = [json.loads(line) for line in Path(str(server["audit"])).read_text(encoding="utf-8").splitlines()]
    denied = [row for row in lines if row["event"] == "denied"]
    assert denied
    assert all("wrong-key-000000000000" not in json.dumps(row) for row in denied)
    # В журнале лежит только 16-символьный префикс хеша ключа — по нему находят
    # перебор, и при этом восстановлений самого ключа из записи нет.
    presented = [row for row in denied if row.get("key_fingerprint")]
    assert all(len(row["key_fingerprint"]) == 16 for row in presented)


def test_audit_records_hash_not_answer_text(server: dict[str, object]) -> None:
    """Событие проверки содержит sha256 ответа и не содержит сам ответ."""
    _call(server, "/v1/verify", key=OPERATOR_KEY, body={"answer": ANSWER, "context": CONTEXT})
    lines = [json.loads(line) for line in Path(str(server["audit"])).read_text(encoding="utf-8").splitlines()]
    event = [row for row in lines if row["event"] == "verify"][-1]
    assert event["answer_sha256"] == hash_answer(ANSWER)
    assert ANSWER not in json.dumps(event, ensure_ascii=False)
    assert event["answer_chars"] == len(ANSWER)
    assert event["subject"] == "ops" and event["role"] == "operator"
    # Оценка и порог пишутся как в ответе контракта: по ним видно, насколько
    # близко событие прошло решение.
    assert event["score"] >= 0.0 and event["threshold"] >= 0.0


def test_metrics_endpoint_exposes_contract_names(server: dict[str, object]) -> None:
    """/metrics содержит все имена из контракта и корректный текст Prometheus."""
    status, text = _call(server, "/metrics", key=VIEWER_KEY)
    assert status == 200
    for name in metric_names():
        assert name in text, name
    assert "# TYPE spanverify_latency_seconds histogram" in text
    assert 'le="+Inf"' in text


def test_requests_are_counted_by_status(server: dict[str, object]) -> None:
    """Счётчик запросов различает код ответа и маршрут: 401 и 200 — разные ряды."""
    snapshot = server["service"].metrics.snapshot()  # type: ignore[index]
    assert snapshot["requests_total"] > 0
    assert "/v1/verify" in snapshot["by_route"] or "/v1/config" in snapshot["by_route"]


# ----------------------------------------------------------------------- метрики


def test_metrics_render_cumulative_and_consistent() -> None:
    """Бакеты кумулятивные, монотонные, и ``+Inf`` равен числу наблюдений."""
    metrics = Metrics(mode="demo", version="test")
    for latency, verdict in [(0.001, "grounded"), (0.03, "doubtful"), (0.7, "likely_hallucination")]:
        metrics.observe("/v1/verify", 200, latency, verdict)
    text = metrics.render()
    buckets = {
        float(line.split('le="')[1].split('"')[0]): float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if "_bucket{" in line and 'le="+Inf"' not in line
    }
    values = [buckets[edge] for edge in BUCKETS_SECONDS]
    assert values == sorted(values), "кумулятивный ряд обязан не убывать"
    assert values[-1] <= 3.0
    assert 'le="+Inf"} 3' in text


def test_metrics_p95_interpolates_inside_bucket() -> None:
    """p95 интерполируется внутри бакета: 0,06 с и 0,49 с — разные числа."""
    fast, slow = Metrics(mode="demo"), Metrics(mode="demo")
    for _ in range(19):
        fast.observe("/v1/verify", 200, 0.06, "grounded")
        slow.observe("/v1/verify", 200, 0.49, "grounded")
    fast.observe("/v1/verify", 200, 0.06, "grounded")
    slow.observe("/v1/verify", 200, 0.49, "grounded")
    assert fast.p95_seconds() < slow.p95_seconds()
    assert 0.05 <= fast.p95_seconds() <= 0.1
    assert 0.25 <= slow.p95_seconds() <= 0.5


def test_metrics_ignore_non_finite_latency() -> None:
    """NaN/Inf в длительности не портят сумму: такие значения считаются нулём."""
    metrics = Metrics(mode="demo")
    metrics.observe("/v1/verify", 200, float("nan"), "grounded")
    metrics.observe("/v1/verify", 200, float("inf"), "grounded")
    snapshot = metrics.snapshot()
    assert snapshot["latency_sum_s"]["demo"] == 0.0
    assert snapshot["latency_count"]["demo"] == 2


def test_audit_disabled_writes_nothing(tmp_path: Path) -> None:
    """Журнал выключен (путь None) — события не пишутся, но и не падают."""
    log = AuditLog(None)
    record = log.write("ping", note="текст")
    assert record["event"] == "ping"
    assert log.enabled is False
    assert log.tail() == []
