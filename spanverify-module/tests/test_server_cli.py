"""Тесты HTTP-контракта и командной строки.

Проверяется то, что увидит пользователь: поля ответа API, коды возврата CLI,
понятные сообщения об ошибках. Сервер поднимается на свободном порту в потоке и
закрывается после теста — тесты не зависят от занятости 8765.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from spanverify.cli import EXIT_ERROR, EXIT_ISSUES, EXIT_OK, main
from spanverify.dataset import generate_pairs, write_pairs
from spanverify.server import DEFAULT_PORT, Service, free_port, make_handler

CONTEXT = "Регламент 343: срок хранения первичных документов составляет 10 лет."
ANSWER_FALSE = "Срок хранения первичных документов составляет 3 года."
ANSWER_TRUE = "Срок хранения первичных документов составляет 10 лет."


@pytest.fixture()
def service():
    """Сервис в демо-режиме на весах из репозитория (без скачивания моделей)."""
    return Service(mode="demo")


@pytest.fixture()
def http_server(service):
    """Поднять HTTP-сервер на свободном порту и вернуть его базовый адрес."""
    handler = make_handler(service)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    httpd.quiet = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str) -> tuple[int, dict, dict]:
    """GET-запрос с возвратом кода, JSON и заголовков."""
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
        payload = json.loads(body) if body else {}
        return response.status, payload, dict(response.headers)


def get_text(url: str) -> tuple[int, str, dict]:
    """GET-запрос с возвратом кода, текста и заголовков (для HTML)."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read().decode("utf-8"), dict(response.headers)


def post(url: str, payload: dict) -> tuple[int, dict, dict]:
    """POST JSON-запроса с возвратом кода, JSON и заголовков."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)


# ------------------------------------------------------------------ сервис


def test_service_health_reports_mode_and_version(service):
    """Здоровье сервиса сообщает режим, версию, порог и отсутствие ошибок."""
    health = service.health()
    assert health["status"] == "ok"
    assert health["mode"] == "demo"
    assert health["version"].count(".") == 2
    assert 0.0 <= health["threshold"] <= 1.0
    assert "weights" in health


def test_service_config_exposes_weights_schema(service):
    """Конфигурация отдаёт схему весов контракта и источник параметров."""
    config = service.config()
    assert set(config) >= {"weights", "threshold", "target_fpr", "isotonic", "head", "seed", "version"}
    assert config["runtime"]["mode"] == "demo"
    assert config["runtime"]["weights_source"] in {"disk", "embedded", "defaults"}


def test_service_model_reports_head_and_folds(service):
    """Сведения о модели содержат голову, признами и число фолдов."""
    model = service.model()
    assert model["mode"] == "demo"
    assert model["head"]["type"] in {"none", "logreg"}
    assert isinstance(model["folds"], list)
    assert model["seed"] == 42


def test_service_verify_matches_contract(service):
    """Разбор запроса сервиса возвращает все поля контракта."""
    body = service.verify({"answer": ANSWER_FALSE, "context": CONTEXT})
    assert set(body) >= {
        "score",
        "is_hallucination",
        "ai_share",
        "ai_share_hard",
        "threshold",
        "spans",
        "latency_ms",
        "mode",
        "stats",
        "verdict",
    }
    assert body["spans"], "подмена числа должна быть найдена"


def test_service_verify_accepts_legacy_text_field(service):
    """Прежнее поле text принимается как синоним answer (совместимость)."""
    body = service.verify({"text": ANSWER_FALSE, "context": CONTEXT})
    assert body["verdict"] in {"doubtful", "likely_hallucination"}


def test_service_verify_requires_answer(service):
    """Без ответа сервис сообщает понятную ошибку."""
    with pytest.raises(ValueError):
        service.verify({"context": CONTEXT})


def test_service_verify_accepts_context_list(service):
    """Контекст можно передать списком фрагментов документа."""
    body = service.verify({"answer": ANSWER_TRUE, "context": [CONTEXT, "Второй абзац документа."]})
    assert body["verdict"] in {"grounded", "doubtful"}


def test_service_verify_rejects_non_object(service):
    """Неверный тип тела запроса — это ошибка клиента."""
    with pytest.raises(ValueError):
        service.verify(["не объект"])


def test_free_port_returns_available_port():
    """Подбор порта возвращает свободный порт (не меньше запрошенного)."""
    probe = free_port(preferred=8765)
    assert probe >= 8765
    assert DEFAULT_PORT == 8765


# ------------------------------------------------------------------ HTTP


def test_health_endpoint(http_server):
    """GET /health отвечает 200 и JSON со статусом ok."""
    status, payload, headers = get(http_server + "/health")
    assert status == 200 and payload["status"] == "ok"
    assert headers["Content-Type"].startswith("application/json")


def test_config_and_model_endpoints(http_server):
    """GET /v1/config и /v1/model отдают 200 и ожидаемые ключи."""
    status, config, _ = get(http_server + "/v1/config")
    assert status == 200 and "weights" in config
    status, model, _ = get(http_server + "/v1/model")
    assert status == 200 and "head" in model


def test_verify_endpoint_contract(http_server):
    """POST /v1/verify возвращает контракт и находит подмену числа."""
    status, payload, _ = post(http_server + "/v1/verify", {"answer": ANSWER_FALSE, "context": CONTEXT})
    assert status == 200
    assert set(payload) >= {"score", "spans", "ai_share", "threshold", "stats"}
    assert payload["spans"]
    assert payload["spans"][0]["label"] in {"doubtful", "likely_hallucination"}


def test_verify_endpoint_with_tokens(http_server):
    """Флаг with_tokens добавляет в ответ покадровый разбор."""
    _, payload, _ = post(
        http_server + "/v1/verify",
        {"answer": ANSWER_FALSE, "context": CONTEXT, "with_tokens": True},
    )
    assert payload["tokens"]
    assert {"index", "text", "risk", "label"} <= set(payload["tokens"][0])


def test_index_page_is_served(http_server):
    """Корень отдаёт интерфейс и не запрещает показ во фрейме."""
    status, body, headers = get_text(http_server + "/")
    assert status == 200
    assert "SpanVerify" in body
    assert headers["Content-Type"].startswith("text/html")
    assert "X-Frame-Options" not in headers
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_unknown_route_returns_404(http_server):
    """Неизвестный маршрут — 404 с JSON-описанием."""
    with pytest.raises(urllib.error.HTTPError) as error:
        get(http_server + "/v1/nope")
    assert error.value.code == 404
    body = json.loads(error.value.read().decode("utf-8"))
    assert "не найден" in body["error"]


def test_bad_json_returns_400(http_server):
    """Некорректный JSON — 400, а не падение сервера."""
    request = urllib.request.Request(
        http_server + "/v1/verify",
        data="{не json}".encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 400


def test_missing_answer_returns_400(http_server):
    """Запрос без ответа — 400 с понятным текстом."""
    with pytest.raises(urllib.error.HTTPError) as error:
        post(http_server + "/v1/verify", {"context": CONTEXT})
    assert error.value.code == 400
    body = json.loads(error.value.read().decode("utf-8"))
    assert "answer" in body["error"]


def test_options_preflight(http_server):
    """CORS-предзапрос OPTIONS обрабатывается."""
    request = urllib.request.Request(http_server + "/v1/verify", method="OPTIONS")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status in {200, 204}
        assert response.headers["Access-Control-Allow-Methods"]


# ------------------------------------------------------------------ CLI


def test_cli_verify_confirmed_answer(monkeypatch, capsys):
    """CLI: подтверждённый ответ — код 0 и вердикт grounded."""
    code = main(["verify", "--answer", ANSWER_TRUE, "--context", CONTEXT])
    output = capsys.readouterr().out
    assert code == EXIT_OK
    assert "ОПОРА НА КОНТЕКСТ ЕСТЬ" in output


def test_cli_verify_json_output(capsys):
    """CLI: --json печатает разбор, пригодный для машинной обработки."""
    code = main(["verify", "--answer", ANSWER_FALSE, "--context", CONTEXT, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_ISSUES
    assert payload["spans"]


def test_cli_verify_tokens_flag(capsys):
    """CLI: --tokens добавляет построчный разбор рисков."""
    main(["verify", "--answer", ANSWER_FALSE, "--context", CONTEXT, "--tokens"])
    output = capsys.readouterr().out
    assert "Токены:" in output
    assert "риск" in output


def test_cli_verify_accepts_files(tmp_path, capsys):
    """CLI: ответ и контекст можно читать из файлов."""
    answer_file = tmp_path / "answer.txt"
    context_file = tmp_path / "context.txt"
    answer_file.write_text(ANSWER_FALSE, encoding="utf-8")
    context_file.write_text(CONTEXT, encoding="utf-8")
    code = main(["verify", "--answer-file", str(answer_file), "--context-file", str(context_file), "--json"])
    assert code == EXIT_ISSUES
    assert json.loads(capsys.readouterr().out)["spans"]


def test_cli_verify_without_answer_is_error(capsys):
    """CLI: без ответа команда вернёт код 2 и объяснит причину."""
    code = main(["verify", "--context", CONTEXT])
    assert code == EXIT_ERROR
    assert "нужен --answer" in capsys.readouterr().err


def test_cli_selftest_passes(capsys):
    """CLI: само-проверка проходит на встроенных примерах."""
    code = main(["selftest"])
    output = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Все проверки пройдены" in output
    assert "ПРОВАЛ" not in output


def test_cli_config_prints_json(capsys):
    """CLI: команда config печатает действующие параметры в JSON."""
    code = main(["config"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert "weights" in payload and "runtime" in payload


def test_cli_evaluate_on_generated_corpus(tmp_path, capsys):
    """CLI: evaluate считает метрики на сгенерированном корпусе."""
    path = write_pairs(generate_pairs(60, seed=77), tmp_path / "pairs.jsonl")
    code = main(["evaluate", "--dataset", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["pairs"] == 60
    assert 0.0 <= payload["tokens"]["f1"] <= 1.0


def test_cli_train_and_calibrate(tmp_path, capsys):
    """CLI: обучение и калибровка создают файл параметров."""
    path = write_pairs(generate_pairs(70, seed=78), tmp_path / "pairs.jsonl")
    out = tmp_path / "weights.json"
    code = main(["train", "--dataset", str(path), "--out", str(out), "--folds", "3"])
    assert code == EXIT_OK
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert pytest.approx(sum(payload["weights"].values())) == 1.0

    code = main(["calibrate", "--dataset", str(path), "--out", str(out)])
    assert code == EXIT_OK
    assert "Калибровка пересчитана" in capsys.readouterr().out


def test_cli_train_generates_dataset_when_missing(tmp_path, capsys):
    """CLI: если корпуса нет, он генерируется по --make-dataset."""
    path = tmp_path / "нет-корпуса.jsonl"
    code = main(
        [
            "train",
            "--dataset",
            str(path),
            "--out",
            str(tmp_path / "w.json"),
            "--make-dataset",
            "40",
            "--folds",
            "2",
        ]
    )
    assert code == EXIT_OK
    assert path.is_file()


def test_cli_train_without_dataset_is_error(tmp_path, capsys):
    """CLI: отсутствие корпуса без генерации — код 2 и объяснение."""
    code = main(["train", "--dataset", str(tmp_path / "нет.jsonl"), "--make-dataset", "0"])
    assert code == EXIT_ERROR
    assert "не найден" in capsys.readouterr().err


def test_cli_demo_runs_end_to_end(tmp_path, capsys):
    """CLI: демонстрационный прогон обучается и печатает итог."""
    code = main(["demo", "--pairs", "50", "--dataset", str(tmp_path / "demo.jsonl"), "--folds", "2"])
    output = capsys.readouterr().out
    assert code == EXIT_OK
    assert "демо-числа" in output.lower() or "Демо-числа" in output


def test_cli_analyze_legacy_branch(capsys):
    """CLI: историческая ветка analyze работает без контекста."""
    text = (
        "В современном мире ключевым аспектом эффективного функционирования "
        "любой организации является комплексная оптимизация процессов."
    )
    code = main(["analyze", text, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code in {EXIT_OK, EXIT_ISSUES}
    assert "verdict" in payload and "share" in payload


def test_cli_without_command_prints_help(capsys):
    """CLI: запуск без команды печатает справку и не падает."""
    code = main([])
    assert code == EXIT_OK
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_unknown_command_is_error():
    """CLI: неизвестная команда завершается кодом 2 (argparse)."""
    with pytest.raises(SystemExit) as error:
        main(["нет-такой-команды"])
    assert error.value.code == 2
