"""Тесты HTTP API: /health, /v1/verify, обработка ошибок."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from spanverify.api import create_server
from spanverify.config import Config
from spanverify.demo_data import make_ai_paragraph


@pytest.fixture(scope="module")
def server():
    config = Config(backend="surrogate", host="127.0.0.1", port=0)
    srv = create_server(config, quiet=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="module")
def base_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def get_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_health_reports_backend(base_url: str):
    status, payload = get_json(f"{base_url}/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["backend"] == "surrogate"
    assert "версия" not in payload and payload["version"]


def test_root_serves_interface(base_url: str):
    with urllib.request.urlopen(base_url + "/", timeout=10) as response:
        html = response.read().decode("utf-8")
    assert response.status == 200
    assert "SpanVerify" in html
    assert "/v1/verify" in html  # интерфейс обращается по относительному адресу


def test_verify_returns_result(base_url: str):
    import random

    text = make_ai_paragraph(random.Random(1), 5)
    status, payload = post_json(f"{base_url}/v1/verify", {"text": text})
    assert status == 200
    result = payload["result"]
    assert 0.0 <= result["ai_fraction"] <= 1.0
    assert result["n_word_tokens"] > 10
    assert isinstance(result["spans"], list)


def test_verify_with_explain_returns_tokens(base_url: str):
    _, payload = post_json(
        f"{base_url}/v1/verify",
        {"text": "Данный метод обеспечивает эффективное решение задачи.", "explain": True},
    )
    assert payload["tokens"]
    assert {"raw", "prob", "flag"} <= set(payload["tokens"][0])


def test_verify_rejects_empty_text(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(f"{base_url}/v1/verify", {"text": "  "})
    assert error.value.code == 400


def test_verify_rejects_bad_threshold(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(f"{base_url}/v1/verify", {"text": "Текст", "threshold": 5})
    assert error.value.code == 400


def test_unknown_route_returns_404(base_url: str):
    with pytest.raises(urllib.error.HTTPError) as error:
        get_json(f"{base_url}/v1/unknown-endpoint")
    assert error.value.code == 404


def test_config_endpoint_returns_parameters(base_url: str):
    status, payload = get_json(f"{base_url}/v1/config")
    assert status == 200
    assert payload["backend"] == "surrogate"
    assert payload["k_neighbors"] >= 1
