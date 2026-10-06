"""Совместимость модуля ``spanverify.api``.

С версии 1.1.0 HTTP-сервис живёт в :mod:`spanverify.server` и работает по
контракту «ответ против контекста». Этот модуль проверяет, что прежние имена
из ``spanverify.api`` продолжают импортироваться: внешние скрипты не должны
ломаться из-за переезда.
"""

from __future__ import annotations

import spanverify.api as api
from spanverify.server import Service as ServerService


def test_api_reexports_service():
    """Имя Service указывает на реализацию из server."""
    assert api.Service is ServerService


def test_api_reexports_helper_functions():
    """Вспомогательные функции сервера доступны через старый модуль."""
    assert callable(api.make_handler)
    assert callable(api.serve)
    assert callable(api.free_port)
    assert api.DEFAULT_PORT == 8765


def test_service_has_contract_methods():
    """Сервис предоставляет методы контракта: health/config/model/verify."""
    for name in ("health", "config", "model", "verify"):
        assert callable(getattr(api.Service, name))
