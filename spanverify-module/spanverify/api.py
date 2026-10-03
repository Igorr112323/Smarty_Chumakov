"""Совместимый псевдоним HTTP-сервиса (реализация — в ``spanverify.server``).

Раньше в этом модуле жил API текстового детектора. С версии 1.1.0 продукт
работает по контракту «ответ против контекста», и сервис переехал в
:mod:`spanverify.server`. Модуль оставлен, чтобы внешний код и скрипты,
импортировавшие ``spanverify.api``, продолжали работать.

    from spanverify.api import Service, make_handler, serve
"""

from .server import DEFAULT_PORT, Service, free_port, make_handler, serve

__all__ = ["Service", "make_handler", "serve", "free_port", "DEFAULT_PORT"]
