# SpanVerify — образ для запуска API и веб-интерфейса в контуре заказчика.
#   docker build -t spanverify .
#   docker run --rm -p 8000:8000 spanverify
#
# Сборка многостадийная (задача 4, промышленный контур):
#
#   builder  — ставит зависимости (только если WITH_HF=1) в отдельный venv;
#   test     — прогоняет selftest собранного модуля: в финальный образ не
#              попадает то, что не умеет запускаться;
#   runtime  — тонкий слой: python:3.11-slim + venv из builder + код.
#
# В финальном образе нет pip-кэша, нет torch (по умолчанию), нет компилятора и
# нет тестов: поверхность атаки и размер меньше. Ядро, CLI, HTTP API и
# веб-интерфейс работают на стандартной библиотеке Python 3.10+, поэтому
# демо-режим не тянет ни одной внешней зависимости.
#
# Для режима реальной модели:
#   docker build --build-arg WITH_HF=1 -t spanverify:hf .
# Веса модели в образ НЕ записываются — их кладут в монтируемый каталог
# HF_HOME (см. docker-compose.yml и docs/ДЕПЛОЙ_ENTERPRISE.md): образ с весами
# весит гигабайты и не проходит проверку лицензий на модели.
#
# Аргументы сборки:
#   WITH_HF=0|1        — поставить torch/transformers (CPU-колёса)
#   UID=10001          — uid непривилегированного пользователя (для rootless
#                        cluster-складываний и readOnlyVolumeMounts с fsGroup)
#   PIP_INDEX_URL=…    — внутренний индекс пакетов, если внешний недоступен

# ------------------------------------------------------------------- зависимостей
FROM python:3.11-slim AS builder

ARG WITH_HF=0
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Отдельным слоем — чтобы зависимости кэшировались отдельно от кода.
COPY requirements-hf.txt /tmp/requirements-hf.txt
RUN if [ "$WITH_HF" = "1" ]; then \
      pip install --upgrade pip \
      && pip install --index-url "$PIP_INDEX_URL" --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-hf.txt \
      && pip freeze > /opt/venv/requirements.lock.txt \
      && cp /opt/venv/requirements.lock.txt /tmp/requirements.lock.txt; \
    else \
      pip install --upgrade pip && printf '# зависимости не ставились: демо-режим на стандартной библиотеке\n' > /tmp/requirements.lock.txt; \
    fi

# ----------------------------------------------------------------------- проверка
FROM python:3.11-slim AS test

WORKDIR /src
COPY spanverify ./spanverify
COPY config ./config
COPY app.py ./
# Проверка собрана на импортах и само-прогоне модуля: если какой-то слой
# (например, auth или metrics) не импортируется или selftest падает, сборка
# обрывается здесь, а не в проде на пятом запросе.
RUN python -c "import spanverify.auth, spanverify.metrics, spanverify.server, spanverify.cli; print('импорты ok', spanverify.__version__)" \
    && python -m spanverify selftest

# ------------------------------------------------------------------------ релизный
FROM python:3.11-slim AS runtime

ARG WITH_HF=0
ARG UID=10001
ARG VERSION=dev

LABEL org.opencontainers.image.title="SpanVerify" \
      org.opencontainers.image.description="Проверка достоверности ответа относительно контекста (HTTP API + веб-интерфейс)" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Igorr112323/Smarty_Chumakov" \
      org.opencontainers.image.base.name="docker.io/library/python:3.11-slim"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SPANVERIFY_HOST=0.0.0.0 \
    SPANVERIFY_PORT=8000 \
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=1 \
    HUGGINGFACE_HUB_CACHE=/models

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /tmp/requirements.lock.txt /app/requirements.lock.txt
COPY spanverify ./spanverify
COPY config ./config
COPY data ./data
COPY app.py ./
COPY README.md ./

RUN mkdir -p /models /var/log/spanverify && \
    useradd --create-home --uid "$UID" --shell /usr/sbin/nologin spanverify && \
    chown -R spanverify:spanverify /app /models /var/log/spanverify

ENV PATH="/opt/venv/bin:$PATH"

USER spanverify

# 8000 — API и интерфейс; метрики и аудит идут тем же портом (см. openapi.yaml).
EXPOSE 8000

# Проба живости — /health: он доступен без ключа доступа именно потому, что
# docker и Kubelet не отправляют заголовков (см. spanverify/auth.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Точка входа — обёртка app.py: она же подбирает порт для .exe-сценария.
# Флаги доступа (--api-keys, --audit-log) передают через SPANVERIFY_ARGS или
# аргументами в compose/k8s (примеры — в docs/ДЕПЛОЙ_ENTERPRISE.md).
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000", "--quiet"]
