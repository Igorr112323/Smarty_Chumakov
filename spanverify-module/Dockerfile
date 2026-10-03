# SpanVerify — образ для запуска API и веб-интерфейса в контуре заказчика.
#   docker build -t spanverify .
#   docker run --rm -p 8000:8000 spanverify
#
# Образ намеренно без torch/transformers: по умолчанию демо-режим.
# Для режима реальной модели соберите с аргументом:
#   docker build --build-arg WITH_HF=1 -t spanverify:hf .

FROM python:3.11-slim

ARG WITH_HF=0
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    SPANVERIFY_HOST=0.0.0.0 \
    SPANVERIFY_PORT=8000

COPY spanverify ./spanverify
COPY config ./config
COPY data ./data
COPY app.py ./
COPY requirements-hf.txt ./

RUN if [ "$WITH_HF" = "1" ]; then pip install --no-cache-dir -r requirements-hf.txt; fi

# Непривилегированный пользователь.
RUN useradd --create-home --uid 10001 spanverify && chown -R spanverify /app
USER spanverify

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000", "--quiet"]
