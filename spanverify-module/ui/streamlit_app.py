"""Опциональный интерфейс на Streamlit (для внутренних демонстраций).

Запуск:
    pip install -r requirements-ui.txt
    streamlit run ui/streamlit_app.py

Основной интерфейс — встроенный (/), он не требует зависимостей и
работает из собранного .exe. Streamlit нужен, когда требуется
интерактивная подстройка параметров и просмотр таблиц.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spanverify import Config, Detector  # noqa: E402

st.set_page_config(page_title="SpanVerify", page_icon="🔍", layout="wide")

st.title("SpanVerify — доля участия ИИ и локализация фрагментов")
st.caption("Локальный анализ. Текст не отправляется во внешние сервисы.")

with st.sidebar:
    st.header("Параметры")
    backend = st.selectbox("Бэкенд", ["surrogate", "hf"], index=0,
                           help="surrogate — демо-конвейер, hf — реальная модель")
    threshold = st.slider("Порог", 0.0, 1.0, 0.5, 0.01)
    calibrate = st.checkbox("Использовать калибратор", value=True)
    show_tokens = st.checkbox("Показать покадровую таблицу", value=False)
    st.divider()
    st.write("**Метрики режима**")
    if backend == "surrogate":
        st.warning("Демо-режим: оценивается работоспособность конвейера, а не достоверность текста.")

text = st.text_area("Текст для проверки", height=280, placeholder="Вставьте документ…")

if st.button("Проверить", type="primary") and text.strip():
    config = Config(backend=backend, threshold=threshold)
    detector = Detector(config, calibrator=True if calibrate else False)
    payload = detector.explain(text, threshold=threshold)

    result = payload["result"]
    ai_fraction = result["ai_fraction"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Доля ИИ (символы)", f"{ai_fraction:.1%}")
    col2.metric("Доля ИИ (слова)", f"{result['ai_fraction_tokens']:.1%}")
    col3.metric("Фрагментов", result["spans_count"])

    st.progress(min(1.0, ai_fraction), text=f"Вердикт: {result['verdict']}")
    for warning in result.get("warnings", []):
        st.warning(warning)

    if result["spans"]:
        st.subheader("Разметка")
        marked = text
        for span in reversed(result["spans"]):
            start, end = span["start_char"], span["end_char"]
            marked = (
                marked[:start]
                + "**==\\[" + marked[start:end] + "\\]==**"
                + marked[end:]
            )
        st.markdown(marked)

        st.subheader("Фрагменты")
        st.dataframe(
            [
                {
                    "№": s["index"] + 1,
                    "начало": s["start_char"],
                    "конец": s["end_char"],
                    "токенов": s["n_tokens"],
                    "средняя вероятность": round(s["mean_prob"], 3),
                    "пик": round(s["peak_prob"], 3),
                    "фрагмент": s["text"][:120],
                }
                for s in result["spans"]
            ],
            use_container_width=True,
        )
    else:
        st.info("Фрагментов выше порога не найдено.")

    if show_tokens:
        st.subheader("Покадровая оценка")
        st.dataframe(payload["tokens"][:600], use_container_width=True)
elif text.strip():
    st.info("Нажмите «Проверить», чтобы запустить анализ.")
