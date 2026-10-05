#!/usr/bin/env python3
"""Сгенерировать фикстуру сабтокенов для тестов сопоставления (пункт 2.1).

Зачем. Тесты привязки сабтокенов к символам ответа должны работать на
**настоящем** byte-level BPE, а не на придуманных смещениях: именно byte-level
BPE включает ведущий пробел в сабтокен и именно на этом «съезжал» пилот.
При этом продукт и его тесты не должны зависеть от сети и от пакета
``tokenizers``.

Решение: токенизатор обучается здесь, один раз, на текстах реальных актов из
``data/corpus_a3/sources``; в репозиторий попадает только результат разбора
(``tests/fixtures/bpe_offsets.json``) — тексты, сабтокены и смещения.

Запуск (нужен пакет ``tokenizers``, см. requirements.txt):

    python scripts/make_alignment_fixture.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "bpe_offsets.json"

# Тексты подобраны так, чтобы покрыть все ловушки сопоставления:
# ведущий пробел, дефис, кавычки-ёлочки, «ё», число с точкой, длинный текст.
TEXTS = [
    "Срок хранения составляет пять лет, если иное не установлено договором.",
    "Документы по личному составу хранятся 50 лет, а бухгалтерская отчётность — 5 лет.",
    "Приказ Минтруда России от 29.10.2021 № 772н «Об утверждении основных требований».",
    "Обращение рассматривается в течение 30 дней со дня регистрации; срок может быть продлён.",
    "Объём вложения — не более 10 МБ; формат — PDF/A-1.",
]


def main() -> int:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError:
        print("нужен пакет tokenizers: pip install -r requirements.txt", file=sys.stderr)
        return 2

    corpus = sorted(glob.glob(str(ROOT / "data" / "corpus_a3" / "sources" / "*.txt")))
    if not corpus:
        print("нет текстов в data/corpus_a3/sources — сначала запустите fetch_npa_corpus.py", file=sys.stderr)
        return 2

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.train(corpus, trainers.BpeTrainer(vocab_size=3000, special_tokens=["<unk>"], show_progress=False))

    long_text = " ".join(TEXTS) * 12  # длинный текст для проверки окон
    payload = {
        "generator": "scripts/make_alignment_fixture.py",
        "tokenizer": "byte-level BPE, vocab=3000, обучен на data/corpus_a3/sources/*.txt",
        "corpus_files": [Path(p).name for p in corpus],
        "samples": [],
    }
    for text in [*TEXTS, long_text]:
        encoded = tokenizer.encode(text)
        payload["samples"].append(
            {
                "text": text,
                "tokens": encoded.tokens,
                "offsets": [list(pair) for pair in encoded.offsets],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано: {OUT} ({OUT.stat().st_size} байт, образцов {len(payload['samples'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
