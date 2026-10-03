"""Тесты корпуса A3: пары на фрагментах реальных документов.

Задание требует пяти проверок, и каждая проверяется здесь **как проверка**, а не как
наблюдение: для проверок есть негативные тесты (испорченный контекст и подменённое
число обязаны находиться), а не только утверждение «проблем нет».

1. каждый ``context`` дословно (после нормализации пробелов) есть в файлах источников;
2. ``answer[start:end] == span`` для каждой метки;
3. при «атрибуции числа» новое число присутствует в другом месте того же документа;
4. документ не попадает более чем в одну часть (train/dev/test);
5. манифест содержит SHA256 файлов и каждой пары, и они воспроизводятся повторно.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_corpus_a import DATASET_VERSION_A3, REAL_MODE_MIX, build, plan_modes
from scripts.corpus_real import (
    check_contexts_in_sources,
    check_number_attribution,
    normalize_text,
    pair_sha256,
)
from spanverify.dataset import read_pairs

TARGET = 200
SEED = 43

# Документы-фикстуры: не выдаются за нормативные акты, нужны только чтобы проверить
# свойства генератора A3 на текстах с датами, сроками, суммами и условиями.
FIXTURE_TEXTS = (
    "Правила хранения документов утверждены приказом от 12.03.2025 № 214. "
    "Срок хранения первичных учётных документов составляет пять лет, если иное не установлено договором. "
    "Документы, содержащие персональные данные работников, хранятся в течение 30 календарных дней, после чего уничтожаются. "
    "Уничтожение производится по акту, за исключением документов с грифом ограниченного доступа. "
    "Журнал учёта событий ведётся не менее 12 месяцев. Контроль исполнения возлагается на начальника отдела.",
    "Регламент защиты информации определяет порядок доступа к сведениям. "
    "Пароль должен содержать не менее 12 символов и меняется каждые 90 дней, если иное не установлено администратором. "
    "Резервная копия создаётся ежедневно в 2 часа ночи. Хранение резервных копий осуществляется 14 календарных дней. "
    "Доступ предоставляется на основании заявки, оформленной по форме приложения № 4.",
    "Инструкция по охране труда вводится в действие с 01.02.2026. "
    "Повторный инструктаж проводится не реже одного раза в 6 месяцев. "
    "Работники обеспечиваются средствами защиты в количестве 2 комплектов. "
    "Проверка знаний проводится комиссией из 3 человек, за исключением работников, прошедших обучение в течение 5 рабочих дней.",
    "Положение о кадровом учёте введено в действие с 01.08.2024. "
    "Личные карточки работников хранятся 50 лет, если иное не установлено архивным законодательством. "
    "Трудовые договоры хранятся в течение 75 лет. "
    "Выдача документов производится в течение 3 рабочих дней по письменному заявлению работника.",
    "Административный регламент устанавливает срок предоставления услуги. "
    "Срок предоставления услуги не должен превышать 15 рабочих дней со дня регистрации заявления. "
    "Уведомление направляется заявителю не позднее 5 рабочих дней. "
    "Отказ оформляется в письменной форме, если заявление подано в электронном виде. "
    "Жалоба рассматривается в течение 30 календарных дней.",
    "Порядок обработки обращений граждан утверждён распоряжением от 05.05.2025 № 88. "
    "Обращение регистрируется в течение 3 рабочих дней. Срок рассмотрения обращения составляет 30 дней, "
    "если иное не установлено законодательством. Продление срока допускается не более чем на 30 дней. "
    "Ответ направляется по адресу, указанному в обращении.",
    "Правила резервного копирования баз данных утверждены приказом от 07.11.2025 № 512. "
    "Полная копия создаётся один раз в 3 суток. Инкрементальная копия создаётся каждые 6 часов. "
    "Срок хранения копий составляет 45 суток, за исключением копий, сданных в архив. "
    "Проверка восстановления выполняется не реже одного раза в 6 месяцев.",
    "Регламент кадрового делопроизводства введён в действие с 01.09.2025. "
    "Приказы по личному составу хранятся 75 лет, если иное не предусмотрено перечнем. "
    "Личные дела увольняются в архив через 3 года. Опись дел составляется ежегодно. "
    "Выдача трудовой книжки производится в течение 3 рабочих дней.",
)


def _write_sources(docs_dir: Path, count: int = 8) -> None:
    """Разложить документы-фикстуры и sources.json загрузчика."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    documents = {}
    for index, text in enumerate(FIXTURE_TEXTS[:count], start=1):
        doc_id = f"eo-0001202501{index:04d}"
        (docs_dir / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")
        documents[doc_id] = {
            "source_url": f"http://publication.pravo.gov.ru/document/0001202501{index:04d}",
            "act_type": "Приказ",
            "act_number": f"{100 + index}",
            "act_date": f"2025-0{index}-15",
            "doc_id": doc_id,
            "extraction": "ocr-tesseract-rus",
        }
    (docs_dir / "sources.json").write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Собрать корпус A3 на фикстурах и вернуть пары, сплиты и манифест."""
    out = tmp_path_factory.mktemp("corpus_a3")
    docs = out / "sources"
    _write_sources(docs)
    manifest = build(docs, out, target=TARGET, seed=SEED, real=True)
    pairs = list(read_pairs(out / "pairs.jsonl"))
    # На маленьком корпусе-фикстуре часть сплитов может быть пустой (read_pairs пустой файл не читает).
    splits = {}
    for name in ("train", "dev", "test"):
        path = out / "splits" / f"{name}.jsonl"
        splits[name] = list(read_pairs(path)) if path.exists() and path.stat().st_size else []

    return {"manifest": manifest, "pairs": pairs, "splits": splits, "docs": docs, "out": out}


def test_plan_reserves_clean_pairs_and_min_per_mode() -> None:
    """План режимов: цель выдержана, чистых ≥40 %, на каждый тип ошибки ≥40 пар."""
    plan = plan_modes(1200, REAL_MODE_MIX)
    counts = {mode: plan.count(mode) for mode in REAL_MODE_MIX}
    assert len(plan) == 1200
    assert counts["faithful"] / len(plan) >= 0.40
    for mode in REAL_MODE_MIX:
        if mode != "faithful":
            assert counts[mode] >= 40, (mode, counts[mode])


def test_every_context_is_verbatim_in_sources(corpus: dict) -> None:
    """Проверка №1: контекст каждой пары дословно найден в файле источника."""
    assert corpus["manifest"]["contexts_problems"] == []
    assert corpus["manifest"]["contexts_checked"] == len(corpus["pairs"])
    sources = {path.stem: normalize_text(path.read_text(encoding="utf-8")) for path in corpus["docs"].glob("*.txt")}
    for pair in corpus["pairs"]:
        source = sources[pair["meta"]["doc_id"]]
        assert normalize_text(pair["context"]) in source, pair["id"]


def test_context_check_catches_tampered_context(corpus: dict) -> None:
    """Проверка №1 не пустая: испорченный контекст обязан быть найден."""
    broken = dict(corpus["pairs"][0])
    broken["context"] = "Этот фрагмент в документах отсутствует."
    report = check_contexts_in_sources([broken], corpus["docs"])
    assert report["checked"] == 1
    assert len(report["problems"]) == 1


def test_spans_match_answer_slices(corpus: dict) -> None:
    """Проверка №2: срез ответа по метке совпадает с размеченным фрагментом."""
    assert corpus["manifest"]["spans_problems"] == []
    labelled = 0
    for pair in corpus["pairs"]:
        texts = pair["meta"]["span_texts"]
        assert len(texts) == len(pair["labels"]), pair["id"]
        for index, (start, end, label) in enumerate(pair["labels"]):
            assert label == 1
            assert pair["answer"][int(start) : int(end)] == texts[index], pair["id"]
            labelled += 1
    assert labelled > 0
    assert corpus["manifest"]["spans_checked"] == labelled


def test_number_attribution_value_is_elsewhere_in_same_document(corpus: dict) -> None:
    """Проверка №3: новое число при «атрибуции числа» есть в другом месте документа."""
    assert corpus["manifest"]["number_attribution_problems"] == []
    checked = corpus["manifest"]["number_attribution_checked"]
    assert checked >= 1
    sources = {path.stem: normalize_text(path.read_text(encoding="utf-8")) for path in corpus["docs"].glob("*.txt")}
    for pair in corpus["pairs"]:
        if pair["meta"]["mode"] != "number_attribution":
            continue
        source = sources[pair["meta"]["doc_id"]]
        start, end = pair["meta"]["fact_span"]
        outside = source[:start] + " " + source[end:]
        assert pair["meta"]["span_texts"][0] in outside, pair["id"]


def test_number_attribution_check_catches_foreign_value(corpus: dict) -> None:
    """Проверка №3 не пустая: значение, которого нет в другом месте, обязано быть найдено."""
    pair = next(item for item in corpus["pairs"] if item["meta"]["mode"] == "number_attribution")
    broken = json.loads(json.dumps(pair))
    broken["meta"]["span_texts"] = ["значение-которого-нет-в-документе"]
    report = check_number_attribution([broken], corpus["docs"])
    assert report["checked"] == 1
    assert len(report["problems"]) == 1


def test_document_is_in_exactly_one_split(corpus: dict) -> None:
    """Проверка №4: документ не попадает более чем в одну часть."""
    assert corpus["manifest"]["shared_groups"] == 0
    seen: dict[str, set[str]] = {}
    for name, pairs in corpus["splits"].items():
        for pair in pairs:
            seen.setdefault(pair["meta"]["group"], set()).add(name)
    assert seen and all(len(parts) == 1 for parts in seen.values())
    assert sum(len(block) for block in corpus["splits"].values()) == len(corpus["pairs"])


def test_manifest_hashes_are_reproducible(corpus: dict) -> None:
    """Проверка №5: хеши файлов и каждой пары в манифесте совпадают при пересчёте."""
    out: Path = corpus["out"]
    manifest = corpus["manifest"]
    assert manifest["dataset_version"] == DATASET_VERSION_A3
    assert manifest["real_documents"] is True
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert actual == expected, name
    for pair in corpus["pairs"]:
        assert manifest["pair_sha256"][pair["id"]] == pair_sha256(pair)
    for path in sorted(corpus["docs"].glob("*.txt")):
        assert manifest["sources_sha256"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebuild_gives_same_numbers(corpus: dict, tmp_path: Path) -> None:
    """Повторная сборка на тех же источниках даёт те же хеши файлов и пар."""
    second = tmp_path / "second"
    docs = second / "sources"
    _write_sources(docs)
    manifest = build(docs, second, target=TARGET, seed=SEED, real=True)
    assert manifest["sha256"] == corpus["manifest"]["sha256"]
    assert manifest["pairs"] == corpus["manifest"]["pairs"]
    assert manifest["balance"] == corpus["manifest"]["balance"]


def test_pairs_have_real_document_metadata(corpus: dict) -> None:
    """Обязательные поля meta заполнены, а документ помечен как настоящий."""
    required = ("source_url", "act_type", "act_number", "act_date", "doc_id")
    for pair in corpus["pairs"]:
        meta = pair["meta"]
        assert meta["synthetic_document"] is False
        assert meta["kind"] == "corpus_a3_real"
        for field in required:
            assert meta.get(field), (pair["id"], field)
        assert pair["context"], pair["id"]
        assert pair["answer"], pair["id"]


def test_pairs_are_unique(corpus: dict) -> None:
    """Одинаковых пар в корпусе нет: ключ «контекст + ответ» уникален."""
    keys = [(pair["context"], pair["answer"]) for pair in corpus["pairs"]]
    assert len(keys) == len(set(keys))


def test_build_real_refuses_empty_and_generated_documents(tmp_path: Path) -> None:
    """Корпус A3 не собирается из пустого каталога и не принимает синтетику."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        build(empty, tmp_path / "out", target=10, seed=1, real=True)

    generated = tmp_path / "gen"
    (generated / "generated").mkdir(parents=True)
    (generated / "generated" / "doc-0001.md").write_text("Синтетический текст." * 3, encoding="utf-8")
    with pytest.raises(ValueError):
        build(generated, tmp_path / "out2", target=10, seed=1, real=True)
