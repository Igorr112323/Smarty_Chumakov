"""Тесты измерений, на которые опираются решения по корпусу v2.

Смысл этих проверок — не «код работает», а «число в отчёте воспроизводимо»: доля актов с
годным текстовым слоем и состояние ветки данных считаются скриптами, и тест закрепляет
и значение, и то, как именно оно получено (знаменатель, разбиение по host, интервал).
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hf_protocol_v2 as v2  # noqa: E402
import text_layer_yield as yield_mod  # noqa: E402
import v2_corpus_status as status_mod  # noqa: E402

status_mod._mode_and_approval = status_mod.mode_and_approval

SOURCES_V1 = ROOT / "data" / "corpus_a3" / "sources" / "sources.json"
MODE_FILE = REPO_ROOT / ".github" / "corpus-v2.mode"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "corpus-v2.yml"
YIELD_REPORT = ROOT / "reports" / "v2" / "text_layer_yield.json"
STATUS_REPORT = ROOT / "reports" / "v2" / "corpus_status.json"
NPA_HOST = "publication.pravo.gov.ru"


def test_wilson_interval_is_exact_for_a_small_share() -> None:
    """Интервал Уилсона обязан быть несимметричным и не выходить за [0; 1]."""
    low, high = yield_mod.wilson_interval(4, 272)
    assert (round(low, 4), round(high, 4)) == (0.0057, 0.0372)
    assert low < 4 / 272 < high
    zero_low, zero_high = yield_mod.wilson_interval(0, 10)
    assert zero_low == 0.0 and 0.0 < zero_high < 0.4
    assert yield_mod.wilson_interval(0, 0) == (0.0, 0.0)


def test_text_layer_share_of_the_real_source_is_measured_not_remembered() -> None:
    """4 документа из 272 скачанных с портала — это то, что даёт режим «только текстовый слой»."""
    docs = yield_mod.load_documents(SOURCES_V1)
    assert len(docs) == 409, "манифест корпуса v1 изменился: сверяйте обоснование режима загрузки"
    npa = [rec for rec in docs if str(rec.get("source_host")) == NPA_HOST]
    part = yield_mod.measure(npa, 500)
    assert part["n_scored"] == 272
    assert part["docs_with_text_layer"] == 4
    assert abs(part["share_by_method"] - 4 / 272) < 1e-6
    assert [round(value, 4) for value in part["ci95_by_method"]] == [0.0057, 0.0372]
    # годность по объёму слоя обязана совпасть с методом: расхождение означало бы,
    # что «OCR» и «пустой слой» в манифестах — разные документы
    assert part["docs_with_usable_layer_chars"] == 4


def test_local_donor_records_are_not_poured_into_the_source_denominator() -> None:
    """Смешивание локального донора с порталом дало «34 % годовых слоёв» — этого больше не будет."""
    docs = yield_mod.load_documents(SOURCES_V1)
    mixed = yield_mod.measure(docs, 500)
    npa_only = yield_mod.measure([rec for rec in docs if str(rec.get("source_host")) == NPA_HOST], 500)
    assert (
        mixed["share_by_method"] > 0.3 > npa_only["share_by_method"]
    ), "доля по всей совокупности обязана заметно превышать долю источника: иначе разбиение по host потеряно"


def test_records_without_a_method_are_excluded_from_both_sides() -> None:
    """Запись без метода извлечения не попадает ни в числитель, ни в знаменатель."""
    rec = {"doc_id": "x", "act_type": "Приказ", "pages_count": 3}
    assert yield_mod.method_of(rec) == ""
    part = yield_mod.measure(
        [rec, {"doc_id": "y", "extraction": "pypdf", "pages_count": 1, "text_layer_chars": 900}], 500
    )
    assert part["n_scored"] == 1 and part["n_without_method"] == 1
    assert part["docs_with_text_layer"] == 1 and part["share_by_method"] == 1.0


def test_manifest_field_shapes_both_supported() -> None:
    """Манифест v1 пишет `extraction` строкой, срезы v2 — словарём: обе формы читаются."""
    flat_doc = {"extraction": "ocr-tesseract-rus", "text_layer_chars": 900, "pages_count": 1}
    nested_doc = {"extraction": {"method": "pypdf", "text_layer_chars": 1200}, "pages_count": 2}
    assert yield_mod.usable(flat_doc, 500) == (False, True)
    assert yield_mod.usable(nested_doc, 500) == (True, True)
    assert yield_mod.method_of(nested_doc) == "pypdf"


def test_sample_is_deterministic_for_a_fixed_seed() -> None:
    """Случайный срез обязан воспроизводиться по зерну (иначе «выборка» не проверяема)."""
    import random

    docs = yield_mod.load_documents(SOURCES_V1)
    first = random.Random(20261011).sample(docs, 100)
    second = random.Random(20261011).sample(docs, 100)
    assert [rec["doc_id"] for rec in first] == [rec["doc_id"] for rec in second]
    assert yield_mod.measure(first, 500)["n_scored"] > 0


def test_yield_report_on_disk_matches_the_script(tmp_path: Path) -> None:
    """Коммитнутый отчёт обязан совпадать с пересчётом — иначе число в документах устарело."""
    if not YIELD_REPORT.is_file():
        pytest.skip("отчёт ещё не сгенерирован: `python3 scripts/text_layer_yield.py --all --json ...`")
    payload = json.loads(YIELD_REPORT.read_text(encoding="utf-8"))
    recomputed = yield_mod.measure(
        [rec for rec in yield_mod.load_documents(SOURCES_V1) if str(rec.get("source_host")) == NPA_HOST],
        payload["min_chars_per_page"],
    )
    stored = payload["by_host"][NPA_HOST]
    for key in ("n_scored", "docs_with_text_layer", "share_by_method", "ci95_by_method"):
        assert stored[key] == recomputed[key], f"{key}: отчёт на диске разошёлся с пересчётом"


def test_mode_switch_accepts_only_the_three_documented_values() -> None:
    assert status_mod.read_mode(MODE_FILE) in {"text-only", "ocr", "paused"}


def test_every_wave_job_waits_for_the_mode_gate() -> None:
    """Пауза обязана останавливать и `chain` (он запрашивает следующую волну), иначе она фиктивна."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = spec["jobs"]
    assert "gate" in jobs, "в workflow корпуса нет job'а `gate`, читающего режим"
    slice_needs = jobs["slice"].get("needs")
    assert slice_needs == "gate" or (isinstance(slice_needs, list) and "gate" in slice_needs)
    chain_needs = jobs["chain"].get("needs")
    assert isinstance(chain_needs, list) and "gate" in chain_needs and "archive" in chain_needs
    for name in ("slice", "assemble", "archive", "chain"):
        condition = jobs[name].get("if", "")
        assert "needs.gate.outputs.run" in condition, f"job {name} запускается мимо гейта режима"
    assert "needs.archive.result" in jobs["chain"].get("if", ""), "цепочка не сверяется с результатом архива"


def test_ocr_is_installed_only_in_the_ocr_mode() -> None:
    """Распознавание ставится только когда режим подтверждён — иначе «решение заказчика» отменяется молча."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'needs.gate.outputs.mode }}" = "ocr"' in text, "установка tesseract не привязана к режиму ocr"
    assert "needs.gate.outputs.mode == 'text-only' && '--text-only'" in text, "флаг --text-only не привязан к режиму"
    assert "--text-only' || ''" in text, "в режиме ocr флаг --text-only обязан не передаваться"


def test_chain_commit_hygiene_is_enforced_in_ci() -> None:
    """Коммит цепочки в ветку данных: только markdown-счётчик, малый размер, проверка на токены и ключи."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "git diff --cached --name-only" in text
    assert "wave/*.md" in text or "wave/" in text
    assert "gh[pousr]_" in text, "цепочка не проверяет дифф на токены GitHub"
    assert "PRIVATE KEY" in text, "цепочка не проверяет дифф на приватные ключи"
    assert "20000" in text, "нет ограничения на размер коммита цепочки"


def test_corpus_status_counts_documents_and_marks_pending_approval(tmp_path: Path) -> None:
    """Состояние ветки данных: счётчик документов + флаг «распознавание не подтверждено заказчиком»."""
    acts = tmp_path / "acts"
    acts.mkdir(parents=True)
    files = [
        ("acts/a.txt", ("текст " * 400).encode("utf-8")),
        ("acts/b.txt", b""),
        ("acts/c.txt", "коротко".encode()),
        ("reports-wave-x.md", "# волна\n".encode()),
    ]
    counts = status_mod.summarize(files)
    assert counts["documents_in_acts"] == 3
    assert counts["documents_nonempty"] == 2
    assert counts["documents_at_least_min_chars"] == 1
    assert counts["wave_reports"] == 1


def test_mode_and_approval_grammar() -> None:
    """`ocr` без отметки `approved YYYY-MM-DD` — не подтверждённый режим; `text-only` — выбор заказчика."""
    cases = {
        "text-only": ("text-only", True),
        "paused": ("paused", False),
        "ocr": ("ocr", False),
        "ocr approved 2026-10-12 решением заказчика (письмо от 12.10)": ("ocr", True),
        "ocr approved tomorrow": ("ocr", False),
        "": ("text-only", True),
    }
    for line, expected in cases.items():
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mode"
            if line:
                path.write_text(line, encoding="utf-8")
            assert status_mod.mode_and_approval(path) == expected, repr(line)


def test_ocr_flag_follows_the_mode_file(tmp_path: Path, monkeypatch) -> None:
    """Флаг `ocr_pending_approval` ставится, пока режим распознавания не подтверждён заказчиком."""
    mode = tmp_path / "mode"
    monkeypatch.setattr(status_mod, "mode_and_approval", lambda path=mode: status_mod._mode_and_approval(path))
    wave = tmp_path / "wave"
    (wave / "acts").mkdir(parents=True)
    out = tmp_path / "status.json"
    cases = {
        "paused": True,
        "ocr": True,
        "ocr approved 2026-10-12 заказчиком": False,
        "text-only": False,
    }
    for value, expected in cases.items():
        mode.write_text(value, encoding="utf-8")
        assert status_mod.mode_and_approval(mode) == (value.split()[0] if value else "text-only", not expected)
        code = status_mod.main(["--dir", str(wave), "--data-branch", "test/branch", "--json", str(out)])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["ocr_pending_approval"] is expected, value
        assert payload["mode_approved_by_customer"] is (not expected), value


def test_committed_status_report_exists_and_says_pending(tmp_path: Path) -> None:
    """Отчёт в репозитории обязаны читать заказчик: он про 84 файла и про ожидание подтверждения."""
    if not STATUS_REPORT.is_file():
        pytest.skip("отчёт генерируется командой `python3 scripts/v2_corpus_status.py --git-ref FETCH_HEAD --json ...`")
    payload = json.loads(STATUS_REPORT.read_text(encoding="utf-8"))
    assert payload["ocr_pending_approval"] is True
    assert payload["load_mode"] == status_mod.read_mode(MODE_FILE)
    assert payload["counts"]["documents_in_acts"] > 0
    assert payload["data_branch"] == "data/corpus-v2-raw"


def test_mode_file_is_a_single_token(tmp_path: Path) -> None:
    """Файл режима читается одной строкой: комментарии и лишние слова в `paused` недопустимы."""
    raw = MODE_FILE.read_text(encoding="utf-8")
    assert len(raw.strip().splitlines()) == 1, "файл режима обязан содержать ровно одну строку"
    token = raw.strip().split()[0]
    assert token in {"text-only", "ocr", "paused"}
    if token != "ocr":
        assert len(raw.strip().split()) == 1, f"лишние слова в режиме {token}: режим читается одним словом"


def test_ocr_mode_requires_the_approval_note_in_ci() -> None:
    """CI тоже требует отметку подтверждения: тихий `paused → ocr` не должен разблокировать волны."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "approved" in text and "[0-9]{4}-[0-9]{2}-[0-9]{2}" in text, "гейт не проверяет отметку решения заказчика"
    assert "text-only|ocr) run=yes" not in text, "гейт по-прежнему включает ocr без подтверждения"


def test_freeze_blocks_while_the_load_mode_is_unapproved(tmp_path: Path, capsys) -> None:
    """Шаг 0 не замораживает выборку, полученная часть которой — в неподтверждённом режиме."""
    pending = tmp_path / "corpus_status.json"
    pending.write_text(
        json.dumps(
            {
                "load_mode": "paused",
                "mode_approved_by_customer": False,
                "ocr_pending_approval": True,
                "counts": {"documents_in_acts": 84},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = Namespace(
        splits=str(ROOT / "data" / "corpus_a3" / "splits"),
        config=str(ROOT / "config" / "hf_experiments_v2.json"),
        seeds="42,43,44,45,46",
        out=str(tmp_path / "отчёт"),
        corpus_status=str(pending),
    )
    code = v2.command_freeze(args)
    text = capsys.readouterr().out
    assert code == 1
    assert "режим загрузки не подтверждён заказчиком" in text
    payload = json.loads((tmp_path / "отчёт" / "freeze.json").read_text(encoding="utf-8"))
    assert payload["corpus_status"]["ocr_pending_approval"] is True
    assert payload["corpus_status"]["documents_in_acts"] == 84


def test_freeze_accepts_an_approved_mode_and_records_it(tmp_path: Path, capsys) -> None:
    """После подтверждения запись о режиме остаётся в заморозке — режим нельзя сменить задним числом."""
    approved = tmp_path / "corpus_status.json"
    approved.write_text(
        json.dumps(
            {
                "load_mode": "ocr",
                "mode_approved_by_customer": True,
                "ocr_pending_approval": False,
                "counts": {"documents_in_acts": 1600},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = Namespace(
        splits=str(ROOT / "data" / "corpus_a3" / "splits"),
        config=str(ROOT / "config" / "hf_experiments_v2.json"),
        seeds="42,43,44,45,46",
        out=str(tmp_path / "отчёт"),
        corpus_status=str(approved),
    )
    v2.command_freeze(args)
    capsys.readouterr()
    payload = json.loads((tmp_path / "отчёт" / "freeze.json").read_text(encoding="utf-8"))
    assert payload["corpus_status"]["mode_approved_by_customer"] is True
    assert not any("не подтверждён заказчиком" in problem for problem in payload["problems"])


def test_chain_fails_instead_of_guessing_about_missing_outputs() -> None:
    """Пустые `before/after` — ошибка, а не «нулевой прогресс»: иначе цепочка молча умирает."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ -z "${BEFORE:-}" ] || [ -z "${AFTER:-}" ]' in text
    assert "числа не доехали" in text, "сообщение об ошибке обязано называть различие двух случаев"


def test_chain_verifies_that_the_counter_branch_moved() -> None:
    """Push без проверки — это «успешный» job, который ничего не запросил."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'git ls-remote origin "refs/heads/$branch"' in text
    assert 'if [ "$remote" != "$pushed" ]' in text
    assert "следующая волна НЕ запрошена" in text
    assert "GITHUB_TOKEN" in text, "в summary обязано быть правило про токен, не порождающий прогоны"


def test_numbers_doc_records_the_withdrawn_chain_claim() -> None:
    """Снятое заявление обязано оставаться снятым в документе с числами."""
    text = (ROOT / "docs" / "ЧИСЛА_v2.md").read_text(encoding="utf-8")
    assert "автоцепочка не работала" in text.lower()
    corpus = (ROOT / "docs" / "CORPUS_v2.md").read_text(encoding="utf-8")
    assert "Автоцепочка волн: измеренное состояние — не работала" in corpus, "раздел с состоянием цепочки обязАН быть"
    assert "ни одного от счётчика" in corpus, "документ обязан содержать проверяемую формулировку факта"
    assert "точная причина не установлена" in corpus, "нельзя фиксировать причину, которую не видели"
