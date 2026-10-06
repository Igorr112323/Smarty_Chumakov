# Доказательства по дефектам D и E: падает на 1.2.0 — проходит на 1.2.1

Дата: 03.10.2026. Проверка воспроизводима двумя командами: код версии 1.2.0 лежит в
теге `v1.2.0`, тесты — в рабочем дереве (они специально используют ленивые импорты,
поэтому на старом коде падают **по поведению**, а не по ошибке импорта).

## Как воспроизвести

```bash
# 1. Код версии 1.2.0 в отдельное дерево, тесты — из рабочей версии
git worktree add --detach /tmp/v120 v1.2.0
cp spanverify-module/tests/test_number_attribution.py \
   spanverify-module/tests/test_dataset_validation.py /tmp/v120/spanverify-module/tests/

# 2. Прогон на коде 1.2.0 (ожидаем падения)
cd /tmp/v120/spanverify-module && python -m pytest -q \
   tests/test_number_attribution.py tests/test_dataset_validation.py

# 3. Прогон на коде 1.2.1 (ожидаем прохождения)
cd spanverify-module && python -m pytest -q \
   tests/test_number_attribution.py tests/test_dataset_validation.py
```

## Вывод команды на коде v1.2.0 (фактический, 13 падений)

```
FAILED tests/test_number_attribution.py::test_measurements_split_by_subject
FAILED tests/test_number_attribution.py::test_number_from_other_object_is_not_grounded
FAILED tests/test_number_attribution.py::test_attribution_finds_correct_subject_among_three
FAILED tests/test_number_attribution.py::test_substitution_of_sentence_in_three_facts_is_flagged
FAILED tests/test_number_attribution.py::test_attribution_requires_two_measurements
FAILED tests/test_number_attribution.py::test_attribution_is_quiet_without_subject
FAILED tests/test_dataset_validation.py::test_foreign_schema_is_rejected
FAILED tests/test_dataset_validation.py::test_missing_required_key_reports_line_number
FAILED tests/test_dataset_validation.py::test_label_bounds_are_checked
FAILED tests/test_dataset_validation.py::test_broken_json_reports_line_number
FAILED tests/test_dataset_validation.py::test_empty_file_is_rejected
FAILED tests/test_dataset_validation.py::test_cli_evaluate_rejects_foreign_dataset
FAILED tests/test_dataset_validation.py::test_zipapp_propagates_exit_codes
```

Ключевые строки из вывода (дословно):

```
# дефект E: CLI возвращает 0 там, где обязан вернуть 2
>       assert code == EXIT_ERROR, code
E       AssertionError: 0
E       assert 0 == 2

# дефект E на релизном артефакте: zipapp тоже терял код возврата
>       assert run("evaluate", "--dataset", str(foreign)) == EXIT_ERROR
E       AssertionError: assert 0 == 2

# дефект D: подмена числа считается подтверждённой
>       assert result.verdict != "grounded", result.verdict
E       AssertionError: grounded
E       assert 'grounded' != 'grounded'

# дефект D: подмена в середине трёхфактного документа не найдена
>       assert result.verdict != "grounded", result.verdict
E       AssertionError: grounded
```

## Вывод на коде v1.2.1 (фактический)

```
$ python -m pytest -q tests/test_number_attribution.py tests/test_dataset_validation.py
.................                                                        [100%]
17 passed
```

## Команды воспроизведения из приёмки на релизном `.pyz` (v1.2.1)

```
$ python spanverify.pyz verify \
    --context "Согласно регламенту, срок хранения первичных документов составляет пять лет. Срок хранения вторичных документов составляет десять лет." \
    --answer  "Срок хранения первичных документов составляет десять лет."
Вердикт: ЕСТЬ СОМНИТЕЛЬНЫЕ ФРАГМЕНТЫ
Фрагменты:
  - 0:57 риск 0.339 [doubtful] 'Срок хранения первичных документов составляет десять лет.'
rc=1
```

```
$ printf '%s\n' '{"id":"x1","question":"тест","answer":"тест","label":1}' > foreign.jsonl
$ python spanverify.pyz evaluate --dataset foreign.jsonl
ошибка: foreign.jsonl: строка 1: корпус не соответствует формату «контекст — ответ».
  - отсутствуют обязательные поля: context
  - неизвестные поля: label, question
  ожидалось: context, answer; допустимы также id, context, answer, labels, meta
  пример корректной строки: {"id": "p1", "context": "текст документа-источника", "answer": "текст ответа", "labels": [[12, 27, 1]], "meta": {"kind": "faithful"}}
rc=2
```

Контрольные исходы того же релизного файла: корректный корпус → `rc=0`
(240 пар, token F1 0.951), подтверждённый ответ → `rc=0` и «ОПОРА НА КОНТЕКСТ ЕСТЬ».

## Что это не доказывает

* Поведение на **реальных** документах не проверялось: правило привязки — текстовое и
  покрыто только синтетическими кейсами (два и три измерения в контексте).
* Режим `hf` с правилом привязки не прогонялся: нет GPU и весов модели.
* Ложные срабатывания измерены только на демо-корпусе (0 из 116 чистых пар).
