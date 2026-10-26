# Карта тестов SpanVerify

Дата: **2026-10-19**. Базовый принятый HEAD: `725824030f24862830c277d9e537c5f05727d8b0`.
Реальная коллекция получена на `ce68caf72c0294630039e6874fd24ef6e53e4c49` после удаления только рабочих промптов. Презентационная правка расширяет существующий documentation-consistency test, не добавляя и не удаляя test nodes.

## Источник и способ проверки

```bash
cd spanverify-module
python -m pytest --collect-only -q
python -m pytest
```

**972 собранных nodes в 77 файлах**, включая параметризованные случаи. Это не число функций: например, YAML-проверки разворачиваются для каждого workflow. Полный список nodeid и количество по каждому файлу сохранены в [TESTS_INVENTORY.json](presentation/TESTS_INVENTORY.json). Названия в таблице — реальные файлы `spanverify-module/tests/`, а не примеры будущих тестов.

Базовое полное выполнение: **908 passed, 2 skipped, 0 failed, 0 errors**, tests_excluded=false. Машинное доказательство: [committed JUnit](../reports/p8/closure/local-suite.xml), [LOCAL_SUITE](../reports/p8/closure/LOCAL_SUITE.json); окончательные CI-проверки презентационного HEAD указываются в квитанции/финальном ответе, а не выдаются за этот исторический запуск.

## Пофайловая карта: проверка → гарантия

Гарантия ниже означает защиту от регрессии в проверенных fixtures/артефактах, **не математическое доказательство**, не новую оценку корпуса и не обещание качества на любом тексте. Число справа — количество реально collected nodes этого файла.

| Файл / группа | Nodes | Что проверяет | Что защищает |
|---|---:|---|---|
| [test_answer_head.py](../spanverify-module/tests/test_answer_head.py) | 17 | Статистики рисков, размеры и finite-значения, сериализация, sklearn fit и full pipeline | Answer-head сохраняет предсказание при round-trip и отвергает неверный вход |
| [test_answer_label.py](../spanverify-module/tests/test_answer_label.py) | 7 | Положительные, отрицательные, пустые и смешанные labels, короткие числовые spans | Метка ответа выводится из разметки без потери короткого факта |
| [test_api.py](../spanverify-module/tests/test_api.py) | 3 | Реэкспорт Service и вспомогательных функций, методы контракта | Публичный совместимый импорт не исчезает |
| [test_backends.py](../spanverify-module/tests/test_backends.py) | 9 | Фабрика/алиасы, surrogate-признаки, пустой ввод, missing HF dependencies, CLI options | Режим выбирается явно; неизвестный backend не принимается |
| [test_calibration.py](../spanverify-module/tests/test_calibration.py) | 15 | PAVA, монотонность/границы, порог под FPR, ручные метрики, CV | Калибровка и выбор порога согласованы с проверяемыми формулами |
| [test_calibration_fix.py](../spanverify-module/tests/test_calibration_fix.py) | 13 | Повторяющиеся x, перестановки, граничные значения, компрессия и сериализация | Регрессии калибровки на равных scores не возвращаются |
| [test_check_numbers.py](../spanverify-module/tests/test_check_numbers.py) | 4 | Распознавание десятичного покрытия, ложные совпадения, намеренно неверное число | Числовой checker ловит подмену, не принимая посторонний текст за метрику |
| [test_chunking.py](../spanverify-module/tests/test_chunking.py) | 23 | Нормализация, overlap/offsets, длинные предложения, top-k/BM25 и union claims | Чанки сохраняют координаты и корректный отбор контекста |
| [test_ci_workflows.py](../spanverify-module/tests/test_ci_workflows.py) | 5 | Полнота workflow/job registry, triggers/status/provenance, Windows size gates; README links и inventory | Документация CI и точная карта suite согласованы с файлами; четыре download URL и локальные ссылки существуют |
| [test_clean_pair_fpr.py](../spanverify-module/tests/test_clean_pair_fpr.py) | 3 | Непустой clean split, доля ложных замечаний и отсутствие spans на корректных парах | Проверяется регрессия ложных обвинений на clean fixtures |
| [test_collect_experiments.py](../spanverify-module/tests/test_collect_experiments.py) | 4 | Чтение committed run files, official fields и внешних срезов, числовой whitelist | Сводки берут числа из источников, а не констант из памяти |
| [test_config.py](../spanverify-module/tests/test_config.py) | 16 | Defaults, типы/unknown keys, environment precedence, overrides, save/load, pyz roots | Конфигурация и поиск runtime-файлов воспроизводимы |
| [test_context_index.py](../spanverify-module/tests/test_context_index.py) | 5 | Сравнение с brute force, дубликаты, пустой контекст, ручная density | Индекс не меняет вычисления по сравнению с полным перебором |
| [test_context_mass_norm.py](../spanverify-module/tests/test_context_mass_norm.py) | 8 | Expected causal mass, равномерное внимание, длина, нулевой фон и lift | Нормировка опоры не смешивает длину документа с силой поддержки |
| [test_context_stack.py](../spanverify-module/tests/test_context_stack.py) | 90 | Offsets, scored predicate, признаки, engine, span helpers, dataset, train/save/load | Интеграция базового конвейера и синтетические QA-fixtures согласованы |
| [test_contract.py](../spanverify-module/tests/test_contract.py) | 21 | Покрытие пунктов ТЗ, provenance чисел, документы, checksum, размеры, CI coverage gate | Требование связано с проверкой и артефактом; неподтверждённое число не принимается |
| [test_contract_provenance.py](../spanverify-module/tests/test_contract_provenance.py) | 2 | Local/CI provenance и отказ при отсутствующем CI artifact | Контрактное доказательство не принимается без происхождения |
| [test_contract_surface.py](../spanverify-module/tests/test_contract_surface.py) | 14 | API/CLI features, лимит текста/тела запроса, пять признаков, confidence и единые gate | Пользовательская поверхность соответствует проверяемому контракту |
| [test_corpus_a.py](../spanverify-module/tests/test_corpus_a.py) | 14 | Генерация корпуса, режимы, разметка/координаты, документная группировка | Синтетические пары и labels внутренне согласованы |
| [test_corpus_a2.py](../spanverify-module/tests/test_corpus_a2.py) | 8 | Разбор фактов/условий, вопросы и промпты, deterministic tasks, dry-run, needs_expert_review | Черновые labels A2 не выдаются за экспертную истину; GPU-генерация здесь не измеряется |
| [test_corpus_a3.py](../spanverify-module/tests/test_corpus_a3.py) | 12 | Реальные контексты A3, режимы/разметка, document-clean split и воспроизводимость | Документы не протекают между частями; план A3 проверяется явно |
| [test_corpus_a3_report.py](../spanverify-module/tests/test_corpus_a3_report.py) | 5 | Содержимое отчёта A3 и его источники | Отчёт не подменяет данные или их происхождение |
| [test_coverage.py](../spanverify-module/tests/test_coverage.py) | 14 | Факты/субъекты, missing/partial/irrelevant, словесные числа, offsets и пустой ввод | Покрытие проверяется по fixtures, а не по подписи UI |
| [test_dataset_validation.py](../spanverify-module/tests/test_dataset_validation.py) | 10 | Плохие labels, offsets, структура и типы данных | Некорректная разметка отвергается до измерения |
| [test_demo_data.py](../spanverify-module/tests/test_demo_data.py) | 7 | Режимы и воспроизводимость демонстрационных данных | Demo-fixtures остаются корректными, не объявляются реальными наблюдениями |
| [test_detector.py](../spanverify-module/tests/test_detector.py) | 12 | Входы Detector, параметры, обработка ошибок и результат | Высокоуровневый Detector сохраняет публичное поведение |
| [test_engine.py](../spanverify-module/tests/test_engine.py) | 3 | Проверка движка на фиксированных примерах | Базовая проверка ответа выполняется end-to-end |
| [test_experiment_official_split.py](../spanverify-module/tests/test_experiment_official_split.py) | 3 | Роли train/dev/test и поля официального split | Эксперимент не переименовывает dev в официальный test |
| [test_external_datasets.py](../spanverify-module/tests/test_external_datasets.py) | 13 | Чтение/преобразование внешних наборов, поля и координаты | Внешняя разметка не заменяется выдуманными offsets |
| [test_external_fetch_and_eval.py](../spanverify-module/tests/test_external_fetch_and_eval.py) | 16 | Fetch/cache/eval fixtures и честные ошибки источников | Импорт и отчётность внешнего набора проверяются без обещания его качества |
| [test_feature_cache.py](../spanverify-module/tests/test_feature_cache.py) | 13 | Ключи/метаданные cache, несовместимость и содержимое | Повторное использование признаков не смешивает несовместимые данные |
| [test_feature_cache_roundtrip.py](../spanverify-module/tests/test_feature_cache_roundtrip.py) | 3 | Запись/чтение и сохранение признаков | Все пять полей переживают dump/load; старый формат выявляется, collect_samples работает |
| [test_fetch_npa_corpus.py](../spanverify-module/tests/test_fetch_npa_corpus.py) | 9 | Получение НПА на fixtures, parsing, ошибки/кэш | Источник реальных документов обрабатывается контролируемо |
| [test_fetch_rushallu.py](../spanverify-module/tests/test_fetch_rushallu.py) | 9 | Загрузка/преобразование RusHallu на fixtures | Поля и разметка источника сохраняются при импорте |
| [test_gold_counts.py](../spanverify-module/tests/test_gold_counts.py) | 2 | Пересчёт committed gold denominators | Знаменатели метрик не дрейфуют незаметно |
| [test_head_applied.py](../spanverify-module/tests/test_head_applied.py) | 8 | Загрузка и влияние head на runtime/eval | Заявленная голова действительно участвует в предсказании |
| [test_hf_chunked.py](../spanverify-module/tests/test_hf_chunked.py) | 6 | HF window fixtures и сборка признаков по offsets | Оконная обработка не теряет координаты текста |
| [test_hf_footprint_measure.py](../spanverify-module/tests/test_hf_footprint_measure.py) | 10 | Состав и размеры модели/библиотек, отчёт и ошибки измерения | Footprint не сводится к размеру одного EXE |
| [test_hf_no_silent_fallback.py](../spanverify-module/tests/test_hf_no_silent_fallback.py) | 5 | Отсутствующие зависимости/модель, реальный backend при доступности | HF не переходит в surrogate без явного сообщения |
| [test_metrics_files.py](../spanverify-module/tests/test_metrics_files.py) | 12 | Синтетическая оговорка, split/перенос, числа/документы и реальная коллекция session | Документация согласована с зарегистрированными числами и количеством тестов |
| [test_no_test_isolation.py](../spanverify-module/tests/test_no_test_isolation.py) | 2 | Поведение --no-test: ошибка без splits, evaluate только train/dev, test_official=null | Запрет официального измерения при --no-test действительно исполняется |
| [test_normalize.py](../spanverify-module/tests/test_normalize.py) | 12 | Нормализация слов/чисел и устойчивость к вариантам записи | Числовые формы сравниваются предсказуемо |
| [test_number_attribution.py](../spanverify-module/tests/test_number_attribution.py) | 8 | Число вместе с субъектом и соседним контекстом | Чужое число документа не подтверждает факт другого субъекта |
| [test_number_words_in_answer.py](../spanverify-module/tests/test_number_words_in_answer.py) | 6 | Числа словами в ответе и их токенизация/атрибуция | Словесное значение не выпадает из проверки факта |
| [test_oof_protocol.py](../spanverify-module/tests/test_oof_protocol.py) | 11 | OOF фолды, роли split и leakage guard | Обучение/подбор ограничены разрешёнными частями |
| [test_oof_public_pipeline.py](../spanverify-module/tests/test_oof_public_pipeline.py) | 6 | Публичный OOF runtime и паритет данных | OOF-свидетельства относятся к публичному конвейеру |
| [test_p1_integration.py](../spanverify-module/tests/test_p1_integration.py) | 6 | P1 pipeline, сохранение и интеграция answer-head | Базовый P1 работает совместно с runtime |
| [test_p1_replay_contract.py](../spanverify-module/tests/test_p1_replay_contract.py) | 13 | Replay P1, артефакты, параметры, предикаты и паритет | Повтор базовой точки не меняет её протокол |
| [test_p2_public_influence.py](../spanverify-module/tests/test_p2_public_influence.py) | 9 | Token-head в публичном pipeline и его влияние | P2 не остаётся только offline-результатом |
| [test_p2_tau.py](../spanverify-module/tests/test_p2_tau.py) | 4 | Выбор/применение token threshold | Токенный порог применяется согласно замороженному правилу |
| [test_p3_localization.py](../spanverify-module/tests/test_p3_localization.py) | 17 | Seed/grow, границы, bridge и fail-closed конфигурация | Локализация воспроизводима и отвергает неверную схему |
| [test_p4_context.py](../spanverify-module/tests/test_p4_context.py) | 14 | Scored/pair boundaries, spec, doc-folds, sklearn export и public round-trip | Контекстные признаки не пересекают пары и совпадают с экспортом |
| [test_p5_span_filter.py](../spanverify-module/tests/test_p5_span_filter.py) | 21 | Pure candidates, finite/edges, no-gold, schema, empty-kept, cross-fit и round-trip | Фильтр не читает gold, пустой kept авторитетен, legacy не меняется |
| [test_p7_engineering.py](../spanverify-module/tests/test_p7_engineering.py) | 6 | Byte-exact intervals, unicode windows, stitching, cosine, share calibration isolation | Оптимизации не теряют токены и не меняют исходную арифметику качества |
| [test_p7_publication.py](../spanverify-module/tests/test_p7_publication.py) | 2 | Raw proof, overlay отчёта, path traversal и checksum отказ | Публикация не теряет доказательства и не принимает опасный ZIP |
| [test_p8_evidence_hygiene.py](../spanverify-module/tests/test_p8_evidence_hygiene.py) | 5 | Provenance, immutable legacy, индекс, frozen hashes, sealed receipt copy | Исторические числа и predictive-файлы не переписываются |
| [test_participation.py](../spanverify-module/tests/test_participation.py) | 7 | Границы/доступность ai_participation, fixture смеси, round-trip, CLI/UI | Поле участия присутствует; fixture не объявляется доказательством авторства |
| [test_review_server.py](../spanverify-module/tests/test_review_server.py) | 7 | Решения ручной проверки, статистика, invalid IDs, deterministic sample | Ручная проверка сохраняет решения без тихой потери |
| [test_r2_share.py](../spanverify-module/tests/test_r2_share.py) | 3 | Изотонный агрегатор доли (дорожка Д, IsotonicFragmentAggregator, monotone PAVA) | Доля AI монотонна по форме, не везением |
| [test_r2_coverage_extra.py](../spanverify-module/tests/test_r2_coverage_extra.py) | 37 | Покрытие H1: r2_share агрегативные пути + engine-интеграция (isolated/isotonic) | Новые ветки R2 (PAVA, transform, estimate/legacy, save/load) и R2-интеграция Verifier покрыты |
| [test_rus_hallu_eval.py](../spanverify-module/tests/test_rus_hallu_eval.py) | 5 | Точные подстроки, две группы метрик, clean pairs, missing data | Внешняя оценка не выдумывает baseline или отсутствующий набор |
| [test_rushallu_metrics.py](../spanverify-module/tests/test_rushallu_metrics.py) | 8 | Character-tokenization, repeated ngrams, LCS, empty reference, optional ROUGE parity | Stdlib-метрики совпадают с формулами и при доступности с библиотекой |
| [test_server_cli.py](../spanverify-module/tests/test_server_cli.py) | 33 | HTTP endpoints/ошибки/CORS, CLI exit codes/files/JSON, fixtures train/calibrate | Публичные HTTP/CLI контракты проверяются реальными локальными запросами |
| [test_similarity_contrast.py](../spanverify-module/tests/test_similarity_contrast.py) | 5 | Peak/background, margin, вырожденные случаи и знаки | Контраст похожести сохраняет свою математическую интерпретацию |
| [test_span_boundaries.py](../spanverify-module/tests/test_span_boundaries.py) | 7 | Shrink, соседние clauses, реальные подстановки и допустимые offsets | Фрагмент не расширяется произвольно на соседний текст |
| [test_span_filter.py](../spanverify-module/tests/test_span_filter.py) | 8 | Короткая маска, числа/даты, token flags и score_pairs | Фильтрация коротких фрагментов согласована между engine и eval |
| [test_split_and_version.py](../spanverify-module/tests/test_split_and_version.py) | 5 | Group split, контрпример row split, единая версия | Группы не пересекаются; версия согласована с артефактами |
| [test_support_distance.py](../spanverify-module/tests/test_support_distance.py) | 7 | Нормировка расстояния, вырожденные случаи и decay | Дистанция опоры имеет согласованный масштаб и монотонность |
| [test_text.py](../spanverify-module/tests/test_text.py) | 13 | Токены/предложения/абзацы, offsets, дефисы и символьные ngrams | Текст покрывается без дыр, координаты соответствуют оригиналу |
| [test_token_head.py](../spanverify-module/tests/test_token_head.py) | 30 | Feature dimensions/NaN, JSON schema/tree, sklearn parity и threshold edges | Экспорт token-head сохраняет предсказания и fail-closed validation |
| [test_token_mapping.py](../spanverify-module/tests/test_token_mapping.py) | 4 | Character-overlap subwords, средние и пустые пересечения | Признаки модели сопоставляются токенам по координатам |
| [test_vectors.py](../spanverify-module/tests/test_vectors.py) | 9 | Hash determinism, нормы/cosine, соседи и masked density | Векторные признаки конечны и согласованы с ожидаемой геометрией |
| [test_verdict_calibration.py](../spanverify-module/tests/test_verdict_calibration.py) | 15 | Frozen config, selection rule, dev/evidence recount, once-only test и overrides | Вердикт связан с сохранённым правилом; test не используется для выбора |
| [test_verdict_rule.py](../spanverify-module/tests/test_verdict_rule.py) | 8 | Single vs two tokens, decisive attribution, legacy и token parity | Вердикт не обвиняет весь ответ по одиночной слабой маске |
| [test_version.py](../spanverify-module/tests/test_version.py) | 1 | Версия пакета и pyproject | Метаданные выпуска не расходятся |
| [test_workflow_limits.py](../spanverify-module/tests/test_workflow_limits.py) | 158 | YAML, job/step timeouts, quoting, concurrency и push metrics | CI не теряет обязательные проверки и ограничения выполнения |
| [test_realgen_freeze.py](../spanverify-module/tests/test_realgen_freeze.py) | 7 | Квоты, dev-хэш, sealed metadata, background 6510 без test-hosting, сплит и run-id provenance | Обычный CI не открывает sealed test_realgen payload; официальный evaluator fail-closed до run-now и разрешения |

**Итого: 77 строки-файла, 972 nodes.** Инвентарь сохраняет каждый nodeid, включая классы и параметры; документационный тест сверяет его со всей реально собранной pytest session при полном запуске.

## Два штатных skip — без новых исключений

| Реальный nodeid | Причина в базовом JUnit | Что осталось проверяемым |
|---|---|---|
| `tests/test_hf_no_silent_fallback.py::test_hf_features_return_model_backend_when_available` | `torch и transformers не установлены` | Остальные HF negative/fail-closed и window fixtures выполняются; реальный offline Windows HF доказан отдельной сохранённой квитанцией |
| `tests/test_rushallu_metrics.py::test_rouge_matches_rouge_score_library` | `библиотека rouge_score не установлена` | Stdlib ROUGE, n-grams, LCS и пустые случаи выполняются; отсутствует лишь сравнение с необязательной библиотекой |

Это существующие dependency-based skips, а не отключённые тесты и не `-k`/`-m` фильтрация. При установленных optional dependencies эти сравнения должны выполняться, поэтому 2 — характеристика указанного окружения, не обязательное число пропусков во всех средах.

## Границы QA и заморозки

- Полный обычный suite включает синтетические train/calibration fixtures. Они не переобучают production bundle и не выполняют новое измерение official177/B.
- Production, пороги, decoder, span-filter и исторические результаты заморожены. Изменён только существующий тест документационной согласованности: сохранены его прежние assertions, добавлены download/evidence links и соответствие inventory реальной session.
- Контрактные, числовые, provenance и frozen-hash защиты продолжают выполняться без ослабления. Все научные ограничения и процессные исключения остаются в [DEVIATIONS](../reports/DEVIATIONS.md).
- CI требует не просто общего зелёного статуса, но фактического SUCCESS финальной read-only проверки. [Реестр CI](ЗАДАНИЯ_CI.md) · [README](../README.md).
