# P6 — результаты приёмочного контура

**Приёмка продукта целиком не пройдена.** Приёмочный контур выполнен и результаты зафиксированы без изменения замороженной модели. Проверенный код: `14a018a1b90b51cc47cd8baf628147ac22e1d8b1`. Обычный CI **39184437556 SUCCESS**, Windows **39184437541 SUCCESS**, public parity/interfaces **39184437524 SUCCESS**. После этого публикуются только receipts/отчёты; inference/runtime/config не меняются. Официальная оценка test177 не выполнялась (инцидент чтения описан ниже).

## Контроль модели и регрессии

- Frozen P5: **122874 B**, SHA-256 `e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`.
- Public dev replay: 181 пара, 13150 полных token-risk записей, несовпадений 0, P1 raw/calibrated/threshold delta 0, cache misses 0. Все span/token поля сравниваются с сохранённым P5, без переобучения.
- Полный локальный suite: **870 passed, 2 skipped, 0 failures/errors**, 316.054 s, 872 узла. Tests excluded: false. Пропуски прежние: отсутствуют torch/transformers и rouge_score в unit-test окружении; настоящий product HF измеряется отдельно на runner.
- Все 22 исходных testcase найдены и PASSED в новом JUnit: [индивидуальные причины и diff](reports/acceptance/historical-22-fixes.md). Исходные положительные grounded/CLI/числовые проверки сохранены; изменено только устаревшее ограничение sentinel-порога.
- Исторические каталоги P2/P3/P4/P5 побайтно не изменены.

## Native benchmark на GitHub runner

Источник: https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39177834081 . По три повтора, cold-process и warm-model; профилирование — отдельными запросами, не входит в эти цифры. Скачивание HF-весов отдельно от cold latency. Время запуска Python/import не включено; cold включает создание verifier и загрузку HF-модели. Синтетический вход — точное повторение первого train-фрагмента, не репрезентативная выборка реальных 2M документов.

| mode | CPU | logical cores | RAM | Python | дополнительные версии |
|---|---|---:|---|---|---|
| demo | AMD EPYC 9V45 96-Core Processor | 4 | 16373452 kB | 3.11.17 | torch not installed, transformers not installed, sentencepiece not installed |
| hf | AMD EPYC 7763 64-Core Processor | 4 | 16377684 kB | 3.11.17 | torch 2.14.1+cpu, transformers 5.20.1, sentencepiece 0.2.2 |

HF: CPU wheels, OMP/MKL/OpenBLAS threads=2. У каждого режима отдельный фиксированный runner на все его повторы; CPU между режимами различаются, поэтому это не apples-to-apples сравнение моделей.

| mode | вход | знаков | состояние | median s | max s | норматив | результат |
|---|---|---:|---|---:|---:|---:|---|
| demo | document-context | 1000 | cold-process | 0.007070 | 0.013610 | 0.03 s | достигнут |
| demo | document-context | 1000 | warm-model | 0.002000 | 0.002011 | 0.03 s | достигнут |
| demo | document-context | 50000 | cold-process | 0.046811 | 0.046911 | 1.0 s | достигнут |
| demo | document-context | 50000 | warm-model | 0.042196 | 0.042376 | 1.0 s | достигнут |
| demo | document-context | 2000000 | cold-process | 2.309570 | 2.316569 | 60.0 s | достигнут |
| demo | document-context | 2000000 | warm-model | 2.312381 | 2.325358 | 60.0 s | достигнут |
| demo | answer-text | 1000 | cold-process | 0.013408 | 0.013634 | 0.03 s | достигнут |
| demo | answer-text | 1000 | warm-model | 0.008582 | 0.008736 | 0.03 s | достигнут |
| demo | answer-text | 50000 | cold-process | 0.545140 | 0.545219 | 1.0 s | достигнут |
| demo | answer-text | 50000 | warm-model | 0.535732 | 0.574278 | 1.0 s | достигнут |
| demo | answer-text | 2000000 | cold-process | timeout/censored | timeout/censored | 60.0 s | не достигнут |
| demo | answer-text | 2000000 | warm-model | timeout/censored | timeout/censored | 60.0 s | не достигнут |
| hf | document-context | 1000 | cold-process | 5.759504 | 5.766686 | 0.03 s | не достигнут |
| hf | document-context | 1000 | warm-model | 1.469454 | 1.481909 | 0.03 s | не достигнут |
| hf | document-context | 50000 | cold-process | 8.605325 | 8.725292 | 1.0 s | не достигнут |
| hf | document-context | 50000 | warm-model | 4.220389 | 4.262032 | 1.0 s | не достигнут |
| hf | document-context | 2000000 | cold-process | 9.524676 | 9.649385 | 60.0 s | достигнут |
| hf | document-context | 2000000 | warm-model | 5.344216 | 5.349799 | 60.0 s | достигнут |
| hf | answer-text | 1000 | cold-process | 7.681688 | 17.582006 | 0.03 s | не достигнут |
| hf | answer-text | 1000 | warm-model | 3.328763 | 3.345397 | 0.03 s | не достигнут |
| hf | answer-text | 50000 | cold-process | 8.131707 | 8.163428 | 1.0 s | не достигнут |
| hf | answer-text | 50000 | warm-model | 3.664375 | 3.680788 | 1.0 s | не достигнут |
| hf | answer-text | 2000000 | cold-process | timeout/censored | timeout/censored | 60.0 s | не достигнут |
| hf | answer-text | 2000000 | warm-model | timeout/censored | timeout/censored | 60.0 s | не достигнут |

Для 2M answer-text все шесть повторов каждого режима не завершились в process budget 75 s; это цензурированные результаты, не выдуманная медиана 75 s. Кроме того, native HF длинного ответа усечён до 512 модельных позиций: token_count/schema не являются доказательством полноценного HF-покрытия. Даже успешную задержку такого пути нельзя выдавать за соблюдение no-truncation.

### Функциональные 2 млн знаков

- Полный context-документ: ровно 2M Unicode chars, 2022 проиндексированных чанка, 3 обработанных retrieved chunks по замороженной политике; короткий ответ. Это не проверка каждого факта документа.
- Отдельный полный answer-запрос в настоящем sandbox CPU (2 CPU, RAM 4034452 kB, Python 3.11.2): **2000000 chars / 3660605 UTF-8 B**, 282830 token records, последняя координата **2000000**, все full risks конечны и в [0,1], все span slices/schema проверены. **363.414889 s**, норма 60 s не достигнута. Модель та же P5, режим demo; это не HF-результат. Receipt: `reports/acceptance/functional-2M-demo-final.json`; первый полный прогон 349.840719 s сохранён отдельно.
- Native HF full-answer без усечения **не достигнут**: [доказательство и ограничение](reports/acceptance/long-answer-limitation.json). Извлечение признаков/окно/чанкинг не менялись ради красивой скорости.

### Профиль узких мест (отдельный диагностический прогон)

- Demo 2M: `_normalised_support` — около 67.44 s self / 67.76 s cumulative из ~74.95 s прерванного профиля; membership в списке content создаёт квадратичную работу.
- HF 2M: `map_token_positions` — около 63.48 s cumulative; `_span_overlap` вызван **53461856** раз, ~47.09 s cumulative. После лимита 512 сохраняется и проблема непокрытого хвоста.
- Эти цифры относятся к профилю с overhead, не к native скорости. Оптимизация/подмена признаков/zero-fill для улучшения результата не применялись.

## CLI / web 8765 / HTTP API

Реальные CLI и HTTP ответы сопоставлены по score/threshold/verdict/spans/ai_share/ai_participation для demo и hf; web возвращает HTML 200, HTTP — JSON 200, CLI-код согласован с verdict. Запросы/ответы/HTML/логи сохранены в runner receipts. Заключительный интерфейсный replay добавляет обязательный number-attribution кейс со span coordinates/label/confidence; это старый контрактный fixture, не подбор по dev.

## Известные смеси AI-share (никакой калибровки)

| известная доля | demo mean ai_share | hf mean ai_share |
|---:|---:|---:|
| 0.00 | 0.072423 | 0.069847 |
| 0.25 | 0.084185 | 0.163155 |
| 0.50 | 0.069070 | 0.144111 |
| 0.75 | 0.033363 | 0.182326 |
| 1.00 | 0.026797 | 0.080941 |

MAE: demo **0.471802**, hf **0.399863**. Обе средние кривые **немонотонны**. Три train-контекста, точные доли символов 0/.25/.5/.75/1, machine-style блок генерируется программно. Это не real-LLM authorship validation. `ai_share` — недостоверность, `ai_participation` — отдельное поле стиля; они не подменяются друг другом. Результат плохой, новое fitting запрещено и не выполнялось.

## Устойчивость по сохранённым режимам

Сопоставление зафиксировано до отчёта чисел: numeric→number_attribution (132 в полном объявленном корпусе, 92 train); contradiction (144 в полном корпусе, 104 train); wording→partial/unconfirmed только proxy; omission→missing/oversight. Числа полного корпуса — объявленные пользователем/манифестом counts, не чтение test177. Не доказана устойчивость к произвольному перефразированию.

| mode | train OOF token F1 | train OOF strict-span F1 | dev token F1 | dev strict-span F1 | dev answer F1 |
|---|---:|---:|---:|---:|---:|
| number_attribution | 0.480000 | 0.356589 | 0.400000 | 0.333333 | 0.375000 |
| contradiction | 0.387960 | 0.213740 | 0.262295 | 0.153846 | 0.307692 |
| partial | 0.223214 | 0.164948 | 0.064516 | 0.000000 | 0.400000 |
| unconfirmed | 0.892045 | 0.913386 | 0.897727 | 0.903226 | 0.928571 |
| missing | 0.721951 | 0.767442 | 0.830189 | 0.880000 | 0.923077 |
| oversight | 0.947020 | 0.972973 | 0.935065 | 1.000000 | 0.941176 |
| excess | 0.926441 | 0.903226 | 0.915663 | 0.875000 | 0.833333 |
| faithful | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Все counts/FPR/деноминаторы — в `robustness.json`. У faithful нет положительных примеров: F1=0 по прежней метрике не является содержательной оценкой clean-качества; смотреть FP/FPR. OOF answer predictions P5 не сохранены полностью: они не заменяются in-sample или P1-другого-прогона; answer-метрики здесь из saved dev.

## Windows RC / окончательная CI-проверка

Итоговые проверки на одном коде `14a018a`:

- [Обычный ci.yml — **SUCCESS**](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39184437556): quality Ruff/Black, полный suite и coverage **85.06%** (не demo quality P5), Linux/Windows, contract и numbers jobs SUCCESS. Tests collected **872**, tests_excluded=false; новых unit skips нет. Существующие conditional pilot/external/scout jobs не отключались и не выдаются за запущенные.
- [Windows workflow — **SUCCESS**](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39184437541): три тестовые Python-матрицы, onefile/onedir/pyz, HTTP/clean-CWD autonomy smoke, проверка embedded Python DLL SHA и frozen P5 SHA, product registry keys before/after, оба строгих size gate SUCCESS.
- [Public interfaces/parity — **SUCCESS**](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39184437524): demo и hf, CLI/web8765/HTTP200; обязательный nonempty span 46..53 «десять », label=doubtful, risk=.5654, confidence=.4346; сохранены реальные HTTP bodies/annotations/artifact links. В `[interfaces-only]` пропущено повторное измерение уже неизменного benchmark, не unit tests. Publisher не создаёт повторные bot-коммиты.
- [Только prerelease RC, не latest](https://github.com/Igorr112323/Smarty_Chumakov/releases/tag/v1.3.1-rc.p6.39184437541): **exe 8985072 B ≤9000000**, **pyz 227008 B ≤600000**, onedir ZIP **9136772 B**. `.pyz` требует Python ≥3.10, самостоятельный `.exe` содержит Python DLL; ZIP не называется 0.6-MB portable package.
- SHA256 exe: `b8b46a30e7b1775e77fb335afad33f55d58a654826c11e0d3bc206079639b5f7`; pyz: `a447ca798651f78317e55433318fd2f2bc982612afada0071defd3f8fdf901e3`; ZIP: `49383d0bd7b2613623bf0f3639504b8b4eae6cf4434aa6f71e07f3bf7c8c5457`. Полный SHA256SUMS опубликован с RC и зеркалируется в `RC-SHA256SUMS.txt`; digest взят из GitHub release API, а не придуман из имени файла.
- Фикс CRLF: `.gitattributes -text` оставляет frozen evidence побайтно исходным. Training fallback запрещён, production auto-train не выполнялся. Неиспользуемые server exe train/evaluate корпуса вынесены из exe; модель/config не удалялись, DLL не стрипились.

Portable copy/update: остановить процесс, проверить SHA256SUMS, заменить полный exe/onedir одним RC целиком; не смешивать config/runtime разных поставок. Frozen `frozen-p5.json` и `p5-runtime.json` включены в RC. Конкретные bytes/hash/runtime/env и JUnit находятся в `reports/acceptance/`; `manifest.json` инвентаризирует артефакты.

HF внешние веса ~530 MB и библиотеки ~500 MB не включены в маленький exe. Маленький default exe — demo; frozen-p5.json/runtime передаются отдельно. Windows HF autonomy с конкретной внешней раскладкой не подтверждена и не выдаётся за измеренный факт. Продуктовые registry keys проверяются до/после; это не глобальный OS registry trace.

## Инцидент изоляции

Первый остановленный regression вызвал старый chunking test, который парсил весь corpus_a3/pairs.jsonl перед срезом 20 строк. Это нарушение запрета чтения, хотя официальная model evaluation/test177 не запускалась и выбора модели по этим данным не было. После исправления тест использует train, audit hook запрещает открывать реальный test177 и общий corpus_a3/pairs.jsonl. Отрицать инцидент было бы неверно. Следующий официальный прогон всё ещё должен быть отдельным санкционированным этапом; в P6 его не запускали.

## Итог

Локальные 22 регрессионные причины устранены, frozen P5 public parity=0, приёмочные измерения опубликованы без завышения статусов. Скорость HF, full native HF long-answer без усечения и калибровка ai_share не достигнуты; полный контракт **не принят**. Матрица: `reports/acceptance/coverage-matrix.{json,md}`; файл hashes/sizes: `reports/acceptance/manifest.json`.


### Отклонённый промежуточный RC на 9d576a8

CI `39180292854` и Windows workflow `39180292839` завершились SUCCESS, но exe **34096674 B** превысил требуемые **9000000 B**. Старый size gate допускал OR с маленьким pyz; P6 требует оба ограничения. RC помечен REJECTED, остаётся prerelease и не становится latest. Тестовые fit-зависимости теперь сохранены только в отдельных test jobs, исключены из packaging job; numpy/sklearn/scipy исключены из exe; size gate требует оба ограничения. Добавлена regression-проверка этой комплектации: окончательный suite собирает **872** узла, его результат должен подтверждаться новым JUnit.

Функциональный повтор на `9d576a8`: **363.414889 s**, 282830 records, последний end=2000000; hash всех token records совпал с первым прогоном: `5e68bd503774557c878ffcbb25a182f3c829dff42b2fe711615f64a73a2fcfda`. Норма 60 s не достигнута. Дерево inference-runtime при исправлении упаковки не меняется.


Следующий strict RC run `39181817291` корректно остановился на **9044022 B** (превышение 44022 B); autonomy и HTTP smoke успешны, релиз **не опубликован**. Ненужные server-launcher демонстрационные train/evaluate корпуса (≈1 MB исходных данных, ≈59 KB сжато) вынесены из exe/onedir, без удаления config/моделей или стрипа DLL. Portable CLI по-прежнему содержит свой demo_pairs. Отдельный regression удерживает эту комплектацию. На `ef39a09` обычный CI `39181817272` SUCCESS, локально **870 passed / 2 skipped / 872** за **314.489 s**, все исходные 22 testcase PASSED ([индивидуальные причины и diff](reports/acceptance/historical-22-fixes.md)). Эта проверка завершена; заключительные run links приведены выше.


## Заключительная публикация

Окончательный HEAD `14a018a` — receipt-only публикация, после неё повторены полный локальный suite (870 passed / 2 прежних skips / 872, 316.054 s), обычный CI, actual public parity/interfaces и fresh Windows RC. Все три workflows SUCCESS на этом точном SHA. Итоговые машинные receipts обновлены после завершения workflows; они не являются новым inference-кодом. Публичные отчёты на GitHub содержат предыдущий уже проверенный code-equivalent snapshot `6926a80`; текущие receipt/JUnit/manifest также включены в выдаваемый архив и итоговые RC notes. Не утверждается clean working tree: post-run отчётные файлы обновлены без нового изменения модели/кода.


Дополнительное прикрепление сводного ZIP/его SHA256/MD к релизу через `uploads.github.com` завершилось **EOF**; ни один из этих трёх дополнительных assets не загружен. Шесть штатных RC-assets (exe, pyz, onedir ZIP, frozen model, runtime config, SHA256SUMS) опубликованы и не затронуты. Полный финальный сводный ZIP и его SHA256 предоставляются в Arena workspace (`reports/p6-acceptance-14a018a.zip`); текущий отчёт также опубликован как RC notes, а текущие JUnit/interfaces — в successful Actions artifacts. Это ограничение дополнительной доставки, не успешная загрузка, и не повод поручать пользователю команды.
