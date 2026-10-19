# P7 — измеренный инженерный результат

19.10.2026. **Инженерная работа и измерения выполнены; весь договор не принят.**
HF50k warm≤1s не достигнут, AI-share-калибровка частичная, официальный test177
не оценивался. Успех workflows не означает достижения всех численных нормативов.

Baseline `14a018a1b90b51cc47cd8baf628147ac22e1d8b1`.
Предиктивная реализация после индекса: `7262f09`; проверенная Windows-поставка
`a85c40a14828141447768d3c78d13aa0ced105eb`. Последующие изменения —
измерительная/документарная инфраструктура, не признаки/модель качества.

## Доставка и прямые доказательства

- [Научная приёмка/benchmark39236657451 — SUCCESS](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39236657451).
- [Full CI895tests b8aa671 — SUCCESS](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39239481352).
- [Второй обычный CI — SUCCESS](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39239475878).
- [Windows39236657579 — SUCCESS](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39236657579).
- [Same-host before/after39232439363 — SUCCESS](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39232439363).
- [Новый RC, prerelease/not latest](https://github.com/Igorr112323/Smarty_Chumakov/releases/tag/v1.3.1-rc.p7.39236657579).
- [Проверенная полная portable HF поставка](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39236657579/artifacts/12180458356), artifact-container637051335B.
- [Исходный архив приёмки](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39236657451/artifacts/12180029914).
- [Финальный архив с отчётом и committed-квитанциями](https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39241808393/artifacts/12181377477), publication39241808393 **SUCCESS**, 4762168B artifact-container.
- [Полный diff от baseline](https://github.com/Igorr112323/Smarty_Chumakov/compare/14a018a1b90b51cc47cd8baf628147ac22e1d8b1...arena%2Fc5f69767-smarty-chumakov).

Финальная artifact-only публикация проверяет SUCCESS исходных runs, неизменность
кода/настроек/train/dev/folds/frozen bundle относительно измеренной реализации,
SHA исходного ZIP, сохраняет raw evidence и добавляет финальные документы.
Она не выдаётся за новый benchmark и не заменяет его. ZIP передаётся через
workflow artifacts, не uploads.github.com release-ZIP. Квитанции: `final-receipts/`.
Stable/latest проверен отдельно: **v1.3.1**, без продвижения RC.

## 1. Заморожено и изменено

Bundle **122874B**, SHA256
`e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892`.
RC asset digest совпадает, файл скопирован байт-в-байт, не перестроен.
P1/P4/span_filter/decoder, коэффициенты, порог **0.5794392523364486**,
исходные folds/корпус/IoU≥.5/greedy/scored predicate/сглаживание неизменны.
Нового обучения модели качества/подбора конфигурации нет. Существующие временные
synthetic/demo fit в тестах сохранены по явному разрешению пользователя.
Единственный новый fit — отдельная train-only PAVA.

Изменения исполнения: индексирование token-position overlaps, кеширование норм,
устранение sentence-prefix hot spots, сохранение eval HF runtime, точный
request-local LRU4096 одинаковых окон, индекс интервалов в двух разных
предикатах (strict overlap / containment начала токена). Ни токены, ни логические
окна не пропускаются; никаких нулевых хвостов/подмен рисков/ослабления правил.
Python≥3.12 сохраняет reference summation, без обещания такого же ускорения.
100000synthetic interval queries совпали с reference; это отдельно от native SLO.

Согласованное исключение: **только >512 model-position ответы** получают
`window512_full_answer_v1` и могут изменить результат. Политика top3context
не изменена; это не обещание Transformer-проверки каждого факта всего документа.
Исторические P2/P3/P4/P5 артефакты/метрики не переписаны.

## 2. Паритет и аудит усечения

Byte golden291747B SHA256
`8e4a06666ef6e854dd6a768d7cc253ec368201cd74ab5485e3693d4955186edd`;
исключён **только latency_ms**, исходный снимок сохранён.
Перед каждой предиктивной оптимизацией — golden и saved-dev replay181/13150,
zero mismatch/P1delta0. Cached replay обозначен cached compatibility,
**не** свежая проверка feature path.

Отдельный реальный old/new HF replay подтвердил byte identity **178коротких**
dev-ответов. Tokenizer audit: train842, длинных22, max1923model positions,
legacy unmapped7250native tokens; dev181, длинных3, max1046, unmapped479.
В этих train/dev коротких ответах пропусков нет. Исключение не расширялось на
иные короткие входы: short path остался legacy-byte-frozen.

| Dev ID | Model positions | Legacy score | Full-window score | Changed records |
|---|---:|---:|---:|---:|
| corpus-a3-00265 |529|0.5188679245283019|0.5794392523364486|418|
| corpus-a3-00114 |1046|0.3969465648854962|0.5794392523364486|623|
| corpus-a3-01041 |641|0.5188679245283019|0.5794392523364486|531|

Все3gold-positive; при неизменном `score>=threshold` это FN→TP.
Исторические P5 answer TP/FP/FN/TN **49/14/47/71** сохраняются.
Версионированный long-v1 audit даёт **52/14/44/71**, не новую независимую оценку
качества и не повод выбирать порог. Подробные hashes/deltas:
`versioned-long-deltas.json`, `accepted-long-contract-annotations.json`,
новые `final-receipts/acceptance/*annotations.json` и workflow HF-contract artifact.

## 3. Полный HF2M и скорость

Actual2M UTF8 answer: **2000000chars/3660605B/282830native token records**,
последний end2000000; все риски finite/bounded, schema/text slices проверены,
все native tokens отображены.3283логических окна,14model calls,
3269точных cache hits; ни одного пропуска логического окна.
SHA всех token records:
`56c3028861b5ea021310a59f98272fa1bff6e33a43755af6fae0bc00f89655a4`.

До кеша один functional прогон1738.994582s, после кеша до индекса125.521545s —
оба **не** достигли60s. На итоговом source реальная SAME-PROCESS проверка
индекса дала **91.417021→39.301874s**, полный public JSON byte-identical после
удаления только latency_ms, SHA
`e86ac2ade7e182d6f6fb03bce53f81680b92f0a3715804bc39d13d833a1ebd35`.
Это отдельная functional equality проверка, не замена трём native repeats.

Ниже current native3cold/3warm (median/max), только train/synthetic.
Latency — вызов verifier по протоколу P6, без download/import, не сеть/браузер
и не время скачивания огромного JSON. Warm — прогрев модели запросом1k,
не кеширование измеряемого полного ответа. Timeout75s относится ко всему worker,
не притворяется успешной медианой. Все24cells/sec1000/limits:
`final-receipts/native-benchmark.json`.

| mode | answer chars | cold median/max s | warm median/max s | limit s | статус |
|---|---:|---:|---:|---:|---|
|demo|1000|0.024435/0.024529|0.015931/0.016051|0.03|достигнут|
|demo|50000|0.706259/0.712470|0.703229/0.704727|1.0|достигнут|
|demo|2000000|29.950421/29.995070|29.876726/30.019292|60.0|достигнут|
|hf|1000|5.343224/5.413219|0.811025/0.814359|0.03|не достигнут|
|hf|50000|12.672036/12.702179|8.168092/8.213496|1.0|не достигнут|
|hf|2000000|42.878149/42.953376|38.769417/38.821635|60.0|достигнут|

Current demo CPU7763, HF9V74;4logical CPUs/16GB-class RAM, Ubuntu24.04,
Python3.11.17; torch2.14.1+cpu/transformers5.20.1/numpy2.4.6/sentencepiece.2.2,
OMP/MKL/OpenBLAS2threads. Это не та же физическая VM, что historical P6.
Поэтому [TIMINGS.md](TIMINGS.md) отдельно публикует48same-host before/after cells:
P6-программа14a018a и P7f7bb718 на одной VM каждого режима (CPU7763).
HF-before50k усекался и не является like-for-like полным baseline.

**Достигнуты:** demo answer1k≤.03/50k≤1/2M≤60s, HF answer2M≤60s без timeout;
demo context нормативы сохранены, HF context2M≤60 и ускорение на paired control.
**Не достигнуты:** HF answer50k warm≤1s (**8.168092median/8.213496max s**),
HF малые context/1k нормативы. Demo context небольшие колебания медиан не скрыты:
2Mwarm+0.42%,50kwarm+.78%,1kcold+.913ms на paired VM. Строгая статистическая
неухудшенность тремя повторами не доказана; заявляется сохранение нормативов.

Новый failed-norm профиль опубликован (`native-profiles.json` + raw `.prof`
workflow artifacts). В paired HF50kwarm: torch.addmm3.282463self-s,
linear1.611283self-s, tree predict_one.746631self-s, cumsum.596300self-s.
Основной остаток — замороженное dense inference; риски/порог ради1s не менялись.
Повторяемый train-вход — best case кеша. Универсальные60s на уникальном2M тексте
не доказаны и не заявлены. Нормативы не переписаны.

## 4. AI-share — отдельный частичный артефакт

[Полные кривые/MAE/метод](CALIBRATION.md), `ai-share-oof-curves.json`.
265train-documents/1325смесей0/.25/.5/.75/1, исходные5document folds,
fit/held PAVA-documents disjoint. Raw frozen; dev/test не использованы.

| mode | pooled OOF MAE до | после | pooled curve monotone |
|---|---:|---:|---|
|demo|0.4497321882850523|0.2989009919597126|нет/нет|
|HF|0.4068236540115673|0.2565365102059551|нет/нет|

Функция PAVA монотонна по raw, pooled-кривая по известной доле не монотонна.
**Статус частично**, не калибровка реального LLM-авторства. Отдельные
`ai_share_calibrator_v1.demo.json/.hf.json` в `final-receipts/`, их SHA проверены.
Opt-in через constructor/`SPANVERIFY_AI_SHARE_CALIBRATOR`, default raw сохранён,
P1 не меняется; несовпадение mode/bundle отвергается. OOF относится к PAVA,
не новой nested валидации ранее замороженной модели качества.

## 5. Windows, размеры и реальная автономность

Exe **8979956B≤9000000**, pyz **232874B≤600000**. Native exe И pyz выполнили
реальный `/v1/verify` HTTP200/backendhf в изолированной копии, с внешними libs,
весами и portable Python, без source checkout/host development site-packages,
с исходящей firewall-блокировкой. Проверки application registry keys неизменны;
это не глобальная трассировка каждого системного ключа. Поставка не требует
пользовательского pip/install/регистрации. Файлы `Start-HF.exe.cmd` /
`Start-HF.pyz.cmd`, `AUTONOMY.json` и полный layout входят в portable ZIP.

Фактический model `pytorch_model.bin` **551290714B** (~525.75MiB), SHA256
`0d2a4a387b0430654bb1f6813436bd31f8e10a06e18ea0e5c41b33120b78458e`.
**libs870588148B** (~830.26MiB), не первоначальная оценка~500MB;
portable Python и приложения дополнительно. Model+libs уже1421878862B.
Не выдаём размер tiny exe за полную HF-поставку. Artifact-container637051335B;
внутренний `p7-hf-portable.zip` SHA256
`c5cd4a93952b9016b8a00748eef8d1b6f2ce2d23f2d6069f7c91185e1dbb5771`.
Полный installed-byte total SDK не измерен, конкретная эталонная4GB-машина
договора не испытывалась — соответствующие строки остаются частичными.

Предыдущие ошибки не скрыты: `/health` был успешным, HF verify500 сначала
`urllib.error`, затем native `tokenizers` DLL loading. Исправлены search paths
частично frozen stdlib packages и portable ABI3/MSVC DLL directories; handles
живут до конца serve. PE Windows-wheel подтвердил зависимость `python3.dll`.
Lossless PYZ zlib6→9 уменьшил размер без удаления bytecode/assertions.
Новые реальные HF200, а не один health, закрывают автономность этой поставки.

## 6. Legacy-инцидент чтения test

На начальном остановленном P6 regression legacy
`tests/test_chunking.py::TestSelectRelevantChunks::test_on_corpus_a3_sample`
разбирал **весь** unsplit corpus_a3/pairs.jsonl, затем срезал первые20.
Следовательно, зарезервированные строки **читались**. Точный timestamp чтения
не восстановлен; исправление5ff79f9a7bed33c00c6278f67d68ddff73f8325c:
18.10.2026в23:34:59UTC (19.10в02:34:59MSK), переход на train split.
Нет утверждения «test никогда не читался». Официальных метрик/обучения/выбора
конфигурации на test не было; frozen bytes/history/replays сохранены.
Guard conftest + P7 runners блокирует aggregate/official reads. Новых чтений
официальных177 в P7 нет. Исторический source-pairs SHA (без нового открытия файла):
`b9a4cf05a9078644dd539470481a1b8079aed2d9201e195c3ecd901276ff7c8e`.
Доказательство: `freeze-and-incident.json`, source diff5ff79f9.

## 7. Тесты, устойчивость, оставшиеся ограничения

Текущая коллекция **895**. Полный локальный b8aa671: **893passed/2skipped,
327.96s, exit0**; skip прежние missing torch/transformers и rouge_score.
Новых исключений/skip нет, tests_excluded=false. Реальный HF проверен отдельными
успешными jobs. Black191files/Ruff/golden и все workflow guards сохранены.
Промежуточные failures тоже сохранены:7262fullsuite rejected60min>40guard;
4fd81b4fullsuite rejected несинхронизированный workflow registry. Исправлены
конфигурация/документация, не тесты; последующий полный895прогон успешен.

Новый publication workflow добавил5существующих параметризованных checks и
2ZIP integrity/path-escape regressions:888→895; это inventory, не подмена
исторических успешных888JUnit. Edge regressions empty/single/Unicode seams/
long docs, отказ при неполном HF mapping, интервальные границы сохранены.

Robustness number/contradiction/paraphrase/omission: **исследовано и
задокументировано, не «качество достигнуто»**. P6 saved OOF/dev mode proxies,
не новые perturbation tests: strict dev span F1 contradiction.1538,
number_attribution.3333, partial0; paraphrase не имеет отдельной gold-mode.
Новых независимых challenge-метрик не выдумываем. Источник:
`reports/acceptance/robustness.json`. Официальный test177 остаётся неоценённым.

[Матрица P7](coverage-matrix.md), [текущая матрица договора](../contract/coverage-matrix.md).
Подняты только строки с новыми источниками; quality/history/неисследованная
hardware/usability/updates приёмка не повышены. Незакрыто: HF50k≤1s и малые
HF-speed нормы; надёжная authorship-калибровка; соответствие прежней оценке
libs~500MB/полный установленный SDK footprint; весь официальный quality-contract.
Это конечный измеренный частичный результат, без обещания отдельной будущей итерации.

Квитанции всех измерений/RC/архивов сохраняются в Git. Финальный documentary HEAD
проходит обычный полный CI; ссылка на его run проверяется после push и сообщается
в результате (самоссылочный SHA в собственном коммите не создаётся).

## Квитанция финальной публикации

Архив документации source`b5eb133` опубликован workflow39241808393, SUCCESS.
SHA256 внутреннего `p7-final-acceptance.zip`: `03eaa6e0379c30a9db4d23da7f0ea0cbd0a66a57ac7952bbeca2064fcbc0f143`.
Artifact-container digest отличается, указан в `final-receipts/published-archive.json`.
Эта post-run квитанция добавлена после публикации, не маскируется как часть
собственного ZIP. Предиктивный код/поставка/измерения не изменены.
