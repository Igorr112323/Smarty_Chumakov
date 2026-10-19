# Внешний HF-runtime: границы поставки и запуск

Дата: 2026-10-19. Это инструкция к уже проверенной поставке, не команда нового обучения или сборки.

## Что входит и что нужно отдельно

Маленькие `spanverify.exe` и `spanverify.pyz` из [stable](https://github.com/Igorr112323/Smarty_Chumakov/releases/tag/p8-stable-20261019) сами по себе запускают **demo**. Для научного HF-режима нужны:

- каталог модели `ai-forever/rugpt3small_based_on_gpt2`: **554 277 308 байт (554.3 MB)**;
- внешние библиотеки: **870 588 148 байт (870.6 MB)**;
- совместимый Python/runtime и, для полностью переносимого Windows-варианта, app-local DLL;
- неизменённые `frozen-p5.json` и `p5-runtime.json` из того же релиза.

Отдельный файл весов занимает 551 290 714 байт; это не размер всего каталога модели. Полный размер SDK неизвестен. **Готовый HF-runtime не является девятым release asset:** опубликованы восемь исходных RC-файлов. Подготовку внешних зависимостей нельзя выдавать за запуск одного EXE без дополнительных файлов.

## Вариант с установленным Python

В отдельном окружении установите зависимости модуля и `spanverify-module/requirements-hf.txt` (нужен доступ к источникам пакетов), заранее подготовьте локальный каталог модели. Скачайте из stable оба JSON: [frozen-p5.json](https://github.com/Igorr112323/Smarty_Chumakov/releases/download/p8-stable-20261019/frozen-p5.json) и [p5-runtime.json](https://github.com/Igorr112323/Smarty_Chumakov/releases/download/p8-stable-20261019/p5-runtime.json). Они должны находиться рядом, без редактирования.

Для PYZ задайте локальный каталог модели и offline-режим, затем запустите (PowerShell):

```powershell
$env:SPANVERIFY_HF_MODEL = (Resolve-Path .\model).Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python .\spanverify.pyz server --mode hf --weights .\frozen-p5.json --port 8765
```

Вместо `python` можно использовать подходящий переносимый `python\python.exe` вместе с уже подготовленными совместимыми библиотеками. Команды не загружают веса автоматически в offline-режиме; неполный каталог или отсутствующие зависимости приводят к явной ошибке, а не к скрытому переходу на demo.

## Проверенный Windows portable layout

```text
HF-folder/
  spanverify.exe
  spanverify.pyz
  frozen-p5.json
  p5-runtime.json
  model/                 # локальная модель и tokenizer
  libs/                  # совместимые Python/HF-библиотеки
  python/python.exe      # совместимый portable Python с Lib/ и DLLs/
```

Из этой папки, после подготовки layout, пример EXE-запуска:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\spanverify.exe --mode hf --model .\model --weights .\frozen-p5.json --runtime-dir .\libs --port 8765
```

В браузере откройте **http://localhost:8765** (либо фактический свободный порт, который сообщил launcher). Проверьте `mode: hf` и `stats.features.backend: hf` в ответе HTTP API. Просто изменение подписи интерфейса не является доказательством HF.

## Уже сохранённые доказательства

- [Windows HF autonomy receipt](../../reports/p8/run39271014161/hf-autonomy/windows-hf-autonomy.json): offline-проверки EXE/PYZ, layout и размеры библиотек.
- [Запрос](../../reports/p8/run39271014161/hf-autonomy/exe-request.json) и [реальный ответ EXE](../../reports/p8/run39271014161/hf-autonomy/exe-response.json).
- [Footprint receipt](../../reports/p8/DELIVERY.json): объём модели и границы измерения.
- [Stable receipt](../../reports/p8/closure/STABLE_RECEIPT.json): восемь файлов без пересборки и их SHA-256.
- [Паспорт](../../reports/ACCEPTANCE_PASSPORT.md) и [ограничения](../../reports/DEVIATIONS.md).

Проверенные размеры относятся к зафиксированному Windows layout. Другая версия Python/torch или другая платформа может дать иной размер и требует отдельной инженерной проверки, а не изменения опубликованных чисел.
