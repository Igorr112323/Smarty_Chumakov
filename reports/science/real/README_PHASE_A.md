# CORPUS-REAL-R1 — Фаза A: источники и заморозка сырого корпуса

**Ветка:** `science/corpus-real-a` от `main 4fff418` (bundle `e0b13c69…` read-only)  
**Статус:** ЗАМОРОЗКА (обновлён после смены маршрута) — источник `LawCodex/ru` pin `5806f92`  
**PR:** #16 `science/corpus-real-a` → `main` (per_phase_pr)

## Источник и лицензия (честный provenance)
- **Источник:** `git clone https://github.com/LawCodex/ru` **pin `@5806f92` `5806f92e6bd2b6802b080de3c71d85ed84d11543`** (actual HEAD при клонировании зафиксирован в манифесте) — 11 770 актов Markdown, снятых сторонним проектом **LawCodex** с официального **actual.pravo.gov.ru** (раздел «Тексты правовых актов с внесёнными изменениями», API `ebpi`) **2026-10-12**. Маскировка под «мы сами скачали с pravo» — **запрещена**, в отчёте provenance честный.
- **Лицензия:** Госакты не объекты авторских прав — **ст.1259 ГК РФ п.6 пп.1** (зафиксировано в `CORPUS_MANIFEST.json`); портал `robots.txt` не ограничивает, условия reuse не найдены; conversion code `scripts/` — CC0
- **Фолбэк:** `irlcode/RusLawOD` — только отдельным стоп-сообщением, пока не использован

## Сбор (выполнено, без детекторных API)
- **≥1600 документов (запас):** после фильтров **7710** документов (из 11 770 актов, FKZ 171 + ФЗ 7378 + Кодексы 25→117 чанков + Конституция1→2 + Законы РФ/РСФСР 149 → после split и дедупа)
- **Фильтры:** длины **2–50к знаков** (минимум 1500 исключён — 3163 отсеяно, >50к исключено — 1011; 10% ≥20к выполнено: **1153/7710 =14.95% ≥20к**, 995 <2к, 2763 2–5к, 2799 5–20к)
- **Дедуп:** по `sha256(норм_текст)`, UTF-8, без бинарей/сканов/обрезок <1500, без web-поисковых API — только скачивание + локальная обработка
- **Кодексы по главам:** опция манифеста **вкл** (`kodeksy_by_chapters_option:true`, `kodeksy_split: ## Главам`): исходный кодекс 0.8–10 MB >50к → split по `## `, иначе потеря типа; `konstitutsiya` аналогично (242k→2 дока)

## Заморозка
- **`reports/science/real/CORPUS_MANIFEST.json`** — на документ: `url` (official viewer `actual.pravo.gov.ru/content/content.html#hash`), `act_name`, `date`, `last_amended`, `identifier`, `kind`, `sha256`, `size` (+ `path`, `type`, `retrieved:2026-10-12`); общий `manifest_sha256: 1d076c90f5af…` и `manifest_sha256_of_entries`; `pin:5806f92`, `pin_full:5806f92e6bd2b…`, `actual_head_sha` зафиксирован
- **`reports/science/real/corpus_raw.jsonl` (145 MB) + `.gz` (20 MB):** компактные JSONL `{id,text,meta}` — коммитится `CORPUS_MANIFEST.json` (10 MB) + `corpus_raw.jsonl.gz` (20 MB, под 128 MB cap), полный `.jsonl` — внешний артефакт (воспроизводится `git clone LawCodex/ru@5806f92` + скрипт `build_manifest_v3.py`)
- **Состав типов (фактический, не изобретать):** `code:117`, `constitution:2`, `constitutional_amendment:6`, `federal_constitutional_law:117`, `federal_law:7378`, `law:90` (171 ФКЗ + 7378 ФЗ + 117 кодексов_чанков + 2 конституция + 90 законы_РФ; указов/постановлений нет — как в источнике)
- **Suite ≥913, пломба P8 зелёная, bundle read-only, main не тронут напрямую**

## Причинат-коды фазы B (на подтверждение, `КОРПУС_v1.md` показан)
8 режимов из `spanverify-module/docs/КОРПУС_v1.md` §1 (B внешний 423 спана: Partial125/Unconfirmed116/Contradiction68/Missing56/Oversight36/Excess22):
- faithful (чистый ≥40%)
- contradiction (значение заменено)
- partial (отброшено условие)
- number_attribution (число другого объекта)
- unconfirmed (добавлено без источника)
- missing (пропущена деталь)
- oversight (потеряно при пересказе)
- excess (лишняя деталь)
→ Ожидаем подтверждения 10 из PDF/JSONL corpus B (показан файл, опция `confirm_file`).

## Кураторский спотчек
20 случайных актов — независимо куратором (из `CORPUS_MANIFEST.json` entries).

## Дальше
Ждём команды на Фазу B (`science/corpus-real-b` от фазы A): детерминированная инъекция 1–3 спанов/пару, 50/50 чистые/с-инъекциями, split-by-doc `real_dev≥300 / real_test≥1200` как corpus B, заморозка `CORPUS_FROZEN.json` + 5% curатор-список doc_id.
