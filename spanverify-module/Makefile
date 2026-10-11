# Обвязка единственной команды каждого шага цикла v2.
#
# Файл нужен ровно для одного: чтобы «замороженная команда» из config/hf_final_config_v2.json
# и из docs/EXPERIMENTS_v2.md была одной и той же командой, а не двумя похожими. Числа в
# отчёты вносятся только из артефактов этой обвязки.
PYTHON ?= python
GRID ?= reports/v2-grid
GRID_BASE ?= reports/v2-grid-base
SPLITS ?= data/corpus_v2/splits
SEEDS ?= 42,43,44,45,46
OUT ?= reports/v2

.PHONY: hf-grid-v2 hf-freeze-v2 hf-curve-v2 hf-sweep-v2 hf-seedcheck-v2 hf-freeze-config-v2 hf-final-v2 hf-verify-v2 corpus-v2-build

## Шаг 0: аудит разбиения новой выборки + фиксация правил (до любых чисел)
hf-freeze-v2:
	$(PYTHON) scripts/hf_protocol_v2.py freeze --splits $(SPLITS) --config config/hf_experiments_v2.json --seeds $(SEEDS) --out $(OUT)

## Предвычисление сетки признаков (одна модель = один каталог: провенанс модели обязан быть однозначным)
hf-grid-v2:
	$(PYTHON) scripts/precompute_features.py --grid --splits $(SPLITS) --out $(GRID)

hf-grid-base-v2:
	$(PYTHON) scripts/precompute_features.py --grid --model ai-forever/rugpt3base --splits $(SPLITS) --out $(GRID_BASE)

## Шаг 1: кривые F1/FPR на val и оба правила порога (test не читается)
hf-curve-v2:
	$(PYTHON) scripts/hf_protocol_v2.py curve --config config/hf_experiments_v2.json --grid-cache $(GRID) --splits $(SPLITS) --out $(OUT)

## Шаг 2: десять гипотез, отбор по val, стоп-правило
hf-sweep-v2:
	$(PYTHON) scripts/hf_protocol_v2.py sweep --config config/hf_experiments_v2.json --grid-cache $(GRID) --splits $(SPLITS) --out $(OUT)

## Проверка, что seed влияет на обучение (std > 0) — обязательна перед шагом 3
hf-seedcheck-v2:
	$(PYTHON) scripts/hf_protocol_v2.py seedcheck --config config/hf_experiments_v2.json --grid-cache $(GRID) --splits $(SPLITS) --seeds $(SEEDS) --out $(OUT)

## Заморозка конфигурации и порога до единственного прогона на test
hf-freeze-config-v2:
	$(PYTHON) scripts/freeze_final_config_v2.py --sweep $(OUT)/sweep.json --freeze $(OUT)/freeze.json --seedcheck $(OUT)/seedcheck.json

## Шаг 3: ровно один прогон на test (команда обязана совпадать с final_command из конфига)
hf-final-v2:
	$(PYTHON) scripts/hf_protocol_v2.py final --config config/hf_final_config_v2.json --grid-cache $(GRID) --splits $(SPLITS) --corpus-name v2 --seeds $(SEEDS) --include-val --freeze $(OUT)/freeze.json --sweep $(OUT)/sweep.json --seedcheck $(OUT)/seedcheck.json --out $(OUT)

## Пересчёт каждого числа манифеста из строк предсказаний
hf-verify-v2:
	$(PYTHON) scripts/hf_protocol_v2.py verify-manifest --manifest $(OUT)/manifest.json --check-splits

## Пары, разбиение и манифест корпуса по скачанным текстам
corpus-v2-build:
	$(PYTHON) scripts/build_corpus_a.py --real --docs data/corpus_v2/sources --out data/corpus_v2 --target 4280 --seed 2026
