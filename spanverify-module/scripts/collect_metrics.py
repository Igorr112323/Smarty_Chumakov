"""Единый источник чисел проекта: ``reports/METRICS.json``.

Правило проекта (после аудита): любое число в документах берётся отсюда, а не из
памяти. Скрипт собирает:

* версию, коммит, seed, дату;
* число собранных тестов и покрытие (из ``reports/coverage.json``, если CI его создал);
* метрики демо-корпуса на **групповом** разделении (обучение и сквозная проверка);
* оценку доли участия ИИ (AUC вне выборки) — калибровка на синтетике;
* кросс-корпусный тест из ``reports/cross_corpus.json`` (создаёт ``scripts/cross_corpus.py``);
* числа пилота из ``reports/pilot/pilot.json`` (создаёт ``scripts/pilot_rugpt3small.py``);
* размеры и SHA256 артефактов релиза, если они лежат в ``release/``.

Запуск::

    python scripts/collect_metrics.py --out reports/METRICS.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spanverify import __version__  # noqa: E402
from spanverify.dataset import read_pairs  # noqa: E402
from spanverify.engine import Verifier  # noqa: E402
from spanverify.features import number_attribution  # noqa: E402
from spanverify.train import train  # noqa: E402

DISCLAIMER = (
    "Демонстрационные метрики получены на синтетическом корпусе и научным "
    "результатом не являются. Для подтверждения требуются 1200 пар "
    "«вопрос–ответ» с экспертной разметкой на реальных регламентах организации."
)

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent


def _git_commit() -> str:
    """Текущий коммит (для трассировки чисел к состоянию кода)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=20
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _test_count() -> int | None:
    """Сколько тестов собирает pytest (быстрая команда, без запуска тестов)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    for line in result.stdout.splitlines():
        parts = line.rsplit(": ", 1)
        if len(parts) == 2 and parts[1].strip().isdigit():
            total += int(parts[1])
    return total or None


def _coverage_percent(path: Path) -> float | None:
    """Покрытие в процентах из отчёта coverage.json (если он есть)."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return round(float(payload["totals"]["percent_covered"]), 2)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _sha256(path: Path) -> str:
    """SHA256 файла артефакта."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_info(release_dir: Path) -> dict | None:
    """Размеры и контрольные суммы артефактов релиза, если они собраны рядом."""
    artifacts = []
    for name in ("spanverify.exe", "spanverify-win.zip", "spanverify.pyz", "SHA256SUMS.txt"):
        path = release_dir / name
        if path.is_file():
            artifacts.append(
                {
                    "name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"dir": str(release_dir), "artifacts": artifacts} if artifacts else None


def _attribution_block(pairs: list[dict], verifier: Verifier) -> dict:
    """Правило привязки числа к объекту: кейсы дефекта D и ложные срабатывания.

    Числа считаются тем же кодом, что обслуживает API (`number_attribution` и
    `Verifier.verify`), поэтому попадают в документы, а не берутся из отчёта.
    """
    context_two = (
        "Согласно регламенту, срок хранения первичных документов составляет пять лет. "
        "Срок хранения вторичных документов составляет десять лет."
    )
    borrowed = verifier.verify("Срок хранения первичных документов составляет десять лет.", context_two)
    own = verifier.verify("Срок хранения первичных документов составляет пять лет.", context_two)
    clean = [pair for pair in pairs if not any(int(label[2]) == 1 for label in pair.get("labels", []))]
    flagged_clean = [pair["id"] for pair in clean if number_attribution(pair["answer"], pair["context"])]
    return {
        "borrowed_number_verdict": borrowed.verdict,
        "borrowed_number_spans": [span.text for span in borrowed.spans],
        "own_number_verdict": own.verdict,
        "clean_pairs": len(clean),
        "false_positives": len(flagged_clean),
        "false_positive_ids": flagged_clean[:10],
        "rule": (
            "число ответа сверяется с измерениями того же объекта документа; "
            "если оно взято у другого объекта — фрагмент doubtful независимо от головы"
        ),
    }


def _dataset_validation_block() -> dict:
    """Проверка схемы корпуса (дефект E): чужие форматы отвергаются с кодом 2."""
    from spanverify.cli import EXIT_ERROR, main
    from spanverify.dataset import DatasetFormatError

    checks: dict[str, object] = {}
    samples = {
        "foreign": '{"id":"x","question":"тест","answer":"тест","label":1}',
        "empty": "",
        "broken_json": '{"id":"x", ',
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in samples.items():
            path = Path(tmp) / f"{name}.jsonl"
            path.write_text(content + "\n", encoding="utf-8")
            try:
                list(read_pairs(path))
                checks[name] = {"rejected": False, "returncode": None}
            except DatasetFormatError:
                checks[name] = {"rejected": True, "returncode": EXIT_ERROR}
    valid = Path(tempfile.mkdtemp()) / "valid.jsonl"
    valid.write_text(
        '{"id":"p","context":"Регламент: срок 10 лет.","answer":"Срок 10 лет.","labels":[[6,8,1]]}\n',
        encoding="utf-8",
    )
    # Вывод команды глушим: он нужен как код возврата, а не как шум в консоли.
    with contextlib.redirect_stdout(io.StringIO()):
        checks["valid"] = {"rejected": False, "returncode": main(["evaluate", "--dataset", str(valid)])}
    return {
        "schema_required": ["context", "answer"],
        "schema_optional": ["id", "labels", "meta"],
        "checks": checks,
    }


def _metrics_tree(metrics: dict) -> dict:
    """Сжать дерево метрик evaluate() до чисел, которые попадают в документы."""
    tokens = metrics["tokens"]
    spans = metrics["spans"]
    answers = metrics["answers"]
    return {
        "tokens": {
            "precision": round(tokens["precision"], 4),
            "recall": round(tokens["recall"], 4),
            "f1": round(tokens["f1"], 4),
            "fpr": round(tokens["fpr"], 4),
            "auc": round(tokens["auc"], 4) if tokens["auc"] == tokens["auc"] else None,
            "n": tokens["tp"] + tokens["fp"] + tokens["tn"] + tokens["fn"],
        },
        "spans": {
            "strict_f1": round(spans["f1"], 4),
            "coverage": round(spans["recall_containment"], 4),
            "soft_f1": round(spans["f1_expanded_labels"], 4),
            "width_ratio": round(spans["mean_width_ratio"], 2),
        },
        "answers": {
            "precision": round(answers["precision"], 4),
            "recall": round(answers["recall"], 4),
            "f1": round(answers["f1"], 4),
            "fpr": round(answers["fpr"], 4),
            "auc": round(answers["auc"], 4) if answers["auc"] == answers["auc"] else None,
            "threshold": round(answers["threshold"], 4),
        },
        # Вердикт уровня ответа: сюда попадает и текстовое правило привязки числа
        # к объекту (дефект D), которое не видно в метриках по токенам.
        "verdicts": {
            "tp": metrics.get("verdicts", {}).get("tp"),
            "fp": metrics.get("verdicts", {}).get("fp"),
            "fn": metrics.get("verdicts", {}).get("fn"),
            "tn": metrics.get("verdicts", {}).get("tn"),
            "precision": metrics.get("verdicts", {}).get("precision"),
            "recall": metrics.get("verdicts", {}).get("recall"),
            "f1": metrics.get("verdicts", {}).get("f1"),
            "fpr": metrics.get("verdicts", {}).get("fpr"),
        },
    }


def _corpus_a_block(verifier: Verifier) -> dict:
    """Метрики нашего корпуса A (управляемые подмены), если он собран.

    Корпус создаётся ``scripts/build_corpus_a.py``; здесь он только измеряется —
    тем же кодом, что обслуживает API. Разбиение по документам, поэтому метрики
    dev и test честные (одна группа не попадает в две части).
    """
    manifest_path = ROOT / "data" / "corpus_a" / "manifest.json"
    if not manifest_path.is_file():
        return {
            "available": False,
            "note": (
                "корпус не собран: python scripts/build_corpus_a.py "
                "--docs data/corpus_a/docs --out data/corpus_a --target 1200 --seed 42"
            ),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block: dict = {
        "available": True,
        "dataset_version": manifest["dataset_version"],
        "pairs": manifest["pairs"],
        "balance": manifest["balance"],
        "splits": manifest["splits"],
        "shared_groups": manifest["shared_groups"],
        "spans_checked": manifest["spans_checked"],
        "sha256_pairs": manifest["sha256"]["pairs.jsonl"],
        "documents": {
            "count": manifest["documents"]["count"],
            "synthetic": manifest["documents"]["synthetic"],
        },
        "by_split": {},
        "by_mode": {},
    }
    for name in ("dev", "test"):
        path = ROOT / "data" / "corpus_a" / "splits" / f"{name}.jsonl"
        if not path.is_file():
            continue
        block["by_split"][name] = _metrics_tree(verifier.evaluate(list(read_pairs(path))))
    test_path = ROOT / "data" / "corpus_a" / "splits" / "test.jsonl"
    if test_path.is_file():
        by_mode: dict[str, list] = {}
        for pair in read_pairs(test_path):
            by_mode.setdefault(pair["meta"]["mode"], []).append(pair)
        for mode, pairs_block in sorted(by_mode.items()):
            metrics = verifier.evaluate(pairs_block)
            block["by_mode"][mode] = {
                "pairs": len(pairs_block),
                "verdict_recall": metrics["verdicts"]["recall"],
                "verdict_fpr": metrics["verdicts"]["fpr"],
                "token_f1": round(metrics["tokens"]["f1"], 4),
                "span_coverage": round(metrics["spans"]["recall_containment"], 4),
            }
    return block


def _corpus_b_block() -> dict:
    """Числа внешнего бенчмарка RusHallu-RAG из отчёта оценщика (без выдумывания)."""
    path = ROOT / "reports" / "rus_hallu_eval.json"
    if not path.is_file():
        return {
            "available": False,
            "note": (
                "отчёт не собран: python scripts/fetch_rushallu.py --out data/external/rushallu "
                "--verify && python scripts/rus_hallu_eval.py --data data/external/rushallu "
                "--mode demo --json reports/rus_hallu_eval.json"
            ),
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "dataset": report["dataset"],
        "citation": report["citation"],
        "mode": report["mode"],
        "pairs": report["pairs"],
        "our_metrics": report["our_metrics"],
        "their_metrics": report["their_metrics"],
        "baseline_comparison": report["baseline_comparison"],
        "baseline_note": report["baseline_note"],
        "disclaimer": report["disclaimer"],
        "role": "только тест: обучение и калибровка на корпусе B запрещены",
    }


def _external_tests_block() -> dict:
    """Числа внешних размеченных наборов из ``reports/external_tests.json``.

    Файл собирает ``scripts/external_eval.py``; здесь он только читается, чтобы
    документы сверялись с фактическими прогонами, а не с текстом отчёта.
    """
    path = ROOT / "reports" / "external_tests.json"
    if not path.is_file():
        return {
            "available": False,
            "note": (
                "прогонов нет: scripts/fetch_external_tests.py --all --out data/external --verify --adapt "
                "и scripts/external_eval.py --dataset ragtruth --task qa --split test --mode demo"
            ),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") or {}
    summary: dict = {"available": True, "runs": {}}
    for key, run in runs.items():
        tokens = run["our_metrics"]["tokens"]
        answers = run["our_metrics"]["verdicts"]
        spans = run["our_metrics"]["spans"]
        theirs = run["their_metrics"]
        summary["runs"][key] = {
            "dataset": run["dataset"],
            "task": run["task"],
            "split": run["split"],
            "mode": run["mode"],
            "label_origin": run["label_origin"],
            "pairs": run["pairs"],
            "limit": run.get("limit"),
            "tokens": tokens,
            "answers": answers,
            "spans": spans,
            "their": {
                name: theirs.get(name)
                for name in ("accuracy", "jaccard_score", "hamming_loss", "rouge1", "rouge2", "rougeL")
            },
            "baseline_extracted": bool(run["baseline"]["extracted"]),
        }
    summary["total_pairs"] = sum(item["pairs"] for item in summary["runs"].values())
    summary["demo_runs"] = sum(1 for item in summary["runs"].values() if item["mode"] == "demo")
    summary["hf_runs"] = sum(1 for item in summary["runs"].values() if item["mode"] == "hf")
    return summary


def collect(dataset: str, seed: int, release_dir: Path, coverage_json: Path) -> dict:
    """Собрать METRICS.json целиком (каждое число — из прогона, а не из памяти)."""
    started = time.time()
    pairs = list(read_pairs(dataset))
    report = train(pairs, mode="demo", seed=seed, dataset_name=dataset)
    end_to_end = report.validation["end_to_end"]
    verifier = Verifier(mode="demo", weights=report.bundle)
    participation = report.participation

    payload: dict = {
        "commands": {
            "train": (
                "python -m spanverify train --dataset data/demo_pairs.jsonl " "--out config/weights.json --seed 42"
            ),
            "evaluate": "python -m spanverify evaluate --dataset data/demo_pairs.jsonl",
            "cross_corpus": "python scripts/cross_corpus.py --out reports/cross_corpus.json",
            "tests": "python -m pytest -q",
            "coverage": "python -m pytest --cov=spanverify --cov-fail-under=85",
        },
        "meta": {
            "version": __version__,
            "commit": _git_commit(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "dataset": dataset,
        },
        "disclaimer": DISCLAIMER,
        "tests": {
            "collected": _test_count(),
            "coverage_percent": _coverage_percent(coverage_json),
            "coverage_gate_percent": 85,
        },
        "demo": {
            "split": report.stats.get("split", {}),
            "in_corpus": _metrics_tree(end_to_end),
            "validation": {
                "precision": round(report.validation["precision"], 4),
                "recall": round(report.validation["recall"], 4),
                "f1": round(report.validation["f1"], 4),
                "fpr": round(report.validation["fpr"], 4),
                "auc": round(report.validation["auc"], 4),
                "signal": report.bundle.meta.get("signal"),
            },
            "bundle": {
                "threshold": round(report.bundle.threshold, 4),
                "weights": report.bundle.weights,
                "span_z": report.bundle.span_z,
                "span_floor": report.bundle.span_floor,
                "span_cap": report.bundle.span_cap,
            },
            "participation": {
                "auc_out_of_fold": participation.get("auc_out_of_fold"),
                "calibrated_on": participation.get("calibrated_on"),
                "rows": participation.get("rows"),
                "features": list((participation.get("payload") or {}).get("features") or []),
            },
            "whole_corpus": _metrics_tree(verifier.evaluate(pairs)),
            "demo_answer_example": {
                "answer": "Срок хранения первичных документов составляет 5 лет.",
                "context": "Регламент: срок хранения первичных документов составляет 10 лет.",
                "verdict": verifier.verify(
                    "Срок хранения первичных документов составляет 5 лет.",
                    "Регламент: срок хранения первичных документов составляет 10 лет.",
                ).verdict,
            },
        },
        "defects": {
            "number_attribution": _attribution_block(pairs, verifier),
            "dataset_validation": _dataset_validation_block(),
        },
        "corpus_a": _corpus_a_block(verifier),
        "corpus_b": _corpus_b_block(),
        "external_tests": _external_tests_block(),
        "cross_corpus": None,
        "pilot": None,
        "release": _release_info(release_dir),
        "duration_s": round(time.time() - started, 1),
    }

    cross_path = ROOT / "reports" / "cross_corpus.json"
    if cross_path.is_file():
        cross = json.loads(cross_path.read_text(encoding="utf-8"))
        payload["cross_corpus"] = {
            "in_corpus_f1": cross["in_corpus"]["tokens"]["f1"],
            "cross_corpus_f1": cross["cross_corpus"]["tokens"]["f1"],
            "drop": cross.get("token_f1_drop"),
            "cross_fpr": cross["cross_corpus"]["tokens"]["fpr"],
            "corpus_b": cross["corpus_b"],
        }

    pilot_path = ROOT / "reports" / "pilot" / "pilot.json"
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        payload["pilot"] = {
            "status": pilot.get("status"),
            "published": pilot.get("published"),
            "wording": pilot.get("wording"),
            "control": pilot.get("control"),
            "pairs": pilot.get("pairs"),
            "tokens": pilot.get("tokens"),
            "bootstrap_iterations": pilot.get("bootstrap_iterations"),
            "auc": {
                layer: {
                    feature: {
                        "auc_oriented": values.get("auc_oriented"),
                        "ci": [values.get("ci_low"), values.get("ci_high")],
                        "auc_fact_oriented": values.get("auc_fact_oriented"),
                        "fact_ci": [values.get("ci_fact_low"), values.get("ci_fact_high")],
                        "delta_mean": values.get("delta_mean"),
                        "delta_ci": [values.get("delta_ci_low"), values.get("delta_ci_high")],
                        "delta_significant": values.get("delta_significant"),
                    }
                    for feature, values in (layer_values or {}).items()
                }
                for layer, layer_values in (pilot.get("auc") or {}).items()
            },
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать reports/METRICS.json")
    parser.add_argument("--dataset", default="data/demo_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/METRICS.json")
    parser.add_argument("--release-dir", default="release")
    parser.add_argument("--coverage-json", default="reports/coverage.json")
    args = parser.parse_args()

    payload = collect(args.dataset, args.seed, Path(args.release_dir), Path(args.coverage_json))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"METRICS.json: {out}")
    print(f"  версия: {payload['meta']['version']} | коммит: {payload['meta']['commit'][:12]}")
    print(f"  тестов: {payload['tests']['collected']} | покрытие: {payload['tests']['coverage_percent']}")
    demo = payload["demo"]
    print(
        f"  демо (групповой сплит, сквозной путь): F1={demo['in_corpus']['tokens']['f1']:.3f} "
        f"FPR={demo['in_corpus']['tokens']['fpr']:.3f}"
    )
    corpus_a = payload["corpus_a"]
    if corpus_a.get("available"):
        test_metrics = corpus_a["by_split"].get("test", {})
        print(
            f"  корпус A: {corpus_a['pairs']} пар, сплиты {corpus_a['splits']}, "
            f"общих групп {corpus_a['shared_groups']}; test token F1="
            f"{test_metrics.get('tokens', {}).get('f1')} вердикт F1={test_metrics.get('verdicts', {}).get('f1')}"
        )
    else:
        print(f"  корпус A: {corpus_a['note']}")
    corpus_b = payload["corpus_b"]
    if corpus_b.get("available"):
        theirs = corpus_b["their_metrics"]
        print(
            f"  корпус B (RusHallu-RAG, режим {corpus_b['mode']}): {corpus_b['pairs']} пар, "
            f"accuracy={theirs['accuracy']} Jaccard={theirs['jaccard_score']} ROUGE-L={theirs['rougeL']}"
        )
    else:
        print(f"  корпус B: {corpus_b['note']}")
    external = payload["external_tests"]
    if external.get("available"):
        print(
            f"  внешние наборы: прогонов {len(external['runs'])} "
            f"(demo {external['demo_runs']}, hf {external['hf_runs']}), пар всего {external['total_pairs']}"
        )
    else:
        print(f"  внешние наборы: {external['note']}")
    if payload["cross_corpus"]:
        cross = payload["cross_corpus"]
        print(
            f"  кросс-корпус: {cross['in_corpus_f1']:.3f} → {cross['cross_corpus_f1']:.3f} "
            f"(падение {cross['drop']:+.3f})"
        )
    if payload["pilot"]:
        print(f"  пилот: статус {payload['pilot']['status']}, опубликован={payload['pilot']['published']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
