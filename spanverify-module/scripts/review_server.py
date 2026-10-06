#!/usr/bin/env python3
"""Сервер ручной проверки корпуса: очередь → решения человека.

Зачем он нужен
--------------

Корпус A собран скриптом, значит его метки нужно выборочно подтвердить глазами
(по плану — 10 % пар), а спаны естественных ответов (A2) проверить полностью.
Скрипт закрывает этот шаг без внешних сервисов: отдаёт страницу с одной парой,
пишет решения в JSONL и показывает прогресс. Ни одна пара не «подтверждается»
автоматически: решение принимает человек, файл решений — его след.

Эндпоинты
---------

* ``GET /`` — страница с очередной непроверенной парой;
* ``GET /api/item/<id>`` — пара в JSON;
* ``POST /decision`` — решение ``{"id": ..., "verdict": "ok"|"wrong"|"skip", "comment": ...}``,
  дописывается в файл решений (JSONL, одна запись на строку);
* ``GET /api/stats`` — прогресс: сколько проверено, сколько осталось, разбивка;

Запуск::

    # 1) очередь на проверку (10 % корпуса A)
    python scripts/review_server.py --make-sample --source data/corpus_a/splits/test.jsonl \\
        --queue data/review_queue/corpus_a_10pct.jsonl --share 0.10 --seed 42

    # 2) сервер
    python scripts/review_server.py --queue data/review_queue/corpus_a_10pct.jsonl \\
        --out data/review_queue/decisions.jsonl --port 8770
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICTS = {"ok", "wrong", "skip"}

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Проверка корпуса — SpanVerify</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f6f7f9; }}
 main {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 .progress {{ color: #667; font-size: 14px; margin-bottom: 16px; }}
 .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #778; margin: 16px 0 4px; }}
 .text {{ white-space: pre-wrap; line-height: 1.5; background: #fafbfc; border: 1px solid #e6e8eb;
         border-radius: 8px; padding: 12px; max-height: 340px; overflow: auto; }}
 mark {{ background: #ffe08a; padding: 0 2px; border-radius: 3px; }}
 .meta {{ color: #667; font-size: 13px; margin-top: 8px; }}
 button {{ font-size: 15px; padding: 10px 18px; margin-right: 8px; border-radius: 8px; border: 1px solid #ccd;
          background: #fff; cursor: pointer; }}
 button.ok {{ background: #e6f6ea; border-color: #9ad3a6; }}
 button.wrong {{ background: #fdeaea; border-color: #e0a1a1; }}
 input[type=text] {{ width: 100%; padding: 10px; margin-top: 8px; border: 1px solid #ccd; border-radius: 8px; }}
 .done {{ font-size: 18px; color: #256b3b; }}
</style>
</head>
<body>
<main>
<h1>Проверка корпуса</h1>
<div class="progress" id="progress">{progress}</div>
<div id="card">{card}</div>
</main>
<script>
async function send(id, verdict) {{
  const comment = document.getElementById('comment') ? document.getElementById('comment').value : '';
  await fetch('decision', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id, verdict, comment}})}});
  location.reload();
}}
document.addEventListener('keydown', (event) => {{
  const id = document.getElementById('card').dataset.id;
  if (!id) return;
  if (event.key === '1') send(id, 'ok');
  if (event.key === '2') send(id, 'wrong');
  if (event.key === '3') send(id, 'skip');
}});
</script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    """Прочитать очередь или решения (если файла нет — пустой список)."""
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def make_sample(source: Path, queue: Path, share: float, seed: int) -> int:
    """Взять случайную долю пар из источника — очередь ручной проверки."""
    pairs = load_jsonl(source)
    if not pairs:
        raise SystemExit(f"нет пар в {source}")
    random.Random(seed).shuffle(pairs)
    count = max(1, round(len(pairs) * share))
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs[:count]), encoding="utf-8")
    return count


def decisions_index(path: Path) -> dict[str, dict]:
    """Последнее решение по каждой паре (перепроверка перезаписывает ответ)."""
    index: dict[str, dict] = {}
    for row in load_jsonl(path):
        index[str(row.get("id"))] = row
    return index


def render_card(pair: dict) -> str:
    """Карточка пары: контекст, ответ с подсветкой метки и кнопки решения."""
    answer = pair.get("answer", "")
    pieces: list[str] = []
    cursor = 0
    for start, end, _label in sorted(pair.get("labels", []), key=lambda item: item[0]):
        start, end = int(start), int(end)
        if start < cursor or end > len(answer):
            continue
        pieces.append(html.escape(answer[cursor:start]))
        pieces.append(f"<mark>{html.escape(answer[start:end])}</mark>")
        cursor = end
    pieces.append(html.escape(answer[cursor:]))
    meta = pair.get("meta", {})
    mode = meta.get("mode", "—")
    taxonomy = meta.get("taxonomy", "—")
    marked = len(pair.get("labels", []))
    return (
        f'<div id="card" data-id="{html.escape(str(pair.get("id", "")))}">'
        f'<div class="label">Контекст</div><div class="text">{html.escape(pair.get("context", ""))}</div>'
        f'<div class="label">Ответ</div><div class="text">{"".join(pieces)}</div>'
        f'<div class="meta">id: {html.escape(str(pair.get("id", "")))} | режим: {html.escape(str(mode))} | '
        f"тип: {html.escape(str(taxonomy))} | меток: {marked}</div>"
        '<div class="label">Решение (1 — верно, 2 — неверно, 3 — пропустить)</div>'
        f'<button class="ok" onclick="send(\'{html.escape(str(pair.get("id", "")))}\', \'ok\')">Верно</button>'
        f'<button class="wrong" onclick="send(\'{html.escape(str(pair.get("id", "")))}\', \'wrong\')">Неверно</button>'
        f'<button onclick="send(\'{html.escape(str(pair.get("id", "")))}\', \'skip\')">Пропустить</button>'
        '<input type="text" id="comment" placeholder="комментарий (необязательно)">'
        "</div>"
    )


def make_handler(queue_path: Path, out_path: Path):  # noqa: ANN201 - фабрика обработчика
    """Собрать обработчик HTTP с замкнутыми путями очереди и решений."""

    class Handler(BaseHTTPRequestHandler):
        """Отдаёт страницу проверки и принимает решения."""

        server_version = "SpanVerifyReview/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - подпись базового класса
            """Не засорять вывод: каждая отдача страницы — это reload браузера."""
            return

        def _send(self, body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload, ensure_ascii=False), status, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - имя задано базовым классом
            """Отдать страницу проверки, одну пару или статистику."""
            path = unquote(self.path.split("?")[0])
            queue = load_jsonl(queue_path)
            done = decisions_index(out_path)
            if path in {"/", "/index.html"}:
                pending = [pair for pair in queue if str(pair.get("id")) not in done]
                progress = (
                    f"проверено {len(done)} из {len(queue)} · осталось {len(pending)}"
                    if queue
                    else "очередь пуста — создайте её ключом --make-sample"
                )
                card = (
                    render_card(pending[0])
                    if pending
                    else '<div class="done">Очередь проверена: непроверенных пар нет.</div>' if queue else ""
                )
                self._send(PAGE.format(progress=html.escape(progress), card=card))
                return
            if path.startswith("/api/item/"):
                item_id = path.removeprefix("/api/item/")
                for pair in queue:
                    if str(pair.get("id")) == item_id:
                        self._json(pair)
                        return
                self._json({"error": "пара не найдена", "id": item_id}, status=404)
                return
            if path == "/api/stats":
                counts: dict[str, int] = {}
                for row in done.values():
                    counts[str(row.get("verdict"))] = counts.get(str(row.get("verdict")), 0) + 1
                self._json(
                    {
                        "queue": len(queue),
                        "checked": len(done),
                        "left": len(queue) - len(done),
                        "decisions": counts,
                    }
                )
                return
            self._json({"error": "не найдено", "path": path}, status=404)

        def do_POST(self) -> None:  # noqa: N802 - имя задано базовым классом
            """Принять решение по паре и дописать его в файл решений."""
            if self.path.split("?")[0] != "/decision":
                self._json({"error": "не найдено", "path": self.path}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError as error:
                self._json({"error": f"некорректный JSON: {error}"}, status=400)
                return
            item_id = str(payload.get("id") or "")
            verdict = str(payload.get("verdict") or "")
            known = {str(pair.get("id")) for pair in load_jsonl(queue_path)}
            if not item_id or item_id not in known:
                self._json({"error": "неизвестный id", "id": item_id}, status=400)
                return
            if verdict not in VERDICTS:
                self._json({"error": f"вердикт должен быть одним из {sorted(VERDICTS)}"}, status=400)
                return
            out_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"id": item_id, "verdict": verdict, "comment": str(payload.get("comment") or "")}
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._json({"saved": record, "left": len(known) - len(decisions_index(out_path))})

    return Handler


def serve(queue: Path, out: Path, host: str, port: int) -> None:
    """Поднять сервер проверки (блокирующий вызов)."""
    handler = make_handler(queue, out)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"[review] очередь: {queue}")
    print(f"[review] решения: {out}")
    print(f"[review] страница: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] остановлено")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Сервер ручной проверки корпуса")
    parser.add_argument("--queue", type=Path, default=ROOT / "data" / "review_queue" / "corpus_a_10pct.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "review_queue" / "decisions.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--make-sample", action="store_true", help="создать очередь из источника и выйти")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "corpus_a" / "splits" / "test.jsonl")
    parser.add_argument("--share", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.make_sample:
        try:
            count = make_sample(args.source, args.queue, args.share, args.seed)
        except SystemExit as error:
            print(f"ошибка: {error}", file=sys.stderr)
            return 2
        print(f"Очередь: {args.queue} ({count} пар из {args.source})")
        return 0

    if not args.queue.is_file():
        print(
            f"ошибка: нет очереди {args.queue}. Создайте её: python scripts/review_server.py "
            f"--make-sample --source {args.source} --queue {args.queue}",
            file=sys.stderr,
        )
        return 2
    serve(args.queue, args.out, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
