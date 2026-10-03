"""Встроенный веб-интерфейс: один HTML-файл без сборки и CDN.

Страница обращается только к относительным адресам (``/v1/verify``),
поэтому корректно работает и локально, и через прокси предпросмотра.
"""

from __future__ import annotations

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpanVerify — локализация фрагментов, написанных ИИ</title>
<style>
  :root { color-scheme: light dark; --bg:#f6f7fb; --card:#fff; --ink:#14181f; --muted:#5b6472;
          --accent:#2f6df6; --ai:#e5484d; --human:#12855f; --border:#dfe3ec; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1218; --card:#171b23; --ink:#eef1f7; --muted:#9aa4b5; --border:#2a303c; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 18px; font-size:13.5px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:16px; }
  textarea { width:100%; min-height:190px; resize:vertical; padding:12px; border-radius:10px;
             border:1px solid var(--border); background:transparent; color:inherit; font:14.5px/1.6 inherit; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-top:12px; }
  button { cursor:pointer; border-radius:9px; border:1px solid var(--border); background:transparent;
           color:inherit; padding:8px 13px; font-size:14px; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; padding:10px 20px; }
  button:disabled { opacity:.55; cursor:progress; }
  .slider { display:flex; align-items:center; gap:8px; font-size:13.5px; color:var(--muted); }
  input[type=range] { width:150px; }
  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12.5px; font-weight:600; }
  .badge.ai { background:rgba(229,72,77,.14); color:var(--ai); }
  .badge.human { background:rgba(18,133,95,.14); color:var(--human); }
  .badge.mixed { background:rgba(245,165,36,.18); color:#a35c00; }
  .gauge { height:14px; border-radius:999px; background:rgba(128,128,128,.18); overflow:hidden; margin:10px 0 6px; }
  .gauge > span { display:block; height:100%; background:linear-gradient(90deg,#12855f,#f5a524,#e5484d); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:12px; }
  .stat { border:1px solid var(--border); border-radius:10px; padding:9px 11px; }
  .stat b { display:block; font-size:18px; }
  .stat span { color:var(--muted); font-size:12.5px; }
  .doc { white-space:pre-wrap; font:14.5px/1.7 inherit; }
  mark { background:rgba(229,72,77,.22); border-bottom:2px solid var(--ai); border-radius:3px; padding:1px 2px; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  th, td { border-bottom:1px solid var(--border); padding:5px 8px; text-align:left; }
  th { color:var(--muted); font-weight:600; }
  .warn { border-left:3px solid #f5a524; background:rgba(245,165,36,.1); padding:9px 12px; border-radius:0 8px 8px 0; font-size:13.5px; margin-top:10px; }
  .muted { color:var(--muted); font-size:13px; }
  .err { border-left:3px solid var(--ai); background:rgba(229,72,77,.1); padding:9px 12px; border-radius:0 8px 8px 0; }
  .hint { font-size:13px; color:var(--muted); margin-top:6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SpanVerify</h1>
  <p class="sub">Локализация фрагментов текста, написанных языковой моделью, и оценка доли участия ИИ.
     Локальный сервис, без передачи текста третьим лицам.</p>

  <div class="card">
    <textarea id="input" placeholder="Вставьте текст для проверки…"></textarea>
    <div class="row">
      <button class="primary" id="run">Проверить достоверность</button>
      <button id="s-ai">Пример: машинный</button>
      <button id="s-hum">Пример: человеческий</button>
      <button id="s-mix">Пример: смешанный</button>
      <label class="slider">порог <input type="range" id="thr" min="0" max="1" step="0.01" value="0.5">
        <span id="thrval">0.50</span></label>
      <label class="slider"><input type="checkbox" id="explain"> покадровая таблица</label>
    </div>
    <div class="hint" id="backend">…</div>
  </div>

  <div class="card" id="summary" style="display:none"></div>

  <div class="card" id="docblock" style="display:none">
    <b>Разметка документа</b>
    <p class="muted">Красным выделены фрагменты, отнесённые к машинной генерации.</p>
    <div class="doc" id="doc"></div>
  </div>

  <div class="card" id="spansblock" style="display:none">
    <b>Найденные фрагменты</b>
    <table id="spans"><thead><tr><th>#</th><th>Символы</th><th>Токенов</th><th>Ср. вероятность</th><th>Фрагмент</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="card" id="tokenblock" style="display:none">
    <b>Покадровая оценка</b>
    <p class="muted">Фрагмент таблицы (первые 400 токенов). reprob — сырая оценка, prob — калиброванная.</p>
    <table id="tokens"><thead><tr><th>#</th><th>Токен</th><th>Сырая</th><th>Калиброванная</th><th>Метка</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<script>
const SAMPLES = {
  ai: "Важно отметить, что данный метод обеспечивает эффективное решение поставленной задачи, что подтверждает эффективность предложенного решения. Следует отметить, что предложенный подход позволяет оптимизировать ключевые процессы, что обеспечивает высокое качество получаемых результатов. Кроме того, реализация механизма обеспечивает повышение общей эффективности, что является ключевым фактором успешной реализации. Предложенный подход представляет собой комплексное решение задачи, что позволяет достичь поставленных целей.",
  hum: "Вчера на семинаре мы прогнали три прогона — цифры разошлись примерно на 7 % между прогонами, а на ноутбуке это считалось 40 минут, а на кластере минуту с небольшим. Пётр предложил иначе: половина датасета оказалась с битыми подписями, пришлось вручную перепроверять 200 с лишним примеров. Честно говоря, ошибка вылезала только при batch_size=3, что странно; спрошу у Ларисы, она это уже делала.",
  mix: "Отчёт сдали в срок, замечаний не было. Важно отметить, что реализация механизма обеспечивает повышение общей эффективности, что подтверждает эффективность предложенного решения. Следует отметить, что ключевой аспект заключается в комплексной оптимизации параметров, что позволяет достичь поставленных целей. Тут вышла заминка: формулу (3.2) я так и не проверил до конца. По моим наблюдениям, гипотезу пришлось отбросить — корреляция оказалась 0.12. В современном мире предложенный подход представляет собой комплексное решение, что свидетельствует о высокой эффективности подхода."
};

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (100 * x).toFixed(1) + " %";
const VERDICT = { likely_ai: ["Скорее всего, текст создан ИИ", "ai"],
                  mixed: ["Смешанный текст", "mixed"],
                  likely_human: ["Скорее всего, текст человеческий", "human"] };

$("thr").addEventListener("input", (e) => { $("thrval").textContent = (+e.target.value).toFixed(2); });
$("s-ai").onclick = () => { $("input").value = SAMPLES.ai; };
$("s-hum").onclick = () => { $("input").value = SAMPLES.hum; };
$("s-mix").onclick = () => { $("input").value = SAMPLES.mix; };

fetch("/health").then((r) => r.json()).then((h) => {
  const parts = ["режим: " + h.backend, "версия " + h.version,
                 h.calibrated ? "калибратор: загружен" : "калибратор: отсутствует"];
  $("backend").textContent = parts.join(" · ") + (h.notice ? " — " + h.notice : "");
}).catch(() => { $("backend").textContent = "сервис недоступен"; });

$("run").onclick = async () => {
  const text = $("input").value;
  if (!text.trim()) { alert("Вставьте текст для проверки."); return; }
  $("run").disabled = true;
  $("run").textContent = "Проверяю…";
  try {
    const res = await fetch("/v1/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, threshold: +$("thr").value, explain: $("explain").checked })
    });
    const data = await res.json();
    if (!res.ok) { showError(data.message || "ошибка"); return; }
    render(data.result);
    if (data.tokens) renderTokens(data.tokens);
  } catch (e) {
    showError(String(e));
  } finally {
    $("run").disabled = false;
    $("run").textContent = "Проверить достоверность";
  }
};

function showError(message) {
  $("summary").style.display = "block";
  $("summary").innerHTML = '<div class="err"><b>Ошибка:</b> ' + esc(message) + "</div>";
  $("docblock").style.display = "none";
  $("spansblock").style.display = "none";
  $("tokenblock").style.display = "none";
}

function render(result) {
  const [label, cls] = VERDICT[result.verdict] || ["—", "mixed"];
  let html = '<b>' + label + '</b> <span class="badge ' + cls + '">' + esc(result.verdict) + "</span>";
  html += '<div class="gauge"><span style="width:' + (100 * result.ai_fraction).toFixed(1) + '%"></span></div>';
  html += '<div class="muted">Доля участия ИИ: ' + pct(result.ai_fraction) +
          " символов · " + pct(result.ai_fraction_tokens) + " слов · порог " + result.threshold.toFixed(3) +
          (result.calibrated ? " · калибровано" : " · без калибровки") + "</div>";
  html += '<div class="stats">' +
    stat(result.n_word_tokens, "слов в документе") +
    stat(result.spans_count, "фрагментов найдено") +
    stat(result.text_length, "символов") +
    stat(result.backend, "бэкенд") + "</div>";
  (result.warnings || []).forEach((w) => { html += '<div class="warn">' + esc(w) + "</div>"; });
  $("summary").style.display = "block";
  $("summary").innerHTML = html;

  const spans = result.spans || [];
  $("docblock").style.display = "block";
  $("doc").innerHTML = markup($("input").value, spans);

  const tbody = $("spans").querySelector("tbody");
  tbody.innerHTML = spans.length ? spans.map((s) =>
    "<tr><td>" + (s.index + 1) + "</td><td>" + s.start_char + "–" + s.end_char + "</td><td>" + s.n_tokens +
    "</td><td>" + s.mean_prob.toFixed(3) + "</td><td>" + esc(s.text.slice(0, 140)) + "</td></tr>").join("")
    : '<tr><td colspan="5" class="muted">Фрагментов выше порога не найдено.</td></tr>';
  $("spansblock").style.display = "block";
}

function stat(value, caption) {
  return '<div class="stat"><b>' + esc(String(value)) + "</b><span>" + esc(caption) + "</span></div>";
}

function markup(text, spans) {
  const ordered = [...spans].sort((a, b) => a.start_char - b.start_char);
  let out = "";
  let cursor = 0;
  for (const s of ordered) {
    const start = Math.max(cursor, s.start_char);
    const end = Math.max(start, s.end_char);
    out += esc(text.slice(cursor, start));
    out += "<mark>" + esc(text.slice(start, end)) + "</mark>";
    cursor = end;
  }
  out += esc(text.slice(cursor));
  return out;
}

function renderTokens(tokens) {
  const tbody = $("tokens").querySelector("tbody");
  tbody.innerHTML = tokens.slice(0, 400).map((t, i) =>
    "<tr><td>" + (i + 1) + "</td><td>" + esc(t.token) + "</td><td>" + t.raw.toFixed(3) +
    "</td><td>" + t.prob.toFixed(3) + "</td><td>" + (t.flag ? "ИИ" : "человек") + "</td></tr>").join("");
  $("tokenblock").style.display = "block";
}
</script>
</body>
</html>
"""
