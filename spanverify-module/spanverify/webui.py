"""Встроенный веб-интерфейс: один HTML-файл без сборки и без CDN.

Страница обращается только к относительным адресам (``/v1/verify``,
``/health``), поэтому работает и из собранного .exe, и из предпросмотра, и
локально без интернета. Слева — документ-контекст, справа — ответ: именно так
формулируется задача «подтверждается ли ответ документом».
"""

from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpanVerify — проверка ответа по документу</title>
<style>
  :root { --bg:#f5f6f8; --card:#fff; --ink:#1c2128; --muted:#6b7280; --line:#e3e6ea;
          --ok:#1f8a4c; --warn:#b57d05; --bad:#c0392b; --accent:#2d5be3; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
  header { background:#fff; border-bottom:1px solid var(--line); padding:14px 22px;
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:19px; margin:0; }
  .sub { color:var(--muted); font-size:13px; }
  main { max-width:1240px; margin:18px auto 60px; padding:0 16px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  label { display:block; font-weight:600; margin-bottom:6px; font-size:13px; }
  textarea { width:100%; min-height:210px; resize:vertical; padding:10px; font:13px/1.5 inherit;
             border:1px solid var(--line); border-radius:8px; background:#fcfcfd; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  button { border:1px solid var(--line); background:#fff; padding:9px 14px; border-radius:8px;
           cursor:pointer; font-size:14px; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.55; cursor:progress; }
  .verdict { display:flex; align-items:center; gap:14px; margin-bottom:12px; flex-wrap:wrap; }
  .badge { padding:6px 12px; border-radius:999px; font-weight:700; font-size:14px; }
  .grounded { background:#e7f6ec; color:var(--ok); }
  .doubtful { background:#fdf3dd; color:var(--warn); }
  .likely_hallucination { background:#fdecea; color:var(--bad); }
  .empty { background:#eef1f5; color:var(--muted); }
  .stats { display:flex; gap:22px; flex-wrap:wrap; }
  .stat b { display:block; font-size:19px; }
  .stat span { color:var(--muted); font-size:12px; }
  .answer { white-space:pre-wrap; font-size:15px; line-height:1.7; }
  mark { background:#ffd9d4; border-bottom:2px solid var(--bad); padding:1px 2px; border-radius:3px; }
  mark.doubtful { background:#ffeec2; border-bottom-color:var(--warn); }
  .errors { color:var(--bad); font-weight:600; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }
  th, td { border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  .hint { color:var(--muted); font-size:12.5px; margin-top:10px; }
  .warnbox { background:#fdf3dd; border:1px solid #f0dcae; color:#6b4c05; border-radius:8px;
             padding:10px 12px; font-size:13px; margin-top:12px; }
  details { margin-top:12px; }
  summary { cursor:pointer; color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>SpanVerify</h1>
  <div class="sub">проверка ответа по документу-источнику · <span id="backend">…</span></div>
</header>

<main>
  <div class="grid">
    <div class="card">
      <label for="context">Документ-контекст (источник фактов)</label>
      <textarea id="context" placeholder="Вставьте фрагмент документа, на который должен опираться ответ…"></textarea>
    </div>
    <div class="card">
      <label for="answer">Ответ, который проверяем</label>
      <textarea id="answer" placeholder="Вставьте ответ языковой модели или сотрудника…"></textarea>
    </div>
  </div>

  <div class="row">
    <button class="primary" id="run">Проверить</button>
    <button id="sample-true">Пример: подтверждённый</button>
    <button id="sample-false">Пример: подмена факта</button>
    <button id="sample-fab">Пример: выдумка</button>
    <label class="slider"><input type="checkbox" id="tokens"> разбор по токенам</label>
    <span class="hint" id="latency"></span>
  </div>

  <div class="card" id="result" style="display:none; margin-top:16px;">
    <div class="verdict">
      <span class="badge" id="badge">—</span>
      <div class="stats">
        <div class="stat"><b id="score">—</b><span>оценка недостоверности</span></div>
        <div class="stat"><b id="thr">—</b><span>порог решения</span></div>
        <div class="stat"><b id="share">—</b><span>доля спорного текста</span></div>
        <div class="stat"><b id="participation">—</b><span>оценка участия ИИ</span></div>
        <div class="stat"><b id="nspans">—</b><span>фрагментов</span></div>
      </div>
    </div>
    <div class="answer" id="marked"></div>
    <div id="warning"></div>
    <div id="spanblock" style="display:none">
      <table><thead><tr><th>Символы</th><th>Риск</th><th>Метка</th><th>Фрагмент</th></tr></thead>
      <tbody id="spans"></tbody></table>
    </div>
    <div id="tokenblock" style="display:none">
      <table><thead><tr><th>#</th><th>Токен</th><th>Энтропия</th><th>Опора</th><th>Плотность</th><th>Риск</th><th>Метка</th></tr></thead>
      <tbody id="tokens-body"></tbody></table>
    </div>
    <details>
      <summary>Технические детали ответа</summary>
      <pre id="raw" style="white-space:pre-wrap; font-size:12px;"></pre>
    </details>
  </div>
  <div class="card errors" id="error" style="display:none; margin-top:16px;"></div>
  <div class="hint">
    Все вычисления выполняются локально на вашем компьютере: приложение не отправляет тексты в интернет.
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = (x) => (100 * x).toFixed(1) + " %";
const VERDICT = {
  grounded: ["Ответ подтверждается документом", "grounded"],
  doubtful: ["Есть сомнительные фрагменты", "doubtful"],
  likely_hallucination: ["Высокий риск недостоверности", "likely_hallucination"],
  empty: ["Пустой ответ", "empty"],
};

const SAMPLES = {
  "sample-true": {
    context: "Регламент 343: срок хранения первичных документов составляет 10 лет. Контроль исполнения возложен на службу делопроизводства.",
    answer: "Срок хранения первичных документов установлен в размере 10 лет.",
  },
  "sample-false": {
    context: "Регламент 343: срок хранения первичных документов составляет 10 лет.",
    answer: "Срок хранения первичных документов составляет 3 года.",
  },
  "sample-fab": {
    context: "Регламент 343: срок хранения первичных документов составляет 10 лет.",
    answer: "Срок хранения первичных документов составляет 10 лет. Дополнительно требуется согласование с внешним аудитором и архивным агентством.",
  },
};

fetch("/health").then((r) => r.json()).then((h) => {
  const parts = ["режим: " + (h.mode || h.backend), "версия " + h.version];
  if (h.calibrated) parts.push("порог " + Number(h.threshold).toFixed(3));
  if (h.head && h.head !== "none") parts.push("голова: " + h.head);
  $("backend").textContent = parts.join(" · ");
  if (h.warning) showWarning(h.warning);
}).catch(() => { $("backend").textContent = "сервис недоступен"; });

for (const [id, sample] of Object.entries(SAMPLES)) {
  $(id).addEventListener("click", () => {
    $("context").value = sample.context;
    $("answer").value = sample.answer;
    $("result").style.display = "none";
  });
}

$("run").addEventListener("click", async () => {
  const answer = $("answer").value;
  if (!answer.trim()) { showError("Введите ответ для проверки."); return; }
  $("run").disabled = true;
  $("error").style.display = "none";
  try {
    const response = await fetch("/v1/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer, context: $("context").value, with_tokens: $("tokens").checked }),
    });
    const data = await response.json();
    if (!response.ok) { showError(data.error || ("ошибка " + response.status)); return; }
    render(data, answer);
  } catch (error) {
    showError("Не удалось выполнить запрос: " + error);
  } finally {
    $("run").disabled = false;
  }
});

function showError(message) { $("error").textContent = message; $("error").style.display = "block"; }
function showWarning(message) { $("warning").innerHTML = '<div class="warnbox">' + esc(message) + "</div>"; }

function render(data, answer) {
  const [label, cls] = VERDICT[data.verdict] || ["—", "empty"];
  $("badge").textContent = label;
  $("badge").className = "badge " + cls;
  $("score").textContent = Number(data.score).toFixed(3);
  $("thr").textContent = Number(data.threshold).toFixed(3);
  $("share").textContent = pct(data.ai_share) + " / " + pct(data.ai_share_hard);
  $("participation").textContent = pct(data.ai_participation);
  $("nspans").textContent = String((data.spans || []).length);
  $("marked").innerHTML = markup(answer, data.spans || []);
  $("warning").innerHTML = "";
  if (data.stats && data.stats.warning) showWarning(data.stats.warning);
  $("latency").textContent = "обработано за " + Number(data.latency_ms).toFixed(0) + " мс";

  const spans = data.spans || [];
  $("spanblock").style.display = spans.length ? "block" : "none";
  $("spans").innerHTML = spans.map((s) => (
    "<tr><td>" + s.start + ":" + s.end + "</td><td>" + Number(s.risk).toFixed(3) +
    "</td><td>" + esc(s.label) + "</td><td>" + esc(s.text) + "</td></tr>"
  )).join("");

  const tokens = data.tokens || [];
  $("tokenblock").style.display = tokens.length ? "block" : "none";
  $("tokens-body").innerHTML = tokens.map((t) => (
    "<tr><td>" + t.index + "</td><td>" + esc(t.text) + "</td><td>" + Number(t.attention_entropy).toFixed(3) +
    "</td><td>" + Number(t.ctx_attention_mass).toFixed(3) + "</td><td>" + Number(t.embedding_density).toFixed(3) +
    "</td><td>" + Number(t.risk).toFixed(3) + "</td><td>" + esc(t.label) + "</td></tr>"
  )).join("");

  $("raw").textContent = JSON.stringify(data.stats || {}, null, 1);
  $("result").style.display = "block";
}

function markup(text, spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start);
  let cursor = 0, html = "";
  for (const span of ordered) {
    const start = Math.max(cursor, span.start);
    const end = Math.max(start, span.end);
    html += esc(text.slice(cursor, start));
    const cls = span.label === "likely_hallucination" ? "" : ' class="doubtful"';
    html += "<mark" + cls + " title=\\"риск " + Number(span.risk).toFixed(3) + " · " + esc(span.label) +
            "\\">" + esc(text.slice(start, end)) + "</mark>";
    cursor = end;
  }
  return html + esc(text.slice(cursor));
}
</script>
</body>
</html>
"""

__all__ = ["INDEX_HTML"]
