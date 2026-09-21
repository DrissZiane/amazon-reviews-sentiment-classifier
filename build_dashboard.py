#!/usr/bin/env python3
"""Dashboard generator.

Builds a single self-contained HTML page (works offline, no CDN/network) that
presents the classification results: headline numbers, descriptive + prediction
visualizations, an emotion comparison, and an interactive, filterable review
table with a live count.

Usage:
    python build_dashboard.py \
        --results outputs/classify_balanced_three_with_nrc.json \
        --out outputs/dashboard.html
"""

import argparse
import gzip
import html
import json
import os
from collections import Counter

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
DATA = "data/Gift_Cards.jsonl.gz"


# --------------------------------------------------------------------------- helpers
def full_dataset_distribution(path=DATA):
    stars = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stars[json.loads(line)["rating"]] += 1
    return stars


def load_results(path):
    return json.load(open(path, encoding="utf-8"))


def bar(label, value, total, color, max_value=None):
    if max_value is None:
        max_value = total
    pct = (value / max_value * 100) if max_value else 0
    # guard: never below 8px so tiny bars can't collapse to zero width
    return f"""
    <div class="hbar-row">
      <div class="hbar-label">{label}</div>
      <div class="hbar-track"><div class="hbar-fill" style="width:{pct:.1f}%;background:{color};min-width:8px"></div></div>
      <div class="hbar-val">{value}<span class="hbar-sub">/{total} · {value/total*100:.1f}%</span></div>
    </div>"""


# --------------------------------------------------------------------------- metrics
def compute_metrics(rows):
    n = len(rows)
    valid = [r for r in rows if r["pred_sentiment"] in CLASSES]
    correct = [r for r in valid if r["correct"]]
    acc = len(correct) / n
    nerr = n - len(valid)
    # truth + predicted distributions
    truth = Counter(r["truth"] for r in rows)
    pred = Counter(r["pred_sentiment"] for r in rows)
    # per-class recall
    recall = {c: [0, 0] for c in CLASSES}
    precision = {c: [0, 0] for c in CLASSES}
    for r in rows:
        if r["truth"] == r["pred_sentiment"] and r["pred_sentiment"] in CLASSES:
            recall[r["truth"]][0] += 1
        recall[r["truth"]][1] += 1
    for r in rows:
        if r["pred_sentiment"] in CLASSES:
            if r["truth"] == r["pred_sentiment"]:
                precision[r["pred_sentiment"]][0] += 1
            precision[r["pred_sentiment"]][1] += 1
    # confusion matrix (rows=truth, cols=pred), plus error col
    mat = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in rows:
        p = r["pred_sentiment"] if r["pred_sentiment"] in CLASSES else None
        mat[r["truth"]][p] = mat[r["truth"]].get(p, 0) + 1
    mat_errors = {t: sum(1 for r in rows if r["truth"] == t and r["pred_sentiment"] not in CLASSES) for t in CLASSES}
    # emotions — restore the thermometer of observed labels, sanitized to the 8
    # NRC emotions (+ "none" for empty, "other" for anything the model invented)
    def _sanitize(e):
        e = (e or "").lower()
        if not e:
            return "none"
        return e if e in EMOTIONS else "other"
    llm_emo = Counter(_sanitize(r.get("pred_emotion")) for r in rows)
    nrc_emo = Counter((r.get("nrc_emotion") or "none").lower() for r in rows)
    emo_agree = sum(1 for r in rows if _sanitize(r.get("pred_emotion")) == (r.get("nrc_emotion") or "none").lower())
    return dict(n=n, acc=acc, correct=len(correct), wrong=n - len(correct), nerr=nerr,
                truth=truth, pred=pred, recall=recall, precision=precision,
                mat=mat, mat_errors=mat_errors, llm_emo=llm_emo, nrc_emo=nrc_emo,
                emo_agree=emo_agree)


# --------------------------------------------------------------------------- html assembly
CSS = """\
:root{
  --bg:#F6F3EC; --surface:#FFFFFF; --surface2:#FBF9F4;
  --ink:#201E1A; --muted:#6E6A5E; --line:#E6E1D4;
  --accent:#C2561F; --accent-ink:#7C3A12;
  --pos:#3E7C4F; --pos-bg:#E4EDE6;
  --neu:#B08A1F; --neu-bg:#F4EED7;
  --neg:#B4453A; --neg-bg:#F3E4E1;
  --err:#8A8A94; --err-bg:#EEEEF1;
  --serif: Georgia,'Iowan Old Style','Times New Roman',serif;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --radius:14px; --shadow:0 1px 2px rgba(32,30,26,.06),0 12px 30px -18px rgba(32,30,26,.25);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.55;padding:0 20px 90px}
.wrap{max-width:1040px;margin:0 auto}
header.mast{padding:56px 0 30px;border-bottom:1px solid var(--line)}
.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent-ink);
  font-weight:700;margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5vw,46px);line-height:1.08;margin:0 0 16px}
.sub{color:var(--muted);font-size:15px;max-width:64ch;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:22px;color:var(--muted);font-size:12.5px}
.meta b{color:var(--ink);font-weight:600}

section.block{margin-top:44px}
.sec-head{display:flex;align-items:baseline;gap:14px;margin-bottom:18px}
.sec-head h2{font-family:var(--serif);font-weight:600;font-size:23px;margin:0}
.sec-note{color:var(--muted);font-size:13px}

.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:26px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:20px;box-shadow:var(--shadow)}
.kpi .lab{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600}
.kpi .num{font-family:var(--serif);font-size:34px;font-weight:600;margin-top:6px;line-height:1}
.kpi .num small{font-size:17px;color:var(--muted);font-weight:400}
.kpi .foot{font-size:12px;color:var(--muted);margin-top:8px}

.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:26px;box-shadow:var(--shadow)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}

.hbar-row{display:grid;grid-template-columns:150px 1fr 120px;gap:12px;align-items:center;margin:10px 0}
.hbar-label{font-size:13px;color:var(--ink);font-weight:500;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hbar-track{background:var(--surface2);border:1px solid var(--line);border-radius:999px;height:20px;overflow:hidden}
.hbar-fill{height:100%;border-radius:999px;transition:width .3s}
.hbar-val{font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}
.hbar-sub{color:var(--muted);font-size:11px;margin-left:6px}

/* grouped truth-vs-pred */
.grow{margin:6px 0}
.grow .gr{display:flex;align-items:center;gap:10px;margin:7px 0}
.grow .g-label{width:96px;font-size:12.5px;font-weight:600}
.grow .g-tracks{flex:1;display:flex;gap:6px}
.grow .b{height:26px;border-radius:6px;min-width:10px;display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:11.5px;font-weight:700;font-variant-numeric:tabular-nums}
.grow .g-legend{display:flex;gap:18px;margin-bottom:10px;font-size:12px;color:var(--muted)}

.cm{width:100%;border-collapse:separate;border-spacing:5px;font-size:13px}
.cm th{font-weight:600;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:4px 8px}
.cm td{text-align:center;padding:14px 6px;border-radius:8px;font-variant-numeric:tabular-nums;font-weight:600}
.cm .c-t{background:#95866a;color:#fff}
.diag{color:#0f0e0c}
.cm .cap{font-size:11px;color:var(--muted);font-weight:500;margin-top:14px;text-align:center}

.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11.5px;font-weight:700;letter-spacing:.02em}
.pill.POSITIVE{background:var(--pos-bg);color:var(--pos)}
.pill.NEUTRAL{background:var(--neu-bg);color:var(--neu)}
.pill.NEGATIVE{background:var(--neg-bg);color:var(--neg)}
.pill.ERR{background:var(--err-bg);color:var(--err)}
.pill.t{background:var(--surface2);color:var(--muted);border:1px solid var(--line)}

.emo-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.emo-grid{grid-template-columns:1fr}}
.emo-h{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:6px}

.verdict{margin-top:20px;background:var(--surface2);border:1px dashed var(--line);border-radius:var(--radius);padding:18px 22px;font-size:14.5px}
.verdict b{font-family:var(--serif);font-size:15.5px}

/* filters + table */
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0}
.filters select,.filters input[type=search]{font:500 13px var(--sans);color:var(--ink);
  background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:8px 12px;
  outline:none}
.filters select:focus,.filters input:focus{border-color:var(--accent)}
.count-badge{margin-left:auto;font-size:13px;color:var(--muted)}
.count-badge b{font-family:var(--serif);font-size:20px;color:var(--ink);font-variant-numeric:tabular-nums}
table.rev{width:100%;border-collapse:collapse;font-size:13.5px}
table.rev th{text-align:left;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--line);padding:10px 12px;position:sticky;top:0;background:var(--surface)}
table.rev td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}
table.rev tr:hover td{background:var(--surface2)}
.stars{color:#D9A218;letter-spacing:1px;font-size:12px;white-space:nowrap}
.rev-t{font-weight:600;line-height:1.3}
.rev-b{color:var(--muted);font-size:12.5px;margin-top:3px;max-width:none;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
em.e{font-style:normal;font-size:11px;color:var(--muted)}
#noRows{padding:26px;text-align:center;color:var(--muted);font-size:14px}

footer{margin-top:60px;padding-top:24px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
footer b{color:var(--ink)}
a{color:var(--accent-ink)}
"""

HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SENTIMENT · Amazon Gift Cards reviews — classifier dashboard</title>
<style>{css}</style></head><body><div class="wrap">"""


def kpi_cards(m):
    wrong = m["n"] - m["correct"]
    cls_d = {"POSITIVE": "pos", "NEUTRAL": "neu", "NEGATIVE": "neg"}
    rec_l = [f'<span class="pill {cls_d[c]}">{c}</span> {m["recall"][c][0]}/{m["recall"][c][1]}' for c in CLASSES]
    return f"""<div class="kpis">
      <div class="kpi"><div class="lab">Overall accuracy</div>
        <div class="num">{m['acc']*100:.1f}<small>%</small></div>
        <div class="foot">{m['correct']} of {m['n']} agree with the rating</div></div>
      <div class="kpi"><div class="lab">Right / Wrong</div>
        <div class="num"><span style="color:var(--pos)">{m['correct']}</span><small> / {wrong}</small></div>
        <div class="foot">{m['nerr']} empty model output(s) count as wrong</div></div>
      <div class="kpi"><div class="lab">Reviews scored</div>
        <div class="num">{m['n']}<small> rws</small></div>
        <div class="foot">balanced ~50 / class · fixed seed</div></div>
      <div class="kpi"><div class="lab">Per-class recall</div><div class="foot" style="margin-top:14px;line-height:2">
        {'<br>'.join(rec_l)}</div></div>
    </div>"""


def confusion(m):
    rows = []
    maxc = max((m["mat"][t][p] for t in CLASSES for p in CLASSES), default=1)
    for t in CLASSES:
        cells = ""
        for p in CLASSES:
            v = m["mat"][t].get(p, 0)
            isd = (t == p)
            intensity = 0.06 + 0.5 * (v / maxc) if v else 0
            base = {"POSITIVE": "86,130,84", "NEUTRAL": "164,133,48", "NEGATIVE": "164,66,55"}[t]
            bg = f"rgba({base},{intensity:.2f})" if v else "transparent"
            ink = "#000" if isd else "inherit"
            cells += f'<td style="background:{bg};color:{ink}">{v}</td>'
        rows.append(f"<tr><th>{t}</th>{cells}</tr>")
    # error column
    errcells = "".join(f'<td style="background:var(--err-bg)">{m["mat_errors"][t]}</td>' for t in CLASSES)
    return f"""<table class="cm">
    <tr><th></th><th style="text-align:center" colspan="3">predicted</th></tr>
    <tr><th></th>{''.join(f'<th>{c}</th>' for c in CLASSES)}<th>error</th></tr>
    {''.join(rows)}
    <tr><th></th>{errcells}</tr>
    </table>
    <div class="cap">rows = correct answer (from rating) · columns = model prediction · color = agreement strength</div>"""


def truth_vs_pred(m):
    col = {"POSITIVE": "pos", "NEUTRAL": "neu", "NEGATIVE": "neg"}
    tmax = max(m["truth"].values(), default=1)
    pmax = max(m["pred"].values(), default=1)
    rows = ""
    for c in CLASSES:
        tv = m["truth"][c]; pv = m["pred"][c]
        tp = tv / tmax * 100; pp = pv / pmax * 100
        ccol = {"POSITIVE": "var(--pos)", "NEUTRAL": "var(--neu)", "NEGATIVE": "var(--neg)"}[c]
        rows += f"""<div class="gr">
          <div class="g-label">{c} <span style="color:var(--muted);font-weight:400">({tv}/{pv})</span></div>
          <div class="g-tracks">
            <div class="b" title="truth {tv}" style="width:{tp:.1f}%;background:{ccol}">{tv}</div>
            <div class="b" title="predicted {pv}" style="width:{pp:.1f}%;background:var(--surface2);color:var(--ink);border:1px solid var(--line)">{pv}</div>
          </div>
        </div>"""
    return f"""<div class="grow">
      <div class="g-legend"><span>■ truth (from rating)</span><span style="color:var(--muted)">▢ predicted</span></div>
      {rows}
    </div>"""


def emo_block(m):
    def ord_keys(cnt):
        present = [e for e in EMOTIONS if cnt.get(e, 0)]
        for extra in ("none", "other"):
            if cnt.get(extra, 0):
                present.append(extra)
        return present

    def bars(cnt, color):
        mx = max(cnt.values(), default=1)
        out = ""
        for e in ord_keys(cnt):
            v = cnt.get(e, 0)
            if e in ("none", "other"):
                color_here = "#8A8A94"
            else:
                color_here = color
            out += f"""<div class="hbar-row">
              <div class="hbar-label">{e.title() if e in EMOTIONS else e}</div>
              <div class="hbar-track"><div class="hbar-fill" style="width:{v/mx*100:.1f}%;min-width:8px;background:{color_here}"></div></div>
              <div class="hbar-val">{v}<span class="hbar-sub">/{m['n']} · {v/m['n']*100:.0f}%</span></div></div>"""
        return out
    agree_pct = m["emo_agree"] / m["n"] * 100
    return f"""<div class="grid2">
      <div class="card">
        <div class="emo-h">LLM primary emotion</div>
        {bars(m["llm_emo"], "#3E7C4F")}
      </div>
      <div class="card">
        <div class="emo-h">NRC word-list primary emotion</div>
        {bars(m["nrc_emo"], "#C2561F")}
      </div>
    </div>
    <div class="verdict"><b>The two takes on emotion agree about {m['emo_agree']} of {m['n']} reviews ({agree_pct:.0f}%).</b>
      The LLM reads tone and nuance; the NRC word list simply counts words, so it over-favors
      “anticipation” (common purchase words like <em>get/will/worth</em> are tagged with it) and
      struggles with sarcasm or body-vs-title tension.</div>"""


def review_row(i, r):
    esc = html.escape
    star = int(r["rating"])
    cls = r["pred_sentiment"] if r["pred_sentiment"] in CLASSES else "ERR"
    truthcls = r["truth"]
    tpill = f'<span class="pill t">truth {truthcls}</span>'
    if cls == "ERR":
        ppill = '<span class="pill ERR">model ∅</span>'
    else:
        ppill = f'<span class="pill {cls}">pred {cls}</span>'
    conf = r.get("confidence")
    cs = f' · conf {conf:.2f}' if isinstance(conf, (int, float)) else ""
    le = esc((r.get("pred_emotion") or "—"))
    ne = esc((r.get("nrc_emotion") or "—"))
    title = esc(r.get("title") or "—")
    text = esc((r.get("text") or "").strip())
    body = f'<div class="rev-b" title="{text}">{text}</div>' if text else '<div class="rev-b">—</div>'
    return f"""<tr data-cls="{cls}" data-truth="{truthcls}" data-emo-llm="{(r.get('pred_emotion') or '').lower()}" data-emo-nrc="{(r.get('nrc_emotion') or '').lower()}" data-conf="{conf if isinstance(conf,(int,float)) else ''}">
      <td class="stars">{"★"*star + "☆"*(5-star)}</td>
      <td><div class="rev-t">{title}</div>{body}
        <em class="e">LLM {le} · NRC {ne}{cs}</em></td>
      <td>{tpill}<br>{ppill}</td>
    </tr>"""


def table_js(rows, n, correct_counts):
    data = []
    for r in rows:
        data.append({
            "cls": (r["pred_sentiment"] if r["pred_sentiment"] in CLASSES else "ERR"),
            "truth": r["truth"], "emo_llm": (r.get("pred_emotion") or "").lower(),
            "emo_nrc": (r.get("nrc_emotion") or "").lower(),
            "html": review_row(0, r),
        })
    data_json = json.dumps(data)
    return f"""<div class="filters">
      <select id="fCls"><option value="all">All predictions</option><option value="POSITIVE">POSITIVE</option>
        <option value="NEUTRAL">NEUTRAL</option><option value="NEGATIVE">NEGATIVE</option><option value="ERR">model error</option></select>
      <select id="fCorr"><option value="all">All outcomes</option><option value="ok">correct</option>
        <option value="bad">mismatched</option></select>
      <select id="fTruth"><option value="all">Any true class</option>
        <option value="POSITIVE">true POSITIVE</option><option value="NEUTRAL">true NEUTRAL</option><option value="NEGATIVE">true NEGATIVE</option></select>
      <select id="fEmo"><option value="all">Any emotion</option>{''.join(f'<option value="{e}">{e}</option>' for e in EMOTIONS)}</select>
      <input type="search" id="fQ" placeholder="Search review text…" autocomplete="off">
      <span class="count-badge"><b id="cnt">0</b> / {n} reviews</span>
    </div>
    <table class="rev"><thead><tr>
      <th style="width:64px">Rating</th><th>Review · title / body · emotions</th><th style="width:150px">Verdict</th>
    </tr></thead><tbody id="tbody"></tbody></table>
    <div id="noRows" style="display:none">No reviews match the current filters.</div>
    <script>
    (function(){{
      var rows = {data_json};
      var els = {{cls:document.getElementById('fCls'),corr:document.getElementById('fCorr'),
        truth:document.getElementById('fTruth'),emo:document.getElementById('fEmo'),
        q:document.getElementById('fQ'),tbody:document.getElementById('tbody'),
        cnt:document.getElementById('cnt'),no:document.getElementById('noRows')}};
      function filt(){{
        var out = rows.filter(function(r){{
          if (els.cls.value!=='all' && r.cls!==els.cls.value) return false;
          if (els.corr.value==='ok' && r.cls!==r.truth) return false;
          if (els.corr.value==='bad' && r.cls===r.truth) return false;
          if (els.truth.value!=='all' && r.truth!==els.truth.value) return false;
          if (els.emo.value!=='all' && r.emo_llm!==els.emo.value && r.emo_nrc!==els.emo.value) return false;
          var q=els.q.value.toLowerCase();
          if (q && (r.html.toLowerCase().indexOf(q)<0)) return false;
          return true;
        }});
        els.cnt.textContent = out.length;
        els.no.style.display = out.length?'none':'block';
        els.tbody.innerHTML = out.map(function(r){{return r.html;}}).join('');
      }}
      [els.cls,els.corr,els.truth,els.emo,els.q].forEach(function(e){{e.addEventListener('input',filt);}});
      filt();
    }})();
    </script>"""


def build(results_path, out_path):
    rows = load_results(results_path)
    m = compute_metrics(rows)
    full = full_dataset_distribution()

    star_rows = ""
    maxs = max(full.values())
    for s in (1, 2, 3, 4, 5):
        v = full[s]
        star_rows += bar(f"★{s} star", v, sum(full.values()), "#C2561F", max_value=maxs)

    html = HEAD.format(css=CSS)
    html += f"""
    <header class="mast">
      <p class="eyebrow">MBAX 6418 · Assignment 1</p>
      <h1>Amazon Gift-Card review sentiment</h1>
      <p class="sub">An LLM classifies a balanced sample of Amazon “Gift Cards” reviews as
      <b>POSITIVE</b> / <b>NEUTRAL</b> / <b>NEGATIVE</b> and names each review’s primary emotion.
      Its predictions are checked against the <em>star rating</em>, which the model never sees.</p>
      <div class="meta">
        <span><b>{m['n']}</b> scored reviews</span>
        <span>balanced across 3 classes</span>
        <span>fixed random seed</span>
        <span>model: <b>Qwen3.6-35B</b> (OpenAI-compatible endpoint)</span>
        <span>single-file dashboard · offline</span>
      </div>
    </header>
    {kpi_cards(m)}

    <section class="block">
      <div class="sec-head"><h2>The data</h2><span class="sec-note">full Gift-Cards category · 152,410 reviews</span></div>
      <div class="card">{star_rows}
        <div class="verdict"><b>Heavily skewed.</b> 84% of reviews are ★5. A model that
        “always says positive” would score near 85% — the lopsidedness must not be mistaken for skill.
        The scored sample is deliberately balanced so every class gets a fair test.</div>
      </div>
    </section>

    <section class="block">
      <div class="sec-head"><h2>Predictions vs the rating</h2><span class="sec-note">truth derived from stars</span></div>
      <div class="card">{truth_vs_pred(m)}
        <div class="verdict" style="margin-top:18px">{confusion(m)}</div>
      </div>
    </section>

    <section class="block">
      <div class="sec-head"><h2>Primary emotion</h2><span class="sec-note">LLM reading vs NRC word list</span></div>
      {emo_block(m)}
    </section>

    <section class="block">
      <div class="sec-head"><h2>Reviews</h2><span class="sec-note">filter the table · counts update live</span></div>
      <div class="card">{table_js(rows, m['n'], m['correct'])}</div>
    </section>

    <footer>
      <b>Data source:</b> Amazon Reviews '23 (“Gift Cards” category), McAuley Lab, UC San Diego —
      <a href="https://amazon-reviews-2023.github.io">amazon-reviews-2023.github.io</a>.
      Predictions from <b>Qwen3.6-35B-A3B-AWQ-4bit</b> via an OpenAI-compatible class endpoint.
      NRC word-list emotion take from the NRC Emotion Lexicon v0.92.
    </footer>
    </div></body></html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote dashboard -> {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="outputs/dashboard.html")
    args = ap.parse_args()
    build(args.results, args.out)
