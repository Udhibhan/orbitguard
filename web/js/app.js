import {
  TIERS, TIER_COLOR, SERIES, CUTOFF, OBJECTS, loadSummary, loadEvents, loadCdms, classifyLog, rulesFired,
  fmtPc, fmtKm, fmtPct, fmtInt, tierBadge, queueChip, esc, suggestAction, calibrateEncounter, pcAtDistance,
  safeSeparation, saveDecision, loadDecisions, lastDecision, decisionsCsv,
} from "./data.js";
import { timeChart, hBars, stackedBars, histogram, confusion, radar, pbar, ttRow } from "./charts.js";
import { buildFacts, requestBrief } from "./brief.js";

const view = document.getElementById("view");
const state = {
  split: readPref("split", "test"),
  summary: null,
  disposers: [],
  triage: { tiers: new Set(TIERS), queue: "All", obj: "All", q: "", sort: "risk", dir: -1, page: 0, escalating: false },
};

function readPref(k, d) { try { return localStorage.getItem(`orbitguard.${k}`) || d; } catch { return d; } }
function writePref(k, v) { try { localStorage.setItem(`orbitguard.${k}`, v); } catch { /* ignore */ } }
function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2200);
}
const TIER_RANK = { Safe: 0, Elevated: 1, Critical: 2 };

// ---------------------------------------------------------------- shell
const ICONS = {
  overview: '<path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0Zm9-5v5l3 3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  triage: '<path d="M4 6h16M4 12h10M4 18h6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  event: '<circle cx="12" cy="12" r="3" fill="currentColor"/><ellipse cx="12" cy="12" rx="9" ry="4" fill="none" stroke="currentColor" stroke-width="1.6" transform="rotate(-25 12 12)"/>',
  warnings: '<path d="M12 3 2 20h20L12 3Zm0 6v5m0 3v.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>',
  models: '<path d="M4 20V10m5 10V4m5 16v-7m5 7V8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  about: '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 11v6m0-9v.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
};
const PAGES = [
  ["overview", "Mission control"], ["triage", "Triage queue"], ["event", "Encounter"],
  ["warnings", "24-hour warnings"], ["models", "Model lab"], ["about", "Method & limits"],
];

function renderNav(active, counts = {}) {
  document.getElementById("links").innerHTML = PAGES.map(([k, label]) => {
    const href = k === "event" ? `#/event/${state.split}/${state.lastEvent || ""}` : `#/${k}`;
    return `<a href="${href}" class="${k === active ? "active" : ""}"><svg viewBox="0 0 24 24">${ICONS[k]}</svg>${label}${counts[k] != null ? `<span class="count">${counts[k]}</span>` : ""}</a>`;
  }).join("");
  document.querySelectorAll("#dsSwitch button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.ds === state.split)));
  document.getElementById("dsNote").innerHTML = state.split === "test"
    ? "Live view: the 2,167 test encounters as an analyst sees them 2 days before closest approach."
    : "Hindsight view: 1,659 held-out training encounters where the real outcome at TCA is known.";
}

function setCrumbs(html) { document.getElementById("crumbs").innerHTML = html; }

document.getElementById("dsSwitch").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-ds]");
  if (!b || b.dataset.ds === state.split) return;
  state.split = b.dataset.ds; writePref("split", state.split);
  state.triage.page = 0;
  const [page] = parseHash();
  if (page === "event") location.hash = "#/triage"; else route();
});
document.getElementById("menuBtn").addEventListener("click", () => document.getElementById("nav").classList.toggle("open"));
document.getElementById("jump").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const id = parseInt(document.getElementById("jumpInput").value, 10);
  if (!Number.isFinite(id)) return;
  for (const sp of [state.split, state.split === "test" ? "val" : "test"]) {
    const { byId } = await loadEvents(sp);
    if (byId.has(id)) { location.hash = `#/event/${sp}/${id}`; document.getElementById("jumpInput").blur(); return; }
  }
  toast(`Event ${id} is not in the live or hindsight sets (train events outside validation are used for fitting only).`);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "/" && !/input|textarea|select/i.test(document.activeElement.tagName)) { ev.preventDefault(); document.getElementById("jumpInput").focus(); }
});

function parseHash() { return (location.hash.replace(/^#\/?/, "") || "overview").split("/"); }

async function route() {
  state.disposers.forEach((d) => { try { d(); } catch { /* ignore */ } });
  state.disposers = [];
  document.getElementById("nav").classList.remove("open");
  const [page, a, b] = parseHash();
  if (!state.summary) state.summary = await loadSummary();
  const ev = await loadEvents(state.split);
  const counts = {
    triage: fmtInt(ev.rows.filter((e) => e.queue !== "Auto-clear").length),
    warnings: fmtInt(state.summary.warnings_24h.zones.under_1km),
  };
  view.innerHTML = "";
  view.scrollTop = 0; window.scrollTo(0, 0);
  try {
    if (page === "triage") { renderNav("triage", counts); await pageTriage(ev); }
    else if (page === "event") {
      const split = a === "val" || a === "test" ? a : state.split;
      if (split !== state.split) { state.split = split; writePref("split", split); }
      const set = await loadEvents(split);
      const id = parseInt(b, 10);
      state.lastEvent = Number.isFinite(id) ? id : topThreats(set.rows, 1)[0].id;
      renderNav("event", counts);
      await pageEvent(set, state.lastEvent, split);
    }
    else if (page === "warnings") { renderNav("warnings", counts); await pageWarnings(ev); }
    else if (page === "models") { renderNav("models", counts); pageModels(); }
    else if (page === "about") { renderNav("about", counts); pageAbout(); }
    else { renderNav("overview", counts); pageOverview(ev); }
  } catch (err) {
    console.error(err);
    view.innerHTML = `<div class="callout bad"><b>Could not render this page.</b> ${esc(err.message)}</div>`;
  }
  view.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);

function topThreats(rows, n) {
  return [...rows].sort((x, y) => TIER_RANK[y.fcTier] - TIER_RANK[x.fcTier] || y.pc - x.pc || y.pred - x.pred || x.miss - y.miss).slice(0, n);
}
function forecastRule(e, S) {
  const pc = 100 * e.pc, pe = 100 * (e.pe + e.pc), tc = 100 * S.tau_crit, te = 100 * S.tau_elev;
  if (e.pc >= S.tau_crit) return `F1 P(Critical) ${pc.toFixed(1)}% ≥ ${tc.toFixed(0)}% → Critical`;
  if (e.pe + e.pc >= S.tau_elev) return `F2 P(Elevated or worse) ${pe.toFixed(1)}% ≥ ${te.toFixed(0)}% → Elevated`;
  return `F3 P(Elevated or worse) ${pe.toFixed(1)}% < ${te.toFixed(0)}% → Safe`;
}
const splitLabel = () => (state.split === "test" ? "Live · T−2 days" : "Hindsight · validation");

// ---------------------------------------------------------------- overview
function pageOverview({ rows }) {
  const S = state.summary;
  setCrumbs(`<b>Mission control</b> · ${splitLabel()}`);
  const n = rows.length;
  const crit = rows.filter((e) => e.fcTier === "Critical").length;
  const elev = rows.filter((e) => e.fcTier === "Elevated").length;
  const review = rows.filter((e) => e.queue === "Analyst review").length;
  const cleared = rows.filter((e) => e.queue === "Auto-clear").length;
  const esc_ = rows.filter((e) => TIER_RANK[e.fcTier] > TIER_RANK[e.curTier]).length;
  const pce = S.pc_engine.test;
  const wl = S.forecast_metrics.models[S.model].workload;
  const mix = Object.fromEntries(OBJECTS.map((o) => [o, rows.filter((e) => e.object === o).length]));

  view.innerHTML = `
    <section class="hero">
      <div class="scene" id="heroScene"></div>
      <div class="hero-copy">
        <div class="eyebrow">Conjunction triage · ESA CDM telemetry</div>
        <h1>Every warning scored. <em>Only the dangerous ones</em> reach a human.</h1>
        <p>OrbitGuard reads each conjunction data message, estimates its collision probability, forecasts where the risk will land at closest approach two days early, and auto-clears what is provably safe.</p>
        <div class="hero-stats">
          <div><b>${fmtPct(wl.pct_cleared)}</b><span>of validation events auto-cleared</span></div>
          <div><b>${wl.critical_auto_cleared} / ${S.forecast_metrics.split.val_tier_counts.Critical}</b><span>Critical events wrongly cleared</span></div>
          <div><b>${fmtPct(pce.tier_accuracy)}</b><span>tier accuracy on ${fmtInt(pce.n_rows)} test CDMs</span></div>
        </div>
      </div>
      <div class="scene-tag">Schematic orbit view · particle mix = real secondary-object shares · red pulses = forecast Critical</div>
    </section>

    <div class="tiles">
      <div class="tile"><div class="label">Encounters in view</div><div class="value">${fmtInt(n)}</div><div class="foot">${splitLabel()}</div></div>
      <div class="tile"><div class="label">${tierBadge("Critical")} at TCA</div><div class="value">${fmtInt(crit)}</div><div class="foot">forecast · manoeuvre planning</div></div>
      <div class="tile"><div class="label">${tierBadge("Elevated")} at TCA</div><div class="value">${fmtInt(elev)}</div><div class="foot">forecast · keep watching</div></div>
      <div class="tile"><div class="label">Escalating</div><div class="value">${fmtInt(esc_)}</div><div class="foot">forecast tier above today's tier</div></div>
      <div class="tile"><div class="label">Analyst review</div><div class="value">${fmtInt(review)}</div><div class="foot">volatile history, never auto-cleared</div></div>
      <div class="tile feature"><div class="label">Auto-clear</div><div class="value">${fmtPct(cleared / n)}</div><div class="foot">${fmtInt(cleared)} encounters off the desk</div><div class="bar-mini" style="width:${(100 * cleared / n).toFixed(1)}%"></div></div>
    </div>

    <div class="grid g-12 mt">
      <div class="card span-7">
        <div class="card-head"><div><h2>Conjunction radar</h2><p>Each dot is one encounter's predicted miss point around your satellite, in the radial / along-track plane. Distance from centre is the miss distance (square-root scale); colour is the forecast tier at TCA. Click a dot to open it.</p></div></div>
        <div class="chart" id="radar"></div>
        <div class="legend"><span><i class="dot" style="background:${TIER_COLOR.Critical}"></i>Critical</span><span><i class="dot" style="background:${TIER_COLOR.Elevated}"></i>Elevated</span><span><i class="dot" style="background:${TIER_COLOR.Safe}"></i>Safe</span><span>dashed rings: 1 km Critical and 5 km Elevated rules</span></div>
      </div>
      <div class="span-5 stack">
        <div class="card">
          <div class="card-head"><div><h2>Top threats</h2><p>Ranked by forecast tier, then P(Critical at TCA)</p></div><a class="btn sm" href="#/triage">Open queue →</a></div>
          <div class="threats" id="threats"></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h2>Where the workload goes</h2><p>Policy: clear if P(Safe at TCA) ≥ θ = ${S.theta.toFixed(3)} and the history is not volatile</p></div></div>
          <div class="chart" id="workload"></div>
        </div>
      </div>
    </div>`;

  const scene = document.getElementById("heroScene");
  import("./scene3d.js").then(({ heroOrbit }) => {
    if (!document.body.contains(scene)) return;
    try { const h = heroOrbit(scene, { mix, nCritical: crit }); state.disposers.push(h.dispose); } catch (e) { console.warn("WebGL unavailable", e); }
  });
  radar(document.getElementById("radar"), rows, { onClick: (e) => (location.hash = `#/event/${state.split}/${e.id}`), maxKm: 30 });
  document.getElementById("threats").innerHTML = topThreats(rows, 6).map((e) => `
    <a class="threat" href="#/event/${state.split}/${e.id}">
      ${tierBadge(e.fcTier)}
      <div class="t-main"><b>Event ${e.id}</b> <span class="chip">${esc(e.object)}</span><div>miss ${fmtKm(e.miss)} km · Pc ${fmtPc(e.risk)} · now ${e.curTier} · ${e.queue}</div></div>
      <div class="t-p">${(100 * e.pc).toFixed(0)}%<small>P(Critical)</small></div>
    </a>`).join("");
  hBars(document.getElementById("workload"), [
    { label: "Manoeuvre queue", value: rows.filter((e) => e.queue === "Queue").length, color: SERIES.blue },
    { label: "Analyst review", value: review, color: SERIES.violet },
    { label: "Auto-cleared", value: cleared, color: TIER_COLOR.Safe },
  ].map((d) => ({ ...d, text: `${fmtInt(d.value)} · ${fmtPct(d.value / n, 0)}` })), { labelW: 130 });
}

// ---------------------------------------------------------------- triage
function pageTriage({ rows }) {
  const T = state.triage;
  setCrumbs(`<b>Triage queue</b> · ${splitLabel()}`);
  const hind = state.split === "val";
  view.innerHTML = `
    <div class="page-head">
      <div><div class="eyebrow">Analyst desk</div><h1>Triage queue</h1>
      <p>Sorted most dangerous first. The probability strip shows P(Safe / Elevated / Critical) at closest approach. Tiers always come from the handbook rules applied to the forecast; the model only supplies numbers.</p></div>
      <div class="row"><button class="btn" id="csvBtn">Export decisions CSV</button><a class="btn" href="data/test_predictions_rows.csv" download>Test predictions (rows)</a></div>
    </div>
    <div class="filters">
      <div class="pillset" id="tierPills">${TIERS.slice().reverse().map((t) => `<button class="pill" aria-pressed="${T.tiers.has(t)}" data-t="${t}"><span class="dot" style="background:${TIER_COLOR[t]}"></span>${t}</button>`).join("")}</div>
      <select class="input" id="qSel">${["All", "Queue", "Analyst review", "Auto-clear"].map((q) => `<option ${q === T.queue ? "selected" : ""}>${q}</option>`).join("")}</select>
      <select class="input" id="oSel">${["All", ...OBJECTS].map((o) => `<option ${o === T.obj ? "selected" : ""}>${o}</option>`).join("")}</select>
      <input class="input" id="qInput" placeholder="Filter event id" value="${esc(T.q)}" style="width:140px">
      <label class="toggle"><input type="checkbox" id="escT" ${T.escalating ? "checked" : ""}>Escalating only</label>
    </div>
    <div class="card flush"><div class="table-wrap" id="tbl"></div><div class="pager" id="pager"></div></div>`;

  const cols = [
    ["id", "Event", (e) => `<b>${e.id}</b>`, (e) => e.id],
    ["fc", "Forecast @TCA", (e) => tierBadge(e.fcTier), (e) => TIER_RANK[e.fcTier]],
    ["cur", "Now", (e) => tierBadge(e.curTier), (e) => TIER_RANK[e.curTier]],
    ...(hind ? [["ft", "Actual @TCA", (e) => tierBadge(e.finalTier), (e) => TIER_RANK[e.finalTier]]] : []),
    ["p", "P(tier at TCA)", (e) => pbar(e), (e) => e.pc],
    ["risk", "P(Critical)", (e) => `<span class="num">${(100 * e.pc).toFixed(1)}%</span>`, (e) => e.pc * 10 + TIER_RANK[e.fcTier] * 100 + e.pred / 100, "num"],
    ["pe", "P(Elev+)", (e) => `<span class="num">${(100 * (e.pe + e.pc)).toFixed(1)}%</span>`, (e) => e.pe + e.pc, "num"],
    ["lr", "Pc now", (e) => `<span class="num">${fmtPc(e.risk)}</span>`, (e) => e.risk, "num"],
    ["miss", "Miss km", (e) => `<span class="num">${fmtKm(e.miss)}</span>`, (e) => -e.miss, "num"],
    ["obj", "Object", (e) => esc(e.object), (e) => e.object],
    ["t", "T−days", (e) => `<span class="num">${e.t.toFixed(2)}</span>`, (e) => -e.t, "num"],
    ["q", "Route", (e) => queueChip(e.queue), (e) => e.q],
    ["dec", "Decision", (e) => { const d = lastDecision(state.split, e.id); return d ? `<span class="chip accent">${d.decision}</span>` : ""; }, () => 0],
  ];
  const draw = () => {
    let v = rows.filter((e) => T.tiers.has(e.fcTier) && (T.queue === "All" || e.queue === T.queue) && (T.obj === "All" || e.object === T.obj));
    if (T.q.trim()) v = v.filter((e) => String(e.id).includes(T.q.trim()));
    if (T.escalating) v = v.filter((e) => TIER_RANK[e.fcTier] > TIER_RANK[e.curTier]);
    const col = cols.find((c) => c[0] === T.sort) || cols[5];
    v.sort((a, b) => (col[3](a) < col[3](b) ? -1 : col[3](a) > col[3](b) ? 1 : 0) * T.dir);
    const per = 50, pages = Math.max(1, Math.ceil(v.length / per));
    T.page = Math.min(T.page, pages - 1);
    const slice = v.slice(T.page * per, T.page * per + per);
    document.getElementById("tbl").innerHTML = `<table class="data"><thead><tr>${cols.map((c) => `<th class="sortable ${c[4] || ""}" data-k="${c[0]}">${c[1]}${T.sort === c[0] ? `<span class="arrow">${T.dir < 0 ? "↓" : "↑"}</span>` : ""}</th>`).join("")}</tr></thead>
      <tbody>${slice.map((e) => `<tr data-id="${e.id}">${cols.map((c) => `<td class="${c[4] || ""}">${c[2](e)}</td>`).join("")}</tr>`).join("") || `<tr><td colspan="${cols.length}" class="empty">No encounters match these filters.</td></tr>`}</tbody></table>`;
    document.getElementById("pager").innerHTML = `<span>${fmtInt(v.length)} encounters · page ${T.page + 1} of ${pages}</span>
      <span class="row"><button class="btn sm" id="prev" ${T.page === 0 ? "disabled" : ""}>← Prev</button><button class="btn sm" id="next" ${T.page >= pages - 1 ? "disabled" : ""}>Next →</button></span>`;
    document.getElementById("prev").onclick = () => { T.page--; draw(); };
    document.getElementById("next").onclick = () => { T.page++; draw(); };
    document.querySelectorAll("#tbl th[data-k]").forEach((th) => (th.onclick = () => {
      if (T.sort === th.dataset.k) T.dir *= -1; else { T.sort = th.dataset.k; T.dir = -1; }
      draw();
    }));
    document.querySelectorAll("#tbl tbody tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = `#/event/${state.split}/${tr.dataset.id}`)));
  };
  document.getElementById("tierPills").onclick = (ev) => {
    const b = ev.target.closest("[data-t]"); if (!b) return;
    T.tiers.has(b.dataset.t) ? T.tiers.delete(b.dataset.t) : T.tiers.add(b.dataset.t);
    b.setAttribute("aria-pressed", String(T.tiers.has(b.dataset.t))); T.page = 0; draw();
  };
  document.getElementById("qSel").onchange = (e) => { T.queue = e.target.value; T.page = 0; draw(); };
  document.getElementById("oSel").onchange = (e) => { T.obj = e.target.value; T.page = 0; draw(); };
  document.getElementById("qInput").oninput = (e) => { T.q = e.target.value; T.page = 0; draw(); };
  document.getElementById("escT").onchange = (e) => { T.escalating = e.target.checked; T.page = 0; draw(); };
  document.getElementById("csvBtn").onclick = () => {
    const n = loadDecisions().length;
    if (!n) { toast("No decisions recorded in this browser yet."); return; }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([decisionsCsv()], { type: "text/csv" }));
    a.download = "orbitguard_decisions.csv"; a.click();
  };
  draw();
}

// ---------------------------------------------------------------- event detail
async function pageEvent(set, id, split) {
  const e = set.byId.get(id);
  if (!e) { view.innerHTML = `<div class="callout bad">Event ${id} is not in this dataset.</div>`; return; }
  setCrumbs(`<a href="#/triage">Triage</a> / <b>Event ${id}</b> · ${split === "test" ? "Live · T−2 days" : "Hindsight · validation"}`);
  view.innerHTML = `<div class="loading"><div class="spinner"></div>Loading CDM history…</div>`;
  const cdms = await loadCdms(split, id);
  const pre = cdms.filter((c) => c.t >= CUTOFF);
  const post = cdms.filter((c) => c.t < CUTOFF);
  const latest = pre[pre.length - 1] || cdms[cdms.length - 1];
  const S = state.summary;
  const hind = split === "val";
  const facts = buildFacts(e, split, latest, forecastRule(e, S));
  const [action] = suggestAction(e.fcTier, e.curTier, !!e.vol, e.queue === "Auto-clear");
  const idx = set.rows.indexOf(e);
  const sorted = topThreats(set.rows, set.rows.length);
  const pos = sorted.indexOf(e);
  const prevId = pos > 0 ? sorted[pos - 1].id : null, nextId = pos < sorted.length - 1 ? sorted[pos + 1].id : null;

  view.innerHTML = `
    <div class="ev-head">
      <div class="ev-title">
        <h1>Event ${e.id}</h1>
        ${tierBadge(e.curTier, "lg")}<span class="ev-arrow">→</span>${tierBadge(e.fcTier, "lg")}
        ${queueChip(e.queue)} <span class="chip">${esc(e.object)}</span>
        ${hind ? `<span class="chip accent">actual at TCA: ${e.finalTier}</span>` : ""}
      </div>
      <div class="row">
        ${prevId != null ? `<a class="btn sm" href="#/event/${split}/${prevId}">← Higher risk</a>` : ""}
        ${nextId != null ? `<a class="btn sm" href="#/event/${split}/${nextId}">Lower risk →</a>` : ""}
      </div>
    </div>
    <div class="card"><div class="kv">
      <div><span>Now (T−${e.t.toFixed(2)} d)</span><b>Pc ${fmtPc(e.risk)}</b></div>
      <div><span>P(Elevated or worse)</span><b>${(100 * (e.pe + e.pc)).toFixed(1)}%</b></div>
      <div><span>P(Critical at TCA)</span><b>${(100 * e.pc).toFixed(1)}%</b></div>
      <div><span>Miss distance</span><b>${fmtKm(e.miss)} km</b></div>
      <div><span>Relative speed</span><b>${(e.speed / 1000).toFixed(2)} km/s</b></div>
      <div><span>Pc Engine (latest CDM)</span><b>${fmtPc(latest.est)}</b></div>
      <div><span>CDMs before decision</span><b>${e.n}</b></div>
      <div><span>Suggested action</span><b>${action}</b></div>
    </div></div>

    <div class="grid g-12 mt">
      <div class="card span-7 flush">
        <div class="card-head"><div><h2>Encounter geometry</h2><p>Your satellite at the centre, in its own radial / along-track / cross-track frame. Blue dots are where each CDM predicted the other object would pass; the red shell is the 1 km Critical zone and the amber shell the 5 km Elevated zone.</p></div></div>
        <div class="scene" id="enc" style="height:520px">
          <div class="scene-tag">Drag to orbit · scroll to zoom · fly-by line is schematic (relative velocity ⟂ miss vector)</div>
          <div class="scene-legend"><span><i style="background:#3987e5"></i>CDM miss estimates</span><span><i style="background:#ff6b6b"></i>Secondary at selected CDM</span><span><i style="background:#9085e9"></i>1σ radial uncertainty</span></div>
          <div class="scene-ui"><button class="btn sm" id="play">▶ Replay CDMs</button><input type="range" class="range" id="cdmSlider" min="0" max="${cdms.length - 1}" value="${pre.length - 1}"><span class="scene-readout" id="readout"></span></div>
        </div>
      </div>
      <div class="span-5 stack">
        <div class="card">
          <div class="card-head"><div><h2>Briefing</h2><p>Facts computed in code; Claude only phrases them</p></div><span id="briefSrc"></span></div>
          <div id="brief"><div class="loading" style="padding:20px 0"><div class="spinner"></div>Writing briefing…</div></div>
          <div class="suggest"><span>Suggested</span><span class="what">${action}</span><span class="sub">${esc(facts.suggestion_rule)}</span></div>
          <input class="input mt" id="note" placeholder="Note for the log (optional)" style="width:100%">
          <div class="decision-row">
            <button class="btn act" data-d="Act">Act · plan manoeuvre</button>
            <button class="btn watch" data-d="Watch">Watch</button>
            <button class="btn clear" data-d="Clear">Clear</button>
          </div>
          <div class="decision-log" id="declog"></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h2>Manoeuvre what-if</h2><p>Add separation at TCA and see Pc and tier respond</p></div></div>
          <input type="range" class="range" id="dv" min="0" max="20000" step="50" value="0" style="width:100%">
          <div class="whatif-out" id="wiOut"></div>
          <p class="note" id="wiNote"></p>
        </div>
      </div>
    </div>

    <div class="grid g-2 mt">
      <div class="card">
        <div class="card-head"><div><h2>Collision probability over time</h2><p>log10 Pc per CDM. The star carries the latest Pc to TCA (the ESA benchmark); the tier forecast comes from the probabilities.</p></div>
        ${hind && post.length ? `<label class="toggle"><input type="checkbox" id="hindT">Reveal after cutoff</label>` : ""}</div>
        <div class="chart" id="pcChart"></div>
        <div class="legend" id="pcLegend"></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h2>Miss distance over time</h2><p>km per CDM; shaded bands are the miss-distance rules</p></div></div>
        <div class="chart" id="missChart"></div>
        <div class="legend"><span><i style="background:${SERIES.blue}"></i>Before the 2-day cutoff</span>${hind ? `<span><i class="dash" style="color:#898781;background:#898781"></i>After cutoff (hindsight)</span>` : ""}</div>
      </div>
    </div>

    <div class="grid g-12 mt">
      <div class="card span-7">
        <div class="card-head"><div><h2>Decision trace</h2><p>Every step is a rule or a model output you can audit</p></div></div>
        <div class="trace" id="trace"></div>
      </div>
      <div class="card span-5">
        <div class="card-head"><div><h2>Pc Engine per CDM</h2><p>P(Critical) for each message, computed from telemetry only (never sees the reported Pc)</p></div></div>
        <div class="chart" id="pcritChart"></div>
      </div>
    </div>`;

  // --- 3D encounter
  const readout = document.getElementById("readout");
  const setReadout = (i) => {
    const c = cdms[i];
    readout.textContent = `CDM ${i + 1}/${cdms.length} · T−${c.t.toFixed(2)} d · miss ${fmtKm(c.miss)} km · Pc ${fmtPc(c.risk)}${c.t < CUTOFF ? " · hindsight" : ""}`;
  };
  setReadout(pre.length - 1);
  const slider = document.getElementById("cdmSlider");
  if (!hind) slider.max = String(pre.length - 1);
  import("./scene3d.js").then(({ encounterScene }) => {
    const host = document.getElementById("enc");
    if (!host) return;
    let sc;
    try {
      sc = encounterScene(host, hind ? cdms : pre, { height: 520, onIndex: (i) => { setReadout(i); slider.value = i; } });
    } catch (err) { host.insertAdjacentHTML("beforeend", `<div class="empty">3D view needs WebGL (${esc(err.message)})</div>`); return; }
    sc.setIndex(pre.length - 1);
    state.disposers.push(sc.dispose);
    slider.oninput = () => sc.setIndex(+slider.value);
    let timer = null;
    document.getElementById("play").onclick = (ev) => {
      if (timer) { clearInterval(timer); timer = null; ev.target.textContent = "▶ Replay CDMs"; return; }
      ev.target.textContent = "❚❚ Pause";
      let i = 0; sc.setIndex(0);
      const last = +slider.max;
      timer = setInterval(() => { i++; if (i > last) { clearInterval(timer); timer = null; ev.target.textContent = "▶ Replay CDMs"; return; } sc.setIndex(i); }, 650);
    };
    state.disposers.push(() => timer && clearInterval(timer));
  });

  // --- charts
  const xMax = Math.max(7, Math.ceil(Math.max(...cdms.map((c) => c.t))));
  const LO = -20;
  const pcPts = (list, key) => list.map((c) => ({ x: c.t, y: Math.max(c[key], LO), raw: c[key], floor: c[key] < LO, c }));
  const tierBands = [
    { y0: -4, y1: 0, color: TIER_COLOR.Critical, label: "Critical (Pc ≥ 1e-4)" },
    { y0: -6, y1: -4, color: TIER_COLOR.Elevated, label: "Elevated" },
    { y0: LO, y1: -6, color: TIER_COLOR.Safe, label: "Safe by Pc" },
  ];
  const drawPc = (showPost) => {
    const series = [
      { name: "Reported Pc (CDM)", color: SERIES.blue, data: pcPts(pre, "risk") },
      { name: "Pc Engine estimate", color: SERIES.aqua, dash: "5 4", data: pcPts(hind && showPost ? cdms : pre, "est") },
      { name: "Carry-forward", color: "#f0efec", dash: "2 4", markers: false, data: [{ x: latest.t, y: Math.max(latest.risk, LO) }, { x: 0, y: Math.max(e.pred, LO) }] },
      { name: "LRP: Pc carried to TCA", color: "#f0efec", star: true, data: [{ x: 0, y: Math.max(e.pred, LO), raw: e.pred }] },
    ];
    if (hind && showPost) series.splice(1, 0, { name: "Reported after cutoff", color: "#898781", dash: "3 3", data: pcPts([latest, ...post], "risk") });
    timeChart(document.getElementById("pcChart"), {
      height: 270, x: [xMax, 0], y: [LO, 0], xLabel: "days to closest approach", yLabel: "log10 Pc",
      bands: tierBands, series, vlines: [{ x: CUTOFF, label: "decision cutoff" }],
      tipFmt: (v) => fmtPc(v), tipExtra: (p) => (p.c ? ttRow("miss", `${fmtKm(p.c.miss)} km`) + ttRow("CDM row", p.c.row) : ""),
      tipTitle: (p) => (p.x === 0 ? "At TCA (LRP carry-forward)" : `T−${p.x.toFixed(2)} days`),
    });
    document.getElementById("pcLegend").innerHTML = series.filter((s) => s.name !== "Carry-forward").map((s) =>
      `<span><i class="${s.star ? "dot" : s.dash ? "dash" : ""}" style="background:${s.color};color:${s.color}"></i>${s.name}</span>`).join("") + `<span>▼ below 1e-20 (floor values reach 1e-30)</span>`;
  };
  drawPc(false);
  const hT = document.getElementById("hindT");
  if (hT) hT.onchange = () => { drawPc(hT.checked); drawMiss(hT.checked); };
  const drawMiss = (showPost) => {
    const list = hind && showPost ? cdms : pre;
    const maxKm = Math.max(6, ...list.map((c) => c.miss / 1000)) * 1.1;
    const s = [{ name: "Miss distance", color: SERIES.blue, data: pre.map((c) => ({ x: c.t, y: c.miss / 1000, c })) }];
    if (hind && showPost) s.push({ name: "After cutoff", color: "#898781", dash: "3 3", data: [latest, ...post].map((c) => ({ x: c.t, y: c.miss / 1000, c })) });
    timeChart(document.getElementById("missChart"), {
      height: 270, x: [xMax, 0], y: [0, maxKm], xLabel: "days to closest approach", yLabel: "km",
      bands: [{ y0: 0, y1: 1, color: TIER_COLOR.Critical, label: "< 1 km Critical" }, { y0: 1, y1: 5, color: TIER_COLOR.Elevated, label: "1–5 km Elevated" }],
      series: s, vlines: [{ x: CUTOFF, label: "decision cutoff" }], tipFmt: (v) => `${v.toFixed(3)} km`,
      tipExtra: (p) => (p.c ? ttRow("reported Pc", fmtPc(p.c.risk)) : ""),
    });
  };
  drawMiss(false);
  const pcList = hind ? cdms : pre;
  timeChart(document.getElementById("pcritChart"), {
    height: 230, x: [xMax, 0], y: [0, 1], xLabel: "days to closest approach", yLabel: "P(Critical)", yFmt: (v) => `${Math.round(v * 100)}%`,
    series: [{ name: "P(Critical) per CDM", color: SERIES.violet, data: pcList.map((c) => ({ x: c.t, y: c.pcrit, c })) }],
    vlines: [{ x: CUTOFF, label: "cutoff" }], tipFmt: (v) => `${(100 * v).toFixed(1)}%`,
    tipExtra: (p) => ttRow("engine tier", p.c.etierName) + ttRow("rule tier (reported Pc)", p.c.tier),
  });

  // --- decision trace
  const cur = rulesFired(e.risk, e.miss);
  const steps = [
    ["Telemetry ingested", `${e.n} CDMs at or before T−2 d`, `latest = ${split === "test" ? "test" : "train"}_data.csv row ${latest.row}; 13 whitelisted columns only`],
    ["Pc Engine", tierBadge(latest.etierName), `estimated Pc ${fmtPc(latest.est)} vs reported ${fmtPc(latest.risk)} · P(Critical) ${(100 * latest.pcrit).toFixed(1)}%`],
    ["Current tier", tierBadge(e.curTier), cur.map((r) => `${r.code} ${r.text}`).join(" · ")],
    ["Forecast at TCA", tierBadge(e.fcTier), `${forecastRule(e, S)} · P(Safe/Elev/Crit) = ${(100 * e.ps).toFixed(1)} / ${(100 * e.pe).toFixed(1)} / ${(100 * e.pc).toFixed(1)}%`],
    ["Volatility check", e.vol ? `<span class="chip review">Analyst review</span>` : `<span class="chip">stable</span>`, e.vol || `risk std ${e.rstd.toFixed(2)}, rise ${e.rrise.toFixed(2)} orders, miss ratio ${e.mratio.toFixed(2)} (all within limits)`],
    ["Auto-clear policy", queueChip(e.queue), `P(Safe) ${e.ps.toFixed(3)} ${e.ps >= S.theta ? "≥" : "<"} θ ${S.theta.toFixed(3)}${e.vol ? " · volatile events are never cleared" : ""}`],
    ["Suggested action", `<b>${action}</b>`, facts.suggestion_rule],
  ];
  if (hind) {
    const fin = rulesFired(e.frisk, e.fmiss);
    steps.push(["Hindsight: actual final CDM", tierBadge(e.finalTier), `row ${e.frow} at T−${e.ftt.toFixed(2)} d: Pc ${fmtPc(e.frisk)}, miss ${fmtKm(e.fmiss)} km · ${fin.map((r) => r.code).join(", ")}`]);
  }
  document.getElementById("trace").innerHTML = steps.map((s, i) => `<div class="trace-step"><div class="n">${i + 1}</div><div><div class="what">${s[0]} ${s[1]}</div><div class="why">${esc(s[2])}</div></div></div>`).join("");

  // --- what-if
  const cal = calibrateEncounter(e.miss, e.risk);
  const need = Math.max(0, safeSeparation(cal) - e.miss);
  const dv = document.getElementById("dv");
  dv.max = String(Math.max(6000, Math.ceil((need * 1.5) / 500) * 500));
  const drawWi = () => {
    const add = +dv.value, d = e.miss + add, lp = pcAtDistance(cal, d), t = classifyLog(lp, d);
    document.getElementById("wiOut").innerHTML = `
      <div><span>Miss at TCA</span><b>${fmtKm(d)} km</b></div>
      <div><span>Pc</span><b>${fmtPc(lp)}</b></div>
      <div><span>Tier</span>${tierBadge(t)}</div>`;
    document.getElementById("wiNote").innerHTML = `+${fmtKm(add, 2)} km of separation. Reaching <b>Safe</b> needs about <b>+${fmtKm(need, 2)} km</b> (miss > 5 km and Pc < 1e-6). Linear-Gaussian encounter calibrated to this event's current Pc ${fmtPc(e.risk)} (effective σ ${fmtM(cal.sigma)}): planning intuition, not a manoeuvre design.`;
  };
  dv.oninput = drawWi; drawWi();

  // --- briefing + decisions
  const drawLog = () => {
    const mine = loadDecisions().filter((d) => d.dataset === split && d.event_id === e.id);
    document.getElementById("declog").innerHTML = mine.slice(-4).reverse().map((d) =>
      `<div>📝 <b>${d.decision}</b> · ${new Date(d.timestamp_utc).toLocaleString()}${d.note ? ` · ${esc(d.note)}` : ""}</div>`).join("");
  };
  drawLog();
  document.querySelectorAll(".decision-row [data-d]").forEach((b) => (b.onclick = () => {
    saveDecision({ timestamp_utc: new Date().toISOString(), dataset: split, event_id: e.id, decision: b.dataset.d,
      suggested_action: action, current_tier: e.curTier, forecast_tier: e.fcTier, p_critical: e.pc, note: document.getElementById("note").value });
    toast(`Logged ${b.dataset.d} for event ${e.id}`);
    drawLog();
  }));
  requestBrief(facts).then((res) => {
    const host = document.getElementById("brief");
    if (!host) return;
    document.getElementById("briefSrc").innerHTML = res.source === "claude"
      ? `<span class="chip accent">Claude · verified</span>` : `<span class="chip">Deterministic template</span>`;
    host.innerHTML = `<div class="brief">${esc(res.text)}</div>${(res.problems || []).map((p) => `<p class="note">ℹ ${esc(p)}</p>`).join("")}`;
  });
  void idx;
}

// ---------------------------------------------------------------- 24-hour warnings
async function pageWarnings({ rows }) {
  const S = state.summary, W = S.warnings_24h, I = S.insights_24h;
  setCrumbs(`<b>24-hour warnings</b>`);
  const live = (await loadEvents("test")).rows;
  const liveCrit = live.filter((e) => e.fcTier === "Critical");
  const liveUnder1 = live.filter((e) => e.miss < 1000).length;
  const liveUnder5 = live.filter((e) => e.miss <= 5000).length;
  const cats = W.hour_bins.slice(0, -1).map((h, i) => `${h}–${W.hour_bins[i + 1]}h`);
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Inside the last day</div><h1>24-hour warnings</h1>
      <p>How many encounters come inside the dangerous distances in the final 24 hours before closest approach, and what drives the alerts. The 24-hour frame is measured on the ${fmtInt(S.counts.train_events)} training events (the test file stops 2 days out); the live panel projects the test set forward with the forecaster.</p></div></div>

    <div class="tiles">
      <div class="tile"><div class="label">Encounters with a CDM &lt; 24 h</div><div class="value">${fmtInt(W.n_events)}</div><div class="foot">${fmtInt(W.n_cdms)} messages issued in that window</div></div>
      <div class="tile"><div class="label">${tierBadge("Critical")} &lt; 1 km</div><div class="value">${fmtInt(W.zones.under_1km)}</div><div class="foot">${fmtPct(W.zones.under_1km / W.n_events)} of 24 h encounters</div></div>
      <div class="tile"><div class="label">${tierBadge("Elevated")} 1–5 km</div><div class="value">${fmtInt(W.zones.km_1_to_5)}</div><div class="foot">${fmtPct(W.zones.km_1_to_5 / W.n_events)} of 24 h encounters</div></div>
      <div class="tile"><div class="label">Final tier Critical</div><div class="value">${fmtInt(I.by_tier.Critical)}</div><div class="foot">Pc rule or miss rule</div></div>
      <div class="tile feature"><div class="label">Live set → forecast Critical at TCA</div><div class="value">${fmtInt(liveCrit.length)}</div><div class="foot">of ${fmtInt(live.length)} test encounters · ${fmtInt(liveUnder1)} already &lt; 1 km, ${fmtInt(liveUnder5)} ≤ 5 km</div></div>
    </div>

    <div class="grid g-12 mt">
      <div class="card span-7">
        <div class="card-head"><div><h2>When the last-day warnings land</h2><p>Encounters by hours to TCA of their latest CDM inside 24 h, stacked by tier</p></div></div>
        <div class="chart" id="hourChart"></div>
        <div class="legend">${TIERS.slice().reverse().map((t) => `<span><i class="dot" style="background:${TIER_COLOR[t]}"></i>${t}</span>`).join("")}</div>
      </div>
      <div class="card span-5">
        <div class="card-head"><div><h2>Generated insights</h2><p>Every sentence computed in code from the data</p></div></div>
        <div class="insights">${I.insights.map((s, i) => `<div class="insight"><div class="ico">${i + 1}</div><div>${esc(s)}</div></div>`).join("")}</div>
      </div>
    </div>

    <div class="grid g-12 mt">
      <div class="card span-7">
        <div class="card-head"><div><h2>Miss distance inside 24 h</h2><p>Latest CDM per encounter; log scale. Lines mark the 1 km and 5 km rules.</p></div></div>
        <div class="chart" id="missHist"></div>
      </div>
      <div class="card span-5">
        <div class="card-head"><div><h2>What triggers a Critical flag</h2><p>Critical encounters inside 24 h by the rule that fired</p></div></div>
        <div class="chart" id="drivers"></div>
        <h3 class="mt">Critical rate by object type</h3>
        <div class="chart mt" id="objRate"></div>
      </div>
    </div>

    <div class="card mt">
      <div class="card-head"><div><h2>Live watchlist: forecast Critical at TCA</h2><p>Test encounters the forecaster expects to end Critical, closest first. These are the manoeuvre candidates for the next 2 days.</p></div></div>
      <div class="table-wrap"><table class="data"><thead><tr><th>Event</th><th>Now</th><th>Forecast @TCA</th><th class="num">Miss km</th><th class="num">P(Critical)</th><th class="num">Pc now</th><th>Object</th><th>Route</th></tr></thead>
      <tbody>${liveCrit.sort((a, b) => a.miss - b.miss).slice(0, 25).map((e) => `<tr data-id="${e.id}"><td><b>${e.id}</b></td><td>${tierBadge(e.curTier)}</td><td>${tierBadge(e.fcTier)}</td><td class="num">${fmtKm(e.miss)}</td><td class="num">${(100 * e.pc).toFixed(1)}%</td><td class="num">${fmtPc(e.risk)}</td><td>${esc(e.object)}</td><td>${queueChip(e.queue)}</td></tr>`).join("")}</tbody></table></div>
    </div>`;
  stackedBars(document.getElementById("hourChart"), cats,
    TIERS.map((t) => ({ name: t, color: TIER_COLOR[t], values: W.hist_by_tier[t] })), { height: 260, xLabel: "hours to TCA", yLabel: "encounters" });
  const edges = W.miss_log_hist.edges;
  histogram(document.getElementById("missHist"), edges, W.miss_log_hist.counts, {
    height: 240, xLabel: "miss distance", ticks: [1, 2, 3, 4, 5], xFmt: (v) => ({ 1: "10 m", 2: "100 m", 3: "1 km", 4: "10 km", 5: "100 km" }[v] || ""),
    colorAt: (m) => (m < 3 ? TIER_COLOR.Critical : m <= Math.log10(5000) ? TIER_COLOR.Elevated : SERIES.blue),
    vlines: [{ x: 3, color: TIER_COLOR.Critical, label: "1 km" }, { x: Math.log10(5000), color: TIER_COLOR.Elevated, label: "5 km" }],
    binLabel: (a, b) => `${fmtM(10 ** a)} – ${fmtM(10 ** b)}`,
  });
  const dr = I.critical_drivers;
  hBars(document.getElementById("drivers"), [
    { label: "miss < 1 km only", value: dr.miss_rule_only, color: TIER_COLOR.Critical },
    { label: "Pc ≥ 1e-4 only", value: dr.pc_rule_only, color: SERIES.violet },
    { label: "both rules", value: dr.both, color: SERIES.orange },
  ], { labelW: 120 });
  hBars(document.getElementById("objRate"), I.by_object_type.map((o) => ({
    label: o.c_object_type, value: o.critical_rate, color: SERIES.blue, text: `${(100 * o.critical_rate).toFixed(1)}% of ${fmtInt(o.events)}`,
  })), { labelW: 110 });
  document.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = `#/event/test/${tr.dataset.id}`)));
  void rows;
}
const fmtM = (m) => (m >= 1000 ? `${(m / 1000).toFixed(m >= 10000 ? 0 : 1)} km` : `${Math.round(m)} m`);

// ---------------------------------------------------------------- model lab
function pageModels() {
  const S = state.summary, P = S.pc_engine, F = S.forecast_metrics;
  setCrumbs(`<b>Model lab</b>`);
  const t = P.test, v = P.val;
  const models = Object.entries(F.models);
  const best = F.models[S.model];
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Evidence</div><h1>Model lab</h1>
      <p>Two models, each scored on data it never saw. The <b>Pc Engine</b> estimates collision probability and tier for every single CDM from telemetry alone; it is scored on all ${fmtInt(t.n_rows)} test rows against their reported risk. The <b>forecaster</b> predicts the tier at closest approach from the history 2 days out, scored on ${fmtInt(F.split.n_val)} held-out events.</p></div>
      <a class="btn" href="data/test_predictions_rows.csv" download>Download test predictions (CSV)</a></div>

    <h2>Pc Engine · per-CDM collision probability</h2>
    <div class="tiles mt">
      <div class="tile feature"><div class="label">Tier accuracy (test rows)</div><div class="value">${fmtPct(t.tier_accuracy)}</div><div class="foot">validation ${fmtPct(v.tier_accuracy)}</div></div>
      <div class="tile"><div class="label">${tierBadge("Critical")} recall</div><div class="value">${fmtPct(t.per_tier.Critical.recall)}</div><div class="foot">precision ${fmtPct(t.per_tier.Critical.precision)}</div></div>
      <div class="tile"><div class="label">${tierBadge("Elevated")} recall</div><div class="value">${fmtPct(t.per_tier.Elevated.recall)}</div><div class="foot">precision ${fmtPct(t.per_tier.Elevated.precision)}</div></div>
      <div class="tile"><div class="label">Pc error (orders of magnitude)</div><div class="value">${t.mae_log_pc_nonfloor.toFixed(2)}</div><div class="foot">physics formula alone: ${t.physics_mae_log_pc_nonfloor.toFixed(1)}</div></div>
      <div class="tile"><div class="label">Pc error on high-risk CDMs</div><div class="value">${t.mae_log_pc_high_risk.toFixed(2)}</div><div class="foot">Pc ≥ 1e-6 · rank corr. ${t.spearman.toFixed(2)}</div></div>
    </div>
    <div class="grid g-12 mt">
      <div class="card span-5"><div class="card-head"><div><h2>Confusion matrix · test CDMs</h2><p>Rows = tier from reported risk, columns = Pc Engine</p></div></div><div class="chart" id="cm"></div></div>
      <div class="card span-7"><div class="card-head"><div><h2>What the engine relies on</h2><p>Increase in Pc error (orders of magnitude) when a feature is shuffled</p></div></div><div class="chart" id="imp"></div>
      <p class="note">Physics features (miss distance over σ, the isotropic and worst-case Pc formulas) carry the most signal; the model learns the correction for the along-track covariance the whitelist leaves out.</p></div>
    </div>

    <h2 class="mt-l">Early-warning forecaster · tier at closest approach</h2>
    <div class="tiles mt">
      <div class="tile feature"><div class="label">Auto-cleared (validation)</div><div class="value">${fmtPct(best.workload.pct_cleared)}</div><div class="foot">95% CI ${best.bootstrap.pct_cleared_ci95.map((x) => fmtPct(x)).join("–")}</div></div>
      <div class="tile"><div class="label">Critical events cleared</div><div class="value">${best.workload.critical_auto_cleared}</div><div class="foot">of ${F.split.val_tier_counts.Critical} · Critical recall ${fmtPct(best.workload.critical_recall)}</div></div>
      <div class="tile"><div class="label">Critical recall incl. review</div><div class="value">${fmtPct(best.critical_recall_with_review)}</div><div class="foot">forecast or volatility flag</div></div>
      <div class="tile"><div class="label">ESA score L ↓</div><div class="value">${best.esa.L.toFixed(2)}</div><div class="foot">95% CI ${best.bootstrap.L_ci95.map((x) => x.toFixed(2)).join("–")} · ${F.split.val_high_risk} high-risk events</div></div>
    </div>
    <div class="grid g-12 mt">
      <div class="card span-7"><div class="card-head"><div><h2>Workload saved at zero missed Critical</h2><p>Share of validation events auto-cleared by each model's policy (θ tuned on train for ≥ 99% Critical recall)</p></div></div>
        <div class="chart" id="wl"></div>
        <div class="table-wrap mt"><table class="data static"><thead><tr><th>Model</th><th class="num">ESA L</th><th class="num">Crit recall</th><th class="num">Crit precision</th><th class="num">Auto-cleared</th></tr></thead>
        <tbody>${models.map(([n, r]) => `<tr><td>${n === S.model ? "<b>OrbitGuard</b>" : esc(n.replace("Baseline ", ""))}</td><td class="num">${r.esa.L.toFixed(3)}</td><td class="num">${r.tiers.critical_recall.toFixed(3)}</td><td class="num">${r.tiers.critical_precision.toFixed(3)}</td><td class="num">${fmtPct(r.workload.pct_cleared)}</td></tr>`).join("")}</tbody></table></div>
        <p class="note">Latest Risk Prediction (LRP) is the benchmark only 12 of 97 teams beat in ESA's 2019 challenge. No regressor we trained improved on its Pc value, so OrbitGuard keeps LRP for the Pc value and adds gradient-boosted tier probabilities. Those drive the forecast tier (rule F1: P(Critical) ≥ ${(100 * S.tau_crit).toFixed(0)}%, F2: P(Elevated or worse) ≥ ${(100 * S.tau_elev).toFixed(0)}%, both tuned out-of-fold on train), which raises Critical recall and clears more events at the same safety level.</p>
      </div>
      <div class="card span-5"><div class="card-head"><div><h2>Where it fails: late surprises</h2><p>${F.volatility.late_surprises} validation events jumped after the cutoff; the volatility flag caught ${F.volatility.late_surprises_flagged}</p></div></div>
        <div class="threats" id="gallery"></div>
      </div>
    </div>`;
  confusion(document.getElementById("cm"), t.confusion, t.labels, { xLabel: "Pc Engine tier", yLabel: "reported tier" });
  const imp = Object.entries(P.feature_importance).slice(0, 10);
  const NAMES = { abs_r_over_sigma: "|radial miss| / σ", max_risk_estimate: "max_risk_estimate", d_over_sigma: "miss / σ", time_to_tca: "time to TCA",
    relative_speed: "relative speed", d_mre: "Δ max_risk_estimate", d_log_c_sigma_r: "Δ chaser σ_r", log_sigma_comb: "combined σ",
    first_log_miss: "first miss distance", d_log_miss: "Δ miss distance", pc_iso_log: "isotropic Pc formula", pc_max_log: "worst-case Pc",
    log_miss: "miss distance", min_log_miss_sofar: "closest miss so far", log_c_sigma_r: "chaser σ_r", log_t_sigma_r: "target σ_r", pos_r: "radial miss", pos_t: "along-track miss", pos_n: "cross-track miss" };
  hBars(document.getElementById("imp"), imp.map(([k, val]) => ({ label: NAMES[k] || k, value: val, text: val.toFixed(2), color: SERIES.blue })), { labelW: 160 });
  hBars(document.getElementById("wl"), models.map(([n, r]) => ({
    label: n.replace("Baseline A: ", "A: ").replace("Baseline B: ", "B: ").replace("OrbitGuard: ", "OrbitGuard: "),
    value: r.workload.pct_cleared, text: `${fmtPct(r.workload.pct_cleared)} · ${r.workload.critical_auto_cleared} Critical cleared`,
    color: n === S.model ? SERIES.blue : "#5d5b56",
  })), { labelW: 220, max: 0.6 });
  const g = S.failure_gallery.slice().sort((a, b) => b.jump_orders - a.jump_orders).slice(0, 6);
  document.getElementById("gallery").innerHTML = g.map((r) => `
    <a class="threat" href="#/event/val/${r.event_id}">
      ${tierBadge(r.final_tier)}
      <div class="t-main"><b>Event ${r.event_id}</b> ${r.volatile_flag ? `<span class="chip review">caught</span>` : `<span class="chip">silent</span>`}<div>${r.current_tier} at cutoff → ${r.final_tier}; miss ${fmtKm(r.latest_miss_m)} → ${fmtKm(r.final_miss_m)} km</div></div>
      <div class="t-p">+${r.jump_orders.toFixed(1)}<small>orders of Pc</small></div>
    </a>`).join("");
}

// ---------------------------------------------------------------- about
function pageAbout() {
  const S = state.summary;
  setCrumbs(`<b>Method & limits</b>`);
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">How it works</div><h1>Method & limits</h1>
      <p>Rules decide, models estimate, Claude phrases, and the analyst decides. Each layer can be audited on its own.</p></div></div>
    <div class="pipeline">
      <div class="pipe"><div class="who code">Code</div><b>Ingest CDMs</b><p>Only the 13 permitted columns are loaded; anything else fails an assertion.</p></div>
      <div class="pipe"><div class="who ml">Model</div><b>Pc Engine</b><p>Gradient-boosted hurdle model on physics features estimates Pc and tier for every CDM.</p></div>
      <div class="pipe"><div class="who ml">Model</div><b>Forecaster</b><p>From the history at T−2 days: P(Safe / Elevated / Critical) at closest approach.</p></div>
      <div class="pipe"><div class="who code">Rules</div><b>Handbook tiers</b><p>Critical: Pc ≥ 1e-4 or miss &lt; 1 km. Elevated: Pc ≥ 1e-6 or miss ≤ 5 km. Otherwise Safe.</p></div>
      <div class="pipe"><div class="who llm">Claude</div><b>Briefing</b><p>Rewrites the facts as 3 sentences; a verifier rejects any number or tier not in the facts.</p></div>
      <div class="pipe"><div class="who human">Analyst</div><b>Act / Watch / Clear</b><p>The suggested action comes from rules A1–A4; the human always records the decision.</p></div>
    </div>
    <div class="grid g-2 mt">
      <div class="card prose">
        <h3 style="margin-top:0">What makes OrbitGuard different</h3>
        <ul>
          <li><b>Two time horizons.</b> A per-message Pc calculator for <i>now</i>, and an early-warning forecaster for <i>closest approach</i>, two days ahead.</li>
          <li><b>Workload with a safety guarantee.</b> The auto-clear threshold is tuned on training data for ≥ 99% Critical recall, then measured on held-out events: ${fmtPct(S.forecast_metrics.models[S.model].workload.pct_cleared)} cleared, none of them Critical.</li>
          <li><b>Physics plus learning.</b> The isotropic and worst-case Pc formulas are features, not the answer; the Pc Engine corrects them (error ${S.pc_engine.test.mae_log_pc_nonfloor.toFixed(1)} vs ${S.pc_engine.test.physics_mae_log_pc_nonfloor.toFixed(1)} orders of magnitude).</li>
          <li><b>Every flag is traceable</b> to rule codes R1–R6 and a CDM row number; the LLM cannot introduce a number.</li>
          <li><b>Manoeuvre what-if</b> shows how much separation turns an encounter Safe.</li>
        </ul>
        <h3>Data decisions</h3>
        <ul>
          <li>Units checked against the distributions: days, metres, m/s; <code>risk</code> = log10 Pc floored at −30.</li>
          <li>Decision cutoff of 2 days. The forecaster's training events need a CDM at or before 2 days and a final CDM within 1 day of TCA (the ESA filter).</li>
          <li>Train and validation are split by <code>event_id</code>, so no encounter appears on both sides.</li>
          <li>A sigma of 6.378e7 m (10 Earth radii) is a "no covariance" placeholder and is treated as missing.</li>
        </ul>
      </div>
      <div class="card prose">
        <h3 style="margin-top:0">Known limits</h3>
        <ul>
          <li><b>Late surprises.</b> ${S.forecast_metrics.volatility.late_surprises} validation events changed sharply after the cutoff, and ${S.forecast_metrics.volatility.late_surprises - S.forecast_metrics.volatility.late_surprises_flagged} looked calm beforehand. Volatile events are never auto-cleared and a human always decides.</li>
          <li><b>Critical is mostly the 1 km rule</b> (${(100 * S.insights_24h.critical_drivers.miss_rule_only / (S.insights_24h.critical_drivers.miss_rule_only + S.insights_24h.critical_drivers.pc_rule_only + S.insights_24h.critical_drivers.both)).toFixed(1)}% of last-day Critical flags). The forecaster carries the latest miss distance forward rather than predicting it.</li>
          <li><b>Few high-risk events</b> (${S.forecast_metrics.split.val_high_risk} in validation), so the ESA score has a wide confidence interval.</li>
          <li><b>No orbits in the data.</b> The 3D views use the real relative miss vectors; the Earth view and fly-by line are schematic.</li>
          <li><b>24-hour frame is retrospective</b> because the test file ends 2 days before TCA.</li>
          <li><b>Decisions are stored in this browser</b> (export them as CSV from the triage page).</li>
        </ul>
        <h3>Reproduce</h3>
        <ul><li><code>python run.py all</code> builds data, trains and scores both models and writes <code>submissions/</code>.</li>
        <li><code>python -m orbitguard.export_web</code> writes the JSON this site reads.</li></ul>
        <p class="note">Data generated ${esc(S.generated_utc)} · ${fmtInt(S.counts.train_rows)} train and ${fmtInt(S.counts.test_rows)} test CDMs.</p>
      </div>
    </div>`;
}

route();
