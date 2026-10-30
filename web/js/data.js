// Data loading, decoding and the rule engine mirrored from orbitguard/tiers.py.
// Every number shown on the site comes from web/data/*.json (exported by orbitguard/export_web.py)
// or from the deterministic functions below.

export const TIERS = ["Safe", "Elevated", "Critical"];
export const OBJECTS = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"];
export const QUEUES = ["Queue", "Analyst review", "Auto-clear"];
export const TIER_COLOR = { Critical: "#d03b3b", Elevated: "#fab219", Safe: "#0ca30c" };
export const SERIES = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", violet: "#9085e9" };
export const CUTOFF = 2.0;
export const HBR_M = 10;

const cache = new Map();

async function getJSON(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

export const loadSummary = () => getJSON("summary.json");

const eventCache = {};
export async function loadEvents(split) {
  if (eventCache[split]) return eventCache[split];
  const raw = await getJSON(`events_${split}.json`);
  const ix = Object.fromEntries(raw.cols.map((c, i) => [c, i]));
  const rows = raw.rows.map((r) => {
    const e = {};
    for (const c of raw.cols) e[c] = r[ix[c]];
    e.curTier = TIERS[e.cur];
    e.fcTier = TIERS[e.fc];
    e.object = OBJECTS[e.obj];
    e.queue = QUEUES[e.q];
    if (e.ft != null) e.finalTier = TIERS[e.ft];
    return e;
  });
  const byId = new Map(rows.map((e) => [e.id, e]));
  eventCache[split] = { rows, byId };
  return eventCache[split];
}

export async function loadCdms(split, id) {
  const raw = await getJSON(`cdms_${split}.json`);
  const list = raw.events[String(id)] || [];
  return list.map((r) => {
    const o = {};
    raw.cols.forEach((c, i) => (o[c] = r[i]));
    o.etierName = TIERS[o.etier];
    o.tier = classifyLog(o.risk, o.miss);
    return o;
  });
}

// ---------- rule engine (identical thresholds to orbitguard/tiers.py) ----------
export function classifyLog(logPc, missM) {
  const known = logPc != null && Number.isFinite(logPc);
  if ((known && logPc >= -4) || missM < 1000) return "Critical";
  if ((known && logPc >= -6) || missM <= 5000 || !known) return "Elevated";
  return "Safe";
}

export function rulesFired(logPc, missM) {
  const out = [];
  const km = (missM / 1000).toFixed(3);
  const known = logPc != null && Number.isFinite(logPc);
  const pcCrit = known && logPc >= -4, missCrit = missM < 1000;
  if (pcCrit) out.push({ code: "R1", text: `Pc ${fmtPc(logPc)} ≥ 1e-4 → Critical` });
  if (missCrit) out.push({ code: "R2", text: `miss ${km} km < 1.0 km → Critical` });
  if (pcCrit || missCrit) return out;
  const pcElev = known && logPc >= -6, missElev = missM <= 5000;
  if (pcElev) out.push({ code: "R3", text: `1e-6 ≤ Pc ${fmtPc(logPc)} < 1e-4 → Elevated` });
  if (missElev) out.push({ code: "R4", text: `1.0 ≤ miss ${km} km ≤ 5.0 km → Elevated` });
  if (!known) out.push({ code: "R5", text: "Pc unavailable → at least Elevated" });
  if (pcElev || missElev || !known) return out;
  return [{ code: "R6", text: `Pc ${fmtPc(logPc)} < 1e-6 and miss ${km} km > 5.0 km → Safe` }];
}

// ---------- formatting ----------
export function fmtPc(logPc) {
  if (logPc == null || !Number.isFinite(logPc)) return "n/a";
  if (logPc <= -29.99) return "≤1e-30";
  return Math.pow(10, logPc).toExponential(2).replace("e+0", "");
}
export const fmtKm = (m, d = 3) => (m == null ? "n/a" : (m / 1000).toFixed(d));
export const fmtPct = (x, d = 1) => (x == null ? "n/a" : `${(100 * x).toFixed(d)}%`);
export const fmtInt = (n) => (n == null ? "n/a" : Number(n).toLocaleString("en-US"));
export const tierBadge = (t, cls = "") => `<span class="tier ${t} ${cls}">${t}</span>`;
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function queueChip(q) {
  const cls = q === "Analyst review" ? "review" : q === "Auto-clear" ? "clear" : "queue";
  const label = q === "Analyst review" ? "Analyst review" : q === "Auto-clear" ? "Auto-clear" : "Queue";
  return `<span class="chip ${cls}">${label}</span>`;
}

// ---------- deterministic action suggestion (mirrors briefing.suggest_action) ----------
export function suggestAction(fc, cur, volatile, autoClear) {
  if (fc === "Critical" || cur === "Critical") return ["Act", "A1: current or forecast tier is Critical → plan manoeuvre options"];
  if (volatile || fc === "Elevated" || cur === "Elevated") return ["Watch", "A2: Elevated tier or volatile history → monitor the next CDMs"];
  if (autoClear) return ["Clear", "A3: forecast Safe, not volatile, passes the auto-clear threshold"];
  return ["Watch", "A4: forecast Safe but below the auto-clear confidence threshold"];
}

// ---------- manoeuvre what-if: linear-Gaussian encounter calibrated to the event's current Pc ----------
// Pc(d) = R² / (2σ²) · exp(-d² / 2σ²). With R = 10 m we solve σ so that Pc(d0) equals the current Pc
// (tight-covariance branch, σ ≤ d0/√2). If the current Pc is above the R = 10 m maximum we keep
// σ = d0/√2 and scale R instead. Then Pc at a new separation follows from the same σ and R.
export function calibrateEncounter(d0, logPc0) {
  const d = Math.max(d0, 1);
  if (logPc0 == null || logPc0 <= -29.99) return { sigma: d / 6, R: HBR_M, floor: true };
  const pcAt = (s, R) => Math.log10((R * R) / (2 * s * s)) - (d * d) / (2 * s * s) / Math.LN10;
  const sMax = d / Math.SQRT2;
  if (logPc0 >= pcAt(sMax, HBR_M)) {
    const R = Math.sqrt(Math.pow(10, logPc0) * Math.E * d * d);
    return { sigma: sMax, R, floor: false };
  }
  // tight-covariance branch (σ < d/√2): the encounter is unlikely because the miss is many σ away,
  // so Pc falls monotonically as separation grows. Pc(σ) increases with σ on this branch.
  let lo = sMax * 1e-6, hi = sMax;
  for (let i = 0; i < 200; i++) {
    const mid = Math.sqrt(lo * hi);
    if (pcAt(mid, HBR_M) < logPc0) lo = mid; else hi = mid;
  }
  return { sigma: Math.sqrt(lo * hi), R: HBR_M, floor: false };
}
export function pcAtDistance(cal, dM) {
  const s = cal.sigma;
  const v = Math.log10((cal.R * cal.R) / (2 * s * s)) - (dM * dM) / (2 * s * s) / Math.LN10;
  return Math.max(-30, Math.min(0, v));
}
// smallest separation (m) at which the encounter is Safe under the rules (miss > 5 km and Pc < 1e-6)
export function safeSeparation(cal) {
  let d = 5001;
  if (pcAtDistance(cal, d) < -6) return d;
  let lo = d, hi = d;
  while (pcAtDistance(cal, hi) >= -6 && hi < 1e8) hi *= 2;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (pcAtDistance(cal, mid) >= -6) lo = mid; else hi = mid;
  }
  return hi;
}

// ---------- decisions (per browser) ----------
const DKEY = "orbitguard.decisions.v1";
export function loadDecisions() {
  try { return JSON.parse(localStorage.getItem(DKEY) || "[]"); } catch { return []; }
}
export function saveDecision(rec) {
  const all = loadDecisions();
  all.push(rec);
  try { localStorage.setItem(DKEY, JSON.stringify(all)); } catch { /* storage blocked: keep in memory only */ }
  return all;
}
export function lastDecision(split, id) {
  const all = loadDecisions().filter((d) => d.dataset === split && d.event_id === id);
  return all.length ? all[all.length - 1] : null;
}
export function decisionsCsv() {
  const all = loadDecisions();
  const cols = ["timestamp_utc", "dataset", "event_id", "decision", "suggested_action", "current_tier", "forecast_tier", "p_critical", "note"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [cols.join(","), ...all.map((d) => cols.map((c) => q(d[c])).join(","))].join("\n");
}
