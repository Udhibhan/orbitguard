// Analyst briefing: facts are computed here (deterministic), /api/brief asks Claude to phrase them and
// verifies the result server-side. Any failure (no key, offline, local static server) -> template.
import { fmtPc, rulesFired, suggestAction } from "./data.js";

export function buildFacts(e, split, engine, forecastRule) {
  const cur = rulesFired(e.risk, e.miss);
  const [action, rule] = suggestAction(e.fcTier, e.curTier, !!e.vol, e.queue === "Auto-clear");
  return {
    event_id: e.id,
    dataset: split === "test" ? "live (test set, decision 2 days before TCA)" : "validation (hindsight available)",
    decision_time_to_tca_days: Number(e.t.toFixed(2)),
    cdms_before_cutoff: e.n,
    object_type: e.object,
    current_tier: e.curTier,
    current_rules_fired: cur.map((r) => `${r.code} ${r.text}`),
    latest_pc: fmtPc(e.risk),
    latest_miss_km: Number((e.miss / 1000).toFixed(3)),
    relative_speed_km_s: Number((e.speed / 1000).toFixed(2)),
    pc_engine_estimate: engine ? fmtPc(engine.est) : "n/a",
    pc_engine_tier: engine ? engine.etierName : undefined,
    forecast_tier: e.fcTier,
    forecast_rule_fired: forecastRule,
    p_critical_at_tca_pct: Number((100 * e.pc).toFixed(1)),
    p_safe_at_tca_pct: Number((100 * e.ps).toFixed(1)),
    volatile: !!e.vol,
    volatile_reasons: e.vol || "none",
    queue: e.queue,
    suggested_action: action,
    suggestion_rule: rule,
  };
}

export function template(f) {
  const s1 = `Event ${f.event_id} is ${f.current_tier} at ${f.decision_time_to_tca_days} days to TCA (${f.current_rules_fired.join("; ")}), with Pc ${f.latest_pc} at a miss distance of ${f.latest_miss_km} km.`;
  const s2 = `The forecaster expects ${f.forecast_tier} at TCA (${f.forecast_rule_fired}), a ${f.p_critical_at_tca_pct}% probability of ending Critical; the Pc Engine's independent estimate for the latest CDM is ${f.pc_engine_estimate}.`;
  const s3 = f.volatile
    ? `The history is volatile (${f.volatile_reasons}), so the event is routed to analyst review and must not be auto-cleared.`
    : `The pre-cutoff history is stable, but late CDMs can still change the picture, so watch the next update.`;
  return `${s1} ${s2} ${s3}`;
}

const memo = new Map();
export async function requestBrief(facts) {
  const key = JSON.stringify(facts);
  if (memo.has(key)) return memo.get(key);
  const fallback = (problems) => ({ text: template(facts), recommendation: facts.suggested_action, source: "template", problems });
  let out;
  try {
    const r = await fetch("api/brief", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ facts }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    out = j.source === "claude" ? j : fallback(j.problems || []);
  } catch (err) {
    out = fallback([`briefing API not reachable (${err.message}); deterministic template shown`]);
  }
  memo.set(key, out);
  return out;
}
