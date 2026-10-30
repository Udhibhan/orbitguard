"""Analyst briefings. Code computes every fact; Claude only PHRASES them.

Guardrails:
  * facts (tiers, reasons, numbers, the Act/Watch/Clear suggestion) are computed in code and sent as JSON;
  * the suggested action comes from a deterministic rule (suggest_action), and the LLM must echo it;
  * verifier: every number in the LLM text must appear in the facts, only tier names present in the
    facts may be mentioned, and the action must match. Anything else falls back to a template;
  * no ANTHROPIC_API_KEY -> template only. Responses are cached on disk by facts hash.
The human analyst makes the decision; the suggestion is advisory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

from . import data
from .tiers import CRITICAL, ELEVATED, SAFE

MODEL = "claude-haiku-5-5"
CACHE_DIR = data.ROOT / "data" / "briefings_cache"
ACTIONS = ("Act", "Watch", "Clear")
THRESHOLD_NUMBERS = [1e-4, 1e-6, 1.0, 5.0, 2.0, 99]  # rule constants the text may legitimately cite

SYSTEM = (
    "You write briefings for a satellite conjunction assessment analyst. You are given FACTS as JSON, "
    "computed by a deterministic rule engine. Rewrite them as exactly 3 plain sentences: (1) current "
    "status and which rule fired, (2) the trend and the forecast at TCA, (3) caveats (volatility, "
    "model uncertainty). Use ONLY numbers that appear in FACTS, copied as written, and never compute, "
    "round differently or invent numbers. Never change or reinterpret a tier. The recommendation field "
    "must be exactly FACTS.suggested_action. The analyst makes the final decision."
)
SCHEMA = {
    "type": "object",
    "properties": {"briefing": {"type": "string"}, "recommendation": {"type": "string", "enum": list(ACTIONS)}},
    "required": ["briefing", "recommendation"],
    "additionalProperties": False,
}


def suggest_action(forecast_tier: str, current_tier: str, volatile: bool, auto_cleared: bool) -> tuple[str, str]:
    """Deterministic suggestion (not a decision): returns (action, rule)."""
    if CRITICAL in (forecast_tier, current_tier):
        return "Act", "A1: current or forecast tier is Critical -> plan manoeuvre options"
    if volatile or ELEVATED in (forecast_tier, current_tier):
        return "Watch", "A2: Elevated tier or volatile history -> monitor the next CDMs"
    if auto_cleared:
        return "Clear", "A3: forecast Safe, not volatile, passes the auto-clear threshold"
    return "Watch", "A4: forecast Safe but below the auto-clear confidence threshold"


def _fmt_pc(log_pc):
    if log_pc is None or (isinstance(log_pc, float) and math.isnan(log_pc)):
        return "unavailable"
    return "<=1e-30" if log_pc <= -30 else f"{10 ** log_pc:.2e}"


def build_facts(ev: dict, pred: dict, current: tuple, forecast: tuple, pc_approx_log=None,
                auto_cleared: bool = False) -> dict:
    """ev: event row (pre-cutoff features); pred: model output row; current/forecast: (tier, reasons)."""
    action, rule = suggest_action(forecast[0], current[0], bool(ev["volatile"]), auto_cleared)
    return {
        "event_id": int(ev["event_id"]),
        "decision_time_to_tca_days": round(float(ev["latest_time_to_tca"]), 2),
        "cdms_before_cutoff": int(ev["n_cdms_pre"]),
        "object_type": str(ev["latest_c_object_type"]),
        "current_tier": current[0], "current_rules_fired": current[1],
        "latest_pc": _fmt_pc(ev["latest_risk"]),
        "latest_miss_km": round(float(ev["latest_miss_distance"]) / 1000, 3),
        "risk_trend_orders_per_day": round(float(ev["risk_slope"]), 2),
        "miss_trend_km_per_day": round(float(ev["miss_slope"]) / 1000, 3),
        "model": pred["model_name"],
        "forecast_pc_at_tca": _fmt_pc(pred["pred_risk"]),
        "forecast_tier": forecast[0], "forecast_rules_fired": forecast[1],
        "p_critical_pct": round(100 * float(pred["p_critical"]), 1),
        "volatile": bool(ev["volatile"]), "volatile_reasons": ev["volatile_reasons"] or "none",
        "independent_pc_approx": _fmt_pc(pc_approx_log),
        "suggested_action": action, "suggestion_rule": rule,
    }


# ---------- verifier ----------
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _numbers_in(obj) -> list[float]:
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    out = []
    for m in NUM_RE.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _matches(x: float, allowed: list[float], token: str) -> bool:
    decimals = len(token.split(".")[1].split("e")[0].split("E")[0]) if "." in token else 0
    for a in allowed:
        for v in (a, -a):
            if math.isclose(x, v, rel_tol=1e-9, abs_tol=1e-12):
                return True
            if "e" not in token.lower() and round(v, decimals) == x:
                return True
            if "e" in token.lower() and v != 0 and math.isclose(x, v, rel_tol=0.051):
                return True
    return False


def verify(text: str, recommendation: str, facts: dict) -> tuple[bool, list[str]]:
    problems = []
    allowed = _numbers_in(facts) + THRESHOLD_NUMBERS + [3]
    for tok in NUM_RE.findall(text):
        try:
            x = float(tok.replace(",", ""))
        except ValueError:
            continue
        if not _matches(x, allowed, tok):
            problems.append(f"number {tok} not in facts")
    mentioned = {t for t in (CRITICAL, ELEVATED, SAFE) if re.search(rf"\b{t}\b", text, re.I)}
    allowed_tiers = {facts["current_tier"], facts["forecast_tier"]}
    if mentioned - allowed_tiers:
        problems.append(f"mentions tier(s) {sorted(mentioned - allowed_tiers)} not in facts")
    if recommendation != facts["suggested_action"]:
        problems.append(f"recommendation {recommendation} != rule suggestion {facts['suggested_action']}")
    return not problems, problems


# ---------- template + LLM ----------
def template(facts: dict) -> str:
    cur = "; ".join(facts["current_rules_fired"])
    s1 = (f"Event {facts['event_id']} is currently {facts['current_tier']} at "
          f"{facts['decision_time_to_tca_days']} days to TCA ({cur}).")
    s2 = (f"Over {facts['cdms_before_cutoff']} CDMs the risk trend is {facts['risk_trend_orders_per_day']} "
          f"orders of magnitude per day and miss distance changes {facts['miss_trend_km_per_day']} km per day; "
          f"{facts['model']} forecasts Pc {facts['forecast_pc_at_tca']} at TCA, i.e. {facts['forecast_tier']}.")
    s3 = (f"Volatility: {facts['volatile_reasons']}." if facts["volatile"]
          else "Pre-cutoff history is stable, but late CDMs can still change the picture.")
    return f"{s1} {s2} {s3}"


def _cache_path(facts: dict):
    h = hashlib.sha256((MODEL + SYSTEM + json.dumps(facts, sort_keys=True)).encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.json"


def brief(facts: dict, use_llm: bool = True) -> dict:
    """Returns {text, recommendation, source: 'claude'|'template', problems}."""
    fallback = {"text": template(facts), "recommendation": facts["suggested_action"], "source": "template",
                "problems": []}
    if not use_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        fallback["problems"] = ["ANTHROPIC_API_KEY not set: template used"] if use_llm else []
        return fallback
    path = _cache_path(facts)
    if path.exists():
        return json.loads(path.read_text())
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": "FACTS:\n" + json.dumps(facts, indent=1)}],
        )
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={resp.stop_reason}")
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    except Exception as e:  # network, auth, refusal, parse: deterministic fallback, never block the analyst
        fallback["problems"] = [f"LLM unavailable: {type(e).__name__}: {e}"[:200]]
        return fallback
    ok, problems = verify(out["briefing"], out["recommendation"], facts)
    result = ({"text": out["briefing"], "recommendation": out["recommendation"], "source": "claude", "problems": []}
              if ok else {**fallback, "problems": ["verifier rejected LLM text: " + "; ".join(problems)],
                          "rejected_text": out["briefing"]})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result
