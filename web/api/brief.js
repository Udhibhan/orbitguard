// POST /api/brief  { facts }  ->  { text, recommendation, source, problems }
//
// Code computes every fact (tiers, numbers, the Act/Watch/Clear suggestion) in the browser and sends them here.
// Claude only phrases them as 3 sentences. A verifier rejects any text that uses a number not present in the
// facts, mentions a tier that is not in the facts, or changes the suggested action. On rejection, missing key,
// or any API error the caller gets source "template" and shows its deterministic template instead.
import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-haiku-5-5";
const ACTIONS = ["Act", "Watch", "Clear"];
const TIERS = ["Safe", "Elevated", "Critical"];
const THRESHOLD_NUMBERS = [1e-4, 1e-6, 1, 5, 2, 24, 99, 3];

const SYSTEM =
  "You write briefings for a satellite conjunction assessment analyst. You are given FACTS as JSON, " +
  "computed by a deterministic rule engine. Rewrite them as exactly 3 plain sentences: (1) current " +
  "status and which rule fired, (2) the trend and the forecast at closest approach (TCA), (3) caveats " +
  "(volatility, model uncertainty) and what to watch next. Use ONLY numbers that appear in FACTS, copied as " +
  "written; never compute, round differently or invent numbers. Never change or reinterpret a tier. The " +
  "recommendation field must be exactly FACTS.suggested_action. The analyst makes the final decision.";

const SCHEMA = {
  type: "object",
  properties: { briefing: { type: "string" }, recommendation: { type: "string", enum: ACTIONS } },
  required: ["briefing", "recommendation"],
  additionalProperties: false,
};

const NUM_RE = /(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?/g;

function numbersIn(obj) {
  const s = typeof obj === "string" ? obj : JSON.stringify(obj);
  return (s.match(NUM_RE) || []).map((t) => parseFloat(t.replace(",", ""))).filter(Number.isFinite);
}

function matches(x, allowed, token) {
  const sci = /e/i.test(token);
  const decimals = token.includes(".") ? token.split(".")[1].split(/e/i)[0].length : 0;
  return allowed.some((a) =>
    [a, -a].some((v) => {
      if (Math.abs(x - v) <= 1e-12 + 1e-9 * Math.abs(v)) return true;
      if (!sci && Number(v.toFixed(Math.min(decimals, 20))) === x) return true;
      return sci && v !== 0 && Math.abs(x - v) / Math.abs(v) <= 0.051;
    })
  );
}

export function verify(text, recommendation, facts) {
  const problems = [];
  const allowed = numbersIn(facts).concat(THRESHOLD_NUMBERS);
  for (const tok of text.match(NUM_RE) || []) {
    const x = parseFloat(tok.replace(",", ""));
    if (Number.isFinite(x) && !matches(x, allowed, tok)) problems.push(`number ${tok} not in facts`);
  }
  const allowedTiers = new Set([facts.current_tier, facts.forecast_tier, facts.pc_engine_tier].filter(Boolean));
  for (const t of TIERS) {
    if (new RegExp(`\\b${t}\\b`, "i").test(text) && !allowedTiers.has(t)) problems.push(`mentions tier ${t} not in facts`);
  }
  if (recommendation !== facts.suggested_action) {
    problems.push(`recommendation ${recommendation} != rule suggestion ${facts.suggested_action}`);
  }
  return problems;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  const facts = req.body && req.body.facts;
  if (!facts || typeof facts !== "object" || !ACTIONS.includes(facts.suggested_action)) {
    res.status(400).json({ error: "body must be { facts } with a suggested_action" });
    return;
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    res.status(200).json({ source: "template", problems: ["ANTHROPIC_API_KEY not set on the server"] });
    return;
  }
  try {
    const client = new Anthropic();
    const resp = await client.messages.create({
      model: MODEL,
      max_tokens: 2000,
      system: SYSTEM,
      output_config: { effort: "low", format: { type: "json_schema", schema: SCHEMA } },
      messages: [{ role: "user", content: "FACTS:\n" + JSON.stringify(facts, null, 1) }],
    });
    if (resp.stop_reason !== "end_turn") throw new Error(`stop_reason=${resp.stop_reason}`);
    const block = resp.content.find((b) => b.type === "text");
    const out = JSON.parse(block.text);
    const problems = verify(out.briefing, out.recommendation, facts);
    if (problems.length) {
      res.status(200).json({ source: "template", problems: ["verifier rejected LLM text: " + problems.join("; ")],
        rejected_text: out.briefing });
      return;
    }
    res.setHeader("Cache-Control", "s-maxage=86400");
    res.status(200).json({ text: out.briefing, recommendation: out.recommendation, source: "claude", model: MODEL,
      problems: [] });
  } catch (e) {
    const kind = e instanceof Anthropic.APIError ? `API ${e.status}` : e.name;
    res.status(200).json({ source: "template", problems: [`LLM unavailable: ${kind}: ${String(e.message).slice(0, 160)}`] });
  }
}
