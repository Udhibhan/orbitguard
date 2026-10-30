# OrbitGuard: satellite collision-risk triage

Airbus hackathon, Challenge #1. OrbitGuard reads ESA Kelvins conjunction data messages (CDMs), estimates the
collision probability of every message, forecasts each encounter's tier at closest approach (TCA) **2 days early**,
and auto-clears what is provably safe so that analysts only see the dangerous ones.

Rules make every tier decision, models supply the numbers, Claude only phrases briefings, and the analyst decides.

## The website (`web/`)

A static site with no build step (ES modules, three.js and d3 from a CDN), plus one optional serverless function
(`web/api/brief.js`) for Claude-written briefings. Pages: Mission control (orbit hero, conjunction radar, top
threats), Triage queue, Encounter (3D RTN geometry with CDM replay, Pc/miss timelines, decision trace,
manoeuvre what-if, briefing, Act/Watch/Clear log), 24-hour warnings, Model lab, and Method & limits.

**Run locally:** `python run.py serve`, then open http://localhost:8000. Briefings fall back to the deterministic
template locally because `python -m http.server` can't run the function.

**Deploy on Vercel:** import the GitHub repo, set **Root Directory = `web`**, Framework preset **Other**, and leave
the build command empty. Optionally add the environment variable `ANTHROPIC_API_KEY` to turn on Claude briefings.
No database is needed: all data is static JSON in `web/data/`, and analyst decisions are stored in the
browser (export them as CSV from the triage page).

## Rebuild the data and models

Put `train_data.csv` and `test_data.csv` in `data/raw/`, then:

```bash
pip install -r requirements.txt          # Python 3.10+
python run.py all                        # data, evaluation, Pc Engine, test predictions, web export, tests
python run.py serve                      # website at http://localhost:8000
python run.py app                        # legacy Streamlit dashboard at http://localhost:8501
```

## Headline results

**Pc Engine (per-CDM Pc and tier, `orbitguard/pcnet.py`).** It is scored on all 24,484 test rows against their reported
`risk`, which it never sees as an input.

| Metric | Test rows | Validation rows |
|---|---|---|
| Tier accuracy | **97.5%** | 98.2% |
| Critical recall / precision | 97.8% / 98.3% | 99.4% / 98.3% |
| Elevated recall / precision | 94.4% / 96.1% | 95.0% / 96.7% |
| Pc error, orders of magnitude (physics formula alone) | **2.66** (16.5) | 3.01 (16.1) |

**Forecaster (tier at TCA from the history at T-2 days; 1,659 held-out events, split by event_id)**

| Model | ESA L | Critical recall | Critical recall incl. analyst review | Auto-cleared @ 0 Critical cleared |
|---|---|---|---|---|
| Baseline A: constant -5 | 9.31 | 0.84 | 0.84 | 0% |
| Baseline B: Latest Risk (LRP) | 0.56 | 0.84 | 0.95 | 45.3% |
| **OrbitGuard: LRP + GBM tier probabilities** | 0.56 | **0.86** | **0.96** | **51.5%** (95% CI 49.0–53.8%) |

- LRP is the benchmark that only 12 of 97 teams beat in ESA's 2019 competition. No regressor we trained improved its Pc value,
  so OrbitGuard keeps LRP for Pc. Gradient-boosted tier probabilities (out-of-fold on train) drive the forecast tier
  (F1: P(Critical) >= 15%, F2: P(Elevated or worse) >= 20%) and the auto-clear policy.
- Submissions: `submissions/test_predictions_rows.csv` (Pc Engine, one row per test CDM),
  `test_predictions_event.csv`, `test_submission_esa.csv`.

## How it works

| Stage | Module | Who decides |
|---|---|---|
| Load CDMs, **13-column whitelist** asserted | `orbitguard/data.py` | code |
| Event features from CDMs **at or before 2 days to TCA** only; target = final CDM risk | `data.py` | code |
| Tiers: Critical (Pc ≥ 1e-4 or miss < 1 km) > Elevated (Pc ≥ 1e-6 or miss ≤ 5 km) > Safe | `orbitguard/tiers.py` | **rules** |
| Per-CDM Pc and tier probabilities (Pc Engine: hurdle GBM on physics features) | `orbitguard/pcnet.py` | model (numbers only) |
| Forecast final log10 Pc + tier probabilities: baselines, GBM forecaster, teammate plug-in | `orbitguard/models.py` | model (numbers only) |
| Forecast tier = rules(pred risk, latest miss), or rules F1/F2 on tier probabilities for the GBM forecaster | `tiers.py`, `models.py` | **rules** |
| Volatility → Analyst review; auto-clear if p_safe ≥ θ | `evaluate.py`, `data.py` | **rules** |
| Independent Pc approximation (cross-check) | `orbitguard/pc.py` | code |
| Briefing: facts JSON → Claude (`claude-haiku-5-5`) → verifier → template fallback | `orbitguard/briefing.py` | LLM phrases only |
| Act / Watch / Clear → `data/decisions.jsonl` | `app.py`, `decisions.py` | **analyst** |

Traceability: every flag in the dashboard shows the rule codes that fired (R1–R6) and the exact CDM row
(`train_data.csv data row N`). The briefing's suggested action comes from rules A1–A4 in code, and the LLM must
repeat it. Any number in the LLM text that isn't in the facts JSON causes the text to be rejected.

## Data decisions (confirmed with the team)

- Units verified from distributions: days, metres, m/s; `risk` = log10 Pc, floored at -30.
- ESA event filter: ≥1 CDM at ≥ 2 d **and** a final CDM ≤ 1 d from TCA. 8,293 of 13,154 train events kept
  (1,212 had no pre-cutoff CDM; 3,649 had no CDM within 1 day of TCA).
- The test CSV stops at 2 days (no labels), so all metrics come from the validation split and the test set only gets
  predictions: `submissions/test_predictions_event.csv`, `test_predictions_rows.csv`, `test_submission_esa.csv`.
- A sigma of 6.378e7 m (10 Earth radii) is a "no covariance" sentinel and is treated as missing.

## Where it fails

1. **Late surprises.** 22 validation events jumped > 2 orders of magnitude or escalated to Critical after the
   cutoff. The volatility flag sent 14 to analyst review; **8 were silent**. Example: event 4699 had a calm history
   at 1.1 km, and its final CDM put the miss at 273 m. No pre-cutoff model sees these. We never auto-clear volatile
   events, and a human always decides.
2. **Critical is almost entirely miss distance.** 98.7% of Critical flags inside 24 h come from the 1 km rule alone.
   All 6 silent Critical escalations were miss distances shrinking below 1 km. We carry the latest miss forward
   rather than forecasting it, which is the biggest modelling gap.
3. **The Pc approximation is poor** (Spearman 0.22 vs dataset risk; it misses 88% of high-risk CDMs). Only radial
   sigmas are allowed, while along-track sigma is typically 10–100× larger. We show this openly and never use it for
   tiers. The worst-case Pc_max does track `max_risk_estimate` (Spearman 0.53), with an implied hard-body radius of
   11.7 m, which supports the assumed HBR of 10 m.
4. **Small high-risk sample:** 13 validation events, so ESA L has a 95% CI of 0.01–1.77.
5. **The 24-hour view is retrospective,** because the test data ends 2 days before TCA.
6. **Volatility rule is broad:** 33% of events are routed to analyst review. That's safe, but it caps the workload saving.

## Repo layout

```
orbitguard/   data.py tiers.py pc.py models.py evaluate.py predict_test.py briefing.py insights.py decisions.py
app.py        Streamlit dashboard (triage, event detail, 24 h warnings, evaluation, about)
models/       CONTRACT.md (teammate plug-in: teammate_predictions.csv or teammate/model.py)
reports/      metrics.json, failure_gallery.json, pc_validation.json, insights_24h.json, PNGs
submissions/  test-set predictions
tests/        tier boundaries, Pc sanity, leakage, whitelist, verifier, plug-in
```

License: Apache-2.0.
