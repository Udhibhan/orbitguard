# OrbitGuard: satellite collision-risk triage

Airbus hackathon, Challenge #1. OrbitGuard helps a **conjunction assessment analyst** decide, about
**2 days before closest approach (TCA)**, whether an event needs manoeuvre planning. It runs on the ESA Kelvins
Collision Avoidance Challenge CDM data.

Rules and statistics make every decision; Claude only phrases briefings; the analyst decides.

## Run it (3 commands)

Put `train_data.csv` and `test_data.csv` in `data/raw/`, then:

```bash
pip install -r requirements.txt          # Python 3.10+
python run.py all                        # build data, evaluate, Pc validation, test predictions, tests
python run.py app                        # dashboard at http://localhost:8501
```

`make data | eval | predict | test | app` do the same where `make` is available. Set `ANTHROPIC_API_KEY`
for Claude-written briefings; without it every briefing uses the deterministic template.

## Headline results (validation: 1,659 held-out events, split by event_id)

| Model | ESA L ↓ (95% CI) | MSE_HR | F2_HR | Critical recall | Auto-cleared @ Critical recall |
|---|---|---|---|---|---|
| Baseline A: constant -5 | 9.31 | 0.35 | 0.04 | 0.84 | 0% |
| **Baseline B: Latest Risk (LRP)** | **0.56** (0.01–1.77) | 0.23 | 0.41 | 0.84 (0.95 incl. analyst review) | **45.3% @ 100%** |
| Teammate model | plugs in via `models/CONTRACT.md` | | | | |

- **Impact:** 45.3% of events (95% CI 42.7–47.7%) are auto-cleared with **0 of 110 Critical events cleared**
  (8 Elevated cleared). The clearing threshold is tuned on the train split only and measured on validation.
- LRP is the benchmark: only 12 of 97 teams beat it in ESA's 2019 competition. Our L of 0.56 is close to ESA's
  published LRP score (~0.69 on their test set).
- There are only 13 high-risk events in validation, so treat L as an indication, not a precise number.

## How it works

| Stage | Module | Who decides |
|---|---|---|
| Load CDMs, **13-column whitelist** asserted | `orbitguard/data.py` | code |
| Event features from CDMs **at or before 2 days to TCA** only; target = final CDM risk | `data.py` | code |
| Tiers: Critical (Pc ≥ 1e-4 or miss < 1 km) > Elevated (Pc ≥ 1e-6 or miss ≤ 5 km) > Safe | `orbitguard/tiers.py` | **rules** |
| Forecast final log10 Pc: baselines + teammate plug-in | `orbitguard/models.py` | model (numbers only) |
| Forecast tier = rules(pred risk, latest miss) | `tiers.py` | **rules** |
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
