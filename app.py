"""OrbitGuard dashboard. Run: streamlit run app.py

Every number on screen is computed by code (rules, statistics, cached reports). Claude only phrases
the per-event briefing, and an analyst records the decision.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from orbitguard import briefing, data, decisions, insights, pc
from orbitguard.models import available_models
from orbitguard.tiers import CRITICAL, ELEVATED, SAFE, TIER_RANK, classify_log, classify_array

st.set_page_config(page_title="OrbitGuard", page_icon="🛰️", layout="wide")

# ---------- design tokens (dataviz reference palette, dark mode) ----------
TIER_COLOR = {CRITICAL: "#d03b3b", ELEVATED: "#fab219", SAFE: "#0ca30c"}
TIER_ICON = {CRITICAL: "🔴", ELEVATED: "🟠", SAFE: "🟢"}
SERIES = "#3987e5"     # categorical slot 1 (blue)
HINDSIGHT = "#898781"  # muted ink for post-cutoff CDMs
INK, INK2, GRID, AXIS, SURFACE = "#f0efec", "#c3c2b7", "#2c2c2a", "#383835", "#1a1a19"
DISPLAY_FLOOR = -15.0  # risk below this is drawn at the floor (hover shows the true value)
REPORTS = data.ROOT / "reports"
DATASETS = {"Test: live view, 2 days before TCA (no ground truth)": "test",
            "Validation: hindsight available (held-out events)": "val"}


def badge(tier: str) -> str:
    return f"{TIER_ICON[tier]} {tier}"


def layout(fig: go.Figure, title: str, height=360, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK)), height=height,
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=12),
                      margin=dict(l=50, r=20, t=50, b=40), hovermode="closest",
                      legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


# ---------- cached data ----------
@st.cache_data(show_spinner="Building event tables…")
def load_events():
    data.build_all()
    return data.load_events()


@st.cache_data
def load_metrics():
    p = REPORTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="Fitting models on the train split…")
def fitted_models():
    train = load_events().query("split == 'train'")
    return {m.name: m.fit(train) for m in available_models()}


@st.cache_data(show_spinner="Scoring events…")
def predictions(split: str, model_name: str) -> pd.DataFrame:
    ev = load_events().query("split == @split").reset_index(drop=True)
    p = fitted_models()[model_name].predict(ev).set_index("event_id").loc[ev["event_id"]].reset_index()
    m = load_metrics()
    theta = m["models"].get(model_name, {}).get("workload", {}).get("theta", np.inf) if m else np.inf
    df = ev.merge(p, on="event_id")
    df["forecast_tier"] = classify_array(df["pred_risk"], df["latest_miss_distance"])
    df["auto_clear"] = (df["p_safe"] >= theta) & ~df["abstain"]
    df["queue"] = np.where(df["abstain"], "Analyst review", np.where(df["auto_clear"], "Auto-clear", "Queue"))
    df.attrs["theta"] = theta
    return df


@st.cache_data(show_spinner="Loading CDMs…")
def cdms_for(dataset_file: str) -> pd.DataFrame:
    return pc.add_pc_columns(data.load_cdms_cached(dataset_file))


@st.cache_data
def load_insights():
    p = REPORTS / "insights_24h.json"
    return json.loads(p.read_text()) if p.exists() else insights.compute()


def dataset_file(split):
    return "test_data" if split == "test" else "train_data"


# ---------- sidebar ----------
PAGES = ["Triage queue", "Event detail", "24-hour warnings", "Model evaluation", "About / limits"]
st.sidebar.title("🛰️ OrbitGuard")
st.sidebar.caption("Collision-risk triage for conjunction assessment analysts. Decisions are made at the 2-day cutoff.")
page = st.sidebar.radio("Page", PAGES, key="page")
ds_label = st.sidebar.radio("Dataset", list(DATASETS), key="ds")
split = DATASETS[ds_label]
model_names = list(fitted_models())
default_model = model_names[-1] if len(model_names) > 2 else "Baseline B: Latest Risk (LRP)"
model_name = st.sidebar.selectbox("Forecast model", model_names, index=model_names.index(default_model))
st.sidebar.caption("Tiers come from fixed rules. Models only forecast the final log10 Pc.")
if load_metrics() is None:
    st.sidebar.warning("No reports/metrics.json yet: run `make eval`.")


def go_detail(eid: int):
    st.session_state.event_id = int(eid)
    st.session_state.page = "Event detail"


# ======================================================================
def page_triage():
    df = predictions(split, model_name)
    st.header("Triage queue")
    st.caption(f"{ds_label} · model: **{model_name}** · sorted by forecast tier, then predicted final risk, "
               "then miss distance.")
    n = len(df)
    c = st.columns(5)
    c[0].metric("Events", f"{n:,}")
    c[1].metric(f"{TIER_ICON[CRITICAL]} Forecast Critical", f"{(df.forecast_tier == CRITICAL).sum():,}")
    c[2].metric(f"{TIER_ICON[ELEVATED]} Forecast Elevated", f"{(df.forecast_tier == ELEVATED).sum():,}")
    c[3].metric("🧑‍🚀 Analyst review (volatile)", f"{df.abstain.sum():,}")
    c[4].metric("✅ Auto-clear candidates", f"{df.auto_clear.mean():.1%}",
                help=f"p_safe ≥ θ={df.attrs['theta']:.3f} (tuned on train for ≥99% Critical recall) and not volatile")

    f = st.columns([1.2, 2, 2, 1.6])
    q = f[0].text_input("Search event_id", "")
    tiers = f[1].multiselect("Forecast tier", [CRITICAL, ELEVATED, SAFE], default=[CRITICAL, ELEVATED, SAFE])
    objs = sorted(df.latest_c_object_type.unique())
    obj = f[2].multiselect("Secondary object type", objs, default=objs)
    queue = f[3].selectbox("Queue", ["All", "Analyst review", "Queue", "Auto-clear"])

    v = df[df.forecast_tier.isin(tiers) & df.latest_c_object_type.isin(obj)]
    if queue != "All":
        v = v[v.queue == queue]
    if q.strip():
        v = v[v.event_id.astype(str).str.contains(q.strip())]
    v = v.assign(_r=v.forecast_tier.map(TIER_RANK)).sort_values(["_r", "pred_risk", "latest_miss_distance"],
                                                               ascending=[False, False, True])
    dec = decisions.load_decisions()
    last_dec = (dec[dec.get("dataset", pd.Series(dtype=str)) == split].groupby("event_id")["decision"].last()
                if len(dec) else pd.Series(dtype=str))
    table = pd.DataFrame({
        "event_id": v.event_id,
        "Forecast": v.forecast_tier.map(badge),
        "Current": v.current_tier.map(badge),
        "Δ": np.select([v.forecast_tier.map(TIER_RANK) > v.current_tier.map(TIER_RANK),
                        v.forecast_tier.map(TIER_RANK) < v.current_tier.map(TIER_RANK)], ["▲", "▼"], "="),
        "Pred Pc @TCA": [f"{10 ** r:.2e}" for r in v.pred_risk],
        "Latest Pc": ["≤1e-30" if r <= -30 else f"{10 ** r:.2e}" for r in v.latest_risk],
        "Miss (km)": (v.latest_miss_distance / 1000).round(3),
        "P(Critical)": v.p_critical.round(3),
        "Queue": v.queue.map({"Analyst review": "🧑‍🚀 Analyst review", "Queue": "📋 Queue", "Auto-clear": "✅ Auto-clear"}),
        "Object": v.latest_c_object_type,
        "Days to TCA": v.latest_time_to_tca.round(2),
        "Decision": [last_dec.get(e, "") for e in v.event_id],
    })
    if split == "val":
        table.insert(3, "Actual final (hindsight)", v.final_tier.map(badge))
    st.caption(f"{len(table):,} events shown. Select a row to open it.")
    sel = st.dataframe(table, hide_index=True, width="stretch", height=480, on_select="rerun",
                       selection_mode="single-row", key="triage_table")
    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        eid = int(table.iloc[rows[0]].event_id)
        st.button(f"Open event {eid} →", on_click=go_detail, args=(eid,), type="primary")


# ======================================================================
def risk_chart(cd: pd.DataFrame, ev, pred, show_post: bool):
    fig = go.Figure()
    for y0, y1, t in [(-4, 0, CRITICAL), (-6, -4, ELEVATED), (DISPLAY_FLOOR - 1, -6, SAFE)]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=TIER_COLOR[t], opacity=0.10, line_width=0,
                      annotation_text=f"{TIER_ICON[t]} {t} (Pc rule)", annotation_position="top left",
                      annotation_font_color=INK2, annotation_font_size=10)
    pre = cd[cd.time_to_tca >= data.CUTOFF_DAYS]
    post = cd[cd.time_to_tca < data.CUTOFF_DAYS]

    def add(d, name, color, dash):
        if d.empty:
            return
        y = d.risk.clip(lower=DISPLAY_FLOOR)
        fig.add_trace(go.Scatter(
            x=d.time_to_tca, y=y, name=name, mode="lines+markers", line=dict(color=color, width=2, dash=dash),
            marker=dict(size=9, symbol=np.where(d.risk < DISPLAY_FLOOR, "triangle-down", "circle"),
                        line=dict(color=SURFACE, width=2)),
            customdata=np.stack([d.risk, d.cdm_row, d.miss_distance], axis=1),
            hovertemplate="t-TCA %{x:.2f} d<br>log10 Pc %{customdata[0]:.2f}<br>miss %{customdata[2]:.0f} m"
                          "<br>CDM row %{customdata[1]}<extra></extra>"))
    add(pre, "CDMs available at decision", SERIES, "solid")
    if show_post:
        add(post, "CDMs after cutoff (hindsight)", HINDSIGHT, "dot")
    fig.add_trace(go.Scatter(x=[ev.latest_time_to_tca, 0], y=[max(ev.latest_risk, DISPLAY_FLOOR), pred.pred_risk],
                             name=f"forecast ({pred.model_name})", mode="lines+markers",
                             line=dict(color=INK, width=2, dash="dash"), marker=dict(size=[0, 12], symbol="star"),
                             hovertemplate="forecast log10 Pc at TCA %{y:.3f}<extra></extra>"))
    fig.add_vline(x=data.CUTOFF_DAYS, line=dict(color=INK, dash="dash", width=1),
                  annotation_text="decision cutoff (2 d)", annotation_font_color=INK)
    layout(fig, "Collision probability vs time to TCA", yaxis_title="log10 Pc",
           xaxis_title="days to TCA")
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(range=[DISPLAY_FLOOR - 0.5, 0])
    return fig


def miss_chart(cd, show_post):
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=1, fillcolor=TIER_COLOR[CRITICAL], opacity=0.10, line_width=0,
                  annotation_text="🔴 < 1 km", annotation_position="top left", annotation_font_color=INK2)
    fig.add_hrect(y0=1, y1=5, fillcolor=TIER_COLOR[ELEVATED], opacity=0.10, line_width=0,
                  annotation_text="🟠 1–5 km", annotation_position="top left", annotation_font_color=INK2)
    for d, name, color, dash in [(cd[cd.time_to_tca >= 2], "at decision", SERIES, "solid"),
                                 (cd[cd.time_to_tca < 2] if show_post else cd.iloc[:0], "after cutoff (hindsight)",
                                  HINDSIGHT, "dot")]:
        if len(d):
            fig.add_trace(go.Scatter(x=d.time_to_tca, y=d.miss_distance / 1000, name=name, mode="lines+markers",
                                     line=dict(color=color, width=2, dash=dash),
                                     marker=dict(size=9, line=dict(color=SURFACE, width=2)),
                                     customdata=d.cdm_row,
                                     hovertemplate="t-TCA %{x:.2f} d<br>miss %{y:.3f} km<br>CDM row %{customdata}"
                                                   "<extra></extra>"))
    fig.add_vline(x=2, line=dict(color=INK, dash="dash", width=1))
    layout(fig, "Miss distance vs time to TCA", yaxis_title="miss distance (km)", xaxis_title="days to TCA")
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(rangemode="tozero")
    return fig


def sphere(center, radius, color, name, opacity=0.25):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return go.Surface(x=x, y=y, z=z, opacity=opacity, showscale=False, name=name, hoverinfo="name",
                      colorscale=[[0, color], [1, color]], showlegend=True)


def geometry_chart(cd, row, show_post):
    """Relative RTN geometry: primary at origin, secondary at (r, t, n) of each CDM. Units: metres."""
    fig = go.Figure()
    hist = cd if show_post else cd[cd.time_to_tca >= 2]
    fig.add_trace(go.Scatter3d(x=hist.relative_position_r, y=hist.relative_position_t, z=hist.relative_position_n,
                               mode="markers+lines", name="predicted miss point, CDM by CDM",
                               marker=dict(size=3, color=SERIES), line=dict(color=SERIES, width=2),
                               customdata=np.stack([hist.time_to_tca, hist.cdm_row], axis=1),
                               hovertemplate="R %{x:.0f} m, T %{y:.0f} m, N %{z:.0f} m<br>t-TCA %{customdata[0]:.2f} d"
                                             "<br>CDM row %{customdata[1]}<extra></extra>"))
    p = (row.relative_position_r, row.relative_position_t, row.relative_position_n)
    fig.add_trace(go.Scatter3d(x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="lines+markers", name="miss vector (selected CDM)",
                               line=dict(color=INK, width=5), marker=dict(size=[6, 4], color=[INK, INK])))
    s = row.sigma_comb
    if np.isfinite(s):
        fig.add_trace(sphere(p, 3 * s, TIER_COLOR[ELEVATED], f"3σ isotropic sphere (σ={s:.0f} m)", 0.15))
        fig.add_trace(sphere(p, s, TIER_COLOR[CRITICAL], "1σ isotropic sphere", 0.35))
    fig.update_layout(scene=dict(xaxis_title="R radial (m)", yaxis_title="T along-track (m)",
                                 zaxis_title="N cross-track (m)", aspectmode="data",
                                 bgcolor=SURFACE, xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                                 zaxis=dict(gridcolor=GRID)))
    return layout(fig, "Encounter geometry: relative RTN miss vector (not trajectories)", height=480)


def page_detail():
    df = predictions(split, model_name)
    ids = df.event_id.tolist()
    default = st.session_state.get("event_id", ids[0])
    if default not in ids:
        default = df.assign(_r=df.forecast_tier.map(TIER_RANK)).sort_values(["_r", "pred_risk"], ascending=False).event_id.iloc[0]
    eid = st.selectbox("Event", ids, index=ids.index(default), key="detail_select")
    st.session_state.event_id = eid
    ev = df[df.event_id == eid].iloc[0]
    cd = cdms_for(dataset_file(split))
    cd = cd[cd.event_id == eid].sort_values("time_to_tca", ascending=False)
    latest = cd[cd.cdm_row == ev.latest_cdm_row].iloc[0]
    cur = classify_log(ev.latest_risk, ev.latest_miss_distance)
    fc = classify_log(ev.pred_risk, ev.latest_miss_distance)

    st.header(f"Event {eid}")
    h = st.columns(5)
    h[0].metric("Current tier (at cutoff)", badge(cur[0]))
    h[1].metric("Forecast tier at TCA", badge(fc[0]))
    h[2].metric("Queue", ev.queue)
    h[3].metric("Latest miss", f"{ev.latest_miss_distance / 1000:.3f} km")
    h[4].metric("Days to TCA at decision", f"{ev.latest_time_to_tca:.2f}")
    show_post = False
    if split == "val":
        show_post = st.toggle("Reveal what happened after the cutoff (hindsight; the analyst could not see this)",
                              value=False)
        if show_post:
            st.info(f"Actual final CDM: {badge(ev.final_tier)} · log10 Pc {ev.final_risk:.2f} · "
                    f"miss {ev.final_miss_distance / 1000:.3f} km · {ev.final_time_to_tca:.2f} d before TCA "
                    f"(CDM row {int(ev.final_cdm_row)})")

    a, b = st.columns(2)
    a.plotly_chart(risk_chart(cd, ev, ev, show_post), width="stretch")
    b.plotly_chart(miss_chart(cd, show_post), width="stretch")

    g, t = st.columns([1.1, 1])
    with g:
        st.plotly_chart(geometry_chart(cd, latest, show_post), width="stretch")
        st.caption("The dataset has no orbits, only the relative position of the secondary at predicted TCA "
                   "(RTN frame of the primary). Each point is one CDM's estimate of the miss point. Spheres use the "
                   "isotropic combined radial sigma, an approximation: real covariances are elongated along-track.")
    with t:
        st.subheader("Rule trace")
        src = f"{dataset_file(split)}.csv data row {int(ev.latest_cdm_row)} (file line {int(ev.latest_cdm_row) + 2})"
        trace = [{"Step": "Current tier", "Input": src, "Tier": badge(cur[0]), "Rules fired": " · ".join(cur[1])},
                 {"Step": "Forecast tier", "Input": f"{ev.model_name}: log10 Pc {ev.pred_risk:.3f} + latest miss",
                  "Tier": badge(fc[0]), "Rules fired": " · ".join(fc[1])}]
        if np.isfinite(latest.pc_approx_log):
            ap = classify_log(latest.pc_approx_log, latest.miss_distance)
            trace.append({"Step": "Independent Pc approx. (HBR 10 m, isotropic)", "Input": src,
                          "Tier": badge(ap[0]), "Rules fired": f"approx Pc {10 ** latest.pc_approx_log:.2e} vs dataset "
                          f"{10 ** latest.risk:.2e} (see Model evaluation: known underestimate)"})
        else:
            trace.append({"Step": "Independent Pc approx.", "Input": src, "Tier": "n/a",
                          "Rules fired": "covariance unavailable (sigma sentinel 6.378e7 m)"})
        trace.append({"Step": "Volatility check", "Input": f"{int(ev.n_cdms_pre)} pre-cutoff CDMs",
                      "Tier": "🧑‍🚀 Analyst review" if ev.volatile else "stable",
                      "Rules fired": ev.volatile_reasons or "none fired"})
        trace.append({"Step": "Auto-clear policy", "Input": f"p_safe {ev.p_safe:.3f} vs θ {df.attrs['theta']:.3f}",
                      "Tier": "✅ eligible" if ev.auto_clear else "not eligible", "Rules fired": ev.queue})
        if show_post:
            fin = classify_log(ev.final_risk, ev.final_miss_distance)
            trace.append({"Step": "Hindsight: actual final", "Input": f"data row {int(ev.final_cdm_row)}",
                          "Tier": badge(fin[0]), "Rules fired": " · ".join(fin[1])})
        st.dataframe(pd.DataFrame(trace), hide_index=True, width="stretch")

        st.subheader("Briefing")
        facts = briefing.build_facts(ev.to_dict(), ev.to_dict(), cur, fc,
                                     latest.pc_approx_log if np.isfinite(latest.pc_approx_log) else None,
                                     bool(ev.auto_clear))
        res = briefing.brief(facts)
        src_lbl = "✍️ Claude (claude-haiku-5-5), verified against facts" if res["source"] == "claude" else "🧾 Deterministic template"
        st.markdown(f"**{src_lbl}**")
        st.write(res["text"])
        st.markdown(f"**Suggested action: {res['recommendation']}** · rule `{facts['suggestion_rule']}`. "
                    "This is a suggestion; the analyst decides.")
        for p_ in res["problems"]:
            st.caption(f"ℹ️ {p_}")
        with st.expander("Facts JSON sent to the LLM (the only source of numbers)"):
            st.json(facts)

    st.subheader("Analyst decision")
    note = st.text_input("Note (optional)", key=f"note_{eid}")
    bc = st.columns([1, 1, 1, 4])
    for col, d, kind in zip(bc, ["Act", "Watch", "Clear"], ["primary", "secondary", "secondary"]):
        if col.button(f"{d}", key=f"dec_{d}_{eid}", type=kind, width="stretch"):
            rec = decisions.log_decision({"dataset": split, "event_id": int(eid), "decision": d, "note": note,
                                          "model": model_name, "current_tier": cur[0], "forecast_tier": fc[0],
                                          "suggested_action": facts["suggested_action"],
                                          "latest_cdm_row": int(ev.latest_cdm_row), "briefing_source": res["source"]})
            st.success(f"Logged {d} at {rec['timestamp_utc']} → data/decisions.jsonl")
    hist = decisions.load_decisions()
    if len(hist):
        hist = hist[(hist.event_id == eid) & (hist.get("dataset") == split)]
        if len(hist):
            st.dataframe(hist, hide_index=True, width="stretch")


# ======================================================================
def page_24h():
    ins = load_insights()
    st.header("24-hour warnings")
    st.caption("Retrospective view over all train events with a CDM < 1 day from TCA (latest such CDM). "
               "The test set stops at 2 days, so there is no live 24 h view.")
    c = st.columns(5)
    c[0].metric("Events < 24 h", f"{ins['n_events']:,}")
    for i, t in enumerate([CRITICAL, ELEVATED, SAFE]):
        c[i + 1].metric(badge(t), f"{ins['by_tier'][t]:,}")
    c[4].metric("Miss < 1 km", f"{ins['under_1km']:,}")
    st.subheader("Insights (computed in code)")
    for s in ins["insights"]:
        st.markdown(f"- {s}")
    a, b = st.columns(2)
    drv = ins["critical_drivers"]
    fig = go.Figure(go.Bar(x=list(drv.values()), y=["miss < 1 km only", "Pc ≥ 1e-4 only", "both"], orientation="h",
                           marker=dict(color=TIER_COLOR[CRITICAL], cornerradius=4), text=list(drv.values()),
                           textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>"))
    a.plotly_chart(layout(fig, "What drives Critical flags (< 24 h)", height=300, xaxis_title="events"),
                   width="stretch")
    obj = pd.DataFrame(ins["by_object_type"])
    fig = go.Figure(go.Bar(x=obj.c_object_type, y=100 * obj.critical_rate, marker=dict(color=SERIES, cornerradius=4),
                           customdata=obj.events, text=[f"{v:.1f}%" for v in 100 * obj.critical_rate],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f}% Critical of %{customdata} events"
                                                                 "<extra></extra>"))
    b.plotly_chart(layout(fig, "Critical rate by secondary object type", height=300, yaxis_title="% Critical"),
                   width="stretch")
    st.dataframe(obj.round(3), hide_index=True, width="stretch")
    st.markdown("**Space weather (F10 terciles)**")
    st.dataframe(pd.DataFrame(ins["by_f10_tercile"]).round(4), hide_index=True, width="stretch")

    cd = cdms_for("train_data")
    w = insights.latest_within_24h(cd)
    w = w[w.tier == CRITICAL].sort_values("miss_distance").head(200)
    st.subheader("Critical events inside 24 h (closest first)")
    st.dataframe(pd.DataFrame({"event_id": w.event_id, "Tier": w.tier.map(badge),
                               "Rules": [" · ".join(r.split(":")[0] for r in classify_log(a_, b_)[1])
                                         for a_, b_ in zip(w.risk, w.miss_distance)],
                               "Miss (km)": (w.miss_distance / 1000).round(3),
                               "log10 Pc": w.risk.round(2), "Hours to TCA": (24 * w.time_to_tca).round(1),
                               "Object": w.c_object_type, "CDM row": w.cdm_row}),
                 hide_index=True, width="stretch", height=320)


# ======================================================================
def page_eval():
    m = load_metrics()
    st.header("Model evaluation (held-out validation events)")
    if m is None:
        st.warning("Run `make eval` first.")
        return
    s = m["split"]
    st.caption(f"Train {s['n_train']:,} / validation {s['n_val']:,} events, split by event_id. "
               f"Validation: {s['val_tier_counts']} · {s['val_high_risk']} high-risk (final log10 Pc ≥ -6).")
    rows = []
    for name, r in m["models"].items():
        b = r["bootstrap"]
        rows.append({"Model": name, "ESA L ↓": f"{r['esa']['L']:.3f}", "L 95% CI": f"[{b['L_ci95'][0]:.2f}, {b['L_ci95'][1]:.2f}]",
                     "MSE_HR": round(r["esa"]["mse_hr"], 3), "F2_HR": round(r["esa"]["f2"], 3),
                     "Crit recall": round(r["tiers"]["critical_recall"], 3),
                     "Crit precision": round(r["tiers"]["critical_precision"], 3),
                     "Crit recall incl. analyst review": round(r["critical_recall_with_review"], 3),
                     "Auto-cleared": f"{100 * r['workload']['pct_cleared']:.1f}%",
                     "Crit recall @ auto-clear": round(r["workload"]["critical_recall"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("ESA L = MSE on true high-risk events ÷ F2 (β=2, threshold -6); lower is better. Only 12 of 97 teams beat "
               "the Latest Risk baseline in ESA's 2019 competition. With 13 high-risk validation events the CI is wide.")

    best = min(m["models"], key=lambda k: m["models"][k]["esa"]["L"])
    w = m["models"][best]["workload"]
    st.subheader("Impact: analyst workload")
    c = st.columns(4)
    c[0].metric("Events auto-cleared", f"{100 * w['pct_cleared']:.1f}%",
                help=f"95% CI {[round(100 * x, 1) for x in m['models'][best]['bootstrap']['pct_cleared_ci95']]}")
    c[1].metric("Critical recall kept", f"{100 * w['critical_recall']:.1f}%")
    c[2].metric("Critical events auto-cleared", w["critical_auto_cleared"])
    c[3].metric("Elevated auto-cleared", w["elevated_auto_cleared"])
    st.caption(f"Model: {best}. Policy: clear if p_safe ≥ θ={w['theta']:.3f} and not volatile, with θ tuned on the TRAIN "
               "split for ≥99% Critical recall and measured here on validation. Of "
               f"{w['n_events']:,} events, {w['n_events'] - w['n_cleared']:,} still need a human; "
               f"{w['n_analyst_review']:,} of those are volatile events routed to analyst review.")

    a, b = st.columns(2)
    a.image(str(REPORTS / "confusion.png"), caption="3-class confusion: true final tier vs forecast tier")
    if (REPORTS / "calibration.png").exists():
        b.image(str(REPORTS / "calibration.png"), caption="Calibration of tier probabilities (models that output them)")
    pers = m["current_tier_persistence"]
    st.caption(f"Reference: 'current tier persists' has accuracy {pers['accuracy']:.3f} and Critical recall "
               f"{pers['critical_recall']:.3f}. Tiers move after the cutoff, which is why forecasting matters.")

    st.subheader("Failure gallery: late surprises")
    gal = json.loads((REPORTS / "failure_gallery.json").read_text())
    vol = m["volatility"]
    st.markdown(f"**{vol['late_surprises']}** validation events jumped > 2 orders of magnitude after the cutoff or escalated "
                f"to Critical. The volatility flag routed **{vol['late_surprises_flagged']}** of them to analyst review; "
                f"**{vol['late_surprises'] - vol['late_surprises_flagged']}** looked calm at the cutoff and no pre-cutoff "
                "model could have seen them coming.")
    g = pd.DataFrame(gal)
    if len(g):
        g["flagged"] = np.where(g.volatile_flag, "🧑‍🚀 caught by volatility flag", "❌ silent miss")
        show = g[["event_id", "flagged", "current_tier", "final_tier", "latest_risk", "final_risk", "jump_orders",
                  "latest_miss_m", "final_miss_m", "volatile_reasons", "latest_cdm_row", "final_cdm_row"]]
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        silent = g[~g.volatile_flag]
        esc = silent[(silent.final_tier == CRITICAL) & (silent.current_tier != CRITICAL)]
        esc = esc.assign(shrink=esc.latest_miss_m - esc.final_miss_m).sort_values("shrink", ascending=False)
        jump = silent.sort_values("jump_orders", ascending=False)
        cases = []
        if len(esc):
            r = esc.iloc[0]
            cases.append((r, f"Silent escalation, event {r.event_id}: {r.current_tier} at the cutoff with a calm history "
                             f"(miss {r.latest_miss_m / 1000:.3f} km). The final CDM put the miss at {r.final_miss_m / 1000:.3f} km, "
                             f"so {r.final_tier} via the 1 km rule. All {len(esc)} silent Critical escalations in validation "
                             "come from miss distance shrinking, not Pc."))
        if len(jump):
            r = jump.iloc[0]
            cases.append((r, f"Silent Pc jump, event {r.event_id}: log10 Pc {r.latest_risk:.2f} at the cutoff, "
                             f"{r.final_risk:.2f} at the final CDM (+{r.jump_orders:.1f} orders of magnitude). LRP predicted "
                             "-6.001 (low risk), so ESA's F2 counts it as a missed high-risk event."))
        for r, msg in cases:
            st.error(msg)
            st.button(f"Open event {r.event_id} (validation)", key=f"fail_{r.event_id}",
                      on_click=lambda e=int(r.event_id): (st.session_state.update(ds=list(DATASETS)[1]), go_detail(e)))

    st.subheader("Independent Pc approximation vs dataset risk")
    pv = json.loads((REPORTS / "pc_validation.json").read_text())
    c = st.columns(4)
    c[0].metric("Spearman (non-floor CDMs)", f"{pv['spearman_non_floor']:.2f}")
    c[1].metric("Median log10 error", f"{pv['median_log_diff_non_floor']:.1f}")
    c[2].metric("High-risk CDMs missed", f"{pv['dataset_high_risk_missed_by_approx']:,} / {pv['dataset_high_risk_cdms']:,}")
    c[3].metric("Implied HBR from max_risk_estimate", f"{pv['pc_max_vs_max_risk_estimate']['median_implied_hbr_m']:.1f} m")
    st.image(str(REPORTS / "pc_validation.png"))
    st.markdown(
        "**Why it diverges (deliberately shown):** only *radial* sigmas are allowed. Radial uncertainty is the smallest "
        f"axis (along-track is typically 10–100× larger), so the isotropic sigma is far too small and "
        "exp(-d²/2σ²) collapses toward zero, giving the median error of ~18 orders of magnitude above. The worst-case Pc_max needs no "
        f"covariance and tracks `max_risk_estimate` (Spearman {pv['pc_max_vs_max_risk_estimate']['spearman']:.2f}), with "
        f"an implied hard-body radius of {pv['pc_max_vs_max_risk_estimate']['median_implied_hbr_m']:.1f} m "
        f"(IQR {pv['pc_max_vs_max_risk_estimate']['iqr_implied_hbr_m'][0]:.1f}–{pv['pc_max_vs_max_risk_estimate']['iqr_implied_hbr_m'][1]:.1f} m), "
        "which supports our 10 m HBR assumption. Tiers therefore use the dataset's `risk`; the approximation is a "
        "cross-check only.")


# ======================================================================
def page_about():
    st.header("About / limits")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname=Helvetica fontcolor="#f0efec" color="#383835"];
  edge [color="#898781"];
  subgraph cluster_code { label="Deterministic code (decides)"; fontcolor="#f0efec"; color="#3987e5";
    cdm [label="CDMs\\n(13 whitelisted columns)" fillcolor="#1a1a19"];
    feat [label="Pre-cutoff features\\n(t-TCA ≥ 2 d)" fillcolor="#1a1a19"];
    model [label="Forecast model\\n(baseline / teammate)" fillcolor="#1a1a19"];
    rules [label="Tier rules\\nCritical / Elevated / Safe" fillcolor="#1a1a19"];
    vol [label="Volatility +\\nauto-clear policy" fillcolor="#1a1a19"];
    facts [label="Facts JSON\\n+ suggested action" fillcolor="#1a1a19"];
  }
  subgraph cluster_llm { label="LLM (reads & phrases only)"; fontcolor="#f0efec"; color="#d95926";
    llm [label="Claude haiku-5-5\\n3-sentence briefing" fillcolor="#1a1a19"]; }
  verify [label="Verifier\\nnumbers ⊆ facts\\nelse template" fillcolor="#1a1a19"];
  human [label="Analyst\\nAct / Watch / Clear" fillcolor="#1a1a19" color="#0ca30c"];
  log [label="decisions.jsonl" fillcolor="#1a1a19"];
  cdm->feat->model->rules; feat->vol; rules->facts; vol->facts; facts->llm->verify->human; facts->human; human->log;
}""")
    st.markdown("""
### Assumptions
- **Units:** `time_to_tca` in days, distances in metres, `relative_speed` in m/s; `risk` = log10 Pc, floored at -30. These were checked against the data.
- **Decision cutoff:** 2.0 days. Events need ≥1 CDM at ≥ 2 d and a final CDM ≤ 1 d from TCA (ESA filter). 3,649 train events were excluded for having no final CDM within 1 day, and 1,212 for having no CDM before the cutoff.
- **Forecast tier** = rules applied to (predicted final log10 Pc, latest pre-cutoff miss distance). The miss distance is carried forward, not forecast.
- **Pc approximation:** isotropic combined sigma σ² = c_sigma_r² + t_sigma_r², small-HBR 2D formula, **HBR = 10 m** (assumed). A sigma of 6.378e7 m is a "no covariance" sentinel and is treated as missing. An unknown Pc never yields Safe.
- **Volatility (routes to analyst review):** clipped-risk std > 1.0, risk rise > 1 order of magnitude, or miss falling below 50% of the first value, all over pre-cutoff CDMs.
- **Auto-clear:** p_safe ≥ θ and not volatile, with θ tuned on train for ≥ 99% Critical recall.

### Known failure modes
- **Late surprises:** some events are calm at 2 days and become Critical in the last day (see the failure gallery). No pre-cutoff model sees them, which is why we never auto-clear volatile events and the human always decides.
- **Critical is almost entirely the 1 km miss rule** (≈99% of Critical events). Predicting the final miss distance matters more than Pc for the tier.
- **Few high-risk events** (13 in validation), so ESA L has a wide confidence interval.
- **Pc approximation underestimates badly** (no along-track or normal sigma). It is shown as a cross-check, never used for tiers.
- **24-hour view is retrospective:** the test set stops at 2 days.
- **The LLM can be wrong.** It never sees anything we don't verify; on any unverifiable number we fall back to a template.
""")


{"Triage queue": page_triage, "Event detail": page_detail, "24-hour warnings": page_24h,
 "Model evaluation": page_eval, "About / limits": page_about}[page]()
