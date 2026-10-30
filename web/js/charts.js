// Small d3 chart kit. Every chart: recessive grid/axes, thin marks, a hover layer, and text in ink tokens.
import * as d3 from "d3";
import { TIER_COLOR, fmtPc } from "./data.js";

const INK = "#f0efec", INK2 = "#c3c2b7", MUTED = "#898781", GRID = "#2c2c2a", SURFACE = "#131416";
const tip = () => document.getElementById("tooltip");

export function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.style.opacity = 1;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { tip().style.opacity = 0; };
export const ttRow = (k, v) => `<div class="tt-row"><span>${k}</span><b>${v}</b></div>`;

// Render now and again whenever the container width changes.
const observers = new WeakMap();
export function mount(el, draw) {
  const run = () => {
    const w = Math.max(240, el.clientWidth);
    if (el._lastW === w) return;
    el._lastW = w;
    el.innerHTML = "";
    draw(w);
  };
  el._lastW = null;
  run();
  if (observers.has(el)) observers.get(el).disconnect();
  const ro = new ResizeObserver(() => run());
  ro.observe(el);
  observers.set(el, ro);
}

function svgIn(el, w, h) {
  return d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", w).attr("height", h);
}
function styleAxis(g) {
  g.selectAll("path.domain").attr("stroke", "#383835");
  g.selectAll(".tick line").attr("stroke", "#383835");
  g.selectAll("text").attr("fill", MUTED).style("font-size", "11px");
}
function gridY(g, y, w) {
  g.append("g").call(d3.axisLeft(y).tickSize(-w).tickFormat("").ticks(5))
    .call((s) => s.select(".domain").remove())
    .call((s) => s.selectAll("line").attr("stroke", GRID));
}

// ---------- time series vs days-to-TCA ----------
// opts: { height, x: [from, to], y: [lo, hi], yLabel, xLabel, bands, series, vlines, yFmt, tipFmt }
// series item: { name, color, data: [{x, y, ...}], dash, markers, star }
export function timeChart(el, opts) {
  mount(el, (W) => {
    const H = opts.height || 260, m = { t: 14, r: 16, b: 34, l: 48 };
    const w = W - m.l - m.r, h = H - m.t - m.b;
    const svg = svgIn(el, W, H);
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain(opts.x).range([0, w]);
    const y = (opts.yScale === "log" ? d3.scaleLog() : d3.scaleLinear()).domain(opts.y).range([h, 0]).clamp(true);
    for (const b of opts.bands || []) {
      const y0 = y(Math.max(b.y0, opts.y[0])), y1 = y(Math.min(b.y1, opts.y[1]));
      g.append("rect").attr("x", 0).attr("width", w).attr("y", y1).attr("height", Math.max(0, y0 - y1))
        .attr("fill", b.color).attr("opacity", 0.09);
      if (b.label) g.append("text").attr("x", 6).attr("y", y1 + 13).text(b.label).attr("fill", INK2).style("font-size", "10.5px").attr("opacity", 0.8);
    }
    gridY(g, y, w);
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(Math.min(8, w / 70))).call(styleAxis);
    g.append("g").call(d3.axisLeft(y).ticks(5, opts.yScale === "log" ? "~s" : undefined).tickFormat(opts.yFmt || null)).call(styleAxis);
    if (opts.xLabel) g.append("text").attr("x", w).attr("y", h + 30).attr("text-anchor", "end").text(opts.xLabel);
    if (opts.yLabel) g.append("text").attr("x", -m.l + 4).attr("y", -4).text(opts.yLabel);
    for (const v of opts.vlines || []) {
      g.append("line").attr("x1", x(v.x)).attr("x2", x(v.x)).attr("y1", 0).attr("y2", h)
        .attr("stroke", v.color || INK2).attr("stroke-dasharray", "4 4").attr("opacity", 0.7);
      if (v.label) g.append("text").attr("x", x(v.x) + 5).attr("y", 10).text(v.label).attr("fill", INK2).style("font-size", "10.5px");
    }
    const line = d3.line().x((d) => x(d.x)).y((d) => y(d.y)).defined((d) => d.y != null && Number.isFinite(d.y));
    const all = [];
    for (const s of opts.series) {
      const pts = s.data.filter((d) => d.y != null && Number.isFinite(d.y));
      if (!pts.length) continue;
      if (!s.star) {
        g.append("path").datum(pts).attr("d", line).attr("fill", "none").attr("stroke", s.color)
          .attr("stroke-width", 2).attr("stroke-dasharray", s.dash || null).attr("stroke-linejoin", "round");
      }
      if (s.markers !== false) {
        g.selectAll(null).data(pts).join("path")
          .attr("d", d3.symbol(s.star ? d3.symbolStar : (d) => (d.floor ? d3.symbolTriangle : d3.symbolCircle), s.star ? 150 : 40))
          .attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})${d.floor ? " rotate(180)" : ""}`)
          .attr("fill", s.color).attr("stroke", SURFACE).attr("stroke-width", 2);
      }
      pts.forEach((d) => all.push({ ...d, s }));
    }
    // crosshair + nearest-point tooltip
    const cross = g.append("line").attr("y1", 0).attr("y2", h).attr("stroke", INK2).attr("opacity", 0).attr("stroke-width", 1);
    const ring = g.append("circle").attr("r", 6).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 2).attr("opacity", 0);
    g.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent")
      .on("mousemove", (ev) => {
        const [mx, my] = d3.pointer(ev);
        let best = null, bd = Infinity;
        for (const p of all) {
          const dd = Math.hypot(x(p.x) - mx, (y(p.y) - my) * 0.35);
          if (dd < bd) { bd = dd; best = p; }
        }
        if (!best || bd > 60) { hideTip(); cross.attr("opacity", 0); ring.attr("opacity", 0); return; }
        cross.attr("x1", x(best.x)).attr("x2", x(best.x)).attr("opacity", 0.35);
        ring.attr("cx", x(best.x)).attr("cy", y(best.y)).attr("opacity", 1);
        const same = all.filter((p) => Math.abs(p.x - best.x) < 1e-6);
        showTip(`<div class="tt-title">${opts.tipTitle ? opts.tipTitle(best) : `T−${best.x.toFixed(2)} d`}</div>` +
          same.map((p) => ttRow(`<span style="color:${p.s.color}">●</span> ${p.s.name}`, (opts.tipFmt || ((v) => v.toFixed(2)))(p.raw ?? p.y, p))).join("") +
          (opts.tipExtra ? opts.tipExtra(best) : ""), ev);
      })
      .on("mouseleave", () => { hideTip(); cross.attr("opacity", 0); ring.attr("opacity", 0); });
  });
}

// ---------- horizontal bars ----------
// items: [{ label, value, color, text, tip }]
export function hBars(el, items, opts = {}) {
  mount(el, (W) => {
    const rowH = opts.rowH || 26, labelW = opts.labelW || 150;
    const H = items.length * rowH + 8;
    const svg = svgIn(el, W, H);
    const max = opts.max ?? d3.max(items, (d) => d.value) ?? 1;
    const x = d3.scaleLinear().domain([0, max || 1]).range([0, W - labelW - 70]);
    const g = svg.selectAll("g").data(items).join("g").attr("transform", (d, i) => `translate(0,${i * rowH + 4})`);
    g.append("text").attr("x", labelW - 10).attr("y", rowH / 2).attr("dy", "0.32em").attr("text-anchor", "end")
      .text((d) => d.label).attr("fill", INK2).style("font-size", "12px");
    g.append("rect").attr("x", labelW).attr("y", rowH * 0.2).attr("height", rowH * 0.6)
      .attr("width", (d) => Math.max(2, x(d.value))).attr("rx", 4).attr("fill", (d) => d.color || "#3987e5");
    g.append("text").attr("x", (d) => labelW + Math.max(2, x(d.value)) + 6).attr("y", rowH / 2).attr("dy", "0.32em")
      .text((d) => d.text ?? d.value).attr("fill", INK).style("font-size", "11.5px").style("font-variant-numeric", "tabular-nums");
    g.append("rect").attr("width", W).attr("height", rowH).attr("fill", "transparent")
      .on("mousemove", (ev, d) => showTip(d.tip || `<div class="tt-title">${d.label}</div>${ttRow("value", d.text ?? d.value)}`, ev))
      .on("mouseleave", hideTip);
  });
}

// ---------- stacked vertical bars ----------
// cats: x labels; stacks: [{ name, color, values[] }]
export function stackedBars(el, cats, stacks, opts = {}) {
  mount(el, (W) => {
    const H = opts.height || 240, m = { t: 12, r: 10, b: 34, l: 44 };
    const w = W - m.l - m.r, h = H - m.t - m.b;
    const svg = svgIn(el, W, H);
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleBand().domain(cats).range([0, w]).padding(0.22);
    const totals = cats.map((_, i) => d3.sum(stacks, (s) => s.values[i]));
    const y = d3.scaleLinear().domain([0, d3.max(totals) || 1]).nice().range([h, 0]);
    gridY(g, y, w);
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).tickSizeOuter(0)).call(styleAxis);
    g.append("g").call(d3.axisLeft(y).ticks(5, "~s")).call(styleAxis);
    if (opts.xLabel) g.append("text").attr("x", w).attr("y", h + 30).attr("text-anchor", "end").text(opts.xLabel);
    if (opts.yLabel) g.append("text").attr("x", -m.l + 4).attr("y", -2).text(opts.yLabel);
    cats.forEach((c, i) => {
      let acc = 0;
      const col = g.append("g");
      stacks.forEach((s, si) => {
        const v = s.values[i];
        if (!v) return;
        const y0 = y(acc), y1 = y(acc + v);
        const top = si === stacks.length - 1 || stacks.slice(si + 1).every((t) => !t.values[i]);
        col.append("path").attr("fill", s.color)
          .attr("d", roundedTop(x(c), y1 + 1, x.bandwidth(), Math.max(0, y0 - y1 - 2), top ? 4 : 0));
        acc += v;
      });
      col.append("rect").attr("x", x(c)).attr("width", x.bandwidth()).attr("y", 0).attr("height", h).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<div class="tt-title">${opts.tipTitle ? opts.tipTitle(c) : c}</div>` +
          stacks.map((s) => ttRow(`<span style="color:${s.color}">●</span> ${s.name}`, s.values[i].toLocaleString())).join("") +
          ttRow("total", totals[i].toLocaleString()), ev))
        .on("mouseleave", hideTip);
    });
  });
}
function roundedTop(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

// ---------- histogram with reference lines (log10 metres on x) ----------
export function histogram(el, edges, counts, opts = {}) {
  mount(el, (W) => {
    const H = opts.height || 220, m = { t: 14, r: 12, b: 34, l: 44 };
    const w = W - m.l - m.r, h = H - m.t - m.b;
    const svg = svgIn(el, W, H);
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain([edges[0], edges[edges.length - 1]]).range([0, w]);
    const y = d3.scaleLinear().domain([0, d3.max(counts) || 1]).nice().range([h, 0]);
    gridY(g, y, w);
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues(opts.ticks || x.ticks(6)).tickFormat(opts.xFmt || null)).call(styleAxis);
    g.append("g").call(d3.axisLeft(y).ticks(5, "~s")).call(styleAxis);
    if (opts.xLabel) g.append("text").attr("x", w).attr("y", h + 30).attr("text-anchor", "end").text(opts.xLabel);
    counts.forEach((c, i) => {
      const x0 = x(edges[i]) + 1, x1 = x(edges[i + 1]) - 1;
      const mid = (edges[i] + edges[i + 1]) / 2;
      const color = opts.colorAt ? opts.colorAt(mid) : "#3987e5";
      g.append("path").attr("fill", color).attr("d", roundedTop(x0, y(c), Math.max(1, x1 - x0), h - y(c), 3));
      g.append("rect").attr("x", x0).attr("width", Math.max(1, x1 - x0)).attr("height", h).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<div class="tt-title">${opts.binLabel ? opts.binLabel(edges[i], edges[i + 1]) : ""}</div>${ttRow("events", c.toLocaleString())}`, ev))
        .on("mouseleave", hideTip);
    });
    for (const v of opts.vlines || []) {
      g.append("line").attr("x1", x(v.x)).attr("x2", x(v.x)).attr("y1", 0).attr("y2", h).attr("stroke", v.color).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
      g.append("text").attr("x", x(v.x) + 5).attr("y", 10).text(v.label).attr("fill", INK2).style("font-size", "11px");
    }
  });
}

// ---------- confusion matrix (sequential blue by row share) ----------
export function confusion(el, matrix, labels, opts = {}) {
  mount(el, (W) => {
    const size = Math.min(W, 420), m = { t: 30, l: 86, r: 8, b: 8 };
    const cell = (size - m.l - m.r) / labels.length;
    const H = m.t + cell * labels.length + m.b;
    const svg = svgIn(el, W, H);
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const ramp = d3.scaleLinear().domain([0, 0.5, 1]).range(["#1a1b1e", "#1c5cab", "#86b6ef"]);
    g.append("text").attr("x", (cell * labels.length) / 2).attr("y", -16).attr("text-anchor", "middle").text(opts.xLabel || "predicted");
    labels.forEach((l, j) => g.append("text").attr("x", j * cell + cell / 2).attr("y", -4).attr("text-anchor", "middle").text(l).attr("fill", INK2));
    matrix.forEach((row, i) => {
      const tot = d3.sum(row) || 1;
      g.append("text").attr("x", -8).attr("y", i * cell + cell / 2).attr("dy", "0.32em").attr("text-anchor", "end").text(labels[i]).attr("fill", INK2);
      row.forEach((v, j) => {
        const share = v / tot;
        g.append("rect").attr("x", j * cell + 1).attr("y", i * cell + 1).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 4)
          .attr("fill", ramp(share))
          .on("mousemove", (ev) => showTip(`<div class="tt-title">true ${labels[i]} → predicted ${labels[j]}</div>${ttRow("rows", v.toLocaleString())}${ttRow("share of true " + labels[i], (100 * share).toFixed(1) + "%")}`, ev))
          .on("mouseleave", hideTip);
        g.append("text").attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 - 6).attr("text-anchor", "middle").attr("dy", "0.32em")
          .text(v.toLocaleString()).attr("fill", share > 0.55 ? "#0b0b0b" : INK).style("font-weight", 600).style("font-size", "13px").style("pointer-events", "none");
        g.append("text").attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 + 11).attr("text-anchor", "middle").attr("dy", "0.32em")
          .text(`${(100 * share).toFixed(1)}%`).attr("fill", share > 0.55 ? "#1a1a19" : MUTED).style("font-size", "10.5px").style("pointer-events", "none");
      });
    });
    svg.append("text").attr("x", 4).attr("y", m.t + (cell * labels.length) / 2).attr("text-anchor", "middle")
      .attr("transform", `rotate(-90,10,${m.t + (cell * labels.length) / 2})`).text(opts.yLabel || "true");
  });
}

// ---------- conjunction radar: every event's predicted miss vector in the primary's R–T plane ----------
// angle = atan2(radial, along-track) (where the threat passes), radius = sqrt-scaled miss distance.
export function radar(el, events, opts = {}) {
  mount(el, (W) => {
    const S = Math.min(W, opts.max || 520), H = S;
    const svg = svgIn(el, W, H);
    const cx = W / 2, cy = H / 2, R = S / 2 - 26;
    const maxKm = opts.maxKm || 30;
    const r = d3.scaleSqrt().domain([0, maxKm]).range([0, R]).clamp(true);
    const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);
    const defs = svg.append("defs");
    const grad = defs.append("radialGradient").attr("id", "radarBg");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#16202f");
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#0d0f13");
    g.append("circle").attr("r", R).attr("fill", "url(#radarBg)").attr("stroke", "#2c2c2a");
    g.append("circle").attr("r", r(5)).attr("fill", TIER_COLOR.Elevated).attr("opacity", 0.07);
    g.append("circle").attr("r", r(1)).attr("fill", TIER_COLOR.Critical).attr("opacity", 0.16);
    for (const k of [1, 5, 10, 20, maxKm]) {
      g.append("circle").attr("r", r(k)).attr("fill", "none").attr("stroke", k === 1 ? TIER_COLOR.Critical : k === 5 ? TIER_COLOR.Elevated : "#2c2c2a")
        .attr("stroke-dasharray", k <= 5 ? "4 3" : null).attr("opacity", k <= 5 ? 0.8 : 1);
      g.append("text").attr("x", 4).attr("y", -r(k) - 3).text(`${k} km`).style("font-size", "10px").attr("fill", k <= 5 ? INK2 : MUTED);
    }
    for (let a = 0; a < 360; a += 30) {
      const t = (a * Math.PI) / 180;
      g.append("line").attr("x2", R * Math.cos(t)).attr("y2", R * Math.sin(t)).attr("stroke", "#1f2024");
    }
    g.append("text").attr("x", R + 4).attr("y", 4).text("+T").attr("fill", INK2).style("font-size", "11px");
    g.append("text").attr("x", -R - 22).attr("y", 4).text("−T").attr("fill", INK2).style("font-size", "11px");
    g.append("text").attr("x", -8).attr("y", -R - 8).text("+R").attr("fill", INK2).style("font-size", "11px");
    g.append("text").attr("x", -8).attr("y", R + 16).text("−R").attr("fill", INK2).style("font-size", "11px");
    // sweep (decorative motion; reduced-motion users get a static frame via CSS)
    const sweep = g.append("path").attr("d", d3.arc()({ innerRadius: 0, outerRadius: R, startAngle: 0, endAngle: 0.5 }))
      .attr("fill", "#3987e5").attr("opacity", 0.07).attr("class", "sweep");
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      let ang = 0;
      const tick = () => { if (!document.body.contains(el)) return; ang = (ang + 0.6) % 360; sweep.attr("transform", `rotate(${ang})`); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }
    const order = { Safe: 0, Elevated: 1, Critical: 2 };
    const pts = events.filter((e) => e.rr != null && e.rt != null).sort((a, b) => order[a.fcTier] - order[b.fcTier]);
    g.append("circle").attr("r", 5).attr("fill", INK);
    g.selectAll(null).data(pts).join("circle")
      .attr("cx", (e) => { const d = Math.hypot(e.rr, e.rt) / 1000; return r(d) * (e.rt / Math.max(1, Math.hypot(e.rr, e.rt))); })
      .attr("cy", (e) => { const d = Math.hypot(e.rr, e.rt) / 1000; return -r(d) * (e.rr / Math.max(1, Math.hypot(e.rr, e.rt))); })
      .attr("r", (e) => (e.fcTier === "Critical" ? 4.5 : e.fcTier === "Elevated" ? 3.4 : 2.4))
      .attr("fill", (e) => TIER_COLOR[e.fcTier]).attr("opacity", (e) => (e.fcTier === "Safe" ? 0.45 : 0.95))
      .attr("stroke", "#0b0c0e").attr("stroke-width", 1).style("cursor", "pointer")
      .on("mousemove", (ev, e) => {
        d3.select(ev.currentTarget).attr("stroke", INK).attr("stroke-width", 2);
        showTip(`<div class="tt-title">Event ${e.id} · ${e.fcTier} forecast</div>${ttRow("miss (RTN)", (e.miss / 1000).toFixed(3) + " km")}${ttRow("R / T", `${(e.rr / 1000).toFixed(2)} / ${(e.rt / 1000).toFixed(2)} km`)}${ttRow("P(Critical) at TCA", (100 * e.pc).toFixed(1) + "%")}${ttRow("object", e.object)}<div class="tt-row"><span>click to open</span></div>`, ev);
      })
      .on("mouseleave", (ev) => { d3.select(ev.currentTarget).attr("stroke", "#0b0c0e").attr("stroke-width", 1); hideTip(); })
      .on("click", (ev, e) => opts.onClick && opts.onClick(e));
  });
}

// ---------- probability strip (stacked p_safe / p_elevated / p_critical) ----------
export function pbar(e) {
  const seg = (p, c) => (p > 0.004 ? `<span style="width:${(100 * p).toFixed(1)}%;background:${c}"></span>` : "");
  return `<div class="pbar" title="P(Safe) ${(100 * e.ps).toFixed(1)}% · P(Elevated) ${(100 * e.pe).toFixed(1)}% · P(Critical) ${(100 * e.pc).toFixed(1)}%">${seg(e.ps, TIER_COLOR.Safe)}${seg(e.pe, TIER_COLOR.Elevated)}${seg(e.pc, TIER_COLOR.Critical)}</div>`;
}

export { d3, fmtPc };
