// three.js scenes. Both return { dispose } so the router can free GPU memory on navigation.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function makeRenderer(el, height) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(el.clientWidth, height);
  el.appendChild(renderer.domElement);
  return renderer;
}

function textSprite(text, color = "#c3c2b7", size = 44) {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  c.width = Math.ceil(ctx.measureText(text).width) + 16;
  c.height = size + 16;
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, 8, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(c.width / c.height, 1, 1);
  sp.renderOrder = 10;
  return sp;
}

function disposeScene(scene, renderer, extra) {
  scene.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { m.map && m.map.dispose(); m.dispose(); });
  });
  renderer.dispose();
  renderer.domElement.remove();
  extra && extra();
}

// ---------------------------------------------------------------------------------------------
// Hero: schematic LEO. The dataset has no orbital elements, so positions are illustrative; the
// particle mix follows the real secondary-object-type shares, and red pulses = forecast Critical events.
// ---------------------------------------------------------------------------------------------
export function heroOrbit(el, { mix, nCritical }) {
  const H = el.clientHeight || 360;
  const renderer = makeRenderer(el, H);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, el.clientWidth / H, 0.1, 100);
  camera.position.set(0, 0.9, 5.4);
  const root = new THREE.Group();
  root.position.x = el.clientWidth > 860 ? 1.35 : 0;
  scene.add(root);

  // Earth: dark sphere + fibonacci dot shell + rim glow
  root.add(new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), new THREE.MeshBasicMaterial({ color: 0x0b1626 })));
  const N = 2600, dots = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(1 - y * y), th = Math.PI * (3 - Math.sqrt(5)) * i;
    dots.set([Math.cos(th) * r * 1.002, y * 1.002, Math.sin(th) * r * 1.002], i * 3);
  }
  const dg = new THREE.BufferGeometry();
  dg.setAttribute("position", new THREE.BufferAttribute(dots, 3));
  root.add(new THREE.Points(dg, new THREE.PointsMaterial({ color: 0x2f5f9e, size: 0.012, transparent: true, opacity: 0.75 })));
  const glow = new THREE.Mesh(new THREE.SphereGeometry(1.09, 64, 64), new THREE.ShaderMaterial({
    transparent: true, side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false,
    vertexShader: "varying vec3 vN; void main(){ vN = normalize(normalMatrix*normal); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
    fragmentShader: "varying vec3 vN; void main(){ float i = pow(0.72 - dot(vN, vec3(0,0,1.0)), 3.0); gl_FragColor = vec4(0.22,0.53,0.9,1.0)*i; }",
  }));
  root.add(glow);

  // debris field
  const types = [["DEBRIS", 0x9a9890], ["UNKNOWN", 0x5d5b56], ["PAYLOAD", 0x3987e5], ["ROCKET BODY", 0xd95926]];
  const total = types.reduce((s, [k]) => s + (mix[k] || 0), 0) || 1;
  const P = 3600, pos = new Float32Array(P * 3), col = new Float32Array(P * 3), orb = [];
  let k = 0;
  for (const [name, hex] of types) {
    const n = Math.round((P * (mix[name] || 0)) / total);
    const c = new THREE.Color(hex);
    for (let i = 0; i < n && k < P; i++, k++) {
      orb.push({ r: 1.1 + Math.pow(Math.random(), 2) * 0.32, inc: (Math.random() * 0.9 + 0.1) * Math.PI * (Math.random() < 0.5 ? 0.55 : 0.98),
        raan: Math.random() * Math.PI * 2, ph: Math.random() * Math.PI * 2, w: 0.08 + Math.random() * 0.06 });
      col.set([c.r, c.g, c.b], k * 3);
    }
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  pg.setAttribute("color", new THREE.BufferAttribute(col, 3));
  pg.setDrawRange(0, k);
  root.add(new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.016, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false })));

  // the protected satellite and its orbit
  const satOrbit = { r: 1.22, inc: 1.71, raan: 0.6 };
  const ring = new THREE.EllipseCurve(0, 0, satOrbit.r, satOrbit.r).getPoints(200);
  const rg = new THREE.BufferGeometry().setFromPoints(ring.map((p) => new THREE.Vector3(p.x, 0, p.y)));
  const ringLine = new THREE.Line(rg, new THREE.LineBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.7 }));
  const orbitFrame = new THREE.Group();
  orbitFrame.rotation.set(satOrbit.inc - Math.PI / 2, satOrbit.raan, 0);
  orbitFrame.add(ringLine);
  const sat = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  orbitFrame.add(sat);
  const pulses = [];
  for (let i = 0; i < Math.min(nCritical, 14); i++) {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.032, 24), new THREE.MeshBasicMaterial({ color: 0xd03b3b, transparent: true, side: THREE.DoubleSide }));
    const a = (i / Math.min(nCritical, 14)) * Math.PI * 2 + 0.3;
    m.position.set(Math.cos(a) * satOrbit.r, 0, Math.sin(a) * satOrbit.r);
    m.rotation.x = Math.PI / 2;
    m.userData.phase = Math.random();
    orbitFrame.add(m);
    pulses.push(m);
  }
  root.add(orbitFrame);

  let t = 0, raf = 0, mx = 0, my = 0;
  const onMove = (e) => { const b = el.getBoundingClientRect(); mx = (e.clientX - b.left) / b.width - 0.5; my = (e.clientY - b.top) / b.height - 0.5; };
  el.addEventListener("pointermove", onMove);
  const tmp = new THREE.Vector3();
  const frame = () => {
    t += REDUCED ? 0 : 0.016;
    for (let i = 0; i < k; i++) {
      const o = orb[i], a = o.ph + t * o.w * 2;
      tmp.set(Math.cos(a) * o.r, 0, Math.sin(a) * o.r);
      const ci = Math.cos(o.inc), si = Math.sin(o.inc), y = tmp.z * si, z = tmp.z * ci;
      const cr = Math.cos(o.raan), sr = Math.sin(o.raan);
      pos[i * 3] = tmp.x * cr - z * sr; pos[i * 3 + 1] = y; pos[i * 3 + 2] = tmp.x * sr + z * cr;
    }
    pg.attributes.position.needsUpdate = true;
    const sa = t * 0.35;
    sat.position.set(Math.cos(sa) * satOrbit.r, 0, Math.sin(sa) * satOrbit.r);
    for (const p of pulses) { const s = 1 + ((t * 0.6 + p.userData.phase) % 1) * 2.2; p.scale.setScalar(s); p.material.opacity = 1 - (s - 1) / 2.2; }
    root.rotation.y = t * 0.03 + mx * 0.3;
    root.rotation.x = 0.18 + my * 0.15;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  };
  frame();
  const ro = new ResizeObserver(() => {
    const w = el.clientWidth, h = el.clientHeight || H;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    root.position.x = w > 860 ? 1.35 : 0;
  });
  ro.observe(el);
  return { dispose: () => { cancelAnimationFrame(raf); ro.disconnect(); el.removeEventListener("pointermove", onMove); disposeScene(scene, renderer); } };
}

// ---------------------------------------------------------------------------------------------
// Encounter view in the primary's RTN frame (scene axes: x = T along-track, y = R radial, z = N cross-track).
// Real data: every CDM's predicted miss point and its isotropic radial sigma. Schematic: the straight fly-by
// line (relative velocity assumed perpendicular to the miss vector at TCA, as in the B-plane).
// ---------------------------------------------------------------------------------------------
export function encounterScene(el, cdms, { height = 460, onIndex } = {}) {
  const renderer = makeRenderer(el, height);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, el.clientWidth / height, 0.01, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = !REDUCED;
  controls.autoRotateSpeed = 0.6;
  renderer.domElement.addEventListener("pointerdown", () => (controls.autoRotate = false));

  const km = cdms.map((c) => new THREE.Vector3((c.tt || 0) / 1000, (c.r || 0) / 1000, (c.n || 0) / 1000));
  const maxR = Math.max(...km.map((v) => v.length()), 1);
  const ext = Math.min(Math.max(6.5, maxR * 1.35), 120);
  camera.position.set(ext * 1.55, ext * 1.05, ext * 1.75);
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(3, 5, 2); scene.add(dl);

  const grid = new THREE.GridHelper(ext * 2, 20, 0x2c2c2a, 0x1c1d20);
  scene.add(grid);
  const axis = (dir, color, label) => {
    const g = new THREE.BufferGeometry().setFromPoints([dir.clone().multiplyScalar(-ext), dir.clone().multiplyScalar(ext)]);
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 })));
    const s = textSprite(label, "#c3c2b7"); s.position.copy(dir.clone().multiplyScalar(ext * 1.06)); s.scale.multiplyScalar(ext * 0.045); scene.add(s);
  };
  axis(new THREE.Vector3(1, 0, 0), 0x6a6964, "T along-track");
  axis(new THREE.Vector3(0, 1, 0), 0x6a6964, "R radial");
  axis(new THREE.Vector3(0, 0, 1), 0x6a6964, "N cross-track");

  // danger zones from the handbook rules
  const zone = (rKm, color, op) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(rKm, 48, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false }));
    const w = new THREE.Mesh(new THREE.SphereGeometry(rKm, 24, 16), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: op * 1.6 }));
    scene.add(m); scene.add(w);
  };
  zone(5, 0xfab219, 0.045);
  zone(1, 0xd03b3b, 0.14);

  // primary satellite (drawn at a readable size; true hard-body radius is ~10 m)
  const sat = new THREE.Group();
  const s = ext * 0.035;
  sat.add(new THREE.Mesh(new THREE.BoxGeometry(s, s, s), new THREE.MeshStandardMaterial({ color: 0xdcdcdc, metalness: 0.4, roughness: 0.4 })));
  for (const sx of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(s * 2.4, s * 0.06, s * 0.9), new THREE.MeshStandardMaterial({ color: 0x1c5cab, metalness: 0.2, roughness: 0.6 }));
    p.position.x = sx * s * 1.8; sat.add(p);
  }
  scene.add(sat);

  // trail of CDM miss points (older = dimmer)
  const trailGeo = new THREE.BufferGeometry().setFromPoints(km);
  scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.45 })));
  const dotGeo = new THREE.SphereGeometry(ext * 0.012, 12, 12);
  const dots = km.map((v, i) => {
    const m = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.35 + 0.65 * (i / Math.max(1, km.length - 1)) }));
    m.position.copy(v); scene.add(m); return m;
  });

  // current-CDM group: secondary, sigma sphere, miss vector, fly-by line
  const cur = new THREE.Group(); scene.add(cur);
  const secondary = new THREE.Mesh(new THREE.OctahedronGeometry(ext * 0.03), new THREE.MeshStandardMaterial({ color: 0xff6b6b, emissive: 0x6b1010, roughness: 0.5 }));
  const sigma = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), new THREE.MeshBasicMaterial({ color: 0x9085e9, transparent: true, opacity: 0.16, depthWrite: false }));
  const missLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xf0efec, dashSize: ext * 0.03, gapSize: ext * 0.02 }));
  const flyLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff8d8d, transparent: true, opacity: 0.55 }));
  const mover = new THREE.Mesh(new THREE.SphereGeometry(ext * 0.018, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffd0d0 }));
  cur.add(secondary, sigma, missLine, flyLine, mover);

  let idx = km.length - 1, dir = new THREE.Vector3(1, 0, 0), speedKmS = 10;
  function setIndex(i) {
    idx = Math.max(0, Math.min(km.length - 1, i));
    const p = km[idx], c = cdms[idx];
    secondary.position.copy(p);
    const sg = Math.hypot(c.cs || 0, c.ts || 0) / 1000;
    sigma.visible = Number.isFinite(sg) && sg > 0;
    if (sigma.visible) { sigma.position.copy(p); sigma.scale.setScalar(Math.min(sg, ext * 2)); }
    missLine.geometry.setFromPoints([new THREE.Vector3(), p]); missLine.computeLineDistances();
    const ph = p.clone().normalize();
    const nAxis = new THREE.Vector3(0, 0, 1);
    dir = nAxis.clone().sub(ph.clone().multiplyScalar(nAxis.dot(ph)));
    if (dir.lengthSq() < 1e-6) dir.set(1, 0, 0);
    dir.normalize();
    flyLine.geometry.setFromPoints([p.clone().addScaledVector(dir, -ext * 1.6), p.clone().addScaledVector(dir, ext * 1.6)]);
    speedKmS = (c.speed || 10000) / 1000;
    dots.forEach((d, j) => d.scale.setScalar(j === idx ? 1.9 : 1));
    onIndex && onIndex(idx);
  }
  setIndex(idx);

  let raf = 0, last = performance.now(), phase = -1;
  const frame = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    // fly-by, slowed down: faster encounters (relative speed in km/s) sweep proportionally faster
    if (!REDUCED) phase += dt * 0.14 * (speedKmS / 7);
    if (phase > 1) phase = -1;
    mover.position.copy(km[idx]).addScaledVector(dir, phase * ext * 1.6);
    const sc = ext * 0.0016 * camera.position.length() / ext;
    sat.scale.setScalar(Math.max(0.6, sc));
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  const ro = new ResizeObserver(() => { const w = el.clientWidth; renderer.setSize(w, height); camera.aspect = w / height; camera.updateProjectionMatrix(); });
  ro.observe(el);
  return {
    setIndex,
    get index() { return idx; },
    dispose: () => { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); disposeScene(scene, renderer); },
  };
}
