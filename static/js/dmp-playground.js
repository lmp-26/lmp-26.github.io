import { DMP_EXAMPLES } from './dmp-examples.js';
import { fk, ikPath } from './panda.js';

// +esm builds resolve three by full URL, so no import map is needed
const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.170.0/+esm';
const ORBIT_URL = 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js/+esm';
const SVG_URL = 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/renderers/SVGRenderer.js/+esm';
const PANDA_URL = './static/resources/panda/';

// Position DMP exactly as run on the robot (t_v2.py): tau*zdot = az*bz*(g - xi) - az*z + F*f, tau*xidot = z.
const BETA_Z = 5.2;
const ALPHA_Z = 4 * BETA_Z;
const TAU = 1.0;
const FORCE = 2.0;
const ALPHA_X = 4.5;
const WEIGHT_SCALE = 20; // generator weights are scaled by 20 before execution
const NB = 11;
const H = 1 / (0.25 / (NB - 1)) ** 2;
const DT = 0.003;
const T_END = 1.0;
// the robot steps until the phase x drops below 0.0112, i.e. t = tau
const N = Math.ceil(T_END / DT) + 1;
const CENTERS = Array.from({ length: NB }, (_, k) => k / (NB - 1));

const DOFS = [
  { key: 'x', name: 'x', hint: 'forward' },
  { key: 'y', name: 'y', hint: 'right' },
  { key: 'z', name: 'z', hint: 'up' },
];
const EXAMPLES = DMP_EXAMPLES;

const ORANGE = '#e68a00';
const SVGNS = 'http://www.w3.org/2000/svg';

// Basis rows are fixed, so precompute x(t) * psi_k(t) / sum(psi) for every step.
const PHI = (() => {
  const phi = new Float64Array(N * NB);
  for (let i = 1; i < N; i++) {
    const ph = Math.min((i * DT) / TAU, 1);
    const x = Math.exp(-ALPHA_X * ph * ph * ph);
    let sum = 0;
    for (let k = 0; k < NB; k++) {
      const p = Math.exp(-H * (ph - CENTERS[k]) ** 2);
      phi[i * NB + k] = p;
      sum += p;
    }
    for (let k = 0; k < NB; k++) phi[i * NB + k] *= x / sum;
  }
  return phi;
})();

// Same explicit Euler update as DMP.step(): position uses z from before the update.
function rollout(w, y0, g) {
  const ys = new Float64Array(N);
  let xi = y0;
  let z = 0;
  ys[0] = xi;
  for (let i = 1; i < N; i++) {
    let f = 0;
    for (let k = 0; k < NB; k++) f += PHI[i * NB + k] * w[k] * WEIGHT_SCALE;
    const zdot = (ALPHA_Z * BETA_Z * (g - xi) - ALPHA_Z * z + FORCE * f) / TAU;
    const xidot = z / TAU;
    z += zdot * DT;
    xi += xidot * DT;
    ys[i] = xi;
  }
  return ys;
}

function el(tag, attrs, parent) {
  const node = document.createElementNS(SVGNS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}

function pathFrom(xs, ys, sx, sy) {
  let d = '';
  for (let i = 0; i < xs.length; i++) d += (i ? 'L' : 'M') + sx(xs[i]).toFixed(1) + ' ' + sy(ys[i]).toFixed(1);
  return d;
}

const root = document.getElementById('dmp-playground');
if (root) init();

function init() {
  const exampleBar = root.querySelector('.dp-examples');
  const subtaskText = root.querySelector('.dp-subtask-text');
  const dofHost = root.querySelector('.dp-dofs');
  const jsonOut = root.querySelector('.dp-json');

  const state = { ex: EXAMPLES[0], w: null, paths: null, base: null };
  const view = { update: () => {}, replay: () => {}, fit: () => {} };

  EXAMPLES.forEach((ex) => {
    const b = document.createElement('button');
    b.className = 'dp-chip';
    b.type = 'button';
    b.textContent = ex.label;
    b.addEventListener('click', () => load(ex));
    exampleBar.appendChild(b);
  });
  root.querySelector('.dp-reset').addEventListener('click', () => load(state.ex));
  root.querySelector('.dp-zero').addEventListener('click', () => {
    state.w = DOFS.map(() => new Array(NB).fill(0));
    changed();
  });
  root.querySelector('.dp-replay').addEventListener('click', () => view.replay());

  const rows = DOFS.map((dof, d) => buildRow(dof, d));

  function load(ex) {
    state.ex = ex;
    state.w = ex.weights.map((r) => r.slice());
    state.base = DOFS.map((_, d) => rollout(new Array(NB).fill(0), ex.start[d], ex.goal[d]));
    exampleBar.querySelectorAll('.dp-chip').forEach((b, i) => b.classList.toggle('is-active', EXAMPLES[i] === ex));
    subtaskText.textContent = ex.subtask;
    changed();
    view.fit();
    view.replay();
  }

  function changed() {
    const ex = state.ex;
    state.paths = DOFS.map((_, d) => rollout(state.w[d], ex.start[d], ex.goal[d]));
    rows.forEach((r) => r.draw());
    const fmt = (r) => '[' + r.map((v) => v.toFixed(2)).join(', ') + ']';
    jsonOut.textContent = '{\n  "weights": [\n    ' + state.w.map(fmt).join(',\n    ') + '\n  ]\n}';
    view.update();
  }

  function buildRow(dof, d) {
    const row = document.createElement('div');
    row.className = 'dp-row';
    row.innerHTML =
      `<div class="dp-dof-label"><b>${dof.name}</b><span>${dof.hint}</span></div>` +
      '<div class="dp-plot dp-editor"></div><div class="dp-plot dp-time"></div>';
    dofHost.appendChild(row);

    // weight editor
    const W = 460, Hh = 150, padL = 34, padR = 10, padT = 10, padB = 22;
    const ed = el('svg', { viewBox: `0 0 ${W} ${Hh}`, class: 'dp-svg', role: 'application', 'aria-label': `${dof.name} weights` }, row.querySelector('.dp-editor'));
    const sx = (t) => padL + t * (W - padL - padR);
    const sy = (w) => padT + (1 - w) / 2 * (Hh - padT - padB);
    [-1, -0.5, 0, 0.5, 1].forEach((v) => {
      el('line', { x1: padL, x2: W - padR, y1: sy(v), y2: sy(v), class: v === 0 ? 'dp-zero-line' : 'dp-grid' }, ed);
      if (v !== 0.5 && v !== -0.5) el('text', { x: padL - 6, y: sy(v) + 4, class: 'dp-tick', 'text-anchor': 'end' }, ed).textContent = v;
    });
    ['early', 'middle', 'late'].forEach((s, i) => {
      el('text', { x: sx([0.15, 0.5, 0.85][i]), y: Hh - 5, class: 'dp-tick', 'text-anchor': 'middle' }, ed).textContent = s;
    });
    const fills = el('path', { class: 'dp-fill' }, ed);
    const forcing = el('path', { class: 'dp-forcing' }, ed);
    const stems = CENTERS.map((c) => el('line', { x1: sx(c), x2: sx(c), class: 'dp-stem' }, ed));
    const handles = CENTERS.map((c, k) => {
      const h = el('circle', { cx: sx(c), r: 7, class: 'dp-handle', tabindex: 0, 'aria-label': `${dof.name} weight ${k}` }, ed);
      h.addEventListener('keydown', (e) => {
        const step = e.key === 'ArrowUp' ? 0.05 : e.key === 'ArrowDown' ? -0.05 : 0;
        if (!step) return;
        e.preventDefault();
        setW(k, state.w[d][k] + step);
      });
      return h;
    });

    function setW(k, v) {
      state.w[d][k] = Math.round(Math.max(-1, Math.min(1, v)) * 100) / 100;
      changed();
    }

    function fromPointer(e) {
      const pt = ed.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const p = pt.matrixTransform(ed.getScreenCTM().inverse());
      const k = Math.round(((p.x - padL) / (W - padL - padR)) * (NB - 1));
      if (k < 0 || k >= NB) return;
      setW(k, 1 - ((p.y - padT) / (Hh - padT - padB)) * 2);
    }
    let dragging = false;
    ed.addEventListener('pointerdown', (e) => {
      dragging = true;
      try { ed.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
      fromPointer(e);
    });
    ed.addEventListener('pointermove', (e) => { if (dragging) fromPointer(e); });
    ed.addEventListener('pointerup', () => { dragging = false; });
    ed.addEventListener('pointercancel', () => { dragging = false; });

    // position over time
    const TW = 460, TH = 150, tL = 46, tR = 10, tT = 10, tB = 22;
    const ts = el('svg', { viewBox: `0 0 ${TW} ${TH}`, class: 'dp-svg' }, row.querySelector('.dp-time'));
    const tx = (t) => tL + (t / T_END) * (TW - tL - tR);
    const axisG = el('g', {}, ts);
    const goalLine = el('line', { x1: tL, x2: TW - tR, class: 'dp-goal-line' }, ts);
    const basePath = el('path', { class: 'dp-base' }, ts);
    const dmpPath = el('path', { class: 'dp-dmp' }, ts);
    [0, 0.5, 1].forEach((t) => {
      el('text', { x: tx(t), y: TH - 5, class: 'dp-tick', 'text-anchor': 'middle' }, ts).textContent = t + ' s';
    });

    const tGrid = Array.from({ length: 201 }, (_, i) => i / 200);
    const simT = Array.from({ length: N }, (_, i) => i * DT);

    function draw() {
      const w = state.w[d];
      let fd = '';
      CENTERS.forEach((c, k) => {
        const ts = tGrid.filter((t) => Math.abs(t - c) <= 0.08);
        const ys = ts.map((t) => w[k] * Math.exp(-H * (t - c) ** 2));
        fd += `M${sx(ts[0]).toFixed(1)} ${sy(0).toFixed(1)}` + pathFrom(ts, ys, sx, sy).replace('M', 'L') +
          `L${sx(ts[ts.length - 1]).toFixed(1)} ${sy(0).toFixed(1)}Z`;
      });
      fills.setAttribute('d', fd);
      const fs = tGrid.map((t) => {
        let num = 0, den = 0;
        for (let k = 0; k < NB; k++) {
          const p = Math.exp(-H * (t - CENTERS[k]) ** 2);
          num += p * w[k];
          den += p;
        }
        return Math.exp(-ALPHA_X * t * t * t) * num / den;
      });
      forcing.setAttribute('d', pathFrom(tGrid, fs, sx, sy));
      CENTERS.forEach((c, k) => {
        stems[k].setAttribute('y1', sy(0));
        stems[k].setAttribute('y2', sy(w[k]));
        handles[k].setAttribute('cy', sy(w[k]));
      });

      const cur = state.ex;
      const dmp = state.paths[d];
      const base = state.base[d];
      let lo = Infinity, hi = -Infinity;
      const scan = (arr) => { for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; } };
      scan(dmp); scan(base); scan([cur.goal[d]]);
      const span = Math.max(hi - lo, 0.05);
      const mid = (hi + lo) / 2;
      lo = mid - span * 0.6;
      hi = mid + span * 0.6;
      const ty = (v) => tT + (1 - (v - lo) / (hi - lo)) * (TH - tT - tB);
      axisG.innerHTML = '';
      [lo + span * 0.1, mid, hi - span * 0.1].forEach((v) => {
        el('line', { x1: tL, x2: TW - tR, y1: ty(v), y2: ty(v), class: 'dp-grid' }, axisG);
        el('text', { x: tL - 6, y: ty(v) + 4, class: 'dp-tick', 'text-anchor': 'end' }, axisG).textContent = v.toFixed(2);
      });
      goalLine.setAttribute('y1', ty(cur.goal[d]));
      goalLine.setAttribute('y2', ty(cur.goal[d]));
      basePath.setAttribute('d', pathFrom(simT, base, tx, ty));
      dmpPath.setAttribute('d', pathFrom(simT, dmp, tx, ty));
    }

    return { draw };
  }

  load(EXAMPLES[0]);
  initView(root.querySelector('.dp-view'), state, view).catch((err) => {
    console.warn('3D view unavailable:', err);
    const host = root.querySelector('.dp-view');
    host.querySelector('.dp-view-fallback').textContent = `3D view could not load: ${err && err.message ? err.message : err}`;
    host.classList.add('dp-view-failed');
  });
}

async function initView(host, state, view) {
  const THREE = await import(THREE_URL);
  const { OrbitControls } = await import(ORBIT_URL);
  const canvas = host.querySelector('canvas');

  // SVG fallback for browsers without WebGL: same scene, fewer polygons, no tube.
  let renderer;
  let svg = false;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  } catch (err) {
    const { SVGRenderer } = await import(SVG_URL);
    renderer = new SVGRenderer();
    renderer.domElement.classList.add('dp-svg-view');
    host.insertBefore(renderer.domElement, canvas);
    canvas.remove();
    host.classList.add('dp-view-svg');
    svg = true;
  }
  const SEG = svg ? 8 : 20;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f5f5);
  if (svg) renderer.setClearColor(0xf5f5f5);
  // SVGRenderer ignores ambient intensity and adds the full light color, so dim the color itself
  scene.add(svg ? new THREE.AmbientLight(0x8a8a8a) : new THREE.AmbientLight(0xffffff, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, svg ? 0.7 : 1.8);
  sun.position.set(1.5, -1, 2);
  scene.add(sun);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = !svg;
  controls.dampingFactor = 0.08;

  const table = new THREE.Mesh(
    new THREE.PlaneGeometry(1.1, 1.1, svg ? 6 : 1, svg ? 6 : 1),
    new THREE.MeshLambertMaterial({ color: svg ? 0xd4cdbf : 0xe9e3d6 })
  );
  table.position.set(0.45, 0, -0.001);
  table.renderOrder = -2;
  scene.add(table);
  const grid = new THREE.GridHelper(1.1, 11, 0xcfc6b4, 0xd9d1c1);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0.45, 0, 0);
  grid.renderOrder = -1;
  scene.add(grid);

  // axis gizmo with HTML labels so both renderers can show them
  const origin = new THREE.Vector3(0.9, -0.45, 0);
  const axisLabels = [];
  [[1, 0, 0, '#d0402b', 'x'], [0, 1, 0, '#3a8f3a', 'y'], [0, 0, 1, '#2f6db5', 'z']].forEach(([a, b, c, col, t]) => {
    const dir = new THREE.Vector3(a, b, c);
    const end = origin.clone().addScaledVector(dir, 0.12);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, end]), new THREE.LineBasicMaterial({ color: col, linewidth: 3 }));
    scene.add(line);
    const span = document.createElement('span');
    span.className = 'dp-axis-label';
    span.style.color = col;
    span.textContent = t;
    host.appendChild(span);
    axisLabels.push({ span, at: origin.clone().addScaledVector(dir, 0.15) });
  });

  // Panda from franka_description visual meshes; each link mesh sits on its URDF/DH frame
  const lod = svg ? 'lo' : 'hi';
  const manifest = await (await fetch(PANDA_URL + 'panda.json')).json();
  const bin = await (await fetch(PANDA_URL + manifest.lods[lod].file)).arrayBuffer();
  const linkObjs = {};
  const arm = new THREE.Group();
  scene.add(arm);
  ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'finger_l', 'finger_r'].forEach((name) => {
    const g = new THREE.Group();
    g.matrixAutoUpdate = false;
    arm.add(g);
    linkObjs[name] = g;
  });
  manifest.lods[lod].parts.forEach((part) => {
    const q = new Uint16Array(bin, part.vOffset, part.vCount * 3);
    const pos = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) pos[i] = part.min[i % 3] + q[i] * part.scale[i % 3];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array(bin, part.iOffset, part.iCount), 1));
    geo.computeVertexNormals();
    // SVG output skips color management and reads brighter, so tone it down there
    const color = new THREE.Color().setRGB(...part.color, THREE.SRGBColorSpace).multiplyScalar(svg ? 0.72 : 1);
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
    if (part.link === 'finger') {
      linkObjs.finger_l.add(new THREE.Mesh(geo, mat));
      linkObjs.finger_r.add(new THREE.Mesh(geo, mat));
    } else {
      linkObjs[part.link].add(new THREE.Mesh(geo, mat));
    }
  });
  linkObjs.link0.matrix.identity();

  const FINGER = 0.04;
  const tmpA = new THREE.Matrix4();
  const tmpB = new THREE.Matrix4();
  function pose(q) {
    const { frames, hand } = fk(q);
    for (let i = 0; i < 7; i++) linkObjs[`link${i + 1}`].matrix.set(...frames[i]);
    linkObjs.hand.matrix.set(...hand);
    tmpA.set(...hand);
    linkObjs.finger_l.matrix.copy(tmpA).multiply(tmpB.makeTranslation(0, FINGER, 0.0584));
    linkObjs.finger_r.matrix.copy(tmpA).multiply(tmpB.makeTranslation(0, -FINGER, 0.0584)).multiply(tmpB.makeRotationZ(Math.PI));
  }

  const startMesh = new THREE.Mesh(new THREE.SphereGeometry(0.014, SEG * 2, SEG), new THREE.MeshLambertMaterial({ color: 0x7a7a7a }));
  const goalMesh = new THREE.Mesh(new THREE.TorusGeometry(0.022, 0.005, SEG, SEG * 3), new THREE.MeshLambertMaterial({ color: 0xe68a00 }));
  scene.add(startMesh, goalMesh);

  const matPath = new THREE.MeshLambertMaterial({ color: 0xff9900, transparent: true, opacity: 0.3 });
  const matTrail = new THREE.MeshLambertMaterial({ color: 0xff9900 });
  const matPathLine = new THREE.LineBasicMaterial({ color: 0xff9900, linewidth: 2, transparent: true, opacity: 0.45 });
  const matTrailLine = new THREE.LineBasicMaterial({ color: 0xff9900, linewidth: 5 });
  const matShadow = new THREE.LineBasicMaterial({ color: 0xe68a00, transparent: true, opacity: 0.2, linewidth: 1.5 });
  const matBase = new THREE.LineDashedMaterial({ color: 0x8a8a8a, dashSize: 0.015, gapSize: 0.01, linewidth: 1.5 });

  const TUBE_SEGS = 240;
  let pathObj = null, trailObj = null, shadow = null, baseLine = null;
  let pts = [], qs = [];

  function drop(obj) {
    if (!obj) return;
    scene.remove(obj);
    obj.geometry.dispose();
  }

  function rebuild() {
    const ex = state.ex;
    pts = [];
    for (let i = 0; i < N; i += 3) pts.push(new THREE.Vector3(state.paths[0][i], state.paths[1][i], state.paths[2][i]));
    qs = ikPath(pts.map((p) => [p.x, p.y, p.z]));
    drop(pathObj); drop(trailObj); drop(shadow); drop(baseLine);
    if (svg) {
      pathObj = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), matPathLine);
      trailObj = new THREE.Line(new THREE.BufferGeometry(), matTrailLine);
    } else {
      const curve = new THREE.CatmullRomCurve3(pts);
      pathObj = new THREE.Mesh(new THREE.TubeGeometry(curve, TUBE_SEGS, 0.0045, 10, false), matPath);
      trailObj = new THREE.Mesh(new THREE.TubeGeometry(curve, TUBE_SEGS, 0.0065, 10, false), matTrail);
    }
    shadow = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(p.x, p.y, 0.001))), matShadow);
    const bpts = [];
    for (let i = 0; i < N; i += 6) bpts.push(new THREE.Vector3(state.base[0][i], state.base[1][i], state.base[2][i]));
    baseLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(bpts), matBase);
    baseLine.computeLineDistances();
    scene.add(pathObj, trailObj, shadow, baseLine);
    startMesh.position.set(...ex.start);
    goalMesh.position.set(...ex.goal);
  }

  // coalesce rebuilds to one per frame while dragging
  let pending = false;
  view.update = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      rebuild();
    });
  };

  view.fit = () => {
    // frame the path, leaning toward the arm base so both stay in view
    const box = new THREE.Box3().setFromPoints(pts.length ? pts : [new THREE.Vector3(...state.ex.start)]);
    box.expandByPoint(new THREE.Vector3(...state.ex.start));
    box.expandByPoint(new THREE.Vector3(...state.ex.goal));
    const c = box.getCenter(new THREE.Vector3());
    const span = box.getSize(new THREE.Vector3()).length();
    controls.target.set(c.x * 0.75, c.y * 0.75, 0.18);
    camera.position.copy(controls.target).add(new THREE.Vector3(1.0, 0.42, 0.8).normalize().multiplyScalar(1.15 + span));
    controls.update();
  };

  let t0 = performance.now();
  view.replay = () => { t0 = performance.now(); };

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, svg ? undefined : false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);

  let visible = true;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; }, { threshold: 0.05 }).observe(host);
  }

  const PLAY_MS = 2800, HOLD_MS = 1000;
  const v = new THREE.Vector3();
  let lastSvg = 0;

  function setTrail(u) {
    if (svg) {
      const k = Math.max(2, Math.round(u * (pts.length - 1)) + 1);
      trailObj.geometry.dispose();
      trailObj.geometry = new THREE.BufferGeometry().setFromPoints(pts.slice(0, k));
    } else {
      trailObj.geometry.setDrawRange(0, Math.round(u * TUBE_SEGS) * 10 * 6);
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || !qs.length) return;
    // SVG redraws the whole DOM; cap it at ~30 fps
    if (svg && now - lastSvg < 33) return;
    lastSvg = now;
    // rAF timestamps can precede a t0 taken mid-frame
    const u = Math.min((Math.max(0, now - t0) % (PLAY_MS + HOLD_MS)) / PLAY_MS, 1);
    const f = u * (qs.length - 1);
    const i = Math.min(Math.floor(f), qs.length - 2);
    const a = f - i;
    pose(qs[i].map((q, j) => q * (1 - a) + qs[i + 1][j] * a));
    setTrail(u);
    controls.update();
    renderer.render(scene, camera);
    axisLabels.forEach(({ span, at }) => {
      v.copy(at).project(camera);
      span.style.transform = `translate(${((v.x + 1) / 2) * host.clientWidth}px, ${((1 - v.y) / 2) * host.clientHeight}px)`;
    });
  }

  resize();
  rebuild();
  view.fit();
  requestAnimationFrame(frame);
}
