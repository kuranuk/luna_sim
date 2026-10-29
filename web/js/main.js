// LunarSim Web UI: Three.js view (world ENU, camera.up = +z), controls, worker messaging.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { resolveConfig, scenarioToConfig } from './sim/config.js';
import { SCENARIOS, SIZES } from './sim/scenarios.js';

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// ------------------------------------------------------------------ scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
$('view').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 5000);
camera.up.set(0, 0, 1);
camera.position.set(-10, -10, 7);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const ambient = new THREE.AmbientLight(0xffffff, 0.25); // display aid only (the Moon has no ambient light)
scene.add(ambient);
const sunLight = new THREE.DirectionalLight(0xffffff, 2.4);
scene.add(sunLight, sunLight.target);
const sunArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 6, 0xffcc33, 1.2, 0.6);
scene.add(sunArrow);

let terrainMesh = null, terrainInfo = null, rover = null, state = null, scenario = null;
const clouds = {}, lidarStats = {};
let trailLine = null, wpLine = null, estLine = null, gnc = null;
const estArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1.2, 0xff5533, 0.4, 0.25);
estArrow.visible = false;
scene.add(estArrow);
const waypoints = [], wpMarkers = new THREE.Group();
scene.add(wpMarkers);

function makeRover(cfg) {
  const g = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ color: 0xd9dde4 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(cfg.wheelbase + 0.25, cfg.track + 0.05, 0.25), mat);
  body.position.z = -0.12;
  g.add(body);
  const wmat = new THREE.MeshLambertMaterial({ color: 0x333840 });
  for (const [x, y] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(cfg.wheelRadius, cfg.wheelRadius, cfg.wheelWidth, 20), wmat);
    w.position.set(x * cfg.wheelbase / 2, y * (cfg.track / 2 + 0.05), -(cfg.originHeight - cfg.wheelRadius));
    g.add(w);
  }
  scene.add(g);
  return g;
}

function buildTerrainMesh(t) {
  if (terrainMesh) { scene.remove(terrainMesh); terrainMesh.geometry.dispose(); }
  const { m, z, stride, res, x0, y0 } = t, step = stride * res;
  const pos = new Float32Array(m * m * 3), col = new Float32Array(m * m * 3).fill(1);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) pos.set([x0 + b * step, y0 + a * step, z[a * m + b]], 3 * (a * m + b));
  const idx = new Uint32Array((m - 1) * (m - 1) * 6);
  let k = 0;
  for (let a = 0; a < m - 1; a++) {
    for (let b = 0; b < m - 1; b++) { // diagonal (i, j)-(i+1, j+1) as in the simulator
      const p00 = a * m + b, p01 = p00 + 1, p10 = p00 + m, p11 = p10 + 1;
      idx.set([p00, p01, p11, p00, p11, p10], k); k += 6;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  terrainMesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: 0x9a9a9a, vertexColors: true }));
  scene.add(terrainMesh);
}

function applySunlit(msg) {
  if (!terrainMesh || msg.m !== terrainInfo.m) return;
  const col = terrainMesh.geometry.attributes.color;
  for (let k = 0; k < msg.lit.length; k++) { const v = msg.lit[k] ? 1 : 0.07; col.array[3 * k] = col.array[3 * k + 1] = col.array[3 * k + 2] = v; }
  col.needsUpdate = true;
  updateSunVisual(msg.sun);
}

function updateSunVisual(sun) {
  const d = new THREE.Vector3(...sun.dir);
  sunLight.position.copy(d.clone().multiplyScalar(100));
  sunLight.intensity = sun.elDeg > 0 ? 2.4 : 0;
  sunArrow.setDirection(d);
  if (rover) sunArrow.position.copy(rover.position).add(new THREE.Vector3(0, 0, 2));
}

function updateCloud(msg) {
  let pts = clouds[msg.name];
  if (!pts) {
    pts = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.05, vertexColors: true }));
    clouds[msg.name] = pts; scene.add(pts);
  }
  const n = msg.count, colors = new Float32Array(n * 3);
  const base = msg.name.startsWith('spad') ? [0.3, 1.0, 0.55] : [0.4, 0.7, 1.0];
  for (let k = 0; k < n; k++) {
    if (msg.labels[k] === 254) { colors.set([1, 0.25, 0.25], 3 * k); continue; }
    const c = 0.35 + 0.65 * Math.min(1, msg.intensity[k] * 3);
    colors.set([base[0] * c, base[1] * c, base[2] * c], 3 * k);
  }
  pts.geometry.setAttribute('position', new THREE.BufferAttribute(msg.world, 3));
  pts.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  pts.geometry.computeBoundingSphere();
  lidarStats[msg.name] = msg;
}

function updateTrail(trail) {
  if (!trail) return;
  const pos = new Float32Array(trail.length * 3);
  trail.forEach((p, k) => pos.set([p[0], p[1], p[2] + 0.05], 3 * k));
  if (!trailLine) { trailLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd166 })); scene.add(trailLine); }
  trailLine.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  trailLine.geometry.computeBoundingSphere();
}

// Python GNC estimate (orange) vs truth (rover + yellow trail).
function updateEstimate(g) {
  gnc = g;
  if (estLine) { scene.remove(estLine); estLine = null; }
  estArrow.visible = !!(g && g.estimate);
  if (!estArrow.visible) return;
  const e = g.estimate;
  estArrow.position.set(e.x, e.y, e.z + 0.6);
  estArrow.setDirection(new THREE.Vector3(Math.cos(e.yaw), Math.sin(e.yaw), 0));
  if (g.estTrail.length > 1) {
    const pos = new Float32Array(g.estTrail.length * 3);
    g.estTrail.forEach((p, k) => pos.set([p[0], p[1], p[2] + 0.08], 3 * k));
    estLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff5533 }));
    estLine.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(estLine);
  }
}

function gncHud() {
  if (!gnc) return '';
  const lines = [`GNC ${gnc.state === 'connected' ? '接続' : '接続中…'} ${gnc.name}  #${gnc.seq}  ${gnc.rttMs.toFixed(0)} ms${gnc.status ? `  ${gnc.status}` : ''}`];
  if (gnc.estimate) {
    const e = gnc.estimate, t = gnc.truth;
    const dyaw = ((e.yaw - t.yaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
    lines.push(`推定誤差 位置 ${Math.hypot(e.x - t.x, e.y - t.y).toFixed(3)} m  高さ ${(e.z - t.z).toFixed(3)} m  方位 ${(dyaw / DEG).toFixed(2)}°`);
  }
  return lines.join('\n');
}

let lastRoverPos = null;
function applyState(st) {
  state = st;
  if (!rover) return;
  const r = st.rover, R = r.R;
  rover.position.set(...r.position);
  rover.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, 0, 0, 0, 1));
  if ($('follow').checked && lastRoverPos) {
    const delta = rover.position.clone().sub(lastRoverPos);
    camera.position.add(delta); controls.target.add(delta);
  }
  lastRoverPos = rover.position.clone();
  sunArrow.position.copy(rover.position).add(new THREE.Vector3(0, 0, 2));
  const lid = Object.values(lidarStats).map((l) => `${l.name}: ${l.count}/${l.beams} 点 (#${l.frame}, ${l.ms ? l.ms.toFixed(0) : '-'} ms)`).join('\n');
  $('hud').textContent = [
    `t = ${st.time.toFixed(2)} s   ${running ? '実行中' : '停止'}`,
    `pos = ${r.position.map((v) => v.toFixed(2)).join(', ')} m`,
    `yaw ${r.yawDeg.toFixed(1)}°  roll ${r.rollDeg.toFixed(1)}°  pitch ${r.pitchDeg.toFixed(1)}°`,
    `slope ${r.slopeDeg.toFixed(1)}°  slip ${r.slip.toFixed(3)}  odo ${r.odometer.toFixed(2)} m`,
    `cmd v ${r.command.v.toFixed(2)} m/s  ω ${(r.command.omega / DEG).toFixed(1)}°/s${r.waypointsRemaining ? `  残り経由点 ${r.waypointsRemaining}` : ''}`,
    r.warnings.length ? `⚠ ${r.warnings.join(', ')}` : '',
    `sun az ${st.sun.azDeg.toFixed(1)}°  el ${st.sun.elDeg.toFixed(2)}°`,
    gncHud(),
    lid,
  ].filter(Boolean).join('\n');
}

// ------------------------------------------------------------------ worker messages
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'built') {
    $('busy').hidden = true;
    terrainInfo = m.terrain;
    buildTerrainMesh(m.terrain);
    if (rover) scene.remove(rover);
    rover = makeRover(scenario.rover);
    for (const k of Object.keys(clouds)) { scene.remove(clouds[k]); delete clouds[k]; delete lidarStats[k]; }
    for (const k of Object.keys(camStats)) delete camStats[k];
    $('cam-panel').hidden = true; $('cameras').innerHTML = ''; $('cam-grid').innerHTML = '';
    if (trailLine) { scene.remove(trailLine); trailLine = null; }
    updateEstimate(null);
    lastRoverPos = null;
    applyState(m.state);
    const p = rover.position;
    camera.position.set(p.x - 9, p.y - 9, p.z + 6); controls.target.copy(p);
    $('build-info').textContent = `${m.terrain.n}² 点 / ${m.terrain.res} m（${(m.terrain.extent[1] - m.terrain.extent[0]).toFixed(0)} m 四方）、クレーター ${m.terrain.craters}、岩 ${m.terrain.rocks}、生成 ${(m.buildMs / 1000).toFixed(1)} s`;
    $('lidars').innerHTML = '';
    for (const l of m.lidars) {
      const lab = document.createElement('label');
      lab.innerHTML = `<input type="checkbox" checked> ${l.name}（${l.beams} ビーム, ${l.rateHz} Hz）`;
      lab.querySelector('input').addEventListener('change', (e) => {
        worker.postMessage({ type: 'lidar', name: l.name, on: e.target.checked });
        if (clouds[l.name]) clouds[l.name].visible = e.target.checked;
      });
      $('lidars').appendChild(lab);
    }
    setupSensorTab(m.sensors);
    drawObjects(m.objects);
    if (m.outsideObjects.length) {
      $('build-info').textContent += `　※ ${m.outsideObjects.join(', ')} は地形の範囲外（下に地面がありません。範囲 512 m で地面の上に立ちます）`;
    }
    window.lunarsimWeb.ready = true;
  } else if (m.type === 'sunlit') applySunlit(m);
  else if (m.type === 'state') { updateEstimate(m.gnc); applyState(m.state); updateTrail(m.trail); }
  else if (m.type === 'gnc') gncEvent(m);
  else if (m.type === 'camera') showCamera(m);
  else if (m.type === 'cameras') setupCameras(m);
  else if (m.type === 'sensor') sensorApplied(m);
  else if (m.type === 'lidar') updateCloud(m);
  else if (m.type === 'export') {
    const url = URL.createObjectURL(new Blob([m.data], { type: m.mime }));
    const a = document.createElement('a');
    a.href = url; a.download = m.filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } else if (m.type === 'error') { $('busy').hidden = true; $('error').textContent = m.message; console.error(m.message); }
};

// ------------------------------------------------------------------ scene objects (analytic cylinders)
let objectGroup = null;
function drawObjects(objects) {
  if (objectGroup) scene.remove(objectGroup);
  objectGroup = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ color: 0xb0b0b0 });
  for (const c of objects || []) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(c.r, c.r, c.zt - c.zb, 48), mat);
    mesh.rotation.x = Math.PI / 2; // Three's cylinder axis is +y; ours is +z
    mesh.position.set(c.cx, c.cy, (c.zb + c.zt) / 2);
    objectGroup.add(mesh);
  }
  scene.add(objectGroup);
}

// ------------------------------------------------------------------ sensor placement tab (取り付け位置・角度・画角)
// One card per LiDAR / camera / light. Mount = body FLU (x forward, y left, z up; origin = body centre,
// rover.originHeight above the ground), ZYX Euler angles with pitch > 0 = looking down, yaw > 0 = left.
// The worker clamps the values and answers with the applied config, which is kept in sensorOverrides
// so that rebuilds, preset changes and the exported config keep it.
const FOV_FIELDS = {
  flash: [['hfovDeg', '水平'], ['vfovDeg', '垂直']],
  spinning: [['vMinDeg', '仰角 下'], ['vMaxDeg', '上']],
  camera: [['hfovDeg', '水平']],
};
const MOUNT_FIELDS = [['x', 'x 前', 'm'], ['y', 'y 左', 'm'], ['z', 'z 上', 'm'], ['rollDeg', 'ロール', '°'], ['pitchDeg', 'ピッチ 下+', '°'], ['yawDeg', 'ヨー 左+', '°']];
const PROJ_LABEL = { azel: '方位・仰角（鉛直軸、ピッチを仰角に加算）', equiangular: '等角（センサごと傾く）', pinhole: 'ピンホール（平面受光）', equidistant: 'f-θ（等距離射影）' };
const KIND_LABEL = { lidar: 'LiDAR', camera: 'カメラ', light: 'ライト' };
let sensorCfgs = { lidar: [], camera: [], light: [] };
const sensorOverrides = {}; // "kind:name" -> { mount, hfovDeg, ... }; matched by name on rebuild
const fmt = (v) => (Math.round(v * 1000) / 1000).toString();
function vfovText(cfg) { // camera: vertical angle from the aspect ratio (square pixels)
  const t = Math.tan(cfg.hfovDeg * DEG / 2) * cfg.height / cfg.width;
  return `垂直 ${(2 * Math.atan(t) / DEG).toFixed(1)}°`;
}
function mountHint(kind, cfg) {
  const h = cfg.mount.z + (scenario ? scenario.rover.originHeight : 0.5);
  const fov = kind === 'camera' ? `、${vfovText(cfg)}` : '';
  return `平地で地面から ${h.toFixed(2)} m、${cfg.mount.pitchDeg >= 0 ? '下向き' : '上向き'} ${Math.abs(cfg.mount.pitchDeg).toFixed(1)}°${fov}`;
}
function sensorCard(kind, cfg) {
  const card = document.createElement('fieldset');
  card.className = 'sensor-card';
  card.dataset.sensor = `${kind}:${cfg.name}`;
  const fov = FOV_FIELDS[kind === 'camera' ? 'camera' : kind === 'lidar' ? cfg.kind : ''] || [];
  card.innerHTML = `<legend>${KIND_LABEL[kind]} ${cfg.name}${kind === 'lidar' ? `（${cfg.kind}）` : ''}</legend>
    <div class="mount-grid">${MOUNT_FIELDS.map(([k, label, unit]) =>
      `<label>${label}<input type="number" step="${unit === 'm' ? 0.05 : 1}" data-mount="${k}" value="${fmt(cfg.mount[k])}"></label>`).join('')}</div>
    ${fov.length ? `<div class="row wrap">画角 ${fov.map(([k, label]) =>
      `<label>${label} <input type="number" step="1" data-fov="${k}" value="${fmt(cfg[k])}">°</label>`).join(' ')}</div>` : ''}
    ${kind === 'lidar' && cfg.kind === 'flash' ? `<label>投影 <select data-proj>${Object.entries(PROJ_LABEL).map(([k, v]) =>
      `<option value="${k}" ${(cfg.projection || 'azel') === k ? 'selected' : ''}>${v}</option>`).join('')}</select></label>` : ''}
    <p class="hint">${mountHint(kind, cfg)}</p>`;
  for (const inp of card.querySelectorAll('input, select')) {
    inp.addEventListener('change', () => {
      const params = { mount: {} };
      for (const i of card.querySelectorAll('input[data-mount]')) params.mount[i.dataset.mount] = +i.value;
      for (const i of card.querySelectorAll('input[data-fov]')) params[i.dataset.fov] = +i.value;
      const proj = card.querySelector('select[data-proj]');
      if (proj) params.projection = proj.value;
      worker.postMessage({ type: 'sensor', kind, name: cfg.name, params });
    });
  }
  return card;
}
function setupSensorTab(sensors) {
  sensorCfgs = structuredClone(sensors);
  const box = $('mounts');
  box.innerHTML = '';
  for (const kind of ['lidar', 'camera', 'light']) for (const cfg of sensorCfgs[kind]) box.appendChild(sensorCard(kind, cfg));
  updateSensorMarkers();
}
function sensorApplied(m) {
  const list = sensorCfgs[m.kind], k = list.findIndex((c) => c.name === m.name);
  if (k >= 0) list[k] = structuredClone(m.cfg);
  const card = document.querySelector(`.sensor-card[data-sensor="${m.kind}:${m.name}"]`);
  if (card) {
    for (const i of card.querySelectorAll('input[data-mount]')) i.value = fmt(m.cfg.mount[i.dataset.mount]);
    for (const i of card.querySelectorAll('input[data-fov]')) i.value = fmt(m.cfg[i.dataset.fov]);
    card.querySelector('.hint').textContent = mountHint(m.kind, m.cfg);
  }
  const o = { mount: { ...m.cfg.mount } };
  for (const key of ['hfovDeg', 'vfovDeg', 'vMinDeg', 'vMaxDeg', 'projection']) if (m.cfg[key] !== undefined) o[key] = m.cfg[key];
  sensorOverrides[`${m.kind}:${m.name}`] = o;
  updateSensorMarkers();
}
function applySensorOverrides(sc) {
  for (const [key, o] of Object.entries(sensorOverrides)) {
    const [kind, name] = key.split(':');
    const c = ((kind === 'lidar' ? sc.lidars : kind === 'camera' ? sc.cameras : sc.lights) || []).find((x) => x.name === name);
    if (c) { Object.assign(c, o, { mount: { ...c.mount, ...o.mount } }); }
  }
}
// Sensor markers on the rover model: box + viewing direction (+x of the mount) + mast to the body.
const MARK_COLOR = { lidar: 0x55c08a, camera: 0x6aa8ff, light: 0xffd166 };
function updateSensorMarkers() {
  if (!rover) return;
  const old = rover.getObjectByName('sensors');
  if (old) rover.remove(old);
  const g = new THREE.Group();
  g.name = 'sensors';
  const poleMat = new THREE.MeshLambertMaterial({ color: 0xb8bec8 });
  for (const kind of ['lidar', 'camera', 'light']) {
    for (const cfg of sensorCfgs[kind]) {
      const m = cfg.mount, color = MARK_COLOR[kind];
      const holder = new THREE.Group();
      holder.position.set(m.x, m.y, m.z);
      holder.rotation.set(m.rollDeg * DEG, m.pitchDeg * DEG, m.yawDeg * DEG, 'ZYX');
      holder.add(new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.08), new THREE.MeshLambertMaterial({ color })));
      holder.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), kind === 'light' ? 0.4 : 0.7, color, 0.15, 0.08));
      g.add(holder);
      if (m.z > 0.15) { // mast from the top of the body
        const len = m.z - 0.0, pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, len, 8), poleMat);
        pole.rotation.x = Math.PI / 2; pole.position.set(m.x, m.y, len / 2);
        g.add(pole);
      }
    }
  }
  rover.add(g);
}

// ------------------------------------------------------------------ camera (GPU-rendered in the worker)
const camStats = {};
function camTile(name) { // one tile per camera in the image panel (created on demand, in config order)
  let fig = document.querySelector(`#cam-grid figure[data-cam="${name}"]`);
  if (!fig) {
    fig = document.createElement('figure');
    fig.dataset.cam = name;
    fig.innerHTML = '<canvas width="640" height="480"></canvas><figcaption></figcaption>';
    $('cam-grid').appendChild(fig);
  }
  return fig;
}
function updateCamPanel() {
  const tiles = [...document.querySelectorAll('#cam-grid figure')].filter((f) => !f.hidden && camStats[f.dataset.cam]);
  $('cam-panel').hidden = !$('cam-show').checked || !tiles.length;
  $('cam-grid').classList.toggle('multi', tiles.length > 1);
}
function showCamera(m) {
  const fig = camTile(m.name), cv = fig.querySelector('canvas');
  if (cv.width !== m.width || cv.height !== m.height) { cv.width = m.width; cv.height = m.height; }
  const img = new ImageData(m.width, m.height);
  for (let k = 0, p = 0; k < m.pixels.length; k++, p += 4) { const g = m.pixels[k]; img.data[p] = img.data[p + 1] = img.data[p + 2] = g; img.data[p + 3] = 255; }
  cv.getContext('2d').putImageData(img, 0, 0);
  camStats[m.name] = m;
  fig.querySelector('figcaption').textContent = `${m.name} #${m.frame}  t=${m.time.toFixed(1)} s  露光 ${(m.exposure * 1e3).toFixed(2)} ms  飽和 ${m.saturated}  GPU ${m.gpuMs.toFixed(0)} + ${m.sensorMs.toFixed(0)} ms`;
  updateCamPanel();
}
function setupCameras(m) {
  const box = $('cameras');
  box.innerHTML = '';
  if (!m.available) { box.innerHTML = `<p class="hint">カメラは無効: ${m.error || 'GPU なし'}</p>`; return; }
  const grid = $('cam-grid'), names = new Set(m.cameras.map((c) => c.name));
  for (const f of [...grid.children]) if (!names.has(f.dataset.cam)) f.remove(); // keep tiles already drawn
  for (const c of m.cameras) {
    const lab = document.createElement('label'), fig = camTile(c.name);
    grid.appendChild(fig); // config order
    fig.hidden = !c.enabled;
    lab.innerHTML = `<input type="checkbox" ${c.enabled ? 'checked' : ''}> ${c.name}（${c.width}×${c.height}, ${c.rateHz} Hz）`;
    lab.querySelector('input').addEventListener('change', (e) => {
      worker.postMessage({ type: 'camera', name: c.name, on: e.target.checked });
      fig.hidden = !e.target.checked; updateCamPanel();
    });
    box.appendChild(lab);
  }
  if (m.lights.length) {
    const lab = document.createElement('label');
    lab.innerHTML = `<input type="checkbox" checked> ヘッドライト（${m.lights.length} 灯）`;
    lab.querySelector('input').addEventListener('change', (e) => worker.postMessage({ type: 'lights', on: e.target.checked }));
    box.appendChild(lab);
  }
  updateCamPanel();
}
$('cam-show').addEventListener('change', updateCamPanel);

// ------------------------------------------------------------------ controls
for (const k of Object.keys(SCENARIOS)) $('preset').add(new Option(k, k));
for (const k of Object.keys(SIZES)) $('size').add(new Option(k, k));
$('size').value = '205 m / 0.2 m';

// ------------------------------------------------------------------ config files (JSON, see sim/config.js)
let customConfig = null; // { obj, base, label } while a config file is active
function setSizeUi(t) {
  const old = $('size').querySelector('option[value="__config"]');
  if (old) old.remove();
  const key = Object.keys(SIZES).find((k) => SIZES[k].n === t.n && SIZES[k].res === t.res);
  if (key) { $('size').value = key; return; }
  const o = new Option(`設定: ${(t.n - 1) * t.res} m / ${t.res} m`, '__config');
  $('size').add(o); $('size').value = '__config';
}
function loadConfig(obj, label) {
  let sc;
  try { sc = resolveConfig(obj); } catch (e) { $('cfg-info').textContent = ''; $('error').textContent = e.message; return false; }
  customConfig = { obj, base: obj.base || 'south_pole_lowsun', label };
  for (const k of Object.keys(sensorOverrides)) delete sensorOverrides[k]; // the file is the new reference
  $('seed').value = sc.seed;
  $('site').textContent = `${sc.site.lat}°, ${sc.site.lon}°`;
  setSunUi(sc.sun);
  setSizeUi(sc.terrain);
  $('cfg-info').textContent = `設定ファイル: ${label}（base ${customConfig.base}、カメラ ${sc.cameras.length}、LiDAR ${sc.lidars.length}、ライト ${sc.lights.length}）`;
  build();
  return true;
}
function currentConfigText() { // the effective scenario (incl. field-of-view changes) as a complete config
  const sc = structuredClone(scenario);
  applySensorOverrides(sc);
  return scenarioToConfig(sc, customConfig ? customConfig.base : $('preset').value);
}
$('cfg-load').addEventListener('click', () => $('cfg-file').click());
$('cfg-file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try { loadConfig(JSON.parse(await f.text()), f.name); } catch (err) { $('error').textContent = `設定ファイルを読めません: ${err.message}`; }
  e.target.value = '';
});
$('cfg-save').addEventListener('click', () => {
  if (!scenario) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([currentConfigText()], { type: 'application/json' }));
  a.download = 'lunarsim_web_config.json'; a.click();
});
$('cfg-edit').addEventListener('toggle', () => { if ($('cfg-edit').open && scenario) $('cfg-text').value = currentConfigText(); });
$('cfg-apply').addEventListener('click', () => {
  try { loadConfig(JSON.parse($('cfg-text').value), 'エディタ'); } catch (err) { $('error').textContent = `JSON の書式エラー: ${err.message}`; }
});
$('cfg-clear').addEventListener('click', () => { customConfig = null; $('cfg-info').textContent = ''; presetChanged(); build(); });

function presetChanged() {
  customConfig = null; $('cfg-info').textContent = '';
  const sc = SCENARIOS[$('preset').value];
  if ($('size').value === '__config') setSizeUi(sc.terrain); // leaving a config file: back to a preset size
  else { const o = $('size').querySelector('option[value="__config"]'); if (o) o.remove(); }
  $('seed').value = sc.seed;
  $('site').textContent = `${sc.site.lat}°, ${sc.site.lon}°`;
  setSunUi(sc.sun);
}
function setSunUi(s) {
  $('sun-az').value = s.azDeg; $('sun-el').value = s.elDeg;
  $('sun-az-v').textContent = s.azDeg; $('sun-el-v').textContent = s.elDeg;
  $('sun-utc').value = s.utc.slice(0, 16);
  document.querySelector(`input[name=sunmode][value=${s.mode === 'astronomical' ? 'astro' : 'manual'}]`).checked = true;
  sunModeChanged();
}
function sunModeChanged() {
  const astro = document.querySelector('input[name=sunmode]:checked').value === 'astro';
  $('sun-manual').hidden = astro; $('sun-astro').hidden = !astro;
}
$('preset').addEventListener('change', presetChanged);

function build() {
  scenario = customConfig ? resolveConfig(customConfig.obj) : structuredClone(SCENARIOS[$('preset').value]);
  if (SIZES[$('size').value]) Object.assign(scenario.terrain, SIZES[$('size').value]);
  scenario.seed = Math.max(0, Math.floor(+$('seed').value || 0));
  applySensorOverrides(scenario);
  const astro = document.querySelector('input[name=sunmode]:checked').value === 'astro';
  scenario.sun = astro ? { ...scenario.sun, mode: 'astronomical', utc: `${$('sun-utc').value}:00Z` }
    : { ...scenario.sun, mode: 'manual', azDeg: +$('sun-az').value, elDeg: +$('sun-el').value };
  $('busy').hidden = false; $('error').textContent = '';
  window.lunarsimWeb.ready = false;
  worker.postMessage({ type: 'build', scenario });
  setRunning(running);
}
$('build').addEventListener('click', build);

let sunTimer = null;
function sendSun() {
  clearTimeout(sunTimer);
  sunTimer = setTimeout(() => {
    const astro = document.querySelector('input[name=sunmode]:checked').value === 'astro';
    worker.postMessage(astro ? { type: 'sun', utc: `${$('sun-utc').value}:00Z` } : { type: 'sun', azDeg: +$('sun-az').value, elDeg: +$('sun-el').value });
  }, 150);
}
for (const id of ['sun-az', 'sun-el']) {
  $(id).addEventListener('input', () => { $(`${id}-v`).textContent = $(id).value; sendSun(); });
}
$('sun-utc').addEventListener('change', sendSun);
for (const r of document.querySelectorAll('input[name=sunmode]')) r.addEventListener('change', () => { sunModeChanged(); sendSun(); });

let running = false;
function setRunning(on) {
  running = on;
  $('run').textContent = on ? '⏸ 停止' : '▶ 実行';
  worker.postMessage({ type: 'run', running: on });
}
$('run').addEventListener('click', () => setRunning(!running));
$('speed').addEventListener('change', () => worker.postMessage({ type: 'speed', speed: +$('speed').value }));
$('reset').addEventListener('click', () => { worker.postMessage({ type: 'reset' }); if (trailLine) { scene.remove(trailLine); trailLine = null; } updateEstimate(null); lastRoverPos = null; });

const keys = new Set();
function sendKeys() {
  if (!scenario) return;
  const v = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0), w = (keys.has('a') ? 1 : 0) - (keys.has('d') ? 1 : 0);
  worker.postMessage({ type: 'cmd', v: v * scenario.rover.maxSpeed, omega: w * scenario.rover.maxYawRateDeg * DEG });
  if ((v || w) && !running) setRunning(true);
}
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (['w', 'a', 's', 'd'].includes(k) && !keys.has(k)) { keys.add(k); sendKeys(); }
  if (k === ' ') { keys.clear(); sendKeys(); e.preventDefault(); }
});
window.addEventListener('keyup', (e) => { const k = e.key.toLowerCase(); if (keys.delete(k)) sendKeys(); });

// Waypoints: click on the terrain while "経路" is on.
const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!$('wp-mode').checked || !terrainMesh || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(terrainMesh)[0];
  if (!hit) return;
  waypoints.push([hit.point.x, hit.point.y]);
  const mk = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff6699 }));
  mk.position.copy(hit.point).add(new THREE.Vector3(0, 0, 0.15));
  wpMarkers.add(mk);
  drawWpLine();
});
function drawWpLine() {
  if (wpLine) { scene.remove(wpLine); wpLine = null; }
  if (waypoints.length < 1 || !state) return;
  const pts = [new THREE.Vector3(...state.rover.position), ...wpMarkers.children.map((c) => c.position)];
  wpLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0xff6699, dashSize: 0.3, gapSize: 0.2 }));
  wpLine.computeLineDistances();
  scene.add(wpLine);
}
$('wp-go').addEventListener('click', () => {
  if (!waypoints.length) return;
  worker.postMessage({ type: 'waypoints', points: waypoints.slice(), speed: scenario.rover.maxSpeed });
  setRunning(true);
});
$('wp-clear').addEventListener('click', () => {
  waypoints.length = 0; wpMarkers.clear(); drawWpLine();
  worker.postMessage({ type: 'waypoints', points: [], speed: 0 });
});

// GNC: connect the worker to a Python `lunarsim gnc` server.
const GNC_URL_KEY = 'lunarsim.web.gncUrl';
let gncConnected = false;
try { $('gnc-url').value = localStorage.getItem(GNC_URL_KEY) || $('gnc-url').value; } catch (e) { /* storage blocked */ }
function gncConnect(on, url = $('gnc-url').value.trim()) {
  if (on) {
    try { localStorage.setItem(GNC_URL_KEY, url); } catch (e) { /* ignore */ }
    worker.postMessage({ type: 'gnc', connect: true, url, period: +$('gnc-period').value });
  } else worker.postMessage({ type: 'gnc', connect: false });
}
function gncEvent(m) {
  const st = $('gnc-status');
  if (m.event === 'connecting') { gncConnected = true; st.textContent = '接続中…'; }
  else if (m.event === 'ready') { st.textContent = `接続: ${m.name}（▶ 実行で走行）`; $('error').textContent = ''; }
  else if (m.event === 'closed') { gncConnected = false; st.textContent = `未接続${m.code !== 1000 && m.code !== 1005 ? `（切断 ${m.code}）` : ''}`; updateEstimate(null); }
  else if (m.event === 'error') { $('error').textContent = m.message; if (m.paused) setRunning(false); }
  $('gnc-connect').textContent = gncConnected ? '切断' : '接続';
}
$('gnc-connect').addEventListener('click', () => gncConnect(!gncConnected));

for (const b of document.querySelectorAll('[data-export]')) {
  b.addEventListener('click', () => {
    const what = b.dataset.export;
    if (what === 'png') { // the 8-bit display images (gamma 2.2), one file per camera shown
      for (const fig of document.querySelectorAll('#cam-grid figure')) {
        if (fig.hidden || !camStats[fig.dataset.cam]) continue;
        fig.querySelector('canvas').toBlob((blob) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${fig.dataset.cam}_preview.png`; a.click(); });
      }
      return;
    }
    if (what === 'pgm') { for (const name of Object.keys(camStats)) worker.postMessage({ type: 'export', what, name }); return; }
    const name = what === 'ply' ? (Object.keys(lidarStats)[0] || 'spad_flash') : undefined;
    worker.postMessage({ type: 'export', what, name });
  });
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

for (const b of document.querySelectorAll('#tabs button')) {
  b.addEventListener('click', () => {
    for (const x of document.querySelectorAll('#tabs button')) x.classList.toggle('active', x === b);
    for (const p of document.querySelectorAll('[data-panel]')) p.hidden = p.dataset.panel !== b.dataset.tab;
  });
}

window.lunarsimWeb = { ready: false, get state() { return state; }, get lidar() { return lidarStats; }, get gnc() { return gnc; }, get camera() { return camStats; },
  worker, build, setRunning, gncConnect };
presetChanged();
build();
const cfgUrl = new URLSearchParams(location.search).get('config'); // e.g. ?config=configs/example.json
if (cfgUrl) {
  fetch(cfgUrl).then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); })
    .then((obj) => loadConfig(obj, cfgUrl)).catch((e) => { $('error').textContent = `設定ファイル ${cfgUrl} を読めません: ${e.message}`; });
}
const autoGnc = new URLSearchParams(location.search).get('gnc'); // e.g. ?gnc=ws://127.0.0.1:8765/
if (autoGnc) { $('gnc-url').value = autoGnc; gncConnect(true, autoGnc); }
