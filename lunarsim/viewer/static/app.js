// LunarSim browser viewer. Talks only to the LunarSim server it is served from.
// World frame: ENU (x East, y North, z Up) used directly in Three.js (camera.up = +z).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const hashToken = new URLSearchParams(location.hash.slice(1)).get('token');
if (hashToken) sessionStorage.setItem('lunarsim-token', hashToken);
const token = sessionStorage.getItem('lunarsim-token');

function showError(msg) {
  const el = $('error');
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 6000);
}

async function api(path, { method = 'GET', body, raw = false } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = JSON.stringify((await res.json()).detail); } catch (_) { /* not JSON */ }
    throw new Error(`${method} ${path}: ${res.status} ${detail}`);
  }
  return raw ? res : res.json();
}

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$('view').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 5000);
camera.up.set(0, 0, 1);
camera.position.set(-12, -12, 8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
// No atmosphere: only a faint fill so shadowed terrain is not pure black in the viewer.
scene.add(new THREE.HemisphereLight(0x8899aa, 0x222222, 0.06));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60, near: 1, far: 3000 });
sun.shadow.bias = -0.0005;
scene.add(sun, sun.target);
const sunArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 6, 0xffcc33, 1.2, 0.6);
scene.add(sunArrow);

const rover = new THREE.Group();
{
  const mat = new THREE.MeshStandardMaterial({ color: 0xd9d9d9, metalness: 0.2, roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.8, 0.35), mat);
  body.position.z = 0.0;
  const mast = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.6), mat);
  mast.position.set(0.45, 0, 0.45);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.45, 0.12), new THREE.MeshStandardMaterial({ color: 0x3366aa }));
  head.position.set(0.5, 0, 0.8);
  const wheelGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.18, 16);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
  for (const [x, y] of [[0.5, 0.45], [0.5, -0.45], [-0.5, 0.45], [-0.5, -0.45]]) {
    const w = new THREE.Mesh(wheelGeo, wheelMat);
    w.position.set(x, y, -0.35);
    rover.add(w);
  }
  rover.add(body, mast, head);
  rover.traverse((o) => { if (o.isMesh) o.castShadow = true; });
}
scene.add(rover);

let terrainMeshes = [];
let lidarPoints = null;

function heightMesh(z, rows, cols, res, x0, y0, color, dz = 0) {
  const pos = new Float32Array(rows * cols * 3);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const k = (i * cols + j) * 3;
      pos[k] = x0 + j * res;
      pos[k + 1] = y0 + i * res;
      pos[k + 2] = z[i * cols + j] + dz;
    }
  }
  const idx = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let n = 0;
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const a = i * cols + j, b = a + 1, c = a + cols, d = c + 1;
      // Same diagonal as the simulator mesh: (i, j)-(i+1, j+1).
      idx[n++] = a; idx[n++] = b; idx[n++] = d;
      idx[n++] = a; idx[n++] = d; idx[n++] = c;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color, roughness: 1.0, metalness: 0.0 }));
  m.receiveShadow = true;
  m.castShadow = true;
  return m;
}

async function loadHeightmap(wid, layer, maxSize) {
  const res = await api(`/worlds/${wid}/heightmap?format=f32&layer=${layer}&max_size=${maxSize}`, { raw: true });
  const h = (k) => res.headers.get(k);
  const z = new Float32Array(await res.arrayBuffer());
  return { z, rows: +h('X-Rows'), cols: +h('X-Cols'), res: +h('X-Resolution-M'), x0: +h('X-Origin-X-M'), y0: +h('X-Origin-Y-M') };
}

// ---------------------------------------------------------------- world state
let wid = null;
let meta = null;
let state = null;
let ws = null;
let running = false;
const cmd = { v: 0, w: 0 };

async function refreshWorlds(select) {
  const { world_ids } = await api('/worlds');
  const sel = $('world');
  sel.innerHTML = '';
  for (const id of world_ids) sel.add(new Option(id.slice(0, 8), id));
  if (select) sel.value = select;
  if (sel.value && sel.value !== wid) await openWorld(sel.value);
  $('create-box').open = world_ids.length === 0;
}

async function openWorld(id) {
  wid = id;
  meta = await api(`/worlds/${id}`);
  state = meta.state;
  for (const m of terrainMeshes) { scene.remove(m); m.geometry.dispose(); }
  terrainMeshes = [];
  if (meta.far_layer) {
    const far = await loadHeightmap(id, 'far', 512);
    terrainMeshes.push(heightMesh(far.z, far.rows, far.cols, far.res, far.x0, far.y0, 0x8a8a8a, -0.05));
  }
  const near = await loadHeightmap(id, 'near', 600);
  terrainMeshes.push(heightMesh(near.z, near.rows, near.cols, near.res, near.x0, near.y0, 0x9a9a9a));
  for (const m of terrainMeshes) scene.add(m);
  const cfg = await api(`/worlds/${id}/config`);
  fillSelect('cam', cfg.cameras.map((c) => c.name));
  // Camera UI only when the world has cameras (LiDAR-only worlds skip the slow renders).
  const hasCam = cfg.cameras.length > 0;
  for (const id of ['cam-row', 'auto-cam-row', 'camera-box', 'cam-hint']) $(id).hidden = !hasCam;
  $('auto-cam').checked = hasCam;
  $('lights').hidden = Object.keys(state.lights).length === 0;
  fillSelect('lidar', cfg.lidars.map((l) => l.name));
  $('lights').innerHTML = '';
  for (const [name, on] of Object.entries(state.lights)) {
    const lab = document.createElement('label');
    lab.innerHTML = `<input type="checkbox" ${on ? 'checked' : ''}> ${name}`;
    lab.querySelector('input').addEventListener('change', async (e) => {
      await api(`/worlds/${id}/lights/${name}`, { method: 'PUT', body: { on: e.target.checked } });
      refreshCamera();
    });
    $('lights').appendChild(lab);
  }
  applyState(state, true);
  connectWs();
  refreshCamera();
  refreshLidar();
}

function fillSelect(id, names) {
  const sel = $(id);
  sel.innerHTML = '';
  for (const n of names) sel.add(new Option(n, n));
  sel.disabled = names.length === 0;
}

function sunDir(azDeg, elDeg) {
  const az = THREE.MathUtils.degToRad(azDeg), el = THREE.MathUtils.degToRad(elDeg);
  return new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el));
}

function applyState(s, snapCamera = false) {
  state = s;
  const r = s.rover;
  const prev = rover.position.clone();
  rover.position.set(...r.position_m);
  const [qw, qx, qy, qz] = r.quaternion_wxyz;
  rover.quaternion.set(qx, qy, qz, qw);
  if ($('follow').checked) {
    const delta = rover.position.clone().sub(prev);
    controls.target.copy(rover.position);
    camera.position.add(snapCamera ? new THREE.Vector3() : delta);
    if (snapCamera) camera.position.copy(rover.position).add(new THREE.Vector3(-10, -10, 6));
  }
  const d = sunDir(s.sun.azimuth_deg, s.sun.elevation_deg);
  sun.target.position.copy(rover.position);
  sun.position.copy(rover.position).addScaledVector(d, 1000);
  sun.intensity = s.sun.elevation_deg > 0 ? 2.2 : 0.0;
  sunArrow.position.copy(rover.position).add(new THREE.Vector3(0, 0, 2));
  sunArrow.setDirection(d);
  if (document.activeElement !== $('sun-az')) $('sun-az').value = s.sun.azimuth_deg.toFixed(0);
  if (document.activeElement !== $('sun-el')) $('sun-el').value = s.sun.elevation_deg.toFixed(1);
  $('sun-az-v').textContent = (+$('sun-az').value).toFixed(0);
  $('sun-el-v').textContent = (+$('sun-el').value).toFixed(1);
  $('hud').textContent = [
    `t = ${s.time_s.toFixed(1)} s`,
    `pos  = ${r.position_m.map((v) => v.toFixed(2)).join(', ')} m`,
    `rpy  = ${r.roll_deg.toFixed(1)}, ${r.pitch_deg.toFixed(1)}, ${r.yaw_deg.toFixed(1)} °`,
    `slope ${r.slope_deg.toFixed(1)}°  slip ${r.slip.toFixed(3)}`,
    `odometer ${r.odometer_m.toFixed(2)} m`,
    r.warnings.length ? `⚠ ${r.warnings.join(', ')}` : '',
  ].join('\n');
}

// ---------------------------------------------------------------- websocket / driving
function connectWs() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const q = token ? `?token=${encodeURIComponent(token)}` : '';
  ws = new WebSocket(`${proto}://${location.host}/ws/${wid}${q}`);
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const m = JSON.parse(ev.data);
    if (m.type === 'state') applyState(m);
    else if (m.type === 'error') showError(m.detail);
  };
  ws.onclose = () => { running = false; $('run').textContent = '▶ Run'; };
}

function send(msg) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }

function setRunning(on) {
  running = on;
  send(on ? { type: 'run', dt_s: 0.2, rate_hz: 5 } : { type: 'stop' });
  $('run').textContent = on ? '⏸ Pause' : '▶ Run';
}
$('run').addEventListener('click', () => setRunning(!running));

const keys = new Set();
function updateCmd() {
  const v = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
  const w = (keys.has('a') ? 1 : 0) - (keys.has('d') ? 1 : 0);
  if (v === cmd.v && w === cmd.w) return;
  cmd.v = v; cmd.w = w;
  send({ type: 'cmd', v_m_s: 0.2 * v, omega_rad_s: 0.35 * w });
  if ((v || w) && !running) setRunning(true);
}
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === ' ') { keys.clear(); cmd.v = cmd.w = NaN; updateCmd(); e.preventDefault(); return; }
  if ('wasd'.includes(k)) { keys.add(k); updateCmd(); }
});
window.addEventListener('keyup', (e) => { keys.delete(e.key.toLowerCase()); updateCmd(); });

// ---------------------------------------------------------------- sun sliders
let sunTimer = null;
for (const id of ['sun-az', 'sun-el']) {
  $(id).addEventListener('input', () => {
    $('sun-az-v').textContent = (+$('sun-az').value).toFixed(0);
    $('sun-el-v').textContent = (+$('sun-el').value).toFixed(1);
    clearTimeout(sunTimer);
    sunTimer = setTimeout(async () => {
      const sunState = await api(`/worlds/${wid}/sun`, {
        method: 'PUT', body: { azimuth_deg: +$('sun-az').value % 360, elevation_deg: +$('sun-el').value },
      });
      applyState({ ...state, sun: sunState });
      refreshCamera();
    }, 250);
  });
}

// ---------------------------------------------------------------- sensors
let camBusy = false;
async function refreshCamera() {
  const name = $('cam').value;
  if (!wid || !name || camBusy) return;
  camBusy = true;
  try {
    const idx = Math.round(state.time_s * 10);
    const res = await api(`/worlds/${wid}/sensors/camera/${name}?format=png&frame_index=${idx}`, { raw: true });
    const url = URL.createObjectURL(await res.blob());
    const img = $('camera-img');
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = url;
    $('camera-cap').textContent = `${name} · t=${state.time_s.toFixed(1)} s · exposure ${(1e3 * +res.headers.get('X-Exposure-S')).toFixed(2)} ms`;
  } catch (e) { showError(e.message); } finally { camBusy = false; }
}

let lidarBusy = false;
async function refreshLidar() {
  const name = $('lidar').value;
  if (!wid || !name || lidarBusy) return;
  lidarBusy = true;
  try {
    const idx = Math.round(state.time_s * 10);
    const res = await api(`/worlds/${wid}/sensors/lidar/${name}?format=bin&frame_index=${idx}`, { raw: true });
    const pts = new Float32Array(await res.arrayBuffer());
    const T = JSON.parse(res.headers.get('X-Pose'));
    const m = new THREE.Matrix4().set(...T.flat());
    const n = pts.length / 4;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const v = new THREE.Vector3();
    for (let k = 0; k < n; k++) {
      v.set(pts[4 * k], pts[4 * k + 1], pts[4 * k + 2]).applyMatrix4(m);
      pos.set([v.x, v.y, v.z], 3 * k);
      const c = 0.3 + 0.7 * Math.min(1, pts[4 * k + 3] * 4);
      col.set([0.2 * c, c, 0.4 * c], 3 * k);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    if (lidarPoints) { scene.remove(lidarPoints); lidarPoints.geometry.dispose(); }
    lidarPoints = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.06, vertexColors: true }));
    scene.add(lidarPoints);
  } catch (e) { showError(e.message); } finally { lidarBusy = false; }
}
$('auto-lidar').addEventListener('change', (e) => { if (!e.target.checked && lidarPoints) lidarPoints.visible = false; });
setInterval(() => {
  if ($('auto-cam').checked) refreshCamera();
  if ($('auto-lidar').checked) { refreshLidar(); if (lidarPoints) lidarPoints.visible = true; }
}, 1000);

// ---------------------------------------------------------------- UI wiring
$('world').addEventListener('change', (e) => openWorld(e.target.value).catch((err) => showError(err.message)));
$('reload').addEventListener('click', () => refreshWorlds().catch((err) => showError(err.message)));
$('create').addEventListener('click', async () => {
  try {
    $('create').disabled = true;
    const r = await api('/worlds', { method: 'POST', body: { config_yaml: $('yaml').value } });
    await refreshWorlds(r.world_id);
  } catch (e) { showError(e.message); } finally { $('create').disabled = false; }
});

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
window.lunarsimViewer = { scene, get state() { return state; } };  // for tests / debugging
refreshWorlds().catch((e) => showError(e.message));
