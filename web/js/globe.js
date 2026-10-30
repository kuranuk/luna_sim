// Whole-Moon globe for the 3D view: shown when zooming out beyond the generated terrain.
// The globe is drawn in the local ENU frame of the site: Moon centre at (0, 0, -R), the site on top,
// north along +y (Moon-fixed ME frame -> ENU rotation). It is lit by the same sun direction as the
// terrain (sun at infinity), so the terminator is where it really is at the simulated time.
// Texture: web/assets/moon/color.jpg if present (equirectangular, longitude -180..180 left to right,
// north up; e.g. NASA SVS "CGI Moon Kit" lroc_color_poles_*), otherwise a procedural placeholder.

import * as THREE from 'three';

import { enuFromMoonFixed, MOON_RADIUS } from './sim/moon.js';

export { MOON_RADIUS };

const EARTH_RADIUS = 6371000;
const SINK = 200; // [m] globe surface below the site plane: the local terrain always draws on top

function rng(seed) { // small deterministic generator for the placeholder texture
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Procedural placeholder (no real lunar features): dark "maria" blobs + craters on light highlands. */
function placeholderTexture() {
  const w = 2048, h = 1024, cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d'), r = rng(20261030);
  g.fillStyle = '#9a9a96'; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 40; k++) { // maria-like dark patches, mostly on the near side (lon -90..90)
    const x = w * (0.25 + 0.5 * r()), y = h * (0.25 + 0.5 * r()), rad = 30 + 140 * r();
    const grd = g.createRadialGradient(x, y, 0, x, y, rad);
    grd.addColorStop(0, 'rgba(60,60,62,0.55)'); grd.addColorStop(1, 'rgba(60,60,62,0)');
    g.fillStyle = grd; g.beginPath(); g.ellipse(x, y, rad * 1.4, rad, 0, 0, 2 * Math.PI); g.fill();
  }
  for (let k = 0; k < 2500; k++) { // craters: power-law sizes, bright rim + dark floor
    const rad = 1.5 + 40 * Math.pow(r(), 6), x = w * r(), y = h * (0.05 + 0.9 * r());
    const sx = 1 / Math.max(0.15, Math.sin(Math.PI * y / h)); // equirectangular stretch towards the poles
    g.strokeStyle = 'rgba(210,210,205,0.5)'; g.lineWidth = Math.max(1, rad * 0.15);
    g.beginPath(); g.ellipse(x, y, rad * sx, rad, 0, 0, 2 * Math.PI); g.stroke();
    g.fillStyle = 'rgba(70,70,70,0.25)'; g.beginPath(); g.ellipse(x + rad * 0.15 * sx, y + rad * 0.15, rad * 0.8 * sx, rad * 0.8, 0, 0, 2 * Math.PI); g.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class Globe {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    this.group.visible = false;
    scene.add(this.group);
    this.material = new THREE.MeshLambertMaterial({ color: 0xffffff, map: placeholderTexture() });
    this.textureInfo = '仮の模様（手続き生成、実際の地形ではありません）';
    new THREE.TextureLoader().load('assets/moon/color.jpg', (tex) => { // optional real map
      tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
      this.material.map = tex; this.material.needsUpdate = true;
      this.textureInfo = `web/assets/moon/color.jpg（${tex.image.width}×${tex.image.height}）`;
      if (this.onTexture) this.onTexture();
    }, undefined, () => { /* keep the placeholder */ });
    // Moon sphere in Three's sphere frame (y = pole; u = 0 at longitude -180): x = X, y = Z, z = -Y.
    this.sphere = new THREE.Mesh(new THREE.SphereGeometry(MOON_RADIUS - SINK, 256, 128), this.material);
    this.group.add(this.sphere);
    const lines = [], R = MOON_RADIUS * 1.0003;
    const p = (la, lo) => { // Moon-fixed -> sphere frame
      const a = la * Math.PI / 180, b = lo * Math.PI / 180;
      return [R * Math.cos(a) * Math.cos(b), R * Math.sin(a), -R * Math.cos(a) * Math.sin(b)];
    };
    for (let la = -60; la <= 60; la += 30) for (let lo = -180; lo < 180; lo += 2) lines.push(...p(la, lo), ...p(la, lo + 2));
    for (let lo = -180; lo < 180; lo += 30) for (let la = -90; la < 90; la += 2) lines.push(...p(la, lo), ...p(la + 2, lo));
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
    this.graticule = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x5577aa, transparent: true, opacity: 0.35 }));
    this.group.add(this.graticule);
    // site marker (constant screen size) and the Earth
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const g = cv.getContext('2d'); g.strokeStyle = '#ff4d4d'; g.lineWidth = 8; g.beginPath(); g.arc(32, 32, 24, 0, 2 * Math.PI); g.stroke();
    this.marker = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), sizeAttenuation: false, depthTest: false }));
    this.marker.scale.set(0.035, 0.035, 1);
    this.marker.visible = false;
    scene.add(this.marker);
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS, 64, 32), new THREE.MeshLambertMaterial({ color: 0x3a6fd8 }));
    this.earth.visible = false;
    scene.add(this.earth);
    this.center = new THREE.Vector3(0, 0, -MOON_RADIUS);
    this.north = new THREE.Vector3(0, 0, 1);
  }

  /** Place the globe for a site { lat, lon } [deg]. */
  setSite(site) {
    const E = enuFromMoonFixed(site.lat, site.lon); // rows E, N, U in Moon-fixed coordinates
    // ENU <- sphere frame: columns = images of the sphere axes x (= X), y (= Z), z (= -Y)
    const col = (v) => [E[0] * v[0] + E[1] * v[1] + E[2] * v[2], E[3] * v[0] + E[4] * v[1] + E[5] * v[2], E[6] * v[0] + E[7] * v[1] + E[8] * v[2]];
    const cx = col([1, 0, 0]), cy = col([0, 0, 1]), cz = col([0, -1, 0]);
    this.group.matrix.set(cx[0], cy[0], cz[0], 0, cx[1], cy[1], cz[1], 0, cx[2], cy[2], cz[2], -MOON_RADIUS, 0, 0, 0, 1);
    this.group.matrixWorldNeedsUpdate = true;
    this.north.set(...cy).normalize(); // Moon's north pole axis in ENU
    this.site = site;
  }

  /** Earth position from the sun state ({ earthDir, earthDistance } in ENU, if available). */
  setEarth(earthDir, earthDistance) {
    if (!earthDir || !earthDistance) { this.earth.visible = false; this.hasEarth = false; return; }
    this.earth.position.set(earthDir[0] * earthDistance, earthDir[1] * earthDistance, earthDir[2] * earthDistance).add(this.center);
    this.hasEarth = true;
  }

  /** Show / hide by camera distance from the site [m]. */
  update(camDist) {
    const far = camDist > 3000;
    this.group.visible = far;
    this.marker.visible = camDist > 30000;
    this.earth.visible = this.hasEarth && camDist > 30000;
  }
}
