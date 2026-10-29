// Analytic scene objects (not part of the height grid): vertical cylinders, e.g. a landmark tower.
// They can stand outside the terrain grid. LiDAR rays, sun shadows (World.sunlit) and the GPU camera
// see them; the rover does not collide with them. Surface model = the regolith's (same reflectance /
// Hapke parameters). Config: scenario.objects = [{ name, type: 'cylinder', x, y [m] (world ENU),
// diameter, height [m], enabled }]; the base is sunk 0.5 m below the lowest ground under the footprint
// (grid clamped at its edge when the object is outside), the top is `height` above the ground at (x, y).

import { heightAt } from './terrain.js';

export const OBJECT_LABEL = 4; // point label of LiDAR hits on objects (terrain: 0..3, false alarm 254)

/** Resolve configs into geometry: { name, type, cx, cy, r, zb, zt, outside }. */
export function buildObjects(scenario, hm) {
  const out = [];
  for (const o of scenario.objects || []) {
    if (o.enabled === false || o.type !== 'cylinder') continue;
    const r = o.diameter / 2, [xa, xb, ya, yb] = hm.extent;
    const clampXY = (x, y) => [Math.min(Math.max(x, xa), xb), Math.min(Math.max(y, ya), yb)];
    const ground = (x, y) => heightAt(hm, ...clampXY(x, y));
    let zmin = ground(o.x, o.y);
    for (let k = 0; k < 16; k++) { // footprint rim
      const a = k * Math.PI / 8;
      zmin = Math.min(zmin, ground(o.x + r * Math.cos(a), o.y + r * Math.sin(a)));
    }
    out.push({ name: o.name, type: 'cylinder', cx: o.x, cy: o.y, r, zb: zmin - 0.5, zt: ground(o.x, o.y) + o.height,
      outside: o.x - r < xa || o.x + r > xb || o.y - r < ya || o.y + r > yb });
  }
  return out;
}

/**
 * Nearest intersection of the unit ray with a finite vertical cylinder (side + top cap) in [t0, t1].
 * Returns { t, n: [nx, ny, nz] } or null. The bottom is underground and not tested.
 */
export function rayCylinder(ox, oy, oz, dx, dy, dz, c, t0, t1) {
  let best = null;
  const px = ox - c.cx, py = oy - c.cy, a = dx * dx + dy * dy;
  if (a > 1e-24) { // side: |p + t d|_xy = r
    const b = px * dx + py * dy, cc = px * px + py * py - c.r * c.r, disc = b * b - a * cc;
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      for (const t of [(-b - sq) / a, (-b + sq) / a]) {
        if (t < t0 || t > t1) continue;
        const z = oz + dz * t;
        if (z < c.zb || z > c.zt) continue;
        if (!best || t < best.t) best = { t, n: [(px + dx * t) / c.r, (py + dy * t) / c.r, 0] };
        break; // the first valid root is the nearest
      }
    }
  }
  if (Math.abs(dz) > 1e-15) { // top cap
    const t = (c.zt - oz) / dz;
    if (t >= t0 && t <= t1 && (!best || t < best.t)) {
      const x = px + dx * t, y = py + dy * t;
      if (x * x + y * y <= c.r * c.r) best = { t, n: [0, 0, dz < 0 ? 1 : -1] };
    }
  }
  return best;
}

/** Terrain caster + objects. trace() / occluded() as HeightfieldCaster; traceHit() also returns the object. */
export class SceneCaster {
  constructor(terrainCaster, objects) {
    this.terrain = terrainCaster;
    this.objects = objects;
    this.hm = terrainCaster.hm;
  }

  /** { t, object (null = terrain), n (object normal or null) }; t = Infinity when nothing is hit. */
  traceHit(ox, oy, oz, dx, dy, dz, t0 = 0, t1 = Infinity) {
    let t = this.terrain.trace(ox, oy, oz, dx, dy, dz, t0, t1), object = null, n = null;
    for (const c of this.objects) {
      const h = rayCylinder(ox, oy, oz, dx, dy, dz, c, t0, Math.min(t, t1));
      if (h && h.t < t) { t = h.t; object = c; n = h.n; }
    }
    return { t, object, n };
  }

  trace(ox, oy, oz, dx, dy, dz, t0 = 0, t1 = Infinity) {
    return this.objects.length ? this.traceHit(ox, oy, oz, dx, dy, dz, t0, t1).t : this.terrain.trace(ox, oy, oz, dx, dy, dz, t0, t1);
  }

  occluded(ox, oy, oz, dx, dy, dz, tMax = Infinity) {
    for (const c of this.objects) if (rayCylinder(ox, oy, oz, dx, dy, dz, c, 0, tMax)) return true;
    return this.terrain.occluded(ox, oy, oz, dx, dy, dz, tMax);
  }
}

/**
 * Sun visibility on each object's side wall: Uint8 (nh x na) per object, rows = heights from zb to zt,
 * columns = angles around the axis (0 = +x, counter-clockwise). 0 where the wall faces away from the sun
 * or the ray to the sun is blocked (terrain or another object). For the GPU camera.
 */
export function objectSunVisibility(objects, caster, sunDir, na = 64, nh = 64) {
  return objects.map((c) => {
    const vis = new Uint8Array(na * nh);
    for (let i = 0; i < nh; i++) {
      const z = c.zb + (c.zt - c.zb) * (i + 0.5) / nh;
      for (let j = 0; j < na; j++) {
        const a = 2 * Math.PI * (j + 0.5) / na, nx = Math.cos(a), ny = Math.sin(a);
        if (nx * sunDir[0] + ny * sunDir[1] <= 0) continue;
        const x = c.cx + (c.r + 2e-3) * nx, y = c.cy + (c.r + 2e-3) * ny;
        vis[i * na + j] = caster.occluded(x, y, z, sunDir[0], sunDir[1], sunDir[2]) ? 0 : 255;
      }
    }
    return { vis, na, nh };
  });
}
