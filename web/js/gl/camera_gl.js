// GPU camera renderer (WebGL2 via Three.js on an OffscreenCanvas, runs in the simulation worker).
// Produces the in-band radiance image [W m^-2 sr^-1] that ../sim/camera.js expose() digitises:
//   L = E_sun band_fraction r(n, s, v) V_sun + Σ_lights r(n, l, v) I(θ) / d² V_light
// r = Hapke (same formula as brdf.py / brdf.js, float precision). V_sun = bilinear interpolation of
// the per-node shadow-ray visibility computed on the CPU with the same raycaster as the LiDAR
// (World.sunlitFull); V_light = distance shadow map rendered from each light. Shading normal =
// interpolated node gradients (as the Python renderer). No ambient light, no earthshine, pinhole
// only. GPU float arithmetic: images are not bit-identical across GPUs (the noise is seeded).
// Three.js is imported from the vendored copy (import maps / SRI do not apply inside workers).

import * as THREE from '../../vendor/three.module.js';
import { HAPKE_GLSL, hapkeUniforms } from '../sim/brdf.js';
import { spotIntensity } from '../sim/camera.js';
import { compose, mountPose } from '../sim/frames.js';

const SHADOW_SIZE = 1024;

const VERT = /* glsl */ `
in vec2 grad;
out vec3 vPos;
out vec2 vGrad;
out vec2 vUv;
uniform vec2 uGrid; // n, unused padding
void main() {
  vPos = position; vGrad = grad; vUv = uv;
  gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
}`;

function fragmentShader(nLights) {
  let lights = '';
  for (let k = 0; k < nLights; k++) {
    lights += `
  {
    vec3 d = uLightPos[${k}] - vPos;
    float dist = length(d);
    vec3 l = d / dist;
    float th = acos(clamp(dot(-l, uLightAxis[${k}]), -1.0, 1.0));
    float t = clamp((th - uLightCone[${k}].x) / (uLightCone[${k}].y - uLightCone[${k}].x), 0.0, 1.0);
    float I = uLightCone[${k}].z * (1.0 - t * t * (3.0 - 2.0 * t));
    if (I > 0.0) {
      vec4 c = uLightVP[${k}] * vec4(vPos, 1.0);
      vec3 ndc = c.xyz / c.w;
      float vis = 1.0;
      if (c.w > 0.0 && all(lessThan(abs(ndc.xy), vec2(1.0)))) {
        float stored = texture(uLightDist${k}, ndc.xy * 0.5 + 0.5).r;
        vis = dist > stored + 0.01 + 0.004 * dist ? 0.0 : 1.0;
      }
      L += hapke(n, l, v) * I / (dist * dist) * vis;
    }
  }`;
  }
  const samplers = Array.from({ length: nLights }, (_, k) => `uniform sampler2D uLightDist${k};`).join('\n');
  return /* glsl */ `
precision highp float;
in vec3 vPos;
in vec2 vGrad;
in vec2 vUv;
out vec4 outColor;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform float uSunE;
uniform sampler2D uLit;
${nLights ? `uniform vec3 uLightPos[${nLights}];\nuniform vec3 uLightAxis[${nLights}];\nuniform vec3 uLightCone[${nLights}];\nuniform mat4 uLightVP[${nLights}];` : ''}
${samplers}
${HAPKE_GLSL}
void main() {
  vec3 n = normalize(vec3(-vGrad, 1.0));
  vec3 v = normalize(uCamPos - vPos);
  float L = 0.0;
  if (uSunE > 0.0) L += uSunE * hapke(n, uSunDir, v) * texture(uLit, vUv).r;
  ${lights}
  outColor = vec4(L, 0.0, 0.0, 1.0);
}`;
}

const DIST_FRAG = /* glsl */ `
precision highp float;
in vec3 vPos;
out vec4 outColor;
uniform vec3 uLightPos;
void main() { outColor = vec4(length(vPos - uLightPos), 0.0, 0.0, 1.0); }`;

const DIST_VERT = /* glsl */ `
out vec3 vPos;
void main() { vPos = position; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }`;

/** World (R, t) + FLU -> Three camera (looks along -z, up +y): x_t = -y_b, y_t = z_b, z_t = -x_b. */
function setCameraPose(cam, R, t) {
  const c = [-R[1], R[2], -R[0], -R[4], R[5], -R[3], -R[7], R[8], -R[6]]; // R_world_three (row-major)
  cam.matrixAutoUpdate = false;
  cam.matrix.set(c[0], c[1], c[2], t[0], c[3], c[4], c[5], t[1], c[6], c[7], c[8], t[2], 0, 0, 0, 1);
  cam.matrixWorld.copy(cam.matrix);
  cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
}

/** OpenCV intrinsics (pixel-centre origin, v down) -> GL projection matrix. */
function projection(K, near, far) {
  const { width: W, height: H, fx, fy, cx, cy } = K;
  return new THREE.Matrix4().set(
    2 * fx / W, 0, 1 - 2 * (cx + 0.5) / W, 0,
    0, 2 * fy / H, 2 * (cy + 0.5) / H - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0);
}

export class GpuCameraRenderer {
  constructor() {
    this.canvas = new OffscreenCanvas(16, 16);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, powerPreference: 'high-performance' });
    const gl = this.renderer.getContext();
    if (!this.renderer.capabilities.isWebGL2 || !gl.getExtension('EXT_color_buffer_float')) {
      this.renderer.dispose();
      throw new Error('WebGL2 + EXT_color_buffer_float が必要です');
    }
    this.scene = new THREE.Scene();
    this.distScene = new THREE.Scene();
    this.camera = new THREE.Camera();
    this.targets = new Map();
    this.world = null;
  }

  dispose() { this.renderer.dispose(); }

  setWorld(world) {
    this.world = world;
    if (this.mesh) { this.scene.remove(this.mesh); this.distScene.remove(this.distMesh); this.mesh.geometry.dispose(); }
    const t = world.terrain, { n, res, x0, y0, z } = t, N = n * n;
    const pos = new Float32Array(3 * N), grad = new Float32Array(2 * N), uv = new Float32Array(2 * N);
    for (let i = 0; i < n; i++) {
      const im = Math.max(i - 1, 0), ip = Math.min(i + 1, n - 1);
      for (let j = 0; j < n; j++) {
        const k = i * n + j, jm = Math.max(j - 1, 0), jp = Math.min(j + 1, n - 1);
        pos[3 * k] = x0 + j * res; pos[3 * k + 1] = y0 + i * res; pos[3 * k + 2] = z[k];
        grad[2 * k] = (z[i * n + jp] - z[i * n + jm]) / ((jp - jm) * res);
        grad[2 * k + 1] = (z[ip * n + j] - z[im * n + j]) / ((ip - im) * res);
        uv[2 * k] = (j + 0.5) / n; uv[2 * k + 1] = (i + 0.5) / n;
      }
    }
    const idx = new Uint32Array(6 * (n - 1) * (n - 1));
    let m = 0;
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - 1; j++) { // diagonal (i, j)-(i+1, j+1), as Heightmap.height_at_mesh
        const a = i * n + j, b = a + 1, c = a + n + 1, d = a + n;
        idx[m++] = a; idx[m++] = b; idx[m++] = c; idx[m++] = a; idx[m++] = c; idx[m++] = d;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('grad', new THREE.BufferAttribute(grad, 2));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeBoundingSphere();
    if (this.litTex) this.litTex.dispose();
    this.litTex = new THREE.DataTexture(new Uint8Array(N), n, n, THREE.RedFormat, THREE.UnsignedByteType);
    this.litTex.magFilter = this.litTex.minFilter = THREE.LinearFilter;
    this.litTex.unpackAlignment = 1;
    this.litTex.flipY = false;
    this.litVersion = -1;
    const lights = world.scenario.lights || [];
    this.lights = lights.map((cfg) => ({ cfg, ...spotIntensity(cfg), cam: new THREE.Camera(),
      rt: new THREE.WebGLRenderTarget(SHADOW_SIZE, SHADOW_SIZE, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true }) }));
    const hu = hapkeUniforms(world.scenario.reflectance);
    const uniforms = {
      uCamPos: { value: new THREE.Vector3() }, uSunDir: { value: new THREE.Vector3() }, uSunE: { value: 0 }, uLit: { value: this.litTex },
      uHapkeA: { value: new THREE.Vector4(...hu.a) }, uHapkeB: { value: new THREE.Vector4(...hu.b) },
    };
    if (lights.length) {
      uniforms.uLightPos = { value: lights.map(() => new THREE.Vector3()) };
      uniforms.uLightAxis = { value: lights.map(() => new THREE.Vector3()) };
      uniforms.uLightCone = { value: lights.map(() => new THREE.Vector3()) };
      uniforms.uLightVP = { value: lights.map(() => new THREE.Matrix4()) };
      this.lights.forEach((l, k) => { uniforms[`uLightDist${k}`] = { value: l.rt.texture }; });
    }
    this.material = new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: fragmentShader(lights.length), uniforms, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
    this.distMaterial = new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: DIST_VERT, fragmentShader: DIST_FRAG,
      uniforms: { uLightPos: { value: new THREE.Vector3() } }, side: THREE.DoubleSide });
    this.distMesh = new THREE.Mesh(geo, this.distMaterial);
    this.distMesh.frustumCulled = false;
    this.distScene.add(this.distMesh);
  }

  target(K) {
    const key = `${K.width}x${K.height}`;
    if (!this.targets.has(key)) {
      this.targets.set(key, { rt: new THREE.WebGLRenderTarget(K.width, K.height, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true }),
        buf: new Float32Array(4 * K.width * K.height) });
    }
    return this.targets.get(key);
  }

  /** Radiance image (Float32Array, row-major, top row first) of camera `cam` at pose (R, t). */
  render(world, cam, pose) {
    if (world !== this.world) this.setWorld(world);
    const u = this.material.uniforms, r = this.renderer;
    if (this.litVersion !== world.sunVersion) {
      this.litTex.image.data.set(world.sunlitFull().lit.map((x) => x * 255));
      this.litTex.needsUpdate = true;
      this.litVersion = world.sunVersion;
    }
    const sun = world.sun;
    u.uSunDir.value.set(...sun.dir);
    u.uSunE.value = sun.el > 0 ? sun.irradiance * cam.cfg.sensor.bandFraction : 0;
    this.lights.forEach((l, k) => {
      const on = world.lightsOn && l.cfg.enabled, lp = compose(world.rover.pose, mountPose(l.cfg.mount));
      u.uLightPos.value[k].set(...lp.t);
      u.uLightAxis.value[k].set(lp.R[0], lp.R[3], lp.R[6]); // light +x in world
      u.uLightCone.value[k].set(l.inner, l.outer, on ? l.i0 : 0);
      if (!on) return;
      const f = 1 / Math.tan(l.outer + 5 * Math.PI / 180), near = 0.05, far = 300;
      setCameraPose(l.cam, lp.R, lp.t);
      l.cam.projectionMatrix.set(f, 0, 0, 0, 0, f, 0, 0, 0, 0, -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0);
      u.uLightVP.value[k].multiplyMatrices(l.cam.projectionMatrix, l.cam.matrixWorldInverse);
      this.distMaterial.uniforms.uLightPos.value.set(...lp.t);
      r.setRenderTarget(l.rt);
      l.cam.projectionMatrixInverse.copy(l.cam.projectionMatrix).invert();
      r.setClearColor(new THREE.Color(1e9, 0, 0), 1); r.clear(); // "infinitely far" where nothing is drawn
      r.render(this.distScene, l.cam);
    });
    const K = cam.K, tg = this.target(K);
    setCameraPose(this.camera, pose.R, pose.t);
    this.camera.projectionMatrix.copy(projection(K, 0.02, 3000));
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    u.uCamPos.value.set(...pose.t);
    r.setRenderTarget(tg.rt);
    r.setClearColor(0x000000, 1); r.clear();
    r.render(this.scene, this.camera);
    r.readRenderTargetPixels(tg.rt, 0, 0, K.width, K.height, tg.buf);
    r.setRenderTarget(null);
    const out = new Float32Array(K.width * K.height);
    for (let row = 0; row < K.height; row++) { // GL rows start at the bottom
      const src = (K.height - 1 - row) * K.width * 4, dst = row * K.width;
      for (let col = 0; col < K.width; col++) out[dst + col] = tg.buf[src + 4 * col];
    }
    return out;
  }
}
