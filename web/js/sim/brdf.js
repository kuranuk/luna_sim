// Hapke bidirectional reflectance [1/sr] (port of lunarsim/core/illumination/brdf.py: Hapke 2002
// H function, Hapke 1984 macroscopic roughness, SHOE; CBOE off, K = 1 by default).
// hapke() is the CPU reference; HAPKE_GLSL is the same formula for the GPU camera renderer.

export const HAPKE_DEFAULT = { w: 0.38, b: 0.25, c: 0.5, bs0: 1.5, hs: 0.05, bc0: 0.0, hc: 0.05, thetaBarDeg: 23.657 };

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function hapkeH(x, w) {
  const gamma = Math.sqrt(1 - w), r0 = (1 - gamma) / (1 + gamma), xs = Math.max(x, 1e-12);
  return 1 / (1 - w * xs * (r0 + 0.5 * (1 - 2 * r0 * xs) * Math.log((1 + xs) / xs)));
}

function roughness(mu0, mu, psi, tb) {
  mu0 = clamp(mu0, 1e-9, 1); mu = clamp(mu, 1e-9, 1);
  if (tb <= 0) return [mu0, mu, 1];
  const tt = Math.tan(tb), chi = 1 / Math.sqrt(1 + Math.PI * tt * tt);
  const i = Math.acos(mu0), e = Math.acos(mu), si = Math.sin(i), se = Math.sin(e);
  const e1 = (x) => Math.exp(-2 / (Math.PI * tt * Math.max(Math.tan(x), 1e-12)));
  const e2 = (x) => Math.exp(-1 / (Math.PI * tt * tt * Math.max(Math.tan(x), 1e-12) ** 2));
  const eta = (x) => chi * (Math.cos(x) + Math.sin(x) * tt * e2(x) / (2 - e1(x)));
  const e1i = e1(i), e1e = e1(e), e2i = e2(i), e2e = e2(e);
  const s2 = Math.sin(psi / 2) ** 2, cpsi = Math.cos(psi), f = Math.exp(-2 * Math.tan(Math.min(psi, Math.PI - 1e-9) / 2));
  const etaI = eta(i), etaE = eta(e);
  if (i <= e) {
    const den = 2 - e1e - (psi / Math.PI) * e1i;
    const m0 = chi * (mu0 + si * tt * (cpsi * e2e + s2 * e2i) / den), m = chi * (mu + se * tt * (e2e - s2 * e2i) / den);
    return [m0, m, m / etaE * mu0 / etaI * chi / (1 - f + f * chi * mu0 / etaI)];
  }
  const den = 2 - e1i - (psi / Math.PI) * e1e;
  const m0 = chi * (mu0 + si * tt * (e2i - s2 * e2e) / den), m = chi * (mu + se * tt * (cpsi * e2i + s2 * e2e) / den);
  return [m0, m, m / etaE * mu0 / etaI * chi / (1 - f + f * chi * mu / etaE)];
}

/** Reflectance [1/sr] for unit normal n, to-light l and to-viewer v (arrays of 3). */
export function hapke(n, l, v, p = HAPKE_DEFAULT) {
  const mu0 = n[0] * l[0] + n[1] * l[1] + n[2] * l[2], mu = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
  if (!(mu0 > 0 && mu > 0)) return 0;
  const cg = clamp(l[0] * v[0] + l[1] * v[1] + l[2] * v[2], -1, 1);
  const den = Math.sqrt(Math.max(1 - mu0 * mu0, 0)) * Math.sqrt(Math.max(1 - mu * mu, 0));
  const psi = Math.acos(clamp(den > 1e-12 ? (cg - mu0 * mu) / den : 1, -1, 1));
  const g = Math.acos(cg), [mu0e, mue, S] = roughness(mu0, mu, psi, p.thetaBarDeg * Math.PI / 180);
  const b = p.b, c = p.c;
  const ph = 0.5 * (1 + c) * (1 - b * b) / (1 - 2 * b * cg + b * b) ** 1.5 + 0.5 * (1 - c) * (1 - b * b) / (1 + 2 * b * cg + b * b) ** 1.5;
  const tg2 = Math.tan(g / 2), bsh = 1 + p.bs0 / (1 + tg2 / p.hs), x = tg2 / p.hc;
  const bcb = 1 + p.bc0 * 0.5 * (1 + (x > 1e-12 ? (1 - Math.exp(-x)) / x : 1)) / (1 + x) ** 2;
  const M = hapkeH(mu0e, p.w) * hapkeH(mue, p.w) - 1;
  return p.w / (4 * Math.PI) * mu0e / (mu0e + mue) * (ph * bsh + M) * bcb * S;
}

/** GLSL (float precision) version of hapke(); uniforms are passed as a vec4 pair. */
export const HAPKE_GLSL = /* glsl */ `
uniform vec4 uHapkeA; // w, b, c, bs0
uniform vec4 uHapkeB; // hs, bc0, hc, theta_bar [rad]
const float PI_ = 3.141592653589793;
float hapkeH(float x, float w) {
  float gamma = sqrt(1.0 - w), r0 = (1.0 - gamma) / (1.0 + gamma), xs = max(x, 1e-6);
  return 1.0 / (1.0 - w * xs * (r0 + 0.5 * (1.0 - 2.0 * r0 * xs) * log((1.0 + xs) / xs)));
}
float e1f(float x, float tt) { return exp(-2.0 / (PI_ * tt * max(tan(x), 1e-6))); }
float e2f(float x, float tt) { float t = max(tan(x), 1e-6); return exp(-1.0 / (PI_ * tt * tt * t * t)); }
float etaf(float x, float tt, float chi) { return chi * (cos(x) + sin(x) * tt * e2f(x, tt) / (2.0 - e1f(x, tt))); }
float hapke(vec3 n, vec3 l, vec3 v) {
  float mu0 = dot(n, l), mu = dot(n, v);
  if (mu0 <= 0.0 || mu <= 0.0) return 0.0;
  float w = uHapkeA.x, b = uHapkeA.y, c = uHapkeA.z, bs0 = uHapkeA.w;
  float hs = uHapkeB.x, bc0 = uHapkeB.y, hc = uHapkeB.z, tb = uHapkeB.w;
  float cg = clamp(dot(l, v), -1.0, 1.0);
  float den = sqrt(max(1.0 - mu0 * mu0, 0.0)) * sqrt(max(1.0 - mu * mu, 0.0));
  float psi = acos(clamp(den > 1e-6 ? (cg - mu0 * mu) / den : 1.0, -1.0, 1.0));
  float g = acos(cg);
  float m0 = clamp(mu0, 1e-6, 1.0), m = clamp(mu, 1e-6, 1.0), mu0e = m0, mue = m, S = 1.0;
  if (tb > 0.0) {
    float tt = tan(tb), chi = 1.0 / sqrt(1.0 + PI_ * tt * tt);
    float i = acos(m0), e = acos(m), si = sin(i), se = sin(e);
    float e1i = e1f(i, tt), e1e = e1f(e, tt), e2i = e2f(i, tt), e2e = e2f(e, tt);
    float s2 = pow(sin(psi / 2.0), 2.0), cpsi = cos(psi), f = exp(-2.0 * tan(min(psi, PI_ - 1e-4) / 2.0));
    float etaI = etaf(i, tt, chi), etaE = etaf(e, tt, chi);
    if (i <= e) {
      float d = 2.0 - e1e - (psi / PI_) * e1i;
      mu0e = chi * (m0 + si * tt * (cpsi * e2e + s2 * e2i) / d);
      mue = chi * (m + se * tt * (e2e - s2 * e2i) / d);
      S = mue / etaE * m0 / etaI * chi / (1.0 - f + f * chi * m0 / etaI);
    } else {
      float d = 2.0 - e1i - (psi / PI_) * e1e;
      mu0e = chi * (m0 + si * tt * (e2i - s2 * e2e) / d);
      mue = chi * (m + se * tt * (cpsi * e2i + s2 * e2e) / d);
      S = mue / etaE * m0 / etaI * chi / (1.0 - f + f * chi * m / etaE);
    }
  }
  float ph = 0.5 * (1.0 + c) * (1.0 - b * b) / pow(1.0 - 2.0 * b * cg + b * b, 1.5)
           + 0.5 * (1.0 - c) * (1.0 - b * b) / pow(1.0 + 2.0 * b * cg + b * b, 1.5);
  float tg2 = tan(g / 2.0), bsh = 1.0 + bs0 / (1.0 + tg2 / hs), x = tg2 / hc;
  float bcb = 1.0 + bc0 * 0.5 * (1.0 + (x > 1e-6 ? (1.0 - exp(-x)) / x : 1.0)) / pow(1.0 + x, 2.0);
  float M = hapkeH(mu0e, w) * hapkeH(mue, w) - 1.0;
  return max(w / (4.0 * PI_) * mu0e / (mu0e + mue) * (ph * bsh + M) * bcb * S, 0.0);
}
`;

export function hapkeUniforms(p = HAPKE_DEFAULT) {
  return { a: [p.w, p.b, p.c, p.bs0], b: [p.hs, p.bc0, p.hc, p.thetaBarDeg * Math.PI / 180] };
}
