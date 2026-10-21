// Seeded random streams (browser port of lunarsim/core/rng.py; NOT bit-identical with Python).
// Each subsystem uses its own named stream, e.g. rng(seed, 'terrain/craters/size'), so changing
// one part of the scenario does not reshuffle the others. Same seed + stream -> same numbers on
// every browser (integer arithmetic only; Math.log/sqrt/cos are IEEE-correctly rounded in V8,
// SpiderMonkey and JavaScriptCore for the ranges used here).

function hash128(str, seed) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return [h1 >>> 0, h2 >>> 0, (h1 ^ 0x9e3779b9) >>> 0, (h2 ^ 0x85ebca6b) >>> 0];
}

/** sfc32 generator with distribution helpers. */
export class Rng {
  constructor(seed, stream) {
    [this.a, this.b, this.c, this.d] = hash128(String(stream), seed >>> 0);
    for (let i = 0; i < 15; i++) this.next();
    this.spare = null;
  }

  next() { // uint32
    let { a, b, c, d } = this;
    const t = (((a + b) >>> 0) + d) >>> 0;
    d = (d + 1) >>> 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    this.a = a; this.b = b; this.c = c; this.d = d;
    return t;
  }

  random() { // [0, 1), 53 bits
    return (this.next() * 2097152 + (this.next() >>> 11)) / 9007199254740992;
  }

  uniform(lo, hi) { return lo + (hi - lo) * this.random(); }

  normal() { // standard normal (Box-Muller, pairs cached)
    if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
    let u = this.random();
    while (u <= 0) u = this.random();
    const v = this.random(), r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  poisson(lam) {
    if (!(lam > 0)) return 0;
    if (lam < 30) { // Knuth
      const L = Math.exp(-lam);
      let k = 0, p = 1;
      do { k++; p *= this.random(); } while (p > L);
      return k - 1;
    }
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * this.normal())); // normal approximation
  }

  gamma(k) { // Marsaglia-Tsang
    if (k < 1) return this.gamma(k + 1) * Math.pow(this.random() || 1e-300, 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = this.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = this.random();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  beta(a, b) { const x = this.gamma(a); return x / (x + this.gamma(b)); }
}

export const rng = (seed, stream) => new Rng(seed, stream);
