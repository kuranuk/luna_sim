// In-place radix-2 complex FFT on Float64Array re/im (2-D, row-major n x n).

function fft1(re, im, off, stride, n, inverse, tre, tim, cosT, sinT) {
  for (let k = 0; k < n; k++) { tre[k] = re[off + k * stride]; tim[k] = im[off + k * stride]; }
  for (let i = 1, j = 0; i < n; i++) { // bit reversal
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = tre[i]; tre[i] = tre[j]; tre[j] = t;
      t = tim[i]; tim[i] = tim[j]; tim[j] = t;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = cosT[k * step], wi = sgn * sinT[k * step];
        const a = i + k, b = a + half;
        const xr = tre[b] * wr - tim[b] * wi, xi = tre[b] * wi + tim[b] * wr;
        tre[b] = tre[a] - xr; tim[b] = tim[a] - xi;
        tre[a] += xr; tim[a] += xi;
      }
    }
  }
  const s = inverse ? 1 / n : 1;
  for (let k = 0; k < n; k++) { re[off + k * stride] = tre[k] * s; im[off + k * stride] = tim[k] * s; }
}

/** 2-D FFT of an n x n array (n a power of two); inverse includes the 1/n² scaling. */
export function fft2(re, im, n, inverse = false) {
  if (n & (n - 1)) throw new Error('fft2: n must be a power of two');
  const tre = new Float64Array(n), tim = new Float64Array(n);
  const cosT = new Float64Array(n / 2), sinT = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) { cosT[k] = Math.cos(2 * Math.PI * k / n); sinT[k] = Math.sin(2 * Math.PI * k / n); }
  for (let r = 0; r < n; r++) fft1(re, im, r * n, 1, n, inverse, tre, tim, cosT, sinT);
  for (let c = 0; c < n; c++) fft1(re, im, c, n, n, inverse, tre, tim, cosT, sinT);
}
