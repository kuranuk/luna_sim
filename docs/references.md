# 文献と採用値

方針: 採用した数値は原典の表・式番号とともにここに記録する。原典で確認できていないものは
「未確認」とし、コード側は設定可能 + `TODO(ref)` にする。

## Phase 1: 基盤地形

| 量 | 採用 | 状態 | 出典 |
|---|---|---|---|
| 自己アフィン面の PSD | `P(k) ∝ k^-(2H+2)`（2 次元等方） | 標準的な関係 | Turcotte (1997) *Fractals and Chaos in Geology and Geophysics*, 2nd ed. |
| RMS 斜度の定義 | `s(L) = RMS(Δz over L) / L` | 定義として採用 | Rosenburg et al. (2011) JGR 116, E02001 |
| Hurst 指数の既定値 0.9 | サンプル値 | **未確認** | TODO(ref): Rosenburg et al. (2011) の地域別 H を確認 |
| RMS 斜度の既定値 5°（基線 1 m） | サンプル値 | **未確認** | TODO(ref): Rosenburg et al. (2011), Kreslavsky et al. (2013) Icarus 226 |

## 以降の Phase（予定）

- クレーター SFD: Trask (1966); Gault (1970); Neukum et al. (2001)
- クレーター形状: Pike (1977); McGetchin et al. (1973); Fassett & Thomson (2014)
- 岩石: Golombek & Rapp (1997)
- 反射: Hapke (2012); Sato et al. (2014); McEwen (1991)
- 太陽: Meeus, *Astronomical Algorithms*; 太陽定数 Kopp & Lean (2011)
