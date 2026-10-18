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

## Phase 2: クレーター・岩石

この環境では arxiv.org 等の原典 PDF に接続できなかったため、以下の多くは**二次文献（引用している論文・要旨）経由**で確認した値。
「二次確認」は原典の頁・式番号まで照合できていないことを示す。

| 量 | 採用（既定値） | 状態 | 出典 |
|---|---|---|---|
| 幾何学的飽和密度 | N(>D) = 1.54 D⁻² km⁻²（D: km）。b = 2 なので D[m]・N[m⁻²] でも係数 1.54 | 二次確認 | Gault (1970) Radio Sci. 5, 273（arXiv:2604.16223 等の引用） |
| 平衡密度 | 幾何学的飽和の 1〜10 %（Gault 1970）、0.69〜3.9 %（Xiao & Werner 2015）→ **既定 2 %** | 二次確認 | Xiao & Werner (2015) JGR Planets 120, 2277; Minton et al. (2019) Icarus（arXiv:1902.07746）での引用 |
| 平衡分布の傾き | b = 2 | 二次確認 | 同上（Hartmann 1984 は b ≈ 1.83） |
| 生成関数の傾き（小クレーター） | 累積 b ≈ 3（power_law モードの既定） | 二次確認 | Neukum (1983) / Neukum et al. (2001); Oetting et al. (2023) は ≤10 m で ≈3、10〜20 m でより急 |
| 新鮮な d/D | **0.15**（Stopar et al. 2017: D < 400 m で 0.11〜0.17; Mahanti et al. 2018: 最新鮮クラス平均 0.15）。SPEC の 0.2 は大きめの単純クレーター（Pike 1977） | 二次確認 | Stopar et al. (2017) Icarus; Mahanti et al. (2018); Pike (1977) Proc. LSC 8 |
| 劣化した d/D | **0.05**（Copernicus 二次クレーターで 0.02〜0.2、多くは ~0.05） | 二次確認 | Basilevsky et al. (2018) |
| 縁の高さ | h = 0.036 D（新鮮）。劣化時は h/d を保って縮小 | 二次確認 | Pike (1977) |
| 噴出物厚さ | ∝ (r/R)^-3（縁高さで連続、4R で 0 にテーパ） | 二次確認 | McGetchin et al. (1973) EPSL 20, 226; t = 0.14 R^0.74 (r/R)^-3（R, t: m） |
| 劣化による丸み | σ = 0.3 g R のガウシアン平滑化（深さ d は保持） | **サンプル値** | 定性的な代替。拡散係数 κ ≈ 5.5 m²/Myr（Fassett & Thomson 2014, 二次確認）を使う年代モデルは未実装 |
| 劣化度 g の分布 | Beta(1, 1)（一様） | **サンプル値** | — |
| 岩石サイズ分布 | F_k(D) = k exp(−q D), q(k) = 1.79 + 0.152/k | 二次確認 | Golombek & Rapp (1997) JGR 102, 4117（Golombek et al. 2003 要旨での記述）。月への同型モデル適用: Li & Wu (2018) JGR Planets 123, doi:10.1029/2017JE005496 |
| 岩石被覆率 k | 0.02 | **サンプル値** | TODO(ref): 月面の地点別 k（Li & Wu 2018 等） |
| 岩石の高さ/直径 | 0.5 | **サンプル値** | TODO(ref) |
| クレーター周辺の岩塊 | k = 0.15 ×(1 − g)、D ≥ 30 m、0.8〜2.5 R、最大径 0.29 D^0.66 | **サンプル値 / 要確認** | TODO(ref): Bart & Melosh (2010) Icarus 209, 337 |

注: q(k) は k が小さいと大きくなる（k = 0.02 で q ≈ 9.4 m⁻¹）ため、k が低い地域では 0.5 m を超える岩は非常にまれになる。これはモデル本来の挙動。

## 以降の Phase（予定）

- 反射: Hapke (2012); Sato et al. (2014); McEwen (1991)
- 太陽: Meeus, *Astronomical Algorithms*; 太陽定数 Kopp & Lean (2011)
