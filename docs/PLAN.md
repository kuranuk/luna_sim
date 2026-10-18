# LunarSim 実装計画 (v2)

> `docs/SPEC.md` に基づく計画。v1 への回答（§9）を反映済み。

## 確定事項（v2）

- **Open3D は使わない**（wheel 約 400 MB + dash/flask 等の不要依存のため）。
- Q1: レイキャストは `core/raycast/` の抽象層を残し、標準は**自前の高さマップレイキャスタ**（max-mipmap DDA、numba があれば JIT、なければ numpy）。
  将来メッシュ系バックエンド（例: 軽量な Embree バインディング）を足せる口だけ残す。
- Q2: **岩石はすべて高さマップに加算**する（メッシュ化しない）。張り出し形状は表現しない。
- Q3: **2 層構成**（ローバー周辺の高解像度パッチ + 広域の低解像度格子）。回答「1」を最初の選択肢と解釈（Q2 と同じ読み方）。Phase 3 のレイキャスタで実装。`max_cells` による上限検証は各層に適用。
- Q4: FastAPI 等は extras（`[server]`）に分離。閉域向けにはフル lock を配布。
- Q5: カメラは**モノクロ**（単一帯域）。
- 依存は CPython 3.10〜3.13 共通の 1 本の lock にする（numpy 2.2 / scipy 1.15 / numba 0.61 系）。

## 0. 前提・調査結果

- 対象リポジトリ: `kuranuk/luna_sim`（空）。既存の `kuranuk/moonsim`（ブラウザ完結の JS 製、LOLA DEM 表示）には手を入れない。
- 開発環境: Python 3.13 / 4 コア / 15 GB RAM / GPU なし（CI 相当として扱う）。
- `open3d` 0.20.0 は cp313 wheel あり（Linux / Windows）。ただし
  - wheel 単体で **約 400 MB**
  - 必須依存に `dash`, `flask`, `werkzeug`, `nbformat`, `ipywidgets`, `configargparse` が入る（Jupyter/Web 可視化用。レイキャストには不要）
  - → 閉域環境の wheelhouse が肥大化する。§9-Q1 で方針を確認したい。

## 1. ディレクトリ構成

```
luna_sim/
  pyproject.toml            # PEP 621, setuptools。extras: [fast] numba / [spice] / [dem] / [server] / [dev]
  requirements.in / requirements.lock   # scripts/update_lock.sh で生成（3.10〜3.13 共通）
  CLAUDE.md  README.md  LICENSES.md  Dockerfile
  docs/  SPEC.md  PLAN.md  frames.md(座標系図)  references.md(文献一覧と採用値)
  lunarsim/
    __init__.py  version.py
    config/                 # pydantic v2 スキーマ（World/Terrain/Crater/Rock/Sun/Camera/Lidar/Rover/Server）
      schema.py  load.py    # YAML → 検証 → 正規化済み dict（ハッシュを World に保持）
    core/                   # FastAPI 非依存。numpy/scipy/pydantic のみで import 可能
      rng.py                # SeedSequence を「名前付きストリーム」で派生（§4）
      frames.py             # ENU / body(FLU) / camera(OpenCV) / sensor、SE3・四元数・オイラー
      units.py              # deg↔rad など（_deg 命名規約の補助）
      terrain/
        heightmap.py        # Heightmap データ型（格子、原点、解像度、z, 補間, 法線, 斜度）
        fractal.py          # スペクトル合成（k^-β, Hurst, RMS 斜度で正規化, 大域傾斜）
        craters.py          # SFD サンプリング、ポアソン配置、形状、劣化、年代順スタンプ
        rocks.py            # Golombek&Rapp 指数モデル、配置、高さマップ加算 / メッシュ生成
        dem_loader.py       # GeoTIFF 読み込み（tifffile 任意）+ 細部重畳
        labels.py           # セマンティックラベル
        mesh.py             # 高さマップ→三角形メッシュ、タイル分割、LOD
      raycast/              # レイキャストの抽象層（§3）
        base.py  heightfield.py   # 高さマップ max-mipmap DDA（numba 任意）
      illumination/
        sun.py              # 手動 / 簡易解析天文 / SPICE（任意）、地球方向
        shadows.py          # シャドウレイ、ホライズンマップキャッシュ、半影サンプリング
        brdf.py             # Hapke(IMSA+SHOE) / Lunar-Lambert(McEwen) / Lambert(テスト用)
        lights.py           # スポットライト（配光、逆二乗、影）
      sensors/
        camera.py           # ピンホール+Brown-Conrady、レンダリング、ステレオ
        camera_noise.py     # 露出、ゲイン、ショット/読み出し/暗電流、飽和、量子化、AE
        lidar.py  lidar_patterns.py  lidar_noise.py
        imu.py              # Phase 8
      rover/
        kinematics.py       # skid-steer / Ackermann
        contact.py          # 車輪接地点から姿勢・車体高さ、転倒判定
        slip.py             # 斜度依存すべり（パラメータ化）
      world.py              # World = config + seed から完全再構築、状態の唯一の置き場
      sim.py                # 固定 dt の決定論ループ、tick カウンタ
    io/                     # png16, npy, ply, pcd, kitti/tum 軌跡, calib ファイル, 簡易 EXR は任意
    api/                    # FastAPI 薄ラッパ（schemas.py, routes_*.py, ws.py, auth.py, registry.py）
    client/                 # lunarsim.client（httpx + websockets）
    viewer/static/          # index.html, js/*.js, vendor/three.module.js 等（同梱）
    cli.py                  # `lunarsim serve|gen-world|render|dataset`
  configs/                  # flat_test.yaml, south_pole_lowsun.yaml, highland_rocky.yaml, ...
  scripts/                  # demo_phaseN.py, generate_dataset.py, build_wheelhouse.sh/.ps1, gen_licenses.py, bench.py
  tests/                    # unit/, statistical/, golden/(小さなハッシュ), perf/(マーカー付き, 既定 skip)
```

モジュール境界のルール:
- `core` → `config` のみに依存。`api`/`client`/`viewer` は `core` を呼ぶだけ（物理ロジック禁止）。
- `raycast` 抽象の上に `illumination` と `sensors` が乗る。バックエンドは差し替え可能。
- import-linter 相当のテスト（`core` が fastapi/uvicorn を import しないこと）を入れる。

## 2. 依存ライブラリ（案）

| 区分 | パッケージ | 理由 | ライセンス |
|---|---|---|---|
| 必須 | numpy, scipy | 配列演算、FFT、統計、補間 | BSD |
| 必須 | pydantic>=2, pyyaml | 設定検証 | MIT |
| 必須 | pillow | PNG(8/16bit) 入出力 | MIT-CMU (HPND) |
| 任意 [fast] | numba | 高さマップレイマーチ・スタンプ・ノイズの高速化。無ければ numpy 実装 | BSD |
| 任意 [server] | fastapi, uvicorn, httpx, websockets | API / クライアント | MIT / BSD |
| 任意 [spice] | spiceypy | 高精度太陽・地球方向（カーネル手動配置） | MIT |
| 任意 [dem] | tifffile | GeoTIFF DEM | BSD |
| 開発 | pytest, hypothesis, ruff, mypy, pip-licenses | テスト・静的解析・ライセンス表 | MIT 等 |

- `fastapi` は SPEC では必須だが、`core` を軽く保つため **extras `[server]`** に分ける案（`pip install lunarsim[server]`）。閉域環境ではフル lock をそのまま使えば同じ。
- 依存は SPEC 外のものを足さない（`httpx` / `websockets` はクライアント・テスト用で追加。いずれも BSD）。
- OpenEXR は LGPL 系ではないが重いので、深度は `.npy` + 16bit PNG（スケール付き）を標準、EXR は出さない（必要なら後で）。

## 3. レイキャスト設計

`RayCaster` 抽象: `cast(origins[N,3], dirs[N,3], t_max) -> hit_t, normal, prim_id, material/label` と `occluded(origins, dirs, t_max) -> bool`。

- ~~Open3D バックエンド~~（v2 で不採用）
- **高さマップ・バックエンド（標準）**（numba / numpy）: max-mipmap（四分木）上の DDA レイマーチ。メモリは高さマップ + mip のみ。2.5D 地形に対して厳密。岩石メッシュは扱えない（岩を高さマップ加算した場合のみ完全）。
- **長距離の太陽影**: 静的太陽方向ではホライズンマップ（方位ビンごとの地平線仰角）をキャッシュして参照、近距離は実レイで精密化。
- テストは両バックエンドで同じ解析解（平面上の箱の影長など）を満たすことを確認。

## 4. 再現性設計

- `core/rng.py`: `root = SeedSequence(seed)`。サブシステムごとに **名前（文字列）→ 安定ハッシュ → spawn key** で独立ストリームを作る（例 `"terrain/fractal"`, `"terrain/craters"`, `"sensor/lidar/front/frame/000123"`）。
  - クレーター設定を変えても岩石配置は変わらない、センサの呼び出し順に依存しない、を保証。
- センサノイズは `(seed, sensor_name, frame_index)` から都度生成 → 同じフレームを何度要求しても同じ結果。
- グローバル `np.random` 禁止を ruff ルール（banned-api）とテストで担保。numba は `fastmath=False`、並列リダクションの順序依存を避ける。
- **ビット一致の保証範囲**: 同一 OS・同一 CPU アーキテクチャ・同一 lock の依存バージョン。異 OS 間は許容誤差付き比較（FFT/BLAS/Embree の実装差のため）。golden テストは SHA-256 ハッシュ + 異環境向けの数値許容比較の 2 段構え。
- World は `config(正規化済み) + seed + コマンド履歴` から完全再構築できる。`World.snapshot()` は YAML + npz（pickle 不使用）。

## 5. 座標系・単位

- World: ローカル ENU（x 東, y 北, z 上）、原点は地形格子の中心（実 DEM モードでは指定緯度経度の局所接平面）。高さマップ `z[i, j]` は **i = y 方向（南→北）, j = x 方向（西→東）** に統一し、画像出力時のみ上下反転。
- Body: FLU（x 前, y 左, z 上）。Camera: OpenCV（x 右, y 下, z 前）。LiDAR: FLU。
- 回転表現: 内部は回転行列 + 四元数（w, x, y, z）。オイラーは ZYX（yaw-pitch-roll）を入出力の便宜のみに使用。
- 太陽方位角: **北から東回り（時計回り）**、仰角は局所水平面から上向き。API は `azimuth_deg`, `elevation_deg`。
- SI 単位。放射照度 W/m²、放射輝度 W/(m²·sr)。カメラは帯域積分の単色（または RGB 3 帯域、§9-Q5）。

## 6. 物理モデルの方針（文献は実装時に原典で確認し `docs/references.md` に採用値・頁を記録）

| 項目 | 方針 | 予定文献（要確認） |
|---|---|---|
| フラクタル地形 | 等方 PSD ∝ k^-β, β = 2H+2。RMS 斜度を基準長で正規化 | Rosenburg et al. 2011 (LOLA roughness), Kreslavsky et al. 2013 |
| クレーター SFD | N(>D)=c·D^-b、逆変換法、[D_min, D_max] で切断。プリセット: 平衡(b≈2) / 生成(より急) | Trask 1966; Gault 1970; Neukum et al. 2001 |
| 形状 | 放物面ボウル + 縁 + 噴出物 ∝ (r/R)^-3、d/D と縁高は劣化度で補間（新鮮 d/D の既定は 0.15、§references） | Pike 1977; McGetchin et al. 1973; Fassett & Thomson 2014（拡散劣化） |
| 劣化 | 劣化度 κt を確率的に割当、ガウシアン拡散で表現（クレーター局所に適用） | Fassett & Thomson 2014 |
| 岩石 | 累積面積率 F_k(D)=k·exp(-q(k)·D), q(k)=1.79+0.152/k | Golombek & Rapp 1997（係数は原典確認） |
| 太陽（簡易） | 月固定系での太陽経度を朔望月で回し、黄道に対する自転軸傾斜 1.54° による季節変化 | Meeus "Astronomical Algorithms"（簡易式） |
| BRDF | Hapke（IMSA + SHOE、粗さ補正）既定、Lunar-Lambert 選択可 | Hapke 2012; Sato et al. 2014 (LROC WAC); McEwen 1991 |
| 太陽定数 | 1361 W/m²（1 AU、距離補正あり） | Kopp & Lean 2011 |

不明・未確定な数値は設定可能にし、`# TODO(ref): 原典で要確認` を残す。数値を断定的に書かない。

## 7. Phase 分割と各 Phase の完了条件

| Phase | 内容 | テスト / デモ |
|---|---|---|
| 1 | pyproject, lock, CLAUDE.md, 設定スキーマ, rng, frames, Heightmap, フラクタル地形, 法線/斜度, PNG16/npy 出力, ruff/mypy/pytest 基盤, wheelhouse/ライセンス スクリプト | frames 往復誤差、PSD 傾き回帰で β 再現、RMS 斜度、ビット一致。`demo_phase1.py` → `out/` に陰影図・斜度図 |
| 2 | クレーター（SFD, 配置, 形状, 劣化, 年代順）, 岩石, ラベル, GT 出力（JSON/CSV） | SFD の KS 検定、密度の期待値、形状断面の解析比較、再現性 |
| 3 | 太陽（手動・簡易天文）, raycast 抽象 + バックエンド, シャドウ, ホライズンマップ, BRDF, ピンホール+歪み, センサノイズ | 箱の影長、太陽方向・周期性、投影/逆投影往復、歪み往復、ノイズ統計。南極低太陽シナリオの画像 |
| 4 | LiDAR（回転/ロゼット/フラッシュ）, ノイズ, 欠測, 強度, ring/time, PLY/PCD | 平面ターゲットで測距誤差統計、検出率、性能計測 |
| 5 | ローバー運動学・接地・すべり・転倒警告, extrinsics, スポットライト, ステレオ | 接地姿勢の解析比較、運動学積分、ライト照度の逆二乗則 |
| 6 | FastAPI REST + WS, トークン認証, スレッドプール, Python クライアント | TestClient による API テスト、決定論的 step |
| 7 | ビューア（Three.js 同梱）, データセット生成（KITTI/TUM/calib） | ヘッドレスで静的ファイル配信確認、データセット形式検証 |
| 8 | SPICE, 実 DEM, IMU, 半影, 地球照, PSR, 性能チューニング, README に性能表 | bench.py の結果 |

各 Phase: 実装 → テスト → デモスクリプト実行（出力画像を提示）→ commit/push。

## 8. 性能見積りとリスク

- 1 km² / 0.25 m = 4000² 格子: FFT 合成は数秒。クレーター数は D_min 次第で数万〜数十万 → numba で局所スタンプ（無い場合 numpy でも bbox 局所演算）。目標「数十秒」は達成見込み。
- 640×480 影あり: 一次レイ 31 万 + シャドウレイ 31 万。高さマップ DDA（numba 4 コア）も 0.3〜1 秒程度の見込み（要計測）。numba 無しの純 numpy は目標未達の可能性大 → 報告対象。
- 32 ライン LiDAR（1800 列 × 32 ≈ 5.8 万レイ）0.2 秒: 見込みあり。
- リスク: 0.05 m 解像度 × 2 km 四方（16 億セル）は単一格子では不可能。**近傍高解像度パッチ + 広域低解像度**の 2 層構成が必要（§9-Q3）。

## 9. 確認したいこと（質問）

v1 の質問と回答は冒頭「確定事項（v2）」を参照。
