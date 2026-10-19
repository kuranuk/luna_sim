# CLAUDE.md — LunarSim 開発メモ

仕様: `docs/SPEC.md` / 計画: `docs/PLAN.md` / 文献と採用値: `docs/references.md`

## コマンド

```bash
# セットアップ（ネットあり）
python3 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.lock
pip install --no-deps -e .

# オフライン導入（閉域）: 先にネットのある機械で wheelhouse を作る
python scripts/build_wheelhouse.py --python-version 3.10 --platform manylinux2014_x86_64
pip install --no-index --find-links wheelhouse/<tag> -r requirements.lock
pip install --no-index --find-links wheelhouse/<tag> --no-deps -e .

# テスト / 静的解析
python -m pytest -q                         # 単体 + 統計テスト
LUNARSIM_PERF=1 python -m pytest -q -s tests/test_perf.py   # 性能計測
ruff check . && ruff format --check .
mypy lunarsim                               # core/config は strict

# 実行
lunarsim gen-world configs/mare_plain.yaml --seed 1 --out out/world
python scripts/demo_phase1.py               # out/phase1/ に画像
python scripts/demo_phase2.py               # out/phase2/ にクレーター・岩石・ラベル・GT
python scripts/demo_phase3.py               # out/phase3/ に南極・低太陽のカメラ画像
python scripts/demo_phase4.py               # out/phase4/ に LiDAR 点群（PLY/PCD）と画像
python scripts/demo_phase5.py               # out/phase5/ にローバー走行・ステレオ・ライト
python scripts/demo_phase6.py               # out/phase6/ にサーバ + クライアント + WS 走行
python scripts/demo_phase7.py               # out/phase7/ にデータセット + ビューアのスクショ
python scripts/demo_phase8.py               # out/phase8/ に半影・地球照・曲率・DEM・PSR・IMU・SPICE
python scripts/bench.py                     # out/bench/ に性能表
lunarsim psr configs/south_pole_lowsun.yaml --stride 4   # 長期日照 / PSR マップ
lunarsim serve                              # API サーバ（127.0.0.1:8000, /docs）
lunarsim serve --world configs/south_pole_lowsun.yaml   # ビューア: /viewer/#token=...
lunarsim dataset configs/dataset_south_pole_sun_sweep.yaml   # データセット生成

# 依存の更新（ネットあり）
scripts/update_lock.sh                      # requirements*.in -> *.lock
python scripts/gen_licenses.py              # LICENSES.md（runtime に GPL 系があれば exit 1）
```

## 座標系（すべて右手系、SI 単位）

```
World: ローカル ENU（原点 = 地形格子の中心）
        z(Up)
        |  y(North)
        | /
        |/____ x(East)

Body (rover): FLU                 Camera: OpenCV optical
        z(Up)                            z(Forward, 光軸)
        |  x(Forward)                   /
        | /                            /____ x(Right)
 y(Left)___|/                           |
                                        y(Down)

R_BODY_FROM_CAMERA: x_cam -> -y_body, y_cam -> -z_body, z_cam -> +x_body
```

- 変換は `lunarsim/core/frames.py` に集約。`R_a_b` / `T_a_b` は「b 系の座標を a 系へ」（`p_a = R_a_b @ p_b`）。
- 回転: 回転行列、四元数 `(w, x, y, z)`（w ≥ 0 に正規化）、オイラーは ZYX（`R = Rz(yaw) Ry(pitch) Rx(roll)`）。
- 方位角: **北から東回り（上から見て時計回り）**。仰角は局所水平面から上向き。
- 高さマップ `z[i, j]`: `i` = 南→北（行 0 が南端）、`j` = 西→東。`x_j = x0 + j*res`, `y_i = y0 + i*res`。
  PNG に書くときだけ北が上になるよう上下反転する（`io/raster.py`）。

## コーディング規約

- `lunarsim/core` は FastAPI・ネットワーク系ライブラリを import しない（`tests/test_architecture.py` で検査）。
  API 層（Phase 6）に物理ロジックを書かない。
- 乱数は `core/rng.py` の `RngFactory(seed).generator("stream/name")` だけを使う。
  `np.random` のグローバル関数禁止（ruff NPY002）。ストリーム名はサブシステム単位で固定し、
  フレーム依存のノイズは `.../frame/{index:06d}` のように名前にフレーム番号を含める。
- `pickle` 禁止（ruff TID251）。`np.load(..., allow_pickle=False)`、設定は `yaml.safe_load` + pydantic。
- 単位は SI（m, s, rad, W/m²）。度を使うフィールド・引数は名前を `_deg` で終える。
  docstring に物理量の単位を書く。
- 物理パラメータには出典をコメントで残す。原典で確認できない値は「サンプル値」と明記し
  `TODO(ref):` を付け、設定で変えられるようにする。出典一覧は `docs/references.md`。
- 型ヒント必須。`core` / `config` は mypy strict。行長 100。
- Python 側はランタイムでネットワークアクセスしない（テレメトリ・自動ダウンロード禁止）。
  例外: ビューアの JS は CDN 可（版固定 + SRI + 同梱版フォールバック、2026-10 ユーザー判断）。
- 依存追加は寛容ライセンス（MIT/BSD/Apache-2.0 等）のみ。Open3D は使わない（wheel 約 400 MB のため）。
  レイキャストは `core/raycast/` の抽象 + 自前の高さマップレイキャスタ（numba 任意）。
- 再現性: 同じ seed + 設定 + `requirements.lock` のバージョンでビット一致。スレッド数で結果を変えない。

## 地形生成の順序と乱数ストリーム

fractal (`terrain/fractal`) → tilt → craters (`terrain/craters/{count,size,position,cluster,degradation}`)
→ rocks (`terrain/rocks/{background,ejecta}`)。クレーターは劣化度 g が大きい（古い）順に刻印し、
新しいクレーターが内部を上書きする。岩石は高さマップに半楕円体として加算（Q2）。

2 層地形（`terrain.near.enabled`）: 遠方層 = `terrain/fractal/far` + 大クレーター（D ≥ 2×粗解像度,
`terrain/craters/large`）、近傍パッチ = 遠方層（クレーター前）の 3 次スプライン補間 + タイル単位の細部
（`terrain/detail/tile/{a}/{b}`）+ 小クレーター（`terrain/craters/small/tile/{a}/{b}`）+ 岩
（`terrain/rocks/background/tile/{a}/{b}`, `terrain/rocks/ejecta/{crater_uid}`）。タイル格子は世界原点に固定、
大クレーターの基準面は遠方層で 1 回だけ当てはめて再利用 → パッチ位置に依らず同じ地形。

## レンダリング

- 地表 = 高さマップの各セルを (i,j)-(i+1,j+1) 対角で 2 三角形に分割したメッシュ（`Heightmap.height_at_mesh` と一致）。
- レイキャスト: max-mipmap 階層走査（`core/raycast/heightfield.py`）。numba（並列）/ numpy の 2 実装は同一アルゴリズム。
  2 層は `LayeredCaster`（近傍パッチの xy 範囲内は近傍層、外は遠方層）。三角形は両面判定。
- 陰影法線 = 格子点の中心差分勾配の双一次補間（滑らか）。影は幾何（三角形）で判定。
- 放射輝度はカメラ帯域内の値 [W m⁻² sr⁻¹]: 太陽 = 放射照度 × `band_fraction` × r(i,e,g)、ライト = r × I(θ)/d²。
  曲率なし（既定）では太陽高度 ≤ 0 で直射光なし。`terrain.curvature` では各点の局所鉛直で判定。環境光ゼロ（地球照は任意）。
- レイキャスタの注意（過去のバグ）: 1 セル内の 2 三角形は各々有効区間で判定してから最小値をとる。
  ほぼ 0 の水平方向成分（< 1e-12）は 0 とみなす（格子線上の無限ループ防止）。
- 乱数: カメラノイズ `sensor/camera/{name}/frame/{index:06d}`。
- 遠方光源は `illumination/visibility.py` の `SkyLighting`（`World.sky`）: 半影（太陽円盤サンプリング、
  `penumbra_samples` = 0 なら従来の影レイ 1 本と同じ判定）、地球照（地球は点光源）、曲率（局所鉛直で地平線判定）。
  `shadow` バッファ = 円盤の可視割合 < 0.5。

## LiDAR

- 主対象は **SPAD フラッシュ LiDAR**（`noise.model: spad`, `pattern.kind: flash`、既定）。光子計数モデルは
  `core/sensors/spad.py`: ライダー方程式で信号光子数、日照地表（影レイ判定）+ DCR で背景率、
  複数イベント型（デッドタイム）/最初の光子だけ型、背景ポアソン分布からのしきい値（誤警報確率）、
  パイルアップによる距離の手前ずれ（補正しない）、誤検出点（`Label.NOISE`, GT は nan）。
- 回転式・ロゼット型 + 単純ガウスモデル（`noise.model: gaussian`）も残す（SPEC の差し替え要件）。
- 点はその計測時刻のセンサ座標（FLU）で出力（走行中の歪みは `SensorMotion` で再現）。
- 乱数: `sensor/lidar/{name}/frame/{index:06d}`。乱数の消費量はヒットの有無に依らず一定。

## 太陽・地形の任意機能（Phase 8）

- 太陽: `sun.ephemeris`（auto: カーネルがあれば SPICE、無ければ簡易モデル）。SPICE は `illumination/spice.py`
  （カーネルプールはプロセス共通なのでロック）。簡易モデルは SPICE と太陽 ≤ 0.07°、地球 ≤ 0.11°（docs/references.md）。
- 実 DEM / 曲率は `terrain/dem.py` の `BaseSurface`: 各層の格子点で解析的に評価して加算（スプライン補間しない）。
  2 層では遠方層の基盤（`_far_base`）に DEM・曲率を含めない（近傍パッチは補間 + 自前で評価）。
  DEM の窓は 16 px 余白をとって 3 次スプラインの前処理をパッチ位置に依らないようにする。全球の等緯度経度 DEM は経度方向に巡回。
- ホライズンマップ / PSR: `illumination/horizon.py`（numba / numpy 同一アルゴリズム、勾配で比較し atan は最後に 1 回）。

## ローバー

- `core/rover/`: 運動学（skid_steer: v, ω / ackermann: v, δ）、静的接地（車輪接地面 3×3 点の最高点に平面当てはめ）、
  斜度依存すべり・横流れ、警告（slope_exceeded / tipover_risk / tipped_over / stalled）。
- 姿勢: pitch > 0 = 機首下げ、roll > 0 = 左側が上。ローバー状態の (x, y) は接地点中心の地上軌跡、yaw は東から反時計回り。
- センサ・ライトの取り付けは `mount`（車体 FLU 座標の xyz + rpy_deg）。カメラは mount = カメラ本体（FLU）で、
  光学座標は `R_BODY_FROM_CAMERA` で変換。ステレオは右カメラ = 左カメラの画像 x 方向に基線長だけずらす。
- `World.step(dt)` で時間を進める。2 層地形では近傍パッチがローバーに追従して再生成される（パッチ寸法の 1/4 以上離れたら）。
- 乱数: ステレオ右カメラは `sensor/camera/{name}/right/frame/{index:06d}`。
- IMU（`core/sensors/imu.py`）: `World.step` はローバーのサブステップを IMU のサンプル時刻で区切り、その時刻の姿勢を記録
  （IMU が無いときのサブステップ列は従来と同じ）。真値は後退差分 + 低域通過（既定 5 Hz・2 次）、ノイズは
  `sensor/imu/{name}/block/{b:06d}`（1024 サンプル単位）なのでステップの分割に依らない。`reset_dynamic` でリセット。

## API（`lunarsim/api`, `lunarsim/client`）

- API 層は薄いラッパ: リクエスト検証 → `World` のメソッド呼び出し → シリアライズ（`lunarsim/io/bundle.py`）のみ。
  物理・幾何ロジックを書かない。重い処理は `run_in_threadpool` + World ごとの `threading.Lock`。
- 既定は 127.0.0.1。非ループバックへのバインドはトークン必須（`ServerSettings.check_exposure`）。
  `/health` と `/version` 以外は `Authorization: Bearer` を検査（WS は `?token=` も可）。
- 描画キャッシュのキーは `(センサ, 目, frame_index, World.version)`。状態を変える操作は `World.version` を増やすこと。
- クライアントは標準ライブラリ `urllib`（REST）+ `websockets`（WS）。httpx は dev（TestClient）専用
  （certifi = MPL-2.0 をランタイムに入れないため）。npz は `allow_pickle=False` で読む。
- 依存の版: fastapi 0.141.1（0.142 以降は opentelemetry-api が必須依存になるため）、starlette 1.7.0 /
  websockets 16.1.1（Python 3.10 対応の最終版）。lock にテレメトリ系が無いことをテストで検査。
- ROS 2 は `lunarsim/bridge/base.py` のインターフェースのみ（未実装）。

## ビューア / データセット（Phase 7）

- ビューア `lunarsim/viewer/static/`（素の ES モジュール、ビルド不要）。Three.js r160 は jsDelivr から読み込み
  （`index.html` で import map を生成、SRI 付き）。読み込み失敗（オフライン・SRI 不一致）時は `boot.js` が
  `vendor/`（CDN と同一バイト、テストで SRI と照合）に切り替えて再読込。`?local`、`serve --no-cdn`、
  `LUNARSIM_VIEWER_CDN=0` では最初から同梱版。CDN のライブラリを足すときは版固定 + `integrity` + 同梱版を揃える。
  3D は ENU（`camera.up = z`）。高さマップは `format=f32&max_size=N`、点群は `format=bin`（float32 xyzI）で取得。
  トークンは URL の `#token=`（サーバへ送られない fragment）→ sessionStorage。
- データセット `lunarsim/datasets/`: 1 つの World を `reset_dynamic()` して変種ごとに同じ軌跡を再生。
  地形と運動は太陽に依存しないので姿勢は全変種で一致（テストで検査）。KITTI 姿勢は左カメラの T_c0_ck。
  ジョブ YAML の `scenario` はジョブ YAML からの相対パス。

## 作業の進め方

Phase ごとに 実装 → テスト → `scripts/demo_phaseN.py` 実行 → commit/push。
判断に迷う仕様はユーザーに選択肢とトレードオフを示して確認する。
