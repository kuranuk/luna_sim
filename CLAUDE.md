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
- ランタイムでネットワークアクセスしない（CDN・テレメトリ・自動ダウンロード禁止）。
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
  太陽高度 ≤ 0 では直射光なし（曲率未考慮の接平面世界のため）。環境光ゼロ。
- レイキャスタの注意（過去のバグ）: 1 セル内の 2 三角形は各々有効区間で判定してから最小値をとる。
  ほぼ 0 の水平方向成分（< 1e-12）は 0 とみなす（格子線上の無限ループ防止）。
- 乱数: カメラノイズ `sensor/camera/{name}/frame/{index:06d}`。

## LiDAR

- 主対象は **SPAD フラッシュ LiDAR**（`noise.model: spad`, `pattern.kind: flash`、既定）。光子計数モデルは
  `core/sensors/spad.py`: ライダー方程式で信号光子数、日照地表（影レイ判定）+ DCR で背景率、
  複数イベント型（デッドタイム）/最初の光子だけ型、背景ポアソン分布からのしきい値（誤警報確率）、
  パイルアップによる距離の手前ずれ（補正しない）、誤検出点（`Label.NOISE`, GT は nan）。
- 回転式・ロゼット型 + 単純ガウスモデル（`noise.model: gaussian`）も残す（SPEC の差し替え要件）。
- 点はその計測時刻のセンサ座標（FLU）で出力（走行中の歪みは `SensorMotion` で再現）。
- 乱数: `sensor/lidar/{name}/frame/{index:06d}`。乱数の消費量はヒットの有無に依らず一定。

## ローバー

- `core/rover/`: 運動学（skid_steer: v, ω / ackermann: v, δ）、静的接地（車輪接地面 3×3 点の最高点に平面当てはめ）、
  斜度依存すべり・横流れ、警告（slope_exceeded / tipover_risk / tipped_over / stalled）。
- 姿勢: pitch > 0 = 機首下げ、roll > 0 = 左側が上。ローバー状態の (x, y) は接地点中心の地上軌跡、yaw は東から反時計回り。
- センサ・ライトの取り付けは `mount`（車体 FLU 座標の xyz + rpy_deg）。カメラは mount = カメラ本体（FLU）で、
  光学座標は `R_BODY_FROM_CAMERA` で変換。ステレオは右カメラ = 左カメラの画像 x 方向に基線長だけずらす。
- `World.step(dt)` で時間を進める。2 層地形では近傍パッチがローバーに追従して再生成される（パッチ寸法の 1/4 以上離れたら）。
- 乱数: ステレオ右カメラは `sensor/camera/{name}/right/frame/{index:06d}`。

## 作業の進め方

Phase ごとに 実装 → テスト → `scripts/demo_phaseN.py` 実行 → commit/push。
判断に迷う仕様はユーザーに選択肢とトレードオフを示して確認する。
