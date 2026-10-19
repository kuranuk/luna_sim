# LunarSim

月面ローバーの航法・誘導・自律走行（SLAM、LiDAR/カメラ認識、経路計画）の開発・評価向けの、
Python 製の軽量な月面ローバーシミュレータです。

- **オフライン・管理者権限なし・GPU なし**で動作（`pip` のみ、ランタイムのネットワークアクセスなし）
- **物理的にもっともらしい**パラメータ（出典は `docs/references.md`）
- **再現性**: seed + YAML 設定から World を完全に再構築（同一 lock でビット一致）

開発状況: **Phase 6**（地形 2 層構成、太陽位置・影・反射、モノクロカメラ / ステレオ、SPAD フラッシュ LiDAR ほか、ローバー運動学・接地・すべり、ライト、FastAPI サーバ + Python クライアント）。
全体計画は [`docs/PLAN.md`](docs/PLAN.md)、仕様は [`docs/SPEC.md`](docs/SPEC.md)。

## インストール

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .
```

閉域環境へは、ネットのある機械で `python scripts/build_wheelhouse.py --python-version 3.10 --platform manylinux2014_x86_64`
（Windows 向けは `--platform win_amd64`）を実行して wheel を集め、`pip install --no-index --find-links ...` で導入します。
`requirements.lock` は CPython 3.10〜3.13（Linux x86_64 / Windows amd64）共通です。

## 使い方

```bash
lunarsim gen-world configs/south_pole_slope.yaml --seed 7 --out out/world
python scripts/demo_phase1.py   # フラクタル地形
python scripts/demo_phase2.py   # クレーター・岩石・ラベル・GT
python scripts/demo_phase3.py   # 南極域・低太陽高度のカメラ画像
python scripts/demo_phase4.py   # SPAD フラッシュ LiDAR / 回転式 LiDAR の点群（PLY/PCD）
python scripts/demo_phase5.py   # ローバー走行（接地・すべり）、ステレオ、ライト
python scripts/demo_phase6.py   # サーバ起動 + Python クライアント + WebSocket 走行
python scripts/demo_phase7.py   # データセット生成（太陽高度 3 条件）+ ビューアのスクリーンショット
python scripts/demo_phase8.py   # 半影・地球照・曲率・実 DEM・PSR マップ・IMU・SPICE 比較
python scripts/bench.py         # 性能計測 → out/bench/bench.{json,md}
```

## 任意機能（Phase 8）

| 機能 | 設定 / 使い方 | 備考 |
|---|---|---|
| 半影（ソフトシャドウ） | `render.penumbra_samples: 16` | 太陽円盤（視半径 0.267°、周辺減光）を決定論的にサンプリング。縁 4 点で判定し、食い違う画素だけ全サンプル。`RenderBuffers.sun_fraction` に可視割合 |
| 地球照 | `sun.earthshine: true`（手動モードは `earth_azimuth_deg` / `earth_elevation_deg`） | 満地球で約 0.14 W/m²。地球は点光源として影を判定 |
| 月面曲率 | `terrain.curvature: true` | 高さに −(x²+y²)/(2R) を加え、照明は各点の局所鉛直で判定（太陽高度が負でも遠方が照らされる） |
| 実 DEM | `terrain.dem: {enabled, path, projection: polar_stereographic\|equirectangular\|local, ...}` | GeoTIFF（`tifffile`、[dem] extra）または `.npy`。DEM が長波長、フラクタル（DEM のナイキスト以下を除去）+ クレーター + 岩が細部。原点 = `site` の緯度経度 |
| SPICE | `sun.ephemeris: auto\|spice`, `sun.spice_kernels: [naif0012.tls, de440s.bsp, pck00011.tpc]` | `spiceypy`（[spice] extra）。カーネルは手動配置（自動ダウンロードしない）。無ければ簡易モデル（SPICE との差: 太陽 ≤ 0.07°、地球 ≤ 0.11°） |
| PSR / 長期日照マップ | `lunarsim psr configs/south_pole_lowsun.yaml --days 346.62 --step-hours 2 --stride 4` | ホライズンマップ + 太陽円盤の弓形面積。`mean_sun_fraction`, `psr`, `earth_visible_fraction` を npy/PNG で出力 |
| IMU | `imus: [{name: imu0, rate_hz: 200}]` → `World.imu_samples()`, `GET /worlds/{id}/sensors/imu/{name}?since_s=`, WS `subscribe: {imus: [...]}`、データセット `imus: [...]`（EuRoC 形式 CSV） | 6 軸。重力 1.6243 m/s²、月の自転、ノイズ・バイアス・ランダムウォーク。ローバーのサブステップを IMU 時刻に合わせる |

## ブラウザビューア

```bash
LUNARSIM_TOKEN=xxxx lunarsim serve --world configs/south_pole_lowsun.yaml
# → http://127.0.0.1:8000/viewer/#token=xxxx を開く（トークンなしで起動した場合は #token 不要）
```

Three.js（r160, MIT）は jsDelivr CDN から読み込みます（版固定 + SRI ハッシュ）。CDN に届かない場合は
同梱の `lunarsim/viewer/static/vendor/`（同一ファイル）に自動で切り替わります。CDN を使わない場合は
`lunarsim serve --no-cdn`（または `LUNARSIM_VIEWER_CDN=0`、URL に `?local`）。
地形（近傍・遠方層）、太陽方向、ローバー、LiDAR 点群、カメラ画像（物理ベース描画）を表示し、
W/A/S/D で走行、スライダで太陽方位・高度、チェックでライトを切り替えます。
ビューア内の 3D 陰影は OpenGL の近似で、右上のカメラ画像がシミュレータの描画結果です。

## データセット生成

```bash
lunarsim dataset configs/dataset_south_pole_sun_sweep.yaml [--out out/datasets/xxx]
```

同じ地形・同じ軌跡を太陽条件（方位・高度・UTC・ライト）ごとに生成します（姿勢は全条件でビット一致）。

```
<out>/dataset.yaml, scenario.yaml, summary.json
<out>/terrain/            near/far_heightmap.npy, craters/rocks .json, geometry.json
<out>/<variation>/
    times.txt
    cameras/<cam>/left|right/NNNNNN.png   16-bit DN
    cameras/<cam>/depth/NNNNNN.npy        float32 Z [m]（inf = 空）
    cameras/<cam>/labels|shadow/NNNNNN.png uint8
    lidars/<lidar>/NNNNNN.bin             KITTI velodyne 形式 float32 x y z intensity（センサ座標）
    poses/<cam>_kitti.txt                 左カメラ、KITTI odometry 形式（T_c0_ck, 3x4）
    poses/body_tum.txt, <sensor>_tum.txt  TUM 形式（世界 ENU, t tx ty tz qx qy qz qw）
    calib/<cam>_kitti.txt                 P0/P1（P1[0,3] = -fx·基線長）, Tr = LiDAR→左カメラ
    calib/calib.yaml                      K, 歪み, T_body_camera, T_body_lidar
    rover_state.csv, metadata.json
```

## サーバ（REST / WebSocket）

```bash
lunarsim serve                                  # 127.0.0.1:8000（OpenAPI: http://127.0.0.1:8000/docs）
LUNARSIM_TOKEN=xxxx lunarsim serve --host 0.0.0.0   # 外部公開はトークン必須（無ければ起動を拒否）
```

```python
from lunarsim.client import LunarSimClient

c = LunarSimClient("http://127.0.0.1:8000", token=None)
wid = c.create_world("configs/south_pole_lowsun.yaml")["world_id"]
c.waypoints(wid, [(5, 0), (5, 5)], speed_m_s=0.2)
c.step(wid, 10.0)
frame = c.camera(wid, "navcam", fmt="npz", buffers=["depth", "labels"])  # dict of arrays
scan = c.lidar(wid, "spad_flash")  # points (N,4), ring, time, ...
with c.stream(wid) as s:  # WebSocket
    s.send(type="subscribe", lidars=["spad_flash"])
    s.receive()
    s.step(0.5)
    print(s.receive()["data"]["points"].shape)
```

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/health`, `/version` | 死活・バージョン（認証不要） |
| POST | `/worlds` | `{"config": {...}}` または `{"config_yaml": "..."}`, `seed` → `world_id` |
| GET / DELETE | `/worlds/{id}` | メタデータ + 状態 / 削除 |
| GET | `/worlds/{id}/heightmap?format=npy\|png16\|f32&layer=near\|far&max_size=N` | 高さマップ（幾何はヘッダ） |
| GET | `/worlds/{id}/craters`, `/rocks` | ground truth |
| PUT | `/worlds/{id}/sun` | `{"azimuth_deg","elevation_deg"}` または `{"utc"}` |
| PUT | `/worlds/{id}/lights/{name}` | `{"on": bool}` |
| POST | `/worlds/{id}/rover/cmd`, `/rover/waypoints` | 速度指令 / ウェイポイント |
| POST | `/worlds/{id}/step` | `{"dt_s"}`（決定論的）→ 状態 |
| GET | `/worlds/{id}/sensors/camera/{name}?format=png\|png16\|npz&buffers=depth,labels&eye=left\|right` | 画像 + GT |
| GET | `/worlds/{id}/sensors/lidar/{name}?format=npz\|ply\|pcd\|json\|bin` | 点群（JSON は小規模のみ） |
| GET | `/worlds/{id}/sensors/imu/{name}?since_s=` | IMU サンプル（t, gyro, accel, 真値, バイアス） |
| GET | `/viewer/` | ブラウザビューア（静的ファイル、`/` はここへリダイレクト） |
| WS | `/ws/{id}` | `cmd` / `waypoints` / `step` / `subscribe` / `run` / `stop` → 状態 JSON + センサ npz（バイナリ） |

重い処理はワーカースレッドで実行（World ごとにロック）するため、描画中もイベントループは応答します（デモで `/health` 最大 14 ms）。

```python
from lunarsim.core.world import World

w = World.from_yaml("configs/mare_plain.yaml", seed=1)
hm = w.heightmap  # z[i, j]（i: 南→北, j: 西→東）, resolution [m]
hm.height_at(1.5, -3.0)  # 双一次補間の標高 [m]
hm.slope(), hm.normals()
```

## 性能

`python scripts/bench.py`（ウォームアップ後の最良値、numba 使用）。4 vCPU クラウドコンテナ、Python 3.13。
「以前」は同じマシンで直前のコミット（Phase 8 + CDN 対応）を計測した値。

| 項目 | 目標 | 以前 | 現在 |
|---|---|---|---|
| 1 km² / 0.25 m の地形生成（4001×4001、クレーター約 10 万個 + 岩石約 6 万個） | 数十秒 | 20.1 s | **7.3 s** |
| 南極シナリオの World 構築（2 層: 2 km/1 m + 100 m/5 cm） | — | 14.0 s | **5.9 s** |
| 640×480 カメラ画像（影あり、2 層地形） | 1 秒以内 | 0.58 s | **0.30 s** |
| 同上 + 半影（太陽円盤 16 サンプル） | — | 0.90 s | **0.55 s** |
| 同上 + 半影 + 地球照 + ライト 2 灯 | — | 1.74 s | **0.93 s** |
| ステレオ 640×480 左右 | — | 1.01 s | **0.66 s** |
| 32 ライン回転式 LiDAR 1 スキャン | 0.2 秒以内 | 0.072 s | **0.057 s** |
| SPAD フラッシュ LiDAR 160×120（背景光の影レイ込み） | — | 0.146 s | 0.136 s |
| ホライズンマップ 501×501 点 × 64 方位（2 km 四方、4 m 間隔） | — | 21 s | 21 s |
| 長期日照 / PSR マップ（1 交点年を 2 時間刻み、4160 時刻） | — | 7.8 s | 6.8 s |
| ローバー走行 10 s（200 Hz IMU あり） | — | 1.96 s | **1.22 s** |

高速化の内容（いずれも numpy 版と同じ式・同じ処理順。numba が無い環境では従来の numpy 版で動く）:

- クレーター・岩石の刻印を numba カーネル化（Python ループ + 小パッチの numpy 呼び出しのオーバーヘッドを除去）。
  numpy 版との差は丸め誤差のみ（≤ 1e−10 m、ラベルは一致）。
- max-mipmap 構築を numba 化（4001² で 2.4 s → 0.34 s、結果は同一）。
- 描画: Hapke 反射、陰影法線の補間、2 層レイキャスト（遠方層・近傍層の区間処理を 1 カーネルに融合）を numba 化。
  画像（DN）は numpy 版とビット一致。
- 2 層地形の細部タイルで、共通のパワースペクトルと振幅をキャッシュ。
- ローバー接地の高さ参照（少数点）を numba 化。
- numba カーネルはディスクキャッシュ（`cache=True`）で、2 回目以降のプロセス起動時に再コンパイルしない。

`fastmath`（ホライズンマップで約 23 % 速い）は浮動小数点演算の並べ替えで CPU 間のビット一致が崩れうるため使っていない。
残るボトルネック: ホライズンマップ（距離方向の走査。傾斜地形では早期打ち切りが効きにくい）、ローバー接地の Python 部分。
numba が無い環境ではカメラ描画が約 2 s で目標（1 s）に届かないため、numba の導入を推奨。

## ライセンス

本体: MIT。依存ライブラリの一覧は [`LICENSES.md`](LICENSES.md)（`scripts/gen_licenses.py` で生成）。
