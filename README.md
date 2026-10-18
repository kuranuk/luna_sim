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
frame = c.camera(wid, "navcam", fmt="npz", buffers=["depth", "labels"])   # dict of arrays
scan = c.lidar(wid, "spad_flash")                                         # points (N,4), ring, time, ...
with c.stream(wid) as s:                                                  # WebSocket
    s.send(type="subscribe", lidars=["spad_flash"])
    s.receive(); s.step(0.5); print(s.receive()["data"]["points"].shape)
```

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/health`, `/version` | 死活・バージョン（認証不要） |
| POST | `/worlds` | `{"config": {...}}` または `{"config_yaml": "..."}`, `seed` → `world_id` |
| GET / DELETE | `/worlds/{id}` | メタデータ + 状態 / 削除 |
| GET | `/worlds/{id}/heightmap?format=npy\|png16&layer=near\|far` | 高さマップ（幾何はヘッダ） |
| GET | `/worlds/{id}/craters`, `/rocks` | ground truth |
| PUT | `/worlds/{id}/sun` | `{"azimuth_deg","elevation_deg"}` または `{"utc"}` |
| PUT | `/worlds/{id}/lights/{name}` | `{"on": bool}` |
| POST | `/worlds/{id}/rover/cmd`, `/rover/waypoints` | 速度指令 / ウェイポイント |
| POST | `/worlds/{id}/step` | `{"dt_s"}`（決定論的）→ 状態 |
| GET | `/worlds/{id}/sensors/camera/{name}?format=png\|png16\|npz&buffers=depth,labels&eye=left\|right` | 画像 + GT |
| GET | `/worlds/{id}/sensors/lidar/{name}?format=npz\|ply\|pcd\|json` | 点群（JSON は小規模のみ） |
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

| 項目 | 目標 | 結果 | 環境 |
|---|---|---|---|
| 1 km² / 0.25 m の地形生成（4001×4001、フラクタルのみ） | 数十秒 | 3.8 s | 4 vCPU クラウドコンテナ、Python 3.13 |
| 同上 + クレーター（平衡分布 2 %, 1〜200 m, 約 3.1 万個を刻印）+ 岩石（約 6 万個） | 数十秒 | 約 14〜16 s | 同上（地形生成に numba 未使用） |
| 640×480 カメラ画像（影あり、2 層地形: 2 km/1 m + 100 m/5 cm） | 1 秒以内 | **0.42〜0.48 s**（numba）/ 1.9〜2.3 s（numpy フォールバック） | 同上 |
| 同上 + ライト 2 灯（影レイ込み） | — | 0.82 s（ステレオ左右で 1.8 s） | 同上 |
| ローバー走行 120 s（0.5 s ステップ、接地計算込み） | — | 約 10 s（ステレオ 4 組の描画を含む） | 同上 |
| API: step / state / カメラ npz（ライト 2 灯）/ LiDAR npz | — | 16 ms / 3 ms / 0.98 s（キャッシュ時 13 ms）/ 0.17 s | 同上、ローカルホスト |
| 南極シナリオの World 構築（2 層、クレーター約 8 万個 + 近傍 0.35 万個） | — | 約 11 s | 同上 |
| 32 ライン回転式 LiDAR 1 スキャン（5.8 万ビーム、ガウスモデル） | 0.2 s 以内 | **0.07 s**（numba） | 同上 |
| SPAD フラッシュ LiDAR 1 フレーム（160×120 画素、背景光の影レイ込み） | — | **0.12〜0.19 s**（numba） | 同上 |

`configs/bench_1km.yaml`、`LUNARSIM_PERF=1 python -m pytest -s tests/test_perf.py`。ノート PC での計測は今後追記します。
地形生成のボトルネックはクレーター刻印ループ（Python のループ + 小パッチの numpy 演算）。numba 化は Phase 8 で検討。
numba が無い環境ではカメラ描画が目標（1 s）に届かない（2.25 s）。主因はベクトル化したレイ走査の反復回数で、numba の導入を推奨。

## ライセンス

本体: MIT。依存ライブラリの一覧は [`LICENSES.md`](LICENSES.md)（`scripts/gen_licenses.py` で生成）。
