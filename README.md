# LunarSim

月面ローバーの航法・誘導・自律走行（SLAM、LiDAR/カメラ認識、経路計画）の開発・評価向けの、
Python 製の軽量な月面ローバーシミュレータです。

- **オフライン・管理者権限なし・GPU なし**で動作（`pip` のみ、ランタイムのネットワークアクセスなし）
- **物理的にもっともらしい**パラメータ（出典は `docs/references.md`）
- **再現性**: seed + YAML 設定から World を完全に再構築（同一 lock でビット一致）

開発状況: **Phase 2**（フラクタル地形、確率分布によるクレーター・岩石、セマンティックラベル、ground truth 出力）。
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
```

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
| 同上 + クレーター（平衡分布 2 %, 1〜200 m, 約 3.1 万個を刻印）+ 岩石（約 6 万個） | 数十秒 | 約 14 s | 同上（numba 未使用） |

`configs/bench_1km.yaml`、`LUNARSIM_PERF=1 python -m pytest -s tests/test_perf.py`。ノート PC での計測は今後追記します。
ボトルネックはクレーター刻印ループ（Python のループ + 小パッチの numpy 演算）。numba 化は Phase 8 で検討。

## ライセンス

本体: MIT。依存ライブラリの一覧は [`LICENSES.md`](LICENSES.md)（`scripts/gen_licenses.py` で生成）。
