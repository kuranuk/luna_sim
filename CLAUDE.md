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

## 作業の進め方

Phase ごとに 実装 → テスト → `scripts/demo_phaseN.py` 実行 → commit/push。
判断に迷う仕様はユーザーに選択肢とトレードオフを示して確認する。
