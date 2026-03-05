# All-Japan-Grid: OpenStreetMap からの日本全国送電網トポロジ自動構築

## Open-Source Geographic Topology of Japan's Transmission Grid — Methods, Reproducibility, and Limitations

---

## 目次 / Table of Contents

1. [概要 / Abstract](#1-概要--abstract)
2. [背景と動機 / Background and Motivation](#2-背景と動機--background-and-motivation)
3. [データ収集パイプライン / Data Collection Pipeline](#3-データ収集パイプライン--data-collection-pipeline)
4. [データモデルと内部表現 / Data Model and Internal Representation](#4-データモデルと内部表現--data-model-and-internal-representation)
5. [トポロジ構築アルゴリズム / Topology Construction Algorithms](#5-トポロジ構築アルゴリズム--topology-construction-algorithms)
6. [電気パラメータの推定 / Electrical Parameter Estimation](#6-電気パラメータの推定--electrical-parameter-estimation)
7. [pandapower/MATPOWER モデル変換 / Model Conversion](#7-pandapowermatpower-モデル変換--model-conversion)
8. [潮流計算ソルバ / Power Flow Solvers](#8-潮流計算ソルバ--power-flow-solvers)
9. [Unit Commitment ソルバ / Unit Commitment Solver](#9-unit-commitment-ソルバ--unit-commitment-solver)
10. [地域間連系線と全国モデル / Inter-Regional Interconnections](#10-地域間連系線と全国モデル--inter-regional-interconnections)
11. [テスト結果と検証 / Test Results and Validation](#11-テスト結果と検証--test-results-and-validation)
12. [限界と既知の問題 / Limitations and Known Issues](#12-限界と既知の問題--limitations-and-known-issues)
13. [再現手順 / Reproducibility Guide](#13-再現手順--reproducibility-guide)
14. [将来の展望 / Future Work](#14-将来の展望--future-work)
15. [参考文献 / References](#15-参考文献--references)

---

## 1. 概要 / Abstract

本プロジェクトは、OpenStreetMap (OSM) のクラウドソーシング地図データから日本全国10地域の送電網地理トポロジを機械的に抽出・構築するオープンソースパイプラインである。変電所約7,000箇所、送電線約40,000本、発電所約19,000箇所を含むデータセットを GeoJSON 形式で提供し、pandapower および MATPOWER 形式への変換、20種の交流潮流計算ソルバ、MILP ベースの Unit Commitment ソルバを含む解析ツール群を同梱する。

**重要**: 本データセットは OSM の公開データを機械処理したものであり、各電力会社・送電事業者・政府機関等の公式情報を正確に反映したものではない。電気パラメータ（インピーダンス、変圧器特性等）は合成推定値であり、本データに基づく潮流計算結果は工学的に有意ではない。

This project presents an open-source pipeline that automatically extracts and constructs the geographic topology of Japan's 10 regional transmission grids from OpenStreetMap (OSM) crowdsourced map data. The resulting dataset contains approximately 7,000 substations, 40,000 transmission lines, and 19,000 power plants in GeoJSON format. The project includes conversion tools for pandapower and MATPOWER formats, 20 AC power flow solvers, and a MILP-based unit commitment solver.

**Important**: This dataset is machine-processed from public OSM data and does not reflect official information from any electric power company. Electrical parameters are synthetic estimates, and power flow results based on this data are not engineering-grade.

---

## 2. 背景と動機 / Background and Motivation

### 2.1 日本の電力系統構造

日本の電力系統は10の地域電力会社（一般送配電事業者）によって運用される地域系統から構成される。

| 地域 | 事業者 | 周波数 | 最高電圧 |
|------|--------|--------|---------|
| 北海道 | 北海道電力 | 50 Hz | 275 kV |
| 東北 | 東北電力 | 50 Hz | 500 kV |
| 東京 | 東京電力 | 50 Hz | 500 kV |
| 中部 | 中部電力 | 60 Hz | 500 kV |
| 北陸 | 北陸電力 | 60 Hz | 500 kV |
| 関西 | 関西電力 | 60 Hz | 500 kV |
| 中国 | 中国電力 | 60 Hz | 500 kV |
| 四国 | 四国電力 | 60 Hz | 500 kV |
| 九州 | 九州電力 | 60 Hz | 500 kV |
| 沖縄 | 沖縄電力 | 60 Hz | 132 kV |

東日本（北海道・東北・東京）は 50 Hz、西日本（中部以西）は 60 Hz で運用され、東京−中部間は3箇所の周波数変換所（新信濃 600 MW、佐久間 300 MW、東清水 300 MW、合計 2,100 MW ＋増強分）で接続される。

### 2.2 公開データの不在

日本では、米国の FERC Form 715 や欧州の ENTSO-E Transparency Platform のような系統モデルの公開義務が存在せず、送電網の電気的パラメータ（インピーダンス、変圧器特性、需要配分等）は非公開である。OCCTO（電力広域的運営推進機関）は連系線容量や需給計画を公開しているが、バスレベルの系統モデルは提供していない。

### 2.3 本プロジェクトの位置づけ

OSM は送電設備の **地理的位置** と **空間的接続関係** を提供する。本プロジェクトは、この地理データから送電網のトポロジ（グラフ構造）を機械的に構築し、合成電気パラメータを付与することで、将来の補完データとの統合に向けた基盤データセットを提供する。

---

## 3. データ収集パイプライン / Data Collection Pipeline

### 3.1 Overpass API クエリ

OSM データは Overpass API を介して取得する。3種類の電力インフラを対象とする。

#### 変電所・開閉所

```
nwr["power"="substation"]({south},{west},{north},{east});
```

osmnx ライブラリ経由でタグ `{"power": "substation"}` を指定し、`features_from_bbox()` で取得する。

#### 送電線

```
nwr["power"~"line|cable"]({south},{west},{north},{east});
```

`power=line`（架空送電線）と `power=cable`（地中・海底ケーブル）の両方を取得する。

#### 発電所

```
[out:json][timeout:120];
(
  nwr["power"="plant"]({south},{west},{north},{east});
);
out center tags;
```

`power=plant`（施設レベル）のみを取得する。`power=generator`（個別発電ユニット、太陽光パネル等）は数が膨大（全国 200,000 超）であるため対象外とした。

### 3.2 バウンディングボックスによる地域分割

各地域のバウンディングボックスは `config/regions.yaml` で定義する。

```yaml
tokyo:
  bounding_box:
    lat_min: 34.8
    lat_max: 37.0
    lon_min: 138.4
    lon_max: 140.9
```

### 3.3 タイル分割による大規模地域の取得

大面積の地域（北海道、東京等）では、Overpass API のタイムアウトを回避するため、バウンディングボックスを N×M グリッドに分割し、タイルごとに順次取得する。

```python
def subdivide_bbox(bbox, rows, cols):
    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append((lon_min + c*lon_step, lat_min + r*lat_step,
                          lon_min + (c+1)*lon_step, lat_min + (r+1)*lat_step))
    return tiles
```

**API礼儀パラメータ**:
- タイムアウト: 300秒/タイル
- クエリ間隔: 3秒
- 最大リトライ: 5回（指数バックオフ、係数 2.0）

### 3.4 重複除去

タイル境界をまたぐ要素は複数タイルで重複取得される。`osmid` カラムによる `drop_duplicates()` で重複を除去する。`osmid` が利用できない場合は `geometry` カラムで代替する。

### 3.5 燃料種別の正規化

OSM の `plant:source` タグは表記揺れが多い。以下のマッピングで正規化する。

```python
FUEL_TYPE_MAP = {
    "coal": "coal", "gas": "gas", "natural_gas": "gas",
    "oil": "oil", "nuclear": "nuclear", "hydro": "hydro",
    "water": "hydro", "wind": "wind", "solar": "solar",
    "photovoltaic": "solar", "biomass": "biomass",
    "geothermal": "geothermal", "pumped_storage": "pumped_hydro",
    "diesel": "oil", ...
}
```

### 3.6 容量値のパース

OSM の `plant:output:electricity` は単位が不統一である。パーサは以下の規則で MW に変換する。

| 入力例 | 解釈 | 結果 |
|--------|------|------|
| `"1000000"` (無単位) | ワット（OSM慣例） | 1.0 MW |
| `"500 MW"` | メガワット | 500.0 MW |
| `"50000 kW"` | キロワット | 50.0 MW |
| `"1.2 GW"` | ギガワット | 1200.0 MW |

1000 以上の無単位値はワットとみなし、1000 未満はそのまま MW として扱う。

---

## 4. データモデルと内部表現 / Data Model and Internal Representation

### 4.1 クラス階層

```
GridNetwork
├── List[Substation]         # ノード（母線）
├── List[TransmissionLine]   # エッジ（送電線）
└── List[Generator]          # 発電機
```

### 4.2 Substation（変電所）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `id` | `str` | `{region}_osm_sub_{osmid}` |
| `name` | `str` | OSM `name` / `name:ja` |
| `latitude`, `longitude` | `float` | WGS-84 座標 |
| `voltage_kv` | `float` | OSM `voltage` タグ（ボルト→kV変換済み） |
| `region` | `str` | 所属地域 |
| `bus_type` | `str` | `PQ`, `PV`, `SLACK` |

Polygon / MultiPolygon ジオメトリはセントロイドに変換する。

### 4.3 TransmissionLine（送電線）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `id` | `str` | `{region}_osm_line_{osmid}` |
| `from_substation_id` | `str` | 始点変電所ID（近傍マッチング） |
| `to_substation_id` | `str` | 終点変電所ID（近傍マッチング） |
| `voltage_kv` | `float` | 電圧階級 |
| `length_km` | `float` | Haversine 距離に基づくポリライン長 |
| `coordinates` | `List[Tuple]` | 経路座標列 |

### 4.4 Generator（発電機）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `id` | `str` | `{region}_gen_{index}` |
| `capacity_mw` | `float` | 発電容量 |
| `fuel_type` | `str` | 正規化済み燃料種別 |
| `connected_bus_id` | `str` | 最近傍変電所（50 km以内） |
| `vm_pu` | `float` | 電圧設定値（p.u.） |

### 4.5 GridNetwork

`GridNetwork` は地域単位でインスタンス化され、`merge_regions()` で全国モデルに統合できる。周波数が異なるネットワークの統合時は `frequency_hz = 0`（混合）に設定される。

---

## 5. トポロジ構築アルゴリズム / Topology Construction Algorithms

### 5.1 端点マッチング（Endpoint Matching）

OSM の送電線は地理的な経路（LineString）として記録されており、どの変電所に接続されるかの情報は含まれない。本パイプラインでは、LineString の始点・終点座標を最近傍の変電所に空間的にマッチングする。

**アルゴリズム**:

```
入力: 送電線 L の始点座標 (lat_s, lon_s), 終点座標 (lat_e, lon_e)
      全変電所リスト S = [(id_1, lat_1, lon_1), ...]
      最大許容距離 d_max = 50 km

for each endpoint (lat, lon) in [(lat_s, lon_s), (lat_e, lon_e)]:
    best_id = argmin_{s ∈ S} haversine(lat, lon, s.lat, s.lon)
    best_dist = haversine(lat, lon, S[best_id].lat, S[best_id].lon)
    if best_dist > d_max:
        reject endpoint (line dropped)
    else:
        assign endpoint → S[best_id]

if from_id == to_id:
    reject (self-loop)
```

**Haversine 距離計算**:

$$d = 2R \cdot \arctan2\left(\sqrt{a}, \sqrt{1-a}\right)$$

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cdot \cos\phi_2 \cdot \sin^2\left(\frac{\Delta\lambda}{2}\right)$$

ここで $R = 6371$ km（地球平均半径, WGS-84）。

**送電線長の計算**: LineString の全頂点間の Haversine 距離の総和として計算する。長さゼロの場合は端点間直線距離にフォールバックし、それでもゼロの場合は最小値 1.0 km を設定する。

### 5.2 電圧値のクリーニング

OSM の `voltage` タグには多くの異常値が含まれる。

```python
VALID_VOLTAGES = [66, 77, 110, 132, 154, 187, 220, 275, 500]  # kV

def _clean_voltage(v_kv):
    if v_kv <= 0:
        v_kv = 66
    elif v_kv > 600:
        v_kv = v_kv % 1000 if v_kv > 1000 else 500
    return min(VALID_VOLTAGES, key=lambda x: abs(x - v_kv))
```

日本の送電系統で使用される標準電圧階級（66 kV 以上）に最近傍スナッピングする。66 kV 未満の配電電圧は対象外とする。

### 5.3 母線電圧推定

OSM 変電所の多くは `voltage` タグを持たない（`vn_kv = 0`）。以下の優先順位で電圧を推定する。

1. 接続された送電線の電圧の最大値を採用
2. 残る零電圧母線にはネットワーク全体の中央値を代入

### 5.4 異電圧線路の変圧器変換

`from_bus` と `to_bus` の電圧比が 1.5 倍以上の線路は、送電線ではなく変圧器として再解釈する。

```python
ratio = max(v_from, v_to) / min(v_from, v_to)
if ratio >= 1.5:
    # 線路を削除し、変圧器を挿入
    pp.create_transformer_from_parameters(net,
        hv_bus=hv_bus, lv_bus=lv_bus,
        sn_mva=sn, vn_hv_kv=hv, vn_lv_kv=lv,
        vkr_percent=vkr, vk_percent=vk, ...)
```

変圧器パラメータは電圧階級に応じた典型値を使用する。

| HV (kV) | Sn (MVA) | vk (%) | vkr (%) |
|----------|----------|--------|---------|
| ≥ 275 | 800 | 14.0 | 0.20 |
| ≥ 154 | 400 | 12.0 | 0.25 |
| ≥ 110 | 200 | 10.0 | 0.30 |
| < 110 | 100 | 8.0 | 0.40 |

### 5.5 孤立成分の橋渡し（Component Bridging）

OSM トポロジは多くの場合、複数の連結成分に分断される。最大成分以外の成分を接続するため、最近傍母線間に推定線路または変圧器を挿入する。

**アルゴリズム**:

```
1. NetworkX グラフを構築（変圧器含む）
2. 連結成分を列挙し、サイズ降順にソート
3. main_comp = 最大成分
4. KD-Tree を main_comp の母線座標で構築
5. for each orphan_comp in comps[1:]:
     for each bus in orphan_comp:
       (dist, nearest_main_bus) = KD-Tree.query(bus.coord)
     if min_dist > 2.7 degrees (≈ 300 km):
       skip (too far)
     if voltage_ratio < 1.5:
       insert estimated line
     else:
       insert estimated transformer
     main_comp = main_comp ∪ orphan_comp
     rebuild KD-Tree
6. Drop any still-disconnected components
```

推定線路パラメータ:
- 220 kV 以上: R = 0.06 Ω/km, X = 0.3 Ω/km, C = 10 nF/km, max_I = 2.0 kA
- 220 kV 未満: R = 0.12 Ω/km, X = 0.4 Ω/km, C = 8 nF/km, max_I = 1.0 kA

### 5.6 発電機の変電所マッチング

発電所は最近傍の変電所に地理的に接続する。KD-Tree による空間探索を使用し、最大許容距離は 50 km とする。

```python
tree = cKDTree(substation_coords)
for gen in generators:
    dist_deg, idx = tree.query((gen.latitude, gen.longitude))
    dist_km = dist_deg * 111.0
    if dist_km <= 50.0:
        gen.connected_bus_id = substations[idx].id
```

---

## 6. 電気パラメータの推定 / Electrical Parameter Estimation

### 6.1 送電線パラメータ参照テーブル

`config/line_types.yaml` に日本の送電線の標準的な電気パラメータを定義する。

| 電圧 (kV) | R (Ω/km) | X (Ω/km) | B (S/km) | Imax (kA) | 導体 | 回線数 |
|-----------|----------|----------|----------|-----------|------|--------|
| 500 | 0.012 | 0.290 | 4.1×10⁻⁶ | 4.0 | ACSR 810mm² ×4 | 2 |
| 275 | 0.028 | 0.325 | 3.85×10⁻⁶ | 2.0 | ACSR 410mm² ×2 | 2 |
| 220 | 0.032 | 0.335 | 3.75×10⁻⁶ | 1.8 | ACSR 410mm² ×2 | 2 |
| 187 | 0.038 | 0.350 | 3.65×10⁻⁶ | 1.5 | ACSR 330mm² ×2 | 2 |
| 154 | 0.050 | 0.380 | 3.5×10⁻⁶ | 1.0 | ACSR 330mm² | 2 |
| 132 | 0.045 | 0.370 | 3.55×10⁻⁶ | 1.2 | ACSR 330mm² | 2 |
| 110 | 0.055 | 0.385 | 3.45×10⁻⁶ | 0.9 | ACSR 240mm² | 2 |
| 77 | 0.100 | 0.395 | 3.3×10⁻⁶ | 0.7 | ACSR 200mm² | 1 |
| 66 | 0.120 | 0.400 | 3.2×10⁻⁶ | 0.6 | ACSR 160mm² | 1 |

出典: OCCTO 広域系統計画文書、TEPCO/KEPCO 設計基準（一般公開値）。

### 6.2 サセプタンス→キャパシタンス変換

pandapower は `c_nf_per_km`（nF/km）を要求するが、工学文献はサセプタンス `B`（S/km）を使用する。変換は周波数依存である。

$$c_{\text{nF/km}} = \frac{B_{\text{S/km}}}{2\pi f} \times 10^9$$

東日本（50 Hz）と西日本（60 Hz）で同じ B 値でも c 値が異なる。

### 6.3 電圧クラス フォールバック

参照テーブルに存在しない電圧クラスの場合、最近傍の電圧クラスのパラメータにフォールバックする。いずれも該当しない場合のジェネリックデフォルト: R = 0.05 Ω/km, X = 0.4 Ω/km, C = 10.0 nF/km, Imax = 1.0 kA。

### 6.4 per-unit 変換とフロア制約

MATPOWER 形式への変換時、per-unit 値に最小値制約を設ける。

| パラメータ | 最小値 | 目的 |
|-----------|--------|------|
| `x_pu` (線路) | 0.005 | ゼロインピーダンス回避 |
| `x_pu` (変圧器) | 0.03 | 現実的な変圧器インピーダンス |
| `r/x` 比 | 0.05 | 極端な R/X 比の回避 |
| `b_pu` 上限 | 5.0 | 数値安定性 |
| `r_pu`, `x_pu` 上限 | 2.0 | 極端なインピーダンスの回避 |

---

## 7. pandapower/MATPOWER モデル変換 / Model Conversion

### 7.1 pandapower ネットワーク構築

`PandapowerBuilder` は `GridNetwork` を pandapower ネットワークに変換する。

**処理手順**:
1. **母線作成**: 各変電所 → `pp.create_bus(vn_kv, name, geodata)`
2. **線路作成**: 各送電線 → `pp.create_line_from_parameters(r, x, c, max_i, length)`
3. **発電機作成**: 各発電機 → `pp.create_gen(bus, p_mw, vm_pu)`
4. **外部系統（スラック）**: `pp.create_ext_grid(bus, vm_pu=1.0)`
5. **母線電圧推定**: 零電圧母線の修正

**スラック母線選択優先順位**:
1. `BusType.SLACK` が明示された変電所
2. 最大容量発電機が接続された母線
3. ネットワーク先頭の母線（フォールバック）

### 7.2 MATPOWER 形式エクスポート

pandapower ネットワークを MATPOWER case struct (`.mat` ファイル) に変換する。潮流計算は実行せず、フラットスタート（Vm = 1.0 p.u., Va = 0°）で出力する。

**負荷配分**: OCCTO 2023年度ピーク需要データに基づき、電圧階級別の重みで各母線に配分する。

```yaml
voltage_weights:
  500: 0.05   # 基幹系統: 少量の負荷
  275: 0.15
  154: 0.30
  66:  0.50   # 配電系統に近い: 大量の負荷
```

目標負荷 = min(地域ピーク需要 × 負荷率 0.85, 総発電容量 × 0.80)

**並列ブランチの統合**: 同一母線間の並列ブランチはアドミタンス加算で統合する。

$$Y_{\text{parallel}} = \sum_k Y_k, \quad Z_{\text{eq}} = \frac{1}{Y_{\text{parallel}}}$$

### 7.3 電圧支持機（同期調相機）

110 kV 以上の発電機未接続母線に、無効電力供給用の仮想同期調相機（P = 0, Q ≠ 0）を挿入する。

| 電圧 (kV) | Qmax (Mvar) | Qmin (Mvar) |
|-----------|-------------|-------------|
| ≥ 275 | 500 | -200 |
| ≥ 154 | 200 | -80 |
| ≥ 110 | 100 | -40 |

---

## 8. 潮流計算ソルバ / Power Flow Solvers

### 8.1 ソルバ概要

20種の AC 潮流計算ソルバを4カテゴリに分類して実装する。

#### pandapower ラッパー（5種）

pandapower の内蔵ソルバをラップし、統一インターフェース `ACMethodResult` で結果を返す。

- `pp_nr` — Newton-Raphson
- `pp_gs` — Gauss-Seidel
- `pp_fdpf_bx` — Fast Decoupled (BX)
- `pp_fdpf_xb` — Fast Decoupled (XB)
- `pp_backward_forward` — Backward/Forward Sweep

#### カスタム Newton-Raphson 系（7種）

PYPOWER 内部行列（Ybus, Sbus, V0）を直接操作するカスタム実装。

- `custom_nr` — 標準 Newton-Raphson
- `custom_nr_linesearch` — 線探索付き NR
- `custom_nr_iwamoto` — 岩本の最適乗数法
- `custom_nr_rectangular` — 直交座標系 NR
- `custom_nr_current` — 電流注入型 NR
- `custom_nr_dishonest` — 不正直 NR（ヤコビアン固定）
- `custom_nr_levenberg` — Levenberg-Marquardt NR

#### カスタム反復法（4種）

- `custom_gs` — Gauss-Seidel
- `custom_gs_accelerated` — 加速 Gauss-Seidel
- `custom_jacobi` — Jacobi 法
- `custom_gs_sor` — SOR (Successive Over-Relaxation)

#### カスタム分離型（4種）

- `custom_fdpf_bx` — Fast Decoupled (BX)
- `custom_fdpf_xb` — Fast Decoupled (XB)
- `custom_decoupled_nr` — 分離型 Newton-Raphson
- `custom_nr_continuation` — 連続法 NR

### 8.2 Newton-Raphson 法の数学的定式化

**潮流方程式**:

$$S_i = V_i \sum_{k=1}^{n} Y_{ik}^* V_k^* = P_i + jQ_i$$

**不整合ベクトル**:

$$F = \begin{bmatrix} \Delta P_{pvpq} \\ \Delta Q_{pq} \end{bmatrix} = \begin{bmatrix} P_{\text{calc}} - P_{\text{spec}} \\ Q_{\text{calc}} - Q_{\text{spec}} \end{bmatrix}$$

**ヤコビアン行列** (極座標形式):

$$J = \begin{bmatrix} \frac{\partial P}{\partial \theta} & \frac{\partial P}{\partial |V|} \\ \frac{\partial Q}{\partial \theta} & \frac{\partial Q}{\partial |V|} \end{bmatrix}$$

**更新ステップ**:

$$J \cdot \Delta x = -F$$

$$\Delta x = \begin{bmatrix} \Delta \theta \\ \Delta |V| / |V| \end{bmatrix}$$

`dSbus_dV` は pandapower/PYPOWER のスパース偏微分関数を使用する。

### 8.3 収束判定と発散検知

- **収束判定**: $\|F\|_\infty < \varepsilon$（デフォルト $\varepsilon = 10^{-8}$）
- **発散検知**: 不整合ノルムが初期値の 10 倍を超えた場合に打ち切り
- **NaN/Inf 検知**: 電圧ベクトルに NaN/Inf が発生した場合に即座に中止
- **特異ヤコビアン**: `spsolve` が `LinAlgError` を発生した場合にキャッチし、失敗理由を記録

### 8.4 統一結果インターフェース

```python
@dataclass
class ACMethodResult:
    converged: bool = False
    iterations: int = 0
    V: Optional[np.ndarray] = None  # 複素電圧ベクトル
    elapsed_sec: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    failure_reason: Optional[str] = None
```

---

## 9. Unit Commitment ソルバ / Unit Commitment Solver

### 9.1 問題定式化

MILP (Mixed-Integer Linear Programming) による Unit Commitment 問題を定式化する。

**決定変数**:
- $u_{g,t} \in \{0, 1\}$ — 発電機 $g$ の時刻 $t$ における起動状態
- $v_{g,t} \in \{0, 1\}$ — 起動インジケータ
- $w_{g,t} \in \{0, 1\}$ — 停止インジケータ
- $p_{g,t} \geq 0$ — 出力（MW）

**目的関数** (総コスト最小化):

$$\min \sum_g \sum_t \left[ C_{\text{fuel}}^g \cdot p_{g,t} + (C_{\text{NL}}^g + C_{\text{labor}}^g) \cdot u_{g,t} + C_{\text{SU}}^g \cdot v_{g,t} + C_{\text{SD}}^g \cdot w_{g,t} \right]$$

### 9.2 制約条件

#### 需給バランス

$$\sum_g p_{g,t} \geq D_t \quad \forall t$$

#### 容量制約

$$P_{\min}^g \cdot u_{g,t} \leq p_{g,t} \leq P_{\max}^g \cdot u_{g,t} \quad \forall g, t$$

#### 起動・停止ロジック

$$v_{g,t} - w_{g,t} = u_{g,t} - u_{g,t-1} \quad \forall g, t$$

#### 最小起動時間 (MUT)

$$\sum_{\tau=t-\text{MUT}+1}^{t} v_{g,\tau} \leq u_{g,t} \quad \forall g, t$$

#### 最小停止時間 (MDT)

$$\sum_{\tau=t-\text{MDT}+1}^{t} w_{g,\tau} \leq 1 - u_{g,t} \quad \forall g, t$$

#### ランプ制約 (Big-M 緩和)

$$p_{g,t} - p_{g,t-1} \leq R_{\text{up}}^g + P_{\max}^g (1 - u_{g,t-1})$$

$$p_{g,t-1} - p_{g,t} \leq R_{\text{down}}^g + P_{\max}^g (1 - u_{g,t})$$

#### 予備力制約

$$\sum_g P_{\max}^g \cdot u_{g,t} \geq D_t (1 + \alpha) \quad \forall t$$

#### メンテナンスウィンドウ

$$u_{g,t} = 0 \quad \forall g, t \in \text{MaintenanceWindow}(g)$$

### 9.3 蓄電設備モデル

揚水発電・バッテリーに対して SOC (State of Charge) 制約を追加する。

$$\text{SOC}_{g,t} = \text{SOC}_{g,t-1} + \eta_{\text{ch}} \cdot p_{\text{ch},g,t} \cdot \Delta t - \frac{p_{\text{dis},g,t}}{\eta_{\text{dis}}} \cdot \Delta t$$

$$p_{g,t} = p_{\text{dis},g,t} - p_{\text{ch},g,t}$$

$$p_{\text{ch},g,t} \leq R_{\text{ch}}^g \cdot z_{\text{ch},g,t}$$

$$p_{\text{dis},g,t} \leq R_{\text{dis}}^g \cdot (1 - z_{\text{ch},g,t})$$

終端 SOC 制約: $\text{SOC}_{g,T} \geq \text{SOC}_{\min} \cdot E_{\max}^g$

### 9.4 ソルババックエンド

HiGHS (HiGHS_CMD) を優先使用し、利用不可の場合は CBC (PuLP バンドル) にフォールバックする。

---

## 10. 地域間連系線と全国モデル / Inter-Regional Interconnections

### 10.1 連系線データ

`data/reference/interconnections.yaml` に OCCTO 公開データに基づく地域間連系線を定義する。

| 連系線 | 区間 | 容量 (MW) | 種別 | 電圧 (kV) |
|--------|------|----------|------|----------|
| 北本連系線 | 北海道↔東北 | 900 | HVDC | 250 |
| 東北東京間連系線 | 東北↔東京 | 5,550 | AC | 500 |
| 東京中部間連系設備 | 東京↔中部 | 2,100 | FC | 275 |
| 中部関西間連系線 | 中部↔関西 | 2,530 | AC | 500 |
| 中部北陸間連系線 | 中部↔北陸 | 1,900 | AC | 275 |
| 関西中国間連系線 | 関西↔中国 | 4,090 | AC | 500 |
| 関西四国間連系線 | 関西↔四国 | 1,400 | AC | 500 |
| 中国四国間連系線 | 中国↔四国 | 1,200 | AC | 500 |
| 関門連系線 | 中国↔九州 | 2,780 | AC | 500 |

沖縄は離島系統であり、本土との連系はない。

### 10.2 全国モデル構築

10地域の MATPOWER ケースを母線番号オフセットで連結し、連系線をタイライン（ブランチ）として追加する。

**連系線インピーダンス** (per-unit, 100 MVA ベース):

| 種別 | R (p.u.) | X (p.u.) | B (p.u.) |
|------|---------|---------|---------|
| AC | 0.005 | 0.05 | 0.02 |
| HVDC | 0.001 | 0.01 | 0.0 |
| FC | 0.002 | 0.02 | 0.0 |

全国モデルではスラック母線を1つ（東京エリア）に統一し、他地域のスラック母線は PV 母線に降格する。

---

## 11. テスト結果と検証 / Test Results and Validation

### 11.1 テストスイート構成

pytest による自動テストスイートを10カテゴリ構成で実装する。

| テストクラス | 対象 | テスト数 |
|-------------|------|---------|
| `TestMethodRegistry` | ソルバレジストリ（20種確認） | 5 |
| `TestSolverInterface` | `ACMethodResult` データクラス | 5 |
| `TestPandapowerNRWrapper` | pp_nr ラッパー | 3 |
| `TestPandapowerGSWrapper` | pp_gs ラッパー | 2 |
| `TestCustomNR` | カスタム NR ソルバ | 4 |
| `TestCustomGS` | カスタム GS ソルバ | 3 |
| `TestNetworkPrep` | ネットワーク前処理 | 7 |
| `TestConvergenceReport` | 収束レポート生成 | 6 |
| `TestSingularJacobianHandling` | 特異ヤコビアン処理 | 2 |
| `TestNaNDetection` | NaN/Inf 検知 | 3 |

### 11.2 3母線テストネットワーク

全ソルバテストは以下の3母線テストネットワークで実施する。

```
Bus 0 (slack, 110kV) ---[10km]--- Bus 1 (gen 40MW, PV)
  |                                 |
[15km]                           [10km]
  |                                 |
Bus 2 (load 60MW+j20Mvar, PQ) ----+
```

線路種別: `149-AL1/24-ST1A 110.0`（pandapower 標準110kV架空線）

### 11.3 検証項目

**カスタム NR ソルバ**:
- 収束確認（`converged == True`）
- 電圧ベクトルの妥当性（複素数、正しい次元）
- 電圧振幅の範囲確認（0.8 < |V| < 1.2 p.u.）
- 収束履歴の単調減少確認
- 最終不整合 < 10⁻⁸

**カスタム GS ソルバ**:
- 反復回数の正確性（max_iter で停止）
- NaN/Inf の非発生確認

**ネットワーク前処理**:
- Ybus の疎行列形式確認（CSC）
- Ybus の正方性確認
- Sbus の次元・型確認（1D 複素配列）
- ref, pv, pq の非重複・全カバー確認
- baseMVA > 0 の確認

**エラーハンドリング**:
- 特異ヤコビアン検出とリカバリ
- V0 への NaN 注入時の早期中止
- Inf 注入時の検出

### 11.4 OSM データに対する潮流計算結果

**重要**: OSM トポロジに合成パラメータを適用した潮流計算は、電気パラメータの不在により**物理的に無意味な結果**を生成する。これは既知かつ想定された制限事項である。

典型的な収束特性:
- **3母線テストケース**: カスタム NR は3-5反復で収束（10⁻⁸ 以下）
- **IEEE テストケース**: pandapower/MATPOWER 標準テストケースでは全20ソルバが正しく動作
- **OSM ベースの地域モデル**: 母線数が多く（数百〜数千）、不完全なトポロジにより収束困難または物理的に無意味な解が得られる

---

## 12. 限界と既知の問題 / Limitations and Known Issues

### 12.1 データの根本的限界

| 欠落データ | 影響 | 補完候補 |
|-----------|------|---------|
| 線路インピーダンス (R, X, B) | 潮流計算不可 | OCCTO パラメータ、文献値 |
| 母線間接続関係 | ヒューリスティックマッチングは誤接続を含む | 手動検証、OCCTO トポロジ |
| 発電機詳細パラメータ | コストカーブ、ランプレート等が欠如 | OCCTO 供給計画、国土数値情報 P03 |
| 需要データ | 母線への需要配分なし | OCCTO 地域需要、県別統計 |
| 変圧器データ | タップ比、巻線構成なし | 合成推定または事業者開示 |
| 開閉器トポロジ | 遮断器レベルの詳細なし | 日本では非公開 |

### 12.2 端点マッチングの脆弱性

50 km の最大許容距離は、大半の接続を捕捉する一方で、偽接続（実際には繋がっていない遠方の変電所への誤マッチ）も生成する。特に変電所密度が低い地方部で問題が顕著である。

### 12.3 OSM データ品質の課題

- **電圧タグの欠損/異常**: `voltage=null`, `voltage=""`, `voltage="yes"`, `voltage="500000;275000"` 等の多様な異常パターン
- **名称の表記揺れ**: 「変電所」「ＳＳ」「S/S」「発電所」「PS」等の混在
- **地域境界問題**: 同名変電所が複数地域に存在する可能性
- **更新の非同期性**: OSM データは更新時期がまちまちで、撤去された設備が残存する場合がある

### 12.4 合成パラメータの限界

- 参照テーブルの値は**電圧階級ごとの代表値**であり、個別線路の導体仕様・回線数・架線形態を反映しない
- 変圧器パラメータは4段階の粗い推定にすぎず、実際のタップ比・巻線構成は不明
- 負荷配分は電圧階級別の重みによる一様配分であり、実際の需要分布を反映しない

### 12.5 まとめ: "地図があるからデータがある" は誤り

地図上に送電線が描かれていることは、電力系統モデルを構築するのに十分なデータが存在することを意味しない。地理データと電気データは根本的に異なるものであり、前者から後者を完全に導出することはできない。

---

## 13. 再現手順 / Reproducibility Guide

### 13.1 環境構築

```bash
# Python 3.10+
git clone https://github.com/lutelute/All-Japan-Grid.git
cd All-Japan-Grid
pip install -r requirements.txt
```

主要依存パッケージ: pandapower ≥3.4, geopandas ≥0.14, scipy ≥1.10, pulp ≥3.3, highspy ≥1.7, fastapi ≥0.110

### 13.2 OSM データ取得

```bash
# 全地域の変電所・送電線取得（タイル分割あり）
python scripts/fetch_subdivided.py --region hokkaido --rows 3 --cols 3
python scripts/fetch_subdivided.py --region tokyo --rows 2 --cols 2
# ... (各地域)

# 発電所取得
python scripts/fetch_plants.py
```

**注意**: Overpass API は負荷分散のため、大量クエリ時には待機時間が発生する。全10地域の取得には数時間を要する場合がある。

### 13.3 MATPOWER モデル構築

```bash
python scripts/build_alljapan_full.py
# 出力: output/matpower_alljapan/{region}.mat (×10) + alljapan.mat
```

### 13.4 テスト実行

```bash
pytest tests/ -v
```

### 13.5 可視化サーバ

```bash
# FastAPI サーバ
uvicorn src.server.app:app --reload
open http://localhost:8000

# GitHub Pages 静的サイト（ローカル）
python -m http.server -d docs 8080
open http://localhost:8080
```

### 13.6 出力ファイル

```
data/
  {region}_substations.geojson    # 変電所 GeoJSON
  {region}_lines.geojson          # 送電線 GeoJSON
  {region}_plants.geojson         # 発電所 GeoJSON
output/matpower_alljapan/
  {region}.mat                    # 地域別 MATPOWER ケース
  alljapan.mat                    # 全国統合 MATPOWER ケース
```

---

## 14. 将来の展望 / Future Work

### 14.1 補完データソースとの統合

| データソース | 提供内容 | アクセス |
|-------------|---------|---------|
| OCCTO | 連系線容量、地域需要、需給計画 | [occto.or.jp](https://www.occto.or.jp/) |
| 国土数値情報 P03 | 発電所位置・容量・燃料 | [nlftp.mlit.go.jp](https://nlftp.mlit.go.jp/ksj/) |
| JEPX | スポット市場価格 | [jepx.jp](http://www.jepx.jp/) |
| PyPSA-Earth / atlite | 再エネ資源、合成系統補完 | [pypsa-earth.readthedocs.io](https://pypsa-earth.readthedocs.io/) |
| MATPOWER テストケース | 検証済みベンチマークモデル | [matpower.org](https://matpower.org/) |

### 14.2 端点マッチングの改善

- OSM の `way` 共有ノードによるトポロジ推定
- 電圧整合性チェック（同一電圧の変電所のみをマッチ候補とする）
- グラフベースの接続性検証

### 14.3 需要モデリング

- 県別・メッシュ別の人口・産業統計に基づく需要配分
- OCCTO 30分値データとの連携
- 時系列負荷カーブの統合

---

## 15. 参考文献 / References

1. OpenStreetMap contributors. *OpenStreetMap*. https://www.openstreetmap.org/
2. OCCTO (電力広域的運営推進機関). *広域系統長期方針*. https://www.occto.or.jp/
3. L. Thurner et al. "pandapower — An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems." *IEEE Transactions on Power Systems*, vol. 33, no. 6, pp. 6510-6521, 2018.
4. R. D. Zimmerman, C. E. Murillo-Sánchez, R. J. Thomas. "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education." *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12-19, 2011.
5. G. Boeing. "OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks." *Computers, Environment and Urban Systems*, vol. 65, pp. 126-139, 2017.
6. J. D. Glover, M. S. Sarma, T. J. Overbye. *Power Systems Analysis and Design*, 6th ed. Cengage Learning, 2017.
7. 国土交通省. *国土数値情報 P03 (発電施設)*. https://nlftp.mlit.go.jp/ksj/
8. JEPX (日本卸電力取引所). https://www.jepx.jp/
9. M. Fiorini et al. "PyPSA-Earth. A New Global Open Energy System Optimization Model Demonstrated in Africa." *Applied Energy*, vol. 341, 2023.

---

## 免責事項 / Disclaimer

本ホワイトペーパーおよび All-Japan-Grid データセットは、公開されている OpenStreetMap のデータを機械的に自動処理して生成したものです。各電力会社・送電事業者・政府機関等の公式情報を正確に反映したものではありません。クラウドソーシングによる地図データおよび自動抽出処理に起因する誤り・欠落・不正確さが含まれる可能性があります。本データの利用は自己責任でお願いいたします。本データの利用により生じたいかなる損害・損失・結果についても、作成者は一切の責任を負いません。

This whitepaper and the All-Japan-Grid dataset are generated automatically by machine processing of publicly available OpenStreetMap data. They do not reflect official information from any electric power company, transmission operator, or government agency. Use at your own risk. The authors assume no liability for any damages arising from the use of this data.

---

*ネットワークデータ: ODbL (OpenStreetMap) / コード: MIT*
