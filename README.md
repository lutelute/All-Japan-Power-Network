```
 █████╗ ██╗     ██╗             ██╗██████╗         ██████╗ ██████╗ ██╗██████╗
██╔══██╗██║     ██║             ██║██╔══██╗       ██╔════╝ ██╔══██╗██║██╔══██╗
███████║██║     ██║             ██║██████╔╝█████╗ ██║  ███╗██████╔╝██║██║  ██║
██╔══██║██║     ██║        ██   ██║██╔═══╝ ╚════╝ ██║   ██║██╔══██╗██║██║  ██║
██║  ██║███████╗███████╗   ╚█████╔╝██║            ╚██████╔╝██║  ██║██║██████╔╝
╚═╝  ╚═╝╚══════╝╚══════╝    ╚════╝ ╚═╝             ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝
```

# All-Japan-Grid

Open Japanese power grid **geographic topology** dataset built from OpenStreetMap.
10 regions, 40,000+ transmission lines, 7,000+ substations, 19,000+ power plants.

OpenStreetMap から機械的に抽出した、日本全国の送電網 **地理トポロジ** データセットです。
10 地域、送電線 40,000 本超、変電所 7,000 箇所超、発電所 19,000 箇所超。

**Live Map / ライブマップ:** https://lutelute.github.io/All-Japan-Grid/

---

## Disclaimer / 免責事項

> **English:**
> This dataset is generated **automatically by machine processing** of publicly available [OpenStreetMap](https://www.openstreetmap.org/) data. It does **not** reflect official information from any electric power company, transmission operator, or government agency. The data may contain errors, omissions, or inaccuracies inherent to crowdsourced mapping and automated extraction. **Use at your own risk.** The authors assume no liability for any damages, losses, or consequences arising from the use of this data. This dataset is provided "as is" without warranty of any kind, express or implied.

> **日本語:**
> 本データセットは、公開されている [OpenStreetMap](https://www.openstreetmap.org/) のデータを **機械的に自動処理** して生成したものです。各電力会社・送電事業者・政府機関等の公式情報を正確に反映したものでは **ありません**。クラウドソーシングによる地図データおよび自動抽出処理に起因する誤り・欠落・不正確さが含まれる可能性があります。**本データの利用は自己責任** でお願いいたします。本データの利用により生じたいかなる損害・損失・結果についても、作成者は一切の責任を負いません。本データセットは明示・黙示を問わず、いかなる種類の保証もなく「現状のまま」提供されます。

---

### Network Preview / ネットワーク プレビュー

<p align="center">
  <img src="https://raw.githubusercontent.com/lutelute/All-Japan-Grid/main/docs/assets/gif/network_ybus_tour_small.gif?v=2" alt="Network + Ybus Tour" width="100%">
</p>

> **Important / 重要:** This dataset provides the **geographic layout** of Japan's transmission infrastructure — where substations and lines are physically located and how they connect spatially. It is **not** a ready-to-use electrical model. See [Limitations](#limitations--what-this-data-is-not--本データの限界) below.
>
> 本データセットは日本の送電インフラの **地理的配置** — 変電所や送電線の物理的な位置と空間的な接続関係 — を提供するものです。そのまま使える電力系統モデルでは **ありません**。詳しくは下記 [Limitations（本データの限界）](#limitations--what-this-data-is-not--本データの限界) を参照してください。

## Dataset / データセット

| Region / 地域 | Substations / 変電所 | Lines / 送電線 | Plants / 発電所 | Frequency / 周波数 |
|--------|------------|-------|--------|-----------|
| Hokkaido / 北海道 | 303 | 1,879 | 436 | 50 Hz |
| Tohoku / 東北 | 738 | 5,112 | 1,311 | 50 Hz |
| Tokyo / 東京 | 1,367 | 8,052 | 7,207 | 50 Hz |
| Chubu / 中部 | 898 | 5,284 | 3,792 | 60 Hz |
| Hokuriku / 北陸 | 273 | 1,604 | 432 | 60 Hz |
| Kansai / 関西 | 1,016 | 5,960 | 1,518 | 60 Hz |
| Chugoku / 中国 | 548 | 3,214 | 1,173 | 60 Hz |
| Shikoku / 四国 | 258 | 1,532 | 688 | 60 Hz |
| Kyushu / 九州 | 1,145 | 6,553 | 2,549 | 60 Hz |
| Okinawa / 沖縄 | 416 | 887 | 32 | 60 Hz |

### File Format / ファイル形式

GeoJSON FeatureCollection per region / 地域ごとの GeoJSON:
```
data/{region}_substations.geojson   # Point/Polygon features（変電所）
data/{region}_lines.geojson         # LineString features（送電線）
data/{region}_plants.geojson        # Point features（発電所）
```

Key properties (substations & lines) / 主なプロパティ（変電所・送電線）:
- `voltage` — OSM voltage in volts / 電圧（ボルト単位、例: `"275000"`）
- `name` / `name:ja` — Facility name / 施設名
- `operator` — Operating utility / 運用事業者
- `cables`, `circuits` — Line specifications / 線路仕様

Key properties (plants) / 主なプロパティ（発電所）:
- `fuel_type` — Normalized: solar, hydro, coal, gas, nuclear, wind, etc. / 燃料種別
- `capacity_mw` — Output capacity in MW (when available) / 発電容量（MW）
- `plant:source` — Raw OSM source tag / OSM 原データのソースタグ
- `name` / `name:ja` — Plant name / 発電所名

### Data Source / データソース

All data is extracted from [OpenStreetMap](https://www.openstreetmap.org/) using the Overpass API.
全データは Overpass API を用いて [OpenStreetMap](https://www.openstreetmap.org/) から抽出しています。

- `power=substation` — Substations, switching stations / 変電所、開閉所
- `power=line` / `power=cable` — Transmission lines / 送電線
- `power=plant` — Power plants / 発電所

License / ライセンス: [ODbL](https://opendatacommons.org/licenses/odbl/) (OpenStreetMap)

## Interactive Map (GitHub Pages) / インタラクティブマップ

The static site at `docs/` renders all regions on a Leaflet.js dark map with voltage-based coloring.
`docs/` 以下の静的サイトで、全地域を Leaflet.js ダークマップ上に電圧別の色分けで表示します。

Voltage filter presets / 電圧フィルタ: 500 kV, 275 kV+, 154 kV+, 110 kV+, 66 kV+, All

```bash
# Local preview / ローカルプレビュー
python -m http.server -d docs 8080
open http://localhost:8080
```

## Limitations — What This Data Is NOT / 本データの限界

OSM provides the **geographic** skeleton of the transmission grid. To build a functioning electrical model (power flow, OPF, UC), the following are required but **missing** from this dataset.

OSM が提供するのは送電網の **地理的** 骨格です。実用的な電力系統モデル（潮流計算、OPF、UC）を構築するには、以下のデータが必要ですが本データセットには **含まれていません**。

| Missing / 不足データ | Why it matters / 重要な理由 | Potential source / 補完候補 |
|---------|---------------|-----------------|
| **Line impedance (R, X, B)** / 線路インピーダンス | Required for any power flow calculation / 潮流計算に必須 | Typical values by voltage class, OCCTO published parameters |
| **From/to bus connectivity** / 母線接続関係 | OSM lines are geographic traces, not bus-bus connections / OSM の線は地理的経路であり母線間接続ではない | Manual verification, OCCTO topology data |
| **Generator details** / 発電機詳細 | Lacks cost curves, min/max output, ramp rates / コストカーブ・出力範囲・ランプレート等が欠如 | OCCTO supply plan, 国土数値情報 P03, JEPX data |
| **Load / demand** / 負荷・需要 | No demand allocation at buses / 母線への需要配分なし | OCCTO area demand, prefecture-level statistics |
| **Transformer data** / 変圧器データ | No tap ratios, impedance, winding configuration / タップ比・インピーダンス・巻線構成なし | Synthetic estimation or utility disclosure |
| **Switching topology** / 開閉器トポロジ | Bus-section / breaker-level detail unavailable / 母線区分・遮断器レベルの詳細なし | Not publicly available in Japan |

### Lessons Learned / 教訓

1. **"地図があるからデータがある" は誤り** — A map showing transmission lines does not imply that the underlying electrical parameters exist. Geographic data and electrical data are fundamentally different.
2. **容量データ ≠ 系統モデル** — Knowing a line is "275 kV" tells you the voltage class but nothing about impedance, thermal rating, or actual connectivity.
3. **Endpoint matching is fragile / 端点マッチングは脆弱** — Heuristic from/to bus estimation from geographic proximity produces many mismatches. A 50 km threshold catches most connections but also creates false links.
4. **Japanese name normalization / 日本語名称の正規化** — `変電所`, `発電所`, `開閉所` have multiple orthographies (kanji/kana/abbreviation). Fuzzy matching is essential.
5. **Null diversity / Null値の多様性** — OSM features may have `voltage=null`, `voltage=""`, `voltage="yes"`, or no voltage tag at all. Robust parsing must handle all cases.
6. **Regional scope & name resolution / 地域スコープと名称解決** — The same substation name can appear in multiple regions. Name-based matching must be scoped to the correct region.
7. **AC power flow on OSM topology produces physically meaningless results / OSMトポロジでの交流潮流計算は物理的に無意味** — Without proper impedance data, generator dispatch, and demand allocation, power flow output is numerical noise, not engineering insight.

## What This Data IS Good For / 本データの活用法

- **Visualization / 可視化**: Interactive maps of Japan's transmission infrastructure by voltage class and region / 電圧階級・地域別の送電インフラ インタラクティブマップ
- **Topology research / トポロジ研究**: Graph-theoretic analysis of network connectivity, redundancy, vulnerability / ネットワーク接続性・冗長性・脆弱性のグラフ理論的分析
- **Geographic reference / 地理的参照**: Substation locations and transmission corridors for spatial analysis / 空間分析のための変電所位置・送電回廊
- **Starting point for synthetic models / 合成モデルの出発点**: Geographic skeleton to be enriched with electrical parameters from other sources / 他ソースの電気パラメータで補完可能な地理的骨格
- **Education / 教育**: Understanding the structure of Japan's 10 regional grids and the 50/60 Hz boundary / 日本の10地域系統と50/60Hz境界の構造理解

## Analysis Tools (Experimental) / 解析ツール（実験的）

The `src/` directory contains power flow and UC solver code. These tools work correctly on **complete** electrical models (e.g. MATPOWER test cases) but produce unreliable results on raw OSM topology due to the missing data described above.

`src/` ディレクトリには潮流計算および UC ソルバのコードが含まれています。これらのツールは **完備された** 電力系統モデル（例: MATPOWER テストケース）では正しく動作しますが、上述の不足データにより、生の OSM トポロジに対しては信頼できない結果を出力します。

They are included as reference implementations for future use when combined with complementary data sources.
補完データソースとの組み合わせを想定した参照実装として収録しています。

### Local Server / ローカルサーバー

```bash
pip install -r requirements.txt
uvicorn src.server.app:app --reload
open http://localhost:8000
```

### Included Tools / 収録ツール

| Module / モジュール | Purpose / 目的 | Status / 状態 |
|--------|---------|--------|
| `src/server/` | FastAPI web server, interactive map / FastAPI ウェブサーバー、インタラクティブマップ | Works / 動作可 |
| `src/powerflow/` | DC/AC power flow via pandapower / pandapower による DC/AC 潮流計算 | Requires electrical parameters / 電気パラメータが必要 |
| `src/ac_powerflow/` | Advanced AC methods / 高度な AC 手法 | Requires electrical parameters / 電気パラメータが必要 |
| `src/uc/` | Unit Commitment (MILP, PuLP + HiGHS) | Requires generators + demand / 発電機・需要データが必要 |
| `src/converter/` | pandapower / MATPOWER export / エクスポート | Works / 動作可 |

## Future Work — Complementary Data Sources / 今後の展望 — 補完データソース

To build a usable electrical model, this geographic topology needs to be combined with:
実用的な電力系統モデルを構築するには、本地理トポロジを以下のデータと組み合わせる必要があります:

| Data source / データソース | What it provides / 提供内容 | Access / アクセス |
|-------------|-----------------|--------|
| **OCCTO** (電力広域的運営推進機関) | Interconnection capacity, area demand, supply-demand plans / 連系線容量、地域需要、需給計画 | [occto.or.jp](https://www.occto.or.jp/) |
| **国土数値情報 P03** | Power plant locations, capacity, fuel type / 発電所位置、容量、燃料種別 | [nlftp.mlit.go.jp](https://nlftp.mlit.go.jp/ksj/) |
| **JEPX** (日本卸電力取引所) | Spot market prices, area price signals / スポット市場価格、エリアプライス | [jepx.jp](http://www.jepx.jp/) |
| **PyPSA-Earth / atlite** | Renewable resource data, synthetic grid enrichment / 再エネ資源データ、合成系統補完 | [pypsa-earth.readthedocs.io](https://pypsa-earth.readthedocs.io/) |
| **MATPOWER test cases** | Validated IEEE/PGLIB models for benchmarking / 検証済みベンチマークモデル | [matpower.org](https://matpower.org/) |
| **Synthetic line parameters** / 合成線路パラメータ | R/X/B estimation by voltage class and conductor type / 電圧階級・導体種別による推定値 | Literature values (e.g. Glover, Sarma & Overbye) |

Contributions and collaborations welcome. If you have access to additional data sources or are working on Japanese grid modeling, please open an issue.

コントリビューションや共同研究を歓迎します。追加のデータソースをお持ちの方、日本の系統モデリングに取り組んでいる方は、ぜひ Issue を作成してください。

## Project Structure / プロジェクト構成

```
data/                  GeoJSON network data (10 regions) / 地域別 GeoJSON
config/regions.yaml    Region metadata / 地域メタデータ（周波数、電圧、バウンディングボックス）
src/
  model/               Data models / データモデル（Substation, TransmissionLine, Generator）
  converter/           pandapower / MATPOWER conversion / 変換
  powerflow/           DC/AC power flow runner (experimental) / 潮流計算（実験的）
  ac_powerflow/        Advanced AC power flow (experimental) / 高度 AC 潮流（実験的）
  uc/                  Unit Commitment solver (experimental) / UC ソルバ（実験的）
  server/              FastAPI web server + GeoJSON loader / ウェブサーバー
  utils/               Geographic utilities / 地理ユーティリティ
examples/              Demo scripts / デモスクリプト
docs/                  GitHub Pages static site / 静的サイト
scripts/               Build tools / ビルドツール（静的サイト生成、OSM 取得）
schemas/               XML schema definitions / XML スキーマ定義
tests/                 pytest test suite / テストスイート
```

## Requirements / 必要環境

Python 3.10+

```bash
pip install -r requirements.txt
```

Key dependencies / 主な依存パッケージ: pandapower, fastapi, pulp, highspy, pyyaml, geopandas

## License / ライセンス

- Network data / ネットワークデータ: [ODbL](https://opendatacommons.org/licenses/odbl/) (OpenStreetMap)
- Code / コード: MIT
