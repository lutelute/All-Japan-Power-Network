# Scripts

データ処理・エクスポート用スクリプト集。

## Generator Data

### export_generators_geojson.py

国土数値情報 P03 (発電所) データを解析し、`data/reference/generator_defaults.yaml` のデフォルトパラメータとマージして GeoJSON を出力する。

```bash
# 全発電機をエクスポート
python scripts/export_generators_geojson.py

# 10MW以上のみ (GitHub Pages 用 — 推奨)
python scripts/export_generators_geojson.py --min-mw 10

# 出力先を指定
python scripts/export_generators_geojson.py --min-mw 10 --output docs/data/generators.geojson
```

**前提条件:**
- `data/generators/P03/P03-13/GML/P03-13-g.xml` が必要
  - 未ダウンロードの場合: `python -c "from src.parser.generator_parser import GeneratorParser; GeneratorParser().download_p03_data()"`
- `data/reference/generator_defaults.yaml` が必要

**出力:**
- `docs/data/generators.geojson` — GeoJSON FeatureCollection
  - 各 Feature に燃料種別のデフォルトパラメータが結合済み
  - GitHub Pages の地図で発電機マーカー + ポップアップとして表示される

## Substation Data

### export_substations_geojson.py

OSM変電所データを全属性付きでエンリッチし、不明な電圧を近接送電線・変電所タイプから推定して GeoJSON を出力する。

```bash
# 全変電所をエクスポート
python scripts/export_substations_geojson.py

# 鉄道用変電所を除外
python scripts/export_substations_geojson.py --exclude-traction
```

**前提条件:**
- `data/osm/{region}_substations.geojson` と `data/osm/{region}_lines.geojson` が必要

**出力:**
- `docs/data/substations.geojson` — 全属性付き GeoJSON (6,962件)

**電圧推定ロジック:**
1. OSM `voltage` タグから取得 (57%)
2. 近接送電線 (1km以内) の電圧から推定 (23%)
3. 変電所タイプから推定: distribution→66kV, traction→25kV, etc. (6%)
4. 推定不能 (14%)

## Data Enrichment Pipeline / エンリッチメント パイプライン

GeoJSON の欠落属性（名称・事業者・燃料種別）を外部データソースで自動補完する。

### Pipeline Scripts (実行順序が重要)

| # | Script | Description | API | Cache |
|---|--------|-------------|-----|-------|
| 1 | `audit_data_quality.py` | プレースホルダ監査（Before/After比較用） | - | - |
| 2 | `enrich_substations_geocode.py` | 変電所名称: Nominatim逆ジオコーディング → `{area}変電所` | Nominatim | - |
| 3 | `enrich_plants_p03.py` | 発電所属性: P03国土数値情報とのマッチング + 事業者名正規化 | - | - |
| 4 | `enrich_overpass_tags.py` | 発電所属性: OSM IDでOverpass APIバッチ取得 | Overpass | `data/cache/overpass_tags.json` |
| 5 | `enrich_plants_geocode.py` | 発電所名称: Nominatim逆ジオコーディング → `{area}発電所` | Nominatim | `data/cache/plants_geocode.json` |
| 6 | `enrich_lines_endpoints.py` | 送電線名称: 端点変電所マッチング → `{from}~{to}線` | - | - |
| 7 | `enrich_all.py` | 上記を正しい順序で一括実行するオーケストレーター | - | - |

### Usage / 使い方

```bash
# 全地域・全ステップ
python scripts/enrich_all.py

# 特定地域のみ
python scripts/enrich_all.py --region okinawa

# ドライラン（実行計画のみ表示）
python scripts/enrich_all.py --dry-run

# 個別ステップ
python scripts/enrich_substations_geocode.py --promote-names --region hokuriku
python scripts/enrich_overpass_tags.py --region tokyo
python scripts/enrich_plants_geocode.py --region kyushu
python scripts/enrich_lines_endpoints.py --region chubu

# 品質監査
python scripts/audit_data_quality.py                  # 全地域
python scripts/audit_data_quality.py --region okinawa  # 特定地域
```

### Dependencies / 依存関係

```
Step 2 (substations)  ← 独立
Step 3 (P03)          ← 独立（P03 GMLが必要）
Step 4 (Overpass)     ← 独立
Step 5 (plants geo)   ← Step 3, 4の後（未解決分のみ対象）
Step 6 (lines)        ← Step 2の後（変電所名が必要）
```

### Rate Limits / API制限

- **Nominatim**: 1.1秒/リクエスト（Usage Policy準拠）。全発電所 ~16,000件で約5時間
- **Overpass**: 100 IDs/バッチ、10秒間隔、HTTP 429/504で指数バックオフ

## Other Scripts

| Script | Description |
|--------|-------------|
| `benchmark_alljapan.py` | 全国ベンチマーク実行 |
| `build_alljapan_full.py` | 全国系統モデル構築 |
| `build_alljapan_matpower.py` | MATPOWER形式エクスポート |
| `build_load_timeseries.py` | 負荷時系列データ構築 |
| `build_static_site.py` | GitHub Pages用静的GeoJSON生成 |
| `fetch_subdivided.py` | OSMデータ取得 (分割) |
| `interactive_grid_maps.py` | インタラクティブ地図生成 |
| `visualize_osm_network.py` | OSMネットワーク可視化 |
