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

## Other Scripts

| Script | Description |
|--------|-------------|
| `benchmark_alljapan.py` | 全国ベンチマーク実行 |
| `build_alljapan_full.py` | 全国系統モデル構築 |
| `build_alljapan_matpower.py` | MATPOWER形式エクスポート |
| `build_load_timeseries.py` | 負荷時系列データ構築 |
| `fetch_subdivided.py` | OSMデータ取得 (分割) |
| `interactive_grid_maps.py` | インタラクティブ地図生成 |
| `visualize_osm_network.py` | OSMネットワーク可視化 |
