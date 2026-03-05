# Reference Data (リファレンスデータ)

系統モデル構築に使うマスターデータ群。外部ソース・文献値をもとに手動で整備する。

## Files

| File | Description | Source |
|------|-------------|--------|
| `generator_defaults.yaml` | 燃料種別ごとの発電機デフォルトパラメータ (ramp rate, startup/shutdown, costs, emissions, etc.) | OCCTO 供給計画、発電コスト検証委員会 (2021)、IEEE Japan |
| `interconnections.yaml` | 地域間連系線の定義と潮流容量 | OCCTO 公開データ |
| `load_profiles.yaml` | 地域別・時間帯別の負荷プロファイル | 各電力会社公開データ |
| `voltage_hierarchy.yaml` | 地域別の電圧階級体系 | OCCTO 系統マスタ |

## generator_defaults.yaml の構造

燃料種別 (`coal`, `lng`, `oil`, `nuclear`, `hydro`, `pumped_hydro`, `geothermal`, `wind`, `solar`, `biomass`, `mixed`, `unknown`) ごとに以下のパラメータを定義:

### Basic
- `name_ja` / `name_en` — 燃料種別名
- `category` — thermal / nuclear / renewable / storage
- `dispatchable` — 出力制御可能かどうか

### Capacity & Ramp
- `typical_unit_mw` — 典型的なユニット出力 (MW)
- `p_min_fraction` — 最低出力比率 (対定格)
- `ramp_up_fraction` / `ramp_down_fraction` — 変化速度 (対定格/h)

### Timing
- `min_up_time_h` / `min_down_time_h` — 最小起動/停止維持時間 (h)
- `startup_time_h` — コールドスタートから同期まで (h)
- `shutdown_time_h` — 解列から完全停止まで (h)

### Costs (JPY)
- `startup_cost_per_mw` / `shutdown_cost_per_mw` — 起動/停止費用 (JPY/MW)
- `fuel_cost_per_mwh` — 燃料費 (JPY/MWh)
- `no_load_cost_per_mw` — 空転費用 (JPY/MW/h)
- `labor_cost_per_h` — 人件費 (JPY/h)

### Efficiency & Emissions
- `heat_rate_kj_per_kwh` — 熱効率 (kJ/kWh)
- `co2_intensity_kg_per_mwh` — CO2排出原単位 (kg/MWh)

### Reliability & Lifecycle
- `planned_outage_rate` / `forced_outage_rate` — 計画/事故停止率
- `capacity_factor` — 年間設備利用率
- `typical_lifetime_years` — 標準耐用年数
- `typical_construction_years` — 建設期間

## 更新手順

1. `generator_defaults.yaml` を直接編集する
2. 変更後に GeoJSON を再エクスポート:
   ```bash
   python scripts/export_generators_geojson.py --min-mw 10
   ```
3. 結果は `docs/data/generators.geojson` に出力される
4. コミット & push で GitHub Pages に反映

## データソース

- [OCCTO 供給計画](https://www.occto.or.jp/kyokyukeikaku/)
- [発電コスト検証委員会 報告書 (2021)](https://www.enecho.meti.go.jp/committee/council/basic_policy_subcommittee/mitoshi/cost_wg/)
- [国土数値情報 P03 発電所データ](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P03.html)
