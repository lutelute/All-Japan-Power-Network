# All-Japan-Power-Network

Open Japanese power grid dataset built from OpenStreetMap.
10 regions, 40,000+ transmission lines, 7,000+ substations.

**Live Map:** https://lutelute.github.io/All-Japan-Power-Network/

## Dataset

| Region | Substations | Lines | Frequency |
|--------|------------|-------|-----------|
| Hokkaido | 303 | 1,879 | 50 Hz |
| Tohoku | 738 | 5,112 | 50 Hz |
| Tokyo | 1,367 | 8,052 | 50 Hz |
| Chubu | 898 | 5,284 | 60 Hz |
| Hokuriku | 273 | 1,604 | 60 Hz |
| Kansai | 1,016 | 5,960 | 60 Hz |
| Chugoku | 548 | 3,214 | 60 Hz |
| Shikoku | 258 | 1,532 | 60 Hz |
| Kyushu | 1,145 | 6,553 | 60 Hz |
| Okinawa | 416 | 887 | 60 Hz |

### File Format

GeoJSON FeatureCollection per region:
```
data/{region}_substations.geojson   # Point/Polygon features
data/{region}_lines.geojson         # LineString features
```

Key properties:
- `voltage` — OSM voltage in volts (e.g. `"275000"`)
- `name` / `name:ja` — Facility name
- `operator` — Operating utility
- `cables`, `circuits` — Line specifications

### Data Source

All network data is extracted from [OpenStreetMap](https://www.openstreetmap.org/) using the Overpass API:
- `power=substation`
- `power=line` / `power=cable`

License: [ODbL](https://opendatacommons.org/licenses/odbl/) (OpenStreetMap)

## Interactive Map (GitHub Pages)

The static site at `docs/` renders all regions on a Leaflet.js dark map with voltage-based coloring.

Voltage filter presets: 500 kV, 275 kV+, 154 kV+, 110 kV+, 66 kV+, All

```bash
# Local preview
python -m http.server -d docs 8080
open http://localhost:8080
```

## Power Flow Analysis

DC/AC power flow via pandapower. Requires the local FastAPI server:

```bash
pip install -r requirements.txt
uvicorn src.server.app:app --reload
open http://localhost:8000
```

Features:
- Region selection + voltage filtering
- DC / AC power flow computation
- Line loading visualization (color-coded)
- MATPOWER `.mat` export

### Example: Run power flow on all regions

```bash
python examples/run_powerflow_all.py
```

## Unit Commitment (UC)

MILP-based day-ahead unit commitment solver using PuLP + HiGHS:

- Demand balance, min up/down time, ramp rate constraints
- Pumped hydro & battery storage with SOC tracking
- Regional decomposition for national-scale problems
- XML/CSV result export

### Example: UC demo

```bash
python examples/uc_demo_visualize.py
```

## Project Structure

```
data/                  GeoJSON network data (10 regions)
config/regions.yaml    Region metadata (frequency, voltage levels, bounding boxes)
src/
  model/               Data models (Substation, TransmissionLine, Generator)
  converter/           pandapower / MATPOWER conversion
  powerflow/           DC/AC power flow runner
  ac_powerflow/        Advanced AC power flow methods
  uc/                  Unit Commitment solver
  server/              FastAPI web server + GeoJSON loader
  utils/               Geographic utilities
examples/              Demo scripts (power flow, UC, visualization)
docs/                  GitHub Pages static site
scripts/               Build tools (static site generator, OSM fetch)
schemas/               XML schema definitions
tests/                 pytest test suite
```

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

Key dependencies: pandapower, fastapi, pulp, highspy, pyyaml, geopandas

## License

- Network data: [ODbL](https://opendatacommons.org/licenses/odbl/) (OpenStreetMap)
- Code: MIT
