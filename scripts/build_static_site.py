#!/usr/bin/env python3
"""Build lightweight static GeoJSON files for GitHub Pages.

Reads the full OSM GeoJSON from data/, filters by voltage tier,
simplifies geometry, and writes compact JSON into docs/data/.

Voltage tiers:
  - 275kv: lines and substations >= 275 kV  (~1.5 MB lines, ~30 KB subs)
  - 154kv: lines and substations >= 154 kV  (~3.8 MB lines, ~75 KB subs)
  - all:   all voltages                      (~12.5 MB lines, ~500 KB subs)

Also generates docs/data/regions.json with region metadata.

Dependencies: pyyaml only (no pandapower needed).
"""

import json
import os
import sys

# Add project root to path so we can import geojson_loader
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.server.geojson_loader import get_all_geojson_light, get_regions_summary, load_all

DOCS_DATA_DIR = os.path.join(PROJECT_ROOT, "docs", "data")

# Voltage tiers: (suffix, min_voltage_kv)
VOLTAGE_TIERS = [
    ("275kv", 275),
    ("154kv", 154),
    ("all", 0),
]


def write_json(path: str, data: dict) -> int:
    """Write compact JSON and return file size in bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    size = os.path.getsize(path)
    return size


def fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.1f} MB"


def main():
    print("Loading OSM GeoJSON data...")
    load_all()

    print(f"Output directory: {DOCS_DATA_DIR}")
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    # Generate regions.json
    regions = get_regions_summary()
    path = os.path.join(DOCS_DATA_DIR, "regions.json")
    size = write_json(path, regions)
    print(f"  regions.json  ({fmt_size(size)}, {len(regions)} regions)")

    # Generate voltage-tiered GeoJSON files
    for suffix, min_kv in VOLTAGE_TIERS:
        for layer in ("lines", "substations"):
            short_layer = "lines" if layer == "lines" else "subs"
            filename = f"{short_layer}_{suffix}.geojson"
            data = get_all_geojson_light(layer, min_voltage_kv=min_kv)
            count = len(data.get("features", []))
            path = os.path.join(DOCS_DATA_DIR, filename)
            size = write_json(path, data)
            kv_label = f">= {min_kv} kV" if min_kv > 0 else "all"
            print(f"  {filename:<25s} ({fmt_size(size):>8s}, {count:>5d} features, {kv_label})")

    print("Done.")


if __name__ == "__main__":
    main()
