#!/usr/bin/env python3
"""Fix plant capacity values: convert W -> MW for unitless OSM values.

The original fetch_plants.py treated unitless capacity values < 10000
as MW, but OSM convention stores plant:output:electricity in watts.
This script retroactively fixes all cached data/{region}_plants.geojson
and regenerates docs/data/plants_*.geojson.

Usage:
    python scripts/fix_plant_capacity.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

# Reasonable MW thresholds for sanity checking
# Largest plant in Japan: Kashiwazaki-Kariwa Nuclear ~8,212 MW
MAX_REASONABLE_MW = 10000


def fix_capacity(val):
    """Fix a capacity_mw value that was incorrectly parsed.

    Values > MAX_REASONABLE_MW are likely in kW (not MW).
    Values > 1,000,000 are likely in W.
    """
    if val is None:
        return None
    if val < 0:
        return None
    if val > 1_000_000:
        # Was in W, should be MW
        return round(val / 1_000_000, 2)
    if val > MAX_REASONABLE_MW:
        # Was in kW, should be MW
        return round(val / 1000, 2)
    return val


def main():
    total_fixed = 0
    total_features = 0

    for region in REGIONS:
        path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fixed = 0
        for feat in data.get("features", []):
            total_features += 1
            cap = feat["properties"].get("capacity_mw")
            if cap is not None and cap > MAX_REASONABLE_MW:
                new_cap = fix_capacity(cap)
                feat["properties"]["capacity_mw"] = new_cap
                fixed += 1

        if fixed > 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            print(f"  {region}: fixed {fixed} / {len(data['features'])} plants")
            total_fixed += fixed
        else:
            print(f"  {region}: OK (no fixes needed)")

    print(f"\nTotal: fixed {total_fixed} / {total_features} plants")
    print("\nNow rebuild static site:")
    print("  python scripts/build_static_site.py")


if __name__ == "__main__":
    main()
