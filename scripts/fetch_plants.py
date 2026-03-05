#!/usr/bin/env python3
"""Fetch power plant (power=plant) data from OpenStreetMap via Overpass API.

Fetches only power=plant (facility-level), NOT power=generator (individual
units like solar panels). This keeps data manageable (~1-2K plants for all
Japan vs 200K+ generators).

Output: data/{region}_plants.geojson

Usage:
    python scripts/fetch_plants.py               # all regions
    python scripts/fetch_plants.py --region tokyo # single region
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "regions.yaml")
OUTPUT_DIR = os.path.join(ROOT, "data")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TIMEOUT = 120
PAUSE = 5  # seconds between regions

# OSM plant:source -> normalized fuel type
FUEL_TYPE_MAP = {
    "coal": "coal", "gas": "gas", "natural_gas": "gas", "oil": "oil",
    "nuclear": "nuclear", "hydro": "hydro", "water": "hydro",
    "wind": "wind", "solar": "solar", "photovoltaic": "solar",
    "biomass": "biomass", "biogas": "biomass", "waste": "waste",
    "geothermal": "geothermal", "tidal": "tidal", "wave": "tidal",
    "battery": "battery", "pumped_storage": "pumped_hydro",
    "diesel": "oil",
}

# Fuel type display colors for the map
FUEL_COLORS = {
    "nuclear": "#ff0000", "coal": "#444444", "gas": "#ff8800",
    "oil": "#884400", "hydro": "#0088ff", "pumped_hydro": "#0044aa",
    "wind": "#00cc88", "solar": "#ffdd00", "geothermal": "#cc4488",
    "biomass": "#668833", "waste": "#996633", "tidal": "#006688",
    "battery": "#aa00ff", "unknown": "#999999",
}


def _log(msg):
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


def load_regions():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("regions", {})


def normalize_fuel(source_str):
    """Normalize plant:source to fuel type."""
    if not source_str:
        return "unknown"
    primary = source_str.split(";")[0].strip().lower()
    return FUEL_TYPE_MAP.get(primary, primary)


def parse_capacity_mw(val):
    """Parse capacity string to MW float.

    OSM convention: plant:output:electricity is in watts (W) when
    no unit suffix is present.  Explicit suffixes (MW, kW, GW) are
    honoured directly.
    """
    if not val:
        return None
    s = str(val).strip().lower().replace(",", "").replace(" ", "")
    try:
        if s.endswith("gw"):
            return round(float(s[:-2]) * 1000, 1)
        elif s.endswith("mw"):
            return round(float(s[:-2]), 1)
        elif s.endswith("kw"):
            return round(float(s[:-2]) / 1000, 2)
        elif s.endswith("w"):
            return round(float(s[:-1]) / 1_000_000, 3)
        else:
            # No unit → assume watts (OSM convention)
            v = float(s)
            return round(v / 1_000_000, 2) if v >= 1000 else round(v, 1)
    except (ValueError, TypeError):
        return None


def fetch_plants_for_bbox(bbox):
    """Fetch power=plant from Overpass API for a bounding box."""
    south, west, north, east = bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_max"]

    query = f"""
    [out:json][timeout:{TIMEOUT}];
    (
      nwr["power"="plant"]({south},{west},{north},{east});
    );
    out center tags;
    """

    for attempt in range(3):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=TIMEOUT + 30)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as e:
            _log(f"  Attempt {attempt+1}/3 failed: {str(e)[:80]}")
            if attempt < 2:
                time.sleep(PAUSE * (attempt + 1))
    return None


def elements_to_geojson(elements, region):
    """Convert Overpass elements to GeoJSON FeatureCollection."""
    features = []
    for el in elements:
        # Get coordinates
        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
        elif "center" in el:
            lon, lat = el["center"].get("lon"), el["center"].get("lat")
        else:
            continue

        if lon is None or lat is None:
            continue

        tags = el.get("tags", {})
        source = tags.get("plant:source", tags.get("generator:source", tags.get("source", "")))
        fuel = normalize_fuel(source)
        capacity = parse_capacity_mw(
            tags.get("plant:output:electricity",
                      tags.get("generator:output:electricity", ""))
        )
        name = tags.get("name", tags.get("name:ja", tags.get("name:en", "")))

        props = {
            "name": name,
            "name:ja": tags.get("name:ja", ""),
            "name:en": tags.get("name:en", ""),
            "operator": tags.get("operator", ""),
            "fuel_type": fuel,
            "plant:source": source,
            "capacity_mw": capacity,
            "voltage": tags.get("voltage", ""),
            "_region": region,
            "_display_name": name or f"plant_{el['id']}",
            "_fuel_color": FUEL_COLORS.get(fuel, FUEL_COLORS["unknown"]),
            "osm_id": el["id"],
            "osm_type": el["type"],
        }

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 6), round(lat, 6)],
            },
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def fetch_region(region, cfg):
    """Fetch plants for a single region."""
    out_path = os.path.join(OUTPUT_DIR, f"{region}_plants.geojson")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            fc = json.load(f)
        n = len(fc.get("features", []))
        _log(f"  {region}: SKIP (cached, {n} plants)")
        return True

    bbox = cfg.get("bounding_box")
    if not bbox:
        _log(f"  {region}: SKIP (no bounding_box)")
        return False

    _log(f"  {region}: fetching...")
    elements = fetch_plants_for_bbox(bbox)
    if elements is None:
        _log(f"  {region}: FAILED")
        return False

    fc = elements_to_geojson(elements, region)
    n = len(fc["features"])
    _log(f"  {region}: {n} plants")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    _log(f"  SAVED: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch power plants from OSM")
    parser.add_argument("--region", help="Single region (default: all)")
    parser.add_argument("--force", action="store_true", help="Overwrite cached files")
    args = parser.parse_args()

    regions = load_regions()

    if args.region:
        if args.region not in regions:
            _log(f"ERROR: '{args.region}' not found")
            sys.exit(1)
        targets = {args.region: regions[args.region]}
    else:
        targets = regions

    if args.force:
        for r in targets:
            p = os.path.join(OUTPUT_DIR, f"{r}_plants.geojson")
            if os.path.exists(p):
                os.remove(p)

    _log(f"\nFetching power=plant for {len(targets)} region(s)...\n")
    start = time.time()
    results = {}

    for region, cfg in targets.items():
        ok = fetch_region(region, cfg)
        results[region] = ok
        time.sleep(PAUSE)

    elapsed = time.time() - start
    _log(f"\nDone in {elapsed:.0f}s")

    # Summary
    total = 0
    for region, ok in results.items():
        p = os.path.join(OUTPUT_DIR, f"{region}_plants.geojson")
        if os.path.exists(p):
            with open(p, "r") as f:
                fc = json.load(f)
            n = len(fc.get("features", []))
            total += n

    _log(f"Total: {total} plants across {len(results)} regions")


if __name__ == "__main__":
    main()
