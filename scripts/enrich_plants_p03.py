#!/usr/bin/env python3
"""Enrich OSM power plant data with 国土数値情報 P03 (発電施設).

Parses the P03 GML file, builds a spatial index, and matches P03 plants
to OSM plants by proximity. Fills in missing name, capacity, and operator.

Usage:
    python scripts/enrich_plants_p03.py
    python scripts/enrich_plants_p03.py --max-distance 1.0  # km threshold
"""

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P03_GML = os.path.join(ROOT, "data", "external", "P03-13", "GML", "P03-13-g.xml")
DATA_DIR = os.path.join(ROOT, "data")

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

NS = {
    "ksj": "http://nlftp.mlit.go.jp/ksj/schemas/ksj-app",
    "gml": "http://www.opengis.net/gml/3.2",
    "xlink": "http://www.w3.org/1999/xlink",
}

# P03 element type -> fuel_type
P03_FUEL_MAP = {
    "ThermalPowerPlant": "thermal",  # further classified by burningType
    "GeneralHydroelectricPowerPlant": "hydro",
    "PumpedStorageHydroelectricPowerPlant": "pumped_hydro",
    "NuclearPowerPlant": "nuclear",
    "GeothermalPowerPlant": "geothermal",
    "WindPowerPlant": "wind",
    "PhotovoltaicPowerPlant": "solar",
    "BiomassPowerStation": "biomass",
}

# Thermal sub-types (burningType or fuel)
THERMAL_FUEL_MAP = {
    "石炭": "coal", "石炭専焼": "coal", "専焼": "coal",
    "LNG": "gas", "ガス": "gas", "天然ガス": "gas",
    "石油": "oil", "重油": "oil", "軽油": "oil",
    "混焼": "mixed", "複合": "mixed",
}

# Operator name normalization (variants → canonical)
# Copied from scripts/export_substations_geojson.py for consistency
OPERATOR_NORMALIZE = {
    "東京電力": "東京電力パワーグリッド",
    "東京電力PG": "東京電力パワーグリッド",
    "関西電力": "関西電力送配電",
    "中部電力": "中部電力パワーグリッド",
    "九州電力": "九州電力送配電",
    "東北電力": "東北電力ネットワーク",
    "北海道電力": "北海道電力ネットワーク",
    "中国電力": "中国電力ネットワーク",
    "四国電力": "四国電力送配電",
    "北陸電力": "北陸電力送配電",
    "沖縄電力": "沖縄電力",
}


def normalize_operator(operator_raw):
    """Normalize operator name to canonical form."""
    if not operator_raw:
        return ""
    op = str(operator_raw).strip()
    # Check direct normalization
    if op in OPERATOR_NORMALIZE:
        return OPERATOR_NORMALIZE[op]
    # Check prefix match
    for prefix, canonical in OPERATOR_NORMALIZE.items():
        if op.startswith(prefix):
            return canonical
    return op


def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def parse_p03(gml_path):
    """Parse P03 GML into list of plant dicts."""
    print(f"  Parsing {gml_path}...")
    tree = ET.parse(gml_path)
    root = tree.getroot()

    # Build point lookup: id -> (lat, lon)
    points = {}
    for pt in root.findall("gml:Point", NS):
        pid = pt.get("{http://www.opengis.net/gml/3.2}id")
        pos = pt.findtext("gml:pos", namespaces=NS)
        if pid and pos:
            parts = pos.strip().split()
            if len(parts) == 2:
                points[pid] = (float(parts[0]), float(parts[1]))

    plants = []
    for elem_type, fuel_type in P03_FUEL_MAP.items():
        for el in root.findall(f"ksj:{elem_type}", NS):
            pos_el = el.find("ksj:position", NS)
            if pos_el is None:
                continue
            href = pos_el.get("{http://www.w3.org/1999/xlink}href", "")
            pid = href.lstrip("#")
            if pid not in points:
                continue

            lat, lon = points[pid]
            name = el.findtext("ksj:nameOfPlant", namespaces=NS) or ""
            owner = el.findtext("ksj:nameOfOwner", namespaces=NS) or ""
            address = el.findtext("ksj:address", namespaces=NS) or ""
            cap_str = el.findtext("ksj:generatingPower", namespaces=NS) or ""
            burning = el.findtext("ksj:burningType", namespaces=NS) or ""

            # Parse capacity (MW)
            capacity = None
            if cap_str:
                try:
                    capacity = float(cap_str)
                except (ValueError, TypeError):
                    pass

            # Refine thermal fuel type
            actual_fuel = fuel_type
            if fuel_type == "thermal":
                actual_fuel = "gas"  # default for thermal
                for key, val in THERMAL_FUEL_MAP.items():
                    if key in burning or key in name:
                        actual_fuel = val
                        break
                # Check name for clues
                if "石炭" in name or "Coal" in name:
                    actual_fuel = "coal"
                elif "LNG" in name or "ガス" in name:
                    actual_fuel = "gas"
                elif "石油" in name or "重油" in name:
                    actual_fuel = "oil"

            plants.append({
                "name": name.strip(),
                "operator": normalize_operator(owner.strip()),
                "address": address.strip(),
                "capacity_mw": capacity,
                "fuel_type": actual_fuel,
                "lat": lat,
                "lon": lon,
                "p03_id": el.get("{http://www.opengis.net/gml/3.2}id"),
            })

    return plants


def match_and_enrich(osm_path, p03_plants, max_dist_km):
    """Match P03 plants to OSM plants by proximity and enrich."""
    with open(osm_path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = fc.get("features", [])
    if not features:
        return 0, 0

    enriched = 0
    matched = 0

    for feat in features:
        coords = feat["geometry"]["coordinates"]
        osm_lon, osm_lat = coords[0], coords[1]
        props = feat["properties"]

        # Find nearest P03 plant
        best_dist = float("inf")
        best_p03 = None
        for p03 in p03_plants:
            d = haversine(osm_lat, osm_lon, p03["lat"], p03["lon"])
            if d < best_dist:
                best_dist = d
                best_p03 = p03

        if best_p03 is None or best_dist > max_dist_km:
            continue

        matched += 1
        changed = False

        # Enrich missing fields
        if not props.get("name", "").strip() and best_p03["name"]:
            props["name"] = best_p03["name"]
            props["_display_name"] = best_p03["name"]
            changed = True

        if props.get("capacity_mw") is None and best_p03["capacity_mw"] is not None:
            props["capacity_mw"] = best_p03["capacity_mw"]
            changed = True

        if not props.get("operator", "").strip() and best_p03["operator"]:
            props["operator"] = normalize_operator(best_p03["operator"])
            changed = True

        fuel = props.get("fuel_type", "unknown")
        if (fuel == "unknown" or not fuel) and best_p03["fuel_type"]:
            props["fuel_type"] = best_p03["fuel_type"]
            changed = True

        if changed:
            props["_enriched_by"] = "p03"
            props["_p03_distance_km"] = round(best_dist, 3)
            enriched += 1

    # Write back
    with open(osm_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    return matched, enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich OSM plants with P03 data")
    parser.add_argument("--max-distance", type=float, default=1.0,
                        help="Max match distance in km (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(P03_GML):
        print(f"  P03 GML not found at {P03_GML}")
        print("  Download instructions:")
        print("    curl -o data/external/P03-13.zip https://nlftp.mlit.go.jp/ksj/gml/data/P03/P03-13/P03-13.zip")
        print("    cd data/external && unzip P03-13.zip")
        print("  Skipping P03 enrichment.")
        return

    p03_plants = parse_p03(P03_GML)
    print(f"  P03 total: {len(p03_plants)} plants")

    # Show P03 breakdown
    from collections import Counter
    fuel_counts = Counter(p["fuel_type"] for p in p03_plants)
    for fuel, count in fuel_counts.most_common():
        print(f"    {fuel}: {count}")

    print(f"\n  Matching with max distance = {args.max_distance} km\n")

    total_matched = 0
    total_enriched = 0

    for region in REGIONS:
        osm_path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
        if not os.path.exists(osm_path):
            continue

        matched, enriched = match_and_enrich(osm_path, p03_plants, args.max_distance)
        total_matched += matched
        total_enriched += enriched
        print(f"  {region:10s}: {matched:4d} matched, {enriched:4d} enriched")

    print(f"\n  TOTAL: {total_matched} matched, {total_enriched} enriched (of {len(p03_plants)} P03 plants)")


if __name__ == "__main__":
    main()
