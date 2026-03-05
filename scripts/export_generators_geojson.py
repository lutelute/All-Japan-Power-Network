#!/usr/bin/env python3
"""Export generator data from P03 GML to GeoJSON with enriched attributes.

Parses the 国土数値情報 P03 dataset, merges fuel-type-based default
parameters from data/reference/generator_defaults.yaml, and writes
a GeoJSON file suitable for the GitHub Pages map.

Usage:
    python scripts/export_generators_geojson.py

Output:
    docs/data/generators.geojson

Options:
    --min-mw FLOAT   Minimum capacity filter (default: 0, i.e. all)
    --output PATH    Output file path (default: docs/data/generators.geojson)
"""

import argparse
import json
import os
import sys

import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.generator_parser import GeneratorParser

DEFAULTS_PATH = "data/reference/generator_defaults.yaml"
OUTPUT_PATH = "docs/data/generators.geojson"

# Category -> marker color mapping (for frontend)
CATEGORY_COLORS = {
    "thermal": "#e94560",
    "nuclear": "#ff6b00",
    "renewable": "#2ecc71",
    "storage": "#3498db",
}

FUEL_TYPE_ICONS = {
    "coal": "coal",
    "lng": "gas",
    "oil": "oil",
    "nuclear": "nuclear",
    "hydro": "hydro",
    "pumped_hydro": "pumped_hydro",
    "geothermal": "geothermal",
    "wind": "wind",
    "solar": "solar",
    "biomass": "biomass",
    "mixed": "mixed",
    "unknown": "unknown",
}


def load_defaults():
    """Load fuel-type default parameters."""
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("fuel_types", {})


def generator_to_feature(gen, defaults):
    """Convert a Generator to a GeoJSON Feature with enriched properties."""
    fuel_defaults = defaults.get(gen.fuel_type, defaults.get("unknown", {}))

    # Compute actual values using defaults
    capacity = gen.capacity_mw
    ramp_up = capacity * fuel_defaults.get("ramp_up_fraction", 0.05)
    ramp_down = capacity * fuel_defaults.get("ramp_down_fraction", 0.05)
    p_min = capacity * fuel_defaults.get("p_min_fraction", 0.30)
    startup_cost = capacity * fuel_defaults.get("startup_cost_per_mw", 10)
    shutdown_cost = capacity * fuel_defaults.get("shutdown_cost_per_mw", 3)

    category = fuel_defaults.get("category", "unknown")

    properties = {
        # Basic info
        "id": gen.id,
        "name": gen.name,
        "operator": gen.operator,
        "region": gen.region,
        "fuel_type": gen.fuel_type,
        "fuel_type_ja": fuel_defaults.get("name_ja", ""),
        "fuel_type_en": fuel_defaults.get("name_en", ""),
        "category": category,
        "dispatchable": fuel_defaults.get("dispatchable", True),
        # Capacity
        "capacity_mw": round(capacity, 2),
        "p_min_mw": round(p_min, 2),
        # Ramp rates
        "ramp_up_mw_per_h": round(ramp_up, 2),
        "ramp_down_mw_per_h": round(ramp_down, 2),
        # Timing
        "min_up_time_h": fuel_defaults.get("min_up_time_h", 1),
        "min_down_time_h": fuel_defaults.get("min_down_time_h", 1),
        "startup_time_h": fuel_defaults.get("startup_time_h", 0),
        "shutdown_time_h": fuel_defaults.get("shutdown_time_h", 0),
        # Costs (JPY)
        "startup_cost_jpy": round(startup_cost, 0),
        "shutdown_cost_jpy": round(shutdown_cost, 0),
        "fuel_cost_per_mwh_jpy": fuel_defaults.get("fuel_cost_per_mwh", 0),
        # Efficiency & emissions
        "heat_rate_kj_per_kwh": fuel_defaults.get("heat_rate_kj_per_kwh", 0),
        "co2_intensity_kg_per_mwh": fuel_defaults.get(
            "co2_intensity_kg_per_mwh", 0
        ),
        # Reliability
        "planned_outage_rate": fuel_defaults.get("planned_outage_rate", 0),
        "forced_outage_rate": fuel_defaults.get("forced_outage_rate", 0),
        "capacity_factor": fuel_defaults.get("capacity_factor", 0),
        # Lifecycle
        "typical_lifetime_years": fuel_defaults.get(
            "typical_lifetime_years", 0
        ),
        "typical_construction_years": fuel_defaults.get(
            "typical_construction_years", 0
        ),
        # Display
        "_color": CATEGORY_COLORS.get(category, "#999999"),
        "_icon": FUEL_TYPE_ICONS.get(gen.fuel_type, "unknown"),
    }

    # Skip generators without valid coordinates
    if gen.latitude == 0.0 and gen.longitude == 0.0:
        return None

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [
                round(gen.longitude, 6),
                round(gen.latitude, 6),
            ],
        },
        "properties": properties,
    }


def main():
    parser = argparse.ArgumentParser(description="Export generators to GeoJSON")
    parser.add_argument(
        "--min-mw",
        type=float,
        default=0,
        help="Minimum capacity in MW (default: 0 = all)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    print("Loading fuel-type defaults...")
    defaults = load_defaults()
    print(f"  Loaded {len(defaults)} fuel types")

    print("Parsing P03 GML data...")
    gen_parser = GeneratorParser()
    generators = gen_parser.parse_gml(
        "data/generators/P03/P03-13/GML/P03-13-g.xml"
    )
    print(f"  Parsed {len(generators)} generators")

    if args.min_mw > 0:
        generators = [g for g in generators if g.capacity_mw >= args.min_mw]
        print(f"  After filter (>= {args.min_mw} MW): {len(generators)}")

    print("Converting to GeoJSON features...")
    features = []
    skipped = 0
    for gen in generators:
        feat = generator_to_feature(gen, defaults)
        if feat is None:
            skipped += 1
            continue
        features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}

    # Summary stats for metadata
    from collections import Counter

    fuel_counts = Counter(f["properties"]["fuel_type"] for f in features)
    region_counts = Counter(f["properties"]["region"] for f in features)
    total_mw = sum(f["properties"]["capacity_mw"] for f in features)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nExported {len(features)} generators ({skipped} skipped)")
    print(f"  Total capacity: {total_mw:,.0f} MW")
    print(f"  Output: {args.output} ({size_mb:.1f} MB)")
    print(f"  By fuel type: {dict(fuel_counts.most_common())}")
    print(f"  By region: {dict(region_counts.most_common())}")


if __name__ == "__main__":
    main()
