#!/usr/bin/env python3
"""Export enriched substation data from OSM GeoJSON with voltage inference.

Reads raw OSM substation GeoJSON files, enriches properties, infers
missing voltage from nearby transmission lines and substation type
heuristics, and outputs a single GeoJSON for GitHub Pages.

Usage:
    python scripts/export_substations_geojson.py

Output:
    docs/data/substations.geojson
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data/osm"
OUTPUT_PATH = "docs/data/substations.geojson"

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

REGION_JA = {
    "hokkaido": "北海道", "tohoku": "東北", "tokyo": "東京",
    "chubu": "中部", "hokuriku": "北陸", "kansai": "関西",
    "chugoku": "中国", "shikoku": "四国", "kyushu": "九州",
    "okinawa": "沖縄",
}

# Operator name normalization (variants → canonical)
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

# Substation type classification
SUBSTATION_TYPE_INFO = {
    "transmission": {
        "category": "transmission",
        "category_ja": "送電用変電所",
        "default_voltage_kv": 154,
    },
    "distribution": {
        "category": "distribution",
        "category_ja": "配電用変電所",
        "default_voltage_kv": 66,
    },
    "traction": {
        "category": "traction",
        "category_ja": "鉄道用変電所",
        "default_voltage_kv": 25,
    },
    "industrial": {
        "category": "industrial",
        "category_ja": "需要家変電所",
        "default_voltage_kv": 66,
    },
    "generation": {
        "category": "generation",
        "category_ja": "発電所構内変電所",
        "default_voltage_kv": 154,
    },
    "converter": {
        "category": "converter",
        "category_ja": "周波数変換所/直流変換所",
        "default_voltage_kv": 275,
    },
    "transition": {
        "category": "transition",
        "category_ja": "架空/地中切替所",
        "default_voltage_kv": 66,
    },
    "switching": {
        "category": "switching",
        "category_ja": "開閉所",
        "default_voltage_kv": 154,
    },
    "compensation": {
        "category": "compensation",
        "category_ja": "調相所",
        "default_voltage_kv": 275,
    },
    "minor_distribution": {
        "category": "minor_distribution",
        "category_ja": "配電塔",
        "default_voltage_kv": 22,
    },
}

# Voltage category labels
VOLTAGE_CATEGORIES = {
    500: "UHV (500kV)",
    275: "EHV (275kV)",
    220: "EHV (220kV)",
    187: "HV (187kV)",
    154: "HV (154kV)",
    132: "HV (132kV)",
    110: "HV (110kV)",
    77: "MV (77kV)",
    66: "MV (66kV)",
}

# Category colors for map
CATEGORY_COLORS = {
    "transmission": "#ff7f0e",
    "distribution": "#2ca02c",
    "traction": "#9467bd",
    "industrial": "#8c564b",
    "generation": "#e94560",
    "converter": "#17becf",
    "transition": "#bcbd22",
    "switching": "#d62728",
    "compensation": "#1f77b4",
    "minor_distribution": "#7f7f7f",
    "unknown": "#cccccc",
}


def normalize_voltage(voltage_raw):
    """Convert OSM voltage string to kV float. Returns None if invalid."""
    if not voltage_raw:
        return None
    s = str(voltage_raw).strip()
    # Remove DC prefix
    if s.lower().startswith("dc"):
        s = s[2:]
    # Handle multiple voltages separated by ; or ,
    if ";" in s or "," in s:
        parts = s.replace(",", ";").split(";")
        voltages = []
        for part in parts:
            v = normalize_voltage(part.strip())
            if v is not None:
                voltages.append(v)
        return max(voltages) if voltages else None
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    if v <= 0:
        return None
    # OSM stores voltage in volts; convert to kV
    # Values > 1000 are definitely in volts (e.g., 275000 → 275 kV)
    # Values 100-1000 are ambiguous: 750 = 750V DC, 500 could be 500kV
    # Heuristic: standard Japanese kV levels that appear raw are > 1000V in OSM
    # Values < 1000 that look like kV (66, 77, 110, etc.) would be stored as
    # 66000, 77000, etc. in OSM. So values < 1000 are likely in volts.
    if v >= 1000:
        return round(v / 1000, 1)
    # Values < 1000 are volts (e.g., 750V DC → 0.75 kV)
    return round(v / 1000, 3)


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


def feature_centroid(feat):
    """Get centroid [lon, lat] from a GeoJSON feature."""
    geom = feat.get("geometry", {})
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Point":
        return coords[:2] if len(coords) >= 2 else None
    elif gtype == "Polygon":
        ring = coords[0] if coords else []
        if not ring:
            return None
        lon = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
        return [round(lon, 6), round(lat, 6)]
    elif gtype == "MultiPolygon":
        all_pts = [c for poly in coords for ring in poly for c in ring]
        if not all_pts:
            return None
        lon = sum(c[0] for c in all_pts) / len(all_pts)
        lat = sum(c[1] for c in all_pts) / len(all_pts)
        return [round(lon, 6), round(lat, 6)]
    return None


def haversine_km(lon1, lat1, lon2, lat2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_line_voltage_index(region):
    """Build spatial index of line endpoint voltages for a region."""
    path = os.path.join(DATA_DIR, f"{region}_lines.geojson")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        v = normalize_voltage(p.get("voltage"))
        if v is None or v < 10:
            continue
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "LineString" and len(coords) >= 2:
            # Use both endpoints
            for c in [coords[0], coords[-1]]:
                points.append((c[0], c[1], v))
    return points


def infer_voltage_from_lines(lon, lat, line_points, max_dist_km=1.0):
    """Find highest voltage from nearby line endpoints."""
    nearby = []
    for lp_lon, lp_lat, lp_v in line_points:
        # Quick bounding box filter (~0.01 deg ≈ 1km)
        if abs(lp_lon - lon) > 0.015 or abs(lp_lat - lat) > 0.015:
            continue
        dist = haversine_km(lon, lat, lp_lon, lp_lat)
        if dist <= max_dist_km:
            nearby.append(lp_v)
    return max(nearby) if nearby else None


def classify_voltage(kv):
    """Map a kV value to the nearest standard voltage bracket."""
    if kv is None:
        return None
    brackets = [500, 275, 220, 187, 154, 132, 110, 77, 66, 33, 22, 6.6]
    best = None
    best_diff = float("inf")
    for b in brackets:
        diff = abs(kv - b)
        if diff < best_diff:
            best_diff = diff
            best = b
    # Only snap if within 15% tolerance
    if best and best_diff / best < 0.15:
        return best
    return round(kv, 1)


def enrich_feature(feat, region, line_points):
    """Enrich a substation feature with full attributes."""
    p = feat.get("properties", {})

    # Get centroid
    centroid = feature_centroid(feat)
    if centroid is None:
        return None

    lon, lat = centroid

    # Voltage
    voltage_kv = normalize_voltage(p.get("voltage"))
    voltage_source = "osm" if voltage_kv is not None else None

    # Infer from nearby lines if missing
    if voltage_kv is None:
        voltage_kv = infer_voltage_from_lines(lon, lat, line_points)
        if voltage_kv is not None:
            voltage_source = "inferred_from_lines"

    # Substation type
    sub_type = p.get("substation") or ""
    # Clean up misplaced data in substation field
    if sub_type and sub_type not in SUBSTATION_TYPE_INFO and len(sub_type) > 20:
        sub_type = ""
    type_info = SUBSTATION_TYPE_INFO.get(sub_type, {})

    # Infer from type if still missing
    if voltage_kv is None and type_info:
        voltage_kv = type_info.get("default_voltage_kv")
        voltage_source = "inferred_from_type"

    # Classify voltage
    voltage_kv_classified = classify_voltage(voltage_kv) if voltage_kv else None

    # Name
    name = p.get("name") or p.get("name:ja") or ""
    name_en = p.get("name:en") or ""
    name_reading = p.get("name:ja-Hira") or p.get("name:ja_rm") or ""

    # Operator
    operator_raw = p.get("operator") or p.get("operator:ja") or ""
    operator = normalize_operator(operator_raw)
    operator_en = p.get("operator:en") or ""
    operator_short = p.get("operator:short") or ""

    # Category
    category = type_info.get("category", "unknown")
    category_ja = type_info.get("category_ja", "不明")

    # Voltage category label
    voltage_label = VOLTAGE_CATEGORIES.get(
        voltage_kv_classified,
        f"{voltage_kv_classified} kV" if voltage_kv_classified else "Unknown"
    )

    # Additional OSM attributes
    frequency = p.get("frequency")
    if frequency:
        try:
            frequency = float(frequency)
        except (ValueError, TypeError):
            frequency = None

    gas_insulated = p.get("gas_insulated")
    if gas_insulated:
        gas_insulated = str(gas_insulated).lower() in ("yes", "true", "1")
    else:
        gas_insulated = None

    rating = p.get("rating") or ""

    properties = {
        # Identity
        "name": name,
        "name_en": name_en,
        "name_reading": name_reading,
        "operator": operator,
        "operator_en": operator_en,
        "operator_short": operator_short,
        "region": region,
        "region_ja": REGION_JA.get(region, region),
        # Electrical
        "voltage_kv": voltage_kv_classified,
        "voltage_source": voltage_source,
        "voltage_label": voltage_label,
        "frequency_hz": frequency,
        "rating": rating,
        # Classification
        "substation_type": sub_type if sub_type in SUBSTATION_TYPE_INFO else "",
        "category": category,
        "category_ja": category_ja,
        "power": p.get("power") or "substation",
        # Physical
        "gas_insulated": gas_insulated,
        "building": p.get("building") or "",
        # Reference
        "ref": p.get("ref") or "",
        "source": p.get("source") or "",
        "website": p.get("website") or "",
        "operator_wikidata": p.get("operator:wikidata") or "",
        # Address
        "addr_city": p.get("addr:city") or "",
        "addr_province": p.get("addr:province") or "",
        # Display
        "_display_name": name or name_en or "(unnamed)",
        "_color": CATEGORY_COLORS.get(category, "#cccccc"),
    }

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(lon, 6), round(lat, 6)],
        },
        "properties": properties,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export enriched substations to GeoJSON"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help=f"Output path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--exclude-traction", action="store_true",
        help="Exclude traction (railway) substations"
    )
    args = parser.parse_args()

    features = []
    stats = {
        "total_raw": 0,
        "voltage_osm": 0,
        "voltage_from_lines": 0,
        "voltage_from_type": 0,
        "voltage_unknown": 0,
    }

    for region in REGIONS:
        sub_path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
        if not os.path.exists(sub_path):
            print(f"  Skip {region}: no data")
            continue

        print(f"Processing {region}...")

        # Build line voltage index
        line_points = build_line_voltage_index(region)
        print(f"  Line voltage index: {len(line_points)} points")

        with open(sub_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        region_count = 0
        for feat in data.get("features", []):
            stats["total_raw"] += 1

            enriched = enrich_feature(feat, region, line_points)
            if enriched is None:
                continue

            p = enriched["properties"]

            # Optional: exclude traction substations
            if args.exclude_traction and p["category"] == "traction":
                continue

            # Track voltage inference stats
            vs = p.get("voltage_source")
            if vs == "osm":
                stats["voltage_osm"] += 1
            elif vs == "inferred_from_lines":
                stats["voltage_from_lines"] += 1
            elif vs == "inferred_from_type":
                stats["voltage_from_type"] += 1
            else:
                stats["voltage_unknown"] += 1

            features.append(enriched)
            region_count += 1

        print(f"  {region}: {region_count} substations")

    geojson = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = os.path.getsize(args.output) / (1024 * 1024)

    # Summary stats
    cat_counts = Counter(f["properties"]["category"] for f in features)
    v_counts = Counter(f["properties"]["voltage_kv"] for f in features)

    print(f"\nExported {len(features)} substations")
    print(f"  File: {args.output} ({size_mb:.1f} MB)")
    print(f"\nVoltage resolution:")
    print(f"  OSM (original):        {stats['voltage_osm']}")
    print(f"  Inferred from lines:   {stats['voltage_from_lines']}")
    print(f"  Inferred from type:    {stats['voltage_from_type']}")
    print(f"  Still unknown:         {stats['voltage_unknown']}")
    resolved = stats["voltage_osm"] + stats["voltage_from_lines"] + stats["voltage_from_type"]
    total = resolved + stats["voltage_unknown"]
    print(f"  Resolution rate:       {resolved}/{total} ({resolved/total*100:.1f}%)")
    print(f"\nBy category:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat}: {cnt}")
    print(f"\nTop voltage classes:")
    for v, cnt in sorted(v_counts.most_common()[:15], key=lambda x: -(x[0] or 0)):
        print(f"  {v} kV: {cnt}")


if __name__ == "__main__":
    main()
