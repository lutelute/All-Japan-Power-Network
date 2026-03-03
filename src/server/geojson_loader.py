"""Load, normalize, and cache OSM GeoJSON data for the web server.

Reads ``data/{region}_{substations|lines|plants}.geojson`` files at startup,
normalizes voltage values (V -> kV), tags each feature with its region,
and caches everything in memory for fast API responses.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "regions.yaml")

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

# Cache: region -> {"substations": FC, "lines": FC, "plants": FC}
_cache: Dict[str, Dict[str, Any]] = {}
_region_config: Dict[str, Any] = {}

# Fuel type display colors
FUEL_COLORS = {
    "nuclear": "#ff0000", "coal": "#444444", "gas": "#ff8800",
    "oil": "#884400", "hydro": "#0088ff", "pumped_hydro": "#0044aa",
    "wind": "#00cc88", "solar": "#ffdd00", "geothermal": "#cc4488",
    "biomass": "#668833", "waste": "#996633", "tidal": "#006688",
    "battery": "#aa00ff", "unknown": "#999999",
}


def _normalize_voltage(voltage_raw: Any) -> Optional[float]:
    """Convert OSM voltage string to kV float.

    OSM stores voltage in volts (e.g. "275000" for 275 kV).
    Returns None if unparseable.
    """
    if voltage_raw is None:
        return None
    s = str(voltage_raw).strip().replace(",", "")
    # Handle multiple voltages separated by ;
    if ";" in s:
        s = s.split(";")[0].strip()
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    # If > 1000, assume it's in volts and convert to kV
    if v > 1000:
        return round(v / 1000, 1)
    return round(v, 1) if v > 0 else None


def _enrich_feature(feature: dict, region: str, layer: str) -> dict:
    """Add region tag and normalize voltage on a GeoJSON feature."""
    props = feature.get("properties", {})
    props["_region"] = region
    props["_region_ja"] = REGION_JA.get(region, region)

    # Normalize voltage
    raw_voltage = props.get("voltage")
    voltage_kv = _normalize_voltage(raw_voltage)
    props["_voltage_kv"] = voltage_kv

    # Enrich name
    name = props.get("name") or props.get("name:ja") or ""
    props["_display_name"] = name

    feature["properties"] = props
    return feature


def _enrich_plant_feature(feature: dict, region: str) -> dict:
    """Enrich a plant GeoJSON feature with region tag and fuel color."""
    props = feature.get("properties", {})
    props["_region"] = region
    props["_region_ja"] = REGION_JA.get(region, region)
    if not props.get("_display_name"):
        props["_display_name"] = props.get("name") or props.get("name:ja") or ""
    fuel = props.get("fuel_type", "unknown")
    if not fuel or fuel.startswith("http") or len(fuel) > 20:
        fuel = "unknown"
        props["fuel_type"] = fuel
    props["_fuel_color"] = FUEL_COLORS.get(fuel, FUEL_COLORS["unknown"])
    feature["properties"] = props
    return feature


def _load_geojson_file(path: str) -> Optional[dict]:
    """Load a single GeoJSON file, return None on error."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_region_config() -> Dict[str, Any]:
    """Load region config from YAML."""
    global _region_config
    if _region_config:
        return _region_config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _region_config = yaml.safe_load(f)
    return _region_config


def load_all() -> Dict[str, Dict[str, Any]]:
    """Load and cache all GeoJSON data at startup.

    Returns:
        Dict mapping region name to {"substations": FC, "lines": FC, "plants": FC, "counts": {...}}
    """
    global _cache
    if _cache:
        return _cache

    data_dir = os.path.abspath(DATA_DIR)

    for region in REGIONS:
        sub_path = os.path.join(data_dir, f"{region}_substations.geojson")
        line_path = os.path.join(data_dir, f"{region}_lines.geojson")
        plant_path = os.path.join(data_dir, f"{region}_plants.geojson")

        sub_fc = _load_geojson_file(sub_path)
        line_fc = _load_geojson_file(line_path)
        plant_fc = _load_geojson_file(plant_path)

        if sub_fc is None and line_fc is None and plant_fc is None:
            continue

        # Enrich features
        if sub_fc and "features" in sub_fc:
            sub_fc["features"] = [
                _enrich_feature(f, region, "substations")
                for f in sub_fc["features"]
            ]
        if line_fc and "features" in line_fc:
            line_fc["features"] = [
                _enrich_feature(f, region, "lines")
                for f in line_fc["features"]
            ]
        if plant_fc and "features" in plant_fc:
            plant_fc["features"] = [
                _enrich_plant_feature(f, region)
                for f in plant_fc["features"]
            ]

        empty_fc = {"type": "FeatureCollection", "features": []}
        _cache[region] = {
            "substations": sub_fc or empty_fc,
            "lines": line_fc or empty_fc,
            "plants": plant_fc or empty_fc,
            "counts": {
                "substations": len((sub_fc or {}).get("features", [])),
                "lines": len((line_fc or {}).get("features", [])),
                "plants": len((plant_fc or {}).get("features", [])),
            },
        }

    return _cache


def get_regions_summary() -> List[Dict[str, Any]]:
    """Return summary info for all loaded regions."""
    cache = load_all()
    config = load_region_config()
    regions_cfg = config.get("regions", {})

    result = []
    for region in REGIONS:
        if region not in cache:
            continue
        cfg = regions_cfg.get(region, {})
        result.append({
            "id": region,
            "name_en": cfg.get("name_en", region.title()),
            "name_ja": REGION_JA.get(region, region),
            "frequency_hz": cfg.get("frequency_hz", 0),
            "substations": cache[region]["counts"]["substations"],
            "lines": cache[region]["counts"]["lines"],
            "plants": cache[region]["counts"].get("plants", 0),
            "bounding_box": cfg.get("bounding_box"),
        })
    return result


def get_geojson(region: str, layer: str) -> Optional[dict]:
    """Get cached GeoJSON for a region and layer.

    Args:
        region: Region id (e.g. "hokkaido").
        layer: "substations" or "lines".

    Returns:
        GeoJSON FeatureCollection or None.
    """
    cache = load_all()
    region_data = cache.get(region)
    if region_data is None:
        return None
    return region_data.get(layer)


def get_all_geojson(layer: str) -> dict:
    """Merge all regions into a single FeatureCollection."""
    cache = load_all()
    features = []
    for region in REGIONS:
        region_data = cache.get(region)
        if region_data and layer in region_data:
            fc = region_data[layer]
            features.extend(fc.get("features", []))
    return {"type": "FeatureCollection", "features": features}


def _simplify_coords(coords: list, step: int = 3) -> list:
    """Downsample coordinate list, keeping first and last."""
    if len(coords) <= 4:
        return coords
    result = coords[::step]
    if result[-1] != coords[-1]:
        result.append(coords[-1])
    return result


def _compact_feature(feat: dict, layer: str) -> dict:
    """Strip heavy OSM properties and simplify geometry for lightweight payload."""
    p = feat.get("properties", {})
    compact_props = {
        "_region": p.get("_region"),
        "_region_ja": p.get("_region_ja"),
        "_display_name": p.get("_display_name"),
    }
    if layer == "plants":
        compact_props["fuel_type"] = p.get("fuel_type", "unknown")
        compact_props["capacity_mw"] = p.get("capacity_mw")
        compact_props["_fuel_color"] = p.get("_fuel_color", "#999999")
    else:
        compact_props["_voltage_kv"] = p.get("_voltage_kv")

    geom = feat.get("geometry", {})
    if layer == "lines" and geom.get("type") == "LineString":
        geom = {
            "type": "LineString",
            "coordinates": _simplify_coords(geom.get("coordinates", [])),
        }
    elif layer in ("substations", "plants") and geom.get("type") == "Polygon":
        # Use centroid instead of full polygon
        ring = geom.get("coordinates", [[]])[0]
        if ring:
            lon = sum(c[0] for c in ring) / len(ring)
            lat = sum(c[1] for c in ring) / len(ring)
            geom = {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]}
    return {"type": "Feature", "properties": compact_props, "geometry": geom}


def get_all_geojson_light(
    layer: str,
    min_voltage_kv: float = 0,
) -> dict:
    """Lightweight all-regions GeoJSON with simplified geometry.

    Strips OSM properties, downsamples line coordinates, and optionally
    filters by minimum voltage to keep payload small enough for the browser.
    """
    cache = load_all()
    features = []
    for region in REGIONS:
        region_data = cache.get(region)
        if region_data is None or layer not in region_data:
            continue
        for feat in region_data[layer].get("features", []):
            kv = feat.get("properties", {}).get("_voltage_kv")
            if min_voltage_kv > 0 and (kv is None or kv < min_voltage_kv):
                continue
            features.append(_compact_feature(feat, layer))
    return {"type": "FeatureCollection", "features": features}
