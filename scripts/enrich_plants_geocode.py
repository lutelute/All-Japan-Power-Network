#!/usr/bin/env python3
"""Enrich unnamed plants with reverse-geocoded area names.

Uses Nominatim to look up the area/neighborhood for each unnamed plant
and constructs a display name like "{area}発電所".

Caches all geocode results to data/cache/plants_geocode.json for
incremental re-runs.

Usage:
    python scripts/enrich_plants_geocode.py                  # all regions
    python scripts/enrich_plants_geocode.py --region okinawa # single region
    python scripts/enrich_plants_geocode.py --skip-cached    # skip already-cached coords
"""

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "plants_geocode.json")

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

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "jpgrid-enrichment/1.0 (https://github.com/lutelute/All-Japan-Grid)"
RATE_LIMIT_SEC = 1.1  # Nominatim: max 1 req/sec


def load_cache():
    """Load geocode cache from disk. Returns empty dict on error."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_cache(cache):
    """Save geocode cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_centroid(feature):
    """Extract (lat, lon) from a GeoJSON feature."""
    geom = feature["geometry"]
    coords = geom["coordinates"]
    if geom["type"] == "Point":
        return coords[1], coords[0]
    elif geom["type"] == "Polygon":
        ring = coords[0]
        lon = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
        return lat, lon
    return None, None


def cache_key(lat, lon):
    """Create a cache key from coordinates."""
    return f"{lat:.6f},{lon:.6f}"


def reverse_geocode(lat, lon):
    """Reverse geocode via Nominatim, return address dict."""
    url = (
        f"{NOMINATIM_URL}?lat={lat}&lon={lon}"
        f"&format=json&accept-language=ja&zoom=16"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return data.get("address", {})
    except Exception:
        return {}


def construct_name(addr):
    """Construct a plant name from address components."""
    # Prefer specific area names
    area = (
        addr.get("neighbourhood")
        or addr.get("suburb")
        or addr.get("quarter")
        or ""
    )
    municipality = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or ""
    )

    if area and area != municipality:
        return f"{area}{municipality}発電所"
    elif municipality:
        return f"{municipality}発電所"
    return ""


def fallback_name(region, lat, lon):
    """Construct a fallback name when geocode returns empty."""
    region_ja = REGION_JA.get(region, region)
    return f"{region_ja}_{lat:.4f}_{lon:.4f}"


def enrich_region(region, cache, skip_cached=False):
    """Enrich unnamed plants in a region. Returns (total, enriched, api_calls)."""
    path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
    if not os.path.exists(path):
        return 0, 0, 0

    with open(path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = fc.get("features", [])
    total = len(features)
    enriched = 0
    api_calls = 0

    for feat in features:
        props = feat["properties"]

        # Skip features that already have a name
        name = (props.get("name") or props.get("name:ja") or "").strip()
        if name:
            continue

        # Skip features already enriched by nominatim
        if props.get("_enriched_by") == "nominatim":
            continue

        lat, lon = get_centroid(feat)
        if lat is None:
            continue

        key = cache_key(lat, lon)

        # Check cache first
        if key in cache:
            if skip_cached:
                continue
            addr = cache[key]
        else:
            # Reverse geocode via API
            addr = reverse_geocode(lat, lon)
            cache[key] = addr
            api_calls += 1
            time.sleep(RATE_LIMIT_SEC)

        constructed = construct_name(addr)
        if constructed:
            props["name"] = constructed
            props["_display_name"] = constructed
            props["_name_source"] = "geocoded"
            props["_enriched_by"] = "nominatim"
            enriched += 1
        else:
            # Fallback name when geocode returns empty
            fb = fallback_name(region, lat, lon)
            props["name"] = fb
            props["_display_name"] = fb
            props["_name_source"] = "fallback"
            props["_enriched_by"] = "nominatim"
            enriched += 1

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    still_unnamed = sum(
        1 for f in features
        if not (f["properties"].get("name") or f["properties"].get("name:ja") or "").strip()
    )
    return total, enriched, api_calls, still_unnamed


def main():
    parser = argparse.ArgumentParser(
        description="Enrich unnamed plants with geocoded area names"
    )
    parser.add_argument("--region", type=str, default=None,
                        help="Single region to process (default: all)")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip features whose coordinates are already in cache")
    args = parser.parse_args()

    regions = [args.region] if args.region else REGIONS

    # Load cache
    cache = load_cache()
    print(f"  Loaded {len(cache)} cached geocode results")

    total_enriched = 0
    total_api_calls = 0

    for region in regions:
        if region not in REGIONS:
            print(f"  Unknown region: {region}")
            continue
        print(f"  Processing {region}...")
        total, enriched, api_calls, still_unnamed = enrich_region(
            region, cache, skip_cached=args.skip_cached
        )
        total_enriched += enriched
        total_api_calls += api_calls
        print(f"    {total} total, {enriched} enriched, "
              f"{api_calls} API calls, {still_unnamed} still unnamed")

        # Save cache after each region (incremental safety)
        save_cache(cache)

    print(f"\n  TOTAL enriched: {total_enriched}, API calls: {total_api_calls}")
    print(f"  Cache size: {len(cache)} entries")


if __name__ == "__main__":
    main()
