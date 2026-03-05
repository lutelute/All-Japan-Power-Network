#!/usr/bin/env python3
"""Enrich unnamed substations with reverse-geocoded area names.

Uses Nominatim to look up the area/neighborhood for each unnamed substation
and constructs a display name like "{area}変電所".

Also picks up name:en as fallback if name/name:ja are missing.

Supports --promote-names to promote _display_name to name for substations
that have been geocoded but not yet officially named.

Usage:
    python scripts/enrich_substations_geocode.py                  # all regions
    python scripts/enrich_substations_geocode.py --region okinawa # single region
    python scripts/enrich_substations_geocode.py --promote-names  # promote display names
"""

import argparse
from collections import Counter
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "jpgrid-enrichment/1.0 (https://github.com/lutelute/All-Japan-Grid)"
RATE_LIMIT_SEC = 1.1  # Nominatim: max 1 req/sec


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
    """Construct a substation name from address components."""
    # Prefer specific area names
    area = (
        addr.get("neighbourhood")
        or addr.get("suburb")
        or addr.get("quarter")
        or addr.get("city_district")
        or ""
    )
    municipality = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or ""
    )

    if area and area != municipality:
        return f"{area}変電所"
    elif municipality:
        return f"{municipality}変電所"
    return ""


def enrich_region(region):
    """Enrich unnamed substations in a region. Returns (total, enriched)."""
    path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
    if not os.path.exists(path):
        return 0, 0

    with open(path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = fc.get("features", [])
    total = len(features)
    enriched = 0

    for feat in features:
        props = feat["properties"]
        name = (props.get("name") or props.get("name:ja") or "").strip()
        if name:
            continue

        # Fallback 1: name:en
        name_en = (props.get("name:en") or "").strip()
        if name_en:
            props["_display_name"] = name_en
            props["_name_source"] = "name:en"
            enriched += 1
            continue

        # Fallback 2: reverse geocode
        lat, lon = get_centroid(feat)
        if lat is None:
            continue

        addr = reverse_geocode(lat, lon)
        constructed = construct_name(addr)
        if constructed:
            props["_display_name"] = constructed
            props["_name_source"] = "geocoded"
            enriched += 1

        time.sleep(RATE_LIMIT_SEC)

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    unnamed = sum(
        1 for f in features
        if not (f["properties"].get("name") or f["properties"].get("name:ja") or "").strip()
    )
    return total, enriched, unnamed


def promote_names_region(region):
    """Promote _display_name to name for unnamed substations. Returns (total, promoted)."""
    path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
    if not os.path.exists(path):
        return 0, 0

    with open(path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = fc.get("features", [])
    total = len(features)

    # Collect all names that will be used (existing + promoted) to detect duplicates
    # First pass: gather existing names and candidate display names
    existing_names = Counter()
    candidates = []
    for feat in features:
        props = feat["properties"]
        name = (props.get("name") or props.get("name:ja") or "").strip()
        if name:
            existing_names[name] += 1
        else:
            display = (props.get("_display_name") or "").strip()
            if display:
                candidates.append(feat)

    # Count how many times each candidate display_name will be used
    candidate_names = Counter(
        (f["properties"].get("_display_name") or "").strip()
        for f in candidates
    )

    # Merge with existing names to get full picture of duplicates
    all_names = Counter()
    all_names.update(existing_names)
    all_names.update(candidate_names)

    # Second pass: assign names with dedup suffixes
    name_usage = Counter(existing_names)  # track how many times each name has been assigned
    promoted = 0
    for feat in candidates:
        props = feat["properties"]
        display = (props.get("_display_name") or "").strip()
        if not display:
            continue

        # Determine if this name needs a suffix
        if all_names[display] > 1:
            name_usage[display] += 1
            count = name_usage[display]
            if count == 1:
                final_name = display
            else:
                final_name = f"{display}_{count}"
        else:
            final_name = display

        props["name"] = final_name
        props["_enriched_by"] = "geocode_promotion"
        promoted += 1

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    return total, promoted


def main():
    parser = argparse.ArgumentParser(description="Enrich substations with geocoded names")
    parser.add_argument("--region", type=str, default=None,
                        help="Single region to process (default: all)")
    parser.add_argument("--promote-names", action="store_true",
                        help="Promote _display_name to name for unnamed substations")
    args = parser.parse_args()

    regions = [args.region] if args.region else REGIONS

    if args.promote_names:
        total_promoted = 0
        for region in regions:
            if region not in REGIONS:
                print(f"  Unknown region: {region}")
                continue
            print(f"  Processing {region}...")
            total, promoted = promote_names_region(region)
            total_promoted += promoted
            print(f"    {total} total, {promoted} promoted")

        print(f"\n  TOTAL promoted: {total_promoted}")
    else:
        total_enriched = 0
        for region in regions:
            if region not in REGIONS:
                print(f"  Unknown region: {region}")
                continue
            print(f"  Processing {region}...")
            total, enriched, still_unnamed = enrich_region(region)
            total_enriched += enriched
            print(f"    {total} total, {enriched} enriched, {still_unnamed} still unnamed")

        print(f"\n  TOTAL enriched: {total_enriched}")


if __name__ == "__main__":
    main()
