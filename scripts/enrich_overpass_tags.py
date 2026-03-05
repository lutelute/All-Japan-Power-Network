#!/usr/bin/env python3
"""Enrich plant features with missing attributes via Overpass API batch tag queries.

Scans plant GeoJSON files for features that have an osm_id but are missing
name, operator, or fuel_type.  Batches osm_ids into groups of 500 and queries
the Overpass API with ``nwr(id:...) out tags`` syntax.  Extracted tags are
normalised (fuel type via FUEL_TYPE_MAP) and written back in-place.

Results are cached to data/cache/overpass_tags.json so subsequent runs skip
already-fetched IDs.

Usage:
    python scripts/enrich_overpass_tags.py                       # all regions
    python scripts/enrich_overpass_tags.py --region okinawa      # single region
    python scripts/enrich_overpass_tags.py --region okinawa --dry-run  # preview only
"""

import argparse
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "overpass_tags.json")

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 300  # seconds for Overpass query
BATCH_SIZE = 500
PAUSE_BETWEEN_BATCHES = 5  # seconds between successful requests

# Exponential backoff for HTTP 429
BACKOFF_INITIAL = 30  # seconds
BACKOFF_MAX = 300  # seconds
BACKOFF_FACTOR = 2.0
MAX_RETRIES = 5

# OSM plant:source -> normalized fuel type (from fetch_plants.py)
FUEL_TYPE_MAP = {
    "coal": "coal", "gas": "gas", "natural_gas": "gas", "oil": "oil",
    "nuclear": "nuclear", "hydro": "hydro", "water": "hydro",
    "wind": "wind", "solar": "solar", "photovoltaic": "solar",
    "biomass": "biomass", "biogas": "biomass", "waste": "waste",
    "geothermal": "geothermal", "tidal": "tidal", "wave": "tidal",
    "battery": "battery", "pumped_storage": "pumped_hydro",
    "diesel": "oil",
}

# Fuel type display colors (from fetch_plants.py)
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


def normalize_fuel(source_str):
    """Normalize plant:source / generator:source to fuel type."""
    if not source_str:
        return "unknown"
    primary = source_str.split(";")[0].strip().lower()
    return FUEL_TYPE_MAP.get(primary, primary)


def load_cache():
    """Load cached Overpass tag results. Returns {osm_id_str: {tags dict}}."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        _log("  WARNING: cache file corrupt, rebuilding")
        return {}


def save_cache(cache):
    """Persist cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))


def needs_enrichment(props):
    """Return True if a feature has osm_id but is missing name/operator/fuel_type."""
    if not props.get("osm_id"):
        return False
    missing_name = not (props.get("name") or "").strip()
    missing_operator = not (props.get("operator") or "").strip()
    unknown_fuel = props.get("fuel_type") == "unknown"
    return missing_name or missing_operator or unknown_fuel


def fetch_overpass_batch(osm_ids):
    """Query Overpass API for tags of a batch of OSM IDs.

    Returns list of elements with their tags, or None on total failure.
    Implements exponential backoff for HTTP 429 responses.
    """
    ids_str = ",".join(str(oid) for oid in osm_ids)
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  nwr(id:{ids_str});
);
out tags;
"""

    backoff = BACKOFF_INITIAL
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=OVERPASS_TIMEOUT + 30,
            )

            if resp.status_code == 429:
                _log(f"    HTTP 429 (rate limited), backing off {backoff:.0f}s "
                     f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)
                continue

            resp.raise_for_status()
            return resp.json().get("elements", [])

        except requests.exceptions.HTTPError:
            _log(f"    HTTP {resp.status_code} on attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)
        except Exception as e:
            _log(f"    Attempt {attempt}/{MAX_RETRIES} failed: {str(e)[:80]}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)

    return None


def apply_tags_to_feature(props, tags, cache_entry):
    """Apply Overpass tags to a feature's properties. Returns True if changed."""
    changed = False

    # Name: only fill if currently empty
    if not (props.get("name") or "").strip():
        name = tags.get("name", tags.get("name:ja", tags.get("name:en", "")))
        if name:
            props["name"] = name
            props["_display_name"] = name
            changed = True
        # Also pick up name:ja and name:en
        if tags.get("name:ja") and not props.get("name:ja"):
            props["name:ja"] = tags["name:ja"]
        if tags.get("name:en") and not props.get("name:en"):
            props["name:en"] = tags["name:en"]

    # Operator: only fill if currently empty
    if not (props.get("operator") or "").strip():
        operator = tags.get("operator", "")
        if operator:
            props["operator"] = operator
            changed = True

    # Fuel type: only update if currently "unknown"
    if props.get("fuel_type") == "unknown":
        source = tags.get("plant:source",
                          tags.get("generator:source",
                                   tags.get("source", "")))
        if source:
            fuel = normalize_fuel(source)
            if fuel != "unknown":
                props["fuel_type"] = fuel
                props["plant:source"] = source
                props["_fuel_color"] = FUEL_COLORS.get(fuel, FUEL_COLORS["unknown"])
                changed = True

    if changed:
        props["_enriched_by"] = "overpass"

    return changed


def collect_features_to_enrich(regions):
    """Scan plant GeoJSON files and collect features needing enrichment.

    Returns list of (region, osm_id, feature_index) tuples and a dict of
    {region: feature_collection}.
    """
    to_enrich = []
    collections = {}

    for region in regions:
        path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            fc = json.load(f)

        collections[region] = fc
        for idx, feat in enumerate(fc.get("features", [])):
            props = feat["properties"]
            if needs_enrichment(props):
                to_enrich.append((region, props["osm_id"], idx))

    return to_enrich, collections


def enrich_regions(regions, dry_run=False):
    """Main enrichment logic across specified regions."""
    _log(f"Scanning plant GeoJSON files for {len(regions)} region(s)...")

    to_enrich, collections = collect_features_to_enrich(regions)
    cache = load_cache()

    # Deduplicate osm_ids (same ID shouldn't appear twice, but just in case)
    all_ids = list({oid for _, oid, _ in to_enrich})

    # Split into cached and uncached
    cached_ids = {oid for oid in all_ids if str(oid) in cache}
    uncached_ids = [oid for oid in all_ids if str(oid) not in cache]

    _log(f"  Features needing enrichment: {len(to_enrich)}")
    _log(f"  Unique OSM IDs: {len(all_ids)}")
    _log(f"  Already cached: {len(cached_ids)}")
    _log(f"  To fetch: {len(uncached_ids)}")
    _log(f"  Batches needed: {(len(uncached_ids) + BATCH_SIZE - 1) // BATCH_SIZE if uncached_ids else 0}")

    if dry_run:
        _log("\n  DRY RUN - no API calls or file changes")
        # Show per-region breakdown
        from collections import Counter
        region_counts = Counter(r for r, _, _ in to_enrich)
        for region in regions:
            count = region_counts.get(region, 0)
            if count:
                _log(f"    {region}: {count} features to enrich")
        return

    # Fetch uncached IDs from Overpass in batches
    if uncached_ids:
        batches = [uncached_ids[i:i + BATCH_SIZE]
                   for i in range(0, len(uncached_ids), BATCH_SIZE)]
        total_batches = len(batches)

        _log(f"\n  Fetching {len(uncached_ids)} IDs in {total_batches} batch(es)...")

        for batch_num, batch in enumerate(batches, 1):
            _log(f"  Batch {batch_num}/{total_batches} ({len(batch)} IDs)...")
            elements = fetch_overpass_batch(batch)

            if elements is None:
                _log(f"    FAILED - skipping batch")
                continue

            _log(f"    Received {len(elements)} elements")

            # Cache results
            for el in elements:
                cache[str(el["id"])] = el.get("tags", {})

            # Also cache IDs that returned no results (empty tags)
            returned_ids = {el["id"] for el in elements}
            for oid in batch:
                if oid not in returned_ids:
                    cache[str(oid)] = {}

            save_cache(cache)

            if batch_num < total_batches:
                time.sleep(PAUSE_BETWEEN_BATCHES)

    # Apply cached tags to features
    _log("\n  Applying tags to features...")
    enriched_count = 0
    region_enriched = {}

    for region, osm_id, feat_idx in to_enrich:
        tags = cache.get(str(osm_id), {})
        if not tags:
            continue

        fc = collections[region]
        feat = fc["features"][feat_idx]
        if apply_tags_to_feature(feat["properties"], tags, cache.get(str(osm_id))):
            enriched_count += 1
            region_enriched[region] = region_enriched.get(region, 0) + 1

    # Write back modified GeoJSON files
    for region, fc in collections.items():
        path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

    # Summary
    _log(f"\n  Enriched {enriched_count} features across {len(region_enriched)} region(s)")
    for region in regions:
        count = region_enriched.get(region, 0)
        if count:
            _log(f"    {region}: {count} enriched")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich plant features via Overpass API batch tag queries")
    parser.add_argument("--region", type=str, default=None,
                        help="Single region to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without making API calls")
    args = parser.parse_args()

    if args.region:
        if args.region not in REGIONS:
            _log(f"ERROR: '{args.region}' not in {REGIONS}")
            sys.exit(1)
        regions = [args.region]
    else:
        regions = REGIONS

    _log(f"\nOverpass tag enrichment for {len(regions)} region(s)\n")
    start = time.time()
    enrich_regions(regions, dry_run=args.dry_run)
    elapsed = time.time() - start
    _log(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
