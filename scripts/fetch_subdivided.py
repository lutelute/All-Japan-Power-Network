#!/usr/bin/env python3
"""Subdivided OSM fetcher for large regions (Hokkaido, Tokyo, etc.).

Splits a large bounding box into a grid of smaller tiles, fetches each
tile sequentially via the Overpass API, then merges results into a
single GeoJSON per region.  This avoids Overpass API timeouts on
country-sized queries.

Usage:
    # Fetch hokkaido split into 3x3 grid
    python scripts/fetch_subdivided.py --region hokkaido --rows 3 --cols 3

    # Fetch tokyo split into 2x2 grid
    python scripts/fetch_subdivided.py --region tokyo --rows 2 --cols 2

    # Fetch tohoku (medium size, 2x2 should suffice)
    python scripts/fetch_subdivided.py --region tohoku --rows 2 --cols 2

    # Fetch remaining regions that haven't completed yet
    python scripts/fetch_subdivided.py --region kansai
    python scripts/fetch_subdivided.py --region kyushu
    python scripts/fetch_subdivided.py --region okinawa
"""

import os
import sys
import time
import argparse
import math

import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml

# ── File-based logging (for sandbox environments) ─────────────────
LOG_PATH = os.environ.get("FETCH_LOG", "/tmp/fetch_subdivided.log")
_log_file = None

def _log(msg):
    """Write to both stdout and a log file."""
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_PATH, "a")
    text = f"{msg}\n"
    sys.stdout.write(text)
    sys.stdout.flush()
    _log_file.write(text)
    _log_file.flush()

# ── Configuration ──────────────────────────────────────────────────

WORKTREE = ".auto-claude/worktrees/tasks/006-convert-kml-to-open-data-format"
CONFIG_PATH = os.path.join(WORKTREE, "config/regions.yaml")
OUTPUT_DIR = os.path.join(WORKTREE, "data/raw/osm")

SUBSTATION_TAGS = {"power": "substation"}
LINE_TAGS = {"power": ["line", "cable"]}

# Overpass API settings — be polite
TIMEOUT = 300  # generous timeout per tile
PAUSE_BETWEEN_QUERIES = 3  # seconds between API calls
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0


def load_region_bbox(region: str) -> dict:
    """Load bounding box from regions.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    regions = config.get("regions", {})
    if region not in regions:
        _log(f"ERROR: Region '{region}' not found in {CONFIG_PATH}")
        _log(f"  Available: {', '.join(regions.keys())}")
        sys.exit(1)
    return regions[region]["bounding_box"]


def subdivide_bbox(bbox: dict, rows: int, cols: int) -> list:
    """Split a bounding box into a grid of smaller tiles.

    Returns list of (lon_min, lat_min, lon_max, lat_max) tuples
    in osmnx format.
    """
    lat_min = bbox["lat_min"]
    lat_max = bbox["lat_max"]
    lon_min = bbox["lon_min"]
    lon_max = bbox["lon_max"]

    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols

    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile_lat_min = lat_min + r * lat_step
            tile_lat_max = lat_min + (r + 1) * lat_step
            tile_lon_min = lon_min + c * lon_step
            tile_lon_max = lon_min + (c + 1) * lon_step
            # osmnx format: (left, bottom, right, top)
            tiles.append((tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max))
    return tiles


def fetch_with_retry(bbox_tuple, tags, label, max_retries=MAX_RETRIES):
    """Fetch OSM features with retry + exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            gdf = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
            return gdf
        except Exception as e:
            error_msg = str(e)
            # "EmptyOverpassResponse" means no features in this tile — that's OK
            if "EmptyOverpassResponse" in error_msg or "no data" in error_msg.lower():
                _log(f"    {label}: no features in this tile (empty response)")
                return gpd.GeoDataFrame()

            wait = BACKOFF_FACTOR ** (attempt - 1) * PAUSE_BETWEEN_QUERIES
            _log(f"    {label}: attempt {attempt}/{max_retries} failed: {error_msg[:80]}")
            if attempt < max_retries:
                _log(f"    Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                _log(f"    FAILED after {max_retries} attempts")
                return None
    return None


def fetch_region_subdivided(region: str, rows: int, cols: int):
    """Fetch a region by subdividing into tiles and merging."""
    _log(f"\n{'='*60}")
    _log(f"  Fetching: {region} ({rows}x{cols} = {rows*cols} tiles)")
    _log(f"{'='*60}")

    # Check if already cached
    sub_path = os.path.join(OUTPUT_DIR, f"{region}_substations.geojson")
    line_path = os.path.join(OUTPUT_DIR, f"{region}_lines.geojson")
    if os.path.exists(sub_path) and os.path.exists(line_path):
        _log(f"  SKIP: {region} already has cached GeoJSON files")
        subs = gpd.read_file(sub_path)
        lines = gpd.read_file(line_path)
        _log(f"  Cached: {len(subs)} substations, {len(lines)} lines")
        return True

    bbox = load_region_bbox(region)
    tiles = subdivide_bbox(bbox, rows, cols)
    total_tiles = len(tiles)

    _log(f"  Bbox: lat [{bbox['lat_min']}, {bbox['lat_max']}]"
          f"  lon [{bbox['lon_min']}, {bbox['lon_max']}]")
    _log(f"  Tiles: {total_tiles}")

    # Configure osmnx
    ox.settings.requests_timeout = TIMEOUT
    ox.settings.use_cache = True
    ox.settings.max_query_area_size = 50 * 1000 * 1000 * 1000  # avoid internal subdivision

    all_subs = []
    all_lines = []
    failed_tiles = []

    for i, tile in enumerate(tiles):
        tile_label = f"tile {i+1}/{total_tiles}"
        _log(f"\n  [{tile_label}] lon=[{tile[0]:.2f},{tile[2]:.2f}] lat=[{tile[1]:.2f},{tile[3]:.2f}]")

        # Fetch substations
        _log(f"    Fetching substations...")
        subs_gdf = fetch_with_retry(tile, SUBSTATION_TAGS, f"{tile_label} subs")
        if subs_gdf is None:
            failed_tiles.append((i, "substations"))
            continue
        if len(subs_gdf) > 0:
            all_subs.append(subs_gdf)
            _log(f"    Got {len(subs_gdf)} substations")

        time.sleep(PAUSE_BETWEEN_QUERIES)

        # Fetch lines
        _log(f"    Fetching lines...")
        lines_gdf = fetch_with_retry(tile, LINE_TAGS, f"{tile_label} lines")
        if lines_gdf is None:
            failed_tiles.append((i, "lines"))
            continue
        if len(lines_gdf) > 0:
            all_lines.append(lines_gdf)
            _log(f"    Got {len(lines_gdf)} lines")

        time.sleep(PAUSE_BETWEEN_QUERIES)

    # Merge results
    _log(f"\n  Merging tiles...")

    if all_subs:
        merged_subs = pd.concat(all_subs, ignore_index=True)
        # Deduplicate by osmid (same substation may appear in overlapping tiles)
        if "osmid" in merged_subs.columns:
            before = len(merged_subs)
            merged_subs = merged_subs.drop_duplicates(subset=["osmid"])
            _log(f"    Substations: {before} → {len(merged_subs)} (deduped)")
        else:
            merged_subs = gpd.GeoDataFrame(
                merged_subs.drop_duplicates(subset=["geometry"]),
                geometry="geometry"
            )
            _log(f"    Substations: {len(merged_subs)}")
    else:
        merged_subs = gpd.GeoDataFrame()
        _log(f"    Substations: 0 (no data)")

    if all_lines:
        merged_lines = pd.concat(all_lines, ignore_index=True)
        if "osmid" in merged_lines.columns:
            before = len(merged_lines)
            merged_lines = merged_lines.drop_duplicates(subset=["osmid"])
            _log(f"    Lines: {before} → {len(merged_lines)} (deduped)")
        else:
            merged_lines = gpd.GeoDataFrame(
                merged_lines.drop_duplicates(subset=["geometry"]),
                geometry="geometry"
            )
            _log(f"    Lines: {len(merged_lines)}")
    else:
        merged_lines = gpd.GeoDataFrame()
        _log(f"    Lines: 0 (no data)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if len(merged_subs) > 0:
        merged_subs.to_file(sub_path, driver="GeoJSON")
    else:
        # Write empty GeoJSON
        gpd.GeoDataFrame(geometry=[]).to_file(sub_path, driver="GeoJSON")

    if len(merged_lines) > 0:
        merged_lines.to_file(line_path, driver="GeoJSON")
    else:
        gpd.GeoDataFrame(geometry=[]).to_file(line_path, driver="GeoJSON")

    _log(f"\n  SAVED: {sub_path}")
    _log(f"  SAVED: {line_path}")

    if failed_tiles:
        _log(f"\n  WARNING: {len(failed_tiles)} tiles failed: {failed_tiles}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Subdivided OSM fetcher for large regions"
    )
    parser.add_argument("--region", required=True, help="Region key (e.g. hokkaido)")
    parser.add_argument("--rows", type=int, default=1, help="Grid rows (default: 1)")
    parser.add_argument("--cols", type=int, default=1, help="Grid cols (default: 1)")
    args = parser.parse_args()

    start = time.time()
    success = fetch_region_subdivided(args.region, args.rows, args.cols)
    elapsed = time.time() - start

    _log(f"\n{'='*60}")
    _log(f"  {args.region}: {'SUCCESS' if success else 'PARTIAL (some tiles failed)'}")
    _log(f"  Elapsed: {elapsed/60:.1f} min")
    _log(f"{'='*60}")


if __name__ == "__main__":
    main()
