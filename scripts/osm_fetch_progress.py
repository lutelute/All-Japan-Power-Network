#!/usr/bin/env python3
"""OSM Fetch Progress Monitor — real-time progress bar for all regions.

Usage:
    python scripts/osm_fetch_progress.py          # one-shot
    python scripts/osm_fetch_progress.py --watch   # auto-refresh every 5s
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Paths
WORKTREE = ".auto-claude/worktrees/tasks/006-convert-kml-to-open-data-format"
OSM_DIR = os.path.join(WORKTREE, "data/raw/osm")
CACHE_DIR = os.path.join(WORKTREE, "cache")

# All 10 regions
REGIONS = [
    ("hokkaido",  "北海道",  "50Hz"),
    ("tohoku",    "東北",    "50Hz"),
    ("tokyo",     "東京",    "50Hz"),
    ("chubu",     "中部",    "60Hz"),
    ("hokuriku",  "北陸",    "60Hz"),
    ("kansai",    "関西",    "60Hz"),
    ("chugoku",   "中国",    "60Hz"),
    ("shikoku",   "四国",    "60Hz"),
    ("kyushu",    "九州",    "60Hz"),
    ("okinawa",   "沖縄",    "60Hz"),
]

BAR_WIDTH = 30


def check_region(region_key):
    """Check fetch status for a region. Returns (status, subs_count, lines_count, size_mb)."""
    sub_path = os.path.join(OSM_DIR, f"{region_key}_substations.geojson")
    line_path = os.path.join(OSM_DIR, f"{region_key}_lines.geojson")

    if os.path.exists(sub_path) and os.path.exists(line_path):
        sub_size = os.path.getsize(sub_path)
        line_size = os.path.getsize(line_path)
        total_mb = (sub_size + line_size) / (1024 * 1024)

        # Count features
        try:
            import geopandas as gpd
            subs = len(gpd.read_file(sub_path))
            lines = len(gpd.read_file(line_path))
        except Exception:
            subs = -1
            lines = -1

        return "done", subs, lines, total_mb
    elif os.path.exists(sub_path):
        return "partial", 0, 0, 0
    else:
        return "pending", 0, 0, 0


def render_bar(fraction, width=BAR_WIDTH):
    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def render_status_icon(status):
    if status == "done":
        return "✅"
    elif status == "partial":
        return "🔄"
    else:
        return "⏳"


def print_progress(use_geopandas=True):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache_count = len(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else 0

    print(f"\033[2J\033[H", end="")  # clear screen
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  OSM Power Grid Fetch Progress                           {now}  ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════════╣")

    done_count = 0
    total_subs = 0
    total_lines = 0
    total_size = 0

    results = []
    for key, name_ja, freq in REGIONS:
        status, subs, lines, size_mb = check_region(key)
        results.append((key, name_ja, freq, status, subs, lines, size_mb))
        if status == "done":
            done_count += 1
            total_subs += subs
            total_lines += lines
            total_size += size_mb

    for key, name_ja, freq, status, subs, lines, size_mb in results:
        icon = render_status_icon(status)
        frac = 1.0 if status == "done" else (0.5 if status == "partial" else 0.0)
        bar = render_bar(frac)
        pct = int(frac * 100)

        if status == "done":
            detail = f"{subs:>4} subs  {lines:>5} lines  {size_mb:>5.1f}MB"
        elif status == "partial":
            detail = "substations fetched, lines pending..."
        else:
            detail = "waiting / API request in progress..."

        print(f"║  {icon} {name_ja:<4} ({key:<8}) {freq}  {bar} {pct:>3}%  {detail:<36} ║")

    # Overall
    overall_frac = done_count / len(REGIONS)
    overall_bar = render_bar(overall_frac, 40)
    overall_pct = int(overall_frac * 100)

    print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Overall: {overall_bar} {overall_pct:>3}%  ({done_count}/{len(REGIONS)} regions)       ║")
    print(f"║  Totals:  {total_subs:>5} substations  {total_lines:>6} lines  {total_size:>6.1f} MB                   ║")
    print(f"║  osmnx cache entries: {cache_count:<6}                                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")

    return done_count, len(REGIONS)


def main():
    parser = argparse.ArgumentParser(description="OSM fetch progress monitor")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 5s")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                done, total = print_progress()
                if done == total:
                    print("\n All regions fetched! 🎉")
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress()


if __name__ == "__main__":
    main()
