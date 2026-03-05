#!/usr/bin/env python3
"""Unified enrichment pipeline for all GeoJSON layers.

Runs all enrichment steps in correct dependency order:
  1. Baseline audit (audit_data_quality.py)
  2. Substation name promotion (enrich_substations_geocode.py --promote-names)
  3. P03 plant enrichment (enrich_plants_p03.py, skip if GML missing)
  4. Overpass batch enrichment (enrich_overpass_tags.py)
  5. Plant geocoding (enrich_plants_geocode.py)
  6. Line endpoint naming (enrich_lines_endpoints.py)
  7. Final audit for validation

Usage:
    python scripts/enrich_all.py                    # full pipeline
    python scripts/enrich_all.py --region okinawa   # single region
    python scripts/enrich_all.py --dry-run           # print plan without running
    python scripts/enrich_all.py --skip-geocode      # skip slow Nominatim step
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]


def _log(msg):
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


def _format_elapsed(seconds):
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def build_steps(region=None, skip_geocode=False):
    """Build the ordered list of enrichment steps.

    Each step is a dict with:
      - name: human-readable step name
      - script: script filename
      - args: list of extra CLI arguments
      - description: what this step does
      - skippable: whether the step can fail gracefully
    """
    region_args = ["--region", region] if region else []

    steps = [
        {
            "name": "1. Baseline Audit",
            "script": "audit_data_quality.py",
            "args": region_args,
            "description": "Scan GeoJSON files and report current placeholder counts",
            "skippable": True,  # audit failure (exit 1) means placeholders exist, expected
        },
        {
            "name": "2. Substation Name Promotion",
            "script": "enrich_substations_geocode.py",
            "args": ["--promote-names"] + region_args,
            "description": "Promote _display_name to name for geocoded substations",
            "skippable": False,
        },
        {
            "name": "3. P03 Plant Enrichment",
            "script": "enrich_plants_p03.py",
            "args": [],
            "description": "Match OSM plants to P03 national dataset (skip if GML missing)",
            "skippable": True,  # gracefully skips when P03 GML not present
        },
        {
            "name": "4. Overpass Batch Tag Enrichment",
            "script": "enrich_overpass_tags.py",
            "args": region_args,
            "description": "Batch query Overpass API for missing name/operator/fuel_type",
            "skippable": False,
        },
        {
            "name": "5. Plant Geocoding",
            "script": "enrich_plants_geocode.py",
            "args": region_args,
            "description": "Reverse-geocode unnamed plants via Nominatim (~4.5h for all)",
            "skippable": False,
        },
        {
            "name": "6. Line Endpoint Naming",
            "script": "enrich_lines_endpoints.py",
            "args": region_args,
            "description": "Construct line names from matched endpoint substations",
            "skippable": False,
        },
        {
            "name": "7. Final Audit",
            "script": "audit_data_quality.py",
            "args": region_args,
            "description": "Validate enrichment results - expect zero placeholders",
            "skippable": True,  # exit 1 if placeholders remain
        },
    ]

    if skip_geocode:
        steps = [s for s in steps if s["script"] != "enrich_plants_geocode.py"]

    return steps


def print_plan(steps):
    """Print the execution plan without running anything."""
    _log("=" * 60)
    _log("ENRICHMENT PIPELINE - EXECUTION PLAN")
    _log("=" * 60)
    _log("")

    for step in steps:
        script_path = os.path.join(SCRIPTS_DIR, step["script"])
        cmd = f"python {step['script']} {' '.join(step['args'])}".strip()
        exists = os.path.exists(script_path)

        status = "READY" if exists else "MISSING"
        _log(f"  {step['name']}")
        _log(f"    Command:     {cmd}")
        _log(f"    Description: {step['description']}")
        _log(f"    Script:      {status}")
        if step["skippable"]:
            _log(f"    On failure:  continue (skippable)")
        else:
            _log(f"    On failure:  abort pipeline")
        _log("")

    _log(f"Total steps: {len(steps)}")
    _log("")


def run_step(step):
    """Run a single enrichment step via subprocess.

    Returns (success: bool, elapsed: float).
    """
    script_path = os.path.join(SCRIPTS_DIR, step["script"])

    if not os.path.exists(script_path):
        _log(f"    ERROR: script not found: {step['script']}")
        return False, 0.0

    cmd = [sys.executable, script_path] + step["args"]
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            timeout=7200,  # 2 hour max per step (geocoding can be long)
        )
        elapsed = time.time() - start

        if result.returncode != 0 and not step["skippable"]:
            _log(f"    FAILED (exit code {result.returncode})")
            return False, elapsed

        return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        _log(f"    TIMEOUT after {_format_elapsed(elapsed)}")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        _log(f"    ERROR: {str(e)[:120]}")
        return False, elapsed


def run_pipeline(steps):
    """Run all enrichment steps in order with timing."""
    total_start = time.time()
    step_results = []

    _log("=" * 60)
    _log("ENRICHMENT PIPELINE - RUNNING")
    _log("=" * 60)
    _log("")

    for i, step in enumerate(steps):
        _log(f"[{i + 1}/{len(steps)}] {step['name']}")
        _log(f"  {step['description']}")
        _log("")

        success, elapsed = run_step(step)
        step_results.append((step["name"], success, elapsed))

        _log("")
        _log(f"  => {'OK' if success else 'FAILED'} in {_format_elapsed(elapsed)}")
        _log("")

        if not success and not step["skippable"]:
            _log(f"  Pipeline ABORTED at step {i + 1}")
            break

    # Print summary
    total_elapsed = time.time() - total_start
    _log("")
    _log("=" * 60)
    _log("PIPELINE SUMMARY")
    _log("=" * 60)
    _log("")
    _log(f"  {'Step':<35} {'Status':>8}  {'Time':>10}")
    _log(f"  {'-' * 35} {'-' * 8}  {'-' * 10}")

    for name, success, elapsed in step_results:
        status = "OK" if success else "FAILED"
        _log(f"  {name:<35} {status:>8}  {_format_elapsed(elapsed):>10}")

    _log(f"  {'-' * 35} {'-' * 8}  {'-' * 10}")
    _log(f"  {'TOTAL':<35} {'':>8}  {_format_elapsed(total_elapsed):>10}")
    _log("")

    failed = [name for name, success, _ in step_results if not success]
    if failed:
        _log(f"  Failed steps: {', '.join(failed)}")
        _log("")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified enrichment pipeline for all GeoJSON layers"
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Single region to process (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running",
    )
    parser.add_argument(
        "--skip-geocode", action="store_true",
        help="Skip slow Nominatim geocoding step",
    )
    args = parser.parse_args()

    if args.region and args.region not in REGIONS:
        _log(f"ERROR: '{args.region}' not in {REGIONS}")
        sys.exit(1)

    steps = build_steps(region=args.region, skip_geocode=args.skip_geocode)

    if args.dry_run:
        print_plan(steps)
        return

    success = run_pipeline(steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
