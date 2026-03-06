#!/usr/bin/env python3
"""Export power flow results as GeoJSON for GitHub Pages visualization.

Runs DC and AC power flow on all 10 regions and exports per-region results
containing bus voltage, line loading, and generation data as GeoJSON files.

Preserves original OSM line geometry (multi-point polylines) and also exports
transformer connections as lines.

Usage::

    PYTHONPATH=. python scripts/export_powerflow_pages.py
"""

import copy
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp

from examples.run_powerflow_all import (
    REGIONS, REGION_JA, REGION_FREQ,
    build_network_from_geojson,
    fix_zero_voltages, insert_transformers, fix_topology,
    select_slack_bus, balance_power, scale_line_ratings,
    prune_dc_infeasible, run_powerflow,
    _get_line_coords, _get_centroid, _find_nearest_sub, _parse_voltage_kv,
)
from src.converter.pandapower_builder import PandapowerBuilder
from src.powerflow.load_estimator import estimate_loads, load_demand_config

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "powerflow")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_osm_line_geometries(region):
    """Load original OSM line geometries and build lookup by (from_sub, to_sub).

    Returns dict mapping (from_sub_id, to_sub_id) -> list of [lon, lat] coords.
    Also returns the reverse mapping for lines matched in opposite direction.
    """
    lines_path = os.path.join(DATA_DIR, f"{region}_lines.geojson")
    subs_path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
    if not os.path.exists(lines_path) or not os.path.exists(subs_path):
        return {}

    # Rebuild sub_coords exactly as build_network_from_geojson does
    with open(subs_path, encoding="utf-8") as f:
        subs_data = json.load(f)

    sub_coords = []
    for i, feat in enumerate(subs_data["features"]):
        lat, lon = _get_centroid(feat)
        if lat is None:
            continue
        sub_id = f"{region}_sub_{i}"
        sub_coords.append((lat, lon, sub_id))

    # Load lines and match endpoints
    with open(lines_path, encoding="utf-8") as f:
        lines_data = json.load(f)

    geom_lookup = {}  # (from_sub_id, to_sub_id) -> [[lon, lat], ...]
    for i, feat in enumerate(lines_data["features"]):
        coords = _get_line_coords(feat)
        if len(coords) < 2:
            continue

        start_lat, start_lon = coords[0]
        end_lat, end_lon = coords[-1]

        from_sub_id = _find_nearest_sub(start_lat, start_lon, sub_coords, 50.0)
        to_sub_id = _find_nearest_sub(end_lat, end_lon, sub_coords, 50.0)

        if not from_sub_id or not to_sub_id or from_sub_id == to_sub_id:
            continue

        # Convert (lat, lon) -> [lon, lat] for GeoJSON
        geojson_coords = [[lon, lat] for lat, lon in coords]

        key = (from_sub_id, to_sub_id)
        rev_key = (to_sub_id, from_sub_id)
        # Store first match; don't overwrite (some parallel lines)
        if key not in geom_lookup and rev_key not in geom_lookup:
            geom_lookup[key] = geojson_coords

    return geom_lookup


def build_and_solve(region, demand_cfg):
    """Build network, solve DC+AC, return (net_dc, dc_result, net_ac, ac_result, build_info)."""
    network = build_network_from_geojson(region)
    if not network or not network.has_elements:
        return None

    builder = PandapowerBuilder()
    build_result = builder.build(network)
    net = build_result.net

    fix_zero_voltages(net)
    n_trafos = insert_transformers(net)
    diag = fix_topology(net)
    select_slack_bus(net)

    total_load = estimate_loads(net, region=region, demand_config=demand_cfg)
    inactive_buses = set(net.bus.index[~net.bus["in_service"]])
    if len(net.load) > 0:
        mask = net.load["bus"].isin(inactive_buses)
        net.load.loc[mask, "in_service"] = False
        total_load = net.load[net.load["in_service"]]["p_mw"].sum()

    balance_power(net, demand_cfg)
    scale_line_ratings(net)
    net.bus["vm_pu"] = 1.0
    if len(net.gen) > 0:
        net.gen["vm_pu"] = 1.0
    if len(net.ext_grid) > 0:
        net.ext_grid["vm_pu"] = 1.0

    # DC
    net_dc = copy.deepcopy(net)
    dc_result = run_powerflow(net_dc, "dc")

    # AC with pruning
    ac_result = {"mode": "ac", "converged": False}
    net_ac = None
    for threshold in [45.0, 30.0, 20.0]:
        net_ac = copy.deepcopy(net)
        n_pruned = prune_dc_infeasible(net_ac, angle_threshold=threshold)
        if n_pruned > 0:
            fix_topology(net_ac)
            select_slack_bus(net_ac)
            scale_line_ratings(net_ac)
        ac_result = run_powerflow(net_ac, "ac")
        if ac_result["converged"]:
            break

    build_info = {
        "n_buses": len(net.bus),
        "n_lines": len(net.line),
        "n_gens": len(net.gen),
        "n_trafos": n_trafos,
        "n_active_buses": diag["n_active_buses"],
        "n_components": diag["n_components"],
        "total_load_mw": float(total_load),
        "total_gen_mw": float(net.gen[net.gen["in_service"]]["p_mw"].sum()) if len(net.gen) > 0 else 0,
    }

    return net_dc, dc_result, net_ac, ac_result, build_info


def _parse_bus_coords(net, idx):
    """Extract (lon, lat) from pandapower bus geo column."""
    if "geo" in net.bus.columns:
        geo_raw = net.bus.at[idx, "geo"]
        if isinstance(geo_raw, str):
            try:
                geo_obj = json.loads(geo_raw)
                coords = geo_obj.get("coordinates", [])
                if len(coords) >= 2:
                    return float(coords[0]), float(coords[1])
            except (json.JSONDecodeError, TypeError):
                pass
        elif hasattr(geo_raw, '__len__') and len(geo_raw) >= 2:
            return float(geo_raw[0]), float(geo_raw[1])
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty and idx in net.bus_geodata.index:
        row = net.bus_geodata.loc[idx]
        return float(row["x"]), float(row["y"])
    return None, None


def _build_bus_name_to_sub_id(net, region):
    """Build mapping from pandapower bus name -> sub_id for geometry lookup.

    Bus names are set from substation names during build_network_from_geojson,
    and sub_ids follow the pattern '{region}_sub_{i}'.
    The bus name in pandapower == substation name. We need the sub_id.
    Since we can't recover the exact sub index from the name alone,
    we use the bus name stored in net.bus and match it to the original
    substation data by coordinate proximity.
    """
    # Build bus_idx -> sub_id mapping using bus coordinates
    subs_path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
    if not os.path.exists(subs_path):
        return {}

    with open(subs_path, encoding="utf-8") as f:
        subs_data = json.load(f)

    # Build sub coordinate list: (lat, lon, sub_id)
    sub_locs = []
    for i, feat in enumerate(subs_data["features"]):
        lat, lon = _get_centroid(feat)
        if lat is None:
            continue
        sub_locs.append((lat, lon, f"{region}_sub_{i}"))

    # For each bus, find nearest sub by coordinate
    bus_to_sub = {}
    for bus_idx in net.bus.index:
        lon, lat = _parse_bus_coords(net, bus_idx)
        if lon is None:
            continue
        best_dist = float("inf")
        best_sub = None
        for slat, slon, sid in sub_locs:
            d = (lat - slat) ** 2 + (lon - slon) ** 2
            if d < best_dist:
                best_dist = d
                best_sub = sid
        if best_sub and best_dist < 0.001:  # ~100m threshold
            bus_to_sub[bus_idx] = best_sub

    return bus_to_sub


def export_bus_geojson(net, mode_label):
    """Export bus results as GeoJSON FeatureCollection."""
    features = []
    for idx in net.bus.index:
        if not net.bus.at[idx, "in_service"]:
            continue
        lon, lat = _parse_bus_coords(net, idx)
        if lon is None or (lon == 0 and lat == 0):
            continue

        vm_pu = float(net.res_bus.at[idx, "vm_pu"]) if idx in net.res_bus.index else 1.0
        va_deg = float(net.res_bus.at[idx, "va_degree"]) if idx in net.res_bus.index else 0.0
        vn_kv = float(net.bus.at[idx, "vn_kv"])

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "name": str(net.bus.at[idx, "name"]),
                "vn_kv": round(vn_kv, 1),
                "vm_pu": round(vm_pu, 4),
                "va_deg": round(va_deg, 2),
            }
        })

    return {"type": "FeatureCollection", "features": features}


def export_line_geojson(net, region, geom_lookup, bus_to_sub):
    """Export line results as GeoJSON FeatureCollection with original OSM geometry."""
    features = []
    geom_hits = 0
    geom_misses = 0

    for idx in net.line.index:
        if not net.line.at[idx, "in_service"]:
            continue

        from_bus = net.line.at[idx, "from_bus"]
        to_bus = net.line.at[idx, "to_bus"]

        from_lon, from_lat = _parse_bus_coords(net, from_bus)
        to_lon, to_lat = _parse_bus_coords(net, to_bus)

        if from_lon is None or to_lon is None:
            continue

        loading = 0.0
        p_mw = 0.0
        if idx in net.res_line.index:
            loading = float(net.res_line.at[idx, "loading_percent"]) if "loading_percent" in net.res_line.columns else 0.0
            p_mw = float(net.res_line.at[idx, "p_from_mw"]) if "p_from_mw" in net.res_line.columns else 0.0

        # Try to find original OSM geometry
        coords = None
        from_sub = bus_to_sub.get(from_bus)
        to_sub = bus_to_sub.get(to_bus)
        if from_sub and to_sub:
            coords = geom_lookup.get((from_sub, to_sub))
            if coords is None:
                # Try reverse direction
                rev = geom_lookup.get((to_sub, from_sub))
                if rev is not None:
                    coords = list(reversed(rev))

        if coords:
            geom_hits += 1
            # Snap endpoints to bus coordinates so lines connect visually
            coords = list(coords)  # copy
            coords[0] = [from_lon, from_lat]
            coords[-1] = [to_lon, to_lat]
        else:
            geom_misses += 1
            coords = [[from_lon, from_lat], [to_lon, to_lat]]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "name": str(net.line.at[idx, "name"]),
                "loading_pct": round(min(loading, 200), 1),
                "p_mw": round(p_mw, 1),
            }
        })

    # Also export transformers as line features
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for idx in net.trafo.index:
            if not net.trafo.at[idx, "in_service"]:
                continue

            hv_bus = net.trafo.at[idx, "hv_bus"]
            lv_bus = net.trafo.at[idx, "lv_bus"]

            hv_lon, hv_lat = _parse_bus_coords(net, hv_bus)
            lv_lon, lv_lat = _parse_bus_coords(net, lv_bus)

            if hv_lon is None or lv_lon is None:
                continue

            loading = 0.0
            p_mw = 0.0
            if hasattr(net, "res_trafo") and idx in net.res_trafo.index:
                loading = float(net.res_trafo.at[idx, "loading_percent"]) if "loading_percent" in net.res_trafo.columns else 0.0
                p_mw = float(net.res_trafo.at[idx, "p_hv_mw"]) if "p_hv_mw" in net.res_trafo.columns else 0.0

            # Try original geometry for transformer (was originally a line)
            coords = None
            from_sub = bus_to_sub.get(hv_bus)
            to_sub = bus_to_sub.get(lv_bus)
            if from_sub and to_sub:
                coords = geom_lookup.get((from_sub, to_sub))
                if coords is None:
                    rev = geom_lookup.get((to_sub, from_sub))
                    if rev is not None:
                        coords = list(reversed(rev))

            if coords is not None:
                # Snap endpoints to bus coordinates
                coords = list(coords)
                coords[0] = [hv_lon, hv_lat]
                coords[-1] = [lv_lon, lv_lat]
            else:
                coords = [[hv_lon, hv_lat], [lv_lon, lv_lat]]

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "name": str(net.trafo.at[idx, "name"]),
                    "loading_pct": round(min(loading, 200), 1),
                    "p_mw": round(p_mw, 1),
                    "is_trafo": True,
                }
            })

    return {"type": "FeatureCollection", "features": features}, geom_hits, geom_misses


def main():
    demand_cfg = load_demand_config()
    summary = {}

    for region in REGIONS:
        print(f"  Processing {region}...", end=" ", flush=True)
        result = build_and_solve(region, demand_cfg)
        if result is None:
            print("SKIP")
            continue

        net_dc, dc_result, net_ac, ac_result, build_info = result

        # Load original OSM line geometries for this region
        geom_lookup = _load_osm_line_geometries(region)

        # Export DC results
        if dc_result["converged"]:
            bus_to_sub_dc = _build_bus_name_to_sub_id(net_dc, region)
            dc_buses = export_bus_geojson(net_dc, "dc")
            dc_lines, dc_hits, dc_misses = export_line_geojson(net_dc, region, geom_lookup, bus_to_sub_dc)
            with open(os.path.join(OUTPUT_DIR, f"{region}_dc_buses.geojson"), "w") as f:
                json.dump(dc_buses, f, separators=(",", ":"))
            with open(os.path.join(OUTPUT_DIR, f"{region}_dc_lines.geojson"), "w") as f:
                json.dump(dc_lines, f, separators=(",", ":"))

        # Export AC results
        if ac_result["converged"]:
            bus_to_sub_ac = _build_bus_name_to_sub_id(net_ac, region)
            ac_buses = export_bus_geojson(net_ac, "ac")
            ac_lines, ac_hits, ac_misses = export_line_geojson(net_ac, region, geom_lookup, bus_to_sub_ac)
            with open(os.path.join(OUTPUT_DIR, f"{region}_ac_buses.geojson"), "w") as f:
                json.dump(ac_buses, f, separators=(",", ":"))
            with open(os.path.join(OUTPUT_DIR, f"{region}_ac_lines.geojson"), "w") as f:
                json.dump(ac_lines, f, separators=(",", ":"))

        dc_status = "OK" if dc_result["converged"] else "FAIL"
        ac_status = "OK" if ac_result["converged"] else "FAIL"
        ac_solver = ac_result.get("solver", "-")

        # Report geometry match rate
        if ac_result["converged"]:
            total = ac_hits + ac_misses
            rate = (ac_hits / total * 100) if total > 0 else 0
            geom_info = f"geom={ac_hits}/{total}({rate:.0f}%)"
        elif dc_result["converged"]:
            total = dc_hits + dc_misses
            rate = (dc_hits / total * 100) if total > 0 else 0
            geom_info = f"geom={dc_hits}/{total}({rate:.0f}%)"
        else:
            geom_info = ""

        n_trafo_feats = 0
        if ac_result["converged"] and hasattr(net_ac, "trafo"):
            n_trafo_feats = len(net_ac.trafo[net_ac.trafo["in_service"]])
        elif dc_result["converged"] and hasattr(net_dc, "trafo"):
            n_trafo_feats = len(net_dc.trafo[net_dc.trafo["in_service"]])

        summary[region] = {
            "name_ja": REGION_JA[region],
            "dc_converged": dc_result["converged"],
            "ac_converged": ac_result["converged"],
            "ac_solver": ac_solver,
            "dc_loss_mw": round(dc_result.get("total_loss_mw", 0), 1),
            "ac_loss_mw": round(ac_result.get("total_loss_mw", 0), 1),
            "ac_vm_min": round(ac_result.get("vm_pu_min", 0), 4),
            "ac_vm_max": round(ac_result.get("vm_pu_max", 0), 4),
            "dc_va_min": round(dc_result.get("va_deg_min", 0), 1),
            "dc_va_max": round(dc_result.get("va_deg_max", 0), 1),
            "dc_max_loading": round(dc_result.get("max_loading_pct", 0), 1),
            "ac_max_loading": round(ac_result.get("max_loading_pct", 0), 1),
            **build_info,
        }

        print(f"DC={dc_status} AC={ac_status} trafos={n_trafo_feats} {geom_info}")

    # Write summary JSON
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    dc_ok = sum(1 for v in summary.values() if v["dc_converged"])
    ac_ok = sum(1 for v in summary.values() if v["ac_converged"])
    print(f"\nDone: DC {dc_ok}/{len(summary)}, AC {ac_ok}/{len(summary)}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
