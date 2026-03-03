"""Power flow orchestration for the web server.

Converts OSM GeoJSON data to a GridNetwork, builds a pandapower model,
runs power flow, and returns results as GeoJSON for visualization.
"""

import copy
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandapower as pp
import pandapower.topology as top
import networkx as nx

from src.converter.pandapower_builder import PandapowerBuilder
from src.converter.matpower_exporter import MATPOWERExporter
from src.powerflow.load_estimator import estimate_loads, load_demand_config
from src.powerflow.powerflow_runner import run_powerflow, PowerFlowResult
from src.server.geojson_parser import build_grid_network

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/Inf with default."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# Cache built networks: region -> (net, grid_network, build_result)
_net_cache: Dict[str, Tuple[Any, Any, Any]] = {}


def _fix_topology(net: Any) -> None:
    """Deactivate isolated components, keep only the largest connected component."""
    try:
        mg = top.create_nxgraph(net, respect_switches=False)
        components = list(nx.connected_components(mg))
        if len(components) <= 1:
            return

        largest = max(components, key=len)
        isolated = set()
        for comp in components:
            if comp != largest:
                isolated.update(comp)

        for bus_idx in isolated:
            if bus_idx in net.bus.index:
                net.bus.at[bus_idx, "in_service"] = False

        for table_name in ("load", "gen", "line"):
            table = getattr(net, table_name, None)
            if table is None or table.empty:
                continue
            if table_name == "line":
                mask = table["from_bus"].isin(isolated) | table["to_bus"].isin(isolated)
            else:
                mask = table["bus"].isin(isolated)
            table.loc[mask, "in_service"] = False

        if not net.ext_grid.empty:
            mask = net.ext_grid["bus"].isin(isolated)
            net.ext_grid.loc[mask, "in_service"] = False

        # Ensure at least one ext_grid
        if net.ext_grid["in_service"].sum() == 0 and len(net.ext_grid) > 0:
            for i, row in net.ext_grid.iterrows():
                if row["bus"] in largest:
                    net.ext_grid.at[i, "in_service"] = True
                    break
            else:
                bus_idx = next(iter(largest))
                pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="slack_recovery")

    except Exception:
        pass


def build_network(
    sub_fc: dict,
    line_fc: dict,
    region: str,
) -> Tuple[Any, Any, Any]:
    """Build a pandapower network from GeoJSON data.

    Returns:
        (pandapower_net, grid_network, build_result)
    """
    if region in _net_cache:
        return _net_cache[region]

    grid_network = build_grid_network(sub_fc, line_fc, region)
    builder = PandapowerBuilder()
    build_result = builder.build(grid_network)
    net = build_result.net

    # Fix zero-voltage buses
    zero_mask = net.bus["vn_kv"] == 0
    if zero_mask.any():
        non_zero = net.bus.loc[~zero_mask, "vn_kv"]
        if len(non_zero) > 0:
            net.bus.loc[zero_mask, "vn_kv"] = float(non_zero.median())

    _net_cache[region] = (net, grid_network, build_result)
    return net, grid_network, build_result


def run_powerflow_for_region(
    sub_fc: dict,
    line_fc: dict,
    region: str,
    mode: str = "dc",
    load_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """Run power flow on a region and return results.

    Args:
        sub_fc: Substations GeoJSON FeatureCollection.
        line_fc: Lines GeoJSON FeatureCollection.
        region: Region identifier.
        mode: "dc" or "ac".
        load_factor: Override for load factor (0.0-1.0).

    Returns:
        Dict with summary, bus_results, line_results.
    """
    # Clear cache for fresh run
    _net_cache.pop(region, None)

    net, grid_network, build_result = build_network(sub_fc, line_fc, region)
    net = copy.deepcopy(net)

    # Load demand config, optionally override load factor
    demand_config = load_demand_config()
    if load_factor is not None:
        demand_config = {**demand_config, "load_factor": load_factor}

    # Allocate loads
    total_load = estimate_loads(net, region, demand_config=demand_config)

    # Fix topology
    _fix_topology(net)

    # Run power flow
    pf_result = run_powerflow(net, mode=mode)

    # Build response
    summary = {
        "converged": pf_result.converged,
        "mode": pf_result.mode,
        "total_loss_mw": round(pf_result.total_loss_mw, 2),
        "max_line_loading_pct": round(pf_result.max_line_loading_pct, 2),
        "total_load_mw": round(total_load, 2),
        "buses": int(net.bus["in_service"].sum()),
        "lines": int(net.line["in_service"].sum()),
        "warnings": pf_result.warnings[:10],
        "build_summary": build_result.summary,
    }

    return {
        "summary": summary,
        "net": net,
        "grid_network": grid_network,
        "build_result": build_result,
        "pf_result": pf_result,
    }


def results_to_bus_geojson(
    net: Any,
    grid_network: Any,
    build_result: Any,
    pf_result: PowerFlowResult,
) -> dict:
    """Convert bus results to GeoJSON for map visualization.

    Returns FeatureCollection with Point features, each carrying vm_pu and p_mw.
    """
    features = []
    if pf_result.res_bus is None:
        return {"type": "FeatureCollection", "features": features}

    # Use bus_map from build_result for reliable sub_id -> bus_idx mapping
    bus_map = build_result.bus_map
    sub_index = {s.id: s for s in grid_network.substations}

    for sub_id, bus_idx in bus_map.items():
        if bus_idx not in pf_result.res_bus.index:
            continue
        sub = sub_index.get(sub_id)
        if sub is None:
            continue

        res = pf_result.res_bus.loc[bus_idx]
        vm_pu = _safe_float(res.get("vm_pu", 1.0), 1.0)
        p_mw = _safe_float(res.get("p_mw", 0.0))

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [sub.longitude, sub.latitude],
            },
            "properties": {
                "name": sub.name,
                "id": sub.id,
                "voltage_kv": sub.voltage_kv,
                "vm_pu": round(vm_pu, 4),
                "p_mw": round(p_mw, 2),
                "bus_idx": int(bus_idx),
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def results_to_line_geojson(
    net: Any,
    grid_network: Any,
    build_result: Any,
    pf_result: PowerFlowResult,
) -> dict:
    """Convert line results to GeoJSON for map visualization.

    Returns FeatureCollection with LineString features carrying loading_percent.
    """
    features = []
    if pf_result.res_line is None:
        return {"type": "FeatureCollection", "features": features}

    # Use line_map from build_result for reliable line_id -> line_idx mapping
    line_map = build_result.line_map
    line_index = {l.id: l for l in grid_network.transmission_lines}
    sub_index = {s.id: s for s in grid_network.substations}

    for line_id, line_idx in line_map.items():
        if line_idx not in pf_result.res_line.index:
            continue
        line = line_index.get(line_id)
        if line is None:
            continue

        res = pf_result.res_line.loc[line_idx]
        loading = _safe_float(res.get("loading_percent", 0.0))
        p_from = _safe_float(res.get("p_from_mw", 0.0))

        # Build coordinates from the line's coordinate list
        coords = []
        if hasattr(line, "coordinates") and line.coordinates:
            # coordinates are (lat, lon), GeoJSON needs [lon, lat]
            coords = [[c[1], c[0]] for c in line.coordinates]
        else:
            # Fallback: straight line between substations
            from_sub = sub_index.get(line.from_substation_id)
            to_sub = sub_index.get(line.to_substation_id)
            if from_sub and to_sub:
                coords = [
                    [from_sub.longitude, from_sub.latitude],
                    [to_sub.longitude, to_sub.latitude],
                ]

        if len(coords) < 2:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "name": line.name,
                "id": line.id,
                "voltage_kv": line.voltage_kv,
                "length_km": line.length_km,
                "loading_percent": round(loading, 2),
                "p_from_mw": round(p_from, 2),
                "line_idx": int(line_idx),
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def export_matpower(
    sub_fc: dict,
    line_fc: dict,
    region: str,
) -> Optional[str]:
    """Export region to MATPOWER .mat file and return the file path.

    Returns:
        Path to the generated .mat file, or None on failure.
    """
    net, _, _ = build_network(sub_fc, line_fc, region)
    net = copy.deepcopy(net)

    # Allocate loads for the export
    estimate_loads(net, region)
    _fix_topology(net)

    exporter = MATPOWERExporter()
    result = exporter.export_region(net, region)
    if result.success:
        return result.mat_path
    return None
