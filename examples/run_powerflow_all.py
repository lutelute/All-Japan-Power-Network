#!/usr/bin/env python3
"""Run DC and AC power flow on all 10 Japanese regional grids.

Builds pandapower networks from OSM-derived GeoJSON data, runs power flow,
and produces a summary dashboard with voltage profiles and line loading.

Usage::

    PYTHONPATH=. python examples/run_powerflow_all.py
"""

import copy
import json
import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import yaml

# Japanese font support
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.converter.pandapower_builder import PandapowerBuilder
from src.model.grid_network import GridNetwork
from src.model.generator import Generator
from src.model.substation import Substation
from src.model.transmission_line import TransmissionLine
from src.powerflow.load_estimator import estimate_loads, load_demand_config, scale_generation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "powerflow_regional")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

REGION_FREQ = {
    "hokkaido": 50, "tohoku": 50, "tokyo": 50,
    "chubu": 60, "hokuriku": 60, "kansai": 60,
    "chugoku": 60, "shikoku": 60, "kyushu": 60, "okinawa": 60,
}

# Default capacity estimates (MW) by fuel type when capacity_mw is missing
_DEFAULT_CAPACITY_MW = {
    "nuclear": 900, "coal": 600, "gas": 400, "oil": 200,
    "oil;gas": 300, "gas;oil": 300, "coal;gas": 400, "gas;coal": 400,
    "coal;gas;oil": 400,
    "hydro": 30, "wind": 20, "solar": 10, "geothermal": 30,
    "biomass": 20, "waste": 5,
}
_DEFAULT_CAPACITY_FALLBACK = 10.0

# Transformer parameters by voltage pair (hv_kv, lv_kv)
_TRAFO_PARAMS = {
    (500, 275): {"sn_mva": 1000, "vk_percent": 12.0, "vkr_percent": 0.25, "pfe_kw": 200, "i0_percent": 0.05},
    (500, 220): {"sn_mva": 800, "vk_percent": 12.0, "vkr_percent": 0.25, "pfe_kw": 180, "i0_percent": 0.05},
    (500, 187): {"sn_mva": 700, "vk_percent": 12.0, "vkr_percent": 0.3, "pfe_kw": 160, "i0_percent": 0.06},
    (500, 154): {"sn_mva": 600, "vk_percent": 12.0, "vkr_percent": 0.3, "pfe_kw": 150, "i0_percent": 0.06},
    (275, 154): {"sn_mva": 500, "vk_percent": 10.0, "vkr_percent": 0.3, "pfe_kw": 120, "i0_percent": 0.06},
    (275, 132): {"sn_mva": 400, "vk_percent": 10.0, "vkr_percent": 0.35, "pfe_kw": 100, "i0_percent": 0.07},
    (275, 110): {"sn_mva": 350, "vk_percent": 10.0, "vkr_percent": 0.35, "pfe_kw": 90, "i0_percent": 0.07},
    (275, 66):  {"sn_mva": 300, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 80, "i0_percent": 0.08},
    (275, 77):  {"sn_mva": 300, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 80, "i0_percent": 0.08},
    (220, 110): {"sn_mva": 300, "vk_percent": 10.0, "vkr_percent": 0.35, "pfe_kw": 80, "i0_percent": 0.07},
    (220, 66):  {"sn_mva": 250, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 70, "i0_percent": 0.08},
    (187, 66):  {"sn_mva": 200, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 60, "i0_percent": 0.08},
    (154, 66):  {"sn_mva": 200, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 60, "i0_percent": 0.08},
    (154, 77):  {"sn_mva": 200, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 60, "i0_percent": 0.08},
    (132, 66):  {"sn_mva": 150, "vk_percent": 8.0, "vkr_percent": 0.5, "pfe_kw": 40, "i0_percent": 0.1},
    (110, 66):  {"sn_mva": 150, "vk_percent": 8.0, "vkr_percent": 0.5, "pfe_kw": 40, "i0_percent": 0.1},
    (77, 66):   {"sn_mva": 100, "vk_percent": 8.0, "vkr_percent": 0.5, "pfe_kw": 30, "i0_percent": 0.1},
}


# ── GeoJSON → GridNetwork conversion ─────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _get_centroid(feature):
    geom = feature["geometry"]
    if geom is None:
        return None, None
    gtype = geom["type"]
    if gtype == "Point":
        return geom["coordinates"][1], geom["coordinates"][0]
    elif gtype == "Polygon":
        coords = geom["coordinates"][0]
    elif gtype == "MultiPolygon":
        coords = geom["coordinates"][0][0]
    else:
        return None, None
    lat = sum(c[1] for c in coords) / len(coords)
    lon = sum(c[0] for c in coords) / len(coords)
    return lat, lon


def _parse_voltage_kv(voltage_raw):
    """Parse OSM voltage string (in volts) to kV.

    Handles semicolon-separated (154000;66000) and comma-separated
    (77000,6600) multi-voltage strings by taking the highest value.
    """
    if not voltage_raw:
        return 0.0
    s = str(voltage_raw).strip()
    # Split on both ; and , (OSM uses both as voltage separators)
    parts = s.replace(",", ";").split(";")
    best_kv = 0.0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except (ValueError, TypeError):
            continue
        kv = v / 1000 if v > 1000 else v
        if kv > best_kv:
            best_kv = kv
    return best_kv


def _get_line_coords(feature):
    """Extract coordinates from a LineString or MultiLineString."""
    geom = feature.get("geometry")
    if not geom:
        return []
    gtype = geom["type"]
    if gtype == "LineString":
        return [(c[1], c[0]) for c in geom["coordinates"]]
    elif gtype == "MultiLineString":
        return [(c[1], c[0]) for c in geom["coordinates"][0]]
    return []


def _find_nearest_sub(lat, lon, sub_coords, max_km):
    """Find nearest substation within max_km."""
    best_id = None
    best_dist = float("inf")
    for slat, slon, sid in sub_coords:
        if abs(slat - lat) > 0.5:  # quick filter
            continue
        d = _haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_id = sid
    return best_id if best_dist <= max_km else None


def build_network_from_geojson(region):
    """Build a GridNetwork from OSM GeoJSON files for a region."""
    freq = REGION_FREQ.get(region, 50)
    network = GridNetwork(region=region, frequency_hz=freq)

    # Load substations
    sub_path = os.path.join(DATA_DIR, f"{region}_substations.geojson")
    if not os.path.exists(sub_path):
        return None
    with open(sub_path, encoding="utf-8") as f:
        subs_data = json.load(f)

    sub_id_map = {}  # feature index → substation id
    for i, feat in enumerate(subs_data["features"]):
        lat, lon = _get_centroid(feat)
        if lat is None:
            continue
        props = feat["properties"]
        name = props.get("name") or f"{region}_sub_{i}"
        voltage_kv = _parse_voltage_kv(props.get("voltage"))
        sub_id = f"{region}_sub_{i}"
        sub_id_map[i] = sub_id

        sub = Substation(
            id=sub_id,
            name=name,
            region=region,
            latitude=lat,
            longitude=lon,
            voltage_kv=max(voltage_kv, 0),
        )
        network.add_substation(sub)

    # Build spatial index of substations for endpoint matching
    sub_coords = []
    for sub in network.substations:
        sub_coords.append((sub.latitude, sub.longitude, sub.id))

    # Load lines and match endpoints to nearest substations
    lines_path = os.path.join(DATA_DIR, f"{region}_lines.geojson")
    if os.path.exists(lines_path):
        with open(lines_path, encoding="utf-8") as f:
            lines_data = json.load(f)

        for i, feat in enumerate(lines_data["features"]):
            props = feat["properties"]
            name = props.get("name") or props.get("_display_name") or f"{region}_line_{i}"
            voltage_kv = _parse_voltage_kv(props.get("voltage"))

            coords = _get_line_coords(feat)
            if len(coords) < 2:
                continue

            start_lat, start_lon = coords[0]
            end_lat, end_lon = coords[-1]

            from_sub_id = _find_nearest_sub(start_lat, start_lon, sub_coords, 50.0)
            to_sub_id = _find_nearest_sub(end_lat, end_lon, sub_coords, 50.0)

            if not from_sub_id or not to_sub_id or from_sub_id == to_sub_id:
                continue

            length_km = 0.0
            for j in range(1, len(coords)):
                length_km += _haversine_km(coords[j-1][0], coords[j-1][1],
                                            coords[j][0], coords[j][1])

            if length_km <= 0:
                continue

            line_id = f"{region}_line_{i}"
            line = TransmissionLine(
                id=line_id,
                name=name,
                from_substation_id=from_sub_id,
                to_substation_id=to_sub_id,
                voltage_kv=max(voltage_kv, 0),
                length_km=length_km,
                region=region,
            )
            try:
                network.add_transmission_line(line)
            except ValueError:
                pass  # duplicate ID, skip

    # Step 1: Load generators from plants GeoJSON
    plants_path = os.path.join(DATA_DIR, f"{region}_plants.geojson")
    if os.path.exists(plants_path):
        with open(plants_path, encoding="utf-8") as f:
            plants_data = json.load(f)

        gen_count = 0
        for i, feat in enumerate(plants_data["features"]):
            lat, lon = _get_centroid(feat)
            if lat is None:
                continue
            props = feat["properties"]

            # Extract capacity
            capacity_mw = None
            raw_cap = props.get("capacity_mw")
            if raw_cap is not None:
                try:
                    capacity_mw = float(raw_cap)
                except (ValueError, TypeError):
                    pass

            fuel = props.get("plant:source") or props.get("fuel_type") or "unknown"
            # Clean fuel string (some have URLs)
            if fuel.startswith("http"):
                fuel = "unknown"

            if capacity_mw is None or capacity_mw <= 0:
                capacity_mw = _DEFAULT_CAPACITY_MW.get(fuel, _DEFAULT_CAPACITY_FALLBACK)

            # Match to nearest substation bus (< 5km)
            nearest_sub = _find_nearest_sub(lat, lon, sub_coords, 5.0)
            if not nearest_sub:
                # Relax to 20km for large plants
                if capacity_mw >= 100:
                    nearest_sub = _find_nearest_sub(lat, lon, sub_coords, 20.0)
                if not nearest_sub:
                    continue

            name = props.get("name") or props.get("_display_name") or f"{region}_plant_{i}"
            gen_id = f"{region}_gen_{i}"

            gen = Generator(
                id=gen_id,
                name=name,
                capacity_mw=capacity_mw,
                fuel_type=fuel,
                connected_bus_id=nearest_sub,
                region=region,
                latitude=lat,
                longitude=lon,
            )
            network.add_generator(gen)
            gen_count += 1

    return network


# ── Post-build network fixes ─────────────────────────────────────────────────

def fix_zero_voltages(net):
    """Fix buses with vn_kv=0 using connected line voltages or defaults."""
    zero_mask = net.bus["vn_kv"] <= 0
    n_zero = int(zero_mask.sum())
    if n_zero == 0:
        return 0

    # Infer from connected lines
    for idx in net.bus.index[zero_mask]:
        connected_lines = net.line[(net.line["from_bus"] == idx) | (net.line["to_bus"] == idx)]
        if connected_lines.empty:
            continue
        # Get voltage from the other end of connected lines
        voltages = []
        for _, line_row in connected_lines.iterrows():
            other_bus = line_row["to_bus"] if line_row["from_bus"] == idx else line_row["from_bus"]
            v = net.bus.at[other_bus, "vn_kv"]
            if v > 0:
                voltages.append(v)
        if voltages:
            net.bus.at[idx, "vn_kv"] = max(voltages)

    # Remaining zero-voltage buses: use median or 66 kV default
    still_zero = net.bus["vn_kv"] <= 0
    if still_zero.any():
        nonzero = net.bus.loc[~still_zero, "vn_kv"]
        fallback = float(nonzero.median()) if len(nonzero) > 0 else 66.0
        net.bus.loc[still_zero, "vn_kv"] = fallback

    fixed = n_zero - int((net.bus["vn_kv"] <= 0).sum())
    return fixed


def _snap_voltage(vn_kv):
    """Snap a voltage to the nearest standard Japanese voltage class."""
    classes = [500, 275, 220, 187, 154, 132, 110, 77, 66]
    if vn_kv <= 0:
        return 66
    return min(classes, key=lambda c: abs(c - vn_kv))


def _get_trafo_params(hv_kv, lv_kv):
    """Get transformer parameters for a voltage pair, with fallback."""
    hv = _snap_voltage(max(hv_kv, lv_kv))
    lv = _snap_voltage(min(hv_kv, lv_kv))
    if hv == lv:
        return None

    key = (hv, lv)
    if key in _TRAFO_PARAMS:
        return _TRAFO_PARAMS[key]

    # Find closest match
    best_key = None
    best_dist = float("inf")
    for k in _TRAFO_PARAMS:
        d = abs(k[0] - hv) + abs(k[1] - lv)
        if d < best_dist:
            best_dist = d
            best_key = k
    if best_key:
        return _TRAFO_PARAMS[best_key]

    # Generic fallback
    return {"sn_mva": 200, "vk_percent": 10.0, "vkr_percent": 0.5, "pfe_kw": 50, "i0_percent": 0.1}


def insert_transformers(net):
    """Replace lines connecting buses at different voltages with transformers."""
    lines_to_remove = []
    trafos_created = 0

    for idx in net.line.index:
        from_bus = net.line.at[idx, "from_bus"]
        to_bus = net.line.at[idx, "to_bus"]
        vn_from = net.bus.at[from_bus, "vn_kv"]
        vn_to = net.bus.at[to_bus, "vn_kv"]

        # Check if voltage ratio > 1.2 (same class lines may have small differences)
        ratio = max(vn_from, vn_to) / max(min(vn_from, vn_to), 0.1)
        if ratio < 1.2:
            continue

        hv_kv = max(vn_from, vn_to)
        lv_kv = min(vn_from, vn_to)
        hv_bus = from_bus if vn_from >= vn_to else to_bus
        lv_bus = to_bus if vn_from >= vn_to else from_bus

        params = _get_trafo_params(hv_kv, lv_kv)
        if params is None:
            continue

        pp.create_transformer_from_parameters(
            net,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            sn_mva=params["sn_mva"],
            vn_hv_kv=hv_kv,
            vn_lv_kv=lv_kv,
            vk_percent=params["vk_percent"],
            vkr_percent=params["vkr_percent"],
            pfe_kw=params["pfe_kw"],
            i0_percent=params["i0_percent"],
            name=f"trafo_{hv_kv:.0f}/{lv_kv:.0f}kV",
        )
        lines_to_remove.append(idx)
        trafos_created += 1

    # Remove replaced lines
    if lines_to_remove:
        net.line = net.line.drop(lines_to_remove)

    return trafos_created


def select_slack_bus(net):
    """Select optimal slack bus: well-connected high-voltage bus with generation."""
    active_buses = net.bus[net.bus["in_service"]]
    if active_buses.empty:
        return None

    # Count connections per bus (lines + trafos)
    connectivity = {}
    for idx in active_buses.index:
        n_conn = 0
        if len(net.line) > 0:
            n_conn += ((net.line["from_bus"] == idx) | (net.line["to_bus"] == idx)).sum()
        if len(net.trafo) > 0:
            n_conn += ((net.trafo["hv_bus"] == idx) | (net.trafo["lv_bus"] == idx)).sum()
        connectivity[idx] = n_conn

    # Aggregate generation per bus
    gen_at_bus = {}
    if len(net.gen) > 0:
        active_gens = net.gen[net.gen["in_service"]]
        for gen_idx in active_gens.index:
            bus = active_gens.at[gen_idx, "bus"]
            if bus in active_buses.index:
                gen_at_bus[bus] = gen_at_bus.get(bus, 0) + active_gens.at[gen_idx, "p_mw"]

    # Score: heavily weight voltage level and connectivity, plus generation
    best_bus = None
    best_score = -1
    for bus_idx in active_buses.index:
        vn_kv = active_buses.at[bus_idx, "vn_kv"]
        conn = connectivity.get(bus_idx, 0)
        gen_mw = gen_at_bus.get(bus_idx, 0)
        # Voltage dominates (500kV >> 66kV), connectivity matters, gen is bonus
        score = vn_kv * 10 + conn * 50 + gen_mw * 0.1
        if score > best_score:
            best_score = score
            best_bus = bus_idx

    if best_bus is not None and len(net.ext_grid) > 0:
        net.ext_grid.at[net.ext_grid.index[0], "bus"] = best_bus

    return best_bus


def fix_topology(net):
    """Fix isolated components and return diagnostics."""
    mg = top.create_nxgraph(net, respect_switches=False)
    components = list(nx.connected_components(mg))
    diag = {
        "n_components": len(components),
        "n_isolated_buses": 0,
        "n_active_buses": int(net.bus["in_service"].sum()),
    }

    if len(components) > 1:
        largest = max(components, key=len)
        isolated = set()
        for comp in components:
            if comp != largest:
                isolated.update(comp)
        diag["n_isolated_buses"] = len(isolated)

        for bus_idx in isolated:
            if bus_idx in net.bus.index:
                net.bus.at[bus_idx, "in_service"] = False
        for tbl in ("load", "gen", "sgen", "line", "trafo"):
            table = getattr(net, tbl, None)
            if table is None or table.empty:
                continue
            if tbl in ("line", "trafo"):
                from_col = "hv_bus" if tbl == "trafo" else "from_bus"
                to_col = "lv_bus" if tbl == "trafo" else "to_bus"
                mask = table[from_col].isin(isolated) | table[to_col].isin(isolated)
            else:
                mask = table["bus"].isin(isolated)
            table.loc[mask, "in_service"] = False
        if not net.ext_grid.empty:
            mask = net.ext_grid["bus"].isin(isolated)
            net.ext_grid.loc[mask, "in_service"] = False
            if net.ext_grid["in_service"].sum() == 0:
                for i, row in net.ext_grid.iterrows():
                    if row["bus"] in largest:
                        net.ext_grid.at[i, "in_service"] = True
                        break
                else:
                    bus_idx = next(iter(largest))
                    pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="slack_recovery")

        diag["n_active_buses"] = int(net.bus["in_service"].sum())

    return diag


def prune_dc_infeasible(net, angle_threshold=45.0):
    """After DC power flow, remove lines/trafos with extreme angle differences.

    Lines with large angle differences across them represent bottlenecks
    that will prevent AC convergence. Iteratively prune and re-run DC
    until the network is clean.
    """
    total_removed = 0
    for _iteration in range(5):  # max 5 rounds
        net_tmp = copy.deepcopy(net)
        try:
            pp.rundcpp(net_tmp)
        except Exception:
            break

        removed_this_round = 0

        # Check lines
        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue
            from_bus = net.line.at[idx, "from_bus"]
            to_bus = net.line.at[idx, "to_bus"]
            if from_bus in net_tmp.res_bus.index and to_bus in net_tmp.res_bus.index:
                angle_diff = abs(
                    net_tmp.res_bus.at[from_bus, "va_degree"]
                    - net_tmp.res_bus.at[to_bus, "va_degree"]
                )
                if angle_diff > angle_threshold:
                    net.line.at[idx, "in_service"] = False
                    removed_this_round += 1

        # Check trafos
        for idx in net.trafo.index:
            if not net.trafo.at[idx, "in_service"]:
                continue
            hv_bus = net.trafo.at[idx, "hv_bus"]
            lv_bus = net.trafo.at[idx, "lv_bus"]
            if hv_bus in net_tmp.res_bus.index and lv_bus in net_tmp.res_bus.index:
                angle_diff = abs(
                    net_tmp.res_bus.at[hv_bus, "va_degree"]
                    - net_tmp.res_bus.at[lv_bus, "va_degree"]
                )
                if angle_diff > angle_threshold:
                    net.trafo.at[idx, "in_service"] = False
                    removed_this_round += 1

        total_removed += removed_this_round
        if removed_this_round == 0:
            break

    return total_removed


def scale_line_ratings(net):
    """Scale line and transformer ratings to prevent unrealistic overloading.

    The synthetic network has limited topology, so a few lines carry
    disproportionate power.  Scale max_i_ka and trafo sn_mva so that
    the network is physically feasible.
    """
    # Run a quick DC to estimate flows
    import copy as _copy
    net_tmp = _copy.deepcopy(net)
    try:
        pp.rundcpp(net_tmp)
    except Exception:
        return

    # Scale lines
    if len(net_tmp.res_line) > 0 and "loading_percent" in net_tmp.res_line.columns:
        for idx in net.line.index:
            if idx in net_tmp.res_line.index:
                loading = net_tmp.res_line.at[idx, "loading_percent"]
                if loading > 100:
                    # Scale up capacity to bring loading to ~80%
                    factor = loading / 80.0
                    net.line.at[idx, "max_i_ka"] = net.line.at[idx, "max_i_ka"] * factor

    # Scale transformers
    if len(net_tmp.res_trafo) > 0 and "loading_percent" in net_tmp.res_trafo.columns:
        for idx in net.trafo.index:
            if idx in net_tmp.res_trafo.index:
                loading = net_tmp.res_trafo.at[idx, "loading_percent"]
                if loading > 100:
                    factor = loading / 80.0
                    net.trafo.at[idx, "sn_mva"] = net.trafo.at[idx, "sn_mva"] * factor


def balance_power(net, demand_config):
    """Ensure generation roughly matches load for convergence."""
    if len(net.load) == 0:
        return

    # Only count active loads and gens
    active_load = net.load[net.load["in_service"]]["p_mw"].sum()
    if active_load <= 0:
        return

    reserve_margin = demand_config.get("reserve_margin", 0.05)
    target_gen = active_load * (1 + reserve_margin)

    if len(net.gen) > 0:
        # Disable generators on out-of-service buses
        inactive_buses = set(net.bus.index[~net.bus["in_service"]])
        gen_inactive_mask = net.gen["bus"].isin(inactive_buses)
        net.gen.loc[gen_inactive_mask, "in_service"] = False

        active_gens = net.gen[net.gen["in_service"]]
        total_capacity = active_gens["max_p_mw"].sum() if "max_p_mw" in active_gens.columns else active_gens["p_mw"].sum()

        if total_capacity > 0:
            scale = min(target_gen / total_capacity, 1.0)
            net.gen.loc[net.gen["in_service"], "p_mw"] = net.gen.loc[net.gen["in_service"], "max_p_mw"] * scale


# ── Power flow execution ─────────────────────────────────────────────────────

def run_powerflow(net, mode="dc"):
    """Run DC or AC power flow and return results."""
    result = {"mode": mode, "converged": False}
    try:
        if mode == "dc":
            pp.rundcpp(net)
            result["converged"] = True
        else:
            # Solver fallback chain
            solvers = [
                {"algorithm": "nr", "init": "dc", "max_iteration": 100, "tolerance_mva": 1e-2},
                {"algorithm": "nr", "init": "flat", "max_iteration": 100, "tolerance_mva": 1e-2},
                {"algorithm": "nr", "init": "dc", "max_iteration": 200, "tolerance_mva": 1e-1},
                {"algorithm": "nr", "init": "dc", "max_iteration": 300, "tolerance_mva": 1.0},
                {"algorithm": "nr", "init": "dc", "max_iteration": 300, "tolerance_mva": 10.0},
                {"algorithm": "fdbx", "init": "dc", "max_iteration": 200, "tolerance_mva": 1.0},
                {"algorithm": "fdxb", "init": "dc", "max_iteration": 200, "tolerance_mva": 1.0},
                {"algorithm": "gs", "init": "dc", "max_iteration": 5000, "tolerance_mva": 10.0},
            ]
            last_err = ""
            for solver_opts in solvers:
                try:
                    pp.runpp(net, numba=False, **solver_opts)
                    if net.converged:
                        result["converged"] = True
                        result["solver"] = solver_opts["algorithm"]
                        break
                except Exception as e:
                    last_err = f"{solver_opts['algorithm']}: {str(e)[:60]}"
                    continue
            if not result["converged"]:
                result["error"] = last_err or "all solvers failed to converge"

        if result.get("converged") or mode == "dc":
            if hasattr(net, "res_line") and len(net.res_line) > 0:
                active = net.line["in_service"]
                res = net.res_line.loc[active]
                result["total_loss_mw"] = float(res["pl_mw"].sum()) if "pl_mw" in res.columns else 0.0
                result["max_loading_pct"] = float(res["loading_percent"].max()) if "loading_percent" in res.columns else 0.0
                result["mean_loading_pct"] = float(res["loading_percent"].mean()) if "loading_percent" in res.columns else 0.0
            if hasattr(net, "res_bus"):
                active_bus = net.bus["in_service"]
                res_bus = net.res_bus.loc[active_bus]
                result["vm_pu_mean"] = float(res_bus["vm_pu"].mean())
                result["vm_pu_min"] = float(res_bus["vm_pu"].min())
                result["vm_pu_max"] = float(res_bus["vm_pu"].max())
                result["va_deg_min"] = float(res_bus["va_degree"].min())
                result["va_deg_max"] = float(res_bus["va_degree"].max())
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ── Dashboard ─────────────────────────────────────────────────────────────────

def plot_dashboard(all_results):
    """Create a summary dashboard figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Japan Regional Power Flow Analysis (OSM GeoJSON)\n"
                 "日本電力系統 地域別潮流計算結果（OSMデータ）",
                 fontsize=16, fontweight="bold", y=0.98)

    regions_sorted = [r for r in REGIONS if r in all_results]
    n = len(regions_sorted)
    x = np.arange(n)
    labels = [f"{REGION_JA[r]}\n{r.title()}" for r in regions_sorted]

    # --- Panel 1: Network size ---
    ax = axes[0, 0]
    buses = [all_results[r]["n_buses"] for r in regions_sorted]
    lines = [all_results[r]["n_lines"] for r in regions_sorted]
    gens = [all_results[r].get("n_gens", 0) for r in regions_sorted]
    trafos = [all_results[r].get("n_trafos", 0) for r in regions_sorted]
    w = 0.2
    ax.bar(x - 1.5*w, buses, w, label="Buses", color="#2196F3", alpha=0.8)
    ax.bar(x - 0.5*w, lines, w, label="Lines", color="#FF9800", alpha=0.8)
    ax.bar(x + 0.5*w, gens, w, label="Gens", color="#4CAF50", alpha=0.8)
    ax.bar(x + 1.5*w, trafos, w, label="Trafos", color="#9C27B0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("(a) Network Size — バス・送電線・発電機・変圧器数")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: DC power flow - bus voltage angle ---
    ax = axes[0, 1]
    for i, r in enumerate(regions_sorted):
        dc = all_results[r].get("dc")
        if dc and dc.get("converged"):
            va_min = dc.get("va_deg_min", 0)
            va_max = dc.get("va_deg_max", 0)
            ax.barh(i, va_max - va_min, left=va_min, height=0.6, color="#4CAF50", alpha=0.7)
            ax.plot([va_min, va_max], [i, i], "k-", linewidth=1.5)
        else:
            ax.barh(i, 0, height=0.6, color="#FFCDD2")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Voltage Angle (degrees)")
    ax.set_title("(b) DC Power Flow — Bus Voltage Angle Range")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 3: Line loading ---
    ax = axes[1, 0]
    dc_loading = []
    dc_labels_line = []
    for r in regions_sorted:
        dc = all_results[r].get("dc")
        if dc and dc.get("converged"):
            dc_loading.append(dc.get("max_loading_pct", 0))
            dc_labels_line.append(REGION_JA[r])
        else:
            dc_loading.append(0)
            dc_labels_line.append(f"{REGION_JA[r]} (N/C)")
    if dc_loading:
        colors_bar = ["#F44336" if v > 100 else "#FF9800" if v > 80 else "#4CAF50" for v in dc_loading]
        ax.barh(range(len(dc_loading)), dc_loading, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(dc_loading)))
        ax.set_yticklabels(dc_labels_line, fontsize=9)
        ax.axvline(100, color="red", linestyle="--", linewidth=1, label="100% limit")
        ax.axvline(80, color="orange", linestyle="--", linewidth=1, label="80% warning")
    ax.set_xlabel("Max Line Loading (%)")
    ax.set_title("(c) DC Power Flow — Maximum Line Loading")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 4: Convergence summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    headers = ["Region", "Buses", "Lines", "Gens", "Trafos", "DC", "AC", "Solver"]
    for r in regions_sorted:
        d = all_results[r]
        dc = d.get("dc", {})
        ac = d.get("ac", {})
        table_data.append([
            f"{REGION_JA[r]} ({r})",
            str(d["n_buses"]),
            str(d["n_lines"]),
            str(d.get("n_gens", 0)),
            str(d.get("n_trafos", 0)),
            "OK" if dc.get("converged") else "FAIL",
            "OK" if ac.get("converged") else "FAIL",
            ac.get("solver", "-") if ac.get("converged") else "-",
        ])

    tbl = ax.table(cellText=table_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = tbl[i + 1, j]
            if val == "OK":
                cell.set_facecolor("#C8E6C9")
            elif val == "FAIL":
                cell.set_facecolor("#FFCDD2")
    ax.set_title("(d) Convergence Summary — 収束結果一覧", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "regional_powerflow_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nDashboard saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    demand_cfg = load_demand_config()
    all_results = {}

    for region in REGIONS:
        print(f"\n{'='*60}")
        print(f"  {REGION_JA[region]} ({region})")
        print(f"{'='*60}")

        # Build network from GeoJSON (includes generators now)
        network = build_network_from_geojson(region)
        if not network or not network.has_elements:
            print(f"  SKIP: no GeoJSON data for {region}")
            continue

        print(f"  GeoJSON: {network.substation_count} substations, "
              f"{network.line_count} lines, "
              f"{network.generator_count} generators "
              f"({network.total_generation_mw:.0f} MW)")

        # Convert to pandapower
        builder = PandapowerBuilder()
        build_result = builder.build(network)
        net = build_result.net

        # Step 2: Fix zero-voltage buses
        n_fixed_v = fix_zero_voltages(net)
        if n_fixed_v > 0:
            print(f"  Fixed {n_fixed_v} zero-voltage buses")

        # Verify no zero-voltage buses remain
        assert (net.bus["vn_kv"] > 0).all(), "Zero-voltage buses remain!"

        # Step 3: Insert transformers at voltage boundaries
        n_trafos = insert_transformers(net)

        n_buses = len(net.bus)
        n_lines = len(net.line)
        n_gens = len(net.gen)
        print(f"  pandapower: {n_buses} buses, {n_lines} lines, "
              f"{n_gens} gens, {n_trafos} trafos, "
              f"{len(build_result.warnings)} warnings")

        # Step 5: Fix topology (keep largest component)
        diag = fix_topology(net)
        print(f"  Components: {diag['n_components']}, "
              f"isolated: {diag['n_isolated_buses']}, "
              f"active: {diag['n_active_buses']}")

        # Step 4: Select optimal slack bus
        slack_bus = select_slack_bus(net)
        if slack_bus is not None:
            slack_name = net.bus.at[slack_bus, "name"]
            slack_vn = net.bus.at[slack_bus, "vn_kv"]
            print(f"  Slack bus: {slack_bus} ({slack_name}, {slack_vn:.0f} kV)")

        # Disable loads on out-of-service buses, then estimate
        # First, remove any pre-existing loads (shouldn't be any)
        # estimate_loads creates on all buses; we'll fix after
        total_load = estimate_loads(net, region=region, demand_config=demand_cfg)
        # Disable loads on out-of-service buses
        if len(net.load) > 0:
            inactive_buses = set(net.bus.index[~net.bus["in_service"]])
            mask = net.load["bus"].isin(inactive_buses)
            net.load.loc[mask, "in_service"] = False
            active_loads = net.load[net.load["in_service"]]
            total_load = active_loads["p_mw"].sum()
        print(f"  Loads allocated: {total_load:.0f} MW across "
              f"{net.load['in_service'].sum()} active buses")

        # Step 6: Balance generation to load
        balance_power(net, demand_cfg)
        total_gen = net.gen["p_mw"].sum() if len(net.gen) > 0 else 0
        print(f"  Generation: {total_gen:.0f} MW ({len(net.gen)} units)")

        # Scale line/trafo ratings to prevent bottlenecks
        scale_line_ratings(net)

        # Set initial flat voltage profile and enforce gen voltage setpoints
        net.bus["vm_pu"] = 1.0
        if len(net.gen) > 0:
            net.gen["vm_pu"] = 1.0
        if len(net.ext_grid) > 0:
            net.ext_grid["vm_pu"] = 1.0

        # DC power flow
        net_dc = copy.deepcopy(net)
        dc_result = run_powerflow(net_dc, "dc")
        if dc_result["converged"]:
            print(f"  DC: converged, loss={dc_result.get('total_loss_mw', 0):.1f} MW, "
                  f"max_loading={dc_result.get('max_loading_pct', 0):.1f}%, "
                  f"angle=[{dc_result.get('va_deg_min', 0):.1f}, "
                  f"{dc_result.get('va_deg_max', 0):.1f}] deg")
        else:
            print(f"  DC: FAILED — {dc_result.get('error', 'unknown')}")

        # AC power flow (with solver fallback chain)
        # Try progressively tighter pruning until convergence
        ac_result = {"mode": "ac", "converged": False}
        for prune_threshold in [45.0, 30.0, 20.0]:
            net_ac = copy.deepcopy(net)
            n_pruned = prune_dc_infeasible(net_ac, angle_threshold=prune_threshold)
            if n_pruned > 0:
                diag_ac = fix_topology(net_ac)
                select_slack_bus(net_ac)
                scale_line_ratings(net_ac)
                print(f"  AC prep (threshold={prune_threshold}°): pruned {n_pruned} branches, "
                      f"{diag_ac['n_active_buses']} active buses remain")
            ac_result = run_powerflow(net_ac, "ac")
            if ac_result["converged"]:
                break
        if ac_result["converged"]:
            print(f"  AC: converged ({ac_result.get('solver','?')}), "
                  f"loss={ac_result.get('total_loss_mw', 0):.1f} MW, "
                  f"V=[{ac_result.get('vm_pu_min', 0):.4f}, "
                  f"{ac_result.get('vm_pu_max', 0):.4f}] pu")
        else:
            print(f"  AC: FAILED — {ac_result.get('error', 'unknown')[:80]}")

        all_results[region] = {
            "n_buses": n_buses,
            "n_lines": n_lines,
            "n_gens": n_gens,
            "n_trafos": n_trafos,
            "n_active_buses": diag["n_active_buses"],
            "topology": diag,
            "dc": dc_result,
            "ac": ac_result,
        }

    # Generate dashboard
    if all_results:
        print(f"\n{'='*60}")
        print("  Generating dashboard...")
        print(f"{'='*60}")
        plot_dashboard(all_results)

    # Print final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — 全地域潮流計算結果")
    print(f"{'='*60}")
    total_buses = sum(r["n_buses"] for r in all_results.values())
    total_lines = sum(r["n_lines"] for r in all_results.values())
    total_gens = sum(r.get("n_gens", 0) for r in all_results.values())
    total_trafos = sum(r.get("n_trafos", 0) for r in all_results.values())
    dc_ok = sum(1 for r in all_results.values() if r["dc"].get("converged"))
    ac_ok = sum(1 for r in all_results.values() if r["ac"].get("converged"))
    print(f"  Total: {total_buses} buses, {total_lines} lines, "
          f"{total_gens} gens, {total_trafos} trafos across "
          f"{len(all_results)} regions")
    print(f"  DC convergence: {dc_ok}/{len(all_results)}")
    print(f"  AC convergence: {ac_ok}/{len(all_results)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
