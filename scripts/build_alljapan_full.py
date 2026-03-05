"""Build FULL-SCALE MATPOWER cases for ALL 10 regions + All-Japan.

Data source: OpenStreetMap (Overpass API) GeoJSON
  - data/{region}_substations.geojson
  - data/{region}_lines.geojson
  - data/{region}_plants.geojson  (generators)

Strategy:
  - Keep ALL buses and lines (not just backbone)
  - Insert transformers at every voltage boundary
  - Bridge disconnected components via nearest-neighbor (estimated connections)
  - Clean corrupt voltage data
  - Full generator set from OSM plants + P03 enrichment

Outputs:
  output/matpower_alljapan/{region}.mat   (x10)
  output/matpower_alljapan/alljapan.mat
"""

import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandapower as pp
import yaml
from scipy.io import savemat
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.server.geojson_parser import build_grid_network
from src.converter.pandapower_builder import PandapowerBuilder
from src.model.generator import Generator
from src.utils.geo_utils import haversine_distance

OUT_DIR = os.path.join(PROJECT_ROOT, "output", "matpower_alljapan")
os.makedirs(OUT_DIR, exist_ok=True)

# Standard transmission voltage levels in Japan (kV) — exclude distribution (< 66 kV)
VALID_VOLTAGES = sorted([66, 77, 110, 132, 154, 187, 220, 275, 500])

# Impedance floors (per-unit on 100 MVA base)
MIN_X_PU = 0.005         # Minimum per-unit reactance for lines
MIN_X_TRAFO = 0.03       # Minimum per-unit reactance for transformers (higher = more realistic)
MIN_RX_RATIO = 0.05      # Minimum r/x ratio

# ── Region config ──
_regions_cfg = yaml.safe_load(
    open(os.path.join(PROJECT_ROOT, "config", "regions.yaml"), encoding="utf-8")
)
ALL_REGIONS = list(_regions_cfg.get("regions", {}).keys())


def _get_region_config(region):
    return _regions_cfg.get("regions", {}).get(region, {})


# ── OCCTO 2023 reference demand (MW) ──
_demand_cfg = yaml.safe_load(
    open(os.path.join(PROJECT_ROOT, "config", "regional_demand.yaml"), encoding="utf-8")
)
REFERENCE_DEMAND = _demand_cfg["regional_peak_demand_mw"]
LOAD_FACTOR = _demand_cfg.get("load_factor", 0.85)
POWER_FACTOR = _demand_cfg.get("power_factor", 0.95)
RESERVE_MARGIN = _demand_cfg.get("reserve_margin", 0.05)
VOLTAGE_WEIGHTS = _demand_cfg.get("voltage_weights", {})


def _clean_voltage(v_kv):
    """Snap to nearest standard transmission voltage (≥ 66 kV)."""
    if v_kv <= 0:
        v_kv = 66
    elif v_kv > 600:
        v_kv = v_kv % 1000 if v_kv > 1000 else 500
    return min(VALID_VOLTAGES, key=lambda x: abs(x - v_kv))


def _get_bus_coords(net, bus_idx):
    """Extract (lat, lon) from bus geo column."""
    if "geo" not in net.bus.columns:
        return None
    geo = net.bus.at[bus_idx, "geo"]
    if geo is None or (isinstance(geo, float) and np.isnan(geo)):
        return None
    if isinstance(geo, str):
        geo = json.loads(geo)
    if isinstance(geo, dict) and "coordinates" in geo:
        lon, lat = geo["coordinates"]
        return (lat, lon)
    return None


def _bridge_components(net):
    """Connect isolated components to the main component via nearest-bus bridging.

    Instead of dropping small clusters, this function creates estimated
    transmission lines or transformers between each disconnected component
    and the nearest bus in the main component.

    Bridge elements are named with 'estimated_bridge_' prefix for traceability.
    """
    mg = pp.topology.create_nxgraph(net, include_trafos=True)
    comps = list(pp.topology.connected_components(mg))
    if len(comps) <= 1:
        return 0

    comps.sort(key=len, reverse=True)
    main_comp = comps[0]

    # Build spatial index of main component buses
    main_coords = []
    main_indices = []
    for bus_idx in main_comp:
        coord = _get_bus_coords(net, bus_idx)
        if coord:
            main_coords.append(coord)
            main_indices.append(bus_idx)

    if not main_coords:
        return 0

    tree = cKDTree(main_coords)
    bridges_added = 0

    for comp in comps[1:]:
        # Find nearest main-component bus to any bus in this component
        best_dist = float("inf")
        best_orphan = None
        best_main = None

        for bus_idx in comp:
            coord = _get_bus_coords(net, bus_idx)
            if coord is None:
                continue
            dist, idx = tree.query(coord)
            if dist < best_dist:
                best_dist = dist
                best_orphan = bus_idx
                best_main = main_indices[idx]

        if best_orphan is None or best_main is None:
            continue

        # Skip if too far (> 300 km ≈ 2.7 degrees)
        if best_dist > 2.7:
            continue

        v_orphan = net.bus.at[best_orphan, "vn_kv"]
        v_main = net.bus.at[best_main, "vn_kv"]
        ratio = max(v_orphan, v_main) / min(v_orphan, v_main) if min(v_orphan, v_main) > 0 else 1

        # Estimate distance in km (rough: 1 degree ≈ 111 km)
        length_km = max(best_dist * 111.0, 0.1)

        if ratio < 1.5:
            # Same voltage — add line
            vn = v_main
            r_per_km = 0.06 if vn >= 220 else 0.12
            x_per_km = 0.3 if vn >= 220 else 0.4
            c_per_km = 10.0 if vn >= 220 else 8.0
            max_i = 2.0 if vn >= 220 else 1.0
            pp.create_line_from_parameters(
                net, from_bus=best_orphan, to_bus=best_main,
                length_km=length_km, r_ohm_per_km=r_per_km,
                x_ohm_per_km=x_per_km, c_nf_per_km=c_per_km,
                max_i_ka=max_i, name=f"estimated_bridge_{bridges_added}",
            )
        else:
            # Different voltage — add transformer
            hv, lv = max(v_orphan, v_main), min(v_orphan, v_main)
            hv_bus = best_orphan if v_orphan >= v_main else best_main
            lv_bus = best_main if v_orphan >= v_main else best_orphan
            if hv >= 275:
                sn, vk, vkr = 800.0, 14.0, 0.20
            elif hv >= 154:
                sn, vk, vkr = 400.0, 12.0, 0.25
            else:
                sn, vk, vkr = 200.0, 10.0, 0.30
            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus,
                sn_mva=sn, vn_hv_kv=hv, vn_lv_kv=lv,
                vkr_percent=vkr, vk_percent=vk,
                pfe_kw=30.0, i0_percent=0.05,
                name=f"estimated_bridge_trafo_{bridges_added}",
            )

        bridges_added += 1
        # Update main component
        main_comp = main_comp | comp
        for bus_idx in comp:
            coord = _get_bus_coords(net, bus_idx)
            if coord:
                main_coords.append(coord)
                main_indices.append(bus_idx)
        tree = cKDTree(main_coords)

    return bridges_added


def _load_plants_as_generators(region):
    """Load generators from OSM plants GeoJSON (+ P03 enrichment).

    Returns list of Generator objects with lat/lon but NOT yet matched
    to substations (connected_bus_id is empty).
    """
    plants_path = os.path.join(PROJECT_ROOT, "data", f"{region}_plants.geojson")
    if not os.path.exists(plants_path):
        return []

    with open(plants_path) as f:
        fc = json.load(f)

    generators = []
    for i, feat in enumerate(fc.get("features", [])):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        cap = props.get("capacity_mw") or props.get("plant:output:electricity")
        if cap is not None:
            try:
                cap = float(str(cap).replace(" MW", "").replace(",", ""))
            except (ValueError, TypeError):
                cap = 0.0
        else:
            cap = 0.0

        if cap <= 0:
            continue

        # Extract coordinates
        coords = geom.get("coordinates", [0, 0])
        if geom.get("type") == "Point":
            lon, lat = coords[0], coords[1]
        elif geom.get("type") in ("Polygon", "MultiPolygon"):
            # Use centroid of first ring
            ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
            lon = sum(c[0] for c in ring) / len(ring)
            lat = sum(c[1] for c in ring) / len(ring)
        else:
            continue

        fuel = props.get("fuel_type", props.get("plant:source", "unknown"))
        if fuel is None:
            fuel = "unknown"
        # Normalize non-standard OSM fuel types to FuelType enum values
        fuel_map = {
            "gas": "lng", "natural_gas": "lng",
            "waste": "biomass", "biofuel": "biomass",
            "battery": "unknown", "bing": "unknown",
            "gsimaps/ort": "unknown", "image": "unknown",
            "biomass_waste": "biomass",
        }
        fuel = fuel_map.get(fuel.lower().strip(), fuel)

        name = props.get("name") or props.get("name:ja") or ""
        if not name:
            name = f"{region}_{fuel}_{i:04d}"

        gen = Generator(
            id=f"{region}_gen_{i:04d}",
            name=name,
            capacity_mw=cap,
            fuel_type=fuel,
            region=region,
            latitude=lat,
            longitude=lon,
            operator=props.get("operator", ""),
            source="osm_plants",
        )
        generators.append(gen)

    return generators


def _match_generators_to_substations(generators, substations, max_distance_km=50.0):
    """Match generators to nearest substation by geographic proximity."""
    if not generators or not substations:
        return generators

    # Build spatial index from substations
    sub_coords = []
    sub_ids = []
    for sub in substations:
        if sub.latitude and sub.longitude:
            sub_coords.append((sub.latitude, sub.longitude))
            sub_ids.append(sub.id)

    if not sub_coords:
        return generators

    tree = cKDTree(sub_coords)

    for gen in generators:
        if gen.latitude == 0 and gen.longitude == 0:
            continue
        dist_deg, idx = tree.query((gen.latitude, gen.longitude))
        dist_km = dist_deg * 111.0  # rough conversion
        if dist_km <= max_distance_km:
            gen.connected_bus_id = sub_ids[idx]
        else:
            print(f"  Generator '{gen.name}' at ({gen.latitude:.4f}, {gen.longitude:.4f}) "
                  f"has no substation within {max_distance_km} km (nearest: {dist_km:.1f} km)")

    return generators


def _build_full_region(region):
    """Build FULL pandapower net for one region (all voltages, all buses)."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    sub_path = os.path.join(data_dir, f"{region}_substations.geojson")
    line_path = os.path.join(data_dir, f"{region}_lines.geojson")
    if not os.path.exists(sub_path) or not os.path.exists(line_path):
        return None, {}

    with open(sub_path) as f:
        sub_fc = json.load(f)
    with open(line_path) as f:
        line_fc = json.load(f)

    cfg = _get_region_config(region)
    f_hz = cfg.get("frequency_hz", 60)

    gn = build_grid_network(sub_fc, line_fc, region)

    # ── Load generators from plants GeoJSON ──
    generators = _load_plants_as_generators(region)
    if generators:
        generators = _match_generators_to_substations(
            generators, gn.substations, max_distance_km=50.0
        )
        for g in generators:
            if g.is_connected and g.capacity_mw > 0:
                try:
                    gn.add_generator(g)
                except ValueError:
                    pass

    net = PandapowerBuilder().build(gn).net

    # ── Clean voltage data ──
    for idx in net.bus.index:
        net.bus.at[idx, "vn_kv"] = _clean_voltage(net.bus.at[idx, "vn_kv"])

    # ── Fix cross-voltage lines → transformers ──
    lines_to_remove = []
    for idx in net.line.index:
        fb = net.line.at[idx, "from_bus"]
        tb = net.line.at[idx, "to_bus"]
        vf = net.bus.at[fb, "vn_kv"]
        vt = net.bus.at[tb, "vn_kv"]
        if min(vf, vt) <= 0:
            continue
        ratio = max(vf, vt) / min(vf, vt)
        if ratio >= 1.5:
            hv, lv = max(vf, vt), min(vf, vt)
            hv_bus = fb if vf >= vt else tb
            lv_bus = tb if vf >= vt else fb
            if hv >= 275:
                sn, vk, vkr = 800.0, 14.0, 0.20
            elif hv >= 154:
                sn, vk, vkr = 400.0, 12.0, 0.25
            elif hv >= 110:
                sn, vk, vkr = 200.0, 10.0, 0.30
            else:
                sn, vk, vkr = 100.0, 8.0, 0.40
            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus,
                sn_mva=sn, vn_hv_kv=hv, vn_lv_kv=lv,
                vkr_percent=vkr, vk_percent=vk,
                pfe_kw=30.0, i0_percent=0.05,
                name=f"auto_trafo_{len(net.trafo)}",
            )
            lines_to_remove.append(idx)

    if lines_to_remove:
        net.line = net.line.drop(lines_to_remove)

    # ── Connect disconnected components via nearest-neighbor bridging ──
    mg = pp.topology.create_nxgraph(net, include_trafos=True)
    comps = list(pp.topology.connected_components(mg))
    comps.sort(key=len, reverse=True)
    n_comps_raw = len(comps)
    n_bridges = _bridge_components(net)
    # After bridging, drop any still-disconnected (no geo, >300km away)
    mg2 = pp.topology.create_nxgraph(net, include_trafos=True)
    comps2 = list(pp.topology.connected_components(mg2))
    comps2.sort(key=len, reverse=True)
    drop_buses = set()
    if len(comps2) > 1:
        for comp in comps2[1:]:
            drop_buses |= comp
        if drop_buses:
            pp.drop_buses(net, drop_buses)

    # ── Set all gen vm_pu to 1.0 ──
    for idx in net.gen.index:
        net.gen.at[idx, "vm_pu"] = 1.0

    # ── Slack bus ──
    slack_bus = net.bus["vn_kv"].idxmax()
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, name="slack")

    # ── Loads: use OCCTO reference demand, capped by available generation ──
    ref_peak = REFERENCE_DEMAND.get(region, 5000)
    total_gen_capacity = net.gen["p_mw"].sum() if len(net.gen) > 0 else ref_peak
    target_load = min(ref_peak * LOAD_FACTOR, total_gen_capacity * 0.80)

    q_factor = math.tan(math.acos(POWER_FACTOR))

    load_buses = [i for i in net.bus.index
                  if i != slack_bus and net.bus.at[i, "vn_kv"] >= 66]
    if not load_buses:
        load_buses = [i for i in net.bus.index if i != slack_bus]

    weights = {}
    for bus_idx in load_buses:
        vn = net.bus.at[bus_idx, "vn_kv"]
        weights[bus_idx] = VOLTAGE_WEIGHTS.get(int(vn), VOLTAGE_WEIGHTS.get(0, 0.5))

    total_weight = sum(weights.values())
    if total_weight > 0:
        for bus_idx in load_buses:
            p_mw = target_load * weights[bus_idx] / total_weight
            pp.create_load(net, bus=bus_idx, p_mw=p_mw, q_mvar=p_mw * q_factor)

    # Name map for tie-line attachment
    name_map = {}
    for idx in net.bus.index:
        name_map[net.bus.at[idx, "name"]] = idx

    n_total_raw = len(net.bus) + len(drop_buses)
    n_still_dropped = len(drop_buses)
    print(f"  {len(net.bus):5d} bus  {len(net.line):5d} line  {len(net.trafo):4d} trafo  "
          f"{len(net.gen):4d} gen  {len(net.load):5d} load  "
          f"({n_comps_raw} comps → {n_bridges} bridges, "
          f"kept {len(net.bus)}/{n_total_raw}, dropped {n_still_dropped})")

    return net, name_map


def _pp_to_mpc_flat(net, f_hz=60, region=None):
    """Convert pandapower net to MATPOWER struct dict (flat start, no PF needed)."""
    baseMVA = 100.0
    n_bus = len(net.bus)
    n_gen = len(net.gen) + len(net.ext_grid)
    n_branch = len(net.line) + len(net.trafo)

    # Compute dispatch factor from reference demand
    ref_peak = REFERENCE_DEMAND.get(region, 5000) if region else 5000
    total_gen_cap = net.gen["p_mw"].sum() if len(net.gen) > 0 else ref_peak
    target_load = min(ref_peak * LOAD_FACTOR, total_gen_cap * 0.80)
    target_gen = target_load * (1 + RESERVE_MARGIN)
    dispatch_factor = min(target_gen / total_gen_cap, 0.90) if total_gen_cap > 0 else 0.5

    bus_data = np.zeros((n_bus, 13))
    bus_idx_map = {}
    for i, pp_idx in enumerate(net.bus.index):
        mp = i + 1
        bus_idx_map[pp_idx] = mp
        bus_data[i, 0] = mp
        bus_data[i, 1] = 1  # PQ
        bus_data[i, 6] = 1  # area
        bus_data[i, 7] = 1.0  # Vm flat start
        bus_data[i, 8] = 0.0  # Va flat start
        bus_data[i, 9] = net.bus.at[pp_idx, "vn_kv"]
        bus_data[i, 10] = 1  # zone
        bus_data[i, 11] = 1.15  # Vmax
        bus_data[i, 12] = 0.85  # Vmin

    for idx in net.load.index:
        bus = net.load.at[idx, "bus"]
        r = bus_idx_map[bus] - 1
        bus_data[r, 2] += net.load.at[idx, "p_mw"]
        bus_data[r, 3] += net.load.at[idx, "q_mvar"]

    for idx in net.gen.index:
        bus = net.gen.at[idx, "bus"]
        bus_data[bus_idx_map[bus] - 1, 1] = 2  # PV

    for idx in net.ext_grid.index:
        bus = net.ext_grid.at[idx, "bus"]
        bus_data[bus_idx_map[bus] - 1, 1] = 3  # Slack

    # Pre-allocate with extra room for voltage support generators
    n_gen_max = n_gen + n_bus
    gen_data = np.zeros((n_gen_max, 21))
    gi = 0
    for idx in net.gen.index:
        bus = net.gen.at[idx, "bus"]
        p = net.gen.at[idx, "p_mw"]
        pg = p * dispatch_factor
        gen_data[gi, 0] = bus_idx_map[bus]
        gen_data[gi, 1] = pg
        gen_data[gi, 2] = 0
        gen_data[gi, 3] = p * 0.6   # Qmax
        gen_data[gi, 4] = -p * 0.3  # Qmin
        gen_data[gi, 5] = 1.0  # Vg
        gen_data[gi, 6] = baseMVA
        gen_data[gi, 7] = 1    # status
        gen_data[gi, 8] = p    # Pmax
        gen_data[gi, 9] = 0    # Pmin
        gi += 1
    for idx in net.ext_grid.index:
        bus = net.ext_grid.at[idx, "bus"]
        gen_data[gi, 0] = bus_idx_map[bus]
        gen_data[gi, 1] = 0
        gen_data[gi, 2] = 0
        gen_data[gi, 3] = 9999
        gen_data[gi, 4] = -9999
        gen_data[gi, 5] = net.ext_grid.at[idx, "vm_pu"]
        gen_data[gi, 6] = baseMVA
        gen_data[gi, 7] = 1
        gen_data[gi, 8] = 9999
        gen_data[gi, 9] = -9999
        gi += 1

    # ── Voltage support (sync condensers) at ≥110 kV buses without generators ──
    gen_buses = set(int(gen_data[i, 0]) for i in range(gi))
    for i in range(n_bus):
        mp_bus = i + 1
        vn = bus_data[i, 9]
        if vn >= 110 and mp_bus not in gen_buses:
            qmax = 500.0 if vn >= 275 else 200.0 if vn >= 154 else 100.0
            gen_data[gi, 0] = mp_bus
            gen_data[gi, 1] = 0
            gen_data[gi, 2] = 0
            gen_data[gi, 3] = qmax
            gen_data[gi, 4] = -qmax * 0.4
            gen_data[gi, 5] = 1.0
            gen_data[gi, 6] = baseMVA
            gen_data[gi, 7] = 1
            gen_data[gi, 8] = 0
            gen_data[gi, 9] = 0
            bus_data[i, 1] = 2  # PV bus
            gen_buses.add(mp_bus)
            gi += 1

    # ── Shunt capacitors at ALL load buses ──
    for i in range(n_bus):
        pd = bus_data[i, 2]
        if pd > 0:
            bus_data[i, 5] = pd * 0.35  # Bs

    branch_data = np.zeros((n_branch, 13))
    bi = 0
    for idx in net.line.index:
        fb = net.line.at[idx, "from_bus"]
        tb = net.line.at[idx, "to_bus"]
        length = net.line.at[idx, "length_km"]
        r_ohm = net.line.at[idx, "r_ohm_per_km"] * length
        x_ohm = net.line.at[idx, "x_ohm_per_km"] * length

        vn = net.bus.at[fb, "vn_kv"]
        z_base = vn ** 2 / baseMVA if vn > 0 else 1.0

        r_pu = r_ohm / z_base
        x_pu = x_ohm / z_base

        if x_pu < MIN_X_PU:
            x_pu = MIN_X_PU
        if r_pu < x_pu * MIN_RX_RATIO:
            r_pu = x_pu * MIN_RX_RATIO

        b_total = net.line.at[idx, "c_nf_per_km"] * length * 2 * np.pi * f_hz * 1e-9
        b_pu = b_total * z_base
        if b_pu > 5.0:
            b_pu = 5.0

        branch_data[bi, 0] = bus_idx_map[fb]
        branch_data[bi, 1] = bus_idx_map[tb]
        branch_data[bi, 2] = r_pu
        branch_data[bi, 3] = x_pu
        branch_data[bi, 4] = b_pu
        branch_data[bi, 5] = max(net.line.at[idx, "max_i_ka"] * vn * np.sqrt(3), 100.0)
        branch_data[bi, 10] = 1
        bi += 1

    for idx in net.trafo.index:
        hv_bus = net.trafo.at[idx, "hv_bus"]
        lv_bus = net.trafo.at[idx, "lv_bus"]
        sn = net.trafo.at[idx, "sn_mva"]
        vn_hv = net.trafo.at[idx, "vn_hv_kv"]
        vn_lv = net.trafo.at[idx, "vn_lv_kv"]
        vk = net.trafo.at[idx, "vk_percent"]
        vkr = net.trafo.at[idx, "vkr_percent"]
        z_pu = (vk / 100.0) * (baseMVA / sn)
        r_pu = (vkr / 100.0) * (baseMVA / sn)
        x_pu = np.sqrt(max(z_pu ** 2 - r_pu ** 2, 1e-8))
        if x_pu < MIN_X_TRAFO:
            x_pu = MIN_X_TRAFO
        if r_pu < x_pu * MIN_RX_RATIO:
            r_pu = x_pu * MIN_RX_RATIO
        bus_base_hv = net.bus.at[hv_bus, "vn_kv"]
        bus_base_lv = net.bus.at[lv_bus, "vn_kv"]
        ratio = (vn_hv / bus_base_hv) / (vn_lv / bus_base_lv) if bus_base_hv > 0 and bus_base_lv > 0 else 1.0
        branch_data[bi, 0] = bus_idx_map[hv_bus]
        branch_data[bi, 1] = bus_idx_map[lv_bus]
        branch_data[bi, 2] = r_pu
        branch_data[bi, 3] = x_pu
        branch_data[bi, 5] = sn
        branch_data[bi, 8] = ratio
        branch_data[bi, 10] = 1
        bi += 1

    # ── Merge parallel branches ──
    branch_raw = branch_data[:bi]
    branch_groups = defaultdict(list)
    for i in range(bi):
        fb, tb = int(branch_raw[i, 0]), int(branch_raw[i, 1])
        key = (min(fb, tb), max(fb, tb))
        branch_groups[key].append(i)

    merged_branches = []
    for key, indices in branch_groups.items():
        if len(indices) == 1:
            row = branch_raw[indices[0]].copy()
        else:
            row = branch_raw[indices[0]].copy()
            y_r_total = 0.0
            y_x_total = 0.0
            b_total = 0.0
            rate_total = 0.0
            has_trafo = False

            for i in indices:
                r_pu = branch_raw[i, 2]
                x_pu = branch_raw[i, 3]
                z2 = r_pu ** 2 + x_pu ** 2
                if z2 > 1e-12:
                    y_r_total += r_pu / z2
                    y_x_total += x_pu / z2
                b_total += branch_raw[i, 4]
                rate_total += branch_raw[i, 5]
                if branch_raw[i, 8] > 0:
                    has_trafo = True

            y2 = y_r_total ** 2 + y_x_total ** 2
            if y2 > 1e-12:
                row[2] = y_r_total / y2
                row[3] = y_x_total / y2

            row[4] = b_total
            row[5] = rate_total
            if has_trafo:
                row[8] = 1.0

        if row[3] < MIN_X_PU:
            row[3] = MIN_X_PU
        if row[2] < row[3] * MIN_RX_RATIO:
            row[2] = row[3] * MIN_RX_RATIO
        if row[3] > 2.0:
            row[3] = 2.0
        if row[2] > 2.0:
            row[2] = 2.0

        merged_branches.append(row)

    branch_data = np.array(merged_branches) if merged_branches else np.zeros((0, 13))

    return {
        "baseMVA": baseMVA,
        "bus": bus_data,
        "gen": gen_data[:gi],
        "branch": branch_data,
        "version": "2",
    }, bus_idx_map


def main():
    print("=" * 70)
    print("  Building FULL-SCALE regional cases + All-Japan")
    print("  (nearest-neighbor bridging for disconnected components)")
    print("  Data source: OpenStreetMap GeoJSON")
    print("=" * 70)

    regions = ALL_REGIONS
    region_nets = {}
    region_mpcs = {}
    region_name_maps = {}
    region_success = {}

    for region in regions:
        cfg = _get_region_config(region)
        f_hz = cfg.get("frequency_hz", 60)
        print(f"\n[{region}]", end=" ", flush=True)

        net, name_map = _build_full_region(region)

        if net is None:
            print("  SKIP (no data)")
            region_success[region] = False
            continue

        mpc, bmap = _pp_to_mpc_flat(net, f_hz, region=region)
        mat_path = os.path.join(OUT_DIR, f"{region}.mat")
        savemat(mat_path, {"mpc": mpc})

        nb = mpc["bus"].shape[0]
        ng = mpc["gen"].shape[0]
        nbr = mpc["branch"].shape[0]
        pd = mpc["bus"][:, 2].sum()
        pg = mpc["gen"][:, 1].sum()
        print(f"\n  → {nb:5d} bus  {ng:4d} gen  {nbr:5d} branch  "
              f"Pd={pd:7.0f} MW  Pg={pg:7.0f} MW  → {os.path.basename(mat_path)}")

        region_success[region] = True
        region_nets[region] = net
        region_mpcs[region] = mpc
        region_name_maps[region] = name_map

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Regional Results")
    print(f"{'=' * 70}")
    total_b = total_br = total_g = 0
    for r in regions:
        if r in region_mpcs:
            m = region_mpcs[r]
            nb = m["bus"].shape[0]
            ng = m["gen"].shape[0]
            nbr = m["branch"].shape[0]
            pd = m["bus"][:, 2].sum()
            pg = m["gen"][:, 1].sum()
            total_b += nb
            total_br += nbr
            total_g += ng
            print(f"  {r:12s}: {nb:5d} bus  {ng:4d} gen  {nbr:5d} branch  "
                  f"Pd={pd:8.1f} MW  Pg={pg:8.1f} MW")
        else:
            print(f"  {r:12s}: SKIP")
    print(f"  {'TOTAL':12s}: {total_b:5d} bus  {total_g:4d} gen  {total_br:5d} branch")

    # ── All-Japan merge ──
    print(f"\n{'=' * 70}")
    print(f"  Building All-Japan model with tie lines")
    print(f"{'=' * 70}")

    ic_path = os.path.join(PROJECT_ROOT, "data", "reference", "interconnections.yaml")
    with open(ic_path) as f:
        ic_data = yaml.safe_load(f)

    all_bus = []
    all_gen = []
    all_branch = []
    bus_offset = 0
    region_bus_offsets = {}
    region_slack_bus = {}

    for region in regions:
        if region not in region_mpcs:
            continue
        mpc = region_mpcs[region]
        nb = mpc["bus"].shape[0]
        area_num = regions.index(region) + 1

        bus = mpc["bus"].copy()
        bus[:, 0] += bus_offset
        bus[:, 6] = area_num
        bus[:, 10] = area_num

        region_bus_offsets[region] = bus_offset

        slack_mask = bus[:, 1] == 3
        if slack_mask.any():
            region_slack_bus[region] = int(bus[slack_mask, 0][0])
        else:
            region_slack_bus[region] = int(bus[0, 0])

        all_bus.append(bus)

        gen = mpc["gen"].copy()
        gen[:, 0] += bus_offset
        all_gen.append(gen)

        branch = mpc["branch"].copy()
        branch[:, 0] += bus_offset
        branch[:, 1] += bus_offset
        all_branch.append(branch)

        bus_offset += nb

    if not all_bus:
        print("  No regions available!")
        return

    all_bus = np.vstack(all_bus)
    all_gen = np.vstack(all_gen)
    all_branch = np.vstack(all_branch)

    # Single slack: Tokyo (or first available)
    primary_slack_region = "tokyo" if "tokyo" in region_slack_bus else list(region_slack_bus.keys())[0]
    primary_slack = region_slack_bus[primary_slack_region]
    for region, slack in region_slack_bus.items():
        if slack != primary_slack:
            row = int(slack - 1)
            if row < len(all_bus) and all_bus[row, 1] == 3:
                all_bus[row, 1] = 2

    # ── Tie lines ──
    baseMVA = 100.0
    tie_lines = []
    for ic in ic_data.get("interconnections", []):
        fr = ic["from_region"]
        to = ic["to_region"]
        cap = ic["capacity_mw"]
        ic_type = ic["type"]
        name = ic.get("name_ja", ic.get("name", ""))

        if fr not in region_slack_bus or to not in region_slack_bus:
            print(f"    {name}: SKIP")
            continue

        from_bus = region_slack_bus[fr]
        to_bus = region_slack_bus[to]

        if ic_type == "HVDC":
            r_pu, x_pu, b_pu = 0.001, 0.01, 0.0
        elif ic_type == "FC":
            r_pu, x_pu, b_pu = 0.002, 0.02, 0.0
        else:
            r_pu, x_pu, b_pu = 0.005, 0.05, 0.02

        from_base_kv = all_bus[int(from_bus - 1), 9]
        to_base_kv = all_bus[int(to_bus - 1), 9]
        ratio = 1.0 if from_base_kv != to_base_kv else 0.0

        tie = np.zeros(13)
        tie[0] = from_bus
        tie[1] = to_bus
        tie[2] = r_pu
        tie[3] = x_pu
        tie[4] = b_pu
        tie[5] = cap
        tie[8] = ratio
        tie[10] = 1

        tie_lines.append(tie)
        print(f"    {name}: bus {int(from_bus)}({fr}) ↔ bus {int(to_bus)}({to}), "
              f"{cap} MW, {ic_type}")

    if tie_lines:
        all_branch = np.vstack([all_branch, np.array(tie_lines)])

    alljapan_mpc = {
        "baseMVA": baseMVA,
        "bus": all_bus,
        "gen": all_gen,
        "branch": all_branch,
        "version": "2",
    }

    aj_path = os.path.join(OUT_DIR, "alljapan.mat")
    savemat(aj_path, {"mpc": alljapan_mpc})

    print(f"\n  All-Japan: {aj_path}")
    print(f"    Buses:    {all_bus.shape[0]}")
    print(f"    Gens:     {all_gen.shape[0]}")
    print(f"    Branches: {all_branch.shape[0]} (incl {len(tie_lines)} tie lines)")
    print(f"    Total Pd: {all_bus[:, 2].sum():.1f} MW")
    print(f"    Total Pg: {all_gen[:, 1].sum():.1f} MW")

    print(f"\n{'=' * 70}")
    print(f"  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
