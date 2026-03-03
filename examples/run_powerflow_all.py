#!/usr/bin/env python3
"""Run DC and AC power flow on all 9 Japanese regional grids.

Builds pandapower networks from KML data, runs power flow,
and produces a summary dashboard with voltage profiles and line loading.

Usage::

    PYTHONPATH=. python scripts/run_powerflow_all.py
"""

import copy
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.converter.pandapower_builder import PandapowerBuilder
from src.parser.kml_parser import KMLParser
from src.powerflow.load_estimator import estimate_loads, load_demand_config

OUTPUT_DIR = "output/powerflow_regional"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu",
]

REGION_JA = {
    "hokkaido": "北海道", "tohoku": "東北", "tokyo": "東京",
    "chubu": "中部", "hokuriku": "北陸", "kansai": "関西",
    "chugoku": "中国", "shikoku": "四国", "kyushu": "九州",
}


def build_network(region: str, config: dict) -> pp.pandapowerNet:
    """Build pandapower network from KML for a region."""
    region_cfg = config["regions"][region]
    parser = KMLParser()
    freq = region_cfg.get("frequency_hz", 0)
    kml_path = f"data/raw/{region}.kml"

    network = parser.parse_file(kml_path, region=region, frequency_hz=freq)
    builder = PandapowerBuilder()
    result = builder.build(network)
    net = result.net

    # Fix zero-voltage buses
    zero_mask = net.bus["vn_kv"] == 0
    if zero_mask.any():
        bus_voltages = {}
        for line in network.lines:
            if line.voltage_kv <= 0:
                continue
            for sid in (line.from_substation_id, line.to_substation_id):
                if sid not in bus_voltages or line.voltage_kv > bus_voltages[sid]:
                    bus_voltages[sid] = line.voltage_kv
        sub_to_idx = {s.id: i for i, s in enumerate(network.substations)}
        for sid, v in bus_voltages.items():
            idx = sub_to_idx.get(sid)
            if idx is not None and idx in net.bus.index and net.bus.at[idx, "vn_kv"] == 0:
                net.bus.at[idx, "vn_kv"] = v
        still_zero = net.bus["vn_kv"] == 0
        if still_zero.any():
            non_zero = net.bus.loc[~still_zero, "vn_kv"]
            if len(non_zero) > 0:
                net.bus.loc[still_zero, "vn_kv"] = float(non_zero.median())

    return net


def fix_topology(net: pp.pandapowerNet) -> dict:
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
        for tbl in ("load", "gen", "line"):
            table = getattr(net, tbl, None)
            if table is None or table.empty:
                continue
            if tbl == "line":
                mask = table["from_bus"].isin(isolated) | table["to_bus"].isin(isolated)
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


def run_powerflow(net: pp.pandapowerNet, mode: str = "dc") -> dict:
    """Run DC or AC power flow and return results."""
    result = {"mode": mode, "converged": False}
    try:
        if mode == "dc":
            pp.rundcpp(net)
        else:
            pp.runpp(net, numba=False)
        result["converged"] = True
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


def plot_dashboard(all_results: dict):
    """Create a summary dashboard figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Japan Regional Power Flow Analysis\n日本電力系統 地域別潮流計算結果",
                 fontsize=16, fontweight="bold", y=0.98)

    regions_sorted = REGIONS
    n = len(regions_sorted)
    x = np.arange(n)
    labels = [f"{REGION_JA[r]}\n{r.title()}" for r in regions_sorted]

    # --- Panel 1: Network size ---
    ax = axes[0, 0]
    buses = [all_results[r]["n_buses"] for r in regions_sorted]
    lines = [all_results[r]["n_lines"] for r in regions_sorted]
    w = 0.35
    ax.bar(x - w/2, buses, w, label="Buses (変電所)", color="#2196F3", alpha=0.8)
    ax.bar(x + w/2, lines, w, label="Lines (送電線)", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Network Size — バス数・送電線数")
    ax.legend()
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
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Voltage Angle (degrees)")
    ax.set_title("DC Power Flow — Bus Voltage Angle Range")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 3: Line loading distribution ---
    ax = axes[1, 0]
    dc_loading = []
    dc_labels_line = []
    for r in regions_sorted:
        dc = all_results[r].get("dc")
        if dc and dc.get("converged"):
            dc_loading.append(dc.get("max_loading_pct", 0))
            dc_labels_line.append(REGION_JA[r])
    if dc_loading:
        colors_bar = ["#F44336" if v > 100 else "#FF9800" if v > 80 else "#4CAF50" for v in dc_loading]
        ax.barh(range(len(dc_loading)), dc_loading, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(dc_loading)))
        ax.set_yticklabels(dc_labels_line, fontsize=9)
        ax.axvline(100, color="red", linestyle="--", linewidth=1, label="100% limit")
        ax.axvline(80, color="orange", linestyle="--", linewidth=1, label="80% warning")
    ax.set_xlabel("Max Line Loading (%)")
    ax.set_title("DC Power Flow — Maximum Line Loading")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 4: Convergence summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    headers = ["Region", "Buses", "Lines", "DC Conv.", "DC Loss(MW)", "AC Conv.", "AC Loss(MW)"]
    for r in regions_sorted:
        d = all_results[r]
        dc = d.get("dc", {})
        ac = d.get("ac", {})
        table_data.append([
            f"{REGION_JA[r]} ({r})",
            str(d["n_buses"]),
            str(d["n_lines"]),
            "OK" if dc.get("converged") else "FAIL",
            f"{dc.get('total_loss_mw', 0):.1f}" if dc.get("converged") else "-",
            "OK" if ac.get("converged") else "FAIL",
            f"{ac.get('total_loss_mw', 0):.1f}" if ac.get("converged") else "-",
        ])

    tbl = ax.table(cellText=table_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    # Color convergence cells
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = tbl[i + 1, j]
            if val == "OK":
                cell.set_facecolor("#C8E6C9")
            elif val == "FAIL":
                cell.set_facecolor("#FFCDD2")
    ax.set_title("Convergence Summary — 収束結果一覧", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "regional_powerflow_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nDashboard saved: {out_path}")


def main():
    config_path = "config/regions.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    all_results = {}

    for region in REGIONS:
        kml_path = f"data/raw/{region}.kml"
        if not os.path.exists(kml_path):
            print(f"SKIP {region}: no KML file")
            continue

        print(f"\n{'='*60}")
        print(f"  {REGION_JA[region]} ({region})")
        print(f"{'='*60}")

        # Build network
        net = build_network(region, config)
        n_buses = len(net.bus)
        n_lines = len(net.line)
        print(f"  Network: {n_buses} buses, {n_lines} lines")

        # Fix topology
        diag = fix_topology(net)
        print(f"  Components: {diag['n_components']}, isolated: {diag['n_isolated_buses']}, active: {diag['n_active_buses']}")

        # Estimate loads from OCCTO demand data
        demand_cfg = load_demand_config()
        total_load = estimate_loads(net, region=region, demand_config=demand_cfg)
        print(f"  Loads allocated: {total_load:.0f} MW across {len(net.load)} buses")

        # DC power flow
        net_dc = copy.deepcopy(net)
        dc_result = run_powerflow(net_dc, "dc")
        if dc_result["converged"]:
            print(f"  DC: converged, loss={dc_result.get('total_loss_mw', 0):.1f} MW, "
                  f"max_loading={dc_result.get('max_loading_pct', 0):.1f}%, "
                  f"angle=[{dc_result.get('va_deg_min', 0):.1f}, {dc_result.get('va_deg_max', 0):.1f}] deg")
        else:
            print(f"  DC: FAILED — {dc_result.get('error', 'unknown')}")

        # AC power flow
        net_ac = copy.deepcopy(net)
        ac_result = run_powerflow(net_ac, "ac")
        if ac_result["converged"]:
            print(f"  AC: converged, loss={ac_result.get('total_loss_mw', 0):.1f} MW, "
                  f"max_loading={ac_result.get('max_loading_pct', 0):.1f}%, "
                  f"V=[{ac_result.get('vm_pu_min', 0):.4f}, {ac_result.get('vm_pu_max', 0):.4f}] pu")
        else:
            print(f"  AC: FAILED — {ac_result.get('error', 'unknown')}")

        all_results[region] = {
            "n_buses": n_buses,
            "n_lines": n_lines,
            "topology": diag,
            "dc": dc_result,
            "ac": ac_result,
        }

    # Generate dashboard
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
    dc_ok = sum(1 for r in all_results.values() if r["dc"].get("converged"))
    ac_ok = sum(1 for r in all_results.values() if r["ac"].get("converged"))
    print(f"  Total: {total_buses} buses, {total_lines} lines across {len(all_results)} regions")
    print(f"  DC convergence: {dc_ok}/{len(all_results)}")
    print(f"  AC convergence: {ac_ok}/{len(all_results)}")


if __name__ == "__main__":
    main()
