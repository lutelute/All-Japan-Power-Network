#!/usr/bin/env python3
"""National UC run: load real P03 generators from XML, solve UC, and visualize.

Loads all generators from the standardized japan_grid_all.xml, filters to
dispatchable units (>= 10 MW), generates a demand profile scaled to total
capacity, runs unit commitment with regional decomposition, and outputs
4 PNG visualizations + CSV/XML result exports.

Usage::

    python scripts/uc_national_run.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter, defaultdict

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Japanese font support
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.model.generator import Generator
from src.uc.models import UCParameters, TimeHorizon, DemandProfile, UCResult
from src.uc.solver import solve_uc
from src.uc.decomposition import create_decomposer
from src.uc.xml_loader import UCXMLLoader
from src.uc.result_exporter import export_uc_result_csv, export_uc_result_xml

# ── Configuration ─────────────────────────────────────────────────
XML_PATH = os.path.join("output", "xml", "japan_grid_all.xml")
UC_CONFIG_PATH = os.path.join("config", "uc_config.yaml")
OUTPUT_DIR = os.path.join("output", "uc_national")

MIN_CAPACITY_MW = 10.0       # Exclude tiny generators from UC
DEMAND_FACTOR = 0.65         # Demand as fraction of total capacity (realistic)
RESERVE_MARGIN = 0.05        # 5% spinning reserve
DECOMPOSITION_THRESHOLD = 200  # Use decomposition above this many generators
NUM_PERIODS = 24

FUEL_COLORS = {
    "nuclear": "#7B2D8E",
    "coal":    "#4A4A4A",
    "lng":     "#E8832A",
    "oil":     "#C44E52",
    "hydro":   "#2196F3",
    "pumped_hydro": "#1565C0",
    "solar":   "#FFD700",
    "wind":    "#4CAF50",
    "biomass": "#8BC34A",
    "geothermal": "#795548",
    "mixed":   "#9E9E9E",
    "unknown": "#BDBDBD",
}

FUEL_ORDER = [
    "nuclear", "coal", "hydro", "pumped_hydro", "geothermal",
    "biomass", "wind", "solar", "lng", "oil", "mixed", "unknown",
]


def load_generators():
    """Load generators from XML and filter to UC-viable units."""
    loader = UCXMLLoader()
    all_gens = loader.load_generators_from_xml(XML_PATH, UC_CONFIG_PATH)
    print(f"Loaded {len(all_gens)} generators from XML")

    # Filter: positive capacity >= threshold, active status
    viable = [
        g for g in all_gens
        if g.capacity_mw >= MIN_CAPACITY_MW and g.status == "active"
    ]
    print(f"Viable for UC (>= {MIN_CAPACITY_MW} MW): {len(viable)} generators")

    # Print summary
    fuel_counts = Counter(g.fuel_type_enum.value for g in viable)
    region_counts = Counter(g.region for g in viable)
    total_cap = sum(g.capacity_mw for g in viable)

    print(f"\n{'Fuel Type':<18} {'Count':>6} {'Capacity (MW)':>14}")
    print("-" * 42)
    for ft in FUEL_ORDER:
        if ft in fuel_counts:
            cap = sum(g.capacity_mw for g in viable if g.fuel_type_enum.value == ft)
            print(f"  {ft:<16} {fuel_counts[ft]:>6} {cap:>14,.0f}")
    print(f"  {'TOTAL':<16} {len(viable):>6} {total_cap:>14,.0f}")

    print(f"\n{'Region':<18} {'Count':>6} {'Capacity (MW)':>14}")
    print("-" * 42)
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        cap = sum(g.capacity_mw for g in viable if g.region == region)
        print(f"  {region:<16} {count:>6} {cap:>14,.0f}")

    return viable


def make_demand_profile(total_capacity_mw):
    """Generate a realistic Japanese 24h demand curve scaled to fleet capacity."""
    # Normalized demand shape: summer weekday pattern
    shape = np.array([
        0.60, 0.57, 0.55, 0.53, 0.55, 0.60,  # 0-5h: nighttime
        0.68, 0.78, 0.87, 0.93, 0.97, 1.00,  # 6-11h: morning ramp
        0.99, 0.98, 0.96, 0.93, 0.90, 0.86,  # 12-17h: afternoon
        0.82, 0.78, 0.74, 0.70, 0.66, 0.63,  # 18-23h: evening
    ])
    peak_demand = total_capacity_mw * DEMAND_FACTOR
    demands = (shape * peak_demand).tolist()
    return demands


def run_uc(generators, demand):
    """Solve the UC problem with decomposition if needed."""
    th = TimeHorizon(num_periods=NUM_PERIODS, period_duration_h=1.0)
    dp = DemandProfile(demands=demand)

    # Load solver config
    with open(UC_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg.get("solver", {})

    params = UCParameters(
        generators=generators,
        demand=dp,
        time_horizon=th,
        reserve_margin=RESERVE_MARGIN,
        solver_name=solver_cfg.get("backend", "HiGHS"),
        solver_time_limit_s=solver_cfg.get("time_limit_s", 300),
        mip_gap=solver_cfg.get("mip_gap", 0.01),
    )

    n_gens = len(generators)
    if n_gens > DECOMPOSITION_THRESHOLD:
        n_regions = len(set(g.region for g in generators))
        print(f"\n{n_gens} generators > {DECOMPOSITION_THRESHOLD} threshold")
        print(f"Using REGIONAL decomposition ({n_regions} regions)")
        decomposer = create_decomposer("regional")
        result = decomposer.solve_decomposed(params)
    else:
        print(f"\n{n_gens} generators <= {DECOMPOSITION_THRESHOLD}, solving directly")
        result = solve_uc(params)

    print(f"\nSolver status: {result.status}")
    print(f"Total cost: ¥{result.total_cost:,.0f}")
    print(f"Solve time: {result.solve_time_s:.2f}s")
    print(f"Generators scheduled: {result.num_generators}")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")
        for w in result.warnings[:5]:
            print(f"  - {w}")

    return result


def plot_dispatch_stack(result, gens, demand, output_path):
    """Stacked area chart of power dispatch by fuel type."""
    hours = np.arange(NUM_PERIODS)
    gen_map = {g.id: g for g in gens}

    stacks = {ft: np.zeros(NUM_PERIODS) for ft in FUEL_ORDER}
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None:
            continue
        ft = g.fuel_type_enum.value
        if ft not in stacks:
            stacks[ft] = np.zeros(NUM_PERIODS)
        stacks[ft] += np.array(sched.power_output_mw[:NUM_PERIODS])

    fig, ax = plt.subplots(figsize=(16, 8))

    bottoms = np.zeros(NUM_PERIODS)
    for ft in FUEL_ORDER:
        if ft in stacks and stacks[ft].sum() > 0:
            ax.fill_between(
                hours, bottoms, bottoms + stacks[ft],
                label=ft.upper(), color=FUEL_COLORS.get(ft, "#999"),
                alpha=0.85, linewidth=0.5, edgecolor="white",
            )
            bottoms += stacks[ft]

    ax.plot(hours, demand[:NUM_PERIODS], "k-", linewidth=2.5, label="Demand", zorder=5)
    ax.plot(hours, demand[:NUM_PERIODS], "ko", markersize=4, zorder=5)

    ax.set_xlim(0, NUM_PERIODS - 1)
    ax.set_ylim(0, max(demand) * 1.15)
    ax.set_xlabel("Hour of Day", fontsize=13)
    ax.set_ylabel("Power Output (MW)", fontsize=13)
    ax.set_title(
        f"24-Hour Generator Dispatch — All Japan ({len(gens)} generators)\n"
        f"Total Capacity: {sum(g.capacity_mw for g in gens):,.0f} MW, "
        f"Peak Demand: {max(demand):,.0f} MW",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, ncol=3, framealpha=0.9)
    ax.set_xticks(hours)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cost_breakdown(result, gens, output_path):
    """Cost breakdown by fuel type — horizontal bar chart."""
    gen_map = {g.id: g for g in gens}

    # Aggregate costs by fuel type
    fuel_costs = defaultdict(lambda: {"fuel": 0, "startup": 0, "noload": 0, "shutdown": 0})
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None:
            continue
        ft = g.fuel_type_enum.value
        fuel_costs[ft]["fuel"] += sched.fuel_cost
        fuel_costs[ft]["startup"] += sched.startup_cost
        fuel_costs[ft]["noload"] += sched.no_load_cost
        fuel_costs[ft]["shutdown"] += sched.shutdown_cost

    # Sort by total cost
    sorted_fuels = sorted(
        fuel_costs.keys(),
        key=lambda ft: sum(fuel_costs[ft].values()),
        reverse=True,
    )
    sorted_fuels = [ft for ft in sorted_fuels if sum(fuel_costs[ft].values()) > 0]

    names = [ft.upper() for ft in sorted_fuels]
    fuel_vals = [fuel_costs[ft]["fuel"] for ft in sorted_fuels]
    startup_vals = [fuel_costs[ft]["startup"] for ft in sorted_fuels]
    noload_vals = [fuel_costs[ft]["noload"] for ft in sorted_fuels]
    shutdown_vals = [fuel_costs[ft]["shutdown"] for ft in sorted_fuels]

    y = np.arange(len(names))
    bar_h = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(y, fuel_vals, bar_h, label="Fuel Cost", color="#E8832A")
    ax.barh(y, noload_vals, bar_h, left=fuel_vals, label="No-Load + Labor", color="#2196F3")
    left2 = np.array(fuel_vals) + np.array(noload_vals)
    ax.barh(y, startup_vals, bar_h, left=left2, label="Startup Cost", color="#C44E52")
    left3 = left2 + np.array(startup_vals)
    ax.barh(y, shutdown_vals, bar_h, left=left3, label="Shutdown Cost", color="#9C27B0")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Cost (¥)", fontsize=13)
    ax.set_title(
        f"Cost Breakdown by Fuel Type — 24h UC Result\n"
        f"Total: ¥{result.total_cost:,.0f}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Total cost labels
    for i, ft in enumerate(sorted_fuels):
        total = sum(fuel_costs[ft].values())
        ax.text(total + max(fuel_vals) * 0.01, i,
                f"¥{total:,.0f}", va="center", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_commitment_heatmap(result, gens, output_path):
    """Heatmap: commitment by region × hour (aggregated load factor)."""
    gen_map = {g.id: g for g in gens}

    # Aggregate by region
    region_output = defaultdict(lambda: np.zeros(NUM_PERIODS))
    region_capacity = defaultdict(float)
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None:
            continue
        r = g.region or "unknown"
        region_output[r] += np.array(sched.power_output_mw[:NUM_PERIODS])
        region_capacity[r] += g.capacity_mw

    regions_sorted = sorted(
        region_output.keys(),
        key=lambda r: region_capacity[r],
        reverse=True,
    )

    # Build load factor matrix
    data = []
    labels = []
    for r in regions_sorted:
        cap = region_capacity[r]
        if cap > 0:
            load_factor = region_output[r] / cap
        else:
            load_factor = np.zeros(NUM_PERIODS)
        data.append(load_factor)
        n_gens = sum(1 for g in gens if g.region == r)
        labels.append(f"{r} ({n_gens} gens, {cap:,.0f} MW)")

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(16, max(6, len(labels) * 0.5 + 2)))

    cmap = plt.cm.YlOrRd
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[0] = [0.95, 0.95, 0.95, 1.0]
    custom_cmap = ListedColormap(cmap_colors)

    im = ax.imshow(data, aspect="auto", cmap=custom_cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(np.arange(NUM_PERIODS))
    ax.set_xticklabels([f"{h:02d}" for h in range(NUM_PERIODS)], fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("Hour of Day", fontsize=13)
    ax.set_title(
        "Regional Load Factor Heatmap — 24h UC Schedule\n"
        "(color intensity = aggregate load factor by region)",
        fontsize=14, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Load Factor (0=idle, 1=full load)", fontsize=11)

    ax.set_xticks(np.arange(-0.5, NUM_PERIODS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(result, gens, demand, output_path):
    """Summary dashboard: 4-panel overview."""
    gen_map = {g.id: g for g in gens}
    hours = np.arange(NUM_PERIODS)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # ── (a) Supply vs Demand ──
    ax = axes[0, 0]
    total_supply = np.zeros(NUM_PERIODS)
    committed_cap = np.zeros(NUM_PERIODS)
    for sched in result.schedules:
        g = gen_map.get(sched.generator_id)
        if g is None:
            continue
        po = np.array(sched.power_output_mw[:NUM_PERIODS])
        cm = np.array(sched.commitment[:NUM_PERIODS])
        total_supply += po
        committed_cap += cm * g.capacity_mw

    demand_arr = np.array(demand[:NUM_PERIODS])
    ax.fill_between(hours, committed_cap, alpha=0.2, color="#2196F3", label="Committed Capacity")
    ax.plot(hours, total_supply, "b-", linewidth=2, label="Total Supply")
    ax.plot(hours, demand_arr, "r--", linewidth=2, label="Demand")
    reserve = demand_arr * (1 + RESERVE_MARGIN)
    ax.plot(hours, reserve, "g:", linewidth=1.5, label=f"Demand + {RESERVE_MARGIN*100:.0f}% Reserve")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.set_title("(a) Demand vs Supply & Reserve", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(hours)

    # ── (b) Cost pie chart ──
    ax = axes[0, 1]
    total_fuel = sum(s.fuel_cost for s in result.schedules)
    total_startup = sum(s.startup_cost for s in result.schedules)
    total_shutdown = sum(s.shutdown_cost for s in result.schedules)
    total_noload = sum(s.no_load_cost for s in result.schedules)

    pie_labels = []
    pie_sizes = []
    pie_colors = []
    for lbl, val, col in [("Fuel", total_fuel, "#E8832A"),
                           ("No-Load+Labor", total_noload, "#2196F3"),
                           ("Startup", total_startup, "#C44E52"),
                           ("Shutdown", total_shutdown, "#9C27B0")]:
        if val > 0:
            pie_labels.append(lbl)
            pie_sizes.append(val)
            pie_colors.append(col)

    if pie_sizes:
        ax.pie(
            pie_sizes, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10},
        )
    ax.set_title(f"(b) Total Cost: ¥{result.total_cost:,.0f}", fontweight="bold")

    # ── (c) Capacity by fuel type ──
    ax = axes[1, 0]
    fuel_gen_count = Counter(g.fuel_type_enum.value for g in gens)
    fuel_cap = defaultdict(float)
    for g in gens:
        fuel_cap[g.fuel_type_enum.value] += g.capacity_mw

    fuels_sorted = [ft for ft in FUEL_ORDER if ft in fuel_cap]
    caps = [fuel_cap[ft] for ft in fuels_sorted]
    colors = [FUEL_COLORS.get(ft, "#999") for ft in fuels_sorted]
    x = np.arange(len(fuels_sorted))

    bars = ax.bar(x, caps, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ft.upper() for ft in fuels_sorted], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Installed Capacity (MW)")
    ax.set_title("(c) Installed Capacity by Fuel Type", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, cap in zip(bars, caps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{cap:,.0f}", ha="center", va="bottom", fontsize=8)

    # ── (d) Statistics table ──
    ax = axes[1, 1]
    ax.axis("off")

    total_energy = sum(s.total_energy_mwh for s in result.schedules)
    total_demand_mwh = sum(demand[:NUM_PERIODS])
    avg_cost = result.total_cost / total_energy if total_energy > 0 else 0
    total_cap = sum(g.capacity_mw for g in gens)
    n_regions = len(set(g.region for g in gens))
    n_fuels = len(set(g.fuel_type_enum.value for g in gens))

    stats = [
        ["Metric", "Value"],
        ["Solver Status", result.status],
        ["Solve Time", f"{result.solve_time_s:.2f} s"],
        ["Total Cost", f"¥{result.total_cost:,.0f}"],
        ["Total Energy", f"{total_energy:,.0f} MWh"],
        ["Total Demand", f"{total_demand_mwh:,.0f} MWh"],
        ["Avg. Cost / MWh", f"¥{avg_cost:,.0f}"],
        ["Peak Demand", f"{max(demand):,.0f} MW"],
        ["Installed Capacity", f"{total_cap:,.0f} MW"],
        ["Fleet Size", f"{len(gens)} generators"],
        ["Regions", f"{n_regions}"],
        ["Fuel Types", f"{n_fuels}"],
        ["Reserve Margin", f"{RESERVE_MARGIN*100:.0f}%"],
    ]

    table = ax.table(cellText=stats[1:], colLabels=stats[0],
                     cellLoc="left", loc="center",
                     colWidths=[0.45, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for j in range(2):
        table[0, j].set_facecolor("#333")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(stats)):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f0f0")

    ax.set_title("(d) UC Result Summary", fontweight="bold", y=0.98)

    fig.suptitle(
        "Unit Commitment — All Japan National Grid",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_generator_matrix(result, gens, output_path):
    """Full 657-generator × 24h commitment matrix — one row per generator.

    Generators are grouped by fuel type (color-coded) and sorted by capacity
    within each group. Cell brightness shows load factor (output / capacity).
    """
    gen_map = {g.id: g for g in gens}

    # Build schedule lookup
    sched_map = {s.generator_id: s for s in result.schedules}

    # Sort generators: by fuel order, then by capacity descending
    fuel_rank = {ft: i for i, ft in enumerate(FUEL_ORDER)}
    sorted_gens = sorted(
        gens,
        key=lambda g: (fuel_rank.get(g.fuel_type_enum.value, 99), -g.capacity_mw),
    )

    n_gens = len(sorted_gens)

    # Build matrices
    load_factor = np.zeros((n_gens, NUM_PERIODS))
    commitment = np.zeros((n_gens, NUM_PERIODS))
    fuel_types = []

    for i, g in enumerate(sorted_gens):
        sched = sched_map.get(g.id)
        fuel_types.append(g.fuel_type_enum.value)
        if sched is None:
            continue
        for t in range(min(NUM_PERIODS, len(sched.power_output_mw))):
            if g.capacity_mw > 0:
                load_factor[i, t] = sched.power_output_mw[t] / g.capacity_mw
            commitment[i, t] = sched.commitment[t] if t < len(sched.commitment) else 0

    # ── Figure: 2-panel (commitment colored by fuel + load factor) ──
    fig, (ax_fuel, ax_lf) = plt.subplots(
        1, 2, figsize=(22, 16),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
    )

    # --- Left panel: Commitment (ON/OFF) colored by fuel type ---
    # Create RGB image: committed cells get fuel color, off cells are white
    rgb_commit = np.ones((n_gens, NUM_PERIODS, 3))  # white background

    from matplotlib.colors import to_rgb
    for i in range(n_gens):
        ft = fuel_types[i]
        color_rgb = to_rgb(FUEL_COLORS.get(ft, "#999"))
        for t in range(NUM_PERIODS):
            if commitment[i, t] > 0.5:
                rgb_commit[i, t, :] = color_rgb

    ax_fuel.imshow(rgb_commit, aspect="auto", interpolation="nearest")
    ax_fuel.set_xlabel("Hour of Day", fontsize=12)
    ax_fuel.set_ylabel(f"Generator Index (n={n_gens}, sorted by fuel → capacity)", fontsize=11)
    ax_fuel.set_xticks(np.arange(NUM_PERIODS))
    ax_fuel.set_xticklabels([f"{h:02d}" for h in range(NUM_PERIODS)], fontsize=8)
    ax_fuel.set_title(
        f"Generator Commitment (ON/OFF) — {n_gens} Generators × 24h\n"
        "Colored by Fuel Type",
        fontsize=13, fontweight="bold",
    )

    # Fuel type group separators and labels
    fuel_boundaries = []
    prev_ft = None
    for i, ft in enumerate(fuel_types):
        if ft != prev_ft:
            if prev_ft is not None:
                fuel_boundaries.append((start_idx, i - 1, prev_ft))
            start_idx = i
            prev_ft = ft
    if prev_ft is not None:
        fuel_boundaries.append((start_idx, n_gens - 1, prev_ft))

    for start, end, ft in fuel_boundaries:
        mid = (start + end) / 2
        # Horizontal separator
        if start > 0:
            ax_fuel.axhline(y=start - 0.5, color="black", linewidth=0.8, alpha=0.6)
            ax_lf.axhline(y=start - 0.5, color="black", linewidth=0.8, alpha=0.6)
        # Label on the right side
        count = end - start + 1
        cap = sum(sorted_gens[j].capacity_mw for j in range(start, end + 1))
        label = f"{ft.upper()}\n({count}, {cap/1000:,.0f}GW)"
        ax_fuel.text(
            -1.5, mid, label,
            ha="right", va="center", fontsize=7,
            color=FUEL_COLORS.get(ft, "#333"), fontweight="bold",
        )

    # Fuel legend
    legend_patches = []
    for ft in FUEL_ORDER:
        if ft in fuel_types:
            legend_patches.append(
                mpatches.Patch(color=FUEL_COLORS.get(ft, "#999"), label=ft.upper())
            )
    ax_fuel.legend(
        handles=legend_patches, loc="lower left", fontsize=8,
        ncol=2, framealpha=0.9, title="Fuel Type", title_fontsize=9,
    )

    # --- Right panel: Load factor heatmap ---
    im = ax_lf.imshow(
        load_factor, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_lf.set_xlabel("Hour of Day", fontsize=12)
    ax_lf.set_xticks(np.arange(NUM_PERIODS))
    ax_lf.set_xticklabels([f"{h:02d}" for h in range(NUM_PERIODS)], fontsize=8)
    ax_lf.set_yticklabels([])
    ax_lf.set_title(
        f"Generator Load Factor — {n_gens} Generators × 24h\n"
        "(Output / Capacity)",
        fontsize=13, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax_lf, shrink=0.6, pad=0.02)
    cbar.set_label("Load Factor (0=idle, 1=full load)", fontsize=10)

    # Statistics annotation
    n_committed = int(commitment.sum())
    n_total_slots = n_gens * NUM_PERIODS
    commit_pct = n_committed / n_total_slots * 100
    avg_lf = load_factor[commitment > 0.5].mean() if (commitment > 0.5).any() else 0

    stats_text = (
        f"Committed slots: {n_committed:,} / {n_total_slots:,} ({commit_pct:.1f}%)\n"
        f"Avg load factor (when ON): {avg_lf:.2f}\n"
        f"Generators: {n_gens} | Hours: {NUM_PERIODS}"
    )
    fig.text(
        0.5, 0.01, stats_text,
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        "Unit Commitment — Individual Generator Schedule Matrix (All Japan)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load generators
    print("=" * 60)
    print("STEP 1: Loading generators from XML")
    print("=" * 60)
    gens = load_generators()
    total_cap = sum(g.capacity_mw for g in gens)

    # Step 2: Generate demand profile
    print("\n" + "=" * 60)
    print("STEP 2: Generating demand profile")
    print("=" * 60)
    demand = make_demand_profile(total_cap)
    print(f"Peak demand: {max(demand):,.0f} MW ({DEMAND_FACTOR*100:.0f}% of {total_cap:,.0f} MW)")
    print(f"Min demand:  {min(demand):,.0f} MW")

    # Step 3: Run UC
    print("\n" + "=" * 60)
    print("STEP 3: Solving Unit Commitment")
    print("=" * 60)
    result = run_uc(gens, demand)

    if not result.schedules:
        print("ERROR: No schedules produced. Exiting.")
        return

    # Step 4: Visualizations
    print("\n" + "=" * 60)
    print("STEP 4: Generating visualizations")
    print("=" * 60)
    plot_dispatch_stack(
        result, gens, demand,
        os.path.join(OUTPUT_DIR, "national_dispatch_stack.png"),
    )
    plot_cost_breakdown(
        result, gens,
        os.path.join(OUTPUT_DIR, "national_cost_breakdown.png"),
    )
    plot_commitment_heatmap(
        result, gens,
        os.path.join(OUTPUT_DIR, "national_commitment_heatmap.png"),
    )
    plot_summary_dashboard(
        result, gens, demand,
        os.path.join(OUTPUT_DIR, "national_summary_dashboard.png"),
    )
    plot_generator_matrix(
        result, gens,
        os.path.join(OUTPUT_DIR, "national_generator_matrix.png"),
    )

    # Step 5: Export results
    print("\n" + "=" * 60)
    print("STEP 5: Exporting results")
    print("=" * 60)
    csv_path = export_uc_result_csv(result, os.path.join(OUTPUT_DIR, "national_uc_result.csv"))
    print(f"  CSV: {csv_path}")
    xml_path = export_uc_result_xml(result, os.path.join(OUTPUT_DIR, "national_uc_result.xml"))
    print(f"  XML: {xml_path}")

    print("\n" + "=" * 60)
    print(f"DONE! All outputs in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
