#!/usr/bin/env python3
"""UC solver demo: run a realistic Japan 9-region scenario and produce visualizations.

Creates a multi-fuel-type generator fleet across 3 regions (Tokyo, Chubu, Kansai),
solves a 24-hour day-ahead unit commitment, and outputs 4 PNG figures:

1. Generator dispatch stack area chart (MW per fuel type over 24h)
2. Cost breakdown bar chart (fuel / startup / no-load per generator)
3. Commitment heatmap (on/off status for each generator × hour)
4. Summary dashboard (demand vs supply, reserve margin, cost pie)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
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
from src.uc.result_exporter import export_uc_result_csv

# ── Color palette ──────────────────────────────────────────────────
FUEL_COLORS = {
    "nuclear": "#7B2D8E",
    "coal":    "#4A4A4A",
    "lng":     "#E8832A",
    "oil":     "#C44E52",
    "hydro":   "#2196F3",
    "solar":   "#FFD700",
    "wind":    "#4CAF50",
    "biomass": "#8BC34A",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "uc")


def make_japan_generators():
    """Create a realistic 15-generator fleet across 3 regions."""
    gens = [
        # ── Tokyo (関東) ──
        Generator(
            id="tokyo_nuc_01", name="柏崎刈羽原発", capacity_mw=1100,
            fuel_type="nuclear", region="tokyo",
            startup_cost=50000, shutdown_cost=20000,
            min_up_time_h=12, min_down_time_h=12,
            fuel_cost_per_mwh=1500, no_load_cost=800, labor_cost_per_h=500,
        ),
        Generator(
            id="tokyo_coal_01", name="磯子火力(石炭)", capacity_mw=600,
            fuel_type="coal", region="tokyo",
            startup_cost=8000, shutdown_cost=3000,
            min_up_time_h=6, min_down_time_h=4,
            ramp_up_mw_per_h=120, ramp_down_mw_per_h=120,
            fuel_cost_per_mwh=4500, no_load_cost=600, labor_cost_per_h=400,
        ),
        Generator(
            id="tokyo_lng_01", name="富津LNG-1", capacity_mw=500,
            fuel_type="lng", region="tokyo",
            startup_cost=5000, shutdown_cost=2000,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=7000, no_load_cost=400, labor_cost_per_h=300,
        ),
        Generator(
            id="tokyo_lng_02", name="富津LNG-2", capacity_mw=500,
            fuel_type="lng", region="tokyo",
            startup_cost=5000, shutdown_cost=2000,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=7200, no_load_cost=400, labor_cost_per_h=300,
        ),
        Generator(
            id="tokyo_oil_01", name="横須賀石油火力", capacity_mw=350,
            fuel_type="oil", region="tokyo",
            startup_cost=3000, shutdown_cost=1500,
            min_up_time_h=2, min_down_time_h=2,
            ramp_up_mw_per_h=250, ramp_down_mw_per_h=250,
            fuel_cost_per_mwh=9000, no_load_cost=300, labor_cost_per_h=200,
        ),
        # ── Chubu (中部) ──
        Generator(
            id="chubu_coal_01", name="碧南火力(石炭)", capacity_mw=700,
            fuel_type="coal", region="chubu",
            startup_cost=9000, shutdown_cost=3500,
            min_up_time_h=6, min_down_time_h=4,
            ramp_up_mw_per_h=140, ramp_down_mw_per_h=140,
            fuel_cost_per_mwh=4200, no_load_cost=650, labor_cost_per_h=400,
        ),
        Generator(
            id="chubu_lng_01", name="知多LNG", capacity_mw=400,
            fuel_type="lng", region="chubu",
            startup_cost=4500, shutdown_cost=1800,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=180, ramp_down_mw_per_h=180,
            fuel_cost_per_mwh=6800, no_load_cost=350, labor_cost_per_h=280,
        ),
        Generator(
            id="chubu_hydro_01", name="奥矢作水力", capacity_mw=200,
            fuel_type="hydro", region="chubu",
            startup_cost=500, shutdown_cost=200,
            min_up_time_h=1, min_down_time_h=1,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=0, no_load_cost=100, labor_cost_per_h=100,
        ),
        Generator(
            id="chubu_solar_01", name="メガソーラー浜松", capacity_mw=150,
            fuel_type="solar", region="chubu",
            startup_cost=0, shutdown_cost=0,
            min_up_time_h=1, min_down_time_h=1,
            fuel_cost_per_mwh=0, no_load_cost=0, labor_cost_per_h=50,
        ),
        # ── Kansai (関西) ──
        Generator(
            id="kansai_nuc_01", name="大飯原発", capacity_mw=900,
            fuel_type="nuclear", region="kansai",
            startup_cost=45000, shutdown_cost=18000,
            min_up_time_h=12, min_down_time_h=12,
            fuel_cost_per_mwh=1500, no_load_cost=750, labor_cost_per_h=480,
        ),
        Generator(
            id="kansai_coal_01", name="舞鶴火力(石炭)", capacity_mw=450,
            fuel_type="coal", region="kansai",
            startup_cost=7000, shutdown_cost=2800,
            min_up_time_h=5, min_down_time_h=4,
            ramp_up_mw_per_h=100, ramp_down_mw_per_h=100,
            fuel_cost_per_mwh=4600, no_load_cost=550, labor_cost_per_h=350,
        ),
        Generator(
            id="kansai_lng_01", name="姫路LNG", capacity_mw=400,
            fuel_type="lng", region="kansai",
            startup_cost=4800, shutdown_cost=1900,
            min_up_time_h=3, min_down_time_h=2,
            ramp_up_mw_per_h=180, ramp_down_mw_per_h=180,
            fuel_cost_per_mwh=7100, no_load_cost=380, labor_cost_per_h=290,
        ),
        Generator(
            id="kansai_hydro_01", name="黒部水力", capacity_mw=250,
            fuel_type="hydro", region="kansai",
            startup_cost=600, shutdown_cost=250,
            min_up_time_h=1, min_down_time_h=1,
            ramp_up_mw_per_h=250, ramp_down_mw_per_h=250,
            fuel_cost_per_mwh=0, no_load_cost=120, labor_cost_per_h=110,
        ),
        Generator(
            id="kansai_wind_01", name="淡路島ウィンドファーム", capacity_mw=100,
            fuel_type="wind", region="kansai",
            startup_cost=0, shutdown_cost=0,
            min_up_time_h=1, min_down_time_h=1,
            fuel_cost_per_mwh=0, no_load_cost=0, labor_cost_per_h=30,
        ),
        Generator(
            id="kansai_oil_01", name="関西石油火力", capacity_mw=300,
            fuel_type="oil", region="kansai",
            startup_cost=2800, shutdown_cost=1200,
            min_up_time_h=2, min_down_time_h=2,
            ramp_up_mw_per_h=200, ramp_down_mw_per_h=200,
            fuel_cost_per_mwh=9500, no_load_cost=280, labor_cost_per_h=180,
        ),
    ]
    return gens


def make_demand_profile():
    """Create a realistic Japanese 24h demand curve (summer weekday)."""
    # Typical shape: low at night, morning ramp, midday peak, evening shoulder
    base = 2500  # MW base load
    demand = [
        2500, 2400, 2300, 2250, 2300, 2500,   # 0-5h: nighttime low
        2800, 3200, 3600, 3900, 4100, 4300,   # 6-11h: morning ramp
        4400, 4500, 4400, 4200, 4000, 3800,   # 12-17h: afternoon peak & decline
        3600, 3400, 3200, 3000, 2800, 2600,   # 18-23h: evening decline
    ]
    return demand


def run_uc():
    """Solve the UC problem and return result + generators."""
    gens = make_japan_generators()
    demand = make_demand_profile()

    th = TimeHorizon(num_periods=24, period_duration_h=1.0)
    dp = DemandProfile(demands=demand)
    params = UCParameters(
        generators=gens,
        demand=dp,
        time_horizon=th,
        reserve_margin=0.05,
    )

    print("Solving UC: 15 generators, 24 hours, 5% reserve margin...")
    result = solve_uc(params)
    print(f"Status: {result.status}")
    print(f"Total cost: ¥{result.total_cost:,.0f}")
    print(f"Solve time: {result.solve_time_s:.2f}s")
    print(f"Generators scheduled: {result.num_generators}")

    return result, gens, demand


def plot_dispatch_stack(result, gens, demand, output_path):
    """Fig 1: Stacked area chart of power dispatch by fuel type."""
    hours = np.arange(24)
    gen_map = {g.id: g for g in gens}

    # Aggregate power output by fuel type
    fuel_order = ["nuclear", "coal", "hydro", "biomass", "wind", "solar", "lng", "oil"]
    stacks = {}
    for ft in fuel_order:
        stacks[ft] = np.zeros(24)

    for sched in result.schedules:
        g = gen_map[sched.generator_id]
        ft = g.fuel_type_enum.value
        if ft not in stacks:
            stacks[ft] = np.zeros(24)
        stacks[ft] += np.array(sched.power_output_mw)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Stack from bottom
    bottoms = np.zeros(24)
    for ft in fuel_order:
        if ft in stacks and stacks[ft].sum() > 0:
            ax.fill_between(
                hours, bottoms, bottoms + stacks[ft],
                label=ft.upper(), color=FUEL_COLORS.get(ft, "#999"),
                alpha=0.85, linewidth=0.5, edgecolor="white",
            )
            bottoms += stacks[ft]

    # Demand line
    ax.plot(hours, demand, "k-", linewidth=2.5, label="Demand", zorder=5)
    ax.plot(hours, demand, "ko", markersize=4, zorder=5)

    ax.set_xlim(0, 23)
    ax.set_ylim(0, max(demand) * 1.15)
    ax.set_xlabel("Hour of Day", fontsize=13)
    ax.set_ylabel("Power Output (MW)", fontsize=13)
    ax.set_title("24-Hour Generator Dispatch Schedule — Japan UC Demo\n"
                 "(Tokyo + Chubu + Kansai, 15 generators)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, ncol=2, framealpha=0.9)
    ax.set_xticks(hours)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cost_breakdown(result, gens, output_path):
    """Fig 2: Horizontal bar chart of cost breakdown per generator."""
    gen_map = {g.id: g for g in gens}

    # Sort by total cost descending
    sorted_scheds = sorted(result.schedules, key=lambda s: s.total_cost, reverse=True)
    # Filter out zero-cost generators
    sorted_scheds = [s for s in sorted_scheds if s.total_cost > 0]

    names = [gen_map[s.generator_id].name for s in sorted_scheds]
    fuel_costs = [s.fuel_cost for s in sorted_scheds]
    startup_costs = [s.startup_cost for s in sorted_scheds]
    noload_costs = [s.no_load_cost for s in sorted_scheds]
    shutdown_costs = [s.shutdown_cost for s in sorted_scheds]

    y = np.arange(len(names))
    bar_h = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(y, fuel_costs, bar_h, label="Fuel Cost", color="#E8832A")
    ax.barh(y, noload_costs, bar_h, left=fuel_costs, label="No-Load + Labor", color="#2196F3")
    left2 = np.array(fuel_costs) + np.array(noload_costs)
    ax.barh(y, startup_costs, bar_h, left=left2, label="Startup Cost", color="#C44E52")
    left3 = left2 + np.array(startup_costs)
    ax.barh(y, shutdown_costs, bar_h, left=left3, label="Shutdown Cost", color="#9C27B0")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Cost (¥)", fontsize=13)
    ax.set_title("Generator Cost Breakdown — 24-Hour UC Result", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add total cost labels
    for i, s in enumerate(sorted_scheds):
        ax.text(s.total_cost + max(fuel_costs) * 0.01, i,
                f"¥{s.total_cost:,.0f}", va="center", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_commitment_heatmap(result, gens, output_path):
    """Fig 3: Heatmap showing on/off commitment status per generator per hour."""
    gen_map = {g.id: g for g in gens}

    # Sort: nuclear first, then coal, lng, etc.
    fuel_priority = {"nuclear": 0, "coal": 1, "lng": 2, "oil": 3,
                     "hydro": 4, "wind": 5, "solar": 6, "biomass": 7}
    sorted_scheds = sorted(
        result.schedules,
        key=lambda s: (fuel_priority.get(gen_map[s.generator_id].fuel_type_enum.value, 99),
                       -gen_map[s.generator_id].capacity_mw),
    )

    names = []
    data = []
    colors_left = []
    for s in sorted_scheds:
        g = gen_map[s.generator_id]
        names.append(f"{g.name} ({g.capacity_mw:.0f}MW)")
        # Encode: 0=off, power_fraction for on
        row = []
        for t in range(24):
            if s.commitment[t] == 1:
                row.append(s.power_output_mw[t] / g.capacity_mw if g.capacity_mw > 0 else 1.0)
            else:
                row.append(0.0)
        data.append(row)
        colors_left.append(FUEL_COLORS.get(g.fuel_type_enum.value, "#999"))

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Custom colormap: white (off) -> light blue -> deep blue (full load)
    cmap = plt.cm.YlOrRd
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[0] = [0.95, 0.95, 0.95, 1.0]  # Off = light gray
    custom_cmap = ListedColormap(cmap_colors)

    im = ax.imshow(data, aspect="auto", cmap=custom_cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    # Fuel type color bars on the left
    for i, c in enumerate(colors_left):
        ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.8, 1, color=c, clip_on=False))

    ax.set_xlabel("Hour of Day", fontsize=13)
    ax.set_title("Generator Commitment & Loading — 24-Hour UC Schedule\n"
                 "(color intensity = load factor)", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Load Factor (0=OFF, 1=Full Load)", fontsize=11)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(result, gens, demand, output_path):
    """Fig 4: Summary dashboard with demand/supply, reserve, cost pie, and stats."""
    gen_map = {g.id: g for g in gens}
    hours = np.arange(24)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (a) Supply vs Demand ──
    ax = axes[0, 0]
    total_supply = np.zeros(24)
    committed_cap = np.zeros(24)
    for sched in result.schedules:
        g = gen_map[sched.generator_id]
        total_supply += np.array(sched.power_output_mw)
        committed_cap += np.array(sched.commitment) * g.capacity_mw

    ax.fill_between(hours, committed_cap, alpha=0.2, color="#2196F3", label="Committed Capacity")
    ax.plot(hours, total_supply, "b-", linewidth=2, label="Total Supply")
    ax.plot(hours, demand, "r--", linewidth=2, label="Demand")
    reserve = np.array(demand) * 1.05
    ax.plot(hours, reserve, "g:", linewidth=1.5, label="Demand + 5% Reserve")
    ax.set_xlabel("Hour"); ax.set_ylabel("MW")
    ax.set_title("(a) Demand vs Supply & Reserve", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xticks(hours)

    # ── (b) Cost pie chart ──
    ax = axes[0, 1]
    total_fuel = sum(s.fuel_cost for s in result.schedules)
    total_startup = sum(s.startup_cost for s in result.schedules)
    total_shutdown = sum(s.shutdown_cost for s in result.schedules)
    total_noload = sum(s.no_load_cost for s in result.schedules)

    labels = []
    sizes = []
    colors = []
    for lbl, val, col in [("Fuel", total_fuel, "#E8832A"),
                           ("No-Load+Labor", total_noload, "#2196F3"),
                           ("Startup", total_startup, "#C44E52"),
                           ("Shutdown", total_shutdown, "#9C27B0")]:
        if val > 0:
            labels.append(lbl)
            sizes.append(val)
            colors.append(col)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title(f"(b) Total Cost Breakdown: ¥{result.total_cost:,.0f}", fontweight="bold")

    # ── (c) Number of committed generators per hour ──
    ax = axes[1, 0]
    n_committed = np.zeros(24)
    for sched in result.schedules:
        n_committed += np.array(sched.commitment)

    bars = ax.bar(hours, n_committed, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Hour"); ax.set_ylabel("# Generators ON")
    ax.set_title("(c) Number of Committed Generators per Hour", fontweight="bold")
    ax.set_xticks(hours)
    ax.set_ylim(0, len(gens) + 1)
    ax.axhline(y=len(gens), color="gray", linestyle="--", alpha=0.5, label=f"Total fleet: {len(gens)}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── (d) Statistics table ──
    ax = axes[1, 1]
    ax.axis("off")

    total_energy = sum(s.total_energy_mwh for s in result.schedules)
    total_demand_mwh = sum(demand)
    avg_cost_per_mwh = result.total_cost / total_energy if total_energy > 0 else 0
    peak_demand = max(demand)
    min_demand = min(demand)
    total_cap = sum(g.capacity_mw for g in gens)

    stats = [
        ["Metric", "Value"],
        ["Solver Status", result.status],
        ["Solve Time", f"{result.solve_time_s:.2f} s"],
        ["Total Cost", f"¥{result.total_cost:,.0f}"],
        ["Total Energy Generated", f"{total_energy:,.0f} MWh"],
        ["Total Demand", f"{total_demand_mwh:,.0f} MWh"],
        ["Avg. Cost / MWh", f"¥{avg_cost_per_mwh:,.0f}"],
        ["Peak Demand", f"{peak_demand:,.0f} MW"],
        ["Min Demand", f"{min_demand:,.0f} MW"],
        ["Total Installed Capacity", f"{total_cap:,.0f} MW"],
        ["Fleet Size", f"{len(gens)} generators"],
        ["Reserve Margin", "5%"],
    ]

    table = ax.table(cellText=stats[1:], colLabels=stats[0],
                     cellLoc="left", loc="center",
                     colWidths=[0.45, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(stats)):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f0f0")

    ax.set_title("(d) UC Result Summary", fontweight="bold", y=0.98)

    fig.suptitle("Unit Commitment Solver — Japan Grid Demo Results",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result, gens, demand = run_uc()

    if not result.is_optimal:
        print(f"WARNING: Solver did not find optimal solution (status={result.status})")
        for w in result.warnings:
            print(f"  - {w}")
        if not result.schedules:
            print("No schedules produced. Exiting.")
            return

    print("\nGenerating visualizations...")
    plot_dispatch_stack(result, gens, demand,
                        os.path.join(OUTPUT_DIR, "uc_dispatch_stack.png"))
    plot_cost_breakdown(result, gens,
                        os.path.join(OUTPUT_DIR, "uc_cost_breakdown.png"))
    plot_commitment_heatmap(result, gens,
                            os.path.join(OUTPUT_DIR, "uc_commitment_heatmap.png"))
    plot_summary_dashboard(result, gens, demand,
                            os.path.join(OUTPUT_DIR, "uc_summary_dashboard.png"))

    # Also export CSV
    csv_path = os.path.join(OUTPUT_DIR, "uc_result.csv")
    export_uc_result_csv(result, csv_path)
    print(f"  Saved CSV: {csv_path}")

    print("\nDone! All outputs in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
