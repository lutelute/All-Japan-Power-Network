"""Visualize OSM-sourced Shikoku power grid from worktree 006 GeoJSON data.

Usage:
    python scripts/visualize_osm_network.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import geopandas as gpd
import numpy as np

# Japanese font setup (macOS)
_jp_font_candidates = [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/b7a6a6575a699e801915b73b9e1e75c74a3404ce.asset/AssetData/YuGothic-Bold.otf",
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/ee89e7987a76cc8cfdff36c96bd7bc77655b343e.asset/AssetData/YuGothic-Medium.otf",
]
for _p in _jp_font_candidates:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break

# Paths to 006 worktree OSM data
WORKTREE = ".auto-claude/worktrees/tasks/006-convert-kml-to-open-data-format"
SUB_PATH = os.path.join(WORKTREE, "data/raw/osm/shikoku_substations.geojson")
LINE_PATH = os.path.join(WORKTREE, "data/raw/osm/shikoku_lines.geojson")
OUTPUT_PATH = "output/osm_shikoku_network.png"


def classify_voltage(v):
    """Classify voltage (in volts) to kV bracket."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    try:
        parts = str(v).split(";")
        max_v = max(float(p.strip()) for p in parts if p.strip())
    except (ValueError, TypeError):
        return 0
    return max_v / 1000.0


VOLTAGE_COLORS = {
    500: "#d62728",   # red
    275: "#ff7f0e",   # orange
    220: "#e377c2",   # pink
    187: "#9467bd",   # purple
    154: "#2ca02c",   # green
    132: "#17becf",   # cyan
    110: "#1f77b4",   # blue
    77:  "#bcbd22",   # yellow-green
    66:  "#8c564b",   # brown
    0:   "#cccccc",   # grey (unknown)
}


def get_voltage_color(kv):
    thresholds = sorted(VOLTAGE_COLORS.keys(), reverse=True)
    for t in thresholds:
        if kv >= t:
            return VOLTAGE_COLORS[t]
    return "#cccccc"


def get_voltage_width(kv):
    if kv >= 500: return 2.5
    if kv >= 275: return 1.8
    if kv >= 220: return 1.5
    if kv >= 187: return 1.2
    if kv >= 154: return 1.0
    if kv >= 110: return 0.7
    if kv >= 66:  return 0.4
    return 0.3


def main():
    print("Loading OSM GeoJSON data...")
    subs = gpd.read_file(SUB_PATH)
    lines = gpd.read_file(LINE_PATH)
    print(f"  Substations: {len(subs)}")
    print(f"  Lines/Cables: {len(lines)}")

    # Parse voltages
    if "voltage" in subs.columns:
        subs["voltage_kv"] = subs["voltage"].apply(classify_voltage)
    else:
        subs["voltage_kv"] = 0

    if "voltage" in lines.columns:
        lines["voltage_kv"] = lines["voltage"].apply(classify_voltage)
    else:
        lines["voltage_kv"] = 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # --- Panel 1: Full network by voltage class ---
    ax1 = axes[0]
    ax1.set_facecolor("#f0f4f8")
    ax1.set_title("四国電力系統 (OSMオープンデータ)", fontsize=16, fontweight="bold")

    # Plot lines by voltage
    for _, row in lines.iterrows():
        kv = row.get("voltage_kv", 0)
        color = get_voltage_color(kv)
        width = get_voltage_width(kv)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            ax1.plot(xs, ys, color=color, linewidth=width, alpha=0.7, zorder=1)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                xs, ys = part.xy
                ax1.plot(xs, ys, color=color, linewidth=width, alpha=0.7, zorder=1)

    # Plot substations
    for _, row in subs.iterrows():
        kv = row.get("voltage_kv", 0)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y
        else:
            c = geom.centroid
            x, y = c.x, c.y
        size = 15 if kv >= 187 else (8 if kv >= 66 else 4)
        color = get_voltage_color(kv)
        ax1.scatter(x, y, s=size, c=color, edgecolors="black",
                    linewidths=0.3, zorder=3, alpha=0.9)

    ax1.set_xlabel("経度")
    ax1.set_ylabel("緯度")
    ax1.set_aspect("equal")

    # Legend for voltage classes
    legend_items = []
    for kv_label, color in [
        ("500 kV", "#d62728"), ("275 kV", "#ff7f0e"), ("220 kV", "#e377c2"),
        ("187 kV", "#9467bd"), ("154 kV", "#2ca02c"), ("110 kV", "#1f77b4"),
        ("66 kV", "#8c564b"), ("不明", "#cccccc"),
    ]:
        legend_items.append(plt.Line2D([0], [0], color=color, linewidth=2, label=kv_label))
    ax1.legend(handles=legend_items, loc="lower left", fontsize=8, title="電圧階級")

    # --- Panel 2: Backbone only (>= 110 kV) ---
    ax2 = axes[1]
    ax2.set_facecolor("#f0f4f8")
    ax2.set_title("基幹系統のみ (110kV以上)", fontsize=16, fontweight="bold")

    for _, row in lines.iterrows():
        kv = row.get("voltage_kv", 0)
        if kv < 110:
            continue
        color = get_voltage_color(kv)
        width = get_voltage_width(kv) * 1.5
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            ax2.plot(xs, ys, color=color, linewidth=width, alpha=0.8, zorder=1)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                xs, ys = part.xy
                ax2.plot(xs, ys, color=color, linewidth=width, alpha=0.8, zorder=1)

    # Plot backbone substations
    for _, row in subs.iterrows():
        kv = row.get("voltage_kv", 0)
        if kv < 110:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y
        else:
            c = geom.centroid
            x, y = c.x, c.y
        size = 40 if kv >= 187 else 20
        color = get_voltage_color(kv)
        ax2.scatter(x, y, s=size, c=color, edgecolors="black",
                    linewidths=0.5, zorder=3, alpha=0.9)

        # Label major substations
        name = row.get("name", "")
        if name and isinstance(name, str) and kv >= 187:
            ax2.annotate(name, (x, y), fontsize=6, ha="left", va="bottom",
                         xytext=(3, 3), textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))

    ax2.set_xlabel("経度")
    ax2.set_ylabel("緯度")
    ax2.set_aspect("equal")

    # Stats text box
    n_lines_backbone = sum(1 for _, r in lines.iterrows() if r.get("voltage_kv", 0) >= 110)
    n_subs_backbone = sum(1 for _, r in subs.iterrows() if r.get("voltage_kv", 0) >= 110)
    stats_text = (
        f"データソース: OpenStreetMap\n"
        f"変電所: {len(subs)} (うち基幹: {n_subs_backbone})\n"
        f"送電線: {len(lines)} (うち基幹: {n_lines_backbone})"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    legend_items_b = []
    for kv_label, color in [
        ("500 kV", "#d62728"), ("275 kV", "#ff7f0e"), ("220 kV", "#e377c2"),
        ("187 kV", "#9467bd"), ("154 kV", "#2ca02c"), ("110 kV", "#1f77b4"),
    ]:
        legend_items_b.append(plt.Line2D([0], [0], color=color, linewidth=2, label=kv_label))
    ax2.legend(handles=legend_items_b, loc="lower left", fontsize=8, title="電圧階級")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
