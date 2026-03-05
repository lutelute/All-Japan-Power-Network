#!/usr/bin/env python3
"""Generate a Ybus sparsity pattern GIF cycling through all regions.

Flow: All Japan (merged) → Hokkaido → Tohoku → ... → Okinawa → All Japan

Output: docs/assets/gif/ybus_tour.gif

Usage:
    python scripts/capture_ybus_gif.py

Requires: pandapower, matplotlib, ffmpeg
"""

import json
import os
import subprocess
import sys
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Japanese font
for _fname in ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Noto Sans CJK JP", "BIZ UDGothic"]:
    if any(f.name == _fname for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _fname
        break

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.server.geojson_parser import build_grid_network
from src.converter.pandapower_builder import PandapowerBuilder

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs", "assets", "gif")
FRAME_DIR = os.path.join(ROOT, ".tmp_ybus_frames")
GIF_FPS = 2  # slower for Ybus — each region holds ~1.5s

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

REGION_COLORS = {
    "hokkaido": "#e6194b", "tohoku": "#3cb44b", "tokyo": "#4363d8",
    "chubu": "#f58231", "hokuriku": "#911eb4", "kansai": "#42d4f4",
    "chugoku": "#f032e6", "shikoku": "#bfef45", "kyushu": "#fabebe",
    "okinawa": "#469990", "all": "#ff7f0e",
}


def _build_net(region_id):
    """Build pandapower net for a single region."""
    sub_path = os.path.join(DATA_DIR, f"{region_id}_substations.geojson")
    line_path = os.path.join(DATA_DIR, f"{region_id}_lines.geojson")
    if not os.path.exists(sub_path) or not os.path.exists(line_path):
        return None
    with open(sub_path, "r", encoding="utf-8") as f:
        sub_fc = json.load(f)
    with open(line_path, "r", encoding="utf-8") as f:
        line_fc = json.load(f)
    network = build_grid_network(sub_fc, line_fc, region_id)
    result = PandapowerBuilder().build(network)
    return result.net


def _ybus_from_net(net):
    """Build Ybus directly from pandapower line table (no power flow).

    Reorders buses by lat+lon (NW→SE) so the sparsity pattern is more
    square / block-diagonal.
    """
    from scipy import sparse as sp
    import numpy as np

    n = len(net.bus)
    if n == 0 or len(net.line) == 0:
        return None, n, len(net.line)

    # Build geographic permutation (lat + lon, descending = NW first)
    lats = np.zeros(n)
    lons = np.zeros(n)
    for idx in range(n):
        geo = net.bus.at[idx, "geodata"] if "geodata" in net.bus.columns else None
        if geo is not None and len(geo) == 2:
            lats[idx] = geo[0]
            lons[idx] = geo[1]
    perm = np.argsort(-(lats + lons))
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)

    Y = sp.lil_matrix((n, n), dtype=complex)
    for _, line in net.line.iterrows():
        if not line.in_service:
            continue
        i, j = inv_perm[int(line.from_bus)], inv_perm[int(line.to_bus)]
        if i >= n or j >= n:
            continue
        r = line.r_ohm_per_km * line.length_km
        x = line.x_ohm_per_km * line.length_km
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y = 1.0 / z
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    Ybus = Y.tocsc()
    return Ybus, n, len(net.line)

    Y = sp.lil_matrix((n, n), dtype=complex)
    for _, line in net.line.iterrows():
        if not line.in_service:
            continue
        i, j = inv_perm[int(line.from_bus)], inv_perm[int(line.to_bus)]
        if i >= n or j >= n:
            continue
        r = line.r_ohm_per_km * line.length_km
        x = line.x_ohm_per_km * line.length_km
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y = 1.0 / z
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    Ybus = Y.tocsc()
    return Ybus, n, len(net.line)


def build_ybus_for_region(region):
    """Build Ybus for a single region, or all regions merged for All-Japan."""
    from scipy import sparse as sp

    if region == "all":
        blocks = []
        all_buses = 0
        all_lines = 0
        for rid in REGIONS:
            net = _build_net(rid)
            if net is None or len(net.bus) == 0:
                continue
            Ybus, nb, nl = _ybus_from_net(net)
            if Ybus is not None and Ybus.nnz > 0:
                blocks.append(Ybus)
            all_buses += nb
            all_lines += nl
        if not blocks:
            return None, all_buses, all_lines
        Ybus = sp.block_diag(blocks, format="csc")
        return Ybus, all_buses, all_lines

    net = _build_net(region)
    if net is None:
        return None, 0, 0
    return _ybus_from_net(net)


def plot_ybus_frame(Ybus, region_label, region_ja, n_bus, n_line, color, frame_path):
    """Plot Ybus sparsity pattern and save as PNG."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if Ybus is not None and Ybus.nnz > 0:
        # Get sparse structure
        coo = Ybus.tocoo()
        n = Ybus.shape[0]
        dot_size = max(0.5, min(8, 800 / n))
        ax.scatter(coo.col, coo.row, s=dot_size, c=color, alpha=0.8, marker="s", linewidths=0)
        ax.set_xlim(-0.5, Ybus.shape[0] - 0.5)
        ax.set_ylim(Ybus.shape[0] - 0.5, -0.5)
        nnz = Ybus.nnz
        n = Ybus.shape[0]
        density = nnz / (n * n) * 100 if n > 0 else 0
        subtitle = f"{n} buses  |  {nnz:,} non-zeros  |  {density:.2f}% density"
    else:
        subtitle = f"{n_bus} buses  |  {n_line} lines  |  (Ybus unavailable)"
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=40, color="#555", fontweight="bold")

    # Title
    title = f"{region_label}"
    if region_ja:
        title += f"  ({region_ja})"
    ax.set_title(title, color="#fff", fontsize=16, fontweight="bold", pad=12)

    # Subtitle
    fig.text(0.5, 0.02, subtitle, ha="center", color="#aaa", fontsize=10)

    # Axis labels
    ax.set_xlabel("Bus index", color="#888", fontsize=9)
    ax.set_ylabel("Bus index", color="#888", fontsize=9)
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")

    # Ybus label
    fig.text(0.5, 0.94, "Ybus Sparsity Pattern", ha="center", color="#e94560",
             fontsize=11, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(frame_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)

    frame_num = 0
    hold_frames = 3  # each region shown for hold_frames/GIF_FPS seconds

    # Tour: All Japan → per-region → All Japan
    tour = ["all"] + REGIONS + ["all"]

    for region in tour:
        print(f"  {region}...", end=" ", flush=True)
        Ybus, n_bus, n_line = build_ybus_for_region(region)

        if region == "all":
            label = "All Japan"
            ja = "全国"
        else:
            label = region.title()
            ja = REGION_JA.get(region, "")
        color = REGION_COLORS.get(region, "#ff7f0e")

        if Ybus is not None:
            print(f"{Ybus.shape[0]} buses, {Ybus.nnz} nnz")
        else:
            print(f"{n_bus} buses (Ybus failed)")

        for _ in range(hold_frames):
            fpath = os.path.join(FRAME_DIR, f"frame_{frame_num:05d}.png")
            plot_ybus_frame(Ybus, label, ja, n_bus, n_line, color, fpath)
            frame_num += 1

    # Build GIF
    print("Building GIF...")
    out_path = os.path.join(OUT_DIR, "ybus_tour.gif")
    palette = os.path.join(FRAME_DIR, "palette.png")

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(GIF_FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%05d.png"),
        "-vf", "palettegen=max_colors=128:stats_mode=diff",
        palette,
    ], capture_output=True)

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(GIF_FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%05d.png"),
        "-i", palette,
        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", out_path,
    ], capture_output=True)

    shutil.rmtree(FRAME_DIR)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone! {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
