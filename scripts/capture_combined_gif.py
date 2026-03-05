#!/usr/bin/env python3
"""Generate a combined Network + Ybus side-by-side GIF.

Left: Leaflet map (region-colored lines via Playwright)
Right: Ybus sparsity pattern (matplotlib)

Both sync frame-by-frame per region.

Output: docs/assets/gif/network_ybus_tour.gif

Usage:
    python scripts/capture_combined_gif.py

Requires: playwright, pandapower, matplotlib, Pillow, ffmpeg
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

# Japanese font for matplotlib
_JP_FONT = None
for _fname in ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Noto Sans CJK JP", "BIZ UDGothic"]:
    if any(f.name == _fname for f in fm.fontManager.ttflist):
        _JP_FONT = _fname
        break
if _JP_FONT:
    plt.rcParams["font.family"] = _JP_FONT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs", "assets", "gif")
FRAME_DIR = os.path.join(ROOT, ".tmp_combined_frames")
SITE_URL = "https://lutelute.github.io/All-Japan-Grid/"

MAP_W, MAP_H = 800, 600
GIF_FPS = 4
HOLD = 6          # frames per region (6/4fps = 1.5s)
HOLD_ALL = 10     # frames for all-Japan (10/4fps = 2.5s)
TRANSITION = 6    # frames during zoom animation

REGIONS = [
    {"id": None,       "label": "All Japan",  "hold": HOLD_ALL},
    {"id": "hokkaido", "label": "Hokkaido",   "hold": HOLD},
    {"id": "tohoku",   "label": "Tohoku",     "hold": HOLD},
    {"id": "tokyo",    "label": "Tokyo",      "hold": HOLD},
    {"id": "chubu",    "label": "Chubu",      "hold": HOLD},
    {"id": "hokuriku", "label": "Hokuriku",    "hold": HOLD},
    {"id": "kansai",   "label": "Kansai",     "hold": HOLD},
    {"id": "chugoku",  "label": "Chugoku",    "hold": HOLD},
    {"id": "shikoku",  "label": "Shikoku",    "hold": HOLD},
    {"id": "kyushu",   "label": "Kyushu",     "hold": HOLD},
    {"id": "okinawa",  "label": "Okinawa",    "hold": HOLD},
    {"id": None,       "label": "All Japan",  "hold": HOLD_ALL},
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
    "okinawa": "#469990",
}


# ── Ybus builder (direct from line impedance, no power flow needed) ──

ALL_REGION_IDS = [
    "hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
    "kansai", "chugoku", "shikoku", "kyushu", "okinawa",
]


def _build_net(region_id):
    """Build pandapower net for a single region."""
    from src.server.geojson_parser import build_grid_network
    from src.converter.pandapower_builder import PandapowerBuilder

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


def build_ybus_for_region(region_id):
    """Build Ybus for a single region, or all regions merged for All-Japan."""
    from scipy import sparse as sp
    import pandapower as pp

    if region_id is None:
        # All-Japan: merge all regions into one big Ybus
        all_buses = 0
        all_lines = 0
        blocks = []
        for rid in ALL_REGION_IDS:
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

    net = _build_net(region_id)
    if net is None:
        return None, 0, 0
    return _ybus_from_net(net)


def render_ybus_png(Ybus, label, region_id, n_bus, n_line, out_path):
    color = REGION_COLORS.get(region_id, "#ff7f0e")
    ja = REGION_JA.get(region_id, "") if region_id else ""

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if Ybus is not None and Ybus.nnz > 0:
        coo = Ybus.tocoo()
        # Scale dot size: smaller matrices get bigger dots
        n = Ybus.shape[0]
        dot_size = max(0.5, min(8, 800 / n))
        ax.scatter(coo.col, coo.row, s=dot_size, c=color, alpha=0.8, marker="s", linewidths=0)
        ax.set_xlim(-0.5, Ybus.shape[0] - 0.5)
        ax.set_ylim(Ybus.shape[0] - 0.5, -0.5)
        n = Ybus.shape[0]
        density = Ybus.nnz / (n * n) * 100 if n > 0 else 0
        subtitle = f"{n} buses | {Ybus.nnz:,} non-zeros | {density:.2f}%"
    elif region_id is None:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=40, color="#555", fontweight="bold")
        subtitle = "10 regions combined"
    else:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=40, color="#555", fontweight="bold")
        subtitle = f"{n_bus} buses | {n_line} lines"

    title = label
    if ja:
        title += f"  ({ja})"
    ax.set_title(title, color="#fff", fontsize=16, fontweight="bold", pad=12)
    fig.text(0.5, 0.02, subtitle, ha="center", color="#aaa", fontsize=10)
    ax.set_xlabel("Bus index", color="#888", fontsize=9)
    ax.set_ylabel("Bus index", color="#888", fontsize=9)
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.text(0.5, 0.94, "Ybus Sparsity Pattern", ha="center", color="#e94560",
             fontsize=11, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def combine_side_by_side(map_png, ybus_png, out_path):
    img_map = Image.open(map_png)
    img_ybus = Image.open(ybus_png)

    # Resize ybus to match map height
    h = img_map.height
    ybus_w = int(img_ybus.width * h / img_ybus.height)
    img_ybus = img_ybus.resize((ybus_w, h), Image.LANCZOS)

    combined = Image.new("RGB", (img_map.width + ybus_w, h))
    combined.paste(img_map, (0, 0))
    combined.paste(img_ybus, (img_map.width, 0))
    combined.save(out_path)


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)
    map_dir = os.path.join(FRAME_DIR, "map")
    ybus_dir = os.path.join(FRAME_DIR, "ybus")
    os.makedirs(map_dir)
    os.makedirs(ybus_dir)

    # ── Pre-render all Ybus PNGs ──
    print("Building Ybus matrices...")
    ybus_cache = {}
    for r in REGIONS:
        rid = r["id"]
        key = rid or "all"
        if key in ybus_cache:
            continue
        Ybus, nb, nl = build_ybus_for_region(rid)
        png = os.path.join(ybus_dir, f"{key}.png")
        render_ybus_png(Ybus, r["label"], rid, nb, nl, png)
        ybus_cache[key] = png
        if Ybus is not None and Ybus.nnz > 0:
            print(f"  {r['label']}: {Ybus.shape[0]} buses, {Ybus.nnz} nnz")
        else:
            print(f"  {r['label']}: {'(all-japan)' if rid is None else '(sparse)'}")

    # ── Playwright: capture map frames ──
    print("\nCapturing map frames...")
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": MAP_W, "height": MAP_H},
            device_scale_factor=2,
        )
        page = await context.new_page()

        print(f"  Loading {SITE_URL}")
        await page.goto(SITE_URL, wait_until="networkidle")
        await asyncio.sleep(3)

        # Switch to area tab
        await page.click('[data-tab="tab-area"]')
        await asyncio.sleep(2)

        # Hide subs + plants
        await page.evaluate("""() => {
            document.querySelectorAll('.layer-cb[data-layer="subs"]').forEach(cb => {
                if (cb.checked) { cb.checked = false; cb.dispatchEvent(new Event("change")); }
            });
            document.querySelectorAll('.layer-cb[data-layer="plants"]').forEach(cb => {
                if (cb.checked) { cb.checked = false; cb.dispatchEvent(new Event("change")); }
            });
        }""")
        await asyncio.sleep(1)

        # Hide sidebar and any markers (pointer in center of Japan)
        await page.evaluate("""() => {
            document.getElementById("sidebar").style.display = "none";
            document.querySelectorAll(".leaflet-marker-icon, .leaflet-marker-shadow").forEach(el => el.style.display = "none");
            if (typeof map !== "undefined") map.invalidateSize();
        }""")
        await asyncio.sleep(1)

        frame_num = 0

        for region in REGIONS:
            rid = region["id"]
            key = rid or "all"
            print(f"  {region['label']}...")

            # Navigate and wait for tiles to fully load
            if rid:
                await page.evaluate(f'selectRegion("{rid}")')
                # Okinawa: zoom into main island (default bbox is too wide)
                if rid == "okinawa":
                    await page.evaluate('map.setView([26.5, 127.8], 10)')
            else:
                await page.evaluate('selectRegion(null); map.setView([35.5, 136.0], 5)')

            # Hide markers again (selectRegion may add them)
            await page.evaluate("""() => {
                document.querySelectorAll(".leaflet-marker-icon, .leaflet-marker-shadow").forEach(el => el.style.display = "none");
            }""")

            # Wait for zoom animation + tile loading
            await asyncio.sleep(3)

            # Take one stable screenshot, reuse for all hold frames
            map_png = os.path.join(map_dir, f"region_{key}.png")
            await page.screenshot(path=map_png)

            # Generate hold frames (map + Ybus perfectly synced)
            for i in range(region["hold"]):
                combined = os.path.join(FRAME_DIR, f"frame_{frame_num:05d}.png")
                combine_side_by_side(map_png, ybus_cache[key], combined)
                frame_num += 1

        await browser.close()

    # ── Build GIF ──
    print("\nBuilding GIF...")
    out_path = os.path.join(OUT_DIR, "network_ybus_tour.gif")
    palette = os.path.join(FRAME_DIR, "palette.png")

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(GIF_FPS),
        "-i", os.path.join(FRAME_DIR, "frame_%05d.png"),
        "-vf", f"scale=-1:-1,palettegen=max_colors=128:stats_mode=diff",
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
    asyncio.run(main())
