"""Generate 3 interactive (zoomable) HTML grid maps using Plotly.

Outputs:
    output/interactive_n1.html       — N-1 contingency status
    output/interactive_voltage.html  — Voltage class coloring
    output/interactive_area.html     — Regional area coloring

Usage:
    python scripts/interactive_grid_maps.py
"""

import os
import sys

import plotly.graph_objects as go
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.kml_parser import KMLParser
from src.model.grid_network import GridNetwork
from src.model.substation import CapacityStatus

# ── Configuration ──────────────────────────────────────────────────

DATA_DIR = "data/raw"
OUTPUT_DIR = "output"
IC_PATH = "data/reference/interconnections.yaml"

REGIONS = {
    "hokkaido":  {"freq": 50, "name_ja": "北海道"},
    "tohoku":    {"freq": 50, "name_ja": "東北"},
    "tokyo":     {"freq": 50, "name_ja": "東京"},
    "chubu":     {"freq": 60, "name_ja": "中部"},
    "hokuriku":  {"freq": 60, "name_ja": "北陸"},
    "kansai":    {"freq": 60, "name_ja": "関西"},
    "chugoku":   {"freq": 60, "name_ja": "中国"},
    "shikoku":   {"freq": 60, "name_ja": "四国"},
    "kyushu":    {"freq": 60, "name_ja": "九州"},
}

AREA_COLORS = {
    "hokkaido": "#1f77b4",
    "tohoku":   "#ff7f0e",
    "tokyo":    "#2ca02c",
    "chubu":    "#d62728",
    "hokuriku": "#9467bd",
    "kansai":   "#8c564b",
    "chugoku":  "#e377c2",
    "shikoku":  "#7f7f7f",
    "kyushu":   "#bcbd22",
}

VOLTAGE_COLORS = {
    500: "#d62728",
    275: "#ff7f0e",
    220: "#e377c2",
    187: "#9467bd",
    154: "#2ca02c",
    132: "#17becf",
    110: "#1f77b4",
    77:  "#bcbd22",
    66:  "#8c564b",
    0:   "#cccccc",
}

N1_COLORS = {
    "zero_capacity_n1_ineligible": "#d62728",   # Red
    "zero_capacity_n1_eligible":   "#ff7f0e",   # Orange
    "available_capacity":          "#2ca02c",    # Green
    "unknown":                     "#aaaaaa",    # Grey
}

N1_LABELS = {
    "zero_capacity_n1_ineligible": "空容量なし (N-1不適格)",
    "zero_capacity_n1_eligible":   "空容量なし (N-1適格)",
    "available_capacity":          "空容量あり",
    "unknown":                     "不明",
}

VOLTAGE_WIDTHS = {500: 3.5, 275: 2.5, 220: 2.0, 187: 1.8, 154: 1.5,
                  132: 1.3, 110: 1.0, 77: 0.8, 66: 0.6, 0: 0.4}


def get_voltage_bracket(kv):
    for t in sorted(VOLTAGE_WIDTHS.keys(), reverse=True):
        if kv >= t:
            return t
    return 0


# ── Load all networks ─────────────────────────────────────────────

def load_national_network():
    parser = KMLParser()
    networks = []
    for region, info in REGIONS.items():
        kml_path = os.path.join(DATA_DIR, f"{region}.kml")
        if not os.path.exists(kml_path):
            print(f"  [SKIP] {kml_path} not found")
            continue
        net = parser.parse_file(kml_path, region, info["freq"])
        print(f"  {region}: {net.substation_count} subs, {net.line_count} lines")
        networks.append(net)
    national = GridNetwork.merge_regions(networks)
    print(f"  National: {national.substation_count} subs, {national.line_count} lines")
    return national


def load_interconnections():
    if not os.path.exists(IC_PATH):
        return []
    with open(IC_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("interconnections", [])


# ── Trace builders ─────────────────────────────────────────────────

def _line_traces_by_group(network, group_fn, color_fn, width_fn, name_fn,
                          legendgroup_fn=None):
    """Build Plotly Scattergeo traces for lines, grouped by a key function."""
    groups = {}
    for line in network.transmission_lines:
        if not line.coordinates or len(line.coordinates) < 2:
            continue
        key = group_fn(line)
        groups.setdefault(key, []).append(line)

    traces = []
    for key in sorted(groups.keys(), key=lambda k: str(k)):
        lats = []
        lons = []
        hovertexts = []
        for line in groups[key]:
            for lat, lon in line.coordinates:
                lats.append(lat)
                lons.append(lon)
            lats.append(None)
            lons.append(None)
            hovertexts.append(
                f"<b>{line.name}</b><br>"
                f"電圧: {line.voltage_kv:.0f} kV<br>"
                f"亘長: {line.length_km:.1f} km<br>"
                f"地域: {REGIONS.get(line.region, {}).get('name_ja', line.region)}<br>"
                f"N-1: {N1_LABELS.get(line.capacity_status.value, '不明')}"
            )
        color = color_fn(key)
        width = width_fn(key)
        lg = legendgroup_fn(key) if legendgroup_fn else str(key)
        trace = go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=color, width=width),
            name=name_fn(key),
            legendgroup=lg,
            hoverinfo="text",
            text=[ht for ht in hovertexts for _ in range(len(groups[key][0].coordinates) + 1)]
                if len(groups[key]) == 1 else None,
            showlegend=True,
        )
        traces.append(trace)
    return traces


def _sub_traces_by_group(network, group_fn, color_fn, size_fn, name_fn,
                         legendgroup_fn=None):
    """Build Plotly Scattergeo traces for substations, grouped by a key function."""
    groups = {}
    for sub in network.substations:
        key = group_fn(sub)
        groups.setdefault(key, []).append(sub)

    traces = []
    for key in sorted(groups.keys(), key=lambda k: str(k)):
        subs = groups[key]
        lats = [s.latitude for s in subs]
        lons = [s.longitude for s in subs]
        texts = [
            f"<b>{s.name}</b><br>"
            f"電圧: {s.voltage_kv:.0f} kV<br>"
            f"地域: {REGIONS.get(s.region, {}).get('name_ja', s.region)}<br>"
            f"ID: {s.id}"
            for s in subs
        ]
        color = color_fn(key)
        sz = size_fn(key)
        lg = legendgroup_fn(key) if legendgroup_fn else str(key)
        trace = go.Scattergeo(
            lat=lats, lon=lons,
            mode="markers",
            marker=dict(size=sz, color=color, line=dict(width=0.5, color="black")),
            name=name_fn(key),
            legendgroup=lg,
            hoverinfo="text",
            text=texts,
            showlegend=False,  # Lines already in legend
        )
        traces.append(trace)
    return traces


def _ic_traces(interconnections, network):
    """Build dashed traces for inter-regional interconnections."""
    traces = []
    for ic in interconnections:
        # Try to find approximate coordinates from route substations
        from_sub = ic.get("route", {}).get("from_substation_ja", "")
        to_sub = ic.get("route", {}).get("to_substation_ja", "")

        # Search for substations with matching names
        from_coord = None
        to_coord = None
        for sub in network.substations:
            if from_sub and from_sub in sub.name:
                from_coord = (sub.latitude, sub.longitude)
            if to_sub and to_sub in sub.name:
                to_coord = (sub.latitude, sub.longitude)

        # If we have FC converter locations, use those
        if from_coord is None and "frequency_converters" in ic:
            fcs = ic["frequency_converters"]
            if fcs:
                loc = fcs[0].get("location", {})
                if loc:
                    from_coord = (loc["latitude"], loc["longitude"])
        if to_coord is None and "frequency_converters" in ic:
            fcs = ic["frequency_converters"]
            if len(fcs) > 1:
                loc = fcs[-1].get("location", {})
                if loc:
                    to_coord = (loc["latitude"], loc["longitude"])

        if from_coord and to_coord:
            ic_type = ic.get("type", "AC")
            dash = "dash" if ic_type == "HVDC" else ("dot" if ic_type == "FC" else "solid")
            traces.append(go.Scattergeo(
                lat=[from_coord[0], to_coord[0]],
                lon=[from_coord[1], to_coord[1]],
                mode="lines+text",
                line=dict(color="#333333", width=3, dash=dash),
                name=f"連系線: {ic['name_ja']} ({ic['capacity_mw']}MW)",
                legendgroup="interconnection",
                hoverinfo="text",
                text=[
                    f"<b>{ic['name_ja']}</b><br>"
                    f"容量: {ic['capacity_mw']} MW<br>"
                    f"電圧: {ic['voltage_kv']} kV<br>"
                    f"種別: {ic_type}",
                    None
                ],
                showlegend=True,
            ))
    return traces


# ── Layout template ────────────────────────────────────────────────

def _make_layout(title):
    return go.Layout(
        title=dict(text=title, font=dict(size=20)),
        geo=dict(
            scope="asia",
            resolution=50,
            projection_type="mercator",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(204, 224, 244)",
            showcountries=True,
            countrycolor="rgb(180, 180, 180)",
            showcoastlines=True,
            coastlinecolor="rgb(150, 150, 150)",
            showlakes=True,
            lakecolor="rgb(204, 224, 244)",
            lonaxis=dict(range=[127, 147]),
            lataxis=dict(range=[26, 46]),
            center=dict(lat=36, lon=137),
            # Fill the full container
            domain=dict(x=[0, 1], y=[0, 1]),
        ),
        legend=dict(
            yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        autosize=True,
    )


# ── Build the 3 maps ──────────────────────────────────────────────

def build_n1_map(network, interconnections):
    """Map 1: N-1 contingency status coloring."""
    traces = []

    # Lines colored by capacity_status
    def group_fn(line): return line.capacity_status.value
    def color_fn(key): return N1_COLORS.get(key, "#aaaaaa")
    def width_fn(key):
        v = {"zero_capacity_n1_ineligible": 2, "zero_capacity_n1_eligible": 2,
             "available_capacity": 1.5, "unknown": 0.8}
        return v.get(key, 1)
    def name_fn(key): return N1_LABELS.get(key, key)

    traces += _line_traces_by_group(network, group_fn, color_fn, width_fn, name_fn)

    # Substations
    def sub_group(sub): return "substation"
    def sub_color(key): return "#333333"
    def sub_size(key): return 4
    def sub_name(key): return "変電所"
    traces += _sub_traces_by_group(network, sub_group, sub_color, sub_size, sub_name)

    # Interconnections
    traces += _ic_traces(interconnections, network)

    fig = go.Figure(data=traces, layout=_make_layout(
        "日本電力系統 — N-1空容量ステータス"
    ))
    return fig


def build_voltage_map(network, interconnections):
    """Map 2: Voltage class coloring."""
    traces = []

    def group_fn(line): return get_voltage_bracket(line.voltage_kv)
    def color_fn(key): return VOLTAGE_COLORS.get(key, "#cccccc")
    def width_fn(key): return VOLTAGE_WIDTHS.get(key, 0.5)
    def name_fn(key): return f"{key} kV" if key > 0 else "不明"

    traces += _line_traces_by_group(network, group_fn, color_fn, width_fn, name_fn)

    # Substations colored by voltage
    def sub_group(sub): return get_voltage_bracket(sub.voltage_kv)
    def sub_color(key): return VOLTAGE_COLORS.get(key, "#cccccc")
    def sub_size(key):
        if key >= 500: return 8
        if key >= 275: return 6
        if key >= 154: return 5
        return 3
    def sub_name(key): return f"{key} kV" if key > 0 else "不明"
    traces += _sub_traces_by_group(network, sub_group, sub_color, sub_size, sub_name)

    # Interconnections
    traces += _ic_traces(interconnections, network)

    fig = go.Figure(data=traces, layout=_make_layout(
        "日本電力系統 — 電圧階級別"
    ))
    return fig


def build_area_map(network, interconnections):
    """Map 3: Regional area coloring."""
    traces = []

    def group_fn(line): return line.region
    def color_fn(key): return AREA_COLORS.get(key, "#999999")
    def width_fn(key): return 1.5
    def name_fn(key):
        return REGIONS.get(key, {}).get("name_ja", key)

    traces += _line_traces_by_group(network, group_fn, color_fn, width_fn, name_fn)

    # Substations colored by region
    def sub_group(sub): return sub.region
    def sub_color(key): return AREA_COLORS.get(key, "#999999")
    def sub_size(key): return 4
    def sub_name(key): return REGIONS.get(key, {}).get("name_ja", key)
    traces += _sub_traces_by_group(network, sub_group, sub_color, sub_size, sub_name)

    # Interconnections
    traces += _ic_traces(interconnections, network)

    fig = go.Figure(data=traces, layout=_make_layout(
        "日本電力系統 — エリア別"
    ))
    return fig


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("Loading all regional KML files...")
    national = load_national_network()
    interconnections = load_interconnections()
    print(f"Loaded {len(interconnections)} interconnections")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nGenerating interactive maps...")

    # Common write_html config: full-width responsive
    html_cfg = dict(
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True},
        default_width="100%",
        default_height="100vh",
    )

    # 1. N-1 map
    fig_n1 = build_n1_map(national, interconnections)
    path_n1 = os.path.join(OUTPUT_DIR, "interactive_n1.html")
    fig_n1.write_html(path_n1, **html_cfg)
    print(f"  [1/3] N-1 map → {path_n1}")

    # 2. Voltage map
    fig_v = build_voltage_map(national, interconnections)
    path_v = os.path.join(OUTPUT_DIR, "interactive_voltage.html")
    fig_v.write_html(path_v, **html_cfg)
    print(f"  [2/3] Voltage map → {path_v}")

    # 3. Area map
    fig_a = build_area_map(national, interconnections)
    path_a = os.path.join(OUTPUT_DIR, "interactive_area.html")
    fig_a.write_html(path_a, **html_cfg)
    print(f"  [3/3] Area map → {path_a}")

    print("\nDone! Open the HTML files in a browser to zoom and explore.")


if __name__ == "__main__":
    main()
