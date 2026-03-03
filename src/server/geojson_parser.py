"""Convert OSM GeoJSON features to GridNetwork for power flow analysis.

Transforms GeoJSON Point features (substations) and LineString features
(transmission lines) into a GridNetwork model that can be fed directly to
PandapowerBuilder.
"""

from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.model.grid_network import GridNetwork
from src.model.substation import Substation, VoltageClass
from src.model.transmission_line import TransmissionLine
from src.utils.geo_utils import (
    find_nearest_point,
    haversine_distance,
    polyline_length,
)

CONFIG_PATH = "config/regions.yaml"

# Maximum distance (km) for matching line endpoints to substations
_MAX_ENDPOINT_DISTANCE_KM = 50.0


def _load_frequency(region: str) -> int:
    """Load frequency_hz for a region from config."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("regions", {}).get(region, {}).get("frequency_hz", 50)
    except Exception:
        return 50


def _parse_voltage_kv(feature: dict) -> float:
    """Extract voltage in kV from a GeoJSON feature.

    The _voltage_kv field is pre-computed by geojson_loader.
    Falls back to parsing the raw 'voltage' property.
    """
    props = feature.get("properties", {})
    v = props.get("_voltage_kv")
    if v is not None and v > 0:
        return float(v)

    raw = props.get("voltage")
    if raw is None:
        return 0.0
    s = str(raw).strip().replace(",", "")
    if ";" in s:
        s = s.split(";")[0].strip()
    try:
        v = float(s)
        return round(v / 1000, 1) if v > 1000 else round(v, 1) if v > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0


def _centroid(coords: list) -> Tuple[float, float]:
    """Compute centroid (lon, lat) from a Polygon coordinate ring."""
    ring = coords[0] if coords and isinstance(coords[0][0], list) else coords
    if not ring:
        return (0.0, 0.0)
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def parse_substations(fc: dict, region: str) -> List[Substation]:
    """Convert a GeoJSON FeatureCollection of substations to Substation objects.

    Handles Point, Polygon, and MultiPolygon geometries. For polygons,
    the centroid is used as the substation location.

    Args:
        fc: GeoJSON FeatureCollection.
        region: Region identifier.

    Returns:
        List of Substation instances.
    """
    substations = []
    features = fc.get("features", [])

    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if geom_type == "Point":
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
        elif geom_type in ("Polygon", "MultiPolygon"):
            poly_coords = coords[0] if geom_type == "MultiPolygon" else coords
            lon, lat = _centroid(poly_coords)
        else:
            continue

        if lon == 0.0 and lat == 0.0:
            continue
        props = feat.get("properties", {})
        voltage_kv = _parse_voltage_kv(feat)
        name = props.get("name") or props.get("name:ja") or f"sub_{i}"
        osm_id = props.get("id", i)

        sub = Substation(
            id=f"{region}_osm_sub_{osm_id}",
            name=name,
            region=region,
            latitude=lat,
            longitude=lon,
            voltage_kv=voltage_kv,
        )
        substations.append(sub)

    return substations


def parse_lines(
    fc: dict,
    region: str,
    substations: List[Substation],
) -> List[TransmissionLine]:
    """Convert a GeoJSON FeatureCollection of lines to TransmissionLine objects.

    Matches line endpoints to nearest substations using spatial proximity.

    Args:
        fc: GeoJSON FeatureCollection with LineString features.
        region: Region identifier.
        substations: Pre-parsed substations for endpoint matching.

    Returns:
        List of TransmissionLine instances.
    """
    if not substations:
        return []

    # Build candidate list for nearest-point matching
    candidates: List[Tuple[str, float, float]] = [
        (s.id, s.latitude, s.longitude) for s in substations
    ]

    lines = []
    features = fc.get("features", [])

    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue

        props = feat.get("properties", {})
        voltage_kv = _parse_voltage_kv(feat)
        osm_id = props.get("id", i)
        name = props.get("name") or props.get("ref") or f"line_{osm_id}"

        # GeoJSON coordinates are [lon, lat]
        start_lon, start_lat = float(coords[0][0]), float(coords[0][1])
        end_lon, end_lat = float(coords[-1][0]), float(coords[-1][1])

        # Match endpoints to nearest substations
        from_id, from_dist = find_nearest_point(
            start_lat, start_lon, candidates, _MAX_ENDPOINT_DISTANCE_KM
        )
        to_id, to_dist = find_nearest_point(
            end_lat, end_lon, candidates, _MAX_ENDPOINT_DISTANCE_KM
        )

        if not from_id or not to_id:
            continue
        if from_id == to_id:
            continue

        # Compute line length from coordinates
        # GeoJSON is [lon, lat], but polyline_length expects (lat, lon)
        lat_lon_coords = [(float(c[1]), float(c[0])) for c in coords]
        length_km = polyline_length(lat_lon_coords)
        if length_km <= 0:
            length_km = haversine_distance(start_lat, start_lon, end_lat, end_lon)
        if length_km <= 0:
            length_km = 1.0  # minimum 1 km fallback

        line = TransmissionLine(
            id=f"{region}_osm_line_{osm_id}",
            name=name,
            from_substation_id=from_id,
            to_substation_id=to_id,
            voltage_kv=voltage_kv,
            length_km=round(length_km, 2),
            region=region,
            coordinates=lat_lon_coords,
        )
        lines.append(line)

    return lines


def build_grid_network(
    sub_fc: dict,
    line_fc: dict,
    region: str,
) -> GridNetwork:
    """Build a GridNetwork from GeoJSON FeatureCollections.

    Args:
        sub_fc: Substations GeoJSON FeatureCollection.
        line_fc: Lines GeoJSON FeatureCollection.
        region: Region identifier.

    Returns:
        A populated GridNetwork.
    """
    freq = _load_frequency(region)
    substations = parse_substations(sub_fc, region)
    lines = parse_lines(line_fc, region, substations)

    network = GridNetwork(region=region, frequency_hz=freq)
    for s in substations:
        try:
            network.add_substation(s)
        except ValueError:
            pass  # skip duplicates
    for l in lines:
        try:
            network.add_transmission_line(l)
        except ValueError:
            pass  # skip duplicates

    return network
