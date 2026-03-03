"""Geographic utility functions for the Japan Grid Pipeline.

Provides Haversine distance calculations, coordinate transforms, and
Japan bounding box validation used throughout the pipeline for computing
transmission line lengths and validating coordinate data.
"""

import math
from typing import List, Tuple

# Earth's mean radius in kilometers (WGS-84)
EARTH_RADIUS_KM = 6371.0

# Japan bounding box for coordinate validation
# Source: config/regions.yaml japan_bounding_box
JAPAN_BBOX = {
    "lat_min": 24.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0,
}


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


def polyline_length(coordinates: List[Tuple[float, float]]) -> float:
    """Calculate the total length of a polyline defined by coordinate pairs.

    Used for computing transmission line lengths from KML LineString coordinates.

    Args:
        coordinates: List of (latitude, longitude) tuples in decimal degrees.

    Returns:
        Total length in kilometers. Returns 0.0 if fewer than 2 points.
    """
    if len(coordinates) < 2:
        return 0.0

    total = 0.0
    for i in range(len(coordinates) - 1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i + 1]
        total += haversine_distance(lat1, lon1, lat2, lon2)

    return total


def is_within_japan(lat: float, lon: float) -> bool:
    """Check if a coordinate falls within Japan's bounding box.

    Uses the national bounding box (lat: 24-46, lon: 122-154) as defined
    in config/regions.yaml. This is a coarse check — points over the ocean
    within these bounds will still pass.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        True if the coordinate is within Japan's bounding box.
    """
    return (
        JAPAN_BBOX["lat_min"] <= lat <= JAPAN_BBOX["lat_max"]
        and JAPAN_BBOX["lon_min"] <= lon <= JAPAN_BBOX["lon_max"]
    )


def is_within_region_bbox(
    lat: float,
    lon: float,
    bbox: dict,
) -> bool:
    """Check if a coordinate falls within a regional bounding box.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        bbox: Dictionary with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns:
        True if the coordinate is within the bounding box.
    """
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def find_nearest_point(
    target_lat: float,
    target_lon: float,
    candidates: List[Tuple[str, float, float]],
    max_distance_km: float = 5.0,
) -> Tuple[str, float]:
    """Find the nearest candidate point to a target coordinate.

    Used for matching generators to their nearest substation and for
    connecting transmission line endpoints to substation nodes.

    Args:
        target_lat: Latitude of the target point.
        target_lon: Longitude of the target point.
        candidates: List of (id, latitude, longitude) tuples.
        max_distance_km: Maximum allowed distance in km. If the nearest
            candidate exceeds this, returns empty string and the distance.

    Returns:
        Tuple of (nearest_id, distance_km). If no candidates exist or
        all exceed max_distance_km, nearest_id is an empty string.
    """
    if not candidates:
        return ("", float("inf"))

    best_id = ""
    best_distance = float("inf")

    for cid, clat, clon in candidates:
        dist = haversine_distance(target_lat, target_lon, clat, clon)
        if dist < best_distance:
            best_distance = dist
            best_id = cid

    if best_distance > max_distance_km:
        return ("", best_distance)

    return (best_id, best_distance)


def dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
    """Convert degrees-minutes-seconds to decimal degrees.

    Some Japanese geographic data sources use DMS format.

    Args:
        degrees: Degree component.
        minutes: Minute component.
        seconds: Second component.

    Returns:
        Coordinate in decimal degrees.
    """
    sign = -1.0 if degrees < 0 else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def decimal_to_dms(decimal_degrees: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to degrees-minutes-seconds.

    Args:
        decimal_degrees: Coordinate in decimal degrees.

    Returns:
        Tuple of (degrees, minutes, seconds).
    """
    sign = -1 if decimal_degrees < 0 else 1
    dd = abs(decimal_degrees)
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    seconds = (dd - degrees - minutes / 60.0) * 3600.0

    return (sign * degrees, minutes, seconds)
