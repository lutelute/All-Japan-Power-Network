"""Synthetic load estimation and generation scaling for power flow analysis.

Distributes regional peak demand across buses proportional to their voltage
class weights, and scales generator output to match total demand plus
reserve margin.  The external grid (slack bus) absorbs residual mismatch.

Usage::

    from src.powerflow.load_estimator import estimate_loads, scale_generation

    estimate_loads(net, region="shikoku", demand_config=cfg)
    scale_generation(net, target_mw=total_demand * 1.05)
"""

import math
from typing import Any, Dict, Optional

import pandapower as pp
import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DEMAND_CONFIG_PATH = "config/regional_demand.yaml"


def load_demand_config(
    config_path: str = DEFAULT_DEMAND_CONFIG_PATH,
) -> Dict[str, Any]:
    """Load regional demand configuration from YAML.

    Args:
        config_path: Path to ``regional_demand.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def estimate_loads(
    net: Any,
    region: str,
    demand_config: Optional[Dict[str, Any]] = None,
    config_path: str = DEFAULT_DEMAND_CONFIG_PATH,
) -> float:
    """Distribute synthetic loads across all buses in the network.

    Each bus receives a load proportional to its voltage-class weight.
    The total load equals the regional peak demand multiplied by the
    configured load factor.

    For national (multi-region) models, the ``zone`` column on each bus
    is used to determine its region and allocate demand accordingly.

    Args:
        net: pandapower network (modified in place).
        region: Region identifier (e.g. ``"shikoku"``).
        demand_config: Pre-loaded config dict.  If ``None``, loaded from
            *config_path*.
        config_path: Fallback path for loading config.

    Returns:
        Total active power (MW) allocated across all buses.
    """
    if demand_config is None:
        demand_config = load_demand_config(config_path)

    peak_demands = demand_config["regional_peak_demand_mw"]
    load_factor = demand_config.get("load_factor", 0.85)
    power_factor = demand_config.get("power_factor", 0.95)
    voltage_weights = demand_config.get("voltage_weights", {})

    # Q/P ratio from power factor
    tan_phi = math.tan(math.acos(power_factor))

    if region == "national":
        return _estimate_loads_national(
            net, peak_demands, load_factor, tan_phi, voltage_weights,
        )

    # Regional model: single region
    peak_mw = peak_demands.get(region)
    if peak_mw is None:
        logger.warning(
            "No peak demand data for region '%s'; skipping load allocation",
            region,
        )
        return 0.0

    target_mw = peak_mw * load_factor
    total_allocated = _allocate_bus_loads(
        net, target_mw, tan_phi, voltage_weights,
    )

    logger.info(
        "Allocated %.1f MW (%.1f MVAr) across %d buses for region '%s'",
        total_allocated,
        total_allocated * tan_phi,
        len(net.bus),
        region,
    )
    return total_allocated


def _estimate_loads_national(
    net: Any,
    peak_demands: Dict[str, float],
    load_factor: float,
    tan_phi: float,
    voltage_weights: Dict,
) -> float:
    """Allocate loads for a national (multi-region) network.

    Uses the ``zone`` column on each bus to determine regional
    membership and applies per-region demand targets.
    """
    total_allocated = 0.0

    if "zone" not in net.bus.columns:
        logger.warning(
            "National network has no 'zone' column; "
            "distributing load uniformly"
        )
        total_peak = sum(peak_demands.values())
        target_mw = total_peak * load_factor
        total_allocated = _allocate_bus_loads(
            net, target_mw, tan_phi, voltage_weights,
        )
        return total_allocated

    # Group buses by zone
    for zone, group in net.bus.groupby("zone"):
        peak_mw = peak_demands.get(zone, 0)
        if peak_mw <= 0:
            continue

        target_mw = peak_mw * load_factor
        bus_indices = group.index.tolist()

        allocated = _allocate_bus_loads_subset(
            net, bus_indices, target_mw, tan_phi, voltage_weights,
        )
        total_allocated += allocated

        logger.info(
            "National model: allocated %.1f MW to zone '%s' (%d buses)",
            allocated, zone, len(bus_indices),
        )

    return total_allocated


def _allocate_bus_loads(
    net: Any,
    target_mw: float,
    tan_phi: float,
    voltage_weights: Dict,
) -> float:
    """Allocate *target_mw* across **all** buses in *net*."""
    bus_indices = net.bus.index.tolist()
    return _allocate_bus_loads_subset(
        net, bus_indices, target_mw, tan_phi, voltage_weights,
    )


def _allocate_bus_loads_subset(
    net: Any,
    bus_indices: list,
    target_mw: float,
    tan_phi: float,
    voltage_weights: Dict,
) -> float:
    """Allocate *target_mw* across a subset of buses.

    The allocation is proportional to each bus's voltage-class weight.
    """
    if not bus_indices:
        return 0.0

    # Compute per-bus weights
    weights = []
    for idx in bus_indices:
        vn_kv = net.bus.at[idx, "vn_kv"]
        # Find the closest matching voltage weight
        w = _voltage_weight(vn_kv, voltage_weights)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight <= 0:
        # Uniform distribution fallback
        total_weight = len(bus_indices)
        weights = [1.0] * len(bus_indices)

    total_allocated = 0.0
    for idx, w in zip(bus_indices, weights):
        p_mw = target_mw * (w / total_weight)
        q_mvar = p_mw * tan_phi

        pp.create_load(
            net,
            bus=idx,
            p_mw=p_mw,
            q_mvar=q_mvar,
            name=f"load_bus_{idx}",
        )
        total_allocated += p_mw

    return total_allocated


def _voltage_weight(vn_kv: float, voltage_weights: Dict) -> float:
    """Look up the voltage weight for a given nominal voltage.

    Falls back to the closest available key if no exact match exists.
    """
    # Try exact integer match first
    key = int(round(vn_kv))
    if key in voltage_weights:
        return voltage_weights[key]

    # Find the closest key
    available = [k for k in voltage_weights if isinstance(k, (int, float)) and k > 0]
    if not available:
        return 0.5  # default

    closest = min(available, key=lambda k: abs(k - vn_kv))
    return voltage_weights[closest]


def scale_generation(net: Any, target_mw: float) -> float:
    """Scale all generator outputs proportionally to meet *target_mw*.

    The external grid (``ext_grid``) absorbs the residual mismatch
    between generation and demand after scaling.

    Args:
        net: pandapower network (modified in place).
        target_mw: Total generation target in MW.

    Returns:
        Actual total generation set (MW) after scaling.
    """
    if len(net.gen) == 0:
        logger.info("No generators to scale; ext_grid will supply all load")
        return 0.0

    total_capacity = net.gen["max_p_mw"].sum()
    if total_capacity <= 0:
        logger.warning("Total generator capacity is zero; skipping scaling")
        return 0.0

    # Scale factor: target / total_capacity, capped at 1.0
    scale = min(target_mw / total_capacity, 1.0)

    net.gen["p_mw"] = net.gen["max_p_mw"] * scale
    # Ensure in_service
    net.gen["in_service"] = True

    actual_total = net.gen["p_mw"].sum()
    logger.info(
        "Scaled %d generators: total=%.1f MW (scale=%.3f, capacity=%.1f MW)",
        len(net.gen),
        actual_total,
        scale,
        total_capacity,
    )
    return actual_total
