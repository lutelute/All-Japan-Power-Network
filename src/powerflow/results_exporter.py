"""Export power flow results to JSON files.

Produces per-region result files and a cross-region summary in
``output/powerflow/``.

Usage::

    from src.powerflow.results_exporter import export_results, export_summary

    export_results(pf_result, net, region="shikoku")
    export_summary(all_results)
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.powerflow.powerflow_runner import PowerFlowResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "output/powerflow"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def export_results(
    pf_result: PowerFlowResult,
    net: Any,
    region: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Export power flow results for a single region.

    Args:
        pf_result: Power flow result from ``run_powerflow()``.
        net: The pandapower network (for supplementary data).
        region: Region identifier.
        output_dir: Output directory path.

    Returns:
        Path to the written JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    data: Dict[str, Any] = {
        "region": region,
        "converged": pf_result.converged,
        "mode": pf_result.mode,
        "summary": {
            "total_buses": len(net.bus),
            "total_lines": len(net.line),
            "total_generators": len(net.gen),
            "total_loads": len(net.load),
            "total_demand_mw": float(net.load["p_mw"].sum()) if len(net.load) > 0 else 0.0,
            "total_generation_mw": (
                float(net.gen["p_mw"].sum()) if len(net.gen) > 0 else 0.0
            ),
            "ext_grid_supply_mw": _ext_grid_supply(pf_result),
            "total_loss_mw": pf_result.total_loss_mw,
            "max_line_loading_pct": pf_result.max_line_loading_pct,
        },
        "warnings": pf_result.warnings,
    }

    # Top loaded lines
    data["top_loaded_lines"] = _top_loaded_lines(pf_result, net, n=10)

    # Generation by fuel type (if available)
    data["generation_by_fuel"] = _generation_by_fuel(net)

    # Interconnection flows (lines connecting different zones)
    data["interconnection_flows"] = _interconnection_flows(pf_result, net)

    output_path = os.path.join(output_dir, f"results_{region}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info("Exported power flow results: %s", output_path)
    return output_path


def export_summary(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Export a cross-region summary of power flow results.

    Args:
        results: Dict mapping region → per-region summary data.
        output_dir: Output directory path.

    Returns:
        Path to the written summary JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "regions_computed": len(results),
        "all_converged": all(
            r.get("converged", False) for r in results.values()
        ),
        "per_region": results,
        "totals": {
            "total_demand_mw": sum(
                r.get("summary", {}).get("total_demand_mw", 0)
                for r in results.values()
            ),
            "total_generation_mw": sum(
                r.get("summary", {}).get("total_generation_mw", 0)
                for r in results.values()
            ),
            "total_loss_mw": sum(
                r.get("summary", {}).get("total_loss_mw", 0)
                for r in results.values()
            ),
        },
    }

    output_path = os.path.join(output_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info("Exported power flow summary: %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ext_grid_supply(pf_result: PowerFlowResult) -> float:
    """Total active power supplied by external grids."""
    if pf_result.res_ext_grid is not None and "p_mw" in pf_result.res_ext_grid.columns:
        return float(pf_result.res_ext_grid["p_mw"].sum())
    return 0.0


def _top_loaded_lines(
    pf_result: PowerFlowResult,
    net: Any,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """Return the *n* most loaded lines."""
    if pf_result.res_line is None or pf_result.res_line.empty:
        return []

    res = pf_result.res_line.copy()
    if "loading_percent" not in res.columns:
        return []

    # Merge with line metadata
    res["name"] = net.line["name"].values if "name" in net.line.columns else ""
    res["from_bus"] = net.line["from_bus"].values
    res["to_bus"] = net.line["to_bus"].values

    top_lines = res.nlargest(n, "loading_percent")
    result = []
    for idx, row in top_lines.iterrows():
        entry = {
            "line_index": int(idx),
            "name": str(row.get("name", "")),
            "loading_percent": round(float(row["loading_percent"]), 2),
            "from_bus": int(row["from_bus"]),
            "to_bus": int(row["to_bus"]),
        }
        if "p_from_mw" in row:
            entry["p_from_mw"] = round(float(row["p_from_mw"]), 2)
        result.append(entry)

    return result


def _generation_by_fuel(net: Any) -> Dict[str, float]:
    """Aggregate generation by fuel type if generator names contain hints."""
    if len(net.gen) == 0:
        return {}

    # pandapower gen table may have a 'type' column or we infer from name
    fuel_totals: Dict[str, float] = {}

    for _, gen_row in net.gen.iterrows():
        name = str(gen_row.get("name", ""))
        p_mw = float(gen_row.get("p_mw", 0.0))
        fuel = "unknown"

        # Simple fuel-type heuristic from Japanese/English name keywords
        name_lower = name.lower()
        fuel_keywords = {
            "nuclear": ["nuclear", "原子力", "原発"],
            "coal": ["coal", "石炭", "火力"],
            "lng": ["lng", "ガス", "gas"],
            "hydro": ["hydro", "水力", "ダム"],
            "wind": ["wind", "風力"],
            "solar": ["solar", "太陽光", "メガソーラー"],
            "geothermal": ["geothermal", "地熱"],
            "biomass": ["biomass", "バイオマス"],
            "oil": ["oil", "石油", "重油"],
        }
        for fuel_type, keywords in fuel_keywords.items():
            if any(kw in name_lower for kw in keywords):
                fuel = fuel_type
                break

        fuel_totals[fuel] = fuel_totals.get(fuel, 0.0) + p_mw

    return {k: round(v, 2) for k, v in sorted(fuel_totals.items())}


def _interconnection_flows(
    pf_result: PowerFlowResult,
    net: Any,
) -> List[Dict[str, Any]]:
    """Identify lines connecting different zones and report power flow."""
    if pf_result.res_line is None or pf_result.res_line.empty:
        return []

    if "zone" not in net.bus.columns:
        return []

    flows = []
    for idx, line_row in net.line.iterrows():
        from_bus = line_row["from_bus"]
        to_bus = line_row["to_bus"]

        from_zone = net.bus.at[from_bus, "zone"] if from_bus in net.bus.index else None
        to_zone = net.bus.at[to_bus, "zone"] if to_bus in net.bus.index else None

        if from_zone and to_zone and from_zone != to_zone:
            res_row = pf_result.res_line.loc[idx] if idx in pf_result.res_line.index else None
            if res_row is not None:
                flow = {
                    "line_index": int(idx),
                    "from_zone": str(from_zone),
                    "to_zone": str(to_zone),
                    "name": str(line_row.get("name", "")),
                }
                if "p_from_mw" in res_row:
                    flow["p_from_mw"] = round(float(res_row["p_from_mw"]), 2)
                if "loading_percent" in res_row:
                    flow["loading_percent"] = round(float(res_row["loading_percent"]), 2)
                flows.append(flow)

    return flows
