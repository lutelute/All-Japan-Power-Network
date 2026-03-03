"""Batch execution engine for AC power flow methods.

Executes all ~20 AC power flow methods against one or more regional networks,
showing progress via tqdm progress bars, with optional parallel execution
using ``ProcessPoolExecutor``.

Functions:

* ``run_all_methods(net, region)`` — Run all methods on one network.
* ``run_batch(regions, parallel, max_workers, output_dir)`` — Run all
  methods across multiple regions with progress visibility.
* ``main()`` — CLI entry point (``python -m src.ac_powerflow.batch_runner``).

Usage::

    # CLI
    PYTHONPATH=. python -m src.ac_powerflow.batch_runner --all-regions
    PYTHONPATH=. python -m src.ac_powerflow.batch_runner --region shikoku
    PYTHONPATH=. python -m src.ac_powerflow.batch_runner --all-regions --parallel

    # Programmatic
    from src.ac_powerflow.batch_runner import run_batch
    results = run_batch(["shikoku", "hokkaido"])
"""

import argparse
import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Default paths (relative to project root).
DEFAULT_CONFIG_PATH = "config/regions.yaml"
DEFAULT_OUTPUT_DIR = "output/ac_powerflow"


# ------------------------------------------------------------------
# Region / network loading
# ------------------------------------------------------------------


def _load_active_regions(config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Load active region names from the YAML configuration.

    Args:
        config_path: Path to ``regions.yaml``.

    Returns:
        Sorted list of region identifiers whose status is ``"active"``.
    """
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        return []

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    regions = config.get("regions", {})
    active = sorted(
        name
        for name, cfg in regions.items()
        if cfg.get("status") == "active"
    )

    logger.info("Loaded %d active regions from %s", len(active), config_path)
    return active


def _build_network(region: str, config_path: str = DEFAULT_CONFIG_PATH) -> Any:
    """Build a pandapower network for a region.

    Follows the same pattern as ``run_powerflow_animation_365.py``'s
    ``compute_region()`` — fully self-contained so it can be used inside
    a ``ProcessPoolExecutor`` worker.

    Args:
        region: Region identifier (e.g. ``"shikoku"``).
        config_path: Path to ``regions.yaml``.

    Returns:
        pandapower network ready for power flow analysis.

    Raises:
        RuntimeError: If the network cannot be built.
    """
    import pandapower as pp
    from src.converter.pandapower_builder import PandapowerBuilder
    from src.server.geojson_loader import load_all
    from src.server.geojson_parser import build_grid_network

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    region_cfg = config["regions"][region]

    # Load from OSM GeoJSON
    cache = load_all()
    if region not in cache:
        raise RuntimeError(f"No GeoJSON data for region '{region}'")
    sub_fc = cache[region]["substations"]
    line_fc = cache[region]["lines"]
    freq = region_cfg.get("frequency_hz", 0)
    network = build_grid_network(sub_fc, line_fc, region=region, frequency_hz=freq)

    # Build pandapower network
    builder = PandapowerBuilder()
    net = builder.build(network).net

    # Fix zero-voltage buses
    zero_mask = net.bus["vn_kv"] == 0
    if zero_mask.any():
        bus_voltages: Dict[str, float] = {}
        for line in network.lines:
            if line.voltage_kv <= 0:
                continue
            for sid in (line.from_substation_id, line.to_substation_id):
                if sid not in bus_voltages or line.voltage_kv > bus_voltages[sid]:
                    bus_voltages[sid] = line.voltage_kv
        sub_to_idx = {s.id: i for i, s in enumerate(network.substations)}
        for sid, v in bus_voltages.items():
            idx = sub_to_idx.get(sid)
            if idx is not None and idx in net.bus.index and net.bus.at[idx, "vn_kv"] == 0:
                net.bus.at[idx, "vn_kv"] = v
        still_zero = net.bus["vn_kv"] == 0
        if still_zero.any():
            non_zero = net.bus.loc[~still_zero, "vn_kv"]
            if len(non_zero) > 0:
                net.bus.loc[still_zero, "vn_kv"] = float(non_zero.median())

    return net


def _check_topology_for_ac(net: Any) -> List[str]:
    """Validate and fix network topology before AC power flow.

    Follows the ``_check_topology()`` pattern from
    ``src/powerflow/powerflow_runner.py``.

    Args:
        net: pandapower network (modified in place).

    Returns:
        List of warning messages.
    """
    import pandapower as pp
    import pandapower.topology as top
    import networkx as nx

    warnings: List[str] = []

    try:
        mg = top.create_nxgraph(net, respect_switches=False)
        components = list(nx.connected_components(mg))

        if len(components) > 1:
            sizes = sorted([len(c) for c in components], reverse=True)
            msg = (
                f"Network has {len(components)} connected components "
                f"(sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''})"
            )
            warnings.append(msg)
            logger.warning(msg)

            # Deactivate isolated buses (not in the largest component)
            largest = max(components, key=len)
            isolated_buses = set()
            for comp in components:
                if comp != largest:
                    isolated_buses.update(comp)

            if isolated_buses:
                for bus_idx in isolated_buses:
                    if bus_idx in net.bus.index:
                        net.bus.at[bus_idx, "in_service"] = False

                # Deactivate loads/gens/lines connected to isolated buses
                for table_name in ("load", "gen", "line"):
                    table = getattr(net, table_name, None)
                    if table is None or table.empty:
                        continue
                    if table_name == "line":
                        mask = (
                            table["from_bus"].isin(isolated_buses)
                            | table["to_bus"].isin(isolated_buses)
                        )
                    else:
                        mask = table["bus"].isin(isolated_buses)
                    table.loc[mask, "in_service"] = False

                # Deactivate ext_grids on isolated buses
                if not net.ext_grid.empty:
                    mask = net.ext_grid["bus"].isin(isolated_buses)
                    net.ext_grid.loc[mask, "in_service"] = False

                msg = (
                    f"Deactivated {len(isolated_buses)} isolated buses "
                    f"and their connected elements"
                )
                warnings.append(msg)
                logger.warning(msg)

                # Ensure at least one ext_grid is in service
                if net.ext_grid["in_service"].sum() == 0 and len(net.ext_grid) > 0:
                    for i, row in net.ext_grid.iterrows():
                        if row["bus"] in largest:
                            net.ext_grid.at[i, "in_service"] = True
                            logger.info(
                                "Re-enabled ext_grid %d on bus %d",
                                i, row["bus"],
                            )
                            break
                    else:
                        bus_idx = next(iter(largest))
                        pp.create_ext_grid(net, bus=bus_idx, vm_pu=1.0, name="slack_recovery")
                        warnings.append(
                            "Created recovery ext_grid on bus in largest component"
                        )

    except Exception as exc:
        msg = f"Topology check failed: {exc}"
        warnings.append(msg)
        logger.warning(msg)

    return warnings


# ------------------------------------------------------------------
# Core execution
# ------------------------------------------------------------------


def run_all_methods(
    net: Any,
    region: str,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Run all ~20 AC power flow methods on one network.

    For pandapower wrappers, passes the network directly.  For custom
    PYPOWER-level solvers, first extracts internal matrices via
    ``prepare_network()``.

    Args:
        net: pandapower network (will be deep-copied for each method).
        region: Region identifier for result tagging.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance.

    Returns:
        List of result dicts, each with keys ``method_id``,
        ``method_name``, ``category``, ``region``, ``result``.
    """
    from src.ac_powerflow.methods import get_all_methods
    from src.ac_powerflow.network_prep import prepare_network
    from src.ac_powerflow.solver_interface import ACMethodResult

    methods = get_all_methods()
    results: List[Dict[str, Any]] = []

    # Prepare network data for custom solvers (done once per region).
    network_data = None
    try:
        network_data = prepare_network(copy.deepcopy(net))
    except Exception as exc:
        logger.warning(
            "Failed to prepare network for custom solvers on %s: %s",
            region, exc,
        )

    try:
        from tqdm import tqdm
        method_iter = tqdm(
            methods,
            desc=f"  {region} methods",
            leave=False,
            unit="method",
        )
    except ImportError:
        method_iter = methods

    for method in method_iter:
        result_record = _run_single_method(
            method=method,
            net=net,
            region=region,
            network_data=network_data,
            max_iteration=max_iteration,
            tolerance=tolerance,
        )
        results.append(result_record)

    return results


def _run_single_method(
    method: Any,
    net: Any,
    region: str,
    network_data: Any,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Execute a single AC power flow method safely.

    Args:
        method: ``MethodDescriptor`` instance.
        net: pandapower network.
        region: Region identifier.
        network_data: ``NetworkData`` for custom solvers (or ``None``).
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance.

    Returns:
        Result dict with keys ``method_id``, ``method_name``,
        ``category``, ``region``, ``result``.
    """
    from src.ac_powerflow.solver_interface import ACMethodResult

    try:
        if method.category == "pandapower":
            # Pandapower wrappers operate on the net object directly.
            net_copy = copy.deepcopy(net)
            result = method.solver_fn(
                net_copy,
                max_iteration=max_iteration,
                tolerance=tolerance,
            )
        else:
            # Custom solvers operate on PYPOWER matrices.
            if network_data is None:
                result = ACMethodResult(
                    converged=False,
                    failure_reason="Network preparation failed — cannot run custom solver",
                )
            else:
                import numpy as np
                result = method.solver_fn(
                    network_data.Ybus,
                    network_data.Sbus,
                    np.copy(network_data.V0),
                    np.array(network_data.ref),
                    np.array(network_data.pv),
                    np.array(network_data.pq),
                    max_iter=max_iteration,
                    tol=tolerance,
                )
    except Exception as exc:
        result = ACMethodResult(
            converged=False,
            failure_reason=f"Unhandled {type(exc).__name__}: {exc}",
        )

    logger.debug(
        "%s on %s: converged=%s, iter=%d, time=%.4fs",
        method.id, region,
        result.converged, result.iterations, result.elapsed_sec,
    )

    return {
        "method_id": method.id,
        "method_name": method.name,
        "category": method.category,
        "region": region,
        "result": result,
    }


def _process_region(
    region: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Build network and run all methods for one region.

    Designed for ``ProcessPoolExecutor`` — fully self-contained with
    all imports inside the function body for spawn-safe execution.

    Args:
        region: Region identifier.
        config_path: Path to ``regions.yaml``.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance.

    Returns:
        List of result dicts for the region (pickle-friendly).
    """
    logger.info("Processing region: %s", region)
    t0 = time.perf_counter()

    try:
        net = _build_network(region, config_path=config_path)
    except Exception as exc:
        logger.error("Failed to build network for %s: %s", region, exc)
        return []

    # Topology check
    _check_topology_for_ac(net)

    # Check for empty network
    active_buses = net.bus[net.bus["in_service"]].shape[0]
    if active_buses == 0:
        logger.warning("Region %s has no active buses — skipping", region)
        return []

    results = run_all_methods(
        net, region,
        max_iteration=max_iteration,
        tolerance=tolerance,
    )

    elapsed = time.perf_counter() - t0
    n_converged = sum(1 for r in results if r["result"].converged)
    logger.info(
        "Region %s complete: %d/%d converged in %.1fs",
        region, n_converged, len(results), elapsed,
    )

    return results


def run_batch(
    regions: List[str],
    parallel: bool = False,
    max_workers: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Run all AC power flow methods across multiple regions.

    Supports both sequential and parallel execution modes.  In parallel
    mode, each worker processes one region using ``ProcessPoolExecutor``.

    Args:
        regions: List of region identifiers to process.
        parallel: If ``True``, use ``ProcessPoolExecutor`` for parallel
            execution across regions.
        max_workers: Maximum number of parallel workers.  If ``None``,
            defaults to the number of regions (capped by available CPUs).
        output_dir: Directory for output files.
        config_path: Path to ``regions.yaml``.
        max_iteration: Maximum solver iterations per method.
        tolerance: Convergence tolerance.

    Returns:
        Combined list of result dicts across all regions.
    """
    from src.ac_powerflow.convergence_report import (
        generate_report,
        print_summary,
        save_report,
    )

    logger.info(
        "Starting batch run: %d regions, parallel=%s, max_workers=%s",
        len(regions), parallel, max_workers,
    )

    all_results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    if parallel and len(regions) > 1:
        all_results = _run_parallel(
            regions,
            config_path=config_path,
            max_workers=max_workers,
            max_iteration=max_iteration,
            tolerance=tolerance,
        )
    else:
        all_results = _run_sequential(
            regions,
            config_path=config_path,
            max_iteration=max_iteration,
            tolerance=tolerance,
        )

    elapsed = time.perf_counter() - t0

    # Generate and save convergence report
    if all_results:
        report = generate_report(all_results)
        report_path = os.path.join(output_dir, "convergence_report.json")
        save_report(report, report_path)
        print_summary(report)
    else:
        logger.warning("No results to report — all regions may have failed.")

    logger.info(
        "Batch run complete: %d results across %d regions in %.1fs",
        len(all_results), len(regions), elapsed,
    )

    return all_results


def _run_sequential(
    regions: List[str],
    config_path: str = DEFAULT_CONFIG_PATH,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Run regions sequentially with progress bars.

    Args:
        regions: Region identifiers.
        config_path: Path to ``regions.yaml``.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance.

    Returns:
        Combined results list.
    """
    all_results: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm
        region_iter = tqdm(regions, desc="Regions", unit="region")
    except ImportError:
        region_iter = regions

    for region in region_iter:
        results = _process_region(
            region,
            config_path=config_path,
            max_iteration=max_iteration,
            tolerance=tolerance,
        )
        all_results.extend(results)

    return all_results


def _run_parallel(
    regions: List[str],
    config_path: str = DEFAULT_CONFIG_PATH,
    max_workers: Optional[int] = None,
    max_iteration: int = 20,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """Run regions in parallel using ProcessPoolExecutor.

    Each worker processes one region independently, following the
    spawn-safe pattern from ``run_powerflow_animation_365.py``.

    Args:
        regions: Region identifiers.
        config_path: Path to ``regions.yaml``.
        max_workers: Maximum worker processes.
        max_iteration: Maximum solver iterations.
        tolerance: Convergence tolerance.

    Returns:
        Combined results list.
    """
    all_results: List[Dict[str, Any]] = []

    effective_workers = max_workers or min(len(regions), os.cpu_count() or 4)

    logger.info(
        "Starting parallel execution: %d regions, %d workers",
        len(regions), effective_workers,
    )

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(regions), desc="Regions (parallel)", unit="region")
    except ImportError:
        pbar = None

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for region in regions:
            future = executor.submit(
                _process_region,
                region,
                config_path=config_path,
                max_iteration=max_iteration,
                tolerance=tolerance,
            )
            futures[future] = region

        for future in as_completed(futures):
            region = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                n_conv = sum(1 for r in results if r["result"].converged)
                logger.info(
                    "Parallel: %s done — %d/%d converged",
                    region, n_conv, len(results),
                )
            except Exception as exc:
                logger.error(
                    "Parallel: %s failed — %s: %s",
                    region, type(exc).__name__, exc,
                )

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return all_results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the AC power flow batch runner.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="ac-powerflow-batch",
        description=(
            "AC Power Flow Batch Runner — Execute all ~20 AC power flow "
            "methods across regional networks with convergence analysis."
        ),
        epilog=(
            "Examples:\n"
            "  PYTHONPATH=. python -m src.ac_powerflow.batch_runner --all-regions\n"
            "  PYTHONPATH=. python -m src.ac_powerflow.batch_runner --region shikoku\n"
            "  PYTHONPATH=. python -m src.ac_powerflow.batch_runner --all-regions --parallel\n"
            "  PYTHONPATH=. python -m src.ac_powerflow.batch_runner --all-regions --parallel --max-workers 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Region selection (mutually exclusive)
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--region",
        type=str,
        metavar="REGION",
        help=(
            "Process a single region. Valid regions: hokkaido, tohoku, "
            "tokyo, chubu, hokuriku, kansai, chugoku, shikoku, kyushu, "
            "okinawa."
        ),
    )
    region_group.add_argument(
        "--all-regions",
        action="store_true",
        help="Process all active regions from config/regions.yaml.",
    )

    # Parallel execution
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help=(
            "Enable parallel execution using ProcessPoolExecutor. "
            "Each worker processes one region."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of parallel workers. "
            "Default: min(num_regions, cpu_count)."
        ),
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=(
            "Output directory for convergence report and results. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=(
            "Path to the regions.yaml configuration file. "
            f"Default: {DEFAULT_CONFIG_PATH}"
        ),
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the AC power flow batch runner.

    Parses command-line arguments, resolves regions, and executes the
    batch run.

    Args:
        argv: Command-line arguments.  If ``None``, uses ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    setup_logging()

    logger.info("AC Power Flow Batch Runner starting")

    # Resolve regions
    if args.all_regions:
        regions = _load_active_regions(config_path=args.config)
    else:
        # Validate the specified region exists
        all_active = _load_active_regions(config_path=args.config)
        if args.region not in all_active:
            # Still allow it — might be an inactive region the user wants
            logger.warning(
                "Region '%s' is not in active regions: %s",
                args.region, all_active,
            )
        regions = [args.region]

    if not regions:
        logger.error("No regions to process — exiting")
        sys.exit(1)

    logger.info("Regions to process: %s", regions)

    # Run batch
    run_batch(
        regions=regions,
        parallel=args.parallel,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    logger.info("AC Power Flow Batch Runner finished")


if __name__ == "__main__":
    main()
