"""Problem decomposition strategies for unit commitment.

Provides strategies to partition large UC problems into smaller
sub-problems, solve each independently, and merge results.  Three
decomposition strategies are available:

- **Regional**: partitions generators by ``generator.region``.
- **Fuel type**: partitions generators by ``generator.fuel_type_enum``.
- **Time window**: splits the time horizon into overlapping chunks.

Usage::

    from src.uc.decomposition import create_decomposer

    decomposer = create_decomposer('regional')
    result = decomposer.solve_decomposed(params)
"""

import abc
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.model.generator import Generator
from src.uc.models import (
    DemandProfile,
    GeneratorSchedule,
    TimeHorizon,
    UCParameters,
    UCResult,
)
from src.uc.solver import solve_uc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Decomposer(abc.ABC):
    """Abstract base class for UC problem decomposition strategies.

    Subclasses must implement :meth:`partition` to split a UC problem
    into independent sub-problems.  The default :meth:`solve_decomposed`
    solves each partition via ``solve_uc`` and merges the results.
    """

    @abc.abstractmethod
    def partition(self, params: UCParameters) -> List[UCParameters]:
        """Partition a UC problem into sub-problems.

        Args:
            params: Full UC problem specification.

        Returns:
            List of independent UCParameters sub-problems.  May return
            ``[params]`` unchanged if decomposition is unnecessary.
        """

    def solve_decomposed(self, params: UCParameters) -> UCResult:
        """Solve a UC problem via decomposition.

        Partitions the problem, solves each sub-problem independently
        using ``solve_uc``, and merges the results.

        Args:
            params: Full UC problem specification.

        Returns:
            Merged UCResult from all sub-problems.
        """
        start = time.monotonic()

        partitions = self.partition(params)
        if not partitions:
            logger.warning(
                "%s: decomposition produced no partitions",
                type(self).__name__,
            )
            return UCResult(
                status="Not Solved",
                warnings=["Decomposition produced no partitions"],
            )

        logger.info(
            "%s: decomposed into %d partition(s)",
            type(self).__name__,
            len(partitions),
        )

        results: List[UCResult] = []
        for i, sub_params in enumerate(partitions):
            logger.info(
                "Solving partition %d/%d (%d generators, %d periods)",
                i + 1,
                len(partitions),
                len(sub_params.generators),
                (
                    sub_params.time_horizon.num_periods
                    if sub_params.time_horizon
                    else 0
                ),
            )
            sub_result = solve_uc(sub_params)
            results.append(sub_result)
            logger.info(
                "Partition %d/%d: status=%s, cost=%.2f",
                i + 1,
                len(partitions),
                sub_result.status,
                sub_result.total_cost,
            )

        merged = self.merge_results(results)
        merged.solve_time_s = time.monotonic() - start

        logger.info(
            "%s: merged result: status=%s, total_cost=%.2f, "
            "solve_time=%.2fs",
            type(self).__name__,
            merged.status,
            merged.total_cost,
            merged.solve_time_s,
        )
        return merged

    def merge_results(self, results: List[UCResult]) -> UCResult:
        """Merge results from multiple sub-problems.

        Concatenates generator schedules, sums costs, and determines
        overall status.  Works correctly for generator-based partitioning
        (Regional, FuelType) where each sub-problem has disjoint
        generators.

        Args:
            results: List of UCResult from each sub-problem.

        Returns:
            Combined UCResult.
        """
        if not results:
            return UCResult(
                status="Not Solved",
                warnings=["No sub-problem results to merge"],
            )

        merged = UCResult()
        merged.schedules = []
        merged.warnings = []
        merged.total_cost = 0.0

        for r in results:
            merged.schedules.extend(r.schedules)
            merged.total_cost += r.total_cost
            merged.warnings.extend(r.warnings)

        # Determine overall status
        statuses = [r.status for r in results]
        if all(s == "Optimal" for s in statuses):
            merged.status = "Optimal"
        elif any(s == "Infeasible" for s in statuses):
            merged.status = "Infeasible"
        elif any(s == "Unbounded" for s in statuses):
            merged.status = "Unbounded"
        else:
            merged.status = next(
                (s for s in statuses if s != "Optimal"),
                "Not Solved",
            )

        # Aggregate gap (worst case across sub-problems)
        gaps = [r.gap for r in results if r.gap is not None]
        if gaps:
            merged.gap = max(gaps)

        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_demand_by_capacity(
    params: UCParameters,
    groups: Dict[str, List[Generator]],
) -> Dict[str, List[float]]:
    """Split demand proportionally based on group capacity.

    Each generator group receives a fraction of total demand proportional
    to its share of total generation capacity.

    Args:
        params: Original UC parameters with full demand profile.
        groups: Mapping of group key to list of generators.

    Returns:
        Mapping of group key to demand list for that group.
    """
    if params.demand is None:
        return {key: [] for key in groups}

    total_cap = sum(g.capacity_mw for gens in groups.values() for g in gens)
    if total_cap <= 0:
        # Equal split if no capacity info
        n = len(groups) or 1
        return {
            key: [d / n for d in params.demand.demands] for key in groups
        }

    result: Dict[str, List[float]] = {}
    for key, gens in groups.items():
        group_cap = sum(g.capacity_mw for g in gens)
        fraction = group_cap / total_cap
        result[key] = [d * fraction for d in params.demand.demands]

    return result


def _build_sub_params(
    generators: List[Generator],
    demand: Optional[DemandProfile],
    params: UCParameters,
) -> UCParameters:
    """Build a sub-problem UCParameters sharing solver config with parent.

    Args:
        generators: Generators for the sub-problem.
        demand: Demand profile for the sub-problem.
        params: Parent UCParameters for solver configuration.

    Returns:
        New UCParameters instance for the sub-problem.
    """
    return UCParameters(
        generators=generators,
        demand=demand,
        time_horizon=params.time_horizon,
        reserve_margin=params.reserve_margin,
        solver_name=params.solver_name,
        solver_time_limit_s=params.solver_time_limit_s,
        mip_gap=params.mip_gap,
        solver_options=params.solver_options,
    )


# ---------------------------------------------------------------------------
# Regional decomposer
# ---------------------------------------------------------------------------


class RegionalDecomposer(Decomposer):
    """Partition generators by region.

    Creates one sub-problem per unique ``generator.region`` value.
    Demand is split proportionally based on each region's total
    generation capacity.
    """

    def partition(self, params: UCParameters) -> List[UCParameters]:
        """Partition by generator region.

        Args:
            params: Full UC problem specification.

        Returns:
            One UCParameters per region with proportional demand.
            Returns ``[params]`` if all generators share the same
            region, the generator list is empty, or interconnections
            are present (national MILP fallback).
        """
        # When interconnections are present, regional decomposition would
        # break the inter-region coupling constraints.  Fall back to a
        # single national MILP so that nodal balance and transmission
        # capacity constraints are handled correctly.
        if params.interconnections:
            logger.info(
                "RegionalDecomposer: Interconnections present: bypassing "
                "regional decomposition for national MILP"
            )
            return [params]

        if not params.generators:
            logger.warning("RegionalDecomposer: no generators to partition")
            return []

        # Group generators by region
        groups: Dict[str, List[Generator]] = defaultdict(list)
        for g in params.generators:
            key = g.region if g.region else "_unassigned"
            groups[key].append(g)

        logger.info(
            "RegionalDecomposer: %d region(s): %s",
            len(groups),
            ", ".join(
                f"{k} ({len(v)} gens)"
                for k, v in sorted(groups.items())
            ),
        )

        # Skip decomposition if only one group
        if len(groups) == 1:
            logger.info(
                "RegionalDecomposer: single region, "
                "no decomposition needed"
            )
            return [params]

        # Split demand proportionally
        demand_splits = _split_demand_by_capacity(params, groups)

        partitions: List[UCParameters] = []
        for key in sorted(groups):
            gens = groups[key]
            sub_demand = (
                DemandProfile(demands=demand_splits[key])
                if params.demand is not None
                else None
            )
            partitions.append(_build_sub_params(gens, sub_demand, params))

        return partitions


# ---------------------------------------------------------------------------
# Fuel-type decomposer
# ---------------------------------------------------------------------------


class FuelTypeDecomposer(Decomposer):
    """Partition generators by fuel type.

    Creates one sub-problem per unique ``generator.fuel_type_enum``
    value.  Demand is split proportionally based on each fuel group's
    total generation capacity.
    """

    def partition(self, params: UCParameters) -> List[UCParameters]:
        """Partition by generator fuel type.

        Args:
            params: Full UC problem specification.

        Returns:
            One UCParameters per fuel type with proportional demand.
            Returns ``[params]`` if all generators share the same
            fuel type or the generator list is empty.
        """
        if not params.generators:
            logger.warning("FuelTypeDecomposer: no generators to partition")
            return []

        # Group generators by fuel type enum value
        groups: Dict[str, List[Generator]] = defaultdict(list)
        for g in params.generators:
            key = g.fuel_type_enum.value
            groups[key].append(g)

        logger.info(
            "FuelTypeDecomposer: %d fuel type(s): %s",
            len(groups),
            ", ".join(
                f"{k} ({len(v)} gens)"
                for k, v in sorted(groups.items())
            ),
        )

        # Skip decomposition if only one group
        if len(groups) == 1:
            logger.info(
                "FuelTypeDecomposer: single fuel type, "
                "no decomposition needed"
            )
            return [params]

        # Split demand proportionally
        demand_splits = _split_demand_by_capacity(params, groups)

        partitions: List[UCParameters] = []
        for key in sorted(groups):
            gens = groups[key]
            sub_demand = (
                DemandProfile(demands=demand_splits[key])
                if params.demand is not None
                else None
            )
            partitions.append(_build_sub_params(gens, sub_demand, params))

        return partitions


# ---------------------------------------------------------------------------
# Time-window decomposer
# ---------------------------------------------------------------------------


class TimeWindowDecomposer(Decomposer):
    """Split the time horizon into overlapping windows.

    Each window includes all generators but covers only a portion of
    the time horizon.  Adjacent windows overlap by ``overlap`` periods
    to improve boundary solutions.  During merge, only the core
    (non-overlapping) periods from each window are kept.

    Args:
        window_size: Number of periods per window (default 12).
        overlap: Number of overlapping periods between adjacent
            windows (default 2).
    """

    def __init__(self, window_size: int = 12, overlap: int = 2) -> None:
        if window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {window_size}"
            )
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) must be < window_size ({window_size})"
            )
        self.window_size = window_size
        self.overlap = overlap

    def partition(self, params: UCParameters) -> List[UCParameters]:
        """Split time horizon into overlapping windows.

        Args:
            params: Full UC problem specification.

        Returns:
            One UCParameters per time window with all generators.
            Returns ``[params]`` if the horizon fits in a single window.
        """
        if params.time_horizon is None or params.demand is None:
            logger.warning(
                "TimeWindowDecomposer: missing time_horizon or demand"
            )
            return []

        total_periods = params.time_horizon.num_periods

        if total_periods <= self.window_size:
            logger.info(
                "TimeWindowDecomposer: %d periods <= window_size %d, "
                "no decomposition needed",
                total_periods,
                self.window_size,
            )
            return [params]

        step = self.window_size - self.overlap
        partitions: List[UCParameters] = []
        start = 0

        while start < total_periods:
            end = min(start + self.window_size, total_periods)
            window_periods = end - start

            sub_th = TimeHorizon(
                num_periods=window_periods,
                period_duration_h=params.time_horizon.period_duration_h,
                start_period=params.time_horizon.start_period + start,
            )
            sub_demand = DemandProfile(
                demands=params.demand.demands[start:end]
            )
            sub_params = UCParameters(
                generators=params.generators,
                demand=sub_demand,
                time_horizon=sub_th,
                reserve_margin=params.reserve_margin,
                solver_name=params.solver_name,
                solver_time_limit_s=params.solver_time_limit_s,
                mip_gap=params.mip_gap,
                solver_options=params.solver_options,
            )
            partitions.append(sub_params)

            start += step
            if end >= total_periods:
                break

        logger.info(
            "TimeWindowDecomposer: %d window(s), size=%d, overlap=%d, "
            "total_periods=%d",
            len(partitions),
            self.window_size,
            self.overlap,
            total_periods,
        )
        return partitions

    def solve_decomposed(self, params: UCParameters) -> UCResult:
        """Solve via time-window decomposition with result stitching.

        Overrides the base implementation to correctly stitch generator
        schedules across time windows, using only core (non-overlapping)
        periods from each window.

        Args:
            params: Full UC problem specification.

        Returns:
            UCResult with stitched schedules spanning the full horizon.
        """
        start_time = time.monotonic()

        partitions = self.partition(params)
        if not partitions:
            logger.warning("TimeWindowDecomposer: no partitions created")
            return UCResult(
                status="Not Solved",
                warnings=["Decomposition produced no partitions"],
            )

        # If no decomposition was needed, solve directly
        if len(partitions) == 1:
            logger.info("TimeWindowDecomposer: single window, solving directly")
            result = solve_uc(partitions[0])
            result.solve_time_s = time.monotonic() - start_time
            return result

        logger.info(
            "TimeWindowDecomposer: solving %d window(s)",
            len(partitions),
        )

        # Solve each window
        results: List[UCResult] = []
        for i, sub_params in enumerate(partitions):
            logger.info(
                "Solving window %d/%d (periods %d-%d)",
                i + 1,
                len(partitions),
                sub_params.time_horizon.start_period,
                sub_params.time_horizon.start_period
                + sub_params.time_horizon.num_periods
                - 1,
            )
            sub_result = solve_uc(sub_params)
            results.append(sub_result)
            logger.info(
                "Window %d/%d: status=%s, cost=%.2f",
                i + 1,
                len(partitions),
                sub_result.status,
                sub_result.total_cost,
            )

        # Stitch results across time windows
        merged = self._stitch_time_windows(params, partitions, results)
        merged.solve_time_s = time.monotonic() - start_time

        logger.info(
            "TimeWindowDecomposer: merged result: status=%s, "
            "total_cost=%.2f, solve_time=%.2fs",
            merged.status,
            merged.total_cost,
            merged.solve_time_s,
        )
        return merged

    def _stitch_time_windows(
        self,
        original_params: UCParameters,
        partitions: List[UCParameters],
        results: List[UCResult],
    ) -> UCResult:
        """Stitch time-window results into a single full-horizon result.

        For each window, the core (non-overlapping) periods are
        extracted and concatenated to form the full-horizon schedule.
        Costs are recomputed from the stitched commitment and power
        output arrays.

        Args:
            original_params: Original full-horizon UCParameters.
            partitions: List of window-level UCParameters.
            results: List of window-level UCResults.

        Returns:
            Merged UCResult spanning the full time horizon.
        """
        merged = UCResult()
        merged.warnings = []

        # Determine overall status
        statuses = [r.status for r in results]
        if all(s == "Optimal" for s in statuses):
            merged.status = "Optimal"
        elif any(s == "Infeasible" for s in statuses):
            merged.status = "Infeasible"
        elif any(s == "Unbounded" for s in statuses):
            merged.status = "Unbounded"
        else:
            merged.status = next(
                (s for s in statuses if s != "Optimal"),
                "Not Solved",
            )

        for r in results:
            merged.warnings.extend(r.warnings)

        # Aggregate gap (worst case)
        gaps = [r.gap for r in results if r.gap is not None]
        if gaps:
            merged.gap = max(gaps)

        # If any sub-problem failed to produce schedules, sum costs and
        # return without stitched schedules.
        if not all(r.schedules for r in results):
            merged.total_cost = sum(r.total_cost for r in results)
            return merged

        # Determine core period ranges for each window.
        #
        # Each window overlaps with its neighbours by ``self.overlap``
        # periods.  The first window discards its trailing overlap
        # (those periods are solved again by the next window with
        # better look-ahead context).  Subsequent windows keep the
        # leading overlap (which provides the boundary solution) and
        # discard only their trailing overlap (except the last window
        # which keeps everything).
        #
        # With step = window_size - overlap this guarantees that the
        # core ranges tile the full horizon without gaps or double-
        # counting.
        step = self.window_size - self.overlap
        core_ranges: List[Tuple[int, int]] = []

        for i, part in enumerate(partitions):
            window_len = part.time_horizon.num_periods
            if i == len(partitions) - 1:
                # Last window: take all local periods
                core_ranges.append((0, window_len))
            else:
                # Non-last windows: take first ``step`` periods
                core_end = min(step, window_len)
                core_ranges.append((0, core_end))

        # Build schedule index per window for fast lookup
        window_sched_maps: List[Dict[str, GeneratorSchedule]] = []
        for r in results:
            smap: Dict[str, GeneratorSchedule] = {}
            for s in r.schedules:
                smap[s.generator_id] = s
            window_sched_maps.append(smap)

        # Stitch per-generator schedules
        gen_schedules: Dict[str, GeneratorSchedule] = {}
        for g in original_params.generators:
            commitment: List[int] = []
            power_output: List[float] = []

            for i, (core_start, core_end) in enumerate(core_ranges):
                gen_sched = window_sched_maps[i].get(g.id)
                if gen_sched is None:
                    # Generator missing from window results (unexpected)
                    commitment.extend([0] * (core_end - core_start))
                    power_output.extend([0.0] * (core_end - core_start))
                else:
                    commitment.extend(
                        gen_sched.commitment[core_start:core_end]
                    )
                    power_output.extend(
                        gen_sched.power_output_mw[core_start:core_end]
                    )

            # Recompute per-generator costs from stitched schedule
            fuel_cost = sum(
                g.fuel_cost_per_mwh * p for p in power_output
            )
            no_load_cost = sum(
                (g.no_load_cost + g.labor_cost_per_h) * u
                for u in commitment
            )
            startup_cost = 0.0
            shutdown_cost = 0.0
            for t in range(len(commitment)):
                if t == 0:
                    # Assume generator starts offline at horizon start
                    if commitment[t] == 1:
                        startup_cost += g.startup_cost
                else:
                    if commitment[t] == 1 and commitment[t - 1] == 0:
                        startup_cost += g.startup_cost
                    if commitment[t] == 0 and commitment[t - 1] == 1:
                        shutdown_cost += g.shutdown_cost

            gen_schedules[g.id] = GeneratorSchedule(
                generator_id=g.id,
                commitment=commitment,
                power_output_mw=power_output,
                startup_cost=startup_cost,
                shutdown_cost=shutdown_cost,
                fuel_cost=fuel_cost,
                no_load_cost=no_load_cost,
            )

        merged.schedules = list(gen_schedules.values())
        merged.total_cost = sum(s.total_cost for s in merged.schedules)

        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Strategy name -> decomposer class mapping
_STRATEGY_MAP: Dict[str, type] = {
    "regional": RegionalDecomposer,
    "fuel_type": FuelTypeDecomposer,
    "time_window": TimeWindowDecomposer,
}


def create_decomposer(strategy: str, **kwargs) -> Decomposer:
    """Create a decomposer instance by strategy name.

    Args:
        strategy: Decomposition strategy name.  One of ``"regional"``,
            ``"fuel_type"``, or ``"time_window"``.
        **kwargs: Additional keyword arguments passed to the decomposer
            constructor (e.g., ``window_size`` and ``overlap`` for
            ``TimeWindowDecomposer``).

    Returns:
        Configured Decomposer instance.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    cls = _STRATEGY_MAP.get(strategy.lower())
    if cls is None:
        valid = ", ".join(sorted(_STRATEGY_MAP))
        raise ValueError(
            f"Unknown decomposition strategy '{strategy}'. "
            f"Valid strategies: {valid}"
        )

    logger.info("Creating decomposer: strategy=%s, kwargs=%s", strategy, kwargs)
    return cls(**kwargs)
