"""Synthetic load and generation data synthesis with existence checks.

Generates deterministic synthetic load and generation data for buses and
generators that lack it, while preserving existing data.  Uses the
regional demand configuration from ``config/regional_demand.yaml`` and
the voltage-weighted distribution pattern from
:mod:`src.powerflow.load_estimator`.

All stochastic operations use :func:`numpy.random.default_rng` with
a configurable seed for reproducibility.  Running twice with the same
seed and input data produces identical results.

Optionally integrates with :class:`~src.db.grid_db.GridDatabase` to
persist synthesised load attributes for reuse across pipeline runs.

Usage::

    from src.reconstruction.data_synthesizer import DataSynthesizer

    synth = DataSynthesizer(seed=42)
    result = synth.synthesize_loads(net, region="shikoku")
    result = synth.synthesize_generation(net, reserve_margin=0.05)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandapower as pp

from src.powerflow.load_estimator import (
    _voltage_weight,
    load_demand_config,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SynthesisResult:
    """Results from synthetic data generation on a pandapower network.

    Attributes:
        loads_created: Number of new loads created on buses that had none.
        loads_skipped: Number of buses that already had loads (preserved).
        total_load_mw: Total active power allocated across all buses.
        total_load_mvar: Total reactive power allocated across all buses.
        generators_scaled: Number of generators whose output was scaled.
        generators_skipped: Number of generators with existing dispatch
            that were preserved.
        total_generation_mw: Total active generation after scaling.
        seed: Random seed used for this synthesis run.
        warnings: Non-fatal issues encountered during synthesis.
    """

    loads_created: int = 0
    loads_skipped: int = 0
    total_load_mw: float = 0.0
    total_load_mvar: float = 0.0
    generators_scaled: int = 0
    generators_skipped: int = 0
    total_generation_mw: float = 0.0
    seed: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, object]:
        """Return a compact summary for logging."""
        return {
            "loads_created": self.loads_created,
            "loads_skipped": self.loads_skipped,
            "total_load_mw": round(self.total_load_mw, 1),
            "total_load_mvar": round(self.total_load_mvar, 1),
            "generators_scaled": self.generators_scaled,
            "generators_skipped": self.generators_skipped,
            "total_generation_mw": round(self.total_generation_mw, 1),
            "seed": self.seed,
            "warnings": len(self.warnings),
        }


class DataSynthesizer:
    """Deterministic synthetic data generator for power network elements.

    Generates load and generation data with reproducible seeding.  All
    random perturbations (load jitter, generation dispatch) use the
    internal :class:`numpy.random.Generator` so that results are
    identical for the same seed and input data.

    Existing data can optionally be preserved via *skip_existing_loads*
    and *skip_existing_generation* flags.

    Args:
        seed: Integer seed for ``numpy.random.default_rng``.
        skip_existing_loads: If ``True``, buses that already have loads
            attached are skipped during load synthesis.
        skip_existing_generation: If ``True``, generators that already
            have non-zero ``p_mw`` dispatch are skipped during
            generation scaling.
        db: Optional :class:`~src.db.grid_db.GridDatabase` for
            persisting synthesised load attributes.
    """

    def __init__(
        self,
        seed: int = 42,
        skip_existing_loads: bool = True,
        skip_existing_generation: bool = True,
        db: Optional[Any] = None,
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._skip_existing_loads = skip_existing_loads
        self._skip_existing_generation = skip_existing_generation
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_loads(
        self,
        net: Any,
        region: str,
        demand_config: Optional[Dict[str, Any]] = None,
        config_path: str = "config/regional_demand.yaml",
    ) -> SynthesisResult:
        """Synthesize loads for buses that lack them.

        Distributes regional demand across buses proportional to their
        voltage-class weight, following the pattern from
        :func:`~src.powerflow.load_estimator.estimate_loads`.  Buses
        that already have attached loads are skipped when
        *skip_existing_loads* is ``True``.

        A small deterministic jitter (0.9--1.1x) is applied to each
        load allocation using the internal RNG so that load profiles
        are realistic while remaining reproducible.

        Args:
            net: pandapower network (modified in place).
            region: Region identifier (e.g. ``"shikoku"``).
            demand_config: Pre-loaded config dict.  If ``None``, loaded
                from *config_path*.
            config_path: Fallback path for loading config.

        Returns:
            SynthesisResult with load allocation statistics.
        """
        result = SynthesisResult(seed=self._seed)

        if demand_config is None:
            try:
                demand_config = load_demand_config(config_path)
            except FileNotFoundError:
                msg = (
                    f"Demand config not found at '{config_path}'; "
                    f"using built-in defaults"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                demand_config = _default_demand_config()

        peak_demands = demand_config.get("regional_peak_demand_mw", {})
        load_factor = demand_config.get("load_factor", 0.85)
        power_factor = demand_config.get("power_factor", 0.95)
        voltage_weights = demand_config.get("voltage_weights", {})

        # Q/P ratio from power factor
        tan_phi = math.tan(math.acos(power_factor))

        # Determine target demand
        if region == "national":
            target_mw = sum(peak_demands.values()) * load_factor
        else:
            peak_mw = peak_demands.get(region)
            if peak_mw is None:
                msg = (
                    f"No peak demand data for region '{region}'; "
                    f"skipping load synthesis"
                )
                result.warnings.append(msg)
                logger.warning(msg)
                return result
            target_mw = peak_mw * load_factor

        # Identify buses that already have loads
        buses_with_loads = self._get_buses_with_loads(net)

        # Determine buses needing synthetic loads
        all_bus_indices = net.bus.index.tolist()
        if self._skip_existing_loads:
            bus_indices = [
                b for b in all_bus_indices if b not in buses_with_loads
            ]
            result.loads_skipped = len(buses_with_loads)
        else:
            bus_indices = all_bus_indices

        if not bus_indices:
            logger.info(
                "All %d buses already have loads — nothing to synthesize",
                len(all_bus_indices),
            )
            # Report existing load totals
            if not net.load.empty:
                result.total_load_mw = float(net.load["p_mw"].sum())
                result.total_load_mvar = float(net.load["q_mvar"].sum())
            return result

        # Scale target down if some buses already carry load
        existing_load_mw = 0.0
        if not net.load.empty and self._skip_existing_loads:
            existing_load_mw = float(net.load["p_mw"].sum())
        remaining_target_mw = max(target_mw - existing_load_mw, 0.0)

        # Allocate loads to buses that need them
        self._allocate_loads(
            net=net,
            bus_indices=bus_indices,
            target_mw=remaining_target_mw,
            tan_phi=tan_phi,
            voltage_weights=voltage_weights,
            region=region,
            result=result,
        )

        # Include existing loads in totals
        result.total_load_mw += existing_load_mw
        if not net.load.empty:
            result.total_load_mvar = float(net.load["q_mvar"].sum())

        logger.info(
            "Load synthesis complete for region '%s': %s",
            region,
            result.summary,
        )

        return result

    def synthesize_generation(
        self,
        net: Any,
        reserve_margin: float = 0.05,
    ) -> SynthesisResult:
        """Scale generator dispatch to match total demand plus reserve.

        Generators with existing non-zero dispatch are optionally
        preserved.  Remaining generators are dispatched proportional
        to their rated capacity, with deterministic jitter applied
        via the internal RNG for realistic dispatch profiles.

        The total generation target is::

            target = total_load_mw * (1 + reserve_margin)

        The ext_grid (slack bus) absorbs the residual mismatch.

        Args:
            net: pandapower network (modified in place).
            reserve_margin: Fraction of additional generation above
                demand (default 0.05, i.e. 5%).

        Returns:
            SynthesisResult with generation scaling statistics.
        """
        result = SynthesisResult(seed=self._seed)

        # Determine total demand
        total_load_mw = 0.0
        if not net.load.empty:
            total_load_mw = float(net.load["p_mw"].sum())

        if total_load_mw <= 0:
            msg = "No loads in network; skipping generation synthesis"
            result.warnings.append(msg)
            logger.warning(msg)
            return result

        target_gen_mw = total_load_mw * (1.0 + reserve_margin)
        result.total_load_mw = total_load_mw

        if net.gen.empty:
            msg = (
                "No generators in network; ext_grid will supply "
                f"all {total_load_mw:.1f} MW demand"
            )
            result.warnings.append(msg)
            logger.info(msg)
            return result

        # Identify generators with existing dispatch
        if self._skip_existing_generation:
            gens_with_dispatch = self._get_gens_with_dispatch(net)
            existing_gen_mw = sum(
                net.gen.at[g, "p_mw"]
                for g in gens_with_dispatch
                if net.gen.at[g, "p_mw"] > 0
            )
            gens_to_scale = [
                g for g in net.gen.index
                if g not in gens_with_dispatch
            ]
            result.generators_skipped = len(gens_with_dispatch)
        else:
            existing_gen_mw = 0.0
            gens_to_scale = list(net.gen.index)

        remaining_target = max(target_gen_mw - existing_gen_mw, 0.0)

        if not gens_to_scale:
            logger.info(
                "All generators already have dispatch — nothing to scale"
            )
            result.total_generation_mw = existing_gen_mw
            return result

        # Scale generators proportional to capacity with jitter
        self._scale_generators(
            net=net,
            gen_indices=gens_to_scale,
            target_mw=remaining_target,
            result=result,
        )

        # Include existing generation in total
        result.total_generation_mw += existing_gen_mw

        # Ensure generators are in service
        for g in gens_to_scale:
            net.gen.at[g, "in_service"] = True

        logger.info("Generation synthesis complete: %s", result.summary)

        return result

    # ------------------------------------------------------------------
    # Internal: load allocation
    # ------------------------------------------------------------------

    def _allocate_loads(
        self,
        net: Any,
        bus_indices: List[int],
        target_mw: float,
        tan_phi: float,
        voltage_weights: Dict,
        region: str,
        result: SynthesisResult,
    ) -> None:
        """Allocate synthetic loads across a subset of buses.

        Each bus receives a load proportional to its voltage-class
        weight, with deterministic jitter from the internal RNG.

        Args:
            net: pandapower network (modified in place).
            bus_indices: Indices of buses to receive loads.
            target_mw: Total active power to allocate (MW).
            tan_phi: Q/P ratio from power factor.
            voltage_weights: Voltage-class weight mapping.
            region: Region identifier for naming.
            result: SynthesisResult to update.
        """
        if not bus_indices or target_mw <= 0:
            return

        # Compute per-bus weights
        weights = []
        for idx in bus_indices:
            vn_kv = net.bus.at[idx, "vn_kv"]
            w = _voltage_weight(vn_kv, voltage_weights)
            weights.append(w)

        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = len(bus_indices)
            weights = [1.0] * len(bus_indices)

        # Generate deterministic jitter factors (0.9 to 1.1)
        jitter = 0.9 + 0.2 * self._rng.random(len(bus_indices))

        # Apply jitter-weighted allocation
        raw_allocations = []
        for i, (idx, w) in enumerate(zip(bus_indices, weights)):
            base_mw = target_mw * (w / total_weight)
            raw_allocations.append(base_mw * jitter[i])

        # Normalise so total matches target exactly
        raw_total = sum(raw_allocations)
        if raw_total > 0:
            scale = target_mw / raw_total
        else:
            scale = 1.0

        for i, idx in enumerate(bus_indices):
            p_mw = raw_allocations[i] * scale
            q_mvar = p_mw * tan_phi

            load_name = f"synth_load_{region}_{idx}"

            pp.create_load(
                net,
                bus=idx,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=load_name,
            )

            result.loads_created += 1
            result.total_load_mw += p_mw

            # Persist to database if available
            if self._db is not None:
                try:
                    self._db.upsert_load_attributes(
                        load_name,
                        bus_id=str(idx),
                        p_mw=p_mw,
                        q_mvar=q_mvar,
                        power_factor=float(
                            1.0 / math.sqrt(1.0 + tan_phi ** 2)
                        ),
                        source=f"synth_seed_{self._seed}",
                    )
                except Exception as exc:
                    msg = f"Failed to persist load '{load_name}' to DB: {exc}"
                    result.warnings.append(msg)
                    logger.warning(msg)

    # ------------------------------------------------------------------
    # Internal: generation scaling
    # ------------------------------------------------------------------

    def _scale_generators(
        self,
        net: Any,
        gen_indices: List[int],
        target_mw: float,
        result: SynthesisResult,
    ) -> None:
        """Scale generator outputs to meet *target_mw*.

        Dispatch is proportional to each generator's maximum capacity
        with deterministic jitter.  No generator exceeds its rated
        capacity.

        Args:
            net: pandapower network (modified in place).
            gen_indices: Indices of generators to scale.
            target_mw: Total active generation target (MW).
            result: SynthesisResult to update.
        """
        if not gen_indices:
            return

        # Gather capacities
        capacities = []
        for g in gen_indices:
            cap = net.gen.at[g, "max_p_mw"]
            if cap <= 0:
                cap = 1.0  # Fallback for generators without rated capacity
            capacities.append(cap)

        total_capacity = sum(capacities)
        if total_capacity <= 0:
            msg = "Total generator capacity is zero; cannot scale"
            result.warnings.append(msg)
            logger.warning(msg)
            return

        # Base scale factor (capped at 1.0 to respect capacity limits)
        base_scale = min(target_mw / total_capacity, 1.0)

        # Deterministic jitter (0.95 to 1.05) for realistic dispatch
        jitter = 0.95 + 0.1 * self._rng.random(len(gen_indices))

        # Apply scaled dispatch
        allocated = 0.0
        for i, g in enumerate(gen_indices):
            cap = capacities[i]
            p_mw = cap * base_scale * jitter[i]
            # Ensure we don't exceed capacity
            p_mw = min(p_mw, cap)
            p_mw = max(p_mw, 0.0)

            net.gen.at[g, "p_mw"] = p_mw
            allocated += p_mw
            result.generators_scaled += 1

        result.total_generation_mw = allocated

        logger.info(
            "Scaled %d generators: total=%.1f MW "
            "(target=%.1f MW, capacity=%.1f MW, scale=%.3f)",
            len(gen_indices),
            allocated,
            target_mw,
            total_capacity,
            base_scale,
        )

    # ------------------------------------------------------------------
    # Internal: existence checks
    # ------------------------------------------------------------------

    @staticmethod
    def _get_buses_with_loads(net: Any) -> Set[int]:
        """Return the set of bus indices that already have loads attached.

        Args:
            net: pandapower network.

        Returns:
            Set of bus indices with at least one load element.
        """
        if net.load.empty:
            return set()
        return set(net.load["bus"].unique())

    @staticmethod
    def _get_gens_with_dispatch(net: Any) -> Set[int]:
        """Return generator indices that already have non-zero dispatch.

        A generator is considered to have existing dispatch if its
        ``p_mw`` value is greater than zero.

        Args:
            net: pandapower network.

        Returns:
            Set of generator indices with existing dispatch.
        """
        if net.gen.empty:
            return set()
        dispatched = net.gen[net.gen["p_mw"] > 0]
        return set(dispatched.index)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------


def _default_demand_config() -> Dict[str, Any]:
    """Return built-in default demand configuration.

    Used as a fallback when ``config/regional_demand.yaml`` is
    unavailable.  Values match the standard OCCTO 2023 data.

    Returns:
        Demand configuration dictionary.
    """
    return {
        "regional_peak_demand_mw": {
            "hokkaido": 3600,
            "tohoku": 13000,
            "tokyo": 52000,
            "chubu": 24000,
            "hokuriku": 5200,
            "kansai": 27000,
            "chugoku": 10500,
            "shikoku": 5500,
            "kyushu": 15500,
            "okinawa": 1600,
        },
        "load_factor": 0.85,
        "power_factor": 0.95,
        "voltage_weights": {
            500: 0.05,
            275: 0.15,
            220: 0.20,
            187: 0.25,
            154: 0.30,
            132: 0.35,
            110: 0.40,
            77: 0.45,
            66: 0.50,
            0: 0.50,
        },
        "reserve_margin": 0.05,
    }
