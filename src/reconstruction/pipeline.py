"""Main reconstruction pipeline orchestrator for the All-Japan Grid model.

Chains the core reconstruction modules in sequence:

1. **Isolator** — detect isolated buses, lines, generators, and loads.
2. **Simplifier** *or* **Reconnector** — apply the configured strategy
   (``"simplify"`` removes isolated elements, ``"reconnect"`` creates
   synthetic connections).
3. **DataSynthesizer** — generate deterministic synthetic load and
   generation data for buses/generators that lack it.

The pipeline is controlled by a :class:`~src.reconstruction.config.ReconstructionConfig`
that specifies the mode, random seed, thresholds, and data-synthesis options.

**Reproducibility guarantee**: running the pipeline twice with the same
configuration and input network produces identical output, because all
stochastic operations use ``numpy.random.default_rng(seed)``.

The pipeline is **idempotent** — safe to re-run on the same network.
Existing data is preserved when *skip_existing_loads* and
*skip_existing_generation* are set in the configuration.

Usage::

    from src.reconstruction.config import ReconstructionConfig
    from src.reconstruction.pipeline import ReconstructionPipeline

    cfg = ReconstructionConfig(mode="simplify", seed=42)
    pipeline = ReconstructionPipeline(cfg)
    result = pipeline.run(net, region="shikoku")
    print(result.summary)
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.reconstruction.config import ReconstructionConfig
from src.reconstruction.data_synthesizer import DataSynthesizer, SynthesisResult
from src.reconstruction.isolator import Isolator, IsolationResult
from src.reconstruction.reconnector import Reconnector, ReconnectionResult
from src.reconstruction.simplifier import Simplifier, SimplificationResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of running the full reconstruction pipeline.

    Aggregates results from isolation detection, reconstruction
    (simplification or reconnection), and data synthesis into a
    single object with a unified summary for logging and auditing.

    Attributes:
        net: The reconstructed pandapower network.
        reconstruction_mode: The mode that was applied (``"simplify"``
            or ``"reconnect"``).
        seed: Random seed used for reproducibility.
        region: Region identifier for the network.
        isolation_result: Detailed isolation detection results.
        simplification_result: Simplification results (only set when
            mode is ``"simplify"``).
        reconnection_result: Reconnection results (only set when
            mode is ``"reconnect"``).
        synthesis_result: Data synthesis results from load and
            generation synthesis combined.
        elapsed_seconds: Total pipeline execution time in seconds.
        warnings: Aggregated warnings from all pipeline stages.
    """

    net: Any = None
    reconstruction_mode: str = ""
    seed: int = 0
    region: str = ""
    isolation_result: Optional[IsolationResult] = None
    simplification_result: Optional[SimplificationResult] = None
    reconnection_result: Optional[ReconnectionResult] = None
    synthesis_result: Optional[SynthesisResult] = None
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, object]:
        """Return a unified summary of the pipeline execution."""
        result: Dict[str, object] = {
            "mode": self.reconstruction_mode,
            "seed": self.seed,
            "region": self.region,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

        if self.isolation_result is not None:
            result["isolation"] = self.isolation_result.summary

        if self.simplification_result is not None:
            result["simplification"] = self.simplification_result.summary

        if self.reconnection_result is not None:
            result["reconnection"] = self.reconnection_result.summary

        if self.synthesis_result is not None:
            result["synthesis"] = self.synthesis_result.summary

        result["warnings"] = len(self.warnings)

        return result


class ReconstructionPipeline:
    """Orchestrates the full network reconstruction workflow.

    Coordinates :class:`~src.reconstruction.isolator.Isolator`,
    :class:`~src.reconstruction.simplifier.Simplifier` /
    :class:`~src.reconstruction.reconnector.Reconnector`, and
    :class:`~src.reconstruction.data_synthesizer.DataSynthesizer` in
    sequence, driven by a :class:`ReconstructionConfig`.

    The pipeline operates on a **deep copy** of the input network by
    default (``copy_network=True``), leaving the original unchanged.
    Set ``copy_network=False`` to modify in place for performance.

    Args:
        config: Reconstruction configuration controlling mode, seed,
            thresholds, and synthesis options.
        copy_network: If ``True`` (default), the pipeline works on a
            deep copy of the input network.
        db: Optional :class:`~src.db.grid_db.GridDatabase` instance
            for persisting synthesised attributes.
    """

    def __init__(
        self,
        config: ReconstructionConfig,
        copy_network: bool = True,
        db: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._copy_network = copy_network
        self._db = db

    @property
    def config(self) -> ReconstructionConfig:
        """Return the pipeline configuration."""
        return self._config

    def run(
        self,
        net: Any,
        region: str = "unknown",
    ) -> PipelineResult:
        """Execute the full reconstruction pipeline.

        Steps:

        1. Optionally deep-copy the input network.
        2. Detect isolated elements via :class:`Isolator`.
        3. Apply the configured reconstruction strategy:
           - ``"simplify"``: remove isolated elements.
           - ``"reconnect"``: create synthetic connections.
        4. Synthesise load and generation data for elements that lack it.
        5. Return aggregated :class:`PipelineResult`.

        Args:
            net: pandapower network to reconstruct.
            region: Region identifier used for load synthesis
                (e.g. ``"shikoku"``, ``"tokyo"``).

        Returns:
            PipelineResult with the reconstructed network and
            per-stage diagnostics.

        Raises:
            ValueError: If the network is empty or the reconstruction
                mode produces an invalid network.
        """
        t0 = time.monotonic()

        result = PipelineResult(
            reconstruction_mode=self._config.mode,
            seed=self._config.seed,
            region=region,
        )

        logger.info(
            "Starting reconstruction pipeline: mode=%s, seed=%d, "
            "region='%s'",
            self._config.mode,
            self._config.seed,
            region,
        )
        logger.info("Configuration: %s", self._config.summary)

        # Step 0: Optionally copy the network
        if self._copy_network:
            net = copy.deepcopy(net)

        result.net = net

        # Step 1: Isolation detection
        isolation_result = self._detect_isolation(net, result)
        result.isolation_result = isolation_result

        # Step 2: Apply reconstruction strategy
        if self._config.mode == "simplify":
            self._apply_simplification(net, isolation_result, result)
        else:
            self._apply_reconnection(net, isolation_result, result)

        # Step 3: Data synthesis (loads + generation)
        self._synthesize_data(net, region, result)

        # Finalise
        result.elapsed_seconds = time.monotonic() - t0

        logger.info(
            "Reconstruction pipeline complete in %.3fs: %s",
            result.elapsed_seconds,
            result.summary,
        )

        return result

    # ------------------------------------------------------------------
    # Stage 1: Isolation detection
    # ------------------------------------------------------------------

    def _detect_isolation(
        self,
        net: Any,
        result: PipelineResult,
    ) -> IsolationResult:
        """Run isolation detection on the network.

        Creates an :class:`Isolator` configured with the minimum
        component size from the pipeline configuration.

        Args:
            net: pandapower network to analyse.
            result: PipelineResult for aggregating warnings.

        Returns:
            IsolationResult with detected isolated elements.
        """
        isolator = Isolator(
            min_component_size=self._config.min_component_size,
        )

        isolation_result = isolator.detect(net)

        # Propagate warnings
        result.warnings.extend(isolation_result.warnings)

        if isolation_result.has_isolation:
            logger.info(
                "Isolation detected: %d buses, %d lines, %d generators",
                len(isolation_result.isolated_buses),
                len(isolation_result.isolated_lines),
                len(isolation_result.isolated_generators),
            )
        else:
            logger.info("No isolation detected — network is fully connected")

        return isolation_result

    # ------------------------------------------------------------------
    # Stage 2a: Simplification
    # ------------------------------------------------------------------

    def _apply_simplification(
        self,
        net: Any,
        isolation_result: IsolationResult,
        result: PipelineResult,
    ) -> None:
        """Apply simplification mode to the network.

        Removes all isolated elements to produce a single connected
        component.

        Args:
            net: pandapower network (modified in place).
            isolation_result: Detection results from Stage 1.
            result: PipelineResult for aggregating results and warnings.
        """
        simplifier = Simplifier(ensure_ext_grid=True)

        try:
            simplification_result = simplifier.simplify(
                net, isolation_result,
            )
        except ValueError as exc:
            msg = f"Simplification failed: {exc}"
            result.warnings.append(msg)
            logger.error(msg)
            # Re-raise — an empty network is unrecoverable in simplify mode
            raise

        result.simplification_result = simplification_result
        result.warnings.extend(simplification_result.warnings)

        logger.info(
            "Simplification applied: %s",
            simplification_result.summary,
        )

    # ------------------------------------------------------------------
    # Stage 2b: Reconnection
    # ------------------------------------------------------------------

    def _apply_reconnection(
        self,
        net: Any,
        isolation_result: IsolationResult,
        result: PipelineResult,
    ) -> None:
        """Apply reconnection mode to the network.

        Creates synthetic transmission lines to reconnect isolated
        elements to the main component and validates the Ybus matrix.

        Args:
            net: pandapower network (modified in place).
            isolation_result: Detection results from Stage 1.
            result: PipelineResult for aggregating results and warnings.
        """
        reconnector = Reconnector()

        reconnection_result = reconnector.reconnect(
            net, isolation_result, self._config,
        )

        result.reconnection_result = reconnection_result
        result.warnings.extend(reconnection_result.warnings)

        logger.info(
            "Reconnection applied: %s",
            reconnection_result.summary,
        )

    # ------------------------------------------------------------------
    # Stage 3: Data synthesis
    # ------------------------------------------------------------------

    def _synthesize_data(
        self,
        net: Any,
        region: str,
        result: PipelineResult,
    ) -> None:
        """Synthesise load and generation data.

        Creates a :class:`DataSynthesizer` with the configured seed
        and synthesis options, then runs load synthesis followed by
        generation scaling.

        Args:
            net: pandapower network (modified in place).
            region: Region identifier for load demand lookup.
            result: PipelineResult for aggregating results and warnings.
        """
        synthesizer = DataSynthesizer(
            seed=self._config.seed,
            skip_existing_loads=self._config.skip_existing_loads,
            skip_existing_generation=self._config.skip_existing_generation,
            db=self._db,
        )

        # Synthesise loads
        load_result = synthesizer.synthesize_loads(net, region=region)

        # Synthesise generation
        gen_result = synthesizer.synthesize_generation(
            net, reserve_margin=self._config.reserve_margin,
        )

        # Merge synthesis results into a single SynthesisResult
        synthesis_result = _merge_synthesis_results(load_result, gen_result)

        result.synthesis_result = synthesis_result
        result.warnings.extend(synthesis_result.warnings)

        logger.info("Data synthesis applied: %s", synthesis_result.summary)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------


def _merge_synthesis_results(
    load_result: SynthesisResult,
    gen_result: SynthesisResult,
) -> SynthesisResult:
    """Merge separate load and generation synthesis results.

    Combines counters, totals, and warnings from the two partial
    results into a single :class:`SynthesisResult`.

    Args:
        load_result: Results from load synthesis.
        gen_result: Results from generation scaling.

    Returns:
        Merged SynthesisResult.
    """
    merged = SynthesisResult(
        loads_created=load_result.loads_created,
        loads_skipped=load_result.loads_skipped,
        total_load_mw=load_result.total_load_mw,
        total_load_mvar=load_result.total_load_mvar,
        generators_scaled=gen_result.generators_scaled,
        generators_skipped=gen_result.generators_skipped,
        total_generation_mw=gen_result.total_generation_mw,
        seed=load_result.seed,
    )

    # Combine warnings from both stages
    merged.warnings = list(load_result.warnings) + list(gen_result.warnings)

    return merged
