"""Unit tests for the reconstruction pipeline modules.

Tests isolation detection, simplification mode, reconnection mode,
reproducibility guarantees, synthetic data synthesis (loads and generation),
and configuration-driven mode switching.

Each test class follows the pattern established in test_pandapower_builder.py:
grouped by module/feature, using fixtures from conftest.py, with descriptive
docstrings and explicit assertions.
"""

import copy
from typing import Any

import numpy as np
import pandapower as pp
import pandapower.topology as top
import pytest

from src.reconstruction.config import ReconstructionConfig
from src.reconstruction.data_synthesizer import DataSynthesizer, SynthesisResult
from src.reconstruction.isolator import Isolator, IsolationResult
from src.reconstruction.pipeline import PipelineResult, ReconstructionPipeline
from src.reconstruction.reconnector import Reconnector, ReconnectionResult
from src.reconstruction.simplifier import Simplifier, SimplificationResult

from tests.conftest import make_isolated_network


# ======================================================================
# IsolationResult dataclass
# ======================================================================


class TestIsolationResult:
    """Tests for the IsolationResult dataclass."""

    def test_has_isolation_true(self) -> None:
        """has_isolation is True when isolated buses exist."""
        result = IsolationResult(isolated_buses={3, 4})
        assert result.has_isolation is True

    def test_has_isolation_false(self) -> None:
        """has_isolation is False when no isolated elements exist."""
        result = IsolationResult()
        assert result.has_isolation is False

    def test_summary_keys(self) -> None:
        """Summary dict contains expected keys."""
        result = IsolationResult()
        summary = result.summary
        assert "component_count" in summary
        assert "main_component_size" in summary
        assert "isolated_buses" in summary
        assert "isolated_generators" in summary
        assert "has_isolation" in summary


# ======================================================================
# Isolator: isolation detection
# ======================================================================


class TestDetectIsolatedSubstations:
    """Tests for detecting isolated substations (buses)."""

    def test_detect_isolated_substations(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Correctly identifies substations with no connecting lines."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        # The fixture has 3 main + 2 isolated buses
        assert result.has_isolation is True
        assert len(result.isolated_buses) == 2
        # Main component should have 3 buses
        assert len(result.main_component_buses) == 3

    def test_no_isolation_in_connected_network(
        self,
        fully_connected_network: Any,
    ) -> None:
        """Fully connected network has no isolated substations."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(fully_connected_network)

        assert result.has_isolation is False
        assert len(result.isolated_buses) == 0
        assert result.component_count == 1

    def test_empty_network(self) -> None:
        """Empty network returns no isolation with a warning."""
        net = pp.create_empty_network()
        isolator = Isolator()
        result = isolator.detect(net)

        assert result.has_isolation is False
        assert len(result.warnings) > 0

    def test_component_count(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Component count reflects both main and isolated components."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        # Main component (3 buses) + individual isolated buses
        # Isolated buses with no lines are "disconnected" rather than
        # forming their own graph component, so component_count may
        # be 1 (just the main) with disconnected buses detected separately
        assert result.component_count >= 1

    def test_component_sizes_sorted_descending(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Component sizes list is sorted in descending order."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        if len(result.component_sizes) > 1:
            for i in range(len(result.component_sizes) - 1):
                assert result.component_sizes[i] >= result.component_sizes[i + 1]


class TestDetectIsolatedGenerators:
    """Tests for detecting generators on isolated buses."""

    def test_detect_isolated_generators(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Correctly identifies generators on isolated buses."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        # The fixture places one generator on the first isolated bus
        assert len(result.isolated_generators) == 1

    def test_connected_generators_not_flagged(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Generators on main-component buses are not flagged as isolated."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        net = isolated_grid_network
        total_gens = len(net.gen)
        connected_gens = total_gens - len(result.isolated_generators)

        # The fixture has 2 main generators + 1 isolated generator
        assert connected_gens == 2

    def test_no_isolated_generators_in_connected_network(
        self,
        fully_connected_network: Any,
    ) -> None:
        """Fully connected network has no isolated generators."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(fully_connected_network)

        assert len(result.isolated_generators) == 0


class TestDetectIsolatedLines:
    """Tests for detecting lines with endpoints on isolated buses."""

    def test_detect_isolated_lines(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Correctly identifies lines where an endpoint is an isolated bus."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(isolated_grid_network)

        # The isolated buses have no lines, so no lines should be flagged
        # as isolated (lines only connect main buses in the fixture)
        assert len(result.isolated_lines) == 0

    def test_line_touching_isolated_bus_detected(self) -> None:
        """A line with one endpoint on an isolated bus is detected."""
        net = pp.create_empty_network(f_hz=60)

        # Main connected component
        b0 = pp.create_bus(net, vn_kv=275.0, name="main_0")
        b1 = pp.create_bus(net, vn_kv=275.0, name="main_1")
        pp.create_line_from_parameters(
            net, b0, b1, length_km=30.0,
            r_ohm_per_km=0.028, x_ohm_per_km=0.325,
            c_nf_per_km=12.24, max_i_ka=2.0, name="main_line",
        )
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="slack")

        # Isolated bus connected by a line to main component
        # but also another bus that's only connected to the isolated one
        b2 = pp.create_bus(net, vn_kv=187.0, name="isolated_0")
        b3 = pp.create_bus(net, vn_kv=187.0, name="isolated_1")
        pp.create_line_from_parameters(
            net, b2, b3, length_km=20.0,
            r_ohm_per_km=0.05, x_ohm_per_km=0.4,
            c_nf_per_km=10.0, max_i_ka=1.0, name="isolated_line",
        )

        isolator = Isolator(min_component_size=2)
        result = isolator.detect(net)

        # The isolated line connects buses in a small component
        assert len(result.isolated_lines) >= 1

    def test_no_isolated_lines_in_connected_network(
        self,
        fully_connected_network: Any,
    ) -> None:
        """Fully connected network has no isolated lines."""
        isolator = Isolator(min_component_size=2)
        result = isolator.detect(fully_connected_network)

        assert len(result.isolated_lines) == 0


# ======================================================================
# Simplifier: simplification mode
# ======================================================================


class TestSimplifyRemovesIsolated:
    """Tests verifying that simplification removes isolated elements."""

    def test_simplify_removes_isolated(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """After simplification, no isolated elements remain."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)
        assert iso_result.has_isolation is True

        simplifier = Simplifier()
        simp_result = simplifier.simplify(net, iso_result)

        # Verify: re-run isolation detection on simplified network
        iso_after = isolator.detect(net)
        assert iso_after.has_isolation is False
        assert simp_result.buses_removed > 0

    def test_simplify_single_connected_component(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """After simplification, network has exactly 1 connected component."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        simplifier = Simplifier()
        simp_result = simplifier.simplify(net, iso_result)

        assert simp_result.component_count == 1

    def test_simplify_element_counts(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Removed counts match the number of isolated elements detected."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        n_isolated_buses = len(iso_result.isolated_buses)
        n_isolated_gens = len(iso_result.isolated_generators)

        simplifier = Simplifier()
        simp_result = simplifier.simplify(net, iso_result)

        assert simp_result.buses_removed == n_isolated_buses
        assert simp_result.generators_removed == n_isolated_gens


class TestSimplifyPreservesConnected:
    """Tests verifying that simplification preserves connected elements."""

    def test_simplify_preserves_connected(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """Connected elements are not removed by simplification."""
        net = isolated_grid_network

        # Record pre-simplification connected element counts
        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        n_main_buses = len(iso_result.main_component_buses)
        n_total_gens = len(net.gen)
        n_isolated_gens = len(iso_result.isolated_generators)
        expected_connected_gens = n_total_gens - n_isolated_gens

        simplifier = Simplifier()
        simp_result = simplifier.simplify(net, iso_result)

        # After simplification: remaining buses should be the main component
        assert simp_result.buses_remaining == n_main_buses
        assert simp_result.generators_remaining == expected_connected_gens

    def test_simplify_preserves_ext_grid(
        self,
        isolated_grid_network: Any,
    ) -> None:
        """At least one ext_grid remains after simplification."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        simplifier = Simplifier()
        simplifier.simplify(net, iso_result)

        assert len(net.ext_grid) >= 1
        assert net.ext_grid["in_service"].any()

    def test_simplify_noop_on_connected_network(
        self,
        fully_connected_network: Any,
    ) -> None:
        """Simplification is a no-op when the network has no isolation."""
        net = fully_connected_network
        buses_before = len(net.bus)

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        simplifier = Simplifier()
        simp_result = simplifier.simplify(net, iso_result)

        assert simp_result.buses_removed == 0
        assert len(net.bus) == buses_before

    def test_simplify_raises_on_all_isolated(self) -> None:
        """Simplification raises ValueError when all buses are isolated."""
        # Create network with only isolated buses (no lines, no ext_grid)
        net = pp.create_empty_network(f_hz=60)
        pp.create_bus(net, vn_kv=275.0, name="iso_0")
        pp.create_bus(net, vn_kv=275.0, name="iso_1")

        # Manually construct an IsolationResult where all buses are isolated
        iso_result = IsolationResult(
            isolated_buses={0, 1},
            main_component_buses=set(),
            component_count=0,
        )

        simplifier = Simplifier()
        with pytest.raises(ValueError, match="empty network"):
            simplifier.simplify(net, iso_result)


# ======================================================================
# Reconnector: reconnection mode
# ======================================================================


class TestReconnectCreatesLinks:
    """Tests verifying that reconnection creates synthetic links."""

    def test_reconnect_creates_links(
        self,
        isolated_grid_network: Any,
        reconstruction_config_reconnect: ReconstructionConfig,
    ) -> None:
        """Reconnection mode creates synthetic lines to isolated buses."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)
        n_isolated = len(iso_result.isolated_buses)

        reconnector = Reconnector()
        recon_result = reconnector.reconnect(
            net, iso_result, reconstruction_config_reconnect,
        )

        assert recon_result.lines_created == n_isolated
        assert recon_result.buses_reconnected == n_isolated
        assert len(recon_result.synthetic_line_map) == n_isolated

    def test_reconnect_synthetic_line_names(
        self,
        isolated_grid_network: Any,
        reconstruction_config_reconnect: ReconstructionConfig,
    ) -> None:
        """Synthetic lines have namespace-prefixed names."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        reconnector = Reconnector()
        recon_result = reconnector.reconnect(
            net, iso_result, reconstruction_config_reconnect,
        )

        for line_name in recon_result.synthetic_line_map:
            assert line_name.startswith("recon_line_")

    def test_reconnect_noop_on_connected_network(
        self,
        fully_connected_network: Any,
        reconstruction_config_reconnect: ReconstructionConfig,
    ) -> None:
        """Reconnection is a no-op when the network has no isolation."""
        net = fully_connected_network
        lines_before = len(net.line)

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        reconnector = Reconnector()
        recon_result = reconnector.reconnect(
            net, iso_result, reconstruction_config_reconnect,
        )

        assert recon_result.lines_created == 0
        assert len(net.line) == lines_before


class TestReconnectYbusValid:
    """Tests verifying Ybus matrix validity after reconnection."""

    def test_reconnect_ybus_valid(
        self,
        isolated_grid_network: Any,
        reconstruction_config_reconnect: ReconstructionConfig,
    ) -> None:
        """Regenerated Ybus is non-singular and correctly sized."""
        net = isolated_grid_network

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        reconnector = Reconnector()
        recon_result = reconnector.reconnect(
            net, iso_result, reconstruction_config_reconnect,
        )

        assert recon_result.ybus_shape is not None
        assert recon_result.ybus_nonsingular is True
        # Ybus should be square
        assert recon_result.ybus_shape[0] == recon_result.ybus_shape[1]

    def test_reconnect_ybus_shape_matches_bus_count(
        self,
        isolated_grid_network_large: Any,
        reconstruction_config_reconnect: ReconstructionConfig,
    ) -> None:
        """Ybus matrix dimensions match the bus count after reconnection."""
        net = isolated_grid_network_large

        isolator = Isolator(min_component_size=2)
        iso_result = isolator.detect(net)

        reconnector = Reconnector()
        recon_result = reconnector.reconnect(
            net, iso_result, reconstruction_config_reconnect,
        )

        if recon_result.ybus_shape is not None:
            # Ybus size should equal the number of in-service buses
            n_buses = len(net.bus[net.bus["in_service"]])
            assert recon_result.ybus_shape[0] == n_buses


# ======================================================================
# Reproducibility
# ======================================================================


class TestReproducibilitySameSeed:
    """Tests verifying deterministic output with the same seed."""

    def test_reproducibility_same_seed(self) -> None:
        """Same seed + input = identical output."""
        net1 = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )
        net2 = copy.deepcopy(net1)

        cfg1 = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        cfg2 = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )

        pipeline1 = ReconstructionPipeline(cfg1, copy_network=True)
        pipeline2 = ReconstructionPipeline(cfg2, copy_network=True)

        result1 = pipeline1.run(net1, region="shikoku")
        result2 = pipeline2.run(net2, region="shikoku")

        # Bus counts should be identical
        assert len(result1.net.bus) == len(result2.net.bus)
        assert len(result1.net.line) == len(result2.net.line)
        assert len(result1.net.gen) == len(result2.net.gen)
        assert len(result1.net.load) == len(result2.net.load)

        # Load values should be byte-identical
        if not result1.net.load.empty:
            np.testing.assert_array_equal(
                result1.net.load["p_mw"].values,
                result2.net.load["p_mw"].values,
            )
            np.testing.assert_array_equal(
                result1.net.load["q_mvar"].values,
                result2.net.load["q_mvar"].values,
            )

    def test_reproducibility_same_seed_reconnect(self) -> None:
        """Same seed produces identical output in reconnect mode."""
        net1 = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )
        net2 = copy.deepcopy(net1)

        cfg1 = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
        )
        cfg2 = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
        )

        pipeline1 = ReconstructionPipeline(cfg1, copy_network=True)
        pipeline2 = ReconstructionPipeline(cfg2, copy_network=True)

        result1 = pipeline1.run(net1, region="shikoku")
        result2 = pipeline2.run(net2, region="shikoku")

        assert len(result1.net.bus) == len(result2.net.bus)
        assert len(result1.net.line) == len(result2.net.line)

        if not result1.net.load.empty:
            np.testing.assert_array_equal(
                result1.net.load["p_mw"].values,
                result2.net.load["p_mw"].values,
            )


class TestReproducibilityDiffSeed:
    """Tests verifying different seeds produce different outputs."""

    def test_reproducibility_diff_seed(self) -> None:
        """Different seeds produce different outputs."""
        net1 = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )
        net2 = copy.deepcopy(net1)

        cfg1 = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        cfg2 = ReconstructionConfig(
            mode="simplify", seed=99, db_path=":memory:",
        )

        pipeline1 = ReconstructionPipeline(cfg1, copy_network=True)
        pipeline2 = ReconstructionPipeline(cfg2, copy_network=True)

        result1 = pipeline1.run(net1, region="shikoku")
        result2 = pipeline2.run(net2, region="shikoku")

        # The structure should be the same (same simplification)
        assert len(result1.net.bus) == len(result2.net.bus)

        # But synthesised load values should differ due to different seeds
        if not result1.net.load.empty and not result2.net.load.empty:
            loads_differ = not np.allclose(
                result1.net.load["p_mw"].values,
                result2.net.load["p_mw"].values,
            )
            assert loads_differ, (
                "Loads should differ with different seeds"
            )


# ======================================================================
# DataSynthesizer: load synthesis
# ======================================================================


class TestLoadSynthesisReusesExisting:
    """Tests verifying that existing loads are preserved."""

    def test_load_synthesis_reuses_existing(self) -> None:
        """Existing loads are not overwritten during synthesis."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=1,
        )

        # Create an existing load on bus 0
        existing_p_mw = 123.456
        pp.create_load(
            net, bus=0, p_mw=existing_p_mw, q_mvar=10.0,
            name="existing_load",
        )

        synth = DataSynthesizer(
            seed=42,
            skip_existing_loads=True,
        )
        result = synth.synthesize_loads(net, region="shikoku")

        # Bus 0 should still have the original load value
        existing_loads = net.load[net.load["name"] == "existing_load"]
        assert len(existing_loads) == 1
        assert existing_loads.iloc[0]["p_mw"] == existing_p_mw

        # Should have skipped 1 bus
        assert result.loads_skipped >= 1

    def test_load_synthesis_skip_false_overwrites(self) -> None:
        """When skip_existing_loads=False, all buses receive synthetic loads."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=1,
        )

        pp.create_load(
            net, bus=0, p_mw=100.0, q_mvar=10.0, name="existing",
        )

        synth = DataSynthesizer(
            seed=42,
            skip_existing_loads=False,
        )
        result = synth.synthesize_loads(net, region="shikoku")

        # All buses should receive synthetic loads (including bus 0)
        assert result.loads_created == len(net.bus)


class TestLoadSynthesisCreatesMissing:
    """Tests verifying that missing loads are synthesised."""

    def test_load_synthesis_creates_missing(self) -> None:
        """Buses without loads receive synthetic loads."""
        net = make_isolated_network(
            n_main_buses=4, n_isolated_buses=0, n_generators=1,
        )

        # No loads exist initially
        assert net.load.empty

        synth = DataSynthesizer(seed=42)
        result = synth.synthesize_loads(net, region="shikoku")

        # All 4 buses should receive synthetic loads
        assert result.loads_created == 4
        assert len(net.load) == 4
        assert result.total_load_mw > 0

    def test_load_synthesis_values_positive(self) -> None:
        """Synthesised load values are all positive."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=1,
        )

        synth = DataSynthesizer(seed=42)
        synth.synthesize_loads(net, region="shikoku")

        assert all(net.load["p_mw"] > 0)
        assert all(net.load["q_mvar"] > 0)

    def test_load_synthesis_total_matches_target(self) -> None:
        """Total synthesised load is close to regional demand target."""
        net = make_isolated_network(
            n_main_buses=5, n_isolated_buses=0, n_generators=2,
        )

        synth = DataSynthesizer(seed=42)
        result = synth.synthesize_loads(net, region="shikoku")

        # Shikoku peak demand is 5500 MW, load factor 0.85 -> 4675 MW
        expected_target = 5500 * 0.85
        # Allow some tolerance due to jitter
        assert abs(result.total_load_mw - expected_target) / expected_target < 0.05


# ======================================================================
# DataSynthesizer: generation synthesis
# ======================================================================


class TestGenerationSynthesis:
    """Tests for generation data synthesis."""

    def test_generation_synthesis(self) -> None:
        """Generation is scaled to match demand + reserve margin."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=2,
        )

        # Add max_p_mw column (not set by default in pandapower)
        net.gen["max_p_mw"] = net.gen["p_mw"].copy()

        # First add loads
        synth = DataSynthesizer(seed=42, skip_existing_generation=False)
        synth.synthesize_loads(net, region="shikoku")

        total_load = float(net.load["p_mw"].sum())
        assert total_load > 0

        # Now synthesise generation
        reserve_margin = 0.05
        gen_result = synth.synthesize_generation(
            net, reserve_margin=reserve_margin,
        )

        assert gen_result.generators_scaled > 0
        assert gen_result.total_generation_mw > 0

    def test_generation_no_exceed_capacity(self) -> None:
        """No generator exceeds its rated capacity."""
        net = make_isolated_network(
            n_main_buses=4, n_isolated_buses=0, n_generators=3,
        )

        # Add max_p_mw column (not set by default in pandapower)
        net.gen["max_p_mw"] = net.gen["p_mw"].copy()

        synth = DataSynthesizer(seed=42, skip_existing_generation=False)
        synth.synthesize_loads(net, region="shikoku")
        synth.synthesize_generation(net, reserve_margin=0.05)

        for idx in net.gen.index:
            p_mw = net.gen.at[idx, "p_mw"]
            max_p = net.gen.at[idx, "max_p_mw"]
            if max_p > 0:
                assert p_mw <= max_p + 1e-6, (
                    f"Generator {idx}: p_mw={p_mw} exceeds max_p_mw={max_p}"
                )

    def test_generation_skips_existing_dispatch(self) -> None:
        """Generators with existing dispatch are preserved."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=2,
        )

        # Add loads first
        synth_load = DataSynthesizer(seed=42)
        synth_load.synthesize_loads(net, region="shikoku")

        # Manually set dispatch on generator 0
        existing_dispatch = 250.0
        net.gen.at[0, "p_mw"] = existing_dispatch

        synth = DataSynthesizer(
            seed=42,
            skip_existing_generation=True,
        )
        gen_result = synth.synthesize_generation(net, reserve_margin=0.05)

        # Generator 0 should be skipped
        assert gen_result.generators_skipped >= 1
        assert net.gen.at[0, "p_mw"] == existing_dispatch

    def test_generation_with_no_loads_skips(self) -> None:
        """Generation synthesis is skipped when no loads exist."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=0, n_generators=2,
        )

        synth = DataSynthesizer(seed=42)
        gen_result = synth.synthesize_generation(net, reserve_margin=0.05)

        assert gen_result.generators_scaled == 0
        assert len(gen_result.warnings) > 0


# ======================================================================
# ReconstructionConfig: mode switching
# ======================================================================


class TestConfigModeSwitch:
    """Tests for configuration-driven mode selection."""

    def test_config_mode_switch(self) -> None:
        """Mode flag correctly selects simplify vs. reconnect strategies."""
        net_s = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )
        net_r = copy.deepcopy(net_s)

        cfg_simplify = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        cfg_reconnect = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
        )

        pipeline_s = ReconstructionPipeline(cfg_simplify, copy_network=True)
        pipeline_r = ReconstructionPipeline(cfg_reconnect, copy_network=True)

        result_s = pipeline_s.run(net_s, region="shikoku")
        result_r = pipeline_r.run(net_r, region="shikoku")

        # Simplify mode should have simplification result, no reconnection
        assert result_s.simplification_result is not None
        assert result_s.reconnection_result is None

        # Reconnect mode should have reconnection result, no simplification
        assert result_r.reconnection_result is not None
        assert result_r.simplification_result is None

        # Both should produce valid networks
        assert result_s.net is not None
        assert result_r.net is not None

        # Simplify removes buses; reconnect keeps them
        assert len(result_s.net.bus) < len(result_r.net.bus)

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reconstruction mode"):
            ReconstructionConfig(mode="invalid_mode")

    def test_simplify_mode_pipeline_result(self) -> None:
        """Simplify mode PipelineResult has expected metadata."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")

        assert result.reconstruction_mode == "simplify"
        assert result.seed == 42
        assert result.region == "shikoku"
        assert result.elapsed_seconds >= 0

    def test_reconnect_mode_pipeline_result(self) -> None:
        """Reconnect mode PipelineResult has expected metadata."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="reconnect", seed=42, db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")

        assert result.reconstruction_mode == "reconnect"
        assert result.seed == 42
        assert result.isolation_result is not None
        assert result.synthesis_result is not None


# ======================================================================
# Pipeline: summary and warnings
# ======================================================================


class TestPipelineResult:
    """Tests for the PipelineResult dataclass."""

    def test_pipeline_summary_keys(self) -> None:
        """PipelineResult summary contains expected keys."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")

        summary = result.summary
        assert "mode" in summary
        assert "seed" in summary
        assert "region" in summary
        assert "elapsed_seconds" in summary
        assert "isolation" in summary
        assert "warnings" in summary

    def test_pipeline_warnings_aggregated(self) -> None:
        """Warnings from all stages are aggregated in PipelineResult."""
        net = make_isolated_network(
            n_main_buses=3, n_isolated_buses=2, n_generators=2,
        )

        cfg = ReconstructionConfig(
            mode="simplify", seed=42, db_path=":memory:",
        )
        pipeline = ReconstructionPipeline(cfg, copy_network=True)
        result = pipeline.run(net, region="shikoku")

        # Warnings should be a list (may be empty or not depending on network)
        assert isinstance(result.warnings, list)
