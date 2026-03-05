"""Tests for the UC solver strategy selection module.

Tests solver strategy selection including:
- High-spec hardware maps to HIGH tier (full MILP, tight gap)
- Mid-spec hardware maps to MID tier (MILP, relaxed gap)
- Low-spec hardware maps to LOW tier (LP relaxation)
- Large generator count triggers decomposition flag
- SolverConfig has valid time_limit, gap, threads values
- Boundary conditions at tier thresholds (cores and RAM)
- SolverConfig validation rejects invalid parameters
- Solver preference selection from available solvers
- Memory estimation for problem size
"""

import pytest

from src.uc.hardware_detector import HardwareProfile
from src.uc.solver_strategy import (
    HIGH_TIER_MIN_CORES,
    HIGH_TIER_MIN_RAM_GB,
    LARGE_PROBLEM_GENERATOR_THRESHOLD,
    MID_TIER_MIN_CORES,
    MID_TIER_MIN_RAM_GB,
    SolverConfig,
    SolverTier,
    _classify_tier,
    _estimate_problem_memory_mb,
    _pick_solver,
    select_strategy,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_profile(
    physical_cores: int = 4,
    logical_cores: int = 8,
    available_ram_gb: float = 16.0,
    total_ram_gb: float = 32.0,
    available_solvers: list = None,
    os_name: str = "Darwin",
    architecture: str = "arm64",
) -> HardwareProfile:
    """Factory for creating HardwareProfile instances in tests."""
    return HardwareProfile(
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        available_ram_gb=available_ram_gb,
        total_ram_gb=total_ram_gb,
        available_solvers=available_solvers or ["HiGHS_CMD", "PULP_CBC_CMD"],
        os_name=os_name,
        architecture=architecture,
    )


def _high_spec_profile() -> HardwareProfile:
    """Create a high-spec hardware profile (>= 4 cores, >= 8 GB RAM)."""
    return _make_profile(
        physical_cores=8,
        logical_cores=16,
        available_ram_gb=32.0,
        total_ram_gb=64.0,
    )


def _mid_spec_profile() -> HardwareProfile:
    """Create a mid-spec hardware profile (>= 2 cores, >= 4 GB RAM)."""
    return _make_profile(
        physical_cores=2,
        logical_cores=4,
        available_ram_gb=6.0,
        total_ram_gb=8.0,
    )


def _low_spec_profile() -> HardwareProfile:
    """Create a low-spec hardware profile (< 2 cores or < 4 GB RAM)."""
    return _make_profile(
        physical_cores=1,
        logical_cores=1,
        available_ram_gb=2.0,
        total_ram_gb=4.0,
    )


# ======================================================================
# TestSolverTierEnum
# ======================================================================


class TestSolverTierEnum:
    """Tests for the SolverTier enumeration."""

    def test_high_tier_value(self) -> None:
        """HIGH tier has value 'high'."""
        assert SolverTier.HIGH.value == "high"

    def test_mid_tier_value(self) -> None:
        """MID tier has value 'mid'."""
        assert SolverTier.MID.value == "mid"

    def test_low_tier_value(self) -> None:
        """LOW tier has value 'low'."""
        assert SolverTier.LOW.value == "low"

    def test_tier_count(self) -> None:
        """There are exactly 3 solver tiers."""
        assert len(SolverTier) == 3


# ======================================================================
# TestSolverConfigValidation
# ======================================================================


class TestSolverConfigValidation:
    """Tests that SolverConfig validates its parameters on init."""

    def test_valid_config_creation(self) -> None:
        """A SolverConfig with valid parameters is created without error."""
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="HiGHS_CMD",
            time_limit_s=600.0,
            mip_gap=0.01,
            threads=4,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="test config",
        )

        assert config.tier == SolverTier.HIGH
        assert config.solver_name == "HiGHS_CMD"
        assert config.time_limit_s == 600.0
        assert config.mip_gap == 0.01
        assert config.threads == 4

    def test_rejects_zero_time_limit(self) -> None:
        """time_limit_s <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="time_limit_s must be positive"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=0.0,
                mip_gap=0.01,
                threads=4,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_rejects_negative_time_limit(self) -> None:
        """Negative time_limit_s raises ValueError."""
        with pytest.raises(ValueError, match="time_limit_s must be positive"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=-10.0,
                mip_gap=0.01,
                threads=4,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_rejects_negative_mip_gap(self) -> None:
        """Negative mip_gap raises ValueError."""
        with pytest.raises(ValueError, match="mip_gap must be between 0 and 1"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=600.0,
                mip_gap=-0.01,
                threads=4,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_rejects_mip_gap_above_one(self) -> None:
        """mip_gap > 1 raises ValueError."""
        with pytest.raises(ValueError, match="mip_gap must be between 0 and 1"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=600.0,
                mip_gap=1.5,
                threads=4,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_rejects_zero_threads(self) -> None:
        """threads < 1 raises ValueError."""
        with pytest.raises(ValueError, match="threads must be >= 1"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=600.0,
                mip_gap=0.01,
                threads=0,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_rejects_negative_threads(self) -> None:
        """Negative threads raises ValueError."""
        with pytest.raises(ValueError, match="threads must be >= 1"):
            SolverConfig(
                tier=SolverTier.HIGH,
                solver_name="HiGHS_CMD",
                time_limit_s=600.0,
                mip_gap=0.01,
                threads=-1,
                use_decomposition=False,
                decomposition_strategy=None,
                use_lp_relaxation=False,
                description="test",
            )

    def test_zero_mip_gap_is_valid(self) -> None:
        """mip_gap=0.0 is accepted (exact optimality)."""
        config = SolverConfig(
            tier=SolverTier.HIGH,
            solver_name="HiGHS_CMD",
            time_limit_s=600.0,
            mip_gap=0.0,
            threads=4,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=False,
            description="test",
        )
        assert config.mip_gap == 0.0

    def test_mip_gap_one_is_valid(self) -> None:
        """mip_gap=1.0 is accepted (boundary)."""
        config = SolverConfig(
            tier=SolverTier.LOW,
            solver_name="PULP_CBC_CMD",
            time_limit_s=60.0,
            mip_gap=1.0,
            threads=1,
            use_decomposition=False,
            decomposition_strategy=None,
            use_lp_relaxation=True,
            description="test",
        )
        assert config.mip_gap == 1.0


# ======================================================================
# TestSelectStrategyHighTier
# ======================================================================


class TestSelectStrategyHighTier:
    """Tests that high-spec hardware maps to HIGH tier."""

    def test_high_spec_returns_high_tier(self) -> None:
        """High-spec hardware (8 cores, 32 GB) maps to HIGH tier."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.HIGH

    def test_high_tier_tight_mip_gap(self) -> None:
        """HIGH tier uses tight MIP gap (1%)."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.mip_gap == 0.01

    def test_high_tier_long_time_limit(self) -> None:
        """HIGH tier uses a generous time limit (600s)."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.time_limit_s == 600.0

    def test_high_tier_multi_threaded(self) -> None:
        """HIGH tier uses multiple solver threads matching physical cores."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.threads == profile.physical_cores
        assert config.threads >= 1

    def test_high_tier_no_lp_relaxation(self) -> None:
        """HIGH tier uses full MILP, not LP relaxation."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_lp_relaxation is False

    def test_high_tier_no_decomposition_small_problem(self) -> None:
        """HIGH tier does not use decomposition for small problems."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None

    def test_high_tier_no_decomposition_large_problem(self) -> None:
        """HIGH tier does not use decomposition even for large problems."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=100, n_periods=48)

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None

    def test_high_tier_has_description(self) -> None:
        """HIGH tier config includes a human-readable description."""
        profile = _high_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert isinstance(config.description, str)
        assert len(config.description) > 0
        assert "HIGH" in config.description


# ======================================================================
# TestSelectStrategyMidTier
# ======================================================================


class TestSelectStrategyMidTier:
    """Tests that mid-spec hardware maps to MID tier."""

    def test_mid_spec_returns_mid_tier(self) -> None:
        """Mid-spec hardware (2 cores, 6 GB) maps to MID tier."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.MID

    def test_mid_tier_relaxed_mip_gap(self) -> None:
        """MID tier uses relaxed MIP gap (5%)."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.mip_gap == 0.05

    def test_mid_tier_shorter_time_limit(self) -> None:
        """MID tier uses a shorter time limit (300s) than HIGH."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.time_limit_s == 300.0

    def test_mid_tier_threads_match_cores(self) -> None:
        """MID tier uses threads matching physical cores."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.threads == profile.physical_cores

    def test_mid_tier_no_lp_relaxation(self) -> None:
        """MID tier uses full MILP, not LP relaxation."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_lp_relaxation is False

    def test_mid_tier_no_decomposition_small_problem(self) -> None:
        """MID tier does not use decomposition for small problems."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None

    def test_mid_tier_has_description(self) -> None:
        """MID tier config includes a human-readable description."""
        profile = _mid_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert isinstance(config.description, str)
        assert len(config.description) > 0
        assert "MID" in config.description


# ======================================================================
# TestSelectStrategyLowTier
# ======================================================================


class TestSelectStrategyLowTier:
    """Tests that low-spec hardware maps to LOW tier."""

    def test_low_spec_returns_low_tier(self) -> None:
        """Low-spec hardware (1 core, 2 GB) maps to LOW tier."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_low_tier_aggressive_mip_gap(self) -> None:
        """LOW tier uses aggressive MIP gap (10%)."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.mip_gap == 0.10

    def test_low_tier_short_time_limit(self) -> None:
        """LOW tier uses a short time limit (120s)."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.time_limit_s == 120.0

    def test_low_tier_uses_lp_relaxation(self) -> None:
        """LOW tier uses LP relaxation (mip=False)."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_lp_relaxation is True

    def test_low_tier_single_core_only(self) -> None:
        """LOW tier with 1 physical core gets threads=1."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.threads == 1

    def test_low_tier_has_description(self) -> None:
        """LOW tier config includes a human-readable description."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert isinstance(config.description, str)
        assert len(config.description) > 0
        assert "LOW" in config.description

    def test_low_spec_few_cores(self) -> None:
        """1 core with sufficient RAM still maps to LOW tier."""
        profile = _make_profile(physical_cores=1, available_ram_gb=16.0)
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_low_spec_low_ram(self) -> None:
        """Sufficient cores but low RAM maps to LOW tier."""
        profile = _make_profile(physical_cores=4, available_ram_gb=3.0)
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW


# ======================================================================
# TestLargeProblemDecomposition
# ======================================================================


class TestLargeProblemDecomposition:
    """Tests that large problems trigger decomposition flag."""

    def test_mid_tier_large_problem_triggers_decomposition(self) -> None:
        """MID tier enables decomposition for >= 50 generators."""
        profile = _mid_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD,
            n_periods=24,
        )

        assert config.use_decomposition is True
        assert config.decomposition_strategy == "regional"

    def test_mid_tier_below_threshold_no_decomposition(self) -> None:
        """MID tier does not decompose below the generator threshold."""
        profile = _mid_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD - 1,
            n_periods=24,
        )

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None

    def test_low_tier_many_periods_triggers_decomposition(self) -> None:
        """LOW tier enables time-window decomposition for > 24 periods."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=48)

        assert config.use_decomposition is True
        assert config.decomposition_strategy == "time_window"

    def test_low_tier_24_periods_no_decomposition(self) -> None:
        """LOW tier does not decompose for exactly 24 periods (small problem)."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None

    def test_low_tier_large_generators_triggers_decomposition(self) -> None:
        """LOW tier enables decomposition for >= 50 generators."""
        profile = _low_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD,
            n_periods=24,
        )

        assert config.use_decomposition is True
        assert config.decomposition_strategy == "time_window"

    def test_high_tier_never_decomposes(self) -> None:
        """HIGH tier does not use decomposition regardless of problem size."""
        profile = _high_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD * 2,
            n_periods=168,
        )

        assert config.use_decomposition is False
        assert config.decomposition_strategy is None


# ======================================================================
# TestSolverConfigParameters
# ======================================================================


class TestSolverConfigParameters:
    """Tests that SolverConfig has valid time_limit, gap, threads values."""

    def test_high_config_valid_time_limit(self) -> None:
        """HIGH tier time_limit_s is positive."""
        config = select_strategy(_high_spec_profile(), 10, 24)
        assert config.time_limit_s > 0

    def test_mid_config_valid_time_limit(self) -> None:
        """MID tier time_limit_s is positive."""
        config = select_strategy(_mid_spec_profile(), 10, 24)
        assert config.time_limit_s > 0

    def test_low_config_valid_time_limit(self) -> None:
        """LOW tier time_limit_s is positive."""
        config = select_strategy(_low_spec_profile(), 10, 24)
        assert config.time_limit_s > 0

    def test_high_config_valid_mip_gap(self) -> None:
        """HIGH tier mip_gap is between 0 and 1."""
        config = select_strategy(_high_spec_profile(), 10, 24)
        assert 0 <= config.mip_gap <= 1

    def test_mid_config_valid_mip_gap(self) -> None:
        """MID tier mip_gap is between 0 and 1."""
        config = select_strategy(_mid_spec_profile(), 10, 24)
        assert 0 <= config.mip_gap <= 1

    def test_low_config_valid_mip_gap(self) -> None:
        """LOW tier mip_gap is between 0 and 1."""
        config = select_strategy(_low_spec_profile(), 10, 24)
        assert 0 <= config.mip_gap <= 1

    def test_high_config_valid_threads(self) -> None:
        """HIGH tier threads >= 1."""
        config = select_strategy(_high_spec_profile(), 10, 24)
        assert config.threads >= 1

    def test_mid_config_valid_threads(self) -> None:
        """MID tier threads >= 1."""
        config = select_strategy(_mid_spec_profile(), 10, 24)
        assert config.threads >= 1

    def test_low_config_valid_threads(self) -> None:
        """LOW tier threads >= 1."""
        config = select_strategy(_low_spec_profile(), 10, 24)
        assert config.threads >= 1

    def test_time_limit_ordering(self) -> None:
        """HIGH tier has longest time limit, LOW tier has shortest."""
        high_cfg = select_strategy(_high_spec_profile(), 10, 24)
        mid_cfg = select_strategy(_mid_spec_profile(), 10, 24)
        low_cfg = select_strategy(_low_spec_profile(), 10, 24)

        assert high_cfg.time_limit_s > mid_cfg.time_limit_s
        assert mid_cfg.time_limit_s > low_cfg.time_limit_s

    def test_mip_gap_ordering(self) -> None:
        """HIGH tier has tightest gap, LOW tier has most relaxed."""
        high_cfg = select_strategy(_high_spec_profile(), 10, 24)
        mid_cfg = select_strategy(_mid_spec_profile(), 10, 24)
        low_cfg = select_strategy(_low_spec_profile(), 10, 24)

        assert high_cfg.mip_gap < mid_cfg.mip_gap
        assert mid_cfg.mip_gap < low_cfg.mip_gap

    def test_solver_name_is_nonempty_string(self) -> None:
        """All tiers produce a non-empty solver_name."""
        for profile_fn in [_high_spec_profile, _mid_spec_profile, _low_spec_profile]:
            config = select_strategy(profile_fn(), 10, 24)
            assert isinstance(config.solver_name, str)
            assert len(config.solver_name) > 0


# ======================================================================
# TestBoundaryConditions
# ======================================================================


class TestBoundaryConditions:
    """Tests tier boundary conditions at exact threshold values."""

    def test_exactly_high_threshold_cores_and_ram(self) -> None:
        """Exactly at HIGH threshold (4 cores, 8 GB) maps to HIGH."""
        profile = _make_profile(
            physical_cores=HIGH_TIER_MIN_CORES,
            available_ram_gb=HIGH_TIER_MIN_RAM_GB,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.HIGH

    def test_one_below_high_cores_maps_to_mid(self) -> None:
        """One core below HIGH threshold with sufficient RAM maps to MID."""
        profile = _make_profile(
            physical_cores=HIGH_TIER_MIN_CORES - 1,
            available_ram_gb=HIGH_TIER_MIN_RAM_GB,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.MID

    def test_just_below_high_ram_maps_to_mid(self) -> None:
        """Just below HIGH RAM threshold with sufficient cores maps to MID."""
        profile = _make_profile(
            physical_cores=HIGH_TIER_MIN_CORES,
            available_ram_gb=HIGH_TIER_MIN_RAM_GB - 0.1,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.MID

    def test_exactly_mid_threshold_cores_and_ram(self) -> None:
        """Exactly at MID threshold (2 cores, 4 GB) maps to MID."""
        profile = _make_profile(
            physical_cores=MID_TIER_MIN_CORES,
            available_ram_gb=MID_TIER_MIN_RAM_GB,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.MID

    def test_one_below_mid_cores_maps_to_low(self) -> None:
        """One core below MID threshold maps to LOW."""
        profile = _make_profile(
            physical_cores=MID_TIER_MIN_CORES - 1,
            available_ram_gb=MID_TIER_MIN_RAM_GB,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_just_below_mid_ram_maps_to_low(self) -> None:
        """Just below MID RAM threshold maps to LOW."""
        profile = _make_profile(
            physical_cores=MID_TIER_MIN_CORES,
            available_ram_gb=MID_TIER_MIN_RAM_GB - 0.1,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_both_below_mid_maps_to_low(self) -> None:
        """Both cores and RAM below MID threshold maps to LOW."""
        profile = _make_profile(
            physical_cores=1,
            available_ram_gb=2.0,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_high_cores_low_ram_maps_to_low(self) -> None:
        """High cores but very low RAM maps to LOW (both conditions needed)."""
        profile = _make_profile(
            physical_cores=16,
            available_ram_gb=1.0,
        )
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.tier == SolverTier.LOW

    def test_decomposition_at_exact_threshold(self) -> None:
        """MID tier at exactly LARGE_PROBLEM_GENERATOR_THRESHOLD triggers decomp."""
        profile = _mid_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD,
            n_periods=24,
        )
        assert config.use_decomposition is True

    def test_decomposition_one_below_threshold(self) -> None:
        """MID tier one generator below threshold does not trigger decomp."""
        profile = _mid_spec_profile()
        config = select_strategy(
            profile,
            n_generators=LARGE_PROBLEM_GENERATOR_THRESHOLD - 1,
            n_periods=24,
        )
        assert config.use_decomposition is False

    def test_low_tier_decomp_at_25_periods(self) -> None:
        """LOW tier at 25 periods (> 24) triggers time-window decomposition."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=25)

        assert config.use_decomposition is True
        assert config.decomposition_strategy == "time_window"

    def test_low_tier_no_decomp_at_24_periods(self) -> None:
        """LOW tier at exactly 24 periods does not trigger decomposition."""
        profile = _low_spec_profile()
        config = select_strategy(profile, n_generators=10, n_periods=24)

        assert config.use_decomposition is False


# ======================================================================
# TestClassifyTier
# ======================================================================


class TestClassifyTier:
    """Tests the internal _classify_tier helper."""

    def test_classify_high(self) -> None:
        """High-spec profile classifies as HIGH."""
        tier = _classify_tier(_high_spec_profile())
        assert tier == SolverTier.HIGH

    def test_classify_mid(self) -> None:
        """Mid-spec profile classifies as MID."""
        tier = _classify_tier(_mid_spec_profile())
        assert tier == SolverTier.MID

    def test_classify_low(self) -> None:
        """Low-spec profile classifies as LOW."""
        tier = _classify_tier(_low_spec_profile())
        assert tier == SolverTier.LOW

    def test_classify_mid_high_cores_low_ram(self) -> None:
        """Many cores but mid-range RAM classifies as MID."""
        profile = _make_profile(physical_cores=8, available_ram_gb=5.0)
        tier = _classify_tier(profile)
        assert tier == SolverTier.MID


# ======================================================================
# TestPickSolver
# ======================================================================


class TestPickSolver:
    """Tests the internal _pick_solver helper."""

    def test_prefers_highs_when_available(self) -> None:
        """HiGHS_CMD is preferred when available."""
        solver = _pick_solver(["PULP_CBC_CMD", "HiGHS_CMD"])
        assert solver == "HiGHS_CMD"

    def test_falls_back_to_cbc(self) -> None:
        """Falls back to PULP_CBC_CMD when HiGHS not available."""
        solver = _pick_solver(["PULP_CBC_CMD"])
        assert solver == "PULP_CBC_CMD"

    def test_fallback_when_no_preferred_solver(self) -> None:
        """Returns HiGHS fallback when no preferred solver found."""
        solver = _pick_solver(["GLPK_CMD"])
        assert solver == "HiGHS"

    def test_empty_solver_list_fallback(self) -> None:
        """Returns HiGHS fallback for empty solver list."""
        solver = _pick_solver([])
        assert solver == "HiGHS"


# ======================================================================
# TestEstimateProblemMemory
# ======================================================================


class TestEstimateProblemMemory:
    """Tests the memory estimation helper."""

    def test_positive_memory_estimate(self) -> None:
        """Memory estimate is always positive for valid inputs."""
        mem_mb = _estimate_problem_memory_mb(10, 24)
        assert mem_mb > 0

    def test_larger_problem_needs_more_memory(self) -> None:
        """More generators/periods -> higher memory estimate."""
        small = _estimate_problem_memory_mb(10, 24)
        large = _estimate_problem_memory_mb(100, 48)
        assert large > small

    def test_linear_scaling_with_generators(self) -> None:
        """Memory scales linearly with generator count."""
        mem_10 = _estimate_problem_memory_mb(10, 24)
        mem_20 = _estimate_problem_memory_mb(20, 24)
        assert abs(mem_20 - 2 * mem_10) < 1e-6

    def test_linear_scaling_with_periods(self) -> None:
        """Memory scales linearly with period count."""
        mem_24 = _estimate_problem_memory_mb(10, 24)
        mem_48 = _estimate_problem_memory_mb(10, 48)
        assert abs(mem_48 - 2 * mem_24) < 1e-6
