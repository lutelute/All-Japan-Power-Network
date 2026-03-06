"""Reconstruction configuration loading and validation.

Provides a dataclass-based configuration object that can be constructed
programmatically or loaded from a YAML file.  All parameters have sensible
defaults that match the values in ``config/reconstruction.yaml``.

Usage::

    from src.reconstruction.config import ReconstructionConfig

    # Programmatic construction with defaults
    cfg = ReconstructionConfig()

    # Override specific parameters
    cfg = ReconstructionConfig(mode="reconnect", seed=123)

    # Load from YAML
    cfg = ReconstructionConfig.from_yaml("config/reconstruction.yaml")
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "config/reconstruction.yaml"

# Valid reconstruction modes
VALID_MODES = ("simplify", "reconnect")


@dataclass
class ReconstructionConfig:
    """Configuration for the network reconstruction pipeline.

    Attributes:
        mode: Reconstruction strategy — ``"simplify"`` to remove isolated
            elements or ``"reconnect"`` to generate synthetic connections.
        seed: Random seed for reproducible synthetic data generation.
        min_reactance_ohm_per_km: Minimum reactance (Ohm/km) for synthetic
            lines.  Prevents Ybus singularity.
        min_component_size: Minimum bus count for a connected component to
            be considered part of the main network.
        max_reconnection_distance_km: Maximum search radius (km) when
            finding a main-component bus for reconnection.
        default_voltage_kv: Fallback voltage (kV) for synthetic lines when
            the isolated bus voltage is unknown.
        reserve_margin: Generation reserve margin (fraction) above total
            demand.
        skip_existing_loads: If ``True``, preserve existing load data during
            synthesis.
        skip_existing_generation: If ``True``, preserve existing generation
            dispatch during synthesis.
        db_path: Path to the SQLite database for grid attribute storage.
    """

    mode: str = "simplify"
    seed: int = 42
    min_reactance_ohm_per_km: float = 0.001
    min_component_size: int = 2
    max_reconnection_distance_km: float = 200.0
    default_voltage_kv: float = 66.0
    reserve_margin: float = 0.05
    skip_existing_loads: bool = True
    skip_existing_generation: bool = True
    db_path: str = "data/grid_attributes.db"

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        self._validate()

    def _validate(self) -> None:
        """Check that all configuration values are within acceptable ranges.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid reconstruction mode '{self.mode}'; "
                f"must be one of {VALID_MODES}"
            )

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(
                f"Seed must be a non-negative integer, got {self.seed!r}"
            )

        if self.min_reactance_ohm_per_km <= 0:
            raise ValueError(
                "min_reactance_ohm_per_km must be positive, "
                f"got {self.min_reactance_ohm_per_km}"
            )

        if self.min_component_size < 1:
            raise ValueError(
                "min_component_size must be >= 1, "
                f"got {self.min_component_size}"
            )

        if self.max_reconnection_distance_km <= 0:
            raise ValueError(
                "max_reconnection_distance_km must be positive, "
                f"got {self.max_reconnection_distance_km}"
            )

        if self.default_voltage_kv <= 0:
            raise ValueError(
                "default_voltage_kv must be positive, "
                f"got {self.default_voltage_kv}"
            )

        if self.reserve_margin < 0:
            raise ValueError(
                "reserve_margin must be non-negative, "
                f"got {self.reserve_margin}"
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "ReconstructionConfig":
        """Load reconstruction configuration from a YAML file.

        Unknown keys in the YAML file are silently ignored to allow
        forward-compatible configuration files.

        Args:
            config_path: Path to the reconstruction YAML config file.

        Returns:
            A validated ``ReconstructionConfig`` instance.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            ValueError: If any parsed value fails validation.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Filter to only known fields to avoid TypeError on unknown keys
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known_fields}

        config = cls(**filtered)

        logger.info(
            "Loaded reconstruction config from '%s': mode=%s, seed=%d",
            config_path,
            config.mode,
            config.seed,
        )
        return config

    @property
    def summary(self) -> Dict[str, object]:
        """Return a dictionary summary of the configuration for logging."""
        return {
            "mode": self.mode,
            "seed": self.seed,
            "min_reactance_ohm_per_km": self.min_reactance_ohm_per_km,
            "min_component_size": self.min_component_size,
            "max_reconnection_distance_km": self.max_reconnection_distance_km,
            "default_voltage_kv": self.default_voltage_kv,
            "reserve_margin": self.reserve_margin,
            "skip_existing_loads": self.skip_existing_loads,
            "skip_existing_generation": self.skip_existing_generation,
            "db_path": self.db_path,
        }
